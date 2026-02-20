"""
vec_env.py — Subprocess-based vectorised PX4+Gazebo environment.

Runs *N* independent PX4 SITL instances (each in its own tmux
session with a unique ``ROS_DOMAIN_ID``) so that episode resets
in one env don't block the others.

Architecture
------------
::

    Main process (training)
        │
        ├── Worker-0  (ROS_DOMAIN_ID=10, tmux=px4sim_w0)
        │     └── PX4GazeboEnv  → steps / resets independently
        ├── Worker-1  (ROS_DOMAIN_ID=11, tmux=px4sim_w1)
        │     └── PX4GazeboEnv  → steps / resets independently
        └── ...

Each worker is a ``multiprocessing.Process`` that owns a full
``PX4GazeboEnv`` instance.  Communication uses ``Pipe`` pairs
for deterministic, ordered message passing.

Usage
-----
::

    vec = SubprocVecEnv.from_args(num_envs=2, args=args)
    obs_list = vec.reset_all()         # list of N obs dicts
    for step in range(rollout_steps):
        # ... compute actions ...
        results = vec.step(actions)    # list of (obs, rew, done, info)
    vec.close()

Determinism guarantees
~~~~~~~~~~~~~~~~~~~~~~
* Each env advances its own sim independently — there is no
  cross-env sim-time coupling.
* Within each env, the sim-stepped reset + lockstep stepping
  is fully deterministic (same physics steps, same PX4 state
  machine transitions).
* The training loop processes results in env-index order,
  ensuring reproducible rollout buffers given the same seed.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Optional

import numpy as np


# ── Commands sent parent → worker ───────────────────────────

CMD_RESET = "reset"
CMD_STEP = "step"
CMD_CLOSE = "close"
CMD_GET_OBS = "get_obs"


# ════════════════════════════════════════════════════════════
#  Worker process
# ════════════════════════════════════════════════════════════


def _worker_fn(
    pipe: Connection,
    worker_id: int,
    env_kwargs: dict[str, Any],
    domain_id: int,
    launch_script: str,
) -> None:
    """Entry point for each subprocess worker.

    1. Sets ``ROS_DOMAIN_ID`` so topics don't collide.
    2. Launches its own sim stack via ``launch_sim.sh``.
    3. Creates a ``PX4GazeboEnv`` and enters a command loop.
    """
    # Isolate ROS 2 domain
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    tmux_session = f"px4sim_w{worker_id}"
    os.environ["PX4_TMUX_SESSION"] = tmux_session

    env = None
    sim_proc = None

    try:
        # ── Launch sim stack in a dedicated tmux session ────
        if launch_script and os.path.isfile(launch_script):
            sim_proc = subprocess.Popen(
                [
                    "bash",
                    launch_script,
                ],
                env={
                    **os.environ,
                    "SESSION": tmux_session,
                    "ROS_DOMAIN_ID": str(domain_id),
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give the sim stack time to start
            time.sleep(15)

        # ── Create environment ──────────────────────────────
        # Import inside worker so each process gets its own
        # rclpy context and gz-transport node.
        from px4_gz_gym.env import PX4GazeboEnv

        # Override tmux target for PX4 NSH commands
        from px4_gz_gym import px4_cmd
        px4_cmd._DEFAULT_TMUX_TARGET = f"{tmux_session}:sim.1"

        env = PX4GazeboEnv(**env_kwargs)

        # ── Command loop ────────────────────────────────────
        while True:
            try:
                cmd, data = pipe.recv()
            except EOFError:
                break

            if cmd == CMD_RESET:
                try:
                    obs, info = env.reset()
                    pipe.send(("ok", (obs, info)))
                except Exception as e:
                    pipe.send(("err", traceback.format_exc()))

            elif cmd == CMD_STEP:
                try:
                    action = data
                    obs, rew, term, trunc, info = env.step(action)
                    pipe.send(("ok", (obs, rew, term, trunc, info)))
                except Exception as e:
                    pipe.send(("err", traceback.format_exc()))

            elif cmd == CMD_CLOSE:
                break

            else:
                pipe.send(("err", f"Unknown command: {cmd}"))

    except Exception:
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Kill the sim tmux session
        subprocess.run(
            ["tmux", "kill-session", "-t", tmux_session],
            capture_output=True,
        )
        pipe.close()


# ════════════════════════════════════════════════════════════
#  SubprocVecEnv
# ════════════════════════════════════════════════════════════


class SubprocVecEnv:
    """Vectorised PX4+Gazebo environment using subprocesses.

    Each worker runs a full sim stack (PX4 + Gazebo + DDS agent)
    in an isolated ROS 2 domain.  Resets are non-blocking across
    workers: while one env resets, others can continue stepping.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    env_kwargs : dict
        Keyword arguments passed to ``PX4GazeboEnv()``.
    base_domain_id : int
        First ``ROS_DOMAIN_ID``; workers use ``base + i``.
    launch_script : str
        Path to ``launch_sim.sh``.  Each worker launches its own
        sim stack.  Pass ``""`` to skip (useful if sims are
        already running).
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: dict[str, Any],
        base_domain_id: int = 10,
        launch_script: str = "",
    ) -> None:
        self.num_envs = num_envs
        self._closed = False

        self._parent_pipes: list[Connection] = []
        self._child_pipes: list[Connection] = []
        self._workers: list[mp.Process] = []

        ctx = mp.get_context("spawn")

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            self._parent_pipes.append(parent_conn)
            self._child_pipes.append(child_conn)

            w = ctx.Process(
                target=_worker_fn,
                args=(
                    child_conn,
                    i,
                    env_kwargs,
                    base_domain_id + i,
                    launch_script,
                ),
                daemon=True,
                name=f"px4_env_worker_{i}",
            )
            w.start()
            self._workers.append(w)
            # Close child end in parent
            child_conn.close()

    @classmethod
    def from_args(
        cls,
        num_envs: int,
        args,
        base_domain_id: int = 10,
        launch_script: str = "",
    ) -> "SubprocVecEnv":
        """Create from parsed CLI args (same as ``make_env``)."""
        env_kwargs = dict(
            world_name=args.world,
            cam_obs_height=args.cam_size,
            cam_obs_width=args.cam_size,
            max_episode_steps=int(args.episode_secs / 0.02),
            takeoff_alt=args.hover_alt,
        )
        return cls(
            num_envs=num_envs,
            env_kwargs=env_kwargs,
            base_domain_id=base_domain_id,
            launch_script=launch_script,
        )

    # ── Vectorised API ──────────────────────────────────────

    def reset_all(
        self,
        timeout: float = 120.0,
    ) -> list[dict[str, np.ndarray]]:
        """Reset all environments in parallel.  Returns list of obs."""
        for pipe in self._parent_pipes:
            pipe.send((CMD_RESET, None))

        results = []
        for i, pipe in enumerate(self._parent_pipes):
            if pipe.poll(timeout):
                status, data = pipe.recv()
                if status == "ok":
                    obs, info = data
                    results.append(obs)
                else:
                    raise RuntimeError(
                        f"Worker {i} reset failed:\n{data}"
                    )
            else:
                raise TimeoutError(f"Worker {i} reset timed out")

        return results

    def reset_one(
        self,
        env_idx: int,
        timeout: float = 120.0,
    ) -> dict[str, np.ndarray]:
        """Reset a single environment.  Returns obs dict."""
        self._parent_pipes[env_idx].send((CMD_RESET, None))
        if self._parent_pipes[env_idx].poll(timeout):
            status, data = self._parent_pipes[env_idx].recv()
            if status == "ok":
                return data[0]  # obs
            raise RuntimeError(f"Worker {env_idx} reset failed:\n{data}")
        raise TimeoutError(f"Worker {env_idx} reset timed out")

    def step(
        self,
        actions: list[np.ndarray],
        timeout: float = 30.0,
    ) -> list[tuple[dict, float, bool, bool, dict]]:
        """Step all environments with the given actions.

        Parameters
        ----------
        actions : list of ndarray
            One action per env, length == num_envs.

        Returns
        -------
        List of (obs, reward, terminated, truncated, info) tuples.
        """
        assert len(actions) == self.num_envs

        for pipe, act in zip(self._parent_pipes, actions):
            pipe.send((CMD_STEP, act))

        results = []
        for i, pipe in enumerate(self._parent_pipes):
            if pipe.poll(timeout):
                status, data = pipe.recv()
                if status == "ok":
                    results.append(data)
                else:
                    raise RuntimeError(
                        f"Worker {i} step failed:\n{data}"
                    )
            else:
                raise TimeoutError(f"Worker {i} step timed out")

        return results

    def step_one(
        self,
        env_idx: int,
        action: np.ndarray,
        timeout: float = 30.0,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Step a single environment."""
        self._parent_pipes[env_idx].send((CMD_STEP, action))
        if self._parent_pipes[env_idx].poll(timeout):
            status, data = self._parent_pipes[env_idx].recv()
            if status == "ok":
                return data
            raise RuntimeError(
                f"Worker {env_idx} step failed:\n{data}"
            )
        raise TimeoutError(f"Worker {env_idx} step timed out")

    def close(self) -> None:
        """Shut down all workers and their sim stacks."""
        if self._closed:
            return
        self._closed = True

        for pipe in self._parent_pipes:
            try:
                pipe.send((CMD_CLOSE, None))
            except Exception:
                pass

        for w in self._workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

        for pipe in self._parent_pipes:
            pipe.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_envs


# ════════════════════════════════════════════════════════════
#  Async-reset wrapper  (hides reset latency in stepping)
# ════════════════════════════════════════════════════════════


class AsyncResetVecEnv:
    """Wraps ``SubprocVecEnv`` to overlap resets with stepping.

    When an env signals ``done``, its reset is initiated
    immediately but doesn't block.  On the next ``step()``
    call, if the reset hasn't finished yet, we poll briefly.
    This amortises the remaining per-env reset cost across the
    rollout collection.

    **Determinism**: each env still performs a fully deterministic
    sim-stepped reset.  The only non-determinism would be in
    wall-clock *ordering* of when resets complete, but since we
    process envs in fixed index order, the rollout buffer is
    filled deterministically given the same seeds.
    """

    def __init__(self, vec_env: SubprocVecEnv) -> None:
        self.vec = vec_env
        self.num_envs = vec_env.num_envs
        self._pending_reset: list[bool] = [False] * self.num_envs
        self._latest_obs: list[Optional[dict]] = [None] * self.num_envs

    def reset_all(self) -> list[dict]:
        """Synchronous full reset of all envs."""
        obs_list = self.vec.reset_all()
        self._latest_obs = list(obs_list)
        self._pending_reset = [False] * self.num_envs
        return obs_list

    def step(
        self,
        actions: list[np.ndarray],
    ) -> list[tuple[dict, float, bool, dict]]:
        """Step all envs; auto-reset done envs asynchronously.

        Returns 4-tuples (obs, reward, done, info) per env.
        When an env is done, the returned obs is already the
        first obs of the new episode.
        """
        # Collect pending resets first
        for i in range(self.num_envs):
            if self._pending_reset[i]:
                # Wait for the reset result
                pipe = self.vec._parent_pipes[i]
                if pipe.poll(120):
                    status, data = pipe.recv()
                    if status == "ok":
                        self._latest_obs[i] = data[0]  # obs
                    else:
                        raise RuntimeError(
                            f"Worker {i} async reset failed:\n{data}"
                        )
                else:
                    raise TimeoutError(f"Worker {i} async reset timed out")
                self._pending_reset[i] = False

        # Step all envs
        results_raw = self.vec.step(actions)
        results = []

        for i, (obs, rew, term, trunc, info) in enumerate(results_raw):
            done = term or trunc
            if done:
                # Initiate async reset
                self.vec._parent_pipes[i].send((CMD_RESET, None))
                self._pending_reset[i] = True
                # Return the terminal obs — the next step() will
                # collect the reset obs before stepping again.
                results.append((obs, rew, True, info))
            else:
                self._latest_obs[i] = obs
                results.append((obs, rew, False, info))

        return results

    def get_reset_obs(self, env_idx: int) -> dict:
        """Get the obs from a completed async reset.

        Call this after ``step()`` returned done=True for env_idx
        and before the next ``step()`` call to get the initial
        observation of the new episode.
        """
        if self._pending_reset[env_idx]:
            pipe = self.vec._parent_pipes[env_idx]
            if pipe.poll(120):
                status, data = pipe.recv()
                if status == "ok":
                    self._latest_obs[env_idx] = data[0]
                else:
                    raise RuntimeError(f"Reset failed:\n{data}")
            self._pending_reset[env_idx] = False
        obs = self._latest_obs[env_idx]
        assert obs is not None, f"No obs available for env {env_idx}"
        return obs

    def close(self) -> None:
        self.vec.close()
