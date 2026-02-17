"""
vec_env.py — In-process vectorised PX4+Gazebo environment for shared worlds.

All *N* PX4 SITL instances run in a **single Gazebo world**.
Stepping Gazebo advances ALL drones simultaneously, so we cannot
step envs independently.  Instead:

    apply_action_only() on ALL envs
            ↓
    ONE step_and_wait()   (shared GzStepController)
            ↓
    observe_only()  on ALL envs

Architecture
------------
::

    Training process
        │
        ├── GzStepController       (one world stepper)
        ├── PX4GazeboEnv[0]        instance_id=0, model=x500_mono_cam_0
        ├── PX4GazeboEnv[1]        instance_id=1, model=x500_mono_cam_1
        └── ...

All instances share a single ``ROS_DOMAIN_ID``.  Topic isolation
is via PX4's DDS namespace (``PX4_UXRCE_DDS_NS=px4_<i>``), so
each env's ``px4_cmd`` calls target the correct autopilot.

Reset strategy
~~~~~~~~~~~~~~
When **any** env signals done, we batch-reset **all** envs
together.  During the reset steps (EKF convergence, offboard
arming, etc.), non-resetting envs would also see Gazebo advance,
so we send them offboard heartbeats to maintain their hold.

For simplicity, the current implementation resets all envs at
the same time — episodes are the same length (truncated at
``max_episode_steps``), so done signals naturally align.

Usage
-----
::

    vec = SharedGzVecEnv.from_args(num_envs=3, args=args)
    obs_list = vec.reset_all()
    for step in range(rollout_steps):
        results = vec.step(actions)
    vec.close()
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np

from px4_gz_gym.gz_step import GzStepController
from px4_gz_gym.env import PX4GazeboEnv
from px4_gz_gym import px4_cmd


class SharedGzVecEnv:
    """In-process vectorised environment sharing one Gazebo world.

    All ``PX4GazeboEnv`` instances live in the main process and
    share a single ``GzStepController``.  Stepping is synchronised:
    actions are applied to ALL envs, then Gazebo is stepped ONCE,
    then observations are collected from ALL envs.

    Parameters
    ----------
    num_envs : int
        Number of parallel drone instances.
    env_kwargs : dict
        Base keyword arguments for ``PX4GazeboEnv``.  ``instance_id``,
        ``model_name``, and ``gz_step_controller`` are set per-env.
    world_name : str
        Gazebo world name (shared across all envs).
    base_model : str
        PX4 model base name (e.g. ``"x500_mono_cam"``).
        Model names become ``{base_model}_{i}``.
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: dict[str, Any],
        world_name: str = "tugbot_depot",
        base_model: str = "x500_mono_cam",
    ) -> None:
        self.num_envs = num_envs
        self._closed = False

        # ── Shared Gazebo stepper ──────────────────────────
        self._gz = GzStepController(world_name=world_name)

        # ── Per-instance env config ────────────────────────
        self._n_gz_steps: int = env_kwargs.get("n_gz_steps", 5)
        self._step_size: float = env_kwargs.get("step_size", 0.004)

        self.envs: list[PX4GazeboEnv] = []
        for i in range(num_envs):
            kw = dict(env_kwargs)
            kw["instance_id"] = i
            kw["model_name"] = f"{base_model}_{i}"
            kw["gz_step_controller"] = self._gz
            kw["world_name"] = world_name
            # Disable per-env RViz for instances > 0 to avoid
            # topic collisions and overhead.
            if i > 0:
                kw["enable_rviz"] = False
            env = PX4GazeboEnv(**kw)
            self.envs.append(env)

    @classmethod
    def from_args(
        cls,
        num_envs: int,
        args,
    ) -> "SharedGzVecEnv":
        """Create from parsed CLI args (same fields as train_hover)."""
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
            world_name=args.world,
        )

    # ════════════════════════════════════════════════════════
    #  Vectorised API
    # ════════════════════════════════════════════════════════

    def reset_all(self) -> list[dict[str, np.ndarray]]:
        """Reset all environments.

        Resets are serialised: each ``env.reset()`` performs its
        own sequence of sim-steps (force-disarm, teleport, EKF
        convergence, arming).  Because they share the same Gazebo
        world, stepping during one env's reset advances physics
        for ALL drones.

        Already-reset envs receive offboard attitude heartbeats
        during each subsequent reset so they don't lose OFFBOARD
        mode while PX4 times out waiting for heartbeats.
        """
        obs_list: list[dict[str, np.ndarray]] = []
        for idx, env in enumerate(self.envs):
            if idx == 0:
                # First env — no previously-reset envs to keep alive.
                obs, _info = env.reset()
            else:
                # Wrap the env's step function to also send heartbeats
                # to all already-reset envs (indices 0..idx-1).
                obs = self._reset_with_heartbeats(idx)
            obs_list.append(obs)
        return obs_list

    def _reset_with_heartbeats(
        self,
        env_idx: int,
    ) -> dict[str, np.ndarray]:
        """Reset env *env_idx* while keeping envs 0..env_idx-1 alive.

        Monkey-patches the shared GzStepController's ``step_and_wait``
        during the reset so that every sim-step batch also publishes
        offboard attitude heartbeats to the previously-reset envs.
        """
        env = self.envs[env_idx]
        original_step = env._gz.step_and_wait

        def _step_with_heartbeats(n: int, step_size: float = 0.004) -> float:
            # Send heartbeats to all already-reset envs before stepping
            for j in range(env_idx):
                px4_cmd.publish_offboard_attitude_heartbeat(
                    instance_id=j,
                )
            return original_step(n, step_size=step_size)

        env._gz.step_and_wait = _step_with_heartbeats  # type: ignore[assignment]
        try:
            obs, _info = env.reset()
        finally:
            env._gz.step_and_wait = original_step  # type: ignore[assignment]
        return obs

    def reset_one(
        self,
        env_idx: int,
    ) -> dict[str, np.ndarray]:
        """Reset a single environment.

        During the reset steps, other envs will also see the
        physics advance.  We send offboard heartbeats to the
        non-resetting envs so they don't lose offboard mode.

        Returns the reset observation for env_idx.
        """
        # Define a custom step function that also sends heartbeats
        # to the non-resetting envs during the reset sim-steps.
        def _step_with_heartbeats(n: int) -> float:
            """Step Gazebo n times while keeping other envs alive."""
            for _ in range(n):
                # Send offboard heartbeats to non-resetting envs
                for j, env in enumerate(self.envs):
                    if j == env_idx:
                        continue
                    px4_cmd.publish_offboard_attitude_heartbeat(
                        instance_id=j,
                    )
                self._gz.step_and_wait(
                    n=1, step_size=self._step_size,
                )
            return self._gz.sim_time

        # Temporarily monkey-patch the env's _gz.step_and_wait to
        # use our heartbeat-sending stepper during reset.
        env = self.envs[env_idx]
        original_step = env._gz.step_and_wait

        def _patched_step(n: int, step_size: float = 0.004) -> float:
            return _step_with_heartbeats(n)

        env._gz.step_and_wait = _patched_step  # type: ignore[assignment]
        try:
            obs, _info = env.reset()
        finally:
            env._gz.step_and_wait = original_step  # type: ignore[assignment]

        return obs

    def step(
        self,
        actions: list[np.ndarray],
        timeout: float = 30.0,
    ) -> list[tuple[dict, float, bool, bool, dict]]:
        """Step all environments with synchronised Gazebo stepping.

        1. Apply actions to all envs (no physics yet).
        2. Step Gazebo ONCE for all drones.
        3. Observe results from all envs.

        Parameters
        ----------
        actions : list of ndarray
            One action per env, length == num_envs.

        Returns
        -------
        List of (obs, reward, terminated, truncated, info) tuples.
        """
        assert len(actions) == self.num_envs

        # ── 1. Apply all actions ────────────────────────────
        for env, action in zip(self.envs, actions):
            env.apply_action_only(action)

        # ── 2. Step Gazebo ONCE ─────────────────────────────
        self._gz.step_and_wait(
            n=self._n_gz_steps,
            step_size=self._step_size,
        )

        # ── 3. Observe all envs ─────────────────────────────
        results: list[tuple[dict, float, bool, bool, dict]] = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.observe_only(action)
            results.append((obs, reward, terminated, truncated, info))

        return results

    def step_one(
        self,
        env_idx: int,
        action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Step a single environment (applies, steps, observes)."""
        env = self.envs[env_idx]
        return env.step(action)

    def close(self) -> None:
        """Shut down all environments."""
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_envs
