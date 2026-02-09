"""
env.py — Gymnasium environment that steps Gazebo Harmonic + PX4 in lockstep.

Every call to ``env.step(action)`` advances the physics simulation by
exactly ``n_gz_steps`` Gazebo steps (configurable).  Because PX4 runs
in lockstep, the autopilot also advances exactly the same amount of
sim-time.

Architecture
------------
::

    env.step(action)
        │
        ├─ 1. publish action  → PX4 (via /fmu/in/* or offboard setpoint)
        ├─ 2. gz_ctrl.step_and_wait(n_gz_steps)   ← deterministic!
        ├─ 3. obs  = sensors.get_obs()
        ├─ 4. rew  = _compute_reward(obs, action)
        └─ 5. return obs, rew, terminated, truncated, info

The environment talks to Gazebo **directly** over ``gz.transport``
(no ROS 2 required in the stepping hot-path).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from px4_gz_gym.gz_step import GzStepController  # noqa: E402
from px4_gz_gym.sensors import GzSensors  # noqa: E402


class PX4GazeboEnv(gym.Env):
    """
    Gymnasium env wrapping PX4-SITL + Gazebo Harmonic with deterministic
    N-step physics advancing.

    Parameters
    ----------
    world_name : str
        Gazebo world name (``<world name="…">`` in the SDF).
    model_name : str
        Spawned model name (default ``"x500_0"``; PX4 appends
        ``_<instance>`` to the base model name).
    n_gz_steps : int
        How many Gazebo physics steps to advance per ``env.step()``.
        With PX4's default ``max_step_size = 0.004 s`` and
        ``n_gz_steps = 25``, each env step is 0.1 s of sim-time (10 Hz).
    step_size : float
        Gazebo ``<max_step_size>`` in seconds.
    action_dim : int
        Dimensionality of the continuous action space.
        Default 4 corresponds to normalised attitude-rate + thrust
        (roll_rate, pitch_rate, yaw_rate, thrust).
    max_episode_steps : int
        Episode length (in env-steps) before truncation.
    render_mode : str | None
        Currently unused (Gazebo provides its own GUI).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_name: str = "default",
        model_name: str = "x500_0",
        n_gz_steps: int = 25,
        step_size: float = 0.004,
        action_dim: int = 4,
        max_episode_steps: int = 1_000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ── config ──────────────────────────────────────────
        self.world_name = world_name
        self.model_name = model_name
        self.n_gz_steps = n_gz_steps
        self.step_size = step_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Sim-time delta per env.step()
        self.dt: float = n_gz_steps * step_size

        # ── spaces ──────────────────────────────────────────
        # Actions: normalised ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Observations: [pos(3) vel(3) quat(4) ang_vel(3) lin_acc(3)]
        obs_high = np.full(GzSensors.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # ── Gazebo transport ────────────────────────────────
        self._gz = GzStepController(world_name=world_name)
        self._sensors = GzSensors(
            world_name=world_name,
            model_name=model_name,
        )

        # ── bookkeeping ─────────────────────────────────────
        self._step_count: int = 0
        self._prev_obs: np.ndarray = np.zeros(
            GzSensors.OBS_DIM,
            dtype=np.float32,
        )

    # ════════════════════════════════════════════════════════
    #  Gymnasium API
    # ════════════════════════════════════════════════════════

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Pause → reset → step once so sensors publish initial data
        self._gz.pause()
        self._gz.reset_world()
        self._gz.step_and_wait(1, step_size=self.step_size)

        self._step_count = 0
        obs = self._sensors.get_obs()
        self._prev_obs = obs
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply *action*, advance the sim by exactly ``n_gz_steps`` physics
        steps, and return the resulting observation.
        """
        action = np.asarray(action, dtype=np.float32)

        # 1. Apply action ────────────────────────────────────
        #    Override this method in a subclass to send the
        #    action to PX4 (e.g. via MAVSDK offboard setpoints,
        #    ROS 2 /fmu/in/vehicle_command, or direct actuator
        #    msgs).  The default is a no-op so the env can be
        #    tested standalone.
        self._apply_action(action)

        # 2. Step Gazebo ─────────────────────────────────────
        sim_time = self._gz.step_and_wait(
            n=self.n_gz_steps,
            step_size=self.step_size,
        )

        # 3. Observe ─────────────────────────────────────────
        obs = self._sensors.get_obs()

        # 4. Reward / termination ────────────────────────────
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        info = self._build_info(sim_time=sim_time)
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Unpause the world so Gazebo doesn't stay frozen."""
        self._gz.unpause()

    # ════════════════════════════════════════════════════════
    #  Override points  (subclass these)
    # ════════════════════════════════════════════════════════

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Send *action* to PX4 / Gazebo.

        Override in a subclass to map the normalised ``[-1, 1]`` action
        to actual offboard setpoints or actuator commands.  Example::

            # Attitude-rate + thrust via MAVSDK
            self.mavsdk.offboard.set_attitude_rate(
                roll_rate_deg_s  = action[0] * 180,
                pitch_rate_deg_s = action[1] * 180,
                yaw_rate_deg_s   = action[2] * 90,
                thrust           = (action[3] + 1) / 2,  # [0, 1]
            )

        The default implementation is a **no-op** — useful for testing
        the stepping pipeline without PX4 in the loop.
        """

    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """
        Compute a scalar reward.

        Default: negative altitude-error for a simple hover-at-1m task.
        Override for your own task.
        """
        target_z = 1.0  # 1 m altitude (ENU up)
        z = obs[2]
        reward = -abs(z - target_z)  # 0 when perfectly at target
        # Small penalty for large actions
        reward -= 0.01 * float(np.sum(action**2))
        return float(reward)

    def _is_terminated(self, obs: np.ndarray) -> bool:
        """
        Check hard termination conditions.

        Default: terminate if the drone is below ground or way too high.
        """
        z = obs[2]
        if z < -0.5:  # crashed below ground
            return True
        if z > 100.0:  # runaway
            return True
        return False

    # ════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════

    def _build_info(
        self,
        sim_time: float | None = None,
    ) -> dict[str, Any]:
        return {
            "sim_time": sim_time or self._gz.sim_time,
            "step_count": self._step_count,
            "dt": self.dt,
            "n_gz_steps": self.n_gz_steps,
            "state": self._sensors.get_state_dict(),
        }

    # ── properties ──────────────────────────────────────────

    @property
    def sim_time(self) -> float:
        """Current simulation time (seconds)."""
        return self._gz.sim_time
