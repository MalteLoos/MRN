"""
env.py — Gymnasium environment that steps Gazebo Harmonic + PX4 in lockstep.

Every call to ``env.step(action)`` advances the physics simulation by
exactly ``n_gz_steps`` Gazebo steps (configurable).  Because PX4 runs
in lockstep, the autopilot also advances exactly the same amount of
sim-time.

Model & sensors
---------------
The environment spawns/connects to a **x500_mono_cam** drone.

* **Observations** (``Dict`` space):
    * ``imu``         — ``(6,)``  float32  — averaged accel + gyro  (50 Hz)
    * ``camera``      — ``(H,W,3)`` uint8  — down-scaled mono-cam  (≈20 Hz, latest)
    * ``position``    — ``(3,)``  float32  — ENU position
    * ``velocity``    — ``(3,)``  float32  — ENU velocity
    * ``orientation`` — ``(4,)``  float32  — quaternion (w,x,y,z)

* **Actions** — ``(3,)`` float32  ∈ [-1, 1]:
    * ``action[0]`` → roll  angle  (scaled to ± max_roll)
    * ``action[1]`` → pitch angle  (scaled to ± max_pitch)
    * ``action[2]`` → thrust       (mapped to [0, 1])

  Sent to PX4 via ``VehicleAttitudeSetpoint`` over the DDS agent.

Architecture
------------
::

    env.step(action)
        │
        ├─ 1. _apply_action  → PX4 attitude setpoint via DDS
        ├─ 2. gz_ctrl.step_and_wait(n_gz_steps)   ← deterministic!
        ├─ 3. obs  = sensors.get_obs()             ← IMU averaged, cam latest
        ├─ 4. rew  = _compute_reward(obs, action)
        └─ 5. return obs, rew, terminated, truncated, info

Camera images, drone pose, and past trajectory are also published to
ROS 2 topics for live RViz visualisation.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from px4_gz_gym.gz_step import GzStepController  # noqa: E402
from px4_gz_gym.sensors import GzSensors  # noqa: E402
from px4_gz_gym import px4_cmd  # noqa: E402


class PX4GazeboEnv(gym.Env):
    """
    Gymnasium env wrapping PX4-SITL + Gazebo Harmonic with deterministic
    N-step physics advancing and attitude (roll / pitch / thrust) control.

    Parameters
    ----------
    world_name : str
        Gazebo world name (``<world name="…">`` in the SDF).
    model_name : str
        Spawned model name (PX4 appends ``_<instance>`` to the base
        model name, e.g. ``"x500_mono_cam_0"``).
    base_model : str
        PX4 model directory name under
        ``$PX4_HOME/Tools/simulation/gz/models/``.
    n_gz_steps : int
        How many Gazebo physics steps to advance per ``env.step()``.
        With ``step_size = 0.004 s`` and ``n_gz_steps = 5``, each
        env step is 0.02 s of sim-time (**50 Hz**).
    step_size : float
        Gazebo ``<max_step_size>`` in seconds.
    max_roll : float
        Maximum roll angle in **radians** mapped from action ∈ [-1, 1].
    max_pitch : float
        Maximum pitch angle in **radians** mapped from action ∈ [-1, 1].
    cam_obs_height / cam_obs_width : int
        Down-scaled camera resolution for the observation.
    max_episode_steps : int
        Episode length (in env-steps) before truncation.
    takeoff_alt : float
        Target altitude (ENU z, metres) for automatic takeoff at the
        beginning of every episode.
    enable_rviz : bool
        Whether to publish camera / pose / trajectory to ROS 2.
    render_mode : str | None
        Currently unused (Gazebo provides its own GUI).
    """

    metadata = {"render_modes": []}

    # ── defaults ───────────────────────────────────────────
    DEFAULT_MAX_ROLL = math.radians(30.0)  # ± 30°
    DEFAULT_MAX_PITCH = math.radians(30.0)  # ± 30°

    def __init__(
        self,
        world_name: str = "default",
        model_name: str = "x500_mono_cam_0",
        base_model: str = "x500_mono_cam",
        n_gz_steps: int = 5,
        step_size: float = 0.004,
        max_roll: float = DEFAULT_MAX_ROLL,
        max_pitch: float = DEFAULT_MAX_PITCH,
        cam_obs_height: int = 128,
        cam_obs_width: int = 128,
        max_episode_steps: int = 2_000,
        takeoff_alt: float = 2.5,
        enable_rviz: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ── config ──────────────────────────────────────────
        self.world_name = world_name
        self.model_name = model_name
        self.base_model = base_model
        self.n_gz_steps = n_gz_steps
        self.step_size = step_size
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.cam_obs_height = cam_obs_height
        self.cam_obs_width = cam_obs_width
        self.max_episode_steps = max_episode_steps
        self.takeoff_alt = takeoff_alt
        self.render_mode = render_mode

        # Path to the SDF used to (re-)spawn the drone.
        _px4_home = os.environ.get("PX4_HOME", "/opt/PX4-Autopilot")
        self.model_sdf_path: str = os.path.join(
            _px4_home,
            "Tools",
            "simulation",
            "gz",
            "models",
            base_model,
            "model.sdf",
        )

        # Sim-time delta per env.step()
        self.dt: float = n_gz_steps * step_size  # 5 × 0.004 = 0.02 s

        # ── action space — 3-D: roll, pitch, thrust ────────
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # ── observation space — Dict ───────────────────────
        self.observation_space = spaces.Dict(
            {
                "imu": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(6,),
                    dtype=np.float32,
                ),
                "camera": spaces.Box(
                    0,
                    255,
                    shape=(cam_obs_height, cam_obs_width, 3),
                    dtype=np.uint8,
                ),
                "position": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "velocity": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "orientation": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        # ── Gazebo transport ────────────────────────────────
        self._gz = GzStepController(world_name=world_name)
        self._sensors = GzSensors(
            world_name=world_name,
            model_name=model_name,
            cam_obs_height=cam_obs_height,
            cam_obs_width=cam_obs_width,
            enable_rviz=enable_rviz,
        )

        # ── bookkeeping ─────────────────────────────────────
        self._step_count: int = 0

    # ════════════════════════════════════════════════════════
    #  Gymnasium API
    # ════════════════════════════════════════════════════════

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # ── 1. Force-disarm & unpause ───────────────────────
        px4_cmd.force_disarm()
        self._gz.unpause()

        # ── 2. Wait for PX4 to confirm disarmed ────────────
        px4_cmd.wait_for_disarm(timeout=5.0)

        # ── 3. Pause & teleport to spawn pose ──────────────
        self._gz.pause()
        self._gz.set_model_pose(
            self.model_name,
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        self._gz.step_and_wait(50, step_size=self.step_size)

        # ── 3b. Restart EKF2 so all filter state is wiped ──
        #    ekf2 stop → start clears covariance, innovation
        #    history, and fault flags from the previous episode.
        px4_cmd.restart_ekf2()

        # ── 4. Clear sensor buffers for fresh episode ───────
        self._sensors.clear_imu_buffer()
        self._sensors.clear_trajectory()

        # ── 5. Unpause & wait for PX4 DDS connection ───────
        px4_cmd.clear_state()
        self._gz.unpause()
        px4_cmd.wait_for_connection(timeout=30.0)

        # ── 6. Stream offboard mode + position setpoints
        #    (PX4 requires OffboardControlMode at ≥ 2 Hz for
        #    ≥ 2 s before accepting the OFFBOARD mode switch) ─
        px4_cmd.stream_setpoints_and_offboard(
            n=100,
            rate_hz=50.0,
            z_enu=self.takeoff_alt,
        )

        # ── 7. Switch to OFFBOARD (position mode for takeoff)
        px4_cmd.switch_to_offboard(timeout=10.0)

        # ── 8. Arm & climb to takeoff altitude ──────────────
        px4_cmd.arm_and_takeoff(
            target_alt=self.takeoff_alt,
            timeout=20.0,
            get_altitude=lambda: float(self._sensors.get_flat_state()[2]),
        )

        # ── 9. Pause again for deterministic stepping ──────
        self._gz.pause()

        self._step_count = 0
        obs = self._sensors.get_obs()
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Apply *action* (roll, pitch, thrust), advance the sim by
        ``n_gz_steps`` physics steps, and return the resulting
        observation dict.
        """
        action = np.asarray(action, dtype=np.float32)

        # 1. Apply action → PX4 attitude setpoint ───────────
        #    publish_attitude_command() already sends both the
        #    OffboardControlMode heartbeat AND the attitude setpoint,
        #    so no separate heartbeat is needed here.
        self._apply_action(action)

        # 2. Step Gazebo ─────────────────────────────────────
        sim_time = self._gz.step_and_wait(
            n=self.n_gz_steps,
            step_size=self.step_size,
        )

        # 3. Observe (IMU averaged, camera latest) ──────────
        obs = self._sensors.get_obs()

        # 4. Reward / termination ────────────────────────────
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        info = self._build_info(sim_time=sim_time)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Disarm and unpause the world."""
        try:
            self._gz.unpause()
        except Exception:
            pass
        finally:
            self._gz.unpause()

    # ════════════════════════════════════════════════════════
    #  Action → PX4 attitude setpoint
    # ════════════════════════════════════════════════════════

    def _apply_action(self, action: np.ndarray) -> None:
        """Map normalised ``[-1, 1]^3`` action to a PX4 attitude
        setpoint sent over the DDS agent.

        ::

            action[0]  →  roll   angle   ∈  [-max_roll,  +max_roll]
            action[1]  →  pitch  angle   ∈  [-max_pitch, +max_pitch]
            action[2]  →  thrust         ∈  [0, 1]

        The current yaw is read from the sensor cache and held constant
        so the drone maintains its heading while the policy controls
        roll / pitch / thrust.
        """
        roll_cmd = float(action[0]) * self.max_roll
        pitch_cmd = float(action[1]) * self.max_pitch
        thrust_cmd = float(np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0))

        # Hold current yaw from latest sensor reading
        state = self._sensors.get_state_dict()
        q = state["orientation"]  # [w, x, y, z]
        current_yaw = math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )

        px4_cmd.publish_attitude_command(
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=current_yaw,
            thrust=thrust_cmd,
        )

    # ════════════════════════════════════════════════════════
    #  Reward / termination  (override in subclass)
    # ════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
    ) -> float:
        """Default: negative altitude-error for a simple hover task."""
        target_z = self.takeoff_alt
        z = float(obs["position"][2])
        reward = -abs(z - target_z)
        reward -= 0.01 * float(np.sum(action**2))
        return float(reward)

    def _is_terminated(self, obs: dict[str, np.ndarray]) -> bool:
        """Default: terminate if below ground or way too high."""
        z = float(obs["position"][2])
        if z < -0.5:
            return True
        if z > 100.0:
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

    @property
    def sim_time(self) -> float:
        """Current simulation time (seconds)."""
        return self._gz.sim_time
