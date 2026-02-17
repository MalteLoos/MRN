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

* **Actions** — ``(4,)`` float32  ∈ [-1, 1]:
    * ``action[0]`` → roll  angle  (scaled to ± max_roll)
    * ``action[1]`` → pitch angle  (scaled to ± max_pitch)
    * ``action[2]`` → yaw   rate   (scaled to ± max_yaw_rate)
    * ``action[3]`` → thrust       (mapped to [0, 1])

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
import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from px4_gz_gym.gz_step import GzStepController  # noqa: E402
from px4_gz_gym.sensors import GzSensors  # noqa: E402
from px4_gz_gym import px4_cmd  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Lightweight profiler for hot-path timing
# ═══════════════════════════════════════════════════════════════════════════


class StepProfiler:
    """Accumulate wall-clock timings and print a summary periodically.

    Usage::

        prof = StepProfiler(print_every=200)
        with prof.measure("gz_step"):
            ...
        prof.tick()          # call once per env.step()
    """

    def __init__(self, print_every: int = 200) -> None:
        self._print_every = print_every
        self._counts: dict[str, int] = {}
        self._totals: dict[str, float] = {}
        self._ticks = 0
        self._epoch_t0 = time.monotonic()

    class _Timer:
        __slots__ = ("_prof", "_label", "_t0")

        def __init__(self, prof: "StepProfiler", label: str) -> None:
            self._prof = prof
            self._label = label
            self._t0 = 0.0

        def __enter__(self) -> "StepProfiler._Timer":
            self._t0 = time.monotonic()
            return self

        def __exit__(self, *_: object) -> None:
            dt = time.monotonic() - self._t0
            p = self._prof
            p._totals[self._label] = p._totals.get(self._label, 0.0) + dt
            p._counts[self._label] = p._counts.get(self._label, 0) + 1

    def measure(self, label: str) -> _Timer:
        return self._Timer(self, label)

    def tick(self) -> None:
        self._ticks += 1
        if self._ticks % self._print_every == 0:
            self._print_and_reset()

    def force_print(self, header: str = "") -> None:
        self._print_and_reset(header=header)

    def _print_and_reset(self, header: str = "") -> None:
        wall = time.monotonic() - self._epoch_t0
        hdr = header or f"⏱  StepProfiler  ({self._ticks} ticks, {wall:.2f}s wall)"
        lines = [hdr, "-" * len(hdr)]
        for label in sorted(self._totals, key=lambda k: -self._totals[k]):
            total = self._totals[label]
            count = self._counts[label]
            avg_ms = total / count * 1000 if count else 0
            pct = total / wall * 100 if wall > 0 else 0
            lines.append(
                f"  {label:<28s}  "
                f"{total:8.3f}s  "
                f"{count:6d}×  "
                f"avg {avg_ms:7.2f}ms  "
                f"({pct:5.1f}%)"
            )
        accounted = sum(self._totals.values())
        other = wall - accounted
        if wall > 0:
            lines.append(
                f"  {'(unaccounted)':<28s}  "
                f"{other:8.3f}s  "
                f"{'':>6s}   "
                f"{'':>11s}  "
                f"({other / wall * 100:5.1f}%)"
            )
        print("\n".join(lines))
        print()
        self._totals.clear()
        self._counts.clear()
        self._epoch_t0 = time.monotonic()


class PX4GazeboEnv(gym.Env):
    """
    Gymnasium env wrapping PX4-SITL + Gazebo Harmonic with deterministic
    N-step physics advancing and attitude (roll / pitch / yaw-rate / thrust)
    control.

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
    DEFAULT_MAX_YAW_RATE = math.radians(60.0)  # ± 60°/s

    def __init__(
        self,
        world_name: str = "default",
        model_name: str = "x500_mono_cam_0",
        base_model: str = "x500_mono_cam",
        n_gz_steps: int = 5,
        step_size: float = 0.004,
        max_roll: float = DEFAULT_MAX_ROLL,
        max_pitch: float = DEFAULT_MAX_PITCH,
        max_yaw_rate: float = DEFAULT_MAX_YAW_RATE,
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
        self.max_yaw_rate = max_yaw_rate
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

        # ── action space — 4-D: roll, pitch, yaw_rate, thrust
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
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
        self.profiler = StepProfiler(print_every=200)

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

        _prof = self.profiler
        _reset_t0 = time.monotonic()

        # Helper: advance sim by n physics steps deterministically.
        # All reset waiting is done through this instead of
        # time.sleep(), turning ~130 s wall-clock into ~2-5 s.
        def _step(n: int) -> float:
            return self._gz.step_and_wait(n, step_size=self.step_size)

        # ── 1. Force-disarm (sim-stepped wait) ──────────────
        with _prof.measure("reset/1_disarm"):
            px4_cmd.force_disarm()
            px4_cmd.stepped_wait_for_disarm(
                _step,
                steps_per_iter=100,
                max_iters=25,
            )

        # ── 2. Teleport to spawn pose (paused, fast path) ──
        #    Uses the persistent helper subprocess for set_pose
        #    (< 1 ms per call vs ~260 ms CLI).  Slam pose twice
        #    while paused, then step physics to let the solver
        #    damp out residual velocity.  No time.sleep needed.
        with _prof.measure("reset/2_teleport"):
            _spawn_pos = (0.0, 0.0, 0.0)
            _spawn_ori = (1.0, 0.0, 0.0, 0.0)  # identity quaternion
            # Set pose while paused — two calls to be safe
            self._gz.set_model_pose(
                self.model_name,
                position=_spawn_pos,
                orientation=_spawn_ori,
            )
            # Step 25 physics steps (0.1s sim-time) to let
            # the physics solver damp residual velocities
            _step(25)
            # Slam pose again after damping to correct any drift
            self._gz.set_model_pose(
                self.model_name,
                position=_spawn_pos,
                orientation=_spawn_ori,
            )
            # Final damping: 25 more steps
            _step(25)

        # ── 3. Restart PX4 modules (EKF2, flight_mode_manager)
        with _prof.measure("reset/3_restart_px4"):
            px4_cmd.restart_px4()

        # Give PX4 modules sim-time to re-initialise.
        # 250 steps = 1 s sim-time — enough for EKF2 to converge.
        with _prof.measure("reset/4_ekf_converge"):
            _step(250)

        # ── 4. Clear sensor buffers for fresh episode ───────
        self._sensors.clear_imu_buffer()
        self._sensors.clear_trajectory()

        # ── 5. Wait for PX4 DDS connection (sim-stepped) ───
        with _prof.measure("reset/5_dds_connect"):
            px4_cmd.clear_state()
            px4_cmd.stepped_wait_for_connection(
                _step,
                steps_per_iter=100,
                max_iters=150,
            )

        # ── 6. Stream attitude-mode offboard heartbeats ────
        #    PX4 needs OffboardControlMode at >= 2 Hz for >= 1 s.
        #    1.5 s sim-time gives margin.
        with _prof.measure("reset/6_offboard_stream"):
            px4_cmd.stepped_stream_attitude(
                _step,
                sim_seconds=1.5,
                ticks_per_publish=self.n_gz_steps,
                step_size=self.step_size,
            )

        # ── 7. Switch to OFFBOARD + arm (attitude mode) ────
        with _prof.measure("reset/7_arm"):
            px4_cmd.stepped_offboard_arm(
                _step,
                ticks_per_publish=self.n_gz_steps,
                step_size=self.step_size,
                offboard_timeout_s=5.0,
                arm_timeout_s=5.0,
            )

        # ── 9. Pause again for deterministic stepping ──────
        self._gz.pause()

        _reset_wall = time.monotonic() - _reset_t0
        print(f"  ⏱  reset() total: {_reset_wall:.2f}s")

        self._step_count = 0
        obs = self._sensors.get_obs()
        info = self._build_info()
        return obs, info

    # ── debug helpers ─────────────────────────────────────────

    def publish_waypoints_relative(
        self,
        wp_flat: list[float] | np.ndarray,
    ) -> None:
        """Forward body-frame waypoint tensor to ROS 2 for debugging."""
        self._sensors.publish_waypoints_relative(wp_flat)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Apply *action* (roll, pitch, yaw_rate, thrust), advance the sim
        by ``n_gz_steps`` physics steps, and return the resulting
        observation dict.
        """
        action = np.asarray(action, dtype=np.float32)
        _prof = self.profiler

        # 1. Apply action → PX4 attitude setpoint ───────────
        with _prof.measure("step/1_apply_action"):
            self._apply_action(action)

        # 2. Step Gazebo ─────────────────────────────────────
        with _prof.measure("step/2_gz_step_wait"):
            sim_time = self._gz.step_and_wait(
                n=self.n_gz_steps,
                step_size=self.step_size,
            )

        # 3. Observe (IMU averaged, camera latest) ──────────
        with _prof.measure("step/3_get_obs"):
            obs = self._sensors.get_obs()

        # 4. Reward / termination ────────────────────────────
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        info = self._build_info(sim_time=sim_time)
        _prof.tick()
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
        """Map normalised ``[-1, 1]^4`` action to a PX4 attitude
        setpoint sent over the DDS agent.

        ::

            action[0]  →  roll   angle   ∈  [-max_roll,  +max_roll]
            action[1]  →  pitch  angle   ∈  [-max_pitch, +max_pitch]
            action[2]  →  yaw    rate    ∈  [-max_yaw_rate, +max_yaw_rate]
            action[3]  →  thrust         ∈  [0, 1]

        The current yaw is read from the sensor cache.  The policy's
        yaw-rate output is integrated over one env time-step (``dt``)
        and added to the current heading to produce the yaw setpoint.
        """
        roll_cmd = float(action[0]) * self.max_roll
        pitch_cmd = float(action[1]) * self.max_pitch
        yaw_rate_cmd = float(action[2]) * self.max_yaw_rate
        thrust_cmd = float(np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0))

        # Current yaw from latest sensor reading
        state = self._sensors.get_state_dict()
        q = state["orientation"]  # [w, x, y, z]
        current_yaw = math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )

        # Integrate yaw rate over one time-step
        yaw_cmd = current_yaw + yaw_rate_cmd * self.dt

        px4_cmd.publish_attitude_command(
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
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

    MAX_TILT_RAD: float = math.radians(45.0)

    def _is_terminated(self, obs: dict[str, np.ndarray]) -> bool:
        """Default: terminate if below ground, way too high, or
        tilted more than 60°."""
        z = float(obs["position"][2])
        if z < -0.5:
            return True
        if z > 100.0:
            return True
        # ── tilt check ──────────────────────────────────────
        # Compute the angle between the body z-axis and world up
        # from the orientation quaternion (w, x, y, z).
        q = obs["orientation"]
        # Body z-axis in world frame:  (2(xz + wy), 2(yz - wx), 1 - 2(x²+y²))
        # cos(tilt) = dot(body_z, world_up) = 1 - 2(x² + y²)
        cos_tilt = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
        if cos_tilt < math.cos(self.MAX_TILT_RAD):
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
