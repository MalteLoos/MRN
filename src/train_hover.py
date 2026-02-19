#!/usr/bin/env python3
"""
train_hover.py — Short-episode hover training for the dual-rate drone policy.

The drone is spawned at its default takeoff altitude and must hold position.
Each episode lasts 7 s of sim-time (350 env-steps at 50 Hz).

Waypoints are placed at the hover position so the policy learns to stay
in place before it is later finetuned for navigation.

Checkpoints are saved every ``--ckpt-every`` epochs.
Metrics are logged to **Weights & Biases in offline mode** so nothing is
uploaded during training — call ``wandb sync <run_dir>`` afterwards.

Usage
-----
    python3 src/train_hover.py                              # defaults
    python3 src/train_hover.py --epochs 500 --device cuda   # custom
    python3 src/train_hover.py --resume runs/hover/ckpt_100.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple  # noqa: F401 — Tuple used in annotations

import numpy as np
import torch

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    DroneAgent,
    DronePolicy,
    ModelConfig,
    RolloutBuffer,
    WaypointBuffer,
)
from profiling_log import get_profiling_logger as _get_profiling_logger, init_profiling_logger


# ═══════════════════════════════════════════════════════════════════════════
#  Hover Environment Wrapper
# ═══════════════════════════════════════════════════════════════════════════


class HoverEnvWrapper:
    """
    Thin wrapper around ``PX4GazeboEnv`` that:

    1. Maps the environment's obs dict to the keys the model expects.
    2. Sets up a trivial 2-waypoint route at the hover position.
    3. Enforces a 7 s episode length.
    4. Signals ``new_frame`` whenever the camera frame changes.
    5. Computes a hover-centric reward.

    Obs mapping
    ───────────
    env key        → model key
    ``camera``     → ``cam``
    ``imu``        → ``imu``
    ``position``   → ``drone_pos``
    ``orientation``→ ``drone_quat``
    ``velocity``   → ``velocity``   (kept for reward)
    """

    def __init__(
        self,
        env,
        hover_alt: float = 2.5,
        episode_seconds: float = 3.0,
        env_dt: float = 0.02,
    ):
        self.env = env
        self.hover_alt = hover_alt
        self.episode_seconds = episode_seconds
        self.max_steps = int(episode_seconds / env_dt)

        self._step_count = 0
        self._prev_cam_id: int | None = None  # track frame identity

    # ── obs remapping ──────────────────────────────────────────────────────

    @staticmethod
    def _remap_obs(
        obs: dict[str, np.ndarray],
        new_frame: bool,
    ) -> dict[str, Any]:
        """Remap env observation keys to model-expected keys."""
        return {
            "cam": obs["camera"],  # (H, W, 3) uint8
            "imu": obs["imu"],  # (6,) float32
            "drone_pos": obs["position"],  # (3,) float32  ENU
            "drone_quat": obs["orientation"],  # (4,) float32  (w,x,y,z)
            "velocity": obs["velocity"],  # (3,) float32  ENU
            "new_frame": new_frame,
        }

    def _detect_new_frame(self, obs: dict[str, np.ndarray]) -> bool:
        """Cheap identity check — cameras don't update every tick."""
        cam_id = id(obs["camera"])
        if cam_id != self._prev_cam_id:
            self._prev_cam_id = cam_id
            return True
        return False

    # ── gymnasium-like API ─────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        obs, _info = self.env.reset()
        self._step_count = 0
        self._prev_cam_id = None
        new_frame = self._detect_new_frame(obs)
        return self._remap_obs(obs, new_frame)

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> Tuple[dict[str, Any], float, bool, dict]:
        # Map the 4-D model action (roll_rate, pitch_rate, yaw_rate, thrust)
        # to the env's 4-D action (roll, pitch, yaw_rate, thrust).
        act_np = np.asarray(action, dtype=np.float32)
        env_action = np.array(
            [act_np[0], act_np[1], act_np[2], act_np[3]],
            dtype=np.float32,
        )

        obs, _env_reward, terminated, truncated, info = self.env.step(env_action)
        self._step_count += 1

        new_frame = self._detect_new_frame(obs)
        mapped = self._remap_obs(obs, new_frame)
        reward = self._compute_hover_reward(mapped, act_np)

        done = terminated or (self._step_count >= self.max_steps)
        return mapped, reward, done, info

    def publish_waypoints_relative(self, wp_flat) -> None:
        """Forward body-frame waypoint tensor to ROS 2 for debugging."""
        if hasattr(self.env, "publish_waypoints_relative"):
            self.env.publish_waypoints_relative(wp_flat)

    # ── hover-centric reward ───────────────────────────────────────────────

    def _compute_hover_reward(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
    ) -> float:
        """
        Reward for holding position at ``hover_alt`` with minimal tilt.

        Components (all negative penalties + alive bonus):
            -  altitude error   (linear, clamped at 3 m)
            -  horizontal drift (linear)
            -  velocity magnitude
            -  tilt penalty     (cosine-based, 0 upright → 2 inverted)
            -  "don't dig deeper" penalty for large tilt + same-sign cmd
            -  action magnitude (energy)
            -  low-thrust penalty
            +  alive bonus
        """
        pos = obs["drone_pos"]  # ENU
        vel = obs["velocity"]

        # altitude error — linear with soft clamp so it doesn't dominate
        raw_alt_err = abs(float(pos[2]) - self.hover_alt)
        alt_err = min(raw_alt_err, 5.0)  # cap at 3 m

        # horizontal drift from origin
        horiz_drift = math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2)

        # velocity penalty
        speed = float(np.linalg.norm(vel))

        # action penalty (exclude thrust)
        act_mag = float(np.sum(action[:3] ** 2))

        # thrust penalty — penalise thrust below 0.35 so the
        # policy learns to keep the motors spinning.  action[3]
        # is in [-1, 1]; actual thrust = (action[3]+1)/2.
        actual_thrust = (float(action[3]) + 1.0) / 2.0  # [0, 1]
        low_thrust_penalty = max(0.0, 0.4 - actual_thrust) * 5.0  # up to 1.75

        # tilt penalty — cos(tilt) = 1 - 2(qx² + qy²) = R_33,
        # tilt_penalty = 0 when upright, grows to 2 when inverted.
        q = obs["drone_quat"]  # (w, x, y, z)
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2)
        tilt_penalty = 1.0 - cos_tilt  # 0 upright, 2 inverted

        # "don't dig deeper" penalty — if already tilted > 30°,
        # punish attitude commands that push further in the same
        # direction.  Extract roll/pitch from the quaternion and
        # penalise when the commanded rate has the same sign as
        # the current lean.
        TILT_THRESH_RAD = math.radians(15.0)
        roll = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx**2 + qy**2),
        )
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))

        same_dir_penalty = 0.0
        if abs(roll) > TILT_THRESH_RAD:
            overlap = float(action[0]) * roll  # >0 means pushing further
            if overlap > 0:
                same_dir_penalty += overlap * 3.0
        if abs(pitch) > TILT_THRESH_RAD:
            overlap = float(action[1]) * pitch
            if overlap > 0:
                same_dir_penalty += overlap * 3.0

        reward = (
            -1.5 * alt_err
            - 0.5 * horiz_drift
            - 0.2 * speed
            - 0.05 * act_mag
            - low_thrust_penalty
            - 0.8 * tilt_penalty
            - same_dir_penalty
            + 0.5  # alive bonus
        )
        return reward


# ═══════════════════════════════════════════════════════════════════════════
#  Dummy Hover Environment (for testing without Gazebo)
# ═══════════════════════════════════════════════════════════════════════════


class DummyHoverEnv:
    """
    A simple physics-free dummy that imitates ``HoverEnvWrapper`` for
    offline testing of the training loop.

    Set ``--dummy`` on the command line to use this instead of the real sim.
    """

    # Gravity constant for simplified dynamics (m/s²).
    # PX4 thrust=0.5 (action[3]=0) should produce hover-neutral
    # acceleration, so gravity_acc must match the thrust scaling.
    #   thrust_force = (action[3]+1)/2 * max_thrust_acc
    #   hover: 0.5 * max_thrust_acc = gravity_acc → max_thrust_acc = 2*G
    # We use G=10 → max_thrust_acc=20 → acc_z = thrust*20 - 10.
    GRAVITY: float = 10.0
    MAX_THRUST_ACC: float = 20.0  # = 2 * GRAVITY

    def __init__(
        self,
        hover_alt: float = 2.5,
        episode_seconds: float = 3.0,
        env_dt: float = 0.02,
        cam_h: int = 128,
        cam_w: int = 128,
    ):
        self.hover_alt = hover_alt
        self.max_steps = int(episode_seconds / env_dt)
        self.cam_h = cam_h
        self.cam_w = cam_w
        self.dt = env_dt
        self._step_count = 0
        # Start at ground level — matches real PX4 env behaviour
        self._pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._frame_counter = 0

    def reset(self) -> dict[str, Any]:
        self._step_count = 0
        self._pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._frame_counter = 0
        return self._obs(new_frame=True)

    def step(self, action):
        act = np.asarray(action, dtype=np.float32)
        # Dynamics with gravity — action[3] maps to thrust:
        #   PX4 thrust = (action[3] + 1) / 2   in [0, 1]
        #   vertical acc = thrust * MAX_THRUST_ACC - GRAVITY
        #   At action[3]=0 → thrust=0.5 → acc=0 (hover-neutral)
        thrust_frac = (float(act[3]) + 1.0) / 2.0
        acc_z = thrust_frac * self.MAX_THRUST_ACC - self.GRAVITY
        acc = np.array(
            [act[0] * 2.0, act[1] * 2.0, acc_z], dtype=np.float32
        )
        self._vel += acc * self.dt
        self._vel *= 0.98  # drag
        self._pos += self._vel * self.dt
        # Ground clamp — can't go below z=0
        if self._pos[2] < 0.0:
            self._pos[2] = 0.0
            self._vel[2] = max(0.0, self._vel[2])
        self._step_count += 1

        # Camera updates at ~30 Hz ≈ every 1-2 env steps
        self._frame_counter += 1
        new_frame = self._frame_counter % 2 == 0

        obs = self._obs(new_frame)
        reward = self._reward(obs, act)
        done = self._step_count >= self.max_steps
        return obs, reward, done, {}

    def _obs(self, new_frame: bool) -> dict[str, Any]:
        return {
            "cam": np.random.randint(
                0, 255, (self.cam_h, self.cam_w, 3), dtype=np.uint8
            ),
            "imu": np.concatenate(
                [
                    np.random.randn(3).astype(np.float32) * 0.1,
                    np.random.randn(3).astype(np.float32) * 0.01,
                ]
            ),
            "drone_pos": self._pos.copy(),
            "drone_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "velocity": self._vel.copy(),
            "new_frame": new_frame,
        }

    def publish_waypoints_relative(self, wp_flat) -> None:
        """No-op — no ROS 2 in dummy mode."""
        pass

    def _reward(self, obs, action):
        pos = obs["drone_pos"]
        vel = obs["velocity"]

        # altitude error — linear with soft clamp (matches HoverEnvWrapper)
        raw_alt_err = abs(float(pos[2]) - self.hover_alt)
        alt_err = min(raw_alt_err, 5.0)

        # horizontal drift from origin
        horiz = math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2)

        # velocity penalty
        speed = float(np.linalg.norm(vel))

        # action penalty (exclude thrust)
        act_mag = float(np.sum(action[:3] ** 2))

        # low-thrust penalty
        thrust = float(action[3] + 1.0) / 2.0
        low_thrust_penalty = max(0.0, 0.4 - thrust) * 5.0

        # tilt penalty
        q = obs["drone_quat"]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2)
        tilt_penalty = 1.0 - cos_tilt

        # "don't dig deeper" penalty (matches HoverEnvWrapper)
        TILT_THRESH_RAD = math.radians(15.0)
        roll_angle = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx**2 + qy**2),
        )
        pitch_angle = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))

        same_dir_penalty = 0.0
        if abs(roll_angle) > TILT_THRESH_RAD:
            overlap = float(action[0]) * roll_angle
            if overlap > 0:
                same_dir_penalty += overlap * 3.0
        if abs(pitch_angle) > TILT_THRESH_RAD:
            overlap = float(action[1]) * pitch_angle
            if overlap > 0:
                same_dir_penalty += overlap * 3.0

        return (
            -1.5 * alt_err
            - 0.5 * horiz
            - 0.2 * speed
            - 0.05 * act_mag
            - low_thrust_penalty
            - 0.8 * tilt_penalty
            - same_dir_penalty
            + 0.5
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Expert (Trainer) Policy — PD controller for hover demonstrations
# ═══════════════════════════════════════════════════════════════════════════


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by the **inverse** of quaternion *q* (w, x, y, z).

    Transforms a world-frame vector into the body frame described by *q*.
    Equivalent to ``WaypointBuffer._quat_rotate_inverse`` but works on
    numpy arrays.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # conjugate (inverse for unit quat): negate t
    tx, ty, tz = -tx, -ty, -tz
    rx = vx + w * tx + (-y * tz - (-z) * ty)
    ry = vy + w * ty + (-z * tx - (-x) * tz)
    rz = vz + w * tz + (-x * ty - (-y) * tx)
    return np.array([rx, ry, rz])


class ExpertHoverPolicy:
    """
    PD controller that demonstrates stable takeoff-and-hover behaviour.

    The expert produces body-rate commands
    ``[roll_rate, pitch_rate, yaw_rate, thrust]``  (all in [-1, 1]) to:

    1. Climb / descend to the target ``hover_alt``.
    2. Drift back to the origin (0, 0) in the horizontal plane.
    3. Stay level (minimise tilt).
    4. Damp yaw rotation.

    Optional Gaussian noise (``noise_std``) is added to each action so
    the resulting demonstrations cover a wider region of state space,
    making the subsequent behavioural-cloning step more robust.
    """

    def __init__(
        self,
        hover_alt: float = 2.5,
        noise_std: float = 0.05,
    ):
        self.hover_alt = hover_alt
        self.noise_std = noise_std

        # Action mapping reminder:
        #   PX4 thrust = (action[3] + 1) / 2
        #   action[3] = 0.0  →  PX4 0.50  →  hover (gravity-neutral)
        #   action[3] = 1.0  →  PX4 1.00  →  full throttle
        #   action[3] = -1.0 →  PX4 0.00  →  zero throttle

        # ── Altitude: cascaded P→PI controller ──────────────────────────
        # The thrust mapping is:  PX4_thrust = (action[3]+1)/2
        #   action[3] =  0.0  → PX4 0.50 → hover-neutral
        #   action[3] = +1.0  → PX4 1.00 → full throttle
        #   action[3] = -1.0  → PX4 0.00 → zero throttle
        #
        # A single PID can't handle a 2.5 m takeoff error AND fine
        # hover: kp*2.5 ≈ full-throttle → overshoot → bounce.
        #
        # Cascade:
        #   OUTER  target_vz = clamp(kp_pos * alt_err, ±max_vz)
        #   INNER  thrust = base + kp_vz*(target_vz - vz) + ki_z*∫(alt_err)
        #
        # This naturally rate-limits the climb (max_vz m/s) so the
        # drone ascends gently and never overshoots.
        # ── Phase 1 (takeoff): fly up with steady thrust until close ────
        self.takeoff_thrust: float = 0.5   # constant thrust during climb
        #   PX4 = (0.35+1)/2 = 0.675 → gentle but definite climb
        self.takeoff_transition: float = 1.5  # switch to PID within this (m)

        # ── Phase 2 (hold): gentle PID for final approach + hover ────────
        self.kp_pos: float = 0.4     # outer: position error → velocity setpoint
        self.max_vz: float = 0.3     # outer: slow final approach (0.3 m/s max)
        self.kp_vz: float = 0.80     # inner: tight velocity tracking
        self.kd_vz: float = 0.30     # inner: strong damping — kills overshoot
        self.ki_z: float = 0.55      # inner: integral for steady-state hold
        self.thrust_base: float = 0.0   # hover-neutral baseline
        self.i_z_max: float = 0.35   # anti-windup clamp

        # ── Horizontal PD gains (body-frame) ─────────────────────────────
        self.kp_xy: float = 0.20     # proportional on horizontal error
        self.kd_xy: float = 0.15     # derivative (damp horizontal vel.)

        # ── Yaw PD hold (lock initial heading) ───────────────────────────
        # The env now sends action[2] as a yaw rate directly to PX4
        # via yaw_sp_move_rate (no integration).  action[2]=1.0 →
        # 60°/s yaw rate.  Gains map heading error (rad) + gyro
        # damping (rad/s) to the [-1,1] action.
        self.kp_yaw: float = 3.0     # heading error → rate command
        self.kd_yaw: float = 1.0     # damp gyro-z (prevents overshoot)

        # ── Controller state (reset each episode) ────────────────────────
        self._i_z: float = 0.0
        self._prev_vz: float | None = None
        self._phase: str = "takeoff"  # "takeoff" or "hold"
        self._target_yaw: float | None = None  # locked on first obs

    def reset(self) -> None:
        """Reset controller state at the start of each episode."""
        self._i_z = 0.0
        self._prev_vz = None
        self._phase = "takeoff"
        self._target_yaw = None

    # ------------------------------------------------------------------

    def compute_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Return ``[roll, pitch, yaw, thrust]`` in [-1, 1]."""
        pos = np.asarray(obs["drone_pos"], dtype=np.float64)
        vel = np.asarray(obs["velocity"], dtype=np.float64)
        quat = np.asarray(obs["drone_quat"], dtype=np.float64)  # w,x,y,z
        imu = np.asarray(obs["imu"], dtype=np.float64)

        # === Altitude — two-phase controller ============================
        alt_err = self.hover_alt - pos[2]  # >0 → need to climb
        vz = vel[2]                        # >0 → climbing

        # Derivative on vertical velocity (used in both phases)
        if self._prev_vz is not None:
            d_vz = (vz - self._prev_vz) / 0.02
        else:
            d_vz = 0.0
        self._prev_vz = vz

        # ── Phase transition: takeoff → hold ──────────────────────────
        if self._phase == "takeoff" and alt_err < self.takeoff_transition:
            self._phase = "hold"
            self._i_z = 0.0   # start integral fresh for hold phase

        # ── Phase: TAKEOFF (constant climb thrust) ────────────────────
        if self._phase == "takeoff":
            # Steady thrust to climb.  Damp vertical velocity so we
            # don't arrive at the transition boundary too fast.
            thrust = self.takeoff_thrust - 0.20 * d_vz

        # ── Phase: HOLD (gentle PID) ──────────────────────────────────
        else:
            # Outer loop: position error → clamped velocity setpoint
            target_vz = np.clip(
                self.kp_pos * alt_err, -self.max_vz, self.max_vz
            )
            vz_err = target_vz - vz

            # Compute thrust before integral update (back-calc anti-windup)
            thrust_raw = (
                self.thrust_base
                + self.kp_vz * vz_err
                - self.kd_vz * d_vz
                + self.ki_z * self._i_z
            )
            thrust_clamped = float(np.clip(thrust_raw, -1.0, 1.0))

            # Only accumulate integral when output is not saturated
            if abs(thrust_raw - thrust_clamped) < 0.01:
                self._i_z += alt_err * 0.02
                self._i_z = max(-self.i_z_max, min(self.i_z_max, self._i_z))

            thrust = thrust_clamped

        # ── Safety net: always prevent ground crash ───────────────────
        # If altitude is very low, guarantee enough thrust to climb.
        if pos[2] < 0.5:
            thrust = max(thrust, 0.30)
        elif pos[2] < self.hover_alt * 0.5 and vz < 0.0:
            thrust = max(thrust, 0.15)

        # === Horizontal PD (in body frame) ==============================
        # World-frame position error  (target = origin)
        err_world = np.array([-pos[0], -pos[1], 0.0])
        vel_xy_world = np.array([vel[0], vel[1], 0.0])

        err_body = _quat_rotate_inv(quat, err_world)
        vel_body = _quat_rotate_inv(quat, vel_xy_world)

        # Action sign conventions (ENU / FLU):
        #   positive action[0] (roll)  → tilt RIGHT → accelerate in -Y_body
        #   positive action[1] (pitch) → nose DOWN  → accelerate in +X_body
        #
        # So to move toward +Y_body error we need NEGATIVE roll (tilt left),
        # and to move toward +X_body error we need POSITIVE pitch (nose down).
        roll = -(self.kp_xy * err_body[1] - self.kd_xy * vel_body[1])
        pitch = self.kp_xy * err_body[0] - self.kd_xy * vel_body[0]

        # === Yaw PD hold (lock initial heading) ========================
        # Extract current yaw from ENU quaternion (w, x, y, z)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        current_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy**2 + qz**2),
        )
        # Lock target heading on the very first observation
        if self._target_yaw is None:
            self._target_yaw = current_yaw

        # Yaw error wrapped to [-π, π]
        yaw_err = self._target_yaw - current_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        yaw_rate_meas = imu[5]  # gyro-z (body-frame, ENU/FLU)
        yaw = self.kp_yaw * yaw_err - self.kd_yaw * yaw_rate_meas

        # === Add exploration noise & clip ===============================
        noise = np.random.randn(4) * self.noise_std
        action = np.array(
            [
                np.clip(roll + noise[0], -1.0, 1.0),
                np.clip(pitch + noise[1], -1.0, 1.0),
                np.clip(yaw + noise[2], -1.0, 1.0),
                np.clip(thrust + noise[3], -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return action


# ═══════════════════════════════════════════════════════════════════════════
#  Expert data collection + Behavioural Cloning
# ═══════════════════════════════════════════════════════════════════════════


def collect_expert_data(
    expert: ExpertHoverPolicy,
    env: HoverEnvWrapper | DummyHoverEnv,
    agent: DroneAgent,
    cfg: ModelConfig,
    num_steps: int,
) -> dict[str, torch.Tensor]:
    """
    Roll out the **expert** PD controller in *env* for *num_steps*
    transitions, recording the observations that the neural policy
    would see together with the expert's actions.

    The ``agent`` is used only for its vision encoder (to produce
    ``vis_feat`` vectors) and waypoint buffer — its actor/critic heads
    are **not** queried.

    Returns
    -------
    dict with keys ``imu``, ``waypoints``, ``vis_feat``, ``expert_actions``,
    ``rewards``, ``dones`` — each a ``(num_steps, *)`` tensor on ``agent.device``.
    """
    agent.reset_vision()
    expert.reset()
    obs = env.reset()

    hover_pos: Tuple[float, float, float] = (0.0, 0.0, env.hover_alt)
    agent.set_route([hover_pos, hover_pos])
    assert agent._buffer is not None

    device = agent.device

    imus: list[torch.Tensor] = []
    waypoints_list: list[torch.Tensor] = []
    vis_feats: list[torch.Tensor] = []
    flows: list[torch.Tensor] = []
    expert_actions: list[torch.Tensor] = []
    rewards_list: list[float] = []
    dones_list: list[bool] = []

    total_reward = 0.0
    episodes = 0

    for step_i in range(num_steps):
        # ── slow path: vision ─────────────────────────────────────────
        if obs.get("new_frame", False) and "cam" in obs:
            agent.update_vision(obs)

        # ── fast-path observation tensors ─────────────────────────────
        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        imu = imu.to(device)

        dp = [float(x) for x in obs["drone_pos"][:3]]
        drone_pos: Tuple[float, float, float] = (dp[0], dp[1], dp[2])
        dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
        drone_quat: Tuple[float, float, float, float] = (dq[0], dq[1], dq[2], dq[3])

        wp_tensor = agent._buffer.current_targets_tensor(
            drone_pos, drone_quat=drone_quat, device=device
        )

        vis_feat = agent.policy._vis_feat_cache
        if vis_feat is None:
            vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)

        # ── expert action ─────────────────────────────────────────────
        expert_act = expert.compute_action(obs)

        # ── store ─────────────────────────────────────────────────────
        imus.append(imu)
        waypoints_list.append(wp_tensor.squeeze(0))
        vis_feats.append(vis_feat.squeeze(0).detach())
        flow = agent.policy._flow_cache
        flows.append(
            flow.squeeze(0).detach().cpu()
            if flow is not None
            else torch.zeros(2, cfg.cam_height, cfg.cam_width)
        )
        expert_actions.append(
            torch.as_tensor(expert_act, dtype=torch.float32, device=device)
        )

        # ── step env with expert action ───────────────────────────────
        next_obs, reward, done, info = env.step(expert_act)
        total_reward += float(reward)
        rewards_list.append(float(reward))
        dones_list.append(done)

        if done:
            obs = env.reset()
            agent.reset_vision()
            expert.reset()
            agent.set_route([hover_pos, hover_pos])
            episodes += 1
        else:
            obs = next_obs

    avg_rew = total_reward / max(episodes, 1)
    print(
        f"  📋 Expert rollout: {num_steps} steps, {episodes} episodes, "
        f"mean reward {avg_rew:+.2f}"
    )

    return {
        "imu": torch.stack(imus),
        "waypoints": torch.stack(waypoints_list),
        "vis_feat": torch.stack(vis_feats),
        "flow": torch.stack(flows),  # (N, 2, H, W) on CPU
        "expert_actions": torch.stack(expert_actions),
        "rewards": torch.tensor(rewards_list, device=device),
        "dones": torch.tensor(dones_list, dtype=torch.float32, device=device),
    }


def bc_pretrain(
    agent: DroneAgent,
    expert_data: dict[str, torch.Tensor],
    optimiser: torch.optim.Optimizer,
    cfg: ModelConfig,
    bc_epochs: int,
    batch_size: int,
    scaler: torch.amp.GradScaler | None = None,  # type: ignore[attr-defined]
) -> dict[str, float]:
    """
    Behavioural-cloning pre-training: teach the actor head to reproduce
    the expert's actions given the same observations.

    Loss = MSE(actor_mean, expert_action)   (on the 4-D action vector)
         + value_coef · MSE(V(s), discounted_return)   (warm-start critic)

    The value targets are computed from the expert rollout rewards using
    simple per-step discounted returns.

    Returns
    -------
    dict  with ``bc_action_loss`` and ``bc_value_loss``.
    """
    import torch.nn as nn

    n = expert_data["imu"].shape[0]
    device = agent.device

    # ── Compute discounted returns for critic warm-start ──────────────
    rewards = expert_data["rewards"]
    dones = expert_data["dones"]
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t].item() + cfg.gamma * running * (1.0 - dones[t].item())
        returns[t] = running
    # normalise
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    total_act_loss = 0.0
    total_val_loss = 0.0
    num_batches = 0

    agent.policy.train()

    for epoch_i in range(bc_epochs):
        indices = torch.randperm(n, device=device)
        epoch_act_loss = 0.0
        epoch_val_loss = 0.0
        epoch_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            obs = {
                "imu": expert_data["imu"][idx],
                "waypoints": expert_data["waypoints"][idx],
                "vis_feat": expert_data["vis_feat"][idx],
            }
            if "flow" in expert_data:
                obs["flow"] = expert_data["flow"][idx.cpu()].to(device)
            target_actions = expert_data["expert_actions"][idx]
            target_returns = returns[idx]

            # forward through the fast encoder + heads
            dev = obs["imu"].device
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device_type=dev.type,
                dtype=torch.float16,
                enabled=(dev.type == "cuda"),
            ):
                fused, _ = agent.policy._encode_fast(obs)
                mu, _ = agent.policy.actor(fused)
                value = agent.policy.critic(fused).squeeze(-1)

            action_loss = torch.nn.functional.mse_loss(mu.float(), target_actions)
            value_loss = torch.nn.functional.mse_loss(value.float(), target_returns)
            loss = action_loss + cfg.value_coef * value_loss

            optimiser.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                optimiser.step()

            epoch_act_loss += action_loss.item()
            epoch_val_loss += value_loss.item()
            epoch_batches += 1

        total_act_loss += epoch_act_loss
        total_val_loss += epoch_val_loss
        num_batches += epoch_batches

        if (epoch_i + 1) % max(bc_epochs // 10, 1) == 0 or epoch_i == 0:
            avg_a = epoch_act_loss / max(epoch_batches, 1)
            avg_v = epoch_val_loss / max(epoch_batches, 1)
            print(
                f"  BC [{epoch_i + 1:4d}/{bc_epochs}]  "
                f"act_loss={avg_a:.6f}  val_loss={avg_v:.6f}"
            )

    nb = max(num_batches, 1)
    return {
        "bc_action_loss": total_act_loss / nb,
        "bc_value_loss": total_val_loss / nb,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════


def make_env(args):
    """Build the hover environment (real, dummy, or vectorised).

    Returns
    -------
    HoverEnvWrapper | DummyHoverEnv
        When ``--num-envs 1`` (default, single-process).
    SubprocVecEnv
        When ``--num-envs N`` with N > 1 (multi-process, each
        worker has its own PX4 + Gazebo stack).
    """
    if args.dummy:
        return DummyHoverEnv(
            hover_alt=args.hover_alt,
            episode_seconds=args.episode_secs,
            cam_h=args.cam_size,
            cam_w=args.cam_size,
        )

    num_envs = getattr(args, "num_envs", 1)

    if num_envs > 1:
        # Multi-process: each worker gets its own sim stack
        from px4_gz_gym.vec_env import SubprocVecEnv

        launch_script = getattr(args, "launch_script", "")
        return SubprocVecEnv.from_args(
            num_envs=num_envs,
            args=args,
            base_domain_id=10,
            launch_script=launch_script,
        )

    # Single-process: direct PX4GazeboEnv
    from px4_gz_gym.env import PX4GazeboEnv

    raw_env = PX4GazeboEnv(
        world_name=args.world,
        cam_obs_height=args.cam_size,
        cam_obs_width=args.cam_size,
        max_episode_steps=int(args.episode_secs / 0.02),
        takeoff_alt=args.hover_alt,
    )
    return HoverEnvWrapper(
        raw_env,
        hover_alt=args.hover_alt,
        episode_seconds=args.episode_secs,
    )


def collect_rollout(
    agent: DroneAgent,
    env: HoverEnvWrapper | DummyHoverEnv,
    buffer: RolloutBuffer,
    rollout_steps: int,
    cfg: ModelConfig,
    expert: ExpertHoverPolicy | None = None,
) -> dict[str, float]:
    """
    Collect ``rollout_steps`` transitions from a **single** env.

    Handles the dual-rate update: vision is only updated when the env
    signals ``new_frame``; the fast path runs every tick.

    If *expert* is provided, the expert's action for each observation
    is computed and stored in the buffer for the BC auxiliary loss.
    """
    buffer.reset()
    agent.reset_vision()

    # Timing accumulators for rollout profiling
    _t_reset = 0.0
    _t_policy = 0.0
    _t_env_step = 0.0
    _t_vision = 0.0
    _t_misc = 0.0
    _n_resets = 0
    _n_vision = 0
    _rollout_t0 = time.time()

    _t0 = time.time()
    obs = env.reset()
    _t_reset += time.time() - _t0
    _n_resets += 1
    if expert is not None:
        expert.reset()

    # Set a trivial hover route (2 waypoints at the target hover position)
    hover_pos: Tuple[float, float, float] = (0.0, 0.0, env.hover_alt)
    agent.set_route([hover_pos, hover_pos])
    assert agent._buffer is not None

    total_reward = 0.0
    episodes = 0
    device = agent.device

    for _ in range(rollout_steps):
        # ── slow path: update vision on new camera frame ──────────
        if obs.get("new_frame", False) and "cam" in obs:
            _tv0 = time.time()
            agent.update_vision(obs)
            _t_vision += time.time() - _tv0
            _n_vision += 1

        # ── prepare fast-path tensors ─────────────────────────────
        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        imu = imu.to(device)

        dp = [float(x) for x in obs["drone_pos"][:3]]
        drone_pos: Tuple[float, float, float] = (dp[0], dp[1], dp[2])
        dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
        drone_quat: Tuple[float, float, float, float] = (dq[0], dq[1], dq[2], dq[3])
        wp_tensor = agent._buffer.current_targets_tensor(
            drone_pos,
            drone_quat=drone_quat,
            device=device,
        )

        # Publish body-frame relative waypoints for RViz debugging
        env.publish_waypoints_relative(wp_tensor.squeeze(0).cpu().numpy())

        vis_feat = agent.policy._vis_feat_cache
        if vis_feat is None:
            vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)

        # Get cached RAFT flow map for FlowEncoder re-training
        flow = agent.policy._flow_cache  # (1, 2, H, W) or None

        # Snapshot GRU hidden state *before* act() updates it
        pre_hidden = agent.policy.get_hidden_state()

        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        _tp0 = time.time()
        action, log_prob, value = agent.policy.act(policy_obs)
        _t_policy += time.time() - _tp0
        action_sq = action.squeeze(0)

        # ── environment step ──────────────────────────────────────
        _te0 = time.time()
        next_obs, reward, done, info = env.step(action_sq.cpu())
        _t_env_step += time.time() - _te0

        # ── store transition ──────────────────────────────────────
        # Flatten hidden (1, 1, H) → (H,) for buffer storage
        hidden_flat = (
            pre_hidden.squeeze(0).squeeze(0)
            if pre_hidden is not None
            else None
        )
        # Compute expert action for BC auxiliary loss
        expert_act_t: torch.Tensor | None = None
        if expert is not None:
            ea = expert.compute_action(obs)  # (4,) numpy, no noise
            expert_act_t = torch.as_tensor(ea, dtype=torch.float32).to(device)
        buffer.store(
            imu=imu,
            waypoints=wp_tensor.squeeze(0),
            vis_feat=vis_feat.squeeze(0),
            action=action_sq,
            log_prob=log_prob.squeeze(0),
            reward=float(reward),
            value=value,
            done=done,
            hidden=hidden_flat,
            flow=flow.squeeze(0) if flow is not None else None,
            expert_action=expert_act_t,
        )
        total_reward += float(reward)

        if done:
            _tr0 = time.time()
            obs = env.reset()
            _t_reset += time.time() - _tr0
            _n_resets += 1
            agent.reset_vision()
            if expert is not None:
                expert.reset()
            hover_pos = (0.0, 0.0, env.hover_alt)
            agent.set_route([hover_pos, hover_pos])
            episodes += 1
        else:
            obs = next_obs

    # ── rollout profiling summary (written to profiling.log) ──────
    _rollout_wall = time.time() - _rollout_t0
    _t_other = _rollout_wall - _t_reset - _t_policy - _t_env_step - _t_vision
    _plog = _get_profiling_logger()
    _plog(f"\n⏱  Rollout profiling  ({rollout_steps} steps, {_rollout_wall:.2f}s wall)")
    _plog(
        f"  {'env.step()':<24s}  {_t_env_step:8.3f}s  ({_t_env_step/_rollout_wall*100:5.1f}%)  {rollout_steps}×  avg {_t_env_step/rollout_steps*1000:.2f}ms"
    )
    _plog(
        f"  {'env.reset()':<24s}  {_t_reset:8.3f}s  ({_t_reset/_rollout_wall*100:5.1f}%)  {_n_resets}×  avg {_t_reset/max(_n_resets,1)*1000:.1f}ms"
    )
    _plog(
        f"  {'policy.act()':<24s}  {_t_policy:8.3f}s  ({_t_policy/_rollout_wall*100:5.1f}%)  {rollout_steps}×  avg {_t_policy/rollout_steps*1000:.2f}ms"
    )
    _plog(
        f"  {'update_vision (RAFT)':<24s}  {_t_vision:8.3f}s  ({_t_vision/_rollout_wall*100:5.1f}%)  {_n_vision}×  avg {_t_vision/max(_n_vision,1)*1000:.2f}ms"
    )
    _plog(
        f"  {'(other/overhead)':<24s}  {_t_other:8.3f}s  ({_t_other/_rollout_wall*100:5.1f}%)"
    )
    # Also trigger the env-internal profiler (writes to same file)
    _inner_env = getattr(env, "env", None)
    _prof = getattr(_inner_env, "profiler", None)
    if _prof is not None:
        _prof.force_print(
            header=f"⏱  Env profiler  ({rollout_steps} steps, {_rollout_wall:.2f}s wall)"
        )

    # ── bootstrap last value ──────────────────────────────────────
    imu = obs["imu"]
    if not isinstance(imu, torch.Tensor):
        imu = torch.as_tensor(imu, dtype=torch.float32).to(device)
    dp = [float(x) for x in obs["drone_pos"][:3]]
    drone_pos = (dp[0], dp[1], dp[2])
    dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
    drone_quat = (dq[0], dq[1], dq[2], dq[3])
    assert agent._buffer is not None
    wp_tensor = agent._buffer.current_targets_tensor(
        drone_pos,
        drone_quat=drone_quat,
        device=device,
    )
    vis_feat = agent.policy._vis_feat_cache
    if vis_feat is None:
        vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)
    with torch.no_grad():
        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        _, _, last_value = agent.policy.act(policy_obs)
    buffer.finish(last_value)

    return {
        "total_reward": total_reward,
        "episodes": max(episodes, 1),
        "mean_reward": total_reward / max(episodes, 1),
    }


def collect_rollout_vec(
    agent: DroneAgent,
    vec_env,
    buffer: RolloutBuffer,
    rollout_steps: int,
    cfg: ModelConfig,
    hover_alt: float = 2.5,
    expert: ExpertHoverPolicy | None = None,
) -> dict[str, float]:
    """Collect transitions from **N parallel envs** round-robin.

    Each env steps independently; done envs are reset inline.
    Transitions from all envs are interleaved into a single
    ``RolloutBuffer`` in deterministic env-index order.

    The buffer must be sized for ``rollout_steps`` total transitions
    (spread across all envs).

    If *expert* is provided, the expert's action for each observation
    is computed and stored in the buffer for the BC auxiliary loss.
    Each env gets its own expert instance (so controller state is
    independent).

    Determinism
    ~~~~~~~~~~~
    * Each env performs its own deterministic sim-stepped reset.
    * Results are always processed in env-index order (0, 1, …, N-1)
      so the buffer contents are reproducible for the same seeds.
    """
    from px4_gz_gym.vec_env import SubprocVecEnv, AsyncResetVecEnv

    num_envs = vec_env.num_envs
    buffer.reset()
    agent.reset_vision()
    device = agent.device

    # ── Initial reset of all envs ────────────────────────────
    if isinstance(vec_env, AsyncResetVecEnv):
        obs_list = vec_env.reset_all()
    else:
        obs_list = vec_env.reset_all()

    # Wrap raw obs in HoverEnvWrapper-style remap
    def _remap(obs_dict: dict) -> dict:
        """Raw PX4GazeboEnv obs → model-expected keys."""
        return {
            "cam": obs_dict["camera"],
            "imu": obs_dict["imu"],
            "drone_pos": obs_dict["position"],
            "drone_quat": obs_dict["orientation"],
            "velocity": obs_dict["velocity"],
            "new_frame": True,  # first frame after reset
        }

    obs_per_env = [_remap(o) for o in obs_list]

    # Per-env expert instances for BC auxiliary actions
    experts_per_env: list[ExpertHoverPolicy | None] = []
    if expert is not None:
        for _ in range(num_envs):
            e = ExpertHoverPolicy(hover_alt=hover_alt, noise_std=0.0)
            e.reset()
            experts_per_env.append(e)
    else:
        experts_per_env = [None] * num_envs

    # Set hover route for agent — target is directly above spawn
    hover_pos: Tuple[float, float, float] = (0.0, 0.0, hover_alt)
    agent.set_route([hover_pos, hover_pos])
    assert agent._buffer is not None

    total_reward = 0.0
    episodes = 0
    steps_collected = 0

    while steps_collected < rollout_steps:
        # ── Compute actions for all envs ─────────────────────
        actions = []
        log_probs = []
        values = []
        imus = []
        wp_tensors = []
        vis_feats = []
        flows = []
        hidden_flats = []

        for i in range(num_envs):
            if steps_collected + i >= rollout_steps:
                break
            obs = obs_per_env[i]

            # Slow path: vision
            if obs.get("new_frame", False) and "cam" in obs:
                agent.update_vision(obs)

            imu = obs["imu"]
            if not isinstance(imu, torch.Tensor):
                imu = torch.as_tensor(imu, dtype=torch.float32)
            imu = imu.to(device)

            dp = [float(x) for x in obs["drone_pos"][:3]]
            drone_pos = (dp[0], dp[1], dp[2])
            dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
            drone_quat = (dq[0], dq[1], dq[2], dq[3])
            wp_tensor = agent._buffer.current_targets_tensor(
                drone_pos,
                drone_quat=drone_quat,
                device=device,
            )

            vis_feat = agent.policy._vis_feat_cache
            if vis_feat is None:
                vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)

            # Get cached RAFT flow map for FlowEncoder re-training
            flow = agent.policy._flow_cache  # (1, 2, H, W) or None

            # Snapshot GRU hidden state *before* act() updates it
            pre_hidden = agent.policy.get_hidden_state()
            hidden_flat = (
                pre_hidden.squeeze(0).squeeze(0)
                if pre_hidden is not None
                else None
            )

            policy_obs = {
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
                "vis_feat": vis_feat,
            }
            action, log_prob, value = agent.policy.act(policy_obs)
            action_sq = action.squeeze(0)

            actions.append(action_sq.cpu().numpy())
            log_probs.append(log_prob.squeeze(0))
            values.append(value)
            imus.append(imu)
            wp_tensors.append(wp_tensor.squeeze(0))
            vis_feats.append(vis_feat.squeeze(0))
            flows.append(flow.squeeze(0) if flow is not None else None)
            hidden_flats.append(hidden_flat)

        n_active = len(actions)
        if n_active == 0:
            break

        # ── Step all active envs in parallel ─────────────────
        if isinstance(vec_env, AsyncResetVecEnv):
            step_results = vec_env.step(
                [np.asarray(a, dtype=np.float32) for a in actions]
                + [np.zeros(4, dtype=np.float32)] * (num_envs - n_active)
            )
        else:
            step_results = vec_env.step(
                [np.asarray(a, dtype=np.float32) for a in actions]
                + [np.zeros(4, dtype=np.float32)] * (num_envs - n_active)
            )

        # ── Store transitions (deterministic env-index order) ─
        for i in range(n_active):
            if steps_collected >= rollout_steps:
                break

            if isinstance(vec_env, AsyncResetVecEnv):
                obs_raw, reward, done, info = step_results[i]
                next_obs_raw = obs_raw
            else:
                obs_raw, reward, term, trunc, info = step_results[i]  # type: ignore[misc]
                done = bool(term or trunc)
                next_obs_raw = obs_raw

            # Compute expert action for BC auxiliary loss
            expert_act_t: torch.Tensor | None = None
            ei = experts_per_env[i]
            if ei is not None:
                ea = ei.compute_action(obs_per_env[i])
                expert_act_t = torch.as_tensor(ea, dtype=torch.float32).to(device)

            buffer.store(
                imu=imus[i],
                waypoints=wp_tensors[i],
                vis_feat=vis_feats[i],
                action=torch.as_tensor(actions[i], dtype=torch.float32).to(device),
                log_prob=log_probs[i],
                reward=float(reward),
                value=values[i],
                done=done,
                hidden=hidden_flats[i],
                flow=flows[i],
                expert_action=expert_act_t,
            )
            total_reward += float(reward)
            steps_collected += 1

            if done:
                if isinstance(vec_env, AsyncResetVecEnv):
                    # Reset obs will be collected on next step
                    reset_obs = vec_env.get_reset_obs(i)
                    obs_per_env[i] = _remap(reset_obs)
                else:
                    reset_obs = vec_env.reset_one(i)
                    obs_per_env[i] = _remap(reset_obs)
                agent.reset_vision()
                hover_pos = (0.0, 0.0, hover_alt)
                agent.set_route([hover_pos, hover_pos])
                ei_done = experts_per_env[i]
                if ei_done is not None:
                    ei_done.reset()
                episodes += 1
            else:
                obs_per_env[i] = _remap(next_obs_raw)

    # ── Bootstrap last value ─────────────────────────────────
    obs = obs_per_env[0]
    imu = obs["imu"]
    if not isinstance(imu, torch.Tensor):
        imu = torch.as_tensor(imu, dtype=torch.float32).to(device)
    dp = [float(x) for x in obs["drone_pos"][:3]]
    drone_pos = (dp[0], dp[1], dp[2])
    dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
    drone_quat = (dq[0], dq[1], dq[2], dq[3])
    assert agent._buffer is not None
    wp_tensor = agent._buffer.current_targets_tensor(
        drone_pos,
        drone_quat=drone_quat,
        device=device,
    )
    vis_feat = agent.policy._vis_feat_cache
    if vis_feat is None:
        vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)
    with torch.no_grad():
        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        _, _, last_value = agent.policy.act(policy_obs)
    buffer.finish(last_value)

    return {
        "total_reward": total_reward,
        "episodes": max(episodes, 1),
        "mean_reward": total_reward / max(episodes, 1),
    }


def ppo_update(
    agent: DroneAgent,
    buffer: RolloutBuffer,
    optimiser: torch.optim.Optimizer,
    cfg: ModelConfig,
    ppo_epochs: int,
    batch_size: int,
    scaler: torch.amp.GradScaler | None = None,  # type: ignore[attr-defined]
    bc_aux_weight: float = 0.0,
) -> dict[str, float]:
    """Run ``ppo_epochs`` of clipped PPO over the filled buffer.

    Parameters
    ----------
    bc_aux_weight : float
        Coefficient for the BC auxiliary loss that anchors the actor's
        mean to the **expert PD controller** actions stored in the
        buffer.  Should decay from ~1.0 → 0.0 over the first portion
        of training.
    """
    import torch.nn as nn

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_bc_aux = 0.0
    num_batches = 0

    for _ in range(ppo_epochs):
        for batch in buffer.sample_batches(batch_size):
            obs = batch["obs"]
            actions = batch["actions"]
            expert_actions = batch["expert_actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            new_log_probs, entropy, values, actor_mu = agent.policy.evaluate_actions(obs, actions)
            values = values.squeeze(-1)

            # policy loss
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = torch.nn.functional.mse_loss(values, returns)

            # ── BC auxiliary loss (decaying anchor to expert PD) ──────
            bc_aux_loss = torch.tensor(0.0, device=values.device)
            if bc_aux_weight > 0.0:
                bc_aux_loss = torch.nn.functional.mse_loss(
                    actor_mu, expert_actions.detach()
                )

            # combined
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy.mean()
                + bc_aux_weight * bc_aux_loss
            )

            optimiser.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                optimiser.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl
            total_bc_aux += bc_aux_loss.item()
            num_batches += 1

    n = max(num_batches, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
        "approx_kl": total_kl / n,
        "bc_aux_loss": total_bc_aux / n,
    }


def save_checkpoint(
    agent: DroneAgent,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    stats: dict,
    path: str,
) -> None:
    """Save model + optimiser state + metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "policy_state_dict": agent.policy.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "stats": stats,
        },
        path,
    )
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(
    agent: DroneAgent,
    optimiser: torch.optim.Optimizer,
    path: str,
) -> int:
    """Load checkpoint and return the epoch number."""
    ckpt = torch.load(path, map_location=agent.device)
    agent.policy.load_state_dict(ckpt["policy_state_dict"])
    optimiser.load_state_dict(ckpt["optimiser_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"  ✅ Resumed from {path}  (epoch {epoch})")
    return epoch


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the drone policy to hover (7 s episodes).",
    )
    # ── training ────────────────────────────────────────────────────────
    p.add_argument(
        "--epochs", type=int, default=200, help="Number of PPO training epochs."
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=32768,
        help="Transitions per rollout (~27 episodes of 150 steps).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for PPO updates."
    )
    p.add_argument(
        "--ppo-epochs", type=int, default=8, help="Gradient passes per rollout."
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")

    # ── environment ─────────────────────────────────────────────────────
    p.add_argument(
        "--world",
        type=str,
        default="tugbot_depot",
        help="Gazebo world name (must match SDF in PX4 worlds dir).",
    )
    p.add_argument(
        "--hover-alt", type=float, default=2.5, help="Target hover altitude (m)."
    )
    p.add_argument(
        "--episode-secs", type=float, default=5.0, help="Episode length in sim-seconds."
    )
    p.add_argument(
        "--cam-size",
        type=int,
        default=128,
        help="Camera observation H=W (must be ≥ 128 for RAFT).",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Use a physics-free dummy env for loop testing.",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel PX4+Gazebo environments (each gets its own sim stack).",
    )
    p.add_argument(
        "--launch-script",
        type=str,
        default="",
        help="Path to launch_sim.sh for auto-launching per-worker sim stacks. "
        "Leave empty if sims are already running.",
    )

    # ── checkpointing & logging ─────────────────────────────────────────
    p.add_argument(
        "--run-dir",
        type=str,
        default="runs/hover",
        help="Directory for checkpoints & wandb logs.",
    )
    p.add_argument(
        "--ckpt-every", type=int, default=10, help="Save checkpoint every N epochs."
    )
    p.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from."
    )
    p.add_argument(
        "--wandb-project", type=str, default="drone-hover", help="Wandb project name."
    )

    # ── behavioural cloning (expert pre-training) ───────────────────────
    p.add_argument(
        "--bc-epochs",
        type=int,
        default=50,
        help="Gradient epochs for the BC pre-training phase (0 = skip).",
    )
    p.add_argument(
        "--bc-rollout-steps",
        type=int,
        default=32768,
        help="Transitions to collect from the expert PD controller.",
    )
    p.add_argument(
        "--bc-noise-std",
        type=float,
        default=0.05,
        help="Gaussian noise σ added to expert actions for diversity.",
    )
    p.add_argument(
        "--skip-bc",
        action="store_true",
        help="Skip the BC phase entirely (e.g. when resuming).",
    )
    p.add_argument(
        "--bc-decay-epochs",
        type=int,
        default=60,
        help="Number of PPO epochs over which the BC auxiliary loss "
        "decays linearly from 1→0 (smooth BC→PPO transition).",
    )

    # ── hardware ────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda", help="'cpu' or 'cuda'.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    init_profiling_logger(args.run_dir)

    # ── Weights & Biases (offline) ──────────────────────────────────────
    import wandb

    os.environ["WANDB_MODE"] = "online"
    wandb.init(
        project=args.wandb_project,
        dir=args.run_dir,
        config=vars(args),
        name=f"hover_{time.strftime('%Y%m%d_%H%M%S')}",
        save_code=True,
    )

    # ── Model config ────────────────────────────────────────────────────
    cfg = ModelConfig(
        cam_height=args.cam_size,
        cam_width=args.cam_size,
        lr=args.lr,
    )
    wandb.config.update(
        {k: v for k, v in cfg.__dict__.items()},
        allow_val_change=True,
    )

    # ── Agent + optimiser ───────────────────────────────────────────────
    agent = DroneAgent(cfg=cfg, device=args.device)
    optimiser = torch.optim.Adam(
        agent.policy.parameters(),
        lr=cfg.lr,
    )

    # Mixed-precision GradScaler for Blackwell / SM 12.0 GPUs where fp32
    # cuBLAS is broken.  On CPU this is simply None.
    scaler: torch.amp.GradScaler | None = None  # type: ignore[attr-defined]
    if args.device.startswith("cuda"):
        scaler = torch.amp.GradScaler()  # type: ignore[attr-defined]

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(agent, optimiser, args.resume)

    # ── Rollout buffer ──────────────────────────────────────────────────
    wp_dim = cfg.waypoint_dim * cfg.num_waypoints
    buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        imu_dim=cfg.imu_dim,
        wp_dim=wp_dim,
        vis_feat_dim=cfg.flow_feature_dim,
        action_dim=cfg.action_dim,
        device=args.device,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        gru_hidden_dim=agent.policy.gru_hidden_dim,
        flow_height=cfg.cam_height,
        flow_width=cfg.cam_width,
    )

    # ── Environment ─────────────────────────────────────────────────────
    env = make_env(args)
    num_envs = getattr(args, "num_envs", 1)
    use_vec_env = num_envs > 1 and not args.dummy

    # ── Parameter summary ───────────────────────────────────────────────
    total_p = sum(p.numel() for p in agent.policy.parameters())
    train_p = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Hover Training — Dual-Rate Policy          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Device:       {args.device:<38s} ║")
    print(f"║  Total params: {total_p:<38,d} ║")
    print(f"║  Trainable:    {train_p:<38,d} ║")
    print(f"║  Frozen (RAFT):{total_p - train_p:<38,d} ║")
    print(f"║  Epochs:       {args.epochs:<38d} ║")
    print(f"║  Rollout steps:{args.rollout_steps:<38d} ║")
    print(f"║  Num envs:     {num_envs:<38d} ║")
    print(f"║  Episode secs: {args.episode_secs:<38.1f} ║")
    print(f"║  Hover alt:    {args.hover_alt:<38.1f} ║")
    print(f"║  Dummy env:    {str(args.dummy):<38s} ║")
    print(f"║  Run dir:      {args.run_dir:<38s} ║")
    print(f"║  BC epochs:    {args.bc_epochs:<38d} ║")
    print(f"║  BC rollout:   {args.bc_rollout_steps:<38d} ║")
    print(f"║  BC decay:     {args.bc_decay_epochs:<38d} ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Behavioural-cloning pre-training (expert → policy) ──────────────
    run_bc = args.bc_epochs > 0 and not args.skip_bc and not args.resume
    if run_bc:
        print("┌──────────────────────────────────────────────────────┐")
        print("│  Phase 1 — Behavioural Cloning from Expert Trainer   │")
        print("└──────────────────────────────────────────────────────┘")
        expert = ExpertHoverPolicy(
            hover_alt=args.hover_alt,
            noise_std=args.bc_noise_std,
        )

        bc_t0 = time.time()
        expert_data = collect_expert_data(
            expert=expert,
            env=env,  # type: ignore[arg-type]
            agent=agent,
            cfg=cfg,
            num_steps=args.bc_rollout_steps,
        )

        bc_stats = bc_pretrain(
            agent=agent,
            expert_data=expert_data,
            optimiser=optimiser,
            cfg=cfg,
            bc_epochs=args.bc_epochs,
            batch_size=args.batch_size,
            scaler=scaler,
        )
        bc_elapsed = time.time() - bc_t0

        wandb.log(
            {
                "bc/action_loss": bc_stats["bc_action_loss"],
                "bc/value_loss": bc_stats["bc_value_loss"],
                "bc/elapsed_s": bc_elapsed,
            },
            step=-1,  # logged before PPO epoch 0
        )

        # Save a post-BC checkpoint so we can resume without re-doing BC
        bc_ckpt = os.path.join(args.run_dir, "post_bc.pt")
        save_checkpoint(agent, optimiser, -1, bc_stats, bc_ckpt)
        print(
            f"  ✅ BC phase done in {bc_elapsed:.1f}s — "
            f"act_loss={bc_stats['bc_action_loss']:.6f}\n"
        )

        del expert_data  # free memory

    # ── Expert PD controller for decaying BC auxiliary loss ───────────
    # Instead of a frozen neural policy copy, anchor the actor directly
    # to the expert PD controller actions stored in the rollout buffer.
    bc_expert: ExpertHoverPolicy | None = None
    if args.bc_decay_epochs > 0:
        bc_expert = ExpertHoverPolicy(hover_alt=args.hover_alt, noise_std=0.0)
        print(
            f"  🎯 Expert PD anchor for BC auxiliary loss "
            f"(decay over {args.bc_decay_epochs} PPO epochs)\n"
        )
    elif args.bc_epochs > 0 and (args.skip_bc or args.resume):
        bc_expert = ExpertHoverPolicy(hover_alt=args.hover_alt, noise_std=0.0)
        print("  ⏭  Skipping BC phase (--skip-bc or --resume), expert anchor active.\n")
    else:
        bc_expert = None  # BC disabled (--bc-epochs 0)

    print("┌──────────────────────────────────────────────────────┐")
    print("│  Phase 2 — PPO Self-Supervised Learning              │")
    print("└──────────────────────────────────────────────────────┘")

    # ── Training loop ───────────────────────────────────────────────────
    best_mean_reward = -float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        # 1. Collect rollout
        if use_vec_env:
            rollout_stats = collect_rollout_vec(
                agent,
                env,  # type: ignore[arg-type]
                buffer,
                args.rollout_steps,
                cfg,
                hover_alt=args.hover_alt,
                expert=bc_expert,
            )
        else:
            rollout_stats = collect_rollout(
                agent,
                env,  # type: ignore[arg-type]
                buffer,
                args.rollout_steps,
                cfg,
                expert=bc_expert,
            )

        # 2. PPO update — with decaying BC auxiliary loss
        ppo_epoch_idx = epoch - start_epoch  # 0-based within this run
        if bc_expert is not None and ppo_epoch_idx < args.bc_decay_epochs:
            bc_w = 1.0 - ppo_epoch_idx / args.bc_decay_epochs
        else:
            bc_w = 0.0

        update_stats = ppo_update(
            agent,
            buffer,
            optimiser,
            cfg,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            scaler=scaler,
            bc_aux_weight=bc_w,
        )

        elapsed = time.time() - t0
        mean_rew = rollout_stats["mean_reward"]

        # 3. Logging
        log_dict = {
            "epoch": epoch,
            "rollout/total_reward": rollout_stats["total_reward"],
            "rollout/mean_reward": mean_rew,
            "rollout/episodes": rollout_stats["episodes"],
            "train/policy_loss": update_stats["policy_loss"],
            "train/value_loss": update_stats["value_loss"],
            "train/entropy": update_stats["entropy"],
            "train/approx_kl": update_stats["approx_kl"],
            "train/bc_aux_loss": update_stats["bc_aux_loss"],
            "train/bc_aux_weight": bc_w,
            "perf/epoch_time_s": elapsed,
            "perf/fps": args.rollout_steps / elapsed,
        }
        wandb.log(log_dict, step=epoch)

        # Console
        bc_tag = f"  bc_w={bc_w:.2f}" if bc_w > 0 else ""
        print(
            f"[{epoch:4d}]  "
            f"reward={mean_rew:+7.2f}  "
            f"π_loss={update_stats['policy_loss']:.4f}  "
            f"v_loss={update_stats['value_loss']:.4f}  "
            f"ent={update_stats['entropy']:.3f}  "
            f"kl={update_stats['approx_kl']:.4f}"
            f"{bc_tag}  "
            f"({elapsed:.1f}s)"
        )

        # 4. Checkpointing
        is_best = mean_rew > best_mean_reward
        if is_best:
            best_mean_reward = mean_rew

        if (epoch + 1) % args.ckpt_every == 0 or is_best:
            ckpt_path = os.path.join(args.run_dir, f"ckpt_{epoch:04d}.pt")
            save_checkpoint(agent, optimiser, epoch, log_dict, ckpt_path)

        if is_best:
            best_path = os.path.join(args.run_dir, "best.pt")
            save_checkpoint(agent, optimiser, epoch, log_dict, best_path)

    # ── Final save ──────────────────────────────────────────────────────
    final_path = os.path.join(args.run_dir, "final.pt")
    save_checkpoint(agent, optimiser, start_epoch + args.epochs - 1, {}, final_path)

    # ── Cleanup ──────────────────────────────────────────────────────────
    if use_vec_env and hasattr(env, "close"):
        env.close()  # type: ignore[union-attr]

    wandb.finish()
    print("\n✅ Training complete.")
    print(f"   Best mean reward: {best_mean_reward:+.3f}")
    print(f"   Checkpoints in:   {args.run_dir}/")
    print(f"   Upload logs:      wandb sync {args.run_dir}/wandb/latest-run")


if __name__ == "__main__":
    main()
