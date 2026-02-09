"""
sensors.py — subscribe to Gazebo-native sensor topics and cache state.

Reads the *same* topics that PX4's GZBridge subscribes to, giving
ground-truth pose / velocity and raw IMU data without an extra ROS 2 hop.

All callbacks are lock-protected; ``get_obs()`` returns a flat
numpy array suitable for a Gymnasium observation space.
"""

from __future__ import annotations

import os
import threading

import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.imu_pb2 import IMU  # noqa: E402
from gz.msgs10.odometry_with_covariance_pb2 import OdometryWithCovariance  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402


class GzSensors:
    """
    Subscribe to Gazebo Harmonic sensor topics for one model.

    Exposes a thread-safe ``get_obs()`` → ``np.ndarray`` with shape
    ``(16,)`` containing::

        [pos_x, pos_y, pos_z,           # 3   ENU  (m)
         vel_x, vel_y, vel_z,           # 3   ENU  (m/s)
         quat_w, quat_x, quat_y, quat_z,  # 4   orientation
         ang_vel_x, ang_vel_y, ang_vel_z,  # 3   body-frame (rad/s)
         lin_acc_x, lin_acc_y, lin_acc_z]  # 3   body-frame (m/s²)
    """

    OBS_DIM = 16

    def __init__(
        self,
        world_name: str = "default",
        model_name: str = "x500_0",
    ) -> None:
        self.world_name = world_name
        self.model_name = model_name

        self._node = Node()
        self._lock = threading.Lock()

        # Cached latest values  (ENU / body-frame)
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._ang_vel = np.zeros(3)
        self._lin_acc = np.zeros(3)

        # ── Odometry  (position + velocity + orientation) ──
        odom_topic = f"/model/{model_name}/odometry_with_covariance"
        self._node.subscribe(
            OdometryWithCovariance,
            odom_topic,
            self._on_odom,
        )

        # ── IMU  (angular velocity + linear acceleration) ──
        imu_topic = (
            f"/world/{world_name}/model/{model_name}"
            f"/link/base_link/sensor/imu_sensor/imu"
        )
        self._node.subscribe(IMU, imu_topic, self._on_imu)

        # ── Ground-truth poses (fallback / multi-model) ────
        pose_topic = f"/world/{world_name}/dynamic_pose/info"
        self._node.subscribe(Pose_V, pose_topic, self._on_pose_v)

    # ── public API ──────────────────────────────────────────

    def get_obs(self) -> np.ndarray:
        """Return a flat (16,) float32 observation vector."""
        with self._lock:
            return np.concatenate(
                [
                    self._pos,
                    self._vel,
                    self._quat,
                    self._ang_vel,
                    self._lin_acc,
                ]
            ).astype(np.float32)

    def get_state_dict(self) -> dict[str, np.ndarray]:
        """Return a human-readable dict copy of the latest state."""
        with self._lock:
            return {
                "position": self._pos.copy(),
                "velocity": self._vel.copy(),
                "orientation": self._quat.copy(),
                "angular_velocity": self._ang_vel.copy(),
                "linear_acceleration": self._lin_acc.copy(),
            }

    # ── callbacks ───────────────────────────────────────────

    def _on_odom(self, msg: OdometryWithCovariance) -> None:
        p = msg.pose_with_covariance.pose
        v = msg.twist_with_covariance.twist.linear
        w = msg.twist_with_covariance.twist.angular
        q = p.orientation
        with self._lock:
            self._pos[:] = [p.position.x, p.position.y, p.position.z]
            self._vel[:] = [v.x, v.y, v.z]
            self._quat[:] = [q.w, q.x, q.y, q.z]
            self._ang_vel[:] = [w.x, w.y, w.z]

    def _on_imu(self, msg: IMU) -> None:
        a = msg.linear_acceleration
        g = msg.angular_velocity
        with self._lock:
            self._lin_acc[:] = [a.x, a.y, a.z]
            self._ang_vel[:] = [g.x, g.y, g.z]

    def _on_pose_v(self, msg: Pose_V) -> None:
        """Fallback – extract our model from the ``Pose_V`` bundle."""
        for pose in msg.pose:
            if pose.name == self.model_name:
                p = pose.position
                q = pose.orientation
                with self._lock:
                    self._pos[:] = [p.x, p.y, p.z]
                    self._quat[:] = [q.w, q.x, q.y, q.z]
                break
