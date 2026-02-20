"""
sensors.py — subscribe to Gazebo-native sensor topics and cache state.

Reads the *same* topics that PX4's GZBridge subscribes to, giving
ground-truth pose / velocity, raw IMU data, and camera images from the
``x500_mono_cam`` model — all without an extra ROS 2 hop in the hot-path.

IMU integration
~~~~~~~~~~~~~~~
Gazebo publishes IMU at the physics rate (~250 Hz with the default
``max_step_size = 0.004 s``).  Between two ``get_obs()`` calls the
samples are buffered and **averaged** so the observation is at the
env-step rate (50 Hz).

Camera
~~~~~~
The mono camera publishes at ~30 Hz (SDF default for x500_mono_cam).
``get_obs()`` returns the latest available frame (down-scaled to
``cam_obs_height × cam_obs_width`` for the neural-network input).

ROS 2 visualisation topics (published from ``get_obs()``):
    /drone/camera/image     (sensor_msgs/Image)       — full-res for RViz
    /drone/imu              (sensor_msgs/Imu)          — 50 Hz averaged
    /drone/pose             (geometry_msgs/PoseStamped) — current pose
    /drone/trajectory       (nav_msgs/Path)            — past trajectory
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque

import cv2
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.imu_pb2 import IMU  # noqa: E402
from gz.msgs10.image_pb2 import Image as GzImage  # noqa: E402
from gz.msgs10.odometry_with_covariance_pb2 import (  # noqa: E402
    OdometryWithCovariance,
)
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402

# ROS 2 — only used for RViz visualisation (publishing only, no spin)
import rclpy  # noqa: E402
from rclpy.node import Node as RosNode  # noqa: E402
from sensor_msgs.msg import Image as RosImage, Imu as RosImu  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402


class GzSensors:
    """
    Subscribe to Gazebo Harmonic sensor topics for one ``x500_mono_cam``
    model and provide dict-observations for a Gymnasium environment.

    Observation dictionary returned by :meth:`get_obs`:
    ====================================================
    =============  =============  ========================
    key            shape          description
    =============  =============  ========================
    ``imu``        ``(6,)``       averaged accel(3)+gyro(3)
    ``camera``     ``(H, W, 3)`` down-scaled RGB (uint8)
    ``position``   ``(3,)``       ENU position (m)
    ``velocity``   ``(3,)``       ENU velocity (m/s)
    ``orientation````(4,)``       quaternion (w, x, y, z)
    =============  =============  ========================

    ROS 2 topics published once per ``get_obs()`` call:
        * ``/drone/camera/image``  — full-resolution camera
        * ``/drone/imu``           — 50 Hz averaged IMU
        * ``/drone/pose``          — drone pose
        * ``/drone/trajectory``    — cumulative flight path
    """

    def __init__(
        self,
        world_name: str = "default",
        model_name: str = "x500_mono_cam_0",
        camera_link: str = "camera_link",
        camera_sensor: str = "camera",
        cam_obs_height: int = 64,
        cam_obs_width: int = 64,
        enable_rviz: bool = True,
        max_traj_len: int = 5_000,
    ) -> None:
        self.world_name = world_name
        self.model_name = model_name
        self.cam_obs_height = cam_obs_height
        self.cam_obs_width = cam_obs_width
        self._enable_rviz = enable_rviz

        self._gz_node = Node()
        self._lock = threading.Lock()

        # ── Cached latest values (ENU / body-frame) ──────────
        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._ang_vel = np.zeros(3, dtype=np.float64)
        self._lin_acc = np.zeros(3, dtype=np.float64)

        # ── IMU ring-buffer for 250→50 Hz integration ───────
        self._imu_buffer: deque[np.ndarray] = deque(maxlen=20)

        # ── Camera ──────────────────────────────────────────
        self._camera_raw: np.ndarray | None = None  # full-res RGB
        self._camera_raw_hw: tuple[int, int] = (0, 0)
        self._camera_new: bool = False  # True when a new frame arrived
        self._camera_obs_cache: np.ndarray | None = None  # cached resize
        self._camera_full_cache: np.ndarray | None = None  # cached copy

        # ── Trajectory history for RViz ─────────────────────
        self._trajectory: list[list[float]] = []
        self._max_traj_len = max_traj_len

        # ── Gazebo subscriptions ────────────────────────────
        # Odometry (position + velocity + orientation)
        odom_topic = f"/model/{model_name}/odometry_with_covariance"
        self._gz_node.subscribe(
            OdometryWithCovariance,
            odom_topic,
            self._on_odom,
        )

        # IMU (angular velocity + linear acceleration)
        imu_topic = (
            f"/world/{world_name}/model/{model_name}"
            f"/link/base_link/sensor/imu_sensor/imu"
        )
        self._gz_node.subscribe(IMU, imu_topic, self._on_imu)

        # Camera
        cam_topic = (
            f"/world/{world_name}/model/{model_name}"
            f"/link/{camera_link}/sensor/{camera_sensor}/image"
        )
        self._gz_node.subscribe(GzImage, cam_topic, self._on_camera)

        # Ground-truth poses (fallback / multi-model)
        pose_topic = f"/world/{world_name}/dynamic_pose/info"
        self._gz_node.subscribe(Pose_V, pose_topic, self._on_pose_v)

        # ── ROS 2 publishers for RViz visualisation ─────────
        self._ros_node: RosNode | None = None
        if self._enable_rviz:
            self._init_ros_publishers()

    # ────────────────────────────────────────────────────────
    #  ROS 2 init  (publishing only — no spin required)
    # ────────────────────────────────────────────────────────

    def _init_ros_publishers(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self._ros_node = RosNode(
            "gz_sensors_viz",
            parameter_overrides=[
                rclpy.Parameter(
                    "use_sim_time",
                    rclpy.Parameter.Type.BOOL,
                    True,
                ),
            ],
        )
        self._pub_cam = self._ros_node.create_publisher(
            RosImage,
            "/drone/camera/image",
            10,
        )
        self._pub_imu = self._ros_node.create_publisher(
            RosImu,
            "/drone/imu",
            10,
        )
        self._pub_pose = self._ros_node.create_publisher(
            PoseStamped,
            "/drone/pose",
            10,
        )
        self._pub_traj = self._ros_node.create_publisher(
            Path,
            "/drone/trajectory",
            10,
        )
        self._pub_wp_rel = self._ros_node.create_publisher(
            Float32MultiArray,
            "/drone/waypoints_relative",
            10,
        )

    # ════════════════════════════════════════════════════════
    #  Public API
    # ════════════════════════════════════════════════════════

    def publish_waypoints_relative(self, wp_flat: list[float] | np.ndarray) -> None:
        """Publish body-frame relative waypoint tensor to ROS 2.

        Parameters
        ----------
        wp_flat : list or ndarray, shape (6,)
            Body-frame relative offsets ``[wp0_x, wp0_y, wp0_z,
            wp1_x, wp1_y, wp1_z]`` as returned by
            ``WaypointBuffer.current_targets_tensor()`` (squeezed).
        """
        if self._ros_node is None:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in wp_flat]
        self._pub_wp_rel.publish(msg)

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return a dict observation (one per env step, ≈ 50 Hz).

        * IMU buffer is averaged and **cleared** so the next step gets
          a fresh window of samples.
        * Camera is the latest available frame, down-scaled to
          ``(cam_obs_height, cam_obs_width, 3)``.
        """
        with self._lock:
            # ── IMU: average buffered samples ───────────────
            if len(self._imu_buffer) > 0:
                imu_avg = np.mean(self._imu_buffer, axis=0).astype(np.float32)
                self._imu_buffer.clear()
            else:
                imu_avg = np.concatenate(
                    [
                        self._lin_acc,
                        self._ang_vel,
                    ]
                ).astype(np.float32)

            # ── Camera: down-scale for obs (cached) ────────
            if self._camera_raw is not None:
                if self._camera_new or self._camera_obs_cache is None:
                    # New frame arrived — resize and cache
                    cam_full = self._camera_raw.copy()
                    cam_obs = cv2.resize(
                        cam_full,
                        (self.cam_obs_width, self.cam_obs_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    self._camera_obs_cache = cam_obs
                    self._camera_full_cache = cam_full
                    self._camera_new = False
                else:
                    # Reuse cached resize (camera ~30 Hz, env ~50 Hz)
                    cam_obs = self._camera_obs_cache
                    cam_full = self._camera_full_cache
            else:
                cam_full = None
                cam_obs = np.zeros(
                    (self.cam_obs_height, self.cam_obs_width, 3),
                    dtype=np.uint8,
                )

            pos = self._pos.copy().astype(np.float32)
            vel = self._vel.copy().astype(np.float32)
            quat = self._quat.copy().astype(np.float32)

        # ── ROS 2 visualisation (outside lock) ──────────────
        if self._enable_rviz and self._ros_node is not None:
            self._publish_rviz(imu_avg, cam_full, pos, quat)

        return {
            "imu": imu_avg,  # (6,)  float32
            "camera": cam_obs,  # (H,W,3) uint8
            "position": pos,  # (3,)  float32
            "velocity": vel,  # (3,)  float32
            "orientation": quat,  # (4,)  float32
        }

    def get_state_dict(self) -> dict[str, np.ndarray]:
        """Return a human-readable dict copy of the latest state
        (does **not** clear the IMU buffer)."""
        with self._lock:
            return {
                "position": self._pos.copy(),
                "velocity": self._vel.copy(),
                "orientation": self._quat.copy(),
                "angular_velocity": self._ang_vel.copy(),
                "linear_acceleration": self._lin_acc.copy(),
            }

    def get_flat_state(self) -> np.ndarray:
        """Flat (16,) state vector for backwards-compat / reward fn."""
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

    def reset_state(self) -> None:
        """Zero all cached velocity / acceleration state for a clean
        episode start.

        Must be called during ``env.reset()`` **after** teleporting
        the model, so that the first observation does not carry stale
        velocity or angular-velocity values from the previous episode
        (the odom callback may not have fired yet at that point).
        """
        with self._lock:
            self._vel[:] = 0.0
            self._ang_vel[:] = 0.0
            self._lin_acc[:] = 0.0
            self._imu_buffer.clear()
            self._trajectory.clear()

    def clear_imu_buffer(self) -> None:
        """Clear buffered IMU readings (call on episode reset)."""
        with self._lock:
            self._imu_buffer.clear()

    def clear_trajectory(self) -> None:
        """Clear the stored trajectory path (call on episode reset)."""
        with self._lock:
            self._trajectory.clear()

    # ════════════════════════════════════════════════════════
    #  Gazebo callbacks
    # ════════════════════════════════════════════════════════

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
            # record position for trajectory
            self._trajectory.append([p.position.x, p.position.y, p.position.z])
            if len(self._trajectory) > self._max_traj_len:
                self._trajectory.pop(0)

    def _on_imu(self, msg: IMU) -> None:
        a = msg.linear_acceleration
        g = msg.angular_velocity
        with self._lock:
            self._lin_acc[:] = [a.x, a.y, a.z]
            self._ang_vel[:] = [g.x, g.y, g.z]
            # buffer sample for averaging
            self._imu_buffer.append(
                np.array(
                    [
                        a.x,
                        a.y,
                        a.z,
                        g.x,
                        g.y,
                        g.z,
                    ],
                    dtype=np.float64,
                )
            )

    def _on_camera(self, msg: GzImage) -> None:
        """Decode Gazebo Image proto → numpy RGB array."""
        h, w = msg.height, msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)
        expected = h * w * 3
        if len(data) != expected:
            return  # unexpected pixel format, skip
        img = data.reshape(h, w, 3)
        with self._lock:
            self._camera_raw = img
            self._camera_raw_hw = (h, w)
            self._camera_new = True

    def _on_pose_v(self, msg: Pose_V) -> None:
        """Fallback — extract our model from the ``Pose_V`` bundle."""
        for pose in msg.pose:
            if pose.name == self.model_name:
                p = pose.position
                q = pose.orientation
                with self._lock:
                    self._pos[:] = [p.x, p.y, p.z]
                    self._quat[:] = [q.w, q.x, q.y, q.z]
                break

    # ════════════════════════════════════════════════════════
    #  ROS 2 RViz publishing
    # ════════════════════════════════════════════════════════

    def _publish_rviz(
        self,
        imu: np.ndarray,
        cam_full: np.ndarray | None,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        """Publish visualisation data to ROS 2 topics.

        Called from ``get_obs()`` (outside the lock) once per env step.
        """
        assert self._ros_node is not None
        now = self._ros_node.get_clock().now().to_msg()

        # ── Camera image (full resolution) ──────────────────
        if cam_full is not None:
            img_msg = RosImage()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "camera_link"
            img_msg.height, img_msg.width = cam_full.shape[:2]
            img_msg.encoding = "rgb8"
            img_msg.is_bigendian = False
            img_msg.step = cam_full.shape[1] * 3
            img_msg.data = cam_full.tobytes()
            self._pub_cam.publish(img_msg)

        # ── IMU ─────────────────────────────────────────────
        imu_msg = RosImu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"
        imu_msg.linear_acceleration.x = float(imu[0])
        imu_msg.linear_acceleration.y = float(imu[1])
        imu_msg.linear_acceleration.z = float(imu[2])
        imu_msg.angular_velocity.x = float(imu[3])
        imu_msg.angular_velocity.y = float(imu[4])
        imu_msg.angular_velocity.z = float(imu[5])
        self._pub_imu.publish(imu_msg)

        # ── Drone pose ──────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])
        pose_msg.pose.orientation.w = float(quat[0])
        pose_msg.pose.orientation.x = float(quat[1])
        pose_msg.pose.orientation.y = float(quat[2])
        pose_msg.pose.orientation.z = float(quat[3])
        self._pub_pose.publish(pose_msg)

        # ── Trajectory path ─────────────────────────────────
        with self._lock:
            traj_copy = list(self._trajectory)
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "map"
        # Down-sample to max 500 points for RViz performance
        step = max(1, len(traj_copy) // 500)
        for pt in traj_copy[::step]:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "map"
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            ps.pose.position.z = float(pt[2])
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self._pub_traj.publish(path_msg)
