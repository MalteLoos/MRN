#!/usr/bin/env python3
"""
dds_pose_relay.py — Publish drone pose from two sources for RViz.

Source 1 — PX4 DDS (NED → ENU converted):
    Subscribes:  /fmu/out/vehicle_local_position_v1, /fmu/out/vehicle_attitude
    Publishes:   /px4/pose  (PoseStamped)  +  TF  map → base_link

Source 2 — Gazebo ground-truth (already ENU):
    Subscribes:  /world/<world>/dynamic_pose/info  (gz-transport)
    Publishes:   /gz/pose   (PoseStamped)  +  TF  map → base_link_gz

Both run continuously so RViz always has a pose, even before PX4
starts publishing.

Usage:
    python3 dds_pose_relay.py --ros-args -p use_sim_time:=true \\
        -p world_name:=tugbot_depot -p model_name:=x500_mono_cam_0
"""

from __future__ import annotations

import math
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

# ── Gazebo transport ────────────────────────────────────────
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node as GzNode  # noqa: E402


# PX4 DDS topics use BEST_EFFORT / VOLATILE
_PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# ═══════════════════════════════════════════════════════════
#  NED → ENU helpers  (for PX4 DDS source)
# ═══════════════════════════════════════════════════════════


def _ned_to_enu_position(x_ned: float, y_ned: float, z_ned: float):
    """Convert NED position to ENU."""
    return y_ned, x_ned, -z_ned


_S = math.sqrt(2.0) / 2.0


def _ned_to_enu_quaternion(qw: float, qx: float, qy: float, qz: float):
    """Convert NED/FRD attitude quaternion to ENU/FLU (MAVROS convention).

    q_enu = NED_ENU_Q * q_ned * AIRCRAFT_BASELINK_Q
    Closed-form with s = √2/2:
        w_out = -s (qw + qz)
        x_out = -s (qx + qy)
        y_out =  s (qy - qx)
        z_out =  s (qz - qw)
    """
    return (
        -_S * (qw + qz),
        -_S * (qx + qy),
        _S * (qy - qx),
        _S * (qz - qw),
    )


# ═══════════════════════════════════════════════════════════
#  ROS 2 Node
# ═══════════════════════════════════════════════════════════


class DdsPoseRelay(Node):
    def __init__(self):
        super().__init__("dds_pose_relay")

        # ── parameters ────────────────────────────────────
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("world_name", "tugbot_depot")
        self.declare_parameter("model_name", "x500_mono_cam_0")

        self._frame = self.get_parameter("frame_id").value
        self._child_frame = self.get_parameter("child_frame_id").value
        world_name = self.get_parameter("world_name").value
        model_name = self.get_parameter("model_name").value

        # ── PX4 DDS state ─────────────────────────────────
        self._px4_pos = (0.0, 0.0, 0.0)
        self._px4_quat = (1.0, 0.0, 0.0, 0.0)
        self._has_px4_pos = False
        self._has_px4_att = False

        # ── Gazebo state ──────────────────────────────────
        self._gz_pos = (0.0, 0.0, 0.0)
        self._gz_quat = (1.0, 0.0, 0.0, 0.0)
        self._gz_lock = threading.Lock()

        # ── PX4 DDS subscribers ───────────────────────────
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._on_local_pos,
            _PX4_QOS,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._on_attitude,
            _PX4_QOS,
        )

        # ── PX4 DDS publishers ────────────────────────────
        self._pub_px4_pose = self.create_publisher(PoseStamped, "/px4/pose", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Gazebo publisher ──────────────────────────────
        self._pub_gz_pose = self.create_publisher(PoseStamped, "/gz/pose", 10)

        # ── Gazebo transport subscription ─────────────────
        self._gz_node = GzNode()
        pose_topic = f"/world/{world_name}/dynamic_pose/info"
        self._model_name = model_name
        self._gz_node.subscribe(Pose_V, pose_topic, self._on_gz_pose)

        # Timer to publish Gz pose at 50 Hz (gz callback is
        # on a different thread, so we relay via timer).
        self.create_timer(0.02, self._publish_gz)

        self.get_logger().info(
            f"PX4 DDS  → /px4/pose + TF {self._frame}→{self._child_frame}"
        )
        self.get_logger().info(
            f"Gz pose  → /gz/pose  + TF {self._frame}→{self._child_frame}_gz  "
            f"(model={model_name}, world={world_name})"
        )

    # ═══════════════════════════════════════════════════════
    #  PX4 DDS callbacks
    # ═══════════════════════════════════════════════════════

    def _on_local_pos(self, msg: VehicleLocalPosition):
        if not (math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z)):
            return
        self._px4_pos = _ned_to_enu_position(msg.x, msg.y, msg.z)
        self._has_px4_pos = True
        self._publish_px4()

    def _on_attitude(self, msg: VehicleAttitude):
        q = msg.q  # [w, x, y, z] in NED/FRD
        self._px4_quat = _ned_to_enu_quaternion(
            float(q[0]), float(q[1]), float(q[2]), float(q[3])
        )
        self._has_px4_att = True
        self._publish_px4()

    def _publish_px4(self):
        if not (self._has_px4_pos and self._has_px4_att):
            return

        now = self.get_clock().now().to_msg()
        ex, ey, ez = self._px4_pos
        qw, qx, qy, qz = self._px4_quat

        # PoseStamped
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self._frame
        pose.pose.position.x = float(ex)
        pose.pose.position.y = float(ey)
        pose.pose.position.z = float(ez)
        pose.pose.orientation.w = float(qw)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        self._pub_px4_pose.publish(pose)

        # TF: map → base_link
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self._frame
        t.child_frame_id = self._child_frame
        t.transform.translation.x = float(ex)
        t.transform.translation.y = float(ey)
        t.transform.translation.z = float(ez)
        t.transform.rotation.w = float(qw)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        self._tf_broadcaster.sendTransform(t)

    # ═══════════════════════════════════════════════════════
    #  Gazebo ground-truth callback  (runs on gz-transport thread)
    # ═══════════════════════════════════════════════════════

    def _on_gz_pose(self, msg: Pose_V):
        """Extract our model from the Pose_V bundle (same as sensors.py)."""
        for pose in msg.pose:
            if pose.name == self._model_name:
                p = pose.position
                q = pose.orientation
                with self._gz_lock:
                    self._gz_pos = (p.x, p.y, p.z)
                    self._gz_quat = (q.w, q.x, q.y, q.z)
                break

    def _publish_gz(self):
        """Timer callback — publish the latest Gz ground-truth pose."""
        if not rclpy.ok():
            return

        with self._gz_lock:
            gx, gy, gz_ = self._gz_pos
            gqw, gqx, gqy, gqz = self._gz_quat

        now = self.get_clock().now().to_msg()

        # PoseStamped on /gz/pose
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self._frame
        pose.pose.position.x = float(gx)
        pose.pose.position.y = float(gy)
        pose.pose.position.z = float(gz_)
        pose.pose.orientation.w = float(gqw)
        pose.pose.orientation.x = float(gqx)
        pose.pose.orientation.y = float(gqy)
        pose.pose.orientation.z = float(gqz)
        self._pub_gz_pose.publish(pose)

        # TF: map → base_link_gz
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self._frame
        t.child_frame_id = self._child_frame + "_gz"
        t.transform.translation.x = float(gx)
        t.transform.translation.y = float(gy)
        t.transform.translation.z = float(gz_)
        t.transform.rotation.w = float(gqw)
        t.transform.rotation.x = float(gqx)
        t.transform.rotation.y = float(gqy)
        t.transform.rotation.z = float(gqz)
        self._tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = DdsPoseRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
