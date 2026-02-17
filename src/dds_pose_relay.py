#!/usr/bin/env python3
"""
dds_pose_relay.py — Publish PX4 DDS vehicle pose for RViz display.

Subscribes to PX4's native DDS topics:
    /fmu/out/vehicle_local_position_v1   (px4_msgs/VehicleLocalPosition)
    /fmu/out/vehicle_attitude         (px4_msgs/VehicleAttitude)

Publishes:
    /px4/pose   (geometry_msgs/PoseStamped)  — pose in ``map`` frame
    TF:  map → base_link                     — for RViz TF tree

PX4 uses NED (North-East-Down) internally; this node converts to ENU
(East-North-Up) so that RViz Z-up conventions work correctly.

Usage:
    ros2 run --prefix 'python3' . dds_pose_relay  # or just:
    python3 dds_pose_relay.py --ros-args -p use_sim_time:=true
"""

from __future__ import annotations

import math

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


# PX4 DDS topics use BEST_EFFORT / VOLATILE
_PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _ned_to_enu_position(x_ned: float, y_ned: float, z_ned: float):
    """Convert NED position to ENU."""
    return y_ned, x_ned, -z_ned


def _ned_to_enu_quaternion(qw: float, qx: float, qy: float, qz: float):
    """Convert NED/FRD attitude quaternion to ENU/FLU (MAVROS convention).

    Reproduces the MAVROS transform chain:
        q_enu = NED_ENU_Q * q_ned * AIRCRAFT_BASELINK_Q
    where
        NED_ENU_Q          = quaternion_from_rpy(π, 0, π/2) = (0, s, s, 0)
        AIRCRAFT_BASELINK_Q = quaternion_from_rpy(π, 0, 0)  = (0, 1, 0, 0)
    with s = √2/2.

    Expanding the two Hamilton products yields the closed-form:
        w_out = -s (qw + qz)
        x_out = -s (qx + qy)
        y_out =  s (qy - qx)
        z_out =  s (qz - qw)
    """
    import math

    s = math.sqrt(2.0) / 2.0
    return (
        -s * (qw + qz),
        -s * (qx + qy),
        s * (qy - qx),
        s * (qz - qw),
    )


class DdsPoseRelay(Node):
    def __init__(self):
        super().__init__("dds_pose_relay")
        # use_sim_time is already declared by the Node base class;
        # pass it via --ros-args -p use_sim_time:=true
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")

        self._frame = self.get_parameter("frame_id").value
        self._child_frame = self.get_parameter("child_frame_id").value

        # Latest state (updated asynchronously)
        self._pos = (0.0, 0.0, 0.0)
        self._quat = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
        self._has_pos = False
        self._has_att = False

        # --- subscribers ---
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

        # --- publishers ---
        self._pub_pose = self.create_publisher(PoseStamped, "/px4/pose", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"Relaying PX4 DDS pose → /px4/pose + TF {self._frame}→{self._child_frame}"
        )

    # ── PX4 callbacks ──────────────────────────────────────

    def _on_local_pos(self, msg: VehicleLocalPosition):
        if not (math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z)):
            return
        self._pos = _ned_to_enu_position(msg.x, msg.y, msg.z)
        self._has_pos = True
        self._publish()

    def _on_attitude(self, msg: VehicleAttitude):
        q = msg.q  # [w, x, y, z] in NED/FRD
        self._quat = _ned_to_enu_quaternion(
            float(q[0]), float(q[1]), float(q[2]), float(q[3])
        )
        self._has_att = True
        self._publish()

    # ── Publish ────────────────────────────────────────────

    def _publish(self):
        if not (self._has_pos and self._has_att):
            return

        now = self.get_clock().now().to_msg()
        ex, ey, ez = self._pos
        qw, qx, qy, qz = self._quat

        # PoseStamped
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self._frame
        pose.pose.position.x = ex
        pose.pose.position.y = ey
        pose.pose.position.z = ez
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        self._pub_pose.publish(pose)

        # TF broadcast
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self._frame
        t.child_frame_id = self._child_frame
        t.transform.translation.x = ex
        t.transform.translation.y = ey
        t.transform.translation.z = ez
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        self._tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = DdsPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
