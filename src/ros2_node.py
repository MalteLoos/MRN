"""
ROS 2 Humble node that wraps the DroneAgent for **inference**.

Subscribes to the topics published by ``GzSensors`` during training or
directly from the Gazebo ↔ ROS 2 bridge:

    /drone/imu          (sensor_msgs/Imu)          — 50 Hz averaged IMU
    /drone/camera/image (sensor_msgs/Image)         — mono-cam image
    /drone/pose         (geometry_msgs/PoseStamped) — drone pose in map

Publishes:
    /fmu/in/offboard_control_mode  (px4_msgs/OffboardControlMode)
    /fmu/in/vehicle_attitude_setpoint  (px4_msgs/VehicleAttitudeSetpoint)

    The node sends **attitude setpoints** (roll, pitch, thrust) directly
    to PX4 over the DDS agent — no intermediate body-rate topic.

    /drone/waypoints    (geometry_msgs/PoseArray) — the two active look-ahead WPs

The node is configured with ``use_sim_time:=true`` so that every
published header stamp comes from the simulation clock.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
)
from std_msgs.msg import Header

from px4_msgs.msg import (
    OffboardControlMode,
    VehicleAttitudeSetpoint,
)

import torch

from model import DroneAgent, ModelConfig


# ---------------------------------------------------------------------------
# QoS profiles
# ---------------------------------------------------------------------------

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------


class DroneAttitudeNode(Node):
    """
    ROS 2 node that fuses IMU + camera + pose, runs the RL policy, and
    publishes **attitude setpoints** (roll, pitch, thrust) to PX4 via
    the DDS agent — stamped with simulation time.
    """

    # Action scaling (must match env.py defaults)
    MAX_ROLL = math.radians(30.0)
    MAX_PITCH = math.radians(30.0)

    def __init__(self) -> None:
        super().__init__("drone_attitude_controller")

        # ---- parameters -----------------------------------------------------
        self.declare_parameter("use_sim_time", True)
        self.declare_parameter("safety_radius", 1.0)
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("model_checkpoint", "")
        self.declare_parameter("deterministic", False)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("max_roll_deg", 30.0)
        self.declare_parameter("max_pitch_deg", 30.0)

        # Waypoints as flat list [x1,y1,z1, x2,y2,z2, ...]
        self.declare_parameter("route", [0.0, 0.0, 5.0, 10.0, 0.0, 5.0])

        safety_radius = self.get_parameter("safety_radius").value
        device = self.get_parameter("device").value
        checkpoint = self.get_parameter("model_checkpoint").value
        self._deterministic = self.get_parameter("deterministic").value
        self._max_roll = math.radians(self.get_parameter("max_roll_deg").value)
        self._max_pitch = math.radians(self.get_parameter("max_pitch_deg").value)

        # ---- build agent ----------------------------------------------------
        cfg = ModelConfig(safety_radius=safety_radius, action_dim=3)
        route = self._parse_route(
            self.get_parameter("route").value  # type: ignore[arg-type]
        )

        self._agent = DroneAgent(cfg, route=route, device=device)
        if checkpoint:
            self._agent.load(checkpoint)
            self.get_logger().info(f"Loaded checkpoint: {checkpoint}")

        # ---- cached latest sensor readings ----------------------------------
        self._latest_imu: Optional[np.ndarray] = None
        self._latest_cam: Optional[np.ndarray] = None
        self._latest_pos: Optional[np.ndarray] = None
        self._latest_quat: Optional[np.ndarray] = None  # (w,x,y,z)

        # ---- subscribers ----------------------------------------------------
        self._sub_imu = self.create_subscription(
            Imu,
            "/drone/imu",
            self._imu_cb,
            SENSOR_QOS,
        )
        self._sub_cam = self.create_subscription(
            Image,
            "/drone/camera/image",
            self._cam_cb,
            SENSOR_QOS,
        )
        self._sub_pose = self.create_subscription(
            PoseStamped,
            "/drone/pose",
            self._pose_cb,
            SENSOR_QOS,
        )

        # ---- PX4 publishers (via DDS agent) ---------------------------------
        self._pub_offboard_mode = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )
        self._pub_attitude = self.create_publisher(
            VehicleAttitudeSetpoint,
            "/fmu/in/vehicle_attitude_setpoint",
            10,
        )

        # ---- waypoint visualisation -----------------------------------------
        self._pub_wp = self.create_publisher(
            PoseArray,
            "/drone/waypoints",
            10,
        )

        # ---- control timer --------------------------------------------------
        rate_hz = self.get_parameter("control_rate_hz").value
        period = 1.0 / rate_hz
        self._timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f"DroneAttitudeNode started  |  rate={rate_hz} Hz  |  "
            f"safety_r={safety_radius} m  |  device={device}  |  "
            f"max_roll={math.degrees(self._max_roll):.0f}°  "
            f"max_pitch={math.degrees(self._max_pitch):.0f}°"
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_route(
        flat: List[float],
    ) -> List[Tuple[float, float, float]]:
        """Convert [x1,y1,z1, x2,y2,z2, ...] → [(x1,y1,z1), ...]."""
        assert (
            len(flat) % 3 == 0 and len(flat) >= 6
        ), "Route must have ≥ 2 waypoints (6 floats)."
        return [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)]

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _imu_cb(self, msg: Imu) -> None:
        self._latest_imu = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )

    def _cam_cb(self, msg: Image) -> None:
        h, w = msg.height, msg.width
        channels = 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, channels)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1].copy()
        self._latest_cam = img

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        self._latest_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        self._latest_quat = np.array(
            [q.w, q.x, q.y, q.z],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Simulation-time header helper
    # ------------------------------------------------------------------

    def _sim_header(self, frame_id: str = "base_link") -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return header

    # ------------------------------------------------------------------
    # PX4 timestamp helper
    # ------------------------------------------------------------------

    def _px4_timestamp(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1_000)

    # ------------------------------------------------------------------
    # Euler ↔ quaternion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_from_quat(q: np.ndarray) -> float:
        """Extract yaw (rad) from quaternion [w, x, y, z]."""
        return math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float):
        """Euler (ZYX) → quaternion [w, x, y, z]."""
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        # Wait until all sensor streams have been received at least once
        if (
            self._latest_imu is None
            or self._latest_cam is None
            or self._latest_pos is None
            or self._latest_quat is None
        ):
            return

        # ---- run policy -----------------------------------------------------
        obs = {
            "cam": self._latest_cam,
            "imu": self._latest_imu,
            "drone_pos": self._latest_pos,
        }
        action = self._agent.step(obs, deterministic=self._deterministic)
        # action: (3,) tensor → [roll, pitch, thrust]  in [-1, 1]

        # ---- scale to physical commands -------------------------------------
        roll_cmd = float(action[0]) * self._max_roll
        pitch_cmd = float(action[1]) * self._max_pitch
        thrust_cmd = float(np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0))

        # Hold current yaw
        current_yaw = self._yaw_from_quat(self._latest_quat)
        w, x, y, z = self._euler_to_quat(roll_cmd, pitch_cmd, current_yaw)

        # ---- publish OffboardControlMode (attitude) -------------------------
        ocm = OffboardControlMode()
        ocm.timestamp = self._px4_timestamp()
        ocm.position = False
        ocm.velocity = False
        ocm.acceleration = False
        ocm.attitude = True
        ocm.body_rate = False
        self._pub_offboard_mode.publish(ocm)

        # ---- publish VehicleAttitudeSetpoint --------------------------------
        att = VehicleAttitudeSetpoint()
        att.timestamp = self._px4_timestamp()
        att.q_d[0] = w
        att.q_d[1] = x
        att.q_d[2] = y
        att.q_d[3] = z
        att.thrust_body[0] = 0.0
        att.thrust_body[1] = 0.0
        att.thrust_body[2] = -thrust_cmd  # NED body-Z down
        self._pub_attitude.publish(att)

        # ---- publish active waypoints for visualisation ---------------------
        self._publish_waypoints()

        # ---- log progress ---------------------------------------------------
        if self._agent.route_finished:
            self.get_logger().info(
                "Route complete — all waypoints reached.",
                throttle_duration_sec=5.0,
            )

    # ------------------------------------------------------------------
    # Waypoint visualisation
    # ------------------------------------------------------------------

    def _publish_waypoints(self) -> None:
        buf = self._agent._buffer
        if buf is None:
            return

        pa = PoseArray()
        pa.header = self._sim_header("map")

        for wp in (buf.wp0, buf.wp1):
            pose = Pose()
            pose.position.x = wp[0]
            pose.position.y = wp[1]
            pose.position.z = wp[2]
            pose.orientation.w = 1.0
            pa.poses.append(pose)

        self._pub_wp.publish(pa)

    # ------------------------------------------------------------------
    # Route hot-reload
    # ------------------------------------------------------------------

    def reload_route(self, flat_route: List[float]) -> None:
        """Change route at runtime."""
        route = self._parse_route(flat_route)
        self._agent.set_route(route)
        self.get_logger().info(f"Route reloaded with {len(route)} waypoints.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DroneAttitudeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
