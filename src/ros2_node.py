"""
ROS 2 Humble node that wraps the DroneAgent.

Subscribes to:
    /drone/imu          (sensor_msgs/Imu)
    /drone/camera/image (sensor_msgs/Image)
    /drone/pose         (geometry_msgs/PoseStamped)   — for waypoint buffer updates

Publishes:
    /drone/cmd_bodyrate (geometry_msgs/TwistStamped)
        angular.x = roll_rate
        angular.y = pitch_rate
        angular.z = yaw_rate
        linear.z  = collective thrust

    /drone/waypoints    (geometry_msgs/PoseArray)     — the two active look-ahead WPs

The node is configured with ``use_sim_time:=true`` so that every published
header stamp comes from the simulation clock (``/clock`` topic).
"""

from __future__ import annotations

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
    TwistStamped,
)
from std_msgs.msg import Header

import torch

from model import DroneAgent, ModelConfig


# ---------------------------------------------------------------------------
# QoS profile for sensor topics (best-effort, keep-last-1 is typical in sim)
# ---------------------------------------------------------------------------

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class DroneBodyRateNode(Node):
    """
    ROS 2 node that fuses IMU + camera + pose, runs the RL policy, and
    publishes body-rate commands stamped with simulation time.
    """

    def __init__(self) -> None:
        super().__init__("drone_bodyrate_controller")

        # ---- parameters -----------------------------------------------------
        self.declare_parameter("use_sim_time", True)
        self.declare_parameter("safety_radius", 1.0)
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("model_checkpoint", "")
        self.declare_parameter("deterministic", False)
        self.declare_parameter("device", "cpu")

        # Waypoints as flat list [x1,y1,z1, x2,y2,z2, ...]
        self.declare_parameter("route", [0.0, 0.0, 5.0,
                                          10.0, 0.0, 5.0])

        safety_radius = self.get_parameter("safety_radius").value
        device = self.get_parameter("device").value
        checkpoint = self.get_parameter("model_checkpoint").value
        self._deterministic = self.get_parameter("deterministic").value

        # ---- build agent ----------------------------------------------------
        cfg = ModelConfig(safety_radius=safety_radius)
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

        # ---- subscribers ----------------------------------------------------
        self._sub_imu = self.create_subscription(
            Imu, "/drone/imu", self._imu_cb, SENSOR_QOS,
        )
        self._sub_cam = self.create_subscription(
            Image, "/drone/camera/image", self._cam_cb, SENSOR_QOS,
        )
        self._sub_pose = self.create_subscription(
            PoseStamped, "/drone/pose", self._pose_cb, SENSOR_QOS,
        )

        # ---- publishers -----------------------------------------------------
        self._pub_cmd = self.create_publisher(
            TwistStamped, "/drone/cmd_bodyrate", 10,
        )
        self._pub_wp = self.create_publisher(
            PoseArray, "/drone/waypoints", 10,
        )

        # ---- control timer --------------------------------------------------
        rate_hz = self.get_parameter("control_rate_hz").value
        period = 1.0 / rate_hz
        self._timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f"DroneBodyRateNode started  |  rate={rate_hz} Hz  |  "
            f"safety_r={safety_radius} m  |  device={device}"
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_route(
        flat: List[float],
    ) -> List[Tuple[float, float, float]]:
        """Convert [x1,y1,z1, x2,y2,z2, ...] → [(x1,y1,z1), ...]."""
        assert len(flat) % 3 == 0 and len(flat) >= 6, (
            "Route must have ≥ 2 waypoints (6 floats)."
        )
        return [
            (flat[i], flat[i + 1], flat[i + 2])
            for i in range(0, len(flat), 3)
        ]

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _imu_cb(self, msg: Imu) -> None:
        self._latest_imu = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float32)

    def _cam_cb(self, msg: Image) -> None:
        # Supports rgb8 and bgr8 encodings; fall back to raw bytes
        h, w = msg.height, msg.width
        channels = 3
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, channels)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1].copy()        # BGR → RGB
        self._latest_cam = img                   # kept as uint8, agent normalises

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._latest_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

    # ------------------------------------------------------------------
    # Simulation-time header helper
    # ------------------------------------------------------------------

    def _sim_header(self, frame_id: str = "base_link") -> Header:
        """Build a Header stamped with the current simulation clock."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()   # sim time when
        header.frame_id = frame_id                       # use_sim_time=true
        return header

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        # Wait until all sensor streams have been received at least once
        if (self._latest_imu is None
                or self._latest_cam is None
                or self._latest_pos is None):
            return

        # ---- run policy -----------------------------------------------------
        obs = {
            "cam": self._latest_cam,
            "imu": self._latest_imu,
            "drone_pos": self._latest_pos,
        }
        action = self._agent.step(obs, deterministic=self._deterministic)
        # action: (4,) tensor → [roll_rate, pitch_rate, yaw_rate, thrust]

        # ---- publish body-rate command --------------------------------------
        cmd = TwistStamped()
        cmd.header = self._sim_header("base_link")
        cmd.twist.angular.x = float(action[0])   # roll  rate
        cmd.twist.angular.y = float(action[1])   # pitch rate
        cmd.twist.angular.z = float(action[2])   # yaw   rate
        cmd.twist.linear.z  = float(action[3])   # thrust
        self._pub_cmd.publish(cmd)

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
    # Route hot-reload (can be called from a service / topic later)
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
    node = DroneBodyRateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
