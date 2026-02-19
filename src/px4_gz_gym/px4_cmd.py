"""
px4_cmd.py — PX4 arming / takeoff / offboard via Micro XRCE-DDS (px4_msgs).

Talks directly to PX4 over the DDS agent — **no MAVROS required**.
This uses the native PX4 ROS 2 interface (``/fmu/in/*``, ``/fmu/out/*``)
so that:

  • ``OffboardControlMode`` is published **separately** from setpoints,
    ensuring PX4 stays in OFFBOARD even if setpoints are delayed.
  • ``VehicleCommand`` replaces MAVROS service calls for arming,
    takeoff, and mode changes.
  • ``TrajectorySetpoint`` carries the actual position/velocity targets.

The module lazily initialises a *single* ROS 2 node (``_PX4Cmd``) that
is reused across episode resets.  ``rclpy.init()`` is called once if
needed.

All public functions are **blocking** and intended to be called from
the Gymnasium ``reset()`` / ``close()`` path (not from a ROS 2
callback).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    VehicleStatus,
)


# ── QoS for PX4 DDS topics ─────────────────────────────────
# PX4 uses BEST_EFFORT / VOLATILE for all uORB-bridged topics.

_PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# ── PX4 nav_state constants ────────────────────────────────

_NAV_STATE_OFFBOARD = 14


# ── MAVLink command IDs (used in VehicleCommand.command) ────

_VEHICLE_CMD_ARM_DISARM = 400  # MAV_CMD_COMPONENT_ARM_DISARM
_VEHICLE_CMD_DO_SET_MODE = 176  # MAV_CMD_DO_SET_MODE
_VEHICLE_CMD_NAV_TAKEOFF = 22  # MAV_CMD_NAV_TAKEOFF
_VEHICLE_CMD_NAV_LAND = 21  # MAV_CMD_NAV_LAND

# PX4 custom main-mode numbers (param2 of DO_SET_MODE)
_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
_PX4_CUSTOM_MAIN_MODE_AUTO = 4
# PX4 custom sub-mode numbers (param3 of DO_SET_MODE, for AUTO)
_PX4_CUSTOM_SUB_MODE_AUTO_LAND = 6


# ── Singleton node ──────────────────────────────────────────


class _PX4Cmd(Node):
    """Thin ROS 2 helper for commanding PX4 via the DDS agent."""

    def __init__(self) -> None:
        super().__init__(
            "px4_gz_gym_cmd",
            parameter_overrides=[
                rclpy.Parameter(
                    "use_sim_time",
                    rclpy.Parameter.Type.BOOL,
                    True,
                ),
            ],
        )

        # ── vehicle status subscription ─────────────────────
        self._status_lock = threading.Lock()
        self._status: Optional[VehicleStatus] = None
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self._status_cb,
            _PX4_QOS,
        )

        # ── publishers ──────────────────────────────────────

        # Offboard control-mode heartbeat  (SEPARATE from setpoints)
        self._offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )

        # Trajectory setpoint  (position / velocity targets)
        self._setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )

        # Attitude setpoint  (roll / pitch / thrust)
        self._attitude_pub = self.create_publisher(
            VehicleAttitudeSetpoint,
            "/fmu/in/vehicle_attitude_setpoint_v1",
            10,
        )

        # Vehicle command  (arm, takeoff, mode-switch, land)
        self._cmd_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

    # ── status callback ────────────────────────────────────

    def _status_cb(self, msg: VehicleStatus) -> None:
        with self._status_lock:
            self._status = msg

    def reset_state(self) -> None:
        """Clear cached status so stale data from a previous
        episode doesn't mislead the connection / arming checks."""
        with self._status_lock:
            self._status = None

    @property
    def armed(self) -> bool:
        with self._status_lock:
            if self._status is None:
                return False
            return self._status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    @property
    def nav_state(self) -> int:
        with self._status_lock:
            if self._status is None:
                return -1
            return int(self._status.nav_state)

    @property
    def connected(self) -> bool:
        """True once we have received at least one VehicleStatus."""
        with self._status_lock:
            return self._status is not None

    @property
    def in_offboard(self) -> bool:
        return self.nav_state == _NAV_STATE_OFFBOARD

    # ── publish helpers ─────────────────────────────────────

    def publish_offboard_control_mode(
        self,
        position: bool = True,
        velocity: bool = False,
        acceleration: bool = False,
        attitude: bool = False,
        body_rate: bool = False,
    ) -> None:
        """Publish an ``OffboardControlMode`` message.

        This is the **offboard heartbeat** — PX4 requires it at ≥ 2 Hz
        to stay in OFFBOARD mode.  It is intentionally **decoupled**
        from setpoints so that a delayed setpoint never causes PX4 to
        exit offboard.
        """
        msg = OffboardControlMode()
        msg.timestamp = self._px4_timestamp()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = acceleration
        msg.attitude = attitude
        msg.body_rate = body_rate
        self._offboard_mode_pub.publish(msg)

    def publish_setpoint(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = -2.5,
    ) -> None:
        """Publish a ``TrajectorySetpoint`` position target.

        PX4 uses NED, so *z* should be **negative** for altitude above
        ground (e.g. ``z = -2.5`` → 2.5 m above home).
        """
        msg = TrajectorySetpoint()
        msg.timestamp = self._px4_timestamp()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float("nan")] * 3  # let PX4 decide
        msg.acceleration = [float("nan")] * 3
        msg.yaw = float("nan")  # hold current heading
        msg.yawspeed = float("nan")
        self._setpoint_pub.publish(msg)

    def publish_attitude_setpoint(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        thrust: float = 0.5,
        yaw_rate: float = 0.0,
    ) -> None:
        """Publish a ``VehicleAttitudeSetpoint``.

        Parameters
        ----------
        roll, pitch, yaw : float
            Desired Euler angles in **radians** (NED body frame).
            The yaw component is baked into ``q_d``.
        thrust : float
            Normalised collective thrust in ``[0, 1]``.  Mapped to
            ``thrust_body[2] = -thrust`` (NED: body-Z points down).
        yaw_rate : float
            Feed-forward yaw rate in **rad/s** (NED: positive = CW
            from above).  Sent via ``yaw_sp_move_rate`` so PX4's
            rate controller tracks it directly.
        """
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self._px4_timestamp()

        # Euler → quaternion  (ZYX intrinsic = aerospace convention)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        msg.q_d[0] = cr * cp * cy + sr * sp * sy  # w
        msg.q_d[1] = sr * cp * cy - cr * sp * sy  # x
        msg.q_d[2] = cr * sp * cy + sr * cp * sy  # y
        msg.q_d[3] = cr * cp * sy - sr * sp * cy  # z

        # Feed-forward yaw rate — PX4's attitude controller adds
        # this to its own heading-error-based yaw rate command.
        msg.yaw_sp_move_rate = float(yaw_rate)

        # Multicopter thrust: body-Z is "down" in NED, so negative = up
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -float(thrust)

        self._attitude_pub.publish(msg)

    def send_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        """Publish a ``VehicleCommand`` (fire-and-forget)."""
        msg = VehicleCommand()
        msg.timestamp = self._px4_timestamp()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param3 = param3
        msg.param4 = param4
        msg.param5 = param5
        msg.param6 = param6
        msg.param7 = param7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._cmd_pub.publish(msg)

    # ── convenience wrappers ────────────────────────────────

    def arm(self) -> None:
        self.send_vehicle_command(
            _VEHICLE_CMD_ARM_DISARM,
            param1=1.0,
        )

    def disarm(self, force: bool = False) -> None:
        self.send_vehicle_command(
            _VEHICLE_CMD_ARM_DISARM,
            param1=0.0,
            param2=21196.0 if force else 0.0,  # magic float = force
        )

    def takeoff(self, altitude: float = 2.5) -> None:
        """Request takeoff to *altitude* m above home."""
        self.send_vehicle_command(
            _VEHICLE_CMD_NAV_TAKEOFF,
            param7=altitude,
        )

    def set_mode_offboard(self) -> None:
        self.send_vehicle_command(
            _VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=float(_PX4_CUSTOM_MAIN_MODE_OFFBOARD),
        )

    def set_mode_auto_land(self) -> None:
        self.send_vehicle_command(
            _VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=float(_PX4_CUSTOM_MAIN_MODE_AUTO),
            param3=float(_PX4_CUSTOM_SUB_MODE_AUTO_LAND),
        )

    def land(self) -> None:
        self.send_vehicle_command(_VEHICLE_CMD_NAV_LAND)

    # ── internal ────────────────────────────────────────────

    def _px4_timestamp(self) -> int:
        """Microsecond timestamp for PX4 messages."""
        return int(self.get_clock().now().nanoseconds / 1_000)


# ── Module-level singleton management ───────────────────────

_node: Optional[_PX4Cmd] = None
_spin_thread: Optional[threading.Thread] = None


def _ensure_node() -> _PX4Cmd:
    """Lazily create (and background-spin) the ROS 2 command node."""
    global _node, _spin_thread
    if _node is not None:
        return _node

    if not rclpy.ok():
        rclpy.init()

    _node = _PX4Cmd()

    # Spin in a daemon thread so callbacks (VehicleStatus, etc.) keep
    # arriving while the main thread is blocked in reset().
    def _spin() -> None:
        assert _node is not None
        try:
            rclpy.spin(_node)
        except Exception:
            pass

    _spin_thread = threading.Thread(target=_spin, daemon=True)
    _spin_thread.start()
    return _node


def shutdown_node() -> None:
    """Destroy the singleton node (call at final env close)."""
    global _node, _spin_thread
    if _node is not None:
        _node.destroy_node()
        _node = None
    if rclpy.ok():
        rclpy.shutdown()
    _spin_thread = None


# ════════════════════════════════════════════════════════════
#  Public API  (called from PX4GazeboEnv)
# ════════════════════════════════════════════════════════════


def clear_state() -> None:
    """Clear cached VehicleStatus so a fresh episode starts clean."""
    node = _ensure_node()
    node.reset_state()


def wait_for_connection(timeout: float = 30.0) -> bool:
    """Block until we receive at least one ``VehicleStatus`` from PX4."""
    node = _ensure_node()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if node.connected:
            return True
        time.sleep(0.2)
    node.get_logger().warn("Timed out waiting for PX4 VehicleStatus")
    return False


def stream_setpoints_and_offboard(
    n: int = 100,
    rate_hz: float = 50.0,
    x: float = 0.0,
    y: float = 0.0,
    z_enu: float = 2.5,
) -> None:
    """Publish *n* offboard-mode + setpoint pairs at *rate_hz*.

    PX4 requires ``OffboardControlMode`` at ≥ 2 Hz for a short period
    before it will accept the OFFBOARD mode switch.  We send both the
    control-mode heartbeat **and** a position setpoint each iteration.

    Parameters
    ----------
    z_enu : float
        Target altitude in ENU (positive-up).  Internally converted to
        NED for ``TrajectorySetpoint``.
    """
    node = _ensure_node()
    dt = 1.0 / rate_hz
    z_ned = -abs(z_enu)  # ENU → NED
    for _ in range(n):
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(x, y, z_ned)
        time.sleep(dt)


def arm_and_takeoff(
    target_alt: float = 2.5,
    timeout: float = 20.0,
    get_altitude=None,
) -> bool:
    """Arm in OFFBOARD mode and climb to *target_alt* via position setpoint.

    PX4's ``NAV_TAKEOFF`` command is unreliable when issued over DDS
    (motors spin at zero and PX4 auto-disarms after ~10 s).  Instead,
    this function arms while already in OFFBOARD mode and lets the
    position controller fly the drone to the commanded altitude.

    **Prerequisite**: ``switch_to_offboard()`` must have been called
    (and succeeded) *before* this function.

    Parameters
    ----------
    target_alt : float
        Target hover altitude in metres (ENU z, positive-up).
    timeout : float
        Max wall-clock seconds to wait for altitude.
    get_altitude : callable, optional
        A zero-arg callable returning the current altitude (ENU m).
        If provided, blocks until the drone is within 0.5 m of
        *target_alt*.  Otherwise returns after the arm command.

    Returns True if arming succeeded.
    """
    node = _ensure_node()
    z_ned = -abs(target_alt)  # ENU → NED

    # Arm (retry a few times — PX4 may reject while still initialising).
    # Keep publishing setpoints while arming so OFFBOARD doesn't time out.
    for _ in range(20):
        if node.armed:
            break
        node.arm()
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(0.0, 0.0, z_ned)
        time.sleep(0.5)

    if not node.armed:
        node.get_logger().warn("Failed to arm after retries")
        return False

    # Climb via position setpoint — the position controller will
    # spin the motors and ascend to z_ned.
    if get_altitude is not None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            alt = get_altitude()
            if alt >= target_alt - 0.5:
                break
            node.publish_offboard_control_mode(position=True)
            node.publish_setpoint(0.0, 0.0, z_ned)
            time.sleep(0.1)

    return True


def switch_to_offboard(timeout: float = 10.0) -> bool:
    """Switch PX4 to OFFBOARD mode.

    Assumes ``OffboardControlMode`` has been streaming (see
    :func:`stream_setpoints_and_offboard`).  Retries until PX4
    reports ``nav_state == OFFBOARD``.
    """
    node = _ensure_node()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if node.in_offboard:
            return True
        # Keep both the heartbeat and setpoint flowing so PX4
        # accepts the mode switch.
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(0.0, 0.0, -2.5)
        node.set_mode_offboard()
        time.sleep(0.25)
    node.get_logger().warn("Failed to switch to OFFBOARD")
    return False


def publish_offboard_heartbeat(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 2.5,
) -> None:
    """Publish an ``OffboardControlMode`` heartbeat to keep PX4 in
    OFFBOARD mode.

    This is **intentionally separate** from the trajectory setpoint.
    Call this every ``env.step()`` so PX4 never times out of offboard,
    even if the RL policy's setpoint publication is slightly delayed.

    A ``TrajectorySetpoint`` hold at the given position is also published
    as a safety fallback — the RL policy's own setpoints (via
    ``_apply_action``) will override it within the same sim step.
    """
    node = _ensure_node()
    z_ned = -abs(z)
    node.publish_offboard_control_mode(position=True)
    node.publish_setpoint(x, y, z_ned)


def force_disarm() -> None:
    """Immediately force-disarm the vehicle (no landing)."""
    node = _ensure_node()
    node.disarm(force=True)


def wait_for_disarm(timeout: float = 5.0) -> bool:
    """Block until PX4 reports disarmed."""
    node = _ensure_node()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not node.armed:
            return True
        node.disarm(force=True)
        time.sleep(0.2)
    return not node.armed


def publish_attitude_command(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    thrust: float = 0.5,
    yaw_rate: float = 0.0,
) -> None:
    """Publish an attitude setpoint to PX4.

    Also publishes an ``OffboardControlMode`` heartbeat with
    ``attitude=True`` so PX4 stays in OFFBOARD attitude mode.

    Parameters
    ----------
    roll, pitch, yaw : float
        Desired Euler angles in radians (NED).
    thrust : float
        Normalised collective thrust ``[0, 1]``.
    yaw_rate : float
        Feed-forward yaw rate in rad/s (NED).  Sent via
        ``yaw_sp_move_rate`` for direct rate tracking.
    """
    node = _ensure_node()
    node.publish_offboard_control_mode(attitude=True, position=False)
    node.publish_attitude_setpoint(roll, pitch, yaw, thrust, yaw_rate=yaw_rate)


def publish_offboard_attitude_heartbeat(
    thrust: float = 0.5,
) -> None:
    """Keep PX4 in OFFBOARD **attitude** mode.

    Publishes ``OffboardControlMode(attitude=True)`` plus a
    neutral attitude setpoint (hover) as a safety fallback.
    The RL policy's own ``publish_attitude_command`` will override
    this within the same sim step.
    """
    node = _ensure_node()
    node.publish_offboard_control_mode(attitude=True, position=False)
    node.publish_attitude_setpoint(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        thrust=thrust,
    )


# ════════════════════════════════════════════════════════
#  Sim-stepped reset helpers  (no wall-clock sleeps)
# ════════════════════════════════════════════════════════
#
# These variants accept a ``step_fn(n)`` callback that advances
# the simulator by *n* physics steps **deterministically**.  All
# waiting is done by stepping the sim forward (fast, < 1 ms per
# batch) instead of ``time.sleep()`` which wastes wall-clock time.
#
# This turns a 60–130 s reset into ~2–5 s wall-clock.
# ════════════════════════════════════════════════════════


def is_connected() -> bool:
    """True if at least one VehicleStatus has been received."""
    return _ensure_node().connected


def is_armed() -> bool:
    """True if PX4 reports armed."""
    return _ensure_node().armed


def is_in_offboard() -> bool:
    """True if PX4 nav_state == OFFBOARD."""
    return _ensure_node().in_offboard


def stepped_wait_for_connection(
    step_fn,
    steps_per_iter: int = 50,
    max_iters: int = 300,
) -> bool:
    """Wait for PX4 DDS connection by stepping the sim forward.

    Each iteration advances ``steps_per_iter`` physics steps (= 0.2 s
    sim-time at 0.004 s step size) instead of wall-clock sleeping.
    Total max wait: 300 × 50 × 0.004 = 60 s sim-time.
    """
    node = _ensure_node()
    for _ in range(max_iters):
        if node.connected:
            return True
        step_fn(steps_per_iter)
    node.get_logger().warn("stepped_wait_for_connection timed out")
    return False


def stepped_stream_setpoints(
    step_fn,
    z_enu: float = 2.5,
    sim_seconds: float = 2.0,
    ticks_per_publish: int = 5,
    step_size: float = 0.004,
) -> None:
    """Stream OffboardControlMode + position setpoints while stepping.

    PX4 requires ``OffboardControlMode`` at ≥ 2 Hz for ≥ 2 s of
    sim-time before it accepts the OFFBOARD mode switch.  This
    function publishes at **50 Hz** (every ``ticks_per_publish``
    physics steps = every 0.02 s sim-time) to keep PX4 happy.

    With defaults: 2.0 s / (5 × 0.004) = 100 publishes, each
    separated by 5 physics steps.  Wall-clock: ~0.1 s.
    """
    node = _ensure_node()
    z_ned = -abs(z_enu)
    n_publishes = int(sim_seconds / (ticks_per_publish * step_size))
    for _ in range(n_publishes):
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(0.0, 0.0, z_ned)
        step_fn(ticks_per_publish)


def stepped_switch_to_offboard(
    step_fn,
    z_enu: float = 2.5,
    ticks_per_publish: int = 5,
    max_sim_seconds: float = 10.0,
    step_size: float = 0.004,
) -> bool:
    """Switch to OFFBOARD mode, stepping sim between retries.

    Publishes setpoints at 50 Hz (every ``ticks_per_publish`` physics
    steps) to avoid PX4's offboard timeout, and sends the mode-switch
    command every 0.2 s of sim-time.
    """
    node = _ensure_node()
    z_ned = -abs(z_enu)
    max_iters = int(max_sim_seconds / (ticks_per_publish * step_size))
    mode_cmd_interval = max(1, int(0.2 / (ticks_per_publish * step_size)))
    for i in range(max_iters):
        if node.in_offboard:
            return True
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(0.0, 0.0, z_ned)
        if i % mode_cmd_interval == 0:
            node.set_mode_offboard()
        step_fn(ticks_per_publish)
    node.get_logger().warn("stepped_switch_to_offboard timed out")
    return False


def stepped_arm_and_takeoff(
    step_fn,
    target_alt: float = 2.5,
    get_altitude=None,
    ticks_per_publish: int = 5,
    step_size: float = 0.004,
    arm_timeout_s: float = 10.0,
    climb_timeout_s: float = 20.0,
) -> bool:
    """Arm and fly to *target_alt* using sim-stepping.

    Setpoints are published every ``ticks_per_publish`` physics steps
    (default: every 5 ticks = 0.02 s sim-time = 50 Hz) so PX4's
    offboard heartbeat never times out.

    The arm command is sent every 0.5 s sim-time until PX4 confirms
    armed.  After arming, the position setpoint drives the drone to
    ``target_alt``.

    Parameters
    ----------
    step_fn : callable(int)
        Advances the simulator by N physics steps.
    ticks_per_publish : int
        Physics steps between each setpoint publish (5 = 50 Hz).
    arm_timeout_s : float
        Max sim-time seconds to wait for arming.
    climb_timeout_s : float
        Max sim-time seconds to wait for altitude.
    """
    node = _ensure_node()
    z_ned = -abs(target_alt)
    dt_per_iter = ticks_per_publish * step_size  # 0.02 s default

    # ── Arm (retry, keep streaming setpoints at 50 Hz) ─────
    arm_iters = int(arm_timeout_s / dt_per_iter)
    arm_cmd_interval = max(1, int(0.5 / dt_per_iter))  # arm cmd every 0.5 s sim
    for i in range(arm_iters):
        if node.armed:
            break
        node.publish_offboard_control_mode(position=True)
        node.publish_setpoint(0.0, 0.0, z_ned)
        if i % arm_cmd_interval == 0:
            node.arm()
        step_fn(ticks_per_publish)

    if not node.armed:
        node.get_logger().warn("stepped_arm_and_takeoff: failed to arm")
        return False

    # ── Climb to target altitude ────────────────────────────
    if get_altitude is not None:
        climb_iters = int(climb_timeout_s / dt_per_iter)
        for _ in range(climb_iters):
            alt = get_altitude()
            if alt >= target_alt - 0.5:
                break
            node.publish_offboard_control_mode(position=True)
            node.publish_setpoint(0.0, 0.0, z_ned)
            step_fn(ticks_per_publish)

    return True


def stepped_stream_attitude(
    step_fn,
    sim_seconds: float = 2.0,
    ticks_per_publish: int = 5,
    step_size: float = 0.004,
    hover_thrust: float = 0.5,
) -> None:
    """Stream **attitude-mode** offboard heartbeats while stepping.

    Publishes ``OffboardControlMode(attitude=True)`` + a neutral
    ``VehicleAttitudeSetpoint`` (level, hover thrust) at 50 Hz.
    PX4 needs ≥ 2 Hz for ≥ 2 s sim-time before it accepts OFFBOARD.
    """
    node = _ensure_node()
    n_publishes = int(sim_seconds / (ticks_per_publish * step_size))
    for _ in range(n_publishes):
        node.publish_offboard_control_mode(attitude=True, position=False)
        node.publish_attitude_setpoint(
            roll=0.0, pitch=0.0, yaw=0.0, thrust=hover_thrust,
        )
        step_fn(ticks_per_publish)


def stepped_offboard_arm(
    step_fn,
    ticks_per_publish: int = 5,
    step_size: float = 0.004,
    offboard_timeout_s: float = 10.0,
    arm_timeout_s: float = 10.0,
    hover_thrust: float = 0.5,
) -> bool:
    """Switch to OFFBOARD **attitude** mode and arm.

    No climb phase — the policy takes over from ground level.
    Publishes neutral attitude setpoints at 50 Hz throughout to
    keep PX4's offboard heartbeat alive.

    Returns True if both offboard switch and arming succeeded.
    """
    node = _ensure_node()
    dt_per_iter = ticks_per_publish * step_size

    def _publish_attitude_heartbeat() -> None:
        node.publish_offboard_control_mode(attitude=True, position=False)
        node.publish_attitude_setpoint(
            roll=0.0, pitch=0.0, yaw=0.0, thrust=hover_thrust,
        )

    # ── Switch to OFFBOARD ──────────────────────────────────
    offboard_iters = int(offboard_timeout_s / dt_per_iter)
    mode_cmd_interval = max(1, int(0.2 / dt_per_iter))  # every 0.2 s sim
    for i in range(offboard_iters):
        if node.in_offboard:
            break
        _publish_attitude_heartbeat()
        if i % mode_cmd_interval == 0:
            node.set_mode_offboard()
        step_fn(ticks_per_publish)

    if not node.in_offboard:
        node.get_logger().warn("stepped_offboard_arm: failed to enter OFFBOARD")
        return False

    # ── Arm ─────────────────────────────────────────────────
    arm_iters = int(arm_timeout_s / dt_per_iter)
    arm_cmd_interval = max(1, int(0.5 / dt_per_iter))  # every 0.5 s sim
    for i in range(arm_iters):
        if node.armed:
            break
        _publish_attitude_heartbeat()
        if i % arm_cmd_interval == 0:
            node.arm()
        step_fn(ticks_per_publish)

    if not node.armed:
        node.get_logger().warn("stepped_offboard_arm: failed to arm")
        return False

    return True


def stepped_wait_for_disarm(
    step_fn,
    steps_per_iter: int = 50,
    max_iters: int = 50,
) -> bool:
    """Wait for PX4 to disarm by stepping sim forward."""
    node = _ensure_node()
    for _ in range(max_iters):
        if not node.armed:
            return True
        node.disarm(force=True)
        step_fn(steps_per_iter)
    return not node.armed


def land_and_disarm(timeout: float = 15.0) -> bool:
    """Switch to AUTO.LAND and wait for disarm (best-effort)."""
    node = _ensure_node()
    node.set_mode_auto_land()
    node.land()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not node.armed:
            return True
        time.sleep(0.3)
    # Force disarm
    node.disarm(force=True)
    time.sleep(0.5)
    return not node.armed


# ════════════════════════════════════════════════════════════
#  PX4 NSH console commands  (via tmux)
# ════════════════════════════════════════════════════════════

# Default tmux target — overridden by SubprocVecEnv workers
# to point at their per-worker tmux session.
_DEFAULT_TMUX_TARGET = "px4sim:sim.1"


def nsh_command(
    cmd: str,
    tmux_target: str | None = None,
) -> None:
    """Send an NSH shell command to the PX4 SITL console.

    PX4 SITL runs inside a tmux pane.  This function uses
    ``tmux send-keys`` to inject a command into that pane's
    NuttShell (nsh) prompt.

    Parameters
    ----------
    cmd : str
        The NSH command to run (e.g. ``"ekf2 stop"``).
    tmux_target : str, optional
        tmux target pane (``session:window.pane``).
        Defaults to ``_DEFAULT_TMUX_TARGET``.
    """
    import subprocess as _sp

    target = tmux_target or _DEFAULT_TMUX_TARGET
    _sp.run(
        ["tmux", "send-keys", "-t", target, cmd, "Enter"],
        check=False,
        capture_output=True,
    )


def restart_ekf2(
    tmux_target: str | None = None,
    settle_time: float = 0.5,
) -> None:
    """Stop and restart EKF2 so all filter state is wiped.

    Call this during ``env.reset()`` after teleporting the drone
    so the estimator starts fresh with no stale covariance,
    innovation history, or fault flags from the previous episode.
    """
    target = tmux_target or _DEFAULT_TMUX_TARGET
    nsh_command("ekf2 stop", tmux_target=target)
    time.sleep(settle_time)
    nsh_command("ekf2 start", tmux_target=target)
    time.sleep(settle_time)


def restart_px4(
    tmux_target: str | None = None,
    settle_time: float = 0.05,
) -> None:
    """Stop and restart all key PX4 modules for a clean episode.

    Unlike ``restart_ekf2`` which only wipes the estimator, this
    stops and restarts **every** module that accumulates state
    between episodes: estimator, commander, controllers, navigator,
    etc.  Gazebo is unaffected — only PX4-internal modules are
    cycled.

    Call this during ``env.reset()`` after teleporting the drone.

    .. note::
       The sleeps here are intentionally minimal (just enough for
       tmux send-keys to be processed).  The caller should use
       sim-stepping to give PX4 the sim-time it needs to
       re-initialise modules and converge EKF2.
    """
    # Modules listed in dependency order (stop in reverse,
    # start in forward order).
    modules = [
        "flight_mode_manager",
        # "navigator",
        # "mc_pos_control",
        # "mc_att_control",
        # "mc_rate_control",
        # "mc_hover_thrust_estimator",
        # "land_detector",
        # "commander",
        "ekf2",
    ]

    # ── Stop all modules ────────────────────────────────────
    target = tmux_target or _DEFAULT_TMUX_TARGET
    for mod in modules:
        nsh_command(f"{mod} stop", tmux_target=target)
        time.sleep(0.02)  # minimal: just enough for tmux delivery

    time.sleep(settle_time)

    # ── Start all modules (reverse order) ───────────────────
    for mod in reversed(modules):
        nsh_command(f"{mod} start", tmux_target=target)
        time.sleep(0.02)  # minimal: just enough for tmux delivery

    time.sleep(settle_time)
