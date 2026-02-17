#!/usr/bin/env bash
# ============================================================
# launch_sim.sh
# Launches the full PX4 SITL simulation stack in a single tmux
# session with lockstep and simulation-time enabled:
#
#   Pane 0 — PX4 SITL  (Gazebo Harmonic, lockstep)
#   Pane 1 — Micro XRCE-DDS Agent  (PX4 ↔ ROS 2 bridge)
#   Pane 2 — MAVROS 2  (MAVLink ↔ ROS 2, /use_sim_time)
#
# Usage:
#   ./launch_sim.sh                        # defaults: gz_x500, domain 0, default world
#   ./launch_sim.sh gz_x500_depth 1         # custom model & domain
#   ./launch_sim.sh gz_x500 0 baylands      # custom model, domain & world/map
#
# Multi-instance:
#   PX4_INSTANCE=0 → starts Gazebo server + PX4 (instance 0)
#   PX4_INSTANCE=1 → PX4_GZ_STANDALONE, connects to existing Gazebo
#   All ports (DDS, MAVLink) are offset by PX4_INSTANCE automatically.
#
# Requirements: tmux, PX4 Autopilot, MicroXRCEAgent, ROS 2,
#               MAVROS 2, Gazebo Harmonic
# ============================================================
set -euo pipefail

# ── Configurable parameters ────────────────────────────────
PX4_MODEL="${1:-gz_x500_mono_cam}"               # PX4 SITL airframe
ROS_DOMAIN_ID="${2:-0}"                         # ROS 2 domain isolation
PX4_GZ_WORLD="${3:-tugbot_depot}"                    # Gazebo world / map name
PX4_HOME="${PX4_HOME:-/opt/PX4-Autopilot}"      # PX4 source tree
PX4_PARAMS_FILE="${PX4_PARAMS_FILE:-/workspace/px4.params}"  # QGC text format params
SESSION="${SESSION:-px4sim}"                    # tmux session name
SKIP_VIZ="${SKIP_VIZ:-0}"                      # 1 = skip RViz/QGC window

# ── Multi-instance: PX4_INSTANCE offsets all ports ─────────
PX4_INSTANCE="${PX4_INSTANCE:-0}"
DDS_PORT=$((8888 + PX4_INSTANCE))                # XRCE-DDS UDP port
# PX4 MAVLink offboard port = 14540 + px4_instance (set by PX4 rcS)
FCU_URL="udp://:$((14540 + PX4_INSTANCE))@127.0.0.1:$((14580 + PX4_INSTANCE))"  # MAVROS ↔ PX4 link
# PX4 multi-instance MAVLink system ID = instance + 1
MAV_SYS_ID=$((PX4_INSTANCE + 1))

# ── Lockstep & simulation-time knobs ──────────────────────
#  PX4_SIM_SPEED_FACTOR=1 is real-time; >1 faster, <1 slower.
#  Lockstep is the PX4 default with Gazebo; we export the var
#  explicitly for clarity and to guard against overrides.
PX4_SIM_SPEED_FACTOR="${PX4_SIM_SPEED_FACTOR:-3}"

# ── Per-instance working directory ────────────────────────
# Each PX4 instance needs its own rootfs to avoid parameter
# file and lock collisions.
BUILD_DIR="${PX4_HOME}/build/px4_sitl_default"
WORKING_DIR="${BUILD_DIR}/instance_${PX4_INSTANCE}"
mkdir -p "${WORKING_DIR}"
# PX4's px4-rc.gzsim sources ./gz_env.sh relative to CWD
ln -sf "${BUILD_DIR}/rootfs/gz_env.sh" "${WORKING_DIR}/gz_env.sh" 2>/dev/null || true

# ── Preamble sourced inside every tmux pane ───────────────
read -r -d '' PREAMBLE <<'SHELL' || true
source /opt/ros/humble/setup.bash
[[ -f /opt/ros_gz_ws/install/setup.bash ]]  && source /opt/ros_gz_ws/install/setup.bash
[[ -f /opt/px4_ros_ws/install/setup.bash ]] && source /opt/px4_ros_ws/install/setup.bash
[[ -f /workspace/install/setup.bash ]]      && source /workspace/install/setup.bash
SHELL

# ── Colour helpers ─────────────────────────────────────────
info()  { echo -e "\033[1;36m[launch_sim]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[launch_sim]\033[0m $*"; }
err()   { echo -e "\033[1;31m[launch_sim]\033[0m $*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────
for cmd in tmux MicroXRCEAgent; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd not found on PATH. Aborting."
        exit 1
    fi
done

if [[ ! -d "$PX4_HOME" ]]; then
    err "PX4_HOME ($PX4_HOME) does not exist. Aborting."
    exit 1
fi

# Kill any previous session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

info "Starting simulation stack …"
info "  PX4 instance:       $PX4_INSTANCE"
info "  Model:              $PX4_MODEL"
info "  World / map:        $PX4_GZ_WORLD"
info "  ROS_DOMAIN_ID:      $ROS_DOMAIN_ID"
info "  DDS port:           $DDS_PORT"
info "  FCU URL:            $FCU_URL"
info "  MAV_SYS_ID:         $MAV_SYS_ID"
info "  MAVROS namespace:   /px4_${PX4_INSTANCE}/mavros"
info "  Working dir:        $WORKING_DIR"
info "  Lockstep:           ON (PX4 default w/ Gazebo)"
info "  PX4_SIM_SPEED_FACTOR: $PX4_SIM_SPEED_FACTOR"
info "  Params file:         $PX4_PARAMS_FILE"
info "  /use_sim_time:      true (MAVROS + ROS nodes)"
info "  SKIP_VIZ:           $SKIP_VIZ"
if [ -n "${PX4_GZ_STANDALONE:-}" ] || [ "$PX4_INSTANCE" -gt 0 ]; then
    info "  GZ standalone:      YES (connecting to existing Gazebo)"
fi
echo ""

# ── Parse px4.params → PX4_PARAM_* env vars ───────────────
# PX4 rcS reads env vars named PX4_PARAM_<name> and applies
# them via `param set` after loading defaults.  This is the
# cleanest way to override SITL parameters.
#
# Format: "vehicle-id  component-id  name  value  type" (tab-sep)
# We skip comment/blank lines and export each param.
# ────────────────────────────────────────────────────────────
PX4_PARAM_EXPORTS=""
PX4_PARAM_EXPORTS+="export PX4_PARAM_SIM_BAT_ENABLE=0"$'\n'
PX4_PARAM_EXPORTS+="export PX4_PARAM_SYS_HAS_MAG=1"$'\n'
info "Forcing SIM_BAT_ENABLE=0 (battery simulation disabled)"
info "Forcing SYS_HAS_MAG=0 (magnetometer disabled)"

# ============================================================
# Pane 1 — PX4 SITL + Gazebo Harmonic  (lockstep)
# ============================================================
# Instance 0: starts Gazebo server + spawns model.
# Instance N>0: PX4_GZ_STANDALONE=1, spawns into existing Gz.
# Each instance gets its own working dir and -i flag so that
# MAVLink, DDS, and lock-file ports don't collide.
# ============================================================

# Build GZ_STANDALONE export.
# If PX4_GZ_STANDALONE is already set in the environment (e.g. by
# launch_parallel.sh), honour it.  Otherwise, default to standalone
# for instances > 0.
if [ -n "${PX4_GZ_STANDALONE:-}" ]; then
    GZ_STANDALONE_EXPORT="export PX4_GZ_STANDALONE=${PX4_GZ_STANDALONE}"
elif [ "${PX4_INSTANCE}" -gt 0 ]; then
    GZ_STANDALONE_EXPORT="export PX4_GZ_STANDALONE=1"
else
    GZ_STANDALONE_EXPORT="unset PX4_GZ_STANDALONE"
fi

# Offset spawn position so drones don't stack on top of each other
# Instance 0 → (0,0), 1 → (0,3), 2 → (0,6), …
SPAWN_Y=$((PX4_INSTANCE * 3))
MODEL_POSE="0,${SPAWN_Y},0,0,0,0"

tmux new-session -d -s "$SESSION" -n sim \
    -x "$(tput cols)" -y "$(tput lines)" \; \
    send-keys "\
${PREAMBLE}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
export PX4_SIM_MODEL=${PX4_MODEL}
export PX4_SIM_SPEED_FACTOR=${PX4_SIM_SPEED_FACTOR}
export PX4_GZ_WORLD=${PX4_GZ_WORLD}
export PX4_GZ_MODEL_POSE=${MODEL_POSE}
${GZ_STANDALONE_EXPORT}
export HEADLESS=${HEADLESS:-1}
export PX4_UXRCE_DDS_PORT=${DDS_PORT}
export PX4_UXRCE_DDS_NS=px4_${PX4_INSTANCE}
${PX4_PARAM_EXPORTS}
# Source Gazebo environment paths (models, worlds, plugins)
source ${BUILD_DIR}/rootfs/gz_env.sh
rm -f ${WORKING_DIR}/parameters.bson
rm -f ${WORKING_DIR}/parameters_backup.bson
echo '──────────────────────────────────────────────────'
echo '  🛩  PX4 SITL  instance=${PX4_INSTANCE}  (lockstep + Gazebo Harmonic)'
echo '──────────────────────────────────────────────────'
cd ${WORKING_DIR} && ${BUILD_DIR}/bin/px4 -i ${PX4_INSTANCE} -w ${WORKING_DIR} ${BUILD_DIR}/etc
" Enter

# Give PX4 / Gazebo a moment to allocate ports
sleep 2

# ============================================================
# Pane 2 — Micro XRCE-DDS Agent
# ============================================================
tmux split-window -t "$SESSION" -v \; \
    send-keys "\
${PREAMBLE}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
echo '──────────────────────────────────────────────────'
echo '  📡  Micro XRCE-DDS Agent  (UDP port $DDS_PORT, instance ${PX4_INSTANCE})'
echo '──────────────────────────────────────────────────'
MicroXRCEAgent udp4 -p ${DDS_PORT}
" Enter

# ============================================================
# Pane 3 — ros_gz_bridge  (Gazebo /clock → ROS 2 /clock)
# ============================================================
# Gazebo publishes clock only on gz-transport; ROS 2 nodes
# using use_sim_time need a /clock topic on the ROS 2 graph.
# This bridge forwards the Gazebo world clock to ROS 2.
#
# Only ONE bridge is needed per ROS_DOMAIN_ID — all instances
# share the same Gazebo world, so we only start it for instance 0.
# ============================================================
if [ "${PX4_INSTANCE}" -eq 0 ]; then
tmux split-window -t "$SESSION" -v \; \
    send-keys "\
${PREAMBLE}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
echo '──────────────────────────────────────────────────'
echo '  ⏱  ros_gz_bridge  (Gazebo clock → ROS 2 /clock)'
echo '──────────────────────────────────────────────────'
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
" Enter
fi

# ============================================================
# Pane 4 — MAVROS 2  (with /use_sim_time:=true)
# ============================================================
# use_sim_time makes MAVROS subscribe to /clock published by
# Gazebo so that all TF stamps, message headers, and timeout
# logic use simulation time — essential for lockstep.
#
# Multi-instance: each MAVROS gets its own ROS 2 namespace
# (px4_<i>/mavros) and targets the correct MAVLink system ID
# (instance + 1).  This prevents topic collisions when all
# instances share one ROS_DOMAIN_ID.
# ============================================================
tmux split-window -t "$SESSION" -v \; \
    send-keys "\
${PREAMBLE}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
echo '──────────────────────────────────────────────────'
echo '  🛰  MAVROS 2  (instance ${PX4_INSTANCE}, ns=px4_${PX4_INSTANCE}/mavros, sysid=${MAV_SYS_ID})'
echo '──────────────────────────────────────────────────'
echo 'Waiting 5 s for PX4 & DDS to settle …'
sleep 5
ros2 launch mavros px4.launch \
    fcu_url:=${FCU_URL} \
    tgt_system:=${MAV_SYS_ID} \
    namespace:=px4_${PX4_INSTANCE}/mavros \
    use_sim_time:=true
" Enter

# ============================================================
# Window 2 — visualisation tools (two panes)
#   Pane 1: RViz2  (use_sim_time + sim.rviz)
#   Pane 2: QGC
# Skipped when SKIP_VIZ=1 (e.g. parallel training)
# ============================================================
if [ "${SKIP_VIZ}" != "1" ]; then

tmux new-window -t "$SESSION" -n viz

# Pane 1 — RViz2
tmux send-keys -t "$SESSION":viz.1 "\
${PREAMBLE}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
echo '──────────────────────────────────────────────────'
echo '  🧭  RViz2  (use_sim_time:=true, sim.rviz)'
echo '──────────────────────────────────────────────────'
rviz2 -d /workspace/src/configs/sim.rviz --ros-args -p use_sim_time:=true
" Enter

# Pane 2 — QGC
tmux split-window -t "$SESSION":viz -v \; \
    send-keys "\
${PREAMBLE}
qgc
" Enter

tmux select-layout -t "$SESSION":viz even-vertical

fi  # SKIP_VIZ

# ── Tidy the layout ───────────────────────────────────────
tmux select-layout -t "$SESSION":sim even-vertical

info "tmux session '${SESSION}' is running."
info "Attach with:  tmux attach -t ${SESSION}"
info "Kill with:    tmux kill-session -t ${SESSION}"
echo ""

# If we're in an interactive terminal, auto-attach
if [[ -t 0 ]]; then
    : #tmux attach -t "$SESSION"
fi
