#!/usr/bin/env bash
# ============================================================
# launch_parallel.sh
# Launches N independent PX4 SITL simulation stacks, each in
# its own tmux session with a unique ROS_DOMAIN_ID, for
# parallel training with SubprocVecEnv.
#
# Usage:
#   ./launch_parallel.sh 2               # 2 parallel sims
#   ./launch_parallel.sh 4 tugbot_depot   # 4 sims, custom world
#
# Each sim stack gets:
#   - tmux session:   px4sim_w0, px4sim_w1, …
#   - ROS_DOMAIN_ID:  shared (topic isolation via DDS namespace)
#   - PX4_INSTANCE:   0, 1, …  (offsets DDS/MAVLink ports)
#
# A dedicated Gazebo server is launched first (session px4sim_gz).
# All PX4 instances connect to it via PX4_GZ_STANDALONE=1.
# QGC and RViz are only started once (with instance 0).
# ============================================================
set -euo pipefail

NUM_ENVS="${1:-2}"
PX4_GZ_WORLD="${2:-tugbot_depot}"
BASE_DOMAIN_ID="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

info()  { echo -e "\033[1;36m[parallel]\033[0m $*"; }
err()   { echo -e "\033[1;31m[parallel]\033[0m $*" >&2; }

GZ_SESSION="px4sim_gz"
PX4_MODEL="${PX4_MODEL:-gz_x500_mono_cam}"

info "Launching ${NUM_ENVS} parallel sim stacks …"
info "  World:          ${PX4_GZ_WORLD}"
info "  Base domain ID: ${BASE_DOMAIN_ID}"
echo ""

# ── Source the same preamble that launch_sim.sh uses ───────
read -r -d '' PREAMBLE <<'SHELL' || true
source /opt/ros/humble/setup.bash
[[ -f /opt/ros_gz_ws/install/setup.bash ]]  && source /opt/ros_gz_ws/install/setup.bash
[[ -f /opt/px4_ros_ws/install/setup.bash ]] && source /opt/px4_ros_ws/install/setup.bash
[[ -f /workspace/install/setup.bash ]]      && source /workspace/install/setup.bash
SHELL

# ============================================================
# Step 1 — Start the Gazebo server explicitly (one instance)
# ============================================================
# By starting Gazebo separately, we guarantee the world is
# running before any PX4 instance tries to connect.  All PX4
# instances then use PX4_GZ_STANDALONE=1.
# ============================================================
BUILD_DIR="/opt/PX4-Autopilot/build/px4_sitl_default"

tmux kill-session -t "${GZ_SESSION}" 2>/dev/null || true
info "Starting Gazebo server (session: ${GZ_SESSION}) …"

tmux new-session -d -s "${GZ_SESSION}" -n gz \
    -x "$(tput cols)" -y "$(tput lines)" \; \
    send-keys "\
${PREAMBLE}
source ${BUILD_DIR}/rootfs/gz_env.sh
export HEADLESS=${HEADLESS:-1}
echo '──────────────────────────────────────────────────'
echo '  🌍  Gazebo Harmonic  (world: ${PX4_GZ_WORLD})'
echo '──────────────────────────────────────────────────'
gz sim --verbose=1 -r -s \${PX4_GZ_WORLDS}/${PX4_GZ_WORLD}.sdf
" Enter

# Wait for Gazebo to be ready (scene/info service available)
info "Waiting for Gazebo world '${PX4_GZ_WORLD}' to become ready …"
for attempt in $(seq 1 60); do
    if gz service -i --service "/world/${PX4_GZ_WORLD}/scene/info" 2>&1 | grep -q "Service providers"; then
        info "  Gazebo world ready (attempt ${attempt})"
        break
    fi
    if [ "$attempt" -eq 60 ]; then
        err "Gazebo world did not become ready after 60 s. Aborting."
        exit 1
    fi
    sleep 1
done

# ============================================================
# Step 2 — Launch PX4 SITL instances (all in standalone mode)
# ============================================================
for i in $(seq 0 $((NUM_ENVS - 1))); do
    SESSION="px4sim_w${i}"

    info "  Worker ${i}: session=${SESSION}  ROS_DOMAIN_ID=${BASE_DOMAIN_ID}  PX4_INSTANCE=${i}"

    # Kill any existing session with this name
    tmux kill-session -t "${SESSION}" 2>/dev/null || true

    # All instances connect to the already-running Gazebo server.
    # Only instance 0 gets the viz window (RViz + QGC); the rest
    # skip it (QGC can only run once, and one RViz is enough).
    if [ "$i" -eq 0 ]; then
        SKIP_VIZ_FLAG=0
    else
        SKIP_VIZ_FLAG=1
    fi

    SESSION="${SESSION}" \
    ROS_DOMAIN_ID="${BASE_DOMAIN_ID}" \
    PX4_GZ_WORLD="${PX4_GZ_WORLD}" \
    PX4_INSTANCE="${i}" \
    PX4_GZ_STANDALONE=1 \
    SKIP_VIZ="${SKIP_VIZ_FLAG}" \
    bash "${SCRIPT_DIR}/launch_sim.sh" \
        "${PX4_MODEL}" \
        "${BASE_DOMAIN_ID}" \
        "${PX4_GZ_WORLD}"

    # Give PX4 time to spawn its model in Gazebo before the next
    if [ $i -lt $((NUM_ENVS - 1)) ]; then
        info "  Waiting 10s before next instance …"
        sleep 10
    fi
done

echo ""
info "All ${NUM_ENVS} sim stacks launched."
info ""
info "All instances share ROS_DOMAIN_ID=${BASE_DOMAIN_ID}"
info "Topics are namespaced: /px4_0/fmu/..., /px4_1/fmu/..."
info ""
info "Train with:"
info "  python3 src/train_hover.py --num-envs ${NUM_ENVS}"
info ""
info "Kill all with:"
info "  tmux kill-session -t ${GZ_SESSION}"
for i in $(seq 0 $((NUM_ENVS - 1))); do
    info "  tmux kill-session -t px4sim_w${i}"
done
