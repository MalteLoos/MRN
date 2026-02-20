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
#   - ROS_DOMAIN_ID:  10, 11, …
#
# Workers share the same Gazebo world file but run in
# completely isolated processes.
# ============================================================
set -euo pipefail

NUM_ENVS="${1:-2}"
PX4_GZ_WORLD="${2:-tugbot_depot}"
BASE_DOMAIN_ID="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

info()  { echo -e "\033[1;36m[parallel]\033[0m $*"; }

info "Launching ${NUM_ENVS} parallel sim stacks …"
info "  World:          ${PX4_GZ_WORLD}"
info "  Base domain ID: ${BASE_DOMAIN_ID}"
echo ""

for i in $(seq 0 $((NUM_ENVS - 1))); do
    DOMAIN_ID=$((BASE_DOMAIN_ID + i))
    SESSION="px4sim_w${i}"

    info "  Worker ${i}: session=${SESSION}  ROS_DOMAIN_ID=${DOMAIN_ID}"

    # Kill any existing session with this name
    tmux kill-session -t "${SESSION}" 2>/dev/null || true

    # Launch using the main script with overridden SESSION and DOMAIN_ID
    SESSION="${SESSION}" \
    ROS_DOMAIN_ID="${DOMAIN_ID}" \
    PX4_GZ_WORLD="${PX4_GZ_WORLD}" \
    bash "${SCRIPT_DIR}/launch_sim.sh" \
        "${PX4_MODEL:-gz_x500_mono_cam}" \
        "${DOMAIN_ID}" \
        "${PX4_GZ_WORLD}"

    # Stagger launches to avoid port conflicts
    if [ $i -lt $((NUM_ENVS - 1)) ]; then
        info "  Waiting 10s before next launch …"
        sleep 10
    fi
done

echo ""
info "All ${NUM_ENVS} sim stacks launched."
info ""
info "Train with:"
info "  python3 src/train_hover.py --num-envs ${NUM_ENVS}"
info ""
info "Kill all with:"
for i in $(seq 0 $((NUM_ENVS - 1))); do
    info "  tmux kill-session -t px4sim_w${i}"
done
