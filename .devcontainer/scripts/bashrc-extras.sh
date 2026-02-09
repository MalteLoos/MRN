#!/usr/bin/env bash
# ============================================================
# Extra shell configuration sourced from ~/.bashrc
# ============================================================

# ── ROS 2 Humble ───────────────────────────────────────────
source /opt/ros/humble/setup.bash

# ── ros_gz bridge workspace ────────────────────────────────
if [[ -f /opt/ros_gz_ws/install/setup.bash ]]; then
    source /opt/ros_gz_ws/install/setup.bash
fi

# ── PX4 ROS 2 messages workspace ──────────────────────────
if [[ -f /opt/px4_ros_ws/install/setup.bash ]]; then
    source /opt/px4_ros_ws/install/setup.bash
fi

# ── User ROS 2 workspace (built inside the container) ─────
if [[ -f /workspace/install/setup.bash ]]; then
    source /workspace/install/setup.bash
fi

# ── PX4 ────────────────────────────────────────────────────
export PX4_HOME="${PX4_HOME:-/opt/PX4-Autopilot}"

# ── Gazebo Harmonic ────────────────────────────────────────
export GZ_VERSION=harmonic
# Add PX4 Gazebo models / worlds / plugins
export GZ_SIM_RESOURCE_PATH="${PX4_HOME}/Tools/simulation/gz/models:${PX4_HOME}/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH:-}"

# ── CUDA ───────────────────────────────────────────────────
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Colcon defaults ────────────────────────────────────────
export COLCON_DEFAULTS_FILE=""
export _colcon_cd_root="/workspace"

# ── Handy aliases ──────────────────────────────────────────
alias cb='cd /workspace && colcon build --symlink-install'
alias cbt='cd /workspace && colcon build --symlink-install && colcon test'
alias sr='source /workspace/install/setup.bash'
alias px4sitl='cd ${PX4_HOME} && make px4_sitl gz_x500'
alias ddsagent='MicroXRCEAgent udp4 -p 8888'
alias tb='tensorboard --logdir /workspace/logs --bind_all'
alias mavros='ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557'

# ── Colcon tab-completion ──────────────────────────────────
if command -v register-python-argcomplete3 &>/dev/null; then
    eval "$(register-python-argcomplete3 colcon 2>/dev/null)" || true
fi

# ── ROS 2 domain (change per-session to isolate simulations)
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "🤖  PX4 + ROS 2 ${ROS_DISTRO} + Gazebo ${GZ_VERSION} + PyTorch CUDA ready"
