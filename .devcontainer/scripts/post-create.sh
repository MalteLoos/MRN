#!/usr/bin/env bash
# ============================================================
# post-create.sh — runs once after the container is created
# ============================================================
set -euo pipefail

echo "🚀  Running post-create setup …"

# ── 0. Fix ownership of volume-mounted directories ────────
#    Docker volumes may have been created by root in a prior run.
sudo chown -R "$(id -u):$(id -g)" \
    "${HOME}/.ros" \
    "${HOME}/.ccache" \
    "${HOME}/.local/share/QGroundControl" \
    2>/dev/null || true

# ── 1. rosdep update (runs as non-root) ────────────────────
rosdep update --rosdistro "${ROS_DISTRO:-humble}" 2>/dev/null || true

# ── 2. Create a user ROS 2 workspace if it doesn't exist ──
ROS2_WS="/workspace"
#if [[ ! -d "${ROS2_WS}/src" ]]; then
#    mkdir -p "${ROS2_WS}/src"
#    echo "📁  Created ROS 2 workspace at ${ROS2_WS}"
#fi

# ── 3. Create an RL training package skeleton ──────────────
#RL_PKG="${ROS2_WS}/src/rl_training"
#if [[ ! -d "${RL_PKG}" ]]; then
#    mkdir -p "${RL_PKG}/rl_training" \
#             "${RL_PKG}/config" \
#             "${RL_PKG}/launch" \
#             "${RL_PKG}/scripts"
#    touch "${RL_PKG}/rl_training/__init__.py"
#    echo "📁  Created RL training package skeleton at ${RL_PKG}"
#fi

# ── 4. Create XDG_RUNTIME_DIR (needed by some Qt apps) ────
if [[ -n "${XDG_RUNTIME_DIR:-}" ]] && [[ ! -d "${XDG_RUNTIME_DIR}" ]]; then
    mkdir -p "${XDG_RUNTIME_DIR}"
    chmod 0700 "${XDG_RUNTIME_DIR}"
fi

# ── 5. Allow X11 access from inside the container ─────────
if command -v xhost &>/dev/null; then
    xhost +local:docker 2>/dev/null || true
fi

# ── 6. Quick sanity checks ────────────────────────────────
echo ""
echo "────────────────────────────────────────────"
echo "  Environment sanity checks"
echo "────────────────────────────────────────────"

set +u
echo -n "  ROS 2:      " && (source /opt/ros/humble/setup.bash && echo "${ROS_DISTRO:-?}")
set -u
echo -n "  Gazebo:     " && (gz sim --version 2>/dev/null | head -1 || echo "not found")
echo -n "  Python:     " && python3 --version
echo -n "  PyTorch:    " && python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not found"
echo -n "  CUDA avail: " && python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A"
echo -n "  GPU name:   " && python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>/dev/null || echo "N/A"
echo -n "  PX4 home:   " && echo "${PX4_HOME:-/opt/PX4-Autopilot}"
echo -n "  MAVROS2:    " && (ros2 pkg list 2>/dev/null | grep -q mavros && echo "installed" || echo "not found")
echo -n "  XRCE-DDS:   " && (MicroXRCEAgent --version 2>/dev/null || echo "installed (check path)")
echo "────────────────────────────────────────────"
echo ""
echo "✅  Post-create setup complete!"
echo ""
echo "  Quick-start commands:"
echo "    # Start PX4 SITL with Gazebo Harmonic"
echo "    cd \${PX4_HOME} && make px4_sitl gz_x500"
echo ""
echo "    # In another terminal – start the DDS agent"
echo "    MicroXRCEAgent udp4 -p 8888"
echo ""
echo "    # In another terminal – launch MAVROS2 (IMU data_raw, GPS, etc.)"
echo "    ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557"
echo ""
echo "    # In another terminal – check ROS 2 topics"
echo "    source /opt/ros/humble/setup.bash"
echo "    ros2 topic list"
echo "    ros2 topic echo /mavros/imu/data_raw"
echo ""
