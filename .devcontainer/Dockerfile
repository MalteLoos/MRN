# ============================================================
# PX4 · ROS 2 Humble · Gazebo Harmonic · PyTorch CUDA
# Dev Container
# ============================================================
#
# Build args let you pin versions without editing the file:
#   docker compose build --build-arg CUDA_VERSION=12.8
#
# Expected image size: ~18-25 GB (GPU toolchain + sim + ML)
# Build time:          ~30-60 min on a fast machine
# ============================================================

ARG CUDA_VERSION=12.8
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

LABEL maintainer="dev" \
      description="PX4 SITL + ROS 2 Humble + Gazebo Harmonic + PyTorch CUDA"

# ── Global environment ──────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble \
    GZ_VERSION=harmonic \
    PX4_HOME=/opt/PX4-Autopilot \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    # Suppress Gazebo / Qt GPU warnings inside containers
    QT_X11_NO_MITSHM=1 \
    MESA_GL_VERSION_OVERRIDE=3.3

SHELL ["/bin/bash", "-c"]

# ── 1. Core system packages ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl wget git sudo ca-certificates tmux \
        lsb-release gnupg2 software-properties-common \
        build-essential cmake pkg-config \
        python3 python3-pip python3-dev python3-venv python3-setuptools \
        vim nano htop tmux unzip gdb \
        # GL / display
        libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 mesa-utils \
        libxkbcommon-x11-0 libxcb-xinerama0 x11-apps dbus-x11 \
        libxcb-cursor0 libxrandr2 libxfixes3 libxi6 \
        # Networking / DDS
        iproute2 iputils-ping net-tools \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── 1b. Non-root user (created early so pip installs go here) ─
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# ── 2. ROS 2 Humble ────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-humble-ament-cmake \
        ros-humble-eigen3-cmake-module \
        ros-humble-geographic-msgs \
        ros-humble-vision-opencv \
        ros-humble-image-transport \
        ros-humble-cv-bridge \
        ros-humble-tf-transformations \
        ros-humble-actuator-msgs \
        # MAVROS2 – MAVLink ↔ ROS 2 bridge (provides imu/data_raw etc.)
        ros-humble-mavros \
        ros-humble-mavros-extras \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# GeographicLib datasets required by MAVROS
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# Initialise rosdep (needs root)
RUN rosdep init || true

# ── 3. Gazebo Harmonic ──────────────────────────────────────
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# ── 4. ros_gz bridge  (source build for Humble ↔ Harmonic) ─
#    Binary ros-humble-ros-gz targets Fortress; we need Harmonic.
#    apt lists were cleared in earlier stages, so refresh before rosdep.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-humble-image-transport-plugins \
        ros-humble-sdformat-urdf \
        ros-humble-xacro \
        ros-humble-gps-msgs \
        ros-humble-vision-msgs \
        libgflags-dev \
    && mkdir -p /opt/ros_gz_ws/src \
    && cd /opt/ros_gz_ws/src \
    && git clone --depth 1 https://github.com/gazebosim/ros_gz.git -b humble \
    && cd /opt/ros_gz_ws \
    && source /opt/ros/humble/setup.bash \
    && export GZ_VERSION=harmonic \
    && rosdep update --rosdistro humble \
    && rosdep install -r --from-paths src -i -y --rosdistro humble \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log src \
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/ros_gz_ws/install/setup.bash" >> /etc/bash.bashrc

# ── 5. Micro XRCE-DDS Agent (PX4 ↔ ROS 2 bridge) ──────────
RUN git clone --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
        /tmp/uxrce-dds-agent \
    && cd /tmp/uxrce-dds-agent \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig /usr/local/lib/ \
    && rm -rf /tmp/uxrce-dds-agent

# ── 6. PX4 Autopilot ───────────────────────────────────────
#    Clone and install dependencies; build the default SITL target.
#    --no-nuttx   → skip embedded toolchain
#    --no-sim-tools → skip gazebo-classic / jMAVSim
RUN git clone --recursive https://github.com/PX4/PX4-Autopilot.git ${PX4_HOME} \
    && cd ${PX4_HOME} \
    && bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && make px4_sitl_default \
    && rm -rf /var/lib/apt/lists/*

# ── 7. PX4 ROS 2 workspace (px4_msgs + px4_ros_com) ────────
RUN mkdir -p /opt/px4_ros_ws/src \
    && cd /opt/px4_ros_ws/src \
    && git clone --depth 1 https://github.com/PX4/px4_msgs.git \
    && git clone --depth 1 https://github.com/PX4/px4_ros_com.git \
    && cd /opt/px4_ros_ws \
    && source /opt/ros/humble/setup.bash \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log src \
    && echo "source /opt/px4_ros_ws/install/setup.bash" >> /etc/bash.bashrc

# ── 8. PyTorch + CUDA + RL libraries ───────────────────────
#    Run as dev user so everything lands in /home/dev/.local/
#    which is on PATH and visible to the runtime user.
USER dev
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
        torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128 \
    && pip3 install --no-cache-dir \
        gymnasium \
        "stable-baselines3[extra]" \
        sb3-contrib \
        tensorboard \
        wandb \
        numpy scipy matplotlib \
        opencv-python-headless \
        transforms3d \
        shapely \
        empy==3.3.4 \
        pyros-genmsg \
        catkin_pkg \
        lark
USER root

# ── 9. Permissions & shell configuration ───────────────────
RUN chown -R dev:dev ${PX4_HOME}

COPY scripts/bashrc-extras.sh /tmp/bashrc-extras.sh
RUN cat /tmp/bashrc-extras.sh >> /home/dev/.bashrc \
    && rm /tmp/bashrc-extras.sh

COPY scripts/.tmux.conf /tmp/.tmux.conf
RUN cat /tmp/.tmux.conf >> /home/dev/.tmux.conf \
    && rm /tmp/.tmux.conf

USER dev
WORKDIR /workspace

CMD ["bash"]
