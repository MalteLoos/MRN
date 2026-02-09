# MRN

## Dev Container – PX4 · ROS 2 Humble · Gazebo Harmonic · PyTorch CUDA

A fully-configured development container for autonomous drone RL research, combining PX4 flight-stack simulation with GPU-accelerated reinforcement learning.

### What's inside

| Layer | Version / Details |
|---|---|
| **Base image** | `nvidia/cuda:12.1.1-devel-ubuntu22.04` |
| **ROS 2** | Humble Hawksbill (desktop) |
| **Gazebo** | Harmonic (with `ros_gz` bridge built from source) |
| **PX4 Autopilot** | `main` branch, SITL pre-built |
| **PX4 ↔ ROS 2** | Micro XRCE-DDS Agent + `px4_msgs` / `px4_ros_com` |
| **MAVROS2** | `ros-humble-mavros` + extras (IMU `data_raw`, GPS, MAVLink topics) |
| **PyTorch** | Latest (CUDA 12.1) |
| **RL libraries** | Gymnasium, Stable-Baselines3, SB3-Contrib, TensorBoard, W&B |

### Prerequisites

- **Docker** ≥ 24 with Compose v2
- **NVIDIA driver** ≥ 525 on the host
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
- **VS Code** with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### Quick start

```bash
# 1. Allow X11 access (for Gazebo / rviz2 GUI)
xhost +local:docker

# 2. Open in VS Code → "Reopen in Container"
#    Or from the command line:
devcontainer up --workspace-folder .

# 3. Inside the container — start PX4 SITL with Gazebo
cd $PX4_HOME && make px4_sitl gz_x500

# 4. In a second terminal — launch the DDS agent (PX4 ↔ ROS 2)
MicroXRCEAgent udp4 -p 8888

# 5. In a third terminal — launch MAVROS2 for IMU/GPS data
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557

# 6. In a fourth terminal — verify ROS 2 topics
ros2 topic list          # should show /fmu/out/* and /mavros/* topics
ros2 topic echo /mavros/imu/data_raw
```

### Project layout

```
.devcontainer/
├── devcontainer.json        # VS Code dev container config
├── Dockerfile               # Multi-layer image build
├── docker-compose.yml       # GPU / display / networking
├── .env                     # Environment variable overrides
└── scripts/
    ├── bashrc-extras.sh     # Shell config (sourced in .bashrc)
    └── post-create.sh       # One-time setup after container creation
ros2_ws/                     # Your ROS 2 workspace (auto-created)
└── src/
    └── rl_training/         # RL training package skeleton
```

### Handy aliases (available in every terminal)

| Alias | Command |
|---|---|
| `px4sitl` | `cd $PX4_HOME && make px4_sitl gz_x500` |
| `ddsagent` | `MicroXRCEAgent udp4 -p 8888` |
| `mavros` | `ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557` |
| `cb` | `colcon build --symlink-install` (in ros2_ws) |
| `sr` | `source ros2_ws/install/setup.bash` |
| `tb` | `tensorboard --logdir /workspace/logs --bind_all` |

### GPU verification

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

### Tips

- **Isolate simulations**: set different `ROS_DOMAIN_ID` values per terminal session.
- **Experiment tracking**: add your `WANDB_API_KEY` to `.devcontainer/.env`.
- **Rebuild only PX4**: `cd $PX4_HOME && make px4_sitl_default` (cached).
- **Custom Gazebo worlds**: add `.sdf` files to `$PX4_HOME/Tools/simulation/gz/worlds/`.