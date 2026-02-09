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
# MRN — Drone Waypoint Navigation via Reinforcement Learning

PPO-based actor-critic model for learning body-rate control to navigate a drone through sequential waypoints.

## Architecture

```
  Camera (64×64 RGB)          IMU (6-D)  +  2 Waypoints (6-D)
        │                              │
  CameraEncoder (CNN)           StateEncoder (MLP)
        │ 128-D                       │ 64-D
        └──────────┬──────────────────┘
             FusionBackbone (MLP 256)
               ┌────┴────┐
          ActorHead    CriticHead
          (4-D μ,σ)      (V(s))
```

| Component | Description |
|---|---|
| **CameraEncoder** | 3-layer CNN → AdaptiveAvgPool → FC(128) |
| **StateEncoder** | MLP encoding IMU (ax,ay,az,gx,gy,gz) + 2 waypoints (x,y,z each) |
| **FusionBackbone** | 2-layer MLP (256 hidden) merging cam + state features |
| **ActorHead** | Gaussian policy outputting body-rates: roll_rate, pitch_rate, yaw_rate, thrust |
| **CriticHead** | MLP → scalar V(s) |

## Waypoint Buffer

The `WaypointBuffer` always keeps **two look-ahead waypoints** visible to the policy:

1. When the drone enters the **safety radius** of waypoint 0, it is marked as reached.
2. Waypoint 1 is promoted to waypoint 0, and the next waypoint from the route queue becomes waypoint 1.
3. If the route is exhausted the last waypoint is duplicated so the buffer stays full.

This two-waypoint look-ahead lets the policy learn to **smooth turns** rather than flying point-to-point.

## Quick Start

```python
from src.model import DroneAgent, ModelConfig

cfg = ModelConfig(safety_radius=1.5)
route = [(0,0,5), (10,0,5), (10,10,5), (0,10,5)]
agent = DroneAgent(cfg, route=route, device="cpu")

obs = {
    "cam": camera_image,       # (H, W, 3) uint8 or float
    "imu": imu_reading,        # (6,) float [ax,ay,az,gx,gy,gz]
    "drone_pos": position,     # (3,) float [x,y,z]
}

action = agent.step(obs)       # → (4,) [roll_rate, pitch_rate, yaw_rate, thrust]
```
