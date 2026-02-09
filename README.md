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