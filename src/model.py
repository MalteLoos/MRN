"""
Drone Waypoint Navigation - Reinforcement Learning Model

A PPO-based actor-critic model for drone navigation through sequential waypoints.
Inputs:  IMU (linear accel + angular vel), Camera image, 2 look-ahead waypoints.
Outputs: Body-rate commands (roll_rate, pitch_rate, yaw_rate, thrust).

The drone always keeps two upcoming waypoints in its buffer. When it enters the
safety radius of the first waypoint the buffer shifts forward, promoting the
second waypoint to the current goal and loading the next one from the route.
Keeping two waypoints visible lets the policy learn to produce smooth,
anticipatory trajectories.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All tuneable hyper-parameters live here."""

    # --- Camera input ---------------------------------------------------------
    cam_channels: int = 3           # RGB
    cam_height: int = 64
    cam_width: int = 64
    cam_feature_dim: int = 128      # CNN output size

    # --- IMU input ------------------------------------------------------------
    imu_dim: int = 6                # 3 accel + 3 gyro

    # --- Waypoint input -------------------------------------------------------
    waypoint_dim: int = 3           # (x, y, z) per waypoint — in body frame
    num_waypoints: int = 2          # always 2 look-ahead waypoints

    # --- Fusion / shared backbone ---------------------------------------------
    fusion_hidden: int = 256
    fusion_layers: int = 2

    # --- Actor (policy) -------------------------------------------------------
    action_dim: int = 4             # roll_rate, pitch_rate, yaw_rate, thrust
    log_std_min: float = -5.0
    log_std_max: float = 0.5

    # --- Waypoint management --------------------------------------------------
    safety_radius: float = 1.0     # metres — when within this, waypoint reached

    # --- Training defaults ----------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


# ---------------------------------------------------------------------------
# Camera encoder (lightweight CNN)
# ---------------------------------------------------------------------------

class CameraEncoder(nn.Module):
    """Encodes an RGB image into a compact feature vector."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.cam_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        # 64 * 4 * 4 = 1024
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, cfg.cam_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W) normalised image in [0, 1].
        Returns:
            features: (B, cam_feature_dim)
        """
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# State encoder  (IMU + waypoints)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Encodes IMU readings and the two upcoming waypoints."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        input_dim = cfg.imu_dim + cfg.waypoint_dim * cfg.num_waypoints
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.output_dim = 64

    def forward(
        self,
        imu: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imu:       (B, imu_dim)          — [ax, ay, az, gx, gy, gz]
            waypoints: (B, num_wp * wp_dim)  — flattened [wp1_x, wp1_y, wp1_z,
                                                          wp2_x, wp2_y, wp2_z]
        Returns:
            features: (B, 64)
        """
        x = torch.cat([imu, waypoints], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-modal fusion backbone
# ---------------------------------------------------------------------------

class FusionBackbone(nn.Module):
    """Merges camera features and state features into a shared representation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cam_feat = cfg.cam_feature_dim  # 128
        state_feat = 64                 # from StateEncoder
        in_dim = cam_feat + state_feat

        layers: list[nn.Module] = []
        for i in range(cfg.fusion_layers):
            layers.append(nn.Linear(in_dim if i == 0 else cfg.fusion_hidden,
                                    cfg.fusion_hidden))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.output_dim = cfg.fusion_hidden

    def forward(
        self,
        cam_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cam_feat:   (B, cam_feature_dim)
            state_feat: (B, 64)
        Returns:
            fused: (B, fusion_hidden)
        """
        return self.net(torch.cat([cam_feat, state_feat], dim=-1))


# ---------------------------------------------------------------------------
# Actor head  — outputs body-rate commands
# ---------------------------------------------------------------------------

class ActorHead(nn.Module):
    """
    Gaussian policy head.

    Outputs mean and log_std for each of the 4 body-rate channels:
        [roll_rate, pitch_rate, yaw_rate, thrust]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mu = nn.Sequential(
            nn.Linear(cfg.fusion_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cfg.action_dim),
            nn.Tanh(),                           # bound means to [-1, 1]
        )
        # Learnable log-std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

    def forward(
        self,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:      (B, action_dim)
            log_std: (B, action_dim) — clamped
        """
        mu = self.mu(fused)
        log_std = self.log_std.clamp(self.cfg.log_std_min,
                                     self.cfg.log_std_max).expand_as(mu)
        return mu, log_std

    def sample(
        self,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        mu, log_std = self.forward(fused)
        std = log_std.exp()
        dist = Normal(mu, std)
        # Re-parameterised sample
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self,
        fused: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for a batch of actions."""
        mu, log_std = self.forward(fused)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Critic head  — estimates state value V(s)
# ---------------------------------------------------------------------------

class CriticHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.fusion_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns: value (B, 1)"""
        return self.net(fused)


# ---------------------------------------------------------------------------
# Full actor-critic policy
# ---------------------------------------------------------------------------

class DronePolicy(nn.Module):
    """
    End-to-end actor-critic policy for waypoint-following drone control.

    Observation dict expected keys
    ──────────────────────────────
    - ``"cam"``       : (B, C, H, W) float32 image in [0, 1]
    - ``"imu"``       : (B, 6) float32  [ax, ay, az, gx, gy, gz]
    - ``"waypoints"`` : (B, 6) float32  [wp1_xyz, wp2_xyz]  — body frame
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.cam_encoder = CameraEncoder(self.cfg)
        self.state_encoder = StateEncoder(self.cfg)
        self.backbone = FusionBackbone(self.cfg)
        self.actor = ActorHead(self.cfg)
        self.critic = CriticHead(self.cfg)

    # ----- helpers --------------------------------------------------------

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cam_feat = self.cam_encoder(obs["cam"])
        state_feat = self.state_encoder(obs["imu"], obs["waypoints"])
        return self.backbone(cam_feat, state_feat)

    # ----- public API -----------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect a single transition (inference mode).

        Returns:
            action   : (B, 4) body-rate command
            log_prob : (B,)   log π(a|s)
            value    : (B, 1) V(s)
        """
        fused = self._encode(obs)
        value = self.critic(fused)
        if deterministic:
            mu, _ = self.actor(fused)
            return mu, torch.zeros(mu.size(0), device=mu.device), value
        action, log_prob = self.actor.sample(fused)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored transitions during PPO update.

        Returns:
            log_prob : (B,)
            entropy  : (B,)
            value    : (B, 1)
        """
        fused = self._encode(obs)
        log_prob, entropy = self.actor.evaluate(fused, actions)
        value = self.critic(fused)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Waypoint buffer manager
# ---------------------------------------------------------------------------

class WaypointBuffer:
    """
    Manages the two-waypoint look-ahead buffer per drone.

    Workflow
    --------
    1. Initialise with a route (ordered list of 3-D waypoints).
    2. Each step, call ``update(drone_pos)`` which returns the current pair
       of target waypoints.
    3. When the drone enters the safety radius of waypoint-0 the buffer
       shifts: wp-0 ← wp-1, wp-1 ← next from route.
    4. If the route is exhausted, the last waypoint is duplicated so the
       buffer always contains exactly two entries (the policy can still
       output smooth decelerating commands).
    """

    def __init__(
        self,
        route: List[Tuple[float, float, float]],
        safety_radius: float = 1.0,
    ):
        assert len(route) >= 2, "Route must contain at least 2 waypoints."
        self.safety_radius = safety_radius
        self._queue: deque[Tuple[float, float, float]] = deque(route)

        # Initialise the two-slot buffer
        self.wp0: Tuple[float, float, float] = self._queue.popleft()
        self.wp1: Tuple[float, float, float] = self._queue.popleft()

        self.reached_count: int = 0
        self.finished: bool = False

    # ----- internal -------------------------------------------------------

    @staticmethod
    def _dist(a: Tuple[float, float, float],
              b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _advance(self) -> None:
        """Shift buffer forward by one waypoint."""
        self.reached_count += 1
        self.wp0 = self.wp1
        if self._queue:
            self.wp1 = self._queue.popleft()
        else:
            # Route exhausted — duplicate last waypoint to keep buffer full
            self.wp1 = self.wp0
            self.finished = True

    # ----- public API -----------------------------------------------------

    def update(
        self,
        drone_pos: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Call once per time-step.

        Args:
            drone_pos: current (x, y, z) of the drone in the world frame.

        Returns:
            (wp0, wp1): the two waypoints the policy should see this step.
        """
        if self._dist(drone_pos, self.wp0) <= self.safety_radius:
            self._advance()
        return self.wp0, self.wp1

    def current_targets_tensor(
        self,
        drone_pos: Tuple[float, float, float],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Returns the two waypoints as a flat (1, 6) tensor ready for the
        policy, after calling ``update``.

        The waypoints are returned in *world frame*.  If you need body-frame
        coordinates, transform them before feeding to the policy.
        """
        wp0, wp1 = self.update(drone_pos)
        data = [*wp0, *wp1]
        return torch.tensor([data], dtype=torch.float32, device=device)

    @property
    def remaining(self) -> int:
        """Waypoints left in the queue (excluding the active pair)."""
        return len(self._queue)


# ---------------------------------------------------------------------------
# High-level agent wrapper  (optional convenience)
# ---------------------------------------------------------------------------

class DroneAgent:
    """
    Ties the policy and waypoint buffer together for rollout collection.

    Usage
    -----
    >>> cfg = ModelConfig(safety_radius=1.5)
    >>> agent = DroneAgent(cfg, route=[(0,0,5), (10,0,5), (10,10,5), (0,10,5)])
    >>> obs = env.reset()
    >>> while not done:
    ...     action = agent.step(obs)
    ...     obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        route: List[Tuple[float, float, float]] | None = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or ModelConfig()
        self.device = torch.device(device)
        self.policy = DronePolicy(self.cfg).to(self.device)
        self._buffer: Optional[WaypointBuffer] = None
        if route is not None:
            self.set_route(route)

    # ----- route management -----------------------------------------------

    def set_route(self, route: List[Tuple[float, float, float]]) -> None:
        """(Re-)initialise the waypoint buffer with a new route."""
        self._buffer = WaypointBuffer(route, self.cfg.safety_radius)

    # ----- step -----------------------------------------------------------

    def step(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Perform one decision step.

        ``obs`` must contain:

        - ``"cam"``       : (H, W, C) or (B, C, H, W) uint8/float image
        - ``"imu"``       : (6,) or (B, 6) float
        - ``"drone_pos"`` : (3,) float — used to update the waypoint buffer

        Returns:
            action: (4,) numpy-compatible body-rate command
        """
        assert self._buffer is not None, "Call set_route() before step()."

        # --- prepare camera --------------------------------------------------
        cam = obs["cam"]
        if not isinstance(cam, torch.Tensor):
            cam = torch.as_tensor(cam, dtype=torch.float32)
        if cam.ndim == 3 and cam.shape[-1] in (1, 3):
            cam = cam.permute(2, 0, 1)         # HWC → CHW
        if cam.ndim == 3:
            cam = cam.unsqueeze(0)              # add batch dim
        if cam.max() > 1.0:
            cam = cam / 255.0
        cam = cam.to(self.device)

        # --- prepare IMU -----------------------------------------------------
        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        if imu.ndim == 1:
            imu = imu.unsqueeze(0)
        imu = imu.to(self.device)

        # --- update waypoint buffer & build tensor ---------------------------
        drone_pos = tuple(float(x) for x in obs["drone_pos"][:3])
        wp_tensor = self._buffer.current_targets_tensor(
            drone_pos, device=self.device,
        )

        # --- forward pass ----------------------------------------------------
        policy_obs = {"cam": cam, "imu": imu, "waypoints": wp_tensor}
        action, log_prob, value = self.policy.act(
            policy_obs, deterministic=deterministic,
        )
        return action.squeeze(0).cpu()

    # ----- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()

    # ----- properties ------------------------------------------------------

    @property
    def route_finished(self) -> bool:
        return self._buffer is not None and self._buffer.finished

    @property
    def waypoints_reached(self) -> int:
        return self._buffer.reached_count if self._buffer else 0

    @property
    def waypoints_remaining(self) -> int:
        return self._buffer.remaining if self._buffer else 0
