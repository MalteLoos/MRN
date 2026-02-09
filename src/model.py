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
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Rollout buffer  (PPO experience storage + GAE)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one epoch of on-policy transitions and computes GAE advantages.

    Replay buffers vs rollout buffers
    ──────────────────────────────────
    • **Replay buffer** (off-policy, e.g. SAC / DDPG):
      Large circular buffer that keeps *all* past transitions.  Mini-batches
      are sampled uniformly (or by priority) regardless of which policy
      collected them.

    • **Rollout buffer** (on-policy, e.g. PPO / A2C):
      Fixed-length buffer that stores *one trajectory* collected by the
      **current** policy.  After the PPO update the data is discarded
      because stale experience would bias the on-policy gradient.

    This class implements the rollout (on-policy) variant, which is the
    correct one for PPO.

    Workflow
    --------
    1.  ``reset()`` at the start of each rollout.
    2.  ``store(...)`` after every env step.
    3.  ``finish(last_value)`` once the rollout is full — computes returns &
        GAE advantages.
    4.  Iterate with ``sample_batches(batch_size)`` during PPO updates.
    """

    def __init__(
        self,
        buffer_size: int,
        cam_shape: Tuple[int, int, int],
        imu_dim: int,
        wp_dim: int,
        action_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Args:
            buffer_size: Number of transitions to collect per rollout epoch.
            cam_shape:   (C, H, W) of the camera observation.
            imu_dim:     Dimensionality of the IMU vector (default 6).
            wp_dim:      Dimensionality of the flattened waypoint vector (default 6).
            action_dim:  Dimensionality of the action (default 4).
            device:      "cpu" or "cuda".
            gamma:       Discount factor.
            gae_lambda:  GAE λ for bias–variance trade-off.
        """
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # ----- pre-allocate storage ------------------------------------------
        self.cam       = torch.zeros(buffer_size, *cam_shape, device=self.device)
        self.imu       = torch.zeros(buffer_size, imu_dim,    device=self.device)
        self.waypoints = torch.zeros(buffer_size, wp_dim,     device=self.device)
        self.actions   = torch.zeros(buffer_size, action_dim, device=self.device)
        self.log_probs = torch.zeros(buffer_size,             device=self.device)
        self.rewards   = torch.zeros(buffer_size,             device=self.device)
        self.values    = torch.zeros(buffer_size,             device=self.device)
        self.dones     = torch.zeros(buffer_size,             device=self.device)

        # ----- computed after finish() ---------------------------------------
        self.advantages = torch.zeros(buffer_size, device=self.device)
        self.returns    = torch.zeros(buffer_size, device=self.device)

        self.ptr = 0
        self.full = False

    # ----- storage --------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer for a new rollout epoch."""
        self.ptr = 0
        self.full = False

    def store(
        self,
        cam: torch.Tensor,
        imu: torch.Tensor,
        waypoints: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Append a single transition.

        All tensor inputs should already be on ``self.device`` and have no
        batch dimension (i.e. squeezed to 1-D / 3-D for cam).
        """
        assert self.ptr < self.buffer_size, "Buffer full — call finish() then reset()."

        i = self.ptr
        self.cam[i]       = cam
        self.imu[i]       = imu
        self.waypoints[i] = waypoints
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value.squeeze()
        self.dones[i]     = float(done)
        self.ptr += 1

        if self.ptr == self.buffer_size:
            self.full = True

    # ----- GAE computation ------------------------------------------------

    def finish(self, last_value: torch.Tensor) -> None:
        """
        Call after the rollout is complete (buffer full **or** episode ended).

        Computes Generalised Advantage Estimation (GAE-λ) and discounted
        returns for every stored transition.

        Args:
            last_value: V(s_{T+1}) — the critic's estimate for the state
                        *after* the last stored transition. Pass 0 if the
                        episode terminated.
        """
        n = self.ptr  # may be < buffer_size if episode ended early
        last_gae = 0.0
        last_val = last_value.squeeze().item()

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - self.dones[t].item()
                next_value = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t].item()
                next_value = self.values[t + 1].item()

            delta = (self.rewards[t].item()
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t].item())
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Normalise advantages (variance reduction)
        adv = self.advantages[:n]
        self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ----- mini-batch sampling --------------------------------------------

    def sample_batches(
        self,
        batch_size: int,
    ):
        """
        Yield shuffled mini-batches of ``batch_size`` from the filled buffer.

        Each yielded dict contains the same keys the policy expects plus
        ``"actions"``, ``"old_log_probs"``, ``"advantages"``, ``"returns"``.
        """
        n = self.ptr
        indices = torch.randperm(n, device=self.device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                "obs": {
                    "cam":       self.cam[idx],
                    "imu":       self.imu[idx],
                    "waypoints": self.waypoints[idx],
                },
                "actions":       self.actions[idx],
                "old_log_probs": self.log_probs[idx],
                "advantages":    self.advantages[idx],
                "returns":       self.returns[idx],
            }


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Proximal Policy Optimisation trainer that ties the rollout buffer, the
    policy, and the update loop together.

    Usage
    -----
    >>> cfg   = ModelConfig()
    >>> agent = DroneAgent(cfg, route=route, device="cuda")
    >>> trainer = PPOTrainer(agent, cfg)
    >>>
    >>> for epoch in range(num_epochs):
    ...     trainer.collect_rollout(env)
    ...     stats = trainer.update()
    ...     print(stats)
    """

    def __init__(
        self,
        agent: DroneAgent,
        cfg: ModelConfig | None = None,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        ppo_epochs: int = 10,
    ):
        self.agent = agent
        self.cfg = cfg or agent.cfg
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        device = str(agent.device)
        cam_shape = (self.cfg.cam_channels, self.cfg.cam_height, self.cfg.cam_width)
        wp_dim = self.cfg.waypoint_dim * self.cfg.num_waypoints

        self.buffer = RolloutBuffer(
            buffer_size=rollout_steps,
            cam_shape=cam_shape,
            imu_dim=self.cfg.imu_dim,
            wp_dim=wp_dim,
            action_dim=self.cfg.action_dim,
            device=device,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        self.optimiser = torch.optim.Adam(
            agent.policy.parameters(), lr=self.cfg.lr,
        )

    # ----- rollout collection ---------------------------------------------

    def collect_rollout(self, env) -> dict:
        """
        Run the current policy in ``env`` for ``rollout_steps`` transitions
        and fill the rollout buffer.

        ``env`` must expose:
            - ``reset() -> obs``
            - ``step(action) -> (obs, reward, done, info)``

        where ``obs`` is a dict with keys ``"cam"``, ``"imu"``,
        ``"drone_pos"``.

        Returns:
            info dict with ``total_reward`` and ``episodes_completed``.
        """
        self.buffer.reset()
        obs = env.reset()
        total_reward = 0.0
        episodes = 0

        for _ in range(self.rollout_steps):
            # --- prepare tensors (single-step, no batch) ---------------------
            cam, imu, wp_tensor = self._obs_to_tensors(obs)

            policy_obs = {
                "cam": cam.unsqueeze(0),
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
            }
            action, log_prob, value = self.agent.policy.act(policy_obs)
            action_sq = action.squeeze(0)

            # --- environment step --------------------------------------------
            action_np = action_sq.cpu()
            next_obs, reward, done, info = env.step(action_np)

            # --- store -------------------------------------------------------
            self.buffer.store(
                cam=cam,
                imu=imu,
                waypoints=wp_tensor.squeeze(0),
                action=action_sq,
                log_prob=log_prob.squeeze(0),
                reward=float(reward),
                value=value,
                done=done,
            )
            total_reward += float(reward)

            if done:
                obs = env.reset()
                episodes += 1
            else:
                obs = next_obs

        # --- bootstrap last value --------------------------------------------
        cam, imu, wp_tensor = self._obs_to_tensors(obs)
        with torch.no_grad():
            policy_obs = {
                "cam": cam.unsqueeze(0),
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
            }
            _, _, last_value = self.agent.policy.act(policy_obs)
        self.buffer.finish(last_value)

        return {"total_reward": total_reward, "episodes_completed": episodes}

    # ----- PPO update -----------------------------------------------------

    def update(self) -> dict:
        """
        Run ``ppo_epochs`` of clipped PPO over the filled rollout buffer.

        Returns:
            dict with ``policy_loss``, ``value_loss``, ``entropy``,
            ``approx_kl``.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_batches = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.sample_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Re-evaluate under current policy
                new_log_probs, entropy, values = (
                    self.agent.policy.evaluate_actions(obs, actions)
                )
                values = values.squeeze(-1)

                # --- policy (actor) loss -----------------------------------
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                1.0 + self.cfg.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value (critic) loss -----------------------------------
                value_loss = F.mse_loss(values, returns)

                # --- total loss --------------------------------------------
                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy.mean())

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(), self.cfg.max_grad_norm,
                )
                self.optimiser.step()

                # --- logging -----------------------------------------------
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                total_kl          += approx_kl
                num_batches       += 1

        n = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss / n,
            "entropy":     total_entropy / n,
            "approx_kl":   total_kl / n,
        }

    # ----- helpers --------------------------------------------------------

    def _obs_to_tensors(self, obs: dict):
        """Convert a raw env obs dict to device tensors (no batch dim)."""
        device = self.agent.device

        cam = obs["cam"]
        if not isinstance(cam, torch.Tensor):
            cam = torch.as_tensor(cam, dtype=torch.float32)
        if cam.ndim == 3 and cam.shape[-1] in (1, 3):
            cam = cam.permute(2, 0, 1)
        if cam.max() > 1.0:
            cam = cam / 255.0
        cam = cam.to(device)

        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        imu = imu.to(device)

        drone_pos = tuple(float(x) for x in obs["drone_pos"][:3])
        wp_tensor = self.agent._buffer.current_targets_tensor(
            drone_pos, device=device,
        )
        return cam, imu, wp_tensor


# ---------------------------------------------------------------------------
# Curriculum  — progressive route difficulty
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """Defines one difficulty level in the training curriculum."""
    name: str
    num_waypoints: int          # how many WPs per episode
    min_spacing: float          # minimum distance between consecutive WPs (m)
    max_spacing: float          # maximum distance
    max_angle_deg: float        # max heading change between consecutive WPs
    safety_radius: float        # waypoint reach threshold (can relax early on)
    altitude_range: Tuple[float, float] = (3.0, 7.0)   # (min_z, max_z)
    success_threshold: float = 0.8   # fraction of WPs reached to "pass"
    min_episodes: int = 50           # minimum episodes before promotion


class CurriculumScheduler:
    """
    Manages a sequence of ``CurriculumStage``s and promotes the agent when
    it consistently succeeds at the current level.

    Design rationale
    ────────────────
    Early stages use **short, straight, 2-waypoint routes** so the policy
    first learns to fly toward a single target.  Later stages add more
    waypoints, sharper turns, and tighter safety radii, forcing the agent
    to exploit the two-waypoint look-ahead for smooth path planning.

    Usage
    -----
    >>> curriculum = CurriculumScheduler.default()
    >>> for epoch in range(total_epochs):
    ...     route = curriculum.generate_route()
    ...     agent.set_route(route)
    ...     agent.cfg.safety_radius = curriculum.current_stage.safety_radius
    ...     # ... run episode ...
    ...     curriculum.report(waypoints_reached, total_waypoints)
    ...     if curriculum.finished:
    ...         break
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        window_size: int = 50,
        on_promote: Optional[Callable[[CurriculumStage, CurriculumStage], None]] = None,
    ):
        """
        Args:
            stages:      Ordered list of difficulty stages.
            window_size: Rolling window for success-rate computation.
            on_promote:  Optional callback ``(old_stage, new_stage)`` called
                         on every promotion.
        """
        assert len(stages) >= 1
        self._stages = stages
        self._idx = 0
        self._window_size = window_size
        self._on_promote = on_promote

        # Rolling history of (reached, total) per episode
        self._history: deque[Tuple[int, int]] = deque(maxlen=window_size)
        self._total_episodes = 0

    # ----- properties -----------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        return self._stages[self._idx]

    @property
    def stage_index(self) -> int:
        return self._idx

    @property
    def num_stages(self) -> int:
        return len(self._stages)

    @property
    def finished(self) -> bool:
        """True once the agent has graduated from the final stage."""
        return self._idx >= len(self._stages)

    @property
    def success_rate(self) -> float:
        """Rolling fraction of waypoints reached over recent episodes."""
        if not self._history:
            return 0.0
        reached = sum(r for r, _ in self._history)
        total = sum(t for _, t in self._history)
        return reached / max(total, 1)

    # ----- route generation -----------------------------------------------

    def generate_route(
        self,
        start: Tuple[float, float, float] = (0.0, 0.0, 5.0),
    ) -> List[Tuple[float, float, float]]:
        """
        Build a random route that respects the current stage constraints.

        Args:
            start: The drone's spawn / take-off position.

        Returns:
            List of (x, y, z) waypoints (length = stage.num_waypoints).
        """
        stage = self.current_stage
        route: List[Tuple[float, float, float]] = []
        prev = start
        heading = random.uniform(0.0, 2.0 * math.pi)

        for _ in range(stage.num_waypoints):
            dist = random.uniform(stage.min_spacing, stage.max_spacing)
            max_delta = math.radians(stage.max_angle_deg)
            heading += random.uniform(-max_delta, max_delta)

            x = prev[0] + dist * math.cos(heading)
            y = prev[1] + dist * math.sin(heading)
            z = random.uniform(*stage.altitude_range)
            wp = (x, y, z)

            route.append(wp)
            prev = wp

        return route

    # ----- episode reporting & promotion ----------------------------------

    def report(self, waypoints_reached: int, total_waypoints: int) -> bool:
        """
        Report the outcome of one episode.

        Args:
            waypoints_reached: How many WPs the drone reached.
            total_waypoints:   Total WPs in the route.

        Returns:
            True if the agent was promoted to the next stage this call.
        """
        if self.finished:
            return False

        self._history.append((waypoints_reached, total_waypoints))
        self._total_episodes += 1

        promoted = False
        stage = self.current_stage
        if (len(self._history) >= stage.min_episodes
                and self.success_rate >= stage.success_threshold):
            promoted = self._promote()

        return promoted

    def _promote(self) -> bool:
        old = self.current_stage
        self._idx += 1
        self._history.clear()

        if self.finished:
            return True

        new = self.current_stage
        if self._on_promote is not None:
            self._on_promote(old, new)
        return True

    # ----- summary --------------------------------------------------------

    def status_dict(self) -> Dict[str, object]:
        """Snapshot for logging / TensorBoard."""
        stage = self.current_stage if not self.finished else self._stages[-1]
        return {
            "curriculum/stage_index": self._idx,
            "curriculum/stage_name": stage.name,
            "curriculum/success_rate": round(self.success_rate, 3),
            "curriculum/total_episodes": self._total_episodes,
            "curriculum/finished": self.finished,
        }

    # ----- built-in presets -----------------------------------------------

    @classmethod
    def default(
        cls,
        window_size: int = 50,
        on_promote: Optional[Callable] = None,
    ) -> "CurriculumScheduler":
        """
        A sensible 5-stage curriculum:

        1. **Hover→Point**   – 2 WPs, short & straight, generous radius.
        2. **Short legs**     – 3 WPs, gentle turns.
        3. **Medium legs**    – 4 WPs, moderate turns & tighter radius.
        4. **Long & twisty**  – 6 WPs, sharp turns.
        5. **Full mission**   – 8 WPs, tight radius, any angle.
        """
        stages = [
            CurriculumStage(
                name="hover_to_point",
                num_waypoints=2,
                min_spacing=3.0,
                max_spacing=6.0,
                max_angle_deg=15.0,
                safety_radius=2.0,
                success_threshold=0.75,
                min_episodes=30,
            ),
            CurriculumStage(
                name="short_legs",
                num_waypoints=3,
                min_spacing=4.0,
                max_spacing=8.0,
                max_angle_deg=30.0,
                safety_radius=1.5,
                success_threshold=0.80,
                min_episodes=50,
            ),
            CurriculumStage(
                name="medium_legs",
                num_waypoints=4,
                min_spacing=5.0,
                max_spacing=12.0,
                max_angle_deg=60.0,
                safety_radius=1.2,
                success_threshold=0.80,
                min_episodes=50,
            ),
            CurriculumStage(
                name="long_and_twisty",
                num_waypoints=6,
                min_spacing=6.0,
                max_spacing=15.0,
                max_angle_deg=90.0,
                safety_radius=1.0,
                success_threshold=0.80,
                min_episodes=80,
            ),
            CurriculumStage(
                name="full_mission",
                num_waypoints=8,
                min_spacing=5.0,
                max_spacing=20.0,
                max_angle_deg=120.0,
                safety_radius=0.8,
                altitude_range=(2.0, 10.0),
                success_threshold=0.85,
                min_episodes=100,
            ),
        ]
        return cls(stages, window_size=window_size, on_promote=on_promote)
