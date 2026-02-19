"""
Drone Waypoint Navigation - Reinforcement Learning Model  (dual-rate)

A PPO-based actor-critic model for drone navigation through sequential
waypoints, split into two update rates:

**Slow path** (camera rate, ~30 Hz):
    A pretrained RAFT-Small optical-flow backbone (frozen) computes dense flow
    between consecutive camera frames.  A small trainable CNN compresses the
    flow map into a compact visual-feature vector that is *cached*.

**Fast path** (IMU rate, ~250 Hz):
    A lightweight MLP ingests the latest IMU sample, the two body-frame
    look-ahead waypoints, **and** the cached visual features.  Its output
    feeds actor + critic heads that produce body-rate commands.

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
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """All tuneable hyper-parameters live here."""

    # --- Camera input ---------------------------------------------------------
    cam_channels: int = 3  # RGB
    cam_height: int = 128  # RAFT-Small needs ≥ 128
    cam_width: int = 128

    # --- Optical-flow / slow path --------------------------------------------
    flow_feature_dim: int = 64  # compressed flow-vector size
    raft_iters: int = 6  # RAFT refinement iterations at inference

    # --- IMU input ------------------------------------------------------------
    imu_dim: int = 6  # 3 accel + 3 gyro

    # --- Waypoint input -------------------------------------------------------
    waypoint_dim: int = 3  # (x, y, z) per waypoint — in body frame
    num_waypoints: int = 2  # always 2 look-ahead waypoints

    # --- Fast-path backbone ---------------------------------------------------
    fast_hidden: int = 256  # hidden size of the fast MLP
    fast_layers: int = 3
    gru_hidden: int = 128   # GRU recurrent hidden size (0 = disable GRU)

    # --- Actor (policy) -------------------------------------------------------
    action_dim: int = 4  # roll_rate, pitch_rate, yaw_rate, thrust
    log_std_min: float = -5.0
    log_std_max: float = 0.5

    # --- Waypoint management --------------------------------------------------
    safety_radius: float = 1.0  # metres — when within this, waypoint reached

    # --- Training defaults ----------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


# ---------------------------------------------------------------------------
# Running observation normalizer
# ---------------------------------------------------------------------------


class RunningNormalizer(nn.Module):
    """Welford online running-mean / variance normalizer.

    Keeps per-feature running statistics and normalises inputs to
    approximately zero mean and unit variance.  Statistics are stored as
    buffers (not parameters) so they survive ``state_dict`` round-trips
    but are never updated by the optimiser.

    The normalizer is updated during rollout collection (``training=True``)
    and frozen during PPO updates / inference (``training=False``).  Call
    ``update(x)`` to incorporate a new sample **and** return the normalised
    value, or ``normalize(x)`` to normalise without updating stats.

    Clip range avoids numerical explosions from rare outliers.
    """

    def __init__(self, dim: int, clip: float = 5.0, epsilon: float = 1e-8):
        super().__init__()
        self.clip = clip
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))  # small init avoids div-by-0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """Update running stats with *x* and return normalised *x*.

        Args:
            x: (\*batch, dim) — the last dimension is normalised.

        Works correctly with any number of leading batch dimensions.
        """
        if self.training:
            flat = x.reshape(-1, x.shape[-1])
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0, unbiased=False)
            batch_count = flat.shape[0]

            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * self.count * batch_count / total
            self.var = m2 / total
            self.count = total

        return self.normalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* using current running statistics (no stat update)."""
        return ((x - self.mean) / (self.var + self.epsilon).sqrt()).clamp(
            -self.clip, self.clip
        )


# ---------------------------------------------------------------------------
# Slow path — optical-flow visual encoder  (runs at camera rate)
# ---------------------------------------------------------------------------


class FlowEncoder(nn.Module):
    """
    Small trainable CNN that compresses a dense optical-flow map (2, H, W)
    into a compact feature vector of size ``flow_feature_dim``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        # 64 * 4 * 4 = 1024
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, cfg.flow_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W) dense optical-flow map.
        Returns:
            features: (B, flow_feature_dim)
        """
        x = self.conv(flow)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SlowVisualEncoder(nn.Module):
    """
    Wraps a **frozen** RAFT-Small backbone and a trainable ``FlowEncoder``.

    Call ``forward(img_prev, img_curr)`` whenever a new camera frame arrives.
    The resulting feature vector should be cached and reused by the fast path
    until the next camera frame.

    RAFT-Small weights are loaded from the official torchvision checkpoint and
    are **not** updated during PPO training.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ----- frozen RAFT-Small backbone ------------------------------------
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        for p in self.raft.parameters():
            p.requires_grad = False
        self.raft.eval()

        # ----- trainable flow encoder ----------------------------------------
        self.flow_encoder = FlowEncoder(cfg)

    @property
    def output_dim(self) -> int:
        return self.cfg.flow_feature_dim

    def compute_flow(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the **frozen** RAFT backbone to produce a dense flow map.

        The result is always detached (no gradient through RAFT).

        Args:
            img_prev: (B, 3, H, W)  previous frame, float32 in [0, 1].
            img_curr: (B, 3, H, W)  current  frame, float32 in [0, 1].

        Returns:
            flow: (B, 2, H, W) — dense optical-flow map (fp32, detached).
        """
        prev_255 = img_prev * 255.0
        curr_255 = img_curr * 255.0

        with torch.no_grad():
            # Run RAFT under fp16 autocast — required on Blackwell GPUs
            # (SM 12.0) where float32 cuBLAS is broken (CUBLAS_STATUS_NOT_INITIALIZED).
            # autocast is a no-op on CPU, so this is safe everywhere.
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device_type=prev_255.device.type, dtype=torch.float16
            ):
                flow_preds = self.raft(
                    prev_255,
                    curr_255,
                    num_flow_updates=self.cfg.raft_iters,
                )
            flow = flow_preds[-1].float()  # (B, 2, H, W)
        return flow

    def encode_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Compress a dense flow map into a compact feature vector.

        This runs the **trainable** ``FlowEncoder`` CNN.  Unlike
        ``compute_flow``, this method supports gradient flow so that
        ``FlowEncoder`` can be trained during PPO / BC updates.

        Args:
            flow: (B, 2, H, W) — dense optical-flow map.
        Returns:
            vis_feat: (B, flow_feature_dim)
        """
        return self.flow_encoder(flow)

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical-flow features from two consecutive RGB frames.
        Convenience wrapper: runs ``compute_flow`` → ``encode_flow``.

        Args:
            img_prev: (B, 3, H, W)  previous frame, float32 in [0, 1].
            img_curr: (B, 3, H, W)  current  frame, float32 in [0, 1].

        Returns:
            vis_feat: (B, flow_feature_dim) — compact visual feature vector.
        """
        flow = self.compute_flow(img_prev, img_curr)
        return self.encode_flow(flow)


# ---------------------------------------------------------------------------
# Fast path — state encoder  (runs at IMU rate)
# ---------------------------------------------------------------------------


class FastStateEncoder(nn.Module):
    """
    MLP + optional GRU that fuses IMU, look-ahead waypoints, **and** the
    cached visual feature on every IMU tick.

    When ``cfg.gru_hidden > 0`` a single-layer GRU sits after the MLP,
    giving the policy *temporal memory* across consecutive ticks.  This
    lets the network implicitly estimate velocities and detect oscillation
    patterns from raw IMU/waypoint sequences rather than relying on an
    explicit velocity input.

    The GRU hidden state must be managed by the caller (``DronePolicy``):
    - passed in and returned at each step during rollout,
    - stored in the rollout buffer for PPO re-evaluation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        state_dim = cfg.imu_dim + cfg.waypoint_dim * cfg.num_waypoints
        in_dim = state_dim + cfg.flow_feature_dim

        layers: list[nn.Module] = []
        for i in range(cfg.fast_layers):
            layers.append(
                nn.Linear(in_dim if i == 0 else cfg.fast_hidden, cfg.fast_hidden)
            )
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

        # Optional GRU temporal layer
        self.use_gru = cfg.gru_hidden > 0
        if self.use_gru:
            self.gru = nn.GRU(
                input_size=cfg.fast_hidden,
                hidden_size=cfg.gru_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.output_dim = cfg.gru_hidden
        else:
            self.output_dim = cfg.fast_hidden

    def forward(
        self,
        imu: torch.Tensor,
        waypoints: torch.Tensor,
        vis_feat: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            imu:       (B, imu_dim)           — [ax, ay, az, gx, gy, gz]
            waypoints: (B, num_wp * wp_dim)   — flattened body-frame waypoints
            vis_feat:  (B, flow_feature_dim)  — cached slow-path output
            hidden:    (1, B, gru_hidden) or None — GRU hidden state

        Returns:
            features: (B, output_dim)
            new_hidden: (1, B, gru_hidden) or None
        """
        x = torch.cat([imu, waypoints, vis_feat], dim=-1)
        x = self.net(x)

        if self.use_gru:
            # GRU expects (B, seq=1, features)
            x_seq = x.unsqueeze(1)
            gru_out, new_hidden = self.gru(x_seq, hidden)
            return gru_out.squeeze(1), new_hidden

        return x, None


# ---------------------------------------------------------------------------
# Actor head  — outputs body-rate commands
# ---------------------------------------------------------------------------


class ActorHead(nn.Module):
    """
    Gaussian policy head.

    Outputs mean and log_std for each of the 4 body-rate channels:
        [roll_rate, pitch_rate, yaw_rate, thrust]
    """

    def __init__(self, cfg: ModelConfig, input_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        _in = input_dim or (cfg.gru_hidden if cfg.gru_hidden > 0 else cfg.fast_hidden)
        self.mu = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cfg.action_dim),
            nn.Tanh(),  # bound means to [-1, 1]
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
        log_std = self.log_std.clamp(
            self.cfg.log_std_min, self.cfg.log_std_max
        ).expand_as(mu)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob, entropy, **and actor mean** for a batch of actions."""
        mu, log_std = self.forward(fused)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mu


# ---------------------------------------------------------------------------
# Critic head  — estimates state value V(s)
# ---------------------------------------------------------------------------


class CriticHead(nn.Module):
    def __init__(self, cfg: ModelConfig, input_dim: int | None = None):
        super().__init__()
        _in = input_dim or (cfg.gru_hidden if cfg.gru_hidden > 0 else cfg.fast_hidden)
        self.net = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns: value (B, 1)"""
        return self.net(fused)


# ---------------------------------------------------------------------------
# Full actor-critic policy  (dual-rate)
# ---------------------------------------------------------------------------


class DronePolicy(nn.Module):
    """
    Dual-rate actor-critic policy for waypoint-following drone control.

    Architecture
    ────────────
    **Slow path** — updated when a new camera frame arrives:
        ``SlowVisualEncoder``  (frozen RAFT-Small → trainable FlowEncoder)
        Produces a visual-feature vector that is cached internally.

    **Fast path** — updated on every IMU tick:
        ``RunningNormalizer``  (online zero-mean, unit-variance for IMU + waypoints)
        → ``FastStateEncoder``  (MLP + optional GRU for temporal memory)
        → ``ActorHead``  (body-rate Gaussian policy)
        → ``CriticHead``  (V(s))

    Observation dict expected keys
    ──────────────────────────────
    For ``update_vision()`` (camera rate):
        - ``"cam_prev"``  : (B, C, H, W) float32 image in [0, 1]
        - ``"cam_curr"``  : (B, C, H, W) float32 image in [0, 1]

    For ``act()`` / ``evaluate_actions()`` (IMU rate):
        - ``"imu"``       : (B, 6) float32  [ax, ay, az, gx, gy, gz]
        - ``"waypoints"`` : (B, 6) float32  [wp1_xyz, wp2_xyz]  — body frame
        - ``"vis_feat"``  : (B, flow_feature_dim) — *optional* at inference;
                            if omitted the internally cached features are used.
        - ``"hidden"``    : (1, B, gru_hidden) — *optional* GRU hidden state;
                            if omitted the internally cached hidden is used.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        # ── Running observation normalizers ──────────────────────────────
        self.imu_norm = RunningNormalizer(self.cfg.imu_dim)
        wp_dim = self.cfg.waypoint_dim * self.cfg.num_waypoints
        self.wp_norm = RunningNormalizer(wp_dim)

        # Slow path (camera rate)
        self.slow_visual = SlowVisualEncoder(self.cfg)

        # Fast path (IMU rate) — MLP + optional GRU
        self.fast_encoder = FastStateEncoder(self.cfg)
        head_dim = self.fast_encoder.output_dim
        self.actor = ActorHead(self.cfg, input_dim=head_dim)
        self.critic = CriticHead(self.cfg, input_dim=head_dim)

        # Cached visual features — filled by update_vision(), consumed by act()
        self._vis_feat_cache: torch.Tensor | None = None

        # Cached RAFT flow map — stored in rollout buffer so FlowEncoder
        # can be re-run with gradients during PPO / BC updates.
        self._flow_cache: torch.Tensor | None = None

        # Cached GRU hidden state — carried across fast-path ticks
        self._gru_hidden_cache: torch.Tensor | None = None

    @property
    def gru_hidden_dim(self) -> int:
        """Dimensionality of the GRU hidden state (0 if GRU disabled)."""
        return self.cfg.gru_hidden if self.fast_encoder.use_gru else 0

    def init_hidden(self, batch_size: int = 1, device: str | torch.device = "cpu") -> torch.Tensor | None:
        """Return a zero-initialised GRU hidden state, or None if GRU is disabled."""
        if not self.fast_encoder.use_gru:
            return None
        return torch.zeros(1, batch_size, self.cfg.gru_hidden, device=device)

    def reset_hidden(self) -> None:
        """Clear the cached GRU hidden state (call on episode reset)."""
        self._gru_hidden_cache = None

    def train(self, mode: bool = True):
        """Override: keep RAFT frozen in eval() mode at all times."""
        super().train(mode)
        self.slow_visual.raft.eval()
        return self

    # ----- slow path (call at camera rate) --------------------------------

    def update_vision(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the slow visual encoder and cache the result.

        Should be called once per camera frame.  Between camera frames the
        fast path reuses the cached feature vector.

        Caches both the raw RAFT flow map (for later re-encoding with
        gradients during PPO / BC updates) and the compressed vis_feat
        (for fast-path action selection during rollout).

        Args:
            img_prev: (B, C, H, W) float32 in [0, 1]
            img_curr: (B, C, H, W) float32 in [0, 1]

        Returns:
            vis_feat: (B, flow_feature_dim) — also stored in internal cache.
        """
        flow = self.slow_visual.compute_flow(img_prev, img_curr)
        self._flow_cache = flow.detach()
        vis_feat = self.slow_visual.encode_flow(flow)
        self._vis_feat_cache = vis_feat.detach()
        return vis_feat

    # ----- fast path helpers ----------------------------------------------

    def _get_vis_feat(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Return visual features from *obs* dict if provided, otherwise fall
        back to the internal cache.
        """
        if "vis_feat" in obs:
            return obs["vis_feat"]
        if self._vis_feat_cache is not None:
            return self._vis_feat_cache
        # No vision yet — return zeros (e.g. first tick before first frame)
        B = obs["imu"].size(0)
        device = obs["imu"].device
        return torch.zeros(B, self.cfg.flow_feature_dim, device=device)

    def _get_hidden(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Return GRU hidden state from obs or the internal cache."""
        if not self.fast_encoder.use_gru:
            return None
        if "hidden" in obs and obs["hidden"] is not None:
            return obs["hidden"]
        return self._gru_hidden_cache

    def _encode_fast(
        self,
        obs: dict[str, torch.Tensor],
        update_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the fast-path encoder (normalise → MLP → GRU).

        Args:
            obs: observation dict with ``imu``, ``waypoints``,
                 and optionally ``vis_feat`` / ``hidden`` / ``flow``.
                 When ``flow`` (B, 2, H, W) is provided, the trainable
                 FlowEncoder is re-run with gradients to produce
                 ``vis_feat``, enabling end-to-end training.
            update_norm: if True, update running normalizer statistics
                         (should be True during rollout, False during
                         PPO update).

        Returns:
            fused:      (B, head_dim) — features for actor/critic heads.
            new_hidden: (1, B, gru_hidden) or None — updated GRU state.
        """
        # If raw flow maps are provided (PPO/BC update), re-run the
        # trainable FlowEncoder so gradients reach its weights.
        if "flow" in obs and obs["flow"] is not None:
            vis_feat = self.slow_visual.encode_flow(obs["flow"])
        else:
            vis_feat = self._get_vis_feat(obs)
        hidden = self._get_hidden(obs)

        # ── Normalise observations ────────────────────────────────────
        imu = obs["imu"]
        wp = obs["waypoints"]
        if update_norm:
            imu = self.imu_norm.update(imu)
            wp = self.wp_norm.update(wp)
        else:
            imu = self.imu_norm.normalize(imu)
            wp = self.wp_norm.normalize(wp)

        fused, new_hidden = self.fast_encoder(imu, wp, vis_feat, hidden)
        return fused, new_hidden

    # ----- public API (fast path — call at IMU rate) ----------------------

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect a single transition (inference mode).

        This is the **fast path** — call at IMU rate.  Make sure
        ``update_vision()`` has been called at least once with the latest
        camera frame pair before the first call.

        Side-effect: updates the internal GRU hidden cache (if GRU enabled)
        and the running normalizer statistics.

        Returns:
            action   : (B, 4) body-rate command  (fp32)
            log_prob : (B,)   log π(a|s)         (fp32)
            value    : (B, 1) V(s)               (fp32)
        """
        dev = obs["imu"].device
        with torch.amp.autocast(  # type: ignore[attr-defined]
            device_type=dev.type,
            dtype=torch.float16,
            enabled=(dev.type == "cuda"),
        ):
            fused, new_hidden = self._encode_fast(obs, update_norm=True)
            # Cache the new hidden state for the next tick
            if new_hidden is not None:
                self._gru_hidden_cache = new_hidden
            value = self.critic(fused)
            if deterministic:
                mu, _ = self.actor(fused)
                return mu.float(), torch.zeros(mu.size(0), device=dev), value.float()
            action, log_prob = self.actor.sample(fused)
        return action.float(), log_prob.float(), value.float()

    def get_hidden_state(self) -> torch.Tensor | None:
        """Return the current cached GRU hidden state (for storage in buffer)."""
        if self._gru_hidden_cache is not None:
            return self._gru_hidden_cache.detach()
        return None

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored transitions during PPO update.

        ``obs["vis_feat"]`` **must** be provided (stored from rollout).
        ``obs["hidden"]`` should be provided when GRU is enabled (the
        hidden state stored during rollout collection).

        Normalizer stats are NOT updated here — only ``normalize()`` is called.

        Returns:
            log_prob  : (B,)    (fp32)
            entropy   : (B,)    (fp32)
            value     : (B, 1)  (fp32)
            actor_mu  : (B, 4)  (fp32) — deterministic actor mean
        """
        dev = obs["imu"].device
        with torch.amp.autocast(  # type: ignore[attr-defined]
            device_type=dev.type,
            dtype=torch.float16,
            enabled=(dev.type == "cuda"),
        ):
            fused, _ = self._encode_fast(obs, update_norm=False)
            log_prob, entropy, mu = self.actor.evaluate(fused, actions)
            value = self.critic(fused)
        return log_prob.float(), entropy.float(), value.float(), mu.float()


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
    def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
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

    @staticmethod
    def _quat_rotate_inverse(
        q: Tuple[float, float, float, float],
        v: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Rotate vector *v* by the inverse of quaternion *q* (w, x, y, z).

        Equivalent to transforming a world-frame vector into the body
        frame described by *q*.
        """
        w, x, y, z = q
        # q_inv for unit quaternion = conjugate
        # v' = q* ⊗ v ⊗ q
        # Expanded form (avoids allocating intermediate quats):
        vx, vy, vz = v
        # t = 2 * cross(q_xyz, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        # Using conjugate: negate q_xyz → negate t
        tx, ty, tz = -tx, -ty, -tz
        # result = v + w*t + cross((-q_xyz), t)
        rx = vx + w * tx + (-y * tz - (-z) * ty)
        ry = vy + w * ty + (-z * tx - (-x) * tz)
        rz = vz + w * tz + (-x * ty - (-y) * tx)
        return (rx, ry, rz)

    def current_targets_tensor(
        self,
        drone_pos: Tuple[float, float, float],
        drone_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        Returns the two waypoints as a flat (1, 6) tensor in the **body
        frame** of the drone, ready for the policy.

        Steps:
        1. ``update(drone_pos)`` — advance / check safety radius.
        2. Subtract ``drone_pos`` → relative world-frame offsets.
        3. Rotate by inverse of ``drone_quat`` → body-frame offsets.

        Args:
            drone_pos:  (x, y, z) world-frame position.
            drone_quat: (w, x, y, z) world-frame orientation quaternion.
            device:     torch device for the output tensor.

        Returns:
            (1, 6) tensor — [wp0_body_xyz, wp1_body_xyz].
        """
        wp0, wp1 = self.update(drone_pos)

        # world-frame offset
        off0 = (wp0[0] - drone_pos[0], wp0[1] - drone_pos[1], wp0[2] - drone_pos[2])
        off1 = (wp1[0] - drone_pos[0], wp1[1] - drone_pos[1], wp1[2] - drone_pos[2])

        # rotate into body frame
        b0 = self._quat_rotate_inverse(drone_quat, off0)
        b1 = self._quat_rotate_inverse(drone_quat, off1)

        data = [*b0, *b1]
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
    Ties the dual-rate policy and waypoint buffer together for rollout
    collection.

    The agent exposes two call paths that mirror the policy's split:

    - ``update_vision(obs)`` — call whenever a **new camera frame** arrives.
      Runs the slow visual encoder (RAFT-Small → FlowEncoder) and caches
      the resulting feature vector inside the policy.

    - ``step(obs)`` — call on **every IMU tick**.  Uses cached visual
      features + fresh IMU / waypoint data to produce a body-rate command.

    Usage
    -----
    >>> cfg = ModelConfig(safety_radius=1.5)
    >>> agent = DroneAgent(cfg, route=[(0,0,5), (10,0,5), (10,10,5), (0,10,5)])
    >>> obs = env.reset()
    >>> while not done:
    ...     if obs.get("new_frame", False):
    ...         agent.update_vision(obs)
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

        # Previous frame for optical-flow computation
        self._prev_frame: Optional[torch.Tensor] = None

        if route is not None:
            self.set_route(route)

    # ----- route management -----------------------------------------------

    def set_route(self, route: List[Tuple[float, float, float]]) -> None:
        """(Re-)initialise the waypoint buffer with a new route."""
        self._buffer = WaypointBuffer(route, self.cfg.safety_radius)

    def reset_vision(self) -> None:
        """Clear cached visual and temporal state (call on episode reset)."""
        self._prev_frame = None
        self.policy._vis_feat_cache = None
        self.policy._flow_cache = None
        self.policy.reset_hidden()

    # ----- image helpers --------------------------------------------------

    @staticmethod
    def _prepare_image(
        raw: torch.Tensor | object,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalise a raw image to (1, C, H, W) float32 in [0, 1]."""
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=torch.float32)
        if raw.ndim == 3 and raw.shape[-1] in (1, 3):
            raw = raw.permute(2, 0, 1)  # HWC → CHW
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)  # add batch dim
        if raw.max() > 1.0:
            raw = raw / 255.0
        return raw.to(device)

    # ----- slow path (camera rate) ----------------------------------------

    @torch.no_grad()
    def update_vision(self, obs: dict) -> torch.Tensor | None:
        """
        Run the slow visual encoder on the latest camera frame.

        ``obs`` must contain ``"cam"`` (current frame).  The agent keeps
        the previous frame internally.  On the very first call (no prior
        frame), the previous frame is set equal to the current one (zero
        flow).

        Returns:
            vis_feat: (1, flow_feature_dim) or ``None`` if skipped.
        """
        cam_curr = self._prepare_image(obs["cam"], self.device)

        if self._prev_frame is None:
            # First frame → duplicate (zero flow baseline)
            self._prev_frame = cam_curr.clone()

        vis_feat = self.policy.update_vision(self._prev_frame, cam_curr)
        self._prev_frame = cam_curr
        return vis_feat

    # ----- fast path (IMU rate) -------------------------------------------

    def step(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Perform one decision step at **IMU rate**.

        ``obs`` must contain:

        - ``"imu"``       : (6,) or (B, 6) float
        - ``"drone_pos"`` : (3,) float — used to update the waypoint buffer

        ``"cam"`` is **not** required here; vision is updated separately
        via ``update_vision()``.

        Returns:
            action: (4,) numpy-compatible body-rate command
        """
        assert self._buffer is not None, "Call set_route() before step()."

        # --- prepare IMU -----------------------------------------------------
        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        if imu.ndim == 1:
            imu = imu.unsqueeze(0)
        imu = imu.to(self.device)

        # --- update waypoint buffer & build tensor ---------------------------
        dp = [float(x) for x in obs["drone_pos"][:3]]
        drone_pos: Tuple[float, float, float] = (dp[0], dp[1], dp[2])

        # orientation (w, x, y, z) — needed for world→body transform
        dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
        drone_quat: Tuple[float, float, float, float] = (dq[0], dq[1], dq[2], dq[3])

        wp_tensor = self._buffer.current_targets_tensor(
            drone_pos,
            drone_quat=drone_quat,
            device=self.device,
        )

        # --- forward pass (fast path only) -----------------------------------
        policy_obs = {"imu": imu, "waypoints": wp_tensor}
        action, log_prob, value = self.policy.act(
            policy_obs,
            deterministic=deterministic,
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

    Dual-rate design
    ────────────────
    Every stored transition corresponds to one **IMU tick** (fast path).
    Camera / optical-flow features are computed at the slower camera rate
    and the resulting ``vis_feat`` vector is stored alongside each transition
    (it stays constant between camera frames).  During PPO updates the
    policy's fast path is re-evaluated using the stored ``vis_feat``, so we
    never need to re-run RAFT — this keeps the update cheap.

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
        imu_dim: int,
        wp_dim: int,
        vis_feat_dim: int,
        action_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        gru_hidden_dim: int = 0,
        flow_height: int = 0,
        flow_width: int = 0,
    ):
        """
        Args:
            buffer_size:  Number of transitions to collect per rollout epoch.
            imu_dim:      Dimensionality of the IMU vector (default 6).
            wp_dim:       Dimensionality of the flattened waypoint vector (default 6).
            vis_feat_dim: Dimensionality of the cached visual feature vector.
            action_dim:   Dimensionality of the action (default 4).
            device:       "cpu" or "cuda".
            gamma:        Discount factor.
            gae_lambda:   GAE λ for bias–variance trade-off.
            gru_hidden_dim: GRU hidden size (0 = no GRU state stored).
            flow_height:  Height of RAFT flow maps (0 = don't store flow).
            flow_width:   Width of RAFT flow maps (0 = don't store flow).
        """
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.gru_hidden_dim = gru_hidden_dim

        # ----- pre-allocate storage ------------------------------------------
        self.imu = torch.zeros(buffer_size, imu_dim, device=self.device)
        self.waypoints = torch.zeros(buffer_size, wp_dim, device=self.device)
        self.vis_feat = torch.zeros(buffer_size, vis_feat_dim, device=self.device)
        self.actions = torch.zeros(buffer_size, action_dim, device=self.device)
        self.expert_actions = torch.zeros(buffer_size, action_dim, device=self.device)
        self.log_probs = torch.zeros(buffer_size, device=self.device)
        self.rewards = torch.zeros(buffer_size, device=self.device)
        self.values = torch.zeros(buffer_size, device=self.device)
        self.dones = torch.zeros(buffer_size, device=self.device)

        # GRU hidden states (stored per-step for PPO re-evaluation)
        if gru_hidden_dim > 0:
            self.hidden_states = torch.zeros(
                buffer_size, gru_hidden_dim, device=self.device
            )
        else:
            self.hidden_states = None

        # RAFT flow maps — stored on **CPU** to save GPU memory.
        # During PPO/BC updates, mini-batches are moved to GPU on-the-fly
        # so the trainable FlowEncoder can be re-run with gradients.
        self.flow_height = flow_height
        self.flow_width = flow_width
        if flow_height > 0 and flow_width > 0:
            self.flow_maps = torch.zeros(
                buffer_size, 2, flow_height, flow_width,
                device="cpu",
            )
        else:
            self.flow_maps = None

        # ----- computed after finish() ---------------------------------------
        self.advantages = torch.zeros(buffer_size, device=self.device)
        self.returns = torch.zeros(buffer_size, device=self.device)

        self.ptr = 0
        self.full = False

    # ----- storage --------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer for a new rollout epoch."""
        self.ptr = 0
        self.full = False

    def store(
        self,
        imu: torch.Tensor,
        waypoints: torch.Tensor,
        vis_feat: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
        hidden: torch.Tensor | None = None,
        flow: torch.Tensor | None = None,
        expert_action: torch.Tensor | None = None,
    ) -> None:
        """
        Append a single transition.

        All tensor inputs should already be on ``self.device`` and have no
        batch dimension.

        Args:
            hidden: (gru_hidden_dim,) — GRU hidden state *before* this step.
                    Required when ``gru_hidden_dim > 0``.
            flow:   (2, H, W) — RAFT flow map for this step (stored on CPU).
                    When provided, FlowEncoder is re-run with gradients
                    during PPO / BC updates.
            expert_action: (action_dim,) — action the expert PD controller
                    would produce for this observation.  Used as the BC
                    auxiliary anchor during PPO.
        """
        assert self.ptr < self.buffer_size, "Buffer full — call finish() then reset()."

        i = self.ptr
        self.imu[i] = imu
        self.waypoints[i] = waypoints
        self.vis_feat[i] = vis_feat
        self.actions[i] = action
        if expert_action is not None:
            self.expert_actions[i] = expert_action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value.squeeze()
        self.dones[i] = float(done)
        if self.hidden_states is not None and hidden is not None:
            self.hidden_states[i] = hidden
        if self.flow_maps is not None and flow is not None:
            self.flow_maps[i] = flow.detach().cpu()
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

            delta = (
                self.rewards[t].item()
                + self.gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
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

            obs_dict: dict[str, torch.Tensor] = {
                "imu": self.imu[idx],
                "waypoints": self.waypoints[idx],
                "vis_feat": self.vis_feat[idx],
            }
            # Include stored RAFT flow maps so FlowEncoder can be
            # re-run with gradients during PPO / BC updates.
            if self.flow_maps is not None:
                obs_dict["flow"] = self.flow_maps[idx.cpu()].to(self.device)
            # Include stored GRU hidden states for re-evaluation
            if self.hidden_states is not None:
                # Stored as (B, gru_hidden) → policy expects (1, B, gru_hidden)
                obs_dict["hidden"] = self.hidden_states[idx].unsqueeze(0)

            yield {
                "obs": obs_dict,
                "actions": self.actions[idx],
                "expert_actions": self.expert_actions[idx],
                "old_log_probs": self.log_probs[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
            }


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """
    Proximal Policy Optimisation trainer for the **dual-rate** policy.

    During rollout collection, every env step is treated as an IMU tick.
    When the env also provides a new camera frame (indicated by
    ``obs["new_frame"]`` being truthy, or ``obs["cam"]`` being present),
    the slow visual encoder is run and its output is cached / stored
    alongside the transition.

    During the PPO update the **fast path** + actor/critic heads are
    re-evaluated.  Raw RAFT flow maps stored in the rollout buffer are
    re-encoded through the trainable ``FlowEncoder`` CNN **with gradients**,
    so FlowEncoder is properly trained alongside the rest of the policy.

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
        wp_dim = self.cfg.waypoint_dim * self.cfg.num_waypoints

        self.buffer = RolloutBuffer(
            buffer_size=rollout_steps,
            imu_dim=self.cfg.imu_dim,
            wp_dim=wp_dim,
            vis_feat_dim=self.cfg.flow_feature_dim,
            action_dim=self.cfg.action_dim,
            device=device,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            gru_hidden_dim=agent.policy.gru_hidden_dim,
            flow_height=self.cfg.cam_height,
            flow_width=self.cfg.cam_width,
        )

        self.optimiser = torch.optim.Adam(
            agent.policy.parameters(),
            lr=self.cfg.lr,
        )

    # ----- rollout collection ---------------------------------------------

    def collect_rollout(self, env) -> dict:
        """
        Run the current policy in ``env`` for ``rollout_steps`` transitions
        and fill the rollout buffer.

        ``env`` must expose:
            - ``reset() -> obs``
            - ``step(action) -> (obs, reward, done, info)``

        where ``obs`` is a dict with keys:
            - ``"imu"``       : (6,) float
            - ``"drone_pos"`` : (3,) float
            - ``"cam"``       : (H, W, C) uint8/float — **when a new frame
                                is available**
            - ``"new_frame"`` : bool — True when ``"cam"`` contains a fresh
                                frame (optional; if absent, the presence of
                                ``"cam"`` is used).

        Returns:
            info dict with ``total_reward`` and ``episodes_completed``.
        """
        self.buffer.reset()
        self.agent.reset_vision()
        obs = env.reset()
        total_reward = 0.0
        episodes = 0

        for _ in range(self.rollout_steps):
            # --- slow path: update vision if new frame available -------------
            has_new_frame = obs.get("new_frame", "cam" in obs)
            if has_new_frame and "cam" in obs:
                self.agent.update_vision(obs)

            # --- prepare fast-path tensors -----------------------------------
            imu, wp_tensor = self._obs_to_tensors(obs)

            # Get cached visual features for storage
            vis_feat = self.agent.policy._vis_feat_cache
            if vis_feat is None:
                vis_feat = torch.zeros(
                    1,
                    self.cfg.flow_feature_dim,
                    device=self.agent.device,
                )

            # Get cached RAFT flow map for FlowEncoder re-training
            flow = self.agent.policy._flow_cache  # (1, 2, H, W) or None

            # Snapshot GRU hidden state *before* act() updates it
            pre_hidden = self.agent.policy.get_hidden_state()

            policy_obs = {
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
                "vis_feat": vis_feat,
            }
            action, log_prob, value = self.agent.policy.act(policy_obs)
            action_sq = action.squeeze(0)

            # --- environment step --------------------------------------------
            action_np = action_sq.cpu()
            next_obs, reward, done, info = env.step(action_np)

            # --- store -------------------------------------------------------
            # Flatten hidden (1, 1, H) → (H,) for buffer storage
            hidden_flat = (
                pre_hidden.squeeze(0).squeeze(0)
                if pre_hidden is not None
                else None
            )
            self.buffer.store(
                imu=imu,
                waypoints=wp_tensor.squeeze(0),
                vis_feat=vis_feat.squeeze(0),
                action=action_sq,
                log_prob=log_prob.squeeze(0),
                reward=float(reward),
                value=value,
                done=done,
                hidden=hidden_flat,
                flow=flow.squeeze(0) if flow is not None else None,
            )
            total_reward += float(reward)

            if done:
                obs = env.reset()
                self.agent.reset_vision()
                episodes += 1
            else:
                obs = next_obs

        # --- bootstrap last value --------------------------------------------
        imu, wp_tensor = self._obs_to_tensors(obs)
        vis_feat = self.agent.policy._vis_feat_cache
        if vis_feat is None:
            vis_feat = torch.zeros(
                1,
                self.cfg.flow_feature_dim,
                device=self.agent.device,
            )
        with torch.no_grad():
            policy_obs = {
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
                "vis_feat": vis_feat,
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
                new_log_probs, entropy, values, _mu = self.agent.policy.evaluate_actions(
                    obs, actions
                )
                values = values.squeeze(-1)

                # --- policy (actor) loss -----------------------------------
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value (critic) loss -----------------------------------
                value_loss = F.mse_loss(values, returns)

                # --- total loss --------------------------------------------
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy.mean()
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(),
                    self.cfg.max_grad_norm,
                )
                self.optimiser.step()

                # --- logging -----------------------------------------------
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                num_batches += 1

        n = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
        }

    # ----- helpers --------------------------------------------------------

    def _obs_to_tensors(self, obs: dict):
        """Convert a raw env obs dict to device tensors (no batch dim)."""
        device = self.agent.device

        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        imu = imu.to(device)

        dp = [float(x) for x in obs["drone_pos"][:3]]
        drone_pos: Tuple[float, float, float] = (dp[0], dp[1], dp[2])

        dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
        drone_quat: Tuple[float, float, float, float] = (dq[0], dq[1], dq[2], dq[3])

        assert self.agent._buffer is not None, "Call agent.set_route() first."
        wp_tensor = self.agent._buffer.current_targets_tensor(
            drone_pos,
            drone_quat=drone_quat,
            device=device,
        )
        return imu, wp_tensor


# ---------------------------------------------------------------------------
# Curriculum  — progressive route difficulty
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """Defines one difficulty level in the training curriculum."""

    name: str
    num_waypoints: int  # how many WPs per episode
    min_spacing: float  # minimum distance between consecutive WPs (m)
    max_spacing: float  # maximum distance
    max_angle_deg: float  # max heading change between consecutive WPs
    safety_radius: float  # waypoint reach threshold (can relax early on)
    altitude_range: Tuple[float, float] = (3.0, 7.0)  # (min_z, max_z)
    success_threshold: float = 0.8  # fraction of WPs reached to "pass"
    min_episodes: int = 50  # minimum episodes before promotion


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
        if (
            len(self._history) >= stage.min_episodes
            and self.success_rate >= stage.success_threshold
        ):
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
