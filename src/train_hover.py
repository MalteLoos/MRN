#!/usr/bin/env python3
"""
train_hover.py — Short-episode hover training for the dual-rate drone policy.

The drone is spawned at its default takeoff altitude and must hold position.
Each episode lasts 7 s of sim-time (350 env-steps at 50 Hz).

Waypoints are placed at the hover position so the policy learns to stay
in place before it is later finetuned for navigation.

Checkpoints are saved every ``--ckpt-every`` epochs.
Metrics are logged to **Weights & Biases in offline mode** so nothing is
uploaded during training — call ``wandb sync <run_dir>`` afterwards.

Usage
-----
    python3 src/train_hover.py                              # defaults
    python3 src/train_hover.py --epochs 500 --device cuda   # custom
    python3 src/train_hover.py --resume runs/hover/ckpt_100.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple  # noqa: F401 — Tuple used in annotations

import numpy as np
import torch

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    DroneAgent,
    DronePolicy,
    ModelConfig,
    RolloutBuffer,
    WaypointBuffer,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Hover Environment Wrapper
# ═══════════════════════════════════════════════════════════════════════════


class HoverEnvWrapper:
    """
    Thin wrapper around ``PX4GazeboEnv`` that:

    1. Maps the environment's obs dict to the keys the model expects.
    2. Sets up a trivial 2-waypoint route at the hover position.
    3. Enforces a 7 s episode length.
    4. Signals ``new_frame`` whenever the camera frame changes.
    5. Computes a hover-centric reward.

    Obs mapping
    ───────────
    env key        → model key
    ``camera``     → ``cam``
    ``imu``        → ``imu``
    ``position``   → ``drone_pos``
    ``orientation``→ ``drone_quat``
    ``velocity``   → ``velocity``   (kept for reward)
    """

    def __init__(
        self,
        env,
        hover_alt: float = 2.5,
        episode_seconds: float = 3.0,
        env_dt: float = 0.02,
    ):
        self.env = env
        self.hover_alt = hover_alt
        self.episode_seconds = episode_seconds
        self.max_steps = int(episode_seconds / env_dt)

        self._step_count = 0
        self._prev_cam_id: int | None = None  # track frame identity

    # ── obs remapping ──────────────────────────────────────────────────────

    @staticmethod
    def _remap_obs(
        obs: dict[str, np.ndarray],
        new_frame: bool,
    ) -> dict[str, Any]:
        """Remap env observation keys to model-expected keys."""
        return {
            "cam": obs["camera"],  # (H, W, 3) uint8
            "imu": obs["imu"],  # (6,) float32
            "drone_pos": obs["position"],  # (3,) float32  ENU
            "drone_quat": obs["orientation"],  # (4,) float32  (w,x,y,z)
            "velocity": obs["velocity"],  # (3,) float32  ENU
            "new_frame": new_frame,
        }

    def _detect_new_frame(self, obs: dict[str, np.ndarray]) -> bool:
        """Cheap identity check — cameras don't update every tick."""
        cam_id = id(obs["camera"])
        if cam_id != self._prev_cam_id:
            self._prev_cam_id = cam_id
            return True
        return False

    # ── gymnasium-like API ─────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        obs, _info = self.env.reset()
        self._step_count = 0
        self._prev_cam_id = None
        new_frame = self._detect_new_frame(obs)
        return self._remap_obs(obs, new_frame)

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> Tuple[dict[str, Any], float, bool, dict]:
        # Map the 4-D model action (roll_rate, pitch_rate, yaw_rate, thrust)
        # to the env's 4-D action (roll, pitch, yaw_rate, thrust).
        act_np = np.asarray(action, dtype=np.float32)
        env_action = np.array(
            [act_np[0], act_np[1], act_np[2], act_np[3]],
            dtype=np.float32,
        )

        obs, _env_reward, terminated, truncated, info = self.env.step(env_action)
        self._step_count += 1

        new_frame = self._detect_new_frame(obs)
        mapped = self._remap_obs(obs, new_frame)
        reward = self._compute_hover_reward(mapped, act_np)

        done = terminated or (self._step_count >= self.max_steps)
        return mapped, reward, done, info

    def publish_waypoints_relative(self, wp_flat) -> None:
        """Forward body-frame waypoint tensor to ROS 2 for debugging."""
        if hasattr(self.env, "publish_waypoints_relative"):
            self.env.publish_waypoints_relative(wp_flat)

    # ── hover-centric reward ───────────────────────────────────────────────

    def _compute_hover_reward(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
    ) -> float:
        """
        Reward for holding position at ``hover_alt`` with minimal tilt.

        Components (all negative penalties + small alive bonus):
            -  altitude error
            -  horizontal drift
            -  velocity magnitude
            -  action magnitude (energy)
            +  alive bonus
        """
        pos = obs["drone_pos"]  # ENU
        vel = obs["velocity"]

        # altitude error (squared — RMS-style)
        alt_err = (float(pos[2]) - self.hover_alt) ** 2
        # horizontal drift from origin
        horiz_drift = math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2)
        # velocity penalty
        speed = float(np.linalg.norm(vel))
        # action penalty (exclude thrust)
        act_mag = float(np.sum(action[:3] ** 2))

        # thrust penalty — penalise thrust below 0.3 so the
        # policy learns to keep the motors spinning.  action[3]
        # is in [-1, 1]; actual thrust = (action[3]+1)/2.
        thrust = float(action[3])
        low_thrust_penalty = max(0.0, 0.35 - abs(thrust)) * 5.0  # up to 1.5

        reward = (
            -1.0 * alt_err
            - 0.5 * horiz_drift
            - 0.2 * speed
            - 0.05 * act_mag
            - low_thrust_penalty
            + 0.1  # alive bonus
        )
        return reward


# ═══════════════════════════════════════════════════════════════════════════
#  Dummy Hover Environment (for testing without Gazebo)
# ═══════════════════════════════════════════════════════════════════════════


class DummyHoverEnv:
    """
    A simple physics-free dummy that imitates ``HoverEnvWrapper`` for
    offline testing of the training loop.

    Set ``--dummy`` on the command line to use this instead of the real sim.
    """

    def __init__(
        self,
        hover_alt: float = 2.5,
        episode_seconds: float = 3.0,
        env_dt: float = 0.02,
        cam_h: int = 128,
        cam_w: int = 128,
    ):
        self.hover_alt = hover_alt
        self.max_steps = int(episode_seconds / env_dt)
        self.cam_h = cam_h
        self.cam_w = cam_w
        self.dt = env_dt
        self._step_count = 0
        self._pos = np.array([0.0, 0.0, hover_alt], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._frame_counter = 0

    def reset(self) -> dict[str, Any]:
        self._step_count = 0
        self._pos = np.array([0.0, 0.0, self.hover_alt], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._frame_counter = 0
        return self._obs(new_frame=True)

    def step(self, action):
        act = np.asarray(action, dtype=np.float32)
        # Extremely simplified dynamics
        acc = np.array(
            [act[0] * 2.0, act[1] * 2.0, (act[3] - 0.5) * 5.0], dtype=np.float32
        )
        self._vel += acc * self.dt
        self._vel *= 0.98  # drag
        self._pos += self._vel * self.dt
        self._step_count += 1

        # Camera updates at ~30 Hz ≈ every 1-2 env steps
        self._frame_counter += 1
        new_frame = self._frame_counter % 2 == 0

        obs = self._obs(new_frame)
        reward = self._reward(obs, act)
        done = self._step_count >= self.max_steps
        return obs, reward, done, {}

    def _obs(self, new_frame: bool) -> dict[str, Any]:
        return {
            "cam": np.random.randint(
                0, 255, (self.cam_h, self.cam_w, 3), dtype=np.uint8
            ),
            "imu": np.concatenate(
                [
                    np.random.randn(3).astype(np.float32) * 0.1,
                    np.random.randn(3).astype(np.float32) * 0.01,
                ]
            ),
            "drone_pos": self._pos.copy(),
            "drone_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "velocity": self._vel.copy(),
            "new_frame": new_frame,
        }

    def _reward(self, obs, action):
        pos = obs["drone_pos"]
        vel = obs["velocity"]
        alt_err = (float(pos[2]) - self.hover_alt) ** 2
        horiz = math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2)
        speed = float(np.linalg.norm(vel))
        act_mag = float(np.sum(action[:3] ** 2))
        thrust = float(action[3] + 1.0) / 2.0
        low_thrust_penalty = max(0.0, 0.3 - thrust) * 5.0
        return -1.0 * alt_err - 0.5 * horiz - 0.2 * speed - 0.05 * act_mag - low_thrust_penalty + 0.1


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════


def make_env(args):
    """Build the hover environment (real, dummy, or vectorised).

    Returns
    -------
    HoverEnvWrapper | DummyHoverEnv
        When ``--num-envs 1`` (default, single-process).
    SubprocVecEnv
        When ``--num-envs N`` with N > 1 (multi-process, each
        worker has its own PX4 + Gazebo stack).
    """
    if args.dummy:
        return DummyHoverEnv(
            hover_alt=args.hover_alt,
            episode_seconds=args.episode_secs,
            cam_h=args.cam_size,
            cam_w=args.cam_size,
        )

    num_envs = getattr(args, "num_envs", 1)

    if num_envs > 1:
        # Multi-process: each worker gets its own sim stack
        from px4_gz_gym.vec_env import SubprocVecEnv

        launch_script = getattr(args, "launch_script", "")
        return SubprocVecEnv.from_args(
            num_envs=num_envs,
            args=args,
            base_domain_id=10,
            launch_script=launch_script,
        )

    # Single-process: direct PX4GazeboEnv
    from px4_gz_gym.env import PX4GazeboEnv

    raw_env = PX4GazeboEnv(
        world_name=args.world,
        cam_obs_height=args.cam_size,
        cam_obs_width=args.cam_size,
        max_episode_steps=int(args.episode_secs / 0.02),
        takeoff_alt=args.hover_alt,
    )
    return HoverEnvWrapper(
        raw_env,
        hover_alt=args.hover_alt,
        episode_seconds=args.episode_secs,
    )


def collect_rollout(
    agent: DroneAgent,
    env: HoverEnvWrapper | DummyHoverEnv,
    buffer: RolloutBuffer,
    rollout_steps: int,
    cfg: ModelConfig,
) -> dict[str, float]:
    """
    Collect ``rollout_steps`` transitions from a **single** env.

    Handles the dual-rate update: vision is only updated when the env
    signals ``new_frame``; the fast path runs every tick.
    """
    buffer.reset()
    agent.reset_vision()
    obs = env.reset()

    # Set a trivial hover route (2 waypoints at the target hover position)
    hover_pos: Tuple[float, float, float] = (0.0, 0.0, env.hover_alt)
    agent.set_route([hover_pos, hover_pos])
    assert agent._buffer is not None

    total_reward = 0.0
    episodes = 0
    device = agent.device

    for _ in range(rollout_steps):
        # ── slow path: update vision on new camera frame ──────────
        if obs.get("new_frame", False) and "cam" in obs:
            agent.update_vision(obs)

        # ── prepare fast-path tensors ─────────────────────────────
        imu = obs["imu"]
        if not isinstance(imu, torch.Tensor):
            imu = torch.as_tensor(imu, dtype=torch.float32)
        imu = imu.to(device)

        dp = [float(x) for x in obs["drone_pos"][:3]]
        drone_pos: Tuple[float, float, float] = (dp[0], dp[1], dp[2])
        dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
        drone_quat: Tuple[float, float, float, float] = (dq[0], dq[1], dq[2], dq[3])
        wp_tensor = agent._buffer.current_targets_tensor(
            drone_pos,
            drone_quat=drone_quat,
            device=device,
        )

        # Publish body-frame relative waypoints for RViz debugging
        env.publish_waypoints_relative(wp_tensor.squeeze(0).cpu().numpy())

        vis_feat = agent.policy._vis_feat_cache
        if vis_feat is None:
            vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)

        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        action, log_prob, value = agent.policy.act(policy_obs)
        action_sq = action.squeeze(0)

        # ── environment step ──────────────────────────────────────
        next_obs, reward, done, info = env.step(action_sq.cpu())

        # ── store transition ──────────────────────────────────────
        buffer.store(
            imu=imu,
            waypoints=wp_tensor.squeeze(0),
            vis_feat=vis_feat.squeeze(0),
            action=action_sq,
            log_prob=log_prob.squeeze(0),
            reward=float(reward),
            value=value,
            done=done,
        )
        total_reward += float(reward)

        if done:
            obs = env.reset()
            agent.reset_vision()
            hover_pos = (0.0, 0.0, env.hover_alt)
            agent.set_route([hover_pos, hover_pos])
            episodes += 1
        else:
            obs = next_obs

    # ── bootstrap last value ──────────────────────────────────────
    imu = obs["imu"]
    if not isinstance(imu, torch.Tensor):
        imu = torch.as_tensor(imu, dtype=torch.float32).to(device)
    dp = [float(x) for x in obs["drone_pos"][:3]]
    drone_pos = (dp[0], dp[1], dp[2])
    dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
    drone_quat = (dq[0], dq[1], dq[2], dq[3])
    assert agent._buffer is not None
    wp_tensor = agent._buffer.current_targets_tensor(
        drone_pos,
        drone_quat=drone_quat,
        device=device,
    )
    vis_feat = agent.policy._vis_feat_cache
    if vis_feat is None:
        vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)
    with torch.no_grad():
        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        _, _, last_value = agent.policy.act(policy_obs)
    buffer.finish(last_value)

    return {
        "total_reward": total_reward,
        "episodes": max(episodes, 1),
        "mean_reward": total_reward / max(episodes, 1),
    }


def collect_rollout_vec(
    agent: DroneAgent,
    vec_env,
    buffer: RolloutBuffer,
    rollout_steps: int,
    cfg: ModelConfig,
    hover_alt: float = 2.5,
) -> dict[str, float]:
    """Collect transitions from **N parallel envs** round-robin.

    Each env steps independently; done envs are reset inline.
    Transitions from all envs are interleaved into a single
    ``RolloutBuffer`` in deterministic env-index order.

    The buffer must be sized for ``rollout_steps`` total transitions
    (spread across all envs).

    Determinism
    ~~~~~~~~~~~
    * Each env performs its own deterministic sim-stepped reset.
    * Results are always processed in env-index order (0, 1, …, N-1)
      so the buffer contents are reproducible for the same seeds.
    """
    from px4_gz_gym.vec_env import SubprocVecEnv, AsyncResetVecEnv

    num_envs = vec_env.num_envs
    buffer.reset()
    agent.reset_vision()
    device = agent.device

    # ── Initial reset of all envs ────────────────────────────
    if isinstance(vec_env, AsyncResetVecEnv):
        obs_list = vec_env.reset_all()
    else:
        obs_list = vec_env.reset_all()

    # Wrap raw obs in HoverEnvWrapper-style remap
    def _remap(obs_dict: dict) -> dict:
        """Raw PX4GazeboEnv obs → model-expected keys."""
        return {
            "cam": obs_dict["camera"],
            "imu": obs_dict["imu"],
            "drone_pos": obs_dict["position"],
            "drone_quat": obs_dict["orientation"],
            "velocity": obs_dict["velocity"],
            "new_frame": True,  # first frame after reset
        }

    obs_per_env = [_remap(o) for o in obs_list]

    # Set hover route for agent — target is directly above spawn
    hover_pos: Tuple[float, float, float] = (0.0, 0.0, hover_alt)
    agent.set_route([hover_pos, hover_pos])
    assert agent._buffer is not None

    total_reward = 0.0
    episodes = 0
    steps_collected = 0

    while steps_collected < rollout_steps:
        # ── Compute actions for all envs ─────────────────────
        actions = []
        log_probs = []
        values = []
        imus = []
        wp_tensors = []
        vis_feats = []

        for i in range(num_envs):
            if steps_collected + i >= rollout_steps:
                break
            obs = obs_per_env[i]

            # Slow path: vision
            if obs.get("new_frame", False) and "cam" in obs:
                agent.update_vision(obs)

            imu = obs["imu"]
            if not isinstance(imu, torch.Tensor):
                imu = torch.as_tensor(imu, dtype=torch.float32)
            imu = imu.to(device)

            dp = [float(x) for x in obs["drone_pos"][:3]]
            drone_pos = (dp[0], dp[1], dp[2])
            dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
            drone_quat = (dq[0], dq[1], dq[2], dq[3])
            wp_tensor = agent._buffer.current_targets_tensor(
                drone_pos, drone_quat=drone_quat, device=device,
            )

            vis_feat = agent.policy._vis_feat_cache
            if vis_feat is None:
                vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)

            policy_obs = {
                "imu": imu.unsqueeze(0),
                "waypoints": wp_tensor,
                "vis_feat": vis_feat,
            }
            action, log_prob, value = agent.policy.act(policy_obs)
            action_sq = action.squeeze(0)

            actions.append(action_sq.cpu().numpy())
            log_probs.append(log_prob.squeeze(0))
            values.append(value)
            imus.append(imu)
            wp_tensors.append(wp_tensor.squeeze(0))
            vis_feats.append(vis_feat.squeeze(0))

        n_active = len(actions)
        if n_active == 0:
            break

        # ── Step all active envs in parallel ─────────────────
        if isinstance(vec_env, AsyncResetVecEnv):
            step_results = vec_env.step(
                [np.asarray(a, dtype=np.float32) for a in actions]
                + [np.zeros(4, dtype=np.float32)] * (num_envs - n_active)
            )
        else:
            step_results = vec_env.step(
                [np.asarray(a, dtype=np.float32) for a in actions]
                + [np.zeros(4, dtype=np.float32)] * (num_envs - n_active)
            )

        # ── Store transitions (deterministic env-index order) ─
        for i in range(n_active):
            if steps_collected >= rollout_steps:
                break

            if isinstance(vec_env, AsyncResetVecEnv):
                obs_raw, reward, done, info = step_results[i]
                next_obs_raw = obs_raw
            else:
                obs_raw, reward, term, trunc, info = step_results[i]  # type: ignore[misc]
                done = bool(term or trunc)
                next_obs_raw = obs_raw

            buffer.store(
                imu=imus[i],
                waypoints=wp_tensors[i],
                vis_feat=vis_feats[i],
                action=torch.as_tensor(actions[i], dtype=torch.float32).to(device),
                log_prob=log_probs[i],
                reward=float(reward),
                value=values[i],
                done=done,
            )
            total_reward += float(reward)
            steps_collected += 1

            if done:
                if isinstance(vec_env, AsyncResetVecEnv):
                    # Reset obs will be collected on next step
                    reset_obs = vec_env.get_reset_obs(i)
                    obs_per_env[i] = _remap(reset_obs)
                else:
                    reset_obs = vec_env.reset_one(i)
                    obs_per_env[i] = _remap(reset_obs)
                agent.reset_vision()
                hover_pos = (0.0, 0.0, hover_alt)
                agent.set_route([hover_pos, hover_pos])
                episodes += 1
            else:
                obs_per_env[i] = _remap(next_obs_raw)

    # ── Bootstrap last value ─────────────────────────────────
    obs = obs_per_env[0]
    imu = obs["imu"]
    if not isinstance(imu, torch.Tensor):
        imu = torch.as_tensor(imu, dtype=torch.float32).to(device)
    dp = [float(x) for x in obs["drone_pos"][:3]]
    drone_pos = (dp[0], dp[1], dp[2])
    dq = [float(x) for x in obs.get("drone_quat", [1, 0, 0, 0])[:4]]
    drone_quat = (dq[0], dq[1], dq[2], dq[3])
    assert agent._buffer is not None
    wp_tensor = agent._buffer.current_targets_tensor(
        drone_pos, drone_quat=drone_quat, device=device,
    )
    vis_feat = agent.policy._vis_feat_cache
    if vis_feat is None:
        vis_feat = torch.zeros(1, cfg.flow_feature_dim, device=device)
    with torch.no_grad():
        policy_obs = {
            "imu": imu.unsqueeze(0),
            "waypoints": wp_tensor,
            "vis_feat": vis_feat,
        }
        _, _, last_value = agent.policy.act(policy_obs)
    buffer.finish(last_value)

    return {
        "total_reward": total_reward,
        "episodes": max(episodes, 1),
        "mean_reward": total_reward / max(episodes, 1),
    }


def ppo_update(
    agent: DroneAgent,
    buffer: RolloutBuffer,
    optimiser: torch.optim.Optimizer,
    cfg: ModelConfig,
    ppo_epochs: int,
    batch_size: int,
    scaler: torch.amp.GradScaler | None = None,  # type: ignore[attr-defined]
) -> dict[str, float]:
    """Run ``ppo_epochs`` of clipped PPO over the filled buffer."""
    import torch.nn as nn

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    num_batches = 0

    for _ in range(ppo_epochs):
        for batch in buffer.sample_batches(batch_size):
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            new_log_probs, entropy, values = agent.policy.evaluate_actions(obs, actions)
            values = values.squeeze(-1)

            # policy loss
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = torch.nn.functional.mse_loss(values, returns)

            # combined
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy.mean()
            )

            optimiser.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.max_grad_norm)
                optimiser.step()

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


def save_checkpoint(
    agent: DroneAgent,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    stats: dict,
    path: str,
) -> None:
    """Save model + optimiser state + metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "policy_state_dict": agent.policy.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "stats": stats,
        },
        path,
    )
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(
    agent: DroneAgent,
    optimiser: torch.optim.Optimizer,
    path: str,
) -> int:
    """Load checkpoint and return the epoch number."""
    ckpt = torch.load(path, map_location=agent.device)
    agent.policy.load_state_dict(ckpt["policy_state_dict"])
    optimiser.load_state_dict(ckpt["optimiser_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"  ✅ Resumed from {path}  (epoch {epoch})")
    return epoch


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the drone policy to hover (7 s episodes).",
    )
    # ── training ────────────────────────────────────────────────────────
    p.add_argument(
        "--epochs", type=int, default=200, help="Number of PPO training epochs."
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="Transitions per rollout (~3 episodes of 350 steps).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for PPO updates."
    )
    p.add_argument(
        "--ppo-epochs", type=int, default=8, help="Gradient passes per rollout."
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")

    # ── environment ─────────────────────────────────────────────────────
    p.add_argument(
        "--world",
        type=str,
        default="tugbot_depot",
        help="Gazebo world name (must match SDF in PX4 worlds dir).",
    )
    p.add_argument(
        "--hover-alt", type=float, default=2.5, help="Target hover altitude (m)."
    )
    p.add_argument(
        "--episode-secs", type=float, default=3.0, help="Episode length in sim-seconds."
    )
    p.add_argument(
        "--cam-size",
        type=int,
        default=128,
        help="Camera observation H=W (must be ≥ 128 for RAFT).",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Use a physics-free dummy env for loop testing.",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel PX4+Gazebo environments (each gets its own sim stack).",
    )
    p.add_argument(
        "--launch-script",
        type=str,
        default="",
        help="Path to launch_sim.sh for auto-launching per-worker sim stacks. "
        "Leave empty if sims are already running.",
    )

    # ── checkpointing & logging ─────────────────────────────────────────
    p.add_argument(
        "--run-dir",
        type=str,
        default="runs/hover",
        help="Directory for checkpoints & wandb logs.",
    )
    p.add_argument(
        "--ckpt-every", type=int, default=10, help="Save checkpoint every N epochs."
    )
    p.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from."
    )
    p.add_argument(
        "--wandb-project", type=str, default="drone-hover", help="Wandb project name."
    )

    # ── hardware ────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda", help="'cpu' or 'cuda'.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    # ── Weights & Biases (offline) ──────────────────────────────────────
    import wandb

    os.environ["WANDB_MODE"] = "online"
    wandb.init(
        project=args.wandb_project,
        dir=args.run_dir,
        config=vars(args),
        name=f"hover_{time.strftime('%Y%m%d_%H%M%S')}",
        save_code=True,
    )

    # ── Model config ────────────────────────────────────────────────────
    cfg = ModelConfig(
        cam_height=args.cam_size,
        cam_width=args.cam_size,
        lr=args.lr,
    )
    wandb.config.update(
        {k: v for k, v in cfg.__dict__.items()},
        allow_val_change=True,
    )

    # ── Agent + optimiser ───────────────────────────────────────────────
    agent = DroneAgent(cfg=cfg, device=args.device)
    optimiser = torch.optim.Adam(
        agent.policy.parameters(),
        lr=cfg.lr,
    )

    # Mixed-precision GradScaler for Blackwell / SM 12.0 GPUs where fp32
    # cuBLAS is broken.  On CPU this is simply None.
    scaler: torch.amp.GradScaler | None = None  # type: ignore[attr-defined]
    if args.device.startswith("cuda"):
        scaler = torch.amp.GradScaler()  # type: ignore[attr-defined]

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(agent, optimiser, args.resume)

    # ── Rollout buffer ──────────────────────────────────────────────────
    wp_dim = cfg.waypoint_dim * cfg.num_waypoints
    buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        imu_dim=cfg.imu_dim,
        wp_dim=wp_dim,
        vis_feat_dim=cfg.flow_feature_dim,
        action_dim=cfg.action_dim,
        device=args.device,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    # ── Environment ─────────────────────────────────────────────────────
    env = make_env(args)
    num_envs = getattr(args, "num_envs", 1)
    use_vec_env = num_envs > 1 and not args.dummy

    # ── Parameter summary ───────────────────────────────────────────────
    total_p = sum(p.numel() for p in agent.policy.parameters())
    train_p = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Hover Training — Dual-Rate Policy          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Device:       {args.device:<38s} ║")
    print(f"║  Total params: {total_p:<38,d} ║")
    print(f"║  Trainable:    {train_p:<38,d} ║")
    print(f"║  Frozen (RAFT):{total_p - train_p:<38,d} ║")
    print(f"║  Epochs:       {args.epochs:<38d} ║")
    print(f"║  Rollout steps:{args.rollout_steps:<38d} ║")
    print(f"║  Num envs:     {num_envs:<38d} ║")
    print(f"║  Episode secs: {args.episode_secs:<38.1f} ║")
    print(f"║  Hover alt:    {args.hover_alt:<38.1f} ║")
    print(f"║  Dummy env:    {str(args.dummy):<38s} ║")
    print(f"║  Run dir:      {args.run_dir:<38s} ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Training loop ───────────────────────────────────────────────────
    best_mean_reward = -float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        # 1. Collect rollout
        if use_vec_env:
            rollout_stats = collect_rollout_vec(
                agent,
                env,  # type: ignore[arg-type]
                buffer,
                args.rollout_steps,
                cfg,
                hover_alt=args.hover_alt,
            )
        else:
            rollout_stats = collect_rollout(
                agent,
                env,  # type: ignore[arg-type]
                buffer,
                args.rollout_steps,
                cfg,
            )

        # 2. PPO update
        update_stats = ppo_update(
            agent,
            buffer,
            optimiser,
            cfg,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            scaler=scaler,
        )

        elapsed = time.time() - t0
        mean_rew = rollout_stats["mean_reward"]

        # 3. Logging
        log_dict = {
            "epoch": epoch,
            "rollout/total_reward": rollout_stats["total_reward"],
            "rollout/mean_reward": mean_rew,
            "rollout/episodes": rollout_stats["episodes"],
            "train/policy_loss": update_stats["policy_loss"],
            "train/value_loss": update_stats["value_loss"],
            "train/entropy": update_stats["entropy"],
            "train/approx_kl": update_stats["approx_kl"],
            "perf/epoch_time_s": elapsed,
            "perf/fps": args.rollout_steps / elapsed,
        }
        wandb.log(log_dict, step=epoch)

        # Console
        print(
            f"[{epoch:4d}]  "
            f"reward={mean_rew:+7.2f}  "
            f"π_loss={update_stats['policy_loss']:.4f}  "
            f"v_loss={update_stats['value_loss']:.4f}  "
            f"ent={update_stats['entropy']:.3f}  "
            f"kl={update_stats['approx_kl']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # 4. Checkpointing
        is_best = mean_rew > best_mean_reward
        if is_best:
            best_mean_reward = mean_rew

        if (epoch + 1) % args.ckpt_every == 0 or is_best:
            ckpt_path = os.path.join(args.run_dir, f"ckpt_{epoch:04d}.pt")
            save_checkpoint(agent, optimiser, epoch, log_dict, ckpt_path)

        if is_best:
            best_path = os.path.join(args.run_dir, "best.pt")
            save_checkpoint(agent, optimiser, epoch, log_dict, best_path)

    # ── Final save ──────────────────────────────────────────────────────
    final_path = os.path.join(args.run_dir, "final.pt")
    save_checkpoint(agent, optimiser, start_epoch + args.epochs - 1, {}, final_path)

    # ── Cleanup ──────────────────────────────────────────────────────────
    if use_vec_env and hasattr(env, "close"):
        env.close()  # type: ignore[union-attr]

    wandb.finish()
    print("\n✅ Training complete.")
    print(f"   Best mean reward: {best_mean_reward:+.3f}")
    print(f"   Checkpoints in:   {args.run_dir}/")
    print(f"   Upload logs:      wandb sync {args.run_dir}/wandb/latest-run")


if __name__ == "__main__":
    main()
