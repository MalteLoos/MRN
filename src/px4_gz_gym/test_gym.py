#!/usr/bin/env python3
"""
test_gym.py — Quick Gymnasium ``check_env`` + manual rollout.

Requires the simulation stack to be running (./launch_sim.sh).

    cd /workspace/src
    python3 -m px4_gz_gym.test_gym
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

# Ensure our envs are registered
import px4_gz_gym.registration  # noqa: E402, F401


def main() -> None:
    print("Creating PX4Gz-v0 …")
    env = gym.make(
        "PX4Gz-v0",
        n_gz_steps=25,  # 25 × 4 ms = 100 ms per env.step()
    )

    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")
    print(f"  dt per step:       {env.unwrapped.dt:.4f} s")
    print()

    # ── reset ───────────────────────────────────────────────
    print("Resetting …")
    obs, info = env.reset()
    print(f"  sim_time after reset: {info['sim_time']:.4f} s")
    print(f"  initial obs: {obs}")
    print()

    # ── rollout ─────────────────────────────────────────────
    n_steps = 20
    print(f"Running {n_steps} env-steps with random actions …\n")
    print(f"{'step':>4}  {'sim_time':>10}  {'reward':>8}  {'pos_z':>8}  {'done':>5}")
    print("-" * 50)

    total_reward = 0.0
    for i in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"{i:4d}  {info['sim_time']:10.4f}  {reward:+8.3f}  "
            f"{obs[2]:+8.3f}  {terminated or truncated}"
        )

        if terminated or truncated:
            print("  → Episode ended.")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
