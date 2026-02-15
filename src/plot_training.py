#!/usr/bin/env python3
"""
Plot training metrics from TensorBoard logs.

Usage:
    python3 plot_training.py --logdir logs --output plots
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(logdir: str) -> dict:
    """Load scalars from TensorBoard event files."""
    ea = EventAccumulator(logdir)
    ea.Reload()

    data = {}
    for key in ea.Tags()["scalars"]:
        events = ea.Scalars(key)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[key] = (steps, values)

    return data


def extract_metrics(data: dict) -> dict:
    """Extract and organize metrics by category."""
    metrics = {
        "curriculum": {},
        "loss": {},
        "reward": {},
        "waypoints": {},
    }

    for key, (steps, values) in data.items():
        if "curriculum" in key:
            metrics["curriculum"][key] = (steps, values)
        elif "loss" in key:
            metrics["loss"][key] = (steps, values)
        elif "reward" in key or "episode" in key:
            metrics["reward"][key] = (steps, values)
        elif "waypoint" in key:
            metrics["waypoints"][key] = (steps, values)

    return metrics


def plot_training(metrics: dict, output_dir: str = "plots") -> None:
    """Generate training plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Success Rate & Curriculum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if "curriculum/success_rate" in metrics["curriculum"]:
        steps, values = metrics["curriculum"]["curriculum/success_rate"]
        ax1.plot(steps, values * 100, "b-o", linewidth=2, markersize=4)
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Success Rate Over Training")
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 105])

    if "curriculum/stage" in metrics["curriculum"]:
        steps, values = metrics["curriculum"]["curriculum/stage"]
        ax2.plot(steps, values, "g-s", linewidth=2, markersize=4)
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Curriculum Stage")
        ax2.set_title("Curriculum Progression")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_success_and_curriculum.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/01_success_and_curriculum.png")
    plt.close()

    # Figure 2: Rewards
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {"train/total_reward": "b", "train/avg_reward": "r"}
    for key, color in colors.items():
        if key in metrics["reward"]:
            steps, values = metrics["reward"][key]
            ax.plot(
                steps,
                values,
                color=color,
                linewidth=2,
                label=key,
                marker="o",
                markersize=3,
            )

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_rewards.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/02_rewards.png")
    plt.close()

    # Figure 3: Losses
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Policy loss
    if "loss/policy" in metrics["loss"]:
        steps, values = metrics["loss"]["loss/policy"]
        ax1.plot(steps, values, "b-o", linewidth=2, markersize=3)
        ax1.set_ylabel("Policy Loss")
        ax1.set_title("Policy (Actor) Loss")
        ax1.grid(alpha=0.3)

    # Value loss
    if "loss/value" in metrics["loss"]:
        steps, values = metrics["loss"]["loss/value"]
        ax2.plot(steps, values, "r-o", linewidth=2, markersize=3)
        ax2.set_ylabel("Value Loss")
        ax2.set_title("Value (Critic) Loss")
        ax2.grid(alpha=0.3)

    # Entropy
    if "loss/entropy" in metrics["loss"]:
        steps, values = metrics["loss"]["loss/entropy"]
        ax3.plot(steps, values, "g-o", linewidth=2, markersize=3)
        ax3.set_ylabel("Entropy")
        ax3.set_xlabel("Global Step")
        ax3.set_title("Policy Entropy")
        ax3.grid(alpha=0.3)

    # KL divergence
    if "loss/approx_kl" in metrics["loss"]:
        steps, values = metrics["loss"]["loss/approx_kl"]
        ax4.plot(steps, values, "purple", linewidth=2, marker="o", markersize=3)
        ax4.set_ylabel("Approx KL")
        ax4.set_xlabel("Global Step")
        ax4.set_title("KL Divergence (Policy Update)")
        ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_losses.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/03_losses.png")
    plt.close()

    # Figure 4: Waypoints & Episodes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if "waypoints/reached" in metrics["waypoints"]:
        steps, values = metrics["waypoints"]["waypoints/reached"]
        ax1.plot(steps, values, "b-o", linewidth=2, markersize=4)
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Waypoints Reached")
        ax1.set_title("Waypoints Reached Per Epoch")
        ax1.grid(alpha=0.3)

    if "train/episodes" in metrics["reward"]:
        steps, values = metrics["reward"]["train/episodes"]
        ax2.plot(steps, values, "orange", linewidth=2, marker="s", markersize=4)
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Episodes Completed")
        ax2.set_title("Episodes Per Epoch")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_waypoints_and_episodes.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/04_waypoints_and_episodes.png")
    plt.close()

    # Figure 5: Reward vs Punishment Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if "train/avg_reward" in metrics["reward"]:
        steps, values = metrics["reward"]["train/avg_reward"]

        # Separate positive and negative rewards
        positive_rewards = np.where(values > 0, values, 0)
        negative_rewards = np.where(values < 0, values, 0)

        # Plot over time
        ax1.plot(
            steps,
            positive_rewards,
            "g-",
            linewidth=2,
            label="Positive (Reward)",
            alpha=0.7,
        )
        ax1.plot(
            steps,
            negative_rewards,
            "r-",
            linewidth=2,
            label="Negative (Punishment)",
            alpha=0.7,
        )
        ax1.fill_between(steps, 0, positive_rewards, color="green", alpha=0.2)
        ax1.fill_between(steps, 0, negative_rewards, color="red", alpha=0.2)
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Reward Value")
        ax1.set_title("Rewards vs Punishments Over Training")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Histogram/distribution
        ax2.hist(values, bins=50, color="blue", alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax2.axvline(
            x=np.mean(values),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(values):.2f}",
        )
        ax2.set_xlabel("Reward Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Reward Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Add statistics text
        pos_pct = (values > 0).sum() / len(values) * 100
        neg_pct = (values < 0).sum() / len(values) * 100
        stats_text = f"Positive: {pos_pct:.1f}%\nNegative: {neg_pct:.1f}%\nMean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}"
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_reward_vs_punishment.png"), dpi=150)
    print(f"✓ Saved: {output_dir}/05_reward_vs_punishment.png")
    plt.close()

    print(f"\n✓ All plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from TensorBoard logs"
    )
    parser.add_argument("--logdir", default="logs", help="TensorBoard log directory")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        print(f"❌ Log directory not found: {args.logdir}")
        return

    print(f"📊 Loading TensorBoard data from: {args.logdir}")
    data = load_tensorboard_data(args.logdir)

    if not data:
        print("❌ No data found in TensorBoard logs")
        return

    print(f"✓ Loaded {len(data)} metrics")
    metrics = extract_metrics(data)
    plot_training(metrics, args.output)


if __name__ == "__main__":
    main()
