#!/usr/bin/env python3
"""
test_stepping.py — Smoke-test for the deterministic Gazebo stepping.

Run the full simulation stack first (./launch_sim.sh), then in another
terminal:

    cd /workspace/src
    python3 -m px4_gz_gym.test_stepping          # default: 25 steps
    python3 -m px4_gz_gym.test_stepping --n 50    # 50 gz-steps per tick
    python3 -m px4_gz_gym.test_stepping --ticks 5 # only do 5 env-steps

The script pauses Gazebo, then calls ``step_and_wait(n)`` in a loop,
printing the sim-time and observations after each env-step.  This
verifies that:

  1. Sim-time advances by exactly ``n * step_size`` each tick.
  2. Sensor observations update between ticks.
  3. PX4 stays in lockstep (you can verify by watching the PX4 console).
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np  # noqa: E402

from px4_gz_gym.gz_step import GzStepController  # noqa: E402
from px4_gz_gym.sensors import GzSensors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test deterministic Gz stepping")
    parser.add_argument("--world", default="default", help="Gazebo world name")
    parser.add_argument("--model", default="x500_0", help="Model name (PX4 appends _0)")
    parser.add_argument("--n", type=int, default=25, help="Gz-steps per tick")
    parser.add_argument("--dt", type=float, default=0.004, help="Physics step size (s)")
    parser.add_argument(
        "--ticks", type=int, default=10, help="Number of env-steps to run"
    )
    args = parser.parse_args()

    print(f"Connecting to world '{args.world}', model '{args.model}' …")
    gz = GzStepController(world_name=args.world)
    sensors = GzSensors(world_name=args.world, model_name=args.model)

    # Allow subscriptions to connect
    time.sleep(1.0)

    print("Pausing world …")
    gz.pause()
    time.sleep(0.5)

    expected_dt = args.n * args.dt
    print(
        f"\nStepping {args.ticks} ticks × {args.n} gz-steps "
        f"(Δt = {expected_dt:.4f} s per tick)\n"
    )
    print(f"{'tick':>4}  {'sim_time':>10}  {'Δt':>8}  {'pos (ENU)':>30}  {'|vel|':>8}")
    print("-" * 75)

    prev_t = gz.sim_time
    for i in range(1, args.ticks + 1):
        wall_start = time.monotonic()
        new_t = gz.step_and_wait(n=args.n, step_size=args.dt)
        wall_elapsed = time.monotonic() - wall_start

        obs = sensors.get_obs()
        pos = obs[0:3]
        vel = obs[3:6]

        dt_actual = new_t - prev_t
        prev_t = new_t

        print(
            f"{i:4d}  {new_t:10.4f}  {dt_actual:8.4f}  "
            f"[{pos[0]:+8.3f} {pos[1]:+8.3f} {pos[2]:+8.3f}]  "
            f"{np.linalg.norm(vel):8.4f}  "
            f"(wall {wall_elapsed*1e3:.1f} ms)"
        )

    print("\nDone.  Unpausing world …")
    gz.unpause()


if __name__ == "__main__":
    main()
