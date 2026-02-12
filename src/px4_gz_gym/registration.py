"""
Auto-registration of PX4-Gazebo Gymnasium environments.

Called by the ``gymnasium.envs`` entry-point so that
``gymnasium.make("PX4Gz-v0")`` just works after ``pip install -e .``.
"""

import gymnasium as gym


def register_envs() -> None:
    # 50 Hz control  (5 × 0.004 s = 0.02 s per env step)
    # x500_mono_cam with roll/pitch/thrust actions
    gym.register(
        id="PX4Gz-v0",
        entry_point="px4_gz_gym.env:PX4GazeboEnv",
        max_episode_steps=2_000,
        kwargs={
            "world_name": "default",
            "model_name": "x500_mono_cam_0",
            "base_model": "x500_mono_cam",
            "n_gz_steps": 5,   # 5 × 0.004 s = 0.02 s → 50 Hz
            "step_size": 0.004,
            "takeoff_alt": 2.5,
        },
    )

    # 10 Hz control variant  (25 × 0.004 s = 0.1 s per env step)
    gym.register(
        id="PX4Gz-10Hz-v0",
        entry_point="px4_gz_gym.env:PX4GazeboEnv",
        max_episode_steps=1_000,
        kwargs={
            "world_name": "default",
            "model_name": "x500_mono_cam_0",
            "base_model": "x500_mono_cam",
            "n_gz_steps": 25,  # 25 × 0.004 s = 0.1 s → 10 Hz
            "step_size": 0.004,
            "takeoff_alt": 2.5,
        },
    )


# Also run at import time so `import px4_gz_gym` is enough.
register_envs()
