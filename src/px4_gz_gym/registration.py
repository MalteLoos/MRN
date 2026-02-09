"""
Auto-registration of PX4-Gazebo Gymnasium environments.

Called by the ``gymnasium.envs`` entry-point so that
``gymnasium.make("PX4Gz-v0")`` just works after ``pip install -e .``.
"""

import gymnasium as gym


def register_envs() -> None:
    gym.register(
        id="PX4Gz-v0",
        entry_point="px4_gz_gym.env:PX4GazeboEnv",
        max_episode_steps=1_000,
        kwargs={
            "world_name": "default",
            "model_name": "x500_0",
            "n_gz_steps": 25,  # 25 × 0.004 s = 0.1 s per env step
            "step_size": 0.004,
            "action_dim": 4,
        },
    )

    gym.register(
        id="PX4Gz-Fast-v0",
        entry_point="px4_gz_gym.env:PX4GazeboEnv",
        max_episode_steps=500,
        kwargs={
            "world_name": "default",
            "model_name": "x500_0",
            "n_gz_steps": 50,  # 50 × 0.004 s = 0.2 s per env step
            "step_size": 0.004,
            "action_dim": 4,
        },
    )


# Also run at import time so `import px4_gz_gym` is enough.
register_envs()
