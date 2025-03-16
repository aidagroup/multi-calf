"""Multi-CALF package."""

import gymnasium as gym
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from src.wrapper.common_wrapper import ResizeObservation
from gymnasium.envs.registration import WrapperSpec

repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
run_path = repo_root / "run"

gym.register(
    id="VisualPendulumNoArrowClassicReward",
    entry_point="src.envs.pendulum_visual:PendulumVisualNoArrowParallelizable",
    disable_env_checker=True,
    order_enforce=False,
    max_episode_steps=200,
    kwargs={
        "render_mode": "rgb_array",
        "costs_coefs": (1.0, 0.1, 0.001),
    },
    additional_wrappers=[
        WrapperSpec(
            name="ResizeObservation",
            entry_point="src.wrapper.common_wrapper:ResizeObservation",
            kwargs={"shape": (64, 64)},
        ),
    ],
)

gym.register(
    id="VisualPendulumNoArrowUpswingReward",
    entry_point="src.envs.pendulum_visual:PendulumVisualNoArrowParallelizable",
    disable_env_checker=True,
    order_enforce=False,
    max_episode_steps=200,
    kwargs={
        "render_mode": "rgb_array",
        "costs_coefs": (2.0, 0.01, 0.00001),
    },
    additional_wrappers=[
        WrapperSpec(
            name="ResizeObservation",
            entry_point="src.wrapper.common_wrapper:ResizeObservation",
            kwargs={"shape": (64, 64)},
        ),
    ],
)
