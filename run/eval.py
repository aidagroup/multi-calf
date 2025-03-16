import numpy as np
import torch
import tyro
import matplotlib
import imageio
import mlflow
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.model import CustomCNN
from src.utilities.mlflow import mlflow_monitoring, MlflowConfig
from src import run_path


@dataclass
class EvalConfig:
    seed: int = 42
    """Random seed for reproducibility"""

    # Environment configuration
    env_id: str = "VisualPendulumClassicReward"
    """Environment ID to use"""
    n_frames_stack: int = 4
    """Number of frames to stack"""

    # Checkpoint and model configuration
    checkpoint_path: Optional[Path] = None
    """Path to the model checkpoint to load"""
    device: str = "cuda:0"
    """Device to use for evaluation"""

    # Evaluation parameters
    n_envs: int = 5
    """Number of environments to evaluate"""
    deterministic: bool = True
    """Whether to use deterministic actions for evaluation"""
    n_steps: int = 200
    """Number of steps to evaluate"""

    # Rendering configuration
    render: bool = True
    """Whether to render the environment"""
    render_fps: int = 20
    """FPS for the output video"""
    output_path: Path = run_path / "artifacts" / "videos"
    """Path to save the output video"""

    # Logging and artifacts
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            experiment_name="pendulum_evaluation",
        )
    )
    """MLflow configuration for experiment tracking"""


@mlflow_monitoring()
def main(config: EvalConfig):
    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    set_random_seed(config.seed)
    matplotlib.use("Agg")

    config.output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from checkpoint: {config.checkpoint_path}")

    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=DummyVecEnv if config.n_envs == 1 else SubprocVecEnv,
    )

    env_for_rendering = env

    env = VecFrameStack(env, n_stack=config.n_frames_stack)
    env = VecTransposeImage(env)

    # Load normalization stats if provided
    vecnormalize_path = config.checkpoint_path.parent.parent / "vecnormalize.pkl"
    if vecnormalize_path.exists():
        print(f"Loading normalization stats from: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(
        config.checkpoint_path,
        env=env,
        custom_objects={
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": dict(
                features_dim=256, num_frames=config.n_frames_stack
            ),
        },
        device=config.device,
    )

    print(f"Model loaded successfully. Starting evaluation for {config.n_steps} steps.")

    if config.render:
        frames = {i: [] for i in range(config.n_envs)}
    episode_rewards = {i: [] for i in range(config.n_envs)}

    obs = env.reset()
    for _ in range(config.n_steps):
        action, _ = model.predict(obs, deterministic=config.deterministic)

        obs, reward, done, info = env.step(action)
        for i in range(config.n_envs):
            episode_rewards[i].append(reward[i])

        if config.render:
            rendered_frames = env_for_rendering.get_images()
            for i in range(config.n_envs):
                if rendered_frames[i] is not None:
                    frames[i].append(rendered_frames[i])

    # Save video
    if config.render:
        video_path = config.output_path / (
            config.env_id
            + "_"
            + config.checkpoint_path.stem.split("_")[1]
            + "_"
            + str(config.seed)
        )
        video_path.mkdir(parents=True, exist_ok=True)

        for i in range(config.n_envs):
            if len(frames[i]) > 0:
                output_file = video_path / f"env_{i:03d}.mp4"
                print(f"Saving video to {output_file} with {len(frames[i])} frames")
                print(f"Frame shape: {frames[i][0].shape}")

                imageio.mimsave(
                    output_file,
                    frames[i],
                    fps=config.render_fps,
                )
                mlflow.log_artifact(output_file)
            else:
                print(f"No frames captured for environment {i}")

    for i in range(config.n_envs):
        print(f"Episode {i} reward: {sum(episode_rewards[i])}")
        mlflow.log_metric(f"episode_reward", sum(episode_rewards[i]), step=i)


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
