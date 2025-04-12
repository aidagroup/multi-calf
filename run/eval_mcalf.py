import numpy as np
import torch
import tyro
import imageio
import mlflow
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Literal
from loguru import logger
import cv2

# Set MuJoCo to use software rendering with OSMesa before importing any mujoco-related libraries
os.environ["MUJOCO_GL"] = "osmesa"

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.utils.mlflow import mlflow_monitoring, MlflowConfig
from src import run_path
from src.wrapper.multicalf import MultiCALFWrapper


@dataclass
class MultiCALFConfig:
    relaxprob_init: float = 1.0
    relaxprob_factor: float = 0.999
    calf_change_rate: float = 0.01


@dataclass
class EvalConfig:
    seed: int = 42
    """Random seed for reproducibility"""

    # Environment configuration
    env_id: str = "Hopper-v4"
    """Environment ID to use"""
    n_frames_stack: int = 4
    """Number of frames to stack (for image-based environments)"""

    # Model type
    model_type: Literal["PPO", "TD3"] = "PPO"
    """Type of the model to load"""

    # Checkpoint and model configuration
    base_checkpoint_path: Optional[Path] = None
    """Path to the base model checkpoint to load"""
    alt_checkpoint_path: Optional[Path] = None
    """Path to the alternative model checkpoint to load"""
    device: str = "cuda:0"
    """Device to use for evaluation"""

    # Environment properties
    is_visual_env: bool = False
    """Whether the environment is visual (uses images as observations)"""

    # Evaluation parameters
    n_envs: int = 5
    """Number of environments to evaluate"""
    deterministic: bool = True
    """Whether to use deterministic actions for evaluation"""
    n_steps: int = 1000
    """Number of steps to evaluate"""

    # Rendering configuration
    render: bool = True
    """Whether to render the environment"""
    render_fps: int = 125
    """FPS for the output video"""
    output_path: Path = run_path / "artifacts" / "videos" / "mcalf"
    """Path to save the output video"""

    mcalf: MultiCALFConfig = field(default_factory=lambda: MultiCALFConfig())
    """Configuration for MultiCALFWrapper"""

    # Logging and artifacts
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            experiment_name="mcalf_evaluation",
        )
    )
    """MLflow configuration for experiment tracking"""


@mlflow_monitoring()
def main(config: EvalConfig):
    if config.base_checkpoint_path is None:
        raise ValueError("base_checkpoint_path must be provided")
    if config.alt_checkpoint_path is None:
        raise ValueError("alt_checkpoint_path must be provided")

    logger.info(f"Setting up evaluation with seed {config.seed}")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    set_random_seed(config.seed)

    config.output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating environment: {config.env_id}")
    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=DummyVecEnv if config.n_envs == 1 else SubprocVecEnv,
    )

    env_for_rendering = env

    logger.info(f"Loading base model from: {config.base_checkpoint_path}")
    model_class = PPO if config.model_type == "PPO" else TD3
    custom_objects = {}

    if config.base_checkpoint_path.name.startswith("ppo"):
        base_model = PPO.load(
            config.base_checkpoint_path,
            env=env,
            device=config.device,
            seed=config.seed,
        )
    elif config.base_checkpoint_path.name.startswith("td3"):
        base_model = TD3.load(
            config.base_checkpoint_path,
            env=env,
            device=config.device,
            seed=config.seed,
        )
    logger.info(f"Loading alternative model from: {config.alt_checkpoint_path}")
    if config.alt_checkpoint_path.name.startswith("ppo"):
        alt_model = PPO.load(
            config.alt_checkpoint_path,
            env=env,
            device=config.device,
            seed=config.seed,
        )
    elif config.alt_checkpoint_path.name.startswith("td3"):
        alt_model = TD3.load(
            config.alt_checkpoint_path,
            env=env,
            device=config.device,
            seed=config.seed,
        )
    logger.info("Wrapping environment with MultiCALFWrapper")
    env = MultiCALFWrapper(
        env,
        model_base=base_model,
        model_alt=alt_model,
        calf_change_rate=config.mcalf.calf_change_rate,
        relaxprob_init=config.mcalf.relaxprob_init,
        relaxprob_factor=config.mcalf.relaxprob_factor,
        seed=config.seed,
    )

    if config.render:
        frames = {i: [] for i in range(config.n_envs)}
        distances = {i: 0 for i in range(config.n_envs)}
    episode_rewards = {i: [] for i in range(config.n_envs)}
    episode_lengths = {i: 0 for i in range(config.n_envs)}
    episode_counts = {i: 0 for i in range(config.n_envs)}

    # Metrics for CALF-specific information
    relaxprob_values = []
    base_action_applied_counts = {i: 0 for i in range(config.n_envs)}
    increase_happened_counts = {i: 0 for i in range(config.n_envs)}

    logger.info(f"Starting evaluation for {config.n_steps} steps")
    obs = env.reset()
    for step in range(config.n_steps):
        action, _ = base_model.predict(obs, deterministic=config.deterministic)

        obs, reward, done, info = env.step(action)

        if step % 100 == 0:
            logger.info(f"Step {step}/{config.n_steps}")

        # Record relaxprob value
        if isinstance(info, list):
            relaxprob_values.append(info[0]["calf.relaxprob"])
        else:
            relaxprob_values.append(info["calf.relaxprob"])

        # Process rewards and info
        for i in range(config.n_envs):
            episode_rewards[i].append(reward[i])
            episode_lengths[i] += 1

            info_i = info[i] if isinstance(info, list) else info
            base_action_applied_counts[i] += int(info_i["calf.base_action_applied"])
            increase_happened_counts[i] += int(info_i["calf.increase_happened"])

            if done[i]:
                episode_counts[i] += 1
                mlflow.log_metric(
                    f"episode_{i}_reward",
                    sum(episode_rewards[i][-episode_lengths[i] :]),
                    step=episode_counts[i],
                )
                mlflow.log_metric(
                    f"episode_{i}_length", episode_lengths[i], step=episode_counts[i]
                )
                episode_lengths[i] = 0

        if config.render:
            rendered_frames = env_for_rendering.get_images()
            for i in range(config.n_envs):
                if rendered_frames[i] is not None:
                    total_reward = sum(episode_rewards[i])
                    distances[i] += obs[i][5] * 0.002
                    # Add text with total reward to the frame
                    frame = rendered_frames[i].copy()
                    reward_text = f"Total Reward: {total_reward:.2f}"
                    distance_text = f"Distance: {distances[i]:.2f}"

                    # First line - text
                    cv2.putText(
                        frame,
                        reward_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        distance_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    frames[i].append(frame)

    # Save video
    if config.render:
        video_path = config.output_path / (
            config.env_id
            + "_"
            + config.base_checkpoint_path.stem.split("_")[1]
            + "_"
            + str(config.seed)
        )
        video_path.mkdir(parents=True, exist_ok=True)

        for i in range(config.n_envs):
            if len(frames[i]) > 0:
                output_file = video_path / f"env_{i:03d}.mp4"
                logger.info(
                    f"Saving video to {output_file} with {len(frames[i])} frames"
                )

                imageio.mimsave(
                    output_file,
                    frames[i],
                    fps=config.render_fps,
                )
                mlflow.log_artifact(output_file)
            else:
                logger.warning(f"No frames captured for environment {i}")

    # Log final metrics
    rewards = [sum(episode_rewards[i]) for i in range(config.n_envs)]
    for i in range(config.n_envs):
        logger.info(f"Environment {i} metrics:")
        logger.info(f"  Total reward: {rewards[i]}")
        logger.info(
            f"  Base action applied: {base_action_applied_counts[i]}/{config.n_steps} ({base_action_applied_counts[i]/config.n_steps:.2%})"
        )
        logger.info(
            f"  Value increases: {increase_happened_counts[i]}/{config.n_steps} ({increase_happened_counts[i]/config.n_steps:.2%})"
        )

        mlflow.log_metric(f"env_{i}_total_reward", rewards[i])
        mlflow.log_metric(
            f"env_{i}_base_action_ratio", base_action_applied_counts[i] / config.n_steps
        )
        mlflow.log_metric(
            f"env_{i}_increase_ratio", increase_happened_counts[i] / config.n_steps
        )

    logger.info(f"Mean reward: {np.mean(rewards)}")
    logger.info(f"Std reward: {np.std(rewards)}")
    mlflow.log_metric("mean_reward", np.mean(rewards))
    mlflow.log_metric("std_reward", np.std(rewards))

    # Log relaxprob progression
    mlflow.log_metric("final_relaxprob", relaxprob_values[-1])


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
