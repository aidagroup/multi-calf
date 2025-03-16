import os
import matplotlib
import numpy as np
import torch
import tyro

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.model import CustomCNN
from src.utils.mlflow import mlflow_monitoring, create_mlflow_logger, MlflowConfig
from dataclasses import dataclass, field
from pathlib import Path
import mlflow
from src import run_path
from typing import Literal


@dataclass
class ExperimentConfig:
    seed: int = 42
    """Random seed for reproducibility"""

    # Environment configuration
    env_id: Literal["VisualPendulumClassicReward", "VisualPendulumUpswingReward"] = (
        "VisualPendulumClassicReward"
    )
    """Environment ID to use"""
    n_envs: int = 8
    """Number of parallel environments to run"""
    single_thread: bool = False
    """Use DummyVecEnv for single-threaded environment"""
    normalize: bool = True
    """Enable reward normalization"""
    n_frames_stack: int = 4
    """Number of frames to stack"""

    # Training parameters
    total_timesteps: int = 300_000
    """Total number of timesteps for training"""
    learning_rate: float = 4e-4
    """Learning rate for the optimizer"""
    n_steps: int = 1000
    """Number of steps to run for each environment per update"""
    batch_size: int = 500
    """Minibatch size for training"""
    gamma: float = 0.99
    """Discount factor for rewards"""
    gae_lambda: float = 0.9
    """Factor for trade-off of bias vs variance in Generalized Advantage Estimation"""
    clip_range: float = 0.2
    """Clipping parameter for PPO"""
    use_sde: bool = False
    """Whether to use State Dependent Exploration"""
    sde_sample_freq: int = -1
    """Sample frequency for SDE"""
    save_model_every_steps: int = 1000
    """Save model checkpoint every N steps"""
    device: str = "cuda:0"
    """Device to use for training"""

    # Logging and artifacts
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            experiment_name=Path(__file__).stem,
        )
    )
    """MLflow configuration for experiment tracking"""
    local_artifacts_path: Path = run_path / "artifacts"
    """Path to store artifacts locally for not searching in mlflow"""


@mlflow_monitoring()
def main(config: ExperimentConfig):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    set_random_seed(config.seed)
    matplotlib.use("Agg")
    local_artifacts_path = (
        config.local_artifacts_path / f"ppo_{config.env_id}_{config.seed}"
    )

    print("Setting up environment...")

    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=DummyVecEnv if config.n_envs == 1 else SubprocVecEnv,
    )
    env = VecFrameStack(env, n_stack=config.n_frames_stack)
    env = VecTransposeImage(env)

    if config.normalize:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        print("Reward normalization enabled.")

    print("Environment setup complete.")

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=256, num_frames=config.n_frames_stack
            ),
        ),
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        seed=config.seed,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        verbose=1,
        device=config.device,
    )
    model.set_logger(create_mlflow_logger())

    mlflow_checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,
        save_path=os.path.join(
            mlflow.get_artifact_uri()[len("file://") :], "checkpoints"
        ),
        name_prefix=f"ppo",
    )

    local_checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,
        save_path=local_artifacts_path / "checkpoints",
        name_prefix="ppo",
    )

    callback = CallbackList(
        [
            local_checkpoint_callback,
            mlflow_checkpoint_callback,
        ]
    )

    print("Starting training...")

    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    print("Training completed.")

    if config.normalize:
        env.save(local_artifacts_path / "vecnormalize_stats.pkl")
        mlflow.log_artifact(
            local_artifacts_path / "vecnormalize_stats.pkl",
        )

    env.close()


if __name__ == "__main__":
    config = tyro.cli(ExperimentConfig)
    main(config)
