import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple

import mlflow
import tyro
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from src.utils.mlflow import MlflowConfig, mlflow_monitoring, create_mlflow_logger

current_dir = Path(__file__).parent


@dataclass
class ExperimentConfig:
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
            experiment_name=current_dir.name,
        )
    )
    local_artifacts_path: Path = current_dir / "artifacts"
    env_id: str = "Pendulum-v1"
    n_envs: int = 1
    gamma: float = 0.99
    learning_rate: float = 0.0003
    buffer_size: int = 1_000_000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    verbose: int = 1
    seed: int = 42
    total_timesteps: int = 1_000_000
    save_model_every_steps: int = 2048
    device: str = "cuda:1"


@mlflow_monitoring()
def main(config: ExperimentConfig):
    # Create the environment
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)
    local_artifacts_path = (
        config.local_artifacts_path / f"td3_{config.env_id}_{config.seed}"
    )
    # Instantiate the agent
    model = TD3(
        "MlpPolicy",
        env,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        policy_delay=config.policy_delay,
        target_policy_noise=config.target_policy_noise,
        target_noise_clip=config.target_noise_clip,
        verbose=config.verbose,
        seed=config.seed,
    )

    model.set_logger(create_mlflow_logger())

    print("Model initialized successfully.")
    # Set up a checkpoint callback to save the model every 'save_freq' steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=local_artifacts_path / "checkpoints",  # Directory to save the model
        name_prefix=f"td3_checkpoint",
    )

    mlflow_checkpoint_callback = CheckpointCallback(
        save_freq=config.save_model_every_steps,  # Save the model periodically
        save_path=os.path.join(
            mlflow.get_artifact_uri()[len("file://") :], "checkpoints"
        ),  # Directory to save the model
        name_prefix=f"td3_{config.env_id}",
    )

    # Instantiate a plotting callback for the live learning curve
    path = (
        local_artifacts_path
        / "logs"
        / f"episode_rewards_td3_{config.env_id}_{config.seed}.csv"
    )
    os.makedirs(str(path.parent), exist_ok=True)

    # Combine both callbacks using CallbackList
    callback = CallbackList([checkpoint_callback, mlflow_checkpoint_callback])

    print("Starting training ...")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)


if __name__ == "__main__":
    config = tyro.cli(ExperimentConfig)
    main(config)
