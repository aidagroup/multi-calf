import pandas as pd
import os
import matplotlib
import numpy as np
import torch
import tyro
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from gymnasium.wrappers import TimeLimit

from src.model import CustomCNN

from src.envs.pendulum_visual import PendulumVisual
from src.envs.pendulum_visual import PendulumVisualNoArrowParallelizable

from src.wrapper.common_wrapper import ResizeObservation
from src.wrapper.common_wrapper import AddTruncatedFlagWrapper
from src.callback.episode_reward_callback import EpisodeRewardCallback

from src.utilities.mlflow import mlflow_monitoring, create_mlflow_logger, MlflowConfig
from dataclasses import dataclass, field
from pathlib import Path
import mlflow
import gymnasium as gym

current_dir = Path(__file__).parent

os.makedirs(current_dir / "artifacts" / "logs", exist_ok=True)


@dataclass
class ExperimentConfig:
    # Experiment setup
    seed: int = 42
    """Random seed for reproducibility"""

    # Execution mode
    notrain: bool = False
    """Skip training and only run evaluation"""
    console: bool = True
    """Disable graphical output for console-only mode"""
    log: bool = True
    """Enable logging and printing of simulation data"""
    verbose: int = 1
    """Verbosity level for logging"""

    # Environment configuration
    parallel_envs: int = 8
    """Number of parallel environments to run"""
    single_thread: bool = False
    """Use DummyVecEnv for single-threaded environment"""
    normalize: bool = True
    """Enable observation and reward normalization"""
    episode_timesteps: int = 1024
    """Maximum number of timesteps per episode"""

    # Visual settings
    image_height: int = 64
    """Height of the observation images"""
    image_width: int = 64
    """Width of the observation images"""

    # Training parameters
    total_timesteps: int = 131072
    """Total number of timesteps for training"""
    learning_rate: float = 4e-4
    """Learning rate for the optimizer"""
    n_steps: int = 1024
    """Number of steps to run for each environment per update"""
    batch_size: int = 512
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
    save_model_every_steps: int = 512
    """Save model checkpoint every N steps"""

    # Evaluation parameters
    loadstep: Optional[int] = None
    """Step to load checkpoint from"""
    eval_checkpoint: Optional[str] = None
    """Path to checkpoint for evaluation"""
    eval_name: Optional[str] = None
    """Experimental name for logging during evaluation"""

    # Logging and artifacts
    mlflow: MlflowConfig = field(
        default_factory=lambda: MlflowConfig(
            tracking_uri="file://" + os.path.join(str(current_dir), "mlruns"),
            experiment_name=current_dir.name,
        )
    )
    """MLflow configuration for experiment tracking"""
    local_artifacts_path: Path = current_dir / "artifacts"
    """Path to store artifacts locally"""


@mlflow_monitoring()
def main(config: ExperimentConfig):
    # Set random seeds for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Check if the --console flag is used
    if config.console:
        matplotlib.use("Agg")  # Use a non-GUI backend to disable graphical output
    else:
        matplotlib.use("TkAgg")
        # pass

    # Train the model if --notrain flag is not provided
    if not config.notrain:

        # Define a global variable for the training loop
        is_training = True

        # Function to create the base environment
        def make_env(seed):
            def _init():
                env = gym.make("VisualPendulumNoArrow-v0")
                # env = LoggingWrapper(env)  # For debugging: log each step. Comment out by default
                env = TimeLimit(env, max_episode_steps=config.episode_timesteps)
                env = ResizeObservation(env, (config.image_height, config.image_width))
                env.reset(seed=seed)
                return env

            return _init

        # Environment setup based on --single-thread flag
        if config.single_thread:
            print("Using single-threaded environment (DummyVecEnv).")
            env = DummyVecEnv([make_env(0)])
        else:
            print("Using multi-threaded environment (SubprocVecEnv).")
            env = SubprocVecEnv(
                [make_env(seed) for seed in range(config.parallel_envs)]
            )

        # Apply VecFrameStack to stack frames along the channel dimension
        env = VecFrameStack(env, n_stack=4)

        # Apply VecTransposeImage
        env = VecTransposeImage(env)

        # Apply reward and observation normalization if --normalize flag is provided
        if config.normalize:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
            print("Reward normalization enabled.")

        env.seed(seed=config.seed)
        obs = env.reset()
        print("Environment reset successfully.")

        # Set random seed for reproducibility
        set_random_seed(config.seed)

        # Define the policy_kwargs to use the custom CNN
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=256, num_frames=4
            ),  # Adjust num_frames as needed
        )

        # Create the PPO agent using the custom feature extractor
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
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
            device="cuda:1",
        )
        model.set_logger(create_mlflow_logger())

        mlflow_checkpoint_callback = CheckpointCallback(
            save_freq=config.save_model_every_steps,  # Save the model periodically
            save_path=os.path.join(
                mlflow.get_artifact_uri()[len("file://") :], "checkpoints"
            ),  # Directory to save the model
            name_prefix=f"ppo_vispendulum_{config.seed}",
        )

        mlflow_plotting_callback = EpisodeRewardCallback(
            save_path=config.local_artifacts_path
            / "logs"
            / f"episode_rewards_ppo_vispendulum_{config.seed}.csv",
        )

        print("Model initialized successfully.")

        # Set up a checkpoint callback to save the model every 'save_freq' steps
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_model_every_steps,  # Save the model periodically
            save_path=config.local_artifacts_path
            / "checkpoints"
            / f"ppo_vispendulum_{config.seed}",  # Directory to save the model
            name_prefix="ppo_vispendulum",
        )
        callback = CallbackList(
            [checkpoint_callback, mlflow_checkpoint_callback, mlflow_plotting_callback]
        )

        print("Starting training ...")

        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        print("Training completed.")
        model.save(
            config.local_artifacts_path
            / "checkpoints"
            / f"ppo_vispendulum_{config.seed}"
        )

        if config.normalize:
            env.save(
                config.local_artifacts_path / "checkpoints" / "vecnormalize_stats.pkl"
            )

        env.close()
        print("Training completed.")
    else:
        print("Skipping training. Loading the saved model...")

        if config.eval_checkpoint:
            model = PPO.load(config.eval_checkpoint)
        elif config.loadstep:
            model = PPO.load(
                config.local_artifacts_path
                / "checkpoints"
                / f"ppo_vispendulum_{config.loadstep}_steps"
            )
        else:
            model = PPO.load(
                config.local_artifacts_path
                / "checkpoints"
                / f"ppo_vispendulum_{config.seed}"
            )

        # Load the normalization statistics if --normalize is used
        if config.normalize:
            env = VecNormalize.load(
                config.local_artifacts_path / "checkpoints" / "vecnormalize_stats.pkl",
                env,
            )
            env.training = False  # Set to evaluation mode
            env.norm_reward = False  # Disable reward normalization for evaluation

    # Visual evaluation after training or loading
    print("Starting evaluation...")

    # Environment for the agent (using 'rgb_array' mode)
    env_agent = DummyVecEnv(
        [
            lambda: AddTruncatedFlagWrapper(
                ResizeObservation(
                    PendulumVisual(render_mode="rgb_array"),
                    (config.image_height, config.image_width),
                )
            )
        ]
    )
    env_agent = VecFrameStack(env_agent, n_stack=4)
    env_agent = VecTransposeImage(env_agent)

    # Environment for visualization (using 'human' mode)
    env_display = PendulumVisual(render_mode="rgb_array" if config.console else "human")

    # Reset the environments
    env_agent.seed(seed=config.seed)

    obs = env_agent.reset()
    env_display.reset(seed=config.seed)

    info_dict = {
        "state": [],
        "action": [],
        "reward": [],
        "accumulated_reward": [],
    }
    accumulated_reward = 0

    for _ in range(1000):
        action, _ = model.predict(obs)
        # action = env_agent.action_space.sample()  # Generate a random action

        # Dynamically handle four or five return values
        result = env_agent.step(action)  # Take a step in the environment

        # env_display.render()

        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        # Handle the display environment
        env_display.step(action)  # Step in the display environment to show animation

        if done:
            obs = env_agent.reset()  # Reset the agent's environment
            env_display.reset()  # Reset the display environment

        accumulated_reward += reward

        info_dict["state"].append(obs)
        info_dict["action"].append(action)
        info_dict["reward"].append(reward)
        info_dict["accumulated_reward"].append(accumulated_reward.copy())

    # Close the environments
    env_agent.close()
    env_display.close()

    df = pd.DataFrame(info_dict)
    if config.eval_name:
        file_name = f"ppo_vispendulum_eval_{config.eval_name}_seed_{config.seed}.csv"
    else:
        file_name = f"ppo_vispendulum_eval_{config.loadstep}_seed_{config.seed}.csv"

    df.to_csv(config.local_artifacts_path / "logs" / file_name)
    mlflow.log_artifact(config.local_artifacts_path / "logs" / file_name)
    print("Case:", file_name)
    print(df.drop(columns=["state"]).tail(2))


if __name__ == "__main__":
    config = tyro.cli(ExperimentConfig)
    main(config)
