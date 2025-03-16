import numpy as np
import torch

from gymnasium import Wrapper
from stable_baselines3.common.logger import configure
from copy import copy
from stable_baselines3 import PPO
from collections import deque
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


class CALFWrapper(Wrapper):
    """
    Description: Critic as a Lyapunov Function, realized as wrapper for a Gymnasium environment.
    """

    def __init__(
        self,
        env,
        base_model,
        alt_model,
        calf_change_rate: float = 0.01,
        init_relaxprob: float = 0.0,
        relaxprob_factor: float = 1.0,
        buffer_size: int = 150,  # Rolling window size
        lr=1e-3,  # Learning rate for online alternative model adaptation
        **kwargs,
    ):
        super().__init__(env)

        self.base_model = copy(base_model)
        self.alt_model = copy(alt_model)
        self.alt_model_original = copy(alt_model)

        self.calf_change_rate = calf_change_rate

        self.init_relaxprob = init_relaxprob
        self.relaxprob_factor = relaxprob_factor

        if "logger" not in kwargs or kwargs["logger"] is None:
            raise ValueError("Logger must be provided for CALFWrapper.")
        self.logger = kwargs["logger"]

        self.debug = kwargs.get("debug", False)

        # self.probability_to_take_base = 1.0
        self.relaxprob = init_relaxprob
        self.probability_increment = 0.005

        # Value change in CALF is checked every `value_change_every_n_steps`
        self.value_change_every_n_steps = 1

        self.last_updated_base_value = 0
        self.last_updated_alt_value = 0

        self.current_base_value = 0
        self.current_alt_value = 0

        self.relaxprob = init_relaxprob

        # Initialize rolling data variables
        self.buffer_size = buffer_size
        self.min_n_samples_to_update_models = int(buffer_size / 2)
        self.base_action_rewards = deque(maxlen=buffer_size)
        self.alt_action_rewards = deque(maxlen=buffer_size)
        self.alt_action_observations = deque(maxlen=buffer_size)
        self.alt_actions = deque(maxlen=buffer_size)

        self.step_count = 0

        self.base_action_count = 0
        self.base_action_percentage = 0

        self.base_value_increase_count = 0
        self.base_value_increase_percentage = 0

        self.alt_value_increase_count = 0
        self.alt_value_increase_percentage = 0

        self.base_value_increase_count_base_action = 0

        self.average_accum_reward_per_base_actions = 0
        self.average_accum_reward_per_alt_actions = 0

        self.last_action_mark = "base"

        self.model_update_discount = 0.99

        self.debug_every_n_steps = 50

        # Optimizer for the alternative model
        # self.optimizer = torch.optim.Adam(self.alt_model.policy.parameters(), lr=lr)

        # Re-define the learning rate
        for param_group in alt_model.policy.optimizer.param_groups:
            param_group["lr"] = lr

    def get_base_value(self, obs):
        with torch.no_grad():
            value_tensor = self.base_model.policy.predict_values(
                self.base_model.policy.obs_to_tensor(obs)[0]
            )
            return value_tensor.item()  # Convert single-element tensor to a scalar

    def get_alt_value(self, obs):
        with torch.no_grad():
            value_tensor = self.alt_model.policy.predict_values(
                self.alt_model.policy.obs_to_tensor(obs)[0]
            )
            return value_tensor.item()  # Convert single-element tensor to a scalar

    def update_relaxprob(self):
        self.relaxprob = self.relaxprob * self.relaxprob_factor

    def increase_relaxprob(self):
        self.relaxprob = np.min([1.0, self.relaxprob + self.probability_increment])

    def decrease_relaxprob(self):
        self.relaxprob = np.max([0.0, self.relaxprob - self.probability_increment])

    def update_alt_model(self):
        """
        Perform a one-step PPO-style gradient descent update for the alternative model.
        """
        if len(self.alt_action_observations) >= self.min_n_samples_to_update_models:
            # Convert inputs to PyTorch tensors
            observations_tensor = torch.tensor(
                np.array(self.alt_action_observations), dtype=torch.float32
            ).to(self.alt_model.device)
            actions_tensor = torch.tensor(
                np.array(self.alt_actions),
                dtype=torch.int64,  # Ensure actions are integers for policy evaluation
            ).to(self.alt_model.device)
            rewards_tensor = torch.tensor(
                list(self.alt_action_rewards), dtype=torch.float32
            ).to(self.alt_model.device)

            # Debug logs
            if self.debug and self.step_count % self.debug_every_n_steps == 0:
                for name, param in self.alt_model.policy.named_parameters():
                    print(f"Initial Norm of {name}: {param.data.norm().item()}")

            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards_tensor):
                cumulative_reward = (
                    reward + self.model_update_discount * cumulative_reward
                )
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.tensor(
                discounted_rewards, dtype=torch.float32
            ).to(self.alt_model.device)

            # Normalize discounted rewards
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-8
            )

            # Get policy log probabilities, entropy, and values
            output = self.alt_model.policy.evaluate_actions(
                observations_tensor, actions_tensor
            )

            # Unpack according to the output signature
            if len(output) == 2:
                log_probs, entropy = output
                values = None  # No values returned
            elif len(output) == 3:
                log_probs, entropy, values = output
            else:
                raise ValueError(
                    f"Unexpected number of outputs from evaluate_actions: {len(output)}"
                )

            # Compute advantages
            if values is not None:
                advantages = discounted_rewards - values.squeeze()
            else:
                advantages = (
                    discounted_rewards  # Treat rewards as advantages if no values
                )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute PPO clipping
            # Ratios between new and old policies
            old_log_probs = log_probs.detach()  # Log probabilities before update
            ratios = torch.exp(log_probs - old_log_probs)
            clip_range = 0.6  # Clipping range for PPO

            # Clipped objective
            clipped_ratios = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(
                ratios * advantages, clipped_ratios * advantages
            ).mean()

            # Compute value loss if values are available
            value_loss = (
                0.1 * ((values - discounted_rewards.clamp(-10, 10)) ** 2).mean()
                if values is not None
                else 0.0
            )

            # Entropy loss to encourage exploration
            entropy_loss = -0.1 * entropy.mean()

            # Total loss
            total_loss = policy_loss + value_loss + entropy_loss

            # Perform gradient descent step
            self.alt_model.policy.optimizer.zero_grad()
            total_loss.backward()
            self.alt_model.policy.optimizer.step()

            # Debug logs
            if self.debug and self.step_count % self.debug_every_n_steps == 0:
                print(f"---DEBUG--- Updated alt_model <------!")
                print(f"Policy Loss: {policy_loss.item():.4f}")
                print(f"Value Loss: {value_loss.item():.4f}")
                print(f"Entropy Loss: {entropy_loss.item():.4f}")
                print(
                    f"Advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}"
                )
                print(
                    f"Discounted Rewards: mean={discounted_rewards.mean().item():.4f}, std={discounted_rewards.std().item():.4f}"
                )
                print(
                    f"Ratios: mean={ratios.mean().item():.4f}, std={ratios.std().item():.4f}"
                )
                print(
                    f"Clipped Ratios: mean={clipped_ratios.mean().item():.4f}, std={clipped_ratios.std().item():.4f}"
                )

                raw_advantages = discounted_rewards - (
                    values.squeeze() if values is not None else 0
                )
                print(
                    f"Raw Advantages: mean={raw_advantages.mean().item():.4f}, std={raw_advantages.std().item():.4f}"
                )

                print(
                    f"Log Probs: mean={log_probs.mean().item():.4f}, std={log_probs.std().item():.4f}"
                )

                for name, param in self.alt_model.policy.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient Norm for {name}: {param.grad.norm().item()}")
                    else:
                        print(f"No Gradient for {name}")

    def update_rolling_data(self, action, obs, reward):
        """
        Update rolling buffers for alternative model actions, rewards, and observations.
        """

        # Ensure reward is a scalar
        if isinstance(reward, np.ndarray) or isinstance(reward, torch.Tensor):
            reward = float(
                reward.item()
            )  # Convert to scalar if it's a numpy array or tensor

        if self.last_action_mark == "alt":
            self.alt_actions.append(action)
            self.alt_action_observations.append(obs)
            self.alt_action_rewards.append(reward)

        if self.last_action_mark == "base":
            self.base_action_rewards.append(reward)
            self.base_action_count += 1

        # Calculate rolling averages
        self.average_accum_reward_per_base_actions = (
            np.mean([r for r in self.base_action_rewards])
            if len(self.base_action_rewards) > 0
            else 0.0
        )
        self.average_accum_reward_per_alt_actions = (
            np.mean([r for r in self.alt_action_rewards])
            if len(self.alt_action_rewards) > 0
            else 0.0
        )

    def debug_prints(self):

        self.base_value_increase_percentage = (
            self.base_value_increase_count
            * self.value_change_every_n_steps
            / self.step_count
            * 100
        )
        self.alt_value_increase_percentage = (
            self.alt_value_increase_count
            * self.value_change_every_n_steps
            / self.step_count
            * 100
        )

        self.base_action_percentage = self.base_action_count / self.step_count * 100

        # self.base_value_increase_percentage_base_action = self.base_value_increase_count_base_action * \
        #     self.value_change_every_n_steps / self.step_count * 100

        # print(f"---DEBUG--- Base value on base action increase percentage: {self.base_value_increase_percentage_base_action:.2f} %")
        print(
            f"---DEBUG--- Base action percentage: {self.base_action_percentage:.2f} %"
        )
        # print(f"---DEBUG--- Base value increase percentage: {self.base_value_increase_percentage:.2f} %")
        # print(f"---DEBUG--- Alt. value increase percentage: {self.alt_value_increase_percentage:.2f} %")
        print(
            f"---DEBUG--- Last updated base value: {self.last_updated_base_value:.3f}"
        )
        print(f"---DEBUG--- Last updated alt. value: {self.last_updated_alt_value:.3f}")
        print(f"---DEBUG--- Current base value: {self.current_base_value:.3f}")
        print(f"---DEBUG--- Current alt. value: {self.current_alt_value:.3f}")
        # print(f"---DEBUG--- Current relax probability: {self.relaxprob*100:.2f} %")
        # print(f"---DEBUG--- Current base model value estimate: {self.current_base_value:.2f}")

    #     print(f"---DEBUG--- Accumulated reward under alt. actions: {self.accum_reward_under_alt:.2f}")
    # print(f"---DEBUG--- Rolling average reward (base): {self.average_accum_reward_per_base_actions:.4f}")
    # print(f"---DEBUG--- Rolling average reward (alt): {self.average_accum_reward_per_alt_actions:.4f}")

    # Ensure there are enough samples to perform a meaningful update
    # if len(self.alt_action_rewards) < self.min_n_samples_to_update_models or len(self.alt_action_observations) < self.min_n_samples_to_update_models:
    #     if self.debug:
    #         print(f"---DEBUG--- Not enough samples to update alt_model: "
    #             f"{len(self.alt_action_rewards)} rewards, {len(self.alt_action_observations)} observations. "
    #             f"Required: {self.min_n_samples_to_update_models}.")

    # obs_tensor = self.alt_model.policy.obs_to_tensor(self.current_obs)[0]

    # # Get the action distribution from the alternative model
    # action_distribution = self.alt_model.policy.get_distribution(obs_tensor)

    # # Inspect available attributes and parameters
    # print("--- Action Distribution Attributes ---")
    # print(dir(action_distribution))  # List all attributes
    # print("--- Action Distribution Parameters ---")
    # for attr in dir(action_distribution):
    #     if not attr.startswith("_"):  # Exclude private attributes
    #         try:
    #             value = getattr(action_distribution, attr)
    #             print(f"{attr}: {value}")
    #         except Exception as e:
    #             print(f"{attr}: Could not retrieve ({e})")

    def get_alt_action(self, obs):

        # Convert observation to tensor
        obs_tensor = self.alt_model.policy.obs_to_tensor(obs)[0]

        # Get the action distribution from the alternative model
        action_distribution = self.alt_model.policy.get_distribution(obs_tensor)

        # Example: Scale the standard deviation of the distribution
        scale_factor = 0.5  # Adjust this value to control spread
        action_distribution.distribution.scale *= scale_factor

        # Sample a perturbed action from the distribution
        alt_action = action_distribution.sample()

        # Clamp the action if the distribution is squashed
        action_low = torch.tensor(
            self.env.action_space.low, device=alt_action.device, dtype=alt_action.dtype
        )
        action_high = torch.tensor(
            self.env.action_space.high, device=alt_action.device, dtype=alt_action.dtype
        )
        alt_action = alt_action.clamp(action_low, action_high)

        # Ensure the action is 1D
        alt_action = alt_action.squeeze(axis=1)  # Remove any extra dimensions

        # Convert to NumPy array and detach from the computation graph
        alt_action = alt_action.detach().cpu().numpy()

        return alt_action

    def step(self, base_action):

        # Update value estimates every `self.value_change_every_n_steps` steps
        if self.step_count % self.value_change_every_n_steps == 0:
            self.current_base_value = self.get_base_value(self.current_obs)
            self.current_alt_value = self.get_alt_value(self.current_obs)

        # Update step counter after value update check
        self.step_count += 1

        base_value_change = self.current_base_value - self.last_updated_base_value
        alt_value_change = self.current_alt_value - self.last_updated_alt_value

        is_base_value_increase = base_value_change >= self.calf_change_rate
        is_alt_value_increase = alt_value_change >= self.calf_change_rate

        if is_base_value_increase:
            self.last_updated_base_value = self.current_base_value
            self.base_value_increase_count += 1 if self.debug else 0

        if is_alt_value_increase:
            self.last_updated_alt_value = self.current_alt_value
            self.alt_value_increase_count += 1 if self.debug else 0

        alt_action, _ = self.alt_model.predict(self.current_obs, deterministic=True)

        # alt_action = self.get_alt_action(self.current_obs)

        # Action filter
        # if is_alt_value_increase and not is_base_value_increase:
        #     action = alt_action
        #     self.last_action_mark = "alt"

        # elif not is_alt_value_increase and is_base_value_increase:
        #     action = base_action
        #     self.last_action_mark = "base"

        # # elif is_alt_value_increase and is_base_value_increase:
        # else:
        # if self.last_updated_base_value < self.last_updated_alt_value:
        # if self.current_base_value > self.current_alt_value:
        if is_base_value_increase:

            if np.random.random() <= self.relaxprob:
                action = alt_action
                self.last_action_mark = "alt"
            else:
                action = base_action
                self.last_action_mark = "base"

        else:

            if np.random.random() <= self.relaxprob:
                action = base_action
                self.last_action_mark = "base"
            else:
                action = alt_action
                self.last_action_mark = "alt"

        # else:
        #     if np.random.random() <= self.relaxprob:
        #         action = alt_action
        #         self.last_action_mark = "alt"

        #     else:
        #         action = base_action
        #         self.last_action_mark = "base"

        # DEBUG
        # action = alt_action

        self.current_obs, reward, terminated, truncated, info = self.env.step(action)

        self.update_rolling_data(action, self.current_obs, reward)
        # self.update_alt_model()
        self.update_relaxprob()

        if self.debug and self.step_count % self.debug_every_n_steps == 0:
            self.debug_prints()
            # print(f"---DEBUG--- base action: {base_action}")
            # print(f"---DEBUG--- alt. action: {alt_action}")
            # print(f"---DEBUG--- action low: {self.env.action_space.low}")
            # print(f"---DEBUG--- action high: {self.env.action_space.high}")

        return self.current_obs.copy(), reward, terminated, truncated, info

    def reset(self, reset_alt_model=False, **kwargs):
        """
        Resets the environment and optionally resets the alternative model to its original state.

        Args:
            reset_alt_model (bool): If True, resets the alternative model to its original checkpoint state.
        """
        self.current_obs, info = self.env.reset(**kwargs)

        # Reset CALF values
        self.last_updated_base_value = self.get_base_value(self.current_obs)
        self.last_updated_alt_value = self.get_alt_value(self.current_obs)

        self.step_count = 0
        self.base_action_count = 0
        self.base_action_percentage = 0
        self.base_value_increase_count = 0
        self.base_value_increase_percentage = 0
        self.alt_value_increase_count = 0
        self.alt_value_increase_percentage = 0
        self.base_value_increase_count_base_action = 0
        self.average_accum_reward_per_base_actions = 0
        self.average_accum_reward_per_alt_actions = 0
        self.last_action_mark = "base"

        # Reset accumulators and rolling windows
        self.accum_reward_under_base = 0.0
        self.accum_reward_under_alt = 0.0
        self.base_action_rewards.clear()
        self.alt_action_rewards.clear()
        self.alt_action_observations.clear()
        self.alt_actions.clear()

        # Optionally reset the alternative model
        if reset_alt_model:
            if self.debug:
                print("---DEBUG--- Resetting alternative model to its original state.")

            self.alt_model = copy(self.alt_model_original)

        return self.current_obs.copy(), info
