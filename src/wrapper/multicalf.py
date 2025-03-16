from gymnasium import Wrapper
import numpy as np
import torch
from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO, TD3
from typing import Optional, Union


class WrappedAgent(ABC):
    def __init__(self, model: Union[PPO, TD3]):
        self.model = model

    @abstractmethod
    def get_value(self, obs: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        pass


class PPOWrappedAgent(WrappedAgent):
    def __init__(self, model: PPO, deterministic: bool = True):
        super().__init__(model)
        self.deterministic = deterministic

    def get_value(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            if obs.ndim == 1:
                tensor_obs = torch.tensor(obs.reshape(1, -1), device=self.model.device)
            else:
                tensor_obs = torch.tensor(obs, device=self.model.device)

            values = self.model.policy.predict_values(tensor_obs).cpu().numpy()

            if obs.ndim == 1:
                return values[0][0]
            else:
                return values

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(obs, deterministic=self.deterministic)[0]


class TD3WrappedAgent(WrappedAgent):
    def __init__(self, model: TD3, deterministic: bool = True):
        super().__init__(model)
        self.deterministic = deterministic

    def get_action(self, obs: np.ndarray) -> float:
        return self.model.policy.predict(obs, deterministic=self.deterministic)[0]

    def get_value(self, obs: np.ndarray) -> float:
        action = self.model.policy.predict(obs, deterministic=True)[0]
        return self.model.critic(obs, action)


def wrap_model(model: Union[PPO, TD3], deterministic: bool = True) -> WrappedAgent:
    if isinstance(model, PPO):
        return PPOWrappedAgent(model, deterministic)
    elif isinstance(model, TD3):
        return TD3WrappedAgent(model, deterministic)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


class MultiCALFWrapper(Wrapper):
    def __init__(
        self,
        env: VecEnv,
        model_base: Union[PPO, TD3],
        model_alt: Union[PPO, TD3],
        calf_change_rate=0.01,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.model_base = wrap_model(model_base)
        self.model_alt = wrap_model(model_alt)
        self.calf_change_rate = calf_change_rate
        self.relaxprob_init = relaxprob_init
        self.relaxprob_factor = relaxprob_factor
        self.relaxprob = float(self.relaxprob_init)
        self.np_rng = np.random.default_rng(seed=seed)

    def step(self, base_action: np.ndarray):
        base_value = self.model_base.get_value(self.obs)
        value_increase = base_value - self.best_value - self.calf_change_rate
        self.best_value = np.where(value_increase >= 0, base_value, self.best_value)

        is_relaxprob_fired = (
            self.np_rng.random(size=value_increase.shape) < self.relaxprob
        )
        is_base_value_increase = value_increase >= 0
        is_base_action_applied = is_base_value_increase != is_relaxprob_fired

        action = np.where(
            is_base_action_applied,
            base_action,
            self.model_alt.get_action(self.obs),
        )
        env_step_output = list(self.env.step(action))
        next_obs, info = env_step_output[0], env_step_output[-1]
        self.obs = np.copy(next_obs)

        if isinstance(info, tuple):
            info = list(info)

        if isinstance(info, list):  # vectorized env
            for i in range(len(info)):
                info[i] |= {
                    "calf.relaxprob": np.copy(self.relaxprob),
                    "calf.increase_happened": (value_increase >= 0)[i, 0],
                    "calf.base_action_applied": is_base_action_applied[i, 0],
                    "calf.action": action[i, :],
                }
        else:  # single env
            info |= {
                "calf.relaxprob": np.copy(self.relaxprob),
                "calf.increase_happened": value_increase >= 0,
                "calf.base_action_applied": is_base_action_applied,
                "calf.action": action,
            }
        env_step_output[-1] = info

        self.relaxprob *= self.relaxprob_factor
        return tuple(env_step_output)

    def reset(self, *args, **kwargs):
        self.relaxprob = float(self.relaxprob_init)
        reset_output = self.env.reset(*args, **kwargs)
        if isinstance(reset_output, tuple):
            self.obs = reset_output[0]
        else:
            self.obs = reset_output

        self.best_value = self.model_base.get_value(self.obs)
        return np.copy(self.obs)
