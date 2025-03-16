import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        self.current_episode_reward += np.sum(rewards)
        if np.any(dones) and np.any(
            [info.get("TimeLimit.truncated", False) for info in infos]
        ):
            self.logger.record("train/episode_reward", self.current_episode_reward)
            self.current_episode_reward = 0

        return True
