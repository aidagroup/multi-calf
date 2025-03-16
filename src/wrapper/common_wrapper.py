import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

from collections import deque
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper
from gymnasium import Wrapper


class AddTruncatedFlagWrapper(Wrapper):
    def step(self, action):
        # Get the original step return values
        result = self.env.step(action)
        # Add `truncated` as False if only four values are returned
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
            return obs, reward, done, truncated, info
        return result


class LoggingWrapper(gym.Wrapper):
    def step(self, action):
        print(f"Action: {action}")

        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)  # Ensure reward is a scalar

        # Log observation, reward, and done flags
        print(
            f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        print(f"Resetting environment with args: {kwargs}")
        return self.env.reset(**kwargs)


class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        # Modify observation space to reflect normalization
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, observation):
        normalized_obs = observation / 255.0  # Example normalization logic
        return normalized_obs


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 3), dtype=np.uint8
        )

    def observation(self, observation):
        # Debug: Check if the observation is empty or not
        if observation is None or observation.size == 0:
            print("Error: Observation is empty or not properly generated.")
            raise ValueError("Observation is empty or not properly generated.")

        # Resize the observation using OpenCV
        resized_observation = cv2.resize(observation, (self.shape[1], self.shape[0]))

        return resized_observation


class ChannelStackWrapper(ObservationWrapper):
    def __init__(self, env, n_stack):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        # Multiply the channel dimension by n_stack
        obs_shape = list(self.observation_space.shape)
        obs_shape[0] *= n_stack
        self.observation_space = Box(
            low=0, high=255, shape=tuple(obs_shape), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Unpack tuple
        obs = self._preprocess_frame(obs)  # Transpose to (C, H, W)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        stacked_obs = np.concatenate(self.frames, axis=0)
        # print(f"DEBUG: Reset stacked observation shape: {stacked_obs.shape}")
        return stacked_obs, info

    def observation(self, observation):
        obs = self._preprocess_frame(observation)  # Transpose to (C, H, W)
        self.frames.append(obs)
        stacked_obs = np.concatenate(self.frames, axis=0)
        # print(f"DEBUG: Step stacked observation shape: {stacked_obs.shape}")
        return stacked_obs

    @staticmethod
    def _preprocess_frame(frame):
        """Ensure the frame is in (C, H, W) format."""
        if frame.shape[-1] == 3:  # If (H, W, C), transpose to (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
        return frame
