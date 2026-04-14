import gymnasium as gym
import numpy as np


class MinFuelWrapper(gym.Wrapper):
    """Penalizes non-zero actions to encourage fuel-efficient policies."""

    def __init__(self, env, fuel_penalty=-0.1):
        super().__init__(env)
        self.fuel_penalty = fuel_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if action != 1:  # 1 = no push
            reward += self.fuel_penalty
        return obs, reward, terminated, truncated, info
