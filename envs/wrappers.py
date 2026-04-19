import gymnasium as gym
import numpy as np


class DiscreteFuelRewardWrapper(gym.Wrapper):
    """Discrete min-fuel: reward = -fuel_cost per non-null action, 0 for no-op."""

    def __init__(self, env, fuel_cost=1.0, goal_bonus=0.0):
        super().__init__(env)
        self.fuel_cost = fuel_cost
        self.goal_bonus = goal_bonus

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = 0.0 if action == 1 else -self.fuel_cost
        if terminated:
            reward += self.goal_bonus
        return obs, reward, terminated, truncated, info


class ContinuousStepsRewardWrapper(gym.Wrapper):
    """Continuous min-steps: reward = -step_cost - action_cost * |a|, +goal_bonus on success."""

    def __init__(self, env, step_cost=1.0, action_cost=0.1, goal_bonus=100.0):
        super().__init__(env)
        self.step_cost = step_cost
        self.action_cost = action_cost
        self.goal_bonus = goal_bonus

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        a = float(np.asarray(action).squeeze())
        reward = -self.step_cost - self.action_cost * abs(a)
        if terminated:
            reward += self.goal_bonus
        return obs, reward, terminated, truncated, info


MinFuelWrapper = DiscreteFuelRewardWrapper
