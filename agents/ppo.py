import os
import numpy as np
import torch
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.logger import configure

from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """PPO via SB3 — works with both discrete and continuous variants."""

    def __init__(self, env, lr=3e-4, gamma=0.99, n_steps=2048,
                 batch_size=64, n_epochs=10, seed=0):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.step_count = 0
        self._episode_start = True
        self._last_value = None
        self._last_log_prob = None

        self.model = SB3PPO(
            "MlpPolicy", env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            seed=seed,
            verbose=0,
        )
        self.model.set_logger(configure(folder=None, format_strings=[]))

    def choose_action(self, state):
        obs_t = torch.FloatTensor([state]).to(self.model.device)
        with torch.no_grad():
            action_t, value_t, log_prob_t = self.model.policy.forward(obs_t)
        self._last_value = value_t
        self._last_log_prob = log_prob_t
        action = action_t.cpu().numpy()[0]
        return int(action) if hasattr(self.model.action_space, 'n') else action

    def learn(self, state, action, reward, next_state, done):
        self.model.rollout_buffer.add(
            obs=np.array([state]),
            action=np.array([action]),
            reward=np.array([reward]),
            episode_start=np.array([self._episode_start]),
            value=self._last_value,
            log_prob=self._last_log_prob,
        )
        self._episode_start = done
        self.model.num_timesteps += 1
        self.step_count += 1

        if self.model.rollout_buffer.full:
            with torch.no_grad():
                last_obs = torch.FloatTensor([next_state]).to(self.model.device)
                _, last_value, _ = self.model.policy.forward(last_obs)
            self.model.rollout_buffer.compute_returns_and_advantage(
                last_values=last_value, dones=np.array([float(done)])
            )
            self.model.train()
            self.model.rollout_buffer.reset()

    def get_config(self):
        return {
            "lr": self.model.learning_rate,
            "gamma": self.model.gamma,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
        }

    def get_metrics(self):
        out = {}
        try:
            nv = self.model.logger.name_to_value
            if "train/value_loss" in nv:
                out["losses/q_loss"] = float(nv["train/value_loss"])
        except AttributeError:
            pass
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path, env):
        agent = cls.__new__(cls)
        agent.model = SB3PPO.load(path, env=env)
        agent.step_count = 0
        agent._episode_start = True
        agent._last_value = None
        agent._last_log_prob = None
        agent.batch_size = agent.model.batch_size
        agent.n_steps = agent.model.n_steps
        return agent
