import os
import numpy as np
from stable_baselines3 import SAC as SB3SAC
from stable_baselines3.common.logger import configure

from .base_agent import BaseAgent


class SACAgent(BaseAgent):
    """SAC via SB3 — continuous action spaces only (continuous_steps / continuous_fuel)."""

    def __init__(self, env, lr=3e-4, gamma=0.99, buffer_size=100_000,
                 batch_size=256, warmup=1000, seed=0):
        self.batch_size = batch_size
        self.warmup = warmup
        self.step_count = 0

        self.model = SB3SAC(
            "MlpPolicy", env,
            learning_rate=lr,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=warmup,
            seed=seed,
            verbose=0,
        )
        self.model.set_logger(configure(folder=None, format_strings=[]))

    def choose_action(self, state):
        action, _ = self.model.predict(state, deterministic=False)
        return action

    def learn(self, state, action, reward, next_state, done):
        self.model.replay_buffer.add(
            np.array([state]), np.array([next_state]),
            np.array([action]), np.array([reward]),
            np.array([done]), [{}],
        )
        self.model.num_timesteps += 1
        self.step_count += 1
        if self.step_count >= self.warmup:
            self.model.train(gradient_steps=1, batch_size=self.batch_size)

    def get_config(self):
        return {
            "lr": self.model.learning_rate,
            "gamma": self.model.gamma,
            "buffer_size": self.model.buffer_size,
            "batch_size": self.batch_size,
            "warmup": self.warmup,
        }

    def get_metrics(self):
        out = {}
        try:
            nv = self.model.logger.name_to_value
            if "train/critic_loss" in nv:
                out["losses/q_loss"] = float(nv["train/critic_loss"])
        except AttributeError:
            pass
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path, env):
        agent = cls.__new__(cls)
        agent.model = SB3SAC.load(path, env=env)
        agent.step_count = 0
        agent.batch_size = agent.model.batch_size
        agent.warmup = agent.model.learning_starts
        return agent
