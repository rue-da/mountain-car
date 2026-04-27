import torch

from .dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN — online net selects action, target net evaluates it."""

    def _q_next(self, ns):
        best_actions = self.online(ns).argmax(dim=1, keepdim=True)
        return self.target(ns).gather(1, best_actions).squeeze(1)
