"""
Deep Q-Network for the discrete MountainCar variants.

Plain DQN (not double / dueling) — the assignment asks for a tabular + a deep
baseline for the discrete action variants, not a novel algorithm. Trained
step-by-step through the shared trainer (no SB3 dependency in this file).
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class _QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, env, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 decay_steps=50_000, buffer_size=50_000, batch_size=64,
                 target_update=500, warmup=1000, hidden=64, seed=0, device=None):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.warmup = warmup
        self.step_count = 0

        self.n_actions = env.action_space.n
        obs_dim = int(np.prod(env.observation_space.shape))

        self.device = torch.device(device or _default_device())
        torch.manual_seed(seed)
        self.online = _QNet(obs_dim, self.n_actions, hidden).to(self.device)
        self.target = _QNet(obs_dim, self.n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=lr)

        self.buffer = deque(maxlen=buffer_size)
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                device=self.device).unsqueeze(0)
            return int(self.online(s).argmax(dim=1).item())

    def learn(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))
        self.step_count += 1

        frac = min(1.0, self.step_count / self.decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

        if len(self.buffer) < max(self.warmup, self.batch_size):
            return

        batch = self._py_rng.sample(self.buffer, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.as_tensor(np.stack(s), device=self.device)
        ns = torch.as_tensor(np.stack(ns), device=self.device)
        a = torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)
        r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=self.device)
        d = torch.as_tensor(np.asarray(d, dtype=np.float32), device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)

    @classmethod
    def load(cls, path, env, device=None):
        agent = cls(env, device=device)
        data = torch.load(path, map_location=agent.device, weights_only=True)
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["online"])
        agent.epsilon = 0.0
        agent.step_count = int(data.get("step_count", 0))
        return agent


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.evaluator import evaluate
    from core.trainer import run
    from envs import make_env

    train_env = make_env("discrete_steps", seed=0, shape=True)
    eval_env = make_env("discrete_steps", seed=100)
    agent = DQNAgent(train_env, decay_steps=60_000, seed=0)

    run(train_env, agent, episodes=600, run_name="_smoke_dqn")
    results = evaluate(eval_env, agent, n=20)
    steps = -np.array(results["rewards"])
    mean_steps = float(steps.mean())
    print(f"[smoke] discrete_steps eval: mean_steps={mean_steps:.1f} "
          f"success={(steps < 200).mean():.2f}")
    assert mean_steps < 200, "agent never reached the flag"
