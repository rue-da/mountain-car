"""
Tabular Q-learning with uniform state discretization.

Target variants: discrete_steps, discrete_fuel. Convergence on the vanilla
discrete_steps variant is hard without reward shaping — pair with
envs.EnergyShapingWrapper (set shape=True in make_env) for reliable training
inside the 200-step truncation.
"""

import os

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, n_bins=40, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, decay_steps=50_000, seed=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self.n_bins = n_bins
        self.n_actions = env.action_space.n
        self.step_count = 0
        self.rng = np.random.default_rng(seed)

        low = env.observation_space.low
        high = env.observation_space.high
        self.bin_edges = [
            np.linspace(low[i], high[i], n_bins + 1)[1:-1] for i in range(len(low))
        ]
        shape = tuple(n_bins for _ in low) + (self.n_actions,)
        self.q_table = self.rng.uniform(-1e-3, 1e-3, shape)

    def _discretize(self, state):
        return tuple(
            int(np.clip(np.digitize(state[i], self.bin_edges[i]), 0, self.n_bins - 1))
            for i in range(len(state))
        )

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[self._discretize(state)]))

    def learn(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        ns = self._discretize(next_state)
        target = reward + (0.0 if done else self.gamma * np.max(self.q_table[ns]))
        self.q_table[s + (action,)] += self.lr * (target - self.q_table[s + (action,)])

        self.step_count += 1
        frac = min(1.0, self.step_count / self.decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    def get_config(self):
        return {
            "n_bins": self.n_bins,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "decay_steps": self.decay_steps,
        }

    def get_metrics(self):
        return {"hyperparams/epsilon": float(self.epsilon)}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            q_table=self.q_table,
            bin_edges=np.array(self.bin_edges, dtype=object),
        )

    @classmethod
    def load(cls, path, env):
        data = np.load(path, allow_pickle=True)
        q = data["q_table"]
        agent = cls(env, n_bins=q.shape[0])
        agent.q_table = q
        agent.bin_edges = [np.asarray(e, dtype=np.float64) for e in data["bin_edges"]]
        agent.epsilon = 0.0
        return agent


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.evaluator import evaluate
    from core.trainer import run
    from envs import make_env

    train_env = make_env("discrete_steps", seed=0, shape=True)
    eval_env = make_env("discrete_steps", seed=100)
    agent = QLearningAgent(train_env, n_bins=40, lr=0.1, decay_steps=80_000, seed=0)

    run(train_env, agent, episodes=800, run_name="_smoke_qlearning")
    results = evaluate(eval_env, agent, n=20)
    print(f"[smoke] discrete_steps eval: mean_steps={results['true_obj_mean']:.1f} "
          f"success_rate={results['success_rate']:.2f}")
    assert results["true_obj_mean"] < 200, "agent never reached the flag"
