import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, n_bins=20, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = env.action_space.n

        low = env.observation_space.low
        high = env.observation_space.high
        self.bins = [np.linspace(low[i], high[i], n_bins) for i in range(len(low))]
        self.q_table = np.zeros([n_bins + 1] * len(low) + [self.n_actions])

    def _discretize(self, state):
        return tuple(np.digitize(state[i], self.bins[i]) for i in range(len(state)))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[self._discretize(state)]))

    def learn(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        ns = self._discretize(next_state)
        target = reward + (0 if done else self.gamma * np.max(self.q_table[ns]))
        self.q_table[s + (action,)] += self.lr * (target - self.q_table[s + (action,)])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
