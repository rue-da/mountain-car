from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state):
        ...

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        ...

    def get_metrics(self):
        """Return a flat dict of TB scalar paths -> values, e.g.
        {"hyperparams/epsilon": 0.1, "losses/q_loss": 0.42}.
        Implementations should reset any per-episode accumulators here."""
        return {}

    def get_config(self):
        """Return a dict of constructor hyperparameters for logging."""
        return {}
