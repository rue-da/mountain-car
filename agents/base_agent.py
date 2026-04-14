from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state):
        ...

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        ...
