import gymnasium as gym
import numpy as np


class TrueObjectiveWrapper(gym.Wrapper):
    """
    Records the *true* objective cost separately from the (possibly shaped / reshaped) reward
    the agent trains on. Populates info on every step:
      - info["true_obj"]: per-step cost under the variant's real objective
      - info["true_obj_cum"]: running sum across the episode
      - info["success"]: True if the goal was reached this step

    Variant → per-step cost:
      discrete_steps      → 1              (minimize #steps)
      discrete_fuel       → 1 if action != 1 else 0  (minimize #non-null actions)
      continuous_steps    → 1              (minimize #steps)
      continuous_fuel     → action**2      (minimize integrated action^2)
    """

    VARIANTS = ("discrete_steps", "discrete_fuel", "continuous_steps", "continuous_fuel")

    def __init__(self, env, variant):
        super().__init__(env)
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {self.VARIANTS}")
        self.variant = variant
        self._cum = 0.0

    def reset(self, **kwargs):
        self._cum = 0.0
        return self.env.reset(**kwargs)

    def _cost(self, action):
        if self.variant in ("discrete_steps", "continuous_steps"):
            return 1.0
        if self.variant == "discrete_fuel":
            return 0.0 if int(action) == 1 else 1.0
        a = float(np.asarray(action).squeeze())
        return a * a

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = self._cost(action)
        self._cum += cost
        info["true_obj"] = cost
        info["true_obj_cum"] = self._cum
        info["success"] = bool(terminated)
        return obs, reward, terminated, truncated, info
