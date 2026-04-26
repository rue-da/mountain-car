import gymnasium as gym


class EnergyShapingWrapper(gym.Wrapper):
    """
    Potential-based reward shaping (Ng et al. 1999). Adds
        bonus = scale * (gamma * phi(s') - phi(s))
    to the environment's reward. Being potential-based, the optimal policy is
    unchanged; only learning speed is affected.

    Default potential phi(s) = |velocity|. This is the standard MountainCar
    shaping cited in the literature — it gives the agent a dense signal for
    building kinetic energy, which is precisely what MountainCar requires
    (the under-powered car must gain speed by rocking up the opposite hill).
    A position-based phi would perversely reward pushing right even when
    gravity stalls the car; a velocity-based phi does not.

    Scale is chosen so per-step bonuses are O(1), competitive with the
    -1/step base reward. Raw |v| is at most 0.07, and per-step |Δv| is
    at most ~0.002, so scale=1000 gives bonuses of order 1-2/step when
    the agent is actively accelerating.
    """

    def __init__(self, env, scale=1000.0, gamma=0.99):
        super().__init__(env)
        self.scale = scale
        self.gamma = gamma
        self._prev_phi = 0.0

    @staticmethod
    def _phi(state):
        return abs(float(state[1]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = 0.0 if terminated else self._phi(obs)
        bonus = self.scale * (self.gamma * phi - self._prev_phi)
        self._prev_phi = phi
        info["shaping_bonus"] = float(bonus)
        return obs, reward + bonus, terminated, truncated, info


class BestPositionWrapper(gym.Wrapper):
    """Sparse bonus for reaching a new rightmost position this episode."""

    def __init__(self, env, scale=50.0):
        super().__init__(env)
        self.scale = scale
        self._best_pos = -float("inf")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._best_pos = float(obs[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = float(obs[0])
        bonus = 0.0
        if pos > self._best_pos:
            bonus = self.scale * (pos - self._best_pos)
            self._best_pos = pos
        info["best_pos_bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info


class GoalBonusWrapper(gym.Wrapper):
    """One-time bonus added on the terminal step (when the agent reaches the goal).

    Not potential-based: this changes the reward structure rather than shaping it.
    Useful as a crutch when the agent struggles to find the goal at all.
    """

    def __init__(self, env, bonus=1000.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward += self.bonus
            info["goal_bonus"] = self.bonus
        return obs, reward, terminated, truncated, info
