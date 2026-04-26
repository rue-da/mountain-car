import gymnasium as gym

from .shaping import BestPositionWrapper, EnergyShapingWrapper, GoalBonusWrapper
from .tracking import TrueObjectiveWrapper
from .wrappers import ContinuousStepsRewardWrapper, DiscreteFuelRewardWrapper

VARIANTS = ("discrete_steps", "discrete_fuel", "continuous_steps", "continuous_fuel")


def make_env(variant, seed=None,
             energy_shaping=False, energy_scale=1000.0, energy_gamma=0.99,
             best_pos_shaping=False, best_pos_scale=50.0,
             goal_bonus=0.0,
             render_mode=None):
    """
    Build one of the four assignment variants. Wrapper order:
      base env → variant reward wrapper → TrueObjectiveWrapper
                 → [EnergyShapingWrapper] → [BestPositionWrapper] → [GoalBonusWrapper]

    TrueObjectiveWrapper sits before all shaping so info["true_obj*"] always
    reports the unshaped, variant-specific objective for fair eval.

    Shaping options (all default off):
      energy_shaping    : potential-based bonus on |velocity|. Optimal-policy-safe.
      best_pos_shaping  : sparse bonus for reaching a new rightmost position.
      goal_bonus        : one-time bonus on terminal step. 0.0 = off.
    """
    if variant == "discrete_steps":
        env = gym.make("MountainCar-v0", render_mode=render_mode)
    elif variant == "discrete_fuel":
        env = DiscreteFuelRewardWrapper(gym.make("MountainCar-v0", render_mode=render_mode))
    elif variant == "continuous_steps":
        env = ContinuousStepsRewardWrapper(
            gym.make("MountainCarContinuous-v0", render_mode=render_mode)
        )
    elif variant == "continuous_fuel":
        env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
    else:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    env = TrueObjectiveWrapper(env, variant=variant)
    if energy_shaping:
        env = EnergyShapingWrapper(env, scale=energy_scale, gamma=energy_gamma)
    if best_pos_shaping:
        env = BestPositionWrapper(env, scale=best_pos_scale)
    if goal_bonus > 0.0:
        env = GoalBonusWrapper(env, bonus=goal_bonus)

    if seed is not None:
        env.reset(seed=seed)
    return env
