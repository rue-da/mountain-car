import gymnasium as gym

from .shaping import EnergyShapingWrapper
from .tracking import TrueObjectiveWrapper
from .wrappers import ContinuousStepsRewardWrapper, DiscreteFuelRewardWrapper

VARIANTS = ("discrete_steps", "discrete_fuel", "continuous_steps", "continuous_fuel")


def make_env(variant, seed=None, shape=False, shape_scale=1000.0, shape_gamma=0.99,
             render_mode=None):
    """
    Build one of the four assignment variants with consistent wrapping:
      base env → reward wrapper (if any) → TrueObjectiveWrapper → [EnergyShapingWrapper]

    Ordering means shaping affects only the agent's training reward; info["true_obj*"]
    always reports the unshaped, variant-specific objective for fair cross-variant eval.
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
    if shape:
        env = EnergyShapingWrapper(env, scale=shape_scale, gamma=shape_gamma)

    if seed is not None:
        env.reset(seed=seed)
    return env
