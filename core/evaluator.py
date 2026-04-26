import numpy as np

from .logger import Logger


def evaluate(env, agent, n=50, run_name=None):
    rewards = []
    true_objs = []
    successes = []
    trajectories = []

    original_epsilon = getattr(agent, "epsilon", None)
    if original_epsilon is not None:
        agent.epsilon = 0  # greedy during eval

    for _ in range(n):
        state, _ = env.reset()
        total_reward = 0
        trajectory = [state.copy()]
        last_info = {}

        while True:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(state.copy())
            last_info = info
            if terminated or truncated:
                break

        rewards.append(total_reward)
        true_objs.append(last_info.get("true_obj_cum", float("nan")))
        successes.append(bool(last_info.get("success", False)))
        trajectories.append(trajectory)

    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    rewards_arr = np.array(rewards, dtype=float)
    true_objs_arr = np.array(true_objs, dtype=float)
    results = {
        "rewards": rewards,
        "true_objs": true_objs,
        "successes": successes,
        "trajectories": trajectories,
        "reward_shaped_mean": float(rewards_arr.mean()),
        "true_obj_mean": float(np.nanmean(true_objs_arr)),
        "true_obj_std": float(np.nanstd(true_objs_arr)),
        "success_rate": float(np.mean(successes)),
    }

    print(f"Eval over {n} episodes: "
          f"true_obj={results['true_obj_mean']:.2f} +/- {results['true_obj_std']:.2f}, "
          f"success_rate={results['success_rate']:.2f}, "
          f"shaped_reward={results['reward_shaped_mean']:.1f}")

    if run_name is not None:
        logger = Logger(run_name)
        logger.log_eval(results)
        logger.close()

    return results
