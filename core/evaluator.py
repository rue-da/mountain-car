import numpy as np


def evaluate(env, agent, n=50):
    rewards = []
    trajectories = []

    original_epsilon = getattr(agent, "epsilon", None)
    if original_epsilon is not None:
        agent.epsilon = 0  # greedy during eval

    for _ in range(n):
        state, _ = env.reset()
        total_reward = 0
        trajectory = [state.copy()]

        while True:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            trajectory.append(state.copy())
            if terminated or truncated:
                break

        rewards.append(total_reward)
        trajectories.append(trajectory)

    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    mean_r = np.mean(rewards)
    print(f"Eval over {n} episodes: mean_reward={mean_r:.1f} +/- {np.std(rewards):.1f}")
    return {"rewards": rewards, "trajectories": trajectories, "mean_reward": mean_r}
