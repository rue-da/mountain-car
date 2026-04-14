from .logger import Logger


def run(env, agent, episodes=1000, run_name="default"):
    logger = Logger(run_name)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        logger.log_episode(ep, total_reward, steps)
        if ep % 100 == 0:
            print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    logger.close()
    return logger
