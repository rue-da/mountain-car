from .logger import Logger


def run(env, agent, episodes=1000, run_name="default"):
    logger = Logger(run_name)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        last_info = {}

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            steps += 1
            last_info = info
            if terminated or truncated:
                break

        true_obj = last_info.get("true_obj_cum")
        success = last_info.get("success")
        agent_metrics = agent.get_metrics()
        logger.log_episode(ep, total_reward, steps,
                           true_obj=true_obj, success=success,
                           agent_metrics=agent_metrics)
        if ep % 100 == 0:
            obj_str = f"{true_obj:.1f}" if true_obj is not None else "n/a"
            print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}, "
                  f"true_obj={obj_str}, success={bool(success)}")

    logger.close()
    return logger
