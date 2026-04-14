import numpy as np
import matplotlib.pyplot as plt


def reward_curve(data, window=50):
    rewards = [d["reward"] for d in data]
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure()
    plt.plot(rewards, alpha=0.3, label="raw")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"avg({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def policy_map(agent, env):
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    policy = np.zeros((len(vel), len(pos)))

    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            policy[i, j] = agent.choose_action(np.array([p, v]))

    plt.figure()
    plt.imshow(policy, origin="lower", aspect="auto",
               extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="coolwarm")
    plt.colorbar(label="Action (0=left, 1=none, 2=right)")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Policy Map")
    plt.tight_layout()
    plt.show()


def trajectory(trajectories, n=5):
    plt.figure()
    for i, traj in enumerate(trajectories[:n]):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=f"ep {i}")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()
