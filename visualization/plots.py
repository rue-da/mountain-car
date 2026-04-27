import numpy as np
import matplotlib.pyplot as plt
import torch


def _resolve_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
        return ax, True
    return ax, False


def reward_curve(data, window=50, ax=None):
    ax, owns = _resolve_ax(ax)
    rewards = [d["reward"] for d in data]
    ax.plot(rewards, alpha=0.3, label="raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"avg({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shaped reward")
    ax.set_title("Reward curve")
    ax.legend()
    if owns:
        plt.tight_layout()
        plt.show()


def true_obj_curve(data, window=50, ax=None):
    ax, owns = _resolve_ax(ax)
    values = [d.get("true_obj") for d in data if d.get("true_obj") is not None]
    if not values:
        ax.text(0.5, 0.5, "no true_obj logged", ha="center", va="center")
        ax.set_axis_off()
        if owns:
            plt.show()
        return
    ax.plot(values, alpha=0.3, label="raw")
    if len(values) >= window:
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(values)), smoothed, label=f"avg({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("True objective (cumulative)")
    ax.set_title("True objective curve")
    ax.legend()
    if owns:
        plt.tight_layout()
        plt.show()


def success_rate_curve(data, ax=None):
    ax, owns = _resolve_ax(ax)
    values = [d.get("success_rate_100") for d in data if d.get("success_rate_100") is not None]
    if not values:
        ax.text(0.5, 0.5, "no success_rate logged", ha="center", va="center")
        ax.set_axis_off()
        if owns:
            plt.show()
        return
    ax.plot(values)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (last 100)")
    ax.set_title("Success rate curve")
    ax.set_ylim(0, 1)
    if owns:
        plt.tight_layout()
        plt.show()


def q_convergence(data, ax=None):
    ax, owns = _resolve_ax(ax)
    values = [(d["episode"], d["q_loss"]) for d in data if d.get("q_loss") is not None]
    if not values:
        ax.text(0.5, 0.5, "no q_loss logged", ha="center", va="center")
        ax.set_axis_off()
        if owns:
            plt.show()
        return
    eps, losses = zip(*values)
    ax.plot(eps, losses, alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q loss")
    ax.set_title("Q convergence")
    if owns:
        plt.tight_layout()
        plt.show()


def policy_map(agent, env, ax=None):
    ax, owns = _resolve_ax(ax)
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    policy = np.zeros((len(vel), len(pos)))

    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            policy[i, j] = agent.choose_action(np.array([p, v]))

    im = ax.imshow(policy, origin="lower", aspect="auto",
                   extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="Action")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Policy map")
    if owns:
        plt.tight_layout()
        plt.show()


def action_heatmap(agent, env, ax=None):
    """Deterministic action per state — works for any SB3 agent."""
    ax, owns = _resolve_ax(ax)
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    actions = np.zeros((len(vel), len(pos)))

    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            obs = np.array([[p, v]], dtype=np.float32)
            action, _ = agent.model.predict(obs, deterministic=True)
            actions[i, j] = float(action.flat[0])

    im = ax.imshow(actions, origin="lower", aspect="auto",
                   extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="Action (force)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Action heatmap")
    if owns:
        plt.tight_layout()
        plt.show()


def entropy_heatmap(agent, env, ax=None):
    """Policy entropy per state — high = uncertain, low = confident (SAC)."""
    ax, owns = _resolve_ax(ax)
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    entropies = np.zeros((len(vel), len(pos)))

    with torch.no_grad():
        for i, v in enumerate(vel):
            for j, p in enumerate(pos):
                obs_t = torch.FloatTensor([[p, v]]).to(agent.model.device)
                _, log_prob = agent.model.actor.action_log_prob(obs_t)
                entropies[i, j] = -log_prob.item()

    im = ax.imshow(entropies, origin="lower", aspect="auto",
                   extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="plasma")
    plt.colorbar(im, ax=ax, label="Entropy (-log prob)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Entropy heatmap")
    if owns:
        plt.tight_layout()
        plt.show()


def phase_portrait(agent, env, n_episodes=10, ax=None):
    """Trajectories overlaid on state space with velocity arrows."""
    ax, owns = _resolve_ax(ax)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        states = [obs.copy()]
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs.copy())
            done = terminated or truncated
        states = np.array(states)
        ax.plot(states[:, 0], states[:, 1], alpha=0.5, linewidth=0.8)
        ax.annotate("", xy=states[-1], xytext=states[-2],
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Phase portrait")
    ax.axvline(x=0.45, color="green", linestyle="--", linewidth=0.8, label="goal")
    ax.legend()
    if owns:
        plt.tight_layout()
        plt.show()


def value_map(agent, env, ax=None):
    ax, owns = _resolve_ax(ax)
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    values = np.zeros((len(vel), len(pos)))

    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            state = np.array([p, v], dtype=np.float32)
            if hasattr(agent, 'online'):
                with torch.no_grad():
                    s = torch.as_tensor(state).unsqueeze(0).to(agent.device)
                    values[i, j] = agent.online(s).max(dim=1).values.item()
            elif hasattr(agent, 'model') and hasattr(agent.model, 'critic'):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(state).unsqueeze(0).to(agent.model.device)
                    if hasattr(agent.model.actor, 'action_log_prob'):
                        action_t, _ = agent.model.actor.action_log_prob(obs_t)
                    else:
                        action_t = agent.model.actor._predict(obs_t, deterministic=True)
                    values[i, j] = agent.model.critic(obs_t, action_t)[0].item()
            elif hasattr(agent, 'model') and hasattr(agent.model.policy, 'predict_values'):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(state).unsqueeze(0).to(agent.model.device)
                    values[i, j] = agent.model.policy.predict_values(obs_t).item()
            else:
                values[i, j] = agent.q_table[agent._discretize(state)].max()

    im = ax.imshow(values, origin="lower", aspect="auto",
                   extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="viridis")
    plt.colorbar(im, ax=ax, label="max Q(s,a)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Value map")
    if owns:
        plt.tight_layout()
        plt.show()
