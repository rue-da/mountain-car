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
    """Mean continuous action per state (SAC/continuous agents)."""
    ax, owns = _resolve_ax(ax)
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 50)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 50)
    actions = np.zeros((len(vel), len(pos)))

    with torch.no_grad():
        for i, v in enumerate(vel):
            for j, p in enumerate(pos):
                obs_t = torch.FloatTensor([[p, v]]).to(agent.model.device)
                actions[i, j] = agent.model.actor._predict(obs_t, deterministic=True).item()

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


def _value_grid(agent, env, n=50):
    """Compute V(s)=max_a Q(s,a) on an n×n (position, velocity) grid.
    Works for QLearningAgent, DQNAgent, and SACAgent."""
    pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], n)
    vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n)
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
                    action_t, _ = agent.model.actor.action_log_prob(obs_t)
                    values[i, j] = agent.model.critic(obs_t, action_t)[0].item()
            else:
                values[i, j] = agent.q_table[agent._discretize(state)].max()
    return pos, vel, values


def value_map(agent, env, ax=None):
    ax, owns = _resolve_ax(ax)
    pos, vel, values = _value_grid(agent, env)
    im = ax.imshow(values, origin="lower", aspect="auto",
                   extent=[pos[0], pos[-1], vel[0], vel[-1]], cmap="viridis")
    plt.colorbar(im, ax=ax, label="max Q(s,a)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Value map")
    if owns:
        plt.tight_layout()
        plt.show()


def value_surface(agent, env, ax=None, n=50):
    """3D surface of V(s)=max_a Q(s,a) over (position, velocity)."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        owns = True
    else:
        owns = False

    pos, vel, values = _value_grid(agent, env, n=n)
    P, V = np.meshgrid(pos, vel)
    surf = ax.plot_surface(P, V, values, cmap="viridis",
                           edgecolor="none", antialiased=True)
    plt.colorbar(surf, ax=ax, shrink=0.6, label="max Q(s,a)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value")
    ax.set_title("Value surface")
    if owns:
        plt.tight_layout()
        plt.show()


def visitation_heatmap(agent, env, n_episodes=20, bins=50, ax=None):
    """2D histogram of visited (position, velocity) states across rollouts."""
    ax, owns = _resolve_ax(ax)
    states = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        states.append(obs.copy())
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs.copy())
            done = terminated or truncated
    states = np.asarray(states)

    low = env.observation_space.low
    high = env.observation_space.high
    H, xedges, yedges = np.histogram2d(
        states[:, 0], states[:, 1], bins=bins,
        range=[[low[0], high[0]], [low[1], high[1]]],
    )
    im = ax.imshow(H.T, origin="lower", aspect="auto",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="magma")
    plt.colorbar(im, ax=ax, label="visit count")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"State visitation ({n_episodes} eps)")
    if owns:
        plt.tight_layout()
        plt.show()


def multi_seed_curve(seed_logs, key="true_obj", window=50, ax=None, label=None):
    """Mean ± 1σ training curve across seeds.

    `seed_logs` is a list of per-seed log lists (Logger.data dicts), as
    produced by `run_matrix`. Episodes that are missing the key are skipped
    via NaN masking, so seeds that occasionally drop a metric still align.
    """
    ax, owns = _resolve_ax(ax)
    series = []
    for log in seed_logs:
        vals = [d.get(key) for d in log]
        vals = [np.nan if v is None else float(v) for v in vals]
        series.append(vals)

    if not series:
        ax.text(0.5, 0.5, "no seeds provided", ha="center", va="center")
        ax.set_axis_off()
        if owns:
            plt.show()
        return

    L = min(len(s) for s in series)
    arr = np.array([s[:L] for s in series], dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    if L >= window:
        kernel = np.ones(window) / window
        mean = np.convolve(mean, kernel, mode="valid")
        std = np.convolve(std, kernel, mode="valid")
        x = np.arange(window - 1, L)
    else:
        x = np.arange(L)

    line, = ax.plot(x, mean, label=label or f"mean ({arr.shape[0]} seeds)")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=line.get_color())
    ax.set_xlabel("Episode")
    ax.set_ylabel(key)
    ax.set_title(f"Multi-seed {key}")
    ax.legend()
    if owns:
        plt.tight_layout()
        plt.show()


def cross_variant_bar(matrix_results, metric="true_obj_mean", ax=None):
    """Bar chart comparing agents across variants from `run_matrix` output.

    `matrix_results` is the dict returned by `core.trainer.run_matrix`,
    keyed by (name, variant). Bars are grouped by variant, with one bar
    per agent name within each group. Error bars use `true_obj_std` when
    metric is "true_obj_mean".
    """
    ax, owns = _resolve_ax(ax)

    names, variants = [], []
    for (name, variant) in matrix_results.keys():
        if name not in names:
            names.append(name)
        if variant not in variants:
            variants.append(variant)

    width = 0.8 / max(len(names), 1)
    x = np.arange(len(variants))

    for i, name in enumerate(names):
        heights, errs = [], []
        for variant in variants:
            agg = matrix_results.get((name, variant))
            if agg is None:
                heights.append(np.nan)
                errs.append(0.0)
                continue
            heights.append(agg[metric])
            errs.append(agg.get("true_obj_std", 0.0) if metric == "true_obj_mean" else 0.0)
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, yerr=errs,
               capsize=3, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Cross-variant comparison ({metric})")
    ax.legend()
    if owns:
        plt.tight_layout()
        plt.show()
