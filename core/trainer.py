import os

import numpy as np

from .evaluator import evaluate
from .logger import Logger, log_run, runs_dir


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


def run_matrix(specs, seeds=(0, 1, 2), episodes=500, eval_episodes=50,
               base_run_name="matrix", verbose=True):
    """
    Train every (variant, agent) combo across multiple seeds and aggregate results.

    `specs` is a list of dicts:
      {
        "name": "ql",                       # short tag used in run_name
        "variant": "discrete_steps",        # passed to make_env
        "agent_cls": QLearningAgent,
        "agent_kwargs": {"n_bins": 40, "decay_steps": 80_000},
        "env_kwargs": {"energy_shaping": True},
      }

    Returns a dict keyed by (name, variant) with:
      "seeds":      list of seeds run
      "logs":       list of per-seed Logger.data lists
      "evals":      list of per-seed evaluator results
      "true_obj_mean":  mean across seeds of per-seed true_obj_mean
      "true_obj_std":   std across seeds of per-seed true_obj_mean
      "success_rate":   mean across seeds of per-seed success_rate
    """
    from envs import make_env  # local import: trainer is imported during package init

    out = {}
    for spec in specs:
        name = spec["name"]
        variant = spec["variant"]
        agent_cls = spec["agent_cls"]
        agent_kwargs = dict(spec.get("agent_kwargs", {}))
        env_kwargs = dict(spec.get("env_kwargs", {}))
        eps = spec.get("episodes", episodes)

        per_seed_logs, per_seed_evals = [], []
        for seed in seeds:
            run_name = f"{base_run_name}/{name}_{variant}_seed{seed}"
            if verbose:
                print(f"\n=== {run_name} ({eps} episodes) ===")

            train_env = make_env(variant, seed=seed, **env_kwargs)
            eval_env = make_env(variant, seed=seed + 10_000)

            kw = dict(agent_kwargs)
            kw.setdefault("seed", seed)
            agent = agent_cls(train_env, **kw)

            log_run(run_name, agent_cls.__name__,
                    env_config={"variant": variant, "seed": seed, **env_kwargs},
                    agent_config=agent.get_config())

            logger = run(train_env, agent, episodes=eps, run_name=run_name)
            results = evaluate(eval_env, agent, n=eval_episodes, run_name=run_name)

            save_path = os.path.join(runs_dir(run_name), "agent")
            try:
                agent.save(save_path)
            except Exception as e:
                if verbose:
                    print(f"  warn: could not save agent at {save_path}: {e}")

            per_seed_logs.append(logger.data)
            per_seed_evals.append(results)

            train_env.close()
            eval_env.close()

        per_seed_means = np.array([r["true_obj_mean"] for r in per_seed_evals])
        per_seed_success = np.array([r["success_rate"] for r in per_seed_evals])
        out[(name, variant)] = {
            "name": name,
            "variant": variant,
            "seeds": list(seeds),
            "logs": per_seed_logs,
            "evals": per_seed_evals,
            "true_obj_mean": float(per_seed_means.mean()),
            "true_obj_std": float(per_seed_means.std()),
            "success_rate": float(per_seed_success.mean()),
        }
        if verbose:
            agg = out[(name, variant)]
            print(f"\n--> {name} on {variant}: "
                  f"true_obj={agg['true_obj_mean']:.2f} ± {agg['true_obj_std']:.2f}, "
                  f"success_rate={agg['success_rate']:.2f} "
                  f"(over {len(seeds)} seeds)")

    return out
