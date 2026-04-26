from torch.utils.tensorboard import SummaryWriter
import csv
import json
import os
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_dir(run_name):
    return os.path.join(PROJECT_ROOT, "runs", run_name)


def log_run(run_name, agent_name, env_config, agent_config):
    path = os.path.join(PROJECT_ROOT, "runs", "runs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "agent": agent_name,
        "env": env_config,
        "hyperparams": agent_config,
    }
    existing = []
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    existing.append(entry)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


class Logger:
    def __init__(self, run_name, success_window=100):
        self.run_name = run_name
        path = runs_dir(run_name)
        os.makedirs(path, exist_ok=True)
        self.writer = SummaryWriter(path)
        self.csv_path = os.path.join(path, "log.csv")
        self.data = []
        self._success_window_size = success_window
        self._success_window = []

    def log_episode(self, episode, reward, steps,
                    true_obj=None, success=None, agent_metrics=None):
        self.writer.add_scalar("train/episode_reward", reward, episode)
        self.writer.add_scalar("train/episode_length", steps, episode)
        row = {"episode": episode, "reward": reward, "steps": steps}

        if true_obj is not None:
            self.writer.add_scalar("train/true_obj", true_obj, episode)
            row["true_obj"] = float(true_obj)

        if success is not None:
            s = 1.0 if success else 0.0
            self.writer.add_scalar("train/success", s, episode)
            self._success_window.append(s)
            if len(self._success_window) > self._success_window_size:
                self._success_window.pop(0)
            rate = sum(self._success_window) / len(self._success_window)
            self.writer.add_scalar("train/success_rate_100", rate, episode)
            row["success"] = bool(success)
            row["success_rate_100"] = rate

        if agent_metrics:
            for key, value in agent_metrics.items():
                if value is None:
                    continue
                self.writer.add_scalar(key, value, episode)
                col = key.split("/", 1)[1] if "/" in key else key
                row[col] = float(value)

        self.data.append(row)

    def log_eval(self, results, step=None):
        """Write evaluator summary scalars under eval/ at the given step."""
        if step is None:
            step = self.data[-1]["episode"] if self.data else 0
        keys = ("reward_shaped_mean", "true_obj_mean", "true_obj_std", "success_rate")
        for k in keys:
            if k in results:
                self.writer.add_scalar(f"eval/{k}", results[k], step)

    def save_csv(self):
        if not self.data:
            return
        all_keys = []
        for row in self.data:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(self.data)

    def close(self):
        self.save_csv()
        self.writer.close()
