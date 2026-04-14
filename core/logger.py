from torch.utils.tensorboard import SummaryWriter
import csv
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Logger:
    def __init__(self, run_name):
        runs_dir = os.path.join(PROJECT_ROOT, "runs", run_name)
        self.writer = SummaryWriter(runs_dir)
        self.data = []
        self.csv_path = os.path.join(runs_dir, "log.csv")
        os.makedirs(runs_dir, exist_ok=True)

    def log_episode(self, episode, reward, steps):
        self.writer.add_scalar("reward", reward, episode)
        self.writer.add_scalar("steps", steps, episode)
        self.data.append({"episode": episode, "reward": reward, "steps": steps})

    def save_csv(self):
        if not self.data:
            return
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.data[0].keys())
            w.writeheader()
            w.writerows(self.data)

    def close(self):
        self.save_csv()
        self.writer.close()
