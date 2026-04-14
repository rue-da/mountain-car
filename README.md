# MountainCar RL Testbed

Testbed for testing reinforcement learning algorithms on the MountainCar-v0 environment from Gymnasium.

## Structure

- **envs/** — Custom reward wrappers (e.g. penalizing fuel usage)
- **agents/** — RL agents. Add new algorithms by subclassing `BaseAgent`
- **core/** — Training loop, evaluation, and logging (TensorBoard + CSV)
- **visualization/** — Reward curves, policy maps, and trajectory plots
- **notebooks/** — Entry point via `main.ipynb`
- **runs/** — Auto-generated TensorBoard event files and CSV logs per training run

## Usage

```bash
pip install -r requirements.txt
```

Run `notebooks/main.ipynb`, then monitor training in real time:

```bash
tensorboard --logdir=runs/
```