# Group split — 4 people, 10 days (due 2026-04-29)

Part 01 (70%) is the code deliverable. Part 02 (30%) is presentation-only. The split below owns Part 01 across the first three people and hands Part 02 to Person D alongside the notebook/report assembly.

Shared contract: nobody touches `envs/`, `core/logger.py`, or the notebook structure without coordinating. Each new agent subclasses `BaseAgent`. Every env is built via `envs.make_env(variant, seed, shape=...)` — never `gym.make` directly. Every training run logs to TensorBoard AND writes the true (unshaped) objective from `info["true_obj*"]`.

## Variants (canonical names, use these strings everywhere)
| variant | base env | reward | true objective |
|---|---|---|---|
| `discrete_steps` | `MountainCar-v0` | −1/step | # steps to goal |
| `discrete_fuel` | `MountainCar-v0` + `DiscreteFuelRewardWrapper` | 0 for no-op, −1 per left/right | # non-null actions |
| `continuous_steps` | `MountainCarContinuous-v0` + `ContinuousStepsRewardWrapper` | −1 − 0.1·\|a\| per step, +100 on goal | # steps to goal |
| `continuous_fuel` | `MountainCarContinuous-v0` | −0.1·a² per step, +100 on goal | ∫ a² |

## Person A — Environments + reward shaping  [DONE, see status at bottom]
Owns `envs/`. Foundation everyone else builds on.
- [x] `envs/wrappers.py` — `DiscreteFuelRewardWrapper`, `ContinuousStepsRewardWrapper`
- [x] `envs/shaping.py` — `EnergyShapingWrapper` (potential-based, preserves optimal policy); added `BestPositionWrapper` (sparse bonus for new rightmost position) and `GoalBonusWrapper` (one-time terminal bonus)
- [x] `envs/tracking.py` — `TrueObjectiveWrapper` (info["true_obj*"], info["success"])
- [x] `envs/factory.py` — `make_env` signature updated with descriptive params: `energy_shaping`, `energy_scale`, `energy_gamma`, `best_pos_shaping`, `best_pos_scale`, `goal_bonus`, `render_mode`
- [ ] Smoke-test notebook section 2 that loops over all 4 variants and renders one random rollout

## Person B — Tabular + discrete deep agents  [IN PROGRESS]

Owns `agents/q_learning.py` (refactor) and `agents/dqn.py` (new). Target variants: `discrete_steps`, `discrete_fuel`.

- [x] 40×40 bins on position/velocity (`n_bins=40`)
- [x] Slower ε decay via `decay_steps` param (configurable, default 50k steps)
- [x] `agents/dqn.py`: custom PyTorch — replay buffer 50k, target network, epsilon-greedy. Exposes `choose_action` for deterministic plotting
- [x] `save(path)` / `load(path)` on both agents
- [x] `get_config()` and `get_metrics()` on both (epsilon, q_loss for DQN) — feeds into TensorBoard and CSV via trainer
- [x] Fixed `bin_edges` dtype bug in `QLearningAgent.load` (object array → float64)
- [ ] Convergence target: `eval_mean_steps < 180` on `discrete_steps` — currently ~196, needs more training or tuning

## Person C — Continuous agents + training infrastructure  [IN PROGRESS]

Owns `agents/sac.py` (new) and `core/trainer.py` (extend). Target variants: `continuous_steps`, `continuous_fuel`.

- [x] `agents/sac.py`: SB3 SAC wrapper exposing `choose_action`, `learn`, `get_config`, `get_metrics`, `save`, `load`
- [x] `agents/sac.py`: q_loss tracked via SB3 internal logger (`train/critic_loss`), flows into CSV and TensorBoard
- [x] `core/trainer.py`: extended to capture `info["true_obj_cum"]` and `info["success"]` per episode, calls `agent.get_metrics()` and passes to logger
- [x] `core/logger.py`: logs `train/episode_reward`, `train/true_obj`, `train/success`, `train/success_rate_100`, agent metrics under their own namespaces; `log_run()` writes to `runs.json`
- [x] `core/trainer.run_matrix(...)` — multi-seed × multi-(variant, agent) runner, returns aggregated `true_obj_mean/std` and `success_rate` per (name, variant)
- [ ] TD3 or DDPG optional baseline — not implemented
- [x] `stable-baselines3>=2.0` already in `requirements.txt`

## Person D — Evaluation, visualization, notebook, report, Part 02  [IN PROGRESS]

Owns `core/evaluator.py` (rewrite), `visualization/plots.py` (extend), `notebooks/main.ipynb` (rewrite), `presentation/` (new), Part 02.

- [x] `core/evaluator.py`: reads `info["true_obj_cum"]` and `info["success"]`; returns `reward_shaped_mean`, `true_obj_mean`, `true_obj_std`, `success_rate`, `trajectories`
- [x] `visualization/plots.py` — implemented:
  - [x] `policy_map(agent)` — best-action color map over (position, velocity)
  - [x] `value_map(agent)` — V(s)=max_a Q(s,a) heatmap (works for QL, DQN, SAC)
  - [x] `phase_portrait(agent, env)` — trajectories through state space with goal marker
  - [x] `action_heatmap(agent)` — continuous action magnitude per state (SAC)
  - [x] `entropy_heatmap(agent)` — policy uncertainty per state (SAC)
  - [x] `reward_curve`, `true_obj_curve`, `success_rate_curve`, `q_convergence`
  - [x] `value_surface` — 3D mesh of V(s)=max_a Q(s,a); reuses `_value_grid` helper shared with `value_map`
  - [x] `visitation_heatmap` — 2D histogram of (position, velocity) visited across rollouts
  - [x] `multi_seed_curve` — mean ± 1σ training curve across seeds, takes `run_matrix` per-seed log lists
  - [x] `cross_variant_bar` — grouped bar chart over `run_matrix` output, error bars from per-seed std
- [x] `notebooks/main.ipynb`: QL, DQN, SAC sections with train/eval/plot; loads from saved files; video recording cells
- [ ] Notebook narrative markdown (methodology, conclusions)
- [ ] `presentation/` — not started
- [ ] Part 02 — not started

## Dependency order
```
A (done) ──┬──► B ──┐
            ├──► C ──┼──► D
            └───────┘
```
B and C can start in parallel as soon as A merges. D's notebook/eval can start immediately against stubbed agents; final plots and cross-variant comparison wait for B + C.

## Submission
- One zip `RLI_22_00 – Group {XY}.zip` — coordinator uploads
- `requirements.txt` pinned, runs clean in fresh venv, no warnings
- Notebook runs end-to-end

## Status
- **2026-04-19** Person A foundation merged. Smoke test across all 4 variants passes. B/C unblocked.
- **2026-04-26** B: QL and DQN implemented and trained on `discrete_steps`. C: SAC implemented and trained on `continuous_steps`. D: evaluator, plots, and notebook updated for all three agents. Remaining: convergence tuning for QL/DQN, run_matrix, missing plots (value_surface, visitation, multi_seed, cross_variant), notebook narrative, presentation, Part 02.
- **2026-04-28** Cross-variant infrastructure landed: `run_matrix` aggregates per-seed eval into mean/std/success_rate; new plots `value_surface`, `visitation_heatmap`, `multi_seed_curve`, `cross_variant_bar`. Smoke tests fixed (`shape=True` → `energy_shaping=True`).
- **2026-04-28** Notebook extended with min-fuel section: QL+DQN on `discrete_fuel` (with `goal_bonus=100` since native reward makes "do nothing" optimal otherwise), SAC on `continuous_fuel`. Final `Cross-variant comparison` cell builds a matrix-shaped dict from the six eval results and plots `true_obj_mean` and `success_rate` with `cross_variant_bar`. Remaining: run the new cells end-to-end and tune, narrative markdown, presentation, Part 02.
