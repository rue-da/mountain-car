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
- [x] `envs/shaping.py` — `EnergyShapingWrapper` (potential-based, preserves optimal policy)
- [x] `envs/tracking.py` — `TrueObjectiveWrapper` (info["true_obj*"], info["success"])
- [x] `envs/factory.py` — `make_env(variant, seed, shape=False)`
- [ ] Smoke-test notebook section 2 that loops over all 4 variants and renders one random rollout

## Person B — Tabular + discrete deep agents
Owns `agents/q_learning.py` (refactor) and `agents/dqn.py` (new). Target variants: `discrete_steps`, `discrete_fuel`.
- Replace uniform binning with **tile coding** (or at least 40×40 bins) on position/velocity
- Slower ε decay (reach ε_min after ~50% of training, not ~500 episodes)
- Should converge to ~−120 to −150 steps on `discrete_steps` with shaping in ≤50k env steps
- `agents/dqn.py`: either SB3 DQN wrapper or pytorch custom (replay buffer 50k, target net, double-DQN optional). Must expose `BaseAgent.choose_action` for plots to work deterministically.
- `save(path)` / `load(path)` on both
- Done when: `python -m agents.q_learning` trains 20k steps on `discrete_steps` and prints `eval_mean_steps < 180`

## Person C — Continuous agents + training infrastructure
Owns `agents/sac.py` (new) and `core/trainer.py` (extend). Target variants: `continuous_steps`, `continuous_fuel`.
- `agents/sac.py`: SB3 SAC wrapper. Add `train(env, total_steps)` method; trainer dispatches to it instead of step-by-step loop when agent has one.
- Optional second continuous baseline (TD3 or DDPG) for comparison in the report
- `core/trainer.run_matrix(variants, agent_factories, seeds)` that runs N×M×K and saves best-by-eval checkpoint to `models/{variant}_{agent}_seed{k}/`
- Extend `core/logger.py` to log both `reward/shaped` AND `objective/true_cum` as separate TB scalars (plus `episode/length` and `success_rate/rolling_100`)
- Add `stable-baselines3>=2.0` to `requirements.txt`

## Person D — Evaluation, visualization, notebook, report, Part 02
Owns `core/evaluator.py` (rewrite), `visualization/plots.py` (extend), `notebooks/main.ipynb` (rewrite), `presentation/` (new), Part 02.
- `core/evaluator.py`: 100 deterministic eps across fixed seeds (0..99). Greedy policy (ε=0 / SAC deterministic). Returns `{mean_steps, std_steps, mean_fuel, std_fuel, success_rate, trajectories}` — **computed from `info["true_obj*"]`, not from shaped reward**. Must run identically across all 4 variants.
- `visualization/plots.py` add:
  - `q_heatmap(agent)` — best-action color map over (position, velocity) [see assignment slides p.29]
  - `value_surface(agent)` — 3D V(s)=max_a Q(s,a) mesh [slides p.31–32]
  - `visitation_heatmap(agent)` [slide p.30]
  - `phase_portrait(trajectories)` with reward-colored traces [slides p.33–34]
  - `multi_seed_curve(logs)` — mean ± std band across seeds
  - `cross_variant_bar(results)` — eval table as grouped bars
- Notebook sections: 1. Setup • 2. Envs + shaping demo • 3. Agents overview • 4. Training (runs the matrix) • 5. Eval • 6. Cross-variant comparison • 7. Conclusions. Each section has markdown narrative for instructional value (the grading rubric weights "Value proposition" and "Outcomes" heavily).
- `presentation/`: paper-style doc — abstract, methodology, results, interpretation, conclusions. Single PDF or pptx.
- Part 02: pick one RL paper/project from the assignment's orientative list (DeepMind data-center cooling, Wayve self-driving, AlphaGo/AlphaZero, JPMorgan LOXM, Atari DQN, etc.). ~5–10 slides: problem framing, state/action/reward design, algorithm choice, results, your critical evaluation.

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
