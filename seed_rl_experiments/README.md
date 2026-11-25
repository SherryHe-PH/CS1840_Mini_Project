# On-policy implementation experiments with Seed RL

This project studies the implementation choices in
[*What Matters in On-Policy Reinforcement Learning?*](https://arxiv.org/abs/2006.05990)
using the original [Google Seed RL](https://github.com/google-research/seed_rl)
TensorFlow learner, agent, losses, and normalization. Experiments use saved
configurations, independent task/seed runs, frozen evaluation, and summaries,
with CPU execution on Harvard OOD.

## Quick start on OOD

Use an allocated **Linux x86_64 CPU session**.
For the full default configuration, start with six CPUs and 16 GB; these are
initial resource requests, not measured per-run requirements. A smoke run uses
one actor and can use a smaller allocation.

Clone the whole repository, then run setup from its experiment folder:

```bash
git clone --branch seed-rl-experiments https://github.com/SherryHe-PH/CS1840_Mini_Project.git
cd CS1840_Mini_Project/seed_rl_experiments
bash scripts/setup_ood.sh
source scripts/ood_env.sh
```

Setup creates a Python 3.8 Conda environment and installs the native
dependencies without administrator access. **No previous `seed-mujoco`
environment is required.** The experiment folder automatically uses the
Seed RL implementation in the parent repository.

The default environment is `seed-experiments`. Reactivate it with
`source scripts/ood_env.sh` in each new terminal; this also restores the CPU,
MuJoCo, compiler, and library paths. `SEED_CONDA_ENV` selects another installed
environment, and `SEED_MINIFORGE` selects another Miniforge installation.

First verify this experiment wrapper, including its separate checkpoint
evaluation process:

```bash
python -m seed_ppo.train --config configs/smoke.yaml \
  --output runs/smoke_toy/seed_0 --timeout 600

python -m seed_ppo.train --config configs/smoke.yaml \
  --env-id HalfCheetah-v2 \
  --output runs/smoke_halfcheetah/seed_0 --timeout 900
```

Each smoke run uses 256 transitions and evaluates two episodes at steps 128 and
256. Successful execution prints `COMPLETE`; inspect `evaluation.csv` and
`summary.json`. The earlier native Seed RL CPU smoke tests passed on OOD;
the new paper-configured wrapper must pass these checks before full experiments.
Always choose a fresh output directory when repeating a run.

## Run experiments

After both smoke tests print `COMPLETE`, run one full seed, or the complete
five-task, three-seed baseline:

```bash
python -m seed_ppo.train --config configs/ppo_default.yaml \
  --env-id HalfCheetah-v2 --seed 0 --output runs/one_seed/HalfCheetah-v2/seed_0

python scripts/run_baseline.py --jobs 1 --output runs/baseline
python scripts/summarize.py --runs runs/baseline --output results/baseline
```

`--jobs` counts independent experiments; each default experiment has four actor
processes, one learner, and an evaluation process at checkpoints. Output includes
resolved YAML-equivalent settings/Gin, source and dependency provenance, native
TensorFlow checkpoints/events, `training.csv`, `evaluation.csv`, and status.
Reports contain `per_seed.csv`, `aggregate.csv`, and `learning_curves.png`.
Native Seed loss/gradient diagnostics remain in the TensorBoard events under
each run's `native/` directory.

For Slurm, create a fixed experiment manifest on OOD, then submit one array
element per task/seed. Supply your account/partition options when required.

```bash
python scripts/run_baseline.py --output runs/baseline_array \
  --manifest plans/baseline.json --dry-run

count=$(python scripts/run_experiments.py --manifest plans/baseline.json --count)
export SEED_MANIFEST="$PWD/plans/baseline.json"
sbatch --array="0-$((count - 1))%3" scripts/slurm_experiments.sbatch
```

The manifest embeds the resolved settings; later YAML edits do not change
submitted runs. `%3` limits concurrent array elements to three. Slurm settings
and the new wrapper's full experiment protocol still need verification on your
allocation. Completed matching runs can be skipped with `--skip-complete`;
interrupted runs require a new output directory.

## Experiment design

The control configuration is saved in `configs/ppo_default.yaml`. It targets
Appendix C, Table 2: a shared 64×64 tanh network, PPO clip 0.2, initial Gaussian
standard deviation 1.0, observation/value normalization, GAE λ=0.95 and γ=0.99,
256 environments, 2,048 transitions per update, ten epochs, minibatches of 64,
Adam learning rate 3e-4 decaying toward zero, and no entropy bonus. Raw Gaussian
samples are retained for likelihoods; a separate simulator copy is clipped to
normalized action bounds and mapped to the task's physical controls.

| Item | Protocol |
| --- | --- |
| Tasks | Hopper-v2, Walker2d-v2, HalfCheetah-v2, Ant-v2, Humanoid-v2 |
| Seeds | 0, 1, 2, independently trained for each configuration |
| Budget | 1M transitions for Hopper/Walker2d/HalfCheetah; 2M for Ant/Humanoid |
| Evaluation | Every 100k transitions: 100 episodes of a frozen stochastic policy, using a separate RNG process |
| Seed score | Mean of checkpoint evaluation means over training |
| Configuration score | Median of the three seed scores; final return reported separately |

Training finishes a complete Seed learner batch: requested 100k and 1M points
occur at 100,352 and 1,001,472 consumed transitions with the default batch size.
Scheduled and actual counts are saved. The runtime uses Gym 0.17.3 / MuJoCo 2.1;
the paper used Gym v1 / MuJoCo 2.0. Results are labeled a simulator port.
The supported thematic subset and remaining differences are documented in
`docs/protocol.md`; this is not the paper's complete search framework.

## Experiments by thematic group

Each group file defines a small Cartesian grid. Only parameters from that group
and the Adam learning rate may vary; all other baseline settings stay fixed.
Run the baseline control and each variant with the same tasks, budgets, seeds,
and evaluation protocol. The provided grids are examples to edit for your
compute budget, not the paper's full search spaces.

| Group | Saved grid | Parameters in the example |
| --- | --- | --- |
| 3.1 Policy losses | `configs/groups/policy_losses.yaml` | PPO clipping epsilon |
| 3.2 Network architecture | `configs/groups/architecture.yaml` | Hidden width, shared/separate networks |
| 3.3 Normalization and clipping | `configs/groups/normalization.yaml` | Observation normalization |
| 3.4 Advantage estimation | `configs/groups/advantage_estimation.yaml` | GAE lambda, value clipping |
| 3.5 Training setup | `configs/groups/training_setup.yaml` | Environment count, epochs |
| 3.6 Timesteps handling | `configs/groups/timesteps.yaml` | Discount factor |
| 3.7 Optimizers | `configs/groups/optimizers.yaml` | Adam epsilon, learning-rate decay |
| 3.8 Regularization | `configs/groups/regularization.yaml` | Entropy coefficient |

Every example also varies learning rate. Inspect the planned configurations and
run count before launching. This architecture example selects one task and all
three seeds:

```bash
python scripts/run_experiments.py \
  --config configs/ppo_default.yaml \
  --group configs/groups/architecture.yaml \
  --envs Hopper-v2 --seeds 0 1 2 \
  --output runs/architecture --manifest plans/architecture.json --dry-run

python scripts/run_experiments.py --manifest plans/architecture.json --count
python scripts/run_experiments.py --manifest plans/architecture.json --jobs 1

python scripts/summarize.py --runs runs/architecture --output results/architecture
```

To submit that same group as a Slurm array:

```bash
count=$(python scripts/run_experiments.py --manifest plans/architecture.json --count)
export SEED_MANIFEST="$PWD/plans/architecture.json"
sbatch --array="0-$((count - 1))%3" scripts/slurm_experiments.sbatch
```

Replace the group-file path and choose new manifest/output paths for another
group. Omit `--envs` and `--seeds` to select all five tasks and seeds 0, 1, 2.
Results stay separate by variant/task/seed, and reporting rejects mixed settings
or runtime provenance within a configuration. Reduced tasks, seeds, or budgets
are useful exploratory runs and are labeled as partial protocols.
