# Implementation mapping

The `seed_rl_experiments/` folder lives inside the existing Seed RL fork and
automatically locates its parent checkout. The original implementation,
`LICENSE`, and `AUTHORS` stay at the repository root. This folder adds the
experiment layer below.

| Local project role | Seed branch implementation |
| --- | --- |
| `ppo/config.py` | `seed_ppo/config.py`: named defaults, validation, group grids, Gin bindings |
| `ppo/model.py` | Upstream `ContinuousControlAgent` in `agents/policy_gradient/modules/continuous_control_agent.py` |
| Policy/value losses | Upstream `GeneralizedOnPolicyLoss`, `GeneralizedAdvantagePolicyLoss`, and policy regularizers |
| `ppo/normalization.py` | Upstream `InputNormalization`, `PopArt` with compensation disabled, and `AverageMeanStd` |
| `ppo/advantages.py` | Upstream `GAE`, called by the native loss/training-step utilities |
| `ppo/train.py` | `seed_ppo/train.py` supervises workers; `worker.py` drives native `Learner.minimize()` |
| `ppo/evaluate.py` | `seed_ppo/evaluate.py`: independent process restoring the native checkpoint |
| `scripts/run_baseline.py` | Same task/seed interface; delegates to the common experiment planner |
| Thematic experiments | Saved YAML grids and immutable, checksummed task manifests |
| Slurm | One manifest entry per array element, all actors/learner on one allocated node |
| Reporting | Measured evaluation CSVs; paper learning-curve score and separate final return |

## Intentional adapters

`runtime.py` creates the original native learner, agent, losses, and optimizer.
It temporarily extends the native action-distribution factory during learner
construction: the baseline uses an ordinary Gaussian, keeping raw samples in
native unrolls. An outer Gym wrapper clips only the simulator copy before Seed's
normalized-to-physical action mapping. The optional tanh choice uses Seed's
existing tanh distribution. This avoids relying on `ClippedIdentity` caching
semantics to recover raw-sample likelihoods.

The worker drives native update calls so it can save a checkpoint at the first
iteration crossing each requested evaluation point. A separate evaluation
process reconstructs the same native object graph and restores `loss_fn`, the
optimizer, and the training counter. It uses the restored training agent's
policy and observation normalizer without updating statistics. Separate RNGs
and environments isolate evaluation; training actors may fill the bounded queue
while the learner waits for evaluation. Queued actor interactions are not part
of the consumed-learner budget.

The optimizer uses the native per-minibatch linear decay schedule. The local
PyTorch port holds its schedule constant within each rollout. Native Seed
observation-normalization update timing also differs, as documented in
`protocol.md`. These choices are kept consistent across all Seed experiments.

## Supported subset

Both policy/value bodies must use equal-width layers, matching the native
agent's interface; nonuniform layer widths are rejected. Activations are tanh,
ReLU, ELU, and swish. The policy-loss choices exposed here are PPO and vanilla
policy gradient. Other configurable fields and group membership are defined in
`seed_ppo/config.py`; they do not claim support for every choice in the paper.

The batched CPU actor path treats time-limit termination as terminal. Setting
`handle_abandoned=True` is rejected: the native special abandonment path requires
one environment per actor and is outside this runner's supported subset.
There is no additional action repetition. Disabling observation normalization
also disables its clipping, retaining the local implementation's option meaning.

## Run records

Each fresh run records resolved configuration, requested and operative Gin,
commands, Python packages, source/Git information, native library hashes, actor
logs, learner progress, evaluation measurements, a final native model/checkpoint,
and terminal status. The supervisor owns worker process groups and cleans them
on completion, failures, timeouts, Ctrl-C, SIGTERM, and SIGHUP. Checkpoints support
separate evaluation; exact interrupted training resume is not implemented by
this wrapper.

Sources: [native learner](https://github.com/google-research/seed_rl/blob/master/agents/policy_gradient/learner.py),
[native agent](https://github.com/google-research/seed_rl/blob/master/agents/policy_gradient/modules/continuous_control_agent.py),
[native loss](https://github.com/google-research/seed_rl/blob/master/agents/policy_gradient/modules/generalized_onpolicy_loss.py),
[action distributions](https://github.com/google-research/seed_rl/blob/master/common/parametric_distribution.py).
