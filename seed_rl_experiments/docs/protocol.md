# Experiment protocol

This branch uses Google Seed RL's TensorFlow on-policy agent and PPO learner.
The experiment wrapper supplies named configurations, CPU process orchestration,
independent evaluation, provenance, and reporting. Its baseline targets the
default settings in Appendix C, Table 2 of *What Matters in On-Policy
Reinforcement Learning?*; a thematic experiment varies the supported choices
within one group, with learning rate considered alongside that group. It does
not reproduce the paper's complete search over all implementation choices.

## Tasks, budget, and evaluation

| Item | Protocol |
| --- | --- |
| Tasks | Hopper-v2, Walker2d-v2, HalfCheetah-v2, Ant-v2, Humanoid-v2 |
| Training seeds | Three independent runs, default IDs 0, 1, 2 |
| Budget | 1M training transitions for Hopper, Walker2d, HalfCheetah; 2M for Ant, Humanoid |
| Checkpoints | Every 100k requested training transitions, plus the requested endpoint |
| Evaluation | 100 episodes from a frozen stochastic policy, with frozen normalization statistics |
| Randomness | An independently seeded evaluation process; evaluation draws do not use learner or actor RNGs |
| Return | Raw undiscounted episode return; evaluation interaction is excluded from the training budget |
| Seed score | Arithmetic mean of the post-zero checkpoint evaluation means |
| Configuration score | Median of the three seed scores |

The final checkpoint return is a separate metric. For example, seed checkpoint
means `[0, 10]`, `[2, 2]`, and `[0, 100]` give seed scores 5, 2, and 50. The
configuration score is 5; its median final return is 10. Reporting does not
substitute final return for the paper's learning-curve score, select the best
seed, or invent reference scores from figures in the paper.

Seed RL consumes complete learner batches. An evaluation requested at transition
`t` runs after the first learner iteration crossing `t`; with iteration size
2,048, the first 100k evaluation occurs at 100,352 consumed training transitions.
A 1M requested budget finishes at 1,001,472. Both scheduled and actual counts are
saved. These are consumed learner transitions; the actor queues may contain
additional environment interaction. Actors can continue collecting while the
frozen evaluation process runs, until the native queue blocks them. That
asynchronous collection is retained from Seed RL; evaluation timing is not a
claim of identical random-number consumption to the local synchronous port.
Runtime smoke tests use smaller budgets,
batches, or episode counts and are explicitly labeled `smoke`.

## Interpreting the simulator port

The tested OOD runtime uses Gym 0.17.3 with the MuJoCo 2.1 engine and
`mujoco-py` 2.1.2.14. The paper used Gym v1 tasks and MuJoCo 2.0. Physics and task
versions therefore differ. A completed study is labeled
`complete_seed_mujoco21_port_protocol`, never an exact numerical replication of
the original benchmark. The stock Seed RL PPO example is also not identical to
the paper's default table; this branch keeps its baseline configuration explicit.

The independent local PyTorch implementation and this Seed RL backend are
separate implementations. In particular, the local port updates observation
statistics after environment steps and caches normalized observations. The Seed
RL learner updates observation statistics at the start of its training batch and
normalizes raw observations through the agent. The Seed RL actor/learner queues
also differ from the local synchronous collection loop. Compare thematic choices
within this backend; comparisons between backends require a separate controlled
study. Equal numeric settings do not establish identical learning dynamics.

## Saved records and validation

Each seed directory contains `config.json`, `metadata.json`, `summary.json`, and
`evaluation.csv`, with native Seed RL checkpoints/logs alongside those records.
Native checkpoint retention keeps the latest checkpoint rather than promising
that every evaluated policy remains available for later re-evaluation. The
recorded evaluation measurements remain in CSV and per-checkpoint JSON files.
The reporting command reads measured evaluations, excludes step-zero scores, and
emits `per_seed.csv`, `aggregate.csv`, and optionally `learning_curves.png`.
Plot lines show the across-seed mean; shaded bands show one population standard
deviation across seeds, not a confidence interval.

Results are grouped by variant and task. The reporter rejects duplicate seeds,
mixed algorithm settings, changed actor layout, incompatible evaluation
timelines, non-finite returns, inconsistent episode counts, and differing source
or dependency provenance within a configuration. Different variants remain
separate rows and plot lines. Missing provenance, fewer than three seeds,
unfinished evaluation schedules, or altered budgets/episode counts receive a
`partial:` status rather than a full-protocol label.

The metadata comparison includes the backend identity, source hash, upstream Git
commit, and package versions. Native library hashes and platform records, when
present, must also agree. Timestamps and process IDs may differ. Preserve
the resolved Conda environment separately when reproducing the CPU graphics
stack: Python package records alone do not describe native Mesa libraries.

Sources: [paper](https://arxiv.org/abs/2006.05990),
[Seed RL PPO configuration](https://github.com/google-research/seed_rl/blob/master/mujoco/gin/ppo.gin),
[Seed RL learner normalization update](https://github.com/google-research/seed_rl/blob/master/agents/policy_gradient/learner.py#L794-L805).
