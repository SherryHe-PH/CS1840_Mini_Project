# Validation status

The user reported passing native Seed RL tests on an x86_64 Harvard OOD CPU
node with TensorFlow 2.4.1, NumPy 1.19.5, Gym 0.17.3, `mujoco-py` 2.1.2.14,
and the MuJoCo 2.1 engine:

- Toy PPO: 256 consumed transitions, eight learner iterations, final checkpoint verified.
- HalfCheetah reset/step: 100 finite random simulation transitions.
- HalfCheetah PPO: 256 consumed transitions, eight learner iterations, final checkpoint verified.

Those tests validate the selected legacy runtime and native backend. They did
not use this new configurable experiment wrapper or its frozen evaluation
process. The quick-start smoke commands must validate the added integration on
OOD before a baseline or parameter study is interpreted as complete.

Local checks exercise configuration and all eight grids, cross-group rejection,
budget rounding, frozen-evaluation scheduling, final-checkpoint selection,
process failure/timeout cleanup, manifest replay/tamper detection, task arrays,
repository preparation using a local Git fixture, clone-and-run backend
discovery, setup without a previous Conda environment, and reporting/provenance
validation. The Slurm check also executes a spooled script copy with its
submission directory. Bash syntax and Python 3.8 grammar are also checked. The real
TensorFlow/custom-gRPC integration check is opt-in because the local host does
not have the Linux TensorFlow 2.4.1 runtime:

```bash
python -m pytest -q tests/study
SEED_RUN_INTEGRATION=1 python -m pytest -q tests/study/test_backend_integration.py
```

The updated bundle passed 65 local tests; the single native-runtime test was
skipped by default. The baseline and all eight group manifests were generated
and validated in dry runs. Branch preparation was exercised against a local
Git fixture, with no remote clone, commit, or push. The setup tests replace
downloads and package installers with fixtures; a fresh OOD installation and
both wrapper smoke tests still need to run on the cluster.

Full-budget learning curves, the 100-episode evaluation protocol at scale, and
site-specific Slurm submission have not been run by this bundle's author.
No baseline results or reference returns are included or fabricated.

The experiment tests live under `tests/study/`, so they do not replace or
automatically collect the fork's unrelated Atari/DMLab integration tests.
