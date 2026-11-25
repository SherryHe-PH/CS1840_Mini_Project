# Repository integration and publishing

`seed_rl_experiments/` belongs inside
[`SherryHe-PH/CS1840_Mini_Project`](https://github.com/SherryHe-PH/CS1840_Mini_Project).
Groupmates clone that repository and follow the quick start in
[`../README.md`](../README.md). They run setup and experiments from
`CS1840_Mini_Project/seed_rl_experiments/`.

## Layout

Keep the original Seed RL source at the repository root, with this experiment
folder alongside `agents/`, `common/`, `grpc/`, and `mujoco/`. The experiment
runtime and activation script find the parent Seed RL checkout automatically.
The adapter uses its TensorFlow learner, network, losses, and normalization.

A complete checkout needs the native `grpc/grpc_cc.so` library as well as the
Python source. Environment setup checks for that library and verifies that the
TensorFlow/custom-gRPC runtime and simulator can load together. Setup installs
the Python and MuJoCo dependencies; it does not compile the gRPC library.

The default Conda environment is `seed-experiments`. A new collaborator can
create it without any previous `seed-mujoco` environment. Generated runs,
results, manifests, native environment records, package metadata, build
artifacts, caches, and local environments should stay out of version control.

## Maintainer publishing

Publish the `seed_rl_experiments/` folder as a directory in the fork. Preserve
the original Seed RL implementation, root README content, `LICENSE`, and `AUTHORS`.
Review the files being committed, including the experiment folder's ignore
rules, and exclude generated output and local environment files.

The optional maintainer helper `scripts/prepare_branch.sh` prepares a new fork
checkout and branch with this nested layout. It copies the clean experiment
folder and prepends a short setup link to the root README, retaining the original
README text. It checks for the native gRPC library and never commits or pushes.
Groupmates using a published checkout do not need this helper.

Commit and push the integration through the repository's normal Git workflow.
These instructions do not establish that any commit or push has happened. If
the folder is published on a branch other than the default branch, update the
README clone command to select that branch until the change is merged.

Before directing collaborators to the quick start, check a clean clone using
that exact clone command. Confirm the nested folder and native gRPC library
are present, run setup, and run both smoke tests from the experiment folder.
The previous native Seed RL CPU tests passed on OOD, as recorded in
[`validation.md`](validation.md). Fresh setup and the configurable wrapper's
complete OOD path still need their own verification; they must not be described
as validated by those earlier tests.

## Experiment manifests

Generate manifests on OOD, where the experiments will run: their output paths
are absolute. Each saved manifest fixes its resolved settings and checksum.
Use a new manifest filename and output root for a new experiment. See the
README for local and Slurm execution commands.
