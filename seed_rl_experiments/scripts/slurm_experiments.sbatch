#!/usr/bin/env bash
#SBATCH --job-name=seed-ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out

# Supply --array=0-LAST%CONCURRENCY and site-specific account/partition at submission.
# Six CPUs cover the default four actors and one learner with room for orchestration.
set -euo pipefail
# Slurm executes a spooled copy of this script, so use the submission directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    project_dir="$SLURM_SUBMIT_DIR"
else
    project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$project_dir"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
if [[ ! "$task_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'Invalid Slurm array task ID: %s\n' "$task_id" >&2
    exit 2
fi
environment_script="${SEED_ENV_SCRIPT:-$project_dir/scripts/ood_env.sh}"
if [[ ! -r "$environment_script" ]]; then
    printf 'Missing environment setup: %s\n' "$environment_script" >&2
    exit 2
fi
# Keep an explicitly supplied interpreter even if the activation script exports
# its own default; native include/library paths still come from that script.
requested_python="${SEED_PYTHON:-}"
source "$environment_script"
python_binary="${requested_python:-${SEED_PYTHON:-$(command -v python)}}"
manifest="${SEED_MANIFEST:?Set SEED_MANIFEST to the generated experiment manifest}"
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$python_binary" scripts/run_experiments.py \
    --manifest "$manifest" --task-id "$task_id" --skip-complete
