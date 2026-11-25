#!/usr/bin/env bash
# Source this file inside an allocated OOD/Slurm CPU session.
_seed_environment() {
    local project_dir miniforge desired_env repo_dir candidate
    project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return
    repo_dir=""
    for candidate in "$project_dir" "$project_dir/.."; do
        if [[ -f "$candidate/agents/policy_gradient/learner.py" &&
              -f "$candidate/mujoco/env.py" && -f "$candidate/grpc/grpc_cc.so" ]]; then
            repo_dir="$(cd "$candidate" && pwd)" || return
            break
        fi
    done
    if [[ -z "$repo_dir" ]]; then
        printf 'Incomplete Seed RL checkout. Clone the complete repository, including grpc/grpc_cc.so.\n' >&2
        return 1
    fi
    miniforge="${SEED_MINIFORGE:-$HOME/.local/seed-miniforge}"
    desired_env="${SEED_CONDA_ENV:-seed-experiments}"
    if [[ ! -r "$miniforge/etc/profile.d/conda.sh" ]]; then
        printf 'Missing Miniforge. Run bash scripts/setup_ood.sh first.\n' >&2
        return 1
    fi
    source "$miniforge/etc/profile.d/conda.sh" || return
    conda activate "$desired_env" || return
    export SEED_REPO_ROOT="$repo_dir"
    export SEED_PYTHON="$CONDA_PREFIX/bin/python"
    export CUDA_VISIBLE_DEVICES=""
    export TF_NUM_INTRAOP_THREADS="${SEED_TF_THREADS:-1}"
    export TF_NUM_INTEROP_THREADS=1
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export MUJOCO_PY_FORCE_CPU=1
    export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
    export CC="${CC:-/usr/bin/gcc}"
    export CXX="${CXX:-/usr/bin/g++}"
    export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
}
_seed_environment
_seed_environment_status=$?
unset -f _seed_environment
return "$_seed_environment_status" 2>/dev/null || exit "$_seed_environment_status"
