#!/usr/bin/env bash
# Rootless setup for a complete Seed RL repository checkout.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name=seed-experiments
clone_name=""
dry_run=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) environment_name="${2:?Missing --env value}"; shift 2 ;;
        --clone) clone_name="${2:?Missing --clone value}"; shift 2 ;;
        --fresh) clone_name=""; shift ;;
        --dry-run) dry_run=true; shift ;;
        *) printf 'Usage: bash scripts/setup_ood.sh [--env NAME] [--clone NAME|--fresh] [--dry-run]\n' >&2; exit 2 ;;
    esac
done
if [[ ! "$environment_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid environment name\n' >&2; exit 2
fi
if [[ -n "$clone_name" && "$environment_name" == "$clone_name" ]]; then
    printf 'Use a new --env name so the validated source environment stays reproducible.\n' >&2; exit 2
fi
miniforge="${SEED_MINIFORGE:-$HOME/.local/seed-miniforge}"
if "$dry_run"; then
    printf 'Miniforge: %s\nEnvironment: %s\nClone if available: %s\n' "$miniforge" "$environment_name" "${clone_name:-none}"
    printf 'Would install Python 3.8, mesalib<25.1, libgl-devel, patchelf and requirements-ood.txt, then check the native runtime.\n'
    exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    printf 'The validated runtime targets Linux x86_64.\n' >&2; exit 2;
}
repo_dir=""
for candidate in "$project_dir" "$project_dir/.."; do
    if [[ -f "$candidate/agents/policy_gradient/learner.py" &&
          -f "$candidate/mujoco/env.py" && -f "$candidate/grpc/grpc_cc.so" ]]; then
        repo_dir="$(cd "$candidate" && pwd)"
        break
    fi
done
[[ -n "$repo_dir" ]] || {
    printf 'Incomplete Seed RL checkout: the learner, mujoco/env.py, and grpc/grpc_cc.so are required.\nClone the complete repository, then run setup from seed_rl_experiments/.\n' >&2
    exit 2
}
if [[ ! -r "$miniforge/etc/profile.d/conda.sh" ]]; then
    installer="$(mktemp /tmp/seed-miniforge.XXXXXX.sh)"
    trap 'rm -f "$installer"' EXIT
    curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$installer"
    mkdir -p "$(dirname "$miniforge")"
    bash "$installer" -b -p "$miniforge"
fi
source "$miniforge/etc/profile.d/conda.sh"
has_environment() {
    conda env list --json | "$miniforge/bin/python" -c \
      'import json, pathlib, sys; name=sys.argv[1]; sys.exit(0 if any(pathlib.Path(p).name == name for p in json.load(sys.stdin)["envs"]) else 1)' "$1"
}
if ! has_environment "$environment_name"; then
    if [[ -n "$clone_name" ]] && has_environment "$clone_name"; then
        conda create -y -n "$environment_name" --clone "$clone_name"
    else
        conda create -y --override-channels -c conda-forge -n "$environment_name" python=3.8 pip=24.0
    fi
fi
conda activate "$environment_name"
python -c 'import sys; assert sys.version_info[:2] == (3, 8), "Python 3.8 is required"'
conda install -y --freeze-installed --override-channels -c conda-forge \
    'mesalib<25.1' libgl-devel patchelf
engine_path="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
if [[ ! -f "$engine_path/bin/libmujoco210.so" ]]; then
    if [[ -e "$engine_path" ]]; then
        printf 'Engine directory exists but is incomplete: %s\n' "$engine_path" >&2
        exit 2
    fi
    archive="$(mktemp /tmp/seed-mujoco210.XXXXXX.tar.gz)"
    curl -fL https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -o "$archive"
    mkdir -p "$engine_path"
    tar -xzf "$archive" -C "$engine_path" --strip-components=1
    rm -f "$archive"
fi
export SEED_CONDA_ENV="$environment_name"
source "$project_dir/scripts/ood_env.sh"
python -m pip install 'pip==24.0' 'setuptools==65.5.0' 'wheel==0.38.4'
python -m pip install --only-binary=mujoco-py -r "$project_dir/requirements-ood.txt"
python -m pip install --no-deps --no-build-isolation -e "$project_dir"
python -m pip check
python "$project_dir/scripts/check_runtime.py" --env-id HalfCheetah-v2
mkdir -p "$project_dir/.runtime"
python -m pip freeze > "$project_dir/.runtime/packages.txt"
conda list --explicit > "$project_dir/.runtime/conda-explicit.txt"
printf '\nSetup complete. Activate in each new terminal with:\n'
printf 'export SEED_CONDA_ENV=%q\nsource scripts/ood_env.sh\n' "$environment_name"
