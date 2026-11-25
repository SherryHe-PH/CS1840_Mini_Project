#!/usr/bin/env bash
# Maintainer helper: add the experiment folder to a new fork checkout/branch.
# Users run directly from a published clone; this helper does not commit or push.
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_url="https://github.com/SherryHe-PH/CS1840_Mini_Project.git"
branch="experiments/seed-ppo-study"
base="origin/master"
target=""
dry_run=0
usage() {
    printf '%s\n' 'Usage: prepare_branch.sh --target NEW_DIRECTORY [--branch NAME] [--base origin/master] [--repo URL] [--dry-run]'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target|--branch|--base|--repo)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --target) target="$2" ;;
                --branch) branch="$2" ;;
                --base) base="$2" ;;
                --repo) repo_url="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$target" ]] || { usage >&2; exit 2; }
git check-ref-format --branch "$branch" >/dev/null
[[ ! -e "$target" && ! -L "$target" ]] || {
    printf 'Refusing an existing target directory: %s\n' "$target" >&2
    exit 2
}
python3 - "$bundle_dir" "$target" <<'PY'
from pathlib import Path
import sys
source, target = [Path(value).resolve() for value in sys.argv[1:]]
try:
    target.relative_to(source)
except ValueError:
    pass
else:
    raise SystemExit('The checkout target must be outside the bundle directory')
if not (source / 'README.md').is_file() or not (source / 'seed_ppo').is_dir():
    raise SystemExit('Incomplete bundle: README.md and seed_ppo/ are required')
PY
if [[ "$dry_run" -eq 1 ]]; then
    printf 'Would clone: '
    printf '%q ' git clone -- "$repo_url" "$target"
    printf '\nWould create branch: '
    printf '%q ' git -C "$target" switch --create "$branch" "$base"
    printf '\nWould copy %s into %s/seed_rl_experiments and prepend a root README quick start.\n' "$bundle_dir" "$target"
    printf 'Would preserve the existing README content, LICENSE, AUTHORS, and Seed RL implementation.\n'
    exit 0
fi
mkdir -p "$(dirname "$target")"
git clone -- "$repo_url" "$target"
if git -C "$target" show-ref --verify --quiet "refs/heads/$branch" || \
   git -C "$target" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    printf 'Branch already exists; choose a new --branch name: %s\n' "$branch" >&2
    exit 2
fi
git -C "$target" switch --create "$branch" "$base"
python3 - "$bundle_dir" "$target" <<'PY'
from pathlib import Path
import shutil
import sys

source, target = [Path(value).resolve() for value in sys.argv[1:]]
required = ['agents/policy_gradient/learner.py', 'mujoco/gin/ppo.gin',
            'grpc/grpc_cc.so', 'README.md', 'LICENSE', 'AUTHORS']
for name in required:
    if not (target / name).is_file():
        raise SystemExit('Checkout is missing the expected upstream file: ' + name)
experiment_dir = target / 'seed_rl_experiments'
if experiment_dir.exists() or experiment_dir.is_symlink():
    raise SystemExit('Refusing an existing experiment directory: ' + str(experiment_dir))
ignored_directories = {'.git', '.venv', '.pytest_cache', '__pycache__', 'runs', 'results',
                       '.runtime', 'plans', '.plans', 'build', 'dist', '.env', '.aws', '.ssh',
                       '.gnupg', '.codex', '.agents'}
files = []
for path in sorted(source.rglob('*')):
    relative = path.relative_to(source)
    if any(part in ignored_directories or part.endswith('.egg-info') for part in relative.parts):
        continue
    if path.name == '.env' or (path.name.startswith('.env.') and
                               path.name not in ('.env.example', '.env.sample')):
        continue
    if path.is_symlink():
        raise SystemExit('Bundle source must not contain symlinks: ' + str(relative))
    if not path.is_file() or path.name == '.DS_Store' or path.suffix in ('.pyc', '.zip'):
        continue
    files.append((path, relative))
for path, relative in files:
    destination = experiment_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
quick_start = '''## On-policy experiments on OOD

From this repository checkout, use an allocated Linux x86_64 CPU session to
set up and test the experiments. No previous `seed-mujoco` environment is needed.

```bash
cd seed_rl_experiments
bash scripts/setup_ood.sh
source scripts/ood_env.sh
```

Follow [the experiment README](seed_rl_experiments/README.md) for smoke tests,
full experiments, and reporting.

---

'''
readme = target / 'README.md'
readme.write_bytes(quick_start.encode('utf-8') + readme.read_bytes())
print('Copied %s files into seed_rl_experiments; upstream implementation and attribution retained.' % len(files))
PY
printf 'Prepared branch %s in %s\n' "$branch" "$target"
printf 'Review the checkout, then commit and push the branch. See seed_rl_experiments/docs/branch.md.\n'
