"""Check clone-and-run path discovery and setup without a previous environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from seed_ppo import runtime


BACKEND_FILES = ("agents/policy_gradient/learner.py", "mujoco/env.py", "grpc/grpc_cc.so")


def make_backend(root):
    for relative in BACKEND_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")


@pytest.mark.parametrize("nested", [False, True])
def test_runtime_discovers_complete_checkout_without_configuration(tmp_path, monkeypatch, nested):
    checkout = tmp_path / "Seed RL checkout"
    project = checkout / "seed_rl_experiments" if nested else checkout
    (project / "seed_ppo").mkdir(parents=True)
    make_backend(checkout)
    monkeypatch.setattr(runtime, "__file__", str(project / "seed_ppo/runtime.py"))
    monkeypatch.delenv("SEED_REPO_ROOT", raising=False)
    assert runtime.seed_root() == checkout


def test_runtime_does_not_hide_an_incomplete_explicit_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "complete"
    make_backend(checkout)
    monkeypatch.setenv("SEED_REPO_ROOT", str(checkout))
    assert runtime.seed_root() == checkout
    with pytest.raises(ValueError, match="Incomplete Seed RL checkout"):
        runtime.seed_root(tmp_path / "missing")
    (checkout / "grpc/grpc_cc.so").unlink()
    with pytest.raises(ValueError, match="grpc/grpc_cc.so"):
        runtime.seed_root()


def setup_fixture(tmp_path, nested=True, installed=()):
    """Replace downloads and installers, while running the actual shell scripts."""
    checkout = tmp_path / "Seed RL checkout"
    project = checkout / "seed_rl_experiments" if nested else checkout
    (project / "scripts").mkdir(parents=True)
    for name in ("setup_ood.sh", "ood_env.sh"):
        shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
    make_backend(checkout)
    miniforge = tmp_path / "miniforge"
    profile = miniforge / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text('''conda() {
    printf 'conda %s\\n' "$*" >> "$SEED_FIXTURE_LOG"
    case "$1" in
        env) cat "$SEED_FIXTURE_ENVS" ;;
        activate)
            export CONDA_PREFIX="$SEED_MINIFORGE/envs/$2"
            export PATH="$CONDA_PREFIX/bin:$PATH"
            ;;
        list) printf '# fixture conda packages\\n' ;;
    esac
}
''')
    (miniforge / "bin").mkdir()
    (miniforge / "bin/python").symlink_to(sys.executable)
    interpreter = miniforge / "envs/seed-experiments/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('''#!/bin/bash
printf 'python %s\\n' "$*" >> "$SEED_FIXTURE_LOG"
printf '%s\\n' "${SEED_REPO_ROOT:-}" > "$SEED_FIXTURE_ROOT"
if [[ "$*" == '-m pip freeze' ]]; then printf 'fixture==1\\n'; fi
''')
    interpreter.chmod(0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uname = binaries / "uname"
    uname.write_text('''#!/bin/bash
case "$1" in -s) printf 'Linux\\n' ;; -m) printf 'x86_64\\n' ;; esac
''')
    uname.chmod(0o755)
    for name in ("curl", "tar"):
        path = binaries / name
        path.write_text("#!/bin/bash\nexit 99\n")
        path.chmod(0o755)
    engine = tmp_path / "mujoco210"
    (engine / "bin").mkdir(parents=True)
    (engine / "bin/libmujoco210.so").write_text("fixture\n")
    import json
    environments = tmp_path / "environments.json"
    environments.write_text(json.dumps({"envs": [str(miniforge / "envs" / name) for name in installed]}))
    log = tmp_path / "calls.txt"
    captured_root = tmp_path / "root.txt"
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("SEED_REPO_ROOT", "SEED_CONDA_ENV")}
    environment.update({
        "PATH": str(binaries) + os.pathsep + environment["PATH"],
        "SEED_MINIFORGE": str(miniforge),
        "MUJOCO_PY_MUJOCO_PATH": str(engine),
        "SEED_FIXTURE_ENVS": str(environments),
        "SEED_FIXTURE_LOG": str(log),
        "SEED_FIXTURE_ROOT": str(captured_root),
    })
    return checkout, project, environment, log, captured_root


@pytest.mark.parametrize("nested", [False, True])
def test_setup_creates_environment_without_a_previous_seed_mujoco(tmp_path, nested):
    checkout, project, environment, log, captured_root = setup_fixture(tmp_path, nested=nested)
    result = subprocess.run(["bash", str(project / "scripts/setup_ood.sh")],
                            env=environment, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text()
    assert "conda create -y --override-channels -c conda-forge -n seed-experiments python=3.8 pip=24.0" in calls
    assert "--clone" not in calls
    assert "python -m pip install --no-deps --no-build-isolation -e " + str(project) in calls
    assert captured_root.read_text().strip() == str(checkout)
    assert "Setup complete" in result.stdout
    assert (project / ".runtime/packages.txt").read_text() == "fixture==1\n"


@pytest.mark.parametrize("source_exists", [False, True])
def test_optional_clone_reuses_source_or_falls_back_to_fresh(tmp_path, source_exists):
    installed = ("seed-mujoco",) if source_exists else ()
    _, project, environment, log, _ = setup_fixture(tmp_path, installed=installed)
    result = subprocess.run(["bash", str(project / "scripts/setup_ood.sh"), "--clone", "seed-mujoco"],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text()
    if source_exists:
        assert "conda create -y -n seed-experiments --clone seed-mujoco" in calls
    else:
        assert "conda create -y --override-channels" in calls
        assert "--clone" not in calls


def test_setup_reuses_existing_target_environment(tmp_path):
    _, project, environment, log, _ = setup_fixture(tmp_path, installed=("seed-experiments",))
    result = subprocess.run(["bash", str(project / "scripts/setup_ood.sh")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "conda create" not in log.read_text()


def test_setup_rejects_incomplete_checkout_before_installing(tmp_path):
    checkout, project, environment, log, _ = setup_fixture(tmp_path)
    (checkout / "grpc/grpc_cc.so").unlink()
    result = subprocess.run(["bash", str(project / "scripts/setup_ood.sh")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Clone the complete repository" in result.stderr
    assert not log.exists()
