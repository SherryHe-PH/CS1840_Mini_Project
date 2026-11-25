"""Atomic run records and reproducible source/runtime identification."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root, arguments):
    try:
        result = subprocess.run(["git", "-C", str(root)] + arguments,
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(config, root):
    own_root = Path(__file__).resolve().parents[1]
    sources = []
    for directory in ("agents", "common", "mujoco", "grpc"):
        sources.extend((root / directory).rglob("*.py"))
    sources.append(root / "grpc/grpc_cc.so")
    sources.extend((own_root / "seed_ppo").glob("*.py"))
    digest = hashlib.sha256()
    seen = set()
    for path in sorted(sources, key=lambda item: str(item)):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        relative = (path.relative_to(root) if root in path.parents else
                    Path("study_adapter") / path.relative_to(own_root))
        digest.update(str(relative).encode())
        digest.update(file_hash(path).encode())
    packages = {}
    for package in ("tensorflow", "tensorflow-probability", "numpy", "gym", "mujoco-py",
                    "gin-config", "PyYAML", "Cython"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    native = {}
    engine = Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210")))
    for name, path in (
        ("mujoco_engine", engine / "bin/libmujoco210.so"),
        ("osmesa", Path(sys.prefix) / "lib/libOSMesa.so"),
    ):
        if path.is_file():
            native[name] = file_hash(path)
    return {
        "framework": "seed_rl", "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages,
        "source_sha256": digest.hexdigest(), "native_libraries": native,
        "upstream_git_commit": git_output(root, ["rev-parse", "HEAD"]),
        "upstream_git_status": git_output(root, ["status", "--short"]),
        "seed_repo": str(root), "command": sys.argv,
        "comparison": "Gym v2 / MuJoCo 2.1 runtime port; not the original Gym v1 / MuJoCo 2.0 benchmark",
        "budget": "transitions consumed by the native learner, rounded up to a full iteration; actors can collect ahead",
        "evaluation": "frozen checkpoint, separate independently seeded process, raw stochastic episode returns",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
