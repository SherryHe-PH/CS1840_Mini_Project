"""Run the batch shell with a substitute interpreter, without Slurm or TensorFlow."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/slurm_experiments.sbatch"


@pytest.mark.parametrize("spooled", [False, True])
def test_array_forwards_manifest_task_and_cpu_environment(tmp_path, spooled):
    capture = tmp_path / "args.txt"
    python = tmp_path / "fake python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SEED_CAPTURE"\n'
                      'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$OMP_NUM_THREADS" "$PWD" >> "$SEED_CAPTURE"\n'
                      'exit 13\n')
    python.chmod(0o755)
    setup = tmp_path / "environment setup.sh"
    setup.write_text('export OMP_NUM_THREADS=99\nexport SEED_PYTHON=/activation/default/python\n')
    manifest = tmp_path / "plan with spaces.json"
    environment = {**os.environ, "SLURM_ARRAY_TASK_ID": "17", "SEED_PYTHON": str(python),
                   "SEED_MANIFEST": str(manifest), "SEED_ENV_SCRIPT": str(setup),
                   "SEED_CAPTURE": str(capture)}
    environment.pop("SLURM_SUBMIT_DIR", None)
    script = SCRIPT
    if spooled:
        script = tmp_path / "slurm_script"
        script.write_text(SCRIPT.read_text())
        environment["SLURM_SUBMIT_DIR"] = str(ROOT)
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode == 13
    assert capture.read_text().splitlines() == [
        "scripts/run_experiments.py", "--manifest", str(manifest), "--task-id", "17",
        "--skip-complete", "", "1", str(ROOT),
    ]


@pytest.mark.parametrize("task", ["-1", "08", "invalid"])
def test_bad_array_id_rejected_before_environment_activation(tmp_path, task):
    result = subprocess.run(["bash", str(SCRIPT)],
                            env={**os.environ, "SLURM_ARRAY_TASK_ID": task,
                                 "SEED_ENV_SCRIPT": str(tmp_path / "absent")},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "Invalid Slurm array task ID" in result.stderr
