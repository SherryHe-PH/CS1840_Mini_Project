"""Validate frozen plans, fresh outputs, process cleanup, and branch preparation."""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("seed_experiment_launcher", ROOT / "scripts/run_experiments.py")
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def test_baseline_plan_has_correct_budgets_and_every_seed_once(tmp_path):
    plan = launcher.build_plan(ROOT / "configs/ppo_default.yaml", None,
                               list(launcher.ENVIRONMENT_BUDGETS), [0, 1, 2], tmp_path / "runs")
    assert len(plan["entries"]) == 15
    observed = {(entry["env_id"], entry["seed"]) for entry in plan["entries"]}
    assert observed == {(env, seed) for env in launcher.ENVIRONMENT_BUDGETS for seed in (0, 1, 2)}
    for entry in plan["entries"]:
        assert entry["config"]["total_timesteps"] == launcher.ENVIRONMENT_BUDGETS[entry["env_id"]]
        assert Path(entry["run_dir"]).parent.name == entry["env_id"]
    assert not (tmp_path / "runs").exists()


def test_manifest_freezes_grid_and_rejects_tampering(tmp_path):
    config_path = tmp_path / "baseline.yaml"
    shutil.copy2(ROOT / "configs/ppo_default.yaml", config_path)
    plan = launcher.build_plan(config_path, ROOT / "configs/groups/architecture.yaml",
                               ["Hopper-v2"], [0, 1], tmp_path / "runs")
    path = tmp_path / "plan.json"
    launcher.write_immutable_json(path, plan)
    config_path.write_text("learning_rate: 0.9\n")
    replay = launcher.read_manifest(path)
    assert replay == plan
    entry = replay["entries"][0]
    command = launcher.entry_command(entry, replay["output_root"], stage=True)
    frozen = json.loads(Path(command[command.index("--config") + 1]).read_text())
    assert frozen == entry["config"]
    assert frozen["learning_rate"] != 0.9
    plan["entries"][0]["config"]["seed"] = 99
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksum"):
        launcher.read_manifest(path)


def test_completed_skip_requires_identical_configuration(tmp_path):
    plan = launcher.build_plan(ROOT / "configs/ppo_default.yaml", None,
                               ["Hopper-v2"], [0], tmp_path / "runs")
    entry = plan["entries"][0]
    run = Path(entry["run_dir"])
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"status": "complete"}))
    (run / "config.json").write_text(json.dumps(entry["config"]))
    assert launcher.select_runs(plan, skip_complete=True) == []
    changed = dict(entry["config"], learning_rate=0.9)
    (run / "config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="exists"):
        launcher.select_runs(plan, skip_complete=True)


def test_dry_run_does_not_create_outputs_and_task_range_is_checked(tmp_path):
    output = tmp_path / "output"
    assert launcher.main(["--config", str(ROOT / "configs/smoke.yaml"), "--envs", "toy_env",
                          "--seeds", "0", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()
    plan = launcher.build_plan(ROOT / "configs/smoke.yaml", None,
                               ["toy_env"], [0], output)
    with pytest.raises(ValueError, match="Task ID"):
        launcher.select_runs(plan, task_id=-1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_cancellation_stops_worker_and_descendant_then_skips_queued_job(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
marker = Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
marker.write_text(str(os.getpid()))
if len(sys.argv) == 2:
    subprocess.Popen([sys.executable, __file__, str(marker.with_suffix('.child')), 'child'])
while True:
    if len(sys.argv) > 2:
        marker.with_suffix('.beat').write_text(str(time.monotonic_ns()))
    time.sleep(.05)
''')
    runner = tmp_path / "runner.py"
    runner.write_text('''
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('seed_experiment_launcher', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = Path(sys.argv[2])
jobs = [(str(seed), [sys.executable, str(root / 'worker.py'), str(root / ('worker%s.pid' % seed))],
         root / ('log%s.txt' % seed)) for seed in (0, 1)]
results, stopped = module.run_jobs(jobs, 1, grace_seconds=.2)
raise SystemExit(128 + stopped if stopped else 0)
''')
    process = subprocess.Popen([sys.executable, str(runner), str(ROOT / "scripts/run_experiments.py"), str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "worker0.beat").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError("Substitute worker did not start")
            time.sleep(.02)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 143, output
        assert not (tmp_path / "worker1.pid").exists()
        parent_pid = int((tmp_path / "worker0.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(parent_pid, 0)
        # A reparented child can remain a zombie briefly. Verify that it stopped
        # doing work without requiring process-table access on a sandboxed host.
        heartbeat = tmp_path / "worker0.beat"
        previous = heartbeat.stat().st_mtime_ns
        time.sleep(.15)
        assert heartbeat.stat().st_mtime_ns == previous
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        for marker in tmp_path.glob("worker0.*"):
            if marker.suffix in (".pid", ".child"):
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def make_upstream_and_bundle(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for name, content in {"README.md": "Original upstream README\n", "LICENSE": "Apache\n",
                          "AUTHORS": "Original authors\n", ".gitignore": "old-rule\n",
                          "agents/policy_gradient/learner.py": "# original learner\n",
                          "mujoco/gin/ppo.gin": "# original gin\n",
                          "grpc/grpc_cc.so": "native library fixture\n"}.items():
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "-c", "init.defaultBranch=master", "init", str(upstream)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    bundle = tmp_path / "bundle"
    (bundle / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/prepare_branch.sh", bundle / "scripts/prepare_branch.sh")
    (bundle / "seed_ppo").mkdir()
    (bundle / "seed_ppo/__init__.py").write_text("# adapter\n")
    (bundle / "README.md").write_text("Study README\n")
    (bundle / ".gitignore").write_text("runs/\n")
    return upstream, bundle


def test_branch_preparation_preserves_upstream_and_dry_run_is_read_only(tmp_path):
    upstream, bundle = make_upstream_and_bundle(tmp_path)
    for name in (".runtime/packages.txt", "plans/submitted.json", ".plans/frozen.json",
                 "seed_on_policy_study.egg-info/PKG-INFO", "build/generated.py", "dist/package.whl",
                 "runs/seed_0/log.txt", "results/aggregate.csv", "__pycache__/cached.pyc",
                 ".aws/credentials", ".ssh/id_private", ".codex/settings.json", ".env", ".env.production"):
        artifact = bundle / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("generated artifact\n")
    source_doc = bundle / "docs/plans.md"
    source_doc.parent.mkdir(parents=True, exist_ok=True)
    source_doc.write_text("Source documentation must be preserved.\n")
    target = tmp_path / "new checkout"
    command = ["bash", str(bundle / "scripts/prepare_branch.sh"), "--repo", str(upstream),
               "--target", str(target), "--branch", "experiments/test"]
    preview = subprocess.run(command + ["--dry-run"], capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert not target.exists()
    prepared = subprocess.run(command, capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stderr
    experiment = target / "seed_rl_experiments"
    assert (experiment / "README.md").read_text() == "Study README\n"
    root_readme = (target / "README.md").read_text()
    assert root_readme.endswith("Original upstream README\n")
    assert "[the experiment README](seed_rl_experiments/README.md)" in root_readme
    assert "cd seed_rl_experiments\nbash scripts/setup_ood.sh\nsource scripts/ood_env.sh" in root_readme
    assert (target / "agents/policy_gradient/learner.py").read_text() == "# original learner\n"
    assert (target / "grpc/grpc_cc.so").read_text() == "native library fixture\n"
    assert (target / "LICENSE").read_text() == "Apache\n"
    assert (target / "AUTHORS").read_text() == "Original authors\n"
    assert (experiment / "docs/plans.md").read_text() == source_doc.read_text()
    for name in (".runtime", "plans", ".plans", "seed_on_policy_study.egg-info", "build", "dist",
                 "runs", "results", "__pycache__", ".aws", ".ssh", ".codex", ".env", ".env.production"):
        assert not (experiment / name).exists()
    assert (target / ".gitignore").read_text() == "old-rule\n"
    assert (experiment / ".gitignore").read_text() == "runs/\n"
    current = subprocess.run(["git", "-C", str(target), "branch", "--show-current"], capture_output=True, text=True, check=True)
    assert current.stdout.strip() == "experiments/test"
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert "existing target" in again.stderr


def test_branch_preparation_requires_native_library_before_copying(tmp_path):
    upstream, bundle = make_upstream_and_bundle(tmp_path)
    subprocess.run(["git", "-C", str(upstream), "rm", "grpc/grpc_cc.so"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "remove native fixture"], check=True, capture_output=True)
    target = tmp_path / "incomplete checkout"
    result = subprocess.run(["bash", str(bundle / "scripts/prepare_branch.sh"), "--repo", str(upstream),
                             "--target", str(target), "--branch", "experiments/test"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "grpc/grpc_cc.so" in result.stderr
    assert not (target / "seed_rl_experiments").exists()
    assert (target / "README.md").read_text() == "Original upstream README\n"


def test_branch_preparation_refuses_existing_experiment_directory(tmp_path):
    upstream, bundle = make_upstream_and_bundle(tmp_path)
    existing = upstream / "seed_rl_experiments/README.md"
    existing.parent.mkdir()
    existing.write_text("Existing study\n")
    subprocess.run(["git", "-C", str(upstream), "add", "seed_rl_experiments"], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "existing experiment fixture"], check=True, capture_output=True)
    target = tmp_path / "existing study checkout"
    result = subprocess.run(["bash", str(bundle / "scripts/prepare_branch.sh"), "--repo", str(upstream),
                             "--target", str(target), "--branch", "experiments/test"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "existing experiment directory" in result.stderr
    assert (target / "seed_rl_experiments/README.md").read_text() == "Existing study\n"
    assert (target / "README.md").read_text() == "Original upstream README\n"
