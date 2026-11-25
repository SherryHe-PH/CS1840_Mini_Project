"""Plan and launch Seed RL baseline/thematic experiments on one node.

Manifests embed resolved configurations, so a submitted array is independent of
later edits to YAML files. Python 3.8 compatible; no TensorFlow import is needed
to plan, inspect, or validate an experiment.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seed_ppo.config import (ENVIRONMENT_BUDGETS, config_from_dict, expand_group,
                             load_config, load_group)
import yaml

SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def parse_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Expected NAME=VALUE, got %r" % item)
        name, value = item.split("=", 1)
        if not name or name in overrides:
            raise ValueError("Empty or duplicate override: %r" % name)
        overrides[name] = yaml.safe_load(value)
    return overrides


def safe_component(value):
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)))


def build_plan(config_path, group_path, environments, seeds, output_root, overrides=None):
    if not environments or len(set(environments)) != len(environments):
        raise ValueError("Specify a nonempty list of distinct environments")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Specify distinct nonnegative integer seeds")
    output_root = Path(output_root).resolve()
    group = load_group(group_path) if group_path else None
    entries = []
    for env_id in environments:
        if not safe_component(env_id):
            raise ValueError("Environment ID cannot be used as a directory name: %r" % env_id)
        for seed in seeds:
            choices = dict(overrides or {}, env_id=env_id, seed=seed)
            base = load_config(config_path, choices)
            variants = expand_group(base, group) if group else [("baseline", base)]
            for variant, configuration in variants:
                if not safe_component(variant):
                    raise ValueError("Invalid variant directory name: %r" % variant)
                config = configuration.as_dict()
                run_dir = output_root
                if group:
                    run_dir = run_dir / variant
                run_dir = run_dir / env_id / ("seed_%s" % seed)
                entries.append({
                    "variant": variant, "env_id": env_id, "seed": seed,
                    "run_dir": str(run_dir), "config": config,
                    "config_sha256": digest(config),
                })
    plan = {"schema_version": SCHEMA_VERSION, "output_root": str(output_root),
            "group": group["group"] if group else "baseline", "entries": entries}
    plan["plan_sha256"] = digest(plan)
    return validate_plan(plan)


def validate_plan(plan):
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != digest(payload):
        raise ValueError("Manifest checksum mismatch; regenerate the plan instead of editing it")
    output_root = Path(plan["output_root"])
    if not output_root.is_absolute():
        raise ValueError("Manifest output_root must be absolute; generate it on the execution machine")
    if not isinstance(plan.get("entries"), list) or not plan["entries"]:
        raise ValueError("Manifest contains no runs")
    observed = set()
    for entry in plan["entries"]:
        config = entry["config"]
        resolved = config_from_dict(config).as_dict()
        if canonical(resolved) != canonical(config) or digest(config) != entry.get("config_sha256"):
            raise ValueError("Resolved configuration/checksum mismatch in manifest")
        if entry["env_id"] != config["env_id"] or entry["seed"] != config["seed"]:
            raise ValueError("Manifest labels disagree with the embedded configuration")
        if not safe_component(entry["variant"]) or not safe_component(entry["env_id"]):
            raise ValueError("Manifest contains an invalid directory component")
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            raise ValueError("Manifest run paths must be absolute")
        try:
            relative = run_dir.resolve().relative_to(output_root.resolve())
        except ValueError:
            raise ValueError("Manifest run directory is outside output_root")
        expected = Path(entry["env_id"]) / ("seed_%s" % entry["seed"])
        if plan["group"] != "baseline":
            expected = Path(entry["variant"]) / expected
        if relative != expected or str(run_dir.resolve()) in observed:
            raise ValueError("Manifest contains an inconsistent or duplicate run path")
        observed.add(str(run_dir.resolve()))
    return plan


def read_manifest(path):
    return validate_plan(json.loads(Path(path).read_text()))


def write_immutable_json(path, value):
    """Publish a complete file without overwriting a different existing plan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".plan-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if canonical(json.loads(path.read_text())) != canonical(value):
                raise ValueError("Refusing to overwrite a different plan: %s" % path)
    finally:
        os.unlink(temporary)


def completed(entry):
    directory = Path(entry["run_dir"])
    try:
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        return summary.get("status") == "complete" and canonical(config) == canonical(entry["config"])
    except (OSError, ValueError, TypeError):
        return False


def select_runs(plan, skip_complete=False, task_id=None):
    entries = plan["entries"]
    if task_id is not None:
        if not 0 <= task_id < len(entries):
            raise ValueError("Task ID must be between 0 and %s" % (len(entries) - 1))
        entries = [entries[task_id]]
    pending = []
    for entry in entries:
        if Path(entry["run_dir"]).exists():
            if skip_complete and completed(entry):
                print("Skipping completed %s" % entry["run_dir"], flush=True)
                continue
            raise ValueError("%s exists; use a fresh output root or --skip-complete for matching completed runs" % entry["run_dir"])
        pending.append(entry)
    return pending


def entry_command(entry, output_root, stage=False):
    config_file = Path(output_root) / ".plans" / (entry["config_sha256"] + ".json")
    if stage:
        write_immutable_json(config_file, entry["config"])
    return [sys.executable, "-m", "seed_ppo.train", "--config", str(config_file),
            "--output", entry["run_dir"]]


@dataclass
class JobResult:
    label: str
    returncode: object
    log_path: Path
    cancelled: bool = False


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def stop_worker(process, grace_seconds):
    signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    signal_group(process, signal.SIGKILL)
    process.wait(timeout=5)


def run_jobs(jobs, max_workers, grace_seconds=15):
    """Run (label, command, log_path) jobs and clean owned groups on signals."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("run_jobs must be called from the main thread")
    stopping = threading.Event()
    stopping_signal = [None]

    def request_stop(signum, frame):
        if stopping_signal[0] is None:
            stopping_signal[0] = signum
            stopping.set()

    def run_one(job):
        label, command, log_path = job
        if stopping.is_set():
            return JobResult(label, None, log_path, True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("x") as log:
                if stopping.is_set():
                    return JobResult(label, None, log_path, True)
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while True:
                        if stopping.is_set():
                            stop_worker(process, grace_seconds)
                            return JobResult(label, process.returncode, log_path, True)
                        try:
                            return JobResult(label, process.wait(timeout=0.1), log_path)
                        except subprocess.TimeoutExpired:
                            pass
                except BaseException:
                    stop_worker(process, grace_seconds)
                    raise
        except OSError as error:
            print("Could not start %s: %s" % (label, error), file=sys.stderr)
            return JobResult(label, 1, log_path)

    previous = {sig: signal.signal(sig, request_stop)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    results = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print("%s %s; log: %s" % (
                        "Stopped" if result.cancelled else "Finished" if result.returncode == 0 else "FAILED",
                        result.label, result.log_path), flush=True)
            except BaseException:
                stopping.set()
                raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return results, stopping_signal[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--envs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--count", action="store_true", help="Print the number of entries in an existing manifest")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print without training; save a manifest if explicitly requested")
    args = parser.parse_args(argv)
    try:
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        if args.manifest and args.manifest.exists():
            if any(value is not None for value in (args.config, args.group, args.envs, args.seeds, args.output)) or args.set:
                raise ValueError("An existing manifest is immutable; omit generation options or choose a new manifest filename")
            plan = read_manifest(args.manifest)
        else:
            if args.task_id is not None or args.count:
                raise ValueError("--task-id and --count require an existing --manifest")
            output = args.output or ROOT / "runs" / (args.group.stem if args.group else "baseline")
            plan = build_plan(args.config or ROOT / "configs/ppo_default.yaml", args.group,
                              args.envs or list(ENVIRONMENT_BUDGETS), args.seeds or [0, 1, 2],
                              output, parse_overrides(args.set))
        if args.count:
            print(len(plan["entries"]))
            return 0
        entries = select_runs(plan, args.skip_complete, args.task_id)
        # Validate every requested destination before publishing or launching.
        if args.manifest and not args.manifest.exists():
            write_immutable_json(args.manifest, plan)
        for entry in entries:
            print(shlex.join(entry_command(entry, plan["output_root"])), flush=True)
        print("Planned %s run(s)%s" % (len(entries),
              "; manifest: " + str(args.manifest) if args.manifest else ""), flush=True)
        if args.dry_run or not entries:
            return 0
        jobs = []
        for entry in entries:
            command = entry_command(entry, plan["output_root"], stage=True)
            label = "%s/%s/seed_%s" % (entry["variant"], entry["env_id"], entry["seed"])
            log_path = Path(plan["output_root"]) / "logs" / (label.replace("/", "_") + ".log")
            jobs.append((label, command, log_path))
        results, stopping_signal = run_jobs(jobs, args.jobs)
        if stopping_signal is not None:
            return 128 + stopping_signal
        return 1 if any(result.returncode for result in results) else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
