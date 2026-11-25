"""Run one Seed RL experiment with local-style configuration and output files."""
import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

from seed_ppo.config import load_config, parse_overrides, to_gin_bindings
from seed_ppo.io import provenance, write_json
from seed_ppo.processes import StopRequested, supervise
from seed_ppo.runtime import evaluation_schedule, seed_root


def worker_commands(config, config_file, run_dir, root, address):
    base = [sys.executable, "-u", "-m", "seed_ppo.worker", "--config", str(config_file),
            "--run-dir", str(run_dir), "--seed-repo", str(root), "--server-address", address]
    commands = {"learner": base + ["--role", "learner"]}
    for index in range(config.num_actors):
        commands["actor_%d" % index] = base + ["--role", "actor", "--actor-id", str(index)]
    return commands


def run(config, run_dir, root, timeout=None, port=0):
    config.validate()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(run_dir / "config.json", config.as_dict())
    write_json(run_dir / "summary.json", {"status": "running", "actual_step": 0})
    try:
        write_json(run_dir / "metadata.json", provenance(config, root))
        packages = sorted("%s==%s" % (item.metadata["Name"], item.version)
                          for item in metadata.distributions())
        (run_dir / "packages.txt").write_text("\n".join(packages) + "\n")
        (run_dir / "requested.gin").write_text("\n".join(to_gin_bindings(config)) + "\n")
        if not port:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        commands = worker_commands(config, run_dir / "config.json", run_dir, root,
                                   "127.0.0.1:%d" % port)
        write_json(run_dir / "commands.json", commands)
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": "", "TF_NUM_INTRAOP_THREADS": str(config.tensorflow_threads),
            "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        })
        own_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = own_root + os.pathsep + environment.get("PYTHONPATH", "")
        print("Running %s seed=%d with %d actor(s); output: %s" % (
            config.env_id, config.seed, config.num_actors, run_dir), flush=True)
        supervise(commands, run_dir, environment, root, timeout)
        summary = json.loads((run_dir / "summary.json").read_text())
        expected = int(math.ceil(config.total_timesteps / config.iteration_size)) * config.iteration_size
        if (summary.get("status") != "complete" or summary.get("actual_step") != expected
                or summary.get("evaluation_count") != len(evaluation_schedule(config))):
            raise RuntimeError("Learner exited without a complete training/evaluation protocol")
        print("COMPLETE: %s; score=%.2f" % (run_dir, summary["score_mean_evaluation_return"]), flush=True)
        return summary
    except BaseException as error:
        current = json.loads((run_dir / "summary.json").read_text())
        current.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, StopRequested)) else "failed",
                       error=repr(error), elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "summary.json", current)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_default.yaml"))
    parser.add_argument("--env-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--seed-repo", type=Path)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changes = parse_overrides(args.set)
    if args.env_id:
        changes["env_id"] = args.env_id
    if args.seed is not None:
        changes["seed"] = args.seed
    config = load_config(args.config, changes)
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.dry_run:
        print(json.dumps({"config": config.as_dict(), "gin_bindings": to_gin_bindings(config),
                          "evaluation_schedule": evaluation_schedule(config)}, indent=2))
        return 0
    try:
        run(config, args.output, seed_root(args.seed_repo), args.timeout, args.port)
    except StopRequested as error:
        print("Stopped by signal %d; workers cleaned up" % error.signum, file=sys.stderr)
        return 128 + error.signum
    except KeyboardInterrupt:
        print("Interrupted; workers cleaned up", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
