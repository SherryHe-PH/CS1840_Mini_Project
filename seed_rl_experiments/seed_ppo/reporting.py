"""Validate and summarize measured Seed RL runs without importing TensorFlow."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
import tempfile
from typing import Any, Dict, List, Optional, Tuple


# Actor layout stays experimental: it can change collection and policy staleness.
EXECUTION_KEYS = {"seed", "output_dir", "run_dir", "device", "tensorflow_threads"}
PORT_BENCHMARK = "seed_mujoco21_port"
PROVENANCE_KEYS = ("framework", "packages", "source_sha256", "upstream_git_commit",
                   "native_libraries", "platform")


@dataclass
class RunResult:
    path: Path
    variant: str
    config: Dict[str, Any]
    evaluations: List[Dict[str, Any]]
    training_complete: bool
    metadata: Optional[Dict[str, Any]] = None

    @property
    def env_id(self):
        return self.config["env_id"]

    @property
    def seed(self):
        return self.config["seed"]

    @property
    def score(self):
        return mean(row["mean_return"] for row in self.evaluations)

    @property
    def final_return(self):
        return self.evaluations[-1]["mean_return"]

    @property
    def provenance_verified(self):
        return (self.metadata is not None
                and self.metadata.get("framework") == "seed_rl"
                and bool(self.metadata.get("upstream_git_commit")))


def _positive_integer(config, field, run_dir):
    value = config[field]
    if type(value) is not int or value <= 0:
        raise ValueError("%s: %s must be a positive integer" % (run_dir, field))
    return value


def evaluation_schedule(config):
    """Evaluate positive interval checkpoints and the requested endpoint."""
    interval, budget = config["eval_interval"], config["total_timesteps"]
    schedule = list(range(interval, budget + 1, interval))
    if not schedule or schedule[-1] != budget:
        schedule.append(budget)
    return schedule


def _infer_variant(run_dir, config, runs_root):
    explicit = config.get("experiment_name", config.get("variant"))
    if explicit and explicit != "baseline":
        return str(explicit)
    if runs_root is None:
        return str(explicit or "baseline")
    # Expected layouts: root/env/seed_N and root/variant/env/seed_N.
    relative = run_dir.relative_to(runs_root)
    variant_parts = relative.parts[:-2]
    return "/".join(variant_parts) if variant_parts else "baseline"


def load_run(run_dir, variant=None, runs_root=None):
    """Read one run, rejecting malformed values and inconsistent records."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("%s: config.json must contain an object" % run_dir)
    required = {"env_id", "seed", "total_timesteps", "eval_interval", "eval_episodes", "benchmark"}
    missing = required - config.keys()
    if missing:
        raise ValueError("%s: missing configuration fields %s" % (run_dir, sorted(missing)))
    if not isinstance(config["env_id"], str) or not config["env_id"]:
        raise ValueError("%s: env_id must be a nonempty string" % run_dir)
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("%s: seed must be a nonnegative integer" % run_dir)
    budget = _positive_integer(config, "total_timesteps", run_dir)
    _positive_integer(config, "eval_interval", run_dir)
    episodes = _positive_integer(config, "eval_episodes", run_dir)
    iteration = None
    if "iteration_size" in config:
        iteration = _positive_integer(config, "iteration_size", run_dir)

    evaluations = []
    required_columns = {"scheduled_step", "actual_step", "mean_return", "std_return", "episodes"}
    with (run_dir / "evaluation.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("%s: evaluation.csv must contain %s" % (run_dir, sorted(required_columns)))
        previous_step, previous_actual = -1, -1
        for raw in reader:
            row = {
                "scheduled_step": int(raw["scheduled_step"]),
                "actual_step": int(raw["actual_step"]),
                "mean_return": float(raw["mean_return"]),
                "std_return": float(raw["std_return"]),
                "episodes": int(raw["episodes"]),
            }
            if raw.get("evaluation_seed", "") != "":
                row["evaluation_seed"] = int(raw["evaluation_seed"])
            scheduled, actual = row["scheduled_step"], row["actual_step"]
            if scheduled < 0 or scheduled <= previous_step or actual < scheduled or actual < previous_actual:
                raise ValueError("%s: evaluation steps must increase and actual_step >= scheduled_step" % run_dir)
            previous_step, previous_actual = scheduled, actual
            if scheduled > budget:
                raise ValueError("%s: evaluation exceeds the requested training budget" % run_dir)
            if row["episodes"] != episodes:
                raise ValueError("%s: measured episode count differs from configuration" % run_dir)
            if row["std_return"] < 0 or not all(math.isfinite(row[key]) for key in ("mean_return", "std_return")):
                raise ValueError("%s: invalid or non-finite evaluation return" % run_dir)
            if scheduled > 0 and iteration is not None:
                expected_actual = ((scheduled + iteration - 1) // iteration) * iteration
                if actual != expected_actual:
                    raise ValueError("%s: actual_step does not match full-iteration checkpoint rounding" % run_dir)
            # Initial evaluations are informative, but never enter the score.
            if scheduled > 0:
                evaluations.append(row)
    if not evaluations:
        raise ValueError("%s: no measured evaluation checkpoints after step zero" % run_dir)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not isinstance(summary, dict):
        raise ValueError("%s: summary.json must contain an object" % run_dir)
    complete = ([row["scheduled_step"] for row in evaluations] == evaluation_schedule(config)
                and summary.get("status") == "complete")
    if "evaluation_count" in summary and summary["evaluation_count"] != len(evaluations):
        raise ValueError("%s: summary evaluation_count differs from evaluation.csv" % run_dir)
    if "actual_step" in summary:
        if type(summary["actual_step"]) is not int or summary["actual_step"] < evaluations[-1]["actual_step"]:
            raise ValueError("%s: summary actual_step precedes recorded evaluation" % run_dir)
        complete = complete and summary["actual_step"] >= budget

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("%s: metadata.json must contain an object" % run_dir)
        if not isinstance(metadata.get("packages"), dict) or not metadata["packages"] or not metadata.get("source_sha256"):
            raise ValueError("%s: metadata must record package versions and source_sha256" % run_dir)
        if metadata.get("framework") not in (None, "seed_rl"):
            raise ValueError("%s: metadata identifies a different framework" % run_dir)
    label = str(variant) if variant is not None else _infer_variant(
        run_dir, config, Path(runs_root) if runs_root is not None else None)
    return RunResult(run_dir, label, config, evaluations, complete, metadata)


def configuration_fingerprint(config):
    comparable = {key: value for key, value in config.items() if key not in EXECUTION_KEYS}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def protocol_status(runs):
    """Name completed port protocols honestly; distinguish every partial study."""
    config, reasons = runs[0].config, []
    if config["benchmark"] != PORT_BENCHMARK:
        reasons.append(str(config["benchmark"]))
    if len(runs) != 3:
        reasons.append("%s_of_3_seeds" % len(runs))
    environment = config["env_id"].split("-v")[0]
    expected_budget = {"Hopper": 1_000_000, "Walker2d": 1_000_000, "HalfCheetah": 1_000_000,
                       "Ant": 2_000_000, "Humanoid": 2_000_000}.get(environment)
    if expected_budget is None:
        reasons.append("outside_five_task_suite")
    elif config["total_timesteps"] != expected_budget:
        reasons.append("short_or_changed_budget")
    if config["eval_interval"] != 100_000:
        reasons.append("changed_eval_interval")
    if config["eval_episodes"] != 100:
        reasons.append("changed_eval_episode_count")
    if not all(run.training_complete for run in runs):
        reasons.append("incomplete_training_or_evaluation")
    if not all(run.provenance_verified for run in runs):
        reasons.append("provenance_not_verified")
    return "partial: " + "; ".join(reasons) if reasons else "complete_seed_mujoco21_port_protocol"


def aggregate_runs(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault((run.variant, run.env_id), []).append(run)
    rows = []
    for (variant, env_id), env_runs in sorted(grouped.items()):
        env_runs.sort(key=lambda run: run.seed)
        label = "%s/%s" % (variant, env_id)
        if len({run.seed for run in env_runs}) != len(env_runs):
            raise ValueError("%s: duplicate seeds" % label)
        if len({configuration_fingerprint(run.config) for run in env_runs}) != 1:
            raise ValueError("%s: mixed configurations; summarize each configuration separately" % label)
        provenance = {
            json.dumps({key: run.metadata.get(key) for key in PROVENANCE_KEYS}, sort_keys=True)
            for run in env_runs if run.metadata is not None
        }
        if len(provenance) > 1:
            raise ValueError("%s: mixed provenance; source and package versions must match" % label)
        timelines = {
            tuple((row["scheduled_step"], row["actual_step"]) for row in run.evaluations)
            for run in env_runs
        }
        if len(timelines) != 1:
            raise ValueError("%s: incompatible evaluation timelines across seeds" % label)
        scores, finals = [run.score for run in env_runs], [run.final_return for run in env_runs]
        rows.append({
            "variant": variant, "env_id": env_id, "seed_count": len(env_runs),
            "seeds": " ".join(str(run.seed) for run in env_runs),
            "median_seed_score": median(scores), "mean_seed_score": mean(scores),
            "std_seed_score": pstdev(scores), "median_final_return": median(finals),
            "mean_final_return": mean(finals), "std_final_return": pstdev(finals),
            "evaluation_checkpoints": len(env_runs[0].evaluations),
            "last_scheduled_step": env_runs[0].evaluations[-1]["scheduled_step"],
            "last_actual_step": env_runs[0].evaluations[-1]["actual_step"],
            "status": protocol_status(env_runs),
        })
    return rows, grouped


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table: %s" % path)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(grouped, output):
    """Show each variant's across-seed mean and population standard deviation."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seed-ppo-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting needs matplotlib; install it or pass --no-plots") from error
    environments = sorted({env_id for _, env_id in grouped})
    columns = min(3, len(environments))
    figure, axes = plt.subplots(math.ceil(len(environments) / columns), columns,
                                figsize=(5 * columns, 3.8 * math.ceil(len(environments) / columns)),
                                squeeze=False)
    for axis, env_id in zip(axes.flat, environments):
        for (variant, task), env_runs in sorted(grouped.items()):
            if task != env_id:
                continue
            values = list(zip(*[[row["mean_return"] for row in run.evaluations] for run in env_runs]))
            averages, deviations = [mean(v) for v in values], [pstdev(v) for v in values]
            steps = [row["actual_step"] for row in env_runs[0].evaluations]
            line, = axis.plot(steps, averages, label="%s (%s seeds)" % (variant, len(env_runs)))
            if len(env_runs) > 1:
                axis.fill_between(steps, [m - s for m, s in zip(averages, deviations)],
                                  [m + s for m, s in zip(averages, deviations)],
                                  color=line.get_color(), alpha=0.15)
        axis.set(title=env_id, xlabel="Training environment transitions",
                 ylabel="Undiscounted episode return")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(environments):]:
        axis.set_visible(False)
    figure.suptitle("Seed RL PPO evaluation curves (MuJoCo 2.1 port)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def summarize(runs_root, output_dir, plots=True):
    runs_root, output_dir = Path(runs_root).resolve(), Path(output_dir)
    directories = sorted(path.parent for path in runs_root.rglob("config.json")
                         if path.parent.name.startswith("seed_"))
    if not directories:
        raise ValueError("No seed_*/config.json files found under %s" % runs_root)
    runs = [load_run(path, runs_root=runs_root) for path in directories]
    aggregates, grouped = aggregate_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = [{
        "variant": run.variant, "env_id": run.env_id, "seed": run.seed,
        "mean_evaluation_return": run.score, "final_evaluation_return": run.final_return,
        "evaluation_checkpoints": len(run.evaluations),
        "last_scheduled_step": run.evaluations[-1]["scheduled_step"],
        "last_actual_step": run.evaluations[-1]["actual_step"],
        "training_complete": run.training_complete, "provenance_verified": run.provenance_verified,
        "run_directory": str(run.path),
    } for run in sorted(runs, key=lambda item: (item.variant, item.env_id, item.seed))]
    write_csv(output_dir / "per_seed.csv", per_seed)
    write_csv(output_dir / "aggregate.csv", aggregates)
    if plots:
        plot_curves(grouped, output_dir / "learning_curves.png")
    return aggregates
