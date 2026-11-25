"""Check measured scores, protocol labels, and experiment separation."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from seed_ppo.reporting import aggregate_runs, load_run, protocol_status, summarize


def make_run(root, seed, returns, variant=None, **overrides):
    base = Path(root) / variant if variant else Path(root)
    directory = base / "HalfCheetah-v2" / ("seed_%s" % seed)
    directory.mkdir(parents=True)
    config = dict(env_id="HalfCheetah-v2", seed=seed, total_timesteps=1_000_000,
                  eval_interval=100_000, eval_episodes=100, benchmark="seed_mujoco21_port",
                  iteration_size=2048, learning_rate=0.0003, num_actors=4, envs_per_actor=64)
    config.update(overrides)
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "metadata.json").write_text(json.dumps({
        "framework": "seed_rl", "packages": {"tensorflow": "2.4.1", "gym": "0.17.3"},
        "source_sha256": "a" * 64, "upstream_git_commit": "b" * 40,
    }))
    with (directory / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scheduled_step", "actual_step", "mean_return", "std_return", "episodes", "evaluation_seed"])
        writer.writerow([0, 0, 1_000_000, 0, config["eval_episodes"], 0])
        for index, value in enumerate(returns, 1):
            step = min(index * config["eval_interval"], config["total_timesteps"])
            actual = ((step + config["iteration_size"] - 1) // config["iteration_size"]) * config["iteration_size"]
            writer.writerow([step, actual, value, 3, config["eval_episodes"], seed * 100 + index])
    (directory / "summary.json").write_text(json.dumps({"status": "complete", "evaluation_count": len(returns)}))
    return load_run(directory, runs_root=Path(root))


def test_score_is_median_of_seed_means_not_final_returns(tmp_path):
    runs = [make_run(tmp_path, 0, [0, 10]), make_run(tmp_path, 1, [2, 2]),
            make_run(tmp_path, 2, [0, 100])]
    rows, _ = aggregate_runs(runs)
    assert [run.score for run in runs] == [5, 2, 50]
    assert rows[0]["median_seed_score"] == 5
    assert rows[0]["median_final_return"] == 10
    assert "incomplete_training_or_evaluation" in rows[0]["status"]


def test_complete_protocol_is_labeled_as_simulator_port(tmp_path):
    runs = [make_run(tmp_path, seed, [100] * 10) for seed in range(3)]
    assert protocol_status(runs) == "complete_seed_mujoco21_port_protocol"
    assert runs[0].evaluations[0]["actual_step"] == 100352
    assert runs[0].evaluations[-1]["actual_step"] == 1001472


def test_smoke_endpoint_and_episode_count_are_not_full_protocol(tmp_path):
    run = make_run(tmp_path, 0, [1, 2], benchmark="smoke", total_timesteps=150,
                   eval_interval=100, eval_episodes=2, iteration_size=32)
    assert run.training_complete
    assert [row["scheduled_step"] for row in run.evaluations] == [100, 150]
    status = protocol_status([run])
    assert status.startswith("partial:")
    assert "smoke" in status and "1_of_3_seeds" in status


def test_different_variants_are_compared_as_separate_configurations(tmp_path):
    baseline = make_run(tmp_path, 0, [1, 2], variant="baseline")
    changed = make_run(tmp_path, 0, [3, 4], variant="architecture_relu", activation="relu")
    rows, groups = aggregate_runs([baseline, changed])
    assert {row["variant"] for row in rows} == {"baseline", "architecture_relu"}
    assert len(groups) == 2


@pytest.mark.parametrize("changes", [{"learning_rate": 0.001}, {"num_actors": 8, "envs_per_actor": 32}])
def test_rejects_mixed_settings_or_actor_layout(tmp_path, changes):
    runs = [make_run(tmp_path, 0, [1, 2]), make_run(tmp_path, 1, [3, 4], **changes)]
    with pytest.raises(ValueError, match="mixed configurations"):
        aggregate_runs(runs)


@pytest.mark.parametrize("key,value", [
    ("packages", {"tensorflow": "different"}), ("source_sha256", "c" * 64),
    ("upstream_git_commit", "d" * 40), ("native_libraries", {"mujoco": "changed"}),
])
def test_rejects_mixed_provenance(tmp_path, key, value):
    runs = [make_run(tmp_path, 0, [1, 2]), make_run(tmp_path, 1, [3, 4])]
    runs[1].metadata[key] = value
    with pytest.raises(ValueError, match="mixed provenance"):
        aggregate_runs(runs)


def test_rejects_duplicate_seeds_and_different_timelines(tmp_path):
    run = make_run(tmp_path, 0, [1, 2])
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate_runs([run, run])
    shorter = make_run(tmp_path, 1, [1])
    with pytest.raises(ValueError, match="incompatible evaluation timelines"):
        aggregate_runs([run, shorter])


def test_missing_summary_or_metadata_is_partial_not_verified(tmp_path):
    run = make_run(tmp_path, 0, [100] * 10)
    (run.path / "summary.json").unlink()
    (run.path / "metadata.json").unlink()
    run = load_run(run.path)
    status = protocol_status([run])
    assert not run.training_complete and not run.provenance_verified
    assert "incomplete_training_or_evaluation" in status
    assert "provenance_not_verified" in status


@pytest.mark.parametrize("column,value,match", [
    ("mean_return", "nan", "non-finite"), ("episodes", "99", "episode count"),
    ("actual_step", "100000", "rounding"),
])
def test_rejects_invalid_measurements(tmp_path, column, value, match):
    run = make_run(tmp_path, 0, [1, 2])
    with (run.path / "evaluation.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, records = reader.fieldnames, list(reader)
    records[1][column] = value
    with (run.path / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match=match):
        load_run(run.path)


def test_summarize_recurses_variants_and_cli_can_skip_plots(tmp_path):
    make_run(tmp_path / "runs", 0, [1, 3], variant="baseline")
    make_run(tmp_path / "runs", 0, [2, 4], variant="architecture_relu", activation="relu")
    output = tmp_path / "results"
    rows = summarize(tmp_path / "runs", output, plots=False)
    assert len(rows) == 2
    assert (output / "aggregate.csv").is_file()
    assert (output / "per_seed.csv").is_file()
    assert not (output / "learning_curves.png").exists()
    script = Path(__file__).resolve().parents[2] / "scripts/summarize.py"
    result = subprocess.run([sys.executable, str(script), "--runs", str(tmp_path / "runs"),
                             "--output", str(output), "--no-plots"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "architecture_relu/HalfCheetah-v2" in result.stdout


def test_plot_contains_real_measurements(tmp_path):
    make_run(tmp_path / "runs", 0, [1, 3])
    output = tmp_path / "results"
    summarize(tmp_path / "runs", output)
    assert (output / "learning_curves.png").stat().st_size > 1000
