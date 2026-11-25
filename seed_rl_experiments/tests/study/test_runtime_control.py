"""Verify update-boundary evaluation and real subprocess cleanup without TF."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest

from seed_ppo.config import PPOConfig
from seed_ppo.processes import supervise
from seed_ppo.runtime import evaluation_schedule, expected_actual_step
from seed_ppo.worker import training_loop


class Counter:
    def __init__(self):
        self.value = 0

    def numpy(self):
        return np.int64(self.value)


class Logger:
    def start(self, callback):
        self.started = True

    def shutdown(self):
        self.stopped = True


class NativeLearnerDouble:
    def __init__(self):
        self.training_iterations = Counter()
        self.optimizer = type("Optimizer", (), {"iterations": Counter()})()
        self.logger = Logger()
        self.manager = type("Manager", (), {"latest_checkpoint": None})()
        self.finalized = False

    def start_actor_loops(self):
        self.accepting_actors = True

    def minimize(self):
        self.training_iterations.value += 1
        self.optimizer.iterations.value += 2

    def learning_rate_fn(self, step):
        return 0.0003

    def additional_logs(self):
        return {}

    def save_checkpoint(self):
        self.manager.latest_checkpoint = "checkpoint-%d" % self.training_iterations.value
        return self.manager.latest_checkpoint

    def finalize(self):
        self.finalized = True
        self.manager.latest_checkpoint = "final-checkpoint"


def small_config():
    return PPOConfig(benchmark="smoke", env_id="toy_env", num_actors=1,
                     envs_per_actor=4, iteration_size=32, minibatch_size=16,
                     num_epochs=1, total_timesteps=96, eval_interval=40,
                     eval_episodes=2).validate()


def test_training_loop_evaluates_frozen_checkpoints_at_first_crossing(tmp_path):
    model, config, calls = NativeLearnerDouble(), small_config(), []

    def evaluate(checkpoint, scheduled, actual):
        calls.append((checkpoint, scheduled, actual))
        assert model.training_iterations.value * 32 == actual
        return {"scheduled_step": scheduled, "actual_step": actual,
                "mean_return": float(len(calls) * 2 - 1), "std_return": 0.,
                "episodes": 2, "evaluation_seed": 123 + scheduled}

    result = training_loop(model, config, tmp_path, evaluate)
    assert calls == [("checkpoint-2", 40, 64), ("checkpoint-3", 80, 96),
                     ("checkpoint-3", 96, 96)]
    assert result["actual_step"] == 96 and result["evaluation_count"] == 3
    assert result["score_mean_evaluation_return"] == 3
    assert result["final_evaluation_return"] == 5
    assert result["checkpoint"] == "final-checkpoint"
    assert model.finalized and model.logger.stopped
    with (tmp_path / "evaluation.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert json.loads((tmp_path / "progress.json").read_text())["actual_step"] == 96


def test_bad_evaluation_cannot_finalize_a_successful_run(tmp_path):
    model, config = NativeLearnerDouble(), small_config()

    def wrong_count(checkpoint, scheduled, actual):
        return {"scheduled_step": scheduled, "actual_step": actual,
                "episodes": 1, "mean_return": 0., "std_return": 0.}

    with pytest.raises(ValueError, match="inconsistent"):
        training_loop(model, config, tmp_path, wrong_count)
    assert not model.finalized
    assert model.logger.stopped


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
@pytest.mark.parametrize("mode", ["complete", "failure", "timeout", "actor_failure"])
def test_supervisor_stops_and_reaps_workers(tmp_path, mode):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import os
from pathlib import Path
import signal
import sys
import time
directory, role, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(directory / (role + '.pid')).write_text(str(os.getpid()))
if role == 'actor' and mode == 'actor_failure':
    raise SystemExit(9)
if role == 'learner' and mode in ('complete', 'failure'):
    deadline = time.monotonic() + 4
    while not (directory / 'actor.pid').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    raise SystemExit(0 if mode == 'complete' else 7)
while True:
    time.sleep(.05)
''')
    commands = {role: [sys.executable, str(worker), str(tmp_path), role, mode]
                for role in ("learner", "actor")}
    try:
        if mode == "complete":
            supervise(commands, tmp_path, os.environ.copy(), tmp_path, timeout=5)
        else:
            expected = TimeoutError if mode == "timeout" else RuntimeError
            with pytest.raises(expected):
                supervise(commands, tmp_path, os.environ.copy(), tmp_path,
                          timeout=1 if mode == "timeout" else 5)
        for role in commands:
            with pytest.raises(ProcessLookupError):
                os.kill(int((tmp_path / (role + ".pid")).read_text()), 0)
    finally:
        for marker in tmp_path.glob("*.pid"):
            try:
                os.killpg(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_endpoint_rounding_for_full_baseline():
    config = PPOConfig().validate()
    assert evaluation_schedule(config)[0] == 100_000
    assert expected_actual_step(config, 100_000) == 100_352
    assert expected_actual_step(config, 1_000_000) == 1_001_472
