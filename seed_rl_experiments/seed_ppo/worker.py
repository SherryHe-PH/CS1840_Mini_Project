"""Native Seed RL learner/actor workers, each in its own Python process."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from seed_ppo.config import load_config
from seed_ppo.io import write_json
from seed_ppo.runtime import (build_learner, configure, environment_factory,
                              evaluation_schedule, seed_root)


def number(value):
    return value.numpy().item() if hasattr(value, "numpy") else value


def training_loop(model, config, run_dir, evaluate):
    """Drive native minimize calls; checkpoint/evaluate only at update boundaries."""
    started = time.monotonic()
    schedule = evaluation_schedule(config)
    next_evaluation, scores, actual_step = 0, [], 0
    checkpoint = None
    logger_started = False
    model.start_actor_loops()
    try:
        with (run_dir / "training.csv").open("w", newline="") as training_file, \
                (run_dir / "evaluation.csv").open("w", newline="") as evaluation_file:
            training_writer = csv.DictWriter(training_file, fieldnames=[
                "actual_step", "learning_rate", "elapsed_seconds"])
            evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=[
                "scheduled_step", "actual_step", "mean_return", "std_return", "episodes", "evaluation_seed"])
            training_writer.writeheader()
            evaluation_writer.writeheader()
            while actual_step < config.total_timesteps:
                model.minimize()
                actual_step = int(number(model.training_iterations)) * config.iteration_size
                if not logger_started:
                    model.logger.start(model.additional_logs)
                    logger_started = True
                learning_rate = float(number(model.learning_rate_fn(model.optimizer.iterations)))
                if not math.isfinite(learning_rate):
                    raise FloatingPointError("Non-finite optimizer learning rate")
                training_writer.writerow({"actual_step": actual_step, "learning_rate": learning_rate,
                                          "elapsed_seconds": time.monotonic() - started})
                training_file.flush()
                write_json(run_dir / "progress.json", {
                    "actual_step": actual_step, "evaluation_count": next_evaluation,
                    "elapsed_seconds": time.monotonic() - started})
                while next_evaluation < len(schedule) and actual_step >= schedule[next_evaluation]:
                    checkpoint = model.save_checkpoint()
                    scheduled_step = schedule[next_evaluation]
                    result = evaluate(checkpoint, scheduled_step, actual_step)
                    if (result["scheduled_step"] != scheduled_step or result["actual_step"] != actual_step
                            or result["episodes"] != config.eval_episodes):
                        raise ValueError("Frozen evaluation returned inconsistent protocol fields")
                    if not all(math.isfinite(result[key]) for key in ("mean_return", "std_return")):
                        raise FloatingPointError("Non-finite frozen evaluation return")
                    evaluation_writer.writerow(result)
                    evaluation_file.flush()
                    scores.append(result["mean_return"])
                    next_evaluation += 1
                    print("%s seed=%d step=%d eval=%.2f (%d episodes)" % (
                        config.env_id, config.seed, actual_step, result["mean_return"], result["episodes"]),
                        flush=True)
            model.logger.shutdown()
            logger_started = False
            model.finalize()
            # finalize() saves again; the native manager retains only one recent
            # checkpoint, so the pre-finalize evaluation prefix can be removed.
            checkpoint = model.manager.latest_checkpoint
            return {
                "status": "complete", "actual_step": actual_step,
                "training_iterations": int(number(model.training_iterations)),
                "evaluation_count": len(scores),
                "score_mean_evaluation_return": sum(scores) / len(scores),
                "final_evaluation_return": scores[-1], "checkpoint": checkpoint,
                "elapsed_seconds": time.monotonic() - started,
            }
    finally:
        if logger_started:
            model.logger.shutdown()


def learn(config, root, run_dir, address):
    model, unused_tf, gin = build_learner(config, root, address, run_dir / "native")
    model.prepare_for_run(logdir=str(run_dir / "native"), restore_latest_checkpoint=False)
    (run_dir / "resolved.gin").write_text(gin.operative_config_str())
    evaluation_directory = run_dir / "evaluations"
    evaluation_directory.mkdir()

    def frozen_evaluation(checkpoint, scheduled_step, actual_step):
        output = evaluation_directory / ("step_%d.json" % scheduled_step)
        command = [sys.executable, "-u", "-m", "seed_ppo.evaluate",
                   "--run-dir", str(run_dir), "--seed-repo", str(root),
                   "--checkpoint", str(checkpoint), "--scheduled-step", str(scheduled_step),
                   "--actual-step", str(actual_step), "--output", str(output)]
        with (evaluation_directory / ("step_%d.log" % scheduled_step)).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(600, config.eval_episodes * 60))
        if result.returncode:
            raise RuntimeError("Frozen evaluation failed; see evaluations/step_%d.log" % scheduled_step)
        return json.loads(output.read_text())

    try:
        return training_loop(model, config, run_dir, frozen_evaluation)
    finally:
        if model.server_initialized:
            model.stop_actor_loops()
        model.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("learner", "actor"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed-repo", type=Path, required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--actor-id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = load_config(args.config)
    root = seed_root(args.seed_repo)
    if args.role == "actor":
        seed = (config.seed + 1 + args.actor_id * 100_003) % (2 ** 31)
        configure(config, root, args.server_address, args.run_dir / "native", "actor",
                  actor_id=args.actor_id, random_seed=seed)
        from seed_rl.common import actor
        actor.actor_loop(environment_factory(config, config.seed))
        return
    try:
        result = learn(config, root, args.run_dir, args.server_address)
        write_json(args.run_dir / "summary.json", result)
    except BaseException as error:
        write_json(args.run_dir / "summary.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
