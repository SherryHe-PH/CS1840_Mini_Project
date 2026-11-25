"""Evaluate a frozen native checkpoint in a separate stochastic RNG process."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from seed_ppo.config import load_config
from seed_ppo.io import write_json
from seed_ppo.runtime import build_learner, environment_factory, seed_root


def evaluate_checkpoint(config, root, checkpoint, scheduled_step, actual_step):
    import numpy as np
    seed = (10_000_000 + config.seed * 100_003 + scheduled_step) % (2 ** 31)
    # Construction creates the same native model/loss/optimizer object paths as
    # training. No server is started and no normalizer update is performed.
    model, tf, unused_gin = build_learner(
        config, root, "127.0.0.1:0", "unused-evaluation-logdir",
        role="evaluate", random_seed=seed)
    checkpoint_object = tf.train.Checkpoint(
        loss_fn=model.loss_fn, optimizer=model.optimizer,
        training_iterations=model.training_iterations)
    checkpoint_object.restore(str(checkpoint)).assert_existing_objects_matched()
    restored_step = int(model.training_iterations.numpy()) * config.iteration_size
    if restored_step != actual_step:
        raise ValueError("Checkpoint step %d does not match requested actual_step %d" %
                         (restored_step, actual_step))
    count = min(config.eval_num_envs, config.eval_episodes)
    create_env = environment_factory(config, seed)
    environments = []
    try:
        for index in range(count):
            environments.append(create_env(index, None))
        observations = np.asarray([environment.reset() for environment in environments], dtype=np.float32)
        quotas = np.full(count, config.eval_episodes // count, dtype=int)
        quotas[:config.eval_episodes % count] += 1
        completed = np.zeros(count, dtype=int)
        episode_returns = np.zeros(count, dtype=np.float64)
        returns = []
        from seed_rl.common import utils
        zeros = tf.zeros([count], tf.float32)
        previous_actions = tf.zeros([count] + list(model.env.action_space.shape), tf.float32)
        state = model.training_agent.initial_state(count)

        @tf.function
        def sample_actions(inputs):
            env_outputs = utils.EnvOutput(
                zeros, tf.zeros([count], tf.bool), inputs,
                tf.zeros([count], tf.bool), tf.zeros([count], tf.int32))
            outputs, unused_state = model.training_agent(
                (previous_actions, env_outputs), state, is_training=False)
            return outputs.action

        while np.any(completed < quotas):
            actions = sample_actions(tf.convert_to_tensor(observations, tf.float32)).numpy()
            for index, environment in enumerate(environments):
                observation, reward, done, unused_info = environment.step(actions[index])
                if not np.isfinite(reward) or not np.isfinite(observation).all():
                    raise FloatingPointError("Non-finite evaluation observation or reward")
                episode_returns[index] += float(reward)
                if done:
                    if completed[index] < quotas[index]:
                        returns.append(float(episode_returns[index]))
                        completed[index] += 1
                    episode_returns[index] = 0
                    observation = environment.reset()
                observations[index] = observation
        return {
            "scheduled_step": scheduled_step, "actual_step": actual_step,
            "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "episodes": len(returns), "evaluation_seed": seed,
        }
    finally:
        for environment in environments:
            environment.close()
        model.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scheduled-step", type=int, required=True)
    parser.add_argument("--actual-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-repo", type=Path)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    config = load_config(args.run_dir / "config.json")
    if args.episodes is not None:
        config = replace(config, eval_episodes=args.episodes).validate()
    result = evaluate_checkpoint(config, seed_root(args.seed_repo), args.checkpoint,
                                 args.scheduled_step, args.actual_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
