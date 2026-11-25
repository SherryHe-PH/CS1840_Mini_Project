"""Small adapters around the original Seed RL learner, losses, and agent.

TensorFlow is imported lazily so planning and reporting work without its runtime.
"""
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import random
import sys

_CONFIGURED = False


def seed_root(explicit=None):
    """Find the backend beside this project or in its containing repository."""
    selected = explicit or os.environ.get("SEED_REPO_ROOT")
    project = Path(__file__).resolve().parents[1]
    candidates = [Path(selected).resolve()] if selected else [project, project.parent]
    required = ("agents/policy_gradient/learner.py", "mujoco/env.py", "grpc/grpc_cc.so")
    for root in candidates:
        if all((root / relative).is_file() for relative in required):
            return root
    checked = ", ".join(str(root) for root in candidates)
    raise ValueError("Incomplete Seed RL checkout (checked %s). Clone the complete "
                     "repository, including grpc/grpc_cc.so, then run from "
                     "seed_rl_experiments/." % checked)


def import_seed(root):
    """Resolve the upstream package even when the Git checkout has another name."""
    root = Path(root).resolve()
    if "seed_rl" in sys.modules:
        loaded = Path(sys.modules["seed_rl"].__file__).resolve().parent
        if loaded != root:
            raise RuntimeError("Another Seed RL checkout is already imported: " + str(loaded))
        return
    spec = importlib.util.spec_from_file_location(
        "seed_rl", str(root / "__init__.py"), submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Seed RL from " + str(root))
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_rl"] = module
    spec.loader.exec_module(module)


def configure(config, root, address, native_dir, role, actor_id=0, random_seed=None):
    """Register native flags/Gin configurables and set per-process random state."""
    global _CONFIGURED
    if _CONFIGURED:
        raise RuntimeError("Build each learner/evaluator in a fresh worker process")
    _CONFIGURED = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(config.tensorflow_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import_seed(root)
    import gin
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(config.tensorflow_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # This import registers the upstream Gin externals and absl flags.
    from seed_rl.mujoco import ppo_main
    from absl import flags
    from seed_ppo.config import to_gin_bindings

    # Gin 0.4 does not expose every TensorFlow activation. Use a unique alias
    # instead of querying registration through a newer Gin convenience API.
    gin.external_configurable(tf.nn.swish, name="study_swish", module="seed_ppo")
    seed = config.seed if random_seed is None else random_seed
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    tf.random.set_seed(seed)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([
            "seed_ppo", "--run_mode=" + ("actor" if role == "actor" else "learner"),
            "--env_name=" + config.env_id, "--server_address=" + address,
            "--logdir=" + str(native_dir), "--num_training_tpus=0",
            "--num_envs=" + str(config.num_envs),
            "--env_batch_size=" + str(config.envs_per_actor),
            "--task=" + str(actor_id), "--num_action_repeats=1",
            "--num_actors_with_summaries=0", "--logtostderr",
        ])
    gin.clear_config()
    gin.parse_config_files_and_bindings([], to_gin_bindings(config))
    return tf, gin, ppo_main


def environment_factory(config, base_seed):
    import gym
    import numpy as np
    from seed_rl.mujoco import env as native_env

    class ClipPolicyActions(gym.Wrapper):
        """Only the simulator copy is clipped; native unrolls retain raw samples."""
        def step(self, action):
            action = np.asarray(action, dtype=self.action_space.dtype)
            return self.env.step(np.clip(action, -1.0, 1.0))

    def create_environment(task, unused_config):
        environment = native_env.create_environment(config.env_id)
        # ToyEnv uses NumPy's process-global RNG, already seeded by configure().
        if not config.env_id.startswith("toy_"):
            environment.seed((base_seed + int(task)) % (2 ** 31))
        return ClipPolicyActions(environment)
    return create_environment


@contextmanager
def distribution_factory(config):
    """Temporarily add an unsquashed Gaussian option to Seed RL's factory.

    The upstream learner constructs its own distribution through a module-level
    factory. The replacement is confined to construction in this worker process;
    all created agents/losses then retain the resulting distribution normally.
    """
    import gym
    import tensorflow as tf
    import tensorflow_probability as tfp
    from seed_rl.common import parametric_distribution as native

    original = native.get_parametric_distribution_for_action_space
    std_fn = native.safe_exp_std_fn(config.initial_std, config.min_std)

    def create(action_space, continuous_config=None):
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("This study supports continuous Box action spaces only")
        native.check_box_space(action_space)
        dimensions = action_space.shape[0]
        if config.action_transform == "tanh":
            return native.normal_tanh_distribution(dimensions, std_fn)

        def raw_gaussian(parameters):
            mean, scale = tf.split(parameters, 2, axis=-1)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(mean, std_fn(scale)), reinterpreted_batch_ndims=1)
        return native.ParametricDistribution(2 * dimensions, raw_gaussian)

    native.get_parametric_distribution_for_action_space = create
    try:
        yield
    finally:
        native.get_parametric_distribution_for_action_space = original


def build_learner(config, root, address, native_dir, role="learner", random_seed=None):
    tf, gin, ppo_main = configure(config, root, address, native_dir, role,
                                  random_seed=random_seed)
    from seed_rl.agents.policy_gradient import learner, learner_config
    from seed_rl.common import utils

    native_config = learner_config.TrainingConfig(
        env_name=config.env_id, num_envs=config.num_envs, num_eval_envs=0,
        env_batch_size=config.envs_per_actor,
        inference_batch_size=config.envs_per_actor,
        unroll_length=config.unroll_length,
        batch_size=config.minibatch_size // config.unroll_length,
        batches_per_step=config.iteration_size // config.minibatch_size,
        epochs_per_step=config.num_epochs, batch_mode=config.batch_mode,
        total_environment_frames=config.total_timesteps,
        server_address=address,
        clip_norm=config.max_grad_norm, prefetch_batches=False,
        block_inference_on_training=True, print_episode_summaries=False,
        log_episode_frequency=1, summary_flush_seconds=10,
        num_action_repeats=1, num_checkpoints=0, num_saved_models=0,
    )

    def create_optimizer(final_iteration):
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            config.learning_rate, max(1, final_iteration),
            config.learning_rate * config.learning_rate_end_fraction)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=schedule, beta_1=config.adam_beta1,
            beta_2=config.adam_beta2, epsilon=config.adam_epsilon)
        return optimizer, schedule

    settings = utils.init_learner_multi_host(0)
    with distribution_factory(config):
        instance = learner.Learner(
            environment_factory(config, config.seed), ppo_main.create_agent,
            create_optimizer, settings=settings, config=native_config,
        )
    return instance, tf, gin


def evaluation_schedule(config):
    schedule = list(range(config.eval_interval, config.total_timesteps + 1, config.eval_interval))
    if not schedule or schedule[-1] != config.total_timesteps:
        schedule.append(config.total_timesteps)
    return schedule


def expected_actual_step(config, scheduled_step):
    return int(math.ceil(scheduled_step / config.iteration_size)) * config.iteration_size
