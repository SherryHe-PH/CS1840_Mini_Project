"""Paper-baseline configuration and bounded thematic grids (Python 3.8)."""
from dataclasses import asdict, dataclass, fields
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


ENVIRONMENT_BUDGETS = {
    "Hopper-v2": 1_000_000,
    "Walker2d-v2": 1_000_000,
    "HalfCheetah-v2": 1_000_000,
    "Ant-v2": 2_000_000,
    "Humanoid-v2": 2_000_000,
}

# Learning rate may be searched alongside every thematic group, as in the paper.
GROUPS = {
    "policy_losses": {"policy_loss", "policy_clip"},
    "architecture": {"hidden_sizes", "shared_network", "activation", "orthogonal_gain",
                     "policy_head_scale", "value_head_scale", "initial_std", "min_std",
                     "action_transform"},
    "normalization": {"normalize_observations", "observation_clip", "normalize_values",
                      "normalize_advantages", "max_grad_norm"},
    "advantage_estimation": {"gae_lambda", "value_clip", "value_loss_coefficient", "batch_mode"},
    "training_setup": {"num_envs", "envs_per_actor", "iteration_size", "num_epochs",
                       "minibatch_size"},
    "timesteps": {"gamma"},
    "optimizers": {"learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon",
                   "learning_rate_end_fraction"},
    "regularization": {"entropy_coefficient"},
}


@dataclass
class PPOConfig:
    benchmark: str = "seed_mujoco21_port"
    experiment_name: str = "baseline"
    thematic_group: str = "baseline"
    env_id: str = "Hopper-v2"
    seed: int = 0
    total_timesteps: Optional[int] = None
    policy_loss: str = "ppo"
    policy_clip: float = 0.2
    hidden_sizes: Tuple[int, ...] = (64, 64)
    shared_network: bool = True
    activation: str = "tanh"
    orthogonal_gain: float = 1.41
    policy_head_scale: float = 0.01
    value_head_scale: float = 1.0
    initial_std: float = 1.0
    min_std: float = 0.001
    action_transform: str = "clip"
    normalize_observations: bool = True
    observation_clip: float = 10.0
    normalize_values: bool = True
    normalize_advantages: bool = False
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    value_clip: Optional[float] = 0.2
    value_loss_coefficient: float = 1.0
    num_actors: int = 4
    envs_per_actor: int = 64
    num_envs: Optional[int] = None
    iteration_size: int = 2048
    num_epochs: int = 10
    minibatch_size: int = 64
    batch_mode: str = "split"
    gamma: float = 0.99
    handle_abandoned: bool = False
    learning_rate: float = 0.0003
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 0.0000001
    learning_rate_end_fraction: float = 0.0
    entropy_coefficient: float = 0.0
    eval_interval: int = 100_000
    eval_episodes: int = 100
    eval_num_envs: int = 10
    device: str = "cpu"
    tensorflow_threads: int = 1

    @property
    def unroll_length(self):
        """Transitions per trajectory, with a separate final bootstrap state."""
        return self.iteration_size // (self.num_envs or self.num_actors * self.envs_per_actor)

    def validate(self):
        for name in ("num_actors", "envs_per_actor", "iteration_size", "num_epochs",
                     "minibatch_size", "eval_interval", "eval_episodes", "eval_num_envs",
                     "tensorflow_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        layout_envs = self.num_actors * self.envs_per_actor
        if self.num_envs is None:
            self.num_envs = layout_envs
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if self.num_envs != layout_envs:
            raise ValueError("num_envs must equal num_actors * envs_per_actor")
        if self.total_timesteps is None:
            if self.env_id not in ENVIRONMENT_BUDGETS:
                raise ValueError("Specify total_timesteps for an environment outside the five-task suite")
            self.total_timesteps = ENVIRONMENT_BUDGETS[self.env_id]
        if type(self.total_timesteps) is not int or self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if self.iteration_size % self.num_envs:
            raise ValueError("iteration_size must be divisible by num_envs")
        if self.iteration_size % self.minibatch_size:
            raise ValueError("iteration_size must be divisible by minibatch_size")
        if self.minibatch_size % self.unroll_length:
            raise ValueError("minibatch_size must be divisible by unroll_length for Seed RL")
        if not self.hidden_sizes or any(type(value) is not int or value <= 0
                                        for value in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        self.hidden_sizes = tuple(self.hidden_sizes)
        if len(set(self.hidden_sizes)) != 1:
            raise ValueError("Seed RL's continuous-control model requires uniform hidden widths")
        for name in ("shared_network", "normalize_observations", "normalize_values",
                     "normalize_advantages", "handle_abandoned"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be true or false")
        if self.handle_abandoned:
            raise ValueError("handle_abandoned=True is unsupported by the batched MuJoCo actor; use False")
        for name in ("policy_clip", "orthogonal_gain", "policy_head_scale", "value_head_scale",
                     "initial_std", "min_std", "observation_clip", "max_grad_norm", "gae_lambda",
                     "value_loss_coefficient", "gamma", "learning_rate", "adam_beta1", "adam_beta2",
                     "adam_epsilon", "learning_rate_end_fraction", "entropy_coefficient"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + " must be a finite number")
        if self.value_clip is not None and (isinstance(self.value_clip, bool) or
                not isinstance(self.value_clip, (int, float)) or not math.isfinite(self.value_clip) or
                self.value_clip < 0):
            raise ValueError("value_clip must be nonnegative or null")
        for name in ("policy_clip", "orthogonal_gain", "initial_std", "min_std", "observation_clip",
                     "max_grad_norm", "learning_rate", "adam_epsilon"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if self.initial_std <= self.min_std:
            raise ValueError("initial_std must exceed min_std")
        for name in ("policy_head_scale", "value_head_scale", "value_loss_coefficient", "entropy_coefficient"):
            if getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative")
        for name in ("gamma", "gae_lambda", "learning_rate_end_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name + " must lie in [0, 1]")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(name + " must lie in [0, 1)")
        if self.policy_loss not in ("ppo", "pg"):
            raise ValueError("policy_loss must be ppo or pg")
        if self.activation not in ("tanh", "relu", "elu", "swish"):
            raise ValueError("activation must be tanh, relu, elu, or swish")
        if self.action_transform not in ("clip", "tanh"):
            raise ValueError("action_transform must be clip or tanh")
        if self.batch_mode not in ("split", "split_with_advantage_recomputation"):
            raise ValueError("batch_mode must be split or split_with_advantage_recomputation")
        if self.device != "cpu":
            raise ValueError("This experiment runner supports CPU execution only")
        if self.benchmark not in ("seed_mujoco21_port", "smoke"):
            raise ValueError("benchmark must be seed_mujoco21_port or smoke")
        if self.thematic_group not in set(GROUPS) | {"baseline", "smoke"}:
            raise ValueError("Unknown thematic_group: " + str(self.thematic_group))
        for name in ("experiment_name", "env_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError(name + " must use only letters, numbers, underscores, dots, or hyphens")
            if value in (".", ".."):
                raise ValueError(name + " cannot be . or ..")
        return self

    def as_dict(self):
        self.validate()
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


def config_from_dict(data):
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a mapping")
    valid = {field.name for field in fields(PPOConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    return PPOConfig(**dict(data)).validate()


def _merge_changes(data, changes):
    result = dict(data)
    result.update(changes)
    if "num_envs" in changes and "envs_per_actor" not in changes:
        actors = result.get("num_actors", PPOConfig.num_actors)
        count = result["num_envs"]
        if type(count) is not int or type(actors) is not int or actors <= 0 or count % actors:
            raise ValueError("num_envs must be divisible by num_actors")
        result["envs_per_actor"] = count // actors
    elif ("num_actors" in changes or "envs_per_actor" in changes) and "num_envs" not in changes:
        result["num_envs"] = None
    return result


def load_config(path, overrides=None):
    """Read a saved YAML/JSON config; overrides use the same field names."""
    source = Path(path)
    text = source.read_text()
    data = (json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a YAML mapping")
    return config_from_dict(_merge_changes(data, overrides or {}))


def parse_overrides(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Expected NAME=VALUE, got " + item)
        name, value = item.split("=", 1)
        if not name or name in result:
            raise ValueError("Override names must be nonempty and unique: " + name)
        result[name] = yaml.safe_load(value)
    return result


def _validate_group(group):
    if not isinstance(group, Mapping):
        raise ValueError("Group file must be a mapping")
    unknown = set(group) - {"name", "group", "parameters"}
    if unknown:
        raise ValueError("Unknown group fields: " + ", ".join(sorted(unknown)))
    name = group.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Group name must be a safe nonempty experiment name")
    thematic = group.get("group")
    if thematic not in GROUPS:
        raise ValueError("group must be one of: " + ", ".join(GROUPS))
    parameters = group.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a nonempty mapping of fields to candidate lists")
    outside = set(parameters) - GROUPS[thematic] - {"learning_rate"}
    if outside:
        raise ValueError("Parameters outside thematic group %s: %s" % (thematic, ", ".join(sorted(outside))))
    if set(parameters) == {"learning_rate"} and thematic != "optimizers":
        raise ValueError("A thematic grid must vary a field in its named group")
    size = 1
    for parameter, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(parameter + " candidates must be a nonempty list")
        fingerprints = [json.dumps(value, sort_keys=True) for value in values]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError(parameter + " candidates contain duplicates")
        size *= len(values)
    if size > 4096:
        raise ValueError("Grid exceeds 4096 configurations; split it into smaller saved grids")
    return {"name": name, "group": thematic, "parameters": dict(parameters)}


def load_group(path):
    return _validate_group(yaml.safe_load(Path(path).read_text()))


def expand_group(config, group):
    """Validate the entire grid before returning deterministic, named variants."""
    config.validate()
    group = _validate_group(group)
    keys = sorted(group["parameters"])
    variants = []
    for index, values in enumerate(itertools.product(*(group["parameters"][key] for key in keys))):
        changes = dict(zip(keys, values))
        payload = json.dumps(changes, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
        variant = "%s_%04d_%s" % (group["name"], index, digest)
        changes.update(experiment_name=variant, thematic_group=group["group"])
        result = config_from_dict(_merge_changes(config.as_dict(), changes))
        variants.append((variant, result))
    return variants


def to_gin_bindings(config):
    """Map supported choices to official SEED configurables.

    The caller must import the official modules and register the TensorFlow
    external configurables, as mujoco/ppo_main.py does. Action sampling is owned
    by the runtime adapter: clipping retains raw Gaussian samples for PPO.
    Actor layout, optimizer schedule, RNG, and evaluation are runtime settings.
    """
    config.validate()
    activation = ("@seed_ppo.study_swish" if config.activation == "swish"
                  else "@tf.nn." + config.activation)
    bindings = {
        "GeneralizedOnPolicyLoss.policy_loss": "@GeneralizedAdvantagePolicyLoss()",
        "GeneralizedAdvantagePolicyLoss.use_importance_weights": repr(config.policy_loss == "ppo"),
        "GeneralizedAdvantagePolicyLoss.ppo_epsilon": repr(config.policy_clip if config.policy_loss == "ppo" else None),
        "GeneralizedAdvantagePolicyLoss.advantage_preprocessor": "@AdvantagePreprocessor()",
        "AdvantagePreprocessor.normalize": repr(config.normalize_advantages),
        "GeneralizedOnPolicyLoss.reward_normalizer": "@PopArt()" if config.normalize_values else "None",
        "PopArt.mean_std_tracker": "@AverageMeanStd()",
        "PopArt.compensate": "False",
        "GeneralizedOnPolicyLoss.regularizer": "@KLPolicyRegularizer()",
        "KLPolicyRegularizer.entropy": repr(config.entropy_coefficient),
        "GeneralizedOnPolicyLoss.advantage_estimator": "@GAE()",
        "GeneralizedOnPolicyLoss.discount_factor": repr(config.gamma),
        "GeneralizedOnPolicyLoss.handle_abandoned_episodes_properly": repr(config.handle_abandoned),
        "GeneralizedOnPolicyLoss.value_ppo_style_clip_eps": repr(config.value_clip),
        "GeneralizedOnPolicyLoss.baseline_cost": repr(config.value_loss_coefficient),
        "GAE.lambda_": repr(config.gae_lambda),
        "ContinuousControlAgent.num_layers_policy": repr(len(config.hidden_sizes)),
        "ContinuousControlAgent.num_layers_value": repr(len(config.hidden_sizes)),
        "ContinuousControlAgent.num_units_policy": repr(config.hidden_sizes[0]),
        "ContinuousControlAgent.num_units_value": repr(config.hidden_sizes[0]),
        "ContinuousControlAgent.activation": activation,
        "ContinuousControlAgent.shared": repr(config.shared_network),
        "ContinuousControlAgent.std_independent_of_input": "True",
        "ContinuousControlAgent.kernel_init": "@study/tf.keras.initializers.Orthogonal()",
        "ContinuousControlAgent.last_kernel_init_policy": "@study/tf.keras.initializers.Orthogonal()",
        "ContinuousControlAgent.last_kernel_init_value": "@study/tf.keras.initializers.Orthogonal()",
        "study/tf.keras.initializers.Orthogonal.gain": repr(config.orthogonal_gain),
        "ContinuousControlAgent.last_kernel_init_policy_scaling": repr(config.policy_head_scale),
        "ContinuousControlAgent.last_kernel_init_value_scaling": repr(config.value_head_scale),
        "ContinuousControlAgent.observation_normalizer": "@InputNormalization()" if config.normalize_observations else "None",
        "InputNormalization.mean_std_tracker": "@AverageMeanStd()",
        "ContinuousControlAgent.correct_observations": "False",
        # The local implementation bypasses both normalization and its clipping
        # when observation normalization is disabled; retain that option meaning.
        "ContinuousControlAgent.input_clipping": repr(config.observation_clip if config.normalize_observations else None),
        "continuous_action_config.action_min_gaussian_std": repr(config.min_std),
        "continuous_action_config.action_gaussian_std_fn": repr("safe_exp"),
        "continuous_action_config.action_std_for_zero_param": repr(config.initial_std),
    }
    return [name + " = " + value for name, value in bindings.items()]
