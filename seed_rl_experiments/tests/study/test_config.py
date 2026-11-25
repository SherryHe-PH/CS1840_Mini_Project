"""Validate experimental protocols and prevent confounded thematic grids."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from seed_ppo.config import (ENVIRONMENT_BUDGETS, GROUPS, PPOConfig, config_from_dict,
                             expand_group, load_config, load_group, parse_overrides,
                             to_gin_bindings)


def test_paper_defaults_have_correct_seed_shapes_and_roundtrip():
    config = load_config(ROOT / "configs/ppo_default.yaml")
    assert config.num_envs == 256
    assert config.unroll_length == 8
    assert config.minibatch_size // config.unroll_length == 8
    assert config.total_timesteps == 1_000_000
    assert config_from_dict(config.as_dict()).as_dict() == config.as_dict()
    humanoid = load_config(ROOT / "configs/ppo_default.yaml", {"env_id": "Humanoid-v2"})
    assert humanoid.total_timesteps == 2_000_000


def test_smoke_is_explicitly_distinct_and_uses_real_evaluation_checkpoints():
    config = load_config(ROOT / "configs/smoke.yaml")
    assert config.benchmark == "smoke"
    assert config.num_envs == 4 and config.unroll_length == 8
    assert config.total_timesteps // config.iteration_size == 8
    assert list(range(config.eval_interval, config.total_timesteps + 1, config.eval_interval)) == [128, 256]
    assert config.eval_episodes == 2


@pytest.mark.parametrize("changes", [
    {"num_actors": 0}, {"num_envs": 250}, {"minibatch_size": 63},
    {"hidden_sizes": [64, 128]}, {"initial_std": 0.001},
    {"normalize_values": "false"}, {"learning_rate": float("nan")},
    {"seed": -1}, {"handle_abandoned": True}, {"device": "cuda"},
])
def test_invalid_runtime_or_algorithm_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/ppo_default.yaml", changes)


def test_all_eight_grids_are_deterministic_and_keep_other_groups_fixed():
    baseline = load_config(ROOT / "configs/ppo_default.yaml")
    paths = sorted((ROOT / "configs/groups").glob("*.yaml"))
    assert len(paths) == 8
    assert {load_group(path)["group"] for path in paths} == set(GROUPS)
    for path in paths:
        group = load_group(path)
        variants = expand_group(baseline, group)
        assert [(name, config.as_dict()) for name, config in variants] == [
            (name, config.as_dict()) for name, config in expand_group(baseline, group)]
        assert len({name for name, _ in variants}) == len(variants)
        for name, config in variants:
            changed = {key for key, value in config.as_dict().items()
                       if value != baseline.as_dict()[key]}
            allowed = GROUPS[group["group"]] | {"learning_rate", "experiment_name", "thematic_group"}
            # Changing total environments derives the actor's equal environment batch.
            assert changed <= allowed


def test_cross_group_parameters_and_duplicate_values_are_rejected():
    config = PPOConfig().validate()
    with pytest.raises(ValueError, match="outside thematic group"):
        expand_group(config, {"name": "bad", "group": "architecture",
                              "parameters": {"gamma": [0.9, 0.99]}})
    with pytest.raises(ValueError, match="duplicates"):
        expand_group(config, {"name": "bad", "group": "regularization",
                              "parameters": {"entropy_coefficient": [0, 0]}})


def test_actor_layout_is_derived_without_silently_accepting_inconsistent_records():
    config = load_config(ROOT / "configs/ppo_default.yaml", {"num_envs": 64})
    assert config.envs_per_actor == 16 and config.num_actors == 4
    record = config.as_dict()
    record["num_envs"] = 128
    with pytest.raises(ValueError, match="must equal"):
        config_from_dict(record)


def test_gin_uses_scaled_shared_heads_and_no_ambiguous_clipped_distribution():
    bindings = "\n".join(to_gin_bindings(PPOConfig()))
    assert "ContinuousControlAgent.shared = True" in bindings
    assert "ContinuousControlAgent.last_kernel_init_policy_scaling = 0.01" in bindings
    assert "study/tf.keras.initializers.Orthogonal.gain = 1.41" in bindings
    assert "GeneralizedOnPolicyLoss.value_ppo_style_clip_eps = 0.2" in bindings
    assert "GeneralizedOnPolicyLoss.handle_abandoned_episodes_properly = False" in bindings
    assert "action_postprocessor" not in bindings


def test_disabling_observation_normalization_also_bypasses_local_input_clip():
    bindings = to_gin_bindings(PPOConfig(normalize_observations=False))
    assert "ContinuousControlAgent.observation_normalizer = None" in bindings
    assert "ContinuousControlAgent.input_clipping = None" in bindings


def test_swish_uses_explicit_study_alias_for_legacy_gin_registry():
    assert "ContinuousControlAgent.activation = @seed_ppo.study_swish" in to_gin_bindings(
        PPOConfig(activation="swish"))


def test_cli_overrides_are_typed_and_unknown_fields_fail():
    assert parse_overrides(["shared_network=false", "value_clip=null"]) == {
        "shared_network": False, "value_clip": None}
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"unknown": 1})
    with pytest.raises(ValueError, match="unique"):
        parse_overrides(["gamma=0.9", "gamma=0.99"])
