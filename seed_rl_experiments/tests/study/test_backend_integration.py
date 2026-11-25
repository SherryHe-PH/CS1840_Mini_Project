"""Opt-in checks against the real Linux Seed RL/TensorFlow installation.

SEED_RUN_INTEGRATION=1 python -m pytest -q tests/study/test_backend_integration.py
The full training/evaluation smoke remains the quick-start integration check.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("SEED_RUN_INTEGRATION") != "1",
    reason="Requires the real OOD TensorFlow/custom-gRPC runtime")


def test_native_model_and_raw_action_likelihoods():
    program = r'''
import numpy as np
from seed_ppo.config import PPOConfig
from seed_ppo.runtime import build_learner, seed_root
config = PPOConfig(benchmark='smoke', env_id='toy_env', num_actors=1,
                   envs_per_actor=4, iteration_size=32, minibatch_size=16,
                   total_timesteps=256, eval_interval=128, eval_episodes=2).validate()
model, tf, gin = build_learner(config, seed_root(), '127.0.0.1:0', 'unused')
distribution = model.parametric_action_distribution(tf.zeros([1, 6]))
raw = distribution.sample(1000).numpy()
assert np.any(np.abs(raw) > 1), 'Samples must remain unbounded before simulator clipping'
log_raw = distribution.log_prob(tf.constant([[2., 0., 0.]])).numpy()
log_clipped = distribution.log_prob(tf.constant([[1., 0., 0.]])).numpy()
assert np.allclose(log_raw - log_clipped, -1.5, atol=1e-5)
assert type(model.training_agent).__module__.startswith('seed_rl.')
assert model.config.server_address == '127.0.0.1:0'
assert model.frames_per_virtual_batch == 32
model.env.close()
print('PASS: native agent, raw Gaussian likelihoods, and batch/address wiring')
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
