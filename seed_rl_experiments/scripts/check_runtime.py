"""Check the CPU TensorFlow/custom-gRPC/simulator dependencies before training."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    from seed_ppo.runtime import import_seed, seed_root
    import_seed(seed_root())
    import numpy as np
    import tensorflow as tf
    from seed_rl import grpc
    from seed_rl.mujoco import toy_env
    if tf.__version__ != "2.4.1" or np.__version__ != "1.19.5":
        raise RuntimeError("Expected TensorFlow 2.4.1 / NumPy 1.19.5")
    assert not tf.config.list_physical_devices("GPU")
    with tf.device("/CPU:0"):
        assert float(tf.reduce_sum(tf.constant([1., 2.])).numpy()) == 3.
    if args.env_id == "toy_env":
        environment = toy_env.ToyEnv()
    else:
        import mujoco_py
        import gym
        environment = gym.make(args.env_id)
    try:
        environment.reset()
        for _ in range(10):
            observation, reward, done, unused_info = environment.step(environment.action_space.sample())
            assert np.isfinite(observation).all() and np.isfinite(reward)
            if done:
                environment.reset()
    finally:
        environment.close()
    print("PASS: CPU TensorFlow, custom gRPC, and " + args.env_id)


if __name__ == "__main__":
    main()
