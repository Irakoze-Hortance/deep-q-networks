import sys
import numpy

# --- SAFETY INJECTION 1: The Expanded NumPy Bridge ---
# This maps the internal structure of NumPy 2.x (Kaggle) to your 1.x (Local)
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

import argparse
import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

gym.register_envs(ale_py)

def make_render_env(env_id: str, policy: str, seed: int, render_mode: str):
    # Change: We now pass render_mode to avoid WSL graphics crashes
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env)
    if policy == "MlpPolicy":
        env = FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def parse_args():
    parser = argparse.ArgumentParser(description="Play Atari using a trained DQN model.")
    parser.add_argument("--model-path", default="dqn_model.zip")
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--policy", default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # --- SAFETY INJECTION 2: Headless Mode Flag ---
    parser.add_argument("--headless", action="store_true", help="Run without a window (best for WSL)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select render mode based on headless flag
    render_mode = "rgb_array" if args.headless else "human"

    print(f"Loading model: {args.model_path}")
    
    env = make_render_env(args.env_id, args.policy, args.seed, render_mode)

    # --- SAFETY INJECTION 3: Stripping the Corrupted Metadata ---
    # This tells SB3 to ignore the Kaggle optimizer/learning rate data 
    # that usually causes the version-mismatch Segfault.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    try:
        model = DQN.load(
            args.model_path, 
            env=env, 
            custom_objects=custom_objects,
            device="cpu" # Start on CPU to avoid CUDA mapping errors
        )
        print("SUCCESS: Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {episode}: reward = {total_reward:.1f}, steps = {steps}")

    env.close()

if __name__ == "__main__":
    main()