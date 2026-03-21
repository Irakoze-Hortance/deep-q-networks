import argparse

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor


gym.register_envs(ale_py)


def make_eval_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = AtariWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def load_model(model_path: str, env):
    return DQN.load(model_path, env=env)


def greedy_action(model: DQN, observation):
    # deterministic=True applies greedy action selection for DQN inference.
    action, _ = model.predict(observation, deterministic=True)
    return action


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play script foundation: load a trained DQN and run greedy action selection."
    )
    parser.add_argument("--model-path", default="dqn_model.zip", help="Path to trained model zip")
    parser.add_argument("--env-id", default="ALE/Adventure-v5", help="Gymnasium Atari env id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_eval_env(args.env_id, args.seed)

    model = load_model(args.model_path, env)

    observation, _ = env.reset(seed=args.seed)
    action = greedy_action(model, observation)

    print(f"Model loaded: {args.model_path}")
    print(f"Environment ready: {args.env_id}")
    print(f"First greedy action: {action}")

    # Member 3 can extend from here for full episode loop, rendering, and metrics.
    env.close()


if __name__ == "__main__":
    main()
