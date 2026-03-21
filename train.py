import argparse
import shutil
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
SEED = 42
TRAIN_TIMESTEPS = 200_000
EVAL_EPISODES = 10


def make_env(seed: int):
    """Create and wrap the Atari environment. CnnPolicy is always used, so
    FlattenObservation is not applied, the CNN handles raw stacked frames."""
    env = gym.make(ENV_ID)
    env = AtariWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def run_sanity_check():
    """Run a random agent for a fixed number of steps and print per-episode
    rewards. If nothing prints, episodes are not terminating, it is a sign that
    something is wrong with the environment or wrapper stack."""
    print("\n Sanity Check: Random Agent")
    env = make_env(seed=0)
    obs, _ = env.reset()
    episodes_seen = 0
    for _ in range(2_000):
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            reward = info.get("episode", {}).get("r", None)
            print(f"  Random episode {episodes_seen + 1} reward: {reward}")
            episodes_seen += 1
            obs, _ = env.reset()
    env.close()
    if episodes_seen == 0:
        raise RuntimeError(
            "Sanity check failed: no episodes completed in 2,000 steps. "
            "Check your environment and wrapper configuration."
        )
    print(f"  Sanity check passed ({episodes_seen} episodes completed).\n")


class RewardLogger(BaseCallback):
    """Log true (unclipped) per-episode rewards and lengths via the Monitor
    info dict. AtariWrapper clips rewards during training, but Monitor records
    the original game score in info['episode']['r'], so we read from there."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN experiments for Atari.")
    parser.add_argument(
        "--experiments-file",
        required=True,
        help="CSV file with experiment rows (lr, gamma, batch_size, epsilon_start, epsilon_end, epsilon_decay).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where models and results are saved.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Experiment number to start from (1-indexed). Use this to resume after adding new rows to the CSV.",
    )
    return parser.parse_args()


def load_experiments(experiments_file: Path):
    required_columns = [
        "lr",
        "gamma",
        "batch_size",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
    ]

    df = pd.read_csv(experiments_file)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns in experiments file: " + ", ".join(missing_columns)
        )

    experiments = df[required_columns].to_dict(orient="records")
    if not experiments:
        raise ValueError("Experiments file contains no rows.")

    for i, exp in enumerate(experiments, start=1):
        exp["lr"] = float(exp["lr"])
        exp["gamma"] = float(exp["gamma"])
        exp["batch_size"] = int(exp["batch_size"])
        exp["epsilon_start"] = float(exp["epsilon_start"])
        exp["epsilon_end"] = float(exp["epsilon_end"])
        exp["epsilon_decay"] = float(exp["epsilon_decay"])

    return experiments


def train_experiments(experiments, output_dir: Path, start_from: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "dqn_model.zip"

    # Load existing results if resuming, so the final CSV stays complete
    results_path = output_dir / "training_results.csv"
    if start_from > 1 and results_path.exists():
        existing_df = pd.read_csv(results_path)
        results = existing_df.to_dict(orient="records")
        print(f"Resuming from experiment {start_from}. Loaded {len(results)} existing result(s).")
    else:
        results = []

    # Determine best result so far from already-completed experiments
    best_result = None
    for row in results:
        if best_result is None or row["Eval Mean Reward"] > best_result["Eval Mean Reward"]:
            best_result = dict(row)

    for i, params in enumerate(experiments, start=1):
        if i < start_from:
            continue

        print(f"\n Experiment {i}")
        print(params)

        env = make_env(seed=SEED + i)
        logger = RewardLogger()

        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=params["lr"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            buffer_size=50_000,
            exploration_initial_eps=params["epsilon_start"],
            exploration_final_eps=params["epsilon_end"],
            exploration_fraction=params["epsilon_decay"],
            seed=SEED + i,
            verbose=1,
            device="cuda",
        )

        model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=logger)

        eval_env = make_env(seed=SEED + 1000 + i)
        eval_mean_reward, eval_std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            return_episode_rewards=False,
        )
        eval_env.close()

        model_path = output_dir / f"dqn_model_exp_{i}.zip"
        model.save(str(model_path))

        avg_training_reward = (
            float(np.mean(logger.episode_rewards)) if logger.episode_rewards else 0.0
        )
        avg_length = (
            float(np.mean(logger.episode_lengths)) if logger.episode_lengths else 0.0
        )

        result_row = {
            "Experiment": i,
            "Policy": "CnnPolicy",
            "Learning Rate": params["lr"],
            "Gamma": params["gamma"],
            "Batch Size": params["batch_size"],
            "Epsilon Start": params["epsilon_start"],
            "Epsilon End": params["epsilon_end"],
            "Epsilon Decay": params["epsilon_decay"],
            "Avg Train Reward (true)": avg_training_reward,
            "Avg Episode Length": avg_length,
            "Eval Mean Reward": float(eval_mean_reward),
            "Eval Std Reward": float(eval_std_reward),
            "Eval Episodes": EVAL_EPISODES,
        }
        results.append(result_row)

        if best_result is None or eval_mean_reward > best_result["Eval Mean Reward"]:
            best_result = dict(result_row)
            shutil.copy2(model_path, best_model_path)
            print(
                f"New best model: Experiment {i} (CnnPolicy) "
                f"with eval mean reward {eval_mean_reward:.3f}"
            )

        env.close()

    df = pd.DataFrame(results)
    if best_result is not None:
        df["Is Best"] = df["Experiment"] == best_result["Experiment"]
    else:
        df["Is Best"] = False

    df.to_csv(output_dir / "training_results.csv", index=False)

    if best_result is not None:
        pd.DataFrame([best_result]).to_csv(output_dir / "best_experiment.csv", index=False)
        print("\nBest Experiment Summary")
        print(pd.DataFrame([best_result]))
        print(f"Best model saved to: {best_model_path}")

    print("\nTraining Complete")
    print(df)


if __name__ == "__main__":
    args = parse_args()
    run_sanity_check()
    selected_experiments = load_experiments(Path(args.experiments_file))
    train_experiments(
        experiments=selected_experiments,
        output_dir=Path(args.output_dir),
        start_from=args.start_from,
    )