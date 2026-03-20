import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import Bas6eCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(ale_py)

SEED = 42
TRAIN_TIMESTEPS = 50000
EVAL_EPISODES = 10


def make_env(seed: int):
    env = gym.make("ALE/Adventure-v5")
    env = AtariWrapper(env)  # Keep preprocessing identical across training and eval.
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# -----------------------------
# Custom Callback for Logging
# -----------------------------
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        if "episode" in self.locals["infos"][0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.episode_lengths.append(self.locals["infos"][0]["episode"]["l"])
        return True


# -----------------------------
# Hyperparameter Experiments
# -----------------------------
experiments = [
    {"lr": 0.001, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.001, "gamma": 0.99, "batch_size": 32},
    {"lr": 0.0005, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.0005, "gamma": 0.97, "batch_size": 64},
    {"lr": 0.0003, "gamma": 0.98, "batch_size": 64},
    {"lr": 0.0002, "gamma": 0.99, "batch_size": 64},
    {"lr": 0.0001, "gamma": 0.98, "batch_size": 128},
    {"lr": 0.0005, "gamma": 0.99, "batch_size": 32},
    {"lr": 0.0001, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.0001, "gamma": 0.99, "batch_size": 32},
]

# -----------------------------
# Training Loop
# -----------------------------
results = []
best_result = None
best_model_path = Path("best_dqn_model.zip")

for i, params in enumerate(experiments):

    print(f"\nStarting Experiment {i+1}")
    print(params)

    # Environment setup
    env = make_env(seed=SEED + i)

    logger = RewardLogger()

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=params["lr"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=50000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        seed=SEED + i,
        verbose=1,
    )

    # Train the agent
    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=logger)

    # Standardized evaluation for consistent experiment-to-experiment rewards.
    eval_env = make_env(seed=SEED + 1000 + i)
    eval_mean_reward, eval_std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        return_episode_rewards=False,
    )
    eval_env.close()

    # Save model for this experiment.
    model_path = Path(f"dqn_model_exp_{i+1}.zip")
    model.save(str(model_path))

    # Collect results
    avg_training_reward = np.mean(logger.episode_rewards) if logger.episode_rewards else 0
    avg_length = np.mean(logger.episode_lengths) if logger.episode_lengths else 0

    result_row = {
        "Experiment": i+1,
        "Learning Rate": params["lr"],
        "Gamma": params["gamma"],
        "Batch Size": params["batch_size"],
        "Avg Train Reward": avg_training_reward,
        "Avg Episode Length": avg_length,
        "Eval Mean Reward": eval_mean_reward,
        "Eval Std Reward": eval_std_reward,
        "Eval Episodes": EVAL_EPISODES,
    }
    results.append(result_row)

    if best_result is None or eval_mean_reward > best_result["Eval Mean Reward"]:
        best_result = dict(result_row)
        shutil.copy2(model_path, best_model_path)
        print(
            f"New best model: Experiment {i+1} with eval mean reward {eval_mean_reward:.3f}"
        )

    env.close()

# Save experiment results
df = pd.DataFrame(results)
if best_result is not None:
    df["Is Best"] = df["Experiment"] == best_result["Experiment"]
else:
    df["Is Best"] = False

df.to_csv("training_results.csv", index=False)
if best_result is not None:
    pd.DataFrame([best_result]).to_csv("best_experiment.csv", index=False)
    print("\nBest Experiment Summary")
    print(pd.DataFrame([best_result]))
    print(f"Best model saved to: {best_model_path}")

print("\nTraining Complete")
print(df)