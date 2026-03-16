import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

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
    #{"lr": 0.001, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.001, "gamma": 0.99, "batch_size": 32},
    {"lr": 0.0005, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.0005, "gamma": 0.99, "batch_size": 32},
    {"lr": 0.0001, "gamma": 0.95, "batch_size": 32},
    {"lr": 0.0001, "gamma": 0.99, "batch_size": 32},
    {"lr": 0.0005, "gamma": 0.97, "batch_size": 64},
    {"lr": 0.0003, "gamma": 0.98, "batch_size": 64},
    {"lr": 0.0002, "gamma": 0.99, "batch_size": 64},
    {"lr": 0.0001, "gamma": 0.98, "batch_size": 128},
]

# -----------------------------
# Training Loop
# -----------------------------
results = []

for i, params in enumerate(experiments):

    print(f"\nStarting Experiment {i+1}")
    print(params)

    # Environment setup
    env = gym.make("ALE/Adventure-v5")
    env = AtariWrapper(env)  # grayscale, resize to 84x84, frame stack
    env = Monitor(env)

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
        verbose=1,
    )

    # Train the agent
    model.learn(total_timesteps=50000, callback=logger)

    # Save model
    model.save(f"dqn_model_exp_{i+1}")

    # Collect results
    avg_reward = np.mean(logger.episode_rewards) if logger.episode_rewards else 0
    avg_length = np.mean(logger.episode_lengths) if logger.episode_lengths else 0

    results.append({
        "Experiment": i+1,
        "Learning Rate": params["lr"],
        "Gamma": params["gamma"],
        "Batch Size": params["batch_size"],
        "Avg Reward": avg_reward,
        "Avg Episode Length": avg_length
    })

    env.close()

# Save experiment results
df = pd.DataFrame(results)
df.to_csv("training_results.csv", index=False)

print("\nTraining Complete")
print(df)