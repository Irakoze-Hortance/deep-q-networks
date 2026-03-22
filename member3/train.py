"""
train.py — GPU-Optimised DQN Training for Google Colab (T4 GPU)
Author: Jean Jabo
Environment: ALE/Pong-v5
Device: CUDA (T4) auto-detected

Fix: optimize_memory_usage=True requires handle_timeout_termination=False
     (SB3 does not support both simultaneously)
"""

import os
import shutil
import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

# ─────────────────────────────────────────
#  Device Detection
# ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*55}")
print(f"  Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────
ENV_ID           = "ALE/Pong-v5"
POLICY           = "CnnPolicy"
SWEEP_TIMESTEPS  = 200_000
BEST_TIMESTEPS   = 500_000
BUFFER_SIZE      = 100_000
LEARNING_STARTS  = 50_000
TRAIN_FREQ       = 4
GRADIENT_STEPS   = 1
TARGET_UPDATE    = 1_000
RESULTS_CSV      = "training_results.csv"
BEST_MODEL_PATH  = "dqn_model"     # SB3 auto-appends .zip

# ─────────────────────────────────────────
#  Jean Jabo — 10 Hyperparameter Experiments
# ─────────────────────────────────────────
experiments = [
    # Exp 1 — High LR, low gamma, small batch (aggressive, unstable)
    dict(lr=0.001,  gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 2 — High LR, high gamma (long-horizon focus, likely unstable)
    dict(lr=0.001,  gamma=0.99, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 3 — Moderate LR, low gamma, small batch
    dict(lr=0.0005, gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 4 — Moderate LR, mid gamma, larger batch, slower decay
    dict(lr=0.0005, gamma=0.97, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.30),
    # Exp 5 — Balanced: moderate LR, high gamma, medium batch
    dict(lr=0.0003, gamma=0.98, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.25),
    # Exp 6 — Low LR, high gamma, tight epsilon end
    dict(lr=0.0002, gamma=0.99, batch_size=64,  eps_start=1.0, eps_end=0.02, eps_decay=0.20),
    # Exp 7 — Very low LR, large batch (most stable, slowest convergence)
    dict(lr=0.0001, gamma=0.98, batch_size=128, eps_start=1.0, eps_end=0.05, eps_decay=0.15),
    # Exp 8 — Reduced eps_start (less early exploration)
    dict(lr=0.0005, gamma=0.99, batch_size=32,  eps_start=0.9, eps_end=0.05, eps_decay=0.20),
    # Exp 9 — Higher eps_end (agent keeps exploring throughout)
    dict(lr=0.0001, gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.10, eps_decay=0.20),
    # Exp 10 — Very slow epsilon decay + lowest LR (Jean Jabo's best candidate)
    dict(lr=0.0001, gamma=0.99, batch_size=32,  eps_start=1.0, eps_end=0.01, eps_decay=0.10),
]

# ─────────────────────────────────────────
#  Reward + Episode Length Logger
# ─────────────────────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


# ─────────────────────────────────────────
#  Environment Factory
# ─────────────────────────────────────────
def make_env(env_id: str = ENV_ID) -> gym.Env:
    env = gym.make(env_id)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


# ─────────────────────────────────────────
#  Single Experiment Runner
# ─────────────────────────────────────────
def run_experiment(exp_index: int, params: dict, timesteps: int):
    print(f"\n{'='*55}")
    print(f"  Experiment {exp_index}/10  [{timesteps:,} steps]")
    print(f"  lr={params['lr']}  gamma={params['gamma']}  batch={params['batch_size']}")
    print(f"  eps: {params['eps_start']} → {params['eps_end']}  decay={params['eps_decay']}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*55}")

    env = make_env()
    logger = RewardLogger()

    model = DQN(
        policy=POLICY,
        env=env,
        device=DEVICE,
        learning_rate=params["lr"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE,
        exploration_initial_eps=params["eps_start"],
        exploration_final_eps=params["eps_end"],
        exploration_fraction=params["eps_decay"],
        optimize_memory_usage=True,          # halves replay buffer RAM
        replay_buffer_kwargs={
            "handle_timeout_termination": False   # required when optimize_memory_usage=True
        },
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=logger, progress_bar=True)

    avg_reward = np.mean(logger.episode_rewards) if logger.episode_rewards else 0.0
    max_reward = np.max(logger.episode_rewards)  if logger.episode_rewards else 0.0
    avg_length = np.mean(logger.episode_lengths) if logger.episode_lengths else 0.0

    env.close()

    save_path = f"dqn_model_exp_{exp_index}"
    model.save(save_path)
    print(f"  Saved  → {save_path}.zip")
    print(f"  Result → Avg Reward: {avg_reward:.2f}  Max: {max_reward:.2f}  Episodes: {len(logger.episode_rewards)}")

    result = {
        "Experiment":         exp_index,
        "Member":             "Jean Jabo",
        "Policy":             POLICY,
        "Learning Rate":      params["lr"],
        "Gamma":              params["gamma"],
        "Batch Size":         params["batch_size"],
        "Epsilon Start":      params["eps_start"],
        "Epsilon End":        params["eps_end"],
        "Epsilon Decay":      params["eps_decay"],
        "Timesteps":          timesteps,
        "Avg Reward":         round(avg_reward, 2),
        "Max Reward":         round(max_reward, 2),
        "Avg Episode Length": round(avg_length, 1),
    }

    return result, model


# ─────────────────────────────────────────
#  PHASE 1 — Sweep all 10 configs
# ─────────────────────────────────────────
print(f"🔍 PHASE 1 — Hyperparameter Sweep ({SWEEP_TIMESTEPS:,} steps each)")
print(f"   Environment : {ENV_ID}")
print(f"   10 × {SWEEP_TIMESTEPS:,} = {10 * SWEEP_TIMESTEPS:,} total steps\n")

results = []

for i, params in enumerate(experiments, start=1):
    result, model = run_experiment(i, params, SWEEP_TIMESTEPS)
    results.append(result)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ─────────────────────────────────────────
#  Identify best experiment
# ─────────────────────────────────────────
df = pd.DataFrame(results)
best_idx    = df["Avg Reward"].idxmax()
best_exp    = int(df.loc[best_idx, "Experiment"])
best_params = experiments[best_exp - 1]

print(f"\n{'='*55}")
print(f"  🏆 Best experiment from sweep: #{best_exp}")
print(f"  Sweep avg reward: {df.loc[best_idx, 'Avg Reward']:.2f}")
print(f"{'='*55}")

# ─────────────────────────────────────────
#  PHASE 2 — Full run on winning config
# ─────────────────────────────────────────
print(f"\n🚀 PHASE 2 — Full run on Experiment #{best_exp} ({BEST_TIMESTEPS:,} steps)\n")

final_result, final_model = run_experiment(best_exp, best_params, BEST_TIMESTEPS)
final_model.save(BEST_MODEL_PATH)
print(f"\n[Saved] Best model → '{BEST_MODEL_PATH}.zip'")

# Update the table with full-run figures for the best experiment
df.loc[best_idx, "Avg Reward"]         = final_result["Avg Reward"]
df.loc[best_idx, "Max Reward"]         = final_result["Max Reward"]
df.loc[best_idx, "Avg Episode Length"] = final_result["Avg Episode Length"]
df.loc[best_idx, "Timesteps"]          = BEST_TIMESTEPS

# ─────────────────────────────────────────
#  Save & Print Results
# ─────────────────────────────────────────
df.to_csv(RESULTS_CSV, index=False)

print(f"\n{'='*55}")
print(f"  ✅ Training Complete — Jean Jabo")
print(f"  Environment  : {ENV_ID}")
print(f"  Best config  : Experiment #{best_exp}")
print(f"  Best model   : {BEST_MODEL_PATH}.zip")
print(f"  Results CSV  : {RESULTS_CSV}")
print(f"{'='*55}\n")

print(df[[
    "Experiment", "Learning Rate", "Gamma", "Batch Size",
    "Epsilon Start", "Epsilon End", "Epsilon Decay",
    "Avg Reward", "Max Reward", "Avg Episode Length"
]].to_string(index=False))
