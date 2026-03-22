"""
play.py — Atari DQN Agent Evaluation Script
Author: Jean Jabo
Environment: ALE/Pong-v5

Responsibilities:
  - Load best trained model (auto-detects from training_results.csv)
  - GreedyQPolicy action selection (deterministic=True)
  - Episode loop with real-time rendering
  - Performance metrics collection and CSV export
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

ENV_ID          = "ALE/Pong-v5"
BEST_MODEL_PATH = "dqn_model.zip"


# ─────────────────────────────────────────
#  GreedyQPolicy — deterministic action selection
# ─────────────────────────────────────────
def greedy_action(model, obs):
    """Always choose argmax Q-value action. No exploration."""
    action, _ = model.predict(obs, deterministic=True)
    return action


# ─────────────────────────────────────────
#  Performance Metrics Collector
# ─────────────────────────────────────────
class MetricsCollector:
    def __init__(self):
        self.episodes = []

    def record(self, episode, total_reward, steps, duration):
        self.episodes.append({
            "Episode":       episode,
            "Total Reward":  total_reward,
            "Steps":         steps,
            "Duration (s)":  round(duration, 2),
            "Reward/Step":   round(total_reward / steps, 4) if steps > 0 else 0,
        })

    def summary(self):
        rewards = [e["Total Reward"] for e in self.episodes]
        steps   = [e["Steps"]        for e in self.episodes]
        return {
            "Mean Reward": round(np.mean(rewards), 2),
            "Std Reward":  round(np.std(rewards),  2),
            "Max Reward":  round(np.max(rewards),  2),
            "Min Reward":  round(np.min(rewards),  2),
            "Mean Steps":  round(np.mean(steps),   2),
        }

    def to_dataframe(self):
        return pd.DataFrame(self.episodes)

    def save(self, path="play_metrics.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[Metrics] Saved → {path}")
        return df


# ─────────────────────────────────────────
#  Resolve best model path automatically
# ─────────────────────────────────────────
def resolve_model(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    # Try without .zip (SB3 adds it automatically)
    if Path(model_path.replace(".zip", "")).with_suffix(".zip").exists():
        return model_path

    # Auto-select from training results
    results_file = Path("training_results.csv")
    if results_file.exists():
        df = pd.read_csv(results_file)
        best_exp = int(df.loc[df["Avg Reward"].idxmax(), "Experiment"])
        candidate = Path(f"dqn_model_exp_{best_exp}.zip")
        if candidate.exists():
            print(f"[Info] Auto-selected model: {candidate}  (Exp #{best_exp})")
            return str(candidate)

    raise FileNotFoundError(
        f"No model found at '{model_path}'. Run train.py first."
    )


# ─────────────────────────────────────────
#  Main Play Loop
# ─────────────────────────────────────────
def play(
    model_path:   str  = BEST_MODEL_PATH,
    env_id:       str  = ENV_ID,
    num_episodes: int  = 5,
    render:       bool = True,
    save_metrics: bool = True,
):
    model_path = resolve_model(model_path)

    print(f"\n{'='*55}")
    print(f"  Model       : {model_path}")
    print(f"  Environment : {env_id}")
    print(f"  Episodes    : {num_episodes}")
    print(f"  Render      : {render}")
    print(f"{'='*55}\n")

    model = DQN.load(model_path)
    print("[Model] Loaded.\n")

    render_mode = "human" if render else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env)

    metrics = MetricsCollector()

    for ep in range(1, num_episodes + 1):
        obs, _    = env.reset()
        done      = False
        total_rew = 0.0
        steps     = 0
        t0        = time.time()

        print(f"─── Episode {ep}/{num_episodes} ───")

        while not done:
            action              = greedy_action(model, obs)
            obs, rew, term, trunc, _ = env.step(action)
            total_rew += rew
            steps     += 1
            done        = term or trunc

            if render:
                env.render()

        duration = time.time() - t0
        metrics.record(ep, total_rew, steps, duration)
        print(f"  Reward: {total_rew:.1f}  Steps: {steps}  Duration: {duration:.1f}s")

    env.close()

    print(f"\n{'='*55}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*55}")
    for k, v in metrics.summary().items():
        print(f"  {k:<18}: {v}")
    print(f"{'='*55}\n")

    if save_metrics:
        df = metrics.save("play_metrics.csv")
        print(df.to_string(index=False))

    return metrics


# ─────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Atari with trained DQN (Jean Jabo)")
    parser.add_argument("--model_path", type=str,  default=BEST_MODEL_PATH)
    parser.add_argument("--env",        type=str,  default=ENV_ID)
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--no_render",  action="store_true")
    parser.add_argument("--no_save",    action="store_true")
    args = parser.parse_args()

    play(
        model_path=args.model_path,
        env_id=args.env,
        num_episodes=args.episodes,
        render=not args.no_render,
        save_metrics=not args.no_save,
    )
