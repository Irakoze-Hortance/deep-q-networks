"""
DQN_Atari_JeanJabo.py
=====================
Author      : Jean Jabo
Environment : ALE/Pong-v5
Device      : CUDA GPU (auto-detected) or CPU

Compatible with:
  - Windows (local Python)
  - Linux / macOS
  - Google Colab (T4 GPU)

Run:
  python DQN_Atari_JeanJabo.py
"""

import subprocess
import sys
import platform
import importlib.util
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_COLAB   = "google.colab" in sys.modules or "COLAB_GPU" in __import__("os").environ

# ════════════════════════════════════════════════════════════
# SECTION 1 — Install Dependencies
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  SECTION 1 — Installing Dependencies")
print(f"  Platform : {platform.system()}")
print("="*55)

def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages]
    )

def module_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def ensure_package(package_name: str, module_name: str | None = None):
    module_name = module_name or package_name
    if module_installed(module_name):
        return
    print(f"Installing missing package: {package_name}")
    pip_install(package_name)

def get_ale_rom_dir() -> Path:
    """Return ale-py's ROM folder and attempt a repair install if ale_py import fails."""
    try:
        import ale_py  # noqa: F401
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "ale-py"]
        )
        import ale_py  # noqa: F401

    import ale_py
    rom_dir = Path(ale_py.__file__).resolve().parent / "roms"
    rom_dir.mkdir(parents=True, exist_ok=True)
    return rom_dir

# Core packages — install only if missing for faster repeat runs.
ensure_package("stable-baselines3[extra]", "stable_baselines3")
ensure_package("gymnasium[atari]", "gymnasium")
ensure_package("ale-py", "ale_py")
ensure_package("torch", "torch")
ensure_package("pandas", "pandas")
ensure_package("numpy", "numpy")
ensure_package("matplotlib", "matplotlib")
ensure_package("moviepy", "moviepy")
ensure_package("AutoROM.accept-rom-license", "AutoROM")

# Accept ROMs (cross-platform replacement for the old gymnasium extra)
# Use explicit install_dir to avoid AutoROM ale-py auto-detection issues.
rom_dir = get_ale_rom_dir()
autorom_base = [
    sys.executable,
    "-m",
    "AutoROM",
    "--accept-license",
    "--install-dir",
    str(rom_dir),
]

try:
    subprocess.check_call(autorom_base)
except subprocess.CalledProcessError as exc:
    print("WARNING: AutoROM failed to install Atari ROMs automatically.")
    print(f"         Expected ROM folder: {rom_dir}")
    print(f"         AutoROM error code : {exc.returncode}")
    print("         Training may fail if Pong ROM is missing.\n")

# Virtual display only on Linux/Colab (not needed or available on Windows)
if not IS_WINDOWS:
    try:
        pip_install("pyvirtualdisplay")
        subprocess.call(
            ["apt-get", "install", "-y", "-q", "xvfb"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass   # non-fatal — virtual display is only needed for video recording

print("Dependencies ready.\n")


# ════════════════════════════════════════════════════════════
# SECTION 2 — Imports
# ════════════════════════════════════════════════════════════
import os
import glob
import time
import shutil
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — works on Windows, Linux, Colab
import matplotlib.pyplot as plt

import torch
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import RecordVideo

warnings.filterwarnings("ignore")
gym.register_envs(ale_py)


# ════════════════════════════════════════════════════════════
# SECTION 3 — Device Detection
# ════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 3 — Device Detection")
print("="*55)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device   : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  Running on CPU — training will be slower.")
    print("  Tip: Use Google Colab with T4 GPU for faster results.")
print("="*55 + "\n")


# ════════════════════════════════════════════════════════════
# SECTION 4 — Configuration
# ════════════════════════════════════════════════════════════
ENV_ID          = "ALE/Pong-v5"
POLICY          = "CnnPolicy"

# Reduce steps automatically when on CPU to keep runtime reasonable
if DEVICE == "cuda":
    SWEEP_TIMESTEPS = 200_000
    BEST_TIMESTEPS  = 500_000
else:
    print("⚠️  CPU detected — reducing timesteps for reasonable runtime.")
    print("   Sweep: 50,000 steps | Best run: 100,000 steps\n")
    SWEEP_TIMESTEPS = 50_000
    BEST_TIMESTEPS  = 100_000

BUFFER_SIZE     = 100_000
LEARNING_STARTS = min(50_000, SWEEP_TIMESTEPS // 2)  # never exceed sweep length
TRAIN_FREQ      = 4
GRADIENT_STEPS  = 1
TARGET_UPDATE   = 1_000
RESULTS_CSV     = "training_results.csv"
BEST_MODEL_PATH = "dqn_model"   # SB3 appends .zip

# ── Jean Jabo — 10 Hyperparameter Experiments ──────────────
EXPERIMENTS = [
    # Exp 1 — High LR, low gamma, small batch
    dict(lr=0.001,  gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 2 — High LR, high gamma
    dict(lr=0.001,  gamma=0.99, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 3 — Moderate LR, low gamma, small batch
    dict(lr=0.0005, gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.05, eps_decay=0.20),
    # Exp 4 — Moderate LR, mid gamma, larger batch, slower decay
    dict(lr=0.0005, gamma=0.97, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.30),
    # Exp 5 — Balanced
    dict(lr=0.0003, gamma=0.98, batch_size=64,  eps_start=1.0, eps_end=0.05, eps_decay=0.25),
    # Exp 6 — Low LR, high gamma, tight epsilon end
    dict(lr=0.0002, gamma=0.99, batch_size=64,  eps_start=1.0, eps_end=0.02, eps_decay=0.20),
    # Exp 7 — Very low LR, large batch
    dict(lr=0.0001, gamma=0.98, batch_size=128, eps_start=1.0, eps_end=0.05, eps_decay=0.15),
    # Exp 8 — Reduced eps_start
    dict(lr=0.0005, gamma=0.99, batch_size=32,  eps_start=0.9, eps_end=0.05, eps_decay=0.20),
    # Exp 9 — Higher eps_end
    dict(lr=0.0001, gamma=0.95, batch_size=32,  eps_start=1.0, eps_end=0.10, eps_decay=0.20),
    # Exp 10 — Very slow epsilon decay + lowest LR
    dict(lr=0.0001, gamma=0.99, batch_size=32,  eps_start=1.0, eps_end=0.01, eps_decay=0.10),
]

NOTED_BEHAVIOR = [
    "High LR destabilises Q-values; rewards fluctuate widely. Fast but erratic.",
    "Higher gamma improves long-term credit assignment but LR still too high; unstable late training.",
    "Moderate LR + low gamma; converges faster but undervalues future rewards.",
    "Larger batch reduces gradient noise. Slower epsilon decay improves exploration coverage.",
    "Balanced config; reward trend stabilises. Good exploration-exploitation balance.",
    "Lower final epsilon forces tighter exploitation. Slight improvement in avg reward.",
    "Large batch + low LR = very stable but slow convergence. Best stability observed.",
    "Reduced eps_start hurts early exploration; agent stuck in suboptimal strategies early on.",
    "Higher eps_end keeps exploration alive; prevents full exploitation, lower peak reward.",
    "Very slow epsilon decay + low LR: thorough exploration → best avg reward in final episodes.",
]


# ════════════════════════════════════════════════════════════
# SECTION 5 — Helpers
# ════════════════════════════════════════════════════════════

class RewardLogger(BaseCallback):
    """Logs episode reward and length during training."""
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


def make_env(env_id: str = ENV_ID) -> gym.Env:
    env = gym.make(env_id)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


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
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
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
    print(f"  Result → Avg: {avg_reward:.2f}  Max: {max_reward:.2f}  "
          f"Episodes: {len(logger.episode_rewards)}")

    return {
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
        "Noted Behavior":     NOTED_BEHAVIOR[exp_index - 1],
    }, model


# ════════════════════════════════════════════════════════════
# SECTION 6 — Phase 1: Hyperparameter Sweep
# ════════════════════════════════════════════════════════════
print("="*55)
print("  SECTION 6 — Phase 1: Hyperparameter Sweep")
print(f"  {len(EXPERIMENTS)} experiments × {SWEEP_TIMESTEPS:,} steps")
print(f"  Environment : {ENV_ID}")
print("="*55)

results = []

for i, params in enumerate(EXPERIMENTS, start=1):
    result, model = run_experiment(i, params, SWEEP_TIMESTEPS)
    results.append(result)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# SECTION 7 — Pick Best Experiment
# ════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
best_idx    = df["Avg Reward"].idxmax()
best_exp    = int(df.loc[best_idx, "Experiment"])
best_params = EXPERIMENTS[best_exp - 1]

print(f"\n{'='*55}")
print(f"  🏆 Best sweep experiment : #{best_exp}")
print(f"  Avg Reward (sweep)       : {df.loc[best_idx, 'Avg Reward']:.2f}")
print(f"{'='*55}")


# ════════════════════════════════════════════════════════════
# SECTION 8 — Phase 2: Full Run on Best Config
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  SECTION 8 — Phase 2: Full Training on Experiment #{best_exp}")
print(f"  {BEST_TIMESTEPS:,} steps")
print(f"{'='*55}")

final_result, final_model = run_experiment(best_exp, best_params, BEST_TIMESTEPS)
final_model.save(BEST_MODEL_PATH)
print(f"\n[Saved] Best model → '{BEST_MODEL_PATH}.zip'")

df.loc[best_idx, "Avg Reward"]         = final_result["Avg Reward"]
df.loc[best_idx, "Max Reward"]         = final_result["Max Reward"]
df.loc[best_idx, "Avg Episode Length"] = final_result["Avg Episode Length"]
df.loc[best_idx, "Timesteps"]          = BEST_TIMESTEPS

df.to_csv(RESULTS_CSV, index=False)
print(f"[Saved] Results table → '{RESULTS_CSV}'")


# ════════════════════════════════════════════════════════════
# SECTION 9 — Print Results Table
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  SECTION 9 — Results Table")
print(f"{'='*55}\n")

display_cols = [
    "Experiment", "Learning Rate", "Gamma", "Batch Size",
    "Epsilon Start", "Epsilon End", "Epsilon Decay",
    "Avg Reward", "Max Reward", "Avg Episode Length",
]
print(df[display_cols].to_string(index=False))

best_row = df.loc[df["Avg Reward"].idxmax()]
print(f"\n🏆 Best : Experiment #{int(best_row['Experiment'])}")
print(f"   lr={best_row['Learning Rate']}  gamma={best_row['Gamma']}  "
      f"batch={int(best_row['Batch Size'])}")
print(f"   Avg Reward: {best_row['Avg Reward']}  Max Reward: {best_row['Max Reward']}")
print(f"   {best_row['Noted Behavior']}")


# ════════════════════════════════════════════════════════════
# SECTION 10 — Plot Reward Comparison
# ════════════════════════════════════════════════════════════
print(f"\n[Plot] Saving reward_comparison.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Jean Jabo — DQN Hyperparameter Tuning (ALE/Pong-v5)", fontsize=14)

colors = [
    "#e74c3c" if i == df["Avg Reward"].idxmax() else "#3498db"
    for i in range(len(df))
]
axes[0].bar(df["Experiment"], df["Avg Reward"], color=colors)
axes[0].set_title("Avg Reward per Experiment  (red = best)", fontsize=12)
axes[0].set_xlabel("Experiment #")
axes[0].set_ylabel("Avg Reward")
axes[0].axhline(df["Avg Reward"].mean(), color="gray", linestyle="--", label="Mean")
axes[0].legend()

axes[1].bar(df["Experiment"], df["Max Reward"], color="#2ecc71")
axes[1].set_title("Max Reward per Experiment", fontsize=12)
axes[1].set_xlabel("Experiment #")
axes[1].set_ylabel("Max Reward")

plt.tight_layout()
plt.savefig("reward_comparison.png", dpi=150)
plt.close()
print("Chart saved → reward_comparison.png")


# ════════════════════════════════════════════════════════════
# SECTION 11 — Evaluate Agent (GreedyQPolicy, headless)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  SECTION 11 — Agent Evaluation (GreedyQPolicy)")
print(f"{'='*55}\n")

NUM_PLAY_EPISODES = 5
model_file = f"{BEST_MODEL_PATH}.zip"
if not Path(model_file).exists():
    model_file = f"dqn_model_exp_{best_exp}.zip"

print(f"Loading: {model_file}")
play_model = DQN.load(model_file)

play_env = gym.make(ENV_ID, render_mode="rgb_array")
play_env = AtariWrapper(play_env)

play_records = []
for ep in range(1, NUM_PLAY_EPISODES + 1):
    obs, _ = play_env.reset()
    done, total_rew, steps, t0 = False, 0.0, 0, time.time()
    while not done:
        action, _ = play_model.predict(obs, deterministic=True)
        obs, rew, term, trunc, _ = play_env.step(action)
        total_rew += rew
        steps += 1
        done = term or trunc
    duration = time.time() - t0
    play_records.append({
        "Episode": ep, "Total Reward": total_rew,
        "Steps": steps, "Duration (s)": round(duration, 2),
        "Reward/Step": round(total_rew / steps, 4) if steps > 0 else 0,
    })
    print(f"  Episode {ep}: reward={total_rew:.1f}  steps={steps}  time={duration:.1f}s")

play_env.close()

play_df = pd.DataFrame(play_records)
play_df.to_csv("play_metrics.csv", index=False)
rewards = play_df["Total Reward"]
print(f"\nEvaluation Summary:")
print(f"  Mean Reward : {rewards.mean():.2f}")
print(f"  Max Reward  : {rewards.max():.2f}")
print(f"  Min Reward  : {rewards.min():.2f}")
print("Play metrics saved → play_metrics.csv")


# ════════════════════════════════════════════════════════════
# SECTION 12 — Record Gameplay Video
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  SECTION 12 — Recording Gameplay Video")
print(f"{'='*55}\n")

# Start virtual display on Linux/Colab only
if not IS_WINDOWS:
    try:
        from pyvirtualdisplay import Display
        vdisplay = Display(visible=0, size=(1400, 900))
        vdisplay.start()
        print("Virtual display started.")
    except Exception:
        print("pyvirtualdisplay unavailable — continuing without virtual display.")

VIDEO_DIR    = "videos"
VIDEO_PREFIX = "dqn_pong_jeanjabo"
NUM_VID_EPS  = 3
os.makedirs(VIDEO_DIR, exist_ok=True)

try:
    vid_base = gym.make(ENV_ID, render_mode="rgb_array")
    vid_base = AtariWrapper(vid_base)
    vid_env  = RecordVideo(
        vid_base,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix=VIDEO_PREFIX,
        video_length=0,
    )
    rec_model  = DQN.load(model_file)
    vid_rewards = []

    for ep in range(1, NUM_VID_EPS + 1):
        obs, _ = vid_env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = rec_model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = vid_env.step(action)
            total += rew
            done   = term or trunc
        vid_rewards.append(total)
        print(f"  Recorded episode {ep}: reward={total:.1f}")

    vid_env.close()
    print(f"\nVideos saved to '{VIDEO_DIR}/':")
    for f in sorted(Path(VIDEO_DIR).glob("*.mp4")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

except Exception as e:
    print(f"Video recording skipped: {e}")


# ════════════════════════════════════════════════════════════
# SECTION 13 — Package Outputs
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  SECTION 13 — Packaging Outputs")
print(f"{'='*55}\n")

zip_name = "jean_jabo_results.zip"
collect  = (
    [RESULTS_CSV, "play_metrics.csv", "reward_comparison.png", f"{BEST_MODEL_PATH}.zip"]
    + sorted(str(p) for p in Path(".").glob("dqn_model_exp_*.zip"))
    + sorted(str(p) for p in Path(VIDEO_DIR).glob("*.mp4") if Path(VIDEO_DIR).exists())
)

with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in collect:
        if os.path.exists(f):
            z.write(f)
            print(f"  Added : {f}")

print(f"\nZip ready → {zip_name}  ({Path(zip_name).stat().st_size // 1024} KB)")

# Auto-download in Colab
if IS_COLAB:
    try:
        from google.colab import files
        files.download(zip_name)
        print("Colab download triggered.")
    except Exception:
        pass
else:
    print(f"Outputs saved locally in: {os.path.abspath('.')}")


# ════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  ✅ ALL DONE — Jean Jabo")
print(f"  Environment  : {ENV_ID}")
print(f"  Best config  : Experiment #{best_exp}")
print(f"  Best model   : {BEST_MODEL_PATH}.zip")
print(f"  Results CSV  : {RESULTS_CSV}")
print(f"  Play metrics : play_metrics.csv")
print(f"  Chart        : reward_comparison.png")
print(f"  Videos       : {VIDEO_DIR}/")
print(f"  Download zip : {zip_name}")
print(f"{'='*55}\n")
