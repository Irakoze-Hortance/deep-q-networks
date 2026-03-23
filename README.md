# Pong DQN - Atari Reinforcement Learning

## Overview

This project trains a Deep Q-Network (DQN) agent to play Pong using Stable Baselines 3 and Gymnasium. We compare CnnPolicy and MlpPolicy architectures across 10 hyperparameter experiments per member and evaluate the best model using a greedy policy.

## Environment

- **Game:** Pong (ALE/Pong-v5)
- **Observation space:** 84x84x4 stacked grayscale frames (via AtariWrapper)
- **Action space:** 6 discrete actions
- **Framework:** Stable Baselines 3 + Gymnasium

## Setup and Installation

This project requires Python 3.10 or higher. It is recommended to use a virtual environment or a conda environment to avoid dependency conflicts.

### 1. Clone the Repository

```bash
git clone https://github.com/Irakoze-Hortance/deep-q-networks.git
cd deep-q-networks
```

### 2. Create and Activate a Virtual Environment

Using conda:
```bash
conda create -n dqn-atari python=3.10 
conda activate dqn-atari
```

Or using venv:
```bash
python -m venv dqn-atari
source dqn-atari/bin/activate        # Linux/Mac
dqn-atari\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py torch numpy
```

To verify your GPU is available for training:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True` if you have a CUDA-enabled GPU. If it prints `False`, training will fall back to CPU automatically but will be significantly slower.

### 4. Verify the Atari Environment

```bash
python -c "import ale_py; import gymnasium as gym; ale_py; gym.register_envs(ale_py); env = gym.make('ALE/Pong-v5'); print('Environment ready')"
```

---

## How to Train

Prepare your experiments CSV file with the hyperparameter configurations you want to test, then run:

```bash
python train.py --experiments-file experiments.csv --output-dir results/
```

The script will print `Training device: cuda (GPU_NAME)` or `Training device: cpu` at startup so you can confirm which hardware is being used before the run begins.

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--experiments-file` | required | Path to CSV with hyperparameter rows |
| `--output-dir` | `.` | Directory to save models and results |
| `--start-from` | `1` | Experiment number to resume from |

The CSV file must contain these columns: `policy`, `lr`, `gamma`, `batch_size`, `epsilon_start`, `epsilon_end`, `epsilon_decay`.

After training completes the following files are saved to the output directory:

- `dqn_model.zip` - the best performing model across all experiments
- `dqn_model_exp_N.zip` - individual model for each experiment N
- `training_results.csv` - full results table with all metrics
- `best_experiment.csv` - single row summary of the best experiment

---

## How to Play

Load the best trained model and watch the agent play in real time:

```bash
python play.py --model-path dqn_model.zip --env-id ALE/Pong-v5 --policy CnnPolicy --episodes 10
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `dqn_model.zip` | Path to the trained model zip file |
| `--env-id` | `ALE/Pong-v5` | Gymnasium environment ID |
| `--policy` | `CnnPolicy` | Must match the policy used during training |
| `--episodes` | `5` | Number of episodes to play |
| `--seed` | `42` | Random seed for reproducibility |

Note: `play.py` opens a native game window using `render_mode="human"`. This requires a display and will not work on headless cloud environments like Kaggle or Google Colab. Run it locally after downloading `dqn_model.zip` from your training environment.

---

## Gameplay Video


[![Pong Agent Gameplay](https://img.youtube.com/vi/dWOCqOm4ej8/0.jpg)](https://www.youtube.com/watch?v=dWOCqOm4ej8)

---

## Hyperparameter Tuning Results

Each group member independently ran 10 experiments varying the learning rate, gamma, batch size, and epsilon parameters. The table below documents each configuration and the observed behavior.


### Member 1: Hortance Irakoze

| Experiment | Policy | lr | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Eval Std Reward | Is Best |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.001 | 0.95 | 32 | 1.0 | 0.10 | 0.20 | 0.000000 | False |
| 2 | CnnPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.10 | 0.20 | 0.000000 | False |
| 3 | CnnPolicy | 0.0005 | 0.95 | 32 | 1.0 | 0.10 | 0.20 | 0.663325 | False |
| 4 | CnnPolicy | 0.0005 | 0.97 | 64 | 1.0 | 0.10 | 0.20 | 1.813836 | False |
| 5 | CnnPolicy | 0.0003 | 0.98 | 64 | 1.0 | 0.10 | 0.20 | 1.374773 | False |
| 6 | CnnPolicy | 0.0002 | 0.99 | 64 | 1.0 | 0.10 | 0.20 | 1.688194 | True |
| 7 | CnnPolicy | 0.0001 | 0.98 | 128 | 1.0 | 0.10 | 0.20 | 2.118962 | False |
| 8 | CnnPolicy | 0.0005 | 0.99 | 32 | 1.0 | 0.10 | 0.20 | 1.833030 | False |
| 9 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 | 0.10 | 0.20 | 0.979796 | False |
| 10 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 0.20 | 0.900000 | False |

### Member 2: Idara Essien

| Experiment | Policy | lr | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Eval Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.0003 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -2.1 | Best result. Moderate lr with small batch produced the most consistent learning. Agent won several points per game. |
| 2 | CnnPolicy | 0.0005 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -6.9 | Slightly higher lr hurt performance compared to 0.0003. Agent less stable during training. |
| 3 | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -10.5 | Low lr learned too slowly at 500k steps. Insufficient gradient updates to converge. |
| 4 | CnnPolicy | 0.0005 | 0.99 | 128 | 1.0 | 0.05 | 0.20 | -11.1 | Larger batch with higher lr underperformed. Reduced replay buffer diversity harmed learning. |
| 5 | CnnPolicy | 0.0003 | 0.95 | 128 | 1.0 | 0.02 | 0.25 | -6.3 | Lower gamma shortened planning horizon. Tighter epsilon end forced more exploitation earlier. |
| 6 | MlpPolicy | 0.0003 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -21.0 | No learning occurred. MLP cannot extract spatial features from flattened pixel observations. |
| 7 | MlpPolicy | 0.0005 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -21.0 | Changing lr had zero effect. Architecture is the bottleneck, not the hyperparameters. |
| 8 | MlpPolicy | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.20 | -21.0 | Same outcome as experiments 6 and 7. MLP stuck at minimum reward regardless of lr. |
| 9 | MlpPolicy | 0.0005 | 0.99 | 128 | 1.0 | 0.05 | 0.20 | -21.0 | Larger batch made no difference. MLP architecture is fundamentally unsuited for pixel input. |
| 10 | MlpPolicy | 0.0005 | 0.95 | 64 | 1.0 | 0.02 | 0.25 | -21.0 | Varying gamma and epsilon also had no effect. Confirms architecture as the sole bottleneck. |


### Member 3: Jean Jabo

| Experiment | Policy | lr | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Eval Mean Reward | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.001 | 0.95 | 32 | 1.0 | 0.05 | 0.2 | -20.82 | High LR destabilises Q-values; rewards fluctuate widely. Fast but erratic. |
| 2 | CnnPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 0.2 | -20.83 | Higher gamma improves long-term credit assignment but LR still too high; unstable late training. |
| 3 | CnnPolicy | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 0.2 | -20.68 | Moderate LR + low gamma; converges faster but undervalues future rewards. |
| 4 | CnnPolicy | 0.0005 | 0.97 | 64 | 1.0 | 0.05 | 0.3 | -19.57 | Best Result. Larger batch reduces gradient noise. Slower epsilon decay improves exploration coverage. |
| 5 | CnnPolicy | 0.0003 | 0.98 | 64 | 1.0 | 0.05 | 0.25 | -20.61 | Balanced config; reward trend stabilises. Good exploration-exploitation balance. |
| 6 | CnnPolicy | 0.0002 | 0.99 | 64 | 1.0 | 0.02 | 0.2 | -20.48 | Lower final epsilon forces tighter exploitation. Slight improvement in avg reward. |
| 7 | MlpPolicy | 0.0001 | 0.98 | 128 | 1.0 | 0.05 | 0.15 | -20.55 | Large batch + low LR = very stable but slow convergence. Best stability observed. |
| 8 | CnnPolicy | 0.0005 | 0.99 | 32 | 0.9 | 0.05 | 0.2 | -20.56 | Reduced eps_start hurts early exploration; agent stuck in suboptimal strategies early on. |
| 9 | MlpPolicy | 0.0001 | 0.95 | 32 | 1.0 | 0.1 | 0.2 | -20.77 | Higher eps_end keeps exploration alive; prevents full exploitation, lower peak reward. |
| 10 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | -20.77 | Very slow epsilon decay + low LR: thorough exploration → best avg reward in final episodes. |

---

## Key Findings

### CnnPolicy vs MlpPolicy

CnnPolicy achieved a best eval mean reward of -2.1 while every MlpPolicy experiment remained stuck at -21.0, the minimum possible score in Pong. Pong requires detecting the ball position, velocity, and paddle locations from pixel frames. Convolutional layers learn spatial filters that identify these features directly from the 84x84 image grid. MlpPolicy flattens the 84x84x4 observation into a 28,224-dimensional vector, discarding all spatial relationships between pixels. No amount of hyperparameter tuning can compensate for this architectural limitation, as confirmed by the fact that changing lr, batch size, gamma, and epsilon across experiments 6 to 10 produced identical -21.0 results every time.

### Learning Rate

lr=0.0003 was the optimal value across all CnnPolicy experiments, producing the best (-2.1) and third best (-6.3) results. lr=0.0001 was too conservative, failing to converge sufficiently within 500k steps. lr=0.0005 updated too aggressively, destabilizing the Q-value estimates and reducing performance.

### Batch Size

batch=64 consistently outperformed batch=128 when learning rate was held constant. Smaller batches sample more diverse transitions from the replay buffer per update, which reduces correlation between training samples and leads to more stable gradient updates.

### Gamma (Discount Factor)

gamma=0.99 outperformed gamma=0.95. Pong rewards sustained sequences of correct actions across many timesteps, so a longer planning horizon captures the true downstream value of each action more accurately.

### Best Configuration

| Parameter | Value |
|---|---|
| Policy | CnnPolicy |
| Learning Rate | 0.0003 |
| Gamma | 0.99 |
| Batch Size | 64 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.20 |
| Training Timesteps | 500,000 |
| Eval Mean Reward | -2.1 |

---

## Individual Contributions

Our workload was planned and divided using a shared Google Doc which can be viewed here: https://docs.google.com/document/d/1gi73VKvuZGYY7jZKym-AEzKgjUgPPz09BM5sdeTElTU/edit?usp=sharing     

### Hortance (Member 1) - Training Script: Core and Early Hyperparameters

| Area | Contributions |
|---|---|
| train.py | Environment setup and initialization, DQN agent configuration and initial structure, MlpPolicy training loop, reward logging system (RewardLogger class), episode length tracking, model saving functionality |
| Experiments | 10 hyperparameter experiments focusing on learning rate and gamma variations |
| Commits | Initial code structure, core functionality, bug fixes, one commit per experiment, inline documentation |

### Idara (Member 2) - Training Script: Advanced Features and Play Script Foundation

| Area | Contributions |
|---|---|
| train.py | CnnPolicy training loop, policy comparison logic (MLP vs CNN), epsilon-greedy exploration parameters |
| play.py | Model loading functionality, environment setup for evaluation, GreedyQPolicy configuration |
| Experiments | 10 hyperparameter experiments focusing on batch size and epsilon variations |
| Commits | Initial code structure, core functionality, bug fixes, one commit per experiment, inline documentation |

### Jean Jabo (Member 3) - Play Script: Visualization and Documentation

| Area | Contributions |
|---|---|
| play.py | Rendering and visualization, episode loop and action selection, performance metrics collection |
| Supporting files | Automated hyperparameter table generation, video recording script, README.md, requirements.txt, config files |
| Experiments | 10 hyperparameter experiments focusing on combined parameter variations |
| Commits | Initial code structure, core functionality, bug fixes, one commit per experiment, inline documentation |

---

## Repository Structure

```
.
├── train.py               # Training script with hyperparameter sweep
├── play.py                # Evaluation and rendering script
├── experiments.csv        # Hyperparameter configurations used for training
├── dqn_model.zip          # Best trained model (CnnPolicy, lr=0.0003)
├── training_results.csv   # Full results table from all 10 experiments
└── README.md              # This file
```