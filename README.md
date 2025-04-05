# Multi-CALF: Critic as a Lyapunov Function

This repository implements the Multi-CALF (Critic As a Lyapunov Function) approach for reinforcement learning, ensuring stability and safety during training and deployment.

## Overview

Multi-CALF is a technique that leverages the critic function as a Lyapunov function to guarantee stability in reinforcement learning agents. The approach facilitates safe exploration and robust policy development by using multiple critics to validate actions before execution.

Key features:
- Safety-aware training with Lyapunov stability guarantees
- Environmentally-conscious action selection
- Support for visual observations in complex control tasks
- Compatible with standard RL algorithms (PPO, TD3)
- Performance tracking via MLflow

## Repository Structure

```
multi-calf/
├── run/                      # Experiment scripts
│   ├── train_ppo.py          # Training script for PPO
│   ├── train_td3.py          # Training script for TD3
│   ├── eval.py               # Standard evaluation script
│   ├── eval_mcalf.py         # Multi-CALF evaluation script
│   ├── artifacts/            # Saved model artifacts
│   └── mlruns/               # MLflow experiment tracking data
├── src
├── pyproject.toml            # Dependencies and configuration
└── uv.lock                   # Lock file for uv package manager
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management - a fast and reliable Python package installer.

### Prerequisites

For rendering functionality, install the following dependencies:

```bash
sudo apt-get install -y libosmesa6-dev libgl1-mesa-dev libglfw3
```

### Setting up with UV

1. **Install UV**:

   ```bash
   # Linux/macOS
   curl -sSf https://astral.sh/uv/install.sh | bash

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/multi-calf.git
   cd multi-calf
   ```

3. **Create and activate the environment**:

   ```bash
   # Create a virtual environment with specific Python version
   uv venv --python=3.12
   
   # Activate the environment
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   
   # Install dependencies from lock file
   uv sync
   ```

## Running Experiments with UV

All experiments can be run using UV's `run` command to ensure proper environment resolution.

### Training

To train a PPO agent:

```bash
uv run run/train_ppo.py 
```

To train a TD3 agent:

```bash
uv run run/train_td3.
```

Command-line parameters provide extensive customization:

```bash
# View all available options
uv run run/train_ppo.py --help
```

### Multi-CALF Evaluation

To evaluate using the Multi-CALF approach (requires two trained models):

```bash
uv run run/eval_mcalf.py \
  --env-id Pendulum-v1 \
  --base-checkpoint-path run/artifacts/ppo_checkpoints/ppo_checkpoint_500000_steps.zip \
  --alt-checkpoint-path run/artifacts/ppo_checkpoints/ppo_checkpoint_250000_steps.zip \
  --mcalf.relaxprob-init 1.0 \
  --mcalf.relaxprob-factor 0.999 \
  --mcalf.calf-change-rate 0.01
```

## Experiment Tracking

View experiment results using MLflow:

```bash
cd run
mlflow ui --port=5000
```

Then navigate to http://localhost:5000 in your browser.

## Troubleshooting UV

If you encounter issues with UV:

1. **Cache problems**: Clear the UV cache
   ```bash
   uv cache clean
   ```

2. **Package resolution issues**: Update UV and retry
   ```bash
   uv self update
   uv sync
   ```

3. **Environment conflicts**: Create a fresh environment
   ```bash
   rm -rf .venv
   uv venv --python=3.12
   uv sync
   ```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
[Citation information for the paper]