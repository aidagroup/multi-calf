# Multi-CALF: Critic as a Lyapunov Function

This repository contains the demo implementation of the Multi-CALF (Critic as a Lyapunov Function) approach described in the paper [TODO PASTE PAPER NAME HERE].

## Project Overview

Multi-CALF is a reinforcement learning approach that uses the critic function as a Lyapunov function to ensure stability and safety during training and deployment. This repository provides a demonstration of how Multi-CALF works in practice.

We have constructed a visual pendulum environment specifically for this demonstration, which provides image-based observations of a pendulum system. This allows for testing the Multi-CALF approach in a setting where the agent must learn from high-dimensional visual inputs rather than low-dimensional state representations.

## Repository Structure

```
multi-calf/
├── run/                      # Scripts for running experiments
│   ├── train_ppo_vispendulum.py  # Training script for PPO on visual pendulum
│   ├── eval.py               # Evaluation script
│   ├── artifacts/            # Saved model artifacts
│   └── mlruns/               # MLflow experiment tracking data
├── src/                      # Source code
│   ├── envs/                 # Environment implementations
│   │   ├── visual_pendulum.py  # Visual pendulum environment
│   │   └── assets/           # Visual assets for environments
│   ├── wrapper/              # Environment wrappers
│   │   ├── calf_wrapper.py   # CALF wrapper implementation
│   │   ├── common_wrapper.py # Common wrapper utilities
│   │   └── multicalf.py      # Multi-CALF implementation
│   ├── utils/                # Utility functions
│   │   └── mlflow.py         # MLflow utilities
│   └── model.py              # Neural network model definitions
├── pyproject.toml            # Project dependencies and configuration
└── uv.lock                   # Lock file for uv package manager
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. uv is a fast, reliable Python package installer and resolver.

### Installing uv

To install uv, run:

```bash
curl -sSf https://astral.sh/uv/install.sh | bash
```

Or on Windows:

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setting up the environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-calf.git
   cd multi-calf
   ```

2. Create a virtual environment with Python 3.13.2:
   ```bash
   uv venv --python=3.13.2
   uv sync
   ```

## Running Experiments

### Training

To train a PPO agent on the visual pendulum environment:

```bash
uv run run/train_ppo_vispendulum.py
```

You can customize the training parameters by modifying the arguments:

```bash
uv run run/train_ppo_vispendulum.py --seed 42 --total-timesteps 500000 --learning-rate 3e-4
```

For the full details about training parameters type
```bash
uv run run/train_ppo_vispendulum.py --help
```

### Evaluation

To evaluate a trained model:

```bash
uv run run/eval.py --checkpoint-path run/artifacts/path/to/your/checkpoint
```

## Tracking Experiments

This project uses MLflow for experiment tracking. You can view the experiment results by running:

```bash
cd run
mlflow ui --port=5000
```

Then navigate to http://localhost:5000 in your web browser.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use this code in your research, please cite:

```
[TODO: Add citation information for the paper]
