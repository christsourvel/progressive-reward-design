# UAV Navigation using Reinforcement Learning: A Systematic Approach to Progressive Reward Function Design

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation code for the paper "UAV Navigation using Reinforcement Learning: A Systematic Approach to Progressive Reward Function Design". The code provides three reward configurations (Goal-Based, Heuristic, Waypoint) for systematic comparison of reward structures in fixed-wing UAV path-following tasks using reinforcement learning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Results](#reproducing-results)
- [Cite Our Work](#cite-our-work)
- [License](#license)
- [Contact](#contact)

## Overview

This repository implements three reward configurations for fixed-wing UAV path following as described in the paper:

- **Waypoint Configuration**: Sequential checkpoint system with progressive rewards
- **Goal-Based Configuration**: Distance shaping with path penalties
- **Heuristic Configuration**: Heuristic reward structure from our previous work

Environments include 2D sine-wave, 2D straight-line, and 3D maneuvers. RL algorithms supported: PPO, SAC, TD3.

## Project Structure

```
progressive-reward-design/
├── 2d_env/                      # 2D sine-wave environment
│   ├── env.py                   # Environment implementation
│   ├── training.py              # Training script
│   ├── validation.py            # Validation script
│   └── validate_unseen_paths.py # Generalization testing
├── 2d_straight_line_env/        # 2D straight-line environment
│   ├── env.py
│   ├── training.py
│   └── validation.py
├── 3d_env/                      # 3D complex maneuvers
│   ├── env.py                   # 3D environment with altitude control
│   ├── training.py
│   ├── validation.py
│   └── validate_unseen_paths.py # 3D unseen paths (helix, descending-S)
├── LICENSE                      # Apache 2.0 License
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/christsourvel/progressive-reward-design.git
   cd progressive-reward-design
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

**Train models:**

```bash
cd 2d_env  # or 2d_straight_line_env or 3d_env
python training.py
```

**Validate models:**

```bash
python validation.py
```

**Test generalization on unseen paths:**

```bash
python validate_unseen_paths.py
```

## Reproducing Results

### Training

All environments support training with PPO, SAC, and TD3 on the three reward configurations. Models are saved to `trained_models/` with VecNormalize statistics for proper evaluation.

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

### Validation

Standard validation generates:

- Episode trajectory plots
- Control signal analysis
- Performance metrics (success rate, path deviation, oscillation index)
- Algorithm comparison plots

### Unseen Path Testing

Tests trained agents on novel geometries (circles, figure-8s, spirals, helixes) to evaluate generalization as reported in the paper.

## Cite Our Work

**Paper Status**: Under review

If you use this code, please reference:

```
Christos Tsourveloudis. "UAV Navigation using Reinforcement Learning: A Systematic Approach to Progressive Reward Function Design"
Implementation available at: https://github.com/christsourveloudis/progressive-reward-design
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```text
Copyright 2025 Christos Tsourveloudis
```

## Contact

For questions or issues:

- GitHub Issues: [progressive-reward-design/issues](https://github.com/christsourvel/progressive-reward-design/issues)
