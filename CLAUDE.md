# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the implementation of **"Client-Master Multiagent Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing"** (ACM: doi.org/10.1145/3768579, arXiv:2402.11653).

**Language:** Python 3 with PyTorch
**Domain:** Multi-Agent Deep Reinforcement Learning (MADRL) for Mobile Edge Computing (MEC) task offloading optimization

## Running Experiments

```bash
# Run main CCM_MADRL algorithm (index 0-9 for multiple trials)
python run.py <index>

# Run benchmark algorithms
python Benchmarks_run.py <index>

# Generate evaluation plots (requires completed experiments)
python plot3.py
python plot4.py
python plotAtTraining.py
```

**HPC/SLURM execution:**
```bash
sbatch CCM_MADRL_MEC.SLURM   # Runs 10 experiments as array job
sbatch Benchmarks.SLURM       # Runs 10 benchmark experiments
```

**Conda environment:** `vCMaDRLMEC`

## Repository Structure

Three experiment directories with identical structure but different hyperparameters:
- `s10lre4e3CCM_MADRL_MEC/` - 10 steps per episode
- `s100lre4e3CCM_MADRL_MEC/` - 100 steps per episode, λ₁=λ₂=0.5
- `s100lre4e3CCM_MADRL_MECLambda5/` - 100 steps, λ₁=1, λ₂=5 (**recommended for analysis**)

Each contains:
- `run.py` - Main entry point
- `CCM_MADRL.py` - Core Client-Master MADRL implementation
- `mec_env.py` - MEC simulation environment
- `Model.py` - Actor/Critic neural network architectures
- `prioritized_memory.py` / `SumTree.py` - Prioritized Experience Replay
- `Benchmarks_*.py` - MADDPG benchmark variants
- `CSV/` - Output metrics (results/, AtTraining/, constraints/)
- `Figures/` - Generated plots

`Submitted_thesis/` contains the PhD thesis LaTeX source (reference material).

## Architecture

**Client-Master MADRL:**
1. **50 Client Agents** - Each has an actor network that outputs: offload decision, compute allocation, transmission power
2. **Central Master (Critic)** - Evaluates offload decisions, selects top-K candidates respecting server constraints (channels, storage)
3. **Prioritized Experience Replay** - SumTree-based priority sampling

**State per agent (7 dims):** transmission power, channel gain, available energy, task size, CPU cycles, deadline, device capability

**Action per agent (3 dims):** offload decision (tanh→threshold), compute allocation, power allocation

**Training:** 2000 episodes, batch size 64, memory capacity 10,000, epsilon decay 1.0→0.01

## Key Configuration (mec_env.py)

```python
LAMBDA_E = 0.5          # Energy cost weight
LAMBDA_T = 0.5          # Latency cost weight
MAX_STEPS = 10/100      # Steps per episode
K_CHANNEL = 10          # Server channels
N_UNITS = 8             # Server processing units
S_E = 400 MB            # Server storage
NUMBERofAGENTS = 50     # Fixed agent count
ENV_SEED = 37           # Reproducibility seed
```

## Known Issues

Energy model in `mec_env.py` has a scaling note (Issue #20) - current calculation doesn't use A_res² per standard E=k*C*f² model. Relative comparisons remain valid.

## Dependencies

PyTorch, NumPy, Matplotlib, Pandas, SciPy
