# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Title:** Gradient Asymmetry and Activation Saturation in Actor-Critic Networks

**Status:** Ongoing research - results and discussions are preliminary

**Continuation of:** [CCM_MADRL_MEC](https://github.com/TesfayZ/CCM_MADRL_MEC) - This repo focuses on analyzing gradient asymmetry and activation saturation observed in that work.

This research investigates why actor networks in actor-critic reinforcement learning stop updating their weights while critics continue learning. The root cause is identified as **tanh activation saturation** in actor output layers.

**Key Finding:** Only 1 out of 16 learning rate combinations converged in experiments. Preventing activation saturation could make many more configurations viable, reducing hyperparameter search costs.

## Repository Structure

```
gradient_asymmetry/
├── paper/                       # LaTeX paper (39 pages)
│   ├── main.tex                # Main paper file
│   ├── main.pdf                # Compiled PDF
│   ├── sections/               # Paper sections
│   └── figures/                # All figures (fig0-fig13)
│
├── ColabExperiments/           # Experiment code (Google Colab)
│   ├── layernorm_experiment/        # LayerNorm mitigation (ongoing)
│   ├── large_actor_experiment/      # Large actor test (preliminary)
│   ├── linear_activation_experiment/ # Linear activations (preliminary)
│   ├── fullnorm_experiment/         # Full normalization (ongoing)
│   ├── *_results/                   # Experiment outputs
│   └── *.zip                        # Packaged experiments for Colab
│
├── README.md                   # Project overview
└── requirements.txt            # Python dependencies
```

## Running Experiments

Experiments are designed to run on Google Colab with GPU:

```bash
# Upload zip file to Colab, then:
!unzip experiment_name.zip
%cd experiment_name
!python run_*_experiment.py
```

Each experiment tracks:
- Stopping episodes (when actors cease updating)
- Gradient magnitudes (actor vs critic)
- Activation saturation ratios
- Pre-activation statistics

## Key Concepts

**Gradient Asymmetry:** Critics have 4-8 orders of magnitude larger gradients than actors due to:
1. Loss function differences (MSE vs policy gradient)
2. Gradient path length (direct vs indirect)
3. Output activation (linear vs tanh)

**Activation Saturation:** When actor learning rates are too high:
- Pre-activation values explode (up to ±10^6)
- tanh outputs saturate at ±1
- Gradients vanish (tanh'(z) ≈ 0)
- Actors stop learning

**Threshold Convention:** Offload decision uses `actor_action[:, 0] >= 0` (tanh output range is [-1, 1], so 0 is the decision boundary)

## Architecture Details

**Actor Network:**
- Input: 7 (state per agent)
- Hidden: 64 → 32 (ReLU) or 512 → 128 (large actor)
- Output: 3 (tanh) - offload decision, compute, power

**Critic Network:**
- Input: 510 (joint state-action + personal state-action)
- Hidden: 512 → 128 (ReLU)
- Output: 1 (linear) - Q-value

**Training:** 2000 episodes, batch size 64, 50 agents

## Mitigation Experiments (Ongoing)

| Experiment | Hypothesis | Status |
|------------|-----------|--------|
| Large Actor | More parameters → prevent saturation | Preliminary |
| LayerNorm | Normalize pre-activations | In progress |
| Linear Activations | Remove ReLU from hidden layers | Preliminary |
| Full Normalization | LayerNorm on all layers | In progress |

## Paper Compilation

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Related Work

This research builds on the CCM-MADRL algorithm for MEC task offloading:
- ACM: doi.org/10.1145/3768579
- arXiv: 2402.11653
- Thesis: Gebrekidan 2024

## Dependencies

PyTorch, NumPy, Matplotlib, Pandas
