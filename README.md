# Gradient Asymmetry and Activation Saturation in Actor-Critic Networks

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **⚠️ Ongoing Research**: This is an active research project. Results and discussions are preliminary and subject to change. Comments, contributions, and collaborations are welcome! Please open an issue or reach out if you'd like to contribute.

## Motivation

During PhD research on the [CCM-MADRL algorithm](https://doi.org/10.1145/3768579) for mobile edge computing, we observed an unexpected phenomenon: **actor networks stopped updating their weights early in training** while critic networks continued learning throughout. This asymmetry was highly sensitive to learning rate configurations.

![Stopping Episodes from Thesis](paper/figures/fig0_thesis_stopping.png)

*Figure: Stopping episodes across learning rate configurations. Each cell shows when actors stopped updating (darker = earlier stopping). Only specific learning rate combinations (bottom-left) allowed training to complete.*

**Only 1 out of 16 learning rate combinations converged.** This means 93.75% of hyperparameter configurations failed—not due to suboptimal learning, but because actors stopped updating entirely. Preventing activation saturation would make many more combinations viable, dramatically reducing hyperparameter search costs and saving compute resources.

This observation motivated a deeper investigation into **why** this happens and **how** to prevent it.

## Overview

This repository contains the research paper, experimental code, and analysis for investigating **gradient asymmetry** between actor and critic networks in actor-critic reinforcement learning architectures. We identify **activation saturation** (specifically tanh saturation in actor output layers) as the root cause of premature actor convergence.

### Key Finding

Actor networks stop updating their weights early in training while critic networks continue learning. This asymmetry is caused by:

1. **Actor**: Uses tanh output activation → saturates at ±1 → vanishing gradients
2. **Critic**: Uses linear output → no saturation → healthy gradient flow

| Actor LR | Critic LR | Actor Stops At | Converged? |
|----------|-----------|----------------|------------|
| 0.1 | any | ~5 episodes | No |
| 0.01 | any | ~5 episodes | No |
| 0.0001 | 0.001 | >2000 episodes | **Yes** |

This phenomenon is **not specific to multi-agent RL**—it applies to any actor-critic architecture using bounded output activations (tanh, sigmoid) for the actor.

## Repository Structure

```
gradient_asymmetry/
├── README.md                    # This file
├── CLAUDE.md                    # Development guidelines
├── requirements.txt             # Python dependencies
│
├── paper/                       # LaTeX paper
│   ├── main.tex                # Main paper file
│   ├── main.pdf                # Compiled PDF
│   ├── references.bib          # Bibliography
│   ├── sections/               # Paper sections
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── background.tex
│   │   ├── methodology.tex
│   │   ├── analysis.tex
│   │   ├── discussion.tex
│   │   └── conclusion.tex
│   └── figures/                # Paper figures
│
└── ColabExperiments/           # Experiment code (Google Colab compatible)
    │
    ├── # Baseline experiments
    ├── colab_notebook.ipynb           # Original experiment notebook
    ├── run_stopping_experiment_colab.py
    │
    ├── # Mitigation experiments
    ├── large_actor_experiment/        # Larger actor network (512→128)
    ├── layernorm_experiment/          # LayerNorm before tanh
    ├── layernorm_experiment_continued/
    ├── fullnorm_experiment/           # Full normalization
    ├── linear_activation_experiment/  # Linear hidden activations
    │
    ├── # Experiment results
    ├── large_actor_results/
    ├── layernorm_results/
    ├── linear_activation_results/
    │
    ├── # Analysis & visualization
    ├── generate_all_figures.py
    ├── plot_analysis.py
    └── figures/
```

## Experiments

### Baseline: Stopping Episode Detection

Tracks when actor gradients vanish across different learning rate configurations.

### Mitigation Strategies Under Investigation

| Experiment | Hypothesis | Status |
|------------|-----------|--------|
| **Large Actor** | More parameters → more gradient paths | In progress |
| **LayerNorm** | Normalize pre-activations → prevent saturation | In progress |
| **Linear Activations** | Remove ReLU → allow negative flow | In progress |
| **Full Normalization** | LayerNorm on all layers | In progress |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/TesfayZ/gradient_asymetry_AND_activation_saturation.git
cd gradient_asymetry_AND_activation_saturation
pip install -r requirements.txt
```

### 2. Run Experiments (Google Colab)

Upload the zip files from `ColabExperiments/` to Google Colab:
- `layernorm_experiment.zip` - LayerNorm mitigation
- `large_actor_experiment.zip` - Large actor network
- `linear_activation_experiment.zip` - Linear activations
- `fullnorm_experiment.zip` - Full normalization

Each contains a Jupyter notebook with experiment code.

### 3. Compile Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Root Cause Analysis

### The Saturation Mechanism

```
Actor output: a = tanh(z)     where z = W·h + b

When |z| > 2:
  tanh(z) ≈ ±1
  tanh'(z) = 1 - tanh²(z) ≈ 0

→ Gradients vanish at output layer
→ No weight updates propagate backward
→ Actor "stops learning"
```

### Why Critics Don't Stop

```
Critic output: Q = W·h + b    (linear, no activation)

∂Q/∂W = h                     (always non-zero)

→ Gradients flow regardless of output magnitude
→ Critic continues learning
```

## Citation

```bibtex
@article{gradient_asymmetry2025,
  title={Gradient Asymmetry and Activation Saturation in Actor-Critic Networks},
  author={Gebrekidan, Tesfay Zemuy and others},
  journal={arXiv preprint},
  year={2025}
}
```

## Related Work

This research builds on the CCM-MADRL algorithm:

```bibtex
@article{ccm_madrl,
  title={Client-Master Multiagent Deep Reinforcement Learning
         for Task Offloading in Mobile Edge Computing},
  author={Gebrekidan, Tesfay Zemuy and Shojafar, Mohammad and
          Pooranian, Zahra and Persico, Valerio and Pescapé, Antonio},
  journal={ACM Transactions on Autonomous and Adaptive Systems},
  year={2024},
  doi={10.1145/3768579}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
