# Gradient Asymmetry and Activation Saturation in Actor-Critic Networks

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Ongoing Research**: Results and discussions are preliminary and subject to change.

> **Continuation of [CCM_MADRL_MEC](https://github.com/TesfayZ/CCM_MADRL_MEC)** — analyzing gradient asymmetry and activation saturation observed in that work.

## Motivation

During PhD research on the [CCM-MADRL algorithm](https://doi.org/10.1145/3768579) for mobile edge computing, we observed that **actor networks stopped updating their weights early in training** while critic networks continued learning. **Only 1 out of 16 learning rate combinations converged** — 93.75% of configurations failed because actors stopped updating entirely due to activation saturation.

![Stopping Episodes from Thesis](paper/figures/fig0_thesis_stopping.png)
*Stopping episodes across learning rate configurations. Darker = earlier stopping. Only the bottom-left corner allowed training to complete.*

## Key Finding

| Component | Output Activation | Gradient Behavior | Stops? |
|-----------|------------------|-------------------|--------|
| **Actor** | tanh (bounded) | Vanishes when saturated | Yes |
| **Critic** | linear (unbounded) | Always flows | No |

High actor learning rates (0.01–0.1) cause tanh saturation within 161–247 episodes. Conservative rates (0.0001) maintain gradient flow throughout 2000 episodes. We measure a **4–8 order of magnitude** gradient asymmetry between actors and critics.

## Repository Structure

```
gradient_asymmetry/
├── paper/                          # LaTeX paper
│   ├── main.tex / main.pdf
│   ├── sections/                   # intro, background, methodology, analysis, discussion, conclusion
│   └── figures/
│
└── ColabExperiments/               # Google Colab experiments
    ├── original_experiment/        # Baseline (seed=42, all 16 LR configs)
    ├── large_actor_experiment/     # 512→128 actor (28× more params)
    ├── layernorm_experiment/       # LayerNorm before tanh output
    ├── linear_activation_experiment/ # Linear hidden activations
    ├── gradient_clipping_experiment/ # Gradient clipping (max_norm=1.0)
    ├── fullnorm_experiment/        # Full normalization (cancelled)
    └── plot_*.py, figures/         # Analysis scripts and plots
```

## Mitigation Experiments

| Experiment | Result |
|------------|--------|
| **Large Actor** (512→128) | Marginal (5–28 episodes delay). Saturation is architectural, not capacity-limited. |
| **LayerNorm** | In progress |
| **Linear Activations** | Failed — pre-activation explosion 17× worse (±18M vs ±1M with ReLU) |
| **Gradient Clipping** (norm=1.0) | Partially effective — helped 2 critic-LR=0.1 configs (8/16 vs 6/16 reaching full training), but cannot fix vanishing actor gradients |
| **Full Normalization** | Cancelled — training prohibitively slow |

## Reproducibility

All experiments use **seed=42** (PyTorch, NumPy, Python random, CUDA, environment). An earlier unseeded run showed the same structural patterns (identical stopping behavior, same 6/16 convergence rate) but with higher reward variance.

## Quick Start

```bash
git clone https://github.com/TesfayZ/gradient_asymetry_AND_activation_saturation.git
cd gradient_asymetry_AND_activation_saturation
pip install -r requirements.txt
```

**Run experiments** — upload zip files from experiment directories to Google Colab. Each contains a Jupyter notebook.

**Compile paper:**
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@article{gradient_asymmetry2025,
  title={Gradient Asymmetry and Activation Saturation in Actor-Critic Networks},
  author={Gebrekidan, Tesfay Zemuy and others},
  year={2025}
}
```

**Related:** [CCM-MADRL](https://doi.org/10.1145/3768579) (Gebrekidan, Stein, Norman — ACM TAAS 2025)

## License

MIT License — see [LICENSE](LICENSE) for details.
