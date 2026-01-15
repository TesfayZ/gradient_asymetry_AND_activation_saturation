"""
Generate All Figures for Gradient Asymmetry Paper
From 16 experiments (4 actor LRs x 4 critic LRs)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Configuration
RESULTS_DIR = 'results'
OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIENT_LRS = [0.0001, 0.001, 0.01, 0.1]
MASTER_LRS = [0.0001, 0.001, 0.01, 0.1]

# Color scheme
COLORS = plt.cm.viridis(np.linspace(0, 1, 4))


def load_all_results():
    """Load all 16 experiment results."""
    all_results = []

    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        return all_results

    for exp_dir in os.listdir(RESULTS_DIR):
        if not exp_dir.startswith('actor_'):
            continue

        results_file = os.path.join(RESULTS_DIR, exp_dir, 'results.json')
        tracking_file = os.path.join(RESULTS_DIR, exp_dir, 'tracking_data.json')

        if os.path.exists(results_file):
            with open(results_file) as f:
                result = json.load(f)

            if os.path.exists(tracking_file):
                with open(tracking_file) as f:
                    result['tracking'] = json.load(f)

            all_results.append(result)
            print(f"Loaded: {exp_dir}")

    return all_results


def get_result(results, actor_lr, critic_lr):
    """Get result for specific LR combination."""
    for r in results:
        if abs(r['actor_lr'] - actor_lr) < 1e-6 and abs(r['critic_lr'] - critic_lr) < 1e-6:
            return r
    return None


# =============================================================================
# FIGURE 1: Stopping Episodes Heatmap
# =============================================================================
def plot_stopping_heatmap(results):
    """4x4 heatmap of stopping episodes."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Create matrix
    matrix = np.zeros((4, 4))
    for i, actor_lr in enumerate(CLIENT_LRS):
        for j, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r:
                matrix[i, j] = r['stopping_episode']

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2000)

    # Labels
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'{lr}' for lr in MASTER_LRS])
    ax.set_yticklabels([f'{lr}' for lr in CLIENT_LRS])
    ax.set_xlabel('Critic Learning Rate', fontsize=12)
    ax.set_ylabel('Actor Learning Rate', fontsize=12)
    ax.set_title('Stopping Episode by Learning Rate Configuration', fontsize=14)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = int(matrix[i, j])
            color = 'white' if val < 1000 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11, color=color, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Episodes Before Actors Stop', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_stopping_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_stopping_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig1_stopping_heatmap")


# =============================================================================
# FIGURE 2: Stopping Episodes Bubble Plot
# =============================================================================
def plot_stopping_bubble(results):
    """Bubble plot with size = stopping episode."""
    fig, ax = plt.subplots(figsize=(10, 8))

    actor_lrs = []
    critic_lrs = []
    stopping_eps = []

    for r in results:
        actor_lrs.append(r['actor_lr'])
        critic_lrs.append(r['critic_lr'])
        stopping_eps.append(r['stopping_episode'])

    # Bubble plot
    scatter = ax.scatter(actor_lrs, critic_lrs,
                        s=[s/3 for s in stopping_eps],  # Scale for visibility
                        c=stopping_eps, cmap='RdYlGn',
                        edgecolors='black', linewidths=1.5, alpha=0.8)

    # Add text labels
    for i in range(len(actor_lrs)):
        ax.annotate(str(stopping_eps[i]), (actor_lrs[i], critic_lrs[i]),
                   fontsize=9, ha='center', va='center', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Actor Learning Rate', fontsize=12)
    ax.set_ylabel('Critic Learning Rate', fontsize=12)
    ax.set_title('Stopping Episode for Different Learning Rate Combinations\n(Bubble size = episodes)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stopping Episode', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_stopping_bubble.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_stopping_bubble.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig2_stopping_bubble")


# =============================================================================
# FIGURE 3: Asymmetry Ratio Over Time
# =============================================================================
def plot_asymmetry_over_time(results):
    """Gradient asymmetry ratio over episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Asymmetry Ratio (Actor/Critic) Over Training', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        for cidx, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r and 'tracking' in r:
                asym = r['tracking'].get('asymmetry_history', [])
                if asym:
                    episodes = [a['episode'] for a in asym]
                    ratios = [a['ratio'] for a in asym]
                    # Clip extreme values for visualization
                    ratios = [min(r, 100) for r in ratios]
                    ax.plot(episodes, ratios, label=f'Critic LR={critic_lr}',
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Asymmetry Ratio')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal gradients')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_asymmetry_over_time.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_asymmetry_over_time.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig3_asymmetry_over_time")


# =============================================================================
# FIGURE 4: Actor vs Critic Gradient Norms
# =============================================================================
def plot_gradient_comparison(results):
    """Compare actor and critic gradient magnitudes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Actor vs Critic Gradient Magnitudes Over Training', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        for cidx, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r and 'tracking' in r:
                grad = r['tracking'].get('gradient_history', [])
                if grad:
                    episodes = [g['episode'] for g in grad]
                    actor_grads = [g['actor_grad'] for g in grad]
                    critic_grads = [g['critic_grad'] for g in grad]

                    ax.plot(episodes, actor_grads, linestyle='-',
                           label=f'Actor (c={critic_lr})', alpha=0.7, linewidth=1.5)
                    ax.plot(episodes, critic_grads, linestyle='--',
                           label=f'Critic (c={critic_lr})', alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_gradient_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_gradient_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig4_gradient_comparison")


# =============================================================================
# FIGURE 5: Per-Layer Gradient Analysis
# =============================================================================
def plot_per_layer_gradients(results):
    """Per-layer gradient analysis for selected experiments."""
    # Select representative experiments
    experiments = [
        (0.0001, 0.001, 'Low actor LR (no stop)'),
        (0.01, 0.001, 'High actor LR (early stop)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Layer Gradient Norms at Final Episode', fontsize=14)

    for idx, (actor_lr, critic_lr, title) in enumerate(experiments):
        ax = axes[idx]
        r = get_result(results, actor_lr, critic_lr)

        if r and 'tracking' in r:
            grad = r['tracking'].get('gradient_history', [])
            if grad:
                final = grad[-1]

                # Actor layers
                actor_layers = list(final.get('actor_per_layer', {}).keys())
                actor_values = list(final.get('actor_per_layer', {}).values())

                # Critic layers
                critic_layers = list(final.get('critic_per_layer', {}).keys())
                critic_values = list(final.get('critic_per_layer', {}).values())

                x = np.arange(max(len(actor_layers), len(critic_layers)))
                width = 0.35

                if actor_values:
                    ax.bar(x[:len(actor_values)] - width/2, actor_values, width,
                          label='Actor', color='steelblue', alpha=0.8)
                if critic_values:
                    ax.bar(x[:len(critic_values)] + width/2, critic_values, width,
                          label='Critic', color='coral', alpha=0.8)

                ax.set_xlabel('Layer')
                ax.set_ylabel('Gradient Norm')
                ax.set_title(f'{title}\n(Actor LR={actor_lr}, Critic LR={critic_lr})')
                ax.legend()
                ax.set_yscale('log')

                # Set x-tick labels
                all_layers = actor_layers if len(actor_layers) >= len(critic_layers) else critic_layers
                ax.set_xticks(x[:len(all_layers)])
                ax.set_xticklabels([l.split('.')[-1] for l in all_layers], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_per_layer_gradients.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_per_layer_gradients.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig5_per_layer_gradients")


# =============================================================================
# FIGURE 6: Final Asymmetry vs Stopping Episode
# =============================================================================
def plot_asymmetry_vs_stopping(results):
    """Correlation between final asymmetry and stopping episode."""
    fig, ax = plt.subplots(figsize=(10, 8))

    stopping_eps = []
    final_asymmetries = []
    colors = []

    for r in results:
        if 'tracking' in r:
            asym = r['tracking'].get('asymmetry_history', [])
            if asym:
                stopping_eps.append(r['stopping_episode'])
                final_asymmetries.append(min(asym[-1]['ratio'], 100))  # Cap for visualization
                colors.append(np.log10(r['actor_lr']))

    scatter = ax.scatter(stopping_eps, final_asymmetries, c=colors, cmap='viridis',
                        s=150, edgecolors='black', alpha=0.8)

    ax.set_xlabel('Stopping Episode', fontsize=12)
    ax.set_ylabel('Final Gradient Asymmetry Ratio (capped at 100)', fontsize=12)
    ax.set_title('Gradient Asymmetry vs Stopping Episode', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log10(Actor LR)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_asymmetry_vs_stopping.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_asymmetry_vs_stopping.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig6_asymmetry_vs_stopping")


# =============================================================================
# FIGURE 7: Saturation Ratio Over Time
# =============================================================================
def plot_saturation_over_time(results):
    """Tanh saturation ratio over episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Actor Output Saturation (|tanh output| > 0.9) Over Training', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        for cidx, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r and 'tracking' in r:
                act = r['tracking'].get('activation_history', [])
                if act:
                    episodes = [a['episode'] for a in act]
                    saturations = [a.get('avg_actor_output_saturation', 0) for a in act]
                    ax.plot(episodes, saturations, label=f'Critic LR={critic_lr}',
                           linewidth=2, alpha=0.8)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Saturation Ratio')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_saturation_over_time.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_saturation_over_time.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig7_saturation_over_time")


# =============================================================================
# FIGURE 8: Pre-activation Statistics
# =============================================================================
def plot_preactivation_stats(results):
    """Pre-activation statistics showing input to tanh."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pre-activation Statistics (Input to Tanh) Over Training', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        for cidx, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r and 'tracking' in r:
                act = r['tracking'].get('activation_history', [])
                if act:
                    episodes = [a['episode'] for a in act]
                    means = [a.get('mean_preact', 0) for a in act]
                    stds = [a.get('std_preact', 0) for a in act]

                    ax.plot(episodes, means, label=f'Mean (c={critic_lr})',
                           linewidth=1.5, alpha=0.8)
                    ax.fill_between(episodes,
                                   [m-s for m,s in zip(means, stds)],
                                   [m+s for m,s in zip(means, stds)],
                                   alpha=0.2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Pre-activation Value')
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Saturation threshold')
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_preactivation_stats.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_preactivation_stats.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig8_preactivation_stats")


# =============================================================================
# FIGURE 9: Saturation Heatmap by LR
# =============================================================================
def plot_saturation_heatmap(results):
    """Heatmap of final saturation by LR combination."""
    fig, ax = plt.subplots(figsize=(8, 7))

    matrix = np.zeros((4, 4))
    for i, actor_lr in enumerate(CLIENT_LRS):
        for j, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r and 'tracking' in r:
                act = r['tracking'].get('activation_history', [])
                if act:
                    matrix[i, j] = act[-1].get('avg_actor_output_saturation', 0)

    im = ax.imshow(matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'{lr}' for lr in MASTER_LRS])
    ax.set_yticklabels([f'{lr}' for lr in CLIENT_LRS])
    ax.set_xlabel('Critic Learning Rate', fontsize=12)
    ax.set_ylabel('Actor Learning Rate', fontsize=12)
    ax.set_title('Final Activation Saturation by Learning Rate', fontsize=14)

    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Saturation Ratio', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_saturation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_saturation_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig9_saturation_heatmap")


# =============================================================================
# FIGURE 10: Saturation vs Stopping Episode
# =============================================================================
def plot_saturation_vs_stopping(results):
    """Correlation between saturation and stopping."""
    fig, ax = plt.subplots(figsize=(10, 8))

    stopping_eps = []
    final_saturations = []
    actor_lrs = []

    for r in results:
        if 'tracking' in r:
            act = r['tracking'].get('activation_history', [])
            if act:
                stopping_eps.append(r['stopping_episode'])
                final_saturations.append(act[-1].get('avg_actor_output_saturation', 0))
                actor_lrs.append(np.log10(r['actor_lr']))

    scatter = ax.scatter(stopping_eps, final_saturations, c=actor_lrs, cmap='viridis',
                        s=150, edgecolors='black', alpha=0.8)

    ax.set_xlabel('Stopping Episode', fontsize=12)
    ax.set_ylabel('Final Saturation Ratio', fontsize=12)
    ax.set_title('Activation Saturation vs Stopping Episode', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log10(Actor LR)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_saturation_vs_stopping.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_saturation_vs_stopping.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig10_saturation_vs_stopping")


# =============================================================================
# FIGURE 11: Learning Curves
# =============================================================================
def plot_learning_curves(results):
    """Learning curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Curves (Episode Reward) by Actor Learning Rate', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        for cidx, critic_lr in enumerate(MASTER_LRS):
            r = get_result(results, actor_lr, critic_lr)
            if r:
                rewards = r.get('all_rewards', [])
                if rewards:
                    # Smooth with moving average
                    window = 50
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    episodes = range(window-1, len(rewards))
                    ax.plot(episodes, smoothed, label=f'Critic LR={critic_lr}',
                           linewidth=1.5, alpha=0.8)

                    # Mark stopping point
                    stop_ep = r['stopping_episode']
                    if stop_ep < len(rewards):
                        ax.axvline(x=stop_ep, linestyle=':', alpha=0.5, color='gray')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (smoothed)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_learning_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig11_learning_curves")


# =============================================================================
# FIGURE 12: Reward at Stopping vs End
# =============================================================================
def plot_reward_comparison(results):
    """Compare reward at stopping point vs end of training."""
    fig, ax = plt.subplots(figsize=(12, 6))

    experiments = []
    reward_at_stop = []
    reward_at_end = []

    for r in sorted(results, key=lambda x: (x['actor_lr'], x['critic_lr'])):
        rewards = r.get('all_rewards', [])
        if rewards:
            stop_ep = min(r['stopping_episode'], len(rewards)-1)
            experiments.append(f"A{r['actor_lr']}\nC{r['critic_lr']}")
            reward_at_stop.append(rewards[stop_ep])
            reward_at_end.append(rewards[-1])

    x = np.arange(len(experiments))
    width = 0.35

    ax.bar(x - width/2, reward_at_stop, width, label='At Stopping', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, reward_at_end, width, label='At End', color='coral', alpha=0.8)

    ax.set_xlabel('Experiment (Actor LR / Critic LR)', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Reward at Stopping Point vs End of Training', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=8, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig12_reward_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig12_reward_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig12_reward_comparison")


# =============================================================================
# FIGURE 13: Per-Agent Stopping Distribution
# =============================================================================
def plot_per_agent_stopping(results):
    """Distribution of when individual agents stopped."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Agent Stopping Episode Distribution (50 agents)', fontsize=14)

    for idx, actor_lr in enumerate(CLIENT_LRS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f'Actor LR = {actor_lr}', fontsize=12)

        data = []
        labels = []

        for critic_lr in MASTER_LRS:
            r = get_result(results, actor_lr, critic_lr)
            if r:
                per_agent = r.get('per_actor_stopped', [])
                # Replace None with max episodes
                per_agent = [p if p is not None else r['total_episodes'] for p in per_agent]
                if per_agent:
                    data.append(per_agent)
                    labels.append(f'C={critic_lr}')

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xlabel('Critic Learning Rate')
        ax.set_ylabel('Stopping Episode')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig13_per_agent_stopping.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig13_per_agent_stopping.pdf'), bbox_inches='tight')
    plt.close()
    print("Generated: fig13_per_agent_stopping")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("GENERATING ALL FIGURES FOR GRADIENT ASYMMETRY PAPER")
    print("=" * 60)

    # Load data
    results = load_all_results()
    print(f"\nLoaded {len(results)} experiments\n")

    # Generate all figures
    print("Generating figures...")
    plot_stopping_heatmap(results)
    plot_stopping_bubble(results)
    plot_asymmetry_over_time(results)
    plot_gradient_comparison(results)
    plot_per_layer_gradients(results)
    plot_asymmetry_vs_stopping(results)
    plot_saturation_over_time(results)
    plot_preactivation_stats(results)
    plot_saturation_heatmap(results)
    plot_saturation_vs_stopping(results)
    plot_learning_curves(results)
    plot_reward_comparison(results)
    plot_per_agent_stopping(results)

    print("\n" + "=" * 60)
    print(f"ALL 13 FIGURES GENERATED IN: {OUTPUT_DIR}/")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
