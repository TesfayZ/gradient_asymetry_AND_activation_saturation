"""
Plot Analysis for Stopping Experiment Results

Run this anytime to visualize completed experiments:
    python3 plot_analysis.py

Generates:
    - plot_num_episodes.png (Figure 5.3)
    - saturation_analysis.png (tanh saturation evidence)
    - asymmetry_analysis.png (gradient asymmetry)
    - per_experiment plots in each result directory
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
OUTPUT_DIR = RESULTS_DIR

CLIENT_LRS = [0.0001, 0.001, 0.01, 0.1]
MASTER_LRS = [0.0001, 0.001, 0.01, 0.1]


def load_all_results():
    """Load results from all completed experiments."""
    all_results = []

    for actor_lr in CLIENT_LRS:
        for critic_lr in MASTER_LRS:
            exp_dir = os.path.join(RESULTS_DIR, f'actor_{actor_lr}_critic_{critic_lr}_run_0')
            results_file = os.path.join(exp_dir, 'results.json')
            tracking_file = os.path.join(exp_dir, 'tracking_data.json')

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    result = json.load(f)

                # Load full tracking data if available
                if os.path.exists(tracking_file):
                    with open(tracking_file, 'r') as f:
                        tracking = json.load(f)
                    result['full_tracking'] = tracking

                all_results.append(result)
                print(f"Loaded: actor_lr={actor_lr}, critic_lr={critic_lr}")

    return all_results


def plot_saturation_analysis(all_results):
    """Plot tanh saturation evidence."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tanh Saturation Analysis - Evidence of Actor Output Saturation', fontsize=14)

    # Plot 1: Saturation ratio over episodes for different LR configs
    ax1 = axes[0, 0]
    for result in all_results:
        tracking = result.get('full_tracking') or {}
        activation_history = tracking.get('activation_history', result.get('activation_history', []))

        if activation_history:
            episodes = [h['episode'] for h in activation_history]
            saturations = [h.get('avg_actor_output_saturation', 0) for h in activation_history]

            if saturations and any(s > 0 for s in saturations):
                label = f"a={result['actor_lr']}, c={result['critic_lr']}"
                ax1.plot(episodes, saturations, label=label, alpha=0.7, linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Saturation Ratio (|output| > 0.9)')
    ax1.set_title('Actor Output Layer Saturation Over Training')
    ax1.legend(fontsize=7, loc='best')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='High saturation threshold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final saturation vs stopping episode
    ax2 = axes[0, 1]
    stopping_eps = []
    final_saturations = []
    colors = []

    for result in all_results:
        tracking = result.get('full_tracking') or {}
        activation_history = tracking.get('activation_history', result.get('activation_history', []))

        if activation_history:
            final_sat = activation_history[-1].get('avg_actor_output_saturation', 0)
            stopping_eps.append(result['stopping_episode'])
            final_saturations.append(final_sat)
            colors.append(result['actor_lr'])

    if stopping_eps:
        scatter = ax2.scatter(stopping_eps, final_saturations, c=np.log10(colors),
                             cmap='viridis', s=150, edgecolors='black', alpha=0.7)
        ax2.set_xlabel('Stopping Episode')
        ax2.set_ylabel('Final Saturation Ratio')
        ax2.set_title('Stopping Episode vs Final Saturation')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('log10(Actor LR)')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Saturation by learning rate (bar chart)
    ax3 = axes[1, 0]
    lr_saturations = {}

    for result in all_results:
        tracking = result.get('full_tracking') or {}
        activation_history = tracking.get('activation_history', result.get('activation_history', []))

        if activation_history and len(activation_history) > 5:
            # Get saturation at episode ~50 (early training)
            early_idx = min(5, len(activation_history)-1)
            early_sat = activation_history[early_idx].get('avg_actor_output_saturation', 0)

            lr_key = result['actor_lr']
            if lr_key not in lr_saturations:
                lr_saturations[lr_key] = []
            lr_saturations[lr_key].append(early_sat)

    if lr_saturations:
        lrs = sorted(lr_saturations.keys())
        means = [np.mean(lr_saturations[lr]) for lr in lrs]
        stds = [np.std(lr_saturations[lr]) if len(lr_saturations[lr]) > 1 else 0 for lr in lrs]

        x = np.arange(len(lrs))
        bars = ax3.bar(x, means, yerr=stds, capsize=5, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(lr) for lr in lrs])
        ax3.set_xlabel('Actor Learning Rate')
        ax3.set_ylabel('Saturation Ratio at Early Training (~50 eps)')
        ax3.set_title('Early Saturation by Learning Rate')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Activation statistics over time
    ax4 = axes[1, 1]

    # Find a high LR experiment with data
    for result in all_results:
        if result['actor_lr'] >= 0.01:
            tracking = result.get('full_tracking') or {}
            activation_history = tracking.get('activation_history', result.get('activation_history', []))

            if activation_history:
                episodes = [h['episode'] for h in activation_history]

                # Extract per-layer stats if available
                mins = []
                maxs = []
                for h in activation_history:
                    actor_act = h.get('actor_activations_sample', {})
                    # Find output layer (fc3)
                    for layer_name, stats in actor_act.items():
                        if 'fc3' in layer_name:
                            mins.append(stats.get('min', 0))
                            maxs.append(stats.get('max', 0))
                            break
                    else:
                        mins.append(0)
                        maxs.append(0)

                if mins and maxs and (any(m != 0 for m in mins) or any(m != 0 for m in maxs)):
                    ax4.fill_between(episodes, mins, maxs, alpha=0.3, label=f"a={result['actor_lr']}")
                    ax4.plot(episodes, mins, 'b-', alpha=0.5)
                    ax4.plot(episodes, maxs, 'r-', alpha=0.5)
                    break

    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='tanh bound (+1)')
    ax4.axhline(y=-1.0, color='b', linestyle='--', alpha=0.7, label='tanh bound (-1)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Actor Output Layer Activation Range')
    ax4.set_title('Output Activation Min/Max (approaches ±1 = saturation)')
    ax4.legend(fontsize=8)
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'saturation_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_gradient_asymmetry(all_results):
    """Plot gradient asymmetry analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Asymmetry Analysis - Actor vs Critic Gradient Flow', fontsize=14)

    # Plot 1: Asymmetry ratio over episodes
    ax1 = axes[0, 0]
    for result in all_results:
        tracking = result.get('full_tracking') or {}
        asymmetry_history = tracking.get('asymmetry_history', result.get('asymmetry_history', []))

        if asymmetry_history:
            episodes = [h['episode'] for h in asymmetry_history]
            ratios = [h['ratio'] for h in asymmetry_history]

            # Filter out inf values
            valid = [(e, r) for e, r in zip(episodes, ratios) if r != float('inf') and r > 0]
            if valid:
                episodes, ratios = zip(*valid)
                label = f"a={result['actor_lr']}, c={result['critic_lr']}"
                ax1.plot(episodes, ratios, label=label, alpha=0.7)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Gradient Ratio (Actor/Critic)')
    ax1.set_title('Gradient Asymmetry Over Training')
    ax1.legend(fontsize=7)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Actor vs Critic gradient magnitudes
    ax2 = axes[0, 1]
    for result in all_results[:4]:  # First 4 experiments
        tracking = result.get('full_tracking') or {}
        asymmetry_history = tracking.get('asymmetry_history', result.get('asymmetry_history', []))

        if asymmetry_history:
            episodes = [h['episode'] for h in asymmetry_history]
            actor_grads = [h['actor_grad'] for h in asymmetry_history]
            critic_grads = [h['critic_grad'] for h in asymmetry_history]

            label = f"a={result['actor_lr']}"
            ax2.plot(episodes, actor_grads, label=f"Actor {label}", linestyle='-', alpha=0.7)
            ax2.plot(episodes, critic_grads, label=f"Critic {label}", linestyle='--', alpha=0.7)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Gradient Magnitude')
    ax2.set_title('Actor vs Critic Gradient Magnitudes')
    ax2.legend(fontsize=7)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stopping episode vs learning rate
    ax3 = axes[1, 0]
    for critic_lr in MASTER_LRS:
        actor_lrs = []
        stopping_eps = []
        for result in all_results:
            if result['critic_lr'] == critic_lr:
                actor_lrs.append(result['actor_lr'])
                stopping_eps.append(result['stopping_episode'])

        if actor_lrs:
            ax3.plot(actor_lrs, stopping_eps, 'o-', label=f"critic_lr={critic_lr}", markersize=10)

    ax3.set_xlabel('Actor Learning Rate')
    ax3.set_ylabel('Stopping Episode')
    ax3.set_title('Stopping Episode vs Actor Learning Rate')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Per-layer gradient comparison
    ax4 = axes[1, 1]

    # Find experiment with per-layer data
    for result in all_results:
        tracking = result.get('full_tracking') or {}
        gradient_history = tracking.get('gradient_history', result.get('gradient_history', []))

        if gradient_history and gradient_history[-1].get('actor_per_layer'):
            sample = gradient_history[-1]
            actor_layers = list(sample['actor_per_layer'].keys())
            actor_values = list(sample['actor_per_layer'].values())
            critic_layers = list(sample['critic_per_layer'].keys())
            critic_values = list(sample['critic_per_layer'].values())

            x_actor = np.arange(len(actor_layers))
            x_critic = np.arange(len(critic_layers))

            ax4.bar(x_actor - 0.2, actor_values, 0.4, label='Actor', alpha=0.7, color='blue')
            ax4.bar(x_critic + 0.2, critic_values, 0.4, label='Critic', alpha=0.7, color='orange')

            ax4.set_xlabel('Layer Index')
            ax4.set_ylabel('Gradient Magnitude')
            ax4.set_title(f'Per-Layer Gradients (a={result["actor_lr"]}, c={result["critic_lr"]})')
            ax4.legend()
            ax4.set_yscale('log')
            break

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'asymmetry_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_stopping_episodes(all_results):
    """Plot Figure 5.3 - Stopping episodes bubble chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    client_lrs = []
    master_lrs = []
    stopping_episodes = []

    for result in all_results:
        client_lrs.append(result['actor_lr'])
        master_lrs.append(result['critic_lr'])
        stopping_episodes.append(result['stopping_episode'])

    if not stopping_episodes:
        print("No results to plot for stopping episodes")
        return

    scatter = ax.scatter(
        client_lrs,
        master_lrs,
        s=[e * 0.5 for e in stopping_episodes],
        c=stopping_episodes,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Episodes', fontsize=12)

    for i in range(len(client_lrs)):
        ax.annotate(
            str(stopping_episodes[i]),
            (client_lrs[i], master_lrs[i]),
            fontsize=9,
            ha='left',
            va='center',
            fontweight='bold'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Client (Actor) Learning Rate', fontsize=12)
    ax.set_ylabel('Master (Critic) Learning Rate', fontsize=12)
    ax.set_title('Stopping Episode for Different Learning Rate Combinations', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'plot_num_episodes.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    # Also save to paper figures
    paper_fig_dir = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
    if os.path.exists(paper_fig_dir):
        plt.savefig(os.path.join(paper_fig_dir, 'plot_num_episodes.png'), dpi=150, bbox_inches='tight')
        print(f"Also saved to: {paper_fig_dir}")

    plt.close()


def plot_reward_curves(all_results):
    """Plot reward curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Reward Curves by Learning Rate Configuration', fontsize=14)

    # Group by actor learning rate
    lr_groups = {lr: [] for lr in CLIENT_LRS}
    for result in all_results:
        lr_groups[result['actor_lr']].append(result)

    for idx, (actor_lr, results) in enumerate(lr_groups.items()):
        ax = axes[idx // 2, idx % 2]

        for result in results:
            rewards = result.get('all_rewards', [])
            if rewards:
                episodes = list(range(len(rewards)))
                label = f"critic_lr={result['critic_lr']}"
                ax.plot(episodes, rewards, label=label, alpha=0.7)

                # Mark stopping episode
                stop_ep = result['stopping_episode']
                if stop_ep < len(rewards):
                    ax.axvline(x=stop_ep, linestyle='--', alpha=0.5)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'Actor LR = {actor_lr}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'reward_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Loading experiment results...")
    print("=" * 60)

    all_results = load_all_results()

    if not all_results:
        print("\nNo completed experiments found!")
        print(f"Looking in: {RESULTS_DIR}")
        return

    print(f"\nLoaded {len(all_results)} experiments")
    print("\nGenerating plots...")

    # Generate all plots
    plot_stopping_episodes(all_results)
    plot_saturation_analysis(all_results)
    plot_gradient_asymmetry(all_results)
    plot_reward_curves(all_results)

    print("\n" + "=" * 60)
    print("All plots generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
