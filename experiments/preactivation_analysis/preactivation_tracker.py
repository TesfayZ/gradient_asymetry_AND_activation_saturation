"""
Pre-Activation Distribution Tracker for Tanh Output Layers

This module tracks the distribution of pre-activation values (inputs to tanh)
in actor networks to detect and visualize saturation behavior.

When pre-activation values are large (|z| > 2), tanh saturates and gradients vanish.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os


class PreActivationTracker:
    """
    Tracks pre-activation values at the tanh output layer of actor networks.

    This helps visualize when and how actors enter the saturation regime,
    which causes gradient vanishing and weight update cessation.

    Usage:
        tracker = PreActivationTracker(actors, log_dir='./logs')

        # Register hooks (call once before training)
        tracker.register_hooks()

        # During training, after forward pass:
        tracker.log_preactivations(episode)

        # After training:
        tracker.plot_saturation_analysis()
        tracker.save_results()
    """

    def __init__(
        self,
        actors: List[nn.Module],
        log_dir: str = './preactivation_logs',
        saturation_threshold: float = 2.0
    ):
        """
        Initialize the pre-activation tracker.

        Args:
            actors: List of actor networks
            log_dir: Directory to save logs and plots
            saturation_threshold: |z| above this is considered saturated
        """
        self.actors = actors
        self.log_dir = log_dir
        self.saturation_threshold = saturation_threshold

        os.makedirs(log_dir, exist_ok=True)

        # Storage for pre-activation values (captured by hooks)
        self.preactivation_buffer = []  # Temporary storage during forward pass
        self.hooks = []

        # History storage
        self.history = {
            'episodes': [],
            'mean_preact': [],
            'std_preact': [],
            'min_preact': [],
            'max_preact': [],
            'pct_saturated': [],  # Percentage of values in saturation region
            'pct_positive_saturated': [],
            'pct_negative_saturated': [],
        }

        # Per-output dimension tracking (offload, compute, power)
        self.per_dim_history = defaultdict(lambda: {
            'mean': [], 'std': [], 'pct_saturated': []
        })

        # Full distribution snapshots at key episodes
        self.distribution_snapshots = {}

    def _hook_fn(self, module, input, output):
        """Hook function to capture pre-activation values."""
        # For the last linear layer, 'input' is what we want
        # (the input to this layer, which becomes pre-activation for tanh)
        if len(input) > 0:
            preact = output.detach().cpu().numpy()  # Output of linear = input to tanh
            self.preactivation_buffer.append(preact)

    def register_hooks(self):
        """Register forward hooks on the output layers of all actors."""
        for actor in self.actors:
            # Find the last linear layer (before tanh)
            for name, module in actor.named_modules():
                if isinstance(module, nn.Linear):
                    # Assume last linear layer is fc3 or similar
                    if 'fc3' in name or 'out' in name.lower():
                        hook = module.register_forward_hook(self._hook_fn)
                        self.hooks.append(hook)
                        print(f"Registered hook on {name}")

        if len(self.hooks) == 0:
            # Fallback: register on all Linear layers and take the last one
            print("Warning: Could not identify output layer. Registering on last Linear layer.")
            for actor in self.actors:
                last_linear = None
                for module in actor.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                if last_linear is not None:
                    hook = last_linear.register_forward_hook(self._hook_fn)
                    self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def log_preactivations(self, episode: int, snapshot: bool = False):
        """
        Log pre-activation statistics from the buffer.

        Call this after a forward pass through the actors.

        Args:
            episode: Current episode number
            snapshot: Whether to save full distribution snapshot
        """
        if len(self.preactivation_buffer) == 0:
            print(f"Warning: No pre-activations captured at episode {episode}")
            return

        # Concatenate all captured pre-activations
        all_preact = np.concatenate([p.flatten() for p in self.preactivation_buffer])

        # Compute statistics
        self.history['episodes'].append(episode)
        self.history['mean_preact'].append(np.mean(all_preact))
        self.history['std_preact'].append(np.std(all_preact))
        self.history['min_preact'].append(np.min(all_preact))
        self.history['max_preact'].append(np.max(all_preact))

        # Saturation analysis
        saturated = np.abs(all_preact) > self.saturation_threshold
        pos_saturated = all_preact > self.saturation_threshold
        neg_saturated = all_preact < -self.saturation_threshold

        self.history['pct_saturated'].append(100 * np.mean(saturated))
        self.history['pct_positive_saturated'].append(100 * np.mean(pos_saturated))
        self.history['pct_negative_saturated'].append(100 * np.mean(neg_saturated))

        # Per-dimension analysis (if 3D output: offload, compute, power)
        if len(self.preactivation_buffer) > 0:
            # Stack all actor outputs
            stacked = np.vstack(self.preactivation_buffer)
            if stacked.shape[1] == 3:
                dim_names = ['offload', 'compute', 'power']
                for i, name in enumerate(dim_names):
                    dim_vals = stacked[:, i]
                    self.per_dim_history[name]['mean'].append(np.mean(dim_vals))
                    self.per_dim_history[name]['std'].append(np.std(dim_vals))
                    self.per_dim_history[name]['pct_saturated'].append(
                        100 * np.mean(np.abs(dim_vals) > self.saturation_threshold)
                    )

        # Save snapshot if requested
        if snapshot:
            self.distribution_snapshots[episode] = all_preact.copy()

        # Clear buffer for next iteration
        self.preactivation_buffer = []

    def detect_saturation(self, threshold_pct: float = 80.0, window: int = 10) -> Dict:
        """
        Detect if outputs have become saturated.

        Args:
            threshold_pct: Percentage of saturated values to trigger detection
            window: Number of recent episodes to check

        Returns:
            Detection results dictionary
        """
        if len(self.history['pct_saturated']) < window:
            return {'detected': False, 'message': 'Not enough data'}

        recent_pct = self.history['pct_saturated'][-window:]
        avg_saturation = np.mean(recent_pct)

        result = {
            'saturated': avg_saturation > threshold_pct,
            'avg_saturation_pct': avg_saturation,
            'recent_saturation': recent_pct,
            'episode': self.history['episodes'][-1] if self.history['episodes'] else None,
        }

        if result['saturated']:
            result['message'] = f"SATURATION DETECTED at episode {result['episode']}: " \
                               f"{avg_saturation:.1f}% of pre-activations are saturated"

        return result

    def get_tanh_gradient_multiplier(self) -> List[float]:
        """
        Compute the effective gradient multiplier due to tanh derivative.

        Returns list of (1 - tanh^2(z)) values showing gradient suppression.
        """
        if len(self.history['mean_preact']) == 0:
            return []

        # Approximate gradient multiplier from mean pre-activation
        multipliers = []
        for mean_z in self.history['mean_preact']:
            tanh_val = np.tanh(mean_z)
            grad_mult = 1 - tanh_val ** 2
            multipliers.append(grad_mult)

        return multipliers

    def plot_saturation_analysis(self, save: bool = True, show: bool = False):
        """
        Create comprehensive saturation analysis plots.

        Args:
            save: Whether to save plots
            show: Whether to display plots
        """
        episodes = self.history['episodes']

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Plot 1: Pre-activation mean and std
        ax1 = axes[0, 0]
        ax1.fill_between(
            episodes,
            np.array(self.history['mean_preact']) - np.array(self.history['std_preact']),
            np.array(self.history['mean_preact']) + np.array(self.history['std_preact']),
            alpha=0.3, label='±1 std'
        )
        ax1.plot(episodes, self.history['mean_preact'], 'b-', label='Mean')
        ax1.axhline(y=self.saturation_threshold, color='r', linestyle='--', label=f'Saturation threshold (±{self.saturation_threshold})')
        ax1.axhline(y=-self.saturation_threshold, color='r', linestyle='--')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Pre-activation value')
        ax1.set_title('Pre-activation Distribution Over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Min/Max pre-activation
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.history['max_preact'], 'r-', label='Max', alpha=0.8)
        ax2.plot(episodes, self.history['min_preact'], 'b-', label='Min', alpha=0.8)
        ax2.axhline(y=self.saturation_threshold, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=-self.saturation_threshold, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Pre-activation value')
        ax2.set_title('Pre-activation Range (Min/Max)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Saturation percentage
        ax3 = axes[0, 2]
        ax3.plot(episodes, self.history['pct_saturated'], 'purple', linewidth=2, label='Total')
        ax3.plot(episodes, self.history['pct_positive_saturated'], 'r-', alpha=0.6, label='Positive saturated')
        ax3.plot(episodes, self.history['pct_negative_saturated'], 'b-', alpha=0.6, label='Negative saturated')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Percentage saturated (%)')
        ax3.set_title('Saturation Percentage Over Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])

        # Plot 4: Effective gradient multiplier
        ax4 = axes[1, 0]
        grad_mult = self.get_tanh_gradient_multiplier()
        if grad_mult:
            ax4.plot(episodes, grad_mult, 'g-', linewidth=2)
            ax4.axhline(y=0.1, color='red', linestyle='--', label='90% gradient suppression')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Gradient multiplier (1 - tanh²)')
            ax4.set_title('Effective Gradient Multiplier from Tanh')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])

        # Plot 5: Per-dimension saturation (if available)
        ax5 = axes[1, 1]
        if self.per_dim_history:
            for dim_name, data in self.per_dim_history.items():
                if data['pct_saturated']:
                    ax5.plot(episodes[:len(data['pct_saturated'])],
                            data['pct_saturated'], label=dim_name, linewidth=2)
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Percentage saturated (%)')
            ax5.set_title('Per-Dimension Saturation (Offload/Compute/Power)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 100])

        # Plot 6: Distribution snapshots (histogram)
        ax6 = axes[1, 2]
        if self.distribution_snapshots:
            snapshot_episodes = sorted(self.distribution_snapshots.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_episodes)))
            for ep, color in zip(snapshot_episodes, colors):
                data = self.distribution_snapshots[ep]
                ax6.hist(data, bins=50, alpha=0.5, color=color, label=f'Episode {ep}', density=True)
            ax6.axvline(x=self.saturation_threshold, color='red', linestyle='--')
            ax6.axvline(x=-self.saturation_threshold, color='red', linestyle='--')
            ax6.set_xlabel('Pre-activation value')
            ax6.set_ylabel('Density')
            ax6.set_title('Pre-activation Distribution Snapshots')
            ax6.legend(fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'No snapshots saved\n(call log_preactivations with snapshot=True)',
                    ha='center', va='center', transform=ax6.transAxes)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.log_dir, 'saturation_analysis.png'), dpi=150)
            print(f"Saved saturation plot to {self.log_dir}/saturation_analysis.png")

        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self):
        """Save tracking results to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.log_dir, 'preactivation_history.csv'), index=False)

        # Save per-dimension data
        for dim_name, data in self.per_dim_history.items():
            df_dim = pd.DataFrame(data)
            df_dim['episode'] = self.history['episodes'][:len(data['mean'])]
            df_dim.to_csv(os.path.join(self.log_dir, f'preactivation_{dim_name}.csv'), index=False)

        print(f"Saved pre-activation results to {self.log_dir}/")

    def get_summary(self) -> str:
        """Get text summary of pre-activation tracking."""
        if len(self.history['episodes']) == 0:
            return "No data collected yet."

        summary = []
        summary.append("=" * 60)
        summary.append("PRE-ACTIVATION TRACKING SUMMARY")
        summary.append("=" * 60)

        summary.append(f"\nTotal episodes tracked: {len(self.history['episodes'])}")
        summary.append(f"Saturation threshold: ±{self.saturation_threshold}")

        summary.append(f"\n--- Pre-activation Statistics ---")
        summary.append(f"Mean (final): {self.history['mean_preact'][-1]:.4f}")
        summary.append(f"Std (final): {self.history['std_preact'][-1]:.4f}")
        summary.append(f"Range: [{self.history['min_preact'][-1]:.2f}, {self.history['max_preact'][-1]:.2f}]")

        summary.append(f"\n--- Saturation Analysis ---")
        summary.append(f"Final saturation: {self.history['pct_saturated'][-1]:.1f}%")
        summary.append(f"Max saturation: {max(self.history['pct_saturated']):.1f}%")

        # When did saturation first exceed 50%?
        for i, pct in enumerate(self.history['pct_saturated']):
            if pct > 50:
                summary.append(f"First >50% saturation: Episode {self.history['episodes'][i]}")
                break

        detection = self.detect_saturation()
        summary.append(f"\n--- Detection Result ---")
        summary.append(f"Saturated: {detection.get('saturated', 'N/A')}")
        if detection.get('saturated'):
            summary.append(f"*** {detection['message']} ***")

        summary.append("=" * 60)

        return "\n".join(summary)


def integrate_with_ccm_madrl(ccm_maddpg_instance, log_dir: str = './preactivation_logs'):
    """
    Helper function to integrate pre-activation tracking with CCM_MADDPG.

    Args:
        ccm_maddpg_instance: Instance of CCM_MADDPG class
        log_dir: Directory for logging

    Returns:
        PreActivationTracker instance (with hooks registered)
    """
    tracker = PreActivationTracker(
        actors=ccm_maddpg_instance.actors,
        log_dir=log_dir
    )
    tracker.register_hooks()
    return tracker


if __name__ == "__main__":
    print("Pre-Activation Tracker Module")
    print("=" * 40)
    print("\nTo use this module, integrate it with your training code:")
    print("""
    from preactivation_tracker import PreActivationTracker, integrate_with_ccm_madrl

    # Initialize and register hooks
    tracker = integrate_with_ccm_madrl(ccmaddpg, log_dir='./logs')

    # In training loop, after forward pass:
    # (actors are called in choose_action or train)
    tracker.log_preactivations(episode=n_episodes, snapshot=(episode % 100 == 0))

    # Check for saturation:
    detection = tracker.detect_saturation()
    if detection['saturated']:
        print(detection['message'])

    # After training:
    tracker.remove_hooks()
    print(tracker.get_summary())
    tracker.plot_saturation_analysis(save=True)
    tracker.save_results()
    """)
