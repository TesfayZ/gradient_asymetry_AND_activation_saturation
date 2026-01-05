"""
Gradient Magnitude Tracking for Actor-Critic Networks

This module provides tools to track and analyze gradient magnitudes
during training of actor-critic reinforcement learning algorithms.

Key metrics tracked:
- Per-layer gradient norms for actors and critics
- Gradient ratio between actor and critic
- Gradient flow through tanh activation layers
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os


class GradientTracker:
    """
    Tracks gradient magnitudes for actor and critic networks during training.

    Usage:
        tracker = GradientTracker(actor, critic, log_dir='./logs')

        # In training loop, after loss.backward():
        tracker.log_gradients(episode, step)

        # After training:
        tracker.plot_gradient_history()
        tracker.save_results()
    """

    def __init__(
        self,
        actors: List[nn.Module],
        critics: List[nn.Module],
        log_dir: str = './gradient_logs',
        track_per_layer: bool = True
    ):
        """
        Initialize the gradient tracker.

        Args:
            actors: List of actor networks
            critics: List of critic networks
            log_dir: Directory to save logs and plots
            track_per_layer: Whether to track gradients per layer
        """
        self.actors = actors
        self.critics = critics
        self.log_dir = log_dir
        self.track_per_layer = track_per_layer

        os.makedirs(log_dir, exist_ok=True)

        # Storage for gradient history
        self.actor_grad_history = defaultdict(list)  # {layer_name: [grad_norms]}
        self.critic_grad_history = defaultdict(list)
        self.actor_total_grad = []
        self.critic_total_grad = []
        self.grad_ratios = []
        self.episodes = []
        self.steps = []

        # Track which layers have tanh activation (output layers in actors)
        self.tanh_layer_names = self._identify_tanh_layers()

    def _identify_tanh_layers(self) -> List[str]:
        """Identify layers that feed into tanh activations."""
        # In typical actor networks, the last linear layer feeds tanh
        tanh_layers = []
        for actor in self.actors:
            for name, module in actor.named_modules():
                if isinstance(module, nn.Linear):
                    # Check if this is likely the output layer
                    if 'fc3' in name or 'output' in name.lower():
                        tanh_layers.append(name)
        return tanh_layers

    def _compute_grad_norm(self, model: nn.Module, per_layer: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Compute gradient norms for a model.

        Args:
            model: Neural network model
            per_layer: Whether to return per-layer norms

        Returns:
            total_norm: Total gradient norm across all parameters
            layer_norms: Dictionary of {layer_name: gradient_norm}
        """
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if per_layer:
                    # Group by layer (remove .weight/.bias suffix)
                    layer_name = name.rsplit('.', 1)[0]
                    if layer_name not in layer_norms:
                        layer_norms[layer_name] = 0.0
                    layer_norms[layer_name] += param_norm ** 2

        total_norm = np.sqrt(total_norm)
        layer_norms = {k: np.sqrt(v) for k, v in layer_norms.items()}

        return total_norm, layer_norms

    def log_gradients(self, episode: int, step: int = 0):
        """
        Log gradient magnitudes for all actors and critics.

        Call this after loss.backward() but before optimizer.step().

        Args:
            episode: Current training episode
            step: Current step within episode
        """
        self.episodes.append(episode)
        self.steps.append(step)

        # Track actor gradients (average across all actors)
        actor_total_norms = []
        actor_layer_norms = defaultdict(list)

        for actor in self.actors:
            total_norm, layer_norms = self._compute_grad_norm(actor, self.track_per_layer)
            actor_total_norms.append(total_norm)

            for layer_name, norm in layer_norms.items():
                actor_layer_norms[layer_name].append(norm)

        avg_actor_total = np.mean(actor_total_norms)
        self.actor_total_grad.append(avg_actor_total)

        for layer_name, norms in actor_layer_norms.items():
            self.actor_grad_history[layer_name].append(np.mean(norms))

        # Track critic gradients
        critic_total_norms = []
        critic_layer_norms = defaultdict(list)

        for critic in self.critics:
            total_norm, layer_norms = self._compute_grad_norm(critic, self.track_per_layer)
            critic_total_norms.append(total_norm)

            for layer_name, norm in layer_norms.items():
                critic_layer_norms[layer_name].append(norm)

        avg_critic_total = np.mean(critic_total_norms)
        self.critic_total_grad.append(avg_critic_total)

        for layer_name, norms in critic_layer_norms.items():
            self.critic_grad_history[layer_name].append(np.mean(norms))

        # Compute gradient ratio (actor/critic)
        if avg_critic_total > 1e-10:
            ratio = avg_actor_total / avg_critic_total
        else:
            ratio = float('inf') if avg_actor_total > 1e-10 else 0.0
        self.grad_ratios.append(ratio)

    def get_current_stats(self) -> Dict:
        """Get current gradient statistics."""
        if len(self.actor_total_grad) == 0:
            return {}

        return {
            'actor_grad_norm': self.actor_total_grad[-1],
            'critic_grad_norm': self.critic_total_grad[-1],
            'grad_ratio': self.grad_ratios[-1],
            'actor_grad_is_zero': self.actor_total_grad[-1] < 1e-8,
            'critic_grad_is_zero': self.critic_total_grad[-1] < 1e-8,
        }

    def detect_gradient_vanishing(self, threshold: float = 1e-6, window: int = 10) -> Dict:
        """
        Detect if gradients have vanished.

        Args:
            threshold: Gradient norm below this is considered vanished
            window: Number of recent episodes to check

        Returns:
            Dictionary with detection results
        """
        if len(self.actor_total_grad) < window:
            return {'detected': False, 'message': 'Not enough data'}

        recent_actor_grads = self.actor_total_grad[-window:]
        recent_critic_grads = self.critic_total_grad[-window:]

        actor_vanished = np.mean(recent_actor_grads) < threshold
        critic_vanished = np.mean(recent_critic_grads) < threshold

        result = {
            'actor_vanished': actor_vanished,
            'critic_vanished': critic_vanished,
            'actor_mean_grad': np.mean(recent_actor_grads),
            'critic_mean_grad': np.mean(recent_critic_grads),
            'asymmetry_detected': actor_vanished and not critic_vanished,
        }

        if result['asymmetry_detected']:
            result['message'] = f"ASYMMETRY DETECTED at episode {self.episodes[-1]}: " \
                               f"Actor gradients vanished while critic continues"

        return result

    def plot_gradient_history(self, save: bool = True, show: bool = False):
        """
        Plot gradient magnitude history.

        Args:
            save: Whether to save plots to log_dir
            show: Whether to display plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Total gradient norms
        ax1 = axes[0, 0]
        ax1.semilogy(self.episodes, self.actor_total_grad, label='Actor (avg)', alpha=0.8)
        ax1.semilogy(self.episodes, self.critic_total_grad, label='Critic', alpha=0.8)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Gradient Norm (log scale)')
        ax1.set_title('Total Gradient Norms: Actor vs Critic')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient ratio
        ax2 = axes[0, 1]
        ax2.plot(self.episodes, self.grad_ratios, color='purple', alpha=0.8)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Equal gradients')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Actor/Critic Gradient Ratio')
        ax2.set_title('Gradient Ratio (Actor/Critic)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, min(10, max(self.grad_ratios) * 1.1)])

        # Plot 3: Actor per-layer gradients
        ax3 = axes[1, 0]
        for layer_name, grads in self.actor_grad_history.items():
            label = f"{layer_name} (OUTPUT)" if layer_name in self.tanh_layer_names else layer_name
            ax3.semilogy(self.episodes[:len(grads)], grads, label=label, alpha=0.8)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Gradient Norm (log scale)')
        ax3.set_title('Actor Per-Layer Gradient Norms')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Critic per-layer gradients
        ax4 = axes[1, 1]
        for layer_name, grads in self.critic_grad_history.items():
            ax4.semilogy(self.episodes[:len(grads)], grads, label=layer_name, alpha=0.8)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Gradient Norm (log scale)')
        ax4.set_title('Critic Per-Layer Gradient Norms')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.log_dir, 'gradient_analysis.png'), dpi=150)
            print(f"Saved gradient plot to {self.log_dir}/gradient_analysis.png")

        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self):
        """Save gradient tracking results to CSV files."""
        # Save total gradients
        df_total = pd.DataFrame({
            'episode': self.episodes,
            'step': self.steps,
            'actor_grad_norm': self.actor_total_grad,
            'critic_grad_norm': self.critic_total_grad,
            'grad_ratio': self.grad_ratios,
        })
        df_total.to_csv(os.path.join(self.log_dir, 'gradient_totals.csv'), index=False)

        # Save per-layer actor gradients
        actor_data = {'episode': self.episodes}
        for layer_name, grads in self.actor_grad_history.items():
            actor_data[f'actor_{layer_name}'] = grads
        df_actor = pd.DataFrame(actor_data)
        df_actor.to_csv(os.path.join(self.log_dir, 'actor_layer_gradients.csv'), index=False)

        # Save per-layer critic gradients
        critic_data = {'episode': self.episodes}
        for layer_name, grads in self.critic_grad_history.items():
            critic_data[f'critic_{layer_name}'] = grads
        df_critic = pd.DataFrame(critic_data)
        df_critic.to_csv(os.path.join(self.log_dir, 'critic_layer_gradients.csv'), index=False)

        print(f"Saved gradient results to {self.log_dir}/")

    def get_summary(self) -> str:
        """Get a text summary of gradient tracking results."""
        if len(self.actor_total_grad) == 0:
            return "No gradient data collected yet."

        summary = []
        summary.append("=" * 60)
        summary.append("GRADIENT TRACKING SUMMARY")
        summary.append("=" * 60)

        summary.append(f"\nTotal episodes tracked: {len(self.episodes)}")
        summary.append(f"Episode range: {self.episodes[0]} - {self.episodes[-1]}")

        summary.append(f"\n--- Actor Gradients ---")
        summary.append(f"Mean: {np.mean(self.actor_total_grad):.6f}")
        summary.append(f"Max:  {np.max(self.actor_total_grad):.6f}")
        summary.append(f"Min:  {np.min(self.actor_total_grad):.6f}")
        summary.append(f"Final: {self.actor_total_grad[-1]:.6f}")

        summary.append(f"\n--- Critic Gradients ---")
        summary.append(f"Mean: {np.mean(self.critic_total_grad):.6f}")
        summary.append(f"Max:  {np.max(self.critic_total_grad):.6f}")
        summary.append(f"Min:  {np.min(self.critic_total_grad):.6f}")
        summary.append(f"Final: {self.critic_total_grad[-1]:.6f}")

        summary.append(f"\n--- Gradient Ratio (Actor/Critic) ---")
        summary.append(f"Mean: {np.mean(self.grad_ratios):.4f}")
        summary.append(f"Final: {self.grad_ratios[-1]:.4f}")

        # Detect vanishing
        detection = self.detect_gradient_vanishing()
        summary.append(f"\n--- Vanishing Gradient Detection ---")
        summary.append(f"Actor vanished: {detection.get('actor_vanished', 'N/A')}")
        summary.append(f"Critic vanished: {detection.get('critic_vanished', 'N/A')}")
        summary.append(f"Asymmetry detected: {detection.get('asymmetry_detected', 'N/A')}")

        if detection.get('asymmetry_detected'):
            summary.append(f"\n*** WARNING: {detection['message']} ***")

        summary.append("=" * 60)

        return "\n".join(summary)


def integrate_with_ccm_madrl(ccm_maddpg_instance, log_dir: str = './gradient_logs'):
    """
    Helper function to integrate gradient tracking with CCM_MADDPG class.

    Args:
        ccm_maddpg_instance: Instance of CCM_MADDPG class
        log_dir: Directory for logging

    Returns:
        GradientTracker instance
    """
    tracker = GradientTracker(
        actors=ccm_maddpg_instance.actors,
        critics=ccm_maddpg_instance.critics,
        log_dir=log_dir
    )
    return tracker


# Example usage and integration code
if __name__ == "__main__":
    print("Gradient Tracker Module")
    print("=" * 40)
    print("\nTo use this module, integrate it with your training code:")
    print("""
    from gradient_tracker import GradientTracker, integrate_with_ccm_madrl

    # Option 1: Direct integration
    tracker = GradientTracker(actors=ccmaddpg.actors, critics=ccmaddpg.critics)

    # Option 2: Helper function
    tracker = integrate_with_ccm_madrl(ccmaddpg, log_dir='./logs')

    # In training loop (after backward, before step):
    tracker.log_gradients(episode=n_episodes, step=current_step)

    # Check for vanishing gradients:
    detection = tracker.detect_gradient_vanishing()
    if detection['asymmetry_detected']:
        print(detection['message'])

    # After training:
    print(tracker.get_summary())
    tracker.plot_gradient_history(save=True)
    tracker.save_results()
    """)
