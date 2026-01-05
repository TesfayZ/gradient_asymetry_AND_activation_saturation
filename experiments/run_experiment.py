"""
Integrated Experiment Runner for Gradient Asymmetry Analysis

This script runs the CCM_MADRL training with gradient and pre-activation
tracking to analyze the differential stopping behavior between actors and critics.

Usage:
    python run_experiment.py --actor_lr 0.0001 --critic_lr 0.001 --episodes 2000 --index 0
"""

import sys
import os
import argparse
import torch
import numpy as np
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These imports will work when the CCM_MADRL code is in the same structure
# Adjust paths as needed for your setup


def parse_args():
    parser = argparse.ArgumentParser(description='Gradient Asymmetry Experiment')

    # Learning rates
    parser.add_argument('--actor_lr', type=float, default=0.0001,
                        help='Learning rate for actor networks')
    parser.add_argument('--critic_lr', type=float, default=0.001,
                        help='Learning rate for critic network')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--steps_per_episode', type=int, default=10,
                        help='Steps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--memory_size', type=int, default=10000,
                        help='Replay memory capacity')

    # Experiment settings
    parser.add_argument('--index', type=int, default=0,
                        help='Experiment index for multiple runs')
    parser.add_argument('--seed', type=int, default=37,
                        help='Random seed')
    parser.add_argument('--n_agents', type=int, default=50,
                        help='Number of agents')

    # Tracking settings
    parser.add_argument('--track_gradients', action='store_true', default=True,
                        help='Enable gradient tracking')
    parser.add_argument('--track_preactivations', action='store_true', default=True,
                        help='Enable pre-activation tracking')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval for saving distribution snapshots')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')

    return parser.parse_args()


def create_output_dirs(base_dir: str, actor_lr: float, critic_lr: float, index: int):
    """Create output directory structure."""
    exp_name = f"lr_a{actor_lr}_c{critic_lr}_run{index}"
    exp_dir = os.path.join(base_dir, exp_name)

    dirs = {
        'base': exp_dir,
        'gradients': os.path.join(exp_dir, 'gradients'),
        'preactivations': os.path.join(exp_dir, 'preactivations'),
        'models': os.path.join(exp_dir, 'models'),
        'plots': os.path.join(exp_dir, 'plots'),
        'csv': os.path.join(exp_dir, 'csv'),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def run_experiment_with_tracking(args):
    """
    Run the CCM_MADRL experiment with gradient and pre-activation tracking.

    This is a template that shows how to integrate the tracking modules.
    Modify according to your actual CCM_MADRL implementation.
    """
    print("=" * 70)
    print("GRADIENT ASYMMETRY EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Actor LR: {args.actor_lr}")
    print(f"  Critic LR: {args.critic_lr}")
    print(f"  Episodes: {args.episodes}")
    print(f"  N Agents: {args.n_agents}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # Create output directories
    dirs = create_output_dirs(args.output_dir, args.actor_lr, args.critic_lr, args.index)
    print(f"\nOutput directory: {dirs['base']}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Import tracking modules
    from gradient_tracking.gradient_tracker import GradientTracker
    from preactivation_analysis.preactivation_tracker import PreActivationTracker

    # =========================================================================
    # INTEGRATION POINT: Replace this section with your actual CCM_MADRL setup
    # =========================================================================

    print("\n" + "=" * 70)
    print("INTEGRATION TEMPLATE")
    print("=" * 70)
    print("""
    To run actual experiments, modify this script to:

    1. Import your CCM_MADRL implementation:
       from CCM_MADRL import CCM_MADDPG
       from mec_env import MecEnv

    2. Initialize environment and agent:
       env = MecEnv(n_agents=args.n_agents, env_seed=args.seed)
       ccmaddpg = CCM_MADDPG(
           env=env,
           actor_lr=args.actor_lr,
           critic_lr=args.critic_lr,
           ...
       )

    3. Initialize trackers:
       grad_tracker = GradientTracker(
           actors=ccmaddpg.actors,
           critics=ccmaddpg.critics,
           log_dir=dirs['gradients']
       )

       preact_tracker = PreActivationTracker(
           actors=ccmaddpg.actors,
           log_dir=dirs['preactivations']
       )
       preact_tracker.register_hooks()

    4. Modify the training loop in CCM_MADRL.train() to call:
       # After loss.backward(), before optimizer.step():
       grad_tracker.log_gradients(episode=self.n_episodes)

       # After forward pass through actors:
       preact_tracker.log_preactivations(
           episode=self.n_episodes,
           snapshot=(self.n_episodes % snapshot_interval == 0)
       )

    5. After training, generate analysis:
       grad_tracker.plot_gradient_history(save=True)
       grad_tracker.save_results()
       print(grad_tracker.get_summary())

       preact_tracker.plot_saturation_analysis(save=True)
       preact_tracker.save_results()
       print(preact_tracker.get_summary())
    """)

    # Demo: Create mock actors/critics to show how tracking works
    print("\n" + "=" * 70)
    print("DEMO: Running with mock networks")
    print("=" * 70)

    # Create simple mock networks for demonstration
    class MockActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(7, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.fc3 = torch.nn.Linear(32, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            return x

    class MockCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(510, 512)
            self.fc2 = torch.nn.Linear(512, 128)
            self.fc3 = torch.nn.Linear(128, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Create mock networks
    actors = [MockActor() for _ in range(args.n_agents)]
    critics = [MockCritic()]

    # Initialize trackers
    grad_tracker = GradientTracker(
        actors=actors,
        critics=critics,
        log_dir=dirs['gradients']
    )

    preact_tracker = PreActivationTracker(
        actors=actors,
        log_dir=dirs['preactivations']
    )
    preact_tracker.register_hooks()

    # Simulate training loop
    print("\nSimulating training with tracking...")
    for episode in range(min(100, args.episodes)):
        # Simulate forward pass
        state = torch.randn(1, 7)
        for actor in actors:
            _ = actor(state)

        critic_input = torch.randn(1, 510)
        for critic in critics:
            q_value = critic(critic_input)

        # Simulate backward pass (mock loss)
        mock_loss = sum(actor(state).sum() for actor in actors) - critics[0](critic_input).sum()
        mock_loss.backward()

        # Log gradients
        grad_tracker.log_gradients(episode=episode)

        # Log pre-activations
        preact_tracker.log_preactivations(
            episode=episode,
            snapshot=(episode % args.snapshot_interval == 0)
        )

        # Zero gradients for next iteration
        for actor in actors:
            actor.zero_grad()
        for critic in critics:
            critic.zero_grad()

        if episode % 20 == 0:
            stats = grad_tracker.get_current_stats()
            print(f"Episode {episode}: Actor grad={stats['actor_grad_norm']:.6f}, "
                  f"Critic grad={stats['critic_grad_norm']:.6f}, "
                  f"Ratio={stats['grad_ratio']:.4f}")

    # Generate analysis
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- Gradient Analysis ---")
    print(grad_tracker.get_summary())
    grad_tracker.plot_gradient_history(save=True, show=False)
    grad_tracker.save_results()

    print("\n--- Pre-activation Analysis ---")
    print(preact_tracker.get_summary())
    preact_tracker.plot_saturation_analysis(save=True, show=False)
    preact_tracker.save_results()

    # Cleanup
    preact_tracker.remove_hooks()

    print("\n" + "=" * 70)
    print(f"Results saved to: {dirs['base']}")
    print("=" * 70)

    return dirs


def main():
    args = parse_args()

    # Run experiment
    result_dirs = run_experiment_with_tracking(args)

    print("\nExperiment completed!")
    print(f"View results in: {result_dirs['base']}")


if __name__ == "__main__":
    main()
