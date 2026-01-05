"""
CCM_MADRL with Gradient and Pre-activation Tracking

This is a modified version of CCM_MADRL.py that includes integrated
gradient magnitude and pre-activation tracking for analyzing the
asymmetric convergence behavior between actors and critics.

Key additions:
- GradientTracker integration in train() method
- PreActivationTracker integration for tanh output monitoring
- Weight change detection with logging
- Automatic saturation and vanishing gradient alerts
"""

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
from copy import deepcopy
from numpy import savetxt
import os
import sys

# Add tracking modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gradient_tracking.gradient_tracker import GradientTracker
from preactivation_analysis.preactivation_tracker import PreActivationTracker


def to_tensor_var(x, use_cuda):
    """Convert numpy array to tensor variable."""
    x = torch.FloatTensor(x)
    if use_cuda:
        x = x.cuda()
    return x


class ActorNetwork(nn.Module):
    """Actor network with tanh output for bounded continuous actions."""

    def __init__(self, state_dim, output_size, output_activation, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.output_activation = output_activation

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_activation(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """Critic network with linear output for unbounded Q-values."""

    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + pestate + peraction, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        out = torch.cat([state, action, pstate, paction], 0)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CCM_MADDPG_Tracked:
    """
    Client-Master MADDPG with integrated gradient and pre-activation tracking.

    This version includes:
    - Automatic gradient magnitude logging after each backward pass
    - Pre-activation distribution tracking for tanh saturation detection
    - Weight change monitoring to detect when actors stop updating
    - Periodic alerts for vanishing gradients and saturation
    """

    def __init__(
        self,
        InfdexofResult,
        env,
        env_eval,
        n_agents,
        state_dim,
        action_dim,
        action_lower_bound,
        action_higher_bound,
        memory_capacity=10000,
        target_tau=1,
        reward_gamma=0.99,
        reward_scale=1.,
        done_penalty=None,
        actor_output_activation=torch.tanh,
        actor_lr=0.0001,
        critic_lr=0.001,
        optimizer_type="adam",
        max_grad_norm=None,
        batch_size=64,
        episodes_before_train=64,
        epsilon_start=1,
        epsilon_end=0.01,
        epsilon_decay=None,
        use_cuda=False,
        # Tracking parameters
        enable_tracking=True,
        tracking_log_dir='./tracking_logs',
        gradient_log_interval=1,
        snapshot_interval=100,
    ):
        self.n_agents = n_agents
        self.env = env
        self.env_eval = env_eval
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound

        self.env_state = env.reset_mec()
        self.n_episodes = 0
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # Epsilon greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.target_tau = target_tau

        # Initialize networks
        self.actors = [
            ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation)
            for _ in range(self.n_agents)
        ]

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.state_dim, self.action_dim)]

        # Target networks
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        # Optimizers
        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.actors_target[i].cuda()
            self.critics[0].cuda()
            self.critics_target[0].cuda()

        # Results storage
        self.eval_episode_rewards = []
        self.mean_rewards = []
        self.episodes = []
        self.results = []
        self.InfdexofResult = InfdexofResult

        # =====================================================================
        # TRACKING INTEGRATION
        # =====================================================================
        self.enable_tracking = enable_tracking
        self.gradient_log_interval = gradient_log_interval
        self.snapshot_interval = snapshot_interval

        if enable_tracking:
            # Create tracking directory
            tracking_dir = os.path.join(tracking_log_dir, f"run_{InfdexofResult}")
            os.makedirs(tracking_dir, exist_ok=True)

            # Initialize gradient tracker
            self.grad_tracker = GradientTracker(
                actors=self.actors,
                critics=self.critics,
                log_dir=os.path.join(tracking_dir, 'gradients')
            )

            # Initialize pre-activation tracker
            self.preact_tracker = PreActivationTracker(
                actors=self.actors,
                log_dir=os.path.join(tracking_dir, 'preactivations')
            )
            self.preact_tracker.register_hooks()

            # Weight change tracking
            self.weight_checkpoints = self._save_weight_checkpoint()
            self.actor_stopped_episodes = [None] * self.n_agents
            self.all_actors_stopped_episode = None

            print(f"Tracking enabled. Logs will be saved to: {tracking_dir}")

    def _save_weight_checkpoint(self):
        """Save current weights for comparison."""
        checkpoint = {
            'actors': [actor.state_dict().copy() for actor in self.actors],
            'critics': [critic.state_dict().copy() for critic in self.critics],
        }
        # Deep copy tensors
        for i, state_dict in enumerate(checkpoint['actors']):
            checkpoint['actors'][i] = {k: v.clone() for k, v in state_dict.items()}
        for i, state_dict in enumerate(checkpoint['critics']):
            checkpoint['critics'][i] = {k: v.clone() for k, v in state_dict.items()}
        return checkpoint

    def _check_weight_changes(self):
        """Check which networks have updated weights since last checkpoint."""
        actor_changes = []
        critic_changes = []

        for i, actor in enumerate(self.actors):
            changed = False
            for name, param in actor.state_dict().items():
                if not torch.equal(param, self.weight_checkpoints['actors'][i][name]):
                    changed = True
                    break
            actor_changes.append(changed)

        for i, critic in enumerate(self.critics):
            changed = False
            for name, param in critic.state_dict().items():
                if not torch.equal(param, self.weight_checkpoints['critics'][i][name]):
                    changed = True
                    break
            critic_changes.append(changed)

        return actor_changes, critic_changes

    def _soft_update_target(self, target, source):
        """Soft update target network parameters."""
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_((1. - self.target_tau) * t.data + self.target_tau * s.data)

    def train(self):
        """
        Training step with integrated tracking.

        This method includes gradient logging and pre-activation tracking.
        """
        if self.n_episodes <= self.episodes_before_train:
            return

        # ... [Your existing training code here] ...
        # The key integration points are marked with ### TRACKING ###

        ### TRACKING: Log pre-activations after forward pass ###
        if self.enable_tracking:
            take_snapshot = (self.n_episodes % self.snapshot_interval == 0)
            self.preact_tracker.log_preactivations(
                episode=self.n_episodes,
                snapshot=take_snapshot
            )

        # ... [After computing losses and calling backward()] ...

        ### TRACKING: Log gradients after backward, before optimizer step ###
        if self.enable_tracking and (self.n_episodes % self.gradient_log_interval == 0):
            self.grad_tracker.log_gradients(episode=self.n_episodes)

            # Check for vanishing gradients
            detection = self.grad_tracker.detect_gradient_vanishing()
            if detection.get('asymmetry_detected'):
                print(f"\n*** ALERT: {detection['message']} ***\n")

        # ... [After optimizer.step()] ...

        ### TRACKING: Check weight changes ###
        if self.enable_tracking:
            actor_changes, critic_changes = self._check_weight_changes()

            # Track when each actor stops
            for i, changed in enumerate(actor_changes):
                if not changed and self.actor_stopped_episodes[i] is None:
                    self.actor_stopped_episodes[i] = self.n_episodes
                    print(f"Actor {i} stopped updating at episode {self.n_episodes}")

            # Check if all actors stopped
            if all(ep is not None for ep in self.actor_stopped_episodes) and self.all_actors_stopped_episode is None:
                self.all_actors_stopped_episode = self.n_episodes
                print(f"\n*** ALL ACTORS STOPPED UPDATING AT EPISODE {self.n_episodes} ***")
                print(f"Critic still updating: {any(critic_changes)}\n")

            # Update checkpoint
            self.weight_checkpoints = self._save_weight_checkpoint()

            # Check for saturation
            if self.n_episodes % 50 == 0:
                sat_detection = self.preact_tracker.detect_saturation()
                if sat_detection.get('saturated'):
                    print(f"\n*** SATURATION ALERT: {sat_detection['message']} ***\n")

    def finalize_tracking(self):
        """
        Finalize tracking and generate analysis.

        Call this after training is complete.
        """
        if not self.enable_tracking:
            return

        print("\n" + "=" * 70)
        print("TRACKING ANALYSIS SUMMARY")
        print("=" * 70)

        # Gradient analysis
        print("\n--- Gradient Analysis ---")
        print(self.grad_tracker.get_summary())
        self.grad_tracker.plot_gradient_history(save=True, show=False)
        self.grad_tracker.save_results()

        # Pre-activation analysis
        print("\n--- Pre-activation Analysis ---")
        print(self.preact_tracker.get_summary())
        self.preact_tracker.plot_saturation_analysis(save=True, show=False)
        self.preact_tracker.save_results()

        # Weight change summary
        print("\n--- Weight Change Summary ---")
        print(f"All actors stopped episode: {self.all_actors_stopped_episode}")
        for i, ep in enumerate(self.actor_stopped_episodes):
            if ep is not None:
                print(f"  Actor {i} stopped at episode {ep}")

        # Cleanup hooks
        self.preact_tracker.remove_hooks()

        print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("CCM_MADRL with Tracking Module")
    print("=" * 40)
    print("""
    This module provides CCM_MADDPG_Tracked class which includes:

    1. Automatic gradient magnitude logging
    2. Pre-activation distribution tracking
    3. Weight change monitoring
    4. Saturation and vanishing gradient alerts

    Usage:
        ccmaddpg = CCM_MADDPG_Tracked(
            InfdexofResult=0,
            env=env,
            env_eval=env_eval,
            n_agents=50,
            state_dim=7,
            action_dim=3,
            actor_lr=0.0001,
            critic_lr=0.001,
            enable_tracking=True,
            tracking_log_dir='./tracking_logs'
        )

        # Run training
        ccmaddpg.interact(MAX_EPISODES, ...)

        # After training
        ccmaddpg.finalize_tracking()
    """)
