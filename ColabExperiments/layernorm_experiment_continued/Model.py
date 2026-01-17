"""
Neural Network Models with Layer Normalization before tanh output.

This experiment tests whether LayerNorm before the tanh activation prevents
the pre-activation explosion that causes saturation and gradient vanishing.

Key change: LayerNorm normalizes pre-activations to ~0 mean, ~1 std before tanh,
keeping outputs in the linear region where gradients flow.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LayerNormActorNetwork(nn.Module):
    """
    Actor network with Layer Normalization before tanh output.

    Architecture: 7 -> 64 -> 32 -> LayerNorm -> tanh -> 3

    The LayerNorm keeps pre-activations bounded, preventing saturation.
    """

    def __init__(self, state_dim, action_dim, output_activation):
        super(LayerNormActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

        # Layer Normalization before tanh - KEY ADDITION
        self.layer_norm = nn.LayerNorm(action_dim)

        self.output_activation = output_activation

    def forward(self, state):
        # Hidden layers with ReLU (standard)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))

        # Output layer: Linear -> LayerNorm -> tanh
        out = self.fc3(out)
        out = self.layer_norm(out)  # Normalize pre-activations

        if self.output_activation == F.softmax:
            out = self.output_activation(out, dim=-1)
        else:
            out = self.output_activation(out)  # tanh on normalized values
        return out


class CriticNetwork(nn.Module):
    """
    Standard Critic network (unchanged - linear output doesn't saturate).

    Architecture: 510 -> 512 -> 128 -> 1 (linear output)
    """

    def __init__(self, state_dim, action_dim, n_state_dim, n_action_dim):
        super(CriticNetwork, self).__init__()
        # Input: joint state-action (500) + individual state-action (10) = 510
        input_dim = state_dim + action_dim + n_state_dim + n_action_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        # Weight initialization
        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        """Single sample forward pass (original behavior)."""
        out = th.cat([state, action, pstate, paction], 0)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)  # Linear output (no saturation)
        return out

    def forward_batched(self, combined_input):
        """Batched forward pass - input already concatenated, shape: (N, input_dim)."""
        out = F.relu(self.fc1(combined_input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)


# Also include the original networks for comparison
class ActorNetwork(nn.Module):
    """Original actor without LayerNorm (for reference)."""

    def __init__(self, state_dim, action_dim, output_activation):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.output_activation = output_activation

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        if self.output_activation == F.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out


class LargeActorNetwork(nn.Module):
    """Large actor (512->128) without LayerNorm (for reference)."""

    def __init__(self, state_dim, action_dim, output_activation):
        super(LargeActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.output_activation = output_activation

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        if self.output_activation == F.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out
