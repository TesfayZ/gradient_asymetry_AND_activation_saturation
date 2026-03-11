"""
Neural Network Models with Input Normalization Only (InputNorm).

This experiment tests whether normalizing inputs alone prevents saturation:
- RunningNorm for input: Normalizes using running mean/std (like BatchNorm stats)
- NO post-activation LayerNorm on hidden layers
- NO LayerNorm before tanh output
- NO LayerNorm on critic output

Key difference from other experiments:
- Original: No normalization at all
- LayerNorm: Only LN before tanh (pre-output normalization)
- FullNorm: RunningNorm input + LN after each hidden layer
- InputNorm: RunningNorm input ONLY, no hidden layer normalization

This isolates the effect of input normalization from hidden layer normalization.

Why RunningNorm instead of LayerNorm for input:
- LayerNorm normalizes across features within a single sample
- RunningNorm tracks population statistics across samples (more stable for RL)
- Better handles the non-stationary nature of RL data distributions
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    """
    Running normalization using exponential moving average of mean and std.

    Similar to BatchNorm's running statistics but applied at inference too.
    This is more suitable for RL where:
    - Batch sizes can be 1 (single sample inference)
    - Data distribution shifts over time
    - We want consistent normalization between training and inference
    """

    def __init__(self, num_features, momentum=0.99, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Running statistics (not trainable parameters)
        self.register_buffer('running_mean', th.zeros(num_features))
        self.register_buffer('running_var', th.ones(num_features))
        self.register_buffer('count', th.tensor(0.0))

    def forward(self, x):
        # x shape: (batch, features) or (features,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if self.training:
            # Update running statistics (detached from computation graph)
            with th.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)

                # Exponential moving average update
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
                self.count += x.shape[0]

        # Normalize using running statistics (both training and inference)
        x_norm = (x - self.running_mean) / (th.sqrt(self.running_var) + self.eps)

        if squeeze_output:
            x_norm = x_norm.squeeze(0)

        return x_norm


class InputNormActorNetwork(nn.Module):
    """
    Actor with Input Normalization only (no hidden layer normalization).

    Architecture: RunningNorm -> 64 -> ReLU -> 32 -> ReLU -> 3 -> tanh

    Key points:
    - RunningNorm balances feature scales using running mean/std
    - NO LayerNorm on hidden layers (unlike FullNorm)
    - NO LN before tanh
    """

    def __init__(self, state_dim, action_dim, output_activation):
        super().__init__()
        # Input normalization using running mean/std
        self.input_norm = RunningNorm(state_dim)

        # Hidden layers WITHOUT post-activation LayerNorm
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        # Output layer - NO LayerNorm before tanh
        self.fc3 = nn.Linear(32, action_dim)

        self.output_activation = output_activation

    def forward(self, state):
        # Normalize inputs using running mean/std
        x = self.input_norm(state)

        # Hidden layers: Linear -> ReLU (no LayerNorm)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output: Linear -> tanh (no normalization)
        x = self.fc3(x)

        if self.output_activation == F.softmax:
            return self.output_activation(x, dim=-1)
        return self.output_activation(x)


class InputNormCriticNetwork(nn.Module):
    """
    Critic with Input Normalization only (no hidden layer normalization).

    Architecture: RunningNorm -> 512 -> ReLU -> 128 -> ReLU -> 1 (linear)

    Key points:
    - RunningNorm balances joint state-action input using running mean/std
    - NO LayerNorm on hidden layers (unlike FullNorm)
    - Linear output (no activation)
    """

    def __init__(self, state_dim, action_dim, n_state_dim, n_action_dim):
        super().__init__()
        # Input: joint state-action (500) + individual state-action (10) = 510
        input_dim = state_dim + action_dim + n_state_dim + n_action_dim

        # Input normalization using running mean/std
        self.input_norm = RunningNorm(input_dim)

        # Hidden layers WITHOUT post-activation LayerNorm
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)

        # Output layer - NO LayerNorm, linear output
        self.fc3 = nn.Linear(128, 1)

        # Weight initialization for output layer
        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        """Single sample forward pass (original behavior)."""
        x = th.cat([state, action, pstate, paction], 0)
        x = self.input_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Linear output

    def forward_batched(self, combined_input):
        """Batched forward pass - input already concatenated, shape: (N, input_dim)."""
        x = self.input_norm(combined_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)


# Include original networks for reference/comparison
class ActorNetwork(nn.Module):
    """Original actor without any normalization (baseline)."""

    def __init__(self, state_dim, action_dim, output_activation):
        super().__init__()
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


class CriticNetwork(nn.Module):
    """Original critic without any normalization (baseline)."""

    def __init__(self, state_dim, action_dim, n_state_dim, n_action_dim):
        super().__init__()
        input_dim = state_dim + action_dim + n_state_dim + n_action_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        out = th.cat([state, action, pstate, paction], 0)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

    def forward_batched(self, combined_input):
        out = F.relu(self.fc1(combined_input))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)
