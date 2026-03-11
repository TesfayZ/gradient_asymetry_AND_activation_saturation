import torch as th
from torch import nn


class AdaptiveUniformTanhActor(nn.Module):
    """
    Actor network with adaptive scaling for uniform tanh output.

    Key idea: Track strict min/max of pre-activations (G_min, G_max) and scale
    them to a target range so that:
    1. Tanh outputs are uniformly distributed by default (good exploration)
    2. Outputs can reach near ±1 when learning genuinely pushes beyond tracked range
    3. Prevents accidental saturation from numerical instability

    With target_range=2:
    - Pre-activations at G_min/G_max map to ±2
    - tanh(±2) ≈ ±0.96, utilizing most of output range
    - tanh'(±2) ≈ 0.07, keeping gradients viable even at extremes
    - Middle values have tanh'(0) = 1.0, full gradient
    """
    def __init__(self, state_dim, output_size, output_activation=None, init_w=3e-3, target_range=2.0):
        super(AdaptiveUniformTanhActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # Target range for scaling: maps [G_min, G_max] -> [-target_range, +target_range]
        self.target_range = target_range

        # Strict min/max tracking per action dimension
        self.register_buffer('g_min', th.full((output_size,), float('inf')))
        self.register_buffer('g_max', th.full((output_size,), float('-inf')))

        # Store output_activation for compatibility (not used, we always use tanh)
        self.output_activation = output_activation

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        z = self.fc3(out)  # Pre-activations [batch, action_dim]

        if self.training:
            with th.no_grad():
                # Compute batch min/max per dimension
                batch_min = z.min(dim=0).values
                batch_max = z.max(dim=0).values

                # Strict min/max: only expand, never shrink (no CPU-GPU sync needed)
                self.g_min = th.min(self.g_min, batch_min)
                self.g_max = th.max(self.g_max, batch_max)

        # Scale [G_min, G_max] -> [-target_range, +target_range]
        g_center = (self.g_max + self.g_min) / 2
        g_half_range = ((self.g_max - self.g_min) / 2).clamp(min=1e-6)

        z_scaled = (z - g_center) / g_half_range * self.target_range

        return th.tanh(z_scaled)

    def __call__(self, state):
        return self.forward(state)

    def get_scaling_stats(self):
        """Return current G_min, G_max for logging."""
        return {
            'g_min': self.g_min.detach().cpu().numpy().tolist(),
            'g_max': self.g_max.detach().cpu().numpy().tolist(),
            'g_range': (self.g_max - self.g_min).detach().cpu().numpy().tolist(),
        }


class CriticNetwork(nn.Module):
    """
    A network for critic - supports both single and batched inputs
    (Same as original - no changes needed for critic)
    """
    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.input_dim = state_dim + action_dim + pestate + peraction
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        """Single sample forward pass (original behavior)."""
        out = th.cat([state, action, pstate, paction], 0)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def forward_batched(self, combined_input):
        """Batched forward pass - input already concatenated, shape: (N, input_dim)."""
        out = nn.functional.relu(self.fc1(combined_input))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)


# Keep original ActorNetwork for reference/comparison
class ActorNetwork(nn.Module):
    """
    Original actor network (for reference)
    """
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
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out

    def __call__(self, state):
        return self.forward(state)
