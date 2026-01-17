import torch as th
from torch import nn


class ActorNetwork(nn.Module):
    """
    Original actor network with ReLU activations: 7 -> 64 -> 32 -> 3
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


class LargeActorNetwork(nn.Module):
    """
    Large actor network with ReLU activations: 7 -> 512 -> 128 -> 3
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=3e-3):
        super(LargeActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

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


class LinearActorNetwork(nn.Module):
    """
    Large actor network with LINEAR hidden activations: 7 -> 512 -> 128 -> 3

    Hypothesis: ReLU only outputs positive values [0, +inf), but rewards are negative.
    Linear activations allow the network to naturally represent negative values
    throughout the computation.

    Note: This makes the hidden layers essentially a composition of linear functions,
    but the tanh output still provides bounded non-linearity for actions.
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=3e-3):
        super(LinearActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.output_activation = output_activation

    def forward(self, state):
        # LINEAR hidden activations (no ReLU)
        out = self.fc1(state)  # Linear
        out = self.fc2(out)    # Linear
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out

    def __call__(self, state):
        return self.forward(state)


class CriticNetwork(nn.Module):
    """
    Original critic network with ReLU activations: 510 -> 512 -> 128 -> 1
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
        out = th.cat([state, action, pstate, paction], 0)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def forward_batched(self, combined_input):
        out = nn.functional.relu(self.fc1(combined_input))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)


class LinearCriticNetwork(nn.Module):
    """
    Critic network with LINEAR hidden activations: 510 -> 512 -> 128 -> 1

    Hypothesis: ReLU only outputs positive values, but Q-values for negative rewards
    should be negative. Linear activations allow natural representation of negative values.
    """
    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(LinearCriticNetwork, self).__init__()
        self.input_dim = state_dim + action_dim + pestate + peraction
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        out = th.cat([state, action, pstate, paction], 0)
        # LINEAR hidden activations (no ReLU)
        out = self.fc1(out)  # Linear
        out = self.fc2(out)  # Linear
        out = self.fc3(out)  # Linear (output)
        return out

    def forward_batched(self, combined_input):
        # LINEAR hidden activations (no ReLU)
        out = self.fc1(combined_input)  # Linear
        out = self.fc2(out)             # Linear
        out = self.fc3(out)             # Linear (output)
        return out

    def __call__(self, state, action, pstate, paction):
        return self.forward(state, action, pstate, paction)
