import torch as th
from torch import nn


class ActorNetwork(nn.Module):
    """
    Original actor network (small): 7 -> 64 -> 32 -> 3
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
    Large actor network with DOUBLE the critic hidden layer sizes: 7 -> 1024 -> 256 -> 3
    (Critic uses 512 -> 128)

    Hypothesis: Larger network capacity may help avoid saturation by:
    1. Distributing learning signal across more parameters
    2. Producing smaller pre-activation values at the output layer
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=3e-3):
        super(LargeActorNetwork, self).__init__()
        # Double the critic hidden layer sizes (critic: 512 -> 128, actor: 1024 -> 256)
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_size)

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


class CriticNetwork(nn.Module):
    """
    A network for critic - supports both single and batched inputs
    Architecture: 510 -> 512 -> 128 -> 1
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
