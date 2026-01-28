"""
Full Normalization Experiment - Testing InputNorm + Post-activation LayerNorm

Hypothesis: The recommended RL normalization strategy combines:
- InputNorm to balance feature scales
- Post-activation LayerNorm after each hidden layer for gradient stability
- NO LayerNorm before tanh (avoids forced saturation)
- NO LayerNorm on critic output (preserves TD error signal)

This approach should:
1. Reduce gradient asymmetry by stabilizing both actor and critic gradients
2. Avoid tanh saturation without the downsides of pre-tanh normalization
3. Preserve magnitude information that's important for RL value estimation

Learning rate combinations (HIGH to LOW - failing cases first):
- Actor LRs: [0.1, 0.01, 0.001, 0.0001]
- Critic LRs: [0.1, 0.01, 0.001, 0.0001]
- Total: 16 experiments

Features:
- Auto-saves to Google Drive every 100 episodes
- Auto-restores from Drive on session restart
- Crash recovery built-in
- Early stopping when all actors stop updating
"""

import os
import sys
import json
import time
import random
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from copy import deepcopy


def set_seed(seed):
    """Set seed for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Add experiment directory to path - Colab environment
for _path in ['/content/fullnorm_experiment',
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fullnorm_experiment')]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from mec_env import MecEnv
except ModuleNotFoundError:
    print("DEBUG: sys.path =", sys.path[:5])
    print("DEBUG: /content contents:", os.listdir('/content') if os.path.exists('/content') else 'NOT FOUND')
    print("DEBUG: fullnorm_experiment exists:", os.path.exists('/content/fullnorm_experiment'))
    if os.path.exists('/content/fullnorm_experiment'):
        print("DEBUG: fullnorm_experiment contents:", os.listdir('/content/fullnorm_experiment'))
    raise

# KEY CHANGE: Import FullNormActorNetwork and FullNormCriticNetwork
from Model import FullNormActorNetwork, FullNormCriticNetwork
from prioritized_memory import Memory
from utils import to_tensor_var, get_device


class Config:
    # Seed for reproducibility
    SEED = 42

    # Learning rates: HIGH to LOW (failing cases first)
    CLIENT_LRS = [0.1, 0.01, 0.001, 0.0001]
    MASTER_LRS = [0.1, 0.01, 0.001, 0.0001]

    # Training parameters
    MAX_EPISODES = 2000
    EPISODES_BEFORE_TRAIN = 1
    NUMBER_OF_EVAL_EPISODES = 3

    # Network
    N_AGENTS = 50
    STATE_DIM = 7
    ACTION_DIM = 3
    MEMORY_CAPACITY = 10000
    BATCH_SIZE = 64

    # Training
    TARGET_TAU = 1
    REWARD_GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 2000

    # GPU
    USE_CUDA = True

    # Tracking intervals
    GRADIENT_LOG_INTERVAL = 10
    ACTIVATION_LOG_INTERVAL = 10

    # Directories - Colab paths (UPDATED for fullnorm)
    RESULTS_DIR = '/content/results/fullnorm_experiment'
    DRIVE_BACKUP_DIR = '/content/drive/MyDrive/fullnorm_results'


class FullNormExperimentAgent:
    """CCM_MADDPG with FullNorm actor/critic and tracking for stopping episode detection."""

    def __init__(self, actor_lr, critic_lr, exp_dir, use_cuda=True):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.exp_dir = exp_dir
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = get_device(self.use_cuda)

        self.env = MecEnv(Config.N_AGENTS, env_seed=Config.SEED)
        self.env_eval = MecEnv(Config.N_AGENTS, env_seed=Config.SEED)

        self.n_agents = Config.N_AGENTS
        self.state_dim = Config.STATE_DIM
        self.action_dim = Config.ACTION_DIM

        self._init_networks()
        self.memory = Memory(Config.MEMORY_CAPACITY)

        self.n_episodes = 0
        self.batch_size = Config.BATCH_SIZE
        self.episodes_before_train = Config.EPISODES_BEFORE_TRAIN
        self.epsilon_start = Config.EPSILON_START
        self.epsilon_end = Config.EPSILON_END
        self.epsilon_decay = Config.EPSILON_DECAY
        self.target_tau = Config.TARGET_TAU
        self.reward_gamma = Config.REWARD_GAMMA
        self.reward_scale = 1.0

        self.all_rewards = []
        self.results = []
        self.Training_results = []
        self.Training_step_rewards = []
        self.Training_episode_rewards = []

        self.gradient_history = []
        self.asymmetry_history = []
        self.activation_history = []

        self.weight_checkpoints = self._save_weight_checkpoint()
        self.actor_stopped_episodes = [None] * self.n_agents
        self.stopping_episode = None
        self.all_actors_stopped = False

        self.preact_buffer = []
        self.hooks = []
        self._register_hooks()

    def _init_networks(self):
        from torch.optim import Adam

        # KEY CHANGE: Use FullNormActorNetwork with InputNorm + post-activation LN
        self.actors = [
            FullNormActorNetwork(self.state_dim, self.action_dim, torch.tanh)
            for _ in range(self.n_agents)
        ]

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim

        # KEY CHANGE: Use FullNormCriticNetwork with InputNorm + post-activation LN
        self.critics = [FullNormCriticNetwork(critic_state_dim, critic_action_dim,
                                              self.state_dim, self.action_dim)]

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
        self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.actors_target[i].cuda()
            self.critics[0].cuda()
            self.critics_target[0].cuda()

        self.zero_state = torch.zeros(self.state_dim, device=self.device)
        self.zero_action = torch.zeros(self.action_dim, device=self.device)

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.preact_buffer.append(output.detach().cpu().numpy())

        for actor in self.actors:
            for name, module in actor.named_modules():
                if 'fc3' in name:
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)

    def _save_weight_checkpoint(self):
        return {
            'actors': [{k: v.clone() for k, v in a.state_dict().items()}
                      for a in self.actors],
            'critics': [{k: v.clone() for k, v in c.state_dict().items()}
                       for c in self.critics]
        }

    def _check_weight_changes(self):
        actor_changes = []
        for i, actor in enumerate(self.actors):
            changed = any(
                not torch.equal(p, self.weight_checkpoints['actors'][i][n])
                for n, p in actor.state_dict().items()
            )
            actor_changes.append(changed)

        critic_changes = []
        for i, critic in enumerate(self.critics):
            changed = any(
                not torch.equal(p, self.weight_checkpoints['critics'][i][n])
                for n, p in critic.state_dict().items()
            )
            critic_changes.append(changed)

        return actor_changes, critic_changes

    def _compute_grad_norms(self):
        actor_total = 0.0
        actor_per_layer = {}

        for actor in self.actors:
            for name, param in actor.named_parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    actor_total += norm ** 2
                    layer = name.rsplit('.', 1)[0]
                    actor_per_layer[layer] = actor_per_layer.get(layer, 0) + norm ** 2

        actor_total = np.sqrt(actor_total / self.n_agents)
        actor_per_layer = {k: np.sqrt(v / self.n_agents) for k, v in actor_per_layer.items()}

        critic_total = 0.0
        critic_per_layer = {}

        for critic in self.critics:
            for name, param in critic.named_parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    critic_total += norm ** 2
                    layer = name.rsplit('.', 1)[0]
                    critic_per_layer[layer] = critic_per_layer.get(layer, 0) + norm ** 2

        critic_total = np.sqrt(critic_total)
        critic_per_layer = {k: np.sqrt(v) for k, v in critic_per_layer.items()}

        return actor_total, critic_total, actor_per_layer, critic_per_layer

    def _log_activation_stats(self):
        if not self.preact_buffer:
            return None

        all_preact = np.concatenate([p.flatten() for p in self.preact_buffer])
        self.preact_buffer = []

        outputs = np.tanh(all_preact)
        saturation_ratio = np.mean(np.abs(outputs) > 0.9)

        stats = {
            'episode': self.n_episodes,
            'mean_preact': float(np.mean(all_preact)),
            'std_preact': float(np.std(all_preact)),
            'min_preact': float(np.min(all_preact)),
            'max_preact': float(np.max(all_preact)),
            'avg_actor_output_saturation': float(saturation_ratio),
            'actor_activations_sample': {}
        }

        stats['actor_activations_sample']['fc3'] = {
            'min': float(np.min(outputs)),
            'max': float(np.max(outputs)),
            'mean': float(np.mean(outputs)),
            'saturation': float(saturation_ratio)
        }

        return stats

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_((1. - self.target_tau) * t.data + self.target_tau * s.data)

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.env.action_higher_bound[i] - self.env.action_lower_bound[i]) / (b - a) \
            + self.env.action_lower_bound[i]
        return x

    def choose_action(self, state, evaluation=False):
        import random
        from mec_env import ENV_MODE, K_CHANNEL, S_E, N_UNITS

        state_var = to_tensor_var(state, self.use_cuda).view(self.n_agents, self.state_dim)

        actor_action = np.zeros((self.n_agents, self.action_dim))
        critic_action = np.zeros(self.n_agents)
        hybrid_action = np.zeros((self.n_agents, self.action_dim))

        with torch.no_grad():
            for agent_id in range(self.n_agents):
                action_var = self.actors[agent_id](state_var[agent_id:agent_id+1, :])
                actor_action[agent_id] = action_var.cpu().numpy()[0]

        if not evaluation:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.n_episodes / self.epsilon_decay)
            noise = np.random.randn(self.n_agents, self.action_dim) * epsilon
            actor_action += noise
            actor_action = np.clip(actor_action, -1, 1)

        hybrid_action = actor_action.copy()
        proposed = np.count_nonzero(actor_action[:, 0] >= 0)
        proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
        sumofproposed = np.sum(state[proposed_indices, 3])

        constraint = K_CHANNEL if ENV_MODE == "H2" else N_UNITS

        if proposed > 0:
            if proposed > constraint or sumofproposed > S_E:
                if not evaluation and np.random.rand() <= epsilon:
                    agent_list = list(range(self.n_agents))
                    random.shuffle(agent_list)
                    randomorder = random.sample(agent_list, constraint)
                    sizeaccepted = np.sum(state[randomorder, 3])
                    while sizeaccepted > S_E and randomorder:
                        randomorder.pop()
                        sizeaccepted = np.sum(state[randomorder, 3])
                    critic_action[randomorder] = 1
                else:
                    with torch.no_grad():
                        states_var = to_tensor_var(state, self.use_cuda).view(self.n_agents, self.state_dim)
                        whole_states = states_var.view(-1)
                        actor_action_var = to_tensor_var(actor_action, self.use_cuda).view(self.n_agents, self.action_dim)
                        whole_actions = actor_action_var.view(-1)

                        critic_action_Qs = np.full(self.n_agents, -np.inf)
                        offload_indices = np.where(actor_action[:, 0] >= 0)[0]

                        if len(offload_indices) > 0:
                            n_off = len(offload_indices)
                            batch_ws = whole_states.unsqueeze(0).expand(n_off, -1)
                            batch_wa = whole_actions.unsqueeze(0).expand(n_off, -1)
                            batch_ps = states_var[offload_indices]
                            batch_pa = actor_action_var[offload_indices]

                            critic_input = torch.cat([batch_ws, batch_wa, batch_ps, batch_pa], dim=1)
                            q_values = self.critics[0].forward_batched(critic_input).squeeze(-1).cpu().numpy()
                            critic_action_Qs[offload_indices] = q_values

                    sorted_indices = np.argsort(critic_action_Qs)[::-1]
                    count = 0
                    size = 0
                    for idx in sorted_indices:
                        if actor_action[idx, 0] >= 0 and count < constraint and size + state[idx, 3] < S_E:
                            critic_action[idx] = 1
                            count += 1
                            size += state[idx, 3]
            else:
                for i in range(self.n_agents):
                    if hybrid_action[i, 0] >= 0:
                        critic_action[i] = 1

        hybrid_action[:, 0] = critic_action

        for n in range(self.n_agents):
            hybrid_action[n][1] = self.getactionbound(-1, 1, hybrid_action[n][1], 1)
            hybrid_action[n][2] = self.getactionbound(-1, 1, hybrid_action[n][2], 2)

        return actor_action, critic_action, hybrid_action

    def append_sample(self, states, actor_actions, critic_actions, rewards, next_states, dones):
        error = 0
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(-1, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents * self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents * self.state_dim)

        nextactor_actions = []
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            nextactor_actions.append(next_action_var.detach())
        nextactor_actions_var = torch.cat(nextactor_actions, dim=1)
        nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents, self.action_dim)
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents * self.action_dim)

        nextperQs = []
        for nexta in range(self.n_agents):
            if nextactor_actions_var[0, nexta, 0] >= 0:
                nextperQs.append(self.critics_target[0](
                    whole_next_states_var[0], whole_nextactor_actions_var[0],
                    next_states_var[0, nexta, :], nextactor_actions_var[0, nexta, :]).detach())
        if len(nextperQs) == 0:
            tar_perQ = self.critics_target[0](
                whole_next_states_var[0], whole_nextactor_actions_var[0],
                self.zero_state, self.zero_action).detach()
        else:
            tar_perQ = max(nextperQs)
        tar = self.reward_scale * rewards_var[0, 0, :] + self.reward_gamma * tar_perQ * (1. - dones)

        cselected = 0
        for a in range(self.n_agents):
            if critic_actions_var[0, a, 0] == 1:
                curr_perQ = self.critics[0](
                    whole_states_var[0], whole_actor_actions_var[0],
                    states_var[0, a, :], actor_actions_var[0, a, :]).detach()
                error += ((tar - curr_perQ) ** 2).cpu().item()
                cselected += 1
        if cselected == 0:
            curr_perQ = self.critics[0](
                whole_states_var[0], whole_actor_actions_var[0],
                self.zero_state, self.zero_action).detach()
            error += ((tar - curr_perQ) ** 2).cpu().item()

        self.memory.addorupdate(error, (states, actor_actions, critic_actions, rewards, next_states, dones))

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        tryfetch = 0
        while tryfetch < 3:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            mini_batch = np.array(mini_batch, dtype=object).transpose()
            if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
                tryfetch += 1
            else:
                break

        if tryfetch >= 3:
            return

        states = np.vstack(mini_batch[0])
        actor_actions = np.vstack(mini_batch[1])
        critic_actions = np.vstack(mini_batch[2])
        rewards = np.vstack(mini_batch[3])
        next_states = np.vstack(mini_batch[4])
        dones = mini_batch[5].astype(int)

        states_var = to_tensor_var(states, self.use_cuda).view(self.batch_size, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(self.batch_size, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(self.batch_size, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(self.batch_size, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(self.batch_size, self.n_agents, self.state_dim)
        dones_var = to_tensor_var(dones, self.use_cuda).view(self.batch_size, 1)
        is_weights_var = to_tensor_var(np.array(is_weights), self.use_cuda).view(self.batch_size)

        whole_states_var = states_var.view(self.batch_size, -1)
        whole_actor_actions_var = actor_actions_var.view(self.batch_size, -1)
        whole_next_states_var = next_states_var.view(self.batch_size, -1)

        nextactor_actions = []
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            nextactor_actions.append(next_action_var)
        nextactor_actions_var = torch.stack(nextactor_actions, dim=1)
        whole_nextactor_actions_var = nextactor_actions_var.view(self.batch_size, -1)

        batch_ws_expanded = whole_next_states_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.state_dim)
        batch_wa_expanded = whole_nextactor_actions_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.action_dim)
        batch_ps = next_states_var.reshape(-1, self.state_dim)
        batch_pa = nextactor_actions_var.reshape(-1, self.action_dim)

        target_critic_input = torch.cat([batch_ws_expanded, batch_wa_expanded, batch_ps, batch_pa], dim=1)
        all_next_q = self.critics_target[0].forward_batched(target_critic_input).view(self.batch_size, self.n_agents)

        zero_input = torch.cat([whole_next_states_var, whole_nextactor_actions_var,
                               self.zero_state.unsqueeze(0).expand(self.batch_size, -1),
                               self.zero_action.unsqueeze(0).expand(self.batch_size, -1)], dim=1)
        zero_next_q = self.critics_target[0].forward_batched(zero_input).squeeze(-1)

        offload_mask = nextactor_actions_var[:, :, 0] >= 0

        masked_next_q = all_next_q.clone()
        masked_next_q[~offload_mask] = float('-inf')
        max_next_q, _ = masked_next_q.max(dim=1)

        any_offload = offload_mask.any(dim=1)
        tar_perQ = torch.where(any_offload, max_next_q, zero_next_q)

        target_q_base = self.reward_scale * rewards_var[:, 0, 0] + self.reward_gamma * tar_perQ * (1. - dones_var.squeeze(-1))

        batch_ws_curr = whole_states_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.state_dim)
        batch_wa_curr = whole_actor_actions_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.action_dim)
        batch_ps_curr = states_var.reshape(-1, self.state_dim)
        batch_pa_curr = actor_actions_var.reshape(-1, self.action_dim)

        current_critic_input = torch.cat([batch_ws_curr, batch_wa_curr, batch_ps_curr, batch_pa_curr], dim=1)
        all_curr_q = self.critics[0].forward_batched(current_critic_input).view(self.batch_size, self.n_agents)

        zero_curr_input = torch.cat([whole_states_var, whole_actor_actions_var,
                                    self.zero_state.unsqueeze(0).expand(self.batch_size, -1),
                                    self.zero_action.unsqueeze(0).expand(self.batch_size, -1)], dim=1)
        zero_curr_q = self.critics[0].forward_batched(zero_curr_input).squeeze(-1)

        selected_mask = critic_actions_var.squeeze(-1) == 1
        any_selected = selected_mask.any(dim=1)

        current_q_list = []
        target_q_list = []
        errors = np.zeros(self.batch_size)

        for b in range(self.batch_size):
            if any_selected[b]:
                selected_indices = torch.where(selected_mask[b])[0]
                for idx in selected_indices:
                    curr_q = all_curr_q[b, idx]
                    current_q_list.append(curr_q * is_weights_var[b])
                    target_q_list.append(target_q_base[b] * is_weights_var[b])
                    errors[b] += ((curr_q - target_q_base[b]) ** 2).detach().cpu().numpy()
            else:
                curr_q = zero_curr_q[b]
                current_q_list.append(curr_q * is_weights_var[b])
                target_q_list.append(target_q_base[b] * is_weights_var[b])
                errors[b] += ((curr_q - target_q_base[b]) ** 2).detach().cpu().numpy()

        current_q = torch.stack(current_q_list)
        target_q = torch.stack(target_q_list)

        critic_loss = torch.nn.MSELoss()(current_q, target_q)
        critic_loss.requires_grad_(True)
        self.critics_optimizer[0].zero_grad()
        critic_loss.backward()
        self.critics_optimizer[0].step()
        self._soft_update_target(self.critics_target[0], self.critics[0])

        for agent_id in range(self.n_agents):
            newactor_actions = []
            for agent_in in range(self.n_agents):
                newactor_action_var = self.actors[agent_in](states_var[:, agent_in, :])
                newactor_actions.append(newactor_action_var)
            newactor_actions_var = torch.stack(newactor_actions, dim=1)
            whole_newactor_actions_var = newactor_actions_var.view(self.batch_size, -1)

            batch_ws_actor = whole_states_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.state_dim)
            batch_wa_actor = whole_newactor_actions_var.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents * self.action_dim)
            batch_ps_actor = states_var.reshape(-1, self.state_dim)
            batch_pa_actor = newactor_actions_var.reshape(-1, self.action_dim)

            actor_critic_input = torch.cat([batch_ws_actor, batch_wa_actor, batch_ps_actor, batch_pa_actor], dim=1)
            all_actor_q = self.critics[0].forward_batched(actor_critic_input).view(self.batch_size, self.n_agents)

            zero_actor_input = torch.cat([whole_states_var, whole_newactor_actions_var,
                                         self.zero_state.unsqueeze(0).expand(self.batch_size, -1),
                                         self.zero_action.unsqueeze(0).expand(self.batch_size, -1)], dim=1)
            zero_actor_q = self.critics[0].forward_batched(zero_actor_input).squeeze(-1)

            actor_offload_mask = newactor_actions_var[:, :, 0] >= 0

            masked_actor_q = all_actor_q.clone()
            masked_actor_q[~actor_offload_mask] = float('-inf')
            max_actor_q, _ = masked_actor_q.max(dim=1)

            any_actor_offload = actor_offload_mask.any(dim=1)
            actor_q_selected = torch.where(any_actor_offload, max_actor_q, zero_actor_q)

            actor_loss = -(actor_q_selected * is_weights_var).mean()
            actor_loss.requires_grad_(True)

            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])

        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i])

    def train_with_tracking(self):
        self.train()

        if self.n_episodes <= self.episodes_before_train:
            return

        if self.n_episodes % Config.GRADIENT_LOG_INTERVAL == 0:
            actor_grad, critic_grad, actor_layers, critic_layers = self._compute_grad_norms()

            ratio = actor_grad / critic_grad if critic_grad > 1e-10 else float('inf')

            self.gradient_history.append({
                'episode': self.n_episodes,
                'actor_grad': actor_grad,
                'critic_grad': critic_grad,
                'actor_per_layer': actor_layers,
                'critic_per_layer': critic_layers,
            })

            self.asymmetry_history.append({
                'episode': self.n_episodes,
                'ratio': ratio,
                'actor_grad': actor_grad,
                'critic_grad': critic_grad,
            })

        if self.n_episodes % Config.ACTIVATION_LOG_INTERVAL == 0:
            stats = self._log_activation_stats()
            if stats:
                self.activation_history.append(stats)

        actor_changes, critic_changes = self._check_weight_changes()

        for i, changed in enumerate(actor_changes):
            if not changed and self.actor_stopped_episodes[i] is None:
                self.actor_stopped_episodes[i] = self.n_episodes

        if all(ep is not None for ep in self.actor_stopped_episodes) and not self.all_actors_stopped:
            self.all_actors_stopped = True
            self.stopping_episode = self.n_episodes
            print(f"\n*** ALL ACTORS STOPPED at episode {self.n_episodes} ***")
            print(f"    Critic still updating: {any(critic_changes)}\n")

        self.weight_checkpoints = self._save_weight_checkpoint()

    def run_episode(self):
        state = self.env.reset_mec()
        done = False
        episode_reward = 0

        while not done:
            actor_action, critic_action, hybrid_action = self.choose_action(state, False)
            next_state, reward, done, _, _ = self.env.step_mec(hybrid_action)

            self.append_sample(state, actor_action, critic_action, reward, next_state, done)
            self.Training_step_rewards.append(np.mean(reward))
            episode_reward += np.mean(reward)

            if not done:
                state = next_state

        self.n_episodes += 1
        self.Training_episode_rewards.append(episode_reward)
        self.all_rewards.append(episode_reward)
        self.Training_step_rewards = []

        return episode_reward

    def evaluate(self, num_episodes=3):
        rewards = []
        for i in range(num_episodes):
            state = self.env_eval.reset_mec(i)
            done = False
            ep_reward = 0
            while not done:
                _, _, hybrid_action = self.choose_action(state, True)
                state, reward, done, _, _ = self.env_eval.step_mec(hybrid_action)
                ep_reward += np.mean(reward)
            rewards.append(ep_reward)
        return np.mean(rewards)

    def save_results(self):
        os.makedirs(self.exp_dir, exist_ok=True)

        results = {
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'stopping_episode': self.stopping_episode or self.n_episodes,
            'total_episodes': self.n_episodes,
            'all_rewards': self.all_rewards,
            'final_reward': self.all_rewards[-1] if self.all_rewards else 0,
            'per_actor_stopped': self.actor_stopped_episodes,
        }

        with open(os.path.join(self.exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        tracking = {
            'gradient_history': self.gradient_history,
            'asymmetry_history': self.asymmetry_history,
            'activation_history': self.activation_history,
        }
        with open(os.path.join(self.exp_dir, 'tracking_data.json'), 'w') as f:
            json.dump(tracking, f, indent=2)

        print(f"Results saved to {self.exp_dir}")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def get_experiment_key(actor_lr, critic_lr, run=0):
    return f'fullnorm_{actor_lr}_critic_{critic_lr}_run_{run}'


def get_status_file():
    return os.path.join(Config.RESULTS_DIR, 'experiment_status.json')


def load_status():
    status_file = get_status_file()
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'in_progress': None}


def save_status(status):
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(get_status_file(), 'w') as f:
        json.dump(status, f, indent=2)


def backup_to_drive():
    """Backup results to Google Drive for persistence."""
    import shutil
    if os.path.exists('/content/drive/MyDrive'):
        os.makedirs(Config.DRIVE_BACKUP_DIR, exist_ok=True)
        if os.path.exists(Config.RESULTS_DIR):
            for item in os.listdir(Config.RESULTS_DIR):
                src = os.path.join(Config.RESULTS_DIR, item)
                dst = os.path.join(Config.DRIVE_BACKUP_DIR, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            print(f"Backed up to Google Drive: {Config.DRIVE_BACKUP_DIR}")


def restore_from_drive():
    """Restore results from Google Drive if available."""
    import shutil
    if os.path.exists(Config.DRIVE_BACKUP_DIR):
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        for item in os.listdir(Config.DRIVE_BACKUP_DIR):
            src = os.path.join(Config.DRIVE_BACKUP_DIR, item)
            dst = os.path.join(Config.RESULTS_DIR, item)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            else:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print(f"Restored from Google Drive: {Config.DRIVE_BACKUP_DIR}")
        return True
    return False


def run_single_experiment(actor_lr, critic_lr, run=0):
    exp_key = get_experiment_key(actor_lr, critic_lr, run)
    exp_dir = os.path.join(Config.RESULTS_DIR, exp_key)

    print(f"\n{'='*60}")
    print(f"FullNorm Experiment: actor_lr={actor_lr}, critic_lr={critic_lr}")
    print(f"{'='*60}")

    agent = FullNormExperimentAgent(actor_lr, critic_lr, exp_dir, Config.USE_CUDA)

    status = load_status()
    status['in_progress'] = exp_key
    save_status(status)

    try:
        while agent.n_episodes < Config.MAX_EPISODES:
            reward = agent.run_episode()

            if agent.n_episodes >= Config.EPISODES_BEFORE_TRAIN:
                agent.train_with_tracking()

            if agent.n_episodes % 10 == 0:
                print(f"Episode {agent.n_episodes}/{Config.MAX_EPISODES} | Reward: {reward:.4f}")

            # Backup to Drive every 100 episodes
            if agent.n_episodes % 100 == 0:
                agent.save_results()
                backup_to_drive()

            if agent.all_actors_stopped and agent.n_episodes > agent.stopping_episode + 100:
                print(f"Early stopping: actors stopped at {agent.stopping_episode}")
                break

        agent.save_results()
        agent.cleanup()

        status = load_status()
        if exp_key not in status['completed']:
            status['completed'].append(exp_key)
        status['in_progress'] = None
        save_status(status)

        backup_to_drive()
        return True

    except Exception as e:
        print(f"Error: {e}")
        raise


def run_all_experiments():
    """Run all 16 experiments with FullNorm actor/critic."""
    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Random seed set to: {Config.SEED}")

    print("="*70)
    print("FULL NORMALIZATION EXPERIMENT")
    print("Testing InputNorm + Post-activation LayerNorm (NO pre-tanh LN)")
    print("="*70)
    print(f"Actor LRs (HIGH to LOW): {Config.CLIENT_LRS}")
    print(f"Critic LRs (HIGH to LOW): {Config.MASTER_LRS}")
    print(f"Total experiments: {len(Config.CLIENT_LRS) * len(Config.MASTER_LRS)}")
    print(f"Max episodes: {Config.MAX_EPISODES}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    # Try to restore from Drive first
    restore_from_drive()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    status = load_status()
    print(f"\nCompleted: {len(status['completed'])} experiments")
    if status['in_progress']:
        print(f"Last in progress (will restart): {status['in_progress']}")

    experiments = []
    for actor_lr in Config.CLIENT_LRS:
        for critic_lr in Config.MASTER_LRS:
            exp_key = get_experiment_key(actor_lr, critic_lr)
            if exp_key not in status['completed']:
                experiments.append((actor_lr, critic_lr))

    print(f"Remaining: {len(experiments)} experiments\n")

    if status['in_progress']:
        for actor_lr, critic_lr in experiments:
            if get_experiment_key(actor_lr, critic_lr) == status['in_progress']:
                experiments.remove((actor_lr, critic_lr))
                experiments.insert(0, (actor_lr, critic_lr))
                break

    for actor_lr, critic_lr in experiments:
        try:
            run_single_experiment(actor_lr, critic_lr)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            backup_to_drive()
            break
        except Exception as e:
            print(f"Experiment failed: {e}")
            backup_to_drive()
            break

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    final_status = load_status()
    print(f"Completed: {len(final_status['completed'])}/{len(Config.CLIENT_LRS)*len(Config.MASTER_LRS)}")
    print("="*70)

    backup_to_drive()


if __name__ == "__main__":
    run_all_experiments()
