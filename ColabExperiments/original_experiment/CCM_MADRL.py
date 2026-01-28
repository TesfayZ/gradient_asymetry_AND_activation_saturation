import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt
from utils import to_tensor_var, get_device
from Model import ActorNetwork, CriticNetwork
from prioritized_memory import Memory
from mec_env import ENV_MODE, K_CHANNEL, S_E, N_UNITS


class CCM_MADDPG(object):
    def __init__(self, InfdexofResult, env, env_eval, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 memory_capacity=10000, target_tau=1, reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.tanh, actor_lr=0.0001, critic_lr=0.001,
                 optimizer_type="adam", max_grad_norm=None, batch_size=64, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=False):
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

        self.memory = Memory(memory_capacity)
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if epsilon_decay is None:
            print("epsilon_decay is None")
            exit()
        else:
            self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = get_device(self.use_cuda)

        self.target_tau = target_tau

        # Create networks
        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation) for _ in range(self.n_agents)]
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

        # Move to GPU if available
        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.actors_target[i].cuda()
            self.critics[0].cuda()
            self.critics_target[0].cuda()

        # Pre-allocate zero tensors for efficiency (used when no agents offload)
        self.zero_state = torch.zeros(self.state_dim, device=self.device)
        self.zero_action = torch.zeros(self.action_dim, device=self.device)

        # Tracking variables
        self.eval_episode_rewards = []
        self.server_episode_constraint_exceeds = []
        self.energy_episode_constraint_exceeds = []
        self.time_episode_constraint_exceeds = []
        self.eval_step_rewards = []
        self.mean_rewards = []

        self.episodes = []
        self.Training_episodes = []

        self.Training_episode_rewards = []
        self.Training_step_rewards = []

        self.InfdexofResult = InfdexofResult
        self.results = []
        self.Training_results = []
        self.serverconstraints = []
        self.energyconstraints = []
        self.timeconstraints = []

    def interact(self, MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES):
        while self.n_episodes < MAX_EPISODES:
            self.env_state = self.env.reset_mec()
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:
                self.evaluate(NUMBER_OF_EVAL_EPISODES)
                self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            self.agent_rewards = [[] for n in range(self.n_agents)]
            done = False
            while not done:
                state = self.env_state
                actor_action, critic_action, hybrid_action = self.choose_action(state, False)
                next_state, reward, done, _, _ = self.env.step_mec(hybrid_action)
                self.Training_step_rewards.append(np.mean(reward))
                if done:
                    self.Training_episode_rewards.append(np.sum(np.array(self.Training_step_rewards)))
                    self.Training_step_rewards = []
                    if self.done_penalty is not None:
                        reward = self.done_penalty
                    self.n_episodes += 1
                else:
                    self.env_state = next_state
                self.append_sample(state, actor_action, critic_action, reward, next_state, done)
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:
                self.train()

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_((1. - self.target_tau) * t.data + self.target_tau * s.data)

    def append_sample(self, states, actor_actions, critic_actions, rewards, next_states, dones):
        """Optimized append_sample with minimal GPU operations for single sample."""
        # Convert to tensors
        states_var = to_tensor_var(states, self.use_cuda).view(self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(self.n_agents, self.state_dim)

        whole_states_var = states_var.view(1, -1).squeeze(0)
        whole_actor_actions_var = actor_actions_var.view(1, -1).squeeze(0)
        whole_next_states_var = next_states_var.view(1, -1).squeeze(0)

        # Batched next actions from all target actors
        with torch.no_grad():
            nextactor_actions_list = []
            for agent_id in range(self.n_agents):
                next_action = self.actors_target[agent_id](next_states_var[agent_id:agent_id+1, :])
                nextactor_actions_list.append(next_action)
            nextactor_actions_var = torch.cat(nextactor_actions_list, dim=0)  # (n_agents, action_dim)
            whole_nextactor_actions_var = nextactor_actions_var.view(-1)

            # Find agents that proposed offload (action[0] >= 0)
            offload_mask = nextactor_actions_var[:, 0] >= 0

            if offload_mask.any():
                # Prepare batched inputs for all offloading agents
                offload_indices = torch.where(offload_mask)[0]
                n_offload = offload_indices.size(0)

                # Expand whole states/actions for each offloading agent
                batch_whole_states = whole_next_states_var.unsqueeze(0).expand(n_offload, -1)
                batch_whole_actions = whole_nextactor_actions_var.unsqueeze(0).expand(n_offload, -1)
                batch_per_states = next_states_var[offload_indices]
                batch_per_actions = nextactor_actions_var[offload_indices]

                # Single batched critic forward pass
                critic_input = torch.cat([batch_whole_states, batch_whole_actions, batch_per_states, batch_per_actions], dim=1)
                all_q_values = self.critics_target[0].forward_batched(critic_input).squeeze(-1)
                tar_perQ = all_q_values.max()
            else:
                # No agents offloading - use zero inputs
                critic_input = torch.cat([whole_next_states_var, whole_nextactor_actions_var, self.zero_state, self.zero_action]).unsqueeze(0)
                tar_perQ = self.critics_target[0].forward_batched(critic_input).squeeze()

            tar = self.reward_scale * rewards_var[0, 0] + self.reward_gamma * tar_perQ * (1. - dones)

        # Compute error for selected agents (critic_action == 1)
        selected_mask = critic_actions_var.squeeze(-1) == 1
        error = torch.tensor(0.0, device=self.device)

        if selected_mask.any():
            with torch.no_grad():
                selected_indices = torch.where(selected_mask)[0]
                n_selected = selected_indices.size(0)

                batch_whole_states = whole_states_var.unsqueeze(0).expand(n_selected, -1)
                batch_whole_actions = whole_actor_actions_var.unsqueeze(0).expand(n_selected, -1)
                batch_per_states = states_var[selected_indices]
                batch_per_actions = actor_actions_var[selected_indices]

                critic_input = torch.cat([batch_whole_states, batch_whole_actions, batch_per_states, batch_per_actions], dim=1)
                curr_q_values = self.critics[0].forward_batched(critic_input).squeeze(-1)
                error = ((tar - curr_q_values) ** 2).sum()
        else:
            with torch.no_grad():
                critic_input = torch.cat([whole_states_var, whole_actor_actions_var, self.zero_state, self.zero_action]).unsqueeze(0)
                curr_perQ = self.critics[0].forward_batched(critic_input).squeeze()
                error = (tar - curr_perQ) ** 2

        error_cpu = error.cpu().item()
        self.memory.addorupdate(error_cpu, (states, actor_actions, critic_actions, rewards, next_states, dones))

    def train(self):
        """Optimized training with batched GPU operations."""
        if self.n_episodes <= self.episodes_before_train:
            return

        # Sample batch
        tryfetch = 0
        while tryfetch < 3:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            mini_batch = np.array(mini_batch, dtype=object).transpose()
            if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
                tryfetch += 1
                if tryfetch >= 3:
                    print("mini_batch = ", mini_batch)
                    exit()
            else:
                break

        # Prepare batch data
        states = np.vstack(mini_batch[0])
        actor_actions = np.vstack(mini_batch[1])
        critic_actions = np.vstack(mini_batch[2])
        rewards = np.vstack(mini_batch[3])
        next_states = np.vstack(mini_batch[4])
        dones = mini_batch[5].astype(int)

        # Convert to tensors - single transfer to GPU
        states_var = to_tensor_var(states, self.use_cuda).view(self.batch_size, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(self.batch_size, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(self.batch_size, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(self.batch_size, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(self.batch_size, self.n_agents, self.state_dim)
        dones_var = to_tensor_var(dones, self.use_cuda).view(self.batch_size, 1)
        is_weights_var = to_tensor_var(np.array(is_weights), self.use_cuda).view(self.batch_size, 1)

        whole_states_var = states_var.view(self.batch_size, -1)
        whole_actor_actions_var = actor_actions_var.view(self.batch_size, -1)
        whole_next_states_var = next_states_var.view(self.batch_size, -1)

        # ==================== CRITIC UPDATE ====================
        # Compute next actions from all target actors (batched per actor)
        # Note: Original does NOT use no_grad here - target networks still compute gradients
        # but they are updated via soft_update only, not through backprop
        nextactor_actions_list = []
        for agent_id in range(self.n_agents):
            next_action = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            nextactor_actions_list.append(next_action)
        nextactor_actions_var = torch.stack(nextactor_actions_list, dim=1)  # (batch, n_agents, action_dim)
        whole_nextactor_actions_var = nextactor_actions_var.view(self.batch_size, -1)

        # Compute target Q-values with masked max
        # Offload mask: agents where action[0] >= 0
        offload_mask = nextactor_actions_var[:, :, 0] >= 0  # (batch, n_agents)

        # Prepare inputs for ALL agent-batch combinations
        # Shape: (batch * n_agents, input_dim)
        batch_indices = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, self.n_agents).reshape(-1)
        agent_indices = torch.arange(self.n_agents, device=self.device).unsqueeze(0).expand(self.batch_size, -1).reshape(-1)

        all_whole_next_states = whole_next_states_var[batch_indices]  # (batch*n_agents, state_dim*n_agents)
        all_whole_next_actions = whole_nextactor_actions_var[batch_indices]
        all_per_next_states = next_states_var.view(-1, self.state_dim)  # (batch*n_agents, state_dim)
        all_per_next_actions = nextactor_actions_var.view(-1, self.action_dim)  # (batch*n_agents, action_dim)

        # Single batched forward pass for all Q-values (no no_grad to match original)
        critic_input = torch.cat([all_whole_next_states, all_whole_next_actions, all_per_next_states, all_per_next_actions], dim=1)
        all_next_q = self.critics_target[0].forward_batched(critic_input).view(self.batch_size, self.n_agents)

        # Apply mask: set non-offloading agents to -inf for max
        masked_next_q = all_next_q.clone()
        masked_next_q[~offload_mask] = float('-inf')

        # Check if any agent offloads per batch
        any_offload = offload_mask.any(dim=1)  # (batch,)

        # Compute max Q for offloading agents, or use zero-input Q for non-offloading batches
        tar_perQ = torch.zeros(self.batch_size, device=self.device)
        tar_perQ[any_offload] = masked_next_q[any_offload].max(dim=1)[0]

        # For batches with no offloading agents, compute Q with zero inputs
        if (~any_offload).any():
            no_offload_indices = torch.where(~any_offload)[0]
            n_no_offload = no_offload_indices.size(0)
            zero_input = torch.cat([
                whole_next_states_var[no_offload_indices],
                whole_nextactor_actions_var[no_offload_indices],
                self.zero_state.unsqueeze(0).expand(n_no_offload, -1),
                self.zero_action.unsqueeze(0).expand(n_no_offload, -1)
            ], dim=1)
            tar_perQ[no_offload_indices] = self.critics_target[0].forward_batched(zero_input).squeeze(-1)

        # Target Q values
        target_q_all = self.reward_scale * rewards_var[:, 0, 0] + self.reward_gamma * tar_perQ * (1. - dones_var.squeeze(-1))

        # Compute current Q-values for selected agents (critic_action == 1)
        selected_mask = critic_actions_var.squeeze(-1) == 1  # (batch, n_agents)

        # Prepare inputs for current Q computation
        all_whole_states = whole_states_var[batch_indices]
        all_whole_actions = whole_actor_actions_var[batch_indices]
        all_per_states = states_var.view(-1, self.state_dim)
        all_per_actions = actor_actions_var.view(-1, self.action_dim)

        critic_input_current = torch.cat([all_whole_states, all_whole_actions, all_per_states, all_per_actions], dim=1)
        all_current_q = self.critics[0].forward_batched(critic_input_current).view(self.batch_size, self.n_agents)

        # Build loss using selected agents only (preserving gradient flow)
        errors = torch.zeros(self.batch_size, device=self.device)
        target_q_list = []
        current_q_list = []
        weight_list = []

        for b in range(self.batch_size):
            batch_selected = selected_mask[b]
            if batch_selected.any():
                selected_q = all_current_q[b][batch_selected]
                n_selected = selected_q.size(0)
                target_expanded = target_q_all[b].expand(n_selected)

                target_q_list.append(target_expanded * is_weights_var[b])
                current_q_list.append(selected_q * is_weights_var[b])
                errors[b] = ((selected_q.detach() - target_q_all[b]) ** 2).sum()
            else:
                # No agents selected - use zero-input Q
                zero_input = torch.cat([whole_states_var[b], whole_actor_actions_var[b], self.zero_state, self.zero_action]).unsqueeze(0)
                curr_q = self.critics[0].forward_batched(zero_input).squeeze()
                target_q_list.append(target_q_all[b:b+1] * is_weights_var[b])
                current_q_list.append(curr_q.unsqueeze(0) * is_weights_var[b])
                errors[b] = ((curr_q.detach() - target_q_all[b]) ** 2)

        current_q = torch.cat(current_q_list)
        target_q = torch.cat(target_q_list)

        # Match original: no detach on target_q, use requires_grad_(True)
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.requires_grad_(True)
        self.critics_optimizer[0].zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critics[0].parameters(), self.max_grad_norm)
        self.critics_optimizer[0].step()
        self._soft_update_target(self.critics_target[0], self.critics[0])

        # ==================== ACTOR UPDATE ====================
        for agent_id in range(self.n_agents):
            # Compute new actions from all actors
            newactor_actions_list = []
            for agent_in in range(self.n_agents):
                new_action = self.actors[agent_in](states_var[:, agent_in, :])
                newactor_actions_list.append(new_action)
            newactor_actions_var = torch.stack(newactor_actions_list, dim=1)  # (batch, n_agents, action_dim)
            whole_newactor_actions_var = newactor_actions_var.view(self.batch_size, -1)

            # Offload mask for new actions
            new_offload_mask = newactor_actions_var[:, :, 0] >= 0  # (batch, n_agents)

            # Compute Q-values for all agent-batch combinations
            all_whole_states_new = whole_states_var[batch_indices]
            all_whole_new_actions = whole_newactor_actions_var[batch_indices]
            all_per_states_new = states_var.view(-1, self.state_dim)
            all_per_new_actions = newactor_actions_var.view(-1, self.action_dim)

            critic_input_new = torch.cat([all_whole_states_new, all_whole_new_actions, all_per_states_new, all_per_new_actions], dim=1)
            all_new_q = self.critics[0].forward_batched(critic_input_new).view(self.batch_size, self.n_agents)

            # Build actor loss with conditional logic
            actor_loss_list = []
            for b in range(self.batch_size):
                batch_offload = new_offload_mask[b]
                if batch_offload.any():
                    offload_q = all_new_q[b][batch_offload]
                    max_q = offload_q.max()
                    actor_loss_list.append(max_q * is_weights_var[b, 0])
                else:
                    # No offloading - use zero-input Q
                    zero_input = torch.cat([whole_states_var[b], whole_newactor_actions_var[b], self.zero_state, self.zero_action]).unsqueeze(0)
                    q_val = self.critics[0].forward_batched(zero_input).squeeze()
                    actor_loss_list.append(q_val * is_weights_var[b, 0])

            actor_loss = -torch.stack(actor_loss_list).mean()
            actor_loss.requires_grad_(True)

            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()
            self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])

        # Update memory priorities
        errors_np = errors.detach().cpu().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors_np[i])

    def save_models(self, path):
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics': [critic.state_dict() for critic in self.critics],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
        }
        torch.save(checkpoint, path)

    def check_parameter_difference(self, model, loaded_state_dict):
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if name in loaded_state_dict:
                if not torch.equal(param, loaded_state_dict[name]):
                    return 1
                else:
                    return 0
            else:
                print(f"Parameter '{name}' is not present in the loaded checkpoint.")
                exit()

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) + self.action_lower_bound[i]
        return x

    def choose_action(self, state, evaluation):
        """Optimized action selection with batched actor inference."""
        state_var = to_tensor_var(state, self.use_cuda).view(self.n_agents, self.state_dim)

        # Batched actor inference
        actor_action = np.zeros((self.n_agents, self.action_dim))
        critic_action = np.zeros((self.n_agents))
        hybrid_action = np.zeros((self.n_agents, self.action_dim))

        with torch.no_grad():
            for agent_id in range(self.n_agents):
                action_var = self.actors[agent_id](state_var[agent_id:agent_id+1, :])
                actor_action[agent_id] = action_var.cpu().numpy()[0]

        if not evaluation:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.n_episodes / self.epsilon_decay)
            noise = np.random.randn(self.n_agents, self.action_dim) * epsilon
            actor_action += noise
            actor_action = np.clip(actor_action, -1, 1)

        # Get critic_action and hybrid actions
        hybrid_action = actor_action.copy()
        proposed = np.count_nonzero(actor_action[:, 0] >= 0)
        proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
        sumofproposed = np.sum(state[proposed_indices, 3])

        if ENV_MODE == "H2":
            constraint = K_CHANNEL
        elif ENV_MODE == "TOBM":
            constraint = N_UNITS
        else:
            print("Unknown env_mode ", ENV_MODE)
            exit()

        if proposed > 0:
            if proposed > constraint or sumofproposed > S_E:
                if not evaluation and (np.random.rand() <= epsilon):
                    # Exploration: random selection
                    agent_list = np.arange(self.n_agents).tolist()
                    random.shuffle(agent_list)
                    randomorder = random.sample(agent_list, constraint)
                    sizeaccepted = np.sum(state[randomorder, 3])
                    while sizeaccepted > S_E:
                        element_to_delete = random.choice(randomorder)
                        randomorder.remove(element_to_delete)
                        sizeaccepted = np.sum(state[randomorder, 3])
                    critic_action[randomorder] = 1
                else:
                    # Exploitation: Q-value based selection
                    with torch.no_grad():
                        states_var = to_tensor_var(state, self.use_cuda).view(self.n_agents, self.state_dim)
                        whole_states_var = states_var.view(-1)
                        actor_action_var = to_tensor_var(actor_action, self.use_cuda).view(self.n_agents, self.action_dim)
                        whole_actions_var = actor_action_var.view(-1)

                        # Batched Q-value computation for all agents that proposed offload
                        offload_mask = actor_action[:, 0] >= 0
                        offload_indices = np.where(offload_mask)[0]

                        if len(offload_indices) > 0:
                            n_offload = len(offload_indices)
                            batch_whole_states = whole_states_var.unsqueeze(0).expand(n_offload, -1)
                            batch_whole_actions = whole_actions_var.unsqueeze(0).expand(n_offload, -1)
                            batch_per_states = states_var[offload_indices]
                            batch_per_actions = actor_action_var[offload_indices]

                            critic_input = torch.cat([batch_whole_states, batch_whole_actions, batch_per_states, batch_per_actions], dim=1)
                            q_values = self.critics[0].forward_batched(critic_input).squeeze(-1).cpu().numpy()

                            # Create full Q-value array with -inf for non-offloading agents
                            critic_action_Qs = np.full(self.n_agents, -np.inf)
                            critic_action_Qs[offload_indices] = q_values
                        else:
                            critic_action_Qs = np.full(self.n_agents, -np.inf)

                    # Sort and select top-K within constraints
                    sorted_indices = np.argsort(critic_action_Qs)[::-1]
                    countaccepted = 0
                    sizeaccepted = 0
                    for agentid in range(self.n_agents):
                        idx = sorted_indices[agentid]
                        if actor_action[idx, 0] >= 0 and countaccepted < constraint and sizeaccepted + state[idx, 3] < S_E:
                            critic_action[idx] = 1
                            countaccepted += 1
                            sizeaccepted += state[idx, 3]
            else:
                # All proposed can be accepted
                for agentid in range(self.n_agents):
                    if hybrid_action[agentid, 0] >= 0:
                        critic_action[agentid] = 1

        hybrid_action[:, 0] = critic_action

        # Scale actions to bounds
        b = 1
        a = -b
        for n in range(self.n_agents):
            hybrid_action[n][1] = self.getactionbound(a, b, hybrid_action[n][1], 1)
            hybrid_action[n][2] = self.getactionbound(a, b, hybrid_action[n][2], 2)

        return actor_action, critic_action, hybrid_action

    def evaluate(self, EVAL_EPISODES):
        if ENV_MODE == "H2":
            constraint = K_CHANNEL
        elif ENV_MODE == "TOBM":
            constraint = N_UNITS
        else:
            print("Unknown env_mode ", ENV_MODE)
            exit()

        for i in range(EVAL_EPISODES):
            self.eval_env_state = self.env_eval.reset_mec(i)
            self.eval_step_rewards = []
            self.server_step_constraint_exceeds = 0
            self.energy_step_constraint_exceeds = 0
            self.time_step_constraint_exceeds = 0
            done = False

            while not done:
                state = self.eval_env_state
                actor_action, critic_action, hybrid_action = self.choose_action(state, True)
                proposed = np.count_nonzero(actor_action[:, 0] >= 0)
                proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
                sumofproposed = np.sum(state[proposed_indices, 3])
                next_state, reward, done, eneryconstraint_exceeds, timeconstraint_exceeds = self.env_eval.step_mec(hybrid_action)
                self.eval_step_rewards.append(np.mean(reward))
                self.energy_step_constraint_exceeds += eneryconstraint_exceeds
                self.time_step_constraint_exceeds += timeconstraint_exceeds

                if proposed > constraint or sumofproposed > S_E:
                    self.server_step_constraint_exceeds += 1

                if done:
                    self.eval_episode_rewards.append(np.sum(np.array(self.eval_step_rewards)))
                    self.server_episode_constraint_exceeds.append(self.server_step_constraint_exceeds / len(self.eval_step_rewards))
                    self.energy_episode_constraint_exceeds.append(self.energy_step_constraint_exceeds / len(self.eval_step_rewards))
                    self.time_episode_constraint_exceeds.append(self.time_step_constraint_exceeds / len(self.eval_step_rewards))
                    self.eval_step_rewards = []
                    self.server_step_constraint_exceeds = 0
                    self.energy_step_constraint_exceeds = 0
                    self.time_step_constraint_exceeds = 0
                    if self.done_penalty is not None:
                        reward = self.done_penalty
                else:
                    self.eval_env_state = next_state

            if i == EVAL_EPISODES - 1 and done:
                mean_reward = np.mean(np.array(self.eval_episode_rewards))
                mean_constraint = np.mean(np.array(self.server_episode_constraint_exceeds))
                mean_energyconstraint = np.mean(np.array(self.energy_episode_constraint_exceeds))
                mean_timeconstraint = np.mean(np.array(self.time_episode_constraint_exceeds))
                self.eval_episode_rewards = []
                self.server_episode_constraint_exceeds = []
                self.energy_episode_constraint_exceeds = []
                self.time_episode_constraint_exceeds = []
                self.mean_rewards.append(mean_reward)
                self.episodes.append(self.n_episodes + 1)
                self.results.append(mean_reward)
                self.serverconstraints.append(mean_constraint)
                self.energyconstraints.append(mean_energyconstraint)
                self.timeconstraints.append(mean_timeconstraint)
                arrayresults = np.array(self.results)
                arrayserver = np.array(self.serverconstraints)
                arrayenergy = np.array(self.energyconstraints)
                arraytime = np.array(self.timeconstraints)
                savetxt('./CSV/results/CCM_MADRL' + str(self.InfdexofResult) + '.csv', arrayresults)
                savetxt('./CSV/Server_constraints/CCM_MADRL' + str(self.InfdexofResult) + '.csv', arrayserver)
                savetxt('./CSV/Energy_constraints/CCM_MADRL' + str(self.InfdexofResult) + '.csv', arrayenergy)
                savetxt('./CSV/Time_constraints/CCM_MADRL' + str(self.InfdexofResult) + '.csv', arraytime)

    def evaluateAtTraining(self, EVAL_EPISODES):
        mean_reward = np.mean(np.array(self.Training_episode_rewards))
        self.Training_episode_rewards = []
        self.Training_episodes.append(self.n_episodes + 1)
        self.Training_results.append(mean_reward)
        arrayresults = np.array(self.Training_results)
        savetxt('./CSV/AtTraining/CCM_MADRL' + self.InfdexofResult + '.csv', arrayresults)
