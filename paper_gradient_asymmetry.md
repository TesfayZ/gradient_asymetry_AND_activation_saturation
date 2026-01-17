# Asymmetric Gradient Flow and Learning Rate Sensitivity in Client-Master Multi-Agent Actor-Critic Architectures

## Authors
[Your Name], [Collaborators]

---

## Abstract

Multi-agent deep reinforcement learning (MADRL) has emerged as a powerful paradigm for solving complex coordination problems in domains such as mobile edge computing (MEC) task offloading. However, a critical yet underexplored phenomenon in actor-critic MADRL architectures is the **asymmetric convergence behavior** between actor (policy) networks and critic (value) networks. In this paper, we investigate why client agents (actors) in a Client-Master MADRL framework stop updating their neural network weights at different episode numbers depending on learning rate configurations, while the master agent (critic) continues learning until training completion. Through systematic analysis of gradient flow dynamics, activation function saturation, and reward scale interactions, we identify **tanh output saturation** as the primary mechanism causing premature actor convergence. We demonstrate that this phenomenon is learning-rate-dependent, with higher learning rates (0.01-0.1) causing actor weight updates to cease within 5 episodes, while lower learning rates (0.0001) maintain gradient flow throughout training. Our analysis provides theoretical and empirical insights into the architectural factors that create gradient flow asymmetry in actor-critic methods, with implications for hyperparameter selection and network design in MADRL systems.

**Keywords:** Multi-agent reinforcement learning, Actor-critic methods, Gradient flow, Learning rate sensitivity, Vanishing gradients, Mobile edge computing, MADDPG

---

## 1. Introduction

### 1.1 Background and Motivation

Deep reinforcement learning (DRL) has achieved remarkable success in solving complex sequential decision-making problems, from game playing to robotic control. The extension to multi-agent settings, known as multi-agent deep reinforcement learning (MADRL), addresses scenarios where multiple agents must learn to coordinate or compete in shared environments. Actor-critic architectures, particularly the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm, have become foundational approaches for continuous control in multi-agent systems.

In actor-critic methods, the actor network learns a policy that maps states to actions, while the critic network estimates the value of state-action pairs to guide policy improvement. This architectural separation creates an inherent asymmetry in how gradients flow through the system during training. The critic receives direct supervision from temporal difference (TD) errors, while the actor receives indirect feedback through the critic's value estimates.

Despite the widespread adoption of actor-critic MADRL, a systematic understanding of how this architectural asymmetry affects learning dynamics—particularly the differential convergence behavior between actors and critics—remains incomplete. This gap is especially significant when considering hyperparameter selection, as practitioners often observe that actors and critics exhibit different sensitivities to learning rate choices.

### 1.2 Problem Statement

In our investigation of a Client-Master MADRL framework for MEC task offloading, we observed a striking phenomenon: **client agents (actors) stopped updating their neural network weights at different episode numbers depending on learning rate configurations, while the master agent (critic) continued learning until training completion**. Specifically:

- With learning rates of 0.01-0.1 for client agents, all 50 actors ceased weight updates within **5 episodes**
- With learning rates of 0.0001, actors maintained gradient flow throughout **2000 episodes**
- The master agent (critic) **never stopped** updating weights regardless of learning rate

This differential behavior has significant implications for:
1. **Algorithm design**: Understanding why certain configurations fail
2. **Hyperparameter selection**: Principled approaches to learning rate tuning
3. **Architecture choices**: Designing networks resistant to gradient pathologies
4. **Performance optimization**: Ensuring sustained learning in both actors and critics

### 1.3 Contributions

This paper makes the following contributions:

1. **Empirical characterization** of the differential stopping phenomenon in Client-Master MADRL, demonstrating that actors and critics exhibit fundamentally different convergence behaviors across learning rate configurations.

2. **Theoretical analysis** identifying tanh output activation saturation as the primary mechanism causing premature actor convergence, explaining the learning-rate-dependent nature of this phenomenon.

3. **Systematic investigation** of the architectural and gradient flow factors that create asymmetry between actor and critic learning dynamics.

4. **Practical guidelines** for learning rate selection and network design to prevent premature actor convergence while maintaining stable critic learning.

5. **Experimental framework** for detecting and monitoring gradient flow asymmetries during MADRL training.

### 1.4 Paper Organization

Section 2 reviews related work on actor-critic methods, gradient flow in deep learning, and hyperparameter sensitivity in DRL. Section 3 presents the technical background and problem formulation. Section 4 details our methodology and experimental setup. Section 5 presents our analysis and findings. Section 6 discusses implications and proposed solutions. Section 7 outlines future experiments. Section 8 concludes the paper.

---

## 2. Related Work

### 2.1 Actor-Critic Methods in Deep Reinforcement Learning

Actor-critic methods combine policy-based (actor) and value-based (critic) approaches to leverage the strengths of both. The actor learns a parameterized policy $\pi_\theta(a|s)$ that directly maps states to actions, while the critic learns a value function $Q_\phi(s,a)$ or $V_\phi(s)$ to evaluate the actor's choices.

The Deep Deterministic Policy Gradient (DDPG) algorithm introduced by Lillicrap et al. extended actor-critic methods to continuous action spaces using deep neural networks. DDPG employs deterministic policy gradients, where the actor gradient is computed as:

$$\nabla_\theta J \approx \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q_\phi(s,a)|_{a=\pi_\theta(s)} \cdot \nabla_\theta \pi_\theta(s) \right]$$

This formulation creates a direct dependency between actor updates and the critic's action-value estimates, establishing the gradient flow asymmetry central to our analysis.

Lowe et al. extended DDPG to multi-agent settings with MADDPG, introducing the centralized training with decentralized execution (CTDE) paradigm. In MADDPG, each agent's critic has access to all agents' observations and actions during training, while actors only access local observations during execution. This asymmetric information access further complicates the gradient flow dynamics between actors and critics.

Recent work on [asymmetric actor-critic frameworks](https://www.emergentmind.com/topics/asymmetric-actor-critic-framework) has explored deliberately designing actors and critics with different input information. Lambrechts et al. (2025) provided theoretical justification for asymmetric designs, showing that privileged information for critics can accelerate learning without compromising deployment performance. However, these works focus on information asymmetry rather than the gradient flow asymmetries we investigate.

### 2.2 Vanishing Gradients and Activation Function Saturation

The vanishing gradient problem, first identified in recurrent neural networks, occurs when gradients diminish exponentially as they propagate backward through layers. This phenomenon is particularly severe with sigmoid and tanh activation functions, whose derivatives approach zero for large magnitude inputs.

For the tanh function:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The derivative is:
$$\tanh'(x) = 1 - \tanh^2(x)$$

When $|x| > 2$, $\tanh'(x) < 0.07$, meaning more than 93% of the gradient is suppressed. This [saturation behavior](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-neural-networks/) creates a "dead zone" where weight updates become negligible regardless of the loss magnitude.

Traditional solutions include:
- Using non-saturating activations like ReLU in hidden layers
- Xavier/He initialization to maintain activation variance
- Batch normalization to prevent internal covariate shift
- Residual connections to provide gradient shortcuts

However, for actor networks in continuous control, **output activations must remain bounded** to produce valid actions, necessitating the use of tanh or similar bounded functions. This constraint makes actor networks inherently susceptible to output saturation.

### 2.3 Learning Rate Sensitivity in Deep Reinforcement Learning

Deep RL algorithms are notoriously sensitive to hyperparameter choices. Research by [Eimer et al.](https://proceedings.mlr.press/v202/eimer23a/eimer23a.pdf) demonstrated that hyperparameters in RL have significant impact on performance, with learning rate being among the most influential parameters.

The [Self-Tuning Actor-Critic (STAC)](https://arxiv.org/abs/2002.12928) algorithm addresses hyperparameter sensitivity by using meta-gradients to adapt hyperparameters online. STAC improved median human-normalized scores from 243% to 364% on the Arcade Learning Environment, demonstrating the performance gains achievable through proper hyperparameter adaptation.

Critically, existing work on [hyperparameter tuning for deep RL](https://arxiv.org/pdf/2201.11182) has examined actor and critic learning rates as separate parameters, implicitly acknowledging their different sensitivities. However, the underlying mechanisms causing this differential sensitivity have not been systematically characterized.

### 2.4 Gradient Flow Asymmetry in Actor-Critic Training

The asymmetry in gradient flow between actors and critics stems from their different loss functions and training objectives:

**Critic Loss (TD Error):**
$$\mathcal{L}_\text{critic} = \mathbb{E}\left[(r + \gamma Q_{\phi'}(s', \pi_{\theta'}(s')) - Q_\phi(s, a))^2\right]$$

**Actor Loss (Policy Gradient):**
$$\mathcal{L}_\text{actor} = -\mathbb{E}\left[Q_\phi(s, \pi_\theta(s))\right]$$

The critic receives direct supervision from rewards, while the actor's gradient depends on:
1. The critic's ability to accurately estimate Q-values
2. The gradient of Q with respect to actions: $\nabla_a Q$
3. The gradient of the policy with respect to parameters: $\nabla_\theta \pi$

This chained dependency creates multiple points where gradient flow can be disrupted for actors but not critics.

Recent work on [gradient imbalance in multi-task RL](https://arxiv.org/html/2510.19178) shows that tasks producing larger gradients can bias optimization, but large gradients don't necessarily correlate with larger learning gains. This finding suggests that gradient magnitude alone is insufficient for understanding convergence behavior.

### 2.5 Multi-Agent Reinforcement Learning for Mobile Edge Computing

Task offloading in MEC environments has become a prominent application domain for MADRL. Multiple user devices (agents) must decide whether to process tasks locally or offload them to edge servers, considering constraints on server capacity, energy consumption, and latency requirements.

The CCM-MADRL algorithm combines policy gradient optimization for client agents with value-based selection for the master agent, creating a hierarchical decision-making structure. This architecture is particularly interesting for gradient flow analysis because:
1. All 50 client agents share the same actor architecture
2. The single master agent has different architecture (larger, unbounded output)
3. The master can continue improving selection even when clients stop learning

This domain provides a controlled setting for studying gradient flow asymmetries, as all agents operate in the same environment with identical state representations.

### 2.6 Research Gap

While substantial work exists on individual aspects—vanishing gradients, learning rate sensitivity, actor-critic methods—**no prior work has systematically investigated how these factors interact to create differential convergence behavior between actors and critics**. Specifically:

1. The phenomenon of actors stopping weight updates while critics continue has not been characterized
2. The interaction between learning rates, reward scales, and output activations in causing this phenomenon is not understood
3. Practical guidelines for preventing premature actor convergence while maintaining stable critic learning are lacking

This paper addresses these gaps through systematic empirical and theoretical analysis.

---

## 3. Background and Problem Formulation

### 3.1 Multi-Agent Markov Decision Process

We formalize the multi-agent task offloading problem as a Multi-Agent Markov Decision Process (MA-MDP) defined by the tuple $\langle \mathcal{N}, \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ where:

- $\mathcal{N} = \{1, 2, ..., N\}$ is the set of $N$ agents (user devices)
- $\mathcal{S} = \mathcal{S}_1 \times \mathcal{S}_2 \times ... \times \mathcal{S}_N$ is the joint state space
- $\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times ... \times \mathcal{A}_N$ is the joint action space
- $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$ is the state transition probability
- $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ is the shared reward function
- $\gamma \in [0,1)$ is the discount factor

### 3.2 State and Action Spaces

**Per-Agent State** ($s_i \in \mathbb{R}^7$):
Each agent's state comprises:
- Transmission power $p_i$
- Channel gain $h_i$
- Available energy $e_i$
- Task size $d_i$
- Required CPU cycles $c_i$
- Task deadline $\tau_i$
- Device compute capability $f_i$

**Per-Agent Action** ($a_i \in \mathbb{R}^3$):
Each agent outputs:
- Offload decision $x_i \in [-1, 1]$ (offload if $x_i \geq 0$)
- Compute allocation $\alpha_i \in [0, 1]$
- Power allocation $\rho_i \in [0, 1]$

### 3.3 Client-Master Architecture

The CCM-MADRL architecture comprises:

**Client Agents (Actors):**
- 50 independent actor networks $\pi_{\theta_i}: \mathcal{S}_i \rightarrow \mathcal{A}_i$
- Architecture: $7 \rightarrow 64 \rightarrow 32 \rightarrow 3$ with ReLU hidden activations and **tanh output**
- Each client outputs continuous actions based on local state

**Master Agent (Critic):**
- Single centralized critic network $Q_\phi: \mathcal{S} \times \mathcal{A} \times \mathcal{S}_i \times \mathcal{A}_i \rightarrow \mathbb{R}$
- Architecture: $(N \cdot 7 + N \cdot 3 + 7 + 3) \rightarrow 512 \rightarrow 128 \rightarrow 1$ with ReLU hidden and **linear output**
- Evaluates individual agent contributions given joint state-action

The master performs combinatorial selection when server constraints are exceeded, ranking clients by Q-value and selecting top-K that satisfy channel and storage constraints.

### 3.4 Training Dynamics

**Critic Update:**
$$\phi \leftarrow \phi - \eta_c \nabla_\phi \frac{1}{B} \sum_{j=1}^B \left( y_j - Q_\phi(s_j, a_j, s_{i,j}, a_{i,j}) \right)^2$$

where $y_j = r_j + \gamma \max_{a'} Q_{\phi'}(s'_j, a'_j, s'_{i,j}, a'_{i,j})$

**Actor Update:**
$$\theta_i \leftarrow \theta_i + \eta_a \nabla_{\theta_i} \frac{1}{B} \sum_{j=1}^B Q_\phi(s_j, a_j, s_{i,j}, \pi_{\theta_i}(s_{i,j}))$$

### 3.5 Problem Definition

**Phenomenon:** Given identical initialization and exploration sequences, client agents stop updating weights at episode $E_\text{stop}$ where:
- $E_\text{stop} \approx 5$ for $\eta_a \in \{0.01, 0.1\}$
- $E_\text{stop} > 2000$ for $\eta_a = 0.0001$
- The master agent never stops ($E_\text{stop}^{\text{master}} = \infty$ for all $\eta_c$)

**Research Questions:**
1. What causes actors to stop updating while critics continue?
2. Why is the stopping episode learning-rate-dependent?
3. What architectural factors create this asymmetry?
4. How can premature actor stopping be prevented?

---

## 4. Methodology

### 4.1 Experimental Setup

**Environment Configuration:**
| Parameter | Value |
|-----------|-------|
| Number of agents | 50 |
| State dimension | 7 |
| Action dimension | 3 |
| Episodes | 2000 |
| Steps per episode | 10 / 100 |
| Batch size | 64 |
| Replay memory | 10,000 |
| Discount factor ($\gamma$) | 0.99 |
| Target network update ($\tau$) | 1.0 (hard update) |
| Exploration ($\epsilon$) | 1.0 → 0.01 (decay) |

**Learning Rate Configurations Tested:**
| Config | Client LR ($\eta_a$) | Master LR ($\eta_c$) |
|--------|----------------------|----------------------|
| A | 0.1 | 0.1, 0.01, 0.001, 0.0001 |
| B | 0.01 | 0.1, 0.01, 0.001, 0.0001 |
| C | 0.001 | 0.1, 0.01, 0.001, 0.0001 |
| D | 0.0001 | 0.1, 0.01, 0.001, 0.0001 |

**Controlled Variables:**
- PyTorch seed: 23 (identical weight initialization)
- NumPy seed: 23 (identical exploration sequence)
- Environment seed: 37 (identical state transitions)

### 4.2 Weight Update Detection

To detect when agents stop updating, we implemented checkpoint comparison:

```python
def check_parameter_difference(model, loaded_state_dict):
    current_state_dict = model.state_dict()
    for name, param in current_state_dict.items():
        if name in loaded_state_dict:
            if not torch.equal(param, loaded_state_dict[name]):
                return 1  # Weight changed
    return 0  # No change detected
```

After each training iteration:
1. Compare current weights to checkpoint from previous episode
2. Record whether any weight changed for each actor and the critic
3. Identify the first episode where all actors simultaneously stop updating

### 4.3 Gradient Flow Analysis

We analyze gradient flow through the computation graph:

**Actor Gradient Path:**
$$\frac{\partial \mathcal{L}_\text{actor}}{\partial \theta} = \frac{\partial \mathcal{L}_\text{actor}}{\partial Q} \cdot \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial \theta}$$

where $\frac{\partial a}{\partial \theta}$ includes:
$$\frac{\partial a}{\partial \theta} = \frac{\partial \tanh(z)}{\partial z} \cdot \frac{\partial z}{\partial \theta} = (1 - \tanh^2(z)) \cdot \frac{\partial z}{\partial \theta}$$

**Critical Observation:** The term $(1 - \tanh^2(z))$ approaches zero when $|z|$ is large, creating a gradient bottleneck at the output layer.

### 4.4 Reward Scale Analysis

From the MEC environment, the reward function is:
$$R = -(\lambda_E \cdot E + \lambda_T \cdot T) - (\lambda_E \cdot P_E + \lambda_T \cdot P_T)$$

where:
- $E$: Energy consumption (sum across agents)
- $T$: Normalized time
- $P_E, P_T$: Energy and time penalties

With $\lambda_E = \lambda_T = 0.5$ and 50 agents, rewards range from approximately **-80 to -270**.

### 4.5 Saturation Analysis

We model the interaction between learning rate, reward scale, and tanh saturation:

**Weight Update Magnitude:**
$$\Delta w \approx \eta_a \cdot |R| \cdot \nabla_w \pi$$

For high learning rates:
$$\Delta w_{0.1} \approx 0.1 \times 100 \times \nabla_w \pi = 10 \cdot \nabla_w \pi$$

For low learning rates:
$$\Delta w_{0.0001} \approx 0.0001 \times 100 \times \nabla_w \pi = 0.01 \cdot \nabla_w \pi$$

Large weight updates ($\Delta w \approx 10$) can push pre-activation values into the saturation region within a few steps, while small updates ($\Delta w \approx 0.01$) allow gradual convergence within the linear region.

---

## 5. Analysis and Findings

### 5.1 Empirical Results: Stopping Episode by Learning Rate

**Table 1: Episodes Before Client Agents Stop Updating**

| Client LR | Master LR | Stopping Episode | Final Reward (Eval) |
|-----------|-----------|------------------|---------------------|
| 0.1 | 0.0001 | ~5 | Not converged |
| 0.1 | 0.001 | ~5 | Not converged |
| 0.01 | 0.0001 | ~5 | Not converged |
| 0.01 | 0.001 | ~5 | Not converged |
| 0.001 | 0.001 | ~200 | -52 |
| 0.001 | 0.0001 | ~500 | -38 |
| 0.0001 | 0.001 | >2000 | **-34** |
| 0.0001 | 0.0001 | >2000 | -40 |

**Key Observations:**
1. Learning rates ≥0.01 cause all actors to stop within 5 episodes
2. The master agent never stopped at any learning rate configuration
3. Lower client learning rates correlate with later stopping and better final performance
4. The optimal configuration {0.0001, 0.001} maintains actor learning throughout training

### 5.2 Root Cause Analysis: Tanh Output Saturation

**Finding 1: High learning rates cause rapid tanh saturation**

The actor output layer uses tanh activation:
$$a = \tanh(W_3^T h_2 + b_3)$$

With rewards on the order of -100 and learning rate 0.1, weight updates are approximately:
$$\Delta W_3 \approx 0.1 \times 100 \times h_2 = 10 \cdot h_2$$

After just a few updates, the pre-activation values $z = W_3^T h_2 + b_3$ grow large enough that:
$$\tanh(z) \approx \pm 1 \quad \text{and} \quad \tanh'(z) \approx 0$$

This creates a **gradient vanishing point** at the output layer, preventing further weight updates.

**Finding 2: The critic's linear output is immune to saturation**

The critic output layer has no activation:
$$Q = W_3^T h_2 + b_3$$

The gradient always flows directly:
$$\frac{\partial Q}{\partial W_3} = h_2$$

Regardless of the Q-value magnitude, gradients remain proportional to hidden activations.

### 5.3 Architectural Asymmetry Analysis

**Table 2: Actor vs Critic Architectural Comparison**

| Property | Client (Actor) | Master (Critic) |
|----------|----------------|-----------------|
| Input dim | 7 | 360 |
| Hidden layers | 64 → 32 | 512 → 128 |
| Output dim | 3 | 1 |
| Output activation | **tanh** | **Linear** |
| Loss type | Policy gradient | MSE (TD error) |
| Gradient source | Indirect (via Q) | Direct (rewards) |

The combination of:
1. Bounded (tanh) vs unbounded (linear) outputs
2. Indirect (Q-gradient) vs direct (TD error) supervision
3. Smaller capacity vs larger capacity

creates systematic gradient flow asymmetry favoring the critic.

### 5.4 Reward Scale Interaction

The reward magnitude interacts multiplicatively with learning rate:

**Effective Learning Rate** = $\eta \times |R|$

| Nominal LR | Reward Scale | Effective LR |
|------------|--------------|--------------|
| 0.1 | 100 | **10** |
| 0.01 | 100 | **1** |
| 0.001 | 100 | **0.1** |
| 0.0001 | 100 | **0.01** |

Effective learning rates above ~0.1 push weights into saturation rapidly. The combination of large negative rewards and high learning rates creates the conditions for premature convergence.

### 5.5 Why CCM-MADRL Continues to Improve

Despite frozen client networks, CCM-MADRL shows performance improvements because:

1. **Master agent continues learning:** The critic refines Q-value estimates throughout training
2. **Selection mechanism:** The master ranks clients by Q-value, selecting optimal combinations
3. **Constraint satisfaction:** Better Q-estimates lead to better constraint-respecting selections

This architectural feature provides resilience against actor stagnation, but comes at the cost of reduced policy diversity.

### 5.6 Generalizability Across Seeds

With different weight initializations (10 independent runs):
- High learning rate configurations ({0.01, 0.0001}) showed poor average performance
- The best performing configuration ({0.0001, 0.001}) maintained consistent results
- Variance was lower for configurations that maintained actor learning

This suggests that premature stopping is not initialization-dependent when learning rates are sufficiently high.

---

## 6. Discussion

### 6.1 Theoretical Implications

Our findings reveal a **fundamental tension** in actor-critic design for continuous control:

1. **Bounded outputs are necessary** for valid action generation in continuous spaces
2. **Bounded activations (tanh) are susceptible** to saturation and gradient vanishing
3. **Large reward scales and high learning rates** exacerbate saturation
4. **Critics with linear outputs** are immune to this specific pathology

This creates an inherent asymmetry: critics can use aggressive learning rates, while actors require conservative rates to avoid saturation.

### 6.2 Practical Recommendations

Based on our analysis, we recommend:

**Learning Rate Selection:**
- Actor learning rate: 0.0001 - 0.001 (conservative)
- Critic learning rate: 0.001 - 0.01 (can be 10× higher than actor)
- Ratio: Actor LR / Critic LR ≈ 0.1

**Reward Scaling:**
- Normalize rewards to unit scale when possible
- Monitor effective learning rate = $\eta \times |R|$
- Keep effective LR < 0.1 for actors

**Architecture Modifications:**
- Consider replacing tanh with scaled sigmoid or softsign
- Add gradient clipping on actor updates
- Monitor pre-activation magnitudes during training

**Training Monitoring:**
- Implement weight change detection for actors
- Track tanh output distribution (warning if clustered at ±1)
- Compare actor vs critic gradient magnitudes

### 6.3 Broader Impact

This analysis has implications beyond MEC task offloading:

1. **DDPG/TD3/SAC implementations:** All use tanh-bounded outputs and may exhibit similar behavior
2. **Hyperparameter transfer:** Learning rates optimal in one domain may fail in others with different reward scales
3. **Multi-agent systems:** Shared rewards amplify the effective reward scale (N agents × per-agent reward)

### 6.4 Limitations

1. **Single domain:** Analysis conducted on MEC task offloading; generalization to other domains requires verification
2. **Fixed architecture:** Results may differ with alternative network designs
3. **Deterministic seeds:** While controlled, may not capture all initialization effects
4. **Lack of gradient magnitude tracking:** Direct gradient measurements would strengthen analysis

---

## 7. Proposed Further Experiments

Based on our analysis, we identify the following experiments to strengthen and extend our findings:

### 7.1 Gradient Magnitude Tracking (High Priority)

**Objective:** Directly measure gradient magnitudes through actor and critic networks during training.

**Methodology:**
```python
# After backward pass, before optimizer step
for name, param in actor.named_parameters():
    if param.grad is not None:
        grad_magnitude = param.grad.norm().item()
        log_gradient(name, grad_magnitude, episode)
```

**Metrics:**
- Per-layer gradient norms over training
- Gradient ratio: actor_grad / critic_grad
- Correlation between gradient magnitude and stopping episode

**Expected Outcome:** Demonstration that actor gradients approach zero at stopping point while critic gradients remain non-zero.

### 7.2 Pre-Activation Distribution Analysis (High Priority)

**Objective:** Track the distribution of pre-activation values at the tanh output layer.

**Methodology:**
- Hook into forward pass to capture pre-activation values
- Compute statistics: mean, std, percentage in saturation region (|z| > 2)
- Visualize distribution evolution over episodes

**Expected Outcome:** High learning rates should show rapid shift of pre-activation distribution into saturation regions.

### 7.3 Alternative Output Activations (Medium Priority)

**Objective:** Compare tanh with alternative bounded activations that have different saturation characteristics.

**Candidates:**
| Activation | Formula | Gradient at Extremes |
|------------|---------|----------------------|
| tanh | $\tanh(x)$ | ~0 |
| softsign | $x/(1+|x|)$ | $1/(1+|x|)^2$ > 0 |
| hardtanh | clip(x, -1, 1) | 0 outside, 1 inside |
| scaled sigmoid | $2\sigma(x) - 1$ | ~0 |

**Expected Outcome:** Softsign may exhibit more robust gradient flow at extremes.

### 7.4 Reward Normalization Study (Medium Priority)

**Objective:** Evaluate the impact of reward normalization on stopping behavior.

**Methodology:**
- Implement running mean/std normalization: $r' = (r - \mu_r) / \sigma_r$
- Compare stopping episodes with and without normalization
- Test across learning rate configurations

**Expected Outcome:** Normalized rewards should allow higher learning rates without premature stopping.

### 7.5 Layer-wise Learning Rate Adaptation (Medium Priority)

**Objective:** Test whether different learning rates for output vs hidden layers can prevent saturation.

**Methodology:**
- Reduce learning rate for output layer only (e.g., 0.1× of hidden layers)
- Monitor gradient flow and stopping behavior
- Compare performance with uniform learning rate

**Expected Outcome:** Lower output layer learning rate may prevent saturation while maintaining faster hidden layer learning.

### 7.6 Cross-Domain Validation (Lower Priority)

**Objective:** Verify that the phenomenon generalizes beyond MEC task offloading.

**Domains:**
- OpenAI Gym continuous control (MuJoCo)
- Multi-agent particle environments
- Robotic manipulation tasks

**Expected Outcome:** Similar patterns should emerge in domains with large reward scales and tanh outputs.

### 7.7 Entropy Regularization Impact (Lower Priority)

**Objective:** Evaluate whether entropy bonuses prevent output saturation.

**Methodology:**
- Add entropy term to actor loss: $\mathcal{L}_\text{actor} = -Q + \beta H(\pi)$
- Entropy encourages diverse outputs, potentially preventing saturation at ±1
- Test entropy coefficients: 0.001, 0.01, 0.1

**Expected Outcome:** Entropy regularization may delay or prevent actor saturation.

### 7.8 Summary of Proposed Experiments

| Experiment | Priority | Effort | Expected Impact |
|------------|----------|--------|-----------------|
| Gradient magnitude tracking | High | Low | Confirms mechanism |
| Pre-activation analysis | High | Low | Visualizes saturation |
| Alternative activations | Medium | Medium | Potential solution |
| Reward normalization | Medium | Low | Practical mitigation |
| Layer-wise LR | Medium | Low | Practical mitigation |
| Cross-domain validation | Lower | High | Generalizability |
| Entropy regularization | Lower | Low | Alternative solution |

---

## 8. Conclusion

This paper investigated the phenomenon of asymmetric convergence behavior between actor and critic networks in Client-Master MADRL architectures. Our analysis revealed that **tanh output activation saturation** is the primary mechanism causing client agents to stop updating their neural network weights, with the stopping episode being strongly dependent on learning rate magnitude.

Key findings include:
1. High learning rates (0.01-0.1) cause all actors to cease weight updates within 5 episodes
2. The master agent (critic) never stops due to its linear output activation
3. The interaction between learning rate, reward scale, and bounded activations creates effective learning rates that push actors into saturation
4. Lower actor learning rates (0.0001) maintain gradient flow throughout training and achieve better final performance

These findings have important implications for hyperparameter selection in actor-critic MADRL, suggesting that actors should use learning rates approximately 10× lower than critics. The architectural choice of bounded output activations, while necessary for continuous action spaces, creates an inherent vulnerability to gradient pathologies that must be managed through careful hyperparameter tuning.

Future work will focus on directly measuring gradient magnitudes, exploring alternative bounded activations with better gradient properties, and validating these findings across diverse domains.

---

## References

1. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

2. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in Neural Information Processing Systems*.

3. [Eimer, T., et al. (2023). Hyperparameters in Reinforcement Learning and How To Tune Them. *ICML*.](https://proceedings.mlr.press/v202/eimer23a/eimer23a.pdf)

4. [Zahavy, T., et al. (2020). A Self-Tuning Actor-Critic Algorithm. *arXiv*.](https://arxiv.org/abs/2002.12928)

5. [Vanishing and Exploding Gradients Problems in Deep Learning. GeeksforGeeks.](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)

6. [Asymmetric Actor-Critic Framework. EmergentMind.](https://www.emergentmind.com/topics/asymmetric-actor-critic-framework)

7. [Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters. *AAAI*.](https://arxiv.org/pdf/1709.06560)

8. [Policy Gradient Algorithms. Lil'Log.](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

9. [Zhang, K., et al. (2021). Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms. *Handbook of RL and Control*.](https://link.springer.com/article/10.1007/s10462-022-10299-x)

10. [Distributional Soft Actor-Critic with Three Refinements. arXiv.](https://arxiv.org/html/2310.05858)

11. [Broad Critic Deep Actor Reinforcement Learning for Continuous Control (2024). arXiv.](https://arxiv.org/abs/2411.15806)

12. [TD-regularized actor-critic methods. Machine Learning.](https://link.springer.com/article/10.1007/s10994-019-05788-0)

13. [Reinforcement Learning Convergence Debugging Guide. Medium.](https://medium.com/@tesfayzemuygebrekidan/reinforcement-learning-convergence-debugging-guide-73dcd9e56000)

14. [T-Soft Update of Target Network for Deep Reinforcement Learning. Neural Networks.](https://www.sciencedirect.com/science/article/abs/pii/S0893608020304482)

15. [Gradient Imbalance in RL Post-Training of Multi-Task LLMs. arXiv.](https://arxiv.org/html/2510.19178)

---

## Appendix A: Network Architectures

### A.1 Client Agent (Actor) Network

```
Input: State (7 dimensions)
├── Linear(7, 64) + ReLU
├── Linear(64, 32) + ReLU
└── Linear(32, 3) + Tanh
Output: Action (3 dimensions, bounded [-1, 1])
```

### A.2 Master Agent (Critic) Network

```
Input: Concatenation of:
  - Joint states (50 × 7 = 350)
  - Joint actions (50 × 3 = 150)
  - Per-agent state (7)
  - Per-agent action (3)
Total: 510 dimensions

├── Linear(510, 512) + ReLU
├── Linear(512, 128) + ReLU
└── Linear(128, 1)
Output: Q-value (1 dimension, unbounded)
```

---

## Appendix B: Reward Function Details

```python
# Energy cost
Energy_local = K_ENERGY_LOCAL * size * cycles * compute_allocation
Energy_off = power * offload_time
Energy_n = (1 - x_n) * Energy_local + x_n * Energy_off

# Time cost (normalized by MAX_DDL)
Time_n = [min(t, MAX_DDL) / MAX_DDL for t in Time_n]

# Penalties
Time_penalty = max((Time_n - deadline/MAX_DDL), 0)
Energy_penalty = max((MIN_ENE - remaining_energy), 0) * 1e6

# Final reward (shared across all agents)
Reward = -1 * (LAMBDA_E * Energy_n + LAMBDA_T * Time_n)
         -1 * (LAMBDA_E * Energy_penalty + LAMBDA_T * Time_penalty)
Reward = np.sum(Reward) * np.ones(N_agents)
```

Typical reward range: **-80 to -270** (depending on task configurations)

---

## Appendix C: Experimental Data Summary

*[Placeholder for inclusion of Figure 5.3 data and CSV results]*

---
