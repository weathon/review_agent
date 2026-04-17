# Distributed Algorithm for Multi-objective Multi-agent Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Multi-objective reinforcement learning (MORL) aims to optimize multiple conflicting objectives for a single agent, where finding Pareto-optimal solutions is NP-hard and existing algorithms are often centralized with high computational complexity, limiting their practical applicability.
Multi-objective multi-agent reinforcement learning (MOMARL) extends MORL to multiple agents, which not only increases computational complexity exponentially due to the global state-action space, but also introduces communication challenges, as agents cannot continuously communicate with a central coordinator in large-scale scenarios.
This necessitates distributed algorithm, where each agent relies only on the information of its neighbors within a limited range rather than depending on the global scale.
To address these challenges, we propose a distributed MOMARL algorithm in which each agent leverages only the state of its $\kappa$-hop neighbors and locally adjusts the weights of multiple objectives through a consensus protocol.
We introduce an approximated policy gradient that reduces the dependency on global actions and a linear function approximation that limits the state space to local neighborhoods.
Each agent $i$'s computational complexity is thus reduced from $\mathcal{O}(|\mathbf{\mathcal{S}}||\mathbf{\mathcal{A}}|)$ with global state-action space in centralized algorithms to $\mathcal{O}(|\mathcal{S}\_{\mathcal{N}^{\kappa}\_{i}}||\mathcal{A}\_{i}|)$ with $\kappa$-neighborhood state and local action space. 
We prove that the algorithm converges to a Pareto-stationary solution at a rate of $\mathcal{O}(1/T)$ and demonstrate in simulations for robot path planning that our approach achieves higher multi-objective values than state-of-the-art method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the challenges of high computational complexity and limited communication in large-scale distributed multi-objective multi-agent reinforcement learning (MOMARL) by proposing a distributed algorithm. The approach introduces an approximate policy gradient and linear function approximation based on local neighborhood information to effectively reduce the state–action dimensionality, and employs a consensus-based protocol to adaptively adjust multi-objective weights, enabling convergence to an approximate Pareto-equilibrium solution using only local communication. Theoretical analysis proves that the algorithm achieves an O(1/T) convergence rate, and experiments on a multi-robot path planning task validate its effectiveness.

### Strengths
1. The problem addressed in this paper is clearly defined, and the motivation is well articulated.
2. A distributed and scalable algorithm is proposed under the MOMARL framework, demonstrating originality.
3. The algorithm design is reasonable, and the overall description is clear.

### Weaknesses
1. The literature review is insufficient: the introduction provides an inadequate discussion of existing MOMARL and distributed RL algorithms, lacking direct comparison with representative works; moreover, the references are relatively outdated and the related work section is missing.

2. The experimental setup is not clearly described, with insufficient details regarding the design of the neighborhood network and other implementation aspects; it is recommended to include comparisons with baseline methods from recent MOMARL studies.

### Questions
1. The paper does not review relevant studies on fully decentralized multi-agent reinforcement learning (MARL). Excluding the multi-objective component, it remains unclear how the proposed method fundamentally differs from existing fully decentralized MARL approaches. The paper lacks a Related Work section to situate its contribution in the broader literature.

2. From the current description, it appears that each agent’s local state space includes information from all nodes, which contradicts the notion of locality. The experimental section should provide more detailed explanations of the experimental design, particularly clarifying how the k-hop neighborhood is defined and implemented.

3. Although the paper claims to develop a fully distributed algorithm, the definition of the Q-function indicates dependence on neighboring agents’ rewards. The authors should clarify what “fully distributed” means in the framework—whether it allows partial information exchange among neighbors or strictly prohibits it.

4. The description of the feature vector mapping is insufficient. It is unclear whether this mapping is a user-defined linear function and whether it serves to restrict which neighbors are included for agent $i$. A more explicit formulation or example would improve clarity.

5. In the second term of Equation (12), the normalization factor should likely be the size of the neighborhood $|N_i|$ rather than the total number of agents $N$. This correction is important for maintaining consistency and correct scaling in the local TD update.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a fully distributed algorithm for multi-objective multi-agent reinforcement learning, where each agent updates its policy using a localized, action-averaged critic over its κ-hop neighborhood and negotiates objective trade-offs via a consensus–Frank–Wolfe update of weights. The authors derive bounds showing that the policy-gradient approximation error decays with neighborhood radius and prove convergence to an $\varepsilon$-Pareto-stationary point under standard assumptions. Experiments on multi-robot path planning report higher per-objective returns and faster optimization (smaller $\|g_t\|_2$) than a centralized MORL baseline.

### Strengths
The paper offers a principled localization of policy gradients to κ-hop neighborhoods with a clear approximation bound that decays as $(\gamma)^{\kappa+1}$ and a proof of convergence to an $\varepsilon$-Pareto-stationary point. Its communication-efficient consensus–Frank–Wolfe weighting—using only neighbor information—translates to practical multi-robot gains, showing higher per-objective returns and faster reduction of $\|g_t\|_2$ than a centralized baseline.

### Weaknesses
- Consensus implementation vs. theory mismatch: Algorithm 2 uses “while-until-exact-consensus,” implying unbounded communication, whereas the analysis assumes a fixed $K_\lambda$; the bounds omit an explicit consensus residual, so finite-round practice either violates assumptions or undermines scalability.
- Core approximation issues: the gradient-error bound lacks explicit dependence on graph topology (degree, spectral gap, neighborhood size), and the critic/TD scaling sums local rewards but divides by global $N$; when $|\mathcal N_i^\kappa|\!\ll\! N$, this compresses magnitudes, distorts variance, and complicates step-size selection (a normalization by $|\mathcal N_i^\kappa|$ would be more coherent).
- Weak empirical support for key claims: only a single centralized baseline is considered, with no Pareto frontier, significance tests, ablations, or communication/runtime reporting; claims of being “near central optimum” and “faster” lack a verifiable upper bound or comparisons to strong MARL baselines.

### Questions
- Could you broaden the empirical study to plot full Pareto frontiers (by sweeping initial weights), compare against strong MARL baselines (e.g., VDN, QMIX, MAPPO, MADDPG), report mean±std over multiple seeds with significance tests, and quantify communication per update and wall-clock time relative to a centralized approach?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a fully distributed actor-critic algorithm for multi-objective multi-agent reinforcement learning. Each agent uses only $\kappa$-hop neighborhood states and its local action, together with an actionaveraged $Q$-function and linear function approximation, to estimate policy gradients. A consensus-plus-Frank-Wolfe procedure adjusts objective weights. The authors prove $O(1 / T)$ convergence to an $\varepsilon$-Pareto-stationary solution under standard assumptions on rewards and network connectivity conditions.

### Strengths
Strengths:


1. The paper replaces the global policy gradient with a local, action-averaged gradient $\nabla_{\theta_i} J_{\text {app }, i}^m(\theta)$ that depends only on the agent's own action and $\kappa$-hop neighborhood. It then proves a geometrically decaying approximation error
$\left\|\nabla_{\theta_i} J_{\mathrm{app}, i}^m(\theta)-\nabla_{\theta_i} J^m(\theta)\right\|_2 \leq \frac{\sqrt{2 R}}{\left(1-\gamma^m\right)^2}\left(\gamma^m\right)^{\kappa+1}$
which makes the approximation transparent and parameterized by $\kappa$.


2. The critic uses a linear approximation of the form
$\hat{Q}^m_i\left(s_{N_i^\kappa}, a_i ; w_i^m\right)=\phi_i\left(s_{N_i^\kappa}, a_i\right)^{\top} w_i^m$
so that each agent only estimates values over its neighborhood state and its own action. This aligns the value approximation with the locality assumption used in the policy-gradient derivation.

3. Distributed weight selection that preserves the convergence rate.
The scalarization/weight-selection step is solved by a consensus plus Frank-Wolfe update over the network. The analysis shows that, despite decentralization, the overall algorithm still achieves an $O(1 / T)$ convergence rate to an $\varepsilon$-Pareto-stationary point, which is nontrivial for multi-objective, multi-agent settings.

### Weaknesses
Weaknesses:

1. The final actor update bundles truncation, sampling, and function-approximation errors into a single term; the paper shows it stays bounded, but does not give a tight characterization of how this bound scales with all inner-loop parameters, so the sharpness of the $O(1 / T)$ claim is partially opaque.

2. The key bound decays as $\left(\gamma^m\right)^{\kappa+1} /\left(1-\gamma^m\right)^2$. When $\gamma^m$ is close to 1 and $\kappa$ must stay small for communication reasons, this term can be large, so the locality that makes the method scalable can simultaneously weaken the approximation guarantee.

3. The distributed weight-update step requires (at least approximate) consensus at every outer iteration. The theory assumes this is done sufficiently well, but the per-iteration communication/synchronization burden is not incorporated into the main convergence complexity statement.

### Questions
1. Normalization in critic updates: The TD-style critic uses localized information. Is the global normalization factor (or averaging step) essential for the contraction argument, or could a purely local normalization reduce variance without breaking the proof?

2. The Frank-Wolfe-based weight update is analyzed as if the subproblem is solved well each round. How does the main convergence bound change if only a fixed, small number of FW steps (and consensus rounds) is used per iteration?

3. The policy-gradient derivation is presented for the softmax/discrete case. Can the same local-action, action-averaged construction be extended to deterministic or Gaussian policies in continuous action spaces while still retaining the $O(1 / T)$ rate?

### Soundness
3

### Presentation
3

### Contribution
3
