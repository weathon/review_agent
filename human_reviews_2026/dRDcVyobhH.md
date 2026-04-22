# Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Group symmetries provide a powerful inductive bias for reinforcement learning (RL), enabling efficient generalization across symmetric states and actions via group-invariant Markov Decision Processes (MDPs). However, real-world environments almost never realize fully group-invariant MDPs; dynamics, actuation limits, and reward design usually break symmetries, often only locally. Under group-invariant Bellman backups for such cases, local symmetry-breaking introduces errors that propagate across the entire state--action space, resulting in global value estimation errors.
To address this, we introduce Partially Group-Invariant MDP (PI-MDP), which selectively applies group-invariant or standard Bellman backups depending on where symmetry holds. This framework mitigates error propagation from locally broken symmetries while maintaining the benefits of equivariance, thereby enhancing sample efficiency and generalizability.
Building on this framework, we present practical RL algorithms -- Partially Equivariant (PE)-DQN for discrete control and PE-SAC for continuous control -- that combine the benefits of equivariance with robustness to symmetry-breaking.
Experiments across Grid-World, locomotion, and manipulation benchmarks demonstrate that PE-DQN and PE-SAC significantly outperform baselines, highlighting the importance of selective symmetry exploitation for robust and sample-efficient RL. Project page: https://pranaboy72.github.io/perl_page/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced a novel partially equivariant reinforcement learning formulation and a novel algorithm to solve RL in imperfect equivariant MDP scenarios. The proposed partially equivariant reinforcement learning formulation, i.e., Partially group-Invariant MDP (PI-MDP), tackles symmetry-broken MDP by decomposition the MDP into an euqivariant MDP and a standard MDP. Later, authors proposed to identify symmetry breaking scenarios (state-action pairs) by dynamic-disagreements. Lastly, two algorithms, PE-DQN and PE-SAC are introduced to solve discrete and continuous PI-MDPs respectively. Experiments show the proposed method outperforms various baselines.

### Strengths
This paper Addressed partially invariant MDP by decomposing it into group-invariant MDP and a standard MDP. Then the two MDPs are merged by a measurable gating function. Beside theory, the paper introduced a forward disagreement measurement to practically estimate the gating. Experiments demonstrated the advantage of the proposed method.

### Weaknesses
Firstly, the paper do not explain well what is a symmetry breaking MDP. Authors gave an equation for group invariant MDP, could authors give equation for symmetry breaking MDP? Furthermore, Figure 1 showed an example of symmetry breaking MDP, I assume the obstacle is observable. Nevertheless, this example to me is more like an extrinsic equivariance [1], where the right subfigure is a new data, rather than a group-transformed seen observation (since the obstacle is not transformed). An equation of symmetry breaking MDP would clarify it.
The forward disagreement measurement seems relied on how good the unconstrained regressor can learn the forward dynamics. If the underlying MDP is group invariant, and the unconstrained regressor learns a bad dynamic, then this regressor will learn to large disagreement with the constrained regressor. In this case, the disagreement measurement fails to measure the equivariance of the MDP. Could authors explain more on this?

[1] A general theory of correct, incorrect, and extrinsic equivariance. D Wang, X Zhu, JY Park, M Jia, G Su, R Platt, R Walters - Advances in Neural Information Processing Systems, 2023

### Questions
Beside questions in weakness, I have following questions:
a. what is the alpha in equation9?
b. Could authors explain the detailed setup of the grid-world task? Especially how it break symmetry.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of symmetry-breaking in equivariant reinforcement learning, where standard equivariant methods fail when environment symmetries are not held. The authors propose a novel framework called the Partially group-Invariant MDP (PI-MDP) and practical algorithms (PE-DQN, PE-SAC) that use a learned gating function to dynamically switch between an equivariant and a standard network, applying equivariance only where symmetries are satisfied. Experiment results demonstrate that the method successfully combines the sample efficiency of equivariance with robustness to symmetry-breaking, outperforming the baselines.

### Strengths
1. The paper is generally well-written. The problem setup is clear, and the main idea is easy to follow.
2.  The core idea of using a local and learned gating mechanism to handle symmetry-breaking is well-motivated.
2. The paper provides a theoretical analysis of the error propagation from local symmetry-breaking.

### Weaknesses
1. The method introduces significant additional complexity: training two dynamics models, with corresponding two policy/value networks and two gating functions. The paper does not conduct the ablation analysis on these components, which would strengthen the impact.

2. The current approach relies on dynamic disagreement to detect symmetry-breaking. This might be less effective for environments where symmetry is broken primarily in the reward function rather than the dynamics. The paper notes this limitation but does not explore alternatives or the performance impact in such scenarios.

3. In lines 262~263, the author states that " We assume those symmetry-breaking disagreements as outliers in the online distribution of $d(s, a)$. We label outliers with $y(s, a) ∈ {0, 1}$ using an online detector." Such a core assumption that its outliers reliably indicate symmetry violations is potentially fragile. For example, in stochastic environments, high variance in transitions could inflate $d(s, a)$ values even in symmetric regions, while in cases where the dynamics models are also inaccurate, leading to an unreliable gating function.

### Questions
1. Was an ablation or sensitivity analysis performed on the parameters used in the proposed method?
2. The paper uses a hard gating strategy for simplicity. Did the authors experiment with a soft, probabilistic gating mechanism?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes incorporating equivariance with reinforcement learning only for certain state-action pairs. The authors propose a partially group-invariant MDP that uses a gating mechanism to switch between a group-invariant or a standard MDP. They bound the optimal Q value so that errors are not propagated when the gating function routes correctly. Experiments on a gridworld domain and continuous control and robotics domains show improved performance over baselines.

Overall, this paper proposes a good way to handle symmetry violations in RL with a gating mechanism, such that one does not have to resort to global notions of exact or approximate equivariance and can use local definitions.

### Strengths
- The authors consider an important problem in equivariance and RL, specifically that symmetry violations are often local in practice. The proposed gating mechanism seems like a good and straightforward approach.
- The PI-MDP formulation is well-motivated and the authors also provide theoretical analysis on the error of the optimal Q function, given the optimal gating.
- The use of disagreement to discover symmetry violations seems like a good choice.
- Experiments are carried out on various domains and the results show that the PI-MDP framework often outperforms exactly equivariant RL.

### Weaknesses
See questions.

### Questions
- The theory is assumed for strict binary gating, while the experiments seem to use a soft gating mechanism for $\lambda_\omega$. Did you find that soft gating works better empirically over strict gating?
- I didn't fully understand why a separate state-only gating function was needed. Can you not just use the Q-gating function $\lambda_\omega$ for the policy? Is this only necessary for continuous actions? I feel like making $\lambda_\zeta$ learnable can potentially lead to mismatch between $\lambda_\omega$ and $\lambda_\zeta$.
- One weakness I can see is that since the PI-MDP framework requires separate critics and policies, using separate networks can require more samples than with a single MDP. For example, if the gating is routed to $M_E$, then only the $Q_E, \pi_E$ networks are optimized and vice versa, leading to lower sample efficiency. This is perhaps why a soft gating mechanism is preferable as well.
- Another question is that if we assume that symmetry violations in certain state-action pairs, then there are large regions where $M_E$ and $M_N$ coincide and thus their respective $Q$ and $\pi$ are the same/similar. Using separate networks can be somewhat parameter inefficient in this case. Have you tried sharing portions of the networks (such as a common trunk) among the critics and policies, and maybe also for the dynamics models?

### Soundness
4

### Presentation
4

### Contribution
3
