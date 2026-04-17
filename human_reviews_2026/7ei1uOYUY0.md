# From Scarcity to Efficiency: Preference-Guided Learning for Sparse-Reward Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
We study the problem of online multi-agent reinforcement learning (MARL) in environments with sparse rewards, where reward feedback is not provided at each interaction but only revealed at the end of a trajectory. This setting, though realistic, presents a fundamental challenge: the lack of intermediate rewards hinders standard MARL algorithms from effectively guiding policy learning. To address this issue, we propose a novel framework that integrates online inverse preference learning with multi-agent on-policy optimization into a unified architecture. At its core, our approach introduces an implicit multi-agent reward learning model, built upon a preference-based value-decomposition network, which produces both global and local reward signals. These signals are further used to construct dual advantage streams, enabling differentiated learning targets for the centralized critic and decentralized actors. In addition, we demonstrate how large language models (LLMs) can be leveraged to provide preference labels that enhance the quality of the learned reward model. Empirical evaluations on state-of-the-art benchmarks, including MAMuJoCo and SMACv2, show that our method achieves superior performance compared to existing baselines, highlighting its effectiveness in addressing sparse-reward challenges in online MARL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenging problem of cooperative multi-agent reinforcement learning (MARL) in sparse-reward environments. The authors propose a novel framework, IMAP, that reframes the problem as one of online inverse preference learning (IPL). The core idea is to use the sparse, episodic reward (e.g., win/loss) not as a direct training signal, but as a source for generating trajectory-level preference labels (e.g., $\sigma_1 > \sigma_2$).

The framework then trains an implicit, decomposed reward model (based on a value-decomposition architecture) to learn dense, per-timestep rewards that are consistent with these preferences. This model then produces both a global reward signal ($R_t$) and agent-specific local reward signals ($r_{i,t}$). These dense rewards are then fed into a "dual-stream" policy gradient algorithm: the centralized critic is trained using a global advantage (from $R_t$), while the decentralized actors are trained using local advantages (from $r_{i,t}$) to improve credit assignment. The paper also explores using LLMs to generate more-nuanced preference labels. The authors provide theoretical analysis to argue that their learned reward function, while not the ground truth, converges to a behaviorally-equivalent function, thus preserving the optimal policy.

### Strengths
- **Significant and Relevant Problem**: The paper tackles a central and difficult challenge in MARL. Finding a way to perform effective credit assignment from sparse, episodic rewards is a highly significant research direction.

- **Novelty of the Core Idea**: The idea of integrating online inverse preference learning (IPL) directly with a value-decomposition network (VDN/QMIX-style) is novel. This is a creative combination of ideas from two different sub-fields (PbRL and MARL).

- **Promising Method for Credit Assignment**: The "dual-advantage" architecture derived from this decomposed reward model is an interesting and plausible approach to solving the multi-agent credit assignment problem. Using the learned local rewards for actors and the global reward for the critic is an elegant concept.

- **Exploration of LLM-based Feedback**: The (brief) exploration of using LLMs to provide more nuanced, qualitative preference feedback beyond simple win/loss signals is a timely and promising research avenue.

### Weaknesses
This paper, while built on promising ideas, suffers from major weaknesses in clarity, technical justification, and empirical evaluation. The work is not ready for publication and requires a major revision.

**Extremely Unclear Presentation**: 

The paper is exceptionally difficult to follow. The writing is convoluted and fails to clearly explain the model's core mechanics.

- The "Dual-Critic" Architecture: The framework appears to use two separate sets of value functions: 1) the reward model's V-nets ($v_\xi$ and $M_w$), which we can call $V_{reward}$, and 2) the policy's separate centralized critic ($V_{\phi}^{tot}$), which we can call $V_{policy}$. Both are trained to estimate the exact same quantity: the value of a state under the current policy. This critical, redundant, and non-standard architectural choice is never explained or justified. The "Reward Learner" (Sec 4) already produces a complete, centralized value function ($V_{reward}$). A logical framework would reuse this as the policy critic. By introducing a second critic ($V_{policy}$) trained on the output of the first, the framework appears inefficient and suggests a poorly integrated design where a standard MAPPO implementation (which comes with $V_{policy}$) was simply "bolted onto" the reward-learning module (which produces $V_{reward}$).

- Convoluted Training Loop: The connection between the "Reward Learner" (Sec 4) and the "Policy Learner" (Sec 5) is poorly articulated. The alternating training process (training $q$ on preference loss, then $v$ on consistency loss, then $\pi$ on PPO loss) is a complex procedure that is not clearly detailed.

- Unclear Motivations: Key theoretical concepts are introduced without any motivation. For example, the Gumbel noise in Theorem 4.2 is stated as an assumption with no explanation for why it is the correct choice (i.e., that it is the theoretical justification for the Bradley-Terry model).

**Significant Technical and Theoretical Inconsistencies**:

- $\tau$ vs. $\beta$: The paper is sloppy with its notation. It uses $\beta$ as the temperature for the soft Bellman operator (Eq. 201) but introduces $\tau$ as the temperature for the Bradley-Terry model (Thm 4.2), setting $\tau=1$ by assumption. These parameters are conceptually identical. Both $\beta$ and $\tau$ function as the temperature parameter controlling the "softness" (or randomness) of an operation rooted in maximum entropy principles. $\beta$ controls the softness of the max in the LSE function ($V(s) = \beta \log \sum \exp(Q/\beta)$), while $\tau$ controls the softness of the choice in the BT model (which is a 2-item softmax: $P(A>B) = \exp(R_A/\tau) / (\exp(R_A/\tau) + \exp(R_B/\tau))$). A coherent theoretical framework would use a single, consistent temperature parameter. This discrepancy suggests a disjointed approach from different sources and a shallow theoretical understanding of the method.

- Theorem 4.2 Discrepancy: Theorem 4.2 is stated differently in the main text (line 252) and the supplementary material (line 785). This is a major red flag regarding the paper's carefulness and correctness.

- Confusing Proofs: As the authors' own pointers note, the proof of Theorem 4.1 (Appendix) contains a "confusing use of sigma" (line 815), making it difficult to verify.

- Proposition 4.1 is Not Novel: Proposition 4.1 (that an additive constant to trajectory-level reward does not change the optimal policy) is a long-established and well-known fact in reward-shaping literature (e.g., potential-based reward shaping). It is not a novel contribution of this paper.

**Unjustified Technical Assumptions**:

- The Shared Mixer Assumption: The paper's strongest assumption is that the global Q-function $Q_{tot}(s,a)$ and the global V-function $V_{tot}(s)$ share the **exact same mixing network** $\mathcal{M}_w$. This is a severe limiting assumption with no theoretical or empirical justification. Why should the mixing function for state-action values be identical to that of state values?

- LLM as a "Reasoner": The use of an LLM to "deduce preference rewards" (as noted in the pointers) is highly suspect. This assumes the LLM can reason about complex, non-trivial coordination dynamics in SMACv2 (or physics in MAMuJoCo, which was omitted). This is a very strong claim that is not substantiated, and the potential for the LLM to have pre-existing (training) knowledge of the environments is not addressed.

**Critically Insufficient Empirical Evaluation**: 

This is the paper's most significant flaw. The authors claim to propose a new SOTA framework for sparse-reward MARL but fail to compare against any of the relevant, state-of-the-art baselines in this specific area.

- Missing Baselines: The experiments are missing comparisons to numerous critical works, including STAS [1], AREL [2], and TAR2 [3]. These methods are all directly designed for reward redistribution in sparse-reward MARL.

- Weak Baselines: The current baselines (SparseMAPPO, SL-MAPPO, Online-IPL) are not representative of the current state of the art.

- Without these comparisons, the empirical results in Table 1 are unsubstantiated and essentially meaningless. The paper does not demonstrate that its approach is any better than existing, simpler reward redistribution techniques.

[1] Chen, Sirui, et al. "STAS: spatial-temporal return decomposition for solving sparse rewards problems in multi-agent reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 16. 2024.

[2] Xiao, Baicen, Bhaskar Ramasubramanian, and Radha Poovendran. "Agent-temporal attention for reward redistribution in episodic multi-agent reinforcement learning." arXiv preprint arXiv:2201.04612 (2022).

[3] Kapoor, Aditya, et al. "$ TAR^ 2$: Temporal-Agent Reward Redistribution for Optimal Policy Preservation in Multi-Agent Reinforcement Learning." arXiv preprint arXiv:2502.04864 (2025).

### Questions
To facilitate a productive discussion, I ask the authors to please address the following points:

- Justify the Dual-Critic System: Please justify the redundant "dual-critic" architecture. Your reward-learning module (Sec 4) already produces a centralized value function, $V_{reward}$ = $M_w$ [$v_\xi$]. Why is this not used as the policy critic? Why is it necessary to introduce a second, separate centralized critic, $V_{\phi}^{tot}$, to be trained on the outputs of the first? This seems highly inefficient and implies a lack of integration between the two components of your framework.

- Justification for the Shared Mixer: Can you provide a clear theoretical or empirical justification for the assumption that $Q_{tot}(s,a)$ and $V_{tot}(s)$ can be factorized by the same mixing network? What is the impact of decoupling them?

- Missing Baselines: Can you justify the omission of critical and SOTA sparse-reward MARL baselines like STAS, AREL, and TAR2? A response to this is essential to evaluate the paper's empirical contribution.

- Temperature Inconsistency: Can you clarify the relationship between the temperature $\beta$ (soft Bellman) and the temperature $\tau$ (BT model)? Do you agree that they are conceptually identical, both controlling the "softness" of their respective MaxEnt operations? If so, why are they treated as separate, unconnected parameters?

- Theorem 4.2 Discrepancy: Can you please address and correct the discrepancy in the statement of Theorem 4.2 between the main text and the supplementary material?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles sparse-reward cooperative MARL by unifying inverse preference learning (IPL) with a CTDE, PPO-style learner. It 1. converts episodic returns into trajectory preferences, 2. learns an implicit reward via a preference-based value-decomposition network and 3. trains with dual advantage streams—a global advantage for the centralized critic and agent-specific local advantages for decentralized actors. It also explores using LLMs to label preferences from trajectory summaries. Experiments on SMACv2 and MAMuJoCo report stronger win-rates/returns and better sample efficiency than baselines

### Strengths
The idea is quite interesting and simple. The decomposition gives local advantages aligned with the global objective, addressing a common weakness of MAPPO with a single global signal.

The theoretical analysis shows that the learned implicit reward converges to a behaviorally indistinguishable surrogate of the ground-truth

The formulation is quite clear and the writing is easy to follow.

### Weaknesses
The theoretical part seems to strong: Theorem 4.2 assumes BT-generated preferences with τ=1, coverage over all trajectory pairs, and N ⁣→ ⁣∞. In practice we only have finite samples. How do you ensure adequate pairwise coverage under realistic budgets, and what is the sampling cost (queries, wall-clock, tokens) required to achieve the stated guarantees? Also, as we may not be dealing with fully converged cases, how sensitive is performance to semi-coverage? 

While effective, LLM labels add potential bias and online cost; Can you quantify the LLM-preference budget (tokens/time) and report win-rate per dollar or per wall-clock hour to substantiate efficiency claims? Also, you seem to be missing the IMAP-LLM for MA-MOJOCO. Any reason for not including the result for that one?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper primarily addresses the sparse reward and credit allocation problem in MARL with episodic terminal rewards. The core idea is to construct trajectory preferences based on the relative ranking of trajectory rewards, and then recover Q-values ​​from the trajectories using the inverse soft Bellman operator of IPL for value decomposition. Building upon MAPPO, dual advantage streams are introduced to achieve finer credit allocation, with corresponding policy enhancement theoretical guarantees provided. Furthermore, for scenarios with clearly structured preference information, LLM is used to construct denser preference signals, further improving the scalability of this method in complex tasks without explicit rewards. Experiments on multiple MARL benchmark tasks effectively demonstrate the performance of the proposed method.

### Strengths
This paper's motivation for applying preference learning to online sparse reward MARL problems is sound. Building upon IPL, it combines value decomposition methods to effectively achieve credit allocation among agents.

Using the inverse soft Bellman operator to recover rewards in Q-space effectively avoids the neglect of environmental state transitions in simple BT models, making it more suitable for value-based RL methods. The consistency of global-local advantages provides a theoretical guarantee for this approach.

This paper considers discrete and continuous control task environments, conducting reasonable baseline comparisons and ablation experiments. It also provides a comparison between rule-based preferences and LLM preferences, demonstrating the potential of LLM preferences in real-world scenarios where dense rewards are difficult to design.

### Weaknesses
W1: The paper defines $r_i (s_i, a_i) = q_i(s_i, a_i) − \gamma E_{p(s’_i|s_i,a_i)}[v_i(s’_i)]$  for each agent. However, in partially observable, coupled dynamical cooperative environments, local state transitions $p(s’_i | s_i, a_i)$ generally depend on the actions of other agents and the global state, making the "local inverse Bellman" assumption somewhat too strong. Especially for the POMDP problem using local observations as input, does this property still hold? It is suggested that the authors further explain what approximations were used in the implementation to handle the local observation problem.

W2: In terms of experiments, the comparison with policy gradient methods with centralized critics like MAPPO is insufficient to fully demonstrate the advantages in credit assignment with sparse rewards, such as FACMAC and HAPPO.

W3: MAMuJoCo tasks are generally dense rewards by default. The article claims a "sparse reward setting" but does not clearly explain how to modify it to only trajectory-level feedback. More specific environmental settings are needed.

### Questions
Q1: A fundamental question with PBRL is how to handle the learning process when early stochastic policies offer similar trajectory rewards or lack exploration, resulting in very little preference information. Are cold-start techniques employed?

Q2: How to extract the corresponding $wi*$ from the value decomposition network of a multi-layer MLP for calculating $\delta_{i,t}^{local}$?

Q3: What are the specific decoding parameter settings for LLM? For example, temperature and top p, and is thinking mode enabled on qwen3-4b?

Q4: This paper only considers constructing the LLM preference reward using structured text to describe the final state, which is merely the final state preference. A natural idea is to input the complete trajectory policy into the LLM to evaluate the trajectory preference. Can the method presented in this paper be extended to this setting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of cooperative multi-agent reinforcement learning in environments with sparse rewards, where feedback is only provided at trajectory termination. The authors propose IMAP, which converts sparse episodic rewards into trajectory preferences and learns implicit dense rewards through inverse preference learning in the Q-function space. Key contributions include transforming sparse rewards into preferences, a CTDE paradigm with dual advantages, theoretical guarantees on reward convergence and policy alignment, and evaluations on MAMuJoCo and SMACv2 benchmarks showing superior performance and efficiency.

### Strengths
- The paper tackles the highly challenging and relevant problem of sparse rewards in online cooperative MARL. This work successfully connects recent advances in offline preference-based RL with the online, multi-agent CTDE paradigm, resulting in a unified and novel learning framework.
- The introduction of dual advantage streams is a key contribution for credit assignment in the sparse-reward setting. By using a global advantage for the centralized critic and distinct local advantages for the decentralized actors, the method effectively separates global coordination from local credit assignment.
- Proposition 4.1 and Theorem 4.2 establish the asymptotic convergence of the learned implicit reward. The key result is that the method converges to a reward function that is "behaviorally indistinguishable" from the ground-truth, ensuring the optimal policy aligns with the true task objective.
- The empirical results are very strong. IMAP-Rule and IMAP-LLM significantly outperform all baselines.

### Weaknesses
- The entire value decomposition framework relies on a simple linear mixing network $\mathcal{M}_w$. This is a strong assumption and is significantly less expressive than the non-linear, monotonic mixing networks used in ubiquitous MARL algorithms like QMIX. The paper provides no justification for this choice or analysis of whether this restrictive structure limits performance.
- The LLM-based component relies on "interpretable features". This is why it is applied to SMACv2 but not MAMuJoCo. This restriction significantly limits the generality of the paper's most novel-looking variant, as many complex problems (like continuous control) do not have easily interpretable, human-readable features.
- The success of IMAP-LLM shifts the burden of "reward engineering" to "prompt engineering". The performance is likely sensitive to the quality of the prompt, such as the "Important Notice" and "Objective". The paper lacks a sensitivity analysis on the prompt's content, making it unclear how much expert knowledge is implicitly required to craft an effective prompt.
- The definition of the local implicit reward $r_i$ in Sec. 5.2 is confusing. The local transition $P(\cdot|s_{i},a_{i})$ is generally not well-defined in a multi-agent setting, as the next state $s'$ depends on the joint action, not just $a_i$. This notation seems to conflict with the CTDE assumptions.
- Theorem 4.2 assumes every possible trajectory pair appears in the preference dataset at least N times, which is impractical for high-dimensional state-action spaces or long horizons.
- No comparison is provided against recent sparse-reward baselines specifically designed for MARL, such as attention-based return decomposition methods mentioned in Section 2 but absent from experiments.
- Notation for global state s sometimes interchanges with joint observation o without consistent clarification.

### Questions
- Theorem 4.2 assumes every possible trajectory pair appears at least N times in the preference dataset. How does your online algorithm (Algorithm 1) ensure sufficient coverage in practice?
- How do you manage the preference buffer P in Algorithm 1? Do preferences expire, and how is sampling performed?
- What is the actual computational overhead of using LLMs for online preference generation, and when does the performance gain justify this cost?
- How sensitive is IMAP to key hyperparameters, particularly the temperature $\beta$, regularization $\varphi$, and the number of IPL gradient steps per iteration?
- How sensitive is IMAP-LLM to prompt design? Have you tested alternative prompts or prompt templates?
- The method generates preferences by sampling and comparing pairs of trajectories. Could you comment on the sample efficiency of this "tournament-style" sampling? Does this pairwise comparison loop introduce a significant sample complexity overhead compared to methods that can learn from individual trajectory returns?

### Soundness
2

### Presentation
3

### Contribution
2
