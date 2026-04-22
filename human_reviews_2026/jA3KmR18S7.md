# Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
The Successor Measure (SM), a powerful method in reinforcement learning (RL), describes discounted future state distributions under a policy, and it has recently been studied using generative modeling techniques. Although SM is a powerful predictive object, it lacks compact representations tailored for online RL. To address this, we introduce Successor Flow Features (SF2), a representation learning framework that bridges SM estimation with policy optimization. SF2 leverages flow-matching generative models to approximate successor measures, while enforcing a structured linear decomposition into a time-invariant embedding and a time-dependent projection. This yields compact, policy-aware state-action features that integrate readily into standard off-policy algorithms like TD3 and SAC. Experiments on DeepMind Control Suite tasks show that SF2 improves sample efficiency and training stability compared to strong successor feature baselines. We attribute these gains to the compact representation induced by flow matching, which reduces compounding errors in long-horizon predictions. The code is available on https://github.com/Shiien/successor-flow-representation-implementation .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SF$^2$, a flow-matching based representation that approximates the successor measure (SM) while enforcing a linear decomposition: time-invariant $\psi(s,a)$ and time-dependent $\zeta(\cdot|k)$. SF$^2$ is trained jointly with off-policy RL via a mixture-structured flow matching loss and a value-alignment term, and shows improved average performance and stability on seven DMC tasks, with explicit analysis of EMA, denoising steps, and runtime trade-offs. The work also offers an interpretation that, in a small-$k$ regime, $\psi$ updates resemble succesor representation (SR)-style TD recursion.

---
**Review summary.** The proposed framework introduces a practically effective bridge between SM and flow matching. Despite some theoretical and scalability gaps, the paper offers solid empirical evidence, clear motivation, and transparent ablation studies. With stronger formal grounding and broader experiments, this work could become a notable contribution to representation learning in RL. However, the reviewer feels that the main contribution lies more in practical engineering and integrative design than in fundamental theoretical advancement. Therefore, the reviewer assigns an initial score of 6 and plans to revisit this rating after the authors address the concerns and questions raised in this review.

### Strengths
The reviewer thinks that the integration of flow matching and SM is novel and potentially impactful. The method makes the flow-based generative paradigm compatible with online RL, which is a meaningful step beyond prior SM works that focused solely on offline or evaluation settings. Additionally, the connection to diffusion spectral representation is insightful. The discussion of mixture distributions in Appendix A is illuminating, showing that flow matching can better capture multimodal structure than VAEs or GANs (It’s a given, perhaps, anyway), lending theoretical and empirical support to the method’s design.

---
**Writing**
- The paper is written in a clear, structured, and mathematically disciplined manner. 
- Equations are well motivated and the derivations are logically consistent. In specific, they provide a clear connection between the mixture structure of the SM and the mixture learning dynamics of Flow Matching.
- This bridige leads to a principled training objective, which is combining direct transition and bootstrap terms weighted by ($1-\gamma$ and $\gamma$). 
- The related work section (section 5) is broad and situates SF² well among SR, SM, and flow-based RL.

---
**Methodology**
- The paper presents a creative connection between SM and flow matching. By enforcing a time-invariant $\psi(s,a)$ and time-dependent $\zeta(\cdot|k)$, the method introduces a new linear factorization of dynamics.
- The semi-gradient derivation in section 3.3 gives a clear heuristic connection to SR, showing that $\psi$ updates naturally resemble a TD-style recursion with a flow-matching twist.
- The design promotes a low-dimensional sufficient statistic for state–action pairs, which provide a clear path to sufficient dimension reduction.
- Algorithm 1 and the embedding into SAC/TD3 are modular, allowing the method to be plugged into standard pipelines without major architectural redesign.

---
**Experiments**
- Across all seven DMC tasks, SF2 outperforms or matches baselines, with clear reporting of mean/std, wall-clock training time, and ablations.
- The consideration of $\gamma =0$ and $\gamma = 0.99$ variants shows how long-horizon modeling contributes to performance. 
- The reviewer thinks that hyperparameter analyses are informative. For example, the study of EMA rate, denoising steps and feature size. These ablations provides practical insight into stability-efficiency trade-offs.
- In section 4.3 and 4.4, ablations results and computational reports are transparent.

### Weaknesses
While the overall empirical presentation is strong, the paper would benefit from a more comprehensive and centralized description of computational details in the main text. Moreover, a clearer breakdown of which components dominate computational load (flow-matching updates vs value alignment vs.actor updates) would make the method’s scalability story much more convincing.

---
**Writing**
- Despite overall clarity, Section 3 contains dense mathematical exposition with several unreferenced variables and occasional overuse of inline equations. Additionally, the reviewer thinks that the interplay between $\psi, \zeta$, and the vector field $u\theta$ could benefit from schematic visualization for intuition.
- Some claims, for example, “SF$^2$ provides compact, expressive, and robust representations,” might seem like overstatements given the limited empirical evidence and absence of stress tests on diverse modalities or noise regimes.

---
**Methodology**
- While the combination of SM and flow matching is powerful, the reviewer thinks that theoretical novelty is somewhat incremental. Each mathmatical part, e.g., flow-mathcing, SM, SDR, has appeared in prior work, so the reviewer thinks that the contribution lies more in enginerring integration than in new theory.
- The derivation in Section 3.3 is heuristic. The claimed connection to SR via the pseudoinverse formulation is intuitive but lacks a formal proof of convergence or representational equivalence. The reviewr thinks that the pseudoinverse handling during training is not described concretely, leaving ambiguity about numerical stability.
- The approximation in Equation (4) that aligns vector fields instead of performing full ODE integration is efficient but somewhat ad-hoc; the implications for consistency and bias in the learned SM are not analyzed theoretically.
- Claims of universal approximation are untested, that is, no experiments validate $\psi$’s robustness under domain shift or adversarial perturbation, which would strengthen the theoretical claims.

---
**Experiments**
- The experiments is limited to state-based DMC tasks. It remains unclear whether the proposed solution scales to high-dimensional observation spaces or more difficult problems, for example, issac gym-based robotics or partially observable environments.
- Comparisons omit several modern flow/diffusion-based RL methods, such as, DSLR, TDFlow, contrastive RL. 
- No qualitative analyses, for example, UMAP, t-SNE, or value map, are presented to interpret what $\psi$ or $\zeta$ actually learn. 
- Some reported variances are large, especially in P-SwU and C-SwU-S, and the authors do not report statistical significance tests or confidence intervals.

### Questions
- Could the authors formalize the small-$k$ connection between $\psi$ updates and SR Bellman recursions? The reviewer wonder that what assumptions does this approximation hold, and what are the limits of this equivalence.
- As the reviewer above-mentioned, some DMC tasks show large performance variance. Could the authors provide the number of seeds and statistical tests (e.g., t-test, bootstrap) to confirm significance?
- The reviewer just wonder how SF2 would extend to pixel-based or image-based input environments. Would $\psi$ and $\zeta$ share encoders or require hierarchical separation?
- Have the authors compared the proposed solutions to recent flow-based RL algorithms?
    - How does the explicit mixture formulation distinguish the proposed solution theoretically or empirically from them?
- How do the authors think that shared encoders or low-rank $\zeta$ to reduce the cost? Is it a good direction or not?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In RL, the successor measure is the probability mass of ending up in a set of states conditioned on starting in some state, taking an action, and then following some policy. It generalizes the well-known idea of the successor representation to continuous state spaces. This paper introduces a method for estimating the successor measure using the generative modelling technique of flow matching. This method is then used as an auxiliary task to shape an online RL agent’s representation. Empirical evaluation on a set of continuous control tasks show that the method increases the final expected return achieved by the agent.

### Strengths
- While prior work appears to have studied how to estimate the successor measure, this work appears to be the first to do so for model-free, online RL agents.
- While I didn’t fully appreciate how the SM estimation part of the algorithm is distinct from prior work, I thought it was interesting how the work integrates a boostrap objective with an generative modelling technique.

### Weaknesses
- I’m not convinced that the experiments substantiate the claims made in the paper. Experiments are only repeated for 5 trials and the paper reports large standard deviations. Given the spread and limited trials, it’s plausible that the novel method does not improve upon the simpler base algorithm (SAC or TD3). Please see “Empirical Design in Reinforcement Learning” for a great reference on why these details matter.
- The paper contains a lot of vague, overclaiming language: “achieves superior performance” (superior in what respect?) “remarkable stability” (what is remarkable about it).
- After going through the experiments in detail and ignoring the statistical issues, I think the best claim would be “our method had the largest final expected return achieved when we stopped training.” While final performance obtained is a valid objective to study, it would be valuable to understand how the termination of training was selected. In the appendix, it is mentioned that training went for 10 million timesteps. This is substantially longer than what is usually used on thest environments (1 million seems more common). Why use the longer time? And, in any case, how can we be sure that performance wouldn’t be different for different cut-off points?
- Abstract claims sample efficiency improves — I don’t see any result (e.g., learning curve) substantiating this claim as only final numbers are given.

### Questions
I don't fully understand how the SM learning auxilliary task differs significantly from prior work on SM learning. I understand the novelty comes from combining with online model-free RL. In addition to responding to the weaknesses raised above, I'd appreciate gaining some insight into the novelty of learning the SM.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Successor Flow Features (SF²), a representation learning framework that integrates flow matching with Successor Measure (SM) to bridge SM estimation and online policy optimization, to enhance sample efficiency of RL training. SF² enforces a structured linear decomposition into a time-invariant embedding and a time-dependent projection, enabling seamless integration with off-policy algorithms like TD3 and SAC. Empirical results on DeepMind Control Suite tasks show improved sample efficiency and training stability over successor feature baselines. The core idea of combining flow-based generative modeling with successor representations is theoretically novel, but several critical gaps limit its comprehensiveness and practical impact.

### Strengths
1. The integration of flow matching with SM for online RL is a creative extension of both generative modeling and successor representation paradigms. The structured linear decomposition (time-invariant embedding + time-dependent projection) addresses SM’s lack of compact, online-friendly representations, which is a non-trivial technical contribution.
2. Experiments on 7 DeepMind Control Suite tasks demonstrate consistent performance gains over standard TD3/SAC and SF-based baselines. The detailed hyperparameter analysis (EMA coefficient, denoising steps, feature size) and ablation studies add rigor to the results.
3. The paper establishes meaningful links between SF² and existing methods (Successor Representation, Diffusion Spectral Representation), providing conceptual clarity and grounding the framework in prior literature.

### Weaknesses
1. Insufficient Justification for Successor Representations in Online RL: The paper frames sample efficiency as a key advantage, but successor representations are inherently designed for reward-dynamics decoupling (enabling generalization/zero-shot transfer). No evidence is provided that SF² outperforms state-of-the-art sample-efficient representation methods (e.g., SPR, CURL) that do not rely on successor paradigms. This raises questions about whether successor representations are necessary for online RL’s core goal of sample-efficient single-task learning.
2. Missing Comparisons with Critical Baselines: The baselines are limited to standard TD3/SAC and SF variants (TD3Sim/SACSim). The absence of comparisons with representative sample-efficient RL methods—particularly SPR (Self-Predictive Representations), which directly targets data efficiency via self-supervised representations—undermines claims of advancing the state-of-the-art in online RL.
3. Unproven Generalization to Pixel-Based Environments: All experiments use structured state features (e.g., joint angles, velocities) from DeepMind Control Suite. The paper provides no analysis or modifications for pixel-based inputs (high-dimensional, unstructured data), which are common in real-world online RL. It remains unclear if SF²’s linear decomposition and flow matching components can adapt to image data.
4. No Evaluation on Discrete Action Spaces: The work exclusively focuses on continuous control tasks with TD3/SAC. Successor representations (e.g., SR) originated in discrete state-action spaces, but SF²’s performance and adaptability to discrete actions are unaddressed. This limits the framework’s generality, as discrete action scenarios (e.g., game playing, recommendation systems) are central to online RL.
5. Complexity: The paper notes SF²’s ~2x longer running time than standard TD3 (e.g., 1300s vs. 659s on AcrobotSwingup).

Reference

SPR: Schwarzer, Max, et al. "Data-Efficient Reinforcement Learning with Self-Predictive Representations." ICLR 2021.

CURL: Laskin, Michael, Aravind Srinivas, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." ICML 2020.

### Questions
1. Does SF² bring better generalization performance?
2. In what RL scenarios is SF² most effective?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Successor Flow Features ($SF^2$), which uses flow-matching technique to approximate Successor Measure (SM). The authors design a linear decomposition of the vector field $u$, separating the time-conditioned and time-invariant components. The time-invariant part is viewed as the learned representation of state action pair and is used by downstream reinforcement learning algorithms. Experiments are conducted on DeepMind Control Suite, and the results demonstrate the superiority of the proposed algorithm.

### Strengths
1.	The linear decomposition of $u$ is the key innovation of this paper and provides a clear architectural insight. It is well-motivated, and both its efficiency and sufficiency are thoroughly discussed.
2.	The proposed method can be used as a plug-in module and can be integrated in a wide range of RL algorithms.
3.	The experiments are clear and comprehensive, covering effectiveness, hyperparameter sensitivity and efficiency. Experimental details are clearly organized in appendices, making them easy enough to be reproduced.
4.	Experimental results clearly demonstrate the superiority of the proposed algorithm over the baseline algorithms, which are strong enough.

### Weaknesses
1.	The presentation of this paper is not sufficiently clear (see Questions for details)
2.	The literature review is somewhat limited. For instance, the paper does not cite Agarwal et al. [1], “Proto Successor Measure: Representing the Behavior Space of an RL Agent” (arXiv:2411.19418, 2024), which appears to be highly relevant to the topic and should be discussed for completeness.

### Questions
1.	Some of the notations used in this paper are not properly introduced. 

    a)	The notation of Successor Measure introduced in lines 135-136 is defined as a probability distribution over state sets, but it is used later as a probability distribution over states (lines 170-171).

    b)	The notation of $ODE(\cdot, \cdot)$ (line 187) is not clearly introduced as a function, although the ODE used by flow matching is described in line 104.
2.	It would be better to provide a complete version of RL training pseudocode in Algorithm 1. The current version includes only value loss, but omits policy loss, which is confusing.
3.	There are existing works that have employed successor measures for RL. The author is encouraged to emphasize the importance of integrating successor measures for online RL representation learning.
4.	The learned representation is only used as input to $Q$ function. Is there a way to allow the policy $\pi$ to also leverage the learnt representation?

### Soundness
3

### Presentation
2

### Contribution
2
