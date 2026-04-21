# Robust Transfer of Safety-Constrained Reinforcement Learning Agents

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement learning (RL) often relies on trial and error, which may cause undesirable outcomes. As a result, standard RL is inappropriate for safety-critical applications. To address this issue, one may train a safe agent in a controlled environment (where safety violations are allowed) and then transfer it to the real world (where safety violations may have disastrous consequences). Prior work has made this transfer safe as long as the new environment preserves the safety-related dynamics. However, in most practical applications, differences or shifts in dynamics between the two environments are inevitable, potentially leading to safety violations after the transfer. This work aims to guarantee safety even when the new environment has different (safety-related) dynamics. In other words, we aim to make the process of safe transfer robust. Our methodology (1) robustifies an agent in the controlled environment and (2) provably provides---under mild assumption---a safe transfer to new environments. The empirical evaluation shows that this method yields policies that are robust against changes in dynamics, demonstrating safety after transfer to a new environment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a problem where a policy is trained in a controlled environment and then transferred  to the real environment. While safety violation is allowed in a controlled environment, unsafe actions may be catastrophic in the real environment. Unlike the previous work that allow policy transfers as long as the new environment preserves safety-related dynamics, this paper explicitly incorporate the dynamics differences or shifts. The authors propose a method to robustifies an agent in the controlled environment and then provably achieves a safe transfer to new environments. The empirical evaluation shows the effectiveness of their proposed method.

### Strengths
- The authors address a very important and interesting problem. I think such problems are fit to some real-world safety-critical problems.
- The Related Work section is concise but well-written and easy to follow.
- The Background section is also easy to follow and well-presented.
- Assumptions are all reasonable and indeed mild.
- I think Theorem 1 is good. The proof seems to be correct and easy to follow.
- Definitions on safety jump-start, $\Delta$ time to safety, and a safe transfer-learning metric are well-presented despite their complicated notions.
- The resulting algorithm presented in Section 5 is technically sound and easy to understand.

### Weaknesses
- In Related work section, I think there is a missing citations:
    - Zhang, Jesse, et al. "Cautious adaptation for reinforcement learning in safety-critical settings." International Conference on Machine Learning. PMLR, 2020.
- Experimental results seem to include only ablation study. I am wondering why Yang et al. (2023) is not compared with their proposed method. If it has been already included as a baseline method, please clarify in the experiment section.
- I think the experimental settings have not fully written in the main paper and appendix. Given the source code is not attached, I feel the reproducibility of this paper is low.

### Questions
- Q1: In Definition 2, what is the definition of $\omega$? I know that the authors mention that the sum of $\omega$ is equal to 1, but I think it not uniquely defined. 
- Q2: Is it possible to compare the authors' proposed method with Yang et al. (2023)? As the authors mention, this previous work seems to be the state-of-the-art method if I understand correctly.

Since I think this paper is overall of high quality, I am happy to increase the score from the current one (i.e., 5: borderline reject) when my concerns are addressed during the rebuttal period.

### Soundness
3

### Presentation
2

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
This paper addresses a critical challenge in safe reinforcement learning: how to transfer an agent trained in a controlled environment to a real-world setting with different dynamics while maintaining safety guarantees. It propose a methodology that makes the transfer process robust against variations in dynamics between source and target environments.

### Strengths
The theoretical work is sound and well-constructed. Theorem 1, which establishes safety guarantees under transfer, is particularly well-developed and properly scoped. The proofs are thorough without being needlessly complex, though some of the assumptions (like the existence of Q-irrelevant abstractions) might prove limiting in practice. The authors demonstrate that their robustified agents can achieve substantial improvements in worst-case performance, with some configurations reaching complete safety transfer. The empirical validation is thorough and well-designed. The authors test their approach across multiple environments and provide meaningful comparisons between different robustification techniques. The ablation studies effectively isolate the contributions of individual components, though more complex environments could have strengthened the conclusions further. Overall, the paper makes a worthwhile contribution to the field of safe reinforcement learning.

### Weaknesses
The paper exhibits some theoretical limitations in its treatment of safety-preserving abstractions and uncertainty quantification. The foundational assumption of Q-irrelevant abstractions $\sigma: S_\odot \rightarrow S_\diamond$ being continuous and bounded (Assumption 2) might become problematic for hybrid or discontinuous systems, while the bounded uncertainty set assumption $\exists\varepsilon \in \mathbb{R}, \forall P, P' \in U_\odot, \|P - P'\| \leq \varepsilon$ fails to capture fat-tailed uncertainty distributions common in real systems. The robustification approach through entropy maximization $r^\diamond_t(s_t, a_t) = r^b_t(s_t, a_t) + \alpha r^H_t(s_t, a_t)$ lacks guarantees about value function Lipschitz constants, potentially compromising stability under state-space perturbations. Moreover, the control switching mechanism $\pi_b(s_t) = \begin{cases} (\pi^\diamond \circ \sigma)(s_t) & \text{if } c_{t-1} > 0 \\ \pi_\odot(s_t) & \text{else} \end{cases}$ introduces potential chattering issues near constraint boundaries, while the p-tail metric $M_{\leq p}(M_\odot, \pi) = \int_B M(M^\odot_P, \pi) dP$ requires additional measure-theoretic assumptions that remain unjustified.

The empirical validation methodology demonstrates concerning limitations in both scope and rigor. The discretization of the uncertainty set using only $N=8$ values for mass and friction parameters provides insufficient coverage of potential failure modes, while the safety jump-start metric $J(\pi^\diamond, M_\odot) = \mathbb{E}[\frac{c(\tau) - c(\tau')}{c(\tau')} | \tau \sim \rho_1, \tau' \sim \rho'_1]$ masks potentially dangerous temporal patterns through trajectory averaging. The environments tested $(M^\diamond_1, M^\diamond_2, M^\diamond_3)$ share similar underlying physics, failing to challenge the framework with fundamentally different dynamics structures. The paper's treatment of model parameterization through simple mass and friction variations $m_i = (0.5 + \frac{i-1}{7})m^\diamond$ and $\eta_i = (0.5 + \frac{i-1}{7})\eta^\diamond$ inadequately captures the complexity of real-world dynamic variations, particularly in systems with coupled parameters or higher-order effects. Furthermore, the Lagrangian multiplier approach for policy imitation lacks theoretical convergence guarantees when combined with robustification techniques, potentially leading to unstable learning dynamics.

### Questions
- In Section 4.1, you assume the existence of a Q-irrelevant abstraction σ that is continuous and bounded. Could you elaborate on how this abstraction can be constructed in practice for high-dimensional state spaces? A concrete example or algorithm would significantly strengthen the paper's practical applicability.

- The bounded uncertainty set assumption (Assumption 3) seems crucial for your theoretical guarantees. Could you provide insights on how to estimate the bound ε in practice?

- Your empirical evaluation uses N=8 values for discretizing the uncertainty set. Could you justify this choice of granularity? And discuss how discretization errors affect your worst-case guarantees?

- The control switching mechanism π_b(s_t) might lead to chattering. Have you analyzed the frequency of policy switches during transfer?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates robust transfer learning of safety-constrained RL problems. The contribution includes: (1) propose a robust transfer learning framework, (2) theoretical analysis on the safety guarantee in transfer across environments with different dynamics, (3) consolidate safe transfer learning metrics and devise methods to train robust agent.

### Strengths
1. The paper is written in an easy-to-understand manner and I appreciate the visual illustration in the paper.  

2. The paper did a good background study on robust transfer learning problems and combine methods & metrics from related literature to form their own transfer learning framework.  

3. The paper clearly defines the assumptions, mapping, metrics used in their framework and elucidates under what circumstances is safety guaranteed for transfer across tasks with different dynamics.

### Weaknesses
1. The paper highlighted that what sets itself apart (from prior work) is that it does not assume uncertainty set is known. However, the uncertainty set is assumed to be bounded by $\epsilon$. I am not too sure if this assumption can be easily verifiable for different set of tasks, especially when $\epsilon$ is related to $\delta$ in theorem 1: when the norm-distance is high, the more we need to "robustify" to have safety assurance. The bound on the norm-distance $\epsilon$ does not seem to be easily quantified.  

2. In theorem 1, the paper concludes that $\delta$ should be increased (i.e. making the policy more robust to dynamics that may be dissimilar to the training environment) to have safety assurance. However, this $\delta$ parameter does not appear in subsequent chapters. In particular, Chapter 5 (robust training) does not seem to reconnect with this $\delta$ parameter and the theoretical safety guarantee. The theory and the actual training algorithm seems a little disjoint in this regard.  

3. Expanding on item 2 above, how is the problem formulation in "formal statement of the problem" connected to the parameter $\delta$? I am also not too sure how does the training algorithm weigh the noise / perturbations (in Entropy Maximization, Random Action Noise & Adversarial Perturbations) such that sufficient $\delta$ can be reached, in order to achieve good safety performance.  

4. The robust training framework depends on the abstraction function $\sigma$. I am not too sure if this abstraction function can be specified easily in practice? Especially when this function needs to connect the dynamics, reward and cost functions (in source and target environments) in accordance to the formulae listed in Definition 2. It may seem straightforward to define $\sigma$ in the empirical analysis section of this paper since source/target envs are largely similar (albeit with different physics parameter) but I guess in practice it may not be that easy.  

5. Chapter 5.1 introduces state abstraction function and I guess this requires substantial feature engineering effort? Is this exploration bonus a bit short-sighted since it only computes the norm-distance between the current state and the next state?  

6. Chapter 5.2 needs to be expanded: the paper mentions that a Lagrangian multiplier $\lambda$ is used to weigh similarity between two policies. I can't figure out how would this enable "student's policy imitate the guide's whenever the cumulative cost is above the safety threshold". In addition, how does the algorithm decide the correct $\lambda$ value to apply during training? These should be further discussed.  

7. The safety gymnasium task analyzed in empirical analysis seems to be only Goal task. Having more tasks may strengthen the paper in this aspect.

### Questions
- Please refer to the bulleted items listed in the Weaknesses section. I'm particularly curious about how does the algorithm quantify and increase $\delta$ to achieve acceptable level of safety.  

- Also, in the last paragraph of Section 6.2, the authors mentioned that "values shown in the tables are the expected costs of one batch of guides". Is there any challenge in getting the expected costs from multiples batches?  

- Is there any reason why the students are limited to off-policy RL algorithms: SAC and DDPG? For deterministic policy, why is DDPG tested over TD3?  

- In Appendix F (Safety Jump-Start Heatmaps), non-deterministic policy seems to be less robust than deterministic policy. Could you discuss why? This seems counterintuitive to me, especially for a CMDP task.

- I'm more than happy to discuss and please let me know if there's anything I misunderstood or missed out.

### Soundness
2

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
4

### Summary
This work seeks to ensure robustly safe transfer learning when the dynamics of a new environment differ from the original environment. The proposed approach (1) strengthens an agent’s robustness within a controlled environment and (2) provides a provable guarantee under some assumptions of safe transfer to new environments. Empirical results show that this method maintains safety after transitioning to a new environment. The authors also propose several metric for safe robust transfer learning, which might be useful.

### Strengths
1. The topic is interesting and significant, and the solution is overall reasonable, the presentation of empirical results is clear and easy to understand.

2. The results are mostly technically correct and bring both theoretical novelty and practical insights.

### Weaknesses
1. The assumptions are not as mild as claimed by the authors, and there are also many hidden assumptions. For example, the paper requests the existence of a robustly safe policy for under constraints only on $\|P-P'\|$.

2. The technical part is confusing. The authors first claim to solve CMDP but later focus on satisfying constraints on every state-action pair $Q^c(s, a)\leq d$, which is a much more conservative objective. The definition of norm and abstract operator is also sloppy.

3. The robustness heavily relies on the action disturbance. There are many other dynamics-perturbation-based robust MDP algorithm such as [1], the authors should consider additional algorithms.

[1] Robust markov decision processes: Beyond rectangularity
V Goyal, J Grand-Clement - Mathematics of Operations Research, 2023

4. The humans knowledge is important in the proposed formulation and is not properly discussed. For example, the abstraction needs to be safety-irrelevant, but sometimes all the observations are related to safety, and sometimes one does not know which observation is safety-irrelevant.

### Questions
1. How is the theorem 1 derived? More specifically, there is a hidden assumption that there exists a $\pi$ that is safe for all $P$ with $\left\|P-P^{\diamond}\right\| \leq \delta$? Solving a robustly safe policy is also a very challenging task, especially for a very large number $\delta$.

2.  Is $P$ a matrix or a nonlinear transition operator? is the state and action space finite or continuous?

3. In the experimental section, why is the mapping from friction and mass to transition dynamics continuous and bounded? For example, in some cases, the friction changes from static friction to kinetic friction, and the change point is usually not continuous. 

4. What exactly are the safety criteria are you considering? in the CMDP problem formulation in line 118, you said the goal is $C(\pi)\leq d$, while later you convert it to the constraints on the $Q^c$ function. They are not equivalent and the $Q^c$ constraints are much more restrictive than the $C(\pi)$ constraints.

Other minor questions related to the notations:

1. What is the superscript ${}^*$ means in line 244?

I will consider raising my score if the question are properly answered.

### Soundness
2

### Presentation
2

### Contribution
3
