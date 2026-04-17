# Finite-time Convergence Analysis of Actor-Critic with Evolving Reward

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Many popular practical reinforcement learning (RL) algorithms employ evolving reward functions—through techniques such as reward shaping, entropy regularization, or curriculum learning—yet their theoretical foundations remain underdeveloped. This paper provides the first finite-time convergence analysis of a single-timescale actor-critic algorithm in the presence of an evolving reward function under Markovian sampling. We consider a setting where the reward parameters may change at each time step, affecting both policy optimization and value estimation. Under standard assumptions, we derive non-asymptotic bounds for both actor and critic errors. Our result shows that an $O(1/\sqrt{T})$ convergence rate is achievable, matching the best-known rate for static rewards, provided the reward parameters evolve slowly enough. this rate is preserved when the reward is updated via a gradient-based rule with bounded gradient and on the same timescale as the actor and critic, offering a theoretical foundation for many popular RL techniques. As a secondary contribution, we introduce a novel analysis of distribution mismatch under Markovian sampling, improving the best-known rate by a factor of $\log^2T$ in the static-reward case.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides a finite-time convergence analysis for single-timescale actor–critic algorithms when the reward function evolves over time. It proves that if the reward changes slowly (in a Lipschitz and bounded manner), the standard $O(1/\sqrt{T})$ convergence rate can still be achieved. The analysis extends known results for static rewards and introduces an improved treatment of distribution mismatch under Markovian sampling.

### Strengths
- The theoretical framework is well-specified and the proofs are systematic; paper is well presented and self-contained
- This work tackles an underexplored theoretical setting and the extension  of the single timescale actor-critic algorithm to evolving rewards bridges a theoretical gap
- Improved rate under Markovian sampling where the tighter control over distribution mismatch (shaving off $\log ^2 T$) is a nice refinement, even if tangential to the main theme

### Weaknesses
- The paper does a poor job of positioning the significance of its work in comparison with prior literature on theoretical analysis with changing dynamics in non-stationary RL, adversarial RL and performative RL (see Questions below)
- The evolving reward model considered here is too restrictive. The paper considers rewards evolve smoothly and slowly, whereas real-world scenarios involve abrupt or feedback-driven changes
- I believe the proof techniques have limited theoretical challenge (see Questions below) with the discounting factor for rewards essentially acting as a soft-sliding-window to forget the past and Assumption 4.1 ensuring sufficient exploration to discover the changed rewards. Further, this setting can be viewed as a two timescale situation by considering the rewards that are sufficiently slow changing to be on slower timescale and the policy learning to be on the faster timescale thereby rendering convergence results here similar to convergence results in two timescale algorithms in stochastic optimization.

### Questions
- Non-Stationary Reinforcement Learning literature [1,2,3] (where the rewards and the transition probabilities are changing) characterize the static/dynamic regret. Could the authors comment further on the choice of the rate of convergence as the metric of analysis?
- Although in the slightly different setting of average reward, two-timescale, natural actor-critic, [4] analyzes the dynamic regret under evolving rewards and transition probabilities with no limitation on how they change
- Techniques to handle changing rewards aided by smoothness and boundedness have been studied in adjacent literature such as Performative RL [5] and adversarial RL [6, 7]. How do methods developed in this work compare with these?

References: \
[1] Cheung, W. C., Simchi-Levi, D., and Zhu, R. Reinforcement learning for non-stationary markov decision processes: The blessing of (more) optimism. ICML 2020. \
[2] Feng, S., Yin, M., Huang, R., Wang, Y.-X., Yang, J., and Liang, Y. Non-stationary reinforcement learning under general function approximation. ICML 2023. \
[3] Mao, W., Zhang, K., Zhu, R., Simchi-Levi, D., and Basar, T. Model-free nonstationary reinforcement learning: Near-optimal regret and applications in multiagent reinforcement learning and inventory control. Management Science 2024. \
[4] Jali, N., Pathak, E., Sharma, S., Qu, G., Joshi, G. Natural Policy Gradient for Average Reward Non-Stationary RL; 2025.
[5] Mandal, D., Triantafyllou, S., and Radanovic, G. Performative reinforcement learning. ICML 2023. \
[6] Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, and Tiancheng Yu. Learning adversarial markov decision processes with bandit feedback and unknown transition. ICML 2020. \
[7] Gergely Neu, Andras Antos, András György, and Csaba Szepesvári. Online markov decision processes under bandit feedback. NeurIPS 2010.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides the finite-time analysis of a single-timescale actor-critic with evolving reward.

### Strengths
-It is the first work considering single-timescale actor-critic with an evolving reward function.

### Weaknesses
This paper raises several significant concerns.

1. Assumption 4.1 is questionable. The definition of $A_\theta$ differs from all cited references, where the expectation is taken over the stationary distribution, whereas the authors instead take the expectation over the discounted state-visitation distribution. This formulation is highly uncommon and requires a strong, well-justified explanation.

 2. The claimed improvement over previous results by a factor of $\log^2 T$ is incomprehensible. The authors attribute this improvement to Proposition 4.8, which they state is independent of the mixing time of the ergodic Markov chain. However, this is not the source of the $\log^2 T$ term in prior works (see Wu et al., 2020; Chen \& Zhao, 2023; Tian et al., 2023; Chen \& Zhao, 2025). The authors should therefore provide a direct comparison with prior analyses and clarify how exactly they eliminate the $\log^2 T$ error term, which was indispensable in earlier studies.

 3. The contribution of this work appears incremental. The only notable difference from previous studies is the introduction of an evolving reward. However, no novel technical tools have been developed to address the evolving-reward setting, as its properties are already guaranteed by the exhaustive Assumption 4.5.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first finite-time convergence analysis of a single-timescale actor-critic algorithm in a setting where the reward function evolves over time. This non-stationary reward setting is relevant to practical RL techniques like reward shaping, entropy regularization, and curriculum learning. The authors analyze a scenario where reward parameters can change at each time step, affecting both policy optimization (actor) and value estimation (critic). Under standard assumptions (linear function approximation for the critic, Lipschitz continuity of policy and reward, sufficient exploration), they derive non-asymptotic bounds for both actor and critic errors. The main result shows that an O(1/√T) convergence rate is achievable, matching the best-known rate for static rewards, provided the total variation of the reward parameters (FT) is sufficiently small (specifically, O(1/T) to preserve the rate).

### Strengths
1. The paper formalizes an important yet under-theorized problem. Providing theoretical guarantees for RL algorithms with evolving rewards directly addresses the gap between empirical practice and theoretical foundations.
2. This is the first work to provide finite-time convergence guarantees for actor-critic methods in the presence of evolving rewards under challenging Markovian sampling. The analysis is non-trivial and represents a substantial step forward in the theory of RL.
3. The novel analysis of distribution mismatch (Proposition 4.8) that improves the bound for the static-reward case is a meaningful secondary contribution, demonstrating the paper's technical depth beyond its primary focus.

### Weaknesses
1. Strong Assumptions:​​ The analysis relies on several strong assumptions that may limit its applicability to very complex environments. The most significant is the ​​linear function approximation​​ for the critic, which is a common but restrictive starting point. The theoretical community is increasingly focused on non-linear (neural network) approximations. The Lipschitz continuity assumptions, while standard, can also be difficult to verify or enforce in practice.
2. ​​Narrow Scope of Reward Evolution:​​ The analysis requires the reward to evolve "slowly enough." While this is a natural necessary condition, it may not capture all interesting practical scenarios where rewards might change in a more abrupt, phase-based manner (e.g., in some curriculum learning setups).

### Questions
What is the convergence result for AC with neural network approximation, as it has been studied by previous theoretical work?

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
3

### Summary
This work proposes an actor–critic algorithm with an evolving reward function and provides a finite-time convergence analysis. By introducing a KL-based regularization term, the authors isolate the dynamic component into a new parameter, denoted as $\varphi$, and derive theoretical results characterizing the algorithm’s performance as $\varphi$ evolves over time.

### Strengths
1. This work addresses the problem by introducing new parameters to capture the changing components, and derives results that characterize the system’s behavior through the evolution of these parameters.

2. The work further provides theoretical guarantees supporting the proposed approach.

### Weaknesses
The novelty of this work appears questionable, and the final results do not align with the stated goals. From the formulation section, the problem is presented as a standard Markov Decision Process (MDP) with a KL-regularized evolving reward function. In this formulation, the evolution of the reward arises solely from the regularization term. If the regularizer is deterministic, the optimal policy should be uniquely determined. However, in the algorithmic and theoretical sections, the authors introduce a new parameter, denoted as $\varphi$, to absorb the evolution of the reward function and update $\varphi$ in an arbitrary manner. This treatment deviates from the original problem formulation, leading to an inconsistency between the theoretical analysis and the stated objectives.

### Questions
1. Why is the parameter $\varphi$ allowed to change arbitrarily? What is the theoretical justification or practical motivation for treating $\varphi$ as a freely evolving variable?

2. Is the evolving reward solely induced by the regularization term, or is the reward function itself assumed to be adversarial or non-stationary? Clarifying this distinction is essential to understand the problem setting and the validity of the proposed analysis.

### Soundness
2

### Presentation
2

### Contribution
2
