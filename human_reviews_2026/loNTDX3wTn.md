# Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Inspired by the human learning and memory system, particularly the interplay between the hippocampus and cerebral cortex, this study proposes a dual-learner framework comprising a fast learner and a meta learner to address continual Reinforcement Learning~(RL) problems. These two learners are coupled to perform distinct yet complementary roles: the fast learner focuses on knowledge transfer, while the meta learner ensures knowledge integration. In contrast to traditional multi-task RL approaches that share knowledge through average return maximization, our meta learner incrementally integrates new experiences by explicitly minimizing catastrophic forgetting, thereby supporting efficient cumulative knowledge transfer for the fast learner. To facilitate rapid adaptation in new environments, we introduce an adaptive meta warm-up mechanism that selectively harnesses past knowledge. We conduct experiments in various pixel-based  and continuous control benchmarks, revealing the superior performance of continual learning for our proposed dual-learner approach relative to baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces FAME (Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning), a dual-learner framework designed to address the challenges of continual Reinforcement Learning (RL). The FAME framework decomposes continual RL into two distinct yet complementary objectives handled by the dual-learner system. Fast Learner and Meta Learner.

**Fast Learner (Knowledge Transfer):** This component focuses on rapidly acquiring knowledge from a new task. It leverages an adaptive meta warm-up mechanism that selectively transfers prior knowledge from the meta learner to facilitate rapid adaptation and circumvent the issue of negative transfer. This transfer is achieved either by directly copying parameters or by adding a Behavior Cloning (BC) regularization during the early training phase.

**Meta Learner (Knowledge Integration):** This component ensures knowledge integration and safeguards against catastrophic forgetting. It incrementally integrates new experiences by minimizing catastrophic forgetting, as formally defined in the paper. This consolidation process enhances the adaptive meta warm-up for subsequent environments.

Besides, the authors provide definitions about MDP Distance and Catastrophic Forgetting and provide theoretical results used in FAME.

### Strengths
1. Outperforming Performances

2. Clear Motivation 

3. Theoretical Foundations

### Weaknesses
1. Hyperparameter settings are required, e.g., $L$ and $N$.

2. Inefficiency to train multiple policies

3. Insufficient Abalation Study for $N$ and $L$

### Questions
I have two primary inquiries concerning the sensitivity and selection process for the critical hyperparameters $L$ (Warm-Up Step) and $N$ (Weight Estimation Step), whose impact is detailed in the ablation studies.

1.  **On the Non-Monotonic Effect of the Warm-Up Step ($L$):**
    The analysis in Table 5 reveals a complex trade-off associated with the warm-up duration ($L$). While a longer $L$ might intuitively suggest improved learning from the meta-learner, the empirical results show a non-monotonic trend. Specifically, extending $L$ to $50 \times 10^4$ yields marginal gains in Forward Transfer (FT) (0.16 to 0.17) but significantly worsens Forgetting (0.72 to 0.78) and degrades the 'Average Performance' on Breakout. This result is counter-intuitive. Why does a prolonged application of Behavior Cloning (BCI appreciate if you let me know this phenomena.

2.  **On a Principled Strategy for Determining $L$ and $N$:**
    The ablation studies  confirm that performance is highly sensitive to the choice of $L$ and $N$. These parameters appear inherently task-dependent. $N$ dictates the fidelity of the approximated state-visitation distribution ($w_k$ or $\mu_k$), while $L$ manages the plasticity-stability trade-off during adaptation. However, the paper does not seem to offer a principled method for their selection beyond empirical tuning within a specific domain. Could the authors share heuristics or a more adaptive strategy for setting these parameters? For instance, in the Meta-World experiments, the warm-up evaluation was set to 10 episodes. What rationale guided this specific choice, and how might such a heuristic generalize to environments with varying task complexities or episode lengths?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes **FAME**, a principled framework for continual reinforcement learning (CRL) that couples a fast learner (responsible for rapid adaptation and knowledge transfer) with a meta learner (responsible for long-term knowledge consolidation and minimizing catastrophic forgetting). The method is theoretically motivated by defining two new foundations for CRL:
1. MDP Distance, a formal measure of environment similarity.
2. Catastrophic Forgetting, quantified across sequential tasks.

**FAME** integrates these foundations through incremental updates and an adaptive meta warm-up strategy based on hypothesis testing, choosing between finetuning, reset, or meta initialization. Experiments on MinAtar, Atari, and Meta-World benchmarks show strong results across both discrete (DQN-based) and continuous (SAC-based) RL settings.

### Strengths
* The paper provides a rigorous definition of MDP similarity and forgetting, which are often treated heuristically in continual RL.
* The dual-learner analogy (hippocampus–cortex interaction) is conceptually elegant and clearly motivates the architectural decomposition.
* The formulation of knowledge integration as minimizing policy-based catastrophic forgetting is mathematically clean and bridges multi-task and continual RL.
* Demonstrates consistent improvement across diverse environments (discrete and continuous), outperforming strong baselines like PackNet, ProgressiveNet, and PT-DQN.

### Weaknesses
* The exposition is mathematically dense, and while the theoretical sections are solid, the algorithmic intuition could be clearer (e.g., Eq. 5).
* Although some hyperparameter studies are referenced in Appendix they are not discussed anough in the main paper.
* It’s unclear how the meta buffer scales with very long task sequences or partially observable settings.

### Questions
Q: Can you elaborate on the connection between the incremental update rule in Eq. (5) and existing multi-task optimization techniques (e.g., EWC, Distillation, or Distral)?
Q: How does FAME behave when task similarity (MDP distance) is low? Is there a threshold beyond which adaptive warm-up consistently defaults to reset?

### Soundness
4

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
3

### Summary
This paper introduces FAME, a biologically inspired framework for Continual Reinforcement Learning that mirrors the complementary roles of the hippocampus and neocortex in human cognition. The framework decomposes continual RL into two coupled processes, where a fast learner that performs rapid adaptation and forward knowledge transfer, and a meta learner that incrementally consolidates knowledge to mitigate catastrophic forgetting. The authors formally define two foundational concepts for continual RL: (1) MDP Distance, which quantifies environment similarity, and (2) a Catastrophic Forgetting metric applicable to both value- and policy-based RL.
On this foundation, FAME employs adaptive meta warm-up and incremental meta updates based on KL and Wasserstein distances to achieve a balance between plasticity and stability. Extensive experiments on pixel-based and continuous control tasks demonstrate consistent improvements in average performance, forward transfer, and resistance to forgetting compared to baselines such as DQN, PPO, PackNet, and ProgressiveNet.

### Strengths
1. The paper introduces a principled dual-learner formulation inspired by human memory systems, offering a new perspective on continual RL.
2. The formalization of MDP Distance and Catastrophic Forgetting metrics fills a long-standing theoretical gap, providing measurable quantities for transfer and forgetting analysis.
3. The adaptive meta warm-up and incremental knowledge integration elegantly combine statistical hypothesis testing with RL optimization, resulting in a flexible yet stable continual learning pipeline.

### Weaknesses
1. The meta learner’s learned representations are opaque and lack interpretability, and the paper does not analyze the memory or computational scaling of maintaining meta buffers as the number of tasks increases.
2. While theoretically elegant, the algorithm involves multiple nested components, like dual learners, adaptive warm-up, statistical tests, and two types of meta-updates, which may hinder practical adoption or extension without significant engineering effort.

### Questions
1. The definition of MDP Distance assumes shared state and action spaces between tasks. How would this metric behave when the state-action distributions of two environments are only partially overlapping or completely distinct?
2. Is there any explicit regularization to prevent the fast learner from overfitting to transient experiences before meta consolidation? If so, how is the trade-off between immediate adaptability and long-term stability controlled?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes new foundations for continual RL by formally defining measures for catastrophic forgetting and MDP distance—addressing a principle way to understand why continual learning algorithms may or may not work. Building on these foundations, the authors develop Principle Fast and Meta Knowledge RL (FAME), a dual-learner framework that decomposes continual RL into two complementary processes: (1) knowledge transfer via adaptive meta warm-up with hypothesis testing, and (2) knowledge integration via incremental updates that explicitly minimize catastrophic forgetting. Extensive experiments across MinAtar, Atari, and Meta-World validate the approach.

### Strengths
- The paper provides the first formal definitions of MDP distance and network level catastrophic forgetting in continual RL, which is directly optimizable.
- The method is sound: It is an adaptive approach considering different knowledge transfer scenarios. It is also flexible, tested on both value-based and policy based algorithms grounded theoretically.
- Experiments are thorough, with diverse domains covering both discrete action spaces (Atari) and continuous action space control (Meta-World). The results shows improvements upon baseline methods.

### Weaknesses
- If I understand it correctly, the method requires storing a small sample of trajectories from every tasks, leading to potential scalability issue. The sampling approach is also very simple, with only 1-2% of the training steps, which might cause poor estimation of $\mu$ when the environment gets complex.

### Questions
- Regarding the weakness, do you think there is a strategy to discard or store trajectories dynamically from the meta buffer to achieve the same continual learning objective?

### Soundness
4

### Presentation
3

### Contribution
3
