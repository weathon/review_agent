# SKARL: Provably Scalable Kernel Mean Field Reinforcement Learning for Variable-Size Multi-Agent Systems

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Scaling multi-agent reinforcement learning (MARL) requires both scalability to large swarms and flexibility across varying population sizes. A promising approach is mean-field reinforcement learning (MFRL), which approximates agent interactions via population averages to mitigate state-action explosion. However, this approximation has limited representational capacity, restricting its effectiveness in truly large-scale settings. In this work, we introduce \underline{S}calable \underline{K}ernel Me\underline{A}n-Field Multi-Agent \underline{R}einforcement \underline{L}earning (SKARL), which lifts this bottleneck by embedding agent interactions into a reproducing kernel Hilbert space (RKHS). This kernel mean embedding provides a richer, size-agnostic representation that enables scaling across swarm sizes without retraining or architectural changes. Furthermore, a cylindrical kernel function is introduced to ensure universal approximation over functional space. For efficiency, we design an implementation based on functional gradient updates with Nyström approximations, which makes kernelized mean-field learning computationally tracable. From the theoretical side, we establish convergence guarantees for both the kernel functionals and the overall SKARL algorithm. Empirically, SKARL trained with 64 agents generalizes seamlessly to deployments ranging from 4 to 256 agents, outperforming MARL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces SKARL (Scalable Kernel MeAn-Field Multi-Agent Reinforcement Learning), a novel framework that integrates mean-field learning with reproducing kernel Hilbert space (RKHS) representations. By using kernel mean embeddings, SKARL provides agents with a rich, high-dimensional feature representation of the entire population distribution, rather than a coarse statistical summary. The authors develop an efficient algorithm using functional gradients and Nyström approximations, proving theoretically that their method is both highly expressive (a universal approximator) and robust to changes in the population (Wasserstein-Lipschitz continuous). Empirically, SKARL outperforms strong MARL baselines in large-scale cooperative tasks.

### Strengths
1. By representing the swarm as a distribution instead of a fixed set of agents, a policy trained with one number of agents (e.g., 64) can be directly deployed in environments with a different number of agents (e.g., 4 to 256) without any retraining. This overcomes a major limitation of traditional methods.

2. Unlike previous mean-field methods that use simple statistics (like averages), SKARL uses kernel mean embeddings in a reproducing kernel Hilbert space (RKHS). This provides a much richer, high-dimensional representation of the agent population, allowing the system to capture complex structural information and higher-order differences between distributions.

3. In experiments, SKARL is shown to achieve superior performance on large-scale cooperative tasks, "consistently outperforming strong MARL baselines" in both cumulative reward and training stability.

### Weaknesses
1. The authors claim that existing mean-field approaches rely on first-order moment statistics, which provide only coarse summaries of the population. This simplification limits expressiveness and hinders adaptation across swarm sizes, since higher-order structural differences between distributions are ignored. I suggest the authors provide more theoretical or empirical discussion on this limitation instead of directly presenting their solutions, which would further strengthen the contributions of this work. For example, why is higher-order information valuable? Why do previous mean-field approaches limit expressiveness?

2. This work improves the previous mean-field Q-function and embeds interactions using a kernel cylindrical representation. The Q-function is equal to the polynomial combinations of KME kernels. This combination increases the computational complexity compared to previous mean-field methods. Does this method scale to a large number of agents (from a kernel computational complexity perspective)?

3. The authors claim that the representation method integrates seamlessly with standard multi-agent value-decomposition methods such as VDN (Sunehag et al., 2017), QMIX (Rashid et al., 2018), and QPLEX (Wang et al., 2020). However, they do not incorporate the representation method into these methods. It is unclear whether the representation method can improve the performance of current value-decomposition methods.

### Questions
Please see the Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SKARL, a new framework for multi-agent reinforcement learning (MARL) that aims to solve the challenges of scalability and flexibility in large swarms. The core idea is to move beyond simple mean-field approximations by representing the entire population of agents as a distribution embedded in a Reproducing Kernel Hilbert Space (RKHS). This allows the policy to learn from rich, size-agnostic features of the swarm, enabling zero-shot generalization to different population sizes. The method is supported by a theoretical analysis and empirical results showing strong performance on large-scale coordination tasks.

### Strengths
1.  The core idea of using kernel mean embeddings to represent the agent population is a powerful and elegant way to create a size-agnostic policy. It's a significant conceptual step up from traditional mean-field methods that rely on simple averages, though not necessarily entirely novel.

2.  The paper is backed by theory, including convergence guarantees and a formal analysis of the zero-shot generalization error. This provides a solid foundation for the empirical results.

3.  The zero-shot transfer experiments demonstrate that a single policy trained on 64 agents can be effectively deployed on swarms of up to 256 agents. This is desirable for real world applications, though standard for MF-based MARL.

### Weaknesses
1.  The paper does not position itself within a growing body of work on kernel methods with mean-field systems and learning on distributions. The novelty of the proposed method is unclear without a discussion of highly relevant works, such as various works on general mean field control that does not rely on first-order approximations [1-2] or specifically kernel-based approaches with mean-field limits [3-8]. This omission makes it difficult for readers to assess the paper's unique contribution.

2.  The experiments are confined to multi-particle navigation tasks. The experiments are also a bit small, training on only $64$ agents, which is a bit limited given the proposed complexity improvements. Moreover, the experiments are limited to a single problem dynamics.

3.  The current ablation studies are useful but don't fully justify the complexity of the proposed RKHS machinery. A missing piece is a comparison against a simpler baseline, such as using a standard deep set, MARL based on mean field control, or a small MLP to process the mean-field statistics. Without this, it's hard to tell if the full power of kernel methods is truly necessary for the observed performance gains.

[1] Carmona, R, et al. Model-free mean-field reinforcement learning: mean-field MDP and mean-field Q-learning. The Annals of Applied Probability 33.6B (2023): 5334-5381.

[2] Mondal, W. U., et al. On the approximation of cooperative heterogeneous multi-agent reinforcement learning (MARL) using mean field control (MFC). Journal of Machine Learning Research 23.129 (2022): 1-46.

[3] Fiedler, C., et al. (2023). Reproducing kernel Hilbert spaces in the mean field limit. Kinetic and Related Models, 16(6), 850-870.

[4] Fiedler, C., et al. (2023). On kernel-based statistical learning theory in the mean field limit. Advances in Neural Information Processing Systems, 36, 20441-20468.

[5] Fiedler, C., et al. (2025). Recent kernel methods for interacting particle systems: first numerical results. European Journal of Applied Mathematics, 36(2), 464-489. 

[6] Cui, K., et al. (2024). Learning Decentralized Partially Observable Mean Field Control for Artificial Collective Behavior. ICLR.

[7] Szabó, Z., et al. (2015). Two-stage sampled learning theory on distributions. In Artificial Intelligence and Statistics (pp. 948-957). PMLR.

[8] Szabó, Z., et al. (2016). Learning theory for distribution regression. Journal of Machine Learning Research, 17(152), 1-40.

### Questions
1.  Could you clarify the novelty of your framework in light of recent work on kernel methods in the mean-field limit and learning on distributions? Specifically, how does your approach relate to or differ from the works above?

2.  To better justify the complexity of the kernel cylindrical functions, could you provide a comparison against a simpler architecture? For instance, a standard MFRL model where the mean-field term is processed by a more expressive neural network (e.g., a small MLP or a Deep Set).

3.  Your framework works for homogeneous swarms. How do you see it adapting to settings with agent heterogeneity, which is a key feature of many complex MARL benchmarks? About the homogeneous agent assumption, what issues prevent one from simply adding heterogeneity via the states?

4. "MFRL simplifies further but lacks multi-scale coordination." This statement is too short for me to understand what exactly is lacking. Can you explain a bit more?

5. Can you quantify or discuss the improvement in approximation over first-order methods such as MFRL?

6. Given the work uses kernel methods, I am curious if the methodology will empirically scale to higher dimensions in states or actions, or if there are any limitations here?

### Soundness
2

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
3

### Summary
This paper proposes SKARL, a reinforcement learning framework that leverages kernel mean embeddings (KME) and mean-field theory to address scalability issues in multi-agent reinforcement learning (MARL). The authors claim that SKARL can handle variable population sizes through a kernelized formulation in a reproducing kernel Hilbert space (RKHS). The paper introduces the notion of kernel cylindrical functionals, derives convergence guarantees using functional analysis (e.g., Frechet and Lions derivatives), and presents experimental results on swarm-like environments to demonstrate scalability and performance.

### Strengths
The paper tackles an important and challenging problem: scalable MARL under population uncertainty. The idea of using kernel mean embeddings for mean-field RL is conceptually interesting and could, in principle, lead to generalizable models across population sizes. The theoretical framework and use of functional-analytic tools (e.g., Lions derivatives, Nyström projection) suggest potentially strong mathematical grounding.

### Weaknesses
1.	Mathematical notation is confusing and not self-contained: 
Many critical mathematical objects—such as the definition of the Lions derivative, the cylindrical functionals, or the exact meaning of $D$ in Eq. (4) are introduced without sufficient explanation. At first glance $D$ appears to represent the number of samples, but later it becomes clear that it actually denotes the number of different kernel components, which is confusing. Similarly, in lines 124–125, the quantity $R_f^i$ appears without prior definition and seems to be an undefined or inconsistent symbolThese ambiguities make the formulation extremely hard to interpret. (See also the questions section)
2.	Lack of intuition and structural explanation:
The paper immediately dives into abstract functional definitions without providing a high-level overview of what the algorithm actually does.
It is unclear whether the kernel functions are fixed or learned, and if learned, how they are trained or parameterized.
A clear intuitive summary—what is being optimized, what role the kernel plays, and how scalability arises—would make the method far more accessible.
At present, the algorithmic pipeline is opaque and difficult to connect to implementation.

### Questions
1.	In Eq. (6), is the variable $x$  equivalent to the state-action pair $(s,a)$?
2.	The parameter $\theta_h$ is introduced but not subsequently used—does Eq. (7) describe an update on $\theta_h$  or on $h$ itself?
3.	Are the kernel functions $g$ trained jointly with the policy/value function, or are they fixed? If they are trainable, how is this implemented in practice?

### Soundness
2

### Presentation
2

### Contribution
2
