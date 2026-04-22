# Goal Reaching with Eikonal-Constrained Hierarchical Quasimetric Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Goal-Conditioned Reinforcement Learning (GCRL) mitigates the difficulty of reward design by framing tasks as goal reaching rather than maximizing hand-crafted reward signals. In this setting, the optimal goal-conditioned value function naturally forms a quasimetric, motivating Quasimetric RL (QRL), which constrains value learning to quasimetric mappings and enforces local consistency through discrete, trajectory-based constraints. We propose Eikonal-Constrained Quasimetric RL (Eik-QRL), a continuous-time reformulation of QRL based on the Eikonal Partial Differential Equation (PDE). This PDE-based structure makes Eik-QRL trajectory-free, requiring only sampled states and goals, while improving out-of-distribution generalization. We provide theoretical guarantees for Eik-QRL and identify limitations that arise under complex dynamics. To address these challenges, we introduce Eik-Hierarchical QRL (Eik-HiQRL), which integrates Eik-QRL into a hierarchical decomposition. Empirically, Eik-HiQRL achieves state-of-the-art performance in offline goal-conditioned navigation and yields consistent gains over QRL in manipulation tasks, matching temporal-difference methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Eikonal-Constrained Quasimetric Reinforcement Learning (Eik-QRL), a novel goal-conditioned reinforcement learning framework that integrates constraints derived from partial differential equations (PDEs), along with its hierarchical extension, Eik-HiQRL. The method reformulates the discrete trajectory constraints in Quasimetric Reinforcement Learning (QRL) into continuous-time constraints governed by the Eikonal PDE, thereby providing a more theoretically grounded and smooth representation of distance-based objectives. The effectiveness of the proposed approach is empirically validated on the ogebnch benchmark.

### Strengths
1.The paper is well-written, with a clear motivation and a high level of completeness in both theoretical and experimental aspects.

2.The paper presents clear theoretical contributions with a rigorous derivation chain from QRL to HJB and finally to the Eikonal equation. Lemma 4.7 establishes the 1-Lipschitz property of the optimal value function, while Theorem 4.8 provides a high-probability value recovery guarantee, ensuring the theoretical soundness of the framework.

3.In terms of algorithmic design, the trajectory-free nature of the method represents a key advantage, particularly evident on stitching datasets, where traditional approaches rely heavily on trajectory reconstruction. The PDE constraint serves as an implicit regularizer, enhancing out-of-distribution estimation. Moreover, the hierarchical design is well-structured: the high-level abstraction operates in a simplified latent space to avoid complex dynamics, while the low-level temporal difference (TD) module compensates for the limitations of Eik-QRL.

4.The experiments are highly comprehensive, covering six categories of environments—PointMaze, AntMaze, HumanoidMaze, AntSoccer, and Manipulation tasks. The paper also introduces an innovative evaluation metric, the collision rate, which addresses a notable gap in the existing literature. Furthermore, extensive ablation studies are conducted, including five additional experiments in the appendix, providing thorough empirical validation of the proposed approach.

### Weaknesses
1.The assumptions are somewhat overly idealized—for instance, the requirement of unit-speed isotropic dynamics (i.e., f(s,a)=a) does not hold in real-world robotic systems, where manipulators typically exhibit complex, nonlinear dynamics. This limits the direct applicability of the theoretical model to practical robotic control scenarios.

2.On the manipulation tasks in the ogebnch benchmark, the proposed method does not outperform previous state-of-the-art algorithms, indicating that its advantages may be less pronounced in environments requiring fine-grained control and complex dynamics.

3.Although Eik-QRL represents a practical compromise that replaces full quasimetric projection with PDE-based constraints for handling high-dimensional and complex dynamical systems, the authors do not provide results on high-dimensional visual environments. As a result, the generality and robustness of their claims remain uncertain without further empirical validation in such settings.

4.The authors are encouraged to clearly specify the differences and relationships among Eik-QRL, QRL, HIQL, and the closely related Eik-HIQL, and to explicitly discuss the advantages and limitations of Eik-QRL relative to these methods. It is recommended to include a comparative table in the related work section to enhance clarity.

5.To the best of my knowledge, QRL has demonstrated experimental effectiveness in online settings; however, it remains unclear how Eik-QRL performs in online training scenarios.

6.Computational efficiency is not reported—for example, there is no comparison of training time or memory consumption. The PDE constraint requires automatic differentiation to compute gradients ‖∇_s d_θ(s,g)‖, which may be more expensive than discrete constraints. Appendix E only states “4 hours on RTX 3090” without providing comparisons to baseline methods.

### Questions
1.Regarding the fundamental issue of the isotropy assumption: Can the authors quantify how much the optimal value function deviates from the 1-Lipschitz property under realistic robotic dynamics? Can they provide a theoretical analysis or bound—even a relaxed one—for cases with non-isotropic dynamics? Since HJB-QRL is theoretically more general, why does it perform poorly in practice, and is there room for improvement?

2.Proposition B.1 — Hierarchical Analysis:The analysis is based on a 1D toy environment (Fig. 5). Can this be extended to higher-dimensional settings?How is the correlation coefficient ρ estimated or controlled in practical environments?

3.Why does the unconstrained version (Eq. 22) outperform the constrained version (Eq. 21)? Table 5 shows that Eik-QRL outperforms Eik-QRL-λ, suggesting that a soft penalty may be more effective than a hard constraint. Does this observation contradict the theoretical motivation of the method?

4.Figure 6 shows Quasimetric-H-actor has the tightest bound. Why should we use Eik-HiQRL instead?

5.The authors consider a continuous formulation of QRL. To my knowledge, a major limitation of QRL lies in its deterministic dynamics assumption. Could the authors clarify whether Eik-QRL is applicable under stochastic dynamics? If not, how could QRL be extended to handle stochastic environments?

6.Statistical significance: Many results exhibit large standard deviations (e.g., Table 1, pointmaze-giant-stitch: 62 ± 22). Were any statistical tests (such as a t-test) conducted to verify the significance of the reported improvements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Eikonal-Constrained Quasimetric RL (Eik-QRL), a novel approach to Goal-Conditioned Reinforcement Learning (GCRL). It builds upon Quasimetric RL (QRL), which frames GCRL as learning a quasimetric (a shortest-path distance function) $d(s, g)$. The key insight of this paper is to reformulate QRL's discrete, trajectory-based local consistency constraint (i.e., $d(s, s') \le \text{cost}$) into a continuous-time Partial Differential Equation (PDE) constraint.

By assuming simplified unit-speed, isotropic dynamics ($f(s,a) = a$), this PDE constraint reduces to the Eikonal equation: $||\nabla_s d_\theta(s, g)|| = 1$. This new formulation, Eik-QRL, is trajectory-free, meaning it only requires sampling states ($s$) and goals ($g$) rather than full state-action-next-state transitions. This makes it an effective regularizer and highly suitable for offline RL.

To address the limitations of the isotropic dynamics assumption in complex environments, the authors propose Eik-Hierarchical QRL (Eik-HiQRL). This method uses the efficient, trajectory-free Eik-QRL as a high-level planner to propose subgoals in a simple abstract space (e.g., $(x, y)$ coordinates) where the Eikonal assumption holds. A separate, standard low-level TD-learning policy is then trained to reach these subgoals in the full, complex state space.

### Strengths
1.  The core idea of connecting quasimetric learning's local consistency to the Eikonal PDE is a creative, insightful, and theoretically sound contribution.
2.  A major practical strength of Eik-QRL is that it is trajectory-free, only requiring i.i.d. state and goal samples. This makes it more data-efficient and better suited for offline learning from unstructured datasets than the original QRL, which requires transition tuples.
3.  The paper clearly identifies the main weakness of its own Eik-QRL formulation (the strong isotropic dynamics assumption) and proposes a very logical and effective solution: use Eik-QRL as a high-level planner in a simple abstract space where the assumption *does* hold, and use a standard model-free controller for the complex low-level dynamics.

### Weaknesses
1.  The paper's strongest results are in navigation tasks (pointmaze, antmaze, humanoidmaze). In the antsoccer and manipulation tasks (Table 2), the performance gains vanish, and Eik-HiQRL is only "comparable" to baselines. The paper acknowledges this but it suggests the method's applicability is currently best suited for tasks where a simple Cartesian abstract space is available.
2.  The success of Eik-HiQRL appears to be highly dependent on the choice of the high-level abstract space $\overline{\mathcal{S}}$, which must be one where the Eikonal (unit-speed) assumption holds. For navigation, this is intuitively the agent's $(x, y)$ coordinates. For manipulation, the paper states this is a "latent space learned end-to-end", but details on *how* this space is learned and *how* it is constrained to satisfy the Eikonal properties are not in the main paper. This choice seems critical and non-trivial for applying the method to new domains.
3.  The paper derives HJB-QRL (Eq. 7) as a general PDE constraint, but quickly simplifies it to Eik-QRL by assuming $f(s,a)=a$. The justification is that HJB-QRL is "ill-conditioned" and still relies on transitions. This simplification is what limits the method to isotropic dynamics. An alternative, underexplored direction would be to keep the HJB-QRL formulation and use a learned local dynamics model.

### Questions
1.  The performance gains are most pronounced in navigation tasks. In manipulation and antsoccer, where the abstract space $\overline{\mathcal{S}}$ is a "latent space learned end-to-end" (for manipulation), the method is only on par with baselines. Could you elaborate on how this latent space is learned? How do you ensure this learned space adheres to the 1-Lipschitz / unit-speed properties required for the Eik-QRL high-level planner to be effective?
2.  Following on from Q1: How sensitive is Eik-HiQRL to the quality of this abstract space? If the abstract space does not accurately reflect the geometry of the underlying state space (e.g., it thinks two states are close when they are separated by a wall), does the Eikonal constraint still provide a useful learning signal?
3.  The jump from HJB-QRL (Eq. 7) to Eik-QRL (Eq. 8) is a key step, motivated by numerical stability and removing the reliance on transition tuples. Did the authors experiment with a middle ground? For example, using the HJB-QRL formulation but with a *learned* local dynamics model to estimate $f(s, a)$ (or $s' - s$), rather than simplifying the dynamics to $f(s,a)=a$?
4.  In Appendix C.2, the paper discusses two optimization methods: the constrained Lagrangian (Eq. 21) and the soft penalty (Eq. 22), ultimately choosing the soft penalty for stability. Does using a soft penalty risk the Eikonal constraint not being fully enforced (i.e., $||\nabla_s d_\theta|| \neq 1$)? If so, how much does this "approximate" enforcement weaken the theoretical guarantees of Theorem 4.8, which rely on the constraint being met?

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
2

### Summary
- This paper introduces Eikonal-Constrained Quasimetric RL (Eik-QRL), which uses a continuous Eikonal constraint instead of the standard local Quasimetric RL local constraint.
- Eik-QRL is derived and and formulated. This involves: (1) moving from QRL to a HJB-QRL continuous time variant, including bye assuming smooth dynamics and a smooth value function; (2) Moving from HJB-QRL to Eik-QRL, which is easier to optimize in practice, by assuming unit-speed, isotropic dynamics.
- Theoretical guarantees are then proved: Under unit-speed isotropic dynamics in a convex space, the optimal value function has unit gradient norm everywhere, and a universal approximator that satisfies the unit gradient constraint will recover the optimal value function.
- The paper notes the benefits of Eik-QRL include trajectory-free estimation and PDE-based regularization, at the cost of strong assumptions (including the 1-Lipschitz property) which may not hold in general MDPs
- Acknowleging the potential limitations in general MDPs, they propose a hierarchical approach that can use a high-level policy to exploit the benefits of Eik-QRL in a “a lightweight, dynamics-agnostic state-space”, while the low-level policy uses standard TD-learning that does not have restrictive assumptions.
- Their methods are empirically tested in the offline GCRL setting. In the simple pointmaze environment, they show improvements in collisions, in more complex antmaze, they show Eik-QRL is on par with QRL despite Eikonal constraints, and show EikHiQRL achieves the strongest results. In humanoidmaze with more complex dynamics still, they also outperform baselines
- In environments where 3rd party objects and categorical variables mean regularity assumptions do not hold, they observe comparable performance to baselines. They identify designing PDE-constrained algorithms for these non-regular, non-isotropic robotic settings as follow up direction.
- Overall, the paper contributes novel theory which enables a practical method that performs very well in certain settings

### Strengths
- The theory and formulation of Eik-QRL is novel and is an interesting perspective on QRL.
- The clarity and writing in the paper is generally very good. The theory is well written, the writing is well-balanced, limitations are well-acknowledged.
- While Eik-QRL does make strong assumptions, it seems plausible these assumptions could hold to some extent in certain scenarios.
- Eik-QRL often shows strong performance on benchmarks versus baselines, including state-of-the art humanoid maze performance. This seems to establish Eik-HiQRL as a SOTA method in the navigation benchmarks.
- Empirically, performance is reasonable in settings where the strong assumptions do not hold. This seems to mitigate some concerns regarding the limitations of the restrictive assumptions of Eik-QRL.

### Weaknesses
**Assumptions of Eik-QRL may limit real-world use-cases.**

The authors have well-acknowledged the limitations of some assumptions made by Eik-QRL. My general concern is that these assumptions could limit the significance and wider impact of this method. While the paper has evaluated their method on a wide range of environments, including some where their assumptions break down, many of these environments seem to be navigation-based - it seems plausible the method is overfitting to these specific toy navigation benchmarks which may not be representative of many realistic settings.

**Experiments and results could be performed and presented in a more systematic manner to help us better understand the strengths and weaknesses of Eik-QRL**

I would like to see improved benchmarking and systematic experiments that better decompose and provide evidence for exactly where and how  Eik-QRL adds value and where and how it struggles. This could add more clarity on the practical value of the method. I will give some examples below.

The authors state three main benefits that Eik-QRL provides. They state trajectory-free as a strong advantage in the introduction, but I cannot see any clear experiments validating this. While they do have evidence of the “improved state coverage” benefit, this evidence is hard to parse in the results.

The baseline methods they compare against all seem to use slightly different components, making it hard to fairly compare and hard to determine what are the core valuable contributions of Eik-HiQRL (is the performance gain simply due to the hierarchy, or is it due to the Eik method? Is Eik particularly suited to a high-level abstract-space, versus standard TD methods or standard QRL?). Ideally, we could easily compare Eik-QRL and Eik-HiQRL to an equivalent QRL and HiQRL to better distinguish where the value add is. If Eik-HiQRL is proposed as their SOTA method, it could be useful to better compare the Eik high-level policy to other standard high-level policies, while keeping the low-level policy method constant.

In general, presenting aggregated may make it easier to parse overall performance differences between methods.

Perhaps a more thorough and systematic approach could be taken to “red-teaming” the Eik method to finding and understand cases where the assumptions are limiting in practice. This could involve: (i) adding another non-regular environment such as CALVIN, or (ii) better determining the extent to which Eik performance gains are mainly due to the ‘navigation’ component in the navigation environments.

**Other**

After strong theory to introduce Eik-QRL, there is minimal theory or discussion to properly justify the choice to use Eik-HiQRL.

### Questions
- Could you explain why exactly PDE acts as an implicit regularizer?
- Could you propose an experiment to demonstrate why “trajectory-free” could be useful in practice?
- How correlated is the success rate metric with the collision avoidance metric? Is the collision avoidance metric providing much extra information?
- Perhaps you could better highlight your strong performance improvements over QRL in the challenging and non-regular environments?
- Could you provide more environment details in the appendix? Ideally including details regarding the extent to which the Eik assumptions hold in each environment.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper considers the problem of goal conditioned quasimetric reinforcement learning (GCQRL). The novel contribution of the paper is to introduce a (Eikonal) PDE constrained formulation. This Eikonal constrained formulation can be viewed as a refinement of the preexisting HJB PDE constrained GCQRL problem, derived by assuming the continuous-time system dynamics to be isotropic. Although this dynamics assumption is simplistic the formulation trades off this with a very desirable trajectory-free nature, requiring only samples from state, goal distributions. The paper establishes that under some Lipschitz dynamics, cost and value function assumptions, for compact state-action spaces, it is possible for a universal quasimetric approximator to approximate the optimal value function under this Eikonal-constrained formulation. Further, to alleviate issues rising from the complex dynamics that most RL problems have and break the assumptions of the Eikonal formulation, the authors propose a hierarchical version of the Eikonal constrained formulation, where a Eikonal GCQRL operates on an abstract (higher level) state space, and a more traditional TD style RL algorithm operates in the actual (lower level) state space. The authors complement their theoretical contributions with empirical evidence demonstrating that their Eikonal formulation consistently outperforms other QRL methods on a variety of robotics tasks.

### Strengths
1. The presented formulation is trajectory-free and only requires one to sample (state, goals) rather than complete trajectory rollouts which I really appreciate.
2. The core part of the presented Eikonal approach boils down to a constrained optimization problem which can be readily solved through a wide suite of existing physics informed ML methods.
3. The experiments in the paper specifically highlight that the formulation performs well in settings where no theoretical statements can be made currently (i.e., complex, non-Lipschitz dynamics).

### Weaknesses
1. The main weakness is the relatively simplistic dynamics assumption of the formulation (i.e., Lipschitzness, and that the continuous time counterpart is unit speed isotropic). However, I feel that this weakness is adequately acknowledged and addressed by the authors.
2. One of the main assumptions for Lemma 4.7 and Theorem 4.8, stated in line 301 says $c(s,g)=1$ on $\mathcal{K}\setminus g$. Does this mean $g$ can be the only suitable goal in $\mathcal{K}$? If so, I feel that is a limitation as for continuous control tasks, several points in a continuous neighborhood can be also be goal states reaching which leads to success for an RL policy.

### Questions
Besides point 2. in the weakness section, please address:
1. The running cost $c(s,a)$ is introduced in the usual state-action sense in Section 3, but is used in the state-goal sense $c(s,g)$ in Section 4. Could you define what $c(s,g)$ means more clearly like you did for the rewards in goal conditioned RL?
2. Could you give some intuition on whether (local) Lipschitzness of the optimal value function in Assumption 4.4 is a reasonable assumption in practice or not?
3. (Minor formatting point) In table 1, in the row ant maze navigate large, the algorithm giving the lowest $\kappa$ is not colored.

### Soundness
4

### Presentation
4

### Contribution
3
