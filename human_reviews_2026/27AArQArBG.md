# MRVF: Multi-Round Value Factorization with Guaranteed Iterative Improvement for Multi-Agent Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 6, 4, 4

## Abstract
Value factorization restricts the joint action value in a monotonic form to enable efficient search for its optimum. However, the representational limitation of monotonic forms often leads to suboptimal results in cases with highly non-monotonic payoff. Although recent approaches introduce additional conditions on factorization to address the representational limitation, we propose a novel theory for convergence analysis to reveal that single-round factorizations with elaborated conditions are still insufficient for global optimality. To address this issue, we propose a novel Multi-Round Value Factorization (MRVF) framework that refines solutions round by round and finally obtains the global optimum. To achieve this, we measure the non-negative incremental payoff of a solution relative to the preceding solution. This measurement enhances the monotonicity of the payoff and highlights solutions with higher payoff, enabling monotonic factorizations to identify them. We evaluate our method in three challenging environments: non-monotonic one-step games, predator-prey tasks, and StarCraft II Multi-Agent Challenge (SMAC). Experiment results demonstrate that our MRVF outperforms existing value factorization methods, particularly in scenarios highly non-monotonic payoff.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the optimality of current value factorization methods in multi-agent reinforcement learning (MARL). It formulates a stable point theory for analyzing convergence to a global optimum, i.e., optimal joint actions in a multi-agent system. The paper provides one-shot examples for all common factorization methods, such as VDN, QMIX, WQMIX, and QPLEX, showing that they can fail to converge to the global optimum due to restricted representations or strong assumptions about the greedy joint action of Q-values, which is commonly but misleadingly assumed to be optimal. As a solution, the paper proposes a multi-round value factorization (MRVF) framework, which "smoothes" the value landscape by clipping local optima until only monotonic convergence to the global optimum is ensured. MRVF is evaluated in one-shot games, a predator-prey task, and the StarCraft Multi-Agent Challenge (SMAC), and demonstrated to be more effective than prior value factorization methods.

### Strengths
**Novelty**

The paper addresses an important problem with a novel theory and approach. Since I am not aware of similar work, I consider the contributions novel.

**Soundness**

The theory is straightforward, and the counterexamples provided to show the suboptimality of current approaches are sound.

**Quality**

The paper is well-written and easy to follow. The approach and examples are nicely illustrated in figures.

**Significance**

The paper provides an important contribution to the MARL community that can move the field forward. I appreciate the variety in benchmark domains from one-shot to SMAC, which provides sufficient empirical evidence that the approach is effective and scalable.

### Weaknesses
SMAC is considered outdated and largely solved, which is also evident in the paper, where MRVF does not dominate the baselines consistently. Furthermore, SMAC is criticized for having deterministic initial states and observations [1,2,3], which has led to more general benchmark suites, such as SMACv2 [2].

**Literature**

[1] Lyu et al., "A Deeper Understanding of State-Based Critics in Multi-Agent Reinforcement Learning", AAAI 2022

[2] Ellis et al., "SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning", NeurIPS Benchmarks 2023

[3] Phan et al., "Attention-Based Recurrence for Multi-Agent Reinforcement Learning under Stochastic Partial Observability", ICML 2023

### Questions
Optimizing over multiple rounds incurs more computational overhead. Do you have data on how this affects the MRVF?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the convergence properties of value-factorization MARL methods and identifies that single-round monotonic factorizations inevitably converge to suboptimal stable points in highly non-monotonic coordination tasks. It introduces MRVF, a multi-round value factorization framework that iteratively refines joint-action estimates by using payoff increments clipped by previous greedy values, guaranteeing strict improvement of the greedy action until the optimal action is reached. The method is evaluated on multiple benchmarks, showing substantial gains in challenging non-monotonic settings and competitive or better results in monotonic ones.

### Strengths
1. The paper introduces a formal definition of stable points in MARL value factorization and proves why existing methods get stuck in suboptimal equilibria.
2. The paper reframes MARL training dynamics similar to multi-round communication or iterative refinement.
3. The paper conducts multiple domains and ablations, especially in non-monotonic cases.

### Weaknesses
See the questions section.

### Questions
1. What is the computational overhead of multiple rounds per timestep in SMAC and larger benchmarks?
2. How sensitive is performance to the number of rounds and clip threshold?
3. How is MRVF compared to other methods, such as MAPPO methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Multi-Round Value Factorization (MRVF), a framework for cooperative multi-agent reinforcement learning that reformulates value decomposition as a multi-round optimization process. Instead of minimizing a single static loss as in WQMIX or QPLEX, MRVF uses the greedy joint action from the previous round as a dynamic baseline and fits only the positive increments in joint-action value, enforcing a strict improvement condition. The authors prove that, under finite action spaces, this procedure guarantees convergence to the globally optimal joint action. Empirical examples illustrate that MRVF avoids suboptimal stable points observed in prior factorization methods.

### Strengths
- The paper provides a theoretical perspective on why existing value factorization methods may converge to suboptimal stable points.

- The proposed multi-round optimization view and strict-improvement condition is sound.

### Weaknesses
- Presentation and notation are not clear. The use of $u$, $\tilde{u}$, and $\bar{u}$ is confusing and imprecise, making the paper hard to follow for readers not already familiar with value-decomposition methods such as WQMIX.

- Related work is incomplete. Many issues discussed such as suboptimal convergence and stability in MARL have been studied before [1], so the contribution appears incremental.

- The difference from WQMIX ($\alpha=0$) and QTRAN is unclear. There is no sufficient analysis or ablation explaining why MRVF performs better on SMAC.


[1] Ye, Jianing, Chenghao Li, Jianhao Wang and Chongjie Zhang. “Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework.” (2022).

### Questions
1. The presentation should be rewritten for clarity. Many notations and terms (e.g., the several variants of $u$, as well as the definition of “value decomposition” itself) are introduced informally and used inconsistently. Please make these concepts explicit and mathematically precise.

2. What is the concrete difference between the proposed multi-round optimization and simply setting $\alpha = 0$ in WQMIX?

3. If WQMIX uses $\alpha = 0$, or equivalently if we use QTRAN (which can be viewed as a weighted VDN with $\alpha = 0$), wouldn’t suboptimality problem disappear entirely? In that case, any non-optimal greedy action would yield a positive loss, and only when the greedy action is the true $\arg\max Q_{jt}$ would the loss be zero (since other entries are masked out). How does the proposed MRVF differ from this behavior?

4. (Minor) The paper claims that QPLEX fails because suboptimal actions are not unstable. However, QPLEX explicitly introduces a stop-gradient trick in its loss to ensure that suboptimal actions do have nonzero gradients, thus avoiding exactly this issue. How does the proposed analysis reconcile with that fact?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MRVF, a framework for cooperative MARL that iteratively monotonizes a non-monotonic joint payoff landscape by training with non-negative payoff increments relative to the previous round’s greedy action.

### Strengths
- This paper is generally well written and easy to follow.

- The stable point lens to analyze greedy-action convergence for value factorization is insightful. This paper shows that WQMIX can prefer suboptimal stable points and QPLEX/ResQ have their own stability failure modes.

- This theoretical contribution seems correct and meaningful, with multi-round clipping target simple yet effective for preventing stable points under idealized mixer.

### Weaknesses
- I'm mostly concerned about the experimental setup. The SMAC is relatively considered outdated, given it's almost 2026 and that there is already SMAC2, some of the baseline selection are also outdated, with the latest being NA2Q, MAPPO and other more recent baselines are missing.

- Also it seems the experimental results are somewhat different from past works, e.g. in NA2Q most of the baselines could reach 60%+ at 1e6 steps while in this papers result, none of them could reach 40% win rate, I wonder if the experimental setup are different? More details on this would help.

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
3
