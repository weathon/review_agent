# Agent-Chained Policy Optimization

- Decision: Reject
- Scores: 2, 4, 0, 6

## Abstract
We study Cooperative Multi-Agent Reinforcement Learning (MARL), where the aim is to train decentralized policies that maximize a shared return. Existing methods typically employ either iterative best-response updates, which converge only to Nash Equilibria (NE) that may be far from the global optimum, or simultaneous learning with centralized critics, which lack convergence guarantees to the optimal joint policy without strong assumptions on 
decomposable value functions.
We introduce the Agent-Chained Belief MDP (AC-BMDP), which reformulates MARL as a serialized decision process where agents act sequentially while maintaining beliefs over actions taken by preceding agents. This enables the definition of agent-specific value functions that are naturally chained together. Building on this framework, we propose Agent-Chained Policy Iteration (ACPI) and prove that it converges to the globally optimal joint policy.
We further develop this framework into a practical actor–critic algorithm, Agent-Chained Policy Optimization (ACPO). On standard benchmarks, ACPO consistently surpasses state-of-the-art baselines, with the performance advantage growing significantly as the number of agents increases.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This works introduces a paradigm for centralised training of actor critic algorithms in multi-agent settings which allows agent-specific value functions to incorporate beliefs over the actions taken by other agents. The authors introduce a formulation for simultaneous actions taken by agents with sequential belief chaining, and prove that it converges to a globally optimal policy.

### Strengths
1. The work is well motivated, as the myopic nature of simultaneous updates in MARL results in inconsistent updates between agents. ACPO is a natural framework to subvert this issue while retaining the benefits of centralised training.
2. The proofs appear correct. (See questions)
3. The authors cover a good selection of benchmarks with continuous and discrete control.
4. The didactic game in Table 1 is easy to follow and a useful example. Figure 1 is also a good explanation for the concept of belief chaining.

### Weaknesses
1. The authors implement ACPO with PPO , but do not indicate whether their method can be extended to Soft Actor Critic[1] or other common actor-critic algorithms. Why was PPO chosen solely? The authors must be transparent about whether this choice was due to implementation difficulties or a lack of performance generalisation.
2. Limited baselines - the authors compare to MAPPO (general MARL AC), HAPPO and HATRPO (BR MARL AC) and QMIX in smacv2. Why were value based methods and more actor critic methods not compared against in these benchmarks? There are existing implementations of VDN, MADDPG, COMIX, and other standard MARL algorithms for the benchmarks provided - if no relevant BR baselines are available, it is at least necessary to understand how ACPO with PPO compares to common MARL baselines. 3. The computational comparison in Figure 5 might be a little unfair, as the BR methods dominate the plot - it is hard to notice that for the 12-hard scenario, ACPO takes seemingly 3-4 times as much as MAPPO to solve the task, despite not reaching a similar multiplier in performance. 
4. Why were the main results obtained with only 3 seeds? Computational constraints are a reasonable answer, but a standard of 5-10 seeds should be the minimum for these results to be considered reliable.

### Questions
1. Can this work for non-cooperative settings? What about mixed objective games?
2. Why does Theorem C.1. use a deterministic policy? Why can a stochastic policy not be used here?
3. Why are the computational comparisons between ACPO and MAPPO so unfavourable? Have the authors missed an implementation detail that could lead to faster runtimes? 
4. Must the agent order always remain the same? What if it were randomized? Does the current ordering not result in asymmetric learning dynamics amongst the agents, even in symmetric games?

### Soundness
2

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
3

### Summary
The paper introduces a new framework for Cooperative Multi-Agent Reinforcement Learning that guarantees convergence to the globally optimal joint policy, overcoming limitations of existing methods that either converge to local Nash equilibria or rely on centralized critics without convergence guarantees.

### Strengths
Originality:

The idea of serializing multi-agent decision making via a belief MDP that maintains consistency with the decentralized setting is novel.
The agent-chained construction elegantly bridges the gap between centralized critics and decentralized value functions, creating a unified formulation that retains theoretical guarantees without restrictive assumptions.
Proving convergence to the global optimum rather than Nash equilibrium represents a significant conceptual advance over HAPPO, HATRPO, and MAPPO.

Quality:

The paper is well-grounded theoretically, with rigorous derivations and clear definitions.
The ablation study and runtime comparison are particularly convincing, showing that performance gains stem from the agent-chaining mechanism rather than auxiliary factors.

Clarity:


Mathematical notation is consistent and readable.
Proof sketches are presented intuitively, with full details deferred to appendices.

Significance:

Theoretically, the paper closes a long-standing gap in cooperative MARL by providing a convergence-guaranteed formulation under the widely-used CTDE paradigm.
Practically, ACPO shows robust scalability as the number of agents increases, which has been a critical bottleneck in MARL.

### Weaknesses
The AC-BMDP belief update assumes tractable distributions over prior agents’ actions. In practice, belief approximation or parameterization details are underspecified, and real-world scalability of belief updates may be challenging.

No comparison to modern off-policy or model-based MARL methods. Although justified for fairness, this limits understanding of ACPO’s off-policy robustness.


The ablation only removes agent-chaining; finer-grained tests could clarify which theoretical component drives the performance gain.

### Questions
The proofs assume deterministic policies; how robust is ACPI’s convergence under stochastic policies used in ACPO?

Could the agent-chained structure be combined with value-decomposition methods to handle credit assignment more efficiently?

The authors mention applications to Multi-Agent LLMs. Could the same belief over prior actions mechanism be re-interpreted as a reasoning chain in communication-based coordination tasks?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces Agent-Chained Policy Optimization (ACPO), a MARL framework based on sequentializing joint actions into micro-steps. It formalizes the process as an Agent-Chained Belief MDP (AC-BMDP), where each agent maintains a belief over preceding agents' actions. A corresponding Agent-Chained Policy Iteration (ACPI) is proved to converge to the globally optimal joint policy. The practical implementation, ACPO, extends PPO with chain-based advantage estimation under CTDE. Experiments on RWARE, SMACv2, and MA-MuJoCo show consistent improvements over MAPPO and HAPPO.

### Strengths
- The paper is well-written and well-structured.

### Weaknesses
The paper’s literature review is insufficient. Sequential decision-making formulations in MARL have already been extensively explored, such as [1-3]. The proposed formulation and algorithm are largely a subset of [3], which already formalized the transformation of an MMDP into a sequential single-agent problem, discussed its global optimality, and applied it to PPO for MARL tasks. Moreover, the paper’s discussion of prior methods and their limitations is less detailed and rigorous than in [3].

[1] Bertsekas, Dimitri P. “Multiagent Rollout Algorithms and Reinforcement Learning.” arXiv preprint arXiv:1910.00120 (2019). 

[2] Wen, Muning, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, J. Wang, and Yaodong Yang. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” arXiv preprint arXiv:2205.14953 (2022).  

[3] Ye, Jianing, Chenghao Li, Jianhao Wang, and Chongjie Zhang. “Towards Global Optimality in Cooperative MARL with the Transformation and Distillation Framework.” (2022).

### Questions
Please discuss the differences between the proposed ACPO and the prior work [1-3].

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors address cooperative Multi-Agent RL (MARL) by introducing Agent-chained belief MDPs and Agent-Chained policy iteration. Together, they contribute to the MARL field by providing a novel paradigm where a cooperative (decentralized) multi-agent problem is reformulated as a serialized decision process where agents act sequentially. The authors demonstrate that an optimal policy of such reformulation also leads to optimal behavior for the original multi-agent problem. Furthermore, the authors demonstrate a practical implemenentation of this approach achieving state-of-the-art empirical results on relevant benchmarks in the field.

### Strengths
- The presented method is presented via both a solid theoretical analysis and an empirical evaluation over several notorious benchmarks in MARL settings, including high-dimensional tasks.
- The paper is clearly written and relatively easy to follow.
- The presented method proves to be scalable as it's easily applicable to currently existing algorithms such as MAPPO. The authors claim that the final practical implementation leads to a similar objective to MAPPO, with the most crucial change being the sequential agent-chaining. This makes it easy to attribute the empirical performance gains primarily to the novel paradigm introduced by the authors.
- The authors also report important considerations on the runtime statistics and computational resources usage (Figure 5 and Appendix J), providing more insights on the reproducibility and scalability of their approach.

### Weaknesses
- The overall empirical experimental evaluation relies on only three seeds. I believe this can considerably hurt the significance of the results, and suggest the authors to increase this to at least 5 (ideally 10) to solidify the empirical findings and claims. This is also evident in Table 2 where there is considerable overlap between the standard errors, with unclear significance over performance gain by one method in particular (only 1 row displays statistically significant higher mean return for ACPO).

### Questions
- Does the requirement of defining a serialized decision process with sequentially acting agent restrict, in practice, the applicability of the algorithm? It's unclear to me whether there exist tasks where this can not be applied, even if considering a belief over unobservable other-agent actions. E.g. for tasks where actions must occur syncronously.
- Could the authors clarify why only 3 seeds have been used for the experimental evaluation and whether this hurts statistical significance particularly for high-dimensional tasks?
- The authors attached an anonymous URL with their codebase in Appendix I. While this would be highly beneficial to the community for reproducibility and further research, the underlying repository seems to be unavailable at the moment. Could and will the authors provide access to it?

### Soundness
3

### Presentation
4

### Contribution
4
