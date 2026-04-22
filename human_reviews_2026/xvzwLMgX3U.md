# Soft Constraints, Strong Solutions: Optimizing Intra-Operator Parallelism for Distributed Deep Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 0, 6, 2

## Abstract
As deep learning models grow in size and complexity, efficiently mapping their computations onto distributed hardware is a central challenge for systems and compiler design. A key technique for addressing this challenge is intra-operator parallelism, which partitions individual operations across multiple devices. To accelerate research on automated intra-operator parallelism, Google curated a benchmark suite of 25 large-scale instances drawn from real production workloads including Graph Network Simulators, U-Nets, diffusion models, and Gemma 1 and Gemma 2 language models, and organized the ASPLOS/EuroSys 2025 Contest on Intra-Operator Parallelism for Distributed Deep Learning. The contest formalized intra-operator parallelism as a constrained combinatorial optimization problem in which each computational-graph node must be assigned an execution strategy that minimizes compute and communication cost while satisfying strict time-varying memory limits. This paper presents the winning solution. We show that relaxing the hard memory constraints enables the problem to be reformulated as a Cost Function Network optimization task. Building on this idea, we develop a solver that combines adaptive penalty-based relaxation with efficient Cost Function Network optimization. The method quickly produces feasible strategies with costs near the global optimum on nearly all benchmark instances, consistently outperforming XLA, the production-grade compiler used in TensorFlow and JAX, often by orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an approximate algorithm for a combinatorial optimization problem derived from intra-operator parallelization in distributed LLM training and inference. Given a graph where each node has an activation time interval and a set of strategies (each with memory usage and time cost), and where each edge's time cost depends on the strategies chosen at its endpoints, the goal is to assign a strategy to every node to minimize total time subject to a memory budget. The proposed algorithm won an ASPLOS contest for this problem. Experiments show the algorithm typically finds solutions with latency between 1× and 2× the optimal.

### Strengths
1. The algorithm is simple yet effective.
2. The optimization problem comes from a real deployment scenario, so the work has practical impact.
3. The algorithm outperforms other contest teams and the production compiler XLA.

### Weaknesses
1. The paper would benefit from a clearer connection between the formal optimization problem and the original intra-operator parallelization task in LLM training/serving.
2. It is surprising that a well-optimized system like XLA can be an order of magnitude slower in some cases; the paper should provide more explanation for this gap.

### Questions
Thanks for submitting to ICLR 2026! I enjoyed reading the paper; it is well written and easy to follow. The algorithm is simple and effective. I have a few suggestions to improve clarity and impact.

**1. Reconnect the optimization problem to intra-operator parallelization.**

The ASPLOS contest abstracts a production problem into a combinatorial formulation, but the paper should better describe the original system-level problem so the paper is self-contained. This will help readers map the mathematical solution back to a concrete scheduling or parallelization strategy in production.

**2. Explain differences with XLA's schedule.**

If the reported cost measures execution latency of the computation graph under a strategy assignment, it is surprising that XLA’s approach can be much slower. An in-depth comparison (or diagnosis) explaining why your solution improves over XLA, for example, differences in objective, search space, heuristics, or memory-time trade-offs, would strengthen the paper.

Overall, this is a solid and practical contribution. Addressing the two points above will make the paper more useful to practitioners and clearer to reviewers.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper presents an intra-operator parallelism solution for a contest, working on a solver for a formulated assignment optimization problem. The key idea is to optimize the objective with relaxed constraints first and greedily adjust the assignment later. It is more of a contest report than an academic paper. It makes little academic contribution: the problem is not new; the method is not novel; the experiment is preliminary.

### Strengths
1. The proposed method is a top-performing solution to a contest.

### Weaknesses
1. The defined problem for intra-operator parallelism is not contributed by the authors, though the problem itself does not contribute novelty and academic significance either. 
2. Authors intentionally design the proposed method (e.g., Techniques 1 and 2 in the proposed method) to fit into the provided workload, while not considering a method for real-case workloads.
3. The idea of solving a relaxed-constraint optimization problem and the adopted algorithm are nothing novel. 
4. The proposed method is not properly and throughly verified. The experiment tried to evaluate a solution for a large-scale problem using a ``low''-configured server. Evaluations on a certain scale computing cluster with realistic distributed learning workloads can be much more convincing.

### Questions
1. I would advise authors to work on realistic system problems on intra-operator parallelism with evaluation on realistic workloads and environments.

### Soundness
1

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
This paper proposes a formal framework for integrating soft constraints—constraints that can be selectively relaxed with associated penalties—into systems that require strong guarantees of correctness and safety. The method achieves promising results on several verification and optimization tasks, showing that it can flexibly handle trade-offs between strict guarantees and optimization goals.

### Strengths
1. This manuscript is well organized, presenting a logically coherent structure .
2. The paper tackles an important and challenging problem, and addresses an interesting topic .
3. The proposed framework is theoretically sound and shows strong empirical outcomes.

### Weaknesses
1. Lack of interpretability or analysis of mechanism. The paper reports strong results but does not explain why the method performs well. A deeper ablation and theoretical intuition would improve clarity.
2. Methodological nature unclear. The approach appears algorithmic rather than learning-based. It would help if the authors clarified whether any learning or adaptation is involved, or whether the framework is purely a deterministic optimization procedure.

### Questions
1. Could the authors provide a more detailed explanation of why your method achieves such good results?
2. Can the authors clarify how the system differs from, or could potentially connect to, deep learning–based constrained optimization methods?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a top-performing solution to the ASPLOS 2025 Contest on Intra-Operator Parallelism for Distributed Deep Learning. The authors propose a solver based on usage-constrained relaxation, where memory requirements are incorporated as soft constraints within the cost model rather than enforced as hard limits. The method uses adaptive weight tuning and greedy post-processing to efficiently generate feasible, low-cost solutions across large computational graphs. The approach achieves state-of-the-art results, significantly outperforming commercial compilers such as XLA and even approaching theoretical lower bounds on several benchmarks.

### Strengths
1. The idea of encoding memory requirements as soft constraints with adaptive penalties is both simple and elegant. It transforms a difficult combinatorial optimization problem into a tractable form while maintaining feasibility.
2. The paper provides extensive experimental validation, including comparisons with commercial compilers and exact solvers, thorough convergence and ablation studies, and a detailed analysis of contest benchmarks.

### Weaknesses
1. The paper reads more like a technical report rather than a research paper. The narrative emphasizes implementation details and experimental results but gives less attention to the high-level conceptual intuition behind the approach.
2. Much of the background context and some of the benchmark and contest results overlap with material already presented in the official ASPLOS contest report.
> Moffitt, Michael D. and Fegade, Pratik, "The ASPLOS 2025 / EuroSys 2025 Contest on Intra-Operator Parallelism for Distributed Deep Learning", Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, 2025.

### Questions
1. How many variables $x_i$ are there in each $c_i$?
2. Why does each node i have a separate $w_i$? It seems $w_i$ is global from the algorithm.
3. For the post-processing step, does the solver traverse the entire computation graph to identify candidate nodes, or is there a targeted strategy for selecting which nodes to revisit?
4. It would be informative to include a latency breakdown across different stages of the solver (e.g., preprocessing, relaxation solving, and greedy refinement) to better understand where most computation time is spent.
5. How sensitive is the final performance to the initial solution? Does the adaptive tuning always converge to similar-quality results regardless of the starting point?

### Soundness
3

### Presentation
1

### Contribution
2
