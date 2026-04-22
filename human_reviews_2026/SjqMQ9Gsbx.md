# Global optimization of graph acquisition functions for neural architecture search

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Graph Bayesian optimization (BO) has shown potential as a powerful and data-efficient tool for neural architecture search (NAS). Most existing graph BO works focus on developing graph surrogate models, i.e., metrics of networks and/or kernels to quantify the similarity between networks. However, optimization of the resulting acquisition functions over graph structures is less studied due to their complexity and formulations over the combinatorial graph search space. This paper presents explicit optimization formulations for graph input spaces, including properties such as reachability and shortest paths, which can then be used to formulate graph kernels and associated acquisition functions. We theoretically prove that the proposed encoding is an equivalent representation of the original graph space and provide a general formulation for neural architecture cells that incorporates node and/or edge-labeled graphs with multiple sources and sinks regardless of connectivity. Numerical results over several NAS benchmarks show that our method efficiently finds the optimal architecture for most cases.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Briefly summarize the paper and its contributions. You can incorporate Markdown and Latex into your review.
This article proposes an equivalent representation of a general labeled graph in an optimized variable space, where each graph corresponds to a unique feasible solution. It further introduces a universal kernel formula to measure graph similarity, which is compatible with the proposed encoding. This method achieves global acquisition optimization based on graph Bayesian optimization in neural structure search.

### Strengths
1.	The paper proposes an equivalent representation of general labeled graphs in the optimization variable space, ensuring that each graph corresponds to a unique feasible solution. Moreover, it introduces a unified kernel formulation that quantifies the similarity between two labeled graphs at the levels of graph structure, node labels, and edge labels.  The advantages over baselines were demonstrated in NAS Bench 101, NAS Bench 201, and NAS Bench 301.
2.	The formulas and derivation proofs in the article are very detailed and accompanied by complete code.

### Weaknesses
1.	The benchmarks used (NAS Bench 101, NAS Bench 201, and NAS Bench 301) are all from before 2022. Similarly, the baseline methods such as GCN, NAS BOT, and NAS BOWL are also from before 2021. No experiments were conducted on the latest benchmarks or with more recent baseline methods.
2.	This paper lacks an analysis of the algorithm's time complexity.
3.	The evaluated benchmark is limited to NAS, lacking experiments on real-world tasks, which makes the contribution relatively limited.

### Questions
1.	Could experiments be added on more recent and broader benchmarks and baselines?
2.	Could an analysis of the algorithm’s time complexity be provided?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
NAS-GOAT casts cell-based neural architecture search as a Mixed-Integer Program in which graph topology, reachability, shortest-path features and a GP acquisition function are jointly optimized. The resulting MIP is solved to global optimality at every BO step, eliminating hand-crafted mutations and providing certificates of optimality under the surrogate model. Experiments on three public NAS benchmarks demonstrate competitive or superior query efficiency versus recent sampling-based or evolutionary BO baselines.

### Strengths
1. The paper is clearly written and easy to follow.
2. The authors design a full condition plan of NAS graph space.
3. The code is supplied, and the hyper-parameters are reported.

### Weaknesses
1. The complexity of the method should be analyzed.
2. The main content in Theorem 1 is more likely a modeling plan of the graph space, but it takes too much space in the paper, which makes readers uncomfortable. In addition, Theorem 1 is unnecessary to be a theorem.
3. The experiments are all conducted on NB101~301, it is better to evaluate the method on more datasets. Besides, the method cannot achieve SOTA in some of cases.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes NAS-GOAT, a framework for globally optimizing graph-based acquisition functions in Bayesian optimization (BO) for neural architecture search (NAS). The authors formulate the graph search space—including reachability, shortest paths, and node/edge labels—as a mixed-integer program (MIP), enabling exact optimization of acquisition functions. The method generalizes prior graph BO formulations (e.g., BoGrape) to handle weakly-connected or disconnected DAGs common in NAS. Experiments on NAS-Bench-101, 201, and 301 show that NAS-GOAT efficiently finds near-optimal architectures, often outperforming or matching state-of-the-art baselines.

### Strengths
++ This method extends graph BO to NAS by relaxing the strong connectivity assumption of BoGrape.

++ Comprehensive experiments on three major NAS benchmarks under both deterministic and noisy settings demonstrate robustness and efficiency.

### Weaknesses
-- The MIP encoding for graph structures builds heavily on BoGrape, with the main adaptation being the relaxation of strong connectivity. While this is non-trivial, the paper could better highlight what specific constraints were modified or added to handle NAS-specific DAGs. 
Specifically, the claim that BoGrape is unsuitable due to strong connectivity is not followed by a clear explanation of how this is resolved beyond "generalizing the graph encoding."

-- I am afraid that this method is not a "plug-and-play" solution. The MIP model must be manually re-derived and re-implemented for each new search space topology. This creates a significant barrier to practical adoption and limits its applicability to new or evolving NAS problems.

### Questions
1. I suggest the authors provide more analyze about the differences between this method and BoGrape. As I am concerned, the contribution of this work lies in the adoption of BoGrape for NAS tasks.

### Soundness
2

### Presentation
3

### Contribution
2
