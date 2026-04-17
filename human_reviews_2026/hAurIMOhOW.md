# Learning to Reduce Search Space for Generalizable Neural Routing Solver

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Constructive neural combinatorial optimization (NCO) has attracted growing research attention due to its ability to solve complex routing problems without relying on handcrafted rules. However, existing NCO methods face significant challenges in generalizing to large-scale problems due to high computational complexity and inefficient capture of structural patterns. To address this issue, we propose a novel learning-based search space reduction method that adaptively selects a small set of promising candidate nodes at each step of the constructive NCO process. Unlike traditional methods that rely on fixed heuristics, our selection model dynamically prioritizes nodes based on learned patterns, significantly reducing the search space while maintaining solution quality. Experimental results demonstrate that our method, trained solely on 100-node instances from a uniform distribution, generalizes remarkably well to large-scale Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) and Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) instances with up to 1 million nodes from the uniform distribution and over 80K nodes from other distributions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an enhancement of several state-of-the-art models for Neural Combinatorial Optimization, improving their generalization on larger-scale instances. Specifically, it builds upon the Attention Model by [Kool et al. 2019.], replacing the computation-heavy attention mechanism with a two-step approach. In the first step, the search space is reduced using a learnable policy. In the second step, the solution is constructed by considering only the nodes retained from the first step.

### Strengths
1. The two-step approach for reducing the search space and selecting nodes is elegant and straightforward.

2. The proposed method achieves better running time than existing solutions for large-scale neural routing solvers.

### Weaknesses
1. This paper begins by claiming that distance-based search space pruning can eliminate globally optimal nodes, making it unsuitable for this domain. Then, the proposed method relied on this distance-based pruning approach.

2. Some important baselines are missing, particularly current state-of-the-art NCO solvers for large-scale routing, such as [1]. Comparing the raw results from that paper with the experiments in this work, it appears that [1] performs better on larger problems. 

3. The proposed method is quite naive. It represents only an incremental improvement over existing state-of-the-art models. All the "ingredients" used in this work are already well known. Even the idea of avoiding an attention bottleneck by computing scores between the [first,last] and the remaining nodes has already been explored in [1].

4. This approach is limited to (Euclidean) routing problems only. For non-Euclidean problems, it is not possible to apply coordinate normalization, and likely not feasible to use the initial pruning or candidate node reduction strategies. Additionally, the idea of using the first and the last node in attention computation makes this approach applicable only to the routing problems. This paper claims to present a "Generalizable neural routing solver", but TSP and CVRP alone are not enough to support such a broad claim, and it seems this approach will not work beyond these problems.



[1] Luo et al. Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems, ICLR 2025

### Questions
In addition to the weaknesses, I would like to raise the following questions:

1. Equation (1), which describes static reduction, is not clear. It introduces a threshold as $1−\gamma$, and later $\gamma$ is defined as 10%. Do you remove the 10% or the 90% of the furthest nodes? If you remove only 10%, that seems like a minimal, almost negligible, amount. On the other hand, if you remove 90% of the furthest nodes, that contradicts your earlier discussion stating that distance-based search space reduction is not an acceptable approach because it prunes many promising nodes. Additionally, what does the parameter $\alpha$ represent in this equation?

2. It seems that after the initial step of static graph reduction, the remaining sparse graph may become disconnected. How do you ensure that the remaining graph stays connected and can still produce a feasible solution?

3. Regarding the experiments, the CVRP results are unclear. There is no reported optimality gap - only objective values are given. Moreover, when comparing with other works, it seems that different datasets are used, making any direct comparison impossible. Additionally, why are the TSP results reported for 1K, 5K, 10K, 50K, and 100K instances, while the CVRP results are limited to 1K, 5K, 7K, and 10K? Where does this large gap come from? Does your model perform poorly on CVRP instances larger than 10K?

4. In Section 4, you explicitly explain that your "model is trained solely on 100-node TSP instances from uniform distribution". Do you train only one model on TSP and test it both on TSP and CVRP, or train two models, one for TSP and another for CVRP?

5. What is the motivation for limiting your training instances to only 100 nodes? Your approach does not require any labeled data, so in theory, you could use any problem size during the training. Does this limitation actually benefit training time or performance? Did you try training on larger instances?

### Soundness
3

### Presentation
3

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
This paper proposes L2R, a hierarchical search space reduction framework for neural routing solvers that combines static sparsification, a lightweight learned dynamic reducer, and a local construction policy trained jointly via REINFORCE. Trained only on 100-node uniform instances, L2R shows strong cross-size generalization up to 1M nodes and robustness across distributions, achieving favorable speed–quality trade-offs versus NCO baselines and competitive gaps with classic heuristics at large scales.

### Strengths
1.	The idea of learning a policy for dynamic search space reduction is valuable, especially for scaling constructive NCO to very large routing instances.

2.	The proposed approach is coherent and empirically effective. The reported cross-size generalization (up to 1M nodes) and cross-distribution robustness are noteworthy, with favorable speed–quality trade-offs.

### Weaknesses
1. Experiments for N < 1000 are missing. Most local models without global awareness degrade on small-scale problems; thus, performance on small- and medium-scale instances should be included to assess performance comprehensively. The L2R model is trained on N = 100, but results on the training scale are not reported. Omitting these makes it difficult to judge overfitting and optimality on the training distribution.

2. Given the hierarchical nature of the decision process, established hierarchical RL methods could be directly applied or adapted to improve credit assignment and training stability for L2R. In this context, the simple joint REINFORCE used in the paper may not be the most suitable or advanced choice.

### Questions
As shown in Table 15, the improvement in SSR accuracy is marginal, raising concerns about whether the RL-based reduction policy sufficiently explores and learns optimal strategies. It is possible that the solver policy rarely visits a substantial portion of optimal actions during training, limiting the reduction policy’s opportunity to learn them. Would incorporating supervised signals (e.g., oracle candidate sets) plausibly increase candidate-set recall and improve overall accuracy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to reduce the search space of constructive neural routing solvers with an additional learning to reduce (L2R) module, which consists of an embedding layer and a self-attention layer. Experiments demonstrate impressive performance when generalizing to large-scale instances.

### Strengths
1.	The idea of learning to reduce the search space is clear and reasonable. The writing of this paper is clear and easy to follow. 
2.	The experimental results on generalizing to large-scale instances are significant and impressive.

### Weaknesses
1.	The methodology that utilizing an embedding layer and a self-attention layer for the purpose of learning to reduce search space with thousands of nodes seems brute. Meanwhile, such architectures are prevailing in constructive neural routing solvers when identifying the importance scores of nodes, which, to some extent, limits the novelty and potential contribution of this work.

### Questions
1.	As described in the paper, the static reduction phase only reduces 10% of candidate nodes, which is a very small part. How do you determine the ratio? Is there any experimental evidence to show the necessarily of static reduction?
2.	What about the performance on small-scale instances like 100-node instances? Does the generalization ability come at the cost of degradation on small-scale instances?

### Soundness
2

### Presentation
3

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
The paper applies RL to the search space reduction problem in NCO. The proposed algorithm is simple to implement and exhibits good generalization capability.

### Strengths
- The proposed algorithm is simple to implement and generalizes well to larger instances and different distributions.

### Weaknesses
- The writing can be improved. The related work section, a large part of section 2, and many experiments were moved to the appendix. The "Impact on Constructive NCOs" paragraph in section 2 provides no useful information, as the main content has been moved to the appendix. Many experiments only have their names mentioned in the main text. The author should at least describe the results and implications of the experiments in the main text. I suggest the authors revise their manuscript to include the necessary information in the main text. 
- There is little algorithmic novelty. The paper simply applies RL to the search space reduction problem. The network architecture was reused or derived from prior works with little justification. For example, following previous works, the context embedding is just the sum of the first and last node embeddings. What is the justification for this? Can a better context embedding improve the performance?
- The proposed algorithm addresses a small and specific problem in NCO. It is a nice engineering trick that unlikely applies to other problems.

### Questions
- Why use RL instead of other learning paradigms? A clearer motivation for using RL and a comparison between RL and SL (and possibly SSL) would improve the quality of this paper.

### Soundness
3

### Presentation
2

### Contribution
2
