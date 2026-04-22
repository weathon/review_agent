# FLOYDNET: A LEARNING PARADIGM FOR GLOBAL RELATIONAL REASONING

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 4, 8

## Abstract
Developing models capable of complex, multi-step reasoning is a central goal in artificial intelligence. While representing problems as graphs is a powerful approach, Graph Neural Networks (GNNs) are fundamentally constrained by their message-passing mechanism, which imposes a local bottleneck that limits global, holistic reasoning. 
We argue that dynamic programming (DP), which solves problems by iteratively refining a global state, offers a more powerful and suitable learning paradigm. We introduce FloydNet, a new architecture that embodies this principle.
In contrast to local message passing, FloydNet maintains a global, all-pairs relationship tensor and learns a generalized DP operator to progressively refine it. This enables the model to develop a task-specific relational calculus, providing a principled framework for capturing long-range dependencies. Theoretically, we prove that FloydNet achieves 3-WL (2-FWL) expressive power, and its generalized form aligns with the k-FWL hierarchy. FloydNet demonstrates state-of-the-art performance across challenging domains: it achieves near-perfect scores (often >99\%) on the CLRS-30 algorithmic benchmark, finds exact optimal solutions for the general Traveling Salesman Problem (TSP) at rates significantly exceeding strong heuristics, and empirically matches the 3-WL test on the BREC benchmark. Our results establish this learned, DP-style refinement as a powerful and practical alternative to message passing for high-level graph reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new architecture replacing the message-passing mechanism in MPNN with a global attentioned-update of node relationships, which aligns with the theoretical k-FWL test. The experiments present satisfyingly good performance on  graph property prediction, neural algorithm reasoning, and real-world applications.

### Strengths
The idea is interesting, which upgrades the attention mechanism in Transformers from the node-level to the edge-level (and further path-level). The experiments are comprehensive, showing the architecture's strong ability over traditional MPNNs on multiple graph tasks, implicating potentially broader applications.

### Weaknesses
1. The main concern is the high training complexity of $O(N^3d)$. Although the authors say the memory usage can be reduced to $O(N^2d)$, I'm afraid that the training time is still huge and infeasible for large graphs. 

2. For experiments, Graph Transformers baselines are lacking, which are more similar to the proposed architecture.

### Questions
1. In line 147, it says "we implemented a dedicated compute kernel that avoids storing these large intermediate tensors, reducing memory usage to $O(N^2 · d_r)$. While this optimization does not alter the theoretical computational cost, it improved training performance by more than 20 times in practice, as hardware memory bandwidth is a limiting factor".  Can you show me the details of the "dedicated compute kernel" and the evidence for the improvement of training performance? I note this critical part is missing in the paper.

2. Can you show the comparison of the training time of your method and other baselines (e.g., MPNN, Transformers)?  I like this work, but I think the training time/complexity should be presented in detail for evaluating the method's practicality.

3. The architecture uses an edge-level attention mechanism. I think it should be mainly compared with Graph Transformers, rather than MPNNs. Can you add some Transformer baselines?

4. Can you explain the relationship between the architecture design and dynamic programming in detail?

5. In the Future Work section, it says, "Ultimately, we envision FloydNet as a core component of future System-2 reasoning systems." Can you give some more explanations or intuitions? And how can we apply it to data structures beyond graphs, e.g., images or natural languages?

6. Some suggestions. 1). I suggest moving the discussion on the expressiveness of the architecture from the Appendix to the main paper. It will enhance the contribution of the paper and make the method more solid; 2). I suggest making the chapter division clearer. For example, the paragraph "Pushing the Limits: The Traveling Salesman Problem (TSP)" and the following paragraph "Settings" should be a subordinate relationship rather than a parallel relationship; 3). I suggest re-plotting all the figures to increase the font size of axes, ticks, and legends for readability.

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
This paper proposes FloydNet, a novel neural architecture for processing graphs that updates intermediate representations of each pair of nodes. It does so via an $O(N^3)-time Pivotal Attention operation that updates each vertex pair by computing a softmax over query-key products along neighboring nodes. In doing so, it evokes the classic Floyd-Warshall algorithm for shortest paths.

The architecture can be further generalized to higher order variants the aggregate over longer paths, at a cost of higher time complexity.

The authors benchmark the theoretical capabilities of the architecture by establishing an equivalence to the 2-WL tests for graph isomorphism, in contrast to GNNs (equivalent to 1-WL).

They exhibit the architecture's reasoning capabilities by comparing it to a wide range of GNN models on benchmarks such as CLRS-30, cycle counting, and Long Range Graph Benchmarks. They further demonstrate the ability of the model to solve synthetic TSP instances more effectively than non-learned heuristics.

### Strengths
FloydNet occupies a distinct middle ground between GNNs, Graph Transformers, and higher-order variants. While the model hinges on softmax attention and lacks a message-passing mechanism, it preserves the permutation-equivariance property and lacks a positional encoding. The motivation for the model is compelling and it is presented clearly and elegantly. 

As far as I am aware, the 2WL equivalence is mathematically sound. This claim effectively situates it within the literature of graph transformers, where the WL-hierarchy is a key quantification of expressivity.

While the computational costs of the attention layer are heavy, the authors nonetheless present a working version of the model with a custom kernel that they test extensively. They demonstrate a compelling scaling with model depth and dataset size.

### Weaknesses
The $O(N^3)$ complexity is a fundamental problem, especially as the construction requires computing a tensor of size $N \times N \times N \times d$. Furthermore, the paper provides relatively little time and memory benchmarking, besides a few noted OOM issues on the CLRS-30 benchmark.

It is unclear whether the baseline models are proper comparisons to FloydNet. For instance, the TSP comparisons are only relative to a single non-learned benchmark, despite the existing of numerous other neural approaches to TSP. For the GNN comparisons, it is unclear whether the alternative approaches are given similar parametric and computational budgets, and transformer-based solutions to graph problems are not included.

### Questions
How do the parameter counts, wall time, and number of training samples compare between FloydNet and other architectures in each benchmark? Including these measurements in tables would help readers understand how much of the improvement can be attributed to the architecture.

Are there plans to release your code in an anonymized form? I was somewhat shocked by how strong the TSP results were---especially the OOD results---and I would appreciate an opportunity to verify that there isn't some kind of contamination or leakage.

Can more context be added to Figure 3(b)? It's unclear to me what the cells and numbers precisely represent?

Why was it necessary to filter out the TSP instances with multiple optimal solutions? Would it be possible to train a FloydNet to find a TSP solution of the correct length, even if that is ambiguous?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces “FloydNet,” a graph model that updates all node pairs via a Floyd–Warshall–style (i–j–k) aggregation, reaching 2-WL-level expressiveness and achieving strong results on several reasoning and TSP-style benchmarks, at the cost of cubic complexity.

### Strengths
- The paper is generally well written and easy to follow.
- You propose a novel solution that appears to provide clear advantages in the evaluated settings.

### Weaknesses
Caveat: This paper lies well beyond my personal expertise. Other reviewers’ points should be clearly prioritized.

**Major Weaknesses:**
- **W1** Scalability: your method’s cubic cost is a clear disadvantage for larger, real-world, and especially sparse graphs. This should be discussed more explicitly. As far as I can tell, the graphs you evaluate on are mostly complete and rather small. It would help to show how your architecture scales to much larger, more sparse graphs and to discuss both training and inference cost (e.g. on  https://arxiv.org/abs/2309.12253).
- **W2** More analysis towards additional computational costs of your methods – how many parameters each network has compared to baselines. It seems an unfair comparison if FloydNet just had a significantly larger parameter&computational budget.
- **W3** There is no limitations section, even though there seems to be enough space. You should explicitly discuss overfitting issues, scalability/cost, and any other known limitations.
- **W4** You filter out about 90% of the cases in the TSP evaluation, which makes the comparison look unfair, since the other solver presumably also handles non-unique solutions. Please clarify why you applied this filter and how FloydNet performs when solutions are not unique.

**Minor Weaknesses:**
- **w5** Table 3 is never referred to in the text.
- **w6** Figure 3b: it is hard to parse what the bars mean. You state “At each step, the colored bar represents the ground-truth maximum-sum subarray,” but there are multiple colors. It would help to name the colors explicitly (e.g. “green for …” and the colors for the endpoints).
- **w7** Font in Figure 4 is too small.

**Expectation Management:**
I am currently inclined to increase the score to 6 or higher if my questions are addressed and the major weaknesses are at least discussed. A higher score would require an evaluation on larger and sparser graphs and a better discussion of the scalability/cost trade-offs, as well as consideration of other reviewers’ opinions.

### Questions
- **Q1** Do you provide details on your kernel for pivotal attention?
- **Q2** In the MST task, Prim without hint significantly outperforms the hint version for slightly larger graphs. Do you have a hypothesis for this behavior?
- **Q3** How do baseline GNN methods perform on the TSP problem?

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
4

### Summary
This paper proposes a new technique for learning on graph structures using edge-centric embeddings, as opposed to the conventional node-centric ones. The authors introduce FloydNet, an algorithmic alignment technique where Floyd-Warshall-type updates are performed on the edges of the graph, with intermediate computations on node triples via attention. They test the method on the CLRS-30 algorithmic benchmark and other graph learning problems, including TSP. The experimental results appear promising, as reported in the paper.

### Strengths
- The idea of applying the Floyd-Warshall dynamic programming algorithm and its alignment for graph algorithms is intuitive and appealing.
- The experimental analysis is well-conducted.
- The proposed PivotAttention is likely a better application of attention mechanisms on graph structures than conventional graph transformers.

### Weaknesses
1.  The proposed approach is subsumed within the K-GNN approach, albeit with a dynamic programming formulation based on the Floyd-Warshall algorithm combined with an attention mechanism. Therefore, it has the same expressivity as K-WL, as noted in the paper. Given this, the additional insights provided by this modification are not entirely clear. While the attention mechanism is useful, how does it impact higher-order aggregations as used in this approach? Is there any intuitive understanding of how pivotal attention benefits, i.e., any insights from the learned attention weights?
1. A clear disadvantage of the approach is the computational complexity of the model, which has been acknowledged in the paper. If this algorithm works in different cases, efficient versions might be of interest. Do authors have any insights from the experiments regarding how this can be alleviated.
1. The TSP experiments are not clearly evaluated. A thorough evaluation would involve following the benchmark TSP setup, comparing with other TSP solvers like MoE [2], SHIELD [1] for OOD generalization.
1. The related work section is highly inadequate. This paper is primarily focused on improving the expressive power of GNNs, but several leading works on this topic are not cited like [3, 4, 5] and others. The authors should have provided a more elaborate discussion, perhaps in the appendix if space is a concern.

## References:

1. Goh, Yong Liang, et al. "SHIELD: Multi-task Multi-distribution Vehicle Routing Solver with Sparsity and Hierarchy." Forty-second International Conference on Machine Learning.

1. Zhou, J., Cao, Z., Wu, Y., Song, W., Ma, Y., Zhang, J., and Xu, C. Mvmoe: Multi-task vehicle routing solver with mixture-of-experts. In International Conference on Machine Learning, 2024.

1. Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural
networks. arXiv preprint arXiv:2002.03155, 2020

1. Dupty, M. H., Dong, Y., & Lee, W. S. PF-GNN: Differentiable particle filtering based approximation of universal graph representations. In International Conference on Learning Representations.

1. Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, and Michael M Bronstein. Improving graph
neural network expressivity via subgraph isomorphism counting. arXiv preprint arXiv:2006.09252, 2020.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
