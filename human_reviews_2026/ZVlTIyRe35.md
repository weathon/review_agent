# Neural Graduated Assignment for Maximum Common Edge Subgraphs

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
The Maximum Common Edge Subgraph (MCES) problem is a crucial challenge with significant implications in domains such as biology and chemistry. Traditional approaches, which include transformations into max-clique and search-based algorithms, suffer from scalability issues when dealing with larger instances. This paper introduces ``Neural Graduated Assignment'' (NGA), a simple, scalable, unsupervised-training-based method that addresses these limitations. Central to NGA is stacking of differentiable assignment optimization with neural components, enabling high-dimensional parameterization of the matching process through a learnable temperature mechanism. We further theoretically analyze the learning dynamics of NGA, showing its design leads to fast convergence, better exploration-exploitation tradeoff, and ability to escape local optima. Extensive experiments across MCES computation, graph similarity estimation, and graph retrieval tasks reveal that NGA not only significantly improves computation time and scalability on large instances but also enhances performance compared to existing methodologies. The introduction of NGA marks a significant advancement in the computation of MCES and offers insights into other assignment problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the *Neural Graduated Assignment* (NGA), a neural optimization framework designed to solve the MCES problem efficiently. Traditional algorithms for solving MCES suffer from scalability issues, especially when dealing with large graphs. The NGA approach overcomes these limitations by leveraging a learnable temperature mechanism and unsupervised training, allowing the algorithm to scale efficiently and avoid computational bottlenecks. The paper provides theoretical justification for the efficacy of NGA, demonstrating its ability to balance exploration and exploitation in the search process. Experimental results show that NGA significantly improves computation time, scalability, and performance compared to existing methods, making it a promising solution for MCES and related problems.

### Strengths
1. Novel Approach: NGA represents advancement in solving the MCES problem by integrating neural components with traditional optimization frameworks. The use of a learnable temperature schedule and the end-to-end trainability of the model is an innovative approach.

2. Theoretical Foundation: The authors provide a strong theoretical analysis of NGA, including its convergence properties and the mechanisms through which it escapes local optima. This adds robustness to the method and makes it more reliable in practical use.

3. Extensive Experiments: The paper includes thorough experimentation across multiple datasets and tasks, such as MCES computation, graph similarity estimation, and graph retrieval, showing that NGA outperforms existing methods in terms of both accuracy and computational efficiency.

### Weaknesses
1. The paper's reliance on modeling MCES as a QAP leads to O(N²) space complexity due to the affinity matrix. For large graphs (e.g., those with more than 100 nodes), this could pose significant memory challenges, making the method less feasible for very large-scale graphs. The paper does not provide an analysis of memory usage, which is a crucial aspect when considering practical applications for large datasets.

2. While the paper claims that NGA is interpretable, the exact mechanisms through which the model learns to assign graph correspondences are not fully clear. The interpretability of the learned assignments, especially in complex graph structures, could be more thoroughly discussed.

3. While the paper compares well against traditional solvers and some learning-based graph matching models (NGM), it could be strengthened by a comparison to other recent neural approaches for combinatorial optimization on graphs, e.g., unsupervised methods for graph matching.

### Questions
1. Could you provide more details on how the learnable temperature schedule in NGA compares to other dynamic temperature annealing methods in optimization? Is there a specific advantage in using this formulation over others?

2. Could you provide an analysis of CPU/GPU memory usage w.r.t. the scale of graphs?

3. How does the performance of NGA change when applied to graphs with significant amounts of noise or missing data? How would NGA perform in scenarios with highly asymmetric graph pairs (e.g., when the graphs have very different numbers of nodes/edges)? Could you analyze the robustness of NGA?

4. What impact does the choice of neural network architecture have on the overall performance of NGA? Would other architectures, like graph transformers, yield better results?

### Soundness
3

### Presentation
3

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
This work tackles the Maximum Common Edge Subgraph (MCES) problem: given two labeled graphs, the goal is to find the largest shared subgraph with the maximum number of common edges. This problem is especially relevant in molecular analysis and is NP-complete. The proposed method, Neural Graduated Assignment (NGA), is an unsupervised, learnable approximation with three main components: (i) it constructs an Association Common Graph (ACG) that only considers node and edge pairs that are label-compatible; (ii) it learns a soft node-to-node assignment that is iteratively refined using a learnable, high-dimensional temperature, which the authors argue improves convergence behavior; and (iii) it discretizes this soft assignment into a final match, using multiple sampled candidates at inference time to encourage exploration. The experiments evaluate NGA both on MCES quality and on downstream tasks such as graph similarity prediction and graph retrieval. NGA outperforms prior baselines under time-constrained scenarios on molecular datasets. Ablation studies support several design choices (use of high-dimensional temperature, sampling strategy, etc.) and provide insight into optimization dynamics.

### Strengths
- The method explicitly balances exploration and exploitation of the solution space.
- The paper provides a clear analysis of the optimization dynamics, which gives provable guarantees for convergence.
- The experiments are thorough: multiple tasks (MCES, similarity, retrieval), strong ablations, and consistent gains over competitive baselines.
- It provides a better trade-off between scalability and performance than the prior works.

### Weaknesses
1. The method is not reusable for more than one pair of graphs at a time.
2. The paper assumes molecular settings, which have particular properties: the ACG is sparse, the node and edge labels are known and meaningful, etc.
3. Runtime is mainly evaluated under a fixed time cap; it would be useful to see quality vs. time curves or difficulty-stratified results.
4. Minor: some citation formatting issues in the appendix (e.g. lines 719,855,885).

### Questions
1. Why would supervised methods perform badly with multiple ground truths? Don't most tasks in deep learning have multiple equally good solutions anyway?
2. How do the methods compare in number of learnable parameters?
3. Can the labels/features be anything? How would the method perform if they are e.g. too fine-grained or noisy? For instance, in molecular datasets we know that the atom types are a good label, but in other instances we might not know what to choose.
4. Relatedly, if the graphs are unlabeled, would it make sense to create the labels by another procedure (e.g. by structural cues) and then run the method? Would this perform better than not giving them any labels?
5. How does complexity change if the ACG is not sparse? In which (non-molecular) settings can that occur?
6. Which (non-molecular) domains are other realistic targets for NGA, and which remain out of scope under the current assumptions?

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
5

### Summary
This paper introduces Neural Graduated Assignment (NGA), a novel unsupervised neural optimization framework for solving the Maximum Common Edge Subgraph (MCES) problem. The authors formulate MCES as a Quadratic Assignment Problem (QAP) via the construction of an Association Common Graph (ACG). The core innovation lies in replacing the fixed temperature parameter in classical Graduated Assignment (GA) with a learnable, high-dimensional temperature parameterization, which enables adaptive exploration and exploitation during optimization. The method is unsupervised, scalable, and theoretically analyzed for its convergence and local optima escape behavior. Extensive experiments on molecular datasets demonstrate that NGA significantly outperforms existing methods in accuracy, scalability, and efficiency, and shows strong performance in downstream tasks like graph similarity computation and retrieval.

### Strengths
1, Novel Formulation and Guarantees: The introduction of the Association Common Graph (ACG) is a crucial contribution. It provides an elegant way to ensure that any valid subgraph extracted from it is a correct common subgraph of the input graphs. This formulation cleanly transforms the MCES problem into a QAP with inherent structural guarantees, a foundational step that enables the subsequent neural optimization.
2, Rigorous Theoretical Underpinning: The paper goes beyond empirical results by providing a solid theoretical analysis. It explains how NGA escapes local optima (Theorem 1) by leveraging the variance of the gradient, and why the product parameterization accelerates convergence (Proposition 2) compared to a scalar one. This theoretical grounding significantly strengthens the methodological claims.

3, Differentiable and Adaptive Optimization Core: The proposed Neural Graduated Assignment (NGA) mechanism is the paper's central innovation. By replacing the static temperature in classical GA with a learnable, high-dimensional parameterization (  β_l = W_1^T  W_2), the method dynamically balances exploration and exploitation. This design eliminates cumbersome manual scheduling and allows the model to adapt its optimization trajectory to the specific problem instance, leading to faster convergence and better performance.

### Weaknesses
1.Limited Discussion on Unlabeled Graphs: The method assumes labeled graphs (node/edge features). While this is reasonable for molecular data, many real-world graphs are unlabeled or partially labeled. The paper does not discuss how NGA might be adapted to such settings, which limits its generalizability.

2, Computational Overhead of ACG: The construction of the ACG is central to the method but may become computationally expensive for very large graphs. The paper does not analyze the scalability of ACG construction in depth, nor does it compare its overhead relative to the overall optimization.

3, Theoretical Assumptions: The theoretical analysis relies on small ∣ β_l ​ ∣ assumptions (Lemma 1), which may not always hold in practice. The empirical distribution of  ∣ β_l ​ ∣  (Fig. 11) shows both small and large values, so the applicability of the theory across all layers is not fully justified.

### Questions
1，ACG Scalability: What is the time and space complexity of constructing the ACG? How does it scale with graph size and label dimensionality? Could approximate or sparse ACG constructions be used for very large graphs?

2， Generalization to Unlabeled Graphs: Have you considered or experimented with unlabeled graphs? Could structural embeddings (e.g., positional encodings) replace or complement label information in such cases?

3， Training Stability: The product parameterization β_l = W_1^T  W_2 can lead to unstable gradients. Did you observe such issues during training? Were any techniques (e.g., gradient clipping, normalization) used to stabilize training?

4, Choice of Parameterization: Why was the product form W_1^T  W_2 ​ chosen over other parameterizations (e.g., MLP or direct scalar)? Was this motivated by empirical performance or theoretical insights?

### Soundness
3

### Presentation
3

### Contribution
3
