# Variational Graph Structure Learning for GNNs by Using Marginal Likelihood

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Learning graph structures for Graph Neural Networks (GNNs) can improve their performance, but it involves a challenging search over the large discrete space of all possible graphs. Prior works often enforce fixed constraints on the graph structure to induce properties such as sparsity, but such rigidity can be overly restrictive and harm performance. Here, we propose a simpler alternative to use the marginal likelihood which naturally favors such properties. We show that a variational formulation with Laplace's method automatically leads to a marginal likelihood based objective over discrete graph structures, which can be optimized efficiently using the Gumbel-Softmax trick. We call this approach the Laplace Approximation-based Graph Structure (LAGS) method, and show empirically that it improves the performance of different base GNNs, including recent state-of-the-art GNNs that outperform graph transformers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Laplace Approximation-based Graph Structure method to address the challenge of learning optimal graph structures for GNNs. The proposed model leverages marginal likelihood as an objective. It uses a variational formulation with Laplace’s method to derive a marginal likelihood objective over discrete graphs, optimized via the Gumbel-Softmax trick. Empirically, it improves performance across base GNNs (GCN, GraphSAGE).

### Strengths
1. Using marginal likelihood avoids rigid, task-specific constraints, enabling automatic regularization that aligns with GNN inductive biases.

2. The variational formulation with Laplace’s method provides a principled link between marginal likelihood and graph structure learning.

3. Ablations show marginal likelihood correlates with generalization and edge importance, providing actionable insights into learned graph quality.

### Weaknesses
1. Calculating and inverting the Hessian (even with approximations) increases computational cost. No comparisons of running time were found. For example, on the ogbn dataset, the running time difference between adding and not adding LAGS, and the runtime of other baseline methods.

2. Scalability depends on kNN/observed graph priors, which may exclude potentially optimal edges outside predefined candidates. Whether this part has defects is not supported by quantitative experimental results.

3. Gains on highly homophilic graphs (e.g., Pubmed) are minimal (~0.3%), suggesting limited utility for a large portion of well-structured data.

4. Excessive hyperparameters increase the cost of parameter tuning, raising concerns about its practicality. Furthermore, no sensitivity experimental results on these hyperparameters were provided.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

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
This paper proposes LAGS (Laplace Approximation-based Graph Structure Learning), a variational framework for learning graph structures in GNNs via marginal likelihood maximization. The key idea is that the marginal likelihood naturally regularizes the learned structure by balancing model fit and complexity. The paper applies Laplace’s approximation and the Gumbel-Softmax trick to enable optimization over discrete graphs, demonstrating consistent performance gains across homophilic and heterophilic datasets.

### Strengths
The paper presents a conceptually clear and theoretically motivated approach connecting Bayesian marginal likelihood to graph structure learning.

The use of Laplace approximation to derive a tractable surrogate objective is technically sound and novel.

Empirical results show consistent improvements over base GNNs.

### Weaknesses
The experimental validation is limited in scope relative to the paper’s motivation. Since the paper emphasizes learning from noisy or unreliable graphs, the absence of experiments on synthetic datasets with controlled graph perturbations is a missed opportunity. Such experiments could directly test robustness to varying noise levels and validate the claimed advantage of the method.

Scalability remains a concern, while approximations for the Hessian are discussed, the empirical section does not provide clear runtime or memory comparisons against competing methods.

The empirical gains of stronger GNNs on standard benchmarks, though consistent, are relatively modest (around 1%–2% on many datasets) and may fall within variance bounds.

### Questions
How does the proposed method scale with the number of nodes and edges?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Accurately learning graph structures is fundamental, as it enables numerous downstream applications and provides substantial benefits for real-world scenarios.

### Strengths
Accurately learning graph structures is fundamental, as it enables numerous downstream applications and provides substantial benefits for real-world scenarios.

### Weaknesses
1. The proposed model is tested only on node classification task, however several types of graph based task exists and are not considered in the paper. This raises a natural question if the approach is limited to node classification and cannot be generalized to other graph based tasks like link prediction, and graph classification?
2. The result table shows the effect of proposed method, LAGS, on GraphSAGE and GCN. The gain is really low compared to GraphSAGE nullifying the contribution from LAGS. Moreover compared  to the other GSL (Graph Structure Learning)  methods like LDS and SUBLIME methods, there is no visible advantage of using LAGS.
3. The formatting in Table 1 is not consistent as it does no explain the meaning of bold face as bold face generally indicate the highest score.
4. Overall the evaluation is weak.

### Questions
see above

### Soundness
1

### Presentation
2

### Contribution
2
