## Summary

The paper proposes Forest-based Graph Learning (FGL), a novel paradigm for semi-supervised node classification that reformulates message passing as transportation over spanning trees. The key insight is that spanning trees achieve global coverage with minimal edges, enabling efficient long-range information propagation. The framework includes: (1) a pre-processing step that augments graphs using pseudo-labels to improve connectivity and homophily, (2) a homophily-biased tree sampler using Wilson's algorithm, (3) a linear-time recursive tree aggregator (Theorem 1), and (4) a forest fusion mechanism. The authors provide theoretical analysis linking edge-homophily estimator accuracy to tree distribution quality (Theorem 2) and demonstrate strong empirical results across 9 benchmarks.

## Strengths

- **Conceptual novelty**: The decomposition of graph learning cost into "(cost per structure) × (number of structures)" and the recognition that spanning trees occupy an optimal point on this Pareto frontier is genuinely insightful and well-articulated. This reframing provides a principled alternative to both deep local models and shallow global attention.

- **Sound theoretical foundation**: Theorem 1's recursive tree aggregator derivation is clean and correct. The proof that any aggregator satisfying Properties (I) and (II) admits efficient O(n) tree DP is a non-trivial technical contribution with meaningful generality (linear attention, RNNs, and SSMs all satisfy these properties).

- **Demonstrated efficiency**: Table 2 shows compelling runtime advantages: FGL achieves 0.005-0.246 sec/epoch across datasets, consistently outperforming Deep GNNs (GCNII: 0.066-2.843 sec/epoch) and Graph Transformers (DIFFormer: 0.029-0.545 sec/epoch, with several GTs hitting OOM on larger graphs).

- **Consistent performance across graph types**: The method achieves competitive results on both homophilous (Cora: 85.46%, Pubmed: 81.00%) and heterophilous (Texas: 91.89%, Wisconsin: 86.27%) benchmarks, demonstrating robustness to varying homophily levels.

## Weaknesses

- **Missing ablation of pre-processing augmentation**: The graph augmentation step (Section 4.1) uses pseudo-labels to add k-nearest-neighbor edges, improving homophily and connectivity. Table 3 shows ablations for tree sampling strategy and module contributions, but critically does not include a baseline that removes the augmentation entirely while keeping the forest mechanism. This makes it difficult to attribute performance gains to the forest paradigm versus the label-informed graph rewiring. For heterophilous datasets where augmentation may add numerous homophilous edges, this isolation is essential.

- **Incomplete baseline comparison for heterophilous graphs**: Table 1 omits purpose-built heterophily methods (e.g., H2GCN, LINKX, GloGNN, ACM-GCN) that are standard benchmarks in this subfield. While Appendix J.9 includes ADPA, GESN, and HiGNN, relegating these to the appendix understates the competitive landscape for the datasets where FGL shows its largest gains (Texas, Wisconsin, Cornell).

- **Theory-practice gap for Theorem 2**: The theorem establishes monotonicity between the score ratio Δ = p/q and expected tree homophily assuming exact edge labels. In practice, edge scores come from a noisy attention estimator trained on limited labeled data. The paper provides no bound on how estimation error degrades the theoretical guarantees, leaving the connection between the theorem and actual performance informal.

- **Hyperparameter complexity**: The framework introduces numerous hyperparameters: NT (number of trees, 4-15), β₁, β₂ (local submodule weights), KL (local layer count), γ (residual coefficient), and k (augmentation neighbors). For small validation sets (Texas: 20 training nodes, Cornell: 20), this creates substantial risk of inadvertent overfitting during tuning, and the paper does not analyze sensitivity rigorously in the main text.

## Nice-to-Haves

- **Inductive setting evaluation**: The current framework operates transductively (the graph is fixed and all nodes are available during training). Extension to inductive learning—where new nodes may appear at test time—would broaden applicability. The pre-processing augmentation and tree sampling both require the full graph, making inductive extension non-trivial.

- **Memory footprint comparison**: While runtime efficiency is well-demonstrated, peak GPU memory usage against O(n²) Graph Transformers would strengthen the efficiency narrative, particularly for large-scale graphs.

- **Analysis of tree quality**: A visualization or quantitative metric showing that sampled trees actually connect distant nodes (rather than reinforcing local neighborhoods) would strengthen the long-range coverage claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Label leakage" characterization**: The harsh reviewer frames the pre-processing pseudo-label generation as "a subtle form of transductive label leakage." This overstates the issue—generating pseudo-labels from training data and applying them to all nodes is standard transductive practice, not improper information leakage. The concern is valid that augmentation should be ablated, but the "leakage" framing is misleading. Removed.

- **Wilson's algorithm O(n²) worst-case complexity**: The critique that Wilson's algorithm can be O(n²) for poorly-connected graphs is technically correct but the paper (1) cites the standard O(τ(p)) bound with τ(p) ≈ O(n) for most graphs, (2) provides a block acceleration algorithm (Algorithm 3) for dense graphs. This is not a weakness requiring emphasis. Removed.

- **"Standard deviations relegated to appendix"**: While true, the variance data is available in Table 10 of Appendix K. This is a presentation preference, not a substantive flaw. Removed.

- **"Numerical instability in Eq. 8"**: The paper applies L₂ normalization (Eq. 10) after aggregation. The speculative concern about subtraction-induced instability without empirical evidence of numerical issues does not warrant inclusion as a weakness. Removed.

- **Missing GERN-GCN comparison in main text**: The paper discusses this in Appendix J.10 and shows favorable results. The novelty positioning concern is addressed in the paper. Removed.

## Novel Insights

The path-decomposition unification (Appendix A.2) reveals that deep local GNNs, infinite-step random walks, and the proposed forest layer all admit a common formulation with different path-weighting schemes: local methods weight paths by local environment (degrees, densities), while forests weight paths by global transport importance (how many spanning trees contain that path). This provides a principled framework for understanding why forests naturally capture long-range structure—they privilege paths that are essential for global connectivity rather than those incident to high-degree nodes. The distinction between "local environmental importance" and "global transport importance" as competing path-weighting philosophies offers a novel lens for analyzing graph learning architectures beyond the local/global dichotomy.

## Suggestions

1. **Add a clear ablation of graph augmentation**: Include a baseline in Table 3 that uses FGL on the unaugmented original graph. This directly addresses the concern that performance gains may stem from the label-informed edge additions rather than the forest mechanism.

2. **Include heterophily-specific baselines in the main comparison table**: Move results from Appendix J.9 (ADPA, GESN, HiGNN) into Table 1 or create a dedicated Table 1b for heterophilous datasets to provide complete context for readers.

3. **Report total wall-clock time including pre-processing**: The efficiency narrative focuses on per-epoch training time but omits the cost of pseudo-label generation, attention estimator training, and tree sampling. A breakdown or total time comparison would make efficiency claims more transparent.

4. **Address the 0.00 standard deviations**: For Texas and Wisconsin, Table 10 reports exactly 0.00 variance, which is implausible. Either report the true variance or explain if this reflects a deterministic setting.

5. **Clarify inductive limitations**: Add a brief discussion of transductive assumptions and potential pathways to inductive learning, as this affects practical deployability for real-world graph applications.