# IRGCL： Information Refinement Graph Contrastive Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Graph contrastive learning (GCL) has emerged as a leading paradigm in unsuper-
vised graph representation learning (UGRL), yet existing contrastive approaches
remain vulnerable to three persistent challenges: noisy features that distort simi-
larity measures, unreliable structures that contain spurious edges, and degree im-
balance that biases representation quality. We propose Information-Refinement
Graph Contrastive Learning (IRGCL), a single-view contrastive learning frame-
work that simultaneously addresses these challenges and effectively generalizes
across key graph learning tasks, including node classification, clustering, and link
prediction. IRGCL integrates three complementary components: (i) structure-
consistent feature selection to filter out redundant or noisy attributes; (ii) high-
confidence structure learning to refine graph neighborhoods; and (iii) degree-
aware focal contrastive learning to balance learning across low- and high-degree
nodes. Extensive experiments on diverse benchmarks demonstrate that IRGCL
consistently outperforms state-of-the-art baselines, and ablation studies confirm
the distinct and complementary benefits of each component, highlighting the ne-
cessity of jointly addressing feature quality, structural reliability, and degree im-
balance. Code is available at https://anonymous.4open.science/r/IRGCL-01F8.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes IRGCL, a single-view graph contrastive learning method designed to simultaneously address three common challenges in graph contrastive learning: feature noise, structural unreliability, and node degree distribution imbalance. The method integrates three synergistic modules, low-rank feature selection, confidence-guided clustering-based structure learning, and degree-aware focal contrastive loss. And IRGCL validates its performance through experiments on multiple graph learning tasks including node classification, node clustering, and link prediction.

### Strengths
Strength 1: The framework establishes a comprehensive architecture that jointly addresses the three key issues in GCL, feature quality, structural reliability, and degree bias, with a clear and well-structured design.

Strength 2: Theoretically proves that the structure learning module monotonically decreases Dirichlet energy and enhances homophily, providing a mathematical foundation for this component.

### Weaknesses
Weakness 1: In Module 1 (Feature Selection), the optimization requires alternating updates between the W and F matrices. This strategy cannot guarantee convergence to a global optimum and is highly likely to settle in a local optimum, making the final feature selection quality strongly dependent on initialization. Moreover, updating W involves matrix inversion, which leads to high computational complexity for high-dimensional features. Ablation studies indicate that the performance gain is relatively small compared to the substantial computational cost.

Weakness 2: In implementation, Module 1 (Feature Selection) operates as an independent "preprocessing" step and lacks tight, end-to-end integration with the subsequent GNN encoder and contrastive learning objective. The feature selection criterion, based on reconstruction error and smoothness, may not align with the final task objective, such as discriminability in node classification.

Weakness 3: Module 3 (Contrastive Learning with Neighborhood & Degree Awareness) has limitations in positive and negative sample selection for low-degree nodes. The core issue for low-degree nodes is the severe shortage of positive samples. Using only one-hop neighbors as positives lacks semantic richness, and since low-degree nodes constitute the majority of the dataset, the sampling process may result in significant loss of valuable information.

Error 1: There is an inaccuracy in the experimental results for the Computers dataset in Table 6.

### Questions
See Weaknesses

### Soundness
3

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
4

### Summary
This paper proposes a principled framework to improve the unsupervised graph contrastive learning (GCL) by replacing noisy random augmentations with structured information refinement. It introduces a three-stage approach: (1) Feature Selection via Low-Rank Approximation, which removes redundant or uninformative node features while preserving structural smoothness; (2) Structure Learning via High-Confidence Clustering, which edits graph edges based on cluster consistency, theoretically guaranteeing monotonically decreasing Dirichlet energy and increasing homophily; and (3) Contrastive Learning with Degree Awareness, which employs a degree-adaptive focal loss and JSD contrastive objective to handle unbalanced neighborhoods. Extensive experiments across benchmarks show that IRGCL achieves state-of-the-art results with strong robustness and interpretability.

### Strengths
1. The proposed low-rank feature selection effectively avoids the randomness in masking or perturbations, providing a stable approach to view enhancement.
2. The adaptive focal weighting mechanism provides an effective solution to the problem of degree imbalance in graph structures.
3. Extensive experiments demonstrate the good performance of IRGCL across diverse datasets.

### Weaknesses
1. The three refinement modules lack a clearly defined systemic coupling. The overall framework appears to be a combination of several relatively independent components. Providing an unified architecture or theoretical framework would be better.
2. The update process of $W$ lacks theoretical guarantees of convergence, which introduces uncertainty regarding the stability of optimization.
3. The method involves a large number of hyperparameters and exhibits relatively high computational complexity, creating a gap with the authors' claim of being "without fussy augmentations."
4. The model mainly relies on local neighborhood sampling and does not explicitly capture global semantics or long-range dependencies.
5. The clustering procedure is relatively complex and computationally demanding, which limits the applicability to relatively large-scale graphs.

### Questions
1. The method involves a large number of hyperparameters. How are they balanced across different modules, and how is stable convergence ensured in updating $W$?
2. How efficient is the proposed method in practice, especially when applied to large-scale graphs in terms of time and memory complexity?
3. Since most modules are designed to enhance smoothness, how does the model prevent over-smoothing and maintain discriminative representations under this design?

### Soundness
2

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
5

### Summary
The authors claim that existing GCLs face three major challenges: noisy feature, unreliable structures, degree imbalance. The proposed IRGCL method tackles these challenges by combining structure-aware feature selection, high-confidence structure learning and degree-aware focal contrast. The authors claim that IRGCL outperforms existing methods.

### Strengths
1. This paper is driven by a clear motivation, which is directly addressed by the proposed method.
2. This paper has a good theoretical analysis.
3. The figures are well presented.

### Weaknesses
1. The term "spurious edges" is mentioned in the motivation but is not explicitly defined. It remains unclear what specific types of edges fall into this category.
2. There is an inconsistency in Section 3.1 regarding the rank of matrix W. It is initially constrained to r (rank(W) = r), but later referred to as k when discussing the number of retained features (rank(W) = k). This ambiguity needs to be resolved.
3. Section 3.2 relies on node silhouettes to identify high-confidence clusters, yet it omits the definition and calculation of this crucial metric.
4. The "degree safeguard" mentioned in Section 3.2 is a critical component for maintaining graph connectivity, but its implementation details are not provided. It is unclear how this mechanism operates to prevent nodes from becoming isolated.
5. The compared baselines are somewhat outdated, and there is a lack of comparison with SOTA approaches. 
6. The improvements in the experiments are marginal. 
7. A fundamental weakness of this work is its lack of novel contributions, as it seems to largely repackage ideas from prior works.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limitations of graph contrastive learning (GCL) under noisy features, unreliable graph structures, and degree imbalance. It proposes IRGCL, a single-view contrastive framework that integrates (i) Laplacian-regularized low-rank feature selection, (ii) confidence-guided clustering for structural refinement, and (iii) degree-aware focal JSD loss for balanced contrastive learning. Experiments on transductive, inductive, and clustering benchmarks demonstrate that IRGCL consistently outperforms other methods.

### Strengths
- The framework is clear, and the method flow is easy to follow.

- The experiments are comprehensive, covering node classification, clustering, and link prediction tasks.

- The codes are provided, enhancing reproducibility.

### Weaknesses
- The motivation is somewhat ambiguous. Feature quality and degree imbalance are intrinsic properties of graph topology (e.g., power-law distributions), rather than specific flaws of contrastive learning. To justify their inclusion, the authors should clarify how these aspects particularly affect the contrastive objective.

- Among the three stated motivations, graph reliability is mentioned but not clearly defined, especially regarding which type of "neighborhood-based contrast" is being referred to. Moreover, the explanation around Lines 58–63 should be elaborated in future revisions.

- Experimental precision is inconsistent (mixture of one- and two-decimal reporting), which slightly reduces the rigor of the empirical section.

- The baselines in Table 4 differ from those in Table 2 and lack GCL methods (e.g., PolyGCL), making it difficult to fully assess comparative effectiveness.

### Questions
- In Figure 4, when the ratio = 0, the results on Cora and CiteSeer also appear to perform good. What accounts for this improvement?

- In Tables 5 and 6, why do the accuracies on Photo and Computers remain higher than most results in Table 2 even after removing other components? Moreover, in Table 6, what causes the Acc on Computers to surge to 94.50%, significantly surpassing all baselines?

### Soundness
2

### Presentation
3

### Contribution
2
