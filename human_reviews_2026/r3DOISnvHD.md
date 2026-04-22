# Are we measuring oversmoothing in graph neural networks correctly?

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Oversmoothing is a fundamental challenge in graph neural networks (GNNs): as the number of layers increases, node embeddings become increasingly similar, and model performance drops sharply. Traditionally, oversmoothing has been quantified using metrics that measure the similarity of neighbouring node features, such as the Dirichlet energy. We argue that these metrics have critical limitations and fail to reliably capture oversmoothing in realistic scenarios. For instance, they provide meaningful insights only for very deep networks, while typical GNNs show a performance drop already with as few as 10 layers.  As an alternative, we propose measuring oversmoothing by examining the numerical or effective rank of the feature representations. We provide extensive numerical evaluation across diverse graph architectures and datasets to show that rank-based metrics consistently capture oversmoothing, whereas energy-based metrics often fail. Notably, we reveal that drops in the rank align closely with performance degradation, even in scenarios where energy metrics remain unchanged. Along with the experimental evaluation, we provide theoretical support for this approach, clarifying why Dirichlet-like measures may fail to capture performance drop and proving that the numerical rank of feature representations collapses to one for a broad family of GNN architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates widely used measures of over-smoothing in graph neural networks such as Dirichlet energy and investigates alternative metrics based on the rank of the feature matrix and argues that rank based measures are better for measuring oversmoothing. The paper reviews existing results on oversmoothign and extends them to rank based measures. Finally, the paper experiments on trained GNNs and shows that the rank based measures correlates better with performance on many datasets.

### Strengths
1. **Significance of problem.** Developing improved tools for understanding the internal mechanisms of GNNs is an important and timely direction.

2. **Insightful discussion of limitations.** The paper provides a thorough critique of standard oversmoothing metrics, clearly articulating where existing measures fall short.

3. **Potential of rank-based metrics.** The proposed rank-based metrics offer a promising way to capture more nuanced behavior than conventional oversmoothing measures, potentially enabling a more fine-grained understanding of representation collapse in GNNs.

### Weaknesses
1. **Lack of relevance** Over-smoothing as a problem has been widely studied, and remedies have been found, the simplest of which is adding a residual connection. This paper proposes a new method to measure a phenomenon that is no longer a problem in practice which weakens its relevance today. This is further shown in Table 4 where little correlation exists between performance and the proposed measures once residual connections are added.
2. **Lack of novelty** A large portion of the paper discusses already established results rather than focusing on what new insight rank based measures can offer.

### Questions
1. What does Section 3.2 offer that prior work does not?
2. For Theorem 5.3 could you maybe expand on “Note that this is implied by the relative metrics [like normalized Dirichlet going to 0], but is actually quite weaker ”. Does the theorem lead to a concrete example where the rank is 1 but the normalized Dirichlet energy or E_proj is not zero? This might help the reader understand in what ways this measure offers a more nuanced perspective.
2. Table1: What is the sample size used to compute each correlation? Can you report confidence intervals (or p-values) for the correlation coefficients?
3. Table 1: In many cases standard measures have a strong negative correlation with performance, sometimes even surpassing some of the rank based rank-based measures in absolute correlation. How do you explain this?

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
3

### Summary
The paper argues that existing energy-based oversmoothing metrics for GNNs (Dirichlet energy and a projection variant) are unreliable in trained networks, and proposes rank-based measures (numerical rank and effective rank) as better metrics. The authors present (i) a unifying framework based on eigenvectors of nonlinear activations, (ii) a linear analysis predicting dominance of the leading eigenspace, and (iii) a nonlinear analysis using the Hilbert projective metric that yields rank collapse under nonnegativity and subhomogeneity conditions. The authors conduct experiments on GCNs and GATs across multiple datasets and small depths demonstrate that rank-based measures correlate well with model accuracy, whereas energy-based metrics often remain insensitive to oversmoothing in some realistic training settings.

### Strengths
The paper is overall clearly written. It clearly points out the failure modes of energy metrics. For example, it shows these metrics are informative mainly in the limit, can be confounded by scaling, and rely on a fixed dominant eigenspace, which often violated by trained GNNs. It also provides a unifying theoretical perspective for previous results on the oversmoothing phenomena measured by Dirichlet-like energies. The nonlinear case with Hilbert-metric argument are technically interesting.

### Weaknesses
Despite strong empirical results, the paper lacks a theoretical justification that rank‑based metrics dominate Dirichlet‑style energy measures as oversmoothing indicators in trained GNNs. The analysis shows when energies vanish and that ranks can collapse, but offers no formal comparison, dominance result, or sensitivity guarantee for rank.

### Questions
Could the authors explain why effective/numerical rank aligns with accuracy more faithfully than Dirichlet‑style energies in trained networks? In the linear setting, at first glance, Thm. 4.1 and Thm. 5.1 suggest that both energy‑like deviations and rank collapse shrink comparably. How do you reconcile this with the empirical gap you observe? Could you provide intuition or formal statement showing a sensitivity advantage for rank metrics?

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
2

### Summary
This paper critiques traditional oversmoothing metrics in graph neural networks (GNNs), such as Dirichlet energy, for their limitations in detecting performance degradation in realistic, trained models. The authors argue these metrics only work asymptotically in deep, untrained networks and propose rank-based alternatives (numerical and effective rank) that measure feature matrix collapse to low rank.

### Strengths
The paper's novel critique of widely used metrics is timely and well-supported, exposing flaws like scale-dependence and eigenspace reliance with examples and simplified theorems. Theoretical contributions, including proofs of rank convergence independent of feature magnitude, extend to GCNs/GATs and offer a unifying eigenvector perspective, and some nonlinear analysis. Empirically, the extensive evaluation across homophilic/heterophilic datasets, depths, activations, and ablations robustly demonstrates rank metrics' superiority, with high correlations and methodological rigor (e.g., log-transformed Pearson analysis). This could practically guide deeper GNN designs.

### Weaknesses
The theoretical analysis is restricted to linear/nonnegative models or shared eigenvectors, limiting generalizability to signed graphs or complex activations. Experiments focus solely on node classification, neglecting graph/edge-level tasks or larger/dynamic graphs, and don't prove causation between rank decay and performance. While rank metrics excel in trained settings, the paper somewhat overlooks scenarios where energy metrics succeed (e.g., untrained asymptotics) and lacks discussion of computational costs or hybrid approaches.

### Questions
None

### Soundness
3

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
This paper argues that dirichlet energy based measures may fail to capture oversmoothing in real application scenarios, so the author introduces rank (numerical or effective rank) as an effective alternative measurement of oversmoothing. The main contributions of the paper are: 1. The authors provide novel rank relaxation functions to measure oversmoothing. 2. The authors compare current oversmoothing metrics with their relaxed rank metrics and demonstrate rank metrics correlate better with performance, 3. The authors theoretically justify the aggregation matrices and nonlinear activation functions will both contribute to the decay of rank, whereas energy-based measures fail to detect oversmoothing. Empirical results shows that the performance drop from GNN is well-correlated with the decrease of rank metric based on comparisons with different energy and rank metrics.

### Strengths
1. This paper provides a strong theoretical proof of the decay of the rank as a measurement of oversmoothing.  
2. Extensive experiments and ablation studies show the rank-based metrics correlate better with performance degradation than energy-based measurement with extension to modern architectures with dropout, residuals, or normalization layers.  
3. Unlike Dirichlet energy, which only shows meaningful trends for very deep networks, the proposed rank metrics can detect degradation even in moderately deep networks (10 or fewer layers).

### Weaknesses
1. Research of related work part is not sufficient as many previous works have already shown energy-based metric is not a proper or sufficient measurement of oversmoothing. For example, [1] has shown that oversmoothing can be mitigated without explicitly dirichlet energy based control.
2. The paper shows rank metrics are effective measurement of oversmoothing but does not provide practical techniques to mitigate it using the rank insight.


[1] Y. Jin and X. Zhu, "Graph Rhythm Network: Beyond Energy Modeling for Deep Graph Neural Networks," 2024 IEEE International Conference on Data Mining (ICDM), Abu Dhabi, United Arab Emirates, 2024, pp. 723-728, doi: 10.1109/ICDM59182.2024.00083.

### Questions
What are certain techniques that can mitigate oversmoothing from the view of rank since the authors show ranks metrics are more effective than energy-based metrics. Also, could the rank metric be leveraged as a regulation metric to prevent oversmoothing, or to achieve better performance?

### Soundness
3

### Presentation
3

### Contribution
2
