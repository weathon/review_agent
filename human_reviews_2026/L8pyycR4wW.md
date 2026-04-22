# Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Fairness has become an arising concern in machine learning with its prevalence in decision-making processes, and the trade-offs between various fairness notions and between fairness and accuracy has been empirically observed. However, the inheritance of such trade-offs, as well as the quantification of the best achievable trade-offs, i.e., the Pareto optimal trade-offs, under varied constraints on fairness notions has been rarely and improperly discussed. Owing to the sub-optimality of fairness interventions, existing work fails to provide informative characterization regarding these trade-offs. In light of existing limitations, in this work, we propose a reformulation of the model-specific (MS) Pareto optimal trade-off, where we frame it as convex optimization problems involving fairness notions and accuracy w.r.t. the confusion vector. Our formulation provides an efficient approximation of the best achievable accuracy under dynamic fairness constraints, and yields systematical analysis regarding the fairness-accuracy trade-off. Going beyond the discussion on fairness-accuracy trade-offs, we extend the discussion to the trade-off between fairness notions, which characterizes the impact of accuracy on the compatibility between fairness notions. Inspired by our reformulation, we propose a last-layer retraining framework with group-dependent bias, and we prove theoretically the superiority of our method over existing baselines. Experimental results demonstrate the effectiveness of our method in achieving better fairness-accuracy trade-off, and that our MS Pareto frontiers sufficiently quantify the two trade-offs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reformulates model-specific Pareto trade-offs for fairness–accuracy and DP–EOd via convex optimization over the confusion vector, and proposes a group-dependent last-layer retraining method with theoretical guarantees and empirical gains.

### Strengths
Clear statement and intuitive novelty;
Multiple datasets evaluated;

### Weaknesses
The problem doesn’t have a single fix. The right mix of fairness and accuracy changes by case. Even on the Pareto front, the final pick depends on the use case, so it isn’t a standalone answer.

Standard fairness contraints already let you tune the balance between fairness and accuracy. So even if this study shows a better way to optimse them together, theoratically, I don‘t think it will change much in what we got after solving the problem.

People have talked about accuracy vs fairness a lot in past years. I don’t think this is the main gap now; some newer work even tries to separate the two.

### Questions
see above

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
3

### Summary
The paper studies Pareto-optimal trade-offs in the fair classification task. It argues that prior model-specific (MS) Pareto frontiers can be suboptimal and may not reflect the true best-achievable trade-offs between accuracy and fairness, or between different fairness notions. To address this challenge, this paper reformulates MS Pareto frontiers as low-dimensional convex programs in terms of a confusion-vector parameterization, extends this to trade-offs between demographic parity and equalized odds under accuracy constraints, and proposes a last-layer retraining framework that introduces group-dependent bias terms. The theoretical comparisons and empirical results on multiple datasets validate the proposed method.

### Strengths
1. The problem formulation is clear. The formation from accuracy/fairness metrics to a linear form of a confusion matrix is clean and leads to tractable convex optimization.
2. The proposed research extends fairness-accuracy trade-off to a general fairness-fairness analysis with accuracy constraints.
3. The theoretical results and experimental results validate the proposed methods.

### Weaknesses
1. It would be promising to discuss more fairness metrics beyond EOd and DP.

### Questions
1. For Assumption 1: it is likely that the ROC curves of a classifier is concave. But it might be common that all group-dependent ROC curves are all concave. Even though there is a empirical validation, I wonder the explicitly conditions behind Assumption 1. Is it possible that if the majority of the group-dependent ROC curves are concave, then Lemma 1 is correct?

### Soundness
3

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
3

### Summary
This paper addresses the fundamental challenge of quantifying and optimizing trade-offs in machine learning fairness. While prior studies have observed empirical trade-offs between fairness and accuracy—and among different fairness notions—they lack a principled characterization of the best achievable (Pareto optimal) trade-offs. To fill this gap, the authors reformulate the model-specific Pareto optimal trade-off as a convex optimization problem based on the confusion vector, enabling efficient approximation of optimal accuracy under dynamic fairness constraints. Their framework also extends analysis to the trade-off between fairness notions, revealing how accuracy influences their compatibility. Building on this formulation, they propose a last-layer retraining method with group-dependent bias and provide theoretical guarantees of its superiority. Experiments validate that the proposed approach achieves improved fairness-accuracy balance and effectively quantifies both types of trade-offs.

### Strengths
1. The proposed low-dimensional convex program over confusion vectors is both efficient and interpretable, providing a principled way to estimate model-specific Pareto frontiers.
2. The last-layer retraining approach with group-dependent biases is easy to implement and consistently outperforms post-processing and other last-layer baselines in the experiments.
3. The paper includes solid theoretical results—such as proofs of suboptimality for prior frontiers, superiority of the proposed retraining method over random flipping, and an analytic formula for accuracy–fairness trade-offs—that reinforce the empirical findings.

### Weaknesses
1. Strong modeling assumptions: The theoretical results rely on restrictive assumptions (e.g., concave ROC curves and equal-variance Gaussian logits), which may not always hold in practice. The robustness of the approach under violations of these assumptions could be further investigated.
2. Dependence on $z_b$ / $z_{ab}$ estimation: The convex program depends on reliable upper-bound estimates obtained from multiple training restarts. The paper does not sufficiently analyze how sensitive the resulting frontier is to these estimates.
3. Notation overload: The manuscript uses a large number of mathematical symbols, which can make it difficult to follow. Including a concise table summarizing all symbols and variables would significantly improve readability.
4. Experimental detail and reproducibility: Some implementation details—such as the number of random initializations for $f$, the sampling strategy for the $\epsilon$-grid, and runtime or computational cost—are missing or only briefly mentioned in the appendix. Providing these would enhance transparency and reproducibility.

### Questions
1. How sensitive is the computed MS Pareto frontier to the estimation of $z_b$ and $z_{ab}$? Could suboptimal $z_b$ estimates (e.g., from local minima) substantially loosen the frontier? Please report the variance across multiple $z_b$ estimates.
2. For datasets or models where ROC curves are not concave, how often does Lemma 1 fail to hold? Are there diagnostic tools to detect when post-processing frontiers may already be tight?
3. The Gaussian equal-variance logit assumption in Theorem 2 is convenient but potentially unrealistic. Can you provide empirical evidence (on your datasets) showing that subgroup logit variances are comparable, or quantify how deviations affect the accuracy–drop bound?
4. What is the computational cost of the proposed approach—particularly solving the convex program for multiple $\epsilon$ samples and $z_b$ initializations? For example, how long does the process take in the ResNet-50 CelebA experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a theoretically grounded and empirically validated framework for characterizing the Pareto-optimal trade-offs in fair classification, specifically focusing on the fairness-accuracy and fairness-fairness (e.g., DP vs. EOd) trade-offs. The authors identify and rigorously address the suboptimality of existing post-processing methods (e.g., FACT, G-STAR) in approximating the true Model-Specific (MS) Pareto frontier. Their core contribution is a reformulation of the problem as a convex optimization over the confusion vector, which bypasses the limitations of specific intervention algorithms. Building on this, they propose a novel last-layer retraining method with group-dependent bias and provide strong theoretical guarantees for its superiority over existing baselines. The experimental results on multiple datasets are comprehensive and convincingly support the claims.

### Strengths
1. The paper correctly identifies a key flaw in prior work: the use of sub-optimal fairness interventions (like post-processing with random flipping) leads to an inaccurate and pessimistic approximation of the MS Pareto frontier. The proposed reformulation using confusion vectors and convex optimization is elegant, efficient, and provides a more meaningful upper bound on achievable performance.
2. The theoretical contributions are substantial, and the empirical Validation is rich.

### Weaknesses
1. The entire theoretical framework in Section 4.1 relies on the assumption that group-dependent ROC curves are concave. While the authors justify this by stating that optimal classifiers are expected to have concave ROC curves, this may not hold perfectly in practice for complex, deep neural networks, especially on real-world data with noise. The paper would be strengthened by a more detailed discussion of the implications when this assumption is violated and perhaps an empirical analysis of the concavity of the ROC curves in their experiments. 
2. The convex optimization for the frontier is performed over a 4-dimensional confusion vector for binary classification. The extension to multi-class (Section 13, Appendix) increases the dimensionality significantly (k classes × 2 groups). The computational cost and feasibility for large k should be briefly discussed. Is the optimization still tractable for, say, 100 classes?
3. In the multi-class extension (Equation 8), the definitions of A_EOd,k' and A_DP,k' could be explained more clearly. The current notation is somewhat dense and could benefit from a sentence or two of intuitive explanation regarding how these matrices capture the worst-case disparity across all classes.

### Questions
Q1: In Lemma 1 and the corresponding proof, you assume concave ROC curves. Could you provide empirical evidence in the appendix (e.g., a plot) showing the ROC curves for your models on one of the datasets to validate that they are indeed approximately concave?
Q2: The hyperparameter λ in your retraining framework (Eq. 6,7) is crucial. Could you provide more details on the sensitivity analysis? How robust is the performance to the choice of λ?
Q3: In the multi-class experiments on the Drug dataset (Table 11, 12), why are there no baselines for the DP-accuracy trade-off other than your own method? Is it because the cited works (DFR, SELF) do not target DP?

### Soundness
3

### Presentation
2

### Contribution
3
