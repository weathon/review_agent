## Summary

This paper proposes a Covariance-Adjusted Support Vector Machine (CSVM) that accounts for class-specific covariance structure in classification. The authors argue that standard SVM's max-margin principle and KKT conditions are valid only under Euclidean geometry, and that the input space (equipped with a Mahalanobis metric) is "Non-Euclidean," requiring transformation via class-specific Cholesky decomposition before classification. An iterative "SM Algorithm" is proposed to estimate population covariance from training data by pseudo-labeling test points and updating covariance estimates until convergence. Empirical results on five binary classification datasets show improvements over standard SVM kernels and global whitening methods (PCA/ZCA).

## Strengths

- **Principled motivation for class-conditional whitening**: The paper provides a coherent (if imprecisely stated) geometric argument for why class-specific covariance adjustment matters for SVM margins, moving beyond ad hoc preprocessing by deriving how the margin in input space becomes a function of intra-class covariance (Eq. 9, 14). This yields the concrete observation that the decision boundary should split the margin space in proportion to class covariances—a testable claim with clear operational implications.

- **SM Algorithm as an iterative estimation procedure**: The proposed algorithm (Section 3) attempts to close the gap between sample and population covariance by iteratively refining pseudo-labels on test data and re-estimating covariance. While it has significant limitations (detailed below), it poses an interesting semi-supervised question: can the structure of test data itself improve covariance estimation? This is a genuinely different angle from static preprocessing.

- **Consistent empirical improvements**: Across five diverse datasets (healthcare, astronomy, quality control, safety), CSVM achieves the highest accuracy and F1 on four out of five datasets, and the highest AUC on three. The consistency of improvement—rather than gains on a single dataset—suggests the method is capturing something real, even if the magnitude and statistical significance remain uncertain.

## Weaknesses

### Major:

- **The SM Algorithm uses test data to update model parameters, making the method transductive and the comparison with inductive baselines unfair.** Step 3(g)–(h) of the SM Algorithm explicitly adds pseudo-labeled test points to the training set and recalculates covariance matrices from the updated data. This means the model parameters (covariance estimates and consequently the decision boundary) depend on the test batch. Standard SVM kernels and PCA/ZCA whitening baselines are purely inductive—they never observe test data. The performance gains may therefore stem from access to the test distribution's marginal structure rather than from the proposed covariance-adjusted geometry. Without an inductive variant of the method (e.g., class-conditional whitening using only training data) as a baseline, the source of improvement cannot be isolated. This is the most critical experimental flaw.

- **The theoretical framework contains imprecise and potentially misleading claims about "Non-Euclidean" spaces and "invalid" KKT conditions.** The input space for standard tabular data is ℝⁿ, which is Euclidean by definition. Using a Mahalanobis metric changes the *metric tensor* but not the underlying vector space; the resulting space is isometric to Euclidean space via a linear transformation—which is precisely what Cholesky decomposition provides. Calling this "Non-Euclidean" conflates a change of metric with a change of topology or vector space structure. Similarly, Lemma 2.3 claims "KKT boundary conditions are not valid" in the input space, but KKT conditions apply to *any* differentiable convex optimization problem. If the objective changes to include Σ, the KKT conditions for that *new* problem involve Σ; this does not render the original KKT conditions "invalid"—it simply means a different optimization problem is being solved. These imprecise claims weaken the theoretical contribution and may mislead readers about the nature of the result.

- **Inconsistency between Lemma 2.2 (two classifiers) and the algorithm's single-classifier inference.** Lemma 2.2 states that a binary problem yields "two unique linear classifiers" in the input space, arising from the two different optimization problems (Eqs. 10–11 vs. 12–13) with different covariance matrices Σ_{y=1} and Σ_{y=-1}. Yet the SM Algorithm (Step 3d–e) produces a single classifier θ_Input^T x + θ₀' = 0 by adjusting only the bias of a standard input-space SVM. The paper does not resolve how the two distinct optimization problems from Lemma 2.2 collapse to a single decision function, nor how the two different margin ratios (one per class perspective) are reconciled into one θ₀' adjustment.

### Minor:

- **No convergence guarantees or stability analysis for the SM Algorithm.** The algorithm iteratively pseudo-labels test data and updates covariance estimates. No proof or empirical analysis is provided showing that this process converges to a stable fixed point, nor is the risk of error propagation analyzed (where early misclassifications corrupt the covariance estimate and subsequent iterations). The convergence criterion ("changes in test data labels are below a certain threshold") is vague, and no sensitivity analysis to this threshold is provided.

- **The theoretical derivation assumes hard-margin SVM (ξᵢ = 0), but the evaluated datasets are not linearly separable.** The paper explicitly sets ξᵢ = 0 in Section 2, deriving the margin and optimization problem for the separable case. However, datasets like Diabetes and Red Wine are unlikely to be linearly separable even after whitening. The paper does not derive or discuss the soft-margin extension, creating a gap between theory and practice.

- **No statistical significance testing for the reported performance improvements.** The tables report point estimates from a single 80/20 split. Some improvements are marginal (e.g., OSHA Accuracy: 0.752 vs. 0.741; Red Wine F1: 0.743 vs. 0.737). Without standard deviations, confidence intervals, or significance tests across multiple runs, it is unclear whether these gains are real or attributable to split variance.

- **Missing comparison with closely related covariance-adjusted baselines.** The paper cites MCVSVM (Zafeiriou et al., 2007), maxi-min margin machine (Huang et al., 2004), and weighted Mahalanobis kernels (Wang et al., 2007), all of which address similar problems of incorporating covariance into SVM. No experimental comparison with these methods is provided. Similarly, QDA—which also performs class-conditional covariance-based classification—is not included. Without these comparisons, it is unclear whether CSVM offers advantages over existing covariance-adjusted approaches.

### Trivial:

- **No runtime comparison provided.** The paper acknowledges higher computational complexity due to Cholesky decomposition and iterative SVM solving, but provides no empirical wall-clock comparison. This is a minor omission since the complexity difference is straightforward to reason about, but practitioners would benefit from knowing the scale of overhead.

## Nice-to-Haves

- **Class-conditional whitening + SVM (inductive, no test data in loop) baseline.** This would cleanly isolate the contribution of the iterative SM refinement from the contribution of class-conditional whitening itself, and would provide a fair inductive comparison point.

- **Ablation study on SM Algorithm components.** How much of the performance gain comes from class-conditional vs. global whitening? How much from the iterative refinement vs. a single-pass approach? How sensitive is the method to the convergence threshold?

- **Visualization on 2D synthetic data** with known covariance disparities, showing the decision boundary splitting the margin in the ratio of class covariance as claimed in Eq. 14. This would be a compelling proof-of-concept that directly validates the core theoretical claim.

- **Comparison with deep metric learning or modern Mahalanobis distance learning methods** to situate the work relative to current standards.

- **Soft-margin derivation** extending the theoretical framework to the non-separable case with slack variables.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Reliance on Sahoo & Maiti (2025) requires more rigorous geometric justification"** (Harsh Critic): Per the hard rules, cited references are assumed to exist and be valid. The criticism of insufficient geometric justification is already captured in the "Non-Euclidean" terminology weakness above; the specific targeting of this citation is removed.

- **"ROC curves without confidence bands"** (Harsh Critic): This is a generic demand for confidence bands on ROC curves, which is not standard practice in the ML community for small-scale evaluations. Moved to trivial/removed.

- **"Comparing with deep metric learning baselines"** (Spark Finder): This is scope creep. The paper operates within classical SVM theory; demanding modern deep baselines goes beyond the paper's stated scope. Moved to nice-to-have.

- **"The paper is well-structured"** (Balanced Reviewer): Generic strength that applies to many papers. Weakened per soft rules.

- **"Comprehensive experimental datasets"** (Balanced Reviewer): Five datasets without significance testing or ablations is adequate but not exceptional. Weakened per soft rules.

## Novel Insights

The most interesting observation emerging from the synthesis of these reviews is that the paper's core contribution may be better understood as an *implicit derivation that class-conditional whitening is the correct metric transformation for SVM*, rather than a discovery about "Non-Euclidean" geometry. The margin-ratio result (Eq. 14)—showing that the decision boundary should split the margin in proportion to θᵀΣ⁻¹θ for each class—is a concrete, testable consequence of this transformation, and it provides a normative answer to *how much* the boundary should shift when classes have unequal dispersion. This is a genuine insight that could be cleanly separated from the problematic "Non-Euclidean" framing and the transductive evaluation issue. If the authors reformulated the contribution as "class-conditional whitening is the metric-correct transformation for SVM, and it implies a specific margin-ratio adjustment," the paper would have a cleaner theoretical story—though the experimental methodology would still need to be corrected.

## Suggestions

1. **Add an inductive baseline**: Implement class-conditional Cholesky whitening using only training data (no iterative test-data inclusion), then apply standard SVM. Compare this against the full SM Algorithm. This single experiment would clarify whether the gains come from the geometry or from transductive access to test data.

2. **Soften and clarify the theoretical claims**: Replace "Non-Euclidean space" with "space equipped with a Mahalanobis metric" or "non-isotropic feature space." Replace "KKT conditions are invalid" with "the standard SVM margin objective is metric-dependent, and under a Mahalanobis metric the margin splits non-equally." These revised claims are true and non-trivial without being overstatements.

3. **Resolve the Lemma 2.2 inconsistency**: Either show that the two optimization problems (Eqs. 10–13) yield the same decision boundary in the transformed space (which would undermine the "two classifiers" claim), or provide a clear inference rule for combining the two classifiers in input space.

4. **Provide convergence analysis or empirical evidence**: At minimum, plot label stability across SM Algorithm iterations for each dataset, and report how many iterations are typically needed. Discuss failure modes when the initial pseudo-labels are substantially wrong.

5. **Extend to soft-margin**: Derive the slack-variable formulation explicitly, since all real-world applications require it and the hard-margin assumption is a gap between theory and practice.