## Summary

This paper proposes a unified MIQP-based clustering framework for building interpretable global surrogate models of Gaussian Process (GP) posteriors. The core idea is to assign new data points to clusters whose members share a common parameter, with cluster assignments optimized by maximizing the log-density of the variational GP posterior. Graph partitioning (for spatial data) and decision tree learning are derived as constrained special cases of this formulation. Experiments on the California Housing dataset and three regression benchmarks demonstrate lower weighted-RMSE compared to k-means and CART, respectively.

---

## Strengths

- **Principled probabilistic objective.** The surrogate loss in Eq. (4)–(5) is derived from the variational GP posterior log-density rather than an ad-hoc distance metric. The inverse-covariance weighting means predictions in high-uncertainty regions contribute less to the objective, which is a non-trivial and principled design choice that distinguishes the method from simple tree distillation onto GP mean outputs.

- **Unifying formulation.** Theorem 4.2 and 4.3 formally show that both spatially-connected graph partitioning and axis-aligned decision trees are representable as linear constraint sets atop the same MIQP core. The DAG-based connectivity encoding for graph partitioning (Theorem 4.3) is a non-obvious technical bridge that makes this unification possible.

- **Honest marginalisation analysis.** Section 4.1 explicitly derives the marginalised objective (Eq. 7), identifies the log-determinant complexity penalty that should in principle discourage over-clustering, and transparently explains why it is dropped for computational reasons. This level of candour about the approximation is valuable.

- **Acknowledgment of scalability limits.** The paper explicitly reports that the MIQP solver failed to prove optimality within hours, and that graph partitioning fails for l = 8 within the time limit. This honesty is appreciated, though it simultaneously highlights an important unresolved challenge.

---

## Weaknesses

### Fatal
None identified. The core formulation is mathematically coherent, but the experimental validation falls well short of substantiating the claims.

### Major

- **Only CART as a tree baseline.** The related work section cites a rich set of optimal tree methods (Bertsimas & Dunn 2017, MurTree, GOSDT, BinOCT, etc.) but the experiments compare only against greedy CART. For a paper whose explicit contribution is an MIQP-based optimal regression tree that outperforms CART because CART is a heuristic, the relevant comparison is against other MIQP/exact optimal tree solvers. Without this, it is impossible to assess whether the GP-weighted objective adds value over simply applying an existing optimal tree solver to the GP posterior mean.

- **No out-of-sample evaluation.** For the graph partitioning experiment the paper explicitly states "new inputs were identical to the inputs used for training," meaning all reported loss values are in-sample. The stated motivation in the introduction—that surrogates are useful when test distributions differ from training distributions—is never tested. For the decision tree experiments, 10-fold cross-validation is used, but it is not clearly stated whether the GP is refitted in each fold, whether CART is also trained as a surrogate to GP posterior means (rather than to original labels), or whether the 10% surrogate-building split is re-drawn per fold. Without this precision the evaluation cannot be reproduced or trusted.

- **No ablation on the covariance-weighted objective.** The central technical novelty is that cluster assignments account for posterior covariance via Σ⁻¹. The paper even derives the simplification to diagonal weights (Section 4.1: "By discarding the covariance among new inputs, our approach becomes equivalent to a regression model trained on the complete data (X, μ) with a weighted squared error"). Yet no experiment compares full Σ⁻¹, diagonal-only, and unweighted objectives. Without this ablation it is impossible to know whether the probabilistic formulation contributes anything over fitting a tree to GP mean predictions.

- **Severe and unexplored scalability.** No proven-optimal solution is found in any experiment, with runtimes of 1–5 hours on problems with 44–418 surrogate samples and depth-3 trees. The graph partitioning formulation fails entirely for l = 8 despite using the granularity aggregation (which reduces the effective problem size). No warm-starting strategy, approximate rounding, or decomposition is explored. For ICLR, an MIQP-based method that cannot be solved to optimality even on small problems and provides no characterisation of the optimality gap is difficult to assess practically.

### Minor

- **Potential subscript error in Eq. (8).** The paper states that "the *i*-th non-empty cluster contains at least n₀ data points" and enforces this via n₀αᵢ ≤ wᵢ₁ + ⋯ + wᵢₗ ≤ nαᵢ, where αᵢ indicates whether the *i*-th cluster is empty. However, under the notation established in Eq. (5), wᵢⱼ = 1 if data point *i* belongs to cluster *j*, so the sum wᵢ₁ + ⋯ + wᵢₗ equals 1 for every data point *i* (enforced by Eq. 5). If *i* is here the cluster index, the sum should run over data points belonging to that cluster, i.e., w₁ᵢ + ⋯ + wₙᵢ. This appears to be a substantive subscript inversion that would make the stated constraint trivially satisfied rather than a minimum cluster-size control.

- **Optimality gap never reported.** All experiments note "feasible solutions that were not proven optimal." Without any gap statistic (e.g., the ratio of upper to lower bound at termination), it is impossible to judge how far the returned solutions are from optimal, and thus whether the objective improvements over CART are meaningful or artifacts of solver incompleteness.

- **Statistical significance of improvements in Table 1.** The improvement for Abalone is 0.0961 ± 0.00274 (CART) vs. 0.0932 ± 0.00362 (MIQP), where the standard deviations substantially overlap. No paired test or significance measure is reported, so the claimed superiority for that dataset is not clearly established.

- **Fixed-depth tree structure not highlighted in claims.** The paper requires specifying the binary tree structure in advance, which means it optimises splits within a given skeleton. This is materially different from full structure learning and should be clearly stated in the abstract and introduction rather than relegated to a brief remark in Section 4.3.

### Tiny

- The semantics of Eq. (4) warrant one sentence of clarification: L(ω, v) is the log-density of the constrained latent approximation Wv under q(f), used as a score; ω is not a random variable in the GP posterior.
- The formula for Σ in Eq. (1) uses K_uf to mean the n × m matrix (rows = new inputs, columns = inducing points), which reverses the more common convention; this should be noted explicitly to avoid confusion.

---

## Nice-to-Haves

- **Visualise learned trees.** Showing actual MIQP-learned decision trees alongside CART trees on the same data would make the interpretability claims concrete and tangible.
- **Warm-starting from CART.** Initialising the MIQP with the CART solution as a feasible starting point is a standard technique for MIP tree solvers and could substantially reduce solver time.
- **Scalability characterisation.** A table or curve showing solver time and optimality gap as a function of n (number of points), l (clusters), and tree depth would be very valuable for practitioners.
- **Out-of-sample surrogate fidelity plot.** A scatter plot of GP posterior mean vs. surrogate prediction on held-out data is the most basic sanity check for a surrogate and would strengthen the empirical narrative.
- **Quantifying interpretability.** Reporting simple structural metrics—number of non-empty leaves, average path length, cluster size distribution—would give concrete evidence that interpretability is preserved alongside accuracy.

---

## Removed Points

*These points were raised in the sub-reviews but are removed or heavily discounted as per the synthesis rules; treat them with caution.*

- **Harsh critic: K_uf covariance formula is dimensionally suspicious.** After tracing the paper's convention (K_uf defined as the Gram matrix of (Z, X), used throughout as n × m via symmetric-index convention), the formula is internally consistent; this is a notational choice, not an error.
- **Harsh critic: Distribution-shift claim is an unsupported empirical claim.** Figure 1 and the surrounding text present this as conceptual motivation and illustration, not an empirical claim requiring validation. Removing as scope creep.
- **Harsh critic: Ethics statement is "too thin."** This is a venue-style concern; substance of the ethics statement is not a technical weakness.
- **Harsh critic: No user study or formal interpretability metric.** User studies are not standard for algorithmic surrogate-model papers at ICLR. The claim to interpretability rests on using fewer parameters and structured forms (trees, connected regions), which is a conventional definition in the XAI literature.
- **All reviewers: Missing related work on global distillation, LIME/SHAP as baselines.** The paper's contribution is specifically the GP-posterior-weighted objective; comparing against global LIME or SHAP (which target individual predictions) is outside scope. Per synthesis rules, missing-reference criticisms are removed as they cannot be verified.
- **Harsh critic: Comparison against k-means is unfair because k-means lacks spatial constraints.** This asymmetry favours the baseline (k-means, unconstrained), making the comparison conservative and intentionally so. Not a weakness.
- **Harsh critic: Overstatement of "interpretability" in title/abstract.** While the claim of interpretability could be better operationalised, this is a common usage in the surrogate-model literature and not a factual error.

---

## Novel Insights

The most non-obvious insight in the paper is the identification of a log-determinant complexity term (Eq. 7) that naturally arises when marginalising the cluster-value parameters v—this term acts as a penalty against over-clustering but is non-quadratic and thus discarded. This represents a precise characterisation of why GP-derived clustering cannot be solved with a pure quadratic objective without some additional cluster-count control mechanism, and it motivates the min-size constraint (Eq. 8) as a tractable substitute. If the notation error in Eq. (8) is resolved and an ablation confirms the penalty's practical significance, this observation could form the basis of a deeper theoretical contribution about surrogate complexity.

---

## Suggestions

1. **Fix or clarify Eq. (8).** Verify and correct the subscript convention for the cluster-size constraint; confirm in the appendix that the implementation matches the intended semantics.
2. **Add an optimal-tree baseline.** Implement or call an existing MIQP/exact tree solver (e.g., the Bertsimas & Dunn formulation) on the same GP posterior means; this directly tests whether the GP-weighted objective provides value beyond optimal tree learning with standard regression targets.
3. **Report optimality gaps.** For every experiment, report the Gurobi MIP gap at termination so readers can assess solution quality.
4. **Clarify CART comparison protocol.** State explicitly: (a) whether CART is trained on original labels or GP posterior means, (b) whether the GP is re-fitted in each CV fold, and (c) how the 90/10 split interacts with cross-validation.
5. **Add covariance ablation.** Compare full Σ⁻¹, diagonal Σ⁻¹, and identity (unweighted) on the decision-tree datasets; this is the minimal experiment needed to validate the paper's core methodological claim.
6. **Provide out-of-sample evaluation.** For at least one dataset, evaluate the surrogate on held-out inputs not used during the clustering phase and report fidelity to GP posterior predictions on those points.