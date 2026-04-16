Now let me read the human review files for calibration.Now I have enough information to write the consolidated review. Let me verify one more thing about the Eq. (8) notation issue.## Summary

This paper proposes a Mixed-Integer Quadratic Programming (MIQP) framework to construct interpretable surrogate models for Gaussian Process (GP) posteriors. Data points are assigned to clusters where members share a common parameter, with the assignment minimizing the weighted squared error from the GP posterior mean (using posterior precision as weights). The authors demonstrate that graph partitioning (with spatial connectivity constraints) and decision tree learning can both be expressed as special cases of this formulation via linear inequality constraints.

---

## Strengths

- **Unified and principled MIQP formulation.** The framework elegantly subsumes graph partitioning and decision tree learning as special cases via Theorems 4.2 and 4.3, providing a coherent theoretical foundation rather than ad-hoc methods.

- **Clean theoretical contributions.** Lemma 4.1 (positive-definiteness of W⊤Σ⁻¹W), Theorem 4.2 (reformulation as positive-definite MIQP), and Theorem 4.3 (equivalence between connected undirected graphs and single-leaf DAGs) ensure well-posedness and enable the solver.

- **Well-motivated posterior variance weighting.** The inverse-variance weighting naturally down-weights uncertain data points, and this is properly grounded in the variational GP posterior structure.

- **Addresses a genuine gap in optimal tree literature.** The paper correctly identifies (Section 2) that existing optimal tree methods handle classification or simple regression metrics but not weighted squared error with continuous variables—a requirement this paper fulfills.

- **Spatial connectivity argument is sound.** The interpretability argument for contiguous spatial clusters (Section 4.2) is plausible and well-motivated: geographically connected clusters produce visually comprehensible summaries of spatial GP posteriors.

---

## Weaknesses

### Fatal
*(None — the fundamental conceptual claim is sound; see Removed Points for the harsh critic's incorrectly raised structural objection.)*

### Major

- **Severe and demonstrated scalability failure.** For the California Housing dataset with l=8 clusters, the authors report: "we were unable to obtain a feasible cluster within the time limit" of 5 hours. Even for l=2 and l=4, the solver returns only feasibility-not-optimality-proven solutions after 5 hours, while k-means converges in seconds. The O(n²) binary variables for ordering constraints (Eq. 10) and the growing number of assignment inequalities (Eq. 15) make scaling to practical problem sizes genuinely infeasible. This is not a theoretical concern — the paper's own experiments demonstrate it. The claimed mitigation ("minimum granularity assumption") is used but no systematic study of its effectiveness is provided.

- **Experimental comparison with decision trees lacks methodological clarity.** Section 5.2 states "90% of each dataset was allocated to obtaining a Gaussian process posterior, while the remaining 10% was used to build a decision tree surrogate model." The paper never explicitly states what target function CART is trained on: is it the GP posterior mean µ (with posterior precision weighting), or the original observed labels? If CART is trained on observed labels while MIQP is trained on the GP posterior mean, the comparison is not apples-to-apples and the headline claim "higher-scoring decision trees compared to CART" cannot be interpreted as evidence that the MIQP formulation is superior as a tree-learning method. This ambiguity is central to the paper's main experimental claim.

- **Comparison against an obsolete baseline for decision trees.** CART (Breiman et al., 1984) is a greedy heuristic from four decades ago, and the paper extensively cites modern exact optimal tree methods (Bertsimas & Dunn, 2017; Demirović et al., 2022 (MurTree); Verwer & Zhang, 2019; Aglin et al., 2020) without comparing against any of them. Since the paper frames itself as an exact/optimal tree search and operates at scales (442–4177 samples) where these methods are tractable, this omission severely undermines the significance of the contribution.

- **Core interpretability claims are not operationally validated.** The abstract claims "significant advantages in enhancing the interpretability of spatial modeling," but interpretability is never formally defined, measured, or evaluated. The evaluation provides: (a) a weighted RMSE surrogate fidelity metric, and (b) one qualitative California Housing visualization. There is no user study, no structural complexity analysis, no comparison of cluster properties (e.g., compactness, balance), and no demonstration that practitioners would find the resulting clusters more interpretable. Lower surrogate fitting loss is not interpretability.

### Minor

- **Dropped log-determinant term unanalyzed.** Eq. (7) shows the exact marginalized objective includes −½ log|W⊤Σ⁻¹W|, which acts as a cluster-complexity penalty. The paper drops this term "to leverage efficient algorithms" (Section 4.1) but provides no empirical or theoretical analysis of the impact on solution quality. On small problems where the full objective is tractable, a comparison would be informative.

- **Notation inconsistency in Eq. (8).** The cluster minimum-size constraint is written as n₀αᵢ ≤ w_{i1}+⋯+w_{il} ≤ nαᵢ, where αᵢ indexes clusters. But the sum w_{i1}+⋯+w_{il} uses the same index i as a data point (fixed at 1 by Eq. 5's partition constraint), not as a sum over data points in cluster i. The intended constraint (that the i-th cluster has at least n₀ data points) would require summing over all data points j assigned to cluster i. This may be a typo, but it is confusing and undermines the stated purpose of the constraint.

- **No analysis of GP quality sensitivity.** The paper uses default hyperparameters throughout, and all inducing points are initialized from k-means. Since the surrogate's quality entirely depends on the quality of µ and Σ from variational inference, poor GP fits could produce misleading clusters. There is no sensitivity analysis.

- **Distributional shift claim unvalidated.** The introduction (and Figure 1) motivates surrogates as superior to direct training "when the distribution of new inputs differs from the training data," but in both experiments, new inputs are drawn from the same distribution (in fact, they are identical to training inputs in Section 5.1). The paper's main selling point is never actually tested.

### Trivial

- The ethics statement is generic given the paper explicitly discusses applications in marketing and risk management with granularity constraints that have fairness implications.

---

## Nice-to-Haves

- Provide scalability curves (time vs. n, l, d) and quality-vs-time tradeoffs for both graph partitioning and decision trees, showing where the method is practically useful.
- Show the actual learned decision trees (features, thresholds, leaf values) so readers can judge interpretability directly.
- Add a surrogate fidelity metric (e.g., R² between surrogate predictions and full GP predictions on held-out inputs) alongside the loss, to establish that the surrogate is a faithful GP approximation.
- Consider LP relaxations or warm-starting from CART solutions to improve solver tractability.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

**Harsh Critic Issue 1 (Claimed "structural" probabilistic interpretation error)**: REMOVED. The critic claims the paper falsely asserts it maximizes posterior probability density because it drops the log-determinant term from Eq. (7). This is incorrect. The paper explicitly states it maximizes L(ω,v) — the joint log-posterior density of the parameter vector v and assignment ω — via Eq. (5). This is standard joint MAP estimation. The log-determinant term in Eq. (7) only arises when one *marginalizes* over v (i.e., computes the probability of ω alone). The paper never claims to maximize the marginal probability of ω; it claims to maximize the joint density L(ω,v), which Eq. (5) does exactly. The paper then *discusses* the marginalized version as an alternative, noting it cannot be expressed in quadratic form. The critic confused joint MAP estimation with marginal likelihood maximization. The abstract's language ("maximizing the probability density of the posterior distribution") is consistent with joint MAP estimation and is not a misrepresentation.

**Harsh Critic Issue regarding unfair comparison with k-means (k-means ignores contiguity and posterior information)**: The criticism that k-means is not a "serious baseline for contiguous spatial partitioning" is noted. However, since k-means is *less capable* than the proposed method (it ignores both contiguity and posterior structure), using it as a baseline is intentionally asymmetric in the authors' favor — this is reasonable to show the method works. What's missing is a *stronger* baseline (e.g., spatial clustering with contiguity constraints), not the removal of k-means.

**Harsh Critic's CART training target claim**: Partially addressed above as a major weakness. The framing as "structurally fatal" is too strong; it is a genuine experimental clarity issue.

---

## Novel Insights

The most genuinely novel observation across all reviews is the combination of (1) posterior variance weighting with (2) exact combinatorial surrogate fitting via MIQP, and specifically the proof that spatial connectivity and axis-aligned decision tree partitions can be expressed as linear inequality constraints on the same MIQP objective. The observation that existing optimal tree literature does not handle weighted squared error with continuous variables (Section 2) is a real and underappreciated gap that this paper partially fills, even if the experimental validation is too limited to establish significance. The connection via Theorem 4.3 between connected undirected graphs and single-leaf DAGs is a clean structural result that may find independent use.

---

## Suggestions

1. **Clarify CART training target**: Explicitly state whether CART in Table 1 is trained on (a) original labels, (b) GP posterior means µ, or (c) GP posterior means with precision weighting. If (a), reframe Table 1 as evidence that GP-posterior-aware training is better — not as a comparison of tree-learning algorithms.

2. **Add OCT or MurTree baseline**: At least one modern exact optimal tree method should be compared, since the paper operates at scales where these are tractable and explicitly cites them.

3. **Demonstrate with a distributional shift experiment**: Create a held-out test set with a different input distribution (e.g., using geographic regions withheld entirely from training) and show that the GP-posterior surrogate generalizes better than a tree trained directly on observed labels. This would validate the paper's core motivating claim from Figure 1.

4. **Report optimality gap**: For all MIQP experiments, report the MIP gap at termination (available from Gurobi). This is a standard reporting requirement for any paper using MIP solvers and reveals how far the feasible solutions are from optimality.

---

## Score and Decision

**Calibration against retrieved papers:**

- *f3TSOXnkXZ* (Output-Constrained Decision Trees): Scores 5, 3, 5, 3 → avg ~4. Rejected. Similar issues: limited baselines (no comparison with optimal tree methods cited in the paper), small datasets, limited experimental evaluation of the core claim. That paper also used MIQP for trees and compared against weak baselines.

- *ghk8lnOYRq* (k-Hyperplane Clustering via MIQP): Scores 5, 5, 5, 5 → avg 5. Rejected. Stronger theoretical contributions (proofs, bounds) but no real-data experiments and algorithmic novelty is debated.

- *GhT6NjiLeA* (Shapley Values for Additive GP): Scores 6, 1, 3, 3 → avg ~3.25. Withdrawn. GP interpretability paper with limited evaluation and overclaimed contributions.

**Positioning**: The paper under review shares the central weaknesses of f3TSOXnkXZ (MIQP decision trees with weak baselines) and GhT6NjiLeA (GP interpretability with overclaimed benefits and limited evaluation). The scalability failure is self-reported and severe. The theoretical contributions are cleaner than GhT6NjiLeA but the experiments are comparable in weakness to both f3TSOXnkXZ and GhT6NjiLeA. The comparison against CART-only — while framed as an advantage — is exactly the type of weak baseline criticism that drove rejections in f3TSOXnkXZ. The paper is at best at the borderline of f3TSOXnkXZ's weaker scores (~3).

**Axis evaluation:**
- *Originality*: Moderate — individual components are not new, but the unification under one MIQP framework for GP surrogates is a genuine contribution.
- *Importance of research question*: Reasonable — GP interpretability is an active area, but the paper addresses a narrow slice of it.
- *Claims well-supported*: Weak — scalability failure is self-demonstrated; interpretability claims are unsupported; CART comparison is ambiguous.
- *Soundness of experiments*: Below average — one spatial dataset, three small UCI datasets, no significance testing, feasibility-but-not-optimality for all MIQP solutions.
- *Clarity of writing*: Acceptable but with notable notation issues (Eq. 8) and missing experimental details.
- *Value to community*: Limited in current form — the severe scalability issues and absence of strong baselines make the practical takeaway unclear.

**Final score: 3.5** — The paper has a genuinely interesting idea and some clean theoretical results, but the experimental evidence is insufficient, the key scalability limitation is demonstrated by the authors themselves, the comparison uses a 40-year-old baseline, and the central interpretability claim is never actually measured. This is below the threshold for publication in its current form, consistent with comparable papers scoring in the 3-4 range.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>