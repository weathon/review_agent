## Summary
The paper proposes a Mixed-Integer Quadratic Programming (MIQP) framework that constructs interpretable surrogate models for Gaussian process (GP) posteriors by assigning new data points to clusters sharing a common parameter value. The clustering objective minimizes a weighted squared error against the GP posterior mean, with weights derived from the full posterior precision matrix Σ⁻¹. Two structured variants are developed: (1) spatially connected graph partitioning via a DAG-based connectivity formulation, and (2) axis-aligned decision tree learning via split/assignment constraints. Experiments on the California Housing dataset and three UCI datasets demonstrate modest improvements over k-means and CART respectively.

---

## Strengths

- **GP-posterior-aware objective with full covariance.** Unlike standard surrogate tree fitting that regresses on raw labels or posterior means, the objective in Eq. (4)–(5) uses the full Σ⁻¹ precision matrix, downweighting uncertain points and capturing posterior correlations. This is a principled and differentiating design choice that most surrogate/distillation methods in the area do not make.

- **Novel DAG-based connected-cluster formulation.** Theorem 4.3 — that a connected undirected graph corresponds exactly to a DAG with one leaf — provides a clean and novel algebraic basis for encoding spatial contiguity as linear MIQP constraints. This is a technically creative contribution that does not appear in prior optimal-tree or spatially-constrained clustering literature.

- **Unified framework spanning two distinct surrogate types.** Both graph partitioning and decision tree learning are derived from the same core MIQP objective by adding structured linear constraints. The paper also handles non-Gaussian likelihoods (Poisson, Bernoulli) in the tree experiments, demonstrating some generality.

- **Transparent acknowledgment of the log-determinant gap.** The paper openly derives the marginalised objective including the log|W⊤Σ⁻¹W| term in Eq. (7) and explicitly states it is dropped for tractability. This honesty is commendable, even though the implications are not fully analyzed.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Only CART as a decision-tree baseline — insufficient to support the core claim.** The paper explicitly positions itself within the optimal-tree literature and even surveys it in the related work (OCT, MurTree, DL8.5, etc.). Yet Table 1 compares only against CART, a 1984 greedy heuristic. It is trivially expected that exact search beats greedy search given enough time. The paper never answers the key question: does the GP-posterior-weighted objective produce better trees than an optimal tree method (e.g., OCT or MurTree) using the same depth and time budget but fitting to GP posterior means with standard MSE? Without this comparison it is impossible to attribute gains to the GP-posterior objective rather than to the use of exact optimization per se.

- **Motivating claim about distribution shift (Figure 1 / Section 1) is never experimentally validated.** Section 1 states: "Compared to a decision tree trained directly on observed data, a surrogate model performs better when the distribution of new inputs differs from the training data." This is the primary motivation for the entire surrogate framework, yet it is never tested. In the decision-tree experiments, both methods see the same 10% evaluation set; there is no experiment with shifted new-input distributions. This undermines the foundational rationale.

- **No ablation on the covariance structure — the central technical contribution.** The paper's key novelty over plain weighted regression is retaining off-diagonal terms in Σ⁻¹. Section 4.1 even notes that ignoring covariance reduces the method to weighted regression on (X, μ). Yet neither the graph partitioning nor the tree experiments include a comparison between: (a) full Σ⁻¹ objective, (b) diagonal weighting only, (c) unweighted MSE, (d) clustering of posterior mean directly. Without this ablation it is unclear whether the full covariance adds any practical value.

- **Scalability is severely limited and inadequately addressed.** The graph partitioning experiment fails to find even a feasible solution for l=8 within 5 hours on 20K points. For the Abalone dataset (n≈4K), the MIQP needs up to 5 hours. The paper offers only a brief disclaimer in the Limitation section without proposing principled approximations (e.g., LP relaxations with rounding, warm-starting from CART, or hierarchical decomposition) or bounding the regime in which the method is feasible. As written, the approach is computationally inapplicable to most real-world GP settings.

- **MIP optimality gaps are never reported.** Table 1 and the graph partitioning results explicitly state "feasible solutions that were not proven optimal." The paper never reports how far these solutions are from the optimum (MIP gap). Given that solving to optimality is the entire motivation for the MIQP approach, reporting only "feasible solutions of unknown quality" substantially weakens the empirical case.

### Minor

- **Notation in Eq. (8) appears to have an indexing error.** Eq. (8) is written as `n_0 α_i ≤ w_{i1} + ... + w_{il} ≤ n α_i`, where α_i is a cluster-level indicator. But from Eq. (5), `w_{i1} + ... + w_{il} = 1` for each data point i by definition. If i indexes clusters here, the sum should be over data points (∑_j w_{ji}), not over cluster assignments of a single point. This ambiguity should be resolved with explicit notation; it affects whether the cluster-size constraint is correctly formulated.

- **The surrogate is evaluated on the same inputs used to construct it.** In Section 5.2, 10% of data are used both to build the surrogate tree and to evaluate its loss. It is unclear whether the loss in Table 1 is in-sample (on the 10% build set) or out-of-sample. For graph partitioning, new inputs are explicitly identical to training inputs. The paper should clarify and ideally evaluate surrogate fidelity on a held-out set distinct from the surrogate-building set.

- **No visualization of learned decision trees.** The paper's central claim is enhanced interpretability, yet no learned tree structure, split rules, or leaf values are shown anywhere. Without seeing the tree, interpretability claims about rule clarity, feature reuse, or split semantics cannot be evaluated.

- **Big-M bound construction is unspecified.** Eq. (6) requires a universal M such that [-M, M]^l contains v̂(ω) for all ω. No constructive bound or practical heuristic is given. Tight big-M values are essential for solver performance and numerical stability in MIQP formulations; this is a standard concern in the MIP literature.

### Tiny

- The paper does not report whether GP hyperparameters (lengthscale, noise) were optimized on the training split or the full dataset, and whether the 10% surrogate-building set was excluded from GP training. These protocol details affect the validity of the comparison.

---

## Nice-to-Haves

- A warm-starting strategy using CART solutions as initial feasible points for the MIQP could dramatically reduce solve times and should be straightforward to implement.
- Reporting cluster statistics alongside loss (e.g., actual cluster count achieved, cluster size distribution, tree depth utilization) would better connect quantitative results to the interpretability motivation.
- A small experiment with covariate-shifted new inputs would directly demonstrate the advantage claimed in Figure 1 and Section 1, and would substantially strengthen the paper's narrative.
- A comparison to spatially-constrained clustering baselines (e.g., REDCAP, Max-P regionalization) would be a more appropriate baseline for the graph partitioning experiment than unconstrained k-means, though k-means can serve as a lower bar.

---

## Removed Points
*These points were flagged for removal or significant weakening. Treat with caution.*

- **[REMOVED] Missing related works (harsh critic Section 2).** Per review instructions, missing related work is excluded since we cannot confirm external references.

- **[REMOVED] K-means comparison is unfair (harsh critic Section 5.1).** K-means lacks both connectivity constraints and GP posterior information; the comparison is intentionally asymmetric in favor of k-means to make a stronger point for the proposed method. This pattern — giving the baseline an intentional advantage — should not be flagged as a weakness.

- **[REMOVED] Σ formula uses K_{uf} on both sides (harsh critic Section 3.1).** This is almost certainly a PDF-to-text parsing artifact. The standard SVGP formula is well-known and the paper's content is otherwise consistent with it.

- **[REMOVED] Ethics statement is too thin.** For an algorithmic methods paper, the level of ethical discussion provided is reasonable for the current norms of the community. Generic calls for deeper ethics engagement are not appropriate here.

- **[WEAKENED → Minor] Probabilistic interpretation of Eq. (4).** The paper says "the posterior probability of v and ω can be approximated by" L(ω,v). While technically the wording is loose (L is the log density of f evaluated at Wv, not a posterior over (v,ω)), the phrase "can be approximated by" provides sufficient hedging. The optimization goal is clear regardless of this framing.

- **[WEAKENED → Nice-to-Have] Claim that no optimal tree solves weighted regression.** The paper states "to the best of our knowledge, no existing work...has yet satisfied this requirement" (weighted squared error with continuous leaf values). While the optimal-tree literature is large, this specific combination with full GP covariance weights is indeed not addressed by the cited prior work. This is partially a novelty claim, partially a framing issue; it should be stated more narrowly but need not be removed.

- **[WEAKENED → Tiny] Minimum-cluster-size constraint is equated with interpretability (harsh critic).** The paper explicitly frames smaller parameter count as one component of interpretability. This is a valid proxy, and the paper does not claim it is exhaustive. Demanding a formal user-grounded interpretability study is not standard in this sub-field.

---

## Novel Insights

The most genuinely novel technical insight in this paper is the combination of two ideas that have not appeared together before: (1) using the full variational GP posterior precision Σ⁻¹ — not just diagonal variance weights — as the Mahalanobis metric for surrogate fitting, which captures output correlations across nearby inputs when constructing clusters; and (2) encoding spatial connectivity through the DAG-with-one-leaf equivalence (Theorem 4.3), enabling linear MIQP constraints for connected-region clustering without cut-based or flow-based relaxations. The synthesis of these into a single problem formulation is the paper's primary intellectual contribution, even if its empirical validation does not yet fully substantiate its practical value.

---

## Suggestions

1. **Add at least one optimal-tree baseline** (e.g., OCT or a depth-3 MurTree) with the same time budget, fitting to GP posterior means with standard MSE, to isolate the value of the full covariance objective vs. exact optimization alone. This is the single most important addition.

2. **Run the ablation: full Σ⁻¹ vs. diagonal vs. unweighted.** A small table or figure showing loss under these three objective variants would directly validate the paper's core novelty claim. This can be done on existing datasets with modest compute.

3. **Report MIP optimality gaps** (e.g., Gurobi's reported MIPGap at termination) for all experiments. This is a one-line addition to the reporting and is essential for interpreting feasibility-only results.

4. **Fix or clarify the indexing in Eq. (8).** Explicitly state the summation is over data points j for each cluster i, i.e., ∑_j w_{ji}, to remove the apparent contradiction with Eq. (5).

5. **Show at least one learned decision tree** (structure, split features/thresholds, leaf values) for one fold of one dataset to ground the interpretability claims concretely.

6. **Include a small covariate-shift experiment** — even a toy 1D GP example — where new inputs are drawn from a different region than training inputs, directly demonstrating the motivation of Section 1 / Figure 1.

---

**Evaluation summary:** The paper offers a technically novel and principled core idea, but its empirical support is the weakest component by a significant margin. With only one baseline per experiment (CART, k-means), no ablations on the key design choice (covariance structure), no validation of the central motivating claim (distribution shift), unreported optimality gaps, and severe scalability limitations that go unresolved, the paper does not currently provide enough evidence for its claims to meet the ICLR bar. The novelty is real and the technical framework is sound; the paper would be substantially stronger with the ablations and additional baselines described above.