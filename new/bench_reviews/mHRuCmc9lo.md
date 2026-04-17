The paper content was provided in the prompt. Let me search for relevant text within it.

Looking at the paper content provided, I can verify:

1. The paper does mention Appendix B for approximate H-calibration ("In the main body of the paper we focus on the H-calibration defined above, but in Appendix B we also discuss scenarios in which only approximate H-calibration is available.")

2. The linear utility assumption (Assumption 2.1) is explicitly stated and the paper acknowledges this limitation in Section 6.

3. The comparison to swap-regret / omniprediction literature is made in Section 4.1 and Section 1.2.

4. The experiments are indeed limited: two datasets, one model, three actions, no error bars.

5. The paper's experiments only test H = {h(v)=v} (self-orthogonality), not decision calibration.

Now let me compose the final review.

## Summary
This paper studies how a conservative decision maker should optimally act on machine learning predictions that carry only partial ("H-") calibration guarantees, rather than full calibration. The authors formulate a minimax problem—maximizing worst-case expected utility over outcome distributions consistent with H-calibration—and characterize the optimal robust decision rule via a duality argument (Theorem 3.1). They show a sharp transition: once the test class H contains decision calibration constraints, the minimax optimal rule collapses to the plug-in best response (Theorems 4.1–4.2), upgrading prior swap-regret guarantees to full minimax optimality. They also derive practical instantiations from properties arising "for free" from squared-loss training (self-orthogonality) and standard post-hoc recalibration (bin-wise calibration), and provide a small-scale empirical evaluation.

## Strengths
- **Principled and elegant problem formulation.** The minimax framework (Eq. 5) for partially calibrated forecasts is a natural and well-motivated interpolation between aggressive best-responding and fully conservative play. The definition of the ambiguity set Q via H-calibration constraints is conceptually clean and directly connected to the multicalibration literature.
- **Surprising and impactful theoretical result.** The collapse of the minimax-optimal rule to plug-in best response under decision calibration (Theorems 4.1–4.2) is a sharp and non-obvious finding. It identifies a tractable, task-specific calibration threshold beyond which no additional robustness is needed, which significantly strengthens the known swap-regret guarantees for decision calibration.
- **Clean general duality characterization.** Theorem 3.1 provides a constructive saddle-point decomposition (dual multipliers → adversarial tilt → pointwise best response) that applies to any finite-dimensional H, yielding an efficiently computable decision rule.
- **Practical connections to standard pipelines.** Propositions 4.4 and 4.5 derive usable robust policies from properties that arise automatically from squared-loss training and cheap post-hoc recalibration, respectively, bridging the abstract framework to practice.

## Weaknesses

### Major:
- **The relationship to prior omniprediction/swap-regret guarantees is insufficiently clarified.** The paper claims to "upgrade" prior swap-regret guarantees to minimax optimality, asserting that previous work only rules out policies of the form ϕ∘a_BR ("actions as a bottleneck"). However, some omniprediction results (e.g., Roth & Shi 2024, Hu & Wu 2024) consider arbitrary downstream policies in a broader loss landscape. The paper does not formally establish whether the minimax optimality notion here is strictly stronger than, equivalent to, or merely a reformulation of these existing guarantees. A precise theorem or counterexample separating the two would substantially strengthen the contribution; without it, the novelty of the main conceptual claim is unclear relative to what was already known.

- **The linear-in-v utility assumption (Assumption 2.1) significantly limits the decision-theoretic scope.** The entire framework—from the ambiguity set Q to the duality and collapse results—relies on utility being linear in the conditional expectation v. This excludes risk-averse, variance-sensitive, and many other practical utility structures. The paper's motivation invokes "high-stakes decision making" and "trustworthy ML," settings where risk aversion is often central. While acknowledged in Section 6, the framing of the results as recovering "full-calibration-style trustworthiness" in the abstract and introduction is misleading without prominently foregrounding this restriction: under non-linear utilities, neither full calibration nor decision calibration ensures that acting on point predictions is optimal, and the minimax construction here does not directly apply.

- **The empirical evaluation is thin and does not test the paper's central theoretical prediction.** The experiments only evaluate H = {h(v) = v} (self-orthogonality from squared loss) on two regression datasets with 3 discrete actions each. No experiment tests the primary theoretical contribution—the collapse under decision calibration—nor the bin-wise calibration instantiation (Proposition 4.5). There are no error bars or confidence intervals; the reported utility differences (e.g., 0.402 vs. 0.410) are small enough to be noise. No comparison is made to other natural robustification approaches (e.g., simple shrinkage, distributionally robust optimization with Wasserstein balls). The experiments also only use d=1, while the paper's key motivation is high-dimensional (d>1) settings where full calibration is intractable.

### Minor:
- **Gap between theory (exact H-calibration) and practice (approximate calibration).** Proposition 4.4 guarantees self-orthogonality only at a first-order stationary point; in practice, neural networks trained with SGD are only approximately stationary, and the moment conditions hold only approximately. Although Appendix B discusses approximate H-calibration, no quantitative bounds connect the approximation level to the robust policy's suboptimality, leaving the practical theory-practice gap unquantified.
- **The finite action set assumption limits the scope of the collapse result.** Theorems 4.1–4.2 require |A| to be finite (so that the decision regions R_a are well-defined indicators). For continuous or very large action sets, the structure of the result could differ substantially; this limitation is not discussed.

### Trivial:
- None worth noting.

## Nice-to-Haves
- Experiments testing the decision-calibration collapse (e.g., post-processing a predictor for decision calibration and verifying that the robust policy coincides with best response).
- Experiments with d > 1 (multiclass outcomes), which is the setting that primarily motivates the paper.
- Comparison to simpler robustification heuristics (shrinkage, Wasserstein DRO, etc.) in the empirical evaluation.
- Finite-sample analysis of how estimation error in the dual multipliers λ* propagates to the robust action.
- Visualization of the adversarial tilt q*(v) vs. v for different H classes, illustrating the transition from full conservatism to best response.

## Removed Points
- **"No empirical validation of H-calibration itself":** While it is true the paper does not measure realized calibration error, Proposition 4.4 guarantees self-orthogonality at stationarity and the calibration split is used to estimate population expectations. This is a secondary concern rather than a core methodological flaw.
- **"Adversaries may be overly pessimistic / consider alternative ambiguity sets":** This is a scope concern—suggesting φ-divergence or Wasserstein constraints is natural, but the paper's framework is specifically about ambiguity sets defined by calibration constraints. This is a design choice, not an error.
- **"The bin-wise calibration robust policy is just best-responding to bin means, which practitioners already do":** While true, the contribution is not the recipe itself but showing that it is *minimax optimal* within the H-calibrated ambiguity set. The optimality guarantee is the new insight.
- **"Regularity conditions for Theorem 3.1 not explicit":** The action set A is finite and u is linear in v, ensuring compactness and continuity. Existence of saddle points follows under these conditions. This is not a real gap.
- **"Comparison to trivial minimax baseline makes contribution look stronger than it is":** The paper motivates its framework relative to this baseline for clarity, not to inflate contributions. The real comparison is to the multicalibration/omniprediction literature, which is explicitly discussed.

## Novel Insights
The "sharp transition" result—that the minimax optimal policy collapses to plug-in best response as soon as H contains the decision-calibration tests, and that no intermediate H-class yields a different policy form—is a genuine and non-obvious structural insight. It implies that decision calibration is not just a useful relaxation of full calibration but is precisely the decision-theoretic "threshold" for trustworthiness, which is both theoretically elegant and practically prescriptive: practitioners need only enforce decision calibration, not full calibration, to guarantee that best-responding to forecasts is optimal.

## Suggestions
- Formally clarify (via theorem or example) the precise relationship between the minimax optimality guarantee here and prior omniprediction/swap-regret guarantees, so readers can assess the novelty of the conceptual contribution.
- Add at least one experiment with a multiclass setting (d > 1) and/or an experiment using decision-calibrated predictors to validate the collapse theorem.
- Report error bars across random seeds and include a comparison to at least one simple robust baseline.
- Prominently note the linear-utility restriction in the abstract and introduction, rather than only in Assumption 2.1 and the limitations section.
- Provide even informal quantitative bounds on robust policy suboptimality under approximate H-calibration, to bridge the theory-practice gap for the self-orthogonality instantiation.

## Calibration and Score

**Calibration anchors:**
- uy4EavBEwl.md (Reconciling Model Multiplicity for Downstream Decision Making): Accept (Poster), scores 6,6,6,6 — similar multicalibration/decision-calibration foundations, theory + limited experiments.
- TId1SHe8JG.md (Provable Uncertainty Decomposition via Higher-Order Calibration): Accept (Spotlight), scores 6,8,8,8 — strong calibration theory, but acknowledged theoretical novelty and well-executed experiments.
- XM7INBbvwT.md (Does Calibration Affect Human Actions?): Reject, scores 3,5,6 — calibration-for-decisions topic with weak experimental methodology.
- MUWkqqH6e7d.md (Human Expertise Really Matters): Withdrawn, scores 5,6,6,6 — calibration + decision-making, dense presentation/limited scope concerns.

This paper has a genuinely novel and elegant theoretical contribution (the collapse theorem under decision calibration, the general H-calibration duality) that advances understanding of partial calibration guarantees. However, the novelty relative to prior decision-calibration/omniprediction work is not cleanly established, the critical linearity assumption limits the practical scope that the paper claims, and the experiments are notably thin—not testing the central theoretical prediction and lacking error bars or baseline comparisons. Relative to "Reconciling Model Multiplicity" (6,6,6,6) and "Higher-Order Calibration" (6,8,8,8), this paper has comparably clean theory but weaker empirical support and a less clearly delineated contribution relative to prior work. I place it below the Reconciling paper (which had clearer novelty) but above the weak calibration experiments papers.

**Score: 6**

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>