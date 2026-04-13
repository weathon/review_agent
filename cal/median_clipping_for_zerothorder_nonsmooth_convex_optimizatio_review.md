=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary
The paper addresses zeroth-order optimization and multi-armed bandits under extremely heavy-tailed symmetric noise (κ > 0, including κ ≤ 1 where the mean may not exist). By combining median gradient estimation with clipping, the authors achieve convergence rates (Õ(d²ε⁻²) for optimization, Õ(√(dT)) for MAB regret) that match optimal bounds for bounded-variance settings, avoiding the degeneration of prior clipping-only methods as κ → 1.

## Strengths
- **Novel extension to κ ≤ 1:** Prior work required κ ∈ (1, 2], which fails when the noise has undefined mean. This paper handles symmetric Cauchy noise and similar distributions where even the first moment diverges—a genuine theoretical advance.
- **Matching optimal rates under symmetry:** The key insight—exploiting symmetry via median estimators to recover variance-like convergence—is technically meaningful and yields rates matching the bounded-variance optimal rates.
- **Unified treatment across settings:** The framework handles both unconstrained/constrained ZO optimization and MAB with consistent methodology, providing a versatile toolkit.
- **Empirical validation in extreme noise regimes:** Figure 3 clearly demonstrates that median-based methods remain stable when κ ≤ 1 while baselines fail, validating the core claim about handling distributions with undefined means.

## Weaknesses
- **Assumption 3 is not justified for the importance-weighted MAB estimator:** The MAB analysis (Theorem 3) relies on Assumption 3 holding for the importance-weighted gradient estimator ĝ_{t,i} = g_{t,i}/x_{k,i}. However, Assumption 3 is stated for the two-point oracle noise φ(ξ|x,y), not for importance-weighted estimators. The distribution of ĝ_{t,i} is a mixture—point mass at 0 with probability 1−x_{k,i} and a scaled continuous component—which is not obviously symmetric. The paper provides no justification that this mixture satisfies Assumption 3. This is a substantive gap in the MAB theoretical claims.

- **Hidden constants blow up as κ → 0:** The variance bound in Lemma 1 has σ² ∝ (4/κ)^(2/κ) which diverges as κ → 0. The median size m = 2/κ + 1 also grows without bound. While the abstract correctly claims "any κ > 0," the practical meaning of this claim diminishes for small κ—the constants become arbitrarily large, and the per-iteration oracle cost grows as Õ(1/κ). The paper should explicitly acknowledge this limitation.

- **MAB experiment contradicts textual claims:** Section 5.1 states "HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does," but Figure 1's right panel shows HTINF achieving ~0.9 probability of best arm selection while the proposed method reaches only ~0.6. The text appears to claim convergence in regret (left panel), but the conflation with "probability" creates confusion about what the method actually achieves.

- **Assumption 3's generality is overstated:** The paper claims Assumption 3 "covers a majority of symmetric absolutely continuous distributions with bounded up to κ-th moments" without rigorous justification. The Cauchy-type tail bound in Eq. (4) imposes a specific algebraic form; the paper should provide examples of distributions that satisfy bounded κ-th moments but fail this density bound, or properly characterize the assumption's scope.

- **Practical gap for unknown κ:** The optimal median size m = 2/κ + 1 requires knowing κ. For κ ≥ 1, the suggestion of m = 3 is reasonable, but for κ < 1—the regime where this method uniquely offers value—no adaptive mechanism exists. A practitioner facing unknown heavy-tailed noise cannot practically deploy the method at its theoretical optimum.

## Nice-to-Haves
- **Dimension scaling validation for MAB:** Experiments use only d = 2 arms. Testing with d ∈ {10, 50, 100} would empirically validate the claimed Õ(√(dT)) scaling.
- **Asymmetry robustness analysis:** Real-world noise often has skew. An empirical study of performance degradation under asymmetric noise would clarify practical applicability beyond the theoretical symmetry assumption.
- **Adaptive median size mechanism:** A heuristic or theoretical approach for selecting m without knowing κ would significantly improve practical utility.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Dimensional scaling discrepancy for Theorem 1:** The critic claimed the new method has an extra factor of d in the deterministic term compared to baselines. Upon verification against Table 1, both methods have d^(1/2)M₂'/ε in the deterministic term for the Lipschitz oracle case, so this criticism does not hold as stated—the notation between M₂' and M₂ requires clarification but the scaling appears consistent.
- **Portfolio experiment baseline criticism:** The Efficient Frontier is criticized as an unfair static baseline. This misses the purpose of the experiment: demonstrating practical applicability on real heavy-tailed data (cryptocurrency), not establishing SOTA in portfolio optimization. The comparison serves as proof-of-concept, not comprehensive benchmarking.

## Novel Insights
The median-of-means approach has deep roots in robust statistics, but its systematic application to zeroth-order optimization under symmetric heavy-tailed noise—particularly achieving optimal rates without variance assumptions—is a genuine contribution. The insight that symmetry enables recovery of variance-like bounds even when variance is undefined (κ < 2) is theoretically meaningful. The interplay between the density bound (Eq. 4) and the moment condition deserves deeper analysis: not all symmetric distributions with bounded κ-th moments satisfy this specific Cauchy-type tail form, and characterizing the gap would strengthen the paper's foundations.

## Suggestions
1. **Address the MAB theoretical gap:** Either provide a lemma proving that the importance-weighted gradient estimator satisfies Assumption 3 under the stated conditions, or modify the algorithm/analysis to handle the mixture distribution that arises from importance weighting.
2. **Clarify MAB experimental claims in Section 5.1:** Explicitly state whether "convergence in probability" refers to regret or arm selection, and reconcile the text with Figure 1's right panel showing HTINF at ~0.9 probability vs. the proposed method at ~0.6.
3. **Add explicit discussion of κ-dependence:** Acknowledge in the main text how the constants scale with κ and what practical range of κ the method is suited for, given the blow-up as κ → 0.
4. **Provide κ-sensitivity experiments:** Show empirical robustness to misspecified m when κ is unknown, demonstrating that the method remains practical even without exact knowledge of the tail index.

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 5.0, 6.0]
Average score: 5.5
Binary outcome: Reject
