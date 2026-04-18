Now I have enough context to write the final review. Let me synthesize all the information.

## Summary

This paper proposes a method to improve the efficiency and accuracy of identifying memorized points in deep models. The authors observe that memorized points are learned later in training (validated empirically across multiple architectures and datasets), develop a simple "Accuracy per Batch" (ApB) proxy computed from a single model's training trajectory that correlates strongly (Pearson < -0.95) with memorization scores, and incorporate this proxy into the standard pseudo Leave-One-Out (pLOO) method to restrict its search space to likely-memorized points. The resulting pLOO_improved reduces computational overhead by over 90% (from ~2000 to ~200 shadow models) and claims to reduce approximation error against the gold-standard LOO baseline by over 65% (RMSE from 35.5 to 12.19).

## Strengths

- **Well-motivated and practically important problem.** The pLOO method is a foundational tool in ML privacy research, and identifying that it is both computationally expensive and inaccurate relative to the LOO baseline is a valuable contribution. The observation that only a fraction of points are memorized (Figure 1), and that running pLOO on the entire dataset is therefore wasteful, is a clear and actionable insight.

- **Simple, adoptable proxy.** The ApB metric (Algorithm 1) requires adding approximately 4 lines of code to a standard training loop and needs only a single model. This low barrier to adoption is a genuine strength. The Pearson correlation of < -0.95 between ApB and pLOO scores across all evaluated models and datasets (Figure 3) is consistently strong.

- **Compelling training dynamics analysis.** The empirical validation that generalized points are learned earlier than memorized ones (Figure 2), across multiple architectures and datasets, is a clear and useful finding. Figure 2 effectively demonstrates the learning trajectory separation between memorized and generalized points.

- **Demonstrated computational savings.** Reducing from 2000 to 200 shadow models is a substantial and concrete efficiency gain. The end-to-end practical improvement is real, regardless of the accuracy claims.

- **Honest discussion of limitations.** The paper explicitly acknowledges the lack of theoretical grounding and the continual learning limitation (Section 7).

## Weaknesses

### Fatal
None.

### Major

- **The central "more accurate" claim is supported by a narrow and potentially biased evaluation.** The paper's headline result—that pLOO_improved reduces RMSE vs. LOO from 35.5 to 12.19 (a 65% reduction)—is derived from a single experiment on VGG-6 (a small, non-standard architecture) on CIFAR-10, on only ~150 points that were *explicitly selected* to maximize disagreement between pLOO and pLOO_improved (Section 5.2: "we run it over 150 points that had the largest difference in memorization scores between the original pLOO and pLOO_improved"). This adversarial selection procedure means the comparison is not representative of typical or average performance. The paper's strong general claims ("erodes the trust in the original pLOO method," "pLOO over-estimates memorization scores") rest on this narrow evidence. The authors also assert that "pLOO and LOO are model-independent methods" and so "we have no reason to believe that our findings will not extend to other models," but this is a claim, not evidence—memorization behavior is known to vary with architecture depth, regularization, and optimization dynamics. Without LOO validation on at least one additional architecture or dataset, or on an unbiased sample of points, the accuracy improvement claim is insufficiently substantiated. This is a significant gap because the "more accurate" half of the paper's dual contribution is what distinguishes it from a pure efficiency improvement.

- **The ApB proxy is validated only against pLOO, not against LOO (the true ground truth).** Since the paper itself argues that pLOO is an inaccurate approximation of true memorization (RMSE 35.5), establishing that ApB correlates strongly with pLOO does not validate that ApB tracks genuine memorization. The paper needs at least a small-scale comparison of ApB vs. LOO scores to verify that the proxy captures true memorization, not just pLOO's biased approximation. For the 150 points where LOO is available, this comparison is straightforward and would significantly strengthen the evidence—but it is not provided.

- **The paper is caught in a conceptual tension regarding pLOO's role.** pLOO is used as the primary benchmark for validating ApB (Section 4.3, Figure 3), yet also portrayed as seriously flawed and untrustworthy (Section 5.3, Section 6). If pLOO is as inaccurate as claimed, then the strong correlation of ApB with pLOO (< -0.95) does not reliably establish ApB as a proxy for *true* memorization. The paper needs to reconcile this: either pLOO is a reasonable proxy (in which case the accuracy-improvement claims are less dramatic), or pLOO is unreliable (in which case correlating ApB with pLOO is insufficient validation). The paper attempts to have it both ways, which undermines the evidential coherence.

### Minor

- **The top-5000 threshold for selecting memorized points lacks justification or sensitivity analysis.** The choice to select the top 5,000 points by ApB score is stated without systematic exploration of how performance varies with this threshold. Given that Figure 1 shows the fraction of memorized points varies substantially across datasets (e.g., CIFAR-10 vs. CIFAR-100 vs. Tiny ImageNet), a fixed k=5000 could over- or under-select. A brief sensitivity analysis would strengthen the practical guidance for users.

- **No comparison with alternative training-dynamics proxies.** Other simple metrics—such as final-epoch loss, learning time (first epoch of correct classification), or forgetting event counts—are natural alternatives to ApB. Without benchmarking against at least one alternative, it is unclear whether ApB is a particularly good choice or whether any training-dynamics metric would suffice. This is not fatal (the proxy works well), but it limits the mechanistic understanding.

- **The "single model" claim for ApB requires clarification.** The training-cycle validation in Section 4.2 uses 50 repeated runs and aggregates across all 50 models to define "learned." But ApB is later presented as requiring only a single model. It is unclear whether the correlation figures in Figure 3 use a single run or an average over 50 runs, and how stable ApB is across different random seeds in the truly single-model regime.

- **The 50-run requirement in Section 4.2 and the computational transparency.** While the core pLOO_improved pipeline genuinely uses fewer models, the overall experimental pipeline involves training 50 models for ApB validation (Section 4.2) plus 200 shadow models. The paper does not provide a wall-clock or FLOPs comparison of the full end-to-end pipeline versus vanilla pLOO, making the headline "90% reduction" a partial picture.

### Trivial

- The paper states "Pearson score < -0.95" which is an unusual way to express a strong negative correlation; the convention would be to state the absolute value is > 0.95 or that the correlation is approximately -0.96 to -0.97.

## Nice-to-Haves

- Comparison of ApB against at least one alternative training-dynamics proxy (e.g., final-epoch loss or forgetting events) on the same benchmarks.
- LOO validation on even a small number of points across a second architecture (e.g., ResNet18/CIFAR-100) to establish that the accuracy improvement generalizes beyond VGG-6/CIFAR-10.
- Sensitivity analysis varying the top-k threshold for ApB-based point selection.
- A direct scatter plot of LOO vs pLOO vs pLOO_improved scores (not just error histograms) for the 150-point evaluation, to reveal whether the improvement is uniform across memorization score ranges.
- A brief analysis of ApB stability across random seeds in the single-model regime.

## Removed Points

- **Questioning the availability/existence of pLOO, FFCV, or other cited tools.** The paper references established methods (pLOO from Feldman & Zhang 2020, FFCV library, etc.). Any concern about their availability is a reviewer knowledge gap, not an author error. — *Removed*

- **Demanding experiments on full ImageNet or billion-parameter models.** The paper explicitly uses CIFAR-10/100 and Tiny ImageNet with standard architectures. Demanding evaluation on production-scale models goes beyond the stated scope. The computational cost discussion is clear about the current evaluation regime. — *Removed*

- **Complaints about lack of theoretical proofs.** The paper is an empirical systems/contribution paper in ML privacy. Theoretical proofs are not standard in this area. The authors acknowledge this limitation in Section 7. — *Removed*

- **Concerns about lack of comparison with membership inference attacks downstream.** The paper's stated scope is improving the memorization identification method itself, not demonstrating downstream utility in MIA. The discussion section mentions this as future impact, not a claimed contribution. — *Removed*

- **Nitpick about "0.04 shards per point" ratio being heuristic.** This ratio is adopted from the original Feldman & Zhang paper. Using the same configuration makes the comparison fair. — *Removed*

- **Continual learning limitation as a major weakness.** The paper acknowledges this clearly in Section 7 and proposes a simple workaround. This is an explicit scope boundary, not an oversight. — *Moved to Nice-to-Have*

- **Demand for error bars or confidence intervals.** Single-run/same-seed evaluation is the norm in this research area (see Feldman & Zhang 2020 and subsequent work). — *Removed*

- **Formatting or style complaints.** — *Removed*

## Novel Insights

The most interesting insight is the conceptual tension the paper itself creates but does not resolve: if pLOO is truly as inaccurate as claimed (RMSE 35.5), then validating the ApB proxy by its correlation with pLOO (< -0.95) provides only a guarantee that ApB tracks *pLOO's biased approximation*, not true memorization. This means the paper's two main claims (ApB is a good proxy, and pLOO is inaccurate) are partially at odds with each other. The proxy would be more convincingly validated against the LOO baseline, even on the 150 points where LOO scores are available—but this comparison is absent. Additionally, the adversarial point selection for the LOO comparison (choosing the 150 points with maximal pLOO vs pLOO_improved disagreement) means the accuracy improvement is demonstrated on a subset specifically designed to favor the new method, leaving typical-case performance uncharacterized.

## Suggestions

1. **Compute ApB-LOO correlation on the 150-point LOO evaluation.** This is the single most impactful addition—the data already exists, and it would resolve the circularity concern.

2. **Run LOO on an unbiased random sample of points (not just adversarially selected ones)** to establish typical-case accuracy. Even 100 randomly selected points would suffice.

3. **Report a sensitivity analysis for the top-k threshold** (e.g., k=1000, 2500, 5000, 10000) showing how RMSE and compute scale.

4. **Clarify whether Figure 3 correlations use single-run or 50-run averaged ApB scores**, and report single-run stability of ApB across seeds.

5. **Add a scatter plot of LOO vs pLOO vs pLOO_improved** scores on the 150 evaluated points, to make the comparison more transparent than the error histogram alone.

6. **Soften the claims about pLOO's unreliability** to match the evidence: the current evaluation shows pLOO is less accurate on a specific adversarial subset of one architecture/dataset, not that it is broadly untrustworthy.

## Score and Decision

**Calibration anchors:**

- **lTh7DEJV5W** (Memorization and Orders of Loss): Similar paper—proposes training-dynamics proxies for memorization with strong empirical correlations but limited validation against true LOO. Scored 3, 3, 3, 8 (avg ~4.25), rejected. The current paper has a similar pattern of correlating with an approximation rather than ground truth, but also contributes a practical efficiency improvement.

- **u9Z6gL5MlL** (Back to Fundamentals): Also studies natural memorization vs artificial memorization on CIFAR datasets. Similar limitations in scale and generalizability. Scored 3, 6, 3, 6 (avg ~4.5), rejected.

- **cMQeDPwSrB** (Memorization Through Curvature): Proposes a single-model proxy for memorization scores, shows cosine similarity with Feldman & Zhang baseline, same issue of validating against pLOO not LOO. Scored 5, 5, 5, 5, 6 (avg ~5.2), rejected. The current paper has a similar core contribution but adds a clear efficiency improvement and an attempt (however narrow) to validate against LOO.

- **KSjPa2eQhG** (Memorization in SSL): Accepted poster, scored 6, 6, 8, 6 (avg ~6.5). This paper had a clearer theoretical framework and broader evaluation.

The current paper's efficiency contribution (reducing 2000 to 200 shadow models) is solid and practically useful. Its core insight—that memorized points can be identified cheaply via ApB—is well-supported by strong correlations across multiple models and datasets. However, the accuracy claims are oversold: the LOO comparison is on a biased subset of a single small architecture, and the proxy validation has a circularity problem (validated against the same pLOO it claims is inaccurate). This places the paper below the accepted poster threshold but above the weakest rejects, which had fundamental methodological errors or near-zero empirical support. The efficiency contribution alone is meaningful, but the accuracy component is insufficiently substantiated to carry the weight claimed.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>