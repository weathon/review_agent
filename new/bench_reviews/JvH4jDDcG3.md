Now I have enough context to write my final review. Let me consolidate based on my careful reading of the paper and the various reviewer inputs.

Key things to verify from the harsh critic:

1. **Theorem claims**: The harsh critic says the theory doesn't support calibration claims. Let me check: Theorem 1 says confidence penalty only occurs in unreliable regions, Theorem 2 says ECE improves. The "mild conditions" in Theorem 2 are indeed deferred to appendix. The critic is partially right - the theoretical grounding IS weak (the "reliable" vs "unreliable" regions are defined by K-means, not by ground truth), but the method empirically works well.

2. **Calibration loss design**: The harsh critic argues the loss is essentially smoothing, not true calibration. This is a fair observation - the target $\hat{q}_k$ is derived from the very overconfident clustering head. But the paper's argument is that averaging within mini-clusters provides a softer target that acts as a regularizer. The critic's point that this is heuristic rather than principled calibration is valid.

3. **ECE metric**: The paper does not specify bin count or binning strategy. The "5× improvement" claim is cherry-picked.

4. **Role of calibration head in clustering improvement**: The ablations don't isolate calibration from dynamic thresholding. This is a valid point.

5. **Proposition 1**: It is indeed near-tautological.

Let me also check: does the paper actually claim to beat supervised baselines? Yes - "CDC-Cal with the same pre-trained model (MoCo-v2) increases accuracy by 0.8% and 2.5% than supervised learning on CIFAR-10 and STL-10". This comparison is somewhat apples-to-oranges since supervised learning here means linear evaluation with MoCo-v2 features, not fully supervised training.

Now for my final review:</think>## Summary

The paper proposes a Calibrated Deep Clustering (CDC) framework that addresses the overconfidence problem in deep clustering through a dual-head architecture (clustering head + calibration head). The calibration head uses K-means mini-clusters on features to create smoothed targets that reduce overconfidence in unreliable regions, while the clustering head leverages calibrated confidences for dynamic per-class pseudo-label selection. A feature prototype-based initialization strategy for the MLP heads is also introduced. Experiments on six benchmarks demonstrate state-of-the-art clustering accuracy and substantially lower ECE compared to prior deep clustering methods.

## Strengths

- **Identifies a genuine and important problem**: The overconfidence issue in deep clustering is real and previously overlooked. The reliability diagrams in Fig. 1 convincingly show that SCAN and SPICE are severely miscalibrated, and the paper makes a compelling case that standard calibration methods (Temperature Scaling, Label Smoothing) are either inapplicable or harmful in the unsupervised setting.

- **Strong empirical clustering performance**: CDC-Cal achieves SOTA clustering accuracy on nearly all benchmarks, with particularly notable improvements on challenging datasets like ImageNet-Dogs (+11.7% over SPICE-3) and Tiny-ImageNet (+4.8%). The method is consistently strong across datasets.

- **Effective initialization strategy**: The feature prototype-based initialization (using K-means centroids to initialize MLP weights) is simple, well-motivated, and demonstrably impactful. Table 2-I shows dramatic differences (e.g., CIFAR-20: 10.4% → 56.4% after initialization), and even the final system degrades substantially without it (61.7% → 44.4%).

- **Thorough ablation analysis**: Table 2 provides ablations covering initialization, threshold strategy, single-head variants, stop-gradient, and K sensitivity. The dual-head design and stop-gradient are shown to meaningfully affect both ECE and ACC.

- **Practical value of calibrated confidence**: The error-rejection metrics (AUROC, AURC, FPR95) in Fig. 3 demonstrate that CDC genuinely improves the ability to separate correct from incorrect predictions, which is practically valuable for deployment decisions in clustering.

## Weaknesses

### Fatal
None.

### Major

- **Theoretical claims overstate what is actually proven**: Theorems 1 and 2 are presented as "solid theoretical guarantees" for calibration improvement, but they rest on key assumptions that are neither stated in the main text nor obviously reasonable. Theorem 1 defines "reliable" and "unreliable" regions based on K-means partitions relative to clustering decision boundaries — not ground-truth labels — and merely shows that average confidence decreases in boundary-crossing regions. It does NOT show that confidence becomes aligned with true accuracy. Theorem 2 claims $ECE^{fcal} \leq ECE^{fclu}$ under unspecified "mild conditions" deferred to the appendix. Since ECE is defined against ground-truth accuracy, and the calibration targets $\hat{q}_k$ are derived from the overconfident clustering head itself, the conditions under which true calibration error improves are far from trivial. The calibration mechanism is effectively a prediction-smoothing heuristic, not a principled estimator of true error probabilities. The paper should present this contribution honestly as a well-motivated heuristic rather than dressing it in claims of "theoretical guarantees."

- **The calibration head's contribution to clustering accuracy is not cleanly isolated**: Table 2 shows that the initialization strategy alone contributes a massive portion of the performance improvement (CIFAR-20: 10.4% → 56.4% just from initialization). The remaining ablations compare CDC-Cal against artificially weak baselines (single-head with overconfident self-selection, randomly initialized heads, fixed global thresholds). There is no ablation that tests a **dynamic per-class thresholding scheme using the clustering head's own confidences** (without calibration) — a natural alternative that could provide similar benefits. Without this control, it remains unclear whether the accuracy gains come from the calibration mechanism per se or from the adaptive sample selection that any reasonable dynamic threshold would provide.

- **ECE measurement protocol is underspecified and the "5×" claim is selectively framed**: The paper does not specify the number of bins, binning strategy, or other ECE computation details. ECE is well-known to be sensitive to bin count and binning method. The "5× on average" improvement claim compares CDC-Cal against extreme outliers (e.g., SPICE-2 on CIFAR-20 at 52.3% ECE vs CDC-Cal at 4.9%), while on Tiny-ImageNet, CDC-Cal's ECE (11.0%) is substantially worse than CC's (3.2%). The claim should be contextualized rather than presented as a headline figure.

### Minor

- **Hyperparameter K requires dataset-specific tuning**: K ranges from 40 to 1000 across the six datasets (a 25× range), and the paper provides no guidance for selecting K on a new dataset without labels. The ±20% sensitivity analysis (Fig. 5) shows modest ACC variation but says nothing about how to set the base value. This limits practical applicability.

- **Proposition 1 is essentially tautological**: Setting MLP weights equal to K-means centroids ensures that the network's initial output preserves nearest-prototype assignments — this is a straightforward consequence of the construction, not a deep theoretical insight. Labeling it alongside Theorems 1 and 2 as part of the "theoretical guarantees" inflates the theoretical contribution.

- **No comparison with adaptive threshold baselines**: The fixed-threshold comparisons (0.99, 0.95, 0.90, 0.80 in Table 2-II) are useful but limited. A comparison against simple adaptive strategies (e.g., percentile-based thresholding, entropy-based selection, or the confidence-aware selection used in semi-supervised learning) would better isolate the contribution of the calibration head.

### Trivial
None.

## Nice-to-Haves

- Report ECE with different bin counts (e.g., 10, 15, 20) and consider classwise-ECE or SCE to strengthen calibration evaluation robustness.
- Provide computational cost analysis (training time, memory) relative to SCAN/SPICE, since online K-means per batch adds overhead.
- Test CDC with alternative self-supervised backbones (e.g., DINO, SwAV) to assess generality beyond MoCo-v2.
- Provide guidance on K selection (e.g., a heuristic based on dataset size, cluster structure, or feature entropy).

## Removed Points

- **ECE comparison with supervised baselines is claimed to be unfair because supervised models don't receive post-hoc calibration**: The paper compares against "Supervised + MoCo-v2" (a standard linear evaluation protocol) and shows CDC-Cal has lower ECE on some datasets. However, this comparison is not misleading per se — it shows that an unsupervised method with explicit calibration can produce better-calibrated predictions than a standard supervised baseline, which is an informative finding. Whether post-hoc calibration would close the gap is an empirical question, not a flaw in the current comparison.

- **Missing standard deviations across multiple runs**: While error bars would strengthen the paper, single-run evaluation is the norm in deep clustering papers (SCAN, SPICE, etc. report single runs). Demanding error bars beyond the community standard is scope creep.

- **Comparison with unsupervised calibration methods like PseudoCal or adaptations of temperature scaling**: These methods are designed for domain adaptation with access to source data, not for pure unsupervised clustering. The paper explicitly discusses why post-calibration methods are inapplicable (no labeled validation set). While applying pseudo-label-based temperature scaling would be an interesting comparison, it is beyond the paper's scope.

- **Concerns about mini-cluster purity and circular dependency**: The neutral reviewer raises whether impure K-means clusters yield noisy calibration targets. This is a valid direction for analysis, but the empirical results (strong ECE reduction and improved ACC) already demonstrate the method works despite impurity. The method is precisely designed to handle this by averaging (smoothing) rather than requiring pure clusters.

- **Scalability to truly large datasets**: The paper already includes ImageNet-Dogs and Tiny-ImageNet. Demanding full ImageNet-1K evaluation when the community standard (SCAN, SPICE, etc.) only evaluates on similar or smaller scales is generic.

- **The "5×" claim cherry-picks SPICE-2 on CIFAR-20**: Partially valid as a major point (the framing is selective), but it's noted above under the ECE protocol concern rather than removed entirely — the "5×" claim is selectively framed, just not entirely fabricated.

## Novel Insights

The most novel insight from the reviews is that the calibration mechanism's theoretical grounding is substantially weaker than presented. The "region-aware penalty" (Theorem 1) only shows confidence reduction in K-means regions near decision boundaries — a geometric property, not a guarantee of alignment with true accuracy. The real empirical contribution is dual: (1) a practical initialization trick that transfers pre-trained discriminability to the clustering head, and (2) a regularized dual-head training scheme that produces smoother, more discriminative confidence estimates. The paper's value lies in these practical contributions and the problem identification (overconfidence in deep clustering), not in the theoretical claims. The fact that the initialization alone explains much of the clustering accuracy gain (Table 2-I) suggests the calibration head's primary marginal contribution is indeed calibration (ECE reduction), while the initialization drives much of the clustering performance.

## Suggestions

- Rewrite the theoretical contribution framing: present the theorems as providing intuition for why smoothing via K-means mini-clusters reduces overconfidence, rather than as "solid guarantees" of true calibration improvement. Be transparent about the gap between what is proven and what is claimed.
- Add an ablation with a dynamic per-class thresholding baseline using the clustering head's own softmax confidences (without the calibration head) to isolate the contribution of calibration to clustering accuracy.
- Contextualize the "5× ECE improvement" claim with per-dataset ECE values and acknowledge datasets where CDC does not achieve the best ECE (e.g., Tiny-ImageNet).

## Evaluation

- **Originality**: High — the problem of calibration in deep clustering is genuinely novel and important.
- **Importance of research question**: High — overconfidence is a real and underappreciated problem with practical consequences.
- **Claims well supported**: Partially — empirical results are strong, but theoretical claims overreach what is actually demonstrated. The causal role of calibration in improving clustering accuracy is under-isolated.
- **Soundness of experiments**: Adequate but could be stronger (missing key ablation, underspecified ECE protocol).
- **Clarity of writing**: Good — the paper is well-organized and clearly describes the method.
- **Value to the community**: Good — practical method for an important problem, though the framing needs correction.

**Calibration against reference papers**: The P^2OT paper (deep imbalanced clustering, scores 6/6/6, accepted poster) introduced a novel problem setting and OT-based solution with strong experiments but had some weaknesses in computational cost and missing comparisons. The COME paper (entropy minimization for test-time adaptation with overconfidence concern, scores 6/6/5/8, accepted poster) addressed overconfidence with a motivated method and good experiments but had limited novelty and missing error bars. The PseudoCal paper (calibration in unsupervised domain adaptation, scores 5/5/6/6, rejected) had heuristic motivation with limited theoretical grounding. The Energy Calibration Head paper (scores 3/5/3/5, rejected) had promising ideas but weak evaluation and unconvincing methodology.

CDC is significantly stronger than PseudoCal and ECH on empirical grounds (6 benchmarks, comprehensive ablations) and identifies a genuinely novel problem (overconfidence in deep clustering). Its main weakness is the overclaimed theory and the under-isolated ablation for clustering accuracy, not fundamental methodological flaws. It is roughly comparable in contribution quality to P^2OT (novel problem, strong results, some weaknesses in ablation completeness) and COME (addresses overconfidence, practical method, good experiments).

**Score: 6** — The paper makes a genuine and important contribution by identifying and addressing overconfidence in deep clustering with a practical and effective method. The empirical improvements are substantial and consistent. However, the theoretical claims are overstated relative to what is actually proven, and the specific contribution of the calibration head to clustering accuracy (vs. the initialization and adaptive thresholding) is insufficiently isolated. These issues prevent a higher score but do not undermine the paper's core value.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>