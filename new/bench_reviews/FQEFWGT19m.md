Now I have all the information needed. Let me synthesize the final review.

## Summary

This paper proposes a two-step multi-task learning (MTL) framework for handling block-wise missing data alongside distribution and posterior heterogeneity. The first step (HBI) imputes missing blocks by extracting disentangled shared and task-specific representations from an anchoring source common to all tasks. The second step (HMTL) decomposes the feature-to-response mapping into shared and task-specific components, using separate encoders and multiple regularizers including orthogonality, imputation downweighting, and disentanglement constraints.

## Strengths

- **Important and underexplored problem formulation**: The simultaneous consideration of block-wise missing data, distribution heterogeneity, and posterior heterogeneity in MTL is genuinely valuable. Prior methods typically handle at most one or two of these challenges. The paper correctly identifies this gap (Section 1, Section 2).

- **Conceptually sound disentanglement strategy**: Separating shared and task-specific representations for both imputation and prediction is well-motivated. The imputation regularizer (Eq. 7) that downweights imputed features relative to observed ones is a particularly practical mechanism to guard against error propagation from the first step.

- **Systematic synthetic experiments across multiple axes**: Settings A–F systematically vary covariance parameters, heterogeneity levels, sample sizes, dimensions, and noise levels (Section 4.1, Figure 5), providing clear evidence for when and why the method excels. The finding that HTL degrades under distribution heterogeneity (Figure 5(b)) while MTL-HMB maintains gains is informative.

- **Quantitative verification of heterogeneity in real data**: The MMD permutation test on the shared MRI sources (p = 10⁻⁶, Section 4.3) provides rigorous justification that distribution heterogeneity exists and must be addressed, and explains why HTL performs worse than STL on Task 2.

- **Visual evidence for disentanglement**: The t-SNE visualization in Figure 7 shows shared representations clustering together across tasks while task-specific representations separate, confirming the architecture captures the intended structure.

## Weaknesses

### Fatal
None.

### Major

- **Insufficient baseline comparisons**: Across all experiments, the method is compared against only two baselines: STL (a trivial lower bound) and HTL (Bica & van der Schaar, 2022). The related work section explicitly cites multiple methods designed for block-wise missing data — Xue & Qu (2021), Xue et al. (2021), Zhou et al. (2021), Gao & Lee (2017), Le Morvan et al. (2021) — yet none are included as baselines. HTL is a transfer learning method not designed for block-wise missingness; beating it shows the method handles heterogeneity but does not establish that it outperforms existing block-wise missing data solutions. Without comparison to these methods, the claim of "superior MTL performance" relative to the actual state of the art is unsupported. This is a significant evidential gap.

- **Narrow applicability vs. generality of claims**: The method assumes T tasks with exactly T+1 sources, where each task observes the anchoring source x₀ᵗ and exactly one task-specific source xₜᵗ (the "diagonal" missing pattern; Section "Problem Description"). While the paper states this assumption explicitly, the title ("Multi-Task Learning for Heterogeneous Multi-Source Block-Wise Missing Data") and abstract promise a general solution. Motivating examples like combining RCTs and observational data (Example 3) do not naturally fit this diagonal pattern. The gap between the generality implied and the narrow setting delivered is significant and cannot be resolved by minor revision.

### Minor

- **Notation errors in core equations**: The term x₀⁰ is used repeatedly (lines 68, 76, 78, 84) where x₀ᵗ is clearly meant — the superscript 0 is not a valid task index. Additionally, in the reconstruction loss (Eq. 1, line 98), the second summation iterates over n₋ₜ samples from tasks r≠t, but all data references use x_{0,i}^t (task t's data) when they should reference data from the other tasks. The intent is recoverable from the figures and surrounding context, but these errors create ambiguity in the method specification and hinder reproducibility.

- **Underpowered real-data evaluation without statistical tests**: The ADNI experiment has 72 and 69 samples per task, with a 20% test split yielding ~14–15 test samples per task. The standard deviations in Table 1 are large relative to means (SD/Mean ≈ 0.22–0.27). While 30 random splits are reported, no confidence intervals or p-values are provided for the RMSE differences between methods, making it impossible to assess whether the claimed "17.28% improvement" in Task 2 is statistically meaningful.

- **No evaluation of imputation quality**: The paper never reports how well HBI imputes the missing blocks (e.g., imputation RMSE on held-out features). If imputation is poor, the second step's gains could be coincidental; if imputation is excellent, simpler imputation methods might suffice. This makes it difficult to attribute the downstream gains specifically to the proposed disentanglement mechanism.

- **No ablation of key architectural choices**: The design choice of extracting shared representations solely from x₀ᵗ (Eq. 3) while using all features for task-specific representations (Eq. 4) is not ablated. Similarly, the orthogonality regularizer (Eq. 6) and the imputation downweighting regularizer (Eq. 7) are not individually evaluated. Without these ablations, it is unclear whether the specific disentanglement architecture matters or whether the improvements come primarily from the two-step pipeline structure.

### Trivial
None.

## Nice-to-Haves

- Comparison against at least one existing block-wise missing data method (e.g., Xue & Qu 2021) to establish the method's value relative to the actual state of the art.
- Statistical significance tests (paired tests or bootstrap CIs) for the ADNI results.
- Ablation study testing variants of the shared/task-specific representation split.
- Demonstration or discussion of what happens when the missing pattern deviates from the strict diagonal structure.
- Evaluation of imputation quality on held-out features in the synthetic experiments.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **MMD p = 10⁻⁶ is "suspicious"**: The harsh reviewer claimed that with n≈70 samples, such an extreme p-value suggests miscalibration. However, with 267-dimensional MRI features and genuine distribution heterogeneity, MMD tests in high dimensions routinely yield very small p-values when real distribution differences exist. This is not suspicious — it is consistent with the significant heterogeneity the paper correctly identifies.

- **x₀⁻ᵗ "never clearly defined as a data object"**: The paper defines x₀⁻ᵗ = {x₀ʳ}_{r≠t} at line 68. While the exact concatenation convention is not spelled out, this is a minor clarity issue, not an undefined term.

- **E_p⁻ᵗ "collapses heterogeneity"**: The reviewer claimed that having a single encoder E_p⁻ᵗ for all tasks r≠t "collapses them into a monolithic group." While this is a valid observation, the architecture already has separate E_pᵗ and E_p⁻ᵗ encoders, and the shared encoder E_c handles the cross-task shared component. This is a design trade-off, not a fundamental flaw.

- **DGP exchangeable covariance makes imputation "trivially easy"**: The correlation structure (ρ)^{0.01|i-j|} is tested with ρ as low as 0.5 (Setting A), where correlations are weak. The imputation task is not trivially easy in all tested regimes. This criticism is overstated.

- **Element-wise squaring is "very mild" nonlinearity**: This is a standard choice for synthetic experiments that allows controlled variation of nonlinearity through α.

- **Example 3 (RCT+observational) is "inappropriate"**: While the diagonal missing pattern doesn't perfectly fit the RCT/observational setting, the example illustrates posterior heterogeneity, which is one of the three challenges addressed. The paper does not claim to directly apply to this setting.

- **Missing appendix / missing proofs**: The DGP for multi-task experiments and the algorithm pseudocode are deferred to the appendix. The parser strips appendices; these exist in the original submission.

## Novel Insights

The observation that standard transfer learning (HTL) degrades to single-task-level performance under distribution heterogeneity (Figure 5(b)), while the proposed imputation-based approach maintains gains, is an important empirical finding. It suggests that addressing distribution heterogeneity through imputation before MTL is more robust than directly transferring representations — a design principle that could inform other multi-task settings with distribution shift.

## Suggestions

- Add comparisons with at least 2–3 block-wise missing data baselines from the cited related work. This is the single most impactful improvement that could strengthen the paper.
- Report imputation quality (RMSE on held-out features) in the synthetic experiments to attribute downstream gains to the imputation mechanism.
- Correct the notation errors: replace x₀⁰ with x₀ᵗ throughout Section 3.1, and fix the reconstruction loss (Eq. 1) second term to use data from tasks r≠t rather than task t.
- Report statistical tests or confidence intervals for the ADNI results, or acknowledge the limited statistical power given the small sample size.

## Calibration Anchors

| Paper Path | Avg Score | Comparison |
|---|---|---|
| /home/wg25r/review_agent/human_reviews/8EyRkd3Qj2.md (CLAP) | 7.50 | Similar topic (block-wise missing + heterogeneity + imputation), but handles more general missing patterns, has better baselines, and more comprehensive evaluation. Our paper is clearly weaker. |
| /home/wg25r/review_agent/human_reviews/GsR3zRCRX5.md (RISE) | 6.17 | Joint imputation+inference with neural processes. Had some weaknesses but solid experiments and clear method. Our paper is weaker due to missing baselines and narrower scope. |
| /home/wg25r/review_agent/human_reviews/ZWthVveg7X.md (PIG) | 4.33 | MTL with imputation and prediction via graph learning. Had analysis issues but reasonable experiments. Similar profile to our paper — reasonable idea with insufficient depth. |
| /home/wg25r/review_agent/human_reviews/a4O528mek9.md (Mul2vec) | 3.00 | Incomplete multi-modal data, poor writing and unclear method. Our paper is clearly better — the method is understandable and the problem is well-motivated. |
| /home/wg25r/review_agent/human_reviews/1gqR7yEqnP.md (Pan for Gold) | 2.20 | Overclaimed, no clear method, terrible writing. Our paper is much better. |
| /home/wg25r/review_agent/human_reviews/OXi1FmHGzz.md (VFL) | 6.00 | Vertical federated learning with missing features. Better baselines and clearer scope. Our paper is weaker. |

**Score reasoning**: The paper is clearly above the low anchors (2.2–3.0) because it has a well-defined method, clear problem formulation, and reasonable experiments. It is clearly below the high anchors (6.0–7.5) because of the narrow scope relative to claims, missing baselines from the actual state of the art, and notation errors. It sits closest to the PIG paper (4.33) which also had a reasonable idea with insufficient experimental depth. The missing baselines for block-wise missing methods and the scope mismatch push it slightly below a borderline paper, as these are the two most important factors for assessing the contribution.

## Score and Decision

The paper addresses an important and underexplored problem — simultaneously handling block-wise missing data, distribution heterogeneity, and posterior heterogeneity in MTL. The disentanglement idea is conceptually sound and the synthetic experiments are systematic. However, two major weaknesses significantly undermine the contribution: (1) the method is compared against only two baselines, neither of which addresses block-wise missing data, despite the paper citing several such methods; (2) the framework is limited to a specific diagonal missing pattern that doesn't match the generality of the title and abstract. These issues, combined with notation errors in core equations and an underpowered real-data evaluation, mean the paper does not convincingly establish its contribution relative to the actual state of the art. The problem and approach have merit, but the evaluation falls short of what is needed.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>