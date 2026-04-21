## Summary

This paper conducts a systematic empirical study of normalization type, position, and pairwise combinations within the Mamba block, evaluating five normalization methods (BN, LN, GN, IN, RMSN) on long-sequence modeling (Breakfast, LRA ListOps) and image classification (ImageNet-100, ImageNet-1k). It finds that post-SSM normalization generally outperforms pre-SSM placement, and that heterogeneous normalization pairs (e.g., IN→LN, RMSN→BN) can outperform homogeneous pairs.

## Strengths

- **Systematic combinatorial sweep across normalization design decisions.** The paper evaluates five normalization types at two positions and all 25 pairwise combinations, producing the most comprehensive controlled map of this design space to date (Tables 1–4). This is substantiated with evidence across both sequence modeling and vision tasks.
- **Discovery that heterogeneous normalization combinations can outperform homogeneous ones.** Specific mixed pairs such as IN→SSM→LN (72.5% on ListOps) and RMSN→SSM→BN (87.3% on ImageNet-100) surpass their uniform counterparts, a potentially useful empirical pattern for practitioners.
- **Timely focus on a real gap.** Existing Mamba variants adopt normalization ad hoc; a controlled ablation is needed and this paper fills that niche.

## Weaknesses

### Fatal
None

### Major
- **Vision validation on ImageNet-1k uses VMamba stripped of its FFN module, weakening practical evidence for the vision recommendation.** Section 4.5 explicitly states that the ImageNet-1k baseline is “the original VMamba’s normalization configuration **without FFN module for fair comparison**.” The 0.3% improvement (70.8% → 71.1%) over this simplified architecture does not demonstrate whether the proposed RMSN→SSM→BN configuration remains beneficial on the full VMamba model that practitioners actually use. Because the paper frames its contribution as delivering “practical recommendations,” this validation gap is a significant structural limitation in the vision half of the study.

### Minor
- **No variance or statistical estimates are reported for any results.** Every entry in Tables 1–5 is a single scalar. Small claimed improvements—e.g., ImageNet-1k 70.8% → 71.1%, ImageNet-100 86.8% → 87.3%—cannot be reliably distinguished from training noise without standard deviations or confidence intervals.
- **L2-norm mechanistic analysis does not explain the headline combinations.** Figure 4 visualizes only four BN-related configurations (None→None, BN→None, None→BN, BN→BN) on ListOps, and Figure 5 examines a single combination (BN→IN) at a single layer. Neither visualization covers the actual best-performing heterogeneous combinations reported in the main results (IN→LN for sequence, RMSN→BN for vision), so the norm-based analysis is disconnected from the paper’s strongest empirical findings.
- **“Training stability” claims rely on indirect evidence.** The abstract, introduction, and conclusions frame the work around improving “training stability” and “robust training dynamics,” but the evidence consists of final test accuracy and post-hoc L2-norm snapshots (Figure 4) rather than direct measurements such as training loss curves, gradient norm trajectories, or divergence/failure rates. Weight-norm distributions are a relevant proxy, but they are insufficient to fully substantiate stability as a primary contribution.
- **Theoretical intuition is anchored on a narrow observation and a tentative disclaimer does not compensate for limited support.** The “harmonic structure” claim is derived from one layer of one combination on one dataset (Figure 5). While Section 4.6 explicitly notes that the explanation “is not intended as an essential explanation,” the proposed practical guidelines would be more credible if accompanied by broader systematic evidence rather than a single anecdotal case.

### Trivial
- **Presentation issues in Section 4.5.** The Table 5 caption contains copy-paste errors (repeating “For vision tasks” and mislabeling the sequence configuration as IN→SSM→IN rather than the reported IN→SSM→LN).

## Nice-to-Haves
- Direct stability measurements (e.g., training loss curves, gradient norms) for representative configurations to substantiate the abstract’s stability framing.
- Variance estimates from multiple independent training runs, at minimum for the marginal comparisons in Tables 4 and 5.
- L2-norm visualizations of the actual best heterogeneous combinations (IN→LN, RMSN→BN) to connect the mechanistic analysis to the headline results.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **“None→SSM→None is a strawman.”** This is a standard ablation baseline. The fact that the model catastrophically fails without normalization (7.0% on Breakfast) is informative about normalization’s necessity, not a dishonest comparison.
- **“Abstract contradicts best configurations.”** The abstract explicitly states that using normalization “after the SSM module (if used only once) and combining different normalization layers before and after the SSM module can enhance training stability.” This is consistent with the results; there is no contradiction.
- **“Original Mamba baseline should use LayerNorm, not RMSN.”** The original Mamba architecture (Gu & Dao, 2023) uses RMSNorm; the paper’s baseline is correct.
- **“Related work is an undifferentiated list” and style/formatting nits.** These are editorial matters outside the core evaluation.
- **“Missing training hyperparameters in the main text.”** The appendix has been stripped by the parser; hyperparameters may be present in the original submission.
- **Complaints about missing statistical tests and per-normalization hyperparameter tuning as fatal flaws.** These are methodological desiderata, but single-run reporting is common in this subfield; they do not invalidate the core empirical sweep.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Re-run the ImageNet-1k validation using the complete VMamba architecture (with FFN). If the proposed normalization cannot outperform the full baseline, the vision recommendation should be explicitly scoped to the studied configuration or revised.
- Extend the L2-norm visualization to cover the top heterogeneous combinations (IN→LN and RMSN→BN) so the mechanistic analysis actually explains the headline results.
- Report variance across at least 3 runs for the key comparisons in Tables 4–5.

## <context>
**Original reviewer signal:** The Harsh Critic called for rejection, citing a structurally flawed ImageNet-1k baseline (missing FFN), unsupported stability claims, absent variance/ significance testing, and a misapplied theoretical citation as fatal to an empirical paper. The Strength Finder supported acceptance, highlighting the systematic sweep, heterogeneous-combination discovery, and timely focus on a real gap.

**What was dropped and why:** Several criticisms were refuted on cross-check. The “strawman baseline” and “abstract contradiction” claims were removed because the baseline is a legitimate ablation and the abstract explicitly addresses both single and combined normalization. The criticism that the original Mamba uses LN rather than RMSN was removed as factually incorrect. Related-work style complaints and appendix-deferred hyperparameters were removed as non-substantive or parser-related.

**Cross-checks performed:** (1) Verified the no-FFN statement in Section 4.5 (line 286): “LN→SSM→LN represents the original VMamba’s normalization configuration without FFN module for fair comparison.” (2) Checked abstract wording: it explicitly covers both single-post-SSM and combined configurations, so the alleged contradiction is a misread. (3) Confirmed the original Mamba uses RMSNorm, not LayerNorm. (4) Verified the L2-norm analysis limits: Figure 4 analyzes only four BN-centric configurations, not the headline heterogeneous combinations. (5) Confirmed single-scalar reporting throughout all tables.

**Severity read:** The surviving weaknesses are mostly minor; the one major issue is the ImageNet-1k validation on a VMamba variant lacking its FFN module, which weakens—but does not invalidate—the practical evidence for the vision recommendation. The core empirical sweep (combinatorial mapping on Breakfast and ImageNet-100) and the ListOps validation (on the original Mamba architecture) remain sound. No single weakness threatens the paper’s entire core claim, though the vision half of the generalization argument is materially weakened.

**Anything else load-bearing:** The paper explicitly downplays its theoretical “intuitive explanation” as “not intended as an essential explanation” (Section 4.6, line 290), which tempers the severity of the mechanistic criticism. The FFN modification appears to be a controlled design choice rather than an unfair comparison (the “fair comparison” phrasing suggests both configurations share the same simplified architecture), but the consequence is that the vision recommendation is untested on the practical full model.
</context>