Now I have verified all the key claims. Let me compile my final review.

## Summary

TabDiff proposes a joint continuous-time diffusion framework for mixed-type tabular data generation that directly models numerical features via Gaussian diffusion and categorical features via masked diffusion in the original data space. Its key innovations are feature-wise learnable noise schedules (parameterized by ρ_i for numerical and k_j for categorical features) that adapt to feature-level distribution heterogeneity, and a mixed-type stochastic sampler that corrects accumulated decoding errors during reverse sampling. The framework also extends to conditional generation via classifier-free guidance for missing value imputation.

## Strengths

- **Principled mixed-type framework operating in original data space**: Unlike TabSyn (which encodes to latent space) and TabDDPM/CoDi (which use discrete-time processes), TabDiff combines continuous-time Gaussian and masked diffusion directly in the data space, avoiding encoding overhead and yielding tighter ELBO. The factorized forward process (Eq. 1) with joint denoising is a clean and natural design.

- **Feature-wise learnable noise schedules are well-motivated**: The observation that tabular features have highly heterogeneous marginal distributions — unlike image channels or text tokens — is correct and sharp. Per-feature schedules (Eq. 10, Eq. 11) are a natural response. The ablation in Table 5 shows clear gains: with the stochastic sampler, Trend error drops from 1.93 (fixed) to 1.80 (learnable), and Figure 2 confirms reduced training loss.

- **Consistent and substantial empirical improvements**: TabDiff outperforms all baselines on 5/7 datasets for Shape and 7/7 for Trend (Tables 1–2). The raw Trend improvements over TabSYN are consistently positive across all datasets, with the average Trend error dropping from 2.33 to 1.80 (22.6% improvement on the correctly computed average).

- **Stochastic sampler contribution is cleanly demonstrated**: Table 5 isolates the contributions of both the stochastic sampler and learnable schedules, showing both contribute meaningfully (e.g., Shape: 1.39→1.17, Trend: 2.29→1.80 when both are enabled).

- **Lightweight CFG extension for conditional generation**: The classifier-free guidance framework (Eqs. 13–16) reuses the unconditional model, requiring only a small additional model for unconditional probabilities over target columns, making conditional generation a natural byproduct rather than a separate pipeline.

## Weaknesses

### Fatal
None.

### Major

- **No analysis of what the learned noise schedules actually learn — the paper's central innovation is unvalidated mechanistically**: Feature-wise learnable noise schedules are presented as the key innovation, with the claimed benefit that they "enable the model to optimally allocate capacity to different features" and "encourages the model to capture inherent correlations during sampling since the model can denoise different features in a flexible order." However, the paper never reports what values ρ_i and k_j converge to, never analyzes whether learned schedules correlate with feature properties (e.g., do features with heavier tails get different ρ values?), and never demonstrates that "flexible denoising order" actually occurs (e.g., no visualization of which features become clean earlier in the reverse process). The ablation (Table 5, Figure 2) only shows that learnable schedules reduce training loss and improve Shape/Trend, but not *why* or *how* they help. Without this analysis, the improvement could simply come from the extra per-feature parameters adding expressiveness rather than from any meaningful adaptation to feature distributions. This gap between the claimed mechanism and the empirical evidence is significant.

- **Inconsistent improvement percentages in Tables 1 and 2 undermine confidence in the precision of experimental reporting**: Several improvement percentages do not match the raw numbers in the tables. Verified discrepancies include: (1) Table 1, Magic Shape: TabSYN=1.03, TabDiff=0.78 → expected improvement (1.03−0.78)/1.03 ≈ 24.3%, but reported as 14.29% (which equals exactly 1/7); (2) Table 1, Diabetes Shape: (1.85−0.89)/1.85 ≈ 51.9%, reported as 46.39%; (3) Table 2, Beijing Trend: (3.13−2.59)/3.13 ≈ 17.3%, reported as 4.4%; (4) Table 2, Diabetes Trend: (3.90−2.20)/3.90 ≈ 43.6%, reported as 37.3%. The correctly computed averages (13.3% Shape, 22.6% Trend) are consistent with the raw numbers, so the underlying results appear sound. However, these errors in derived quantities — particularly the 14.29% = 1/7 entry, which strongly suggests a copy-paste or spreadsheet error — undermine confidence in the care taken with the experimental section.

### Minor

- **Baseline comparison fairness**: Most baseline results are copied from Zhang et al. (2024) while only TabSYN is reproduced by the authors. Additionally, several baselines (GReaT, STaSy) show OOM on larger datasets, and the averages are computed over different numbers of datasets for different methods, making them not directly comparable. However, TabDiff's improvements over the competitive TabSYN (which was reproduced) are large and consistent enough that this is unlikely to change the overall ranking.

- **Imputation evaluation only against TabSYN**: Table 4 compares TabDiff only against TabSYN for missing value imputation, with XGBoost as a reference only. For a claimed contribution (conditional generation via CFG), a comparison against more generative baselines would strengthen the evaluation.

- **Ablation conducted on a single dataset**: Table 5 and Figure 2 appear to report ablations only on the Adult dataset. The relative contributions of stochastic sampling vs. learnable schedules may differ across datasets with different characteristics.

- **Notational inconsistency between Eq. 5 and Algorithm 2**: Eq. 5 trains μ_θ^num to predict ε (noise), but Algorithm 2 line 12 uses μ_θ^num in the formula dx = (x − μ_θ^num)/σ, which is the standard form when the denoiser predicts x_0 (clean data). While the ε-prediction and x_0-prediction parameterizations are equivalent via x̂_0 = x_t − σ·ε̂, the algorithm box as written is misleading about what the model outputs.

### Trivial

- The "≈" in "α_0 ≈ 1 and α_1 ≈ 0" (Section 2.2) is technically appropriate for the general exponential parameterization but becomes exact equality under the specific schedule in Eq. 11. The wording is defensible in context.

## Nice-to-Haves

- Report the learned ρ_i and k_j values per feature for at least one dataset, and correlate them with feature properties (distribution shape, entropy, number of categories). This would directly validate the core mechanism claim.
- Visualize the learned noise schedules (σ(t) curves) for several features to make the core innovation visually concrete.
- Visualize per-feature denoising dynamics (which features denoise first/last) to validate the "flexible denoising order" claim.
- Ablate the stochastic sampler separately for numerical vs. categorical features to understand where error correction matters most.
- Report computational cost (training time, sampling time, memory) for practical adoption considerations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unfair comparison with baselines because baselines not re-run"**: Per the rules, I should not treat asymmetry that favors baselines (not the authors' method) as a weakness. However, the OOM exclusion issue is a different concern (affects average comparability) and is retained as a Minor weakness above.

- **Missing related works / concurrent work Mueller et al. (2024)**: The paper actually does cite Mueller et al. (2024) in Section 3 (Related Work), noting it "also proposed feature-wise diffusion schedules, but the model still relies on encoding to continuous latent space." The harsh critic's claim about missing related work is partially addressed by the paper itself.

- **Shape and Trend metrics are proprietary/not standard**: This is a scope-creep criticism. The paper follows the evaluation protocol of Zhang et al. (2024), which is the established benchmark in this area. Requesting different metrics is a nice-to-have, not a weakness.

- **MLE unreliability**: The authors themselves acknowledge this limitation in Section 4.3: "methods with varying performance on data fidelity metrics might have very close MLE scores." This is already addressed.

- **CFG normalization for categorical features**: The harsh critic notes that p̃_θ from Eq. 16 is not guaranteed to be normalized. This is a well-known property of CFG in log-probability space and is typically handled by softmax/renormalization. The omission of explicit mention is a trivial presentation issue.

- **Formatting/notation nitpicks**: Issues about the "≈" in boundary conditions and other minor notational inconsistencies are trivial.

- **Missing appendix proofs/details**: The parser strips appendices; these exist in the original submission.

## Novel Insights

The most important insight from this review is the disconnect between TabDiff's mechanism claims and its evidence. The paper asserts that learnable schedules "adapt to feature heterogeneity" and enable "flexible denoising order," but the experimental section only demonstrates that they improve metrics — not why. This is a pattern common in diffusion model papers where a parameterized component improves results but the interpretability of what was learned is neglected. For TabDiff specifically, a simple analysis of learned ρ_i and k_j values (which are readily available post-training) and their correlation with feature statistics could transform this from an empirical improvement into a genuine mechanistic insight, potentially revealing which feature properties matter most for schedule design.

## Suggestions

- Report the learned schedule parameters (ρ_i, k_j) for each feature on at least one dataset, ideally alongside feature statistics (skewness, kurtosis, entropy). This is low-effort (the values are already learned) but high-impact for validating the core claim.
- Fix the four incorrect improvement percentages in Tables 1 and 2 (Magic Shape, Diabetes Shape, Beijing Trend, Diabetes Trend).
- Re-run at least TabDDPM and CoDi (the most competitive diffusion baselines after TabSYN) under identical conditions to strengthen the comparison.
- Visualize per-feature denoising trajectories during sampling to demonstrate whether the "flexible denoising order" claim holds.

## Score and Decision

**Calibration anchors:**

1. **CDTD** (QPtoBPn4lZ, avg 5.5, Accept Poster): Concurrent work on tabular diffusion with adaptive noise schedules. Very similar topic and approach. CDTD was criticized for limited novelty and incremental contributions, receiving scores of 6, 5, 5, 6. TabDiff is arguably stronger: it operates in original data space (avoiding encoding overhead), includes the stochastic sampler and CFG contributions, and shows more consistent improvements. However, TabDiff has the calculation errors and lacks schedule analysis.

2. **TabSyn** (4Ay23yeuz0, avg 6.75, Accept Oral): The prior SOTA that TabDiff compares against. TabSyn was accepted as oral despite being somewhat incremental, because it was well-executed with strong results. TabDiff is comparable in contribution level but has the reporting errors and missing analysis.

3. **Block Diffusion** (tyEyYT267x, avg 8.0, Accept Oral): A much stronger paper with genuine novelty, thorough variance analysis, and strong results. TabDiff is clearly below this level — it lacks the depth of mechanistic understanding that Block Diffusion provides.

4. **TimeAutoDiff** (zB6uMznFuZ, avg 3.0, Reject): Weak paper with limited novelty and significant methodology concerns. TabDiff is clearly above this level.

TabDiff sits between CDTD (5.5) and TabSyn (6.75). It is stronger than CDTD due to its cleaner framework design and additional contributions (stochastic sampler, CFG), but weaker than TabSyn due to the calculation errors and missing analysis of the core mechanism. The paper makes genuine contributions and shows strong results, but the gap between mechanism claims and evidence, combined with the reporting errors, holds it back from a higher score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>