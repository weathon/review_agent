## Summary

This paper introduces a novel post-hoc out-of-distribution (OOD) detector that exploits the local plasticity of Kolmogorov-Arnold Networks (KANs). The core idea is to compare the activation outputs of a KAN trained on in-distribution (InD) data against those of its identically initialized untrained counterpart; samples that activate regions unchanged by training receive low InD scores. The authors pair this mechanism with a dataset-partitioning ensemble to capture joint feature distributions, and they report strong empirical results across seven benchmarks spanning image and tabular medical domains, including state-of-the-art overall AUROC on six of the seven.

## Strengths

- **Genuinely novel architectural angle for OOD detection.** The paper makes a fresh connection between spline-based local plasticity and OOD detection via initialization drift (Section 2.2, Eq. 2–4). The 1-D toy example (Figure 2) cleanly illustrates how the detector peaks on trained regions and attenuates elsewhere.
- **Strong and broad empirical results.** The KAN detector achieves the highest reported overall AUROC on the OpenOOD CIFAR-10 (94.12 vs. 93.37 for NAC; Table 1), ImageNet-200 FS (71.46 vs. 67.18 for ASH; Table 2), and ImageNet-1K FS (78.52 vs. 76.28 for NAC; Table 2) benchmarks, and ranks in the top three on all tabular medical benchmarks (Tables 3–5).
- **Unusual robustness to training-set size.** While competitors such as KNN and VIM collapse when training data is reduced to 0.1%, the KAN detector remains stable (93.21 AUROC on CIFAR-10 at 0.1% data; Table 6), a practically desirable property.
- **Effective partitioning strategy.** The proposed k-means-based partitioning to overcome the marginal-distribution limitation of univariate KAN activations yields a large empirical gain (CIFAR-10 AUROC rises from 46.08 without partitioning to 94.12 with 10 partitions; Table 7, Figure 3).

## Weaknesses

### Fatal
None. The algorithmic implementation is sound; the primary issues are in theoretical exposition and experimental completeness.

### Major

- **Incorrect theoretical factorization in Eq. 5 undermines the mechanistic narrative.** The paper claims Eq. 5 is an exact rewrite of Eq. 3 using Eq. 1:  
  $\Delta_{p,q}(x_p) = \sum_i |c_{p,q,i}^{\text{trained}} - c_{p,q,i}^{\text{untrained}}| \cdot B_i(x_p)$.  
  This step is invalid. Because B-spline basis functions overlap and coefficient differences can have mixed signs, $|\sum_i \Delta c_i B_i(x_p)| \neq \sum_i |\Delta c_i| B_i(x_p)$ in general (triangle inequality). The subsequent narrative—that absolute coefficient differences isolate “InD storage locations” while $B_i(x_p)$ acts as a spatial mask—rests on this false equality. The detector actually computes Eq. 3 directly, so the method works, but the theoretical explanation mischaracterizes its own mechanism. The authors should either correct the derivation or explicitly frame Eq. 5 as an approximation with stated assumptions.
  
- **Missing isolation of KAN specificity.** The central causal claim is that KAN *local plasticity* uniquely enables this detection strategy. Yet there is no baseline that applies the identical trained-vs-untrained protocol to a conventional MLP or even a single linear layer on the same latent features. Without this ablation, the paper cannot disentangle whether the gains stem from (i) the general protocol of comparing a feature-space network against its initialization, (ii) an implicit density-smoothing effect, or (iii) KAN-specific properties. The histogram ablation (Section 3.3) shows spline smoothing outperforms hard binning, but it does not isolate the architectural property being advertised.

- **Unexplained discrepancy between Table 1 and Table 6 for NAC.** Table 1 reports NAC’s CIFAR-10 overall AUROC as **93.37**, while Table 6 reports NAC at 100% training data as **87.05**—a drop of more than six points with no explanation offered in the text. If Table 6 uses a different subset of OOD datasets, a different averaging protocol, or a different evaluation setup, this must be stated explicitly; otherwise the dataset-size robustness narrative is undermined.

### Minor

- **Multi-layer feature access is not equalized across all baselines.** The paper transparently states that the KAN detector leverages multiple latent backbone layers following NAC (Liu et al., 2024a; Section 3.1). However, many standard OpenOOD baselines (MDS, KNN, VIM, ASH, Gram) are conventionally evaluated on the penultimate layer only. Because NAC also uses multi-layer integration, the head-to-head comparison with NAC is fair, but the broader leaderboard comparison conflates architectural novelty with richer feature access. The paper should either restrict KAN to the penultimate layer and report both conditions, or re-run all key baselines with the same multi-layer tensor.
- **Abstract overstates the Age benchmark.** The abstract claims “superior performance … across all seven benchmarks,” but the Age benchmark (Table 4) shows all methods near 50% AUROC—essentially chance. Ranking in the top three there is statistically meaningless and should not be folded into a blanket superiority claim.
- **Missing implementation details.** The main text omits the depth, width, and grid parameters of the KAN used in experiments (only the toy example specifies 1 layer × 200 coefficients). It also does not specify whether k-means partitioning is performed in raw input space, backbone latent space, or label space, which matters for reproducibility.

### Trivial

- Minor formatting inconsistencies in table headers do not affect readability.

## Nice-to-Has

- Add an MLP-initialization baseline trained on the same latent features and scored by $|\phi_{\text{trained}}(x) - \phi_{\text{untrained}}(x)|$ to test whether local plasticity is necessary.
- Report inference latency and parameter counts; the partitioning scheme requires $\mathcal{P}$ forward passes, which is costlier than single-pass baselines.
- Include a latent-space visualization (e.g., t-SNE colored by KAN InD score) to verify that high scores correlate with true InD density rather than spurious artifacts.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **“Unfair comparison due to multi-layer features” (full removal).** The criticism was partially overstated. The paper is transparent about using multi-layer features (Section 3.1) and explicitly follows Liu et al. (2024a), whose NAC method also uses multi-layer integration. The KAN-vs-NAC comparison is therefore fair. The broader leaderboard concern is downgraded to Minor because the paper does not hide its use of multi-layer features.
- **“MLPs lack locality entirely.”** The harsh critic correctly notes that MLPs exhibit some parameter-level locality (e.g., ReLU dead zones). However, KANs possess explicit input-space spline locality that is qualitatively different from the implicit, distributed locality of MLPs. The paper’s contrast, while slightly overstated, is not fundamentally misleading.
- **Missing appendix proofs and references.** Per instructions, these sections are stripped by the parser and exist in the original submission.
- **Formatting, grammar, and typographical criticisms.** These are parser artifacts, not author errors.

## Novel Insights

The paper’s most interesting insight—largely orthogonal to its own framing—is that comparing a shallow feature-space network against its random initialization can serve as a strong OOD detector. Whether this effect is primarily due to KAN-specific local plasticity or to the broader phenomenon of “initialization drift” in over-parameterized feature regressors remains an open question that the experiments do not fully resolve.

## Suggestions

1. **Correct or qualify Eq. 5.** Either derive a valid expansion or explicitly state that Eq. 5 is an interpretive upper bound/approximation valid when coefficient differences share the same sign.
2. **Explain the Table 1 vs. Table 6 NAC discrepancy.** If the evaluation protocols differ, state the difference clearly in the text.
3. **Add an MLP-with-identical-protocol baseline** in the ablation section to isolate the contribution of KAN-specific local plasticity.
4. ** temper the abstract claim** about “all seven benchmarks” to acknowledge the Age benchmark chance-level results.

## Score and Decision

**Calibration anchors used:**
- **High:** `/home/wg25r/review_agent/human_reviews/xUO1HXz4an.md` (NegLabel, avg 7.5, Accept spotlight) — stronger theory and more comprehensive baselines than the current paper; this paper sits below it due to the theoretical flaw in Eq. 5 and missing MLP ablation.
- **Medium:** `/home/wg25r/review_agent/human_reviews/ydlDRUuGm9.md` (KAN theory, avg 6.25, Accept poster) — comparable domain (KANs), cleaner theory but weaker empirical scope. The current paper has broader experiments but a theoretical error, placing it slightly below.
- **Medium:** `/home/wg25r/review_agent/human_reviews/ym0ubZrsmm.md` (SSOD, avg 5.33, Accept poster) — strong empirical results with some reviewer concerns about missing comparisons and ID-accuracy trade-offs. The current paper is stronger empirically and lacks the ID-accuracy drawback.
- **Low:** `/home/wg25r/review_agent/human_reviews/6Z8rZlKpNT.md` (Normalizing flows for OOD, avg 3.4, Reject) — rejected for missing baselines and poor organization. The current paper is substantially stronger in both experimental breadth and clarity.

**Reasoning:** The paper introduces a genuinely new OOD detection mechanism and backs it with strong, broad empirical results. The theoretical error in Eq. 5 is real and should be corrected, but it does not invalidate the algorithm or the empirical findings. The missing MLP baseline and the unexplained Table 1/Table 6 discrepancy are notable gaps that keep the paper from scoring higher. Relative to the anchors, it is above the rejected low-end papers and comparable to the accepted poster-tier work, though the theoretical flaw prevents it from reaching the high-end spotlight scores.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>