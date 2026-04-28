## Summary
This paper introduces Learned Reference-based Diffusion Sampler (LRDS), a variational diffusion method that learns a multi-modal reference distribution from mode-localized MCMC samples to address hyperparameter sensitivity in existing samplers. The method comes in two variants: GMM-LRDS (Gaussian Mixture Model reference) and EBM-LRDS (Energy-Based Model reference). Experiments demonstrate GMM-LRDS achieves substantially lower mode weight estimation errors than competing methods in dimensions up to d=64, while EBM-LRDS better captures complex non-Gaussian geometries on 2D benchmarks.

## Strengths
- **Strong high-dimensional performance with GMM-LRDS**: Table 2 shows GMM-LRDS maintains low mode weight estimation error (2.7% ± 0.8% at d=32, 4.1% ± 0.6% at d=64) while competing variational methods (LV-PIS, LV-DDS) degrade to 30%+ errors, indicating genuine mode collapse in baselines.
- **Fair and informed baseline comparison**: Section 5 explicitly ensures all competing methods receive the same prior information (mode locations via $\hat{\pi}^{\text{ref}}$) for hyperparameter tuning, preventing the common pitfall of comparing an informed method against uninformed baselines.
- **Clear identification of reference distribution sensitivity**: Figure 1 effectively isolates why existing methods (LV-PIS, LV-DDS) are sensitive to the reference variance hyperparameter $\sigma$, with error rising from ~0 to ~0.35 as $\sigma$ deviates from optimal, motivating the learned reference approach.
- **EBM-LRDS captures complex geometries**: Figure 3 demonstrates EBM-LRDS successfully samples from the Rings distribution (three concentric ring modes) where GMM-LRDS produces scattered samples that fail to capture the sharp ring geometry.

## Weaknesses

### Fatal
None

### Major
- **EBM-LRDS validated only on 2D distributions**: The EBM-LRDS variant, introduced to handle "harder sampling problems" where GMMs fail (Section 3.3), is evaluated exclusively on 2D toy problems (Rings in Figure 3, Checkerboard in Figure 6). No results demonstrate EBM-LRDS outperforming GMM-LRDS in higher dimensions (d > 10) where GMM approximations typically degrade. This leaves the claim that EBM-LRDS addresses challenging geometries unsupported for the high-dimensional settings where the method would be most needed. This is a significant gap given that similar 2D-only evaluation was heavily criticized in comparable sampling papers (e.g., hvT2vfxD84, avg score 2.00).

- **Narrow applicability due to known-mode assumption**: The entire LRDS pipeline requires initializing MCMC chains at target mode locations to obtain reference samples $\hat{\pi}^{\text{ref}}$ (Section 3, Step (a)). While the paper explicitly states this assumption (Introduction, line 23), it fundamentally restricts applicability to problems where modes are analytically known or easily identified. The motivation citing Bayesian statistics and molecular dynamics is overstated since mode locations are rarely known in these fields. This limitation is structural to the method design and significantly narrows the scope of the claimed contribution to "sampling for multi-modal distributions with known modes." Comparable papers with warm-start assumptions received borderline scores (e.g., 5MyDW1hzL9, avg 4.50).

### Minor
- **No computational efficiency analysis**: LRDS introduces an additional pre-training stage (learning $\pi^{\text{ref}}$ via EM or EBM training) before the main diffusion sampler optimization. The Discussion acknowledges this "computational cost" (line 240), but Section 5 provides no runtime comparisons, FLOP counts, or wall-clock time metrics against baselines. Given that competing methods like LV-PIS or LV-DDS do not require learning a reference model, it is unclear if the accuracy gains justify the increased training overhead. Missing runtime analysis is a common weakness in sampling papers that typically prevents scores from reaching the 7+ range (e.g., mzlScdrryB, avg 5.20).

- **Limited robustness analysis to mode location errors**: Since the method assumes known mode locations, there is no evaluation of how LRDS performs when mode locations are perturbed or approximate. In practice, mode locations might only be approximately known; testing sensitivity to initialization errors would strengthen the practical relevance of the method.

### Trivial
- **Figure 5 caption ambiguity**: The caption states "Reasonable results could not be obtained with PDDS due to numerical issues" but does not clarify whether this applies to all methods shown or only PDDS.

## Nice-to-Haves
- Include at least one experiment with d ≥ 16 where GMM-LRDS fails (e.g., highly non-Gaussian modes) and EBM-LRDS succeeds, to validate the EBM contribution in higher dimensions.
- Add a table or plot comparing training and sampling wall-clock time for LRDS vs. baselines to quantify the accuracy-efficiency trade-off.
- Discuss how LRDS could be combined with global optimization techniques to relax the "known mode" assumption in future work.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Criticism about "not yet released" models or benchmarks**: None present in the reviews.
- **Criticism about missing appendix or proofs**: The parser strips appendices; these exist in the original submission.
- **Criticism about unfair comparison favoring baselines**: The asymmetry in this paper favors baselines (they are given the same prior information), which is appropriate.
- **Generic strength about "important problem"**: Dropped as per guidelines—only concrete, evidence-backed strengths retained.
- **Harsh critic's point about abstract being "misleading"**: The abstract accurately states the assumption; this is a presentation nitpick rather than a substantive issue.
- **Harsh critic's point about Section 6 contradicting the assumption**: The Discussion mentions protein Boltzmann sampling as future work, not a current claim—this is a minor framing issue, not a contradiction.

## Novel Insights
The paper's core insight—that variational diffusion samplers' hyperparameter sensitivity stems from using a fixed, uninformative reference distribution—is well-supported by Figure 1's ablation. However, this observation builds directly on Richter et al. (2023)'s Log-Variance framework rather than introducing fundamentally new theory. The genuine contribution is the practical pipeline for learning a multi-modal reference from mode-localized samples, which is methodologically sound but scope-limited. No insights beyond the paper's own contributions emerged from the reviews.

## Suggestions
1. **Temper claims about applicability**: Reframe the contribution as a solver for "multi-modal distributions with known modes" rather than a general multi-modal sampler. This aligns the scope with the assumptions and prevents overclaiming.
2. **Add high-dimensional EBM validation**: Include at least one experiment with d ≥ 16 demonstrating EBM-LRDS's advantage over GMM-LRDS for non-Gaussian mode geometries.
3. **Report computational costs**: Add runtime comparisons (training time, sampling time, or energy evaluations) to allow readers to assess the accuracy-efficiency trade-off.

## Score and Decision

**Calibration anchors consulted:**
- **High-scoring (≥6)**: XTHQqS7ObC (avg 6.50, Accept Poster) - Proximal Diffusion Neural Sampler with strong experiments on both continuous and discrete tasks, clear novelty, and good empirical validation. This paper has stronger empirical breadth than the paper under review.
- **Medium-scoring (~5)**: 38Ey1FrSDt (avg 5.00, Reject) - Adaptive Destruction Processes for Diffusion Samplers with trainable destruction processes but limited continuous-time analysis. 5MyDW1hzL9 (avg 4.50, Reject) - Sampling with warm starts, explicitly assumes known mode locations, similar scope limitation to this paper. mzlScdrryB (avg 5.20, Reject) - Dyson Diffusion Model with missing computational complexity analysis.
- **Low-scoring (≤4)**: hvT2vfxD84 (avg 2.00, Reject) - Importance Weighted Score Matching for Diffusion Samplers, heavily criticized for 2D-only evaluation on a sampling paper. This paper's EBM-LRDS limitation is similar but less severe since GMM-LRDS has strong high-dim results.

**Positioning**: The paper under review has stronger high-dimensional results (GMM-LRDS up to d=64) than the medium-scoring anchors, but the EBM-LRDS 2D-only limitation and the known-mode assumption prevent it from reaching the high-scoring range. The GMM-LRDS contribution is solid and well-validated, comparable to papers scoring 5.5-6.0. However, the structural scope limitation (known modes required) is similar to 5MyDW1hzL9 (4.50), and the missing runtime analysis is similar to mzlScdrryB (5.20). The EBM-LRDS gap is less severe than hvT2vfxD84's 2D-only evaluation since GMM-LRDS carries the main contribution.

**Final score**: 5.5 — This is a borderline paper with genuine strengths (strong GMM-LRDS results, fair baselines, clear motivation) but notable weaknesses (EBM-LRDS not validated in high dimensions, narrow applicability due to known-mode assumption, missing efficiency analysis). The score reflects that the core contribution is sound but the scope must be aligned with assumptions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>