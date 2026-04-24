Now let me search for calibration papers to establish score anchors.Now I have enough information to synthesize the final review. Let me also check one more calibration for borderline papers in the method/application space.Now I have all the information I need. Let me compose the final synthesized review.

---

## Summary

This paper introduces Corruption2Self (C2S), a score-based self-supervised MRI denoising framework grounded in a Generalized Denoising Score Matching (GDSM) loss. GDSM extends denoising score matching to the ambient noise setting—where only noisy observations are available—and formally unifies DSM, ADSM, and Noisier2Noise as special cases (Remark 1, Section 3.1). The framework incorporates a reparameterization of noise levels for training stability and a detail refinement extension to mitigate over-smoothing. Experiments on M4Raw (real noise) and fastMRI (simulated noise) demonstrate state-of-the-art performance among self-supervised methods and competitive results against supervised baselines.

---

## Strengths

- **GDSM provides a clean theoretical unification** (Theorem 1 / Remark 1, Section 3.1): The paper formally proves that GDSM recovers the conditional expectation E[X_{t_target} | X_t] from noisy data alone, and shows that DSM (σ_target = σ_t_data), ADSM (σ_target = 0), and Noisier2Noise (fixed noise level, σ_target = 0) are all special cases. This conceptual unification is a genuine contribution regardless of the incremental distance from ADSM.

- **Single-contrast results on real MRI are genuinely strong (Table 2)**: C2S with detail refinement achieves 32.77/0.919 on M4Raw T1, outperforming all self-supervised methods (next best is Noise2Noise at 32.59/0.911) and equaling or exceeding supervised methods SwinIR (32.53/0.913), Restormer (32.35/0.912), and supervised Noise2Noise (32.59/0.911) — all trained on noisy labels, without any clean targets. This directly validates the main claim that C2S bridges the self-supervised/supervised gap in real MRI.

- **Detail refinement is statistically validated (Table 1)**: Paired t-tests show significant PSNR/SSIM improvements across all three M4Raw contrasts (e.g., T1 PSNR: 34.56→34.89, p=0.001; FLAIR SSIM: 0.812→0.818, p=0.005). This substantiates the non-zero σ_target strategy for mitigating over-smoothing.

- **Comprehensive empirical coverage**: The evaluation spans two public datasets (M4Raw with real noise, fastMRI with simulated noise), three contrasts, two noise levels, and a large basket of baselines including classical (NLM, BM3D), supervised (SwinIR, Restormer, Noise2Noise), and seven self-supervised methods. This breadth is above average for MRI denoising papers.

- **Robustness to noise level estimation errors (Appendix H)**: The paper demonstrates stable performance with ±50% estimation error, enabling practical "blind" denoising via standard noise estimation tools.

---

## Weaknesses

### Fatal
None.

### Major

- **No direct ADSM + same architecture baseline**: By Remark 1, GDSM with σ_target = 0 is mathematically identical to ADSM (Daras et al., 2024). The paper correctly notes this, but then provides no experiment comparing "ADSM trained directly with the same U-Net/NVC-MSA architecture" against C2S. The performance advantage over Noisier2Noise (a special fixed-noise-level case) shows that continuous noise levels help, but this does not isolate what reparameterization and architecture contribute *beyond* directly applying ADSM. Without this baseline, the novelty of GDSM as a new method—rather than as a new application of ADSM—is not experimentally substantiated. The paper risks being characterized as "ADSM + reparameterization + U-Net applied to MRI."

- **Multi-contrast claim of "outperforming supervised methods" is not fairly supported (Table 5 vs. its framing)**: Table 5 compares multi-contrast C2S (which receives additional contrast inputs) to *single-contrast* supervised baselines (Noise2Noise, BM3D). The introduction states C2S achieves "state-of-the-art performance among both self-supervised and supervised methods" in this setting, but there is no multi-contrast supervised baseline. A trained multi-contrast Noise2Noise or SwinIR would likely narrow or close this gap. Importantly, the *core* single-contrast results in Table 2 already support the claim of competitiveness with supervised methods—so this issue primarily affects the multi-contrast framing rather than the paper's overall conclusion.

### Minor

- **Ablation table (Table 4a) metrics are not directly comparable to main results (Table 2)**: Table 4a reports validation PSNR of 34.43 dB for T1 with reparameterization at 200 epochs, while Table 2 reports test PSNR of 32.77 dB. The paper explicitly notes these are 200-epoch validation results (3-rep avg labels vs. 6-rep avg test labels), but the magnitude of the ablation effect (31.14 → 34.43, a 3.3 dB gap) cannot be confirmed to reflect fully converged test performance. It would be useful to confirm that models *without* reparameterization remain strictly inferior at convergence, not just at 200 epochs, to rule out reparameterization being purely a speed benefit.

- **PDFS σ=13/255 PSNR result is selectively reported**: In Table 3, C2S (w/ detail refinement) achieves PSNR of 30.91 vs. Recorrupted2Recorrupted's 30.95—the only setting where a baseline outperforms C2S on PSNR. The paper correctly discloses this and justifies via SSIM advantage (0.756 vs. 0.752), but the discussion should acknowledge this more directly rather than using the SSIM metric selectively to claim best performance.

### Trivial
None beyond parser artifacts.

---

## Nice-to-Haves

- A direct comparison of "ADSM (Daras et al.) + our U-Net/NVC-MSA architecture" vs. C2S would clarify precisely what the reparameterization and continuous noise level add over the prior work. This experiment is the most valuable addition the authors could make.
- A multi-contrast supervised baseline (e.g., Noise2Noise trained with multi-contrast inputs) in Table 5 would properly substantiate the multi-contrast claim.
- An analysis of convergence with and without reparameterization at full training (1000 epochs) would confirm whether reparameterization is a speed improvement or also a final-performance improvement.
- A case where the Gaussian noise assumption breaks down (very low-SNR region) would strengthen the honesty of the robustness claims.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic "Rician noise / no VST experiment"**: The paper explicitly acknowledges the Gaussian approximation and its conditions (SNR > 2), mentions VST as an option, and provides robustness empirical evidence (Appendix H). The criticism that VST is never applied is correct, but the paper frames this as empirical robustness rather than theoretical correctness—and the SNR > 2 condition is generally met in the MRI data used. This is within the paper's stated scope; it is a nice-to-have, not a critical flaw.

- **"Reparameterization idea is not unique"**: The harsh critic argues the reparameterization idea appears in prior score-based models (Karras et al., 2022, which the paper cites). The paper itself cites this and uses similar weighting—this is a case of proper attribution, not overclaiming novelty. Removed as it mischaracterizes the paper.

- **"All visual comparisons show success; show a failure case"**: The request for a failure case is a useful suggestion but belongs in nice-to-haves, not a weakness.

- **Strength Finder — "comprehensive comparison landscape"**: This is too generic (applies to any paper with a large table). Removed as not paper-specific.

- **Strength Finder — "Robustness to noise level estimation" as a core strength**: The appendix-based robustness result is useful but is specifically an engineering property, not a core claim. Kept as supporting evidence rather than a standalone strength.

---

## Novel Insights

The most valuable observation across the reviews is the potential of ADSM—originally developed to reduce memorization in large generative models—as a principled foundation for self-supervised discriminative denoising. The reparameterization insight (shifting sampling from noise level *t* to the added-noise increment *τ*) is a clean engineering trick that likely has applicability beyond MRI to any ambient-noise score matching setting. The detail refinement extension (non-zero σ_target as a tuneable PSNR-vs.-perceptual-quality dial) is simple and practically significant for medical imaging, where oversmoothing can destroy diagnostically relevant fine structure. The formal unification of Noisier2Noise, ADSM, and DSM under a single GDSM family is conceptually clarifying for the field, even if GDSM's primary operating mode (σ_target = 0) is mathematically equivalent to ADSM.

---

## Suggestions

1. **Add ADSM + same architecture as an ablation baseline** — this is the single most important experiment to add. It directly validates whether the performance improvement comes from the reparameterization/continuous-noise-levels or from the architecture, and separates the theoretical contribution from engineering choices.
2. **Either add a multi-contrast supervised baseline to Table 5 or soften the multi-contrast "beats supervised" framing** — the single-contrast Table 2 already supports the competitive-with-supervised claim, so the multi-contrast headline is not needed; or it could be properly substantiated with a fair comparison.
3. **Report ablation results at convergence (not just 200 epochs)** — a training curve showing that models without reparameterization do not close the gap even with extended training would strongly validate reparameterization as a capability, not just a speed, improvement.

---

## Score and Decision

**Calibration anchors retrieved:**

| Path | Avg Score | Comparison |
|---|---|---|
| OdnqG1fYpo (Moner, MRI unsupervised INR) | **7.50** (Accepted Spotlight) | Higher-novelty problem (no training data at all), harder task; this paper is less novel but more theoretically grounded |
| jsBhmOCKYs (Denoising as Adaptation) | **5.80** (Accepted Poster) | Applied denoising framework using diffusion models; similar empirical scope; C2S has more theoretical depth |
| z8PcUSKXXN (Random Noise Injection Denoising) | **5.75** (Accepted Poster) | Applied denoising framework for generalization; good results; similar contribution level |
| 2XBBumBGeP (sRGB Real Noise Modeling) | **6.50** (Accepted Poster) | Noise modeling with strong empirical results; comparable breadth; similar gap to theory |
| JZgqoOu4Ml (Diffusion priors for Bayesian 3D) | **4.00** (Rejected) | Applied diffusion priors with weaker baselines and limited validation; this paper is clearly stronger |
| mbPvdO2dxb (Meta-Guided Diffusion for medical imaging) | **5.00** (Rejected) | Zero-shot MRI with diffusion models; missing key experiments; this paper has better experimental support |

**Assessment relative to anchors**: This paper is clearly above the 4.00–5.00 range (it has real theory, real experiments, and competitive results). It sits in the zone of accepted posters at 5.75–6.50. Its theoretical contribution (GDSM unification) and single-contrast real-MRI results place it closer to 6.0, but the missing ADSM baseline and multi-contrast comparison gap prevent it from scoring higher. The single-contrast Table 2 results are genuinely strong and the paper is more theoretically grounded than the ~5.80 denoising-as-adaptation paper. I place the paper at **6.0**, corresponding to a marginal accept — solid contribution, real gaps addressable in revision.

**Axis evaluation:**
- *Originality*: Moderate — GDSM is a real generalization of ADSM, but the relationship is close enough that novelty hinges on the engineering/application contributions
- *Importance of research question*: High — self-supervised MRI denoising with no clean labels is a clinically important problem
- *Claims well supported*: Largely yes for single-contrast, less so for multi-contrast supervised comparison
- *Soundness of experiments*: Good — real and simulated noise, multiple contrasts, comprehensive baselines; gap is the missing ADSM baseline
- *Clarity of writing*: Good — well-organized, theory and experiments clearly presented
- *Value to research community*: Moderate-to-high — the GDSM unification and MRI-specific contributions are useful

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>