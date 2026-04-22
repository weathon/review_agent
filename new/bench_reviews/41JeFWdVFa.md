Now I have sufficient information to write the final review. Let me consolidate my analysis.

## Summary

LDP proposes a lightweight denoising autoencoder (DAE) plug-in that improves single-image super-resolution generalization via LR prediction-based cyclic regularization. The method models degradation within a DAE framework, uses LR high-frequency components (y_hf) as a condition to distinguish different LR images from the same HR, and operates in two modes: as a training-time auxiliary loss for fine-tuning SR models, or as an inference-time posterior sampling correction for diffusion models.

## Strengths

- **Consistent improvements across diverse architectures and degradations (Table 3):** LDP fine-tuning improves all four baselines (FeMaSR, StableSR, SwinIR, MambaIR) across all five synthetic degradation categories, with PSNR gains from +0.05 dB to +2.16 dB. This breadth of evaluation across GAN, diffusion, transformer, and state-space model architectures provides strong evidence for architecture-agnostic applicability.

- **Dual-mode applicability with empirical validation:** LDP functions both as a training-time loss (Tables 3–4) and as an inference-time posterior sampling correction (Table 5). The inference-time application requires no retraining of the diffusion model, which is practically valuable. For example, StableSR+LDP shows MANIQA gains of +0.0092 and MUSIQ gains of +1.45 on RealSR (Table 5).

- **Lightweight and efficient design:** Section 4.1 specifies 642K parameters and ~16 hours of training on a single RTX A6000, significantly more efficient than competing degradation models like Lway that the paper notes introduces "significant computational overhead due to its large model size."

- **Good diagnostic validation against trivial collapse (Table 2):** The paper explicitly tests whether LDP collapses to simple downsampling by measuring similarity between generated LR and downsampled SR images. LDP shows significantly lower similarity than DRN (which behaves almost identically to bicubic downsampling), confirming the model is performing non-trivial degradation modeling.

- **Patch-dependent noise schedule (Eq. 7):** Assigning each patch a random timestep enables spatially varying degradation modeling, a meaningful departure from prior methods assuming uniform degradation across the image.

## Weaknesses

### Fatal

None.

### Major

- **LR_hf conditioning creates potential information leakage that is not experimentally tested.** The condition $y_{hf} = y - y\downarrow_{s'}\uparrow_{s'}$ (Eq. 4) provides the denoiser with direct access to the high-frequency content of the target LR image $y$. This means the predicted LR $y'$ can match $y$ partially by reading information from $y_{hf}$ rather than requiring the SR output $x'$ to encode it through genuine degradation modeling. The authors acknowledge this in Section 6 — "the generated LR image inevitably retains information from the input LR high-frequency components" — but treat it as a minor limitation rather than a structural concern warranting experimental validation. **No ablation tests the conditioning mechanism** (e.g., replacing $y_{hf}$ with noise, removing it entirely, or measuring gradient attribution from each input). Without this, it is impossible to assess whether the cycle consistency genuinely regularizes the SR model or is trivially satisfied through the condition. This matters because if the cycle consistency loss is largely satisfied through $y_{hf}$, the core mechanism claim — that LDP constrains the SR solution space via meaningful degradation modeling — is weakened, even though the empirical improvements may still hold.

- **The "generalization to unseen degradations" claim is partially overclaimed.** The paper trains LDP with BSRGAN degradation patterns and fine-tunes SR models with BSRGAN degradation patterns (Section 4.1). The synthetic testing uses "bsrgan plus" (BSRGAN + Real-ESRGAN), which overlaps with the BSRGAN training distribution — this is not genuinely unseen. On real-world benchmarks (Table 4), results are inconsistent: FeMaSR+LDP degrades on CLIPIQA for RealSR (−0.1163) and DPED (−0.1960), and on MUSIQ for DPED (−5.07). The authors explain these degradations by arguing "such metrics may favor visually striking but structurally inaccurate results," but this selective interpretation of metrics is not fully convincing without additional evidence (e.g., user studies). A stronger test of generalization would evaluate on degradation types genuinely outside the training distribution.

### Minor

- **The "learned filters approximate blur kernels" claim (Abstract, Section 3.2) is unsupported by evidence.** The abstract states the denoiser "uses learned filters to approximate blur kernels," and Section 3.2 says it "estimates the blur kernel." No analysis of the learned convolutional filters, comparison to known blur kernels, or validation of this mechanistic claim is provided. Given the information leakage concern, the denoiser may not need to learn meaningful degradation-specific kernels at all. This is an overclaim in the framing rather than a flaw in the method.

- **Limited baselines for LR prediction comparison (Tables 1–2).** DRN only handles bicubic degradation and takes no conditional input; DualSR struggles with diverse degradations. The comparison establishes that LDP is better than two weak baselines, but does not demonstrate competitiveness against stronger degradation models. However, the paper's main contribution is the plug-in framework, not LR prediction per se.

- **Missing ablations for key design choices.** The patch-dependent timestep (Eq. 7), the noise level range [500, 1000], and alternatives to the $y_{hf}$ conditioning mechanism are not ablated. The ablation study (Section 5) only varies loss terms and $\tau$.

- **Mixed results in posterior sampling mode (Table 5).** Many metrics show marginal or negative improvements when LDP is applied to pretrained diffusion models (e.g., LDM+LDP degrades on MANIQA, CLIPIQA, MUSIQ for RealSR; UPSR+LDP degrades on QAlign for RealSRSet). The inference-time benefits are less convincing than the fine-tuning benefits.

### Trivial

None.

## Nice-to-Haves

- An ablation replacing $y_{hf}$ with random noise or removing it entirely would directly test how much the conditioning contributes versus the noisy HR input, and would substantially strengthen or clarify the mechanistic claims.
- Evaluation on a genuinely out-of-distribution degradation pipeline (e.g., a degradation model not used during training) would more convincingly support the "unseen degradations" claim.
- Visualization or analysis of the learned convolutional filters to validate whether they approximate known blur kernels.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic's claim that the paper does not support unpaired degradation modeling as a limitation inherent to y_hf conditioning requiring "addressing it or proposing an alternative conditioning strategy":** This is scope creep — the paper explicitly acknowledges this as a limitation (Section 6), and demanding a solution to an acknowledged limitation goes beyond what the paper claims to address.

- **Harsh critic's request for failure case analysis:** While always useful, this is a generic request applicable to any paper. Not including failure cases is standard practice and not a substantive weakness.

- **Harsh critic's complaint about the "noise-subtraction technique relegated to Appendix E":** Appendices are legitimate locations for supplementary details; this is not a weakness of the paper.

- **Strength Finder's claim about "universal hyperparameter configuration" as a strength:** This is generic and not strongly evidenced — the ablation only tests on SwinIR with one dataset, not across all architectures.

- **Strength Finder's claim about "real-world benchmark validation" as a strong supporting strength:** This conflicts with the verified Major weakness showing inconsistent results on real-world benchmarks. The improvements are present but mixed, not consistently validating generalization.

## Novel Insights

The paper introduces an interesting design tension: the y_hf condition is both necessary (to distinguish different LR images from the same HR, satisfying the paper's own Criterion 2) and potentially problematic (creating information leakage that may undermine the cycle consistency mechanism). This trade-off between discriminativeness and independence of the conditioning signal is a fundamental challenge for conditional degradation models that the field has not adequately addressed. The paper's empirical success despite this tension suggests that even partial cycle consistency (constraining primarily the low-frequency domain) provides meaningful regularization for SR models, which is a useful empirical finding even if the claimed degradation modeling mechanism is not fully validated.

## Suggestions

- Conduct the critical ablation: train LDP with $y_{hf}$ replaced by Gaussian noise of the same dimensionality, and compare the resulting fine-tuning improvements. This single experiment would clarify whether the method's benefits come from genuine degradation modeling or the condition shortcut.
- Moderate the "unseen degradations" language to "degradations beyond the training distribution" or similar, since the synthetic test set partially overlaps with training.
- Either provide evidence for the "approximate blur kernels" claim (filter visualization, comparison to known kernels) or remove/rephrase it as a design motivation rather than a demonstrated property.

## Score and Decision

**Calibration anchors compared:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| pvq53fGnRq (plug-in posterior diffusion SR) | 5.0 | Reject | LDP is more comprehensive with consistent gains; pvq53fGnRq had marginal improvements and theoretical issues |
| IOmPy7P1y4 (SAVL, degradation representation for RWSR) | 5.6 | Accept Poster | Similar scope and limitations — both model degradation for SR, both have limited validation of the degradation representation mechanism |
| 66Ad0i78lW (DM-SR, bridging distribution gap for SR) | 5.0 | Accept Poster | Similar quality of contribution — practical plug-in approach with some unsupported mechanistic claims |
| 9T1agMpZ8i (DGMS, domain generalization for Mamba SR) | 2.5 | Withdrawn/Reject | LDP is clearly better — has real empirical improvements across multiple architectures, not just overclaimed generalization |
| 8CDZkq0ayI (UCD, unconditional discriminator) | 2.5 | Withdrawn/Reject | Shares the "shortcut/leakage" concern but LDP has much stronger empirical evidence |
| fu0NN8GRQ7 (VAE-CycleGAN) | 2.0 | Reject | Cycle consistency shortcut concern, but LDP is far stronger empirically |
| 7UfZAxKo5K (bidirectional cycle consistency) | 7.0 | Accept Poster | Much stronger validation of the cycle consistency mechanism with theoretical analysis; LDP doesn't reach this level |

LDP sits in the medium range: it has real, consistent empirical improvements (stronger than the rejected pvq53fGnRq) but has an unresolved structural concern about information leakage in the conditioning mechanism (weaker than the accepted 7.0+ papers that validate their mechanisms thoroughly). The paper is comparable to IOmPy7P1y4 (5.6) which had similar limitations in validating degradation representations. The missing conditioning ablation is the key gap preventing a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>