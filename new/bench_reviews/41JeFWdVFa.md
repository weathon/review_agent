The paper content is already provided in the user message. Let me now carefully verify key claims from the reviewers against the paper content I have.

Key points to verify:

1. **Harsh Critic Issue #1 (noisy HR ≈ noisy LR imported from DR2 without justification)**: The paper does cite Wang et al. (2023b) (DR2) and states "it leverages a property of diffusion models, where after noise is added, high-resolution (HR) images and LR features become aligned." The paper uses timesteps [500, 1000] which is stated. The paper doesn't provide theoretical justification or empirical verification of this alignment in the SR setting. This is a valid concern.

2. **Harsh Critic Issue #2 (y_hf condition)**: The paper defines y_hf = y - y↓s'↑s' and argues for three criteria. The concern about whether y_hf retains too much information (near-complete LR) is interesting but needs careful thought. The paper explicitly notes the condition should NOT be the LR image itself. The high-frequency residual still contains significant structural info. This is a valid concern about potential information leakage.

3. **Harsh Critic Issue #3 (evaluation limited to BSRGAN-style degradations)**: The paper trains on BSRGAN degradations and tests on BSRGAN/Real-ESRGAN synthetic data PLUS real-world datasets (RealSR, RealSRSet, DPED). The claim of "unseen degradations" is partially validated by real-world tests, but the synthetic tests are within-distribution. The wording could be more careful. This is a partially valid concern.

4. **Harsh Critic Issue #4 (circularity with LR input)**: This is a valid and important concern. The loss L_FT_sym uses y_hf derived from the same LR y that serves as the target. But the paper explicitly addresses this in the limitations section: "It does not support unpaired degradation modeling, as the generated LR image inevitably retains information from the input LR high-frequency components." The paper acknowledges this. However, the concern that improvements might come from "shortcut copying" rather than genuine degradation modeling is substantive.

5. **Harsh Critic Issue #5 (weak baselines for LR prediction)**: The comparison with DRN and DualSR in Table 1 is a proof-of-concept comparison, not a comprehensive baseline evaluation. DRN was designed for bicubic degradation. However, this comparison is only for LR prediction validation, not the main contribution. The main experiments compare baseline vs. baseline+LDP, which is appropriate.

6. **Neutral Reviewer's concern about posterior sampling**: Looking at Table 5, LDM+LDP does show some regressions (NIQE +0.179, MANIQA -0.0094, CLIPIQA -0.0245 on RealSR). ResShift+LDP shows near-zero gains. This is a valid observation that the inference-time mode is less effective.

7. **Missing computational cost analysis**: The paper states 642k parameters but doesn't report FLOPs or inference time. This is a minor weakness.

8. **Harsh Critic's claim about no comparison with simpler regularizers**: This is a valid concern. There's no ablation comparing LDP to a simple LR-consistency loss with a standard downsampler.

Let me now write the final review.</think>Now I have enough information to write the consolidated review.

## Summary

LDP proposes a lightweight denoising autoencoder (DAE) plug-in for single-image super-resolution that improves generalization to unseen degradations via LR prediction-based cyclic regularization. It models the SISR degradation process within a DAE framework, leveraging a diffusion-model property that noisy HR and LR features become aligned, and conditions on LR high-frequency components to distinguish different LR images from the same HR. LDP can operate as a training-time loss or an inference-time post-processing module for diffusion models.

## Strengths

- **Principled and creative framework**: Integrating degradation modeling into a DAE by exploiting noise-induced alignment of HR and LR features is novel. The use of LR high-frequency components as a condition to resolve the one-to-many mapping is well-motivated, and Table 2 demonstrates LDP-generated LR images have low similarity to simple downsampled SR outputs, confirming it does not collapse to trivial downsampling.

- **Broad empirical validation across architectures**: LDP is evaluated on four SR architectures (FeMaSR, StableSR, SwinIR, MambaIR) and four diffusion models (LDM, StableSR, ResShift, UPSR) across five synthetic degradation types and three real-world datasets. The consistent improvements—particularly StableSR+LDP on Hybrid (+2.16 PSNR, Table 3)—demonstrate genuine practical utility as a plug-in.

- **Lightweight and flexible design**: With only 642K parameters and dual-mode applicability (training-time loss and inference-time correction), LDP introduces minimal architectural overhead and can be applied to diverse SR models without modifying their architecture. The proposed loss components (L_sym^FT and L_fre) can be configured with universal hyperparameters (τ=100, λ=1).

- **Comprehensive ablation analysis**: Tables 6-7 and Appendix F provide ablations on loss terms, τ, patch size, and frequency band selection, showing the contributions of each component.

## Weaknesses

### Major:

- **The core conceptual justification—noisy HR/LR alignment—is borrowed from DR2 without adaptation-level validation for SR degradation modeling**: The paper's central mechanism relies on the DR2 observation that adding sufficient noise aligns HR and LR distributions, enabling "denoising noisy HR features" to approximate "denoising noisy LR features." However, DR2 established this property in a specific face restoration context, while LDP applies it to general SISR with blur+noise+compression+downsampling degradation chains. The paper provides neither theoretical justification nor empirical verification (e.g., measuring HR_t vs LR_t feature distances across timesteps) that this alignment holds under BSRGAN-style degradations. The arbitrary choice of timestep range [500, 1000] lacks principled justification. The success of the method could be driven by the cyclic regularizer rather than by the claimed diffusion-based alignment. This undermines the methodological narrative, even if the method itself works empirically.

- **Potential information leakage through the y_hf condition**: The condition y_hf = y − y↓s'↑s' retains most LR structural information (removing only the lowest frequencies). Since the LDP loss compares the predicted LR to the same y from which y_hf is derived, there is a circularity: y_hf provides the model direct access to LR information that it must reconstruct. The paper acknowledges this in the limitations ("the generated LR image inevitably retains information from the input LR high-frequency components"), but does not confront its implication for interpreting the improvements. An ablation randomizing or removing y_hf is needed to determine how much performance comes from genuine degradation modeling versus shortcut copying from the condition. Without this, the claims about LDP as an "independent forward model" are weakened.

- **Inconsistent effectiveness of the inference-time (posterior sampling) mode**: Table 5 reveals that the posterior sampling mode—one of the paper's two claimed contributions—yields mixed or negligible results. LDM+LDP degrades on multiple metrics (RealSR: NIQE +0.179, MANIQA −0.009, CLIPIQA −0.025; DPED: NIQE +0.046). ResShift+LDP shows near-zero gains across all datasets (e.g., RealSR: PSNR unchanged at 22.83, DPED QAlign +0.001). Only StableSR shows consistent improvement. This substantially weakens the claim that LDP is an effective inference-time correction module.

- **Lack of comparison with closely related degradation-modeling methods in SR fine-tuning experiments**: Tables 3-4 compare baselines with and without LDP but include no comparison to DRN, DualSR, Lway, or CorrectFilter used as auxiliary regularizers—methods that share the same goal of improving SR generalization via degradation modeling. The only comparison to DRN and DualSR (Table 1) is in the LR prediction task, where LDP has an inherent advantage (trained on the same degradation family used for testing). The absence of these baselines in the main SR experiments makes it impossible to assess whether LDP is better than simpler degradation-modeling alternatives.

### Minor:

- **FeMaSR+LDP occasionally worsens perceptual quality metrics**: Table 3 shows LPIPS increases for FeMaSR+LDP on Blur (+0.0031) and Hybrid (+0.0063), and Table 4 shows multiple no-reference metrics degrading on RealSR and DPED. The paper attributes this to GAN artifacts being rewarded by perceptual metrics, but this is an ad hoc explanation rather than a principled solution, and it means LDP is not universally beneficial across architectures.

- **Evaluation is largely confined to BSRGAN/Real-ESRGAN synthetic degradation families**: While the paper tests on real-world datasets (RealSR, RealSRSet, DPED), the core synthetic benchmarks (Down, Noise, Blur, JPEG, Hybrid) are all generated from the same degradation generators (BSRGAN, Real-ESRGAN) used for training LDP. The claim of "generalization to unseen degradations" would be better supported by testing on qualitatively different degradation operators (e.g., spatially varying blur, sensor noise patterns) that are outside the BSRGAN family.

- **Missing computational cost analysis beyond parameter count**: The paper claims LDP is "lightweight" (642K parameters) but provides no FLOPs, inference time, or training overhead comparisons. This is particularly relevant for the posterior sampling mode, which requires running LDP at every denoising step.

## Nice-to-Haves

- An ablation where y_hf is removed, randomized, or replaced by a degraded version would clarify the extent to which LDP's improvements come from genuine degradation modeling versus information leakage from the condition.

- A comparison with a simple LR-consistency baseline (e.g., using a fixed downsampling model as regularizer rather than the DAE-based LDP) would isolate the value added by the noise-based corruption and denoising mechanism.

- Empirical or theoretical validation of the noise-alignment property under SR degradations (e.g., measuring HR_t vs LR_t distances across timesteps) would strengthen the methodological foundation.

## Removed Points

- **Demand for comparison with "simpler baselines" like a direct HR→LR CNN**: The spark reviewer suggests comparing to a "lightweight CNN degradation network" trained directly. While conceptually useful, this conflates two separate contributions (the DAE framework vs. any learning-based degradation model). The paper's contribution includes the cyclic regularizer design, and demanding a separate full ablation of the DAE design vs. a vanilla CNN goes beyond reasonable scope. Moved to nice-to-have.

- **Demand for statistical significance tests**: For large-scale image benchmark evaluations with deterministic SR models, single-run evaluation is standard practice. Requesting confidence intervals is not customary in this community.

- **Claim that DRN/DualSR comparisons are "unfair"**: DRN was designed for bicubic SR, not for diverse degradations—the comparison in Table 1 specifically demonstrates that LDP handles diverse degradations better than methods designed for simpler settings, which is precisely the point. DualSR's configuration is described as per its original paper. This is not an unfair comparison but a demonstration of LDP's broader applicability.

- **Formatting and style complaints**: Removed per instructions.

- **Missing related works references**: Per instructions, I cannot confirm the existence of cited or uncited references, so I do not flag missing related work.

- **Demand for user study**: Requesting human evaluation for a method that proposes an algorithmic regularizer is outside standard practice; no-reference and reference metrics are the established evaluation framework in this field.

- **Nitpick about the paper acknowledging limitations as a weakness**: The paper's honest limitation statement shouldn't be double-counted as a separate weakness beyond the substantive concern it reveals.

## Novel Insights

The circularity between y_hf (derived from LR input y) and the cyclic loss (comparing predicted LR to y) is acknowledged by the authors but its implications are underexplored. If y_hf provides sufficiently rich information about y, then the LDP network could learn to reconstruct y largely from y_hf without ever learning the forward degradation model x→y. This means: (1) the improvements during fine-tuning may primarily come from an auxiliary loss that says "your SR output, when combined with LR information, should be consistent with the LR input"—essentially a sophisticated data-consistency term—rather than from a principled degradation constraint; (2) this would also explain why the posterior sampling mode is less effective, since in that context the cyclic loss provides less new information to the already-conditioned diffusion model. This distinction between "LR-consistency regularizer" and "learned forward degradation model" is the most important interpretive question the paper leaves unresolved.

## Suggestions

- **Critical**: Add an ablation where y_hf is replaced by a randomized or zeroed-out version to quantify how much performance comes from the condition's information content. This is the single most important experiment for validating the method's claims.

- Conduct an ablation comparing LDP to a straightforward LR-consistency regularizer (e.g., applying a simple bicubic downsampling + small CNN on SR outputs with L1 loss to y) to isolate the value of the DAE/noise-based design.

- Provide at least one evaluation on degradation types genuinely outside the BSRGAN/Real-ESRGAN family to support the "unseen degradation" claim.

- Report FLOPs and wall-clock time for LDP in both modes.

## Score and Decision

Calibration against related papers:
- **ClearSR** (scores 5/5/6/5, rejected): Similar plug-in for SR generalization, similar concerns about novelty and computational overhead. LDP is somewhat more principled but shares incremental-novelty concerns.
- **RealDGen** (scores 5/6/8/5, accepted poster): Addresses similar SR generalization problem with a stronger methodological contribution and more thorough evaluation. Stronger novelty bar.
- **ConsisSR** (scores 3/5/6/5, withdrawn/rejected): Diffusion SR consistency method; concerns about incremental novelty and weak baselines.
- **FedSR** (scores 5/5/3/5, withdrawn/rejected): Training-free SR plugin, weak novelty.
- **Res-Captioner** (scores 5/6/3/3, withdrawn/rejected): Plug-and-play module for SR generalization with limited novelty.
- **BtT6o5tfHu** (scores 6/8/6, accepted poster): Strong mathematical contribution to diffusion SR, plug-and-play.

LDP has a creative and well-motivated framework with consistent empirical gains on synthetic benchmarks and real-world datasets. Its weaknesses—unverified core assumption, potential information leakage, inconsistent inference-mode results, and limited baseline comparisons—are notable but not fatal. The method clearly works as an LR-consistency regularizer even if the mechanistic story is imperfect. Compared to ClearSR (rejected at avg ~5.25) and ConsisSR (rejected at avg ~4.75), LDP has stronger and more consistent empirical results. Compared to RealDGen (accepted at ~6), LDP has weaker novelty and methodological depth but is more broadly applicable as a plug-in. The paper sits between reject and borderline-accept, with the unvalidated core assumption and the inconsistent posterior sampling results being the main concerns.

MY FINAL SCORE: 5.5
MY FINAL DECISION: Reject