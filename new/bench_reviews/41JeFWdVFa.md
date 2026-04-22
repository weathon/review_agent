Now I have a thorough understanding of the paper. Let me also check the claim about LR_hf information content (the "circularity" concern) and the inference-time claim more carefully.

The key claims from the harsh critic I need to verify:

1. **LR_hf circularity**: The harsh critic claims LR_hf contains "most of the discriminative texture information" and creates near-circular information flow. Let me verify: For s=4, s²=16. So LR_hf = y - y↓_{16}↑_{16}. The LR image y has resolution H/s × W/s from HR (H×W). Downsampling by s²=16 from the LR resolution would give incredibly tiny images. Wait - I need to re-read. The Eq 4 says y_{hf} = y - y↓_{s²}↑_{s²}. But y is the LR image (resolution H/s × W/s). So y↓_{s²} would be H/(s·s²) × W/(s·s²) = H/(4·16) × W/(4·16) if s=4... that's H/64 × W/64, extremely tiny. Then upsampling back gives a heavily blurred version of y. The high-frequency component y_{hf} retains most of y's information.

Actually wait - looking again at line 162: "For a scale factor of s = 4, the key hyperparameters are s' = 2". Let me check if there's an s' that replaces s². Let me look more carefully at what s² means in context.

Looking at line 4 (Eq. 4): y_{hf} = y - y↓_{s²}↑_{s²}. For s=4, this would be 16-fold. But the paper also mentions s'=2 at line 162. Let me check if s' might be used instead of s² in practice.

Actually, re-reading line 94: "Its input is the high-frequency component of the LR image, computed by subtracting the s²-fold downsampled-then-upsampling LR image from the original LR image". And line 162 mentions s' = 2 separately from Eq. 4. So s' seems like a different parameter.

The harsh critic's complaint about s² making LR_hf contain nearly all information seems valid in principle - when s=4, 16-fold downsampling of the LR image and upsampling back creates a heavily low-pass filtered version, so the residual y_{hf} would contain nearly all the high-frequency structure. However, the paper explicitly states this is by design in the three criteria (discriminative, not the LR itself, simple). The question is whether this is truly "near-circular" or just informative conditioning.

The harsh critic's concern about near-circular information flow is interesting but not entirely fair. The conditioning signal LR_hf is not the same as giving the model the LR image directly - it's a high-frequency component. The model still needs to learn the degradation mapping. The concern about "shortcut learning" is somewhat addressed by the fact that the model must still denoise noisy HR features to produce LR predictions.

2. **Dual-mode claim**: The paper claims LDP works as both a training loss and inference post-processing. Looking at the experiments:
- Tables 3-4: fine-tuning mode (training-time)
- Table 5: posterior sampling mode (inference-time, diffusion models only)

The harsh critic claims the inference-time mode is only demonstrated for diffusion models. Checking the paper: Section 3.3 describes "Diffusion Posterior Sampling with LDP" using DPS (Eq. 17), which is indeed only for diffusion models. The abstract and introduction claim "inference post-processing step to correct artifacts" which is verified only for diffusion models. For non-diffusion models, only the fine-tuning mode is shown.

This is a valid concern - the claim is somewhat overgeneralized.

3. **Generalization to unseen degradations**: The synthetic test uses BSRGAN.plus (BSRGAN + Real-ESRGAN patterns). While these are from the same family, Real-ESRGAN adds additional degradation patterns not in BSRGAN training. The real-world datasets (RealSR, DPED, RealSRSet) are genuinely unseen. So the "unseen" generalization claim has some basis, but it's not from truly out-of-distribution degradation types.

4. **Ablations**: The ablations only vary loss terms (Table 6-7) and τ. No ablation of removing DPM conditioning, removing noise, or comparing with cycle consistency baseline. This is a valid gap.

5. **Posterior sampling results**: Looking at Table 5, many improvements are indeed marginal (e.g., CLIPIQA 0.5353→0.5354 for ResShift on RealSR).

Now let me compose my final review.</think>Now I have all the information needed. Let me compose the final review.

## Summary

LDP proposes a lightweight (642K parameter) denoising autoencoder plug-in for single-image super-resolution that improves generalization to unseen degradations via LR cyclic consistency. It conditions degradation prediction on the LR high-frequency component (LR_hf), uses patch-wise noise addition to model spatially-varying degradation, and can be applied either as a fine-tuning auxiliary loss or as an inference-time posterior sampling correction for diffusion-based SR models.

## Strengths

- **Consistent improvements across diverse SR architectures**: Table 3 demonstrates PSNR gains on synthetic benchmarks across four architecturally distinct SR models (FeMaSR, StableSR, SwinIR, MambaIR) and five degradation types, with particularly notable gains for StableSR (e.g., +2.16 dB on Hybrid, +1.74 dB on JPEG). This broad applicability supports the "plug-and-play" nature of the contribution.

- **Lightweight and efficient design**: The LDP module has only 642K parameters and trains in 16 hours on a single RTX A6000 (Section 4.1), making it practical for real use and substantially more efficient than per-image optimization methods like DualSR.

- **Innovative degradation model evaluation**: Table 2 provides a useful diagnostic by measuring similarity between degradation model outputs and simple downsampling, demonstrating that LDP avoids trivial collapse (PSNR 25–28 vs. DRN's 31–35), unlike DRN which collapses to near-bicubic behavior.

- **Dual application modes**: The framework supports both training-time regularization and inference-time posterior sampling for diffusion models (Eq. 17), with both modes empirically validated (Tables 3–5).

## Weaknesses

### Fatal
None.

### Major

- **Insufficient ablation isolating the conditioning signal from the architecture**: The ablation study (Tables 6–7) only varies loss terms and the τ hyperparameter. It does not test what happens when LR_hf conditioning is removed, when noise addition is removed (t=0), or when a simpler cycle-consistency baseline (learnable downsampler + L_fre) is used instead of the full DAE framework. Since LR_hf = y − y↓_{s²}↑_{s²} (Eq. 4) retains most of the LR image's information (especially for s=4 where s²=16-fold downsampling creates an extreme low-pass filter), this conditioning alone may account for a significant portion of LDP's effectiveness. Without these ablations, it is impossible to determine whether the DAE architecture and diffusion alignment mechanism contribute meaningfully beyond what the informative conditioning signal already provides. This directly undermines the paper's core claim that reinterpreting degradation modeling via denoising autoencoders (Section 3.1) is the mechanism driving improvements.

- **Inference-time post-processing claim is overgeneralized**: The abstract and introduction state LDP can be applied "as an inference post-processing step to correct artifacts" generically. However, Section 3.3 only describes this mode for diffusion models via DPS (Eq. 17), and Table 5 exclusively evaluates diffusion models (LDM, StableSR, ResShift, UPSR) in this mode. For all non-diffusion baselines (SwinIR, MambaIR, FeMaSR), LDP is used only during fine-tuning as an auxiliary loss (Section 4.3). The claim of dual-mode applicability to "arbitrary SR models" (Figure 1 caption) is unsupported for non-diffusion models.

- **Posterior sampling results are marginal at best for most diffusion models**: Table 5 shows that many of the reported improvements under posterior sampling are within measurement noise. For example, ResShift+LDP: CLIPIQA on RealSR changes from 0.5353→0.5354, MUSIQ on DPED changes from 44.63→44.59. LDM+LDP worsens NIQE on all three datasets. UPSR+LDP degrades QAlign on DPED (2.271→2.257). These results do not convincingly demonstrate that inference-time LDP correction provides meaningful improvements, which weakens the second claimed contribution.

### Minor

- **Generalization claims are tested within the BSRGAN degradation family**: The synthetic benchmarks use BSRGAN.plus (BSRGAN + Real-ESRGAN degradation patterns), which overlap with the training distribution. The real-world datasets (RealSR, DPED, RealSRSet) provide some out-of-distribution evidence, but no systematically diverse OOD degradation types (e.g., sensor-specific noise, video compression, unusual blur kernels) are evaluated. The generalization claim is partially supported but not as broadly as the title and abstract suggest.

- **Real-world results are mixed for FeMaSR**: On DPED (Table 4), FeMaSR+LDP degrades on NIQE (+0.659), MANIQA (−0.039), MUSIQ (−5.07), and QAlign (−0.167). The paper selectively attributes CLIPIQA degradation to metrics "favoring visually striking but structurally inaccurate results," but this explanation is not applied consistently—when LDP improves CLIPIQA, the metric is treated as valid.

- **The s² scale choice in LR_hf computation is not justified or ablated**: Eq. 4 uses s²-fold downsampling to compute the low-frequency component of y (the LR image itself). For s=4, this means 16-fold downsampling before interpolation, which creates an extreme low-pass filter leaving LR_hf with nearly all the LR image's information. This choice is central to the design but not analyzed or compared against alternatives (e.g., s-fold or smaller factors) in the main paper.

### Trivial
None.

## Nice-to-Haves

- Ablations removing LR_hf conditioning and noise addition (t=0) would clarify how much improvement comes from the DAE architecture vs. the informative conditioning signal.

- Quantifying the information content of LR_hf (e.g., reconstruction PSNR from y↓_{s²}↑_{s²} + y_{hf} = y) would assess the circularity concern directly.

- Demonstration of inference-time correction for at least one non-diffusion model, or appropriate qualification of the dual-mode claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"LR_hf conditioning creates near-circular information flow, undermining the degradation modeling claim"** (Harsh Critic Issue #1): While this raises a legitimate question, the claim that it is "near-circular" overstates the case. The model must still learn to denoise corrupted HR features and predict the correct degradation pattern—the conditioning provides information about *which* degradation, not the degradation mapping itself. The concern is better framed as an ablation gap (moved to Major weakness above) rather than a fundamental circularity that invalidates the paper.

- **Formatting issues**: All formatting/typo complaints are parser artifacts and removed per rules.

- **Missing references/appendices**: References to Appendix F for ablations of patch size, frequency band, and scale factor are stripped by the parser—they exist in the original submission.

- **Reproducibility concerns**: Complaints about undisclosed hyperparameters are removed per rules—τ, λ values, and training details are specified in the paper.

## Novel Insights

The paper raises an interesting design question about conditioning signal informativeness in degradation models: using LR_hf as a condition satisfies multiple criteria simultaneously (not the LR itself, discriminative for different degradations, simple to compute), but its high information content may reduce the burden on the learned degradation model. This tension between informativeness and non-triviality of the conditioning signal is a subtle design tradeoff that the community should consider when building conditional degradation models, and it is not fully resolved by this paper.

## Suggestions

- Add ablations that remove DPM conditioning (LR_hf) and noise addition (t=0) to isolate whether the DAE architecture itself contributes beyond the informative conditioning signal. A simple cycle-consistency baseline with a learnable downsampler + L_fre would also be informative.
- Qualify the "inference post-processing step" claim to apply only to diffusion-based SR models, or demonstrate corrective post-processing for at least one non-diffusion architecture.
- Consider testing on genuinely out-of-distribution degradation types (e.g., camera ISP noise, video compression) to strengthen the generalization claim.

## Evaluation

**Originality**: The idea of integrating a lightweight DAE conditioned on LR high-frequency components as a degradation-aware regularizer for SR is a reasonable and somewhat novel contribution. The connection to diffusion alignment (DR2) provides theoretical motivation, though the single-step DAE deviates significantly from full diffusion models.

**Research Question Importance**: Improving generalization to unseen degradations in SISR is an important and practical problem.

**Claim Support**: The training-time mode is well-supported (Table 3). The inference-time mode for diffusion models is weakly supported at best (Table 5 marginal gains). The core architectural claim (DAE as degradation model) is not isolated from the conditioning signal in ablations.

**Soundness of Experiments**: Synthetic evaluations are within the BSRGAN family. Real-world evaluations show mixed results, particularly for FeMaSR. A key ablation (removing conditioning) is missing.

**Clarity**: The paper is generally well-written and organized, though the dual-mode claim could be more precisely stated.

**Value**: If the ablation gap is addressed and the conditioning signal is not the sole driver of improvements, this could be a useful plug-in contribution. As-is, the value is partially established but uncertain.

## Score Calibration

- **High anchor**: Papers like GenDR (avg 6.0) and SAVL (avg 5.6) — degradation-aware SR with strong experimental frameworks and novel representations, both accepted as posters. LDP's experimental results are broadly similar in scope.
- **Medium anchor**: Plug-in IQC for posterior diffusion SR (avg 5.0, rejected) — similar marginal gains in posterior sampling but with stronger theoretical claims. KernelFusion (avg 4.0) — zero-shot blind SR with overclaimed "unrestricted kernel" paradigm and limited evaluation. LDP has stronger training-time results than these.
- **Low anchor**: DGMS (avg 2.5) and DSPO (avg 2.5) — weak papers with overclaimed contributions, poor presentation, and limited novelty. LDP is clearly above these.

LDP's training-time fine-tuning results (Table 3) are solid and consistent across models. Its weaknesses are the missing ablation isolating architectural contribution, overclaimed dual applicability, and marginal posterior sampling results. These are significant but not paper-killing — the training-time results stand on their own merit. Compared to SAVL (5.6), LDP has weaker ablations. Compared to GenDR (6.0), LDP has similar scope but more uncertainty about core mechanism. Compared to IQC (5.0, rejected), LDP has better fine-tuning results but similar posterior sampling issues.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>