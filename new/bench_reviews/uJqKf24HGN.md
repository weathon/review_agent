Good — the paper on parameter-matched comparison between diffusion and GAN for SR was scored 5.75 and rejected. This helps calibrate: when fairness of comparison is the core concern, papers tend to score in the 5-6 range. But UniCon's case is somewhat different — the paper *does* provide UniCon-Half for some tasks and the efficiency story is strong regardless of parameter count.

Let me now finalize the review.

## Summary

UniCon proposes a unidirectional information flow paradigm for training control adapters in large-scale diffusion models. Instead of the bidirectional interaction in ControlNet (where gradients flow through both the adapter and the diffusion model), UniCon directs information one-way from the frozen diffusion model to a trainable adapter that produces the final output, eliminating gradient computation for the diffusion model. This yields substantial training efficiency gains (~50% VRAM reduction, ~2.3× training speedup on DiT) and enables training adapters with double the parameter count under the same resource budget.

## Strengths

- **Genuine and well-documented training efficiency gains**: The VRAM and speedup results in Figure 6 are the paper's strongest contribution. The per-component breakdowns (Weight, Activation, Gradient, Optimizer VRAM; FP/BP time) are carefully measured under controlled conditions (same GPU, same batch size, pre-computed features). UniCon cuts gradient VRAM nearly in half and achieves ~2.3× training speedup on DiT — these are practical, impactful improvements.

- **Architecture-agnostic design validated on both architectures**: UniCon applies equally to U-Net (SD) and transformer (DiT) architectures, with detailed architectural diagrams (Figure 2) and Table 2 results on both PixArt-α (DiT) and StableDiffusion-2.1 across five conditioning tasks. This addresses a real limitation of ControlNet's encoder-focused design for transformer models.

- **Ablation isolating unidirectional flow on SR task**: Table 1c, for the Full adapter on SR, shows PSNR 36.53→37.34 and FID 23.04→20.34 when switching from bidirectional (✗) to unidirectional (✓) flow with the same adapter architecture, directly demonstrating the paradigm's contribution.

- **UniCon-Half already outperforms ControlNet on SR tasks with comparable parameters**: For DiT SR, UniCon-Half achieves PSNR 35.64 vs. ControlNet's 34.82, and FID 22.07 vs. 26.43 (Table 2). For SD SR, UniCon-Half achieves PSNR 34.38 vs. ControlNet's 31.66. This shows the architectural advantage is not purely from increased capacity.

- **ZeroFT connector design with empirical validation**: Table 1b shows ZeroFT outperforms ZeroMLP and ShareAttn on both Canny (SSIM 0.5426 vs. 0.5343, FID 52.31 vs. 55.22) and SR (FID 22.07 vs. 22.99) tasks. The addition of element-wise multiplication alongside addition is a non-obvious improvement.

## Weaknesses

### Fatal

None.

### Major

- **Data reporting errors in Table 1c undermine the central ablation for the SR task's bidirectional baselines**: The SR (PSNR) section contains SSIM-scale controllability values for the Skip-Layer ✗ (0.5053) and Decoder ✗ (0.5458) rows. These values are in the 0–1 range, not the ~30–40 PSNR range, and they exactly match (across all columns: controllability, FID, Clip-IQA, MAN-IQA, MUSIQ, Clip-Score) the Canny results from Table 1a and Table 2 respectively — e.g., 0.5053 matches the Canny Full row from Table 1a, and 0.5458 matches the DiT Canny UniCon row from Table 2. This is a confirmed copy-paste error. While the Full ✗ (PSNR 36.53) and Full ✓ (PSNR 37.34) rows appear correct and the key ablation is preserved, the absence of valid bidirectional baselines for Skip-Layer and Decoder on SR makes the ablation incomplete for assessing whether unidirectional flow benefits *all* adapter architectures or only the Full one. This matters because the paper already notes that unidirectional flow did not help the Skip-Layer design — but this conclusion cannot be verified for the SR task due to the data errors.

- **Main quality comparisons in Table 2 are not parameter-matched for 3 of 5 DiT tasks and 2 of 5 SD tasks**: UniCon copies the *full* diffusion model as its adapter while ControlNet copies only the encoder portion, meaning UniCon has ~2× the adapter parameters. UniCon-Half (parameter-matched) is reported only for SR and deblur-downsampling tasks, not for the Canny, Depth, and Pose tasks where the largest quality improvements are claimed (e.g., DiT Canny: SSIM 0.4748→0.5458, a 15% improvement). The SR results where UniCon-Half *is* reported show the parameter-matched advantage is much smaller than the headline numbers (DiT SR: PSNR +0.82 for UniCon-Half vs. ControlNet, vs. +2.52 for UniCon-Full). Without parameter-matched baselines for Canny, Depth, and Pose, it is impossible to isolate the architectural contribution from the capacity advantage for these tasks. The paper frames "double the parameter volume" as a feature (which it legitimately is under the fixed-resource framing), but the Table 2 comparisons should not be read as like-for-like architectural comparisons for tasks lacking UniCon-Half.

### Minor

- **SUPIR-UniCon claim (Section 4.3, Figure 8) lacks quantitative validation**: The section concludes that UniCon "effectively addresses" SUPIR's scaling limitation, but provides only qualitative images with no metrics, no comparison with SUPIR+ControlNet under matched conditions, and no implementation details (training configuration, how SD3 was used). This section reads as a proof-of-concept demo rather than evidence.

- **Inconsistency between abstract and body regarding VRAM reduction**: The abstract states "reduces GPU memory usage by one-third" while the introduction states "saves half of the video memory (VRAM) usage" and Section 4.2 describes "saving nearly half the storage required for gradients." A one-third reduction (33%) and a half reduction (50%) are materially different claims. The Figure 6 data appears to support a reduction closer to half for DiT, making "one-third" an understatement rather than an overclaim, but the inconsistency should be resolved.

- **"Existing adapters primarily implement control within the encoder part, so their response to control signals lacks pixel-level precision" (Introduction) slightly misrepresents ControlNet**: ControlNet's zero convolutions inject residuals into the U-Net's middle and decoder blocks as well, not only the encoder. The claim that ControlNet is *only* encoder-focused is an overstatement, though ControlNet's *trainable copy* is primarily the encoder part.

### Trivial

None.

## Nice-to-Haves

- Parameter-matched (UniCon-Half) results for Canny, Depth, and Pose tasks on both DiT and SD would substantially strengthen the paper's quality-improvement claims and clarify the architectural vs. capacity contribution.
- Comparison with ControlNet-XS, which is cited in related work and directly addresses the same scaling problem with a different approach, would contextualize UniCon's contribution.
- Training loss curves or convergence dynamics: with 100K training steps and 2M images, showing that all models are converged at evaluation would strengthen fairness of comparisons.
- A deeper mechanistic analysis of *why* unidirectional flow improves quality (e.g., analyzing the role and magnitude of ZeroFT conditioning signals throughout training) would substantiate the claim beyond the ablation numbers.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"ControlNet copies only the encoder portion, UniCon has ~double the parameters"** — While the parameter asymmetry is real and relevant, the paper *does* address this with UniCon-Half for SR tasks and explicitly acknowledges in text (Section 4.3): "UniCon-Half, with only half the parameters, performs notably worse than the full-parameter UniCon but still performs better than ControlNet with a comparable parameter number. Notably, even the full-parameter UniCon has a lower training computational cost than ControlNet." The incomplete reporting of UniCon-Half is kept as a Major weakness above, but the framing that the paper entirely ignores parameter matching is softened since it partially addresses the concern.

- **"T2I-Adapter performs very poorly on SR (PSNR 18.94), inflating UniCon's relative advantage"** — This is a weak baseline concern, but the paper does not primarily compare UniCon against T2I-Adapter for SR; the main baseline is ControlNet. T2I-Adapter's poor SR performance is consistent with its known limitations for pixel-level tasks, not an artifact. The comparison favoring the baseline does not harm the author's method.

- **"No confidence intervals / variance reported"** — This is a generic weakness. For large-scale benchmark evaluations with 1000 test images, single-run evaluation is standard in this community. Moved to nice-to-have.

- **"The adapter must effectively re-learn layer processing due to ZeroFT conditioning, negating trainable-parameters advantage"** — This is speculative. The ablation shows the method works; the theoretical concern about re-learning is not supported by evidence and partially addressed by the zero-initialization of connectors.

- **"The paper does not analyze training dynamics or convergence behavior"** — This is a nice-to-have for an empirical paper, not a substantive methodological flaw.

- **"Skip-Layer design incompatible with unidirectional flow — explanation is vague"** — The paper provides an intuitive explanation ("skip-layer design compromising the output capability of the copied diffusion model"), and the empirical evidence supports this. A deeper explanation would be nice but is not a methodological flaw.

- **"The 'Full ✗' vs 'Full ✓' comparison is the most important number in the paper yet is buried"** — This is a presentation criticism, not a methodological one. The data is there and accessible.

## Novel Insights

The paper's key insight — that by routing the adapter to produce the final output rather than inject residuals, one can freeze the diffusion model and eliminate gradient computation — is a genuine paradigm shift in adapter design. The most interesting tension is that UniCon's primary *provable* contribution is efficiency (which stands independent of parameter counts), yet the paper's framing emphasizes quality improvement. The SR parameter-matched results show a real but modest architectural advantage (+0.82 PSNR for UniCon-Half over ControlNet), while the full UniCon's larger gains (+2.52 PSNR) come from legitimately applying the freed resources to double adapter capacity. The honest summary is: UniCon is primarily an efficiency method that *enables* quality improvement through capacity scaling, rather than a method that intrinsically produces better quality at matched capacity. This is still a valuable contribution — the efficiency story alone is compelling — but the framing should be adjusted accordingly.

## Suggestions

- Correct Table 1c by replacing the erroneous SR (PSNR) Skip-Layer ✗ and Decoder ✗ rows with actual SR data. This is essential for the paper's credibility.
- Report UniCon-Half results for all five tasks on both architectures, not just SR/deblur-downsampling. This is the single most impactful addition that would resolve the primary concerns.
- Harmonize the abstract's "one-third" VRAM reduction claim with the "half" stated in the introduction and supported by Figure 6 data.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Relation to UniCon |
|-------|-----------|-------------------|
| Würstchen (gU58d5QeGv.md) | 8.0 | Genuine paradigm with rigorous efficiency claims and quality validation — well above UniCon |
| MGFR (m9RNBZewW2.md) | 7.33 | Dual-control adapter with good results and dataset contribution — above UniCon on novelty/completeness |
| CTRL (svp1EBA6hA.md) | 6.5 | RL-based ControlNet alternative, limited comparisons but conceptually novel — similar tier but more complete |
| SaRA (wGVOxplEbf.md) | 6.2 | Efficient diffusion fine-tuning with concerns about assumptions — comparable, slightly above |
| Diffusion vs GAN fair comparison (46mbA3vu25.md) | 5.75 | Fair comparison study with missing metrics — similar concerns about comparison fairness |
| IFAdapter (25l4SWH2eS.md) | 5.75 | Instance feature adapter with moderate novelty — comparable tier |
| 3D-Adapter (C0HDYvGwol.md) | 5.6 | Plugin module with limited scope — comparable |
| MDiT (leBbjaUxut.md) | 5.0 | Strong efficiency claims but split reviews on generalizability — UniCon has stronger efficiency evidence but similar comparison concerns |
| APCtrl (yPxhj1FKhG.md) | 3.67 | ControlNet alternative with unfair comparison table — UniCon is clearly above this |
| ELR-Diffusion (edx7LTufJF.md) | 2.5 | Data inconsistencies across tables, cherry-picked results — UniCon shares a data error concern but is otherwise much stronger |

UniCon sits between the medium-scoring adapter/efficiency papers (5-6 range) and the higher-scoring ones (6.5-7). Its efficiency contribution is genuine and well-documented, placing it above papers with only questionable quality claims. However, the data error in the key ablation table and the incomplete parameter-matched comparisons are substantive issues that prevent it from scoring higher. Compared to CTRL (6.5) and SaRA (6.2), UniCon has a stronger efficiency story but weaker quality-claim validation. Compared to the Diffusion vs GAN fair comparison study (5.75), UniCon has a more novel contribution but similar concerns about comparison fairness. I place UniCon at 5.5 — borderline, with the efficiency contribution being real and significant but the quality claims lacking sufficient evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>