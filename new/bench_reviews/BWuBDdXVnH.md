---

## Summary

ControlAR introduces a framework for spatial-control-guided image generation in autoregressive (AR) models. The core technical contribution is **conditional decoding**: rather than prepending control tokens (prefilling), each image token is fused additively with its spatially aligned control token before being passed through selected sequence layers. The paper demonstrates this approach on two AR backbones (LlamaGen and AiM) for class-to-image generation and on LlamaGen-XL for text-to-image generation across five control types. A multi-resolution variant (MR-ControlAR) additionally enables variable-resolution generation within a bounded training range.

---

## Claims and Support

| Claim | Supported? | Notes |
|-------|-----------|-------|
| Conditional decoding is more effective and efficient than prefilling | **Partially** | Fig. 2c-d provides clear evidence in a single setting (ImageNet canny, LlamaGen-B). No additional settings tested. |
| ControlAR "surpasses" ControlNet++ | **Overstated** | Tab. 2 shows ControlAR is *worse* than ControlNet++ on ADE20K segmentation (39.95 vs 43.64 mIoU), lineart SSIM (79.22 vs 83.99), and depth RMSE (29.01 vs 28.32). Win is not uniform across all metrics. Tab. 3 FID wins are genuine and mostly consistent. |
| ControlAR is a general framework for AR controllable generation | **Partially** | Two backbones in C2I; only one backbone (LlamaGen-XL) for T2I. Functional breadth is real but narrower than "general" implies. |
| Arbitrary-resolution image generation | **Overstated** | The method operates within a bounded range (384–1024 px, (H/16)×(W/16) ≤ 2304) and requires explicit multi-resolution training plus RoPE length adjustment. This is variable-resolution in a trained range, not arbitrary. |
| ViT/DINO encoder is superior to CNN | **Partially** | Tab. 4 shows ViT ≥ CNN empirically; the reasoning that sequence modeling of ViTs "matches" AR is speculative and not isolated experimentally. |
| Negligible computational overhead | **Partially** | Decoding vs prefilling overhead clearly shown (Fig. 2d). No end-to-end inference comparison vs the base AR model or vs diffusion baselines. |

---

## Strengths

- **Conditional decoding is a genuinely novel and efficient mechanism for AR controllable generation.** The 1:1 positional alignment of control tokens with image tokens during decoding is conceptually clean, motivated, and avoids explicit positional learning. Fig. 2c-d quantitatively shows faster convergence and nearly 2× lower training cost versus prefilling.

- **Cross-backbone compatibility demonstrated concretely.** The paper shows the same conditional decoding mechanism applied to both Transformer-based LlamaGen and Mamba-based AiM in C2I, with quantitative results for both. This is a concrete, non-trivial demonstration that the design is backbone-agnostic.

- **Parameter-efficiency vs. ControlVAR.** LlamaGen-L (343M) achieves FID 7.69 for canny C2I on ImageNet, matching ControlVAR's VAR-d30 (2.0B, FID 7.85) with only 16.7% of parameters (Tab. 1). This is a meaningful efficiency finding.

- **Strong T2I FID results.** Tab. 3 shows ControlAR achieves best FID on 5/6 T2I tasks over all diffusion baselines including ControlNet++, with notable gains on Hed (10.53 vs 15.01), COCOStuff segmentation (14.51 vs 19.29), and Depth (14.61 vs 16.66). This is a genuine empirical contribution.

- **MR-ControlAR maintains consistency across aspect ratios without additional modules.** Fig. 6b shows MR-ControlAR achieves near-constant SSIM (~85.5) across four aspect ratios (1:1 to 2:1) while single-resolution ControlAR degrades to ~80.5 at 2:1. This is a useful capability enabled naturally by the token-alignment property.

---

## Weaknesses

### Fatal
*(None. The method works and produces real results. FUNDAMENTAL ISSUES not triggered.)*

---

### Major

1. **The "surpasses ControlNet++" claim is unsupported by the actual data.** The abstract and introduction claim ControlAR "surpasses previous state-of-the-art controllable diffusion models, e.g., ControlNet++." But Tab. 2 shows ControlAR is second-best (underlined) on three out of six controllability metrics where ControlNet++ holds the best score: ADE20K segmentation mIoU (39.95 vs 43.64), lineart SSIM (79.22 vs 83.99), and depth RMSE (29.01 vs 28.32). The conclusion appropriately softens to "very competitive," but the abstract does not. This inconsistency weakens the paper's framing. The claim should be restated as: ControlAR is **competitive with and often better than** ControlNet++ across the evaluated metrics.

2. **Full fine-tuning requirement undermines the "flexible framework" positioning.** Table 6 shows a significant gap between full fine-tuning (F1: 34.15, FID: 10.64) and alternatives: freeze (F1: 30.62, FID: 13.67) and LoRA (F1: 32.90, FID: 13.20). This means deploying ControlAR requires modifying all base AR model weights for each control type — unlike ControlNet, which leaves the Stable Diffusion weights frozen and operates as a plug-in module. This has real practical implications: switching to a different or larger base AR model necessitates full retraining, and combining multiple control types becomes expensive. The paper acknowledges this in Table 6 but does not discuss the implication relative to diffusion-model approaches.

3. **The central mechanistic claim — that conditional decoding is generally superior to prefilling — rests on a single experimental setting.** Fig. 2c presents the key evidence comparing the two strategies using only ImageNet canny edges on LlamaGen-B. This is sufficient justification for using decoding in this paper, but the paper presents the comparison as a general principle for controllable AR generation. No evidence is presented for T2I settings, other control types, or other backbones. Furthermore, there is no control for whether prefilling was comparably optimized (e.g., placement of control tokens, compression ratio, prompt length).

4. **The "arbitrary-resolution" contribution is overstated.** The paper uses "arbitrary-resolution" throughout (abstract, intro, Sec. 3.4) but Sec. 4.2 reveals the actual constraint: resolutions are sampled from 384 to 1024, subject to (H/16)×(W/16) ≤ 2304, and the model requires explicit multi-resolution training plus RoPE positional encoding range extension. The quantitative evaluation in Fig. 6b covers only 4 aspect ratios up to 2:1, reports only SSIM for one control type (hed), and reports no FID across resolutions. This is a "variable-resolution within a trained range" capability, not arbitrary resolution in the broad sense.

5. **AR-to-AR baseline comparison is too thin to substantiate controllable-AR positioning.** Tab. 1 compares ControlAR against ControlVAR using FID only (with values "estimated from its histograms," acknowledged with *). No controllability metrics (F1-Score, RMSE) are compared against ControlVAR — which is the only direct AR-level baseline. Given that the paper positions ControlAR as the solution for next-token AR controllable generation, an incomplete comparison against the closest prior work is a meaningful gap.

---

### Minor

- **Missing inference latency comparison against diffusion baselines.** Fig. 2d shows training cost relative to prefilling, but there is no wall-clock inference time comparison against ControlNet or ControlNet++. AR models generate tokens sequentially; for long sequences (512×512 = 1024 tokens), this can be substantially slower than a diffusion model with a fixed number of denoising steps. Without this, "efficiency" claims about the approach as a whole are incomplete.

- **Cross-attention underperformance is unexplained.** Tab. 5 shows addition outperforms cross-attention for control fusion (F1: 34.01 vs 30.86). The authors attribute this to "cross-attention needing to first understand the positional relationship," which is speculative and not supported by any analysis. Given that cross-attention is the standard fusion mechanism in diffusion-based controllable generation, a more careful analysis (e.g., convergence curves, attention map visualization) would strengthen this design choice.

- **Arbitrary-resolution evaluation breadth is narrow.** Fig. 6b reports SSIM at only 4 aspect ratios for a single control type (hed), without FID. The claim that MR-ControlAR "ensures that the generation of images with different resolution ratios is not impaired" (Sec. 4.2) needs broader quantitative support — at minimum, an FID comparison across resolutions and coverage of more control types.

---

### Trivial

- The one-position offset design in Eq. 4 ("displacement by one position") is presented as unmotivated and has no ablation. Even a brief experiment comparing zero-offset vs. one-offset would help justify this design choice.

---

## Nice-to-Haves

- **Inference speed benchmarks**: Report time-per-image at 512×512 for ControlAR vs ControlNet++ on equivalent hardware to help practitioners make architecture choices.
- **Catastrophic forgetting evaluation**: Since full fine-tuning is required, reporting unconditional image generation FID before and after fine-tuning would address whether the base AR model's diversity is preserved.
- **Failure case analysis**: All visualizations show successful results. Including failure cases with low control consistency would give a more honest view of where the method struggles.
- **Out-of-range resolution extrapolation test**: Stress-testing MR-ControlAR at extreme aspect ratios beyond the 2:1 training range would clarify the true boundary of the "arbitrary" resolution claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Missing comparison with a specific competing AR controllable generation paper (CAR)**: The human-finder reviewer cites a competing paper "CAR (Controllable Autoregressive Modeling, sicB10feCQ)" and criticizes its absence. Per the hard rule, missing related works are not flagged since the existence of this specific paper cannot be independently verified here. Removed.

- **Reproducibility concern about undisclosed hyperparameters / implementation details** (Harsh Critic, Sec. 3.3: "too vague for exact reproduction"): The paper states "we evenly replace the conditional sequence layer three times" and Tab. 5 shows the ablation with positions 1,5,9 for LlamaGen-B's 12 layers, which is consistent. The detail is sufficient for understanding. Removed as a hyperparameter reproducibility nitpick.

- **Doubting the efficiency claim because the control encoder MACs are not contextualized** (Harsh Critic Claim 6): The paper reports 0.05T MACs for the control encoder at 512×512. The inference latency comparison vs diffusion models is a legitimate minor weakness (kept in Minor section), but the specific framing of "0.05T MACs not contextualized" is a detail-level nitpick. The broader inference comparison concern is retained.

- **Generic strength: "the paper addresses a timely and important problem"** (Human Finder): Removed as generic.

- **Generic strength: "well-motivated design choices"** (Neutral reviewer): This applies to any paper with ablations. Removed as generic.

---

## Novel Insights

The key insight—that fusing control information as an additive perturbation to image tokens at select decoding layers (rather than extending the context window) is both more efficient and more effective for AR controllable generation—is a non-obvious and practically significant contribution. The observation that this alignment-via-addition operates analogously to positional encodings, embedding spatial correspondence into the generation process rather than forcing the model to learn it from an extended sequence, offers a useful conceptual framework for understanding why prefilling fails in the AR setting despite succeeding in LLM contexts. The natural extension to variable-resolution generation as a byproduct of token-length alignment (without needing separate resolution-aware prompting) is an elegant and under-appreciated architectural consequence of this design.

---

## Suggestions

1. **Restate the abstract and introduction claims accurately**: Replace "surpasses" with "is competitive with and in many settings outperforms" ControlNet++. Tab. 2 clearly shows mixed results; the text should reflect this.
2. **Replace "arbitrary-resolution" with "variable-resolution within the trained range (384–1024 px)"** throughout, and add an explicit statement of the token-budget constraint.
3. **Extend the decoding-vs-prefilling comparison** to at least one T2I task and one additional backbone, and report FID/controllability alongside training convergence curves.
4. **Report FID and SSIM across multiple resolution ratios and multiple control types** in Fig. 6 to properly evaluate MR-ControlAR.
5. **Include a single-table inference latency comparison** (ControlAR vs ControlNet vs ControlNet++ at 512×512, same hardware) to resolve the open efficiency question.
6. **Add an unconditional FID baseline** before and after full fine-tuning to assess forgetting of the base AR model's generation diversity.

---

## Score and Decision

**Originality**: Moderate-to-good. The conditional decoding idea is novel for AR models and cleanly motivated; individual components (ViT encoder, addition fusion) are standard. The per-token spatial alignment property naturally enabling variable-resolution generation is a genuinely new observation.

**Importance of research question**: High. Spatial controllability is essential for practical image generation, and the AR paradigm is rapidly gaining ground; this is a timely and under-addressed problem.

**Claims vs. support**: Weak. The abstract overreaches meaningfully (uniform superiority over ControlNet++, "arbitrary" resolution), and the central mechanistic argument (conditional decoding is generally better than prefilling) rests on one experimental setting.

**Soundness of experiments**: Moderate. Ablations are solid; T2I comparisons are broad in control type but limited in backbone; C2I comparisons against AR alternatives are incomplete.

**Clarity**: Good overall, with a few under-specified design choices (offset ablation, layer selection rationale).

**Value to the research community**: Moderate-to-high. The method works well in practice and fills a clear gap. The full fine-tuning requirement is a real limitation relative to ControlNet's plug-in design, but the conditional decoding mechanism is useful knowledge for future work.

Overall this is a borderline-accept contribution: the idea is real, the results are strong in FID, but the claims are materially overstated and key evidence is narrower than the paper implies.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>