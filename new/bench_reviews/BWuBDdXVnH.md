Now I have a thorough understanding of the paper and have verified the reviewer claims against the actual text. Let me write the final consolidated review.

---

## Summary

ControlAR proposes "conditional decoding" — a token-level additive fusion of spatial control tokens with image tokens during the autoregressive decoding phase — as an alternative to conditional prefilling for injecting spatial controls into autoregressive image generation models. The method avoids the quadratic sequence-length overhead of prefilling, enables arbitrary-resolution generation by varying control token length, and demonstrates competitive performance against diffusion-based controllable models (e.g., ControlNet++) across diverse spatial conditions (edges, depth, segmentation).

## Strengths

- **Conceptually clean mechanism with strong empirical validation.** The conditional decoding formulation (Eq. 4) is simple — projecting control tokens and adding them to the residual stream at selected intermediate layers — yet Figure 2(c)–(d) provides clear evidence that it outperforms conditional prefilling in both convergence quality (FID ~25 vs ~35, F1-Score ~18 vs ~28) and training efficiency (~59% less GPU memory, ~96% less time per epoch).
- **Competitive results across multiple control modalities.** Tables 1–3 show ControlAR achieving strong FID and conditional consistency scores on C2I (7.69 FID with Canny, 4.19 FID with Depth on ImageNet using LlamaGen-L) and T2I tasks, often matching or exceeding diffusion-based methods like ControlNet++ on conditional consistency metrics (e.g., 85.63 vs 80.97 SSIM on HED edges).
- **Systematic ablations across encoder selection, fusion strategy, and training paradigm.** Table 4's comparison of CNN vs. ImageNet-supervised ViT vs. DINOv2 initialization — finding that DINOv2's more diverse pre-training better suits T2I tasks — is a practically useful insight. Table 5 revealing that injecting at every layer degrades FID (11.75 vs 10.64) is a non-trivial finding about the control-fidelity trade-off.
- **Arbitrary-resolution generation is a genuine extension.** Unlike methods constrained to fixed grid sizes, conditional decoding naturally maps control token count to output token count. Figure 6(b) demonstrates MR-ControlAR maintaining ~85.5 SSIM across aspect ratios from 1:1 to 2:1 after multi-resolution training.

## Weaknesses

### Fatal
*None.* The core claims — that conditional decoding outperforms prefilling for spatial control injection, and that the resulting method can compete with diffusion-based control models — are substantiated by the reported experiments.

### Major

- **Missing text-alignment metrics for text-to-image evaluations undermine T2I claims.** Tables 2 and 3 report FID and conditional consistency (mIoU, F1-Score, SSIM, RMSE) but omit any text-image alignment metric such as CLIPScore, HPSv2, or TIFA. For a claimed T2I controllable generation task, this is a substantive gap: the current metrics only demonstrate that the model follows the spatial control map and produces plausible images. Without alignment scores, it cannot be determined whether the text prompt meaningfully conditions generation or is effectively ignored. Several comparable generative papers at ICLR have been penalized for this exact omission (e.g., RauUgiw7VX scored 5/3/5/6 with a reviewer explicitly requesting FID + CLIP score; rH6IZIXqZG scored 6/5/3 with the same criticism). Including text-alignment evaluation would be necessary to support the full T2I claim.

- **The cross-paradigm SOTA comparison (AR vs. diffusion) is structurally asymmetric and risks overclaiming.** The paper positions its results against ControlNet++ across FID and conditional consistency (Tables 2–3). However, as the harsh critic notes, ControlNet++ is a ~30M parameter adapter fine-tuned on a frozen SD1.5 backbone originally trained on web-scale data, while ControlAR requires full fine-tuning of a 775M–1.1B parameter AR model on a narrower subset. The paper does frame this as a paradigm-level comparison, which is fair for demonstrating that AR can be competitive. But claiming to "surpass" diffusion-based SOTA without discussing the asymmetry in training data scale, parameter update scope, and prior pre-training quality risks overstatement. This does not invalidate the results but requires more honest framing.

### Minor

- **Training convergence of the conditional prefilling baseline may be incomplete.** Figure 2(c) shows prefilling at ~100 epochs still performing worse than decoding. However, because prefilling doubles sequence length, it changes optimization dynamics (attention scaling, gradient flow). The paper provides no evidence that the prefilling baseline was trained to a matched convergence point with an appropriate learning rate schedule. While this does not negate conditional decoding's advantage (the gap is large), a more thorough baseline study would strengthen the motivation. The comparison is sufficient for the paper but leaves room for the reviewer's concern.

- **Efficiency claims are limited to the prefilling comparison and do not address full training costs.** The paper frames ControlAR as "efficient" (Abstract, Sec. 1) and demonstrates efficiency gains over prefilling in Figure 2(d) and over cross-attention in Table 5 — these are correct and well-supported. However, Table 6 shows full fine-tuning is mandatory (FID degrades from 10.64 to 13.20/13.67 with LoRA/freezing). The paper does not report total GPU-hours or training budget relative to the diffusion baselines it claims to compete with. The efficiency framing is accurate within its intended scope (vs. prefilling) but could mislead readers about the total computational cost of training ControlAR from scratch.

### Trivial

- **Minor notation and indexing clarity issue in Eq. 4.** The displacement formulation `[c + C1, I1 + C2, I2 + C3, ...]` offsets control tokens by one position relative to image tokens. The text states this "allows the model to make autoregressive predictions with control information corresponding to the next image token," but the causal relationship between index $C_l$ and $I_l$ could be more explicitly spelled out for readers unfamiliar with residual-stream conditioning.

## Nice-to-Have

- Report total training compute (GPU-hours) for ControlAR to contextualize the full fine-tuning cost.
- Provide additional qualitative results showing arbitrary-resolution generation at extreme aspect ratios (e.g., 3:1) or outside the 384–1024 training range to demonstrate practical limits of RoPE extrapolation.
- A discussion of why injecting control at every layer (1~12) degrades FID compared to sparse injection (layers 1, 5, 9) — the observation in Table 5 is interesting but unanalyzed.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Original: "Uncomparable baselines invalidate SOTA claim" — Reframed as a Major concern rather than a fatal issue.** The harsh critic argued the comparison is "fundamentally flawed" and "invalidates the headline claim." However, cross-paradigm comparison between AR and diffusion methods is standard practice in generative modeling literature, and the paper does report competitive results. The asymmetry is real but does not invalidate the core contribution. Moved to a Major overclaiming concern rather than a dismissal.

2. **Original: "Efficiency claim is directly contradicted by full fine-tuning requirement" — Weakened.** The paper's efficiency claims are explicitly about conditional decoding vs. conditional prefilling (Abstract: "Compared to prefilling tokens, using conditional decoding significantly strengthens the control capability... but also maintains the model efficiency"), not about training cost vs. diffusion baselines. The harsh critic conflated two separate efficiency framings. The valid point — that total training cost for full fine-tuning is not reported — is moved to a Minor concern.

3. **Original: Claims about RoPE extrapolation being "not novel."** The paper does acknowledge the technique ("adjust the parameter settings of the rotational position encoding... increasing its maximum sequence length to 2304" in Sec. 4.2). The contribution is that conditional decoding *enables* this capability for AR models, which is a legitimate integration. Framing this as a weakness misreads the paper's own acknowledgment.

4. **Original: Missing hyperparameter details (epochs, GPU-hours) are a "critical gap."** Reproduced as a Minor, not a Major, concern. The paper does report optimizer, learning rate, batch size, and image sizes. The missing epoch count and GPU hours are standard reproducibility issues but do not undermine the results.

5. **Any criticism about the paper's formatting artifacts, garbled text, or parser issues.** These are parser errors, not author errors.

## Novel Insights

The paper makes a practical engineering contribution by demonstrating that simple residual-stream additive fusion of control tokens (rather than complex cross-attention adapters or multi-scale feature pyramids) is sufficient for strong spatial conditioning in autoregressive image models. The key insight is that by avoiding sequence-length expansion entirely, the method sidesteps both the computational bottleneck of prefilling and the representation-learning challenge of mapping 2D spatial features into token space via attention. The ablation showing that over-frequent control injection (every layer) degrades generation quality while too-sparse injection underperforms is a non-trivial architectural finding — it suggests that AR models have intrinsic spatial priors that are disrupted by excessive external conditioning, a trade-off not present in diffusion models where the U-nature naturally processes multi-scale features. This tension between control adherence and prior preservation is worth articulating more clearly as it informs future work in this space.

## Suggestions

1. **Add a text-alignment metric (CLIPScore is most standard) to Table 2/3 for T2I tasks.** This is the single most impactful change that would strengthen the T2I claims substantially.
2. **Reframe the SOTA comparison section** to transparently acknowledge the training-paradigm asymmetry with diffusion methods. Use language such as "competitive with" or "comparable to" rather than "surpasses" when the training regimes differ significantly.
3. **Report total training epochs and approximate GPU-hours** for the reported experiments to allow readers to contextualize the full fine-tuning cost.
4. **Clarify the indexing in Eq. 4** by explicitly stating which image token prediction is conditioned on which control token (e.g., "to predict $I_l$, the model receives $I_{l-1} + C_l$ as input").
5. **Add a brief analysis** of why per-layer injection (1~12) degrades FID relative to sparse injection (1, 5, 9) — this finding has architectural implications worth discussing.

## Score and Decision

I calibrated this paper against several anchors:
- **High-scoring AR papers** (kGvXIlIVLM, CCA for guidance-free AR, scored 8/8/6/6/8/6): Those had more comprehensive theoretical grounding and cleaner ablation narratives. ControlAR is solid but simpler.
- **Borderline controllable generation papers** (HCUksccuFx: 5/5/6; rYhDcQudVI: 5/5/5/5; svp1EBA6hA: 6/8/6/6): These share similar profiles — clear mechanism, strong experiments but some evaluation gaps or overclaims. ControlAR sits alongside them.
- **Low-scoring papers** (Fk4Op9wpEp: 3/3/3/3): ControlAR is clearly stronger — its mechanism is novel, experiments are well-structured, and results are competitive.
- **Papers penalized for missing text metrics** (RauUgiw7VX: 5/3/5/6): This closely matches ControlAR's situation on the T2I evaluation gap.

The paper presents a genuinely useful and well-demonstrated method with competitive results, but the missing text-alignment evaluation for T2I and the somewhat overframed SOTA comparison prevent it from reaching the 7–8 tier. It is stronger than typical borderline 5s due to the breadth of its ablation studies and the clarity of its mechanism, but the T2I metric gap is a substantive weakness.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>