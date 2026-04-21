Now I have all the evidence I need. Let me compose the final review.

---

## Summary

ControlAR introduces a framework for controllable image generation with autoregressive models by encoding spatial controls (edges, depth, segmentation) into tokens and fusing them with image tokens via *conditional decoding* — adding control tokens to image tokens at intermediate layers — rather than prepending them as context. This approach is shown to be both more effective and more efficient than conditional prefilling, and it naturally enables arbitrary-resolution generation since the number of decoded tokens is determined by the control token sequence length. The method is evaluated on class-to-image (C2I) and text-to-image (T2I) tasks across five control modalities.

## Strengths

- **Conditional decoding is a genuine and well-supported insight.** Figure 2(c) shows conditional decoding converges faster and to better FID/F1 than prefilling, while Figure 2(d) shows 59.1% less memory and 96.3% less training time. This is a clean, useful finding: per-token fusion via addition avoids the quadratic cost of prepending tokens while improving control capability. This comparison alone constitutes a meaningful contribution.

- **Strong C2I results with excellent parameter efficiency.** Table 1 shows ControlAR with LlamaGen-L (343M params) achieves FID 7.69 on Canny, outperforming ControlVAR-d30 (2.0B params) at FID 7.85, using only 16.7% of the parameters. The method also works across Transformer-based (LlamaGen) and Mamba-based (AiM) backbones, demonstrating architectural generality.

- **Creative use of conditional decoding for arbitrary-resolution generation.** By varying the control token sequence length, ControlAR can generate images at non-fixed resolutions without additional modules. MR-ControlAR with multi-resolution training maintains high SSIM (~85.5) across aspect ratios from 1:1 to 2:1 (Figure 6b), while standard ControlAR degrades to ~80.5 at 2:1.

- **Competitive conditional consistency on T2I tasks.** Table 2 shows ControlAR beats ControlNet++ on COCOSTuff mIoU (37.49 vs. 34.56), Hed SSIM (85.63 vs. 80.97), and Canny F1 (37.08 vs. 37.04), demonstrating that the AR-based control mechanism can match or exceed diffusion-based methods on control fidelity.

## Weaknesses

### Fatal
None.

### Major

- **T2I FID comparisons do not isolate the control framework's contribution from base model quality.** Tables 2–3 compare ControlAR (built on LlamaGen-XL, 775M) against methods built on SD1.4/SD1.5. The paper's abstract claims ControlAR "surpasses previous state-of-the-art controllable diffusion models," but LlamaGen-XL may simply produce better unconditional images than SD1.5. Without reporting base-model-only FID for LlamaGen-XL vs. SD1.5 on the same evaluation set, it is impossible to determine whether the large FID gaps in Table 3 (e.g., 14.51 vs. 19.29 on COCOSTuff, 10.53 vs. 15.01 on Hed) come from the control framework or the underlying generative model. This matters because the paper's central claim is about the effectiveness of the *control mechanism*, not about AR models being better generative models than diffusion. The conditional consistency metrics in Table 2 are more diagnostic of control quality (and show mixed results), but the FID-based "surpassing" claim is confounded.

- **Full fine-tuning requirement is an unacknowledged practical limitation that changes the value proposition relative to ControlNet.** Table 6 reveals that full fine-tuning (F1: 34.15, FID: 10.64) substantially outperforms freezing (30.62, 13.67) and LoRA (32.90, 13.20). This means each control type requires a separate full copy of the base model (~775M parameters for LlamaGen-XL). ControlNet, by contrast, adds a lightweight module (~361M) on top of a *frozen* base, allowing one base model to serve multiple controls. The paper says ControlAR "easily expands autoregressive models with strong control capability" (Section 1, contributions), but expansion requires complete retraining and model duplication. This trade-off should be explicitly acknowledged as a limitation.

- **Arbitrary-resolution generation — presented as the "most importantly" contribution (Section 1) — is under-evaluated.** Figure 6(b) only reports SSIM at four resolution ratios with no FID evaluation at any non-square resolution. SSIM measures conditional consistency but says nothing about image quality at novel resolutions. Given that AR models trained at fixed resolution are known to degrade on out-of-distribution sequence lengths, FID at non-training resolutions is precisely the evaluation needed to support the claim that "MR-ControlAR can ensure that the generation of images with different resolution ratios is not impaired" (Section 4.2).

### Minor

- **The cross-attention vs. addition ablation in Table 5 is incomplete.** Cross-attention is only tested at the 1st layer, while addition is tested at 1st, 1+5+9, and all 12 layers. The 1st-layer comparison (30.86 vs. 34.01 F1) does support addition's superiority, but the paper's broader conclusion that "direct addition proves more efficacious than cross-attention" would be strengthened by testing cross-attention at multiple layers to rule out that multi-layer cross-attention could close the gap.

- **The abstract's "surpasses" claim overstates the T2I conditional consistency results.** Table 2 shows ControlAR wins on some metrics (COCOSTuff mIoU, Hed SSIM, Canny F1) but loses on others (ADE20K mIoU: 39.95 vs. 43.64; Lineart SSIM: 79.22 vs. 83.99; Depth RMSE: 29.01 vs. 28.32). The body text's "comparable or even better" framing is more accurate than the abstract's "surpasses."

- **ControlVAR's FID values in Table 1 are "estimated from histograms" (marked with *), making the C2I comparison imprecise.** While acknowledged in the caption, this limits the strength of the efficiency claim relative to ControlVAR.

### Trivial
None.

## Nice-to-Haves

- FID evaluation at multiple resolutions for MR-ControlAR to substantiate the arbitrary-resolution quality claim.
- Base-model-only FID (without controls) for LlamaGen-XL vs. SD1.5 on the T2I evaluation sets, to contextualize the FID improvements.
- Cross-attention fusion tested at layers 1, 5, and 9 to complete the ablation.
- Ablation on the positional displacement in Eq. 4 — what happens without the shift?
- Investigate whether a more thorough LoRA hyperparameter search could narrow the gap with full fine-tuning, which would significantly improve the practical value proposition.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic: "ControlAR as the autoregressive counterpart to ControlNet" framing is misleading due to full fine-tuning.** The paper does not explicitly claim to be a drop-in ControlNet replacement. It frames itself as a framework for "integrating spatial controls into autoregressive image generation models." The full fine-tuning issue is real (kept above), but the analogy itself is not misleading — the paper is solving a similar problem in a different paradigm.

- **Harsh critic: The analogy to positional encodings is misleading.** The paper says conditional decoding is "similar to positional encodings" (abstract) in the sense that both add per-position signals to tokens. The analogy is about the *mechanism* (addition rather than prepending), not about the *content* of the signal. This is a reasonable analogy, not a misleading one.

- **Harsh critic: ViT superiority claim over CNN is overconvincing.** Table 4 actually shows ViT-S outperforms CNN on 4 of 6 metrics (Canny F1, Depth RMSE, Depth FID, Hed SSIM), DINOv2-S outperforms on T2I metrics, and DINOv2-B is best overall. The paper's claim that "a ViT model, pre-trained on a large amount of data, is more adept at modeling sequences" is supported by the overall trend, even if individual cells are close. The paper also provides a reasonable explanation (pre-training data match). This is a minor nitpick, not a substantive weakness.

- **Harsh critic: Figure 5 is cherry-picked.** Highlighting failure modes of baselines is standard practice in qualitative comparisons. The quantitative results in Tables 2–3 provide the fair comparison.

- **Strength finder: "Surpassing diffusion-based SOTA on T2I FID" is listed as a strength.** This is confounded by base model differences (verified above as a Major weakness), so this strength is moved here. The conditional consistency competitiveness is kept as a strength.

- **Harsh critic: Novelty claim that "control-to-image generation remains largely unexplored within AR models" is overstated given ControlVAR.** The paper does cite ControlVAR and differentiates it: "ControlVAR employs next-scale prediction to jointly model control and image, but is still different from next-token prediction in autoregressive generation." This is a legitimate distinction — next-scale prediction (VAR-style) is architecturally different from next-token prediction (LlamaGen-style). The novelty framing is reasonable.

## Novel Insights

The conditional decoding vs. conditional prefilling comparison reveals a generalizable principle: for autoregressive models, injecting spatial control signals via per-token addition at intermediate layers is both more effective and more efficient than prepending them as context. This insight arises because prepending increases sequence length (and thus quadratic attention cost) while providing only indirect, global conditioning; per-token addition preserves sequence length and provides direct, spatially-aligned conditioning. This principle likely extends beyond image generation to any autoregressive task requiring spatial or structured conditioning.

## Suggestions

- Report base-model FID without controls for both LlamaGen-XL and SD1.5 on the T2I evaluation sets, which would either validate the "surpassing" claim or appropriately contextualize it.
- Evaluate MR-ControlAR with FID at non-square resolutions to substantiate the headline arbitrary-resolution contribution.
- Add an explicit discussion of the full fine-tuning trade-off and its practical implications for deployment (one model per control type vs. ControlNet's shared base).
- Soften the abstract's "surpasses" to "is competitive with or surpasses" to accurately reflect the mixed conditional consistency results.

## Evaluation

**Originality:** The conditional decoding mechanism is a genuine and non-obvious insight. The application to arbitrary-resolution generation is creative. However, the overall framework (encode controls → inject into model) follows the ControlNet paradigm, so novelty is moderate. **6/10**

**Importance of research question:** Controllable generation for AR image models is timely and important given the rapid progress of AR models. **7/10**

**Whether claims are well supported:** The core conditional decoding vs. prefilling claim is well supported. The T2I "surpassing" claim is confounded. The arbitrary-resolution claim is under-evaluated. **5/10**

**Soundness of experiments:** Good ablations on the control mechanism itself, but the main comparison (T2I) has a confound, and the headline feature (arbitrary resolution) lacks FID evaluation. **5/10**

**Clarity:** The paper is generally well-written with clear figures and explanations. The distinction between conditional prefilling and decoding is clearly presented. **7/10**

**Value to community:** The conditional decoding insight and its application to arbitrary-resolution generation provide useful tools for the growing AR image generation community. **6/10**

## Calibration

Anchors compared against:

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| CCA (AR visual generation) | kGvXIlIVLM | 7.0 | Stronger theoretical grounding, cleaner experiments. ControlAR is below this due to confounded T2I comparisons and under-evaluated arbitrary-resolution. |
| Relay Diffusion (multi-resolution) | qTlcbLSm4p | 7.0 | Novel finding + SOTA on well-controlled comparisons. ControlAR is below due to confounds in main evaluation. |
| NOVA (AR video generation) | JE9tCwe3lp | 6.5 | Similar level of contribution in a different domain. ControlAR has similar methodological insight but weaker evaluation rigor. |
| Build-A-Scene (spatial control for diffusion) | gg6dPtdC1C | 5.75 | Borderline accept with interesting but limited scope. ControlAR has broader evaluation but with confounds. |
| Prodigy (overclaimed parameter-free learner) | WpQbM1kBuy | 4.25 | Overclaimed with unfair comparisons. ControlAR has a more genuine core contribution (conditional decoding), so above this. |
| EfficientQAT (full fine-tuning undermines efficiency) | 6Mdvq0bPyG | 3.0 | Full fine-tuning contradicts efficiency claims. ControlAR's case is less severe — efficiency isn't its main claim, and the full fine-tuning is about effectiveness. |
| H-QLoRA (claims efficiency, actually slower) | B4S1GAMBLG | 2.0 | Core efficiency claim directly contradicted. ControlAR doesn't make deployment efficiency claims, so well above this. |

ControlAR sits above the overclaiming/reject papers (~4-5) because the conditional decoding insight is genuine and the C2I results are strong and unconfounded. It sits below the borderline-accept and accept papers (~6-7) because its main T2I evaluation is confounded and its headline feature (arbitrary resolution) lacks FID evaluation. The paper makes real contributions but oversells its T2I results and under-evaluates its most novel feature.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>