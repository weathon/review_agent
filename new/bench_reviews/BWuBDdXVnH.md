Now let me run calibration searches to find anchor papers for scoring.Now I have enough to write the final consolidated review.

---

## Summary

ControlAR introduces a framework for adding spatial control (edges, depth maps, segmentation masks) to autoregressive (AR) image generation models. The key contributions are: (1) a lightweight ViT-based control encoder that transforms spatial control images into control token sequences, (2) "conditional decoding" — additive fusion of control tokens into image tokens at select AR model layers during generation (rather than prefilling), and (3) an extension enabling arbitrary-resolution AR generation by adjusting the control token count. The method is validated on both Transformer-based (LlamaGen) and Mamba-based (AiM) AR models and compared against diffusion-based controllable methods like ControlNet and ControlNet++.

---

## Strengths

- **Conditional decoding clearly outperforms prefilling both in quality and efficiency**: Fig. 2(c)-(d) shows conditional decoding achieves better FID (~25 vs ~35 at convergence) and F1-Score, while requiring only +27% GPU memory and +17% training time over the no-control baseline, versus +102% and +130% for prefilling. This is the central technical contribution and is strongly supported.

- **Multi-architecture generalization**: Table 1 demonstrates ControlAR works on both LlamaGen (Transformer) and AiM (Mamba) across canny edge and depth control, establishing that conditional decoding is architecture-agnostic and not a LlamaGen-specific trick.

- **Comprehensive ablation studies**: Tables 4–6 systematically cover control encoder choice (CNN vs. ViT vs. DINOv2), fusion strategy (cross-attention vs. addition, layer placement), and training strategy (freeze vs. LoRA vs. full fine-tune), providing actionable design guidance.

- **Addresses a genuinely underexplored problem**: Controllable generation for AR image generation models is largely unexplored. This paper is an early, principled investigation into adapting ControlNet-like spatial control to the next-token-prediction paradigm.

- **MR-ControlAR shows meaningful improvement for non-square resolutions**: Fig. 6(b) shows multi-resolution training maintains ~85.5 SSIM across resolutions from 1:1 to 2:1 aspect ratios, while standard ControlAR degrades to ~80.5 SSIM at 2:1 — a practically useful result requiring no extra parameters.

---

## Weaknesses

### Fatal
None.

### Major

- **The abstract's claim "surpasses previous state-of-the-art controllable diffusion models, e.g., ControlNet++" is inaccurate as stated.** Table 2 confirms ControlNet++ beats ControlAR on Seg-ADE20K (43.64 vs. 39.95 mIoU), Lineart (83.99 vs. 79.22 SSIM), and Depth (28.32 vs. 29.01 RMSE). Table 3 confirms ControlNet (not even ControlNet++) beats ControlAR on Canny FID (14.73 vs. 17.51). ControlAR wins on FID for hed, lineart, depth, and seg-COCOStuff, but the headline "surpasses" is selectively true at best. The win pattern is mixed and moderate ("competitive" would be accurate); the abstract oversells it.

- **Base model scale asymmetry makes comparative attribution difficult.** All T2I diffusion baselines (ControlNet, ControlNet++, T2I-Adapter, etc.) are built on SD1.5, while ControlAR uses LlamaGen-XL with a T5 text encoder (775M parameters). The paper never reports the unconditional FID of LlamaGen-XL on these benchmarks, making it impossible to determine whether the FID improvements in Table 3 reflect a better control mechanism or simply a stronger base generative model. This isn't necessarily fatal — ControlAR is genuinely proposed as an AR solution — but the paper does not provide evidence to disaggregate these factors, and the claims should be scoped accordingly.

### Minor

- **The sole AR-vs-AR comparison (Table 1, ControlVAR) is compromised by the noted histogram estimation.** The footnote acknowledges "ControlVAR's FID values are estimated from its histograms." The comparison lacks conditional consistency metrics (F1-Score, RMSE) for ControlVAR, which is attributed to ControlVAR not releasing them. While the paper is transparent about this, the only head-to-head AR comparison is both imprecise in FID and incomplete in controllability metrics.

- **Full fine-tuning is required for competitive results, with no analysis of forgetting.** Table 6 shows the performance gap between full fine-tuning and LoRA is large (F1: 34.15 vs. 32.90; FID: 10.64 vs. 13.20), and freezing is substantially worse. Unlike ControlNet's frozen-backbone approach, ControlAR requires retraining the AR sequence model's weights per control type. The paper does not evaluate whether full fine-tuning degrades the base model's text-conditional or unconditional generation capability — a non-trivial question with shared weights.

- **Layer spacing ablation is narrow.** Table 5 ablates only the 1-st, 1/5/9-th, and all-12 options. The paper does not explore varying the number of conditional layers (e.g., 2 or 4 instead of 3) or different spacing patterns, leaving the sensitivity to this design choice unclear.

- **Arbitrary-resolution evaluation is limited to a single metric (SSIM) on one control type (hed edge).** No FID or image quality metric is provided across non-square resolutions, making the "arbitrary-resolution image generation" contribution harder to quantify than the other results.

### Trivial

- The paper uses method-specific evaluation datasets (each method evaluated on its own task-matched subset), which introduces mild selection bias in Tables 2–3. This is common practice but worth noting.

---

## Nice-to-Haves

- Unconditional or no-control FID of LlamaGen-XL on the evaluation benchmarks would allow readers to disaggregate base model quality from control mechanism effectiveness.
- Analysis of whether full fine-tuning causes forgetting of general text-conditional generation capability would clarify the practical trade-off vs. frozen-backbone methods.
- A brief discussion of how ControlAR would handle multi-condition fusion (multiple simultaneous controls), which is a natural extension of the method.
- FID and quality metrics at non-square resolutions for the MR-ControlAR evaluation would strengthen the arbitrary-resolution claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Conditional decoding is a known approach presented as novel decoding paradigm"** *(Harsh Critic)*: While additive conditioning at intermediate layers is not new in the diffusion world (FiLM, ControlNet feature injection), the specific framing in the context of AR next-token prediction — contrasting it empirically with prefilling as the natural AR alternative — provides genuine value and a clear contribution. The paper explicitly frames the comparison against prefilling, not against prior diffusion conditioning work. Removed as overcritical; the AR-specific framing has independent merit.

- **"The 'surprising' arbitrary-resolution claim is trivial"** *(Harsh Critic)*: While the mechanism is a natural consequence of token-count alignment, the practical implication (removing AR's fixed-resolution constraint without architectural changes) is non-trivial for the AR image generation community, where models like LlamaGen have been restricted to 256×256. Weakened — the claim is perhaps overemphasized but not dishonest.

- **"Comparison with ControlNet++ has FID wins from stronger base model" attributed as unfair** *(Harsh Critic)*: Per the hard rules, asymmetry in base model scale that favors the *baseline* (SD1.5 is a well-optimized, widely validated model; LlamaGen is newer and arguably less mature) should not be treated as unfair to the authors. The concern that FID gains track base model quality rather than control mechanism is retained but moved to Major, not characterized as disqualifying.

- **"Claim that conditional decoding avoids positional correspondence learning is unrigorous"** *(Harsh Critic)*: The paper's claim is a characterization of the inductive bias, not a theorem. The positional information is implicitly encoded by token ordering in the fused sequence, which does simplify the mapping relative to cross-attention over a separate prefix. Removed as nitpick.

- **Requests for missing related works** *(Harsh Critic)*: Removed per hard rules — cannot verify existence.

- **Request for FID comparison vs. diffusion methods at non-square resolutions** *(Harsh Critic)*: Scope creep; the arbitrary-resolution section is specifically about releasing AR models from fixed-resolution constraints, not about beating diffusion baselines at those resolutions. Kept as Nice-to-Have only.

---

## Novel Insights

The most genuinely insightful observation from the review pool is the **base model disaggregation problem**: ControlAR's FID improvements over diffusion baselines in Tables 2–3 may reflect the stronger base generative model (LlamaGen-XL + T5) rather than the conditional decoding mechanism itself. The absence of unconditional/no-control FID for LlamaGen-XL on these benchmarks means no reader can determine whether ControlAR's improvement is primarily in *conditioning* or in *generation*. This is a subtle but important methodological gap that the authors should address with a single additional number per task.

---

## Suggestions

1. **Rewrite the abstract and introduction claims**: replace "surpasses ControlNet++" with "achieves competitive performance with ControlNet++, exceeding it on FID and several control conditions while falling short on others." The current phrasing will be challenged in review.

2. **Report unconditional FID of LlamaGen-XL** on the T2I benchmarks (ADE20K, COCOStuff, MultiGen-20M subsets) to allow readers to attribute performance differences to the control mechanism rather than the base model.

3. **Evaluate forgetting**: measure text-conditional generation FID/CLIP score before and after ControlAR full fine-tuning to quantify the cost of retraining the sequence model.

4. **Add FID for MR-ControlAR** at different aspect ratios (alongside SSIM) to more fully characterize the arbitrary-resolution contribution.

---

## Score and Decision

**Calibration:**

- **HART** (hybrid AR tokenizer for 1024px generation, avg ~6.8, Accept-Poster): Strong engineering contribution to AR image generation with good ablations and competitive results, some open questions. ControlAR is comparable in contribution scope but with more comparison-validity concerns.

- **CCA** (guidance-free AR visual generation, avg ~7, Accept-Oral): Strong theoretical motivation and impressive single-epoch fine-tuning results. Higher novelty and cleaner contribution than ControlAR.

- **Low-end papers** (TCIG: avg ~1.5; GenLLaVA: avg ~4.25; yPxhj1FKhG APCtrl: avg ~3.7): These lack genuine contribution or have fatal comparison issues. ControlAR is substantially above these — it addresses a real, underexplored problem with a working method.

- **Ctrl-U** (uncertainty-aware reward modeling for controlled image generation, avg 6.0, Accept-Poster): Solid engineering paper in the controllable generation space, comparable quality. Scored 6 by humans.

**Assessment**: ControlAR is a solid engineering contribution addressing a genuinely underexplored problem (controllable AR generation). Its core technical contribution (conditional decoding vs. prefilling) is well-validated empirically with comprehensive ablations and multi-architecture generalization. The main issues are: (1) overclaimed superiority in the abstract; (2) structural comparison weakness that makes FID gains hard to attribute to the control mechanism; (3) incomplete AR-vs-AR comparison. These are meaningful but non-fatal concerns that don't invalidate the contribution. The paper is above HART-level in practical impact (ControlAR is the first strong solution to a clear gap; HART is an incremental improvement on an existing strong paradigm) but below CCA in theoretical clarity.

Positioning between Ctrl-U (avg 6.0) and HART (avg 6.8), closer to the lower end given the overclaim issues: **6.0**.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>