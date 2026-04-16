Now let me look at the calibration papers referenced by the Human Finder reviewer.Now I have enough information to write the consolidated review. Let me verify one key claim from the harsh critic about LPIPS for captioning by cross-referencing the ablation figure.

Based on the paper reading: Figure 3 mentions "Captioning (LPIPS)" but Figure 6 (ablation on the same captioning task) clearly uses "CIDEr/F1r" as the metric. Since Figure descriptions are extracted from image alt-text (which is often AI-generated or OCR-based), the "LPIPS" label in Figure 3 is almost certainly a PDF parsing artifact. I'll remove that criticism.

---

## Summary

This paper proposes the first neural image compression framework specifically designed for cloud-hosted Multimodal Large Language Models (MLLMs). It introduces a lightweight transform-neck module that adapts compressed image latents directly to an intermediate layer of the MLLM's visual encoder — bypassing full image reconstruction — alongside a novel surrogate loss (combining cross-entropy and distillation terms) that enables training by back-propagating only through the visual encoder, sidestepping the billion-parameter LLM. Three deployment scenarios are supported: fixed human-perception codec (d1), jointly updated multi-task codec (d2), and machine-only perception codec (d3).

---

## Strengths

- **Timely, well-motivated, and genuinely novel problem.** The paper is, to the best of its reviewers' knowledge, the first to tackle image compression specifically for MLLMs. The core justification — that existing coding-for-machine methods requiring full-MLLM backprop are infeasible at billion-scale — is sound and clearly stated in Section 2.2. This alone gives the paper a useful claim on novelty.
- **Elegant surrogate loss design.** The progressive three-stage training schedule (CE only → CE + distillation → distillation only) is well-motivated, and Figure 7 provides compelling qualitative evidence that CE focuses on foreground semantics while distillation corrects global feature alignment. The ablation in Figure 6(b) demonstrates that neither term alone is sufficient, validating the progressive combination.
- **Strong reported empirical gains.** The method achieves up to 60–80% bit-rate reduction over ELIC/TIC reconstruction baselines at equivalent task accuracy, and ~95% reduction in kMAC/pixel versus the post-processing baseline (Table 3). These are substantial numbers.
- **Practical flexibility.** The three application scenarios (d1/d2/d3) meaningfully cover real deployment configurations, from backward-compatible human-viewing-preserving codecs to fully machine-optimized pipelines.
- **Resource efficiency and lightweight design.** Training on a single RTX 4090 with only 13M additional parameters makes the method highly accessible. The complexity comparison in Table 3 is concrete and informative.
- **Reasonable breadth of evaluation.** Results span 6 MLLMs (LLaMA-Adapter, Honeybee, Shikra, V2L-Tokenizer, mPLUG-Owl2, Osprey), 4 tasks (captioning, VQA, REC, few-shot classification), and 2 codec architectures (ELIC, TIC), providing encouraging evidence of generalization.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Critically sparse bit-rate evaluation (only 2 operating points).** Figure 3 evaluates rate-accuracy trade-offs at only ~0.1 and ~0.2 bpp. This is insufficient to establish a proper rate-accuracy curve, and the headline "60–80% bit-rate reduction" claim relies on a very narrow operating range. Without at least 4–5 rate points spanning a broader range (e.g., 0.05–0.4 bpp), the BD-rate characterization of the proposed method is unverifiable. The VVC comparison (cited in the abstract as a key baseline) appears only in the supplementary (Section A.2), which was not available for review; the main-body evidence for the headline claim thus rests on two points, which is weak.

- **Thin baseline coverage relative to the "coding-for-machine" framing.** The paper explicitly positions itself against prior coding-for-machine literature, but the empirical comparison is only against (i) naive reconstruction and (ii) a post-processing U-Net with the same surrogate loss. There is no direct comparison to feature coding approaches (e.g., compressing intermediate CLIP features directly) or to a codec+post-processing chain using a traditional codec (VVC/H.265). The paper argues these are impractical for MLLMs due to full-model backprop cost, which may well be true — but then the paper should be presented as a practical alternative in a constrained deployment setting, not as a general performance winner over the "coding-for-machines" design space. As written, the comparison does not substantiate the broader superiority claims.

### Minor

- **Overclaiming in abstract and conclusion.** The abstract states the framework is "applicable to various MLLMs… and multiple application scenarios" and the conclusion says the surrogate loss "ensures downstream task performance." These are overclaims. The evidence supports "works well on tested MLLMs and tasks," and the loss provides a training signal that empirically helps, but does not guarantee downstream performance. The paper would be stronger with more calibrated language.

- **Dependence on access to the visual encoder at training time.** The transform-neck training requires access to the partial visual encoder C'. In practice, cloud MLLM providers may not expose intermediate visual encoder activations or architecture details. The paper does not discuss this practical constraint or what users should do if the visual encoder is inaccessible or updated post-deployment. This is a real deployment concern that deserves at least a paragraph.

- **Sparse justification for ImageNet-only surrogate training across diverse downstream tasks.** Phase 1 training uses only ImageNet classification with text labels. The tasks evaluated (VQA, captioning, REC) go well beyond classification semantics. While the results show empirical transfer, the paper provides no analysis of whether the ImageNet vocabulary creates a systematic bias (e.g., better performance on object-centric images vs. relational or text-heavy images). The Section 4.5 ablation on loss components is good but does not address the domain gap from training data.

- **Encoding-side complexity is unreported.** Table 3 reports only decoding-side complexity. For scenarios d2 and d3 where the encoder is re-trained, the encoding-side computational budget (on the end device) is equally important and should be reported.

### Trivial

- The few-shot classification setup is custom (5-way 1-shot) rather than reproducing the original paper's setting, because the code was inaccessible. This is acknowledged honestly, but reduces the comparability of that task to published benchmarks.

---

## Nice-to-Haves

- **BD-rate analysis** over a denser set of rate points (at minimum 4–5 points, ideally covering 0.05–0.5 bpp) would turn Figure 3 into a rigorous evaluation.
- **Sensitivity analysis on the CE loss** (varying ImageNet label count m, or swapping to a different classification dataset) to test how robust the text-bridging mechanism is.
- **Feature coding baseline:** compressing intermediate CLIP ViT-L/14 features directly would provide a much more direct comparison and help position the latent-domain transform-neck approach against the feature-coding paradigm in Figure 1(c).
- **Failure case analysis** showing where the method underperforms relative to uncompressed input (e.g., at very low bitrates, fine-grained spatial tasks) would build trust and inform future work.
- **Cross-codec transfer test** (e.g., transform-neck trained with ELIC applied to TIC at inference) would significantly strengthen the generality claims.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**1. [Harsh Critic] Captioning metric in Figure 3 is LPIPS (potential fatal flaw).**
*Removed because:* Figure 6, which performs ablations on the same captioning task using the same method, clearly reports "CIDEr/F1r" as the evaluation metric. The "LPIPS" label in Figure 3's description comes from PDF-extracted image alt-text and is almost certainly a parsing artifact. The paper elsewhere uses standard text-generation metrics for captioning. This is not a real paper flaw.

**2. [Harsh Critic] Phase 2 CE loss disappearance invalidates bridging.**
*Removed because:* The paper explicitly explains the two-phase progressive strategy (Section 3.5). Phase 1 uses CE to bootstrap the transform-neck toward text-aligned features; Phase 2 uses rate-distortion + distillation for codec joint optimization. Dropping CE in Phase 2 is a deliberate and reasonable design choice — distillation directly constrains feature matching to the visual encoder output, making a separate CE text-alignment term redundant once the geometry is established. The ablation validates the overall design.

**3. [Harsh Critic] Evidential claim that the paper's broader "general applicability" claims are unsupported because training signal is ImageNet classification only.**
*Weakened (per soft rules — already partially addressed in paper):* The paper does cover 6 MLLMs, 4 tasks, and 2 codecs, and explicitly notes in Section 4.1 that the shared CLIP visual encoder is the key generalization driver. The "general" claim is mostly warranted with slightly overclaimed language, not a substantive methodological failure. Retained in weakened form as a minor overclaiming issue.

**4. [Harsh Critic] No confidence intervals / variance reporting for MLLM evaluations.**
*Moved to Nice-to-Have:* Single-run evaluation is the standard practice for large-scale MLLM benchmarks (SEED-Bench, MMBench, POPE, RefCOCO). Requiring confidence intervals is not standard in this field. The concern is noted but not a core weakness.

**5. [Human Finder] Missing ICM baselines such as Omni-ICM and TransTIC.**
*Removed per hard rules:* Do not raise missing related works, as we cannot confirm their existence or relevance without external sources.

**6. [Spark] VVC baseline is in abstract but not in main body.**
*Partially addressed:* The paper explicitly references Section A.2 for VVC comparisons. Since the appendix was not included in the review submission, this is a legitimate minor concern about the headline claim depending on supplementary material — already captured in the "sparse bit-rate evaluation" major weakness above.

---

## Novel Insights

The most genuinely insightful observation across the reviews is the progressive loss decomposition finding: the cross-entropy term reduces feature-matching error in *foreground semantics* (objects, regions) while the distillation term reduces global *feature topology* error (Figure 7). This complementarity explains why neither term alone is sufficient, and why CE must precede distillation in the curriculum — a design insight that generalizes beyond this specific paper to any adapter training scenario where a task network must be aligned to both a semantic label space and a target feature geometry simultaneously.

---

## Suggestions

1. **Add at least 3 more rate points** (e.g., 0.05, 0.15, 0.3, 0.4 bpp) to Figure 3, and report BD-rate reduction relative to the reconstruction baseline for each task.
2. **Move the VVC comparison from the appendix into the main body** (or Table 2) to substantiate the headline claim.
3. **Add a discussion paragraph on deployment assumptions**, specifically: what should practitioners do if the visual encoder is inaccessible or updated? A simple framing of "this method requires one-time white-box access to C' for training, after which inference is black-box" would suffice.
4. **Include a feature coding baseline** (e.g., directly quantizing CLIP ViT-L/14 layer-3 features) to empirically position against the Figure 1(c) paradigm the paper argues against.
5. **Report encoder complexity** (kMAC/pixel for g_a after retraining under d2/d3) to complete the complexity story for the end-device setting.

---

## Score and Decision

**Calibration:**

- **3D0mOtnHGR** (Preprocessing for Machine Vision, Withdrawn/Reject, 3/3/5 avg ~3.7): A weaker paper with unclear value proposition, limited generalization, and no novel training insight. This paper is clearly above that.
- **x33vSZUg0A** (Multi-task Representation Compression, Accept Poster, 3/5/8 avg ~5.3): Novel framework, adequate experiments, some clarity issues. Comparable in ambition; the paper under review has a clearer problem framing and practical deployment motivation.
- **ODiY6pbHZQ** (Oryx MLLM, Accept Poster, 6/6/6/6): More comprehensive evaluation, stronger empirical coverage; this paper is somewhat below that level due to sparse rate evaluation and baseline gaps.
- **UQJ7CDW8nb** (LLaVA-Mini, Accept Poster, 8/6/6/6 avg ~6.5): Higher technical rigor and more extensive evaluation; the paper under review is below this.

**Assessment:** The paper occupies genuine novelty territory (first MLLM compression paper), has a well-motivated and efficient method, and shows cross-task/cross-MLLM empirical transfer. The major gaps — two-point rate evaluation, thin coding-for-machine baselines, and modest overclaiming — place it above the rejected preprocessing paper (3.7) but below the more rigorously evaluated accepted posters (6–6.5). The contribution is real but evaluation rigor is below ICLR poster standard. This lands at a **borderline weak accept / borderline reject**.

**Score: 5.5** — The problem is novel and the approach is sound, but the evaluation is too sparse and the comparative baselines too limited to fully substantiate the headline claims. With a denser rate-accuracy evaluation, a feature-coding baseline, and the VVC comparison moved into the main body, this would comfortably be a 6.5+.

**MY FINAL SCORE: <pineapple>5.5</pineapple>**
**MY FINAL DECISION: <orange>Reject</orange>**