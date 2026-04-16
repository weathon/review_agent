## Summary

This paper presents the first framework for adapting neural image compression latents to Multimodal Large Language Models (MLLMs). The key contributions are a lightweight transform-neck that bridges compressed latents directly to an intermediate layer of the MLLM's visual encoder (bypassing full image reconstruction), and a surrogate loss combining distillation and cross-entropy that enables training without back-propagating through the billion-parameter LLM. The framework supports three scenarios: fixed codec for human perception, multi-task codec optimization, and machine-only codec optimization.

## Strengths

- **Novel and timely problem formulation.** The paper identifies a real gap—existing coding-for-machines methods require back-propagation through downstream networks—and proposes the first framework specifically targeting image compression for MLLMs. The motivation of cloud-hosted MLLMs with bandwidth-constrained end devices is practical and increasingly relevant.

- **Clean and practical technical approach.** The transform-neck is simple (linear projection + self-attention + FFN, ~13M parameters), operates in the latent domain to avoid full-resolution decoding, and the surrogate loss requires only the visual encoder during training, enabling training on a single RTX 4090. This is a meaningful practical advantage over end-to-end methods that would need to back-propagate through the entire MLLM.

- **Strong complexity advantage documented.** Table 3 provides a clear comparison showing ~95% reduction in kMAC/pixel compared to the Post-processing baseline (52.8 vs. 1017.96 kMAC/pixel) with ~80% fewer parameters (13M vs. 64M). This is a concrete and convincing efficiency claim.

- **Broad experimental coverage across tasks and models.** The paper evaluates on four tasks (captioning, VQA, REC, few-shot classification) with four different MLLMs, plus two additional non-CLIP-visual-encoder MLLMs (mPLUG-Owl2, Osprey), and two different codecs (ELIC, TIC). This demonstrates the method's applicability beyond a single setting.

- **Well-structured exploration of three deployment scenarios.** The (d1)/(d2)/(d3) framework covers practical deployment considerations from human-priority to machine-priority use cases, giving practitioners meaningful trade-off options.

## Weaknesses

### Major

- **The headline "60–80% bit-rate reduction" claim is not convincingly supported in the main experiments.** The abstract and Section 1 state "up to 60–80% bit-rate reductions under the same recognition accuracy over existing image codecs (e.g. ELIC and VVC intra coding)." However, the main results (Figure 3) show rate-accuracy curves at 0.05–0.2 bpp without explicit iso-accuracy comparisons or BD-rate calculations, which are standard in image compression literature. The VVC comparison is relegated to the appendix. Without quantitative BD-rate or iso-accuracy tables in the main text, this headline figure is not substantiated by the primary evidence the paper presents. This matters because the rate savings claim is the paper's central selling point.

- **Limited baseline comparisons.** The paper compares against only two baselines: naïve reconstruction and a Post-processing U-Net. Missing are: (1) a standard image enhancement method (e.g., SwinIR or similar) applied to decoded images, which would be a more competitive baseline than the U-Net; (2) a simpler CLIP-side adapter approach (e.g., fine-tuning a small adapter on top of CLIP for compressed images without modifying the codec); and (3) a feature coding baseline, which the paper discusses conceptually (Figure 1c) but never implements. The Post-processing baseline is structurally biased toward being expensive (full-resolution decoding + U-Net), which inflates the complexity advantage. Without stronger baselines, it is difficult to assess whether the transform-neck architecture is truly superior or simply the only serious alternative considered.

- **The surrogate loss is only indirectly validated as a proxy for MLLM performance.** The core claim—that optimizing distillation + cross-entropy loss at the visual encoder reliably preserves downstream MLLM output quality—is supported only by showing task performance at test time. The paper does not provide: (1) a correlation analysis between surrogate loss and MLLM task metrics during training; (2) an experiment with even a small MLLM where end-to-end back-propagation is feasible as an oracle comparison; or (3) analysis of when the proxy fails. The cross-entropy component uses ImageNet labels and CLIP text embeddings, while the evaluated downstream tasks (captioning, VQA, REC) involve open-ended language generation far beyond ImageNet classification. The ablation (Figure 6b) shows that CE alone fails and distillation alone fails, but does not establish that the combined surrogate predicts MLLM task performance well—it merely shows it produces better task metrics than either component alone. This is a meaningful evidential gap for the paper's core technical contribution.

### Minor

- **Generality claims are overstated relative to evaluation.** The paper claims the framework is "broadly applicable to a wide range of neural image codecs and MLLMs, regardless of their architectures." In practice, the main experiments use four MLLMs all sharing the same CLIP ViT-L/14 visual encoder, and Section 4.6 experiments on non-CLIP encoders require retraining. The claim of "universality" across MLLMs sharing the visual encoder is unsurprising since training only involves the visual encoder—it is an expected property, not a surprising finding. This does not invalidate the method, but the framing overstates what is actually demonstrated.

- **The captioning metric in Figure 3 is labeled "LPIPS,"** a perceptual image similarity metric, while Figure 6 uses "CIDF1r" (likely CIDEr) for the same task. This inconsistency is confusing and raises questions about which metric is primary for captioning evaluation. Standard captioning metrics (CIDEr, BLEU, SPICE) should be clearly reported.

- **Progressive training hyperparameters (E₁=20, E₂=40, α:β=1:100, γ:δ=60:1) lack sensitivity analysis.** These are stated as "empirically set" without exploration of alternatives. While this is common practice, the γ:δ ratio fundamentally determines the human/machine trade-off in scenario (d2), and its sensitivity is never analyzed.

- **Evaluation tasks skew toward semantic-level understanding** (captioning, VQA, classification) rather than tasks requiring fine-grained spatial detail (OCR, document understanding, dense prediction). It remains unclear how well the transform-neck preserves fine-grained visual information at low bitrates.

### Trivial

- No variance or confidence intervals are reported across tasks. While single-run evaluation is common in MLLM literature, this limits confidence in small observed differences, particularly for Figure 3 where some curves are close together.

## Nice-to-Haves

- A correlation analysis between surrogate loss values and downstream MLLM task metrics across training would strengthen the validation of the proxy objective.
- An experiment with a smaller MLLM (where end-to-end back-propagation is tractable) to directly compare the surrogate loss approach against full back-propagation.
- Evaluation on tasks requiring fine-grained spatial understanding (e.g., OCR, segmentation) to test the limits of latent-domain adaptation.
- Higher bitrate evaluations (0.3–1.0 bpp) to characterize performance across the full rate-accuracy spectrum.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Only two bitrate points evaluated" (Spark reviewer):** This is factually incorrect. The paper trains four ELIC models at four λ values (λ = 0.004, 0.008, 0.016, 0.032), corresponding to different rate points, and Figure 3 shows curves with multiple points, not just two.

- **"No VVC comparison in main paper" (Harsh Critic):** While VVC results are in the appendix (Section A.2) rather than the main text, this is a presentation choice, not an absence. The paper does provide VVC comparison. The concern about lacking explicit iso-accuracy comparisons in main text remains valid.

- **"The method is impractical because one could just extract features on-device" (Harsh Critic reviewer, similar to Preprocessing Enhanced review):** This ignores the core use case. Extracting CLIP features on an edge device still requires running the full visual encoder, which may be too resource-intensive for constrained end devices. The paper's scenario involves end devices with limited compute that transmit compressed data to a server running the MLLM. Running CLIP on-device defeats the bandwidth efficiency purpose at low bitrates, as CLIP features are much larger than compressed latents.

- **"Full end-to-end training should be compared using PEFT or LoRA" (Harsh Critic):** The paper explicitly argues that involving even parts of the MLLM in training is prohibitively expensive (Section 1, 2.2). While PEFT could theoretically reduce this cost, it would still require back-propagation through the MLLM, which the paper's approach deliberately avoids. This is a scope difference, not a methodological flaw.

- **"Reproducibility concerns about undisclosed hyperparameters" (general category):** The paper specifies key hyperparameters (E₁=20, E₂=40, α:β=1:100, γ:δ=60:1), codec architecture (ELIC with N=320, N_h=192), and training dataset (ImageNet). This is adequate for reproducibility in this domain.

## Novel Insights

The paper's most interesting conceptual finding is that the intermediate representations of a neural image codec can be adapted directly to the intermediate layers of a visual encoder via a lightweight transform-neck, bypassing both full image reconstruction and the downstream LLM. This aligns with a broader insight emerging in the MLLM efficiency literature: that visual information can be significantly compressed or transformed at the encoder/connector level while preserving downstream task performance. The progressive training strategy—starting with CE loss to establish semantic alignment before introducing precise feature distillation—reflects an intuition that classification-level "what" alignment should precede feature-level "how" alignment, which is a sensible design principle for multi-stage adaptation. However, the empirical gap between (d1) and (d3) performance suggests that significant further gains are available when the codec is retrained for machine perception, highlighting a practical tension: the most performant setting (d3) abandons image reconstruction, which may limit real-world adoption where human viewing remains a requirement.

## Suggestions

1. **Add explicit BD-rate or iso-accuracy calculations** in the main text to substantiate the "60–80% rate reduction" claim, or tone down the claim to match what the figures directly support.
2. **Include a stronger image-domain enhancement baseline** (e.g., SwinIR or another established method) to better contextualize the transform-neck's advantage over image-domain approaches.
3. **Report a correlation analysis** between surrogate loss values and downstream MLLM task metrics across training checkpoints to validate the proxy objective.
4. **Clarify captioning metrics**: use standard captioning metrics (CIDEr, BLEU, METEOR) consistently in the main figures rather than LPIPS.
5. **Discuss limitations honestly**: acknowledge that the method requires retraining for different visual encoders and that fine-grained spatial tasks may be more sensitive to low-rate compression artifacts.

## Score and Decision

**Calibration reasoning:** I compared this paper against several anchors:

- **Low anchors**: Preprocessing Enhanced Image Compression for Machine Vision (3,3,5 → Reject): Similar coding-for-machines topic but much weaker method and more severe overclaiming. The current paper is substantially stronger—cleaner formulation, more comprehensive evaluation, genuine MLLM novelty.
- **Mid anchors**: LVP (5,6,5,6 → Reject) and EMMA (6,5,5 → Reject): These are lightweight adapter papers for MLLMs with incremental novelty. The current paper has a more novel problem setting and is better validated, but shares similar concerns about novelty depth and baseline comparisons.
- **High anchors**: DEEM (6,8,8,8,6 → Spotlight) and Idempotence and Perceptual Image Compression (6,8,8,8 → Spotlight): These have stronger theoretical grounding and/or more compelling empirical evidence. The current paper is below this tier due to overclaiming and limited baselines.

The paper sits between the mid and low anchors. It has a genuinely novel and well-motivated problem formulation, a clean and practical solution, and demonstrated complexity savings. However, the overclaiming on rate reductions, limited baselines, and the evidential gap in validating the surrogate loss as a reliable proxy prevent it from meeting a clear acceptance threshold. These are not fatal flaws, but they meaningfully weaken the core claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>