Now I have enough context. Let me synthesize all the reviews into one authoritative final review.

Key things to verify:
1. The harsh critic raises the concern that the surrogate loss doesn't guarantee task preservation. Let me check: The paper does show empirical results that it works across 4 tasks and 6 MLLMs. The claim "ensures downstream task performance" in the abstract is indeed strong, but the empirical evidence does support effectiveness. This is a valid concern about overclaiming but not a fatal flaw.

2. Universality claim: The paper says "the transform-neck trained with our surrogate loss exhibits a degree of universality, since it is readily applicable to multiple MLLMs that share the same visual encoder, without the need for retraining." This is actually an important distinction - the paper does NOT claim universality across all MLLMs regardless of visual encoder. It specifically claims universality ACROSS MLLMS SHARING THE SAME VISUAL ENCODER. But as the spark reviewer notes, it's unclear whether the same single transform-neck was used across all four CLIP-based tasks, or whether separate ones were trained.

Let me re-read the paper text carefully: "For each of our scenarios (d1), (d2) and (d3), separate transform-necks are trained on ImageNet dataset for individual λ values." This means separate transform-necks PER λ value and per scenario. But across MLLMs sharing the same CLIP ViT-L/14 visual encoder, they use the same transform-neck. This is stated in Section 4.1: "since we consider MLLMs sharing the same visual encoder, we do not need to train separate systems for the different MLLMs or tasks."

3. Baselines: The paper compares against Reconstruction and Post-processing. Missing VVC curves in main figures, missing standard codec comparisons, missing feature compression baselines. The VVC comparison is mentioned in the abstract as being in Sections 4.2 and A.2.

4. The 95% reduction in kMAC/pixel claim - this is specifically comparing the decoding complexity of the transform-neck vs. Post-processing, not claiming 95% end-to-end savings. The abstract says "nearly 95% reduction in decoding kMAC/pixel as compared to performing full image reconstruction followed by enhancing the reconstructed image." This is carefully scoped to the decoding component.

5. The harsh critic's concern about "surrogate loss does not guarantee MLLM task preservation" - while theoretically valid, the paper demonstrates empirical effectiveness across 4 diverse tasks and 6 MLLMs. The claim is empirically supported, though the language could be more measured. This is a valid criticism of overclaim, not a fatal flaw.

6. The "universality" claim: The paper states "exhibits a degree of universality" which is more measured. And for non-CLIP encoders, they do retrain, which they're transparent about. The claim is that MLLMs sharing the same visual encoder can share the same transform-neck - which IS demonstrated (4 MLLMs share CLIP ViT-L/14 and use the same neck).

Now let me check calibration papers:
- "Preprocessing Enhanced Image Compression for Machine Vision" (scores 3,3,5) - rejected/withdrawn. Similar domain (image compression for machines), weaker method, limited baselines.
- "LLaVA-Mini" (scores 8,6,6,6) - accepted poster. Related MLLM efficiency work, strong results.
- "LLaVA-PruMerge" (scores 3,3,5,3) - withdrawn/rejected. MLLM token reduction, limited evaluation.
- MLLM visual encoder papers generally scored 3-6.

This paper sits in an interesting space - it's genuinely novel (first to address compression for MLLMs), has practical significance, and shows strong empirical results. But it has overclaiming issues and limited baselines. It's better than the rejected compression-for-machines paper but not as polished as LLaVA-Mini.

I'd rate this around 5-6. It's a solid contribution with practical importance but the overclaiming and limited baselines detract from the evaluation.</think>Now I have sufficient calibration context. Let me write the final consolidated review.

## Summary

This paper proposes the first framework for adapting neural image compression latents to Multimodal Large Language Models (MLLMs). The key idea is a lightweight "transform-neck" module that directly maps compressed image latents into the intermediate layers of an MLLM's visual encoder, bypassing full image reconstruction. A surrogate loss (combining distillation and cross-entropy terms computed only through the visual encoder) enables training without back-propagating through the billion-parameter LLM. The framework supports three scenarios: frozen human-perception codec (d1), joint human+machine optimization (d2), and machine-only optimization (d3). Results across four tasks and six MLLMs show significant bitrate savings and computational advantages over reconstruction-based baselines.

## Strengths

- **Novel and timely problem formulation**: Being the first to tackle neural image compression specifically for MLLM-based inference is a genuine contribution. The practical motivation (bandwidth-constrained edge-to-cloud with MLLMs) is clear and increasingly important.

- **Practical training solution**: The surrogate loss design that avoids back-propagation through the entire MLLM is well-motivated and enables training on a single RTX 4090 with 24GB memory. This is a meaningful practical advantage over prior coding-for-machines approaches that require back-propagation through the downstream network, which is infeasible for billion-scale models.

- **Strong empirical results**: Consistent and substantial improvements over "Reconstruction" baseline across four diverse tasks (captioning, VQA, REC, few-shot classification) and six MLLMs, with up to 60-80% bitrate reduction claims verified in the figures. The d3 scenario approaching uncompressed performance is notable.

- **Useful flexibility across scenarios**: The three scenarios (d1, d2, d3) with corresponding training objectives and the documented reconstruction quality trade-offs (Figure 4) provide practical deployment flexibility. The ~95% kMAC/pixel savings in the decoder portion is well-supported by Table 3.

- **Good ablations**: The training objective ablation (Figure 6b), partial visual encoder ablation (Figure 6a), and codec generality ablation (Figure 6c) provide useful design insights. The MSE heatmap visualization (Figure 7) offers intuitive justification for the progressive training strategy.

## Weaknesses

### Fatal

None.

### Major

- **Overclaimed universality and generality**: The paper states the framework is "broadly applicable to a wide range of neural image codecs and MLLMs, regardless of their architectures" and that the transform-neck "exhibits a degree of universality, since it is readily applicable to multiple MLLMs that share the same visual encoder, without the need for retraining." While the method works across MLLMs sharing the CLIP ViT-L/14 encoder (demonstrated on 4 MLLMs), for non-CLIP encoders (mPlug-Owl2, Osprey in Section 4.6), the system is re-trained entirely — this does not validate universality but rather demonstrates the method's applicability with retraining. The "regardless of their architectures" claim is contradicted by the paper's own evidence. This distinction matters because a key selling point is avoiding per-MLLM retraining. The paper should more precisely scope the universality claim to "MLLMs sharing the same visual encoder" and make clear that different encoders require retraining.

- **Limited baseline comparisons for rate-accuracy tradeoffs**: The paper compares against only two baselines: "Reconstruction" (naïve decoded image → MLLM) and "Post-processing" (U-Net adapter on decoded images). Missing comparisons include: (1) standard codecs (JPEG, WebP, VVC) at matching bitrates — VVC is mentioned in the abstract but the main body lacks explicit VVC rate-accuracy curves, with details deferred to the appendix; (2) other coding-for-machines approaches adapted as fairly as possible (the paper dismisses them as infeasible for MLLMs, but a baseline that trains only through the visual encoder using task-specific losses would provide a direct comparison); (3) simple resolution downsampling baselines. The "60-80% bitrate reduction" headline claim is primarily established against an outdated "Reconstruction" baseline that was never designed for machine consumption. These gaps make it hard to assess the real magnitude of improvement.

- **Surrogate loss effectiveness is empirically demonstrated but not analytically or correlationally justified**: The central claim is that a surrogate loss (distillation + cross-entropy) computed only through the visual encoder suffices to preserve MLLM task performance. While the results confirm this empirically for the tested configurations, no analysis shows that minimizing this surrogate objective correlates with actual downstream task metrics across training (e.g., loss-vs-task-accuracy plots). The cross-entropy term uses ImageNet classification labels and CLIP text encodings, yet the downstream tasks (captioning, VQA, REC) involve generative, open-ended reasoning far beyond classification. The paper does not analyze whether this semantic mismatch matters, nor does it test sensitivity to the choice of classification dataset or text encoder. This is not fatal because the method works, but it weakens the claims of generality.

### Minor

- **Separate transform-necks per bitrate and scenario**: A different transform-neck must be trained for each λ (bitrate point) and each scenario (d1/d2/d3). This increases deployment complexity, which is somewhat at odds with the universality framing. The paper could mention this limitation more prominently.

- **Layer selection heuristic for visual encoder bypass**: The decision to skip the first two layers of CLIP ViT is based on an ablation (Figure 6a) performed only on the captioning task. Effects on other tasks are not verified, and no principled mechanism for selecting the injection point is offered for other visual encoders.

- **Progressive training hyperparameters lack sensitivity analysis**: The epoch thresholds E₁=20, E₂=40, the α:β=1:100 ratio, and the γ:δ=60:1 ratio for scenario d2 are all stated as "empirically set" with no sensitivity analysis, limiting confidence in their robustness.

- **No statistical significance reporting**: Results are presented as single runs without error bars, confidence intervals, or variance across seeds.

- **Bitrate evaluation range is narrow**: All main experiments operate at 0.05-0.2 bpp. Behavior at moderate-to-high bitrates is not shown, making the "60-80% bitrate reduction" claim hard to contextualize.

### Trivial

- The abstract claims "ensures downstream task performance" — while the results support this, "ensures" is too strong given the empirical-only validation.

## Nice-to-Haves

- Correlation analysis between surrogate loss values and downstream task metrics during training to validate the proxy.
- Comparison against a simple image downsampling baseline at matched bitrates.
- Sensitivity analysis of transform-neck performance when varying the classification dataset used for the CE term.
- Evaluation on a more demanding task (e.g., fine-grained recognition or long-form reasoning) to stress-test the framework.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"95% complexity reduction is misleading without end-to-end analysis"**: The paper explicitly scopes this claim to "decoding kMAC/pixel" and compares against "performing full image reconstruction followed by enhancing the reconstructed image" — not claiming end-to-end system savings. The comparison is specific and fairly stated in context (Table 3). Removing because the critic mischaracterized the scope of the claim.

- **"No end-to-end upper bound experiment (full MLLM backprop)"**: The paper's core motivation is that full MLLM backprop is infeasible (billion-scale models). Demanding such an experiment as a baseline contradicts the paper's stated constraint. This is analogous to demanding a "fair" comparison that favors the baseline, which the rules say to weaken/remove.

- **"Reproducibility concerns about model availability"**: The paper cites specific, published models (LLaMA-Adapter, Honeybee, Shikra, V2L-Tokenizer, mPlug-Owl2, Osprey, ELIC). Per the rules, cited models are assumed to exist.

- **"Few-shot classification setup is ad hoc due to unavailable code"**: The paper acknowledges this and provides details in the supplementary material. This is a minor implementation detail, not a methodological flaw.

- **"The paper is not conceptually radical; latent-level coding and feature coding exist"**: The paper explicitly discusses these prior works (Section 2.2) and differentiates itself on the key point that existing methods require backpropagation through the downstream network, which is infeasible for MLLMs. This is not a valid weakness.

- **"No failure case analysis"**: While failure cases could improve the paper, their absence does not undermine the core claims given the breadth of experiments across 4 tasks and 6 MLLMs. Moved to nice-to-have.

## Novel Insights

The progressive training strategy (CE then CE+distillation then distillation-only) is an interesting design choice that the MSE heatmap analysis (Figure 7) helps explain: CE loss focuses alignment on foreground object regions first, while distillation loss provides global feature alignment. This suggests that the two losses serve complementary roles — CE provides semantic direction while distillation refines local feature fidelity — which is a potentially useful insight beyond the immediate contribution.

## Suggestions

- Scope the universality claim precisely: state that the same transform-neck works across MLLMs sharing the same visual encoder without retraining, while different visual encoders require retraining. Remove the "regardless of architectures" phrasing from the abstract.

- Add at least one comparison with a standard codec (e.g., JPEG or VVC) at matched bitrates in the main results, even if only for one task, to contextualize the bitrate savings claims.

- Provide a loss-vs-task-metric correlation plot during training to strengthen the argument that the surrogate loss is a reasonable proxy for downstream task performance.

- Report whether a single transform-neck (trained once) is used across all four CLIP-based MLLM experiments, or whether separate necks are trained per task. If shared, report per-task performance variance; if separate, adjust the universality framing accordingly.

## Evaluation

**Originality**: High — this is the first work to address neural image compression specifically for MLLMs, a genuinely novel problem setting.

**Importance of research question**: High — edge-to-cloud MLLM inference with image compression is a practically important and growing problem.

**Claims well supported**: Partially — empirical results are strong, but universality/generality claims exceed what the evidence supports, and baseline comparisons are limited.

**Soundness of experiments**: Moderate — the design is reasonable and ablations are informative, but limited baselines and lack of statistical reporting are weaknesses.

**Clarity**: Good — the paper is well-organized with clear notation and helpful figures.

**Value to community**: High — opens a new direction at the intersection of image compression and MLLM efficiency.

## Score and Decision

**Calibration**: Compared to "Preprocessing Enhanced Image Compression for Machine Vision" (scores 3,3,5; rejected) — this paper is substantially stronger in novelty (first to address MLLMs), more rigorous in methodology, and broader in evaluation. Compared to "LLaVA-Mini" (scores 8,6,6,6; accepted poster) — this paper has comparable practical motivation but weaker experimental rigor (limited baselines, narrower evaluation scope). Compared to "LLaVA-PruMerge" (scores 3,3,5,3; withdrawn) — this paper is stronger in problem novelty and empirical breadth, though it shares some overclaiming tendencies. The paper is stronger than the rejected compression-for-machines paper but has meaningful weaknesses in overclaiming and limited baselines that prevent it from reaching the level of accepted poster papers like LLaVA-Mini.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>