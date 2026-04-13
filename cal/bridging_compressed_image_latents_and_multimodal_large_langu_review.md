=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary

This paper presents the first neural image compression framework tailored for Multimodal Large Language Models (MLLMs). The authors propose a lightweight "transform-neck" that adapts compressed image latents directly to the MLLM's visual encoder, bypassing full image reconstruction, and a "surrogate loss" combining distillation and cross-entropy terms that enables training without back-propagating through the billion-parameter LLM. The framework supports three scenarios: using a fixed human-perception codec, jointly optimizing for human and machine perception, or optimizing purely for machine perception.

## Strengths

- **Practical solution to an important deployment challenge:** The paper correctly identifies that existing coding-for-machines methods require back-propagation through the full downstream network, which is infeasible for billion-parameter MLLMs. The surrogate loss design (Section 3.4) provides a tractable training solution achievable on a single RTX 4090, making MLLM-aware compression practically viable.

- **Significant computational efficiency gains:** Table 3 demonstrates a ~95% reduction in decoding kMAC/pixel compared to image-domain post-processing (from ~900 kMAC/pixel to ~53 kMAC/pixel by operating in the latent domain rather than pixel space), with the transform-neck requiring only 13M parameters versus 64M for the post-processing baseline.

- **Comprehensive multi-task, multi-model evaluation:** The experiments span four distinct tasks (captioning, VQA, REC, few-shot classification) across four different MLLMs (LLaMA-Adapter, Honeybee, Shikra, V2L-Tokenizer), plus two additional non-CLIP MLLMs (mPlug-Owl2, Osprey) in generalization tests. Figure 3 shows consistent improvements over reconstruction baselines across all tasks.

- **Flexible application scenarios:** The three-scenario framework (d1: fixed codec, d2: joint optimization, d3: machine-only optimization) accommodates diverse real-world deployment constraints, as formalized in Table 1 with corresponding loss functions.

## Weaknesses

- **Potential training-evaluation data overlap:** The transform-neck is trained on ImageNet (Section 4.1: "separate transform-necks are trained on ImageNet dataset"), and one of the four evaluation tasks is few-shot classification on ImageNet (Table 2). While train/test splits may differ, the surrogate cross-entropy loss directly optimizes ImageNet class semantics—meaning the transform-neck learns representations specifically tuned to ImageNet categories before evaluation. The paper does not acknowledge this potential overlap or clarify how the splits were separated, raising concerns about potential experimental bias in one of four reported tasks.

- **Captioning metric inconsistency:** Figure 3 labels the captioning y-axis as "LPIPS (Score)," but LPIPS is an image quality metric, not a captioning metric. Figure 6 uses "CIDF1r" for the same task. This inconsistency undermines confidence in the reported captioning results and suggests possible confusion in figure preparation.

- **Transform-neck architecture lacks justification:** Section 3.3 describes the transform-neck as "a linear projection, a self-attention mechanism, a feed-forward layer, and two layer norms" but provides no justification for this specific architecture. The ablation studies in Section 4.5 explore where to connect to the CLIP encoder but not whether one self-attention layer is optimal, why not use a simple MLP, or how architecture depth affects performance.

- **Spatial dimension alignment unexplained:** The compressed latents $\tilde{y}$ have spatial dimensions $H/f \times W/f$ (with stride $f$), while CLIP ViT-L/14 expects patch tokens with specific spatial arrangement. The paper never explains how spatial alignment between codec latents and visual encoder intermediate features is achieved—a non-trivial implementation detail affecting reproducibility.

- **Cross-entropy loss CLIP-dependence for non-CLIP MLLMs:** The surrogate loss uses the CLIP text encoder to compute the cross-entropy term (Section 3.4), creating implicit CLIP bias even when the target MLLM uses a different visual encoder. Figure 8 shows smaller performance gaps for non-CLIP MLLMs (mPlug-Owl2, Osprey) compared to CLIP-based models, but the paper provides no analysis of whether this CLIP-anchored loss is suboptimal for such architectures.

- **Hyperparameter sensitivity unanalyzed:** The progressive training schedule uses fixed values ($E_1=20$, $E_2=40$, $\alpha:\beta=1:100$, $\gamma:\delta=60:1$ for d2) with no sensitivity analysis. Given that Figure 6(b) shows the balance between CE and distillation losses significantly impacts performance, readers cannot assess robustness of these choices.

- **Scenario (d3) decoder handling unclear:** For scenario (d3) where reconstruction quality is disregarded, the paper does not explicitly state whether the decoder $g_s$ weights are updated during Phase 2 joint optimization. This affects parameter count and training cost comparisons.

## Nice-to-Haves

- **BD-rate reporting:** Standard Bjøntegaard delta-rate calculations would enable more precise bitrate-reduction quantification than visually reading curves, though the presented rate-accuracy curves are sufficient for relative comparisons.

- **Confidence intervals on performance metrics:** Statistical significance tests or confidence intervals would strengthen claims of improvement, particularly for tasks with higher variance (few-shot classification, VQA).

- **Failure case analysis:** No examples or analysis of when the method underperforms relative to baselines are provided, limiting understanding of reliability boundaries.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Baseline insufficiency criticism:** The harsh critic demanded comparison to coding-for-machines methods. However, the paper explicitly argues these methods require back-propagation through the full downstream network—infeasible for billion-parameter MLLMs. The Post-processing baseline (U-Net trained with surrogate loss) is a fair alternative approach. Demanding comparison to methods that cannot be practically applied is scope creep.

- **"Universality is trivial by construction" criticism:** While transfer across CLIP-based MLLMs is indeed expected (since they share the same visual encoder), the paper's empirical verification across multiple MLLM architectures still has value. The wording could be more modest, but this is not a critical flaw.

- **"First-ever study overclaim" criticism:** The paper clearly positions itself relative to prior coding-for-machines work and explains the gap addressed. While related areas (VLM efficiency, token reduction) exist, this specific problem—neural compression designed for MLLMs—appears to be genuinely novel.

- **Token reduction method comparison:** Section A.6 discusses token reduction as orthogonal and combinable. Demanding combined experiments is a nice-to-have, not a core weakness, since the paper's contribution is the compression framework itself.

- **Testing at more extreme compression rates:** The 0.1-0.2 bpp range is reasonable for the target application (transmitting to cloud MLLMs). Testing at even lower rates is beyond the stated scope.

## Novel Insights

The key insight from the paper's empirical analysis (Figure 7) is that cross-entropy loss reduces feature matching errors primarily in foreground object regions, while distillation loss reduces global matching errors. The progressive training strategy combines these complementary behaviors, which is a meaningful finding about how different loss components shape feature alignment in neural compression. Additionally, the observation that operating on compressed latents directly (rather than reconstructing images) preserves sufficient semantic information for MLLM tasks while dramatically reducing computation is an important finding for edge-cloud MLLM deployment architectures.

## Suggestions

- Explicitly clarify train/test splits for ImageNet to address the potential data overlap concern, or acknowledge this limitation if overlap exists.
- Correct the captioning metric label in Figure 3 and clarify which captioning metrics (CIDEr, SPICE, etc.) were used.
- Add a brief discussion of spatial dimension handling between codec latents and visual encoder features in Section 3.3.
- Include a small ablation on transform-neck architecture (depth, heads) to justify design choices.
- Clarify whether decoder weights are frozen in scenario (d3).

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
