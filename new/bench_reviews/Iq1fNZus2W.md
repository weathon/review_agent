Now I have a thorough understanding of the paper and the calibration anchors. Let me compile the final review.

## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), a framework for efficient multi-condition control in Diffusion Transformers. PKA decomposes the costly full-attention used in the "concatenate-and-attend" paradigm into two specialized modules: Position-Aligned Attention (PAA) for spatial conditions (one-to-one patch correspondence, reducing O(N²) to O(N)) and Keyword-Scoped Attention (KSA) for subject-driven conditions (using text-keyword attention masks to restrict subject-to-image attention to semantically relevant regions). A Condition Cache further enables KV reuse across denoising steps when condition tokens only self-attend, and an early-timestep sampling strategy accelerates fine-tuning convergence.

## Strengths

- **Principled sparsity analysis motivates the design**: Figures 2–3 provide concrete empirical evidence that spatial condition attention is concentrated along the diagonal and subject condition attention is localized to keyword-relevant regions. This data-driven justification is stronger than most prior DiT efficiency work that applies general sparsity heuristics without analyzing condition-specific structure.

- **Substantial efficiency gains demonstrated**: Figures 7–8 show up to 10× inference speedup and 5.12× VRAM reduction for the attention module relative to UniCombine's full attention, with the gap widening as condition count increases (3.9× at 4 conditions, 6.46× at 8, 10× at 16). This scaling property is critical for enabling multi-condition generation at scale.

- **Quality improvements at evaluated condition counts**: Table 1 shows PKA achieves the best FID and SSIM on all three tasks (e.g., FID 52.99 vs. 61.03/72.03 on Subject-Canny, SSIM 0.553 vs. 0.493/0.406), and best subject consistency (CLIP-I 0.945 vs. 0.912/0.878, DINOv2 0.926 vs. 0.901/0.867). These are non-trivial improvements, not merely matching baselines.

- **Condition Cache is a clean and practical engineering contribution**: By structuring condition tokens to self-attend only (§3.2, Figure 4b), their KV can be computed once and reused across all denoising steps (Figure 4a). This is orthogonal to the attention decomposition itself and provides compounding efficiency gains.

- **KSA threshold ε provides a tunable efficiency-quality knob**: Figure 10 shows increasing ε from 0 to 0.4 reduces VRAM from 368MB to 242MB with only subtle detail-level changes (chair legs, motorcycle windshield), suggesting graceful degradation rather than catastrophic failure.

## Weaknesses

### Fatal

None.

### Major

- **Quality is not evaluated at the condition counts where headline efficiency gains are claimed.** All quality metrics (Table 1, Figure 6) are measured at 2–3 conditions (Subject-Canny, Subject-Depth, Canny-Depth). The headline "10× speedup" and "5.12× VRAM reduction" (abstract, conclusion, §4.2.1) come from the 16-condition regime (Figures 7–8). There is zero evidence that PKA produces acceptable outputs at 8 or 16 conditions. The framing in the abstract — "up to 10× speedup… all while maintaining or improving generative quality" — implicitly links quality preservation to the 10× figure, but no such link is established. This disconnect undermines the paper's core claim that efficiency is achieved *while maintaining quality* at the scale where efficiency matters most.

- **Spatial controllability degradation on Subject-Canny is non-trivial and underreported.** Table 1 shows F1 drops from 0.551 (UniCombine) to 0.414 (Ours) — a 25% relative decrease on the metric that directly measures whether spatial conditions actually control the output. The paper characterizes this as "a minor exception of a narrow margin" (§4.2.3), but a 25% relative loss on a primary controllability metric is not narrow; it is a meaningful capability cost of PAA's strict one-to-one spatial alignment, which eliminates long-range spatial influence by construction. This tradeoff should be honestly acknowledged and analyzed rather than dismissed.

### Minor

- **Condition self-attention restriction carries an information-pathway cost that is not discussed.** To enable the Condition Cache, condition tokens (SP, SJ) are restricted from attending to noisy image tokens X (§3.2, Figure 4b). In full-attention baselines, conditions can attend to X, allowing their representations to adapt to what the model has already generated. The paper presents the cache as a pure efficiency win without acknowledging this capability loss. In practice, the quality improvements in Table 1 suggest this pathway may not be critical, but the design tradeoff should be discussed.

- **PAA ablation reports only latency/VRAM, not controllability metrics.** Given the F1 drop noted above, the PAA ablation (Figure 9) should include F1 or similar spatial controllability measures alongside efficiency, rather than relying only on visual inspection and latency comparisons with SWA variants.

- **Keyword selection for KSA is under-specified.** The paper states 𝕂 "typically contains just 1 to 2 tokens" (§3.2.2) and that "each image caption contains a descriptive keyword" (§4.1), but does not clarify whether keyword selection is manual, automatic, or derived from the caption. Understanding failure modes when keyword selection is poor matters for reproducibility and practical use.

- **KSA mask staleness across timesteps is not analyzed.** Equation 3 computes mask M^t at timestep t and reuses it at t+1. The paper invokes "temporal consistency" but provides no analysis of how the mask evolves or when staleness becomes problematic — a simple visualization of masks at different timesteps would address this.

- **Early-timestep sampling ablation (Figure 11) provides only visual comparison without quantitative metrics**, making it difficult to assess the actual magnitude of benefit beyond visual inspection.

### Trivial

None.

## Nice-to-Haves

- Quality evaluation at 4, 8, and 16 conditions would firmly anchor the headline efficiency numbers and significantly strengthen the paper.
- Analysis of the PAA controllability gap: specifically, which spatial features (long edges, global symmetry) require cross-position attention that PAA cannot provide.
- End-to-end wall-clock speedup measurement, including VAE and text encoder time, so readers can translate module-level speedup to realistic application-level improvement.
- A hybrid PAA design allowing a small cross-position window (diagonal ±1–2 neighbors) could potentially close much of the F1 gap at minimal additional cost.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"PAA is more like gating than attention"** — The harsh critic argues Eq. 2 is closer to position-dependent gating than conventional attention. While PAA does compute a scalar weight per position (single Q, K pair), it remains within the Softmax(QK^T/√d)V family and serves the same function of routing information from condition to image tokens. Calling it attention is defensible and standard in the literature.

- **"Training details of baselines may differ"** — The paper states "To ensure a fair comparison, we fine-tune the FLUX.1 model using LoRA" and references official baselines (OminiControl2, UniCombine). This is a generic reproducibility nitpick; the baselines are published methods with their own training procedures.

- **"Perturbation experiment is a well-known phenomenon"** — The harsh critic notes that early steps dominating coarse structure is well-known. While true, the perturbation analysis (Figure 5) still provides direct quantitative evidence specific to multi-condition control and justifies the sampling strategy. The contribution is in applying this insight, not discovering it.

- **Generic request for "more models" or "larger datasets"** — The paper evaluates on three tasks with six metrics across two baselines, which is adequate for the stated scope.

## Novel Insights

The paper's most insightful observation is the *qualitative difference in sparsity structure between condition types*: spatial conditions are position-diagonal (justifying one-to-one patch correspondence) while subject conditions are keyword-localized (justifying semantic masking). This distinction — that not all conditions should be treated the same way in attention — is a useful design principle that could extend beyond DiTs to other multimodal architectures where heterogeneous condition types interact with a shared representation.

## Suggestions

- At minimum, run a small-scale quality evaluation (FID, F1, CLIP-I) at 4 and 8 conditions using synthetic multi-condition tasks. Even a table with 20–50 samples per setting would partially anchor the efficiency claims.
- Explicitly acknowledge the F1 tradeoff on Subject-Canny as a design cost of PAA's strict locality, and discuss whether a hybrid (PAA + small window) could mitigate it.
- Add a sentence to §3.2 acknowledging that the condition self-attention restriction trades off condition-to-image information flow for cache efficiency, but note that empirical results suggest this flow is not critical for the evaluated tasks.

## Score and Decision

**Calibration anchors:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| MoGA (sparse attention for DiT) | /home/wg25r/review_agent/human_reviews_2026/0hy9kJ1ULB.md | 7.0 | PKA is weaker: MoGA has learnable routing + comprehensive evaluation across video lengths, while PKA has no quality evaluation at high condition counts |
| HyCa (hybrid caching for DiT) | /home/wg25r/review_agent/human_reviews_2026/URbsHlTK8c.md | 7.0 | PKA is weaker: HyCa is training-free, evaluated across multiple models (FLUX, HunyuanVideo, Qwen), and has near-lossless quality claims verified |
| SLA (sparse+linear attention for DiT) | /home/wg25r/review_agent/human_reviews_2026/eD8IPvNoZB.md | 5.0 | PKA is comparable: both have principled sparsity decomposition, real speedups, and similar scope limitations |
| LowDiff (efficient diffusion sampling) | /home/wg25r/review_agent/human_reviews_2026/KtocialdxS.md | 2.67 | PKA is clearly stronger: PKA has genuine empirical analysis motivating design and better baselines |
| ScaleWeaver (efficient controllable T2I) | /home/wg25r/review_agent/human_reviews_2026/xrqTCdA7PS.md | 3.0 | PKA is stronger: PKA has a more principled design and better evaluation, though both share the efficiency/quality evaluation gap concern |
| EVCtrl (efficient control adapter) | /home/wg25r/review_agent/human_reviews_2026/0CQnhxpE7w.md | 5.5 | PKA is comparable: both show real speedups for controllable generation with some evaluation gaps |
| N5RV691l3H (reward-guided distillation) | /home/wg25r/review_agent/human_reviews_2026/N5RV691l3H.md | 2.67 | PKA is clearly stronger: PKA has real technical novelty in sparsity analysis, not just engineering combinations |

PKA sits above the low anchors (2-3 range papers overclaim efficiency without principled analysis) and the medium-low anchors (ScaleWeaver at 3.0 conflates backbone gains with method gains). It is comparable to SLA (5.0) and EVCtrl (5.5) — all papers with principled efficiency ideas but evaluation gaps. It falls below the high anchors (7.0) because those papers verify quality at the scales where efficiency is claimed, and/or are more broadly applicable. Given the meaningful but addressable major weaknesses (quality evaluation gap at high condition counts, underreported controllability tradeoff), and the genuine strengths (principled sparsity analysis, real speedups, quality improvements at evaluated counts), the paper is borderline.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>