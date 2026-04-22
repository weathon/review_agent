## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), an efficient attention framework for multi-condition control in Diffusion Transformers. PKA decomposes full attention into two specialized modules: Position-Aligned Attention (PAA), which restricts spatial condition attention to one-to-one patch-aligned pairs (reducing complexity from O(N²) to O(N)), and Keyword-Scoped Attention (KSA), which confines subject-driven attention to keyword-activated image regions. A Condition Cache reuses K,V projections of condition tokens across denoising steps, and an early-timestep sampling strategy concentrates training on high-noise phases. The method achieves up to 10× inference speedup and 5.12× VRAM reduction for the attention module compared to UniCombine's full attention.

## Strengths

- **Principled, empirically grounded design**: PAA and KSA are motivated by direct attention pattern analysis—Figure 2 shows spatial condition attention concentrates along the diagonal (supporting one-to-one PAA), and Figure 3 shows subject-driven attention activates only keyword-relevant regions (supporting scoped KSA). This is not heuristic; it is evidence-based decomposition.

- **Strong main experimental results**: Table 1 shows PKA achieves the best FID (52.99 vs 61.03 UniCombine on Subject-Canny; 62.08 vs 70.22 on Subject-Depth; 53.01 vs 67.40 on Canny-Depth), best SSIM, best subject consistency (CLIP-I, DINOv2) across nearly all tasks. On Canny-Depth, PKA even wins on controllability (F1: 0.411 vs 0.369, MSE: 114 vs 250). These are substantial improvements, not marginal.

- **Condition Cache as a natural architectural consequence**: By restricting condition tokens to self-attention within their groups (Section 3.2, Figure 4a-b), the K,V projections can be cached after the first denoising step. This is an elegant design that directly follows from the decomposition.

- **Early-timestep sampling with empirical support**: Figure 5 shows SSIM drops sharply under "High-to-low" perturbation but not "Low-to-high," and Figure 11 confirms early-biased sampling (μ > 0) yields superior control fidelity, validating the design.

- **Scalability advantage grows with condition count**: Figure 7 shows speedup increases from ~3.90× to 10× as conditions increase, making the approach increasingly valuable for the most demanding multi-condition scenarios.

## Weaknesses

### Fatal
None.

### Major

- **Quantitative ablations for core modules are absent**: The ablations for PAA (Figure 9) and KSA (Figure 10) report only latency and VRAM numbers alongside single-image qualitative comparisons—no FID, SSIM, F1, CLIP-I, or DINOv2 metrics are reported at each ablation point. For a paper that trades attention expressiveness for efficiency, the most essential experiment is demonstrating that the proposed restrictions (one-to-one attention in PAA; keyword-scoped masking in KSA) do not silently degrade quality in ways not captured by the end-to-end table. Without quantitative ablations, the reader cannot assess the individual contribution of each module or the quality cost of each restriction. This is the central experimental gap.

- **Headline efficiency claims are not contextualized against the most relevant efficient baseline**: The 10× speedup and 5.12× VRAM reduction are measured exclusively against UniCombine's full-attention mechanism (Section 4.2.1: "compared to the full-attention mechanism in UniCombine"). The paper states it "surpasses the performance of OminiControl2" but does not plot OminiControl2 in Figures 7–8 or quantify the efficiency advantage over it. OminiControl2 is the contemporary efficient baseline (using dynamic token pruning and input downsampling per Section 2.2), making it the most relevant comparison point. Without this data, readers cannot assess the practical impact of PKA over the state of the art in efficient multi-condition control.

### Minor

- **F1 degradation on Subject-Canny is underreported**: The controllability F1 drops from 0.551 (UniCombine) to 0.414 (Ours) on Subject-Canny—a 25% relative decrease. The paper calls this "a minor exception of a narrow margin" (Section 4.2.3), which understates the magnitude. Spatial controllability is precisely what PAA is designed to preserve, and a 25% drop in that metric deserves honest acknowledgment. That said, PKA still dramatically outperforms OminiControl2 on this metric (0.414 vs 0.192), and the Canny-Depth F1 actually improves (0.411 vs 0.369), so the overall controllability picture is mixed rather than poor.

- **Condition Cache quality impact is not ablated**: Caching K,V projections of condition tokens from the first denoising step (Section 3.2, Figure 4a) assumes condition representations do not need to evolve as the noisy image changes across steps. In full attention, condition K,V projections are recomputed at each step, allowing contextualization by the evolving noisy image. While the strong main results provide indirect evidence the cache does not substantially hurt quality, a direct comparison (with vs. without cache) would validate this key assumption.

### Trivial
None.

## Nice-to-Haves

- KSA mask drift analysis: showing how the keyword-activated mask changes across denoising steps would strengthen the temporal consistency assumption.
- Failure cases: given the F1 drop on Subject-Canny, showing and discussing failure modes would improve credibility.
- Reporting the exact values of μ and δ for the shifted logit-normal distribution would aid reproducibility.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "μ and δ not specified is critical for reproducibility"**: This falls under the rule removing nitpicks about reproducibility such as undisclosed hyperparameters. The paper specifies the constraints (μ > 0, δ > 1) and the default ε = 0.2 is given. While exact values would be helpful, this is not a critical flaw.

- **Harsh critic: "test set size not reported makes FID uninterpretable"**: This is a minor reproducibility concern. FID comparisons between methods using the same test set are still valid for relative ranking, even if the absolute FID value depends on sample size. This is standard practice in the field.

- **Harsh critic: "CLIP-T consistently worse undermines 'comparable text fidelity' claim"**: The differences are tiny (0.349 vs 0.352, 0.348 vs 0.350, 0.353 vs 0.354). "Comparable" is an accurate characterization of differences this small.

- **Harsh critic: "attention visualization does not establish off-diagonal regions are redundant"**: This is partially valid but the main results in Table 1 provide the empirical validation that the restricted attention preserves quality. The visualizations motivate the design; the experiments validate it.

- **Harsh critic: "early-timestep sampling comparison unfair if baselines don't use it"**: The paper states all baselines are fine-tuned with the same training setup. The early-timestep sampling is part of the proposed method's contribution, not an unfair advantage.

- **Strength finder: "maintaining or improved quality" as a core strength**: This strength partially conflicts with the verified F1 degradation on Subject-Canny. While overall quality is strong, the unqualified "maintaining or improving" claim is somewhat overstated. The strength is retained but weakened.

- **Strength finder: "PAA outperforms sliding window attention on efficiency"**: This is a presentation strength but the comparison lacks quality metrics, making it a weaker claim. Retained as a supporting strength only.

## Novel Insights

The paper reveals an important structural asymmetry in multi-condition DiT attention: spatial conditions exhibit strictly local (diagonal) attention patterns, while subject-driven conditions exhibit semantically sparse but non-local patterns. This observation justifies treating them with fundamentally different attention mechanisms (PAA vs. KSA) rather than applying a single sparsification strategy uniformly. The interplay between the Condition Cache and the decomposition is also notable—by isolating condition tokens to self-attention only, caching becomes a free consequence of the architecture rather than an additional engineering trick.

## Suggestions

- **Add quantitative ablations for PAA and KSA**: Report FID, SSIM, F1, CLIP-I, DINOv2 for PAA vs. full attention vs. SWA, and for KSA at ε = 0, 0.2, 0.4. This is the single most impactful improvement that can be made.

- **Plot OminiControl2 on Figures 7–8**: Even a single data point or bar would contextualize the efficiency claims against the most relevant efficient baseline.

- **Reframe the F1 result on Subject-Canny honestly**: Acknowledge the 25% relative drop and discuss the tradeoff between efficiency and spatial controllability, rather than calling it a "narrow margin."

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| MoGA (efficient sparse attention for DiTs) | `/home/wg25r/review_agent/human_reviews_2026/0hy9kJ1ULB.md` | 7.0 | Stronger than PKA: comprehensive ablations, GPU kernel integration, larger-scale experiments |
| SLA (sparse+linear attention for DiTs) | `/home/wg25r/review_agent/human_reviews_2026/eD8IPvNoZB.md` | 5.0 | Comparable: similar efficiency-quality tradeoff, SLA has better ablations but less novel decomposition |
| CreatiDesign (multi-condition attention masks) | `/home/wg25r/review_agent/human_reviews_2026/Wtda8HpVp2.md` | 5.0 | Comparable: condition-specific attention, also flagged for missing ablations |
| EVCtrl (efficient control adapter) | `/home/wg25r/review_agent/human_reviews_2026/0CQnhxpE7w.md` | 5.5 | Comparable: practical efficiency gains with some gaps |
| ECM (plug-and-play control + early sampling) | `/home/wg25r/review_agent/human_reviews_2026/DDcrkBwzec.md` | 4.5 | Weaker than PKA: limited novelty, weaker experiments |
| DraftAttention (overclaimed sparse attention) | `/home/wg25r/review_agent/human_reviews_2026/jUNmW3s45i.md` | 2.5 | Much weaker than PKA: overclaimed speedup with no end-to-end quality comparison |

PKA is clearly above the low-scoring anchors (DraftAttention at 2.5 had fundamental validation failures) and below the high-scoring anchors (MoGA at 7.0 had comprehensive ablations and engineering depth). It sits in the medium band alongside SLA (5.0) and CreatiDesign (5.0), which share similar profiles: novel condition-specific designs with meaningful results but incomplete ablation rigor. PKA's main results are arguably stronger than SLA's (more convincing quality improvements), but the missing quantitative ablations and the inflated efficiency framing pull it down. I place it slightly above the 5.0 anchors due to the strong Table 1 results and principled design motivation, but below 6.0 due to the ablation gap and overclaiming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>