Now I have sufficient calibration data. Let me write the final review.

## Summary
This paper introduces AdaCache, a training-free adaptive caching mechanism for Video Diffusion Transformers that dynamically reuses residual computations based on content complexity, complemented by Motion Regularization (MoReg) to allocate compute based on motion content. The method achieves up to 4.7× speedup on Open-Sora with minimal quality degradation on the slow variant, evaluated across three Video DiT backbones.

## Strengths
- **Strong empirical speedups across multiple architectures**: Table 1 demonstrates AdaCache achieves 2.24×–3.70× speedups on Open-Sora, Open-Sora-Plan, and Latte while AdaCache-slow maintains or slightly improves VBenCh scores (e.g., Open-Sora: 79.66 vs 79.22 baseline; Open-Sora-Plan: 80.50 vs 80.39 baseline).
- **Content-dependent caching mechanism is well-motivated**: Section 3 provides empirical evidence (Figure 2) that different videos have varying denoising complexity, justifying the adaptive caching approach over fixed schedules used in prior work (PAB, Δ-DiT).
- **Motion Regularization addresses a video-specific concern**: Section 4.3 introduces MoReg to modulate caching based on motion content, with Table 2a showing the motion-gradient component improves VBenCh (83.50 vs 83.36 without gradient), and Figure 6 qualitatively demonstrating reduced temporal artifacts.

## Weaknesses

### Fatal
None

### Major
- **Overclaimed "without sacrificing quality" assertion**: The Abstract and Conclusion claim AdaCache grants speedups "without sacrificing the generation quality," but Table 1 shows AdaCache-fast (the variant delivering headline 2.24×–3.70× speedups) has lower VBenCh scores than baseline on two of three models: Open-Sora-Plan drops from 80.39 to 75.83 (−4.56 points), and Latte drops from 77.40 to 76.26 (−1.14 points). The claim holds only for AdaCache-slow (1.2×–2.2× speedup). This misalignment between the central claim and evidence undermines the paper's credibility, though the data itself is sound and the slow variant does deliver quality preservation.

### Minor
- **Motion Regularization benefits are inconsistent across models**: The Abstract states MoReg "significantly improves the generation quality consistently," but Table 1 shows highly variable gains: +3.47 VBenCh points for Open-Sora-Plan (75.83→79.30), but only +0.09 for Open-Sora (79.39→79.48) and +0.21 for Latte (76.26→76.47). Characterizing this as "consistently significant" overstates the robustness of MoReg, suggesting it may be architecture-dependent rather than solving a general video motion problem.
- **Maximum speedup claim comes from different experimental setup than main comparison**: The prominently highlighted 4.7× speedup (Figure 1, Table 2a) is measured at 100-step Open-Sora 720p generation, while Table 1's main comparison against competing methods uses each model's standard benchmark settings (30 steps for Open-Sora, 150 for Open-Sora-Plan, 50 for Latte). While Section 5.1 transparently states this distinction, there is no direct comparison of AdaCache against baselines at the 100-step setting, making it unclear whether the 4.7× advantage is unique to AdaCache or if baselines would show different speedups at that configuration.

### Trivial
None

## Nice-to-Haves
- **Threshold sensitivity analysis**: The codebook thresholds for caching rates appear to work across models, but analyzing whether these require per-model tuning would strengthen the "plug-and-play" claim.
- **Overhead breakdown**: Reporting the specific latency cost of the distance metric computation and memory access would help verify the "negligible overhead" claim, especially for high-resolution latents where memory bandwidth could be a bottleneck.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Fragmented Evaluation Protocols (Harsh Critic)**: The critic claims the inconsistent step counts prevent fair assessment. However, the paper explicitly states in Section 5.1 that it uses each model's standard benchmark settings, which is appropriate practice. The 100-step ablation is for analyzing AdaCache's behavior, not for unfair comparison. This is scope creep—demanding non-standard evaluation protocols.

- **Claim that Equation 4 overhead is unaccounted for (Harsh Critic)**: The latency measurements in Table 1 and Table 2 include all computation since they measure end-to-end generation time. The critic's concern about memory bandwidth bottlenecks is speculative and not evidenced in the results.

- **Cache metric justification criticism (Harsh Critic)**: The critic claims the cosine distance justification is theoretical while empirical difference is minimal. However, Table 2c shows L1 (83.40) vs Cosine (83.19) with L1 being better, supporting the paper's reasoning. This is a minor point the paper already addresses.

- **Generic strengths from Strength Finder**: Removed strengths like "High-Value Problem" and "Intuitive Mechanism" without specific evidence. Kept only strengths with concrete table/figure citations.

## Novel Insights
The paper's observation that "not all videos are created equal" in terms of denoising complexity is well-supported by Figure 2's L1-distance histograms showing unique variation patterns across video sequences. The Motion Regularization component's use of motion-gradient as an early-predictor of latter-step motion (Section 4.3, Eq 8) is a thoughtful design that addresses the unreliability of motion estimates in early diffusion steps—a detail that distinguishes it from simpler motion-aware approaches.

## Suggestions
- Revise the Abstract and Conclusion to accurately reflect the speedup-quality trade-off: high speedups (AdaCache-fast) incur some quality cost, while quality preservation (AdaCache-slow) yields more modest speedups. This calibration would strengthen rather than weaken the paper's credibility.
- Consider adding a brief discussion of when MoReg provides substantial benefits versus marginal gains, potentially correlating with motion characteristics of the generated content.

## Calibration and Scoring

I retrieved calibration anchors across quality bands:

**High-scoring anchors (avg ≥ 6):**
- HyCa (7.0): Achieves 5.56× speedup on HunyuanVideo with "near-lossless" quality, strong ODE-based novelty
- FastVMT (6.0): 3.43× speedup for video motion transfer with comprehensive baselines and ablations
- MoAlign (6.0): Motion-centric alignment with user study and multiple benchmarks

**Medium-scoring anchors (avg ~5-5.5):**
- ScalingCache (5.0): 2.5-3.1× speedup with 0.5% VBench drop, missing some related works noted as weakness
- BWCache (5.5): 2.6× speedup on video DiTs with block-wise caching, memory overhead concern
- PreciseCache (5.5): 2.6× speedup with 0.1-0.6% VBench drops across backbones
- DiCache (5.5): Sample-specific caching based on feature variation patterns

**Low-scoring anchors (avg ≤ 4):**
- SRDiffusion (4.0): Rejected for limited novelty, questionable VBench claims, missing related work
- SemCache (4.5): Rejected for adaptive semantic caching with insufficient validation

**Positioning:** AdaCache is most comparable to ScalingCache (5.0), BWCache (5.5), and PreciseCache (5.5) in terms of contribution type (training-free caching for video DiTs) and speedup range. AdaCache's maximum 4.7× speedup exceeds these anchors, but the overclaiming issue ("without sacrificing quality" when fast variant does sacrifice quality on 2/3 models) is a credibility concern similar to issues noted in lower-scoring papers. The method itself is sound and AdaCache-slow does deliver on quality preservation, preventing this from being a fatal flaw.

Relative to anchors: AdaCache has higher speedups than ScalingCache (5.0) and BWCache (5.5) but similar overclaiming concerns. It lacks the comprehensive validation of FastVMT (6.0) or the strong novelty framing of HyCa (7.0). The paper sits between the 5.0-5.5 range anchors, slightly above ScalingCache due to better empirical results but below BWCache due to the claim-evidence mismatch.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>