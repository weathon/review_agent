## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), a framework for efficient multi-condition control in Diffusion Transformers that decomposes full attention into Position-Aligned Attention (PAA) for spatial conditions (one-to-one token correspondence) and Keyword-Scoped Attention (KSA) for subject-driven conditions (keyword-activated mask-based pruning), plus an early-timestep sampling strategy. PKA achieves up to 10× inference speedup and 5.12× VRAM reduction for the attention module while maintaining competitive generation quality.

## Strengths

- **Empirically grounded design from attention sparsity analysis**: Figures 2 and 3 provide concrete evidence that spatial condition attention concentrates along the diagonal and subject condition attention is keyword-correlated, directly motivating the PAA and KSA decomposition rather than proposing arbitrary architectural changes.

- **Significant and measured efficiency gains**: Table 1 shows best FID across all three tasks (52.99, 62.08, 53.01 vs. next-best 61.03, 70.22, 67.40), Figure 7 shows 3.90×–10× speedup vs. UniCombine, and Figure 8 shows 2.46×–5.12× VRAM reduction. These are real, measured latency/VRAM improvements, not just FLOP estimates.

- **Clean condition-type taxonomy and decomposition**: The spatial-aligned vs. subject-driven categorization is intuitive and maps cleanly to PAA (O(N) attention via one-to-one alignment, Eq. 2) and KSA (masked sparse attention via keyword relevance, Eq. 3–4). Figure 4 effectively communicates the full architecture.

- **Condition Cache mechanism**: By restricting condition tokens to self-attention within their own condition type (Figure 4b), K/V projections are computed once at the first denoising step and cached across all subsequent steps (Figure 4a). This is a practical optimization directly enabled by the PKA design.

- **Perturbation analysis providing empirical insight**: Figure 5's SSIM perturbation analysis demonstrates that visual conditions exert dominant influence during early (high-noise) denoising stages, providing genuine empirical motivation for the early-timestep sampling strategy.

- **Tunable KSA threshold**: Figure 10 shows a graceful efficiency-fidelity trade-off as ε varies from 0 (16.59s, 368MB) to 0.4 (15.26s, 242MB), with only subtle detail changes, demonstrating KSA is not brittle.

## Weaknesses

### Fatal
None.

### Major

- **Early-timestep sampling lacks quantitative validation despite being a claimed contribution**: The paper lists early-timestep sampling as one of three main contributions, yet only provides visual comparison in Figure 11. No quantitative metrics (FID, controllability, consistency) are reported for different μ and δ values, and the specific hyperparameter values used in the main experiments are not stated anywhere in the paper (only the constraints μ > 0, δ > 1 are given). Without quantitative ablation, this contribution remains unsupported by evidence. (Section 3.3, Figure 11)

- **Canny controllability degradation on Subject-Canny is significant and mischaracterized**: In Table 1, the F1 score for Canny controllability on Subject-Canny drops from 0.551 (UniCombine) to 0.414 (PKA)—a 25% relative decrease. The paper describes this as "the minor exception of a narrow margin" (Section 4.2.3), but this is not narrow; it represents a meaningful loss in one of the two core control modalities. This trade-off is likely a direct consequence of PAA's diagonal-only attention discarding cross-position spatial interactions. The paper should honestly acknowledge this as a real efficiency-controllability trade-off rather than minimize it. Note that PKA does outperform UniCombine on Canny-Depth F1 (0.411 vs. 0.369), so the Canny result is mixed across tasks, making honest discussion even more important.

### Minor

- **PAA's diagonal-only assumption is structurally aggressive and under-validated**: PAA discards all off-diagonal attention between spatial conditions and the noisy image. While Figure 2 shows diagonal concentration, it also shows non-negligible off-diagonal activation. The ablation in Figure 9 reports latency and VRAM but no quality metrics (FID, F1) for PAA vs. full attention, leaving the spatial fidelity cost of the diagonal-only assumption unquantified. The evaluation covers only three task combinations, which is insufficient to establish generality. (Section 3.2.1, Figure 9)

- **KSA mask recomputation frequency is ambiguous**: Eq. 3 computes the mask at timestep t and Eq. 4 applies it at t+1, but the paper does not specify whether the mask is recomputed every step, every k steps, or once and reused across the entire denoising trajectory. This affects both efficiency claims and quality, since recomputing every step reduces the efficiency gain while computing once may degrade quality as the image evolves. (Section 3.2.2)

- **Efficiency gains are not decomposed by component**: The condition cache (computing K/V once, reusing across steps) is a significant efficiency optimization that is directly enabled by the PKA decomposition. While they are intertwined, the paper does not report how much of the speedup/VRAM reduction comes from the attention sparsification (PAA/KSA) versus the cross-step caching, making it hard to attribute the gains precisely. (Section 3.2, Figures 7–8)

- **Limited baseline diversity**: Only two baselines are used (OminiControl2 and UniCombine), both from the same attention-based interaction paradigm. No comparison with methods from other paradigms (e.g., ControlNet-style or IP-Adapter-style adapted for DiTs) is provided, limiting the scope of the evaluation. (Section 4.1)

### Trivial
None.

## Nice-to-Haves

- Quantitative ablation for early-timestep sampling (report FID/controllability for specific μ and δ values)
- KSA mask recomputation frequency ablation (once vs. every step vs. every k steps)
- Isolated condition cache ablation to attribute efficiency gains
- Failure case analysis showing when PAA's diagonal-only attention produces spatially incoherent results
- Reporting speedup relative to OminiControl2 specifically (the paper says it "surpasses" OminiControl2 in Figure 7 but does not state the specific speedup numbers)

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Abstract claims 'maintaining or improving generative quality' contradicted by Table 1"**: The paper does achieve the best FID across all tasks and best subject consistency. The "maintaining or improving" claim is substantially supported by most metrics; only one metric (Canny F1 on one task) shows degradation. The abstract claim is overbroad but not fundamentally contradicted.

- **"Complexity formula O(c²n²) is incorrect"**: The intro uses O(c²n²) as a simplified characterization. Eq. 1 shows the attention is over (M+N+NI)² tokens. When the number of condition tokens (c·n) dominates text and image tokens, O(c²n²) is a reasonable approximation. This is a simplification, not an error.

- **"No variance or confidence intervals reported"**: Single-run evaluation is standard practice in this community for large-scale generation benchmarks. Requesting confidence intervals for every metric is a nice-to-have, not a weakness.

- **"Training data size not specified"**: The paper says "a subset from Subject200K" which is sufficient to understand the experimental setup. This is a minor reproducibility detail, not a substantive weakness.

- **"Speedup not reported relative to OminiControl2"**: The paper does compare against OminiControl2 in Figure 7 and explicitly states "our approach also surpasses the performance of OminiControl2" (Section 4.2.1). The specific speedup numbers relative to OminiControl2 are not stated in text, but the comparison is present in the figures.

- **"Missing comparison with ControlNet/IP-Adapter style methods"**: These are UNet-based methods that use feature-level fusion, a fundamentally different paradigm. The paper explicitly scopes itself to DiT-based attention interaction methods. Requesting comparisons across paradigms is scope creep.

## Novel Insights

The paper's perturbation analysis (Figure 5) revealing that visual conditions exert dominant influence during early denoising stages is a genuinely useful empirical insight that could inform future work beyond this specific method. The condition-type taxonomy (spatial-aligned vs. subject-driven) with corresponding attention sparsity patterns (diagonal vs. keyword-correlated) is a clean conceptual framework that could generalize to other condition types and architectures.

## Suggestions

- Add a quantitative table for the early-timestep sampling ablation (even just FID/F1 for 2–3 μ values) and state the exact μ and δ used in the main experiments.
- Replace "narrow margin" with an honest acknowledgment of the Canny F1 trade-off on Subject-Canny, and discuss how this relates to PAA's diagonal-only assumption.
- Clarify the KSA mask recomputation schedule in the method section (is it every step, every k steps, or once?).
- In the ablation section, report quality metrics (not just efficiency) for the PAA comparison against full attention and SWA.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to PKA |
|-------|------|-----------|-------------------|
| SANA | /home/wg25r/review_agent/human_reviews/N8Oj1XhtYZ.md | 8.50 | Much stronger: comprehensive ablations, deployed system, broader scope. PKA is clearly below this tier. |
| PT-DiT | /home/wg25r/review_agent/human_reviews/lTrrnNdkOX.md | 6.40 | Comparable: both propose efficient sparse attention for DiTs with competitive quality. PT-DiT has broader task evaluation; PKA has stronger efficiency numbers but more ablation gaps. |
| SEA | /home/wg25r/review_agent/human_reviews/JbcwfmYrob.md | 6.67 | Comparable: both propose sparse attention with practical efficiency gains. PKA is more domain-specific but has the early-timestep sampling validation gap. |
| HASA | /home/wg25r/review_agent/human_reviews/Hjk1tWIdvL.md | 5.00 | Weaker than PKA: similar sparse attention idea but more ablation gaps and less compelling results. PKA has stronger empirical backing. |
| SparseDM | /home/wg25r/review_agent/human_reviews/3kADTLbKmm.md | 4.00 | Much weaker: overclaimed efficiency (50% MACs → 1.2× GPU), missing critical ablations. PKA has real measured speedups and better main results. |
| ELR-Diffusion | /home/wg25r/review_agent/human_reviews/edx7LTufJF.md | 2.50 | Far weaker: doubtable efficiency claims, trivial runtime comparisons, missing LoRA baseline. PKA is clearly superior. |

PKA sits comfortably above the low-scoring papers (which have doubtable or overclaimed efficiency and minimal ablations) and in the range of medium-scoring accepted papers (PT-DiT at 6.4, SEA at 6.67). PKA's efficiency gains are real and well-measured, its quality is strong on most metrics, and the design is well-motivated. However, the early-timestep sampling contribution lacks quantitative validation (a claimed contribution with no quantitative evidence), and the Canny F1 trade-off is mischaracterized. These are meaningful but not fatal issues. Compared to PT-DiT and SEA, PKA has comparable strengths and comparable (but different) ablation gaps. I place it slightly below PT-DiT and SEA due to the more severe validation gap on one of three claimed contributions.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>