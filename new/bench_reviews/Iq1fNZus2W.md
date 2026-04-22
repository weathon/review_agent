Now I have all the information needed to write the final review. Let me compile everything.

## Summary

The paper proposes PKA (Patch-wise and Keyword-Aware Attention), a framework for efficient multi-condition control in Diffusion Transformers. The key insight is that full attention over concatenated condition and image tokens is largely redundant: spatial conditions exhibit diagonal-dominant attention (motivating Position-Aligned Attention, PAA), while subject conditions activate only keyword-relevant image regions (motivating Keyword-Scoped Attention, KSA). A Condition Cache mechanism avoids recomputing condition KV projections across denoising steps, and an early-timestep sampling strategy accelerates training convergence. Experiments demonstrate significant efficiency gains and quality improvements over OminiControl2 and UniCombine on multi-conditional generation tasks.

## Strengths

- **Principled empirical motivation grounded in attention analysis.** Figures 2–3 provide concrete evidence that spatial conditions produce diagonal-dominant attention and subject conditions produce keyword-localized activation. This directly justifies PAA's O(N) one-to-one design (Eq. 2) and KSA's masked scoping (Eq. 3–4), making the efficiency gains principled rather than arbitrary.

- **Substantial efficiency gains with quality improvements on most metrics.** Table 1 shows PKA improves FID from 61.03→52.99 (Subject-Canny), 70.22→62.08 (Subject-Depth), and 67.40→53.01 (Canny-Depth) over UniCombine, while simultaneously achieving 3.9× speedup at 4 conditions (Figures 7–8). The efficiency–quality co-improvement is a strong result.

- **Clean modular decomposition matching condition types to attention patterns.** Separating spatial-aligned (PAA) from subject-driven (KSA) conditions, with Condition Cache as a complementary optimization, creates an extensible framework. Figure 9 confirms PAA outperforms sliding window attention on both latency (13.63s vs 14.00s) and VRAM (237MB vs 276MB).

- **Early-timestep sampling with empirical justification.** Figure 5's perturbation analysis shows High-to-Low perturbation degrades SSIM far more than Low-to-High, providing causal evidence that visual conditions dominate early denoising. Figure 11 validates that μ=0.5, δ=1.5 converges faster and produces better results at 8K iterations.

## Weaknesses

### Fatal

None.

### Major

- **Headline efficiency claims are for condition counts never quality-evaluated.** The abstract and conclusion prominently claim "up to 10× inference speedup and 5.12× VRAM reduction... all while maintaining or improving generative quality." However, all quality evaluations (Table 1, Figure 6) use only 2–3 conditions, where speedup is approximately 3.9× (Figure 7, at 4 conditions). The paper never demonstrates that PKA maintains acceptable quality at 8 or 16 conditions — the very regimes where its efficiency gains are largest. The "all while" phrasing in both the abstract (line 19) and conclusion (line 322) explicitly couples the peak efficiency numbers with the quality claim, creating a misleading impression. A reader expecting 10× speedup at equivalent quality would find the claim unsupported. This matters because the practical value of the method hinges on whether quality is preserved at the scale where efficiency gains matter most.

- **PAA's one-to-one restriction causes a significant controllability gap on Subject-Canny, dismissed too casually.** In Table 1, Subject-Canny F1 drops from 0.551 (UniCombine) to 0.414 (PKA) — a 25% relative decrease. The paper characterizes this as "the minor exception of a narrow margin" (Section 4.2.3, line 259), but a 25% controllability gap is neither minor nor narrow. This is the one regime where the core claim that restricted attention preserves controllability is contradicted by the data. Notably, on Canny-Depth (purely spatial conditions), PKA improves F1 from 0.369→0.411, suggesting the issue arises specifically from the interaction of PAA with subject conditions rather than from PAA alone. This nuance is important but unanalyzed. The PAA ablation (Figure 9) also lacks quantitative controllability metrics (F1, MSE) comparing PAA against full attention or SWA alternatives.

### Minor

- **KSA ablation lacks quantitative quality metrics across ε values.** Figure 10 shows latency/VRAM reduction across ε thresholds but reports no CLIP-I or DINOv2 scores. The text claims "the generated image remains highly faithful to the reference" at ε=0.4 (line 308), but this is a visual-only judgment. Without quantitative consistency metrics, the claim of a "graceful trade-off" (line 308) is unsupported on the quality side of the trade-off.

- **Keyword token selection for KSA is underspecified.** The paper states 𝕂 "typically contains just 1 to 2 tokens" (line 138) but does not explain how these keyword tokens are identified — whether automatically or manually. Since the training data is curated to "contain a descriptive keyword" (line 206), this suggests the keywords come from captions, but the selection mechanism is never described. If manual, this is a practical limitation; if automatic, the method should be specified.

- **Early-timestep sampling hyperparameters lack systematic justification.** Only three (μ, δ) settings are tested in Figure 11. The chosen values (μ=0.5, δ=1.5) are not derived from the perturbation analysis in Figure 5 but appear to be hand-tuned.

### Trivial

None.

## Nice-to-Haves

- Quality evaluation at higher condition counts (8, 16) to validate the headline efficiency claims in the regimes where they matter most. Even a limited qualitative study would substantially strengthen the paper.
- Quantitative controllability metrics (F1, MSE) in the PAA ablation (Figure 9) to complement the visual comparison with full attention and SWA.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"FID scores are relatively high"**: This is a generic observation not specific to this paper; FID values depend on the dataset and task, and the paper provides proper comparisons against baselines.
- **"No standard multi-condition benchmark is used"**: There is no widely-adopted standard benchmark for multi-condition generation; the paper constructs a reasonable evaluation setup. Criticizing the absence of a non-existent benchmark is scope creep.
- **"Unclear whether baselines are retrained under identical conditions"**: The paper states "To ensure a fair comparison, we fine-tune the FLUX.1 model using LoRA" (line 208), implying all methods use the same training setup. While not 100% explicit about baselines, this is a standard assumption in the field.
- **"Condition Cache prevents conditions from interacting with each other or the noisy image"**: This is an architectural design choice, and the quality results in Table 1 show it works well. The theoretical concern is addressed by the empirical results.
- **"Perturbation analysis is expected behavior, not novel"**: While it may seem intuitive that early timesteps matter more, the paper provides quantitative evidence (Figure 5) specific to visual conditions in multi-condition DiTs, which is a genuine contribution to understanding.
- **Strength removed: "KSA exploits temporal consistency to avoid per-step mask recomputation"**: While true, this is a minor implementation detail rather than a core strength of the paper.
- **Strength removed: "PAA outperforms sliding window attention"**: While valid, this is a supporting ablation result rather than a standalone strength.

## Novel Insights

The decomposition of multi-condition attention into type-specific sparsity patterns (diagonal for spatial, keyword-localized for subject) is a genuinely useful organizational insight for the field. The Condition Cache insight — that conditions performing only self-attention can have their KV projections frozen across denoising steps — is elegant and likely generalizable beyond this specific framework. The perturbation analysis (Figure 5) quantitatively confirms that visual conditions dominate early timesteps, which, while perhaps intuitive, had not been empirically demonstrated for multi-condition DiTs.

## Suggestions

- Add even a small-scale quality evaluation at 8+ conditions (e.g., 4 spatial + 4 subject conditions) to bridge the gap between the headline efficiency claims and the quality evidence. This would significantly strengthen the paper's core narrative.
- Report F1/MSE numbers in the PAA ablation (Figure 9) and CLIP-I/DINOv2 numbers in the KSA ablation (Figure 10) to make the quality-efficiency trade-offs quantitative rather than purely visual.
- Soften the "all while maintaining or improving" language to clearly separate the efficiency claim (which holds at all condition counts) from the quality claim (which is demonstrated at 2–3 conditions).

## Evaluation

**Originality**: The attention-pattern analysis and the PAA/KSA decomposition are original and well-motivated. The early-timestep sampling is a useful but more incremental contribution. The Condition Cache is a clean but straightforward insight.

**Importance of research question**: Multi-condition control in DiTs is a timely and practically important problem. The quadratic scaling of the concatenate-and-attend paradigm is a real bottleneck.

**Claims support**: The quality claims are well-supported at 2–3 conditions but unsupported at the condition counts where the largest efficiency gains are reported. The controllability claim has a notable exception that is dismissed too lightly.

**Experimental soundness**: Experiments are reasonable with proper baselines and ablations, but the ablation studies lack quantitative quality metrics, and the evaluation is limited to 2–3 conditions.

**Clarity**: The paper is well-organized and clearly written. The decomposition into PAA and KSA is easy to follow.

**Value to community**: The principled attention analysis and the modular framework are valuable contributions that could inform future work on efficient multi-condition generation.

## Calibration Anchors

- **SANA** (avg 8.5, Oral): Comprehensive redesign of DiT pipeline with linear attention. Much more thorough evaluation and system-level contribution. PKA is clearly below SANA due to narrower scope and the overclaiming issue.
- **Differential Transformer** (avg 8.0, Oral): Novel sparse attention mechanism with strong theoretical grounding. PKA has less theoretical depth but more practical efficiency gains.
- **PT-DiT** (avg 6.4, Accept Poster): Also analyzes attention sparsity in DiTs and proposes proxy-tokenized sparse attention. Very similar profile — both use attention pattern analysis to motivate efficient designs. PT-DiT was accepted as poster with scores 6–8. PKA has comparable motivation but the overclaiming (10× at 16 conditions without quality validation) and the dismissed F1 gap are additional weaknesses not present in PT-DiT.
- **Precise Parameter Localization** (avg 6.2, Accept Poster): Analyzes attention sparsity in diffusion models for parameter localization. Similar analytical approach, accepted as poster.
- **CDIM** (avg 5.0, Reject): Claims 10–50× speedup for conditional diffusion but speedup mainly comes from DDPM→DDIM switch (not novel), with weak baselines. PKA is clearly stronger than CDIM — the architectural innovation is genuine, quality improvements are real, and baselines are proper.
- **ELR-Diffusion** (avg 2.5, Withdrawn): Claims memory/parameter reduction but quality possibly not preserved, with serious methodology issues. PKA is far above this.
- **Pixel-Aware Accelerated Reverse Diffusion** (avg 3.0, Reject): Claims 4× speedup with poor quality validation and outdated comparisons. PKA is clearly stronger.

PKA sits between the medium-efficiency papers (CDIM at 5.0, Δ-DiT at 5.5) and the accepted sparse-attention DiT papers (PT-DiT at 6.4, Parameter Localization at 6.2). The principled motivation and genuine quality improvements place it above the rejected efficiency papers, but the overclaiming of the 10× figure without quality validation at that regime, and the dismissed 25% F1 gap, pull it below PT-DiT. A score of 6.0 reflects this positioning.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>