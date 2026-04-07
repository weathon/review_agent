## Summary

Chunk-GRPO proposes grouping consecutive flow-matching timesteps into chunks optimized jointly rather than step-by-step, motivated by (1) inaccurate advantage attribution when uniform advantages are assigned across all timesteps, and (2) the temporal dynamics of flow matching where different timesteps contribute differently to the final image. The paper introduces a chunk-level importance ratio and uses relative L₁ distance to guide chunk boundaries.

## Strengths

- **Novel insight on temporal dynamics (Figure 3):** The paper provides compelling empirical evidence that relative L₁ distance exhibits consistent, prompt-invariant temporal patterns throughout flow-matching generation. This is a genuine domain-specific insight that informs principled chunk segmentation rather than arbitrary grouping.

- **Mathematical foundation for gradient smoothing (Appendix A, Eq. 41-44):** The key theoretical insight is that chunk-level optimization applies a unified importance weight across all timesteps within a chunk, smoothing gradient fluctuations that arise from unequal individual importance ratios in step-level GRPO. This provides a principled reason why chunking improves optimization stability.

- **Consistent improvements across multiple metrics and reward models:** Tables 1, 4, and 5 show improvements on HPSv3, ImageReward, PickScore, and GenEval across different reward model configurations. Table 5 shows Chunk-GRPO achieves gains 3× larger than Dance-GRPO on GenEval (improvement of 0.03 vs. 0.01), suggesting benefits extend beyond preference alignment.

- **Ablation demonstrates temporal-dynamics chunking matters:** Table 3 shows that temporal-dynamics-guided chunking [2,3,4,7] achieves 15.236 HPSv3 vs. 15.115 for the best uniform chunk configuration [2,2,...,2], a meaningful gap that validates the core hypothesis about dynamic-aware segmentation.

## Weaknesses

- **Inconsistent baseline reporting across tables undermines confidence:** Table 1 reports Dance-GRPO baseline as HPSv3=15.080 while Table 4 reports HPSv3=14.612 for the same baseline. These represent different runs/training configurations, but the paper does not explain this discrepancy. Without consistent baselines, readers cannot reliably compare across experiments.

- **No statistical significance or multiple-run variance:** All results are reported as single point estimates. The main improvement (15.080 → 15.236 on HPSv3, ~1% relative gain) is small enough that run-to-run variance could be meaningful. No standard deviations or confidence intervals are provided.

- **Missing results in Table 5:** The GenEval table omits Chunk-GRPO rows despite the text claiming superiority. The paper states "Chunk-GRPO achieves a performance gain of 0.03" but the actual results must be inferred from text rather than shown directly in the table.

- **Incomplete comparison to closely related work:** TempFlow-GRPO (He et al., 2025) is mentioned in the related work section as introducing "temporal-aware weighting across denoising steps"—directly relevant to the core contribution—but no experimental comparison is provided. The distinction between "temporal-aware weighting" and "temporal-dynamic-guided chunking" is not empirically validated against this baseline.

- **Weighted sampling trades preference alignment for semantic degradation:** Table 2 shows Chunk-GRPO with weighted sampling degrades WISE overall score from 0.76 to 0.73, with notable drops in Biology (0.68→0.64), Physics (0.69→0.65), and Chemistry (0.68→0.62). The paper acknowledges this but does not provide a principled solution or deeper analysis of the trade-off.

## Nice-to-Haves

- **Cross-model validation:** Testing on architectures other than FLUX (e.g., SDXL, Stable Diffusion 3) would strengthen claims about the generality of temporal dynamics.

- **Advantage variance visualization:** Empirically plotting advantage variance per timestep during actual training would strengthen the motivation beyond the illustrative toy example in Figure 2.

- **Reconcile adaptive vs. fixed chunking:** The adaptive strategy (Appendix C.5) achieves lower HPSv3 (14.810 in Table 8) than fixed temporal chunking (15.236). This is counterintuitive and deserves explanation—why does a supposedly more principled adaptive approach underperform?

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"23% improvement" claim interpretation:** The harsh reviewer claims the 23% figure is "misleading." However, checking the numbers: Flux (13.804) → Dance-GRPO (15.080) gives 1.276 improvement. Flux → Chunk-GRPO (15.236) gives 1.432 improvement. The additional gain is 0.156, which is 0.156/1.276 ≈ 12.2% relative to Dance-GRPO's gain—closer to 12% than 23%. While potentially overstated, this is a quibble about marketing language rather than a substantive flaw in the method.

- **"Figure 2 conflates intermediate visual quality with policy quality":** The harsh reviewer argues this example is misleading. However, Figure 2 is illustrative of the *concept* that uniform advantage assignment can be suboptimal—it's not claimed as rigorous proof. The real theoretical justification is in Appendix A.

- **"Single baseline/model is inadequate scope":** While broader baselines would strengthen the paper, Dance-GRPO is a strong representative of the GRPO family for T2I, and validating on FLUX (a state-of-the-art model) is reasonable scope. Additional baselines are a "nice-to-have" rather than a core weakness.

- **"Adaptive chunking results not reported":** Table 8 *does* report adaptive chunking results. The harsh reviewer appears to have missed this.

- **"User study methodology concerns":** 9 participants and 40 prompts with 72.5% win rate provides supporting evidence. While not ideal, it's a meaningful validation beyond automated metrics.

- **"Geometric mean not justified for importance ratio":** The mathematical analysis in Appendix A shows this choice leads to smoother gradients. The harsh reviewer's concern is addressed by the theoretical framework.

## Novel Insights

The paper makes a genuinely novel connection between action chunking in robotics and timestep grouping in flow-matching generation, but the deeper insight is the temporal dynamics analysis (Figure 3). The observation that L₁_rel patterns naturally segment trajectories into meaningful phases—and that these patterns remain consistent across prompts—suggests flow matching has intrinsic "developmental stages" that standard RL optimization ignores. This could inform future work on curriculum-style training, dynamic compute allocation across denoising steps, or even architecture design where different network capacities are allocated to different phases.

## Suggestions

- **Add statistical significance:** Report mean ± std over multiple random seeds (at least 3) to confirm improvements are real rather than noise.

- **Explain baseline discrepancy:** Add a note explaining why Tables 1 and 4 show different Dance-GRPO baselines (e.g., different hyperparameters for different reward models) or correct if unintended.

- **Complete Table 5:** Include the Chunk-GRPO results rows directly in the GenEval table rather than only describing them in text.

- **Add TempFlow-GRPO comparison:** Even a single metric comparison against TempFlow-GRPO would help differentiate "temporal-aware weighting" from "temporal-dynamic-guided chunking."