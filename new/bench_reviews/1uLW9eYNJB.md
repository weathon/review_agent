Now I have a good understanding of the calibration papers and their scores. Let me synthesize my final review.

## Summary

This paper proposes Mixture of Shards (MoS), a parameter-efficient finetuning method that extends LoRA by combining inter-layer and intra-layer parameter sharing of vector shards from global pools, combined with four differentiation strategies (subset selection, pair dissociation, vector sharding, shard privatization) to counteract the performance degradation caused by excessive sharing. Experiments on LLaMA2-7B/13B and LLaMA3.2-3B demonstrate approximately 8× parameter savings compared to vanilla LoRA while maintaining comparable performance.

## Strengths

1. **Principled motivation through sharing-differentiation analysis**: The paper provides useful empirical evidence (Table 1) that pure parameter sharing can hurt performance and that differentiation strategies (especially subset selection) reverse this degradation. This principled analysis directly motivates the MoS design rather than relying purely on heuristic choices.

2. **Novel architectural design**: The combination of global shard pools, pair dissociation (separate A and B pools), vector sharding, and shard privatization into a unified framework is technically coherent and introduces genuinely new ideas beyond existing sharing methods like VeRA and PRoLoRA. Pair dissociation in particular is a creative observation that decoupling A and B vector pools increases combinatorial diversity at essentially no parameter cost.

3. **Preserves LoRA's practical advantages**: Unlike some alternatives, MoS maintains LoRA's ability to merge weights into the pretrained model for zero-cost inference and supports low-cost task switching—important for the multi-LoRA serving scenario motivating the work.

4. **Ablation study provides interpretability**: Table 2 shows clear contributions from pair dissociation (~1.1 avg point drop) and shard privatization (~1.1 avg point drop), while vector sharding contributes less (~0.4 point drop), giving practical guidance on which components matter most.

## Weaknesses

### Major:

1. **The efficiency comparison is incomplete and the "8× savings" claim is overstated** — The headline 8× claim is calibrated only against vanilla LoRA, not against the strongest parameter-sharing baseline (PRoLoRA). At 5M parameters, MoS achieves an average of 36.39 vs. PRoLoRA's 36.03—a gap of only 0.36 points. This is not established as statistically significant (no error bars or multiple seeds). The VeRA comparison is acknowledged as infeasible due to OOM at matched budgets, so VeRA sits outside the comparison curve entirely. Without characterizing the efficiency frontier against all relevant baselines (particularly PRoLoRA at varying parameter budgets), the claim that MoS "unleashes" parameter efficiency relative to peer methods is stronger than the evidence warrants. The claim should be scaled to what the experiments actually show: moderate improvement over PRoLoRA at one parameter budget on one model, and substantial savings relative to vanilla LoRA.

2. **No statistical significance testing or multiple-seed results for main comparisons** — All results in Tables 1–3 are single-run. The improvements over PRoLoRA are small (0.36–0.94 avg points depending on the comparison). Whether these differences survive seed variation is unknown. The appendix mentions additional seeds for LLaMA3.2-3B but these are not used to support the main claims or reported in the main tables. For claims about efficiency gains, especially with small absolute differences, basic uncertainty quantification is essential.

3. **Misleading "MoE-like routing" terminology** — The paper repeatedly describes the shard selection mechanism as "MoE-like routing" (Abstract, Section 3, Fig. 2). However, the indices $\mathbf{I}_a^k$ and $\mathbf{I}_b^k$ are randomly initialized and frozen during training (Section 3.2: "randomly sampled during initialization, remains fixed during the finetuning process"). This is static random sparse selection, not learned input-dependent routing. Calling this "MoE-like" overclaims the mechanism and may confuse readers expecting dynamic gating; the framing should be revised to accurately reflect the mechanism.

4. **No training cost, memory, or inference overhead analysis** — The paper's primary motivation is reducing GPU memory for multi-LoRA serving, yet no measurement of actual training time, peak GPU memory during training, or inference latency is provided. MoS introduces global pools, index matrices, and shard assembly operations that add overhead. If MoS at rank 16/32 trains as slowly as LoRA at rank 64, the "8× parameter savings" framing could be misleading from a practical cost perspective. The "nearly cost-free" characterization of differentiation strategies (Sec. 3.3–3.5) is asserted but not verified with actual wall-clock or memory measurements.

### Minor:

5. **Selective reporting on LLaMA2-13B** — Table 3 drops TyDi QA and HumanEval benchmarks from the evaluation, noting "LoRA does not yield consistent improvements." This omission of tasks where LoRA underperforms—particularly from the main model where MoS's improvements are already modest—makes it difficult to assess MoS's robustness across tasks. The paper should report all tasks or explain why individual tasks are unreliable rather than silently dropping them.

6. **The sharing-differentiation principle is empirically supported only narrowly** — The principle that "excessive sharing degrades performance and differentiation reverses it" (Sec. 2) is demonstrated only through one artificially constructed "pure sharing" baseline on one model at one parameter budget, plus internal ablations within MoS itself. While intuitively compelling, the paper presents this as a validated general design rule, which is an overreach from the evidence provided.

7. **Missing comparison with inter-layer-only and intra-layer-only sharing in MoS** — MoS combines both inter-layer and intra-layer sharing, but the paper does not systematically isolate the contribution of each. This makes it impossible to determine how much of MoS's improvement comes from the novel combination versus from either sharing scheme alone.

## Nice-to-Haves

- **Hyperparameter sensitivity analysis** for shard size, private/public ratio, and pool size would improve practical utility.
- **Actual multi-LoRA serving experiment** measuring GPU memory and throughput with frameworks like S-LoRA would directly validate the paper's primary motivation.
- **Comparison with simply applying LoRA to fewer layers at higher rank** as an alternative parameter-saving strategy.

## Removed Points

- **Claim that VeRA is fundamentally impractical due to OOM** — The paper notes VeRA OOMs at matched parameter budgets, and the harsh critic suggests this is an "implementation detail." However, OOM at higher ranks IS a legitimate practical limitation of VeRA: if a method requires substantially more memory to reach the same parameter count, that affects deployment. The paper's framing is reasonable, though they should be clearer that this is an implementation-level finding rather than a fundamental limitation.

- **Demand for AdaLoRA/DoRA baselines** — The paper explicitly justifies excluding AdaLoRA (variable rank complicates multi-LoRA serving, which is the paper's stated scope) and compares against the most relevant parameter-sharing baselines (VeRA, Tied LoRA, PRoLoRA). Adding every LoRA variant as a baseline is scope creep for a paper focused on parameter sharing.

- **Demand for experiments on other model families** — The paper evaluates on LLaMA2-7B, LLaMA2-13B, and LLaMA3.2-3B (appendix). While more model diversity would strengthen the paper, the current evaluation covers multiple scales, which is adequate for establishing the method's basic viability.

- **Formatting and notation nitpicks** — The notation "4/8" and "16/32" is explained in a footnote and may be confusing, but this is a presentation choice, not a methodological flaw.

- **Claims about missing related work** — Cannot verify without external sources.

## Novel Insights

The decoupling of A and B vector pools (pair dissociation) to increase combinatorial diversity at zero parameter cost is a genuinely insightful observation: most LoRA-sharing methods treat vector pairs as inseparable units, but MoS shows that breaking this coupling yields substantial gains. The paper also provides the first systematic, if narrow, demonstration that pure parameter sharing can *hurt* LoRA performance, and that subset selection across a shared pool is a more effective differentiation strategy than random scaling.

## Suggestions

1. **Report results with standard deviations across at least 3 random seeds** for the main comparisons in Table 2, particularly the MoS vs. PRoLoRA comparison at 5M parameters.
2. **Reframe the "8× savings" claim** to be explicit that it is relative to vanilla LoRA only, and provide an analogous analysis versus PRoLoRA.
3. **Rename the mechanism from "MoE-like routing" to "static index-based selection"** or similar, to accurately reflect that the routing is random and frozen, not learned.
4. **Report training time and peak GPU memory** for MoS vs. LoRA and PRoLoRA at matched parameter budgets, to validate the practical efficiency claims.

## Calibration

I compared MoS against papers in the same parameter-sharing LoRA space:
- **VeRA** (ICML 2024): Scores 8, 5, 8, 8. Accepted as poster. Clean, simple method with strong empirical results across multiple seed runs.
- **RaSA**: Scores 8, 8, 6, 6. Accepted as poster. Similar approach (rank-sharing across layers) with theoretical grounding and empirical validation.
- **Bi-Share LoRA**: Scores 3, 6, 6, 6. Rejected. Similar motivation (inter + intra sharing) but with thinner empirical support and missing practical analysis.
- **ShareLoRA**: Scores 3, 5, 3, 5, 5. Withdrawn/rejected. Modest improvements, limited evaluation.
- **GIFT**: Scores 5, 5, 5. Rejected. Novel perspective but modest gains.

MoS is stronger than Bi-Share LoRA (which was rejected) in that it has a cleaner analysis framework and better experimental protocol (more benchmarks, principled ablations). However, MoS has the same core weaknesses flagged in Bi-Share LoRA reviews: missing training/inference cost analysis, no multi-LoRA serving validation, and small absolute improvements. MoS is weaker than VeRA and RaSA (which were accepted) because those had more rigorous experimental validation and cleaner framing. MoS's "MoE-like" terminology is misleading, and the 8× claim is overblown relative to the evidence.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>