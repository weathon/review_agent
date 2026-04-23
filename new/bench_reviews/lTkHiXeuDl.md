Now I have all the information needed. Let me compose the final review.

## Summary

HMoRA proposes a Hierarchical Mixture of LoRA Experts method for multi-task LLM fine-tuning that combines token-level and task-level routing in a depth-dependent manner (more token-level in shallow layers, more task-level in deep layers), introduces a Constrained Generalized Jensen-Shannon (CGJS) auxiliary loss that jointly promotes routing certainty and load balance, and offers optional lightweight designs. The method outperforms full fine-tuning on 5/7 benchmarks with only 3.9% trainable parameters, and the CGJS loss enables the task router to differentiate 73.68% of unseen MMLU sub-tasks without task labels.

## Strengths

- **CGJS auxiliary loss is a well-motivated and empirically supported contribution.** The insight that standard load balancing loss trades certainty for balance, and that a constrained JS divergence can jointly optimize both, is clearly articulated. Table 1 shows consistent improvements with CGJS over both no auxiliary loss (+0.82 avg for soft routing, +0.85 avg for top-k) and standard load balancing loss. Figure 3 visually confirms that CGJS achieves simultaneously lower certainty entropy and higher balance entropy across layers.

- **Quantitative task differentiation study is compelling.** Section 4.3 reports that the task router with CGJS differentiates 42/57 (73.68%) unseen MMLU sub-tasks, compared to 7/57 (12.28%) with load balancing loss and 0/57 with no auxiliary function. This is a meaningful and non-obvious result that supports the claim of unsupervised task differentiation.

- **Consistent outperformance of baselines.** HMoRA w/LW (3.90% params) outperforms MoLoRA (3.82% params) by 0.86 avg points at nearly identical parameter budgets (Table 2), and outperforms full fine-tuning on 5/7 benchmarks. HMoRA w/o LW outperforms full fine-tuning on all 7 benchmarks.

- **Systematic routing analysis.** The entropy-based analysis of routing certainty and balance (Figure 3) provides genuine mechanistic insight into how different auxiliary losses affect expert specialization, going beyond typical MoE papers that only report end-task accuracy.

- **Practical lightweight designs with quantified trade-offs.** The lightweight options reduce trainable parameters from 6.31% to 3.90% and training time from ~1618s to ~1018s per 1k steps (Figure 2c), with only marginal accuracy loss (64.16% → 63.88% avg).

## Weaknesses

### Fatal
None.

### Major

- **Key ablation for the hierarchical routing design is deferred to the appendix.** The paper's title and primary contribution is the *hierarchical* combination of routing granularities, yet the main text ablation (Section 4.3, Table 3) only removes L_aux from the task router. The hierarchical routing ablation—comparing the depth-dependent α schedule against fixed α or pure single-granularity routing—is referenced only in passing: "Experiments in Appendix E.5 demonstrate that increasing α^(l) with l improves model performance" (Section 3.2) and "we find that setting ε > 0 generally leads to better performance" (Section 4.3). For a paper whose central claim is the hierarchical design, this ablation should be in the main text with sufficient detail for readers to assess whether the hierarchical structure itself matters versus simply having both routing types available. The ablation reportedly exists in the appendix, but its absence from the main presentation is a significant gap.

- **No standard deviations or confidence intervals reported despite 5 runs and small performance margins.** The paper states "each experiment is repeated 5 times, and we report the mean" (Section 4), yet no variance measures appear anywhere. The headline improvement of HMoRA w/LW over Full FT is 0.73 avg points (63.88 vs. 63.15), and over MoLoRA is 0.86 points. Without variance, readers cannot assess whether these improvements are statistically meaningful or within noise. This affects all main results tables.

### Minor

- **The hierarchical routing α formula (Eq. 8) introduces multiple hyperparameters (ε, μ, β_low, β_high) whose sensitivity is insufficiently analyzed in the main text.** The appendix reportedly contains ablations, but the main text provides only a brief summary. Given that the routing schedule has four tunable parameters, readers need more evidence that performance is robust to their choices rather than overfit to the evaluation benchmarks.

- **t-SNE visualization for task differentiation is selective.** Figure 4 shows only 3 pairs of tasks (6 tasks total) out of 57 MMLU sub-tasks. While the quantitative study in Appendix E.8 addresses this more rigorously (42/57 tasks), the main text visualization could mislead readers about the generality of the clustering effect, as t-SNE is known to exaggerate clustering structure.

- **The generalization claim for unseen tasks is only weakly tested.** The unseen tasks are MMLU sub-tasks, which are knowledge-QA style benchmarks arguably within the distribution of Flan v2 training tasks. Testing on more substantially different task types (e.g., code generation, mathematical reasoning) would more convincingly demonstrate generalization.

### Trivial
None.

## Nice-to-Haves

- A matched parameter-budget experiment scaling baselines (e.g., MoLoRA with more experts or higher rank) to HMoRA's 6.31% budget, to further isolate architectural contribution from capacity.
- Scaling results to larger base models (7B+), where the task encoder overhead becomes proportionally smaller and the hierarchical routing hypothesis may differ.
- Per-expert semantic analysis showing what each expert specializes in, beyond t-SNE of gate values.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unfair parameter-count comparison" (from Harsh Critic):** The critic argues HMoRA w/o LW (6.31%) has 65% more params than MoLoRA (3.82%). However, the paper also provides HMoRA w/LW at 3.90%—nearly identical to MoLoRA's 3.82%—and HMoRA w/LW still outperforms MoLoRA by 0.86 points. A matched comparison exists; the critic ignored the w/LW variant.

- **"Missing related works" suggestions:** Per rules, these are removed as I cannot verify the existence of suggested references.

- **"Reproducibility of baseline hyperparameter search" (from Harsh Critic):** The paper states "we performed a hyperparameter search for these baselines and report the best results." Demanding detailed search spaces and computational budgets for baseline tuning is a nitpick about reproducibility impractical to include in a submission.

- **"Computational overhead of the TaskEncoder" (from Harsh Critic):** While this would be informative, quantifying the overhead of a lightweight Transformer encoder relative to the base LLM is a nice-to-have, not a core flaw. The paper does report training times in Figure 2c.

- **Formatting/presentation nitpicks:** Removed per rules.

## Novel Insights

The CGJS loss elegantly reframes the routing auxiliary loss problem as one of jointly constraining two entropy terms—balance (entropy of the average distribution) and certainty (average entropy of individual distributions)—via a clipped JS divergence. This is more principled than the standard load balancing loss, which only addresses balance at the cost of certainty. The insight that this loss produces an emergent clustering effect on task router outputs, enabling unsupervised task differentiation (73.68% vs. 12.28%), is the paper's most surprising and potentially impactful finding, and it suggests that the CGJS formulation could be useful beyond the MoE-LoRA setting for any routing problem where both specialization and coverage matter.

## Suggestions

- Move the hierarchical routing ablation (Appendix E.5) into the main text, specifically comparing: (a) hierarchical α schedule vs. fixed α (e.g., α=0.5), (b) pure token-level routing + CGJS, (c) pure task-level routing + CGJS. This would directly substantiate the paper's title claim.
- Report standard deviations (or at least min/max) across the 5 runs for all main tables, given the small margins.
- When claiming generalization to unseen tasks, test on task types more distinct from Flan v2 (e.g., code, math) rather than MMLU sub-tasks which are knowledge-QA tasks similar to Flan v2's distribution.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison to HMoRA |
|-------|-----------|-------------------|
| H-QLoRA | 2.0 | Much weaker: numerically identical to baseline, overclaimed efficiency while being slower. HMoRA has genuine novelty and consistent improvements. |
| Mixture-of-Adapters | 3.0 | Weaker: overclaimed SOTA without key baselines, weak ablation. HMoRA has better experimental design and a more principled contribution. |
| MoRE | 4.0 | Comparable scope (MoE+LoRA multi-task) but less novelty. HMoRA's CGJS loss and task differentiation analysis go beyond MoRE. HMoRA is somewhat stronger. |
| ELREA | 5.8 | Similar quality: LoRA MoE with gradient-clustering routing. Both have reasonable results and weaknesses. HMoRA has stronger analytical contributions (entropy analysis, task differentiation). |
| MeteoRA | 6.2 | Comparable: MoE+LoRA multi-task. MeteoRA has the forward acceleration contribution; HMoRA has CGJS loss and hierarchical routing. Roughly similar. |
| ReMoE | 6.6 | Stronger: cleaner single contribution (differentiable ReLU routing) with good results and strong ablations. HMoRA has more components but weaker evidence for its central claim. |
| Lingual-SMoE | 7.5 | Stronger: hierarchical routing for MoE with clear ablations in main text and large improvements (>5% BLEU). HMoRA has smaller margins and thinner main-text ablations. |

HMoRA sits between the 4.0 and 6.2 anchors. It has genuine contributions (CGJS loss is well-motivated and empirically supported; task differentiation is compelling) but the evidence for its namesake hierarchical routing contribution is thin in the main text, and the small margins lack variance reporting. It's stronger than MoRE (4.0) due to more novel contributions and better analysis, but weaker than MeteoRA/ELREA (~6.0) due to the thin main-text ablation of its central claim. A score of 5.5 reflects this: a borderline paper with real contributions that need stronger empirical support for its primary claim.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>