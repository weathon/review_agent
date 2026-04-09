## Summary

The paper introduces CANON (Conditional Advantage Estimation), a method that regroups sampled responses into two equal-sized groups based on a target metric (e.g., entropy or length) and computes advantage through both inter-group comparison (identifying which metric trend correlates with higher reward) and intra-group comparison (selecting superior responses within the same trend). The mixing parameter μ balances these components, with DR.GRPO recovered as a special case (μ=0.5). Experiments across three LLMs demonstrate that CANON-Inter improves math accuracy by 1.9 points over DR.GRPO, CANON-Intra improves high-complexity logic by 5.2 points, and a length-weighted variant (CANON-Eff) achieves a superior Pareto frontier in the performance–efficiency trade-off.

## Strengths

- **Principled generalization of existing methods.** The decomposition of advantage estimation into inter-group and intra-group components is a clean conceptual contribution. The formal derivation showing DR.GRPO as a special case (Eq. 7, when μ=0.5 and groups are equal-sized) provides a clear theoretical grounding and positions CANON as a proper generalization rather than an ad hoc modification.

- **Task-adaptive behavior through a single mechanism.** The finding that inter-group advantage exploits known metric–reward correlations (benefiting math) while intra-group advantage encourages exploration from the disadvantaged group (benefiting complex logic) is insightful. The analysis in Figure 2f and Figure 6, showing that intra-group advantage produces positive reflection gains that correlate with logic performance improvements, provides mechanistic understanding beyond simple accuracy numbers.

- **Strong efficiency results with a favorable Pareto frontier.** CANON-Eff dominates all baselines across the entire efficiency frontier (Figure 4c). The practical significance of 45.5% token reduction at the same performance level and 2.63× performance at low token budgets is substantial. Notably, CANON-Eff remains stable where Length Reward(+) collapses (performance drops from 54.8 to 22.5 when coefficient changes from 0.004 to 0.005), demonstrating meaningful robustness gains.

- **Selective amplification is empirically validated.** Table 4 directly compares CANON against naive advantage scaling (A=A*2), showing that simple amplification hurts logic performance (25.1 vs. 26.2 baseline) while CANON-Inter achieves 57.6 on math. Table 12 shows random regrouping fails to improve, confirming that meaningful metric-based grouping is essential. Together, these ablations substantiate the claim that CANON's benefit comes from selective metric-specific amplification rather than generic signal boosting.

## Weaknesses

### Major:

- **Scheduling strategy selection introduces unprincipled model-specific tuning.** The paper tests four scheduling strategies and selects different ones per model: Cosin-First-Inter-Later-Intra for Qwen2.5-7B and Llama3.1-8B, and First-Inter-Later-Intra for Qwen2.5-1.5B (Section 5.2, lines 555–556). No guidance is provided for selecting a schedule for a new model. While Table 10 shows a monotonic relationship between μ and task performance (higher μ → better math, worse logic), the *functional form* of the schedule (cosine vs. linear, accuracy-based vs. step-based) requires its own tuning. This undermines the claim that CANON is a drop-in improvement over DR.GRPO, since practitioners must search over scheduling strategies rather than simply setting μ.

- **The inter/intra tension reveals task-dependent benefits rather than universal improvement.** CANON-Inter outperforms on math but underperforms on logic; CANON-Intra shows the opposite pattern (Table 1). CANON-Dynamic resolves this only by carefully tuned scheduling. This suggests the core method's advantage is fundamentally task-contingent: the metric signal that helps one domain may hurt another, and without knowing the target distribution in advance, there is no principled way to set μ. The paper does not analyze what task properties predict which advantage type will dominate, limiting practical applicability.

### Minor:

- **The "preference-free" framing in the abstract is slightly overstated.** The abstract states CANON works "without presuming its direction," but Section 4.3 introduces α to explicitly bias the model toward shorter responses. While α is presented as an optional efficiency control, the abstract claims it as part of CANON's contributions ("When applied to response length, CANON further improves token efficiency"). The base method (CANON-Inter/Intra) is genuinely direction-free in discovering metric trends, but CANON-Eff explicitly encodes a directional prior. A clearer distinction between the discovery mechanism and the directed efficiency extension would strengthen the framing.

- **Efficiency evaluation (Table 3) is limited to math benchmarks.** The length-weighted CANON-Eff results are only reported for six math benchmarks, with no evaluation on the high-complexity logic tasks where CANON-Intra excelled. Given that CANON-Intra already produces 36.6% shorter responses on logic tasks (Table 1), the interaction between efficiency weighting and logic performance remains unexplored. This is a gap in assessing the generality of the efficiency gains.

- **Theorem 2's independence assumption warrants empirical validation.** The theorem assumes conditions c₁ and c₂ are independent (P(o∈C₁∩C₂)=P(o∈C₁)P(o∈C₂)). In practice, entropy and response length are often correlated in LLM rollouts. While the theorem correctly characterizes the theoretical selective amplification property, the paper provides no empirical check of metric correlations during training, leaving open whether CANON-Entropy inadvertently influences length distributions beyond what the entropy grouping alone would predict.

### Trivial:

- The notation switches between C_q^+ and G_q^+ across Sections 4.1 and 4.2 without explicit clarification; consistent use of one symbol would aid readability.

## Nice-to-Haves

- Statistical significance tests or confidence intervals across multiple random seeds for the key claims (e.g., the 1.9-point math improvement), though single-run evaluation is standard in this setting.
- An empirical measurement of the actual advantage magnitude ratio (|Â_inter|/|Â_DR.GRPO|) during training to directly validate Theorem 1's amplification prediction.
- Ablation of all four scheduling strategies across all three models in the main text, rather than reporting only the selected best strategy per model.
- Evaluation on a model at a larger scale (e.g., 32B+) to validate scalability claims implied by the title "Large Reasoning Models."

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Computational overhead of sorting/regrouping.** With G=16, sorting is O(16 log 16), which is negligible compared to the forward/backward pass costs of the LLM itself. The harsh critic themselves acknowledged this is "small for G=16." This is a nitpick about trivial implementation details.

- **Weakness: Table 1 readability issues.** This is a PDF parser artifact, not a paper problem. Per hard rules, formatting nitpicks are removed.

- **Weakness: Missing comparison with Chen et al. 2025b (Seed-GRPO).** Per hard rules, I cannot confirm the existence or relevance of this as a missing baseline from external sources.

- **Weakness: No evaluation on non-verifiable reward tasks.** The paper is explicitly about RLVR (Reinforcement Learning with Verifiable Rewards). Criticizing absence of evaluation on learned reward models is scope creep.

- **Weakness: Dataset heterogeneity for Llama3.1-8B.** The paper clearly explains the rationale: Llama's weak math capability requires a simpler dataset (Appendix C.5). The within-model comparison between methods is fair since all methods trained on Llama use the same dataset.

- **Weakness: Need for larger models (32B+).** Generic "test on more models" criticism. The paper already tests 3 models from 2 families at 2 scales.

- **Weakness: Incomplete comparison with recent advantage shaping baselines.** The paper compares against Entropy Adv (Cheng et al., 2025) and Clip-Cov (Cui et al., 2025) as entropy baselines, and Length Reward(+/*) as efficiency baselines. The claim of missing comparisons is not substantiated.

- **Strength: "The paper is well-written / comprehensive evaluation."** Generic strengths that would apply to many papers. Weakened per soft rules.

## Novel Insights

The inter-group/intra-group decomposition reveals a fundamental tension in RLVR training: exploitation of known metric–reward correlations (via inter-group comparison) benefits in-domain performance but suppresses exploration needed for out-of-distribution generalization, while encouraging exploration from the disadvantaged group (via intra-group comparison) enables breakthroughs on complex tasks at the cost of in-domain efficiency. This trade-off, visible in the training dynamics (Figure 2: CANON-Inter stably decreases entropy while CANON-Intra's logic performance surges only after reflection gains cross zero at ~90 steps), suggests that the optimal RLVR training trajectory is inherently non-monotonic—early exploitation followed by late exploration—and that GRPO-style flat baselines are fundamentally limited in capturing this phased structure.

## Suggestions

- Provide decision criteria for scheduling strategy selection (e.g., based on model capability or training accuracy range) to reduce the practical tuning burden. The observation that accuracy-based scheduling works well for Qwen2.5-1.5B (accuracy range 0–0.6) but not for higher-accuracy models is a starting point.
- Include logic benchmark results for CANON-Eff (Table 3) to complete the efficiency evaluation across both task domains.
- Add a brief empirical analysis of entropy–length correlation in rollout distributions during training to validate or qualify Theorem 2's independence assumption.
- When presenting CANON-Dynamic results, report all four scheduling strategies for at least one model (in the main text or appendix) rather than only the selected best, to demonstrate the sensitivity to schedule choice.

---

**Axis Evaluations:**

- **Novelty:** Moderate-to-high. The conditional regrouping mechanism and the inter/intra decomposition are clean and non-obvious; the DR.GRPO-as-special-case result is meaningful. The method is a genuine conceptual advance over simple reward/advantage shaping.

- **Technical soundness:** Good. Theorems are correct and well-motivated. The independence assumption in Theorem 2 is strong but acknowledged. Empirical ablations (Table 4, Table 12) validate core claims.

- **Empirical support:** Strong on math reasoning and efficiency; good on logic reasoning. The efficiency Pareto frontier result is particularly compelling. Gaps exist in efficiency evaluation on logic tasks and in transparency around scheduling strategy selection.

- **Significance:** Significant for the RLVR community. The practical efficiency gains (45.5% token reduction) are meaningful for deployment, and the theoretical positioning provides a foundation for future work on metric-aware advantage estimation.

- **Clarity:** Generally good. The method is well-motivated and the paper flows logically. Minor issues with notation consistency and the framing transition between the preference-free base method and the directed efficiency variant.