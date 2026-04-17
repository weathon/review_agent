Now let me finalize my assessment. I've read the paper carefully, the various reviewer inputs, and calibration papers. Let me now write the consolidated review.

Key points to verify/decide:

1. **"No directional prior" claim**: The paper repeatedly says CANON "amplifies the impact of the target metric without presuming its direction" (Abstract, §1, §4.2, §4.2 Theorem 2 discussion). The harsh critic argues this is inaccurate because CANON *learns* the direction from reward statistics. Let me check: The paper says "without presuming its direction" vs "does not predefine a preference for the magnitude of the metric." Looking at it carefully, the mechanism does split by high/low metric values and then uses reward differences to determine which direction is beneficial. The paper's framing distinguishes between (a) knowing which direction is beneficial a priori (which CANON doesn't do) vs (b) learning which direction is beneficial from data (which CANON does). This is a genuine distinction from methods that hard-code "shorter is better" or "higher entropy is better." The claim is somewhat overstated—CANON encodes a directional prior but derives it from data rather than prescribing it—but it's not entirely wrong. I'll include this as a moderate weakness, not a fatal one.

2. **Theorem 1 rendering**: The theorem as rendered does have OCR issues, with the numerator and denominator appearing identical. However, the paper states it's proved in Appendix E. The equation (6) shows ratios of what appears to be inter-group advantage magnitude to DR.GRPO advantage magnitude. The notation "pure" likely should read "inter" vs "GRPO". This is a presentation issue. The harsh critic's concern about the theorem being "tautological" seems to stem from the OCR corruption rather than a fundamental error. However, the theorem's claim about equal-sized groups being uniquely optimal for amplification is not well-justified in the main text. This is a moderate weakness.

3. **Theorem 2**: The claim that CANON based on c1 doesn't amplify the influence of an independent condition c2 is meaningful but the theorem statement is vague. The independence assumption between entropy and length is indeed questionable since correlation exists, but the theorem is about conditions, not metrics. This is worth noting but not fatal.

4. **Experimental fairness in efficiency comparisons**: The CANON-Eff experiments do use a different max length (3072 vs 8192) for some comparisons, which is explicitly mentioned and is part of the efficiency setup. The baselines for efficiency also use max length 3072 (Clip Length section). The comparison against baselines that also use DR.GRPO as the base does mean CANON-Eff has a different advantage estimator. This concern has some merit but the paper is comparing the full methods, not just the length-weighting component.

5. **Statistical significance**: No standard deviations or multiple seeds. This is a valid concern but is common in RLVR papers with large models. Minor weakness.

6. **Model-specific scheduling**: This is valid—the scheduling requires per-model tuning, which somewhat undermines the "prior-free" narrative. But the paper acknowledges this and it's a practical engineering choice. Moderate weakness.

7. **Llama3.1-8B results are weak**: Valid observation that the method doesn't help much on weak models.

Now let me calibrate the score. Looking at comparison papers:
- SCoRe (RL for self-correction, Oral, scores 8) - much stronger contribution
- VinePPO (RL for reasoning, Reject, scores 3-6, avg ~5) - simpler method, rejected
- F0GNv13ojF (RL reward design for LLM reasoning, Reject, scores 3-6, avg ~5) - similar area, rejected for novelty concerns
- EAPO/SXUMYMETIR (Entropy advantage estimation, Withdrawn/Reject, scores 3) - weak paper with limited novelty
- 90UhF7e8jo/GAGE (Goal achievement guided exploration, Reject, scores 5-6) - similar concern about hyperparameter sensitivity

CANON is stronger than EAPO (clearer motivation, more experiments), stronger than the rejected RL reward design papers (F0GNv13oj), but has some legitimate concerns about its theoretical backing and framing. It's in the range of 5-6 given the novelty is reasonable but not groundbreaking, experiments are comprehensive but with some fairness concerns, and the theoretical claims need more careful presentation.

Let me assign a score of 5.5 - the method is sensible, empirically validated across multiple models and tasks, but the theoretical contributions have issues and the "no directional prior" framing is somewhat misleading.</think>## Summary

CANON proposes a conditional advantage estimation method for RLVR that splits per-query sampled responses into two groups based on a training metric (e.g., entropy or length), then computes inter-group and intra-group advantage signals. The inter-group advantage identifies which metric trend yields higher accuracy, while the intra-group advantage selects better responses within a group. The authors show DR.GRPO is a special case (μ=0.5), empirically demonstrate improvements on math and logic benchmarks across three models, and show CANON-Eff achieves a favorable Pareto frontier for token efficiency.

## Strengths

- **Simple and practical idea with a unifying view.** The decomposition of advantage into inter-group and intra-group components is intuitive and implementable. The proof that DR.GRPO equals μ=0.5 under equal-size grouping (Eq. 7) provides a meaningful conceptual unification: DR.GRPO uniformly weights both signals, while CANON allows flexible control.

- **Comprehensive empirical coverage.** Experiments span three models (Qwen2.5-Math 1.5B/7B, Llama3.1-8B), six math benchmarks, and three logic reasoning subsets. Both performance and efficiency dimensions are evaluated, with budget-performance and cost-performance curves providing valuable practical insight for a field increasingly concerned with inference cost.

- **Strong efficiency results.** CANON-Eff achieves meaningful token reduction with minimal performance loss. The 45.5% token reduction at the same performance level and the 2.63× improvement in low-budget scenarios are practically significant, and the Pareto frontier analysis in Figure 4 is a clear contribution.

- **Insightful analysis of inter/intra dynamics.** Section 6 provides useful mechanistic analysis (Figure 5 showing hierarchical entropy trends under different μ values; Figure 6 showing how scheduling reconciles exploration vs. exploitation). The qualitative insight that CANON-Inter favors exploitation and CANON-Intra favors exploration, and that scheduling can combine both, is a valuable finding for RLVR practitioners.

## Weaknesses

### Major:

- **The "no directional prior" claim is overstated and misleading.** The paper repeatedly frames CANON as "amplifying the impact of the target metric without presuming its direction" (Abstract, §1, §4.2). But CANON does encode a directional prior—it just *learns* that direction from empirical reward differences rather than fixing it a priori. When the lower-entropy group has higher average reward, CANON-Inter gives positive advantage to low-entropy responses, which is a lower-is-better prior on entropy. When the higher-entropy group has higher reward, the opposite prior emerges. This is a legitimate and useful property (data-driven rather than hand-coded direction), but it is fundamentally different from "no directional prior." Additionally, §4.3 explicitly introduces α < 1 to penalize longer responses—this *is* a hard-coded directional prior. The core contribution (learning direction from data) is real and meaningful, but the framing distorts it.

- **Theoretical results are poorly presented and partially unconvincing.** Theorem 1 as rendered has a tautological-looking ratio (identical numerator and denominator), likely due to notation issues, making it impossible to evaluate. The surrounding prose says equal-size groups maximize the advantage signal ratio, but the proof is relegated to Appendix E and the main text provides no intuition. Theorem 2's selective amplification claim rests on an independence assumption (P(o ∈ C₁ ∩ C₂) = P(C₁)P(C₂)) that is unrealistic when c₁ and c₂ correspond to entropy and length, which are strongly correlated in practice. The theorem claims the ratio of inter-group to DR.GRPO advantage is "a constant" under independence, but this is stated without derivation in the main text and is not obviously true given that reward statistics conditioned on c₂ do affect variance. Since these theorems are the only formal justification for the paper's theoretical claims, their current presentation undermines their credibility.

- **CANON-Dynamic requires model-specific scheduling, undermining the "prior-free" narrative.** Different models require different scheduling strategies for μ: cosine annealing for Qwen2.5-Math-7B and Llama3.1-8B, but accuracy-based scheduling for Qwen2.5-Math-1.5B (Table 2). The paper acknowledges this ("different models may have different numbers of parameters and different levels of capability. A specifically designed strategy is acceptable for better performance in practice") but this essentially concedes that CANON-Dynamic requires per-model tuning—the same kind of engineering effort the paper argues prior methods need. This doesn't invalidate CANON, but it weakens the claim that CANON avoids the brittleness of hand-crafted priors.

- **Efficiency experiments do not cleanly isolate the length-weighting contribution from the advantage estimator.** In §5.3, all baseline efficiency methods (Clip Length, Length Reward +, Length Reward ×) are built on top of DR.GRPO, while CANON-Eff uses CANON's regrouped advantage estimator plus the α-weighting. Since CANON's vanilla advantage is already shown to be stronger than DR.GRPO (§5.1-5.2), the reported efficiency gains may partly reflect the better underlying estimator rather than the length-weighting scheme. Additionally, the baseline hyperparameter sweeps are sparse near the good region (e.g., Length Reward + going from coefficient 0.004 to 0.005 drops accuracy from 54.8 to 22.5), while CANON-Eff is tuned more densely. The Pareto-frontier claim is directionally supported but not rigorously established.

### Minor:

- **No variance or confidence intervals** are reported. On small benchmarks (AIME24/25 has ~30 problems, evaluated @10), a 2-3 point accuracy difference could be within noise. This is standard for the field but limits confidence in small margins.

- **The Llama3.1-8B results** show very low absolute performance (0.0-2.0% on AIME, 7-12% on Olympiad), and in some cases CANON-Dynamic underperforms DR.GRPO on specific metrics (AIME25: 0.3 → 0.0). This suggests CANON's benefits are contingent on the base model having sufficient capability.

- **RoPE modification** (extending context from 4096 to 16384) is applied to all models, but it is a non-trivial architectural change whose interaction with CANON is not analyzed. If baselines also use this modification, it should be stated explicitly.

- **The 16-rollout budget** (§5.1) is large; CANON requires enough samples per query for meaningful binary splits. No ablation tests how performance degrades with fewer rollouts (e.g., 4 or 8), which matters for practical adoption.

### Trivial:

- Formatting artifacts from PDF parsing (mangled equations in Theorem 1, subscripts) make some parts hard to read, but these are clearly not author errors.

## Nice-to-Haves

- **Sensitivity analysis for μ**: Showing performance curves as μ varies from 0 to 1 in small increments (not just 0, 0.5, 1.0) would clarify how sensitive CANON is to this parameter and whether a fixed, well-chosen μ can achieve comparable performance to the heuristic schedules.

- **Ablation on number of groups**: The binary split is justified by Theorem 1, but no experiment tests whether 3-4 groups (quartiles) would help or hurt. This would confirm the binary design is principled rather than arbitrary.

- **Evaluation on non-math tasks**: CANON is evaluated exclusively on tasks with binary verifiable rewards; testing on code generation or other RLVR domains would strengthen generality claims.

- **Isolating the advantage estimator vs. length-weighting**: Running CANON-Eff without the regrouping (i.e., DR.GRPO + α-weighting) would clarify how much of the efficiency gain comes from the advantage structure vs. the weighting scheme.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "fatal" overclaim about no-directional-prior.** While the "no directional prior" framing is indeed overstated (see Major weakness #1), it is not a fatal error that invalidates the paper. The *actual* contribution—learning direction from data rather than hard-coding it—is real and distinguishable from prior work. The claim needs correcting, not rejection.

- **Concern that CANON's α-weighting (§4.3) contradicts the "no prior" claim.** While true, this is a designed extension for efficiency control and is presented as such ("by slightly reducing the weight of longer responses"). It's not hidden and doesn't invalidate CANON's core mechanism at μ=0 or μ=1.

- **Human Finder's concern about "missing concurrent methods" (DAPO).** The paper compares against the most relevant advantage estimation methods (GRPO, DR.GRPO, RLOO, ReMax, REINFORCE++) and two entropy-related baselines. DAPO focuses on a different aspect (importance sampling) and is not a direct advantage estimation competitor. This is not a missing baseline.

- **Concern about statistical significance on small benchmarks.** While valid, single-run evaluation is the norm in RLVR papers at this scale. Flagging this as minor is appropriate; demanding confidence intervals for 7B-8B model training runs is above community standards.

- **Spark's request for "per-benchmark breakdown for all three models."** Table 1 provides per-benchmark breakdown for the main model (7B); Tables 2-3 show aggregated metrics for the other models. This is standard practice and not a critical omission.

## Novel Insights

The inter-group/intra-group decomposition reveals an important structural insight about RLVR dynamics: that the same metric (e.g., entropy) can have opposing benefits at different training stages (exploitation early, exploration later), which explains why CANON-Dynamic's scheduling outperforms either component alone. This mirrors similar findings in RL more broadly (curriculum-style exploration-exploitation tradeoffs), but the specific instantiation in the advantage estimation framework for LLM reasoning is novel and practically useful. The observation that CANON-Inter naturally favors exploitation (low-entropy/high-certainty correct answers) while CANON-Intra encourages exploration (giving more advantage to correct answers from the lower-reward group) provides a principled way to control the exploration-exploitation tradeoff in RLVR without explicit entropy bonuses or penalties.

## Suggestions

1. **Reframe the core claim**: Replace "without presuming its direction" with "without hard-coded directional preferences—the beneficial direction is inferred from reward statistics." This is more accurate and still distinct from prior work.

2. **Add a fixed-μ comparison for CANON-Dynamic**: Run experiments with a single well-chosen μ value (e.g., 0.7) across all models and compare to the scheduling results. If the gap is small, it strengthens the method's practicality; if large, it acknowledges the scheduling dependency honestly.

3. **Include CANON-Eff built on DR.GRPO** as an ablation to isolate the efficiency contribution of the α-weighting from the advantage estimator improvement.

4. **Provide intuition for Theorems 1 and 2** in the main text, and verify that the independence assumption in Theorem 2 is approximately satisfied for entropy vs. length in practice (report the correlation).

## Score and Decision

**Calibration anchors:**

- SCoRe (self-correction via RL, Oral, score 8): Much stronger contribution with clear novelty and thorough analysis.
- F0GNv13ojF (RL reward design for LLM reasoning, Reject, avg ~5): Similar domain, rejected for novelty concerns and limited methodological advance.
- EAPO/SXUMYMETIR (entropy advantage estimation, Withdrawn/Reject, score 3): Weak paper with heuristics and limited experiments.
- GAGE/90UhF7e8jo (goal achievement exploration, Reject, avg ~5.5): Similar concern about hyperparameter sensitivity undermining generality claims.
- Length correlations/sNtDKdcI1f (RLHF length study, Reject, avg 6): Well-executed empirical analysis but limited novelty.

CANON is above EAPO (clearer motivation, more thorough experiments, actually works) and above F0GNv13ojF (more novel methodological contribution, broader evaluation). It is below GAGE-level concerns (the scheduling issue is real but the core method is more principled). The overclaim about "no directional prior" and the theoretical presentation issues are the main detractors. The empirical contribution is solid and the efficiency results are practically valuable.

**Score: 5.5** — The core idea is sound and empirically validated, but the main conceptual claim is overstated, the theoretical presentation has issues, and the scheduling introduces the kind of per-model tuning the paper argues against. The contribution is meaningful but not yet at the threshold for a strong accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>