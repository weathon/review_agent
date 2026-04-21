Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final review.

## Summary

The paper introduces a routing framework that allocates preference annotation instances to either human annotators or an LM by training a Performance Prediction Model (PPM) to predict reward model performance under arbitrary routing configurations, then selecting the configuration that maximizes predicted performance. The PPM is trained on MULTIPREF, a new 10K-instance preference dataset with both human and GPT-4 annotations, and is shown to produce hybrid datasets that outperform both 100% human and 100% synthetic annotations on RewardBench (7–13% absolute improvement) across four datasets, while requiring only 20–70% human annotations.

## Strengths

- **Substantial RewardBench improvements from hybrid routing**: Table 3 shows the best hybrid mix achieves 70.6% on RewardBench for MULTIPREF (vs. 60.4% human, 66.5% synthetic), 79.7% for Helpsteer2 (vs. 72.4%/65.8%), 66.8% for AlpacaFarm (vs. 55.0%/60.9%), and 72.2% for ChatArena (vs. 59.0%/71.6%). These are consistent, large improvements across all four datasets.

- **The MULTIPREF dataset is a genuine contribution**: A 10,461-instance preference dataset with both human (4 annotators per instance, 34.8% qualification pass rate) and GPT-4 annotations under controlled guidelines fills a gap and serves as seed data for the entire framework. The quality control efforts (65% filter rate, majority voting) are well-designed (Table 1).

- **The "moderation trend" analysis (§5) provides actionable, data-centric insight**: Table 5 shows that instances with moderate safety concern (gain 0.085), moderate intent complexity (gain 0.030), and moderate BERTScore (gain 0.194) benefit most from human annotation. The interpretation—simple cases don't need humans, hard cases are equally difficult for both—is intuitive and novel. This insight has value independent of the PPM framework.

- **The framework transfers to unseen datasets**: Figure 4 demonstrates that the routing strategy (using the PPM trained only on MULTIPREF) outperforms random hybrid selection across Helpsteer2, ChatArena, and AlpacaFarm at most annotation budgets, providing evidence that the routing decisions generalize.

- **Clean and well-motivated problem formulation**: Equation 1 and the overall optimization framing in §2 are clearly stated, and the budget-constrained routing in Algorithm 1 is directly applicable to practical scenarios.

## Weaknesses

### Fatal
None.

### Major

- **PPM prediction accuracy is not validated on transfer datasets, breaking the causal chain for the generalization claim**: The PPM is trained and validated only on MULTIPREF (Spearman ρ = 0.673 on 16 held-out candidates). When applied to Helpsteer2, ChatArena, and AlpacaFarm, the paper reports the downstream reward model performance of the routed hybrid datasets but never reports whether the PPM's predictions are accurate on these new datasets. The paper claims "our method generalizes well to all three" (§4.2), but the evidence only shows that the routing *works* (produces better RewardBench scores than random), not that it works *because the PPM predicts well*. It is possible that the routing still produces decent hybrid datasets simply because adding some human annotations at a non-trivial budget is generally helpful—which the random baseline already partially demonstrates. Without a version of Figure 3 for a transfer dataset, the paper cannot attribute the improvements to the PPM's predictive quality on new distributions. This gap matters because the PPM is the paper's core methodological contribution.

- **No comparison against simple routing heuristics**: The analysis in §5 reveals that a small number of features—moderate safety concern, moderate intent complexity, moderate BERTScore, and a few subject areas—capture most of the "gain" from routing instances to humans. A natural and important baseline is a simple rule-based router (e.g., "route to human if safety concern is moderate OR BERTScore ∈ [0.33, 0.67]") that requires no PPM, no candidate sampling, and no RM training. If such a heuristic matches PPM-based routing, the PPM machinery is unnecessary overhead. The paper discusses the moderation trend as an insight but never tests whether acting on it directly suffices. This comparison would determine whether the contribution is the *framework* (PPM + routing) or merely the *finding* that certain instance types benefit from human labels—a distinction that significantly affects the paper's impact.

- **Downstream task improvements are marginal**: Table 4 (Best-of-N evaluation) shows improvements from the hybrid mix that are often tiny: Helpsteer2 Average goes from 52.6 (100% human) to 52.8 (hybrid)—a 0.2-point gain; AlpacaFarm goes from 53.1 to 53.3; ChatArena shows no improvement (53.9 for both). Only MULTIPREF shows a meaningful gain (48.3 → 50.5). The RewardBench improvements are large (7–13%), but the routing directly optimizes RewardBench, making it the metric most likely to show gains. The downstream evaluation is the more meaningful test, and the signal there is weak. The paper reports averages of 3 runs with no standard deviations, making it impossible to assess whether the small improvements are real or within noise. This limits the practical significance of the framework.

### Minor

- **The ChatArena RewardBench/Best-of-N discrepancy is acknowledged but not investigated**: The paper notes an "opposite correlation between RewardBench and Best-of-N evaluation in the ChatArena case" (§4.3) and defers investigation to future work. However, if optimizing for RewardBench can hurt downstream performance on some distributions, this is a fundamental concern about the optimization target—not merely a dataset-specific anomaly. A deeper investigation (e.g., checking correlation between RewardBench and Best-of-N across candidate datasets) would strengthen the paper.

- **The quadratic PPM is potentially overparameterized relative to training data**: With ~30+ tag features, a quadratic model includes all linear, squared, and interaction terms—roughly ~500 parameters—trained on only 200 candidate datasets. While it outperforms the linear model on 16 held-out samples (Spearman ρ = 0.673 vs. 0.515), this improvement is not tested for statistical significance, and 16 samples provides little power. The concern is not that the PPM fails on MULTIPREF (it appears to work there), but that this overparameterization may limit its reliability on out-of-distribution datasets—which is precisely where the paper claims generalization.

- **Human–LM agreement rate is not reported**: The paper acknowledges that when human and LM agree on the label, routing is irrelevant (§2.1), but never reports the agreement rate on MULTIPREF or the transfer datasets. This directly determines the ceiling of routing improvement and is a basic dataset characterization that would strengthen the analysis.

### Trivial
None.

## Nice-to-Haves

- A cost-effectiveness analysis quantifying the marginal value of each human annotation relative to 100% synthetic, since on some datasets (MULTIPREF, AlpacaFarm) 100% synthetic already outperforms 100% human by a substantial margin.

- Investigation of how the framework scales beyond 7K instances, since all experiments are conducted at the same size to control for dataset size effects.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Misleading framing about reducing human annotation cost"** (Harsh Critic): The critic argues the paper's framing is misleading because 100% synthetic already beats 100% human on some datasets, so the more relevant question is whether adding human on top of synthetic justifies its cost. However, the paper's claim is that hybrid outperforms both sources—including synthetic—and uses only 20–70% human annotations. The framing of reducing human annotation cost while improving quality is supported by the results. The cost-effectiveness angle is a valid nice-to-have, not a flaw in the framing.

- **"The 'hybrid beats 100% human' bar is low due to mediocre human annotation quality"** (Harsh Critic): This is partially true—on MULTIPREF, 100% synthetic (66.5) already beats 100% human (60.4) on RewardBench. However, the hybrid approach also beats 100% synthetic, and on Helpsteer2 (where human quality is high), the hybrid still improves over both. The bar being "low" on some datasets does not invalidate the framework.

- **"Algorithm 1 doesn't ensure diversity across candidates"** (Harsh Critic): This is a minor observation about the candidate sampling algorithm. The paper generates 200 candidates for training and 500 for routing, which provides sufficient coverage in practice. This is a presentation concern, not a methodological flaw.

- **"Rationale for z_i=0 as human is counterintuitive"** (Harsh Critic): This is a notation preference, not a substantive issue.

- **"Demand for DPO/policy model performance"** (Harsh Critic, §8): The paper explicitly acknowledges this limitation (§8) and even reports DPO results in the appendix. This is already addressed by the authors.

- **"Missing references to related work"** (per instructions, do not mention missing related works).

- **"Formatting/style nitpicks, typos, grammar"** (per instructions, remove these).

## Novel Insights

The most interesting insight is that the paper's strongest contributions may be its dataset and analysis rather than its framework. The MULTIPREF dataset and the §5 "moderation trend" analysis provide independent, actionable value: they tell practitioners *what kinds of instances* benefit from human annotation. The PPM framework is one possible mechanism for exploiting this insight, but without validating that the PPM's predictions are accurate on transfer datasets or comparing against a heuristic router, the paper cannot conclusively demonstrate that the PPM is the right mechanism—only that the insight itself (some instances benefit from humans) is correct. This suggests the paper's impact may be more data-centric than methodological.

## Suggestions

- **Validate PPM predictions on at least one transfer dataset**: Generate a small set of candidate datasets from Helpsteer2 or AlpacaFarm, train RMs on them, and compare the PPM's predicted vs. actual RewardBench performance (a version of Figure 3 for a transfer dataset). This directly tests whether the PPM generalizes, which is the core methodological question.

- **Add a simple heuristic baseline**: Implement a rule-based router using the top-5 features from Table 5 (e.g., route to human if BERTScore ∈ [0.33, 0.67] OR safety concern is moderate). This determines whether the PPM adds value beyond the insight itself.

- **Report standard deviations across the 3 runs** for Table 4, or at minimum compute confidence intervals for the key comparisons (hybrid vs. 100% human, hybrid vs. 100% synthetic).

- **Report the human–LM agreement rate** on MULTIPREF and transfer datasets as basic dataset characterization.

## Evaluation

**Originality**: The routing framework concept is novel and well-formulated. The PPM-based approach is a reasonable instantiation, though the idea of selectively combining human and AI annotations has roots in active learning and learning-to-defer. The MULTIPREF dataset and the moderation trend analysis are original contributions.

**Importance of research question**: High. The question of how to efficiently combine human and AI preference annotations is practically important for RLHF pipelines.

**Claims well-supported**: Partially. The RewardBench improvements are well-supported. The generalization claim is supported by routing results but not by PPM validation on transfer datasets. The downstream improvements are weakly supported. The missing heuristic baseline leaves the necessity of the PPM unestablished.

**Soundness of experiments**: Reasonable on MULTIPREF, but the generalization experiments have a gap (no PPM validation on transfer datasets) and the downstream evaluation lacks statistical rigor.

**Clarity of writing**: Good. The problem formulation is clear, the algorithm is well-described, and the analysis section is well-structured.

**Value to the community**: Moderate to high. The MULTIPREF dataset, the moderation trend insight, and the framework code release would all be valuable resources. The framework itself is useful if the PPM generalizes, which needs further validation.

## Score and Decision

**Calibration anchors used:**

1. **Probabilistic Learning to Defer** (avg 8.0, Oral, `/home/wg25r/review_agent/human_reviews/zl0HLZOJC9.md`): Human-AI routing with deferral, principled handling of missing annotations, workload control. Stronger than current paper due to principled guarantees and comprehensive baselines.

2. **Trust or Escalate** (avg 8.0, Oral, `/home/wg25r/review_agent/human_reviews/UHPnqSTBPO.md`): Selective routing between LLM judges and humans with provable agreement guarantees. Much stronger theoretical grounding than current paper.

3. **Anyprefer** (avg 6.5, Accept Poster, `/home/wg25r/review_agent/human_reviews/WpZyPk79Fu.md`): Preference data synthesis framework with extensive experiments across 21 datasets. More comprehensive experiments than current paper, but more engineering-oriented.

4. **Active Teacher Selection for RLHF** (avg 5.25, Reject, `/home/wg25r/review_agent/human_reviews/2eIembMRQJ.md`): Bandit-based annotator selection. Similar weaknesses to current paper—applicability concerns, weak baselines—but less empirical support.

5. **Data-Centric Human Preference Optimization** (avg 5.25, Reject, `/home/wg25r/review_agent/human_reviews/2Cg4YrsCMA.md`): Data-centric approach to preference learning. Weaker than current paper in experiments and contributions.

6. **LLM Bandit** (avg 5.25, Reject, `/home/wg25r/review_agent/human_reviews/rEqETC88RY.md`): Dynamic routing framed as bandit. Similar level but with clarity issues.

7. **What can we learn from Harry Potter?** (avg 2.0, Withdrawn, `/home/wg25r/review_agent/human_reviews/3ZdGSTxKuy.md`): Overclaimed contribution, narrow experiments. Much weaker than current paper.

The current paper sits between the 5.25 rejects (similar gaps in baselines/validation) and the 6.5 accept (less comprehensive but more principled). It is stronger than the 5.25 papers due to its substantial RewardBench improvements, dataset contribution, and transfer experiments, but weaker than the 6.5 paper due to missing heuristic baselines and marginal downstream improvements. The core methodological claim (PPM-based routing) has significant evidential gaps that keep it below the acceptance threshold for a top venue.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>