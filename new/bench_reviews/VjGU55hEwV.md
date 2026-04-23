## Summary

RLIE proposes a framework that combines LLM-generated natural language rules with regularized logistic regression for global weighting, iteratively refined via hard-example mining. The paper's most notable contribution is a systematic hierarchical evaluation (E1–E4) comparing direct linear inference against three levels of LLM-augmented inference, finding that the simplest strategy (linear-only) consistently outperforms all LLM-augmented variants across six datasets.

## Strengths

- **Systematic hierarchical evaluation design (E1–E4):** The paper isolates the contribution of each information type injected into the LLM (rules only → rules+weights → rules+weights+linear prediction). This is a clean experimental design that directly addresses how learned rules should be utilized, and the finding that linear-only inference (E1) outperforms all LLM-augmented strategies (E2–E4) is a genuinely useful empirical result for the neuro-symbolic community (Table 2, Section 5.2).

- **Consistent improvement over baselines on a single backbone:** Even when restricted to the same DeepSeek-V3 backbone used by all baselines, RLIE achieves the best Accuracy/F1 on all six datasets (Table 1: 70.7 vs 69.3 on Reviews, 82.3 vs 80.5 on Dreaddit, 67.0 vs 65.4 on Headlines, 63.0 vs 50.0 on Citations, 90.7 vs 85.2 on LLM Detect, 65.6 vs 63.5 on Retweets). This consistency across diverse tasks supports the generalizability claim.

- **Practical ternary judgment scheme:** The $z_{i,j} \in \{-1, 0, +1\}$ design (Section 3.1, Eq. 1) explicitly models rule coverage through abstention, reducing forced misclassifications and enabling sparse combinations—a practical design absent from prior work like HypoGeniC.

- **Low variance and robustness to hyperparameters:** The paper demonstrates low variance across runs (Section 5.1) and stable performance across a wide range of coverage thresholds $\gamma \in [0.1, 0.5]$ (Table 4), supporting practical deployability.

- **Illustrative case study of iterative refinement:** Table 3 shows rules evolving from generic to specific patterns across three rounds, with training accuracy improving from 0.625 to 0.700, providing qualitative evidence for the refinement mechanism.

## Weaknesses

### Fatal
None.

### Major

- **Multi-backbone reporting inflates headline numbers while baselines use a single backbone:** Table 1 reports RLIE results for three backbones (Qwen3-Next-80B, Qwen3-235B, DeepSeek-V3) and bolds the best per dataset, while all baselines (Zero-shot, Few-shot, IO Refinement, HypoGeniC) are run exclusively on DeepSeek-V3. On Reviews and Retweets, the bolded RLIE numbers (71.4, 66.5) come from Qwen3-235B, not DeepSeek-V3 (70.7, 65.6). Critically, even the DeepSeek-V3 RLIE results still beat all baselines on all six datasets, so the "superior overall performance" conclusion is intact—but the magnitude of the advantage is inflated on 2/6 datasets, and the per-dataset backbone selection creates an asymmetric comparison. The paper should report the full backbone×method matrix or, at minimum, present the single-backbone comparison as the primary result.

- **The conclusion that LLMs are unreliable at "fine-grained, controlled probabilistic integration" is overclaimed:** The paper states this finding "aligns with the observation that LLMs excel at semantic generation and interpretation but are less reliable at fine-grained, controlled probabilistic integration" (Section 6) and refers to "the deficiency of LLMs to perform fine controlled inference" (Section 5.2). However, this conclusion rests on a single prompting strategy for E2–E4: presenting rules, weights, and predictions as text in a zero-shot prompt. No alternative prompting approaches are tested—no chain-of-thought reasoning about how to integrate weighted evidence, no few-shot demonstrations of correct probabilistic reasoning, no structured output formats. The empirical observation (E1 > E2–E4) is valid; the generalization to a fundamental LLM limitation is not fully supported without testing at least one alternative prompting strategy.

### Minor

- **Small dataset sizes without significance testing:** All experiments use 200/200/300 train/validation/test splits (Section 4.3). On 300 test samples, the difference between RLIE and the best baseline on several datasets is within 1–3 points (e.g., Headlines: 67.0 vs 65.4; Retweets: 65.6 vs 63.5). While RLIE's consistency across all six datasets is meaningful, the absence of any significance testing (e.g., paired bootstrap, McNemar's test) leaves individual dataset comparisons uncertain. The consistency across six datasets partially mitigates this, but the paper should acknowledge the limited statistical power.

- **Circularity between rule generation and rule evaluation:** Both rule generation and ternary judgment ($z_{i,j}$) are performed by the same LLM (Section 3.1). This creates a systematic bias risk: the LLM that proposes a rule also decides whether it applies, with no independent ground truth for rule satisfaction. The paper does not discuss this circularity or investigate whether LLM rule-application judgments are reliable. A human annotation study on a subset of $z_{i,j}$ judgments would strengthen confidence in the learned rule features.

- **Incremental novelty of the framework:** Each component of RLIE—LLM rule generation, logistic regression with elastic net, hard-example mining for iterative refinement—is individually well-established. The contribution is the combination and the E1–E4 evaluation, which is meaningful but modest in novelty. The iterative refinement is essentially a form of boosting (find hard examples, generate new rules, reweight) but this connection is not discussed.

- **Inconsistency in reported LLM usage:** Section 4.3 states "All experiments involving LLMs utilized gpt-4o-mini," yet Table 1's Backbone column shows DeepSeek-V3, Qwen3-Next-80B, and Qwen3-235B. This discrepancy is confusing and should be clarified—it is unclear whether gpt-4o-mini was used for some auxiliary role, or whether the experimental details section was not updated to reflect the actual models used.

### Trivial
None.

## Nice-to-Haves

- Running all baselines on all three backbones would fully address the multi-backbone concern and could reveal interesting backbone×method interactions.
- Testing alternative prompting strategies (chain-of-thought, few-shot demonstrations for weighted reasoning) for E2–E4 would substantially strengthen or qualify the "LLMs struggle with probabilistic integration" conclusion.
- Per-instance confusion analysis comparing E1 vs E2–E4 predictions would reveal whether LLM-augmented inference systematically overrides correct linear predictions or fails on mutually difficult cases.
- Formal definition of the "generalizable methods" criterion used to exclude LoRA from the bolded comparison (Table 1 note), rather than the informal "fails to generalize on complex reasoning tasks."

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Unfair baseline comparison invalidates the core claim" (Harsh Critic Issue 1):** The harsh critic claims the backbone selection confound "undermines the paper's central empirical claim." Verified against the paper: even with DeepSeek-V3 alone, RLIE beats all baselines on all 6 datasets. The multi-backbone reporting inflates magnitude on 2/6 datasets but does not change the ranking or conclusion. Downgraded from fatal to major (reporting fairness).

- **"LLMs Cannot Do Probabilistic Integration" conclusion "not established" (Harsh Critic Issue 2):** The harsh critic characterizes the paper as claiming "LLMs Cannot Do Probabilistic Integration." Verified against the paper: the actual language is more measured—"less reliable at fine-grained, controlled probabilistic integration" and "non-trivial." However, the Section 5.2 phrase "deficiency of LLMs to perform fine controlled inference" is indeed too strong without testing alternative prompting strategies. Kept as major but with corrected framing.

- **"LoRA baseline dismissal is post hoc" (Harsh Critic Section 4.2):** The harsh critic claims the "generalizable" criterion is post hoc. Verified against the paper: LoRA achieves 94.1 and 99.7 on two tasks but 51.5, 54.4, 52.1, 51.4 on the other four—near chance level. This pattern clearly supports poor cross-task generalization. The characterization is defensible, though the criterion should be formally defined. Moved to nice-to-have.

- **"The combination is not novel—each component is standard" (Harsh Critic Section 1):** While true that each component is individually standard, the combination plus the E1–E4 evaluation constitutes a genuine contribution. The novelty concern is valid but not disqualifying. Kept as minor.

- **"Missing experiments: run baselines on all backbones, test alternative prompting, significance tests" (Harsh Critic Missing Experiments):** The first two are addressed above. Significance testing is a valid concern but moved to minor since the consistency across six datasets provides some protection. All three moved to nice-to-have/minor.

- **"Connection to boosting theory not discussed" (Harsh Critic Section 3.3):** This is a reasonable observation but more of a discussion point than a weakness. Moved to minor.

- **"Abstain (0) treated as midway between +1 and −1" (Harsh Critic Section 3.2):** The logistic regression naturally handles this through learned weights; if abstention needs different treatment, the model can learn appropriate weights. This is a design choice, not a flaw. Removed.

- **"Missing proofs/appendix" (Harsh Critic):** Removed per rules—parser strips appendix content from all papers.

## Novel Insights

The most valuable insight from the review process is the recognition that the multi-backbone reporting issue, while real, is less damaging than it appears at first glance. The core claim of "superior overall performance" is actually supported by the single-backbone (DeepSeek-V3) results alone, since RLIE-DeepSeek-V3 beats every baseline on every dataset. The real contribution of this paper is not the "RLIE beats baselines" finding—which might be expected given a more powerful aggregation scheme—but the E1 > E2–E4 result, which is counterintuitive and has direct practical implications for neuro-symbolic system design: delegating global aggregation to classical models while using LLMs only for local semantic judgments is a replicable and well-supported engineering principle.

## Suggestions

- Present the single-backbone (DeepSeek-V3) comparison as the primary result in Table 1, with multi-backbone results in an appendix or supplementary table. This eliminates the asymmetry concern while preserving the full information.
- Add at least one alternative prompting strategy for E2–E4 (e.g., chain-of-thought or few-shot demonstrations) to test whether the E1 > E2–E4 finding is robust to prompt design. This would significantly strengthen the paper's main qualitative conclusion.
- Report paired significance tests (even simple McNemar's tests) for the key comparisons in Tables 1 and 2, given the small test set sizes.
- Clarify the apparent inconsistency between Section 4.3 ("gpt-4o-mini") and the Backbone column in Table 1.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LLM + symbolic engine for geometry | /home/wg25r/review_agent/human_reviews_2026/1sffPGGQyT.md | 7.0 | RLIE is below this: InternGeometry has a stronger empirical result (IMO-level) and cleaner methodology |
| Reasoning with Sampling | /home/wg25r/review_agent/human_reviews_2026/Vsgq2ldr4K.md | 7.5 | RLIE is well below this: simpler method, more surprising finding, broader evaluation |
| LLM constraints + logistic regression for clinical data | /home/wg25r/review_agent/human_reviews_2026/vKSSZHTdNP.md | 4.67 | RLIE is comparable: similar combination of LLM priors + linear model, similar small-dataset concerns, but RLIE has a cleaner experimental design (E1-E4) |
| AutoRule for RL rewards | /home/wg25r/review_agent/human_reviews_2026/HUNajlG9Tp.md | 5.50 | RLIE is comparable: both automate rule extraction, but AutoRule has stronger downstream task results; RLIE has the E1-E4 insight |
| RuleSHAP for LLM bias | /home/wg25r/review_agent/human_reviews_2026/s2vWrgO4OA.md | 4.50 | RLIE is comparable or slightly above: RuleSHAP has similar incremental novelty concerns |
| CLoVE/GloVE rule extraction | /home/wg25r/review_agent/human_reviews_2026/PRR120c01e.md | 4.00 | RLIE is above this: CLoVE has more circularity issues and weaker evaluation |
| LLM Horn rule extraction (unfair baselines) | /home/wg25r/review_agent/human_reviews_2026/iCJG36rclz.md | 1.50 | RLIE is clearly above: the Horn rule paper has truly unfair baselines and overclaimed novelty |
| Small data deep learning (incoherent) | /home/wg25r/review_agent/human_reviews_2026/wAb8vtEZfM.md | 1.20 | RLIE is far above: this is an incoherent paper |

RLIE sits in the 4.5–5.5 range based on calibration. It has a genuine finding (E1 > E2–E4) and consistent baseline improvements, but the multi-backbone reporting asymmetry, overclaimed explanation, and small datasets without significance testing are real methodological issues. It is above the clearly flawed papers (1–3 range) but below the solid accept papers (7+ range). Compared to the most similar anchor (vKSSZHTdNP, 4.67), RLIE has a cleaner experimental design and a more impactful finding, but also has the multi-backbone reporting issue that the clinical data paper doesn't have. I place it at 5.0—borderline, leaning reject due to the combination of methodological concerns.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>