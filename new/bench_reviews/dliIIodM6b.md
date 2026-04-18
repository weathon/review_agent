Now I have enough context from related papers. Let me synthesize the final review.

**Calibration assessment:**

- CREAM (self-rewarding LM with consistency regularization): Scores 6,6,6,8 → Accept (Poster)
- SeRA (self-reviewing using implicit reward margins): Scores 6,6,6,6 → Accept (Poster)
- Meta-Rewarding (LLM-as-meta-judge): Scores 3,6,5,6 → Reject
- AIPO (iterative DPO for length exploitation): Scores 3,3,5 → Withdrawn/Reject
- iREPO (implicit reward pairwise for self-alignment): Scores 3,3,5,3,3 → Withdrawn/Reject

The DICE paper is in a similar space to SeRA and CREAM. The harsh critic raises a serious conceptual concern about circularity. Let me verify this against the paper.

Looking at the paper carefully, the "circularity" argument needs to be examined. The key insight is: when you use the current policy's implicit reward to rank its own samples, you're creating *on-policy* preference data. The preference labels come from the implicit reward, which reflects the *current policy's learned alignment*, but the key benefit comes from using on-policy samples rather than off-policy samples. The paper does cite Guo et al. (2024) and Tajwar et al. (2024) for this motivation. Additionally, the preference labels are not purely circular — they provide a *relative ordering* among diverse on-policy samples based on the current policy's reward, which, when combined with experience replay from the original human data, creates a new training signal. The circularity concern is partially valid but overstated: the process does introduce new information (on-policy data) and the implicit reward evolves across iterations (since the reference and policy models change).

However, the evaluation concern is serious — only AlpacaEval 2 and Arena-Hard, both LLM-judged, are used. No capability benchmarks, no human eval, no safety evaluations.

Let me now craft the final review.</think>## Summary

This paper proposes DICE (self-alignment with DPO Implicit rEwards), a method that uses the implicit reward model available from DPO training to bootstrap further alignment of language models. Specifically, the implicit reward r(x,y) = β log(π_θ(y|x)/π_ref(y|x)) is used to rank the model's own generated responses, construct new preference data, and run additional DPO rounds. Two refinements are introduced: length-regularized reward shaping to mitigate length bias in the preference data, and experience replay that mixes generated data with original human preference data. Experiments on AlpacaEval 2 and Arena-Hard show 8%+ improvements in length-controlled win rate over DPO-tuned base models.

## Strengths

- **Simple and well-motivated core idea**: Using DPO's own implicit reward for self-bootstrapping is an elegant observation — no separate reward model or external LLM judge is needed. This is a genuine insight that the paper motivates clearly (Section 3, Figure 1).

- **Strong empirical improvements on reported benchmarks**: DICE achieves substantial improvements (>8% LC win rate) on AlpacaEval 2 across two base models, and DICE-Llama3-8B surpasses Gemini Pro on the leaderboard (Table 2). The gains are consistent and large relative to baselines.

- **Effective practical techniques**: Length-regularized reward shaping addresses a real problem (Figure 2 shows clear length bias in vanilla implicit rewards), and the ablation in Table 4 confirms α* found via Eq. (6) performs best. The experience replay ablation (Figure 3) shows a clear U-shaped pattern with γ=0.5 performing best, supporting the motivation from continual learning.

- **Compatibility with other DAP algorithms**: Table 3 demonstrates that DICE-generated data improves IPO, KTO, and Hinge-based fine-tuning beyond their offline baselines, showing generality within the preference-tuning family.

- **Favorable comparison with trained scalar reward model**: Table 5 shows the implicit reward achieves 0.698 alignment rate with GPT-4o labels versus 0.624 for an internal RM trained on the same data, providing empirical justification for using implicit rewards in this setting.

- **Honest limitation discussion**: The paper acknowledges that improvement does not continue beyond ~3 iterations (Section 5), and that the method depends on a well-trained initial DPO model.

## Weaknesses

### Fatal
None.

### Major

- **Evaluation relies exclusively on LLM-judged benchmarks, with no capability or safety benchmarks**: All primary results (Tables 1–4, Figure 3) use AlpacaEval 2 and Arena-Hard, which are instruction-following benchmarks judged by other LLMs (GPT-4-Turbo and Mistral-Large). The paper makes strong claims about "improving alignment" and being a "general purpose approach that can improve alignment for any single DPO-tuned base model" (Abstract, Section 1), but does not verify whether the alignment gains come at the cost of reasoning, factual accuracy, or safety. Papers in this space (e.g., CREAM, SeRA) have been asked to evaluate on broader benchmarks. The risk is that DICE is optimizing for patterns that these specific LLM judges reward rather than genuine alignment improvements. This directly undermines the breadth of the claimed contribution.

- **Limited model scale and diversity**: Only two base models (Zephyr-7B-beta and Llama-3-8B-DPO) are tested, both closely tied to the UltraFeedback/Zephyr training pipeline. The paper claims DICE is "a general purpose approach that can improve alignment for any single DPO-tuned base model" (Section 1), but does not test on different model scales (e.g., 70B), different architectures, or models trained on different preference datasets. Related work in this area (Meta-Rewarding) has been criticized for testing on only one model. The generalizability of the claim is not well-supported by the evidence.

- **Missing strong self-alignment baselines**: The paper compares against Offline DPO and LLM-as-a-Judge but does not compare against important recent self-alignment methods such as SPPO or SimPO. The LLM-as-a-Judge baseline uses a coarse 0–5 scoring template (which the authors themselves note may harm performance), and the paper does not compare against an iterative DPO pipeline using an external reward model (beyond the small-scale Section 4.4 comparison). Given that the core claim is that implicit rewards are competitive without external feedback, comparison with methods like SPPO (which uses on-policy sampling with external RMs) or SimPO is needed to situate DICE's contribution. The Section 4.4 comparison against IntIRM only evaluates reward agreement on 500 examples (Table 5), not the full iterative training pipeline, making it insufficient to establish competitiveness.

### Minor

- **Incomplete analysis of iterative behavior**: The paper only reports 2 iterations and acknowledges degradation beyond ~3 iterations (Section 5). Given that DICE is explicitly an iterative method (Algorithm 1), this is a gap. The paper should report and analyze the trajectory at iterations 3–4, including what goes wrong (reward hacking, length issues, distribution collapse).

- **No reference model update ablation**: Algorithm 1 sets π_ref^(t) = π_θ(t-1) each iteration, changing the implicit reward definition itself. This is an important design choice, but the paper does not compare against keeping the original reference model fixed. Different reference model strategies could significantly affect results and our understanding of the mechanism.

- **α* optimization is heuristic without sensitivity analysis**: The objective in Eq. (6) minimizes average absolute length difference, which assumes the original UltraFeedback distribution is ideal. Table 4 tests α=0 and 2α*, but the paper does not examine whether α* generalizes across domains, model sizes, or iterations, or provide deeper justification for this specific debiasing criterion.

- **The IntIRM comparison (Section 4.4) overclaims**: The conclusion that "the implicit reward is a competitive option in self-alignment settings" (end of Section 4.4) is drawn from a small-scale experiment (500 pairs, agreement with GPT-4o labels) using ERM-555k trained on different data. The comparison does not test the full iterative training pipeline and is too narrow to support the broad claim.

- **No computational cost analysis**: DICE requires sampling K=16 responses per prompt and running α-search each iteration. While the paper provides training details, it does not quantify compute cost relative to baselines, which is important for practical assessment.

### Trivial

- The notation in Eq. (6) uses 𝔼(|y_w| - |y_l|) but the text describes "average absolute difference," creating minor ambiguity about whether the objective uses signed or absolute length difference.

## Nice-to-Haves

- Evaluations on capability benchmarks (MMLU, GSM8K, HumanEval, TruthfulQA) to verify alignment improvements don't regress reasoning/safety.
- Human evaluation on a sample of outputs to validate that AlpacaEval improvements reflect genuine human preference.
- Analysis of response diversity/entropy across iterations (mode collapse risk).
- Results beyond 3 iterations with analysis of failure modes.
- Per-category breakdown on AlpacaEval or Arena-Hard to show which tasks benefit and which don't.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "conceptual circularity" argument** (Critical Issue #1): While thought-provoking, this overstates the problem. The paper's approach is not purely circular — it uses on-policy sampling from the current policy (which generates *new* data points not seen in training) and orders them by the evolving implicit reward. The implicit reward changes across iterations because both the policy and reference models change. The key insight is that on-policy preference data provides a different training signal than offline data, which is well-established in the literature (Guo et al., 2024; Tajwar et al., 2024). The concern that the method optimizes "what the model already likes" is partially addressed by experience replay, which anchors training to human preference data. While a deeper theoretical understanding of why this bootstrapping works would strengthen the paper, the circularity concern is not as fatal as claimed.

- **Demand for comparison with PPO/RLHF baseline**: The paper explicitly scopes its contribution to the iterative DPO framework (Section 1, Figure 1). Comparing with a full RLHF+PPO pipeline is outside this scope and would require training a separate reward model, which is the approach the paper is trying to avoid.

- **Demand for confidence intervals on benchmarks**: Single-run evaluation is standard practice for AlpacaEval 2 and Arena-Hard in this research community. The demand for bootstrapped confidence intervals is a nice-to-have, not a substantive weakness.

- **Human finder's suggestion about missing SPPO/SimPO comparisons**: While a valid point, SPPO and SimPO use external reward models, which the paper explicitly scopes out. The comparison with LLM-as-a-Judge (also self-referential) and offline DPO (same data) are the most relevant given the paper's stated scope. This is moved to the minor weaknesses as a missing baseline concern.

## Novel Insights

The implicit reward model from DPO not only provides a free reward function but turns out to be surprisingly competitive with explicitly trained scalar reward models on the model's own generated data (Table 5, 0.698 vs 0.624 IntIRM agreement with GPT-4o). This suggests that the implicit reward has a structural advantage in evaluating its own on-policy distribution — it naturally captures what the model "knows" about preference ordering without the distributional mismatch that a separately trained RM may suffer. This is underexplored in the paper and could have deeper implications for understanding when and why self-referential reward signals work.

## Suggestions

1. Add evaluations on at least 2-3 capability benchmarks (MMLU, GSM8K, TruthfulQA) to verify no regression in reasoning or safety — this would significantly strengthen the alignment claims.
2. Compare against SPPO or a similar iterative method with an external RM in the same experimental setting to better contextualize DICE's contribution.
3. Analyze the method beyond 2 iterations (report iter 3 and if possible iter 4) and explain the failure mode when improvement plateaus — this directly addresses the "bootstrapping" framing.
4. Ablate the reference model update strategy (fixed vs. updating each iteration) to clarify what drives the improvements.

## Score and Decision

**Calibration**: I compared this paper against papers in the same research area:
- CREAM (self-rewarding LM with consistency regularization): Scores 6,6,6,8 → Accept (Poster) — similar self-alignment space, evaluated on similar benchmarks, but with broader evaluation and theoretical analysis.
- SeRA (self-reviewing using implicit reward margins): Scores 6,6,6,6 → Accept (Poster) — very close topic (using implicit rewards for iterative DPO), with similar evaluation scope but more models tested.
- Meta-Rewarding: Scores 3,6,5,6 → Reject — criticized for limited model coverage, relying on LLM-judge evaluation, and insufficient analysis of iterative saturation.
- AIPO: Scores 3,3,5 → Withdrawn/Reject — iterative DPO with length exploitation mitigation, but weak baselines and overclaimed.

DICE is between SeRA/CREAM (accepted at 6) and Meta-Rewarding (rejected, score 3-6). DICE's core idea is clean and well-motivated, and the empirical improvements are substantial on the reported benchmarks. However, the evaluation is narrow (only two LLM-judged benchmarks, two base models), missing strong contemporaneous baselines, and hasn't verified that improvements generalize beyond AlpacaEval/Arena-Hard. These issues are similar to those that hurt Meta-Rewarding. The paper is stronger than Meta-Rewarding in having two base models and cleaner ablations, but weaker than SeRA/CREAM in breadth of evaluation and depth of analysis. The conceptual concern about circularity is real but manageable with proper framing. On balance, the paper needs significant revisions to evaluation scope and baseline comparisons before it meets the bar for a top venue.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>