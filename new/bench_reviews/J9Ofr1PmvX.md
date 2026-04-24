 ## Summary
UNSTAR proposes a machine unlearning method for LLMs that generates "anti-samples"—paraphrased questions paired with incorrect answers and misleading rationales—to drive selective forgetting. The paper evaluates the approach on the WPU benchmark and claims superior balanced performance compared to eight baselines, along with a novel capability for fine-grained targeted unlearning that preserves related knowledge.

## Strengths
- **Creative use of rationale generation for unlearning.** Repurposing STaR-style reasoning to generate step-by-step justifications for incorrect answers is an inventive twist on self-improvement methods, applied to the unlearning domain for the first time. Evidence: Algorithm 1 formalizes the iterative generation loop; Section 3 describes the "rationalization" process; Table 3 provides concrete examples of generated paraphrases and justifications.
- **Attention to anti-sample quality controls.** The paper includes semantic-divergence filtering (via Levenshtein distance and MiniLM embeddings) and near-correct answer detection to curate generated paraphrases. Evidence: Section 3 subsections on "Semantically Divergent Questions" and "Near-Correct Incorrect Answers."
- **Clear algorithmic formalization.** The full iterative procedure, including forget-set anti-sample handling and retain-set preservation, is laid out explicitly in Algorithm 1, aiding reproducibility.

## Weaknesses

### Fatal
None.

### Major
- **Gross internal contradictions between text and table in Figure 2 invalidate the empirical comparison.** The narrative describing Figure 2 presents values that directly conflict with the accompanying table. For example: the text states DI achieves 84 on Model Utility, but the table reports 40; the text states UNSTAR Response Quality is 92, but the table reports 100; the text states UNSTAR Hallucination Avoidance is 83, but the table reports 100. These are not minor discrepancies—they span 3 of the 5 metrics for UNSTAR and 1 for a baseline. Because all metrics are normalized ("by the maximum across all methods"), these contradictions make it impossible to know which values are correct and undermine confidence in the entire empirical evaluation.
- **Central claim of fine-grained targeted unlearning is unsubstantiated by quantitative evidence.** Contribution ❸ and the abstract claim that fine-grained, association-specific unlearning is "not achievable by previous works." The only evidence is a single qualitative cherry-picked example (Table 4). There are no quantitative retention metrics for related facts, no baselines evaluated on the fine-grained task, and the main WPU experiments evaluate only coarse targeted unlearning. Without such evidence, the distinguishing capability is anecdotal.
- **Overclaiming the novelty of anti-samples.** The abstract and introduction frame anti-samples as a largely "untapped" third "pillar" of unlearning (alongside methods and loss functions). However, prior approximate unlearning methods already employ negatively labeled or counterfactual data (e.g., gradient ascent on forget data, random labels, refusal-response distillation). The paper does not rigorously establish what makes anti-samples—beyond the specific mechanism of rationale-augmented paraphrases—a conceptually distinct pillar rather than a data-augmentation strategy.

### Minor
- **Figure 3 measures convergence on training paraphrases.** Figure 3 plots "Unlearning Efficacy" against iterations on the same synthetically generated paraphrases that the model is iteratively fine-tuned to fail. This is acceptable as a convergence diagnostic but should not be interpreted as generalization evidence. The paper does evaluate on held-out WPU data in Figure 2, so this is a presentation concern rather than a fatal flaw.
- **RL policy-gradient formulation is disconnected from the actual algorithm.** Equations (1)–(2) frame the method as approximating policy gradients with indicator rewards, yet Algorithm 1 performs filtered supervised fine-tuning rather than policy-gradient optimization with reward baselines. The theoretical framing adds little and risks confusing readers about what the method actually does.
- **No ablation isolating the contribution of rationales.** The STaR-inspired rationales are a highlighted novelty, but the paper never tests whether fine-tuning on incorrect answers *with* rationales outperforms fine-tuning on incorrect answers *without* them.

### Trivial
None.

## Nice-to-Haves
- Quantitative fine-grained evaluation: measure and report retention rates for related facts about the target entity when unlearning a specific association, with baseline comparisons.
- Failure-case analysis: include examples where UNSTAR over-forgets related information or fails to unlearn the target, to complement the single positive example in Table 4.
- Raw absolute metrics with standard deviations, rather than normalized scores alone.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Circular training invalidating all efficacy claims:** The critic claims the *entire* evaluation is circular because the model is trained on synthetic paraphrases until it fails them. This is incorrect for the main WPU evaluation (Figure 2/table), which tests on the original, human-written forget-set QA pairs. The circularity applies only to Figure 3's convergence plot, which is a minor issue.
- **Anti-samples are just negative supervision:** While largely correct that anti-samples overlap with existing techniques, this is a matter of framing rather than a fatal flaw. The specific mechanism of paraphrase + incorrect answer + rationale does have inventive elements.
- **Claim that fine-grained retention mechanism is unexplained:** The paper does attribute retention to the retain-set fine-tuning and semantic filtering in Section 3; while deeper analysis would be nice, absence of mechanism analysis is not a major flaw per se.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Fix the text-table contradictions in Figure 2 immediately. The text and table must report the same numbers.
- Add quantitative metrics for fine-grained unlearning: when unlearning Harry Potter→Hogwarts, measure what fraction of related Harry Potter and Hogwarts questions are still answered correctly, and compare against baselines on the same task.
- Run an ablation that trains on incorrect answers without generated rationales to isolate whether the STaR-inspired component actually improves unlearning.

## Score and Decision

**Calibration comparison:**
- **High anchor (8.00):** `/home/wg25r/review_agent/human_reviews/51WraMid8K.md` — Probabilistic evaluation framework for LLM unlearning with novel metrics, strong theory, and comprehensive experiments. UNSTAR is far below this in terms of theoretical rigor, metric design, and empirical reliability.
- **Medium-high anchor (6.00):** `/home/wg25r/review_agent/human_reviews/huo8MqVH6t.md` — Gradient-ascent analysis with G-effect metric; clean, focused contribution. UNSTAR has a more creative method but far worse reporting consistency.
- **Medium anchor (5.25):** `/home/wg25r/review_agent/human_reviews/TLBPjECC5D.md` — Zero-shot unlearning claim, strong but slightly overstated. UNSTAR's overclaiming on fine-grained unlearning is more severe because it is purely anecdotal.
- **Low anchor (4.50):** `/home/wg25r/review_agent/human_reviews/liqUhMECuY.md` — TMU, an unlearning method with weak presentation and logical issues but consistent tables. UNSTAR's method is more interesting, but its text-table contradictions are a more serious empirical integrity problem.
- **Low anchor (3.00):** `/home/wg25r/review_agent/human_reviews/hwXUmwJAq5.md` — UGradSL with fundamentally flawed evaluation methodology. UNSTAR does not have conceptual misunderstandings this deep, and its core method is more creative.

**Positioning:** UNSTAR sits between the low anchors (3–4.5) and the medium anchors (5–6). While the anti-sample idea with rationales is genuinely creative, the text-table contradictions in the main results are a severe empirical integrity issue, and the fine-grained claim is entirely anecdotal. These issues are more damaging than the presentation problems in the 4.5 anchor (TMU) and more serious than the overstated claims in the 5.25 anchor. The paper is below the threshold for acceptance in its current form.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>