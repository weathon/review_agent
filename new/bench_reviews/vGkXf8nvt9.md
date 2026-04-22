## Summary

The paper introduces Forget-to-Focus (F2F), a two-stage protocol that first applies machine unlearning on a "forget set" of general-domain data (with an optional domain-specific "retain set" for stability via gradient descent), then fine-tunes on domain-specific data. The central claim is that removing irrelevant pretraining knowledge before adaptation improves domain specialization. Experiments span five LLMs (0.6B–72B parameters) across coding, medical, and mathematical domains, showing that GA+GD unlearning followed by SFT consistently outperforms standard fine-tuning, DAPT, LoRA, and CurlLoRA.

## Strengths

- **Creative reframing of unlearning**: Repurposing machine unlearning from a privacy tool to a domain adaptation intervention is a novel and well-motivated idea. The paper identifies a real problem (negative transfer from irrelevant pretraining knowledge) and proposes a principled, modular solution.

- **Extensive and convincing empirical gains for GA+GD+SFT**: Across all 5 models and 3 domains, GA+GD+SFT consistently outperforms standard SFT (e.g., Qwen 0.6B HumanEval: 31.71→42.07; LLaMA 8B HumanEval: 56.71→60.37; Qwen 72B HumanEval: 71.12→78.50). The scale of experiments (0.6B to 72B) and diversity of domains is a significant strength.

- **DAPT comparison strengthens the "unlearning helps beyond just more training" argument**: F2F consistently outperforms DAPT (another two-stage approach that continues pretraining on domain data). For Qwen 72B HumanEval, F2F achieves 78.50 vs. DAPT's 72.50, suggesting the unlearning mechanism contributes beyond simply adding more domain exposure.

- **GA-only ablation provides partial causal evidence**: On coding tasks, GA-only (σ=0, no retain set) + SFT already outperforms SFT alone (Qwen 0.6B HumanEval: 40.02 vs. 31.71), demonstrating that gradient ascent on the forget set contributes to improvements independently of the retain set.

- **Informative forget set quality ablation**: The comparison of BC-Select vs. BC-Mixed vs. BC-Cosine (Table 3) reveals important sensitivity to forget set composition, with BC-Select consistently best and BC-Mixed (contaminated with domain data) performing worst, providing practical guidance.

- **Multiple unlearning algorithms tested**: Figure 3 compares GA+GD, GA-only, GA+KL, and NPO, showing the protocol generalizes beyond a single unlearning method.

- **Mechanistic analysis via CKA/SVCCA**: The representation geometry analysis (Section 4.5) shows F2F pushes representations further from both the base and unlearned models compared to standard fine-tuning, supporting the claimed mechanism of reducing negative transfer.

## Weaknesses

### Fatal
None.

### Major

- **The retain set confounds the core causal claim for the best-performing variant**: Section 3.3 states "The retain set is a small subset of the fine-tuning data," meaning the GA+GD variant gives the model additional supervised exposure to domain-specific data during the unlearning phase. The strongest results come from this confounded variant. The cleanest test—GA-only (σ=0, no retain set)—shows inconsistent benefits: it helps coding (Qwen 0.6B HumanEval: 40.02 vs. SFT 31.71) but hurts medical QA (Qwen 0.6B PubMedQA: 58.80 vs. baseline tuning 62.60). The critical missing control is replacing the retain set with non-domain general data to disentangle unlearning effects from domain pre-exposure. The DAPT comparison partially mitigates this concern (DAPT gives more domain exposure yet underperforms F2F), but does not fully resolve it since DAPT and F2F use different data and compute budgets. Without this control, the paper cannot definitively attribute GA+GD's consistent improvements to unlearning rather than domain data pre-exposure.

- **Calibration claim in the abstract is unsupported by evidence in the main text**: The abstract states F2F "helps improved calibration on medical QA tasks, reducing overconfidence and mitigating reliability issues that persist under standard fine-tuning," and the conclusion claims "improves calibration on sensitive QA." However, no calibration metric (ECE, Brier score, reliability diagrams) appears anywhere in the main paper. A claim this specific and prominent, appearing in both abstract and conclusion, requires corresponding evidence in the main text.

### Minor

- **"Consistently outperforms" should be qualified**: The abstract claims F2F "consistently outperforms standard fine-tuning," but this is specifically true for GA+GD+SFT. GA-only+SFT does not consistently outperform SFT (e.g., Qwen 0.6B PubMedQA: 58.80 vs. 62.60). The claim should be more precise about which variant consistently outperforms.

- **Theoretical analysis has limited explanatory power for the empirical results**: The Proposition assumes strong convexity, β-smoothness, and an orthogonal V⊕U decomposition—none of which hold for billion-parameter LLMs. While the paper acknowledges using "a convex linear surrogate to clarify the mechanism," it still draws firm conclusions like "increasing the forget to retain ratio λ/σ tightens the starting distance for fine-tuning" that rely on these violated assumptions. The theory provides intuition but should not be presented as supporting firm predictions.

- **Number of unlearning steps (Tu) not reported**: The hyperparameter section specifies learning rates, batch sizes, and weights but does not state how many unlearning steps were performed for each model, making it difficult to assess the compute overhead of the unlearning phase.

### Trivial
None.

## Nice-to-Haves

- A control experiment replacing the retain set with non-domain data (e.g., a general BookCorpus subset) would definitively resolve the retain set confound and significantly strengthen the paper.
- A perturbation analysis comparing GA on forget data vs. random noise of equal magnitude would clarify the mechanism behind GA-only improvements.
- Reporting calibration metrics (ECE, Brier score) alongside accuracy for medical QA would substantiate the calibration claim.
- Reporting Tu and total FLOPs for each method would clarify the computational cost-benefit tradeoff.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "Compute and data budget are uncontrolled" as a structural issue**: Per rules, undisclosed hyperparameters and trivial implementation details are nitpicks. The DAPT comparison also adds compute before fine-tuning, serving as a partial compute control. Downgraded from structural to minor.

- **Harsh critic: "Statistical significance and variance entirely absent" as evidential**: While true, single-run evaluation is the norm for large-scale LLM papers. Downgraded from evidential to minor, as this is a field-standard practice rather than a specific flaw of this paper.

- **Harsh critic: "Forget sets are too small to meaningfully remove knowledge"**: This is an alternative hypothesis about mechanism, not a flaw. The paper shows empirically that the protocol works regardless of the precise mechanism. Whether gradient ascent on 100 samples truly "removes knowledge" or acts as regularization is an interesting question but does not invalidate the empirical results.

- **Strength finder: "Improved calibration on safety-critical tasks"**: Conflicts with the verified major weakness that no calibration metrics appear in the main text. Dropped.

- **Harsh critic: "QLoRA weights unclear for Qwen 72B"**: This is an implementation detail that falls under the reproducibility nitpick rule.

- **Harsh critic: Gemma 2B's 0.00 performance "confusingly framed"**: The paper actually explains this clearly in Section 4.1, noting that "the rows with large drops correspond to intermediate unlearning checkpoints rather than the final tuned models." This is a misreading by the harsh critic.

## Novel Insights

The GA-only vs. GA+GD split reveals an important tension at the heart of this work: pure unlearning (GA-only) is insufficient and sometimes harmful for domain adaptation, while the best results come from a variant (GA+GD) that combines unlearning with domain data pre-exposure. This suggests the mechanism may not be "removing irrelevant knowledge creates capacity" as claimed, but rather "perturbing the model away from generalist features while simultaneously anchoring it to domain features" — a subtler and more nuanced mechanism than the paper articulates. The forget set quality ablation (BC-Select vs. BC-Mixed) further supports this: when the forget set accidentally includes domain data, performance degrades, suggesting the forget set's role may be to provide a directional perturbation away from domain-irrelevant features rather than to literally "remove" knowledge.

## Suggestions

- Add a single control experiment: run GA+GD with a retain set of non-domain general text instead of domain fine-tuning data. If performance remains similar to GA+GD with the domain retain set, the stability-preserving role of the retain set is confirmed; if it drops, the domain pre-exposure confound is confirmed. Either way, this resolves the central ambiguity.
- Remove or substantially soften the calibration claim from the abstract and conclusion until calibration metrics are provided in the main text.
- Qualify "consistently outperforms" to specify that this holds for GA+GD+SFT specifically, and note that GA-only+SFT shows domain-dependent results.

## Score and Decision

**Calibration anchors used:**

1. `/home/wg25r/review_agent/human_reviews/gc8QAQfXv6.md` — Function vectors for catastrophic forgetting (avg 9.0, Oral). Much stronger theoretical grounding and clean mechanism. Our paper is well below this.

2. `/home/wg25r/review_agent/human_reviews/51WraMid8K.md` — Probabilistic unlearning evaluation (avg 8.0, Oral). Novel evaluation methodology with strong formal grounding. Our paper is below this due to the confound issue.

3. `/home/wg25r/review_agent/human_reviews/KzSGJy1PIf.md` — SURE: selective unlearning via representation erasure (avg 5.67, Poster). Similar unlearning topic with gradient reversal, accepted at poster level despite scalability concerns. Our paper has more extensive LLM experiments but also has the retain set confound.

4. `/home/wg25r/review_agent/human_reviews/fr7cLDfNNU.md` — Interleaved ensemble unlearning for backdoor defense (avg 5.5, Withdrawn). Two-stage unlearning+fine-tuning, fairness concerns about comparison. Our paper is comparable but with the confound issue.

5. `/home/wg25r/review_agent/human_reviews/0OB3RVmTXE.md` — Concept resurgence in diffusion models (avg 4.0, Withdrawn). Identified important phenomenon but lacked mitigation and used only one baseline. Our paper has stronger empirical scope.

6. `/home/wg25r/review_agent/human_reviews/hwXUmwJAq5.md` — UGradSL unlearning (avg 3.0, Reject). Fundamentally incorrect evaluation methodology. Our paper is clearly above this.

7. `/home/wg25r/review_agent/human_reviews/nNQmZGjEVe.md` — CalibRAG (avg 4.25, Reject). Overclaimed calibration in abstract without proper metrics. Similar pattern to our calibration overclaim.

The paper sits above the clearly-flawed unlearning papers (avg 3-4) due to its extensive experiments and creative idea, but below the strong Oral/Spotlight papers (avg 7-9) due to the retain set confound and unsupported calibration claim. It is comparable to the SURE paper (5.67, Poster) and the interleaved unlearning paper (5.5, Withdrawn), but with a more extensive experimental scale. The retain set confound is a genuine concern that limits confidence in the core claim, but the paper provides partial mitigations (GA-only ablation, DAPT comparison) and the empirical gains are substantial.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>