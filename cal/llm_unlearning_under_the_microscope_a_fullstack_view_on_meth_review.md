=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
This paper offers a broad empirical re-examination of LLM unlearning rather than a new unlearning algorithm. It organizes 12 stateful unlearning methods into three families, argues that standard MCQ-based evaluation on WMDP is incomplete, proposes an additional “Open-QA” evaluation based on entailment scoring, and studies robustness under several post-unlearning attacks (relearning, downstream fine-tuning, quantization, and jailbreaks). The central value of the paper is its attempt to shift the field from narrow answer-selection metrics toward a fuller behavioral view of unlearning.

## Strengths
- The paper surfaces a specific and important failure mode of current evaluation: MCQ-based unlearning success can hide severe generative degradation. This is not just asserted; the paper supports it with concrete examples and diagnostics. In Appendix B, Table A1 shows NPO/RMU changing the selected MCQ answer while producing incoherent generations, and Fig. A1 links this to altered logit structure.
- The comparison across 12 methods under a unified lens is genuinely useful because it does more than tabulate scores: it proposes a method-family-level framing—divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning—and uses that framing to interpret UE/UT and robustness tradeoffs.
- The robustness section is broader than most single-attack evaluations. The paper separately studies in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreaks, and the distinction between relearning-style and downstream-adaptation-style robustness is a meaningful conceptual contribution even if the evidence is still limited.
- The paper contains a few insightful mechanistic probes rather than only benchmark plots. In particular, the logit analysis for divergence-driven methods and the loss-landscape comparison between TAR and RMU+LAT provide plausible explanations for some observed behaviors.
- The discussion of rejection-based methods is more nuanced than the usual “they underperform on WMDP MCQ” narrative. The paper identifies that DPO-like rejection methods can look much better under its additional generative-oriented utility view than under MMLU-style MCQ alone, which is a useful corrective.

## Weaknesses

### Fatal
- **The paper’s headline “Open-QA” UE metric does not actually evaluate open-ended generation in the way the paper claims.** This is the most serious issue because it weakens the central methodological contribution. Section 4 frames Open-QA as moving “beyond MCQ” and as evaluating “free-form responses,” but Appendix A states: “Before generating answers for ES evaluation, we add a few-shot prompt consisting of 2 demonstration examples... The purpose is solely to ensure that the model outputs remain restricted to the given options (A–D).” That means the proposed UEOpen-QA on WMDP is not truly open-ended QA; it is still constrained to MCQ-style outputs, just scored through generation plus NLI rather than argmax over choices. This does not fully invalidate the observation that generation behavior can differ from MCQ selection, but it substantially overstates the claim that the paper has introduced a genuinely open-ended alternative for forget-set evaluation.

### Major:
- **The robustness analysis after relearning/fine-tuning is incomplete because it tracks mostly post-attack UE, not the corresponding post-attack utility.** The paper itself correctly notes in the quantization subsection that robustness should be interpreted through the full UE–UT tradeoff, since degraded or restored capability can create a false picture of robustness. However, for in-domain relearning and out-of-domain fine-tuning in Section 5 / Fig. 2, the reported analysis focuses on post-attack UE (UEMCQ and UEOpen-QA) without the matching post-attack UT. This makes it hard to tell whether a rise in forget-set performance reflects true recovery of forgotten knowledge, generic capability restoration, or other broad changes in the model. Since several family-level conclusions are drawn from these comparisons, the missing utility controls materially limit interpretability.
- **The empirical basis is too narrow for several of the paper’s broad family-level conclusions.** Most substantive claims are based on one model family/size (Llama-3 8B Instruct) and primarily one unlearning benchmark/domain (WMDP-Bio), with only limited supplementary MUSE evidence in the appendix. The paper repeatedly generalizes at the method-family level—for example, that divergence-driven methods are more robust to in-domain relearning while representation methods are more robust to out-of-domain fine-tuning—but with this scope those claims should be presented as observations in this setting rather than general properties of the families.
- **The entailment-score-based evaluation is under-validated given how central it is to the paper.** The paper uses ES as the main new measurement lens for UEOpen-QA and discusses it as exposing failures hidden by MCQ, but there is no human validation or calibration study showing that ES tracks the intended notions of knowledge retention, refusal, or harmful answer generation on this task. This matters especially because rejection-style outputs, evasive outputs, and malformed outputs may interact with NLI scoring in non-obvious ways.
- **The attack-strength choices for robustness are fixed rather than stress-tested.** For relearning and out-of-domain fine-tuning, the paper uses specific step counts, learning rates, and task mixtures (Appendix A), but robustness rankings may depend substantially on attack strength. Since robustness is a central axis of comparison, some sensitivity analysis over attack budget would be important to establish that the rankings are not artifacts of a single attack configuration.

### Minor
- **The taxonomy is useful but not especially deep as a research contribution on its own.** The grouping is coherent and serviceable, but it is primarily descriptive; it does not establish sharp formal boundaries or a stronger theoretical account of why these families should behave differently. The paper’s strongest contribution is the evaluation critique, not the taxonomy itself.
- **Some explanatory claims in the robustness section are too causal for the evidence shown.** For example, the paper says RobJA aligns more closely with RobReL because both are “worst-case adversarial testing.” The figure supports correlation, but not this mechanism specifically; alternative shared causes are also plausible.
- **The paper leaves some benchmark/objective mismatch issues underexplained for rejection-based methods.** The authors rightly note that MCQ may understate rejection-based methods, but the paper could more directly discuss how a benchmark whose nominal “correct answer” is hazardous factual content may misalign with methods optimized to refuse, and how the proposed ES handles refusals.
- **There is a clear drafting issue in the introduction (“To tackle (Q)”) that suggests the key research question placeholder was not fully resolved.** This is not substantive, but it does affect polish and precision of presentation.

### Trivial
- None.

## Nice-to-Haves
- Add a small human evaluation or adjudicated sample to validate that ES agrees with human judgments about whether a response truly reveals forgotten knowledge, safely refuses, or is simply nonsensical.
- Report post-attack UT alongside post-attack UE for relearning and downstream fine-tuning, mirroring the paper’s own argument in the quantization section that robustness should be assessed on the UE–UT tradeoff.
- Evaluate at least one additional model scale or architecture and one additional domain/benchmark to support family-level generalization claims.
- Add attack-budget sensitivity analyses for relearning and fine-tuning (steps, learning rate, data volume).
- Clarify explicitly how refusals are scored under ES and whether this structurally advantages or disadvantages rejection-based methods.
- Include a few more side-by-side generations across families showing coherent refusal, successful forgetting, and generative collapse.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Code/model release is not explicit, so reproducibility is questionable.”** Removed under the instruction to avoid reproducibility criticisms rooted in release status or missing large artifacts.
- **Formatting/parser complaints about garbled equations/tables.** Removed because the user explicitly noted formatting artifacts from PDF extraction, and style/format nitpicks should not be counted against the paper.
- **Claims that comparisons are unfair because some robustness baselines/related methods are omitted.** Removed because missing related-work/baseline complaints cannot be reliably verified here, and the paper already includes 12 methods; this is not a well-grounded core criticism from the text alone.
- **Strong claim that Table A1’s degenerate outputs might just be prompt-format artifacts rather than optimization pathology.** Weakened/removed as a main weakness because the paper provides supporting evidence from logits (Fig. A1), so this is not just an unsupported anecdote.
- **Generic requests for confidence intervals/significance tests.** Moved out; while useful, this is not standard enough here to be a core flaw relative to the paper’s main claims.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that the work is strongest when it is read as exposing a mismatch between *selection-style* and *behavior-style* evaluation, but weaker when it over-claims that its replacement metric fully resolves that mismatch. The paper compellingly shows that MCQ can hide collapse, refusal, or malformed generation; however, its own WMDP “Open-QA” UE setup still preserves the multiple-choice scaffold by forcing A–D-form outputs. So the true contribution is less “we replace MCQ with open-ended evaluation” and more “we reveal that decoding behavior and answer selection can diverge, and evaluation should inspect both.” A second useful insight is that the paper itself implicitly applies a stronger standard to quantization robustness (joint UE–UT analysis) than to relearning/fine-tuning robustness; bringing that same standard across all attacks would materially strengthen the work and likely sharpen, or revise, some of the method-family conclusions.

## Suggestions
- Rework the central evaluation claim to accurately describe UEOpen-QA as a generation-based but still option-constrained evaluation on WMDP, unless the authors can add a truly open-ended forget-set evaluation without A–D restrictions.
- Add a validation study for ES: e.g., human labels on a representative subset, or comparison against a stricter factuality/refusal annotation protocol.
- For every robustness attack in Section 5, report both post-attack UE and post-attack UT. This is the most important experimental fix.
- Temper family-level language throughout the paper to reflect the actual empirical scope, or add experiments on another model/domain to justify stronger generalization.
- Add sensitivity analyses over attack strength for relearning and downstream fine-tuning.
- Clarify, with examples, how ES treats refusals, evasions, and malformed generations, especially for rejection-based methods.
- Expand the mechanistic analysis across all three families, not just selected methods, if the goal is to claim actionable design principles rather than descriptive benchmarking.



# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
