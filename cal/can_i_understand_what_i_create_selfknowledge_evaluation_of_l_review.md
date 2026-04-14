=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
Now I have enough to write the consolidated review. Let me synthesize carefully.

---

## Summary

This paper introduces a "First Generate, Then Evaluate" self-knowledge evaluation framework for LLMs and LMMs, inspired by the inverse of Feynman's principle: can a model accurately verify properties of content it itself generated? The authors evaluate seven LLMs and two LMMs across nine tasks (word counting, math, code, theorem proving, factual recall, ArXiv ID retrieval, grammar, SQL-type operations), finding consistent and often severe self-knowledge gaps. Additional analyses examine attention-mechanism correlates of self-knowledge performance, in-context vs. out-of-context evaluation, and whether fine-tuning on self-generated data improves downstream GSM-8k performance.

---

## Strengths

- **Genuinely novel evaluation axis.** The "First Generate, Then Evaluate" paradigm cleanly operationalizes a consistency-based notion of self-knowledge that differs from existing instruction-following benchmarks and calibration studies. Framing the verify step as a separate inference call (no shared context) is a deliberate and meaningful design choice that isolates cross-run consistency.

- **Breadth of coverage across task types.** The framework spans counting, symbolic math, code execution, theorem inequalities, factual recall, and multimodal perception — spanning generation modalities that existing benchmarks treat separately. Extending the same framework to LMMs (Section 5) without modification is a concrete generalization that few self-evaluation papers attempt.

- **The in-context ablation (Table 6) is genuinely informative.** By comparing no-context, in-context, and in-context-with-noise conditions, the paper reveals a stark capability split: GPT-4 and Gemma achieve 100% in the in-context setting while Llama2 and Llama3 remain at 0%. This is a surprising and concrete finding — two 7–8B open-source models fail at word-counting even when the paragraph is right in front of them — and this stratification is not self-evident from existing benchmarks.

- **Attention-score correlation (Table 5) offers a falsifiable mechanistic hypothesis.** The observation that the gap between the initial self-knowledge score and the attention-based score is smaller for better-performing models suggests a structured relationship between attention alignment and behavioral consistency. While correlational, it is a specific, quantitative claim that motivates future mechanistic work.

- **Self-improvement via self-generated fine-tuning.** The finding that Llama3 improves GSM-8k accuracy by +3.08% when fine-tuned on its own correctly-labeled math generations — without any external curated data — is the most practically significant result in the paper and points toward data-efficient self-improvement pipelines.

---

## Weaknesses

### Fatal
None. The issues below are serious but collectively recoverable with reframing and additional experiments.

### Major

- **The core metric conflates generation failure with verification failure.** For the total word count task (and analogously for Code, Designate Count), the ground truth `a` is the *requested* property (e.g., 56 words), not the *actual* property of the generated text. If the model generates a 63-word paragraph (generation failure) and then correctly reports 63 words (verification success), the paper scores this as a self-knowledge failure (63 ≠ 56). Conversely, if the model generates exactly 56 words but reports 60, it is also a failure — but a verification failure. The paper never disaggregates these two failure modes. This conflation means the self-knowledge score in Table 1 is uninterpretable as a diagnostic: a score of 0.00 could reflect that all models fail at *generation*, all models fail at *verification*, or some mixture. Without separating Step 0 (did generation succeed?) from Step 1 (conditional on success, does verification agree?), none of the core quantitative claims can be taken at face value. This is the most pressing issue in the paper.

- **The Facts and ArXiv tasks measure hallucination consistency, not self-knowledge.** The paper asks the model to name a celebrity born on a given date (or an arXiv paper from a given month), then asks in a separate run whether the generated name/title was indeed born/published then. Since the model routinely hallucinated, a high consistency score in the Facts task (Table 1: Mistral 0.92, GPT-4 0.71) simply means the model reliably repeats the same hallucination — not that it possesses accurate factual self-knowledge. The ArXiv task (where all models score near 0.00–0.13) could reflect either inconsistent hallucination or rare factual correctness — the paper cannot distinguish these. The paper acknowledges this implicitly ("models usually show good consistency under this test" for Facts) but does not resolve the ambiguity in the metric design.

- **Multiple counterintuitive results are left unexplained and cast doubt on measurement validity.** Llama2-7B-Chat scores 0.88 on Math (Table 1) — higher than any other model — while GPT-4 scores only 0.24. GPT-3.5 scores 0.51 on Code while GPT-4 scores 0.08. In Table 3, Llama2 scores 0.99 on "Add first word" and 0.93 on Grammar, far above GPT-4. These orderings are not just surprising — they are inconsistent with every established capability ranking. Without investigation (e.g., do simpler tasks get generated, do models have systematic biases toward certain output lengths or answer words?), these results suggest the metric may be measuring unintended artifacts rather than self-knowledge. The paper offers no analysis.

- **Fine-tuning claims are overstated relative to the data.** The paper states "all models have improved accuracies when tuning on its own data" — but the data show GPT-3.5 improves by +0.04% (correct) and −0.06% (wrong), Gemma by +0.11%/+0.19%. These are well within GSM-8k's known single-run variance (typically ±0.5–1.5%). Only Llama3's +3.08% is robust. The paper dismisses GPT-3.5's negative result as an "outlier" due to black-box fine-tuning, but this is post-hoc rationalization. The broader claim about self-improvement as a "promising direction" should be supported by statistical tests or multiple runs, or scoped down to Llama3 where the gain is plausible.

### Minor

- **Sample size (n=100) lacks statistical reporting.** For tasks where scores are very close across models, the ±~10% confidence interval for n=100 means many pairwise comparisons are not statistically distinguishable. No confidence intervals, standard deviations, or significance tests are reported anywhere. The paper should at minimum report uncertainty ranges so readers can judge which model differences are real.

- **The attention analysis (Section 6.1) relies on unjustified hyperparameter choices.** The top 15% threshold, the use of only the last layer, and averaging across all heads are choices made without ablation or justification. Table 5 contains only 5 data points (one per open-source model), and the "additive effect" narrative is a post-hoc description of a monotone trend with 5 values. This section's conclusions should be framed as hypotheses, not findings.

- **The stochastic resonance explanation for noise-boosted performance (GPT-3.5, Qwen) is unsupported.** The paper acknowledges this is a conjecture ("We conjecture…"), but the conjecture is borrowed from a neuroscience/signal-processing context without any demonstration that the mechanism applies here. The section should simply acknowledge the result is unexplained.

- **The claim that "model achieves its highest accuracy when tuning on its self-generated content" (Section 6.3) is contradicted by its own Table 7.** For Llama2, tuning on GPT-3.5 correct data yields 25.32%, which equals or exceeds tuning on Llama2's own correct data (24.91%). The text overstates the consistency of this finding.

### Tiny

- The abstract uses hedged language ("may be due to," "may enhance") for findings that the paper presents as concrete results in the body; the abstract should be more specific about what was actually found.

- The section introductions (4.2.1–4.2.7) contain generic motivation text that adds length without information; these could be condensed substantially.

---

## Nice-to-Haves

- **Disaggregate generation vs. verification failures.** Adding a Step 0 check (e.g., for word count: verify the actual word count of the generated text programmatically) would make the metric interpretable and allow the paper to cleanly distinguish "model can't generate content with property P" from "model can't verify property P of content it generated." This is the single most impactful experiment the authors could add.

- **Human baseline on a subset of tasks.** The paper claims humans would score ~100% by construction ("a self-knowledgeable model should receive an accuracy of nearly 100% easily"), but this is asserted, not measured. Even a small human study (e.g., 20 participants on the word count task) would validate the framework's ceiling and the "misalignment with human attention" narrative.

- **Control for task difficulty in math.** If Llama2 generates simpler math questions than GPT-4, the high Llama2 self-knowledge math score could reflect task difficulty matching rather than better self-knowledge. A comparison against a fixed math question set (e.g., GSM-8k problems) as a control would help isolate this confound.

- **Attention analysis across layers and heads.** The current analysis uses only the last layer averaged across heads. Probing across all layers and presenting a heatmap would substantially strengthen (or weaken) the mechanistic claim.

- **Metric correlation with established capability benchmarks.** Computing Spearman rank correlation between the per-model average self-knowledge score (Table 1) and their MMLU/GSM-8k performance would test whether self-knowledge actually predicts broader capability — or reveals what the metric truly captures.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"The Feynman framing is philosophically misleading" (Critic):** This is a style/framing critique. The paper is internally consistent about its interpretation of the principle as "can I verify what I created," and applying the inverse of a principle as a research framing is a legitimate rhetorical device. Removed as a framing nitpick.

- **"In-context evaluation is expected by construction and tells us little" (Critic):** Incorrect. The in-context ablation is precisely the right control: it reveals the maximum performance ceiling when memory is not an issue, distinguishing a "memory/access" failure mode from a "reasoning/counting" failure mode. The result that Llama2/Llama3 fail even with in-context access is informative, not trivial.

- **"The claim of novelty is not fully substantiated given self-consistency, calibration, and metacognition literature":** Per review instructions, missing related works cannot be cited as weaknesses when external sources cannot be confirmed. Removed.

- **"The Dual-Generating Strategy is not well-motivated" (Critic):** The paper clearly frames this as an alternative evaluation path when the original answer `a` is unavailable (Section 4.3). The motivation is present and reasonable. Removed.

- **"Confidence intervals for large-scale benchmarks" demand:** GSM-8k evaluation at standard single-run with 1319 test samples (the full test set) is standard practice in the field; requiring statistical tests is not standard for this community. However, for the self-knowledge tasks with n=100 and borderline differences, this requirement is appropriate and retained in weaknesses.

- **"The consistency-based formulation (Eq. 3) is unjustified because the model doesn't know sentence reordering preserves preposition count" (Critic):** This misunderstands the design. Equation 3 doesn't require the model to *know* the transformation preserves the count — it requires that the transformation objectively preserves it (which sentence reordering does). The model's consistency across the original and transformed input is what's being measured. Removed as a misreading.

---

## Novel Insights

The most genuinely novel observation is the radical capability split revealed by the in-context ablation (Table 6): Llama2-7B-Chat and Llama3-8B-Instruct score exactly 0.00 on in-context word-counting even when the paragraph is present in the same context window, while GPT-4 achieves 1.00. This is not merely a "GPT-4 is better" finding — it suggests that certain open-source models of this generation lack a fundamental low-level text-processing capability (counting words in a presented paragraph) that is unrelated to language modeling quality broadly defined, and that this failure is categorical rather than graded. The attention correlation analysis, while methodologically thin, raises an interesting hypothesis: that behavioral self-knowledge consistency may be a downstream signal of how well a model's internal attention computation tracks relevant tokens during generation — a hypothesis that, if validated, would connect external behavioral metrics to internal mechanistic structure.

---

## Suggestions

1. **Implement a programmatic generation-success check.** For counting tasks (Total Count, Designate Count, Code execution), use an external verifier (simple tokenizer for word count, Python interpreter for code) to determine whether the generation step actually succeeded. Report self-knowledge scores conditional on generation success — this is the metric that actually captures verification ability.

2. **Restructure the Facts/ArXiv tasks to include a ground-truth oracle.** For Facts, check whether the named celebrity actually was born on the given date using a lookup table; for ArXiv, check whether the generated paper ID exists. This separates "consistent hallucination" from "accurate self-knowledge" and makes these tasks informative.

3. **Investigate the anomalous model rankings.** The Llama2 math and GPT-3.5/GPT-4 code inversions are the most important empirical findings in the paper — because if they reflect artifacts (e.g., Llama2 generates trivial math questions), they undermine the metric's validity; if they reflect genuine capability dissociations, they are the most interesting result in the paper. Either way, they demand investigation.

4. **Narrow the fine-tuning claims** to Llama3 (where +3.08% is potentially robust) and present GPT-3.5 and Gemma results as null or suggestive. Run Llama3 experiments with multiple random seeds to confirm the gain is reliable.

5. **Report confidence intervals** for all n=100 tasks, at minimum as footnotes or error bars in tables.

---

**Evaluation on key axes:**

- **Novelty:** Moderate. The specific "generate-then-verify across separate inference calls" framing for self-knowledge evaluation is genuinely new, and the multimodal extension adds value. The overall idea of consistency checking is not new.
- **Technical soundness:** Weak-to-moderate. The metric design has a real and impactful conflation flaw (generation vs. verification failure), and the attention analysis is ad hoc. Some tasks are more cleanly designed (math, code) than others (total word count, Facts, ArXiv).
- **Empirical support:** Weak. 100 samples per task with no statistical reporting, several striking results left unexplained, and fine-tuning gains that are mostly within noise margins.
- **Significance:** Moderate potential. If the metric conflation and counterintuitive orderings are resolved, the framework could become a useful lightweight diagnostic for LLM introspection. As currently presented, the framework's diagnostic validity is unclear.
- **Clarity:** Moderate. The framework is clearly described and the paper is readable, but the repeated generic motivational paragraphs per task and several inconsistencies between text and tables (Section 6.3 vs. Table 7) reduce precision.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 1.0, 3.0]
Average score: 2.5
Binary outcome: Reject
