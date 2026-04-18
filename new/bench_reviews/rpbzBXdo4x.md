Now I have enough context from calibration papers. Let me synthesize my final review.

Key calibration papers:
- "To CoT or not to CoT?" (Accept Poster, scores 6/8/6) - most directly comparable, meta-analysis showing CoT helps mainly on math/symbolic reasoning. That paper was accepted despite similar questions about generalizability across models.
- "Visual LLMs Exhibit Human-Level Cognitive Flexibility" (Withdrawn/Reject, scores 5/3/3) - rejected for overstating human-model parallels, no validation, poor methodology.
- "ReCogLab" (Accept Poster, scores 6/3/6/5) - accepted for framework + careful evaluation but had weaknesses in task selection justification.
- "Quantifying AI Psychology" (Reject, scores 5/5/3/8) - rejected for overstatement and weak methodology.

This paper is substantially stronger than the rejected psychology-to-LLM papers. It has:
1. Real, large, and practically significant empirical findings
2. A principled framework (psychology-based heuristic)
3. Honest reporting of cases where the heuristic doesn't work
4. Multiple models tested
5. Novel task construction

But it also has real weaknesses:
- The o1-preview vs GPT-4o headline comparison is confounded
- Lack of mechanistic analysis (what does CoT actually cause models to do?)
- The causal claim about "thinking" vs prompt engineering is not well isolated
- Overgeneralization from the cross-model patterns that are actually heterogeneous

Let me write the final review.</think>## Summary

This paper investigates when chain-of-thought (CoT) prompting systematically reduces LLM/LMM performance by drawing on cognitive psychology: tasks where verbal thinking or deliberation hurts human performance may also hurt models. The authors identify six such task categories from the psychology literature, adapt them for model evaluation, and find that three—implicit statistical learning (artificial grammar), face recognition (verbal overshadowing), and classification with pattern exceptions—show significant CoT-induced performance drops across multiple models (e.g., up to 36.3% absolute for o1-preview vs. GPT-4o zero-shot, 331% more learning rounds for exceptions). The other three tasks (logical inconsistency, spatial intuition, feature aggregation) do not show the predicted effect, which the authors attribute to meaningful differences between human and model capabilities.

## Strengths

- **Novel and practically important research question.** Identifying when CoT harms rather than helps is urgent given the default deployment of inference-time reasoning in frontier models. The paper provides concrete, large-magnitude failure cases that practitioners should know about.

- **Principled hypothesis generation.** Using well-established cognitive psychology findings to systematically predict CoT failure modes is a creative and intellectually rich approach that goes beyond ad-hoc task exploration. The paper generates falsifiable predictions and honestly reports cases where the heuristic does not hold.

- **Substantial empirical scope.** Nine+ models (open- and closed-source) tested across six carefully constructed tasks, with large-scale procedurally generated datasets (4400 grammar problems, 500 face-recognition problems, 2400 vehicles, etc.). The effort in scaling psychological paradigms to model evaluation is itself a contribution.

- **Large and consistent negative CoT effects on three tasks.** The drops are not marginal: 23.1% absolute for GPT-4o on ISL, 8.80% for Llama 3.1 70B, 14.40% absolute for Claude 3 Opus on face recognition, and 331% more learning rounds for GPT-4o on CDE. These are practically significant findings with clear deployment implications.

- **Thoughtful analysis of mismatches.** The paper's honest treatment of the three tasks where the heuristic fails—attributing these to specific human-model capability differences—is analytically careful and increases intellectual credibility.

## Weaknesses

### Fatal
None.

### Major

- **The headline 36.3% comparison (o1-preview vs. GPT-4o zero-shot) conflates model capability differences with CoT effects.** The paper's most prominent figure compares two different models (o1-preview has built-in CoT; GPT-4o does not). Architecture, training data, and capacity differences all contribute to this gap. The within-model GPT-4o comparison (87.50% → 64.40%, a 23.1% drop) is still large and meaningful, but the abstract and introduction foreground the cross-model number. This inflates the perceived impact and conflates two distinct factors. The authors do present both, but the framing prioritizes the confounded comparison.

- **Lack of mechanistic analysis of why CoT hurts.** Across all three negative-effect tasks, the paper reports performance differences but does not analyze the actual content of model CoT outputs. Without examining what models write during CoT—do they verbalize incorrect rules? shift from pattern-matching to rule-search? override in-context evidence?—the claim that this mirrors human cognitive failures (rule bias, verbal overshadowing) remains speculative. The paper acknowledges this gap in §5 ("alternative explanation") but does not address it empirically. This limits the contribution from "we identified when CoT hurts" (empirical and valuable) to "we identified why CoT hurts in a way analogous to humans" (not established).

- **CoT is not isolated from broader prompt changes.** Zero-shot and CoT conditions differ in multiple ways beyond "thinking": instruction framing, verbosity, answer format positioning, and the model's distributional exposure to CoT-style prompts during training. The paper does not include controls (e.g., non-reasoning filler text matched to CoT length, instruction-only variants) to disentangle whether the effect is specific to reasoning or is a more general prompt-engineering artifact. The Tree-of-Thought result on ISL (Appendix A.4) is a partial step but only on one task.

### Minor

- **Cross-model heterogeneity is under-analyzed relative to the narrative.** Claude 3.5 Sonnet shows a slight *improvement* on ISL with CoT (−1.8%, non-significant), GPT-4o and Llama 3.1 70B show large drops, Gemini shows moderate ones. The paper's narrative of "consistent decreases" papers over meaningful variation. The Discussion could more explicitly address which model properties (size, training for reasoning, instruction tuning) moderate the CoT-harm effect.

- **Face recognition task validity concerns.** Synthetic faces generated from identical textual descriptions may not adequately test "verbal overshadowing" because the task structure may bias models toward the very verbalizable attributes that overshadowing is supposed to interfere with. The connection to human verbal overshadowing—where faces are hard *because* their distinguishing features resist verbalization—is weakened when stimuli are generated from verbal descriptions. Some models also perform below chance (answering "all same person"), making percentage-point drops in this regime harder to interpret.

- **No trivial algorithmic baselines for the grammar task.** The ISL task gives models 15 training examples and asks classification. It is unclear whether zero-shot GPT-4o is doing anything like "implicit statistical learning" or simply performing string similarity matching. A simple nearest-neighbor or Levenshtein-distance baseline would contextualize whether high zero-shot performance reflects genuine grammar learning.

### Trivial

- The paper uses p-values with very large sample sizes (4400 items), making almost any difference "significant." The authors appropriately emphasize effect sizes, but some p-values are reported to four decimal places (e.g., p < 0.0001) which overstates precision.

## Nice-to-Haves

- Error analysis of CoT traces on the three negative-effect tasks (what do models actually write, and where do they go wrong?)
- Length-matched or non-reasoning filler controls to isolate the "thinking" component from verbosity/format effects
- Few-shot CoT or alternative CoT prompt phrasings to test robustness of the negative effect across prompting strategies
- Confidence intervals per task/model rather than just point estimates and p-values

## Removed Points

These points were flagged for removal or significant weakening:

- **Data contamination / memorization concern (from Human Finder):** The claim that psychology task structures might be in the paper's training data is speculative. The paper carefully constructs novel procedural stimuli (random FSGs, synthetic faces, generated vehicle lists) rather than using well-known psychology stimuli verbatim. Removed as unsupported.

- **Multiple comparisons correction (from Spark):** While always good practice, this is a standard methodological nicety rather than a core flaw. The effect sizes are large enough that they would survive correction. Moved to Nice-to-Have.

- **Missing related works (from Human Finder):** Flagged as not verifiable—I cannot confirm existence of specific papers not cited by the authors.

- **Llama 3.1 70B apartment task anomaly (from Spark):** The near-collapse (42%→6%) is indeed striking, but the paper does note this was because the model "was often unable to return an answer after deliberating in the CoT condition," which is a different phenomenon from reasoning-induced degradation. However, this is already discussed in the paper and does not invalidate the overall findings for other models. Kept as a mention in minor weaknesses but not elevated.

- **"Not yet released" / availability doubts about models or datasets:** Per instructions, all cited models and datasets are assumed to exist.

- **Formatting/style nitpicks (from Spark):** Removed per instructions.

- **Reproducibility concerns about hyperparameters (from Spark):** Removed per instructions as trivial implementation details.

## Novel Insights

The most insightful finding is the partial success of the psychology heuristic: it works for three task types where language and generalization bias create shared vulnerabilities for humans and models (implicit learning, verbal-overshadowing, exception-laden rules), but fails for three others where human-specific limitations (working memory constraints, lack of motor simulation) don't apply to models. This asymmetry is itself informative: it suggests that CoT-harm transfers from humans to models *specifically* when the bottleneck is a shared representational limitation (language's inability to encode certain information), not when the bottleneck is a shared processing limitation. This distinction—representational vs. processing bottlenecks—is not made explicit in the paper but could guide future work on predicting CoT failures.

## Suggestions

- **Reframe the headline finding.** Lead with the within-model GPT-4o comparison (23.1% drop) rather than the confounded o1 vs. GPT-4o comparison, and report the cross-model comparison separately. The within-model result is still strong.
- **Add CoT content analysis.** Even a qualitative analysis of 20-50 CoT traces per negative-effect task would substantially strengthen the mechanistic claim and connect the results more concretely to the psychology literature.
- **Test at least one alternative CoT prompt per task** to assess whether the effect is robust to prompt wording, or specific to the particular instructions used.

## Score and Decision

**Calibration comparison:**

- "To CoT or not to CoT" (Accept Poster, scores 6/8/6): Similar topic, broad empirical scope, but more comprehensive model-wise analysis and cleaner causal framing. That paper had a clearer contribution (meta-analysis + controlled experiments). The current paper has a more creative framing (psychology heuristic) but weaker causal attribution.
- "ReCogLab" (Accept Poster, scores 6/3/6/5): Similar approach of psychology-inspired evaluation framework. Weaker reviewer consensus but accepted for novelty and framework value.
- "Visual LLMs Cognitive Flexibility" (Withdrawn/Reject, 5/3/3): Overclaimed human-model parallels without validation—this is a pitfall the current paper largely avoids by honestly reporting mismatches.
- "Quantifying AI Psychology" (Reject, 5/5/3/8): Weak methodology and overclaiming—substantially worse than the current paper.

The current paper is stronger than the rejected psychology-LLM papers because it has real, large, and novel empirical findings. But it is weaker than "To CoT or not to CoT" because (1) the causal interpretation is less well-isolated, (2) mechanistic analysis is absent, and (3) the headline comparison is confounded. The paper makes a genuine contribution in identifying task characteristics that predict CoT failure, but the psychology analogy is used more as inspiration than validated theory, and the paper could be clearer about this distinction.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>