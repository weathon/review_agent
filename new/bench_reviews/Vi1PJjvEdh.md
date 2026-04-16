Now I have a good picture of the calibration landscape. Let me compile the final review.

## Summary

This paper introduces a "self-knowledge evaluation framework" for LLMs and LMMs inspired by Feynman's principle ("what I cannot create, I do not understand"). The framework uses a "First Generate, Then Evaluate" pipeline: models generate content meeting specific constraints (e.g., paragraphs with exactly N words), and then, in a separate run, are asked to verify properties of that content. Across 7 LLMs and 2 LMMs on 9 tasks (word counting, designated word counting, facts, arXiv IDs, math, theorem proving, code, grammar, SQL-like operations), the paper finds that models perform poorly, which is interpreted as lacking "self-knowledge." Additional experiments explore in-context evaluation with noise, attention-based analysis, and fine-tuning on self-generated data.

## Strengths

- **Simple, reproducible evaluation recipe**: The "first generate, then evaluate" pattern and content-reuse with invariance transformations (e.g., moving a sentence and checking preposition count consistency) is easy to implement and generates test cases without human annotation. This is a practical diagnostic contribution regardless of the "self-knowledge" framing.
- **Breadth of task design and model coverage**: Testing 7 LLMs and 2 LMMs across 9 diverse synthetic tasks provides a useful empirical catalog of model failure modes under this generate-then-verify paradigm. The SQL-like "add/delete/change" word-index operations and preposition invariance tests (Section 4.4) are particularly creative diagnostics for local symbolic manipulation.
- **In-context vs. no-context evaluation distinction** (Section 6.2): The finding that models achieve near-perfect accuracy when the generation context is provided in-context (GPT-4: 0.03 → 1.00), and that performance degrades with noise, is informative. It helps characterize whether failures stem from inability to reconstruct information or from lack of explicit recall.

## Weaknesses

### Major:

- **Core construct conflation between instruction-following failures and "self-knowledge"**: The central claim is that this framework evaluates models' "self-knowledge"—their ability to understand what they create. However, the primary metric ℐ(a = â) conflates two distinct failure modes: (1) the model fails to follow the generation instruction (e.g., generates 63 words instead of 56), and (2) the model fails to correctly verify a property of its own output (e.g., counts incorrectly). A model that generates 63 words and then correctly reports "63" would be scored as a failure (56 ≠ 63), while a model that generates 63 words and hallucinates the answer as "56" would be scored as correct. The paper never decomposes error types across any task, making it impossible to know what the scores actually measure. This is not just a labeling issue—much of the paper's novelty and interpretive claims ("models lack self-knowledge," "misalignment with human attention mechanisms") depend on establishing that a uniquely "self-referential" ability is being measured, rather than generic counting or instruction-following ability.

- **No comparison between self-generated and externally-generated content**: To justify the framing of "self-knowledge," the paper must show that models perform differently on their own generated content versus externally provided content. Without this control, the results are indistinguishable from the known fact that "LLMs are bad at exact counting and symbolic manipulation of text." All tasks—word counting, keyword frequency, preposition counting, ArXiv ID retrieval—measure capabilities that could be tested on any text, not just self-generated text. This missing experiment directly undermines the paper's central novelty claim.

- **Attention-based "human-like mechanism" analysis is speculative and unsupported** (Section 6.1): The paper defines an ad-hoc "attention-based score" (top 15% of tokens by attention to a keyword, with no justification for the 15% threshold), then extrapolates from small numerical differences (0.04–0.21) to a two-phase "additive effect" narrative about "misalignment with human attention mechanisms" and "less concentration." No human data on attention or task performance is collected or cited. No causal intervention (e.g., attention head ablation) is performed. The conclusion that models performing better on self-knowledge tasks "behave more similarly to humans" is asserted without any measured human baseline. This is speculation presented as an empirical finding in both the abstract and conclusion.

### Minor:

- **Fine-tuning claims overinterpret marginal results** (Section 6.3): GSM-8k improvements range from 0.04% (GPT-3.5) to 3.08% (Llama3), with no error bars, confidence intervals, or statistical significance tests. The claim that "self-improving is a promising direction" based on improvements of this magnitude, across single runs with no control for data difficulty or overlap, is not well supported. Notably, fine-tuning on "wrong" data also improves Llama2 performance (0.80%→1.21%), which the paper does not adequately explain.

- **Thin LMM evaluation**: Only 2 multimodal models tested on 3 basic perception tasks (counting, color, position), with no comparison to performance on externally-sourced images, making it unclear whether the "self-knowledge" framing adds any signal beyond known LMM perception limitations.

- **Missing failure mode analysis across all tasks**: The paper treats all task failures as binary (correct/incorrect) without analyzing whether models fail at generation, verification, or both. For the word-counting tasks specifically, no word-counting convention is specified (punctuation, hyphenated words), creating potential ambiguity. The ArXiv ID task likely involves hallucinated IDs, making this a test of hallucination consistency rather than any form of "knowledge."

- **Overclaim in abstract/conclusion**: The abstract states findings indicate gaps "may be due to misalignment with human attention mechanisms," and the conclusion frames this as established rather than speculative. The paper also claims the framework is "comprehensive" despite testing only synthetic, narrow tasks focused on counting and consistency.

## Nice-to-Haves

- Decomposing generation vs. verification errors would clarify what the metrics actually measure and make the results more actionable.
- Running the verification tasks on externally-sourced text (e.g., human-written paragraphs) would establish whether the "self" aspect adds any distinct signal.
- Causal attention interventions (e.g., masking or amplifying attention to the designated keyword) would move the Section 6.1 analysis from correlational to causal.
- Testing statistical significance of fine-tuning improvements with multiple seeds would strengthen Section 6.3.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Human baseline comparison**: While the paper's claim of near-100% human performance is unsupported, demanding a formal human study is scope creep beyond what's needed; the core problem is that "self-knowledge" is not isolated from basic task difficulty, not that humans aren't tested.

- **Reproducibility concerns about prompt design / hyperparameters**: The prompt templates are provided in the appendix; this is a standard level of detail for a benchmarks/datasets paper.

- **Formatting/style nitpicks**: Removed per rules.

- **Claim that the in-context evaluation "undercuts the self-knowledge claim"**: The fact that models perform well in-context is actually relevant and informative—it tells us something about how these abilities manifest. It doesn't make the original evaluation meaningless.

## Novel Insights

The most interesting finding is the dramatic gap between no-context and in-context evaluation (e.g., GPT-4: 0.03 → 1.00). This reveals that models' inability to reconstruct or verify properties of their own outputs is largely about the accessibility of information in context, not about any "internal knowledge." This finding itself undermines the self-knowledge framing—what's being measured is mostly whether the relevant information is still in the context window, not whether the model "understands" its own creation. The paper does not fully grapple with this implication.

## Suggestions

1. **Decouple generation failure from verification failure**: Report what percentage of the model's generated paragraphs actually meet the specified constraint (e.g., how often does the model actually generate exactly N words?). Then report verification accuracy separately on only the correctly-generated items. This would clarify whether the "self-knowledge" gap is about generation incompetence, verification incompetence, or both.

2. **Add an external-content control**: Run the same verification tasks (word counting, preposition counting, SQL operations) on human-written paragraphs. If models perform equally poorly on external text, the "self-knowledge" framing is not warranted; if they perform significantly worse on their own outputs, the framing has legs.

3. **Temper the interpretive claims**: Replace "misalignment with human attention mechanisms" with "correlation between attention patterns and task performance" in the absence of causal evidence or human data. Move this from a central finding to a preliminary observation.

4. **Revise the scoring metric**: Use the *actual* property of the generated text (e.g., actual word count) rather than the *instructed* property as the ground truth. This would cleanly separate instruction-following from verification ability.

## Score and Decision

This paper sits at a similar conceptual space to the "Generative AI Paradox" paper (scores: 6,8,6,8, avg ~7, accepted as poster) and "Looking Inward" (scores: 6,6,6,8, avg ~6.5, accepted as poster), both of which study whether models "understand" what they generate. However, both of those papers had clearer experimental designs that isolated the phenomenon they studied—Generative AI Paradox directly compared generation vs. understanding on the same tasks with human baselines, and Looking Inward had a cross-model experimental design to isolate "privileged access." This paper, by contrast, does not isolate "self-knowledge" from generic task difficulty, conflates instruction-following failures with verification failures, and makes speculative claims about human attention alignment without human data. Its empirical findings (LLMs are bad at counting/verification on their own text) are not new observations, and the proposed framework doesn't establish that a distinct "self-knowledge" construct is being measured.

At the lower end, papers like "Words and Deeds" (scores: 3,6,8,8, avg ~6.25, accepted) had a similar problem (measuring consistency rather than a deep construct) but had cleaner task design. "From Imitation to Introspection" (scores: 6,6,3,8, avg ~5.75, rejected) made strong claims about "self-consciousness" that reviewers found unsupported—similar to this paper's unsupported claims about "self-knowledge" and human attention alignment. "Logical Consistency" (scores: 3,5,6,6,8, rejected, avg ~5.6) had empirical evaluation without clear theoretical grounding, similar to this paper.

This paper's novelty is in the generate-then-evaluate setup and creative task designs, but its core conceptual claim is significantly undermined by the conflation of instruction-following and verification, lack of external-content controls, and unsupported interpretive claims. The empirical findings, while interesting as diagnostic demonstrations, do not establish the distinct construct they claim to.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>