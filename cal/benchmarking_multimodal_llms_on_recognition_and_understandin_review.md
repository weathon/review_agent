=== CALIBRATION EXAMPLE 42 ===

# Final Consolidated Review
## Summary

ChemTable introduces a benchmark of 1,382 real-world chemical tables from peer-reviewed chemistry literature (2015–2024), annotated with cell layouts, logical structures, domain-specific labels, and ~9,886 QA instances. It evaluates multimodal LLMs on two tasks—table recognition (structure/content extraction with fine-grained retrieval and molecular recognition) and table understanding (descriptive and reasoning QA)—revealing that while models handle layout parsing reasonably well, they struggle substantially with molecular structures, symbolic conventions, and fine-grained cell-level alignment.

## Strengths

- **Domain-specific benchmark filling a genuine gap.** Existing table benchmarks (WikiTQ, FinQA, SciTab, MMTab) lack the combination of pictorial modality (molecular diagrams), textual modality, and domain-specific semantics that chemical tables require. Table 1 clearly positions ChemTable as the only benchmark combining all these properties, and the choice of chemistry as a domain is well-motivated by its uniquely dense multimodal structure.

- **Multi-faceted evaluation protocol with domain-aware metrics.** The use of the Tanimoto coefficient for molecular graph comparison (rather than raw edit distance) is a technically sound adaptation for SMILES structural isomorphism (Section 4.1), and the three fine-grained recognition tasks (Value Retrieval, Position Retrieval, Molecular Recognition) go beyond standard TEDS to diagnose specific failure modes.

- **Insightful failure analysis beyond aggregate scores.** The qualitative case studies (Appendix M) identify concrete, distinct failure modes—visual-style hallucination (M.2), domain-specific footnote misreading (M.3), and correct intermediate reasoning with wrong final output mapping (M.4)—that provide actionable targets for model improvement. The unanswerable question analysis (Table 5) testing model self-awareness is a notable design choice.

- **Comprehensive model coverage.** The paper evaluates 7–10 MLLMs spanning both open-source and proprietary families, includes human baselines, and provides additional analyses on input modality impact (Figure 5), multi-hop degradation (Table 6), and CoT effects (Figure 8).

## Weaknesses

### Major:

- **Dataset size inconsistency undermines confidence in reported statistics.** The abstract states "over 1,300 tables," Table 2 reports 1,382 images, and Appendix D.3 claims "1,500 fully annotated chemical table images." Appendix N states 1,382 tables. The discrepancy between the 1,500 in D.3 and 1,382 elsewhere is unexplained and should be reconciled—readers cannot determine which number is the true dataset size.

- **Difficulty filtering via a single model may introduce systematic bias.** Section 3.3.4 discards questions that Qwen-2.5-7B answers correctly on the first attempt. This creates a benchmark biased toward questions that one specific model family finds difficult, potentially over-representing failure modes particular to that architecture and under-representing challenges that other model families face. The paper does not analyze what types of questions were filtered or whether this skews the difficulty distribution.

- **Incomplete human baselines make the claimed "human-model gap" unreliable for key task categories.** Appendix L states "Human performance is only reported for tasks that require chemical expertise or complex reasoning; purely descriptive element-level tasks are not annotated by humans." Table 4 shows "-" for all descriptive human scores. Yet the abstract claims models "fall short of human-level performance" broadly. Without human baselines on descriptive tasks (especially Molecular Recognition, where models score as low as 11.63), the magnitude of the gap for these subtasks is unknown.

### Minor:

- **GPT-4.1-nano as evaluator lacks per-category agreement analysis.** While Appendix G reports 96.8% overall agreement with human judgment on a 20% sample, the agreement is not broken down by question type. Agreement likely varies between descriptive questions (easier to judge) and domain-specific chemistry reasoning (harder). If GPT-4.1-nano systematically misjudges chemistry-specific answers, model rankings could be affected.

- **Molecular formula bottleneck claim lacks controlled analysis.** Figure 4 shows performance declining as molecular formula count increases, but no confound analysis is provided—tables with many molecular structures may also have more complex layouts, spanning cells, or additional rows. The claim that "molecular formulas remain a key bottleneck" (Section 4.2c) would be strengthened by controlling for table structural complexity.

- **Specialized model comparisons are relegated to appendices.** Figure 3 compares MLLMs to DECIMER for molecular recognition, and Figure 14 compares ChemVLM and Table-LLaVA, but these are not in the main results. Given that the paper's core claim is about domain-specific limitations, including at least one specialized model in the main tables would better support whether the gap is about general vs. domain-adapted architectures.

- **Unanswerable question scoring in main results is unclear.** Table 5 analyzes model behavior on unanswerable questions separately, but it is unclear how (or whether) unanswerable questions factor into the main accuracy scores in Tables 3 and 4. If unanswerable questions are included in the QA pool but a model incorrectly answers them, this could penalize models that attempt to help vs. those that refuse.

- **GPT-4.1 used for both question generation and answer evaluation creates a potential circularity.** GPT-4.1 generates reasoning questions (Section 3.1.2), rewrites questions for diversity (Section 3.3.4), and GPT-4.1-nano evaluates answers (Section 5.1). While different model tiers are used, this pipeline creates a single-vendor dependency that may share systematic biases across generation and evaluation.

- **No discussion of potential train-test contamination.** MLLMs evaluated here may have been trained on the same chemistry literature from which tables were drawn (2015–2024, high-impact journals). The paper does not discuss this risk or attempt to mitigate it, which is important for a benchmark claiming to reveal genuine model limitations.

### Trivial:

- **Section ordering inconsistency.** Section 3.3.3 (Reasoning Questions) appears before 3.3.1 (Task Definition) and 3.3.2 (Descriptive Questions) in the text flow, making the section structure confusing.

- **Appendix D.3 describes a three-stage pipeline yielding "1,500" tables, but the main dataset uses 1,382.** It is unclear whether 118 tables were excluded and why.

## Nice-to-Haves

- **Fine-tuning or adaptation experiments on ChemTable.** The paper only evaluates existing models; showing that training on ChemTable data improves performance would demonstrate the benchmark's utility beyond evaluation.

- **Controlled ablation isolating chemical domain factors.** A comparison against a general-domain table benchmark (e.g., SciTab) using the same models would empirically validate that chemical tables are uniquely challenging rather than just generally harder.

- **Statistical significance testing for close score differences.** While most gaps are large enough to be meaningful, some comparisons (e.g., GPT-5 at 93.11 vs. GPT-4.1 at 92.94 on Annotation Description) would benefit from confidence intervals.

- **Clear train/validation/test splits with contamination prevention guidelines.** The paper mentions releasing the benchmark but does not specify whether designated splits exist or how to prevent overfitting in future work.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Table 3/Table 4 formatting issues** (Harsh Critic): These are PDF parser artifacts, not actual paper problems. The instructions explicitly state "formatting artifacts are parser issues, not paper problems."

- **Model versioning/API dates for proprietary models** (Harsh Critic): Per hard rules, I cannot question the existence of cited models, and demanding exact API versioning is a nitpick about trivial implementation details.

- **Reproducibility concerns about closed-source model evaluation** (Positive Reviewer): Evaluating proprietary models via API is standard practice in the field; this is not a novel reproducibility concern specific to this paper.

- **Missing related works** (Harsh Critic): Per hard rules, I should not mention missing related works as I cannot confirm their existence.

- **Generalizability to other chemistry subfields or other scientific domains** (Harsh Critic): This is scope creep. The paper explicitly focuses on experimental chemistry tables and should be evaluated on whether it does that well, not on whether it also covers computational chemistry or biology.

- **Negative impact/dual-use discussion** (Harsh Critic): The paper includes an ethics statement. Demanding broader impact analysis of chemical data extraction misuse is beyond the scope of a benchmark paper.

- **Leaderboard infrastructure** (Spark Finder): Nice for long-term impact but not a requirement for a benchmark paper.

## Novel Insights

The multi-hop reasoning case study (Appendix M.4) reveals a particularly interesting failure pattern: models can correctly execute all intermediate reasoning steps—locating the right row, verifying constraints—yet produce the wrong final answer by mapping to an incorrect schema field. This suggests the bottleneck is not in reasoning capacity per se, but in output schema grounding, a failure mode distinct from hallucination or lack of knowledge. This pattern, where correct intermediate computation yields incorrect final output, may indicate that current MLLMs treat structured table cells as semantically interchangeable tokens rather than schema-bound slots, pointing to a specific architectural limitation in how decoder-only models map tabular structure to output format.

## Suggestions

- Reconcile the dataset size numbers (1,300/1,382/1,500) explicitly: state which is the final benchmark size, explain why Appendix D.3 mentions 1,500, and whether any tables were excluded after annotation.

- Report human baselines for at least Molecular Recognition and Visual Description subtasks—these are where the largest model gaps exist, and without human scores the claim about "falling short of human-level" is unsupported for these categories.

- Analyze the difficulty filter's impact: report what fraction of questions was removed, what types they were, and whether the filtered dataset's difficulty profile differs meaningfully from the pre-filter version.

- Move at least one specialized model comparison (DECIMER or ChemVLM) into the main results tables to directly support the claim about domain-specific limitations.

- Clarify how unanswerable questions are scored in the main evaluation—whether they are excluded from accuracy computation or count as incorrect if answered.

## Evaluation Summary

- **Novelty**: Moderate — the benchmark addresses a genuine and underexplored gap (chemical tables with molecular structures), but the contribution is primarily a dataset rather than a methodological advance; the domain-aware evaluation adaptations (Tanimoto coefficient, Function-Based QA) add some technical novelty.

- **Technical soundness**: Fair — the evaluation protocol is well-designed, but the dataset size inconsistency, potential difficulty-filtering bias, and incomplete human baselines reduce confidence in some reported claims.

- **Empirical support**: Moderate to strong — extensive model coverage and insightful failure analyses provide substantial evidence; however, the lack of controlled ablations and limited specialized model comparisons weaken the causal claims about what specifically makes chemical tables hard.

- **Significance**: Moderate to high — the benchmark fills an important gap and reveals clear failure modes; however, without demonstrating that the benchmark enables model improvement (via fine-tuning or adaptation), its practical impact beyond evaluation is prospective rather than demonstrated.

- **Clarity**: Moderate — generally well-structured with useful appendices, but section ordering issues, the dataset size inconsistency, and unclear scoring of unanswerable questions create avoidable confusion.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0]
Average score: 4.5
Binary outcome: Reject
