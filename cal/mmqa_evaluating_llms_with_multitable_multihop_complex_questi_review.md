=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary

This paper introduces MMQA, a benchmark for evaluating LLMs on multi-table, multi-hop question answering tasks, addressing the limitation of existing single-table benchmarks. The authors construct a dataset of 3,312 instances derived from Spider, propose a Multi-Table Retrieval (MTR) method using question decomposition and iterative relevance scoring, and evaluate proprietary and open-source LLMs across retrieval, Text-to-SQL, QA, and key selection tasks. The paper demonstrates that current LLMs significantly lag behind human performance on multi-table reasoning.

## Strengths

- **Comprehensive evaluation framework**: The paper introduces a multi-faceted evaluation covering Multi-Table Retrieval, Text-to-SQL, Multi-Table QA, and Primary/Foreign Key Selection. The fine-grained tasks (PKS, FKS) provide diagnostic insight into *where* models fail—e.g., Foreign Key Selection accuracy is critically low (~28% for GPT-4), revealing that schema understanding remains a bottleneck.
- **Empirical rigor with human baseline**: The inclusion of human performance (89.8 EM for QA, 96.5% PKS accuracy) provides meaningful calibration. The analysis of table length impact (Figure 4) identifies a practical degradation "elbow point" around 800 rows, offering actionable guidance for deployment.
- **Dataset complexity validated**: Figure 3 demonstrates MMQA requires significantly more reasoning steps (7.46 average) compared to existing benchmarks (TableBench: 6.28, HybridQA: 3.15), substantiating the claim that multi-table reasoning is genuinely harder.

## Weaknesses

- **GPT-4-turbo circularity in data generation and evaluation**: The paper uses GPT-4-turbo to generate natural language questions from SQL (Section 3.1), uses it again for Partial Match evaluation (Section 4.1), and evaluates GPT-4 and O1-preview on the same data. This creates two layers of potential bias: (1) GPT-derived models may have systematic advantages on GPT-generated questions, and (2) GPT-4-turbo evaluating other models (including itself in related experiments) introduces conflict of interest. The paper does not acknowledge or analyze this issue.

- **Text-to-SQL evaluation metric is inappropriate**: The paper evaluates Text-to-SQL using ROUGE-1, ROUGE-L, and BLEU—n-gram overlap metrics that do not capture SQL semantic correctness. Two syntactically different queries can be functionally equivalent. The standard practice in Text-to-SQL literature (Spider, BIRD, SPINACH) is execution accuracy or exact set match. Without execution accuracy, the reported scores (e.g., O1-preview achieving BLEU of 7.58) are nearly uninterpretable for SQL correctness.

- **"First" claim overstates novelty**: The introduction states "to the best of our knowledge, our research is the first to introduce a multi-table and multi-hop QA benchmark." However, Related Work acknowledges Pal et al. (2023) "pioneer a multi-table pre-training task of answering questions over multiple tables." The paper differentiates itself by claiming cell-level granularity and multi-hop focus, but the blanket "first" claim should be qualified.

- **MTR comparison is unfair by construction**: MTR uses GPT-4-turbo for question decomposition while all baselines (BM25, TF-IDF, DTR, SGPT, TableLlama) do not. The reported gains (Table 4) cannot be attributed to MTR's iterative retrieval logic alone—GPT-4-turbo decomposition is itself a powerful advantage. There is no ablation showing MTR's framework alone (without GPT-4 decomposition) compared to baselines augmented with the same decomposition.

- **Reasoning steps metric undefined**: The paper prominently claims MMQA requires 7.46 average reasoning steps (Figure 3), central to the difficulty argument. Yet "reasoning step" is never formally defined—whether it counts SQL operations, entity traversals, or something else is unclear, making the complexity comparison unverifiable.

- **Context window handling unexplained**: Tables average >1,800 rows (Table 1), but the paper never explains how these fit into LLM context windows. Presumably truncation or sampling occurs, but this confounds the "table length" analysis—performance degradation could stem from information loss rather than reasoning difficulty.

- **Confusing statistic in Table 1**: The "Avg primary keys per table" is reported as 3.35 (2-table) and 2.41 (3-table). By definition, a relational table has one primary key (possibly composite). This statistic likely conflates candidate keys with primary keys, or reflects annotation inconsistency, and needs clarification.

- **Table 5 labeling error**: Two rows are both labeled "O1-preview*" with different scores—one should clearly be "O1-preview†" (one-shot). While minor, this undermines confidence in result reporting.

## Nice-to-Haves

- **Execution accuracy for Text-to-SQL**: Replace ROUGE/BLEU with database execution accuracy to enable meaningful SQL quality assessment. This would require running generated queries against the database.

- **Data contamination analysis**: Since MMQA derives from Spider (a widely-used training dataset), quantify potential pretraining overlap. Models may have memorized Spider schemas, inflating performance.

- **End-to-end pipeline evaluation**: Retrieval is evaluated separately (Table 4) from QA (Tables 5-6). Evaluating the full pipeline (retrieval → reasoning) would demonstrate real-world applicability.

- **Schema heterogeneity testing**: The MTR relevance score β is 1 if columns share names, 0 otherwise. Real databases often use different naming conventions (e.g., `student_id` vs `stuID`). Testing on heterogeneous schemas would validate robustness.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Copy-paste error in Table 6**: The harsh critic claims GPT-4* values in Table 6 match Gemini-pro* from Table 5 exactly, suggesting data fabrication. Upon verification, the values differ (PM: 32.72 vs 31.48; several columns are identical but not all). While suspiciously similar, this is not conclusive evidence of copy-paste error.

- **Dataset size concerns**: Demanding larger datasets or more table configurations would be scope creep. The current size (~3,300 samples, 2-3 tables) is appropriate for a focused benchmark.

- **Missing specialized Text-to-SQL baselines**: While DIN-SQL or DAIL-SQL would strengthen comparisons, their absence doesn't invalidate the benchmark contributions.

- **Generic novelty criticism about incremental MTR**: The claim that MTR is "just engineering" undervalues the empirical contribution. Question decomposition + iterative retrieval is a reasonable baseline that the paper provides.

## Novel Insights

The Foreign Key Selection accuracy (26-34% for proprietary LLMs, vs. 96%+ for humans) reveals a specific, underexplored bottleneck: LLMs struggle to infer cross-table relationships from schema metadata alone. This suggests future work should focus on schema-aware pretraining or explicit schema encoding, rather than scaling model size. Additionally, the "elbow point" at 800 rows in Figure 4 implies context window constraints currently limit practical deployment more than reasoning capacity—a finding that would benefit from controlled experiments isolating context length from reasoning hops.

## Suggestions

- Define "reasoning step" formally in the paper or appendix (e.g., number of JOIN operations, number of entity lookups) to make complexity claims verifiable.
- Add execution accuracy for Text-to-SQL; even a subset would strengthen evaluation validity.
- Ablate MTR's question decomposition by comparing MTR (without decomposition) against baselines with decomposition added, to isolate the framework's contribution.
- Clarify how tables with 1,800+ rows are presented to LLMs—full context, sampling, or truncation—to contextualize the length analysis.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
