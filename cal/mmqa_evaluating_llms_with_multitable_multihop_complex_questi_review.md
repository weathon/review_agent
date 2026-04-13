=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
## Summary
This paper introduces MMQA, a benchmark built to evaluate LLMs on multi-table, multi-hop reasoning over relational data, with subtasks spanning retrieval, QA, Text-to-SQL, and primary/foreign key selection. The paper also proposes MTR, an iterative retrieval method based on question decomposition plus table connectivity, and shows sizable retrieval gains over the included baselines. The direction is timely and potentially impactful, but several parts of the empirical validation and benchmark methodology are not yet strong enough to fully support the paper’s broader claims.

## Strengths
- **The benchmark covers a genuinely broader problem setting than standard single-table table QA benchmarks.** MMQA explicitly evaluates multiple stages of multi-table reasoning—retrieval, QA, Text-to-SQL, primary key selection, and foreign key selection—rather than collapsing everything into a single answer metric. This is a meaningful design choice because the results expose distinct weaknesses, e.g. much lower foreign-key selection than primary-key selection in Tables 5–6.
- **The paper operationalizes multi-table reasoning at multiple granularities.** The benchmark is not only about final answers: it probes table-level retrieval, column-level schema understanding (PK/FK), and cell-level answering. That diagnostic framing is more informative than many existing table benchmarks.
- **MTR shows large empirical gains on the retrieval subtask as evaluated here.** In Table 4, MTR(TableLlama-7b) improves substantially over TableLlama-7b alone across Top-2/5/10 for both 2-table and 3-table settings; the ablation without question decomposition is also noticeably worse on Top-2, supporting that decomposition contributes.
- **The benchmark results reveal a substantial gap between current LLMs and humans on the proposed tasks.** Even the strongest model reported remains far below human performance on QA and especially on key selection, suggesting the benchmark is challenging in a way that may be useful for future work.
- **The paper includes an initial analysis of long-table effects.** The table-length study is not fully conclusive, but it points to an interesting difference between answering and SQL generation as table size grows, which could motivate follow-up work on context management for tabular reasoning.

## Weaknesses
###: Fatal

### Major:
- **The Text-to-SQL evaluation is not technically sound for the claims being made.** The paper evaluates SQL generation with ROUGE-1, ROUGE-L, and BLEU (Section 3.3, Section 4.1, Tables 5–6), but these are poor proxies for SQL correctness. Semantically equivalent SQL can have low lexical overlap, and lexically similar SQL can still be wrong. Since Text-to-SQL is presented as a core benchmark subtask, this substantially weakens that part of the empirical contribution and makes the paper’s conclusions about model SQL ability unreliable.
- **The benchmark’s broader claims about “real-world” multi-table reasoning are under-validated given the construction pipeline.** Section 3.1 states that MMQA is built from Spider by selecting samples, synthesizing SQL through 45 manually crafted templates, and then using GPT-4-turbo to generate natural-language questions. That can still yield a useful benchmark, but the paper does not provide enough analysis to show that the resulting questions are natural, diverse, or genuinely require the intended reasoning rather than reflecting template structure. As written, the benchmark is promising but not yet fully validated as evidence of real-world multi-table complexity.
- **The practical value of MTR is only partially established because retrieval is evaluated in isolation, not connected to downstream task performance.** The paper shows strong gains on retrieval (Table 4), and separately reports QA/SQL/PK/FK results for various LLMs (Tables 5–6), but does not show whether using MTR retrieval actually improves end-task multi-table QA relative to baseline retrieval or oracle retrieval. This leaves the central systems story incomplete: strong retrieval numbers do not by themselves show improved end-to-end reasoning.
- **The retrieval comparison is not fully controlled, so the “SOTA” framing should be toned down.** MTR relies on GPT-4-turbo for question decomposition and also uses fine-tuned retrieval models trained on off-the-shelf single-table QA data, whereas several baselines in Table 4 are simpler retrievers or base models without the same pipeline support. The gains are real on the reported setup, but the comparison does not cleanly isolate whether improvement comes from the retrieval formulation itself versus extra decomposition/supervision.
- **There are unresolved clarity/consistency issues in the dataset description that matter for a benchmark paper.** Section 3.1 says “we randomly select a total of 5,000 samples from Spider,” yet later says the final benchmark consists of “3,312 tables in total with the corresponding natural language question, SQL query, gold answer...” Table 1 mixes table counts and question-category counts in a way that is hard to reconcile. For a benchmark submission, the precise unit of evaluation—tables, examples, databases, or question-table sets—needs to be stated clearly.

### Minor
- **The human baseline presentation is incomplete and not cleanly aligned with model reporting.** Tables 5–6 provide human PM / SQL / PKS / FKS values, but the QA columns omit human EM despite the surrounding discussion emphasizing human-vs-model gaps. This does not invalidate the benchmark, but it makes a recurring claim harder to assess directly.
- **The partial-match QA metric depends on GPT-4-turbo as an evaluator, while GPT-4 is itself among the evaluated models.** The paper notes that PM is scored by GPT-4-turbo using an appendix prompt. This is not necessarily invalid, but it does introduce evaluator dependence and would benefit from stronger validation or a human-checked subset.
- **The table-table relevance function in MTR is very simplistic relative to the schema reasoning motivation.** Section 3.2 defines table relevance via column overlap, effectively a binary score of 1/0 depending on overlap with previously retrieved tables. This is a coarse approximation to joinability and may miss semantically related columns with different names or over-credit spurious overlaps.
- **Some result reporting appears internally inconsistent and should be checked carefully.** In Table 4, the “w/o QD” row reports a Top-5 2-table F1 of 74.6, which is higher than the full MTR row’s 73.0, inconsistent with the text claiming question decomposition provides improvements. This looks like either a reporting typo or a calculation issue and should be corrected.
- **The long-table analysis is interesting but not yet robust enough to support a strong claim about an “elbow point.”** The experiment uses only 50 samples per bucket and the main paper does not provide uncertainty estimates or enough detail on table serialization/context handling, so the observed drop around 800 rows should be treated as suggestive rather than conclusive.

### Trivial
- **Some interpretation text is speculative beyond what the experiments show.** For example, attributing model differences to “advanced feature engineering capabilities” or specific schema-understanding mechanisms is not directly supported by the presented evidence.
- **There are a few local explanation issues around schema semantics.** In the Figure 1 walkthrough, the text says “Head ID is both the foreign key and the primary key of the table,” which is schema-specific and too casually stated for a benchmark that evaluates key understanding.

## Nice-to-Haves
- Add qualitative error analysis by failure mode: wrong table retrieval, wrong key identification, wrong join chain, arithmetic/counting error, or answer formatting.
- Provide a human evaluation or manual audit of question naturalness and reasoning necessity for a subset of GPT-generated questions.
- Include a finer-grained analysis of 2-table vs 3-table difficulty to verify that multi-hop structure itself is the main source of challenge.
- Strengthen the decomposition analysis by measuring decomposition quality and correlating decomposition errors with retrieval failures.
- Extend the benchmark beyond 2- and 3-table cases in future versions; this would increase realism, though the current scope is still a reasonable starting point.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claims doubting the existence/release/availability or reproducibility of cited models/tools** — removed per instruction.
- **Complaints about missing related work or unnamed external baselines** — removed because they require external verification. In particular, suggestions that the paper must compare to specific additional systems not discussed in the paper are better treated as optional future strengthening, not core weaknesses.
- **Formatting/parser artifacts and duplicated figure/table captions** — removed as non-substantive.
- **Strong reproducibility complaints about omitted low-level implementation details** — weakened/removed unless they directly undermine the main claim. Here, the more substantive retained issue is not missing hyperparameters but missing evaluation design details that affect interpretation (e.g., end-to-end pipeline connection, SQL metrics, dataset accounting).
- **A criticism that the retrieval pool uses only tables from the same database as the question** — removed because the paper actually states “We build a table pool that includes all tables of MMQA” (Section 3.3), so that specific complaint is not supported by the text.
- **A claim that the paper is merely unfair to baselines because the baselines are weaker** — removed in its stronger form. The retained concern is narrower and factual: the comparison is not controlled for decomposition/supervision, not that baseline weakness alone makes it invalid.

## Novel Insights
The most interesting aspect of the paper is not just that multi-table QA is hard, but that the benchmark decomposes this difficulty into schema-centric subskills that appear disproportionately weak: foreign key selection remains poor even for the strongest models, suggesting that failures may stem less from generic language understanding and more from brittle relational schema modeling. At the same time, the paper’s own method implicitly acknowledges this by solving retrieval with decomposition plus schema connectivity heuristics rather than relying on one-shot end-to-end reasoning. This points to a potentially important research direction: multi-table reasoning may benefit more from explicit intermediate structure induction than from scaling raw prompting alone.

## Suggestions
- Replace or supplement ROUGE/BLEU with execution-based Text-to-SQL evaluation; this is the single most important empirical fix.
- Add an end-to-end experiment comparing QA with baseline retrieval, MTR retrieval, and oracle retrieval, so the value of MTR is demonstrated beyond retrieval F1.
- Clarify dataset accounting precisely: number of examples, number of distinct tables, number of databases, and how the 5,000 selected Spider samples relate to the final MMQA instances.
- Add validation of benchmark quality: at minimum, a manual audit of generated questions for naturalness, diversity, and whether multi-hop reasoning is actually necessary.
- Tone down “real-world” and “SOTA” claims unless they are more carefully qualified by the synthetic construction and comparison setup.
- Verify and correct Table 4 / associated discussion, especially the apparent inconsistency in the ablation row.
- Provide a small set of worked failure cases showing the full chain from question to retrieved tables to predicted answer/SQL, which would make the benchmark more diagnostically useful.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
