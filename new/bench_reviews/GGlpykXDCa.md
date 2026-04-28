## Summary
This paper introduces MMQA, a benchmark for evaluating LLMs on multi-table multi-hop question answering, along with MTR, a novel multi-table retrieval method. The benchmark requires models to identify table relationships via foreign/primary keys and perform multi-step reasoning across 2-3 table samples derived from Spider. The MTR method uses question decomposition and iterative retrieval considering both question-table and table-table relevance, achieving SOTA retrieval performance.

## Strengths
- **Genuine multi-table focus**: Unlike prior benchmarks (WikiSQL, HybridQA, Spider) that are single-table or table-text, MMQA explicitly requires multi-table reasoning with foreign/primary key identification. Table 2 clearly positions this gap, and the 7.46 average reasoning steps (Figure 3) exceeds prior datasets.
- **MTR retrieval method shows measurable gains**: Table 4 demonstrates MTR (with TableLlama-7b) achieves Top-2 F1 of 68.3% on 2-table tasks, substantially outperforming SGPT-54B (46.9%) and DTR (33.3%). The question decomposition ablation shows QD contributes ~4-5 F1 points.
- **Granular diagnostic evaluation**: The framework decomposes performance into four subtasks (Retrieval, Text-to-SQL, QA, Key Selection), revealing that Foreign Key Selection (~30% for SOTA models) is a major bottleneck compared to QA (~50% EM for O1-preview).
- **Table length analysis provides empirical insight**: Figure 4 identifies a performance cliff at ~800 rows for multi-table QA, where all models drop sharply, while Text-to-SQL degrades gradually (likely because SQL generation focuses on schema, not content).

## Weaknesses

### Fatal
None

### Major
- **Text-to-SQL evaluation uses string-similarity metrics instead of Execution Accuracy**: Section 3.3 (lines 250-251) and Tables 5-6 evaluate Text-to-SQL using Rouge-1, Rouge-L, and BLEU. The Text-to-SQL community standard (Spider, BIRD) is Execution Accuracy—whether the generated SQL returns the correct result when executed. Two syntactically different queries can be semantically identical (low Rouge, correct execution), while minor token variations can yield high Rouge but broken SQL. This metric choice makes the Text-to-SQL results difficult to interpret for the database community and weakens claims about model SQL generation capability. Notably, calibration paper C0QYbFfpdd.md argues Execution Accuracy has limitations in the LLM era (diverse correct queries exist), but it still assumes execution-based evaluation is the baseline—the MMQA paper does not use execution at all.

- **Data contamination risk from Spider-derived construction without mitigation**: Section 3.1 (lines 95-96) states MMQA is synthesized directly from the Spider database. Spider has been the dominant Text-to-SQL training dataset for 5+ years; models evaluated (including fine-tuned open-source LLMs and proprietary models trained on web corpora) have almost certainly seen Spider schemas and SQL patterns. The paper provides no contamination analysis, train/test separation verification, or schema obfuscation. Calibration paper VStKtgGXUc.md (jqBench, avg 5.33) similarly uses Spider, and a reviewer explicitly notes "Spider has had potential data contamination issues since its creation" and recommends BIRD instead. Without contamination controls, it is unclear whether models are reasoning over multi-table structures or recalling memorized Spider patterns.

### Minor
- **Retrieval baselines are outdated for a 2025 submission**: Section 4.2 compares MTR against BM25, TF-IDF, and DTR (2021). Modern dense retrieval methods (Contriever, E5, specialized RAG retrievers) are not included. While MTR's SOTA claim holds against the tested baselines, the absence of contemporary retrievers makes the "state-of-the-art" positioning less convincing.

- **Human baseline may be inflated due to annotation design**: Section 3.1 states annotation was performed by "two human experts" who also defined the reasoning steps and keys. When annotators evaluate data they helped construct, performance is artificially elevated compared to blind evaluation. The reported ~90% Exact Match (Tables 5-6) is unusually high for complex multi-hop reasoning and may not represent a reliable ceiling for the benchmark's true difficulty.

### Trivial
- **Terminology inconsistency between text and tables**: Section 3.1 states "3,312 tables in total," but Table 1 clarifies this as 2,591 (2-table) + 721 (3-table) *samples*. This creates ambiguity about whether the benchmark has 3,312 unique database schemas or 3,312 question-table-set instances drawn from fewer underlying schemas.

## Nice-to-Haves
- Add Execution Accuracy alongside Rouge/BLEU for Text-to-SQL to align with community standards and enable semantic equivalence evaluation.
- Include a contamination-controlled subset (e.g., from databases not in Spider, or with schema obfuscation) to isolate reasoning from memorization.
- Compare MTR against modern dense retrievers (e.g., Contriever, E5, or RAG frameworks like LangChain's retrievers) to strengthen the SOTA claim.
- Provide qualitative error analysis on Foreign Key Selection failures (e.g., confusing semantic similarity with referential integrity) to justify the utility of this subtask.
- Clarify schema diversity: quantify how many unique database schemas are represented in the 3,312 samples to assess whether retrieval is schema recognition or generalizable retrieval.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that Rouge/BLEU "invalidates" Text-to-SQL results**: While using string-similarity metrics is a limitation, calibration paper C0QYbFfpdd.md (avg 4.00) argues the opposite—that Execution Accuracy is inadequate for LLM-era NL2SQL because diverse semantically-equivalent queries exist. The community debate means Rouge/BLEU is suboptimal but not "invalidating." Moved to Major weakness with appropriate framing.

- **Harsh Critic's claim that contamination "undermines the core claim"**: Many accepted papers use Spider-derived data (jqBench at 5.33, TQA-Bench at 4.00). Contamination is a legitimate concern but does not automatically invalidate results. Moved to Major weakness with calibrated severity.

- **Strength Finder's "Human Performance Baseline for Gap Analysis"**: This strength is undermined by the annotation bias concern (annotators created the data). Removed.

- **Strength Finder's generic statements**: "Benchmark Novelty in Multi-Table Scenarios" and "Superior Retrieval Performance" are kept with specific evidence; generic phrasing about "important problem" was removed.

- **Harsh Critic's concern about MTR stopping condition (Algorithm 1)**: The paper states "When 0 is assigned, stop iterations" where β assigns 0 for no column overlap. This is a design choice for joinable tables; the critic's concern about semantic joins without column name overlap is valid but speculative without empirical evidence. Moved to Nice-to-Have for future work.

- **Harsh Critic's concern about GPT-4-turbo self-bias in evaluation**: The paper uses GPT-4-turbo for both question generation and answer evaluation. While potential bias exists, this is common practice (calibration papers use similar setups). Moved to Nice-to-Have for discussion.

## Novel Insights
The paper's table length analysis (Figure 4) reveals a previously undocumented "elbow point" at ~800 rows for multi-table QA, where performance drops sharply across all models. This contrasts with Text-to-SQL, which degrades gradually—suggesting QA requires content access while SQL generation focuses on schema. This finding has practical implications for context window management in table reasoning systems. Additionally, the Foreign Key Selection bottleneck (~30% for SOTA models vs. ~50% for QA) indicates that schema understanding, not answer generation, is the primary barrier to multi-table reasoning—a diagnostic insight enabled by the granular evaluation framework.

## Suggestions
1. **Add Execution Accuracy for Text-to-SQL**: Even a subset evaluation (e.g., 500 samples) with execution-based metrics would significantly strengthen the Text-to-SQL results and align with community standards. Report both string-similarity and execution-based scores to show correlation/divergence.

2. **Conduct contamination analysis**: Either (a) create a contamination-controlled subset from databases not in Spider, (b) filter model training data to exclude Spider, or (c) perform a schema obfuscation experiment (rename columns/tables) to test whether models rely on schema memorization. Report results separately.

3. **Expand retrieval baselines**: Include at least one modern dense retriever (e.g., Contriever, E5, or a fine-tuned retriever from 2023-2024) to strengthen the SOTA claim for MTR.

4. **Clarify schema diversity**: Report the number of unique database schemas underlying the 3,312 samples. If diversity is low (e.g., <100 unique schemas), acknowledge this limitation and discuss implications for generalizability.

5. **Add qualitative error analysis for Key Selection**: Show 5-10 examples where models fail Foreign Key Selection, categorizing failure modes (e.g., semantic mismatch, missing join path, confusion with primary keys). This would justify the subtask's utility and guide future improvements.

## Score and Decision

**Calibration Process:**

1. **Topic-based anchors**: Retrieved multi-table/table benchmark papers:
   - TQA-Bench (hxEHr5gJBY.md): avg 4.00, Withdrawn—multi-table QA from WorldBank/Data.gov/BIRD, criticized for missing related work and contamination concerns
   - OpenDataBench (qvqtSIhhFt.md): avg 3.00, Reject—real-world multi-table benchmark, limited novelty
   - SPARTA (8KE9qvKhM4.md): avg 5.00, Accept Poster—table-text multi-hop QA with scalable generation
   - TaTToo (zc1ezBrr5m.md): avg 6.00, Accept Poster—tabular reasoning with novel PRM method
   - Relational Transformer (rpPtgMC5s9.md): avg 6.00, Accept Poster—multi-table retrieval architecture with zero-shot capability

2. **Quality-based anchors**: Retrieved papers with similar strength/weakness patterns:
   - C0QYbFfpdd.md (avg 4.00): Argues Execution Accuracy is inadequate for NL2SQL—shows community debate on Text-to-SQL metrics
   - joh5J1nYAE.md (avg 2.50): Benchmark with flawed metrics and model bias—rejected for methodological issues
   - GFDSGlEks2.md (avg 4.67): Data contamination study—rejected for narrow scope
   - VStKtgGXUc.md (avg 5.33, Accept Poster): Uses Spider dataset; reviewer notes "Spider has had potential data contamination issues"

3. **Range anchoring**:
   - **High (≥6)**: TaTToo (6.00), Relational Transformer (6.00), ProfBench (6.50)—all have novel methods + strong experiments
   - **Medium (~5)**: SPARTA (5.00), QTgx2ThiAb (5.00), jqBench (5.33)—solid benchmarks with some limitations, accepted as posters
   - **Low (≤4)**: TQA-Bench (4.00), OpenDataBench (3.00), CoTabBench (4.00)—benchmarks with significant methodological flaws, rejected/withdrawn

**Positioning**: The MMQA paper sits between SPARTA (5.00) and TQA-Bench (4.00). Like SPARTA, it offers a genuine multi-table contribution with empirical results. Like TQA-Bench, it has contamination concerns and metric limitations. However, MMQA's MTR method is a novel contribution (unlike TQA-Bench's incremental approach), and the Text-to-SQL metric issue, while real, is debated in the community (C0QYbFfpdd.md shows Execution Accuracy is not universally accepted as gold standard). The paper is stronger than OpenDataBench (3.00) and MultiTableQA (3.00, withdrawn) which had limited novelty or artificial data construction.

**Final Score**: Compared to anchors, MMQA warrants a **5.0**—borderline accept/poster. It has genuine contributions (multi-table benchmark + MTR method) but methodological limitations (Text-to-SQL metrics, contamination risk) that prevent a 6.0. It is clearly above the 3.0-4.0 rejected benchmarks.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>