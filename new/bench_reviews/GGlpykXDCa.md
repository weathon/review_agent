Now I have sufficient calibration data. Let me compile my findings and write the final review.

Key calibration anchors:
- **Spider 2.0** (real-world text-to-SQL benchmark): Oral, 8/8/8/8 — very high quality, real data, rigorous evaluation with execution accuracy
- **CofCA** (multi-hop QA benchmark): Poster, 8/5/5/6 — accepted with some issues around metric validity
- **DSBench** (data science benchmark): Poster, 6/8/8/6/6 — accepted, has multi-table but real data
- **MDBench** (synthetic multi-document reasoning): Rejected, 5/3/3/5 — synthetic benchmark with quality and methodology concerns
- **LAIA-SQL** (multi-table QA with text-to-SQL): Rejected (withdrawn), 3/6/3/8 — directly comparable paper

The paper under review is closer to MDBench and LAIA-SQL in quality and issues. The Text-to-SQL evaluation with BLEU/ROUGE is a real methodological flaw (the field standard is execution accuracy). The GPT-4 circular dependency is genuine. The data source inconsistency (Spider vs. Wikipedia in Table 2) is a real issue. The MTR method's comparison isn't clean. However, the benchmark idea is genuinely useful and fills a real gap.

---

## Summary
MMQA introduces a multi-table, multi-hop question answering benchmark built on Spider databases, evaluating LLMs across five subtasks: multi-table retrieval, text-to-SQL generation, multi-table QA, and primary/foreign key selection. The paper also proposes a multi-table retrieval method (MTR) that decomposes multi-hop questions into sub-questions and iteratively retrieves tables by combining question-table and table-table relevance. Experiments reveal a large gap between current LLMs and human performance on multi-table understanding tasks.

## Strengths

- **Genuinely fills an underexplored niche**: Most existing table QA benchmarks (WikiSQL, WikiTableQuestions, Spider, BIRD) evaluate single-table scenarios. MMQA is among the first to package multi-table, multi-hop QA evaluation with explicit foreign/primary key tasks. The specific diagnostic subtasks (PKS, FKS) go beyond simple answer accuracy to expose *which* aspect of multi-table reasoning models fail at—this is a concrete contribution most comparable benchmarks do not offer.

- **Multi-table retrieval component is novel and empirically supported**: The MTR method logically combines question-table relevance (via sub-question decomposition) with table-table relevance (via column overlap propagation), and Table 4 shows clear, consistent improvements over prior retrievers (BM25, TF-IDF, DTR, SGPT, TableLlama) across multiple K values and both 2-table and 3-table splits. The ablation removing question decomposition (w/o QD) provides at least partial mechanistic evidence.

- **Table-length elbow point finding is practically informative**: The observation in Figure 4 that QA performance drops sharply past ~800 rows while Text-to-SQL remains more stable (because SQL operates on headers, not content) is a concrete and useful empirical finding for practitioners designing LLM-based table systems.

- **Both 2-table and 3-table difficulty tiers are included**: Reporting results separately for 2-table and 3-table subsets gives granularity about difficulty scaling, a design choice most QA benchmarks omit.

## Weaknesses

### Fatal
*(None that would completely invalidate the paper's existence as a benchmark resource, but there is one structural methodological error that severely undermines the text-to-SQL contribution.)*

### Major

- **Text-to-SQL is evaluated with ROUGE/BLEU, which are inappropriate metrics for SQL.** This is a field-level standard violation. ROUGE and BLEU measure lexical overlap; semantically equivalent SQL queries can have low lexical overlap, and lexically close but semantically wrong queries score highly. The text-to-SQL community uniformly uses execution accuracy and/or exact-set-match (EX/EM over query result sets) as primary metrics—see Spider, BIRD, and Spider 2.0. Since Text-to-SQL is positioned as one of the benchmark's four core evaluation tasks, this metric choice renders conclusions about model SQL capability on MMQA unreliable. All comparative statements about model SQL ability (Tables 5 and 6) rest on these flawed metrics and should not be trusted.

- **MTR's headline result is confounded by GPT-4-turbo question decomposition not available to baselines.** MTR uses GPT-4-turbo as its first-stage question decomposer. Baselines like BM25, DTR, SGPT, and TableLlama receive no equivalent decomposition assistance. The ablation "w/o QD" confirms that removing decomposition significantly hurts performance—which means a large fraction of MTR's gain may come from the decomposer, not from the retrieval algorithm itself. The paper does not compare against a baseline that also uses GPT-4-turbo decomposition but uses a simpler retrieval strategy. Thus the "SOTA" claim for MTR as a retrieval method is not cleanly supported.

- **Data source inconsistency undermines benchmark clarity.** Section 3.1 explicitly states MMQA is built over Spider database ("We develop the MMQA benchmark over Spider database"). Yet Table 2 (benchmark comparison table) lists MMQA's data source as "Wikipedia." This is not a parser artifact; Spider databases themselves come from many crowdsourced domains, not primarily Wikipedia. This ambiguity affects readers' understanding of the benchmark's provenance and contamination risk.

- **Human performance comparison is incompletely specified and partially reported.** Tables 5 and 6 show human performance only for PM (QA) and two text-to-SQL metrics, leaving EM and other columns blank. The paper states annotators' agreement is "used as human performance," but annotators who created the data have advantages not available to models (familiarity with schema, knowledge of annotation conventions). The protocol for human evaluation under matched task conditions is not described, weakening the central human-vs-LLM finding.

### Minor

- **GPT-4-turbo is used for question generation, question decomposition in MTR, and partial-match evaluation of other LLMs including GPT-4.** This creates multiple circular dependencies. Most critically, using GPT-4-turbo to evaluate GPT-4's partial-match scores with no calibration against human judgment is a validity concern. No validation of the GPT-4-turbo judge against human ratings is provided.

- **Dataset size discrepancy is unexplained.** The paper states 5,000 samples were selected from Spider, but Table 1 lists 2,591 (2-table) + 721 (3-table) = 3,312 items. The filtering criteria responsible for the ~34% reduction in samples are not described.

- **Table-table relevance in MTR is a coarse binary column-name overlap heuristic.** The β score is 1 if any column name exactly matches between tables, 0 otherwise. This will fail when join keys are semantically related but have different names (e.g., "Employee_ID" vs. "emp_id"), a common real-world schema issue. The paper does not test MTR under such conditions, limiting generalizability claims.

- **The "elbow at 800 rows" finding is based on only 50 samples per bucket with no confidence intervals.** While the pattern in Figure 4 is suggestive, the small per-bucket sample size and absence of statistical testing mean this finding is descriptive rather than statistically established.

- **End-to-end pipeline evaluation is absent.** MTR retrieval and downstream QA/SQL tasks are evaluated independently. The paper does not show that better MTR retrieval causally leads to better downstream QA, leaving the utility of the retrieval improvement for the full pipeline unestablished.

### Trivial

- The "reasoning steps" complexity metric in Figure 3 is defined only implicitly. Without a cross-dataset consistent counting protocol, the comparison to other benchmarks is illustrative rather than rigorous.

## Nice-to-Haves

- Replace ROUGE/BLEU for Text-to-SQL with execution accuracy computed over the Spider database instances (SQL is executable since the underlying databases are available). This would be a significant upgrade to validity.
- Add an ablation comparing MTR with GPT-4 decomposition against a simpler retriever also augmented with GPT-4 decomposition, to isolate the retrieval algorithm's contribution.
- Include a small-scale independent human evaluation (non-annotators) to establish a more realistic human baseline under controlled conditions.
- Investigate semantic similarity (embedding-based) for the table-table relevance score β instead of binary column-name overlap, to test robustness on non-obvious foreign key names.
- Provide an error analysis categorizing QA failures (wrong join reasoning, incorrect key identification, arithmetic errors) to make benchmark results more actionable.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **"Real-world framing" as a major weakness** (Harsh Critic): The critic argues that the benchmark fails to prove its questions reflect a "realistic user-query distribution." However, the paper's primary claim is about evaluation of multi-table reasoning, not about distributional realism of the question source. The Spider databases span 138 domains and the benchmark's construction is no worse than Spider itself on this axis. This criticism is scope creep.

- **Novelty claim is "too broad"** (Harsh Critic): The critic calls for a narrower novelty claim. While this is a fair writing suggestion, it is not a substantive weakness; multi-table + multi-hop + key selection evaluation is genuinely new as a packaged benchmark. Moved to Nice-to-Haves.

- **Counting protocol for "reasoning steps" across benchmarks is incomparable** (Harsh Critic): The reasoning step comparison is informal but is clearly presented as illustrative, not as a rigorous metric. This is a minor clarity issue, not a substantive weakness.

- **Weakness about the benchmark not isolating "multi-table reasoning" vs. prompt overload** (Harsh Critic): This is a meta-criticism that would apply to virtually any benchmarking study; it is not specific enough to be actionable and does not invalidate the benchmark.

- **"Missing related work"** (Multiple reviewers): Per hard rules, missing related works are not flagged.

- **Reproducibility concerns about hyperparameters and temperature settings** (Harsh Critic, minor): Averaging three runs at temperature 0.7 and reporting ±standard deviation is reasonable practice for stochastic LLM evaluation. This does not merit a weakness.

## Novel Insights

The most genuinely novel empirical insight in this paper is the decomposition of table-length failure modes: the "elbow point" near 800 rows where multi-table QA performance drops sharply while Text-to-SQL remains more stable. The mechanistic explanation—that SQL generation attends only to table headers while QA requires attending to cell content—is a concrete, actionable finding that differentiates performance degradation modes across tasks. This insight is not present in prior table benchmarks and could guide future work on LLM table-processing architectures (e.g., selective cell retrieval vs. full table ingestion).

## Suggestions

1. **Fix the Text-to-SQL metric before submission**: Add execution accuracy as the primary metric. The Spider databases that underlie MMQA are available, making execution-based evaluation directly feasible.
2. **Add a "GPT-4-turbo augmented baseline" to Table 4**: A simple retriever (e.g., TableLlama) that also receives GPT-4-turbo question decomposition would clarify whether MTR's gains are from the algorithm or the decomposer.
3. **Resolve the data source inconsistency in Table 2**: Correct "Wikipedia" to "Spider (multiple crowdsourced domains)" or add a clarifying footnote.
4. **Extend human evaluation to matched-condition non-annotators**: Even 50 questions would substantially strengthen the human-vs-LLM comparison.
5. **Report the filtering criteria for the 5,000→3,312 reduction**: One paragraph in Section 3.1 would suffice.

---

## Score and Decision

**Calibration reasoning:**

- *Spider 2.0* (Oral, 8s): Uses real enterprise data, execution accuracy, novel realistic benchmark — significantly stronger than MMQA on all dimensions.
- *CofCA* (Poster, avg ~6): Multi-hop QA benchmark, accepted. Has some metric concerns (GPT-4 judge) but uses counterfactual data to address contamination. Comparable in concept to MMQA but somewhat stronger in evaluation rigor.
- *DSBench* (Poster, avg ~6.8): Real-world data science benchmark with multi-table tasks, accepted. Uses real data from competitions, stronger ecological validity.
- *MDBench* (Rejected, avg ~4): Synthetic multi-document benchmark. Rejected partly due to quality concerns and limited rigor, weakly similar to MMQA's template-based construction.
- *LAIA-SQL* (Rejected, avg ~5): Multi-table QA + text-to-SQL paper. Shares the SQL metric flaw and circular GPT-4 dependency with MMQA.

MMQA's gap: the benchmark idea is more complete than MDBench (more subtasks, larger scale, better annotation). However, it shares the LAIA-SQL problem of using BLEU/ROUGE for SQL (a material flaw) and a confounded retrieval comparison. The data source inconsistency and incomplete human evaluation further weaken confidence. The paper is more valuable as a benchmark contribution than as a methods paper, but the evaluation methodology holes are real.

Positioning: This paper sits between MDBench (rejected ~4) and CofCA (accepted ~6). The multi-table retrieval and PKS/FKS tasks are genuinely novel additions. The elbow-point finding is interesting. But the SQL metric flaw is more serious than any flaw in the accepted CofCA, and the retrieval comparison is genuinely confounded. I place this at **4.5**: better than a clear reject (there's a real benchmark contribution) but the methodological issues prevent acceptance without revision.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>