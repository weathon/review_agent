Now let me look at the calibration papers for scoring.## Summary
MMQA introduces a multi-table, multi-hop question-answering benchmark built from Spider relational databases, designed to evaluate LLMs across five tasks: multi-table retrieval, text-to-SQL generation, multi-table QA, and primary/foreign key selection. The paper also proposes MTR, an iterative retrieval method using GPT-4-turbo question decomposition combined with a fine-tuned table retriever. Experiments show a large human–LLM gap, with the best model (O1-preview) reaching ~51% exact match vs. ~90% human performance on the 2-table QA task.

---

## Claims and Support

**Claim 1: MMQA meaningfully evaluates multi-table, multi-hop reasoning beyond single-table benchmarks.**
- *Partially supported.* The benchmark does contain multi-table (2- and 3-table) examples with FK/PK annotations derived from Spider schemas, and Figure 3 reports 7.46 average reasoning steps vs. 2–6 for prior datasets. However, the methodology for counting reasoning steps is not defined in the main paper, and the questions are synthesized from 45 manually crafted SQL templates with no analysis of template diversity, linguistic naturalness, or heuristic answerability. The multi-table structure is real; the "genuine multi-hop reasoning" claim is asserted rather than validated.

**Claim 2: MMQA provides a comprehensive evaluation framework across multiple aspects.**
- *Partially supported.* The paper covers retrieval, QA, text-to-SQL, and key selection. However, text-to-SQL is evaluated with Rouge-1, Rouge-L, and BLEU—surface string metrics that are not valid measures of SQL correctness. The QA "Partial Match" uses GPT-4-turbo as an opaque judge without calibration or inter-annotator agreement. The framework is broad but not uniformly rigorous.

**Claim 3: Current LLMs remain far behind humans on multi-table reasoning.**
- *Partially supported.* Numerical gaps are large (O1-preview ~51% EM vs. ~90% human PM). However, human performance is computed from the same two experts who annotated the data (§3.1: "We compute the results of experts as human performance"), which inflates human scores through construction-phase familiarity. The relationship between inter-human agreement (82–86%, Table 1) and the high human task scores in Tables 5–6 is unexplained. The directional conclusion is plausible, but the magnitude is not securely established.

**Claim 4: MTR achieves SOTA performance on MMQA.**
- *Partially supported.* Table 4 shows MTR (TableLlama-7b) outperforms all tested baselines. However, MTR uses GPT-4-turbo for upstream question decomposition; baselines receive no equivalent assistance. The gain is benchmark-internal (the dataset is introduced in the same paper). "SOTA" should be scoped to "best among tested methods on MMQA."

**Claim 5: Question decomposition is a key contributor to MTR performance.**
- *Well supported.* The w/o QD ablation in Table 4 shows clear drops in all P/R/F1 metrics (e.g., Top-2 2-table precision drops from 72.3 to 65.3), providing direct evidence.

**Claim 6: An "elbow point" around 800 rows substantially degrades LLM performance.**
- *Partially supported.* Figure 4 shows a visual inflection around 800 rows across four models on both 2-table and 3-table subsets. However, each bucket contains only 50 samples, there are no error bars or significance tests, and table length may be confounded with domain/schema complexity. The pattern is suggestive but not a robustly established breakpoint.

---

## Strengths

- **Targets a real and important gap.** Most tabular benchmarks are single-table; real-world relational databases require multi-table operations. MMQA is the first to comprehensively address this with FK/PK annotations, a retrieval task, and multi-hop QA in a single evaluation suite.
- **FK/PK annotation layer.** Annotating primary and foreign keys per table is genuinely useful and under-explored in prior benchmarks, enabling column-level schema understanding evaluation.
- **Clear ablation on question decomposition.** The w/o QD row in Table 4 provides a clean, interpretable contribution of the decomposition module to retrieval quality.
- **Table-length analysis.** The observation in Figure 4—that QA performance drops sharply past ~800 rows while text-to-SQL degrades more gracefully—is a practical insight, even if exploratory.
- **Multi-granularity evaluation design.** Evaluating at table (retrieval), column (key selection), and cell (QA answer) levels provides complementary diagnostic information about where models fail.

---

## Weaknesses

### Fatal
*(None that independently invalidate the entire benchmark, but the two major issues below together seriously undermine the paper's value as a reliable evaluation artifact.)*

### Major

- **Text-to-SQL evaluation is methodologically invalid (§3.3, §4.1, Tables 5–6).** The paper uses Rouge-1, Rouge-L, and BLEU over SQL strings to measure SQL generation quality. These metrics are not valid for SQL: semantically equivalent queries can differ completely in surface form, and syntactically similar strings can produce wrong results. Execution Accuracy (EX) is the standard on Spider-derived benchmarks and is required to support any claim about SQL generation quality. Because text-to-SQL is one of five headline evaluation dimensions, this is not a minor metric choice—it renders one of the benchmark's key axes uninterpretable.

- **Data contamination from Spider is unaddressed.** The entire MMQA dataset is constructed from Spider databases and SQL queries, which have been publicly available since 2018. Many of the evaluated LLMs (GPT-4, GPT-3.5, O1-preview) were almost certainly trained on Spider. The paper does not partition examples by Spider train/dev/test splits, analyze overlap, or present any contamination analysis. All performance numbers—particularly for text-to-SQL—may be inflated by memorization of Spider schemas and queries, undermining the benchmark's validity.

- **Human evaluation protocol is inflated by construction-phase familiarity.** Human performance is computed from the same two experts who annotated the dataset and wrote the questions (§3.1). These annotators have intimate familiarity with the templates and schemas, making their "performance" scores incomparable to model evaluations. A proper human baseline would require held-out annotators who did not participate in construction. The headline gap (e.g., 50% vs. 90%) is likely overstated.

- **MTR comparison is confounded; "SOTA" claim is overstated.** MTR uses GPT-4-turbo for question decomposition while no baseline receives equivalent assistance. It is not possible to attribute the retrieval gains to the MTR retrieval formulation versus the upstream LLM decomposition. Including a decomposition-only baseline or giving baselines the same sub-questions would be necessary to isolate the contribution.

### Minor

- **Template-based synthesis lacks diversity/naturalness validation.** Questions are generated from 45 manually crafted SQL templates via GPT-4-turbo. No analysis of template distribution, lexical diversity, or whether questions can be answered by schema-pattern matching rather than genuine reasoning is provided. This raises the concern that the benchmark tests template familiarity rather than compositional reasoning.

- **Binary β table-relevance mechanism is brittle.** MTR computes table-table relevance β as 1/0 based on exact column-name overlap. This fails for semantically linked columns with different names (e.g., `student_id` vs. `stu_id`) and stops iteration prematurely when β=0, potentially missing valid joins via aliased or renamed keys.

- **Dataset size terminology is misleading.** The paper says "3,312 tables in total" (§3.1 line 147) but Table 1 lists "Total Tables" as 2591 and 721, which are actually the number of 2-table and 3-table question instances (confirmed by §4.1: "2591 samples"). The unit of counting shifts confusingly between "tables" and "samples."

- **Internal inconsistency in Section 4.2 commentary.** The text states "MTR combined with TableLlama-7b while showing commendable recall with a score of 64.7%, lagged in precision"—but Table 4 shows MTR (TableLlama-7b) achieves precision 72.3% and recall 64.7%. It is recall that is lower, not precision. This reversal suggests unreliable result interpretation in the main analysis.

### Trivial

- **Temperature 0.7 for benchmark evaluation.** Using temperature 0.7 for proprietary models introduces avoidable generation variance in a capability-evaluation setting. Temperature 0 is conventional.
- **Reasoning steps methodology not defined in main text.** Figure 3 reports step counts but the counting procedure is not explained in the main paper (possibly in the appendix).

---

## Nice-to-Haves

- **Evaluate scalability beyond 3 tables.** Real-world schemas routinely involve 5–10+ tables; testing 4-table and 5-table settings would strengthen the "reflects real-world complexity" claim.
- **Error breakdown by failure mode.** Categorizing errors by failure type (wrong key identification, wrong reasoning order, retrieval failure, calculation error) would provide actionable insights for future model development.
- **Extend the long-table analysis.** The average row count in the 2-table subset is 1833, yet Figure 4 only shows results up to 1000 rows. Extending to the realistic full range would validate the "elbow at 800" observation more robustly.
- **LLM-as-retriever baseline.** A simple baseline of prompting GPT-4 to directly select relevant tables from schemas—without MTR's machinery—would serve as the natural missing comparison.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"MTR gains unfair because MTR uses a stronger model"** (from harsh reviewer's framing of asymmetric comparison): The asymmetry here disfavors baselines (not the authors), but in this case the confound actually inflates the authors' method, so the criticism is retained above rather than removed under that rule.
- **Concern about 2-3 table limit being insufficient**: Weakened to nice-to-have since a first benchmark paper establishing the framework with 2–3 tables is a reasonable scope.
- **Speculation about MTR discussion attributing performance to "advanced feature engineering capabilities"**: Weak writing, but this is a presentation/framing issue rather than a factual error; removed as a pure stylistic nitpick.
- **Criticism that MMQA's data source is listed as "Wikipedia" in Table 2 while the paper says it's Spider/crowdsourced**: This is likely inherited metadata from Spider's original Wikipedia-derived content; while confusing, it is not a paper-level factual error and was removed as a minor table formatting artifact.

---

## Novel Insights

The paper surfaces a practically useful and underexplored phenomenon: LLM performance on multi-table QA drops sharply past ~800 table rows, while text-to-SQL generation degrades more gradually—consistent with the hypothesis that SQL generation relies primarily on schema headers (short, invariant context) whereas cell-level QA must process full row content. This differential degradation pattern, if validated at scale, has direct implications for how practitioners should think about context-window limitations in database-linked LLM applications.

---

## Suggestions

1. **Replace Rouge/BLEU with Execution Accuracy for SQL.** This is the most critical fix. Without it, the text-to-SQL dimension of the benchmark is uninformative.
2. **Conduct a contamination analysis.** Partition MMQA by Spider train/dev/test splits and report whether LLM performance differs across partitions. Report SQL template origin for each MMQA instance.
3. **Use non-author annotators for human baseline.** Recruit CS-background annotators who did not participate in construction to establish a credible human performance ceiling.
4. **Provide a decomposition-only retrieval baseline.** Give all baselines the same GPT-4-turbo decomposed sub-questions as MTR and report their retrieval performance to isolate what MTR's retrieval formulation actually contributes.
5. **Report question diversity metrics.** Template coverage (number of templates per split), n-gram diversity, and a qualitative sample analysis would support the "genuine multi-hop" claim.
6. **Extend long-table analysis to full dataset range with bootstrapped CIs.** At minimum, extend Figure 4 to 1800+ rows and report bootstrap confidence intervals.

---

## Score and Decision

**Calibration anchors:**

| Paper | Type | Decision | Scores |
|---|---|---|---|
| Spider 2.0 (XmProj9cPs) | SQL benchmark, real data, proper metrics | Accept Oral | 8,8,8,8 |
| Chain-of-Table (4L0xnS4GQM) | Table reasoning method, solid empirical results | Accept Poster | 5,6,5,6 |
| MDBench (KNkalZnq3f) | Synthetic benchmark, limited quality validation | Withdrawn/Reject | 5,3,3,5 |
| TAMO (Mi45HjlVRj) | Table reasoning method, incremental novelty | Reject | 5,5,5,8 |

**Comparison reasoning:**

- MMQA is more novel in its framing than MDBench (genuine FK/PK multi-table gap, a real underexplored area) and has a more coherent evaluation framework with an ablatable method component. This places it above MDBench's average score (~4).
- However, MMQA falls substantially below Spider 2.0 on rigor: Spider 2.0 uses real enterprise data, execution-based metrics, proper curation, and validated human evaluation. MMQA uses a synthetic template pipeline, wrong SQL metrics, and inflated human scores.
- Compared to Chain-of-Table (5–6 range), MMQA has a more important research question but weaker methodology and a key metric flaw in what is supposed to be a benchmark paper. Benchmark papers are held to a higher standard on metric validity than methods papers.
- The combination of (1) invalid SQL metric for a benchmark paper, (2) unaddressed Spider contamination, and (3) inflated human baseline places this paper below the accept threshold but above outright rejection.

**Score: 4.5 — Reject**

The contribution addresses a genuinely important gap and the multi-table retrieval experiments are the strongest part. However, a benchmark paper that uses surface string metrics to evaluate SQL correctness has a structural flaw at its core, and the unaddressed contamination concern undermines the validity of all downstream performance conclusions. These are fixable in a major revision but are not minor in the current submission.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>