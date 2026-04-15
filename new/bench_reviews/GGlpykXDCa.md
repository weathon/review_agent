Now let me search for calibration papers.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

MMQA introduces a multi-table, multi-hop question answering benchmark built atop Spider databases, together with a comprehensive evaluation framework spanning five tasks (multi-table retrieval, text-to-SQL, QA, primary key selection, foreign key selection) and a novel iterative retrieval method (MTR). The benchmark includes 3,312 tables covering both 2-table and 3-table subsets, and experiments across proprietary and open-source LLMs reveal substantial human–model performance gaps, motivating this as a challenging new testbed for multi-table relational reasoning.

---

## Claims and Support

| Claim | Verdict | Notes |
|---|---|---|
| MMQA fills a gap for multi-table multi-hop QA evaluation | **Partially supported** | The benchmark covers real multi-table tasks, but is synthetic (45 SQL templates + GPT-4 paraphrase); no validation that diversity reflects real-world use |
| Benchmark reflects real-world complexity better than single-table benchmarks | **Unsupported** | Spider-derived, template-generated benchmark ≠ real-world. The jump from "multi-table joins exist" to "mirrors real-world" is not evidenced. |
| "First" multi-table multi-hop QA benchmark | **Partially supported** | Novel combination of tasks is plausible, but the "first" claim is too aggressive given cited prior multi-table work |
| Higher average reasoning steps (7.46) shows greater complexity | **Partially supported** | Numerically true, but "reasoning steps" is not defined rigorously across datasets; likely reflects template design rather than independently validated difficulty |
| Comprehensive evaluation across retrieval, QA, SQL, key selection | **Well-supported** | Table 4–6 present all five sub-task results |
| Current LLMs substantially lag human performance | **Partially supported** | Tables 5–6 show large gaps, but human protocol is underspecified and SQL evaluated via Rouge/BLEU rather than execution accuracy |
| MTR achieves SOTA vs. listed baselines on MMQA | **Partially supported** | Table 4 shows MTR (TableLlama-7b) best among listed methods; but the analysis section contains a factual contradiction (see below), and ablations do not isolate the table-table relevance component |
| Question decomposition (QD) is vital for MTR | **Well-supported** | The w/o QD ablation shows clear drops across metrics |
| Joint question-table + table-table relevance is why MTR works | **Unsupported** | No ablation removes the β (table-table) term independently; improvement attributed to joint scoring without evidence |
| LLM performance drops sharply after 800 rows (elbow) | **Partially supported** | Visual pattern in Figure 4 is consistent; but based on only 50 samples per bucket with no error bars or controls |

---

## Strengths

- **Genuine gap in the literature addressed**: MMQA is the first benchmark to jointly evaluate multi-table retrieval, QA, text-to-SQL, primary key selection, and foreign key selection in a single framework. No prior work covers this combination with explicit PK/FK annotation.

- **Human–model gap is clear and substantial across all tasks**: O1-preview achieves only ~50.78 EM vs. human 89.8 on 2-table QA (Table 5); ~49.53% accuracy on primary key selection vs. human 96.5%. The gap is consistent across 2-table and 3-table subsets, providing strong evidence that the benchmark is non-trivial.

- **MTR ablation establishes the value of question decomposition**: The w/o QD row in Table 4 shows meaningful improvements (e.g., Top-2 precision: 65.3→72.3, recall: 62.3→64.7), and this is the cleanest empirical result in the paper.

- **Multi-granularity task design**: Evaluating at table level (retrieval), column level (PKS/FKS), and cell level (QA) in one unified framework is a principled design choice that goes beyond single-metric benchmarks.

---

## Weaknesses

### Fatal
*(None — the paper presents a functional benchmark and genuine results; no single issue invalidates all claims.)*

### Major

- **Internal contradiction in MTR analysis (factual error in the paper's own results)**: Section 4.2 explicitly states that MTR(TableLlama-7b) "lagged in precision, indicating a tendency to retrieve a broader set of tables," yet Table 4 shows MTR(TableLlama-7b) achieves **72.3% precision** at Top-2, which is the *highest* precision of any method. This is not an ambiguous misinterpretation — the text directly contradicts the numbers presented one paragraph above it. This calls into question the reliability of the qualitative analysis throughout Section 4.

- **Text-to-SQL evaluated with lexical metrics (Rouge/BLEU) instead of execution accuracy**: This is below the current standard for the text-to-SQL community (Spider and BIRD both use execution accuracy as the primary metric). Rouge/BLEU penalizes semantically identical SQL with different syntax (different join orderings, column aliasing, subquery vs. JOIN equivalents) while potentially rewarding syntactically similar but semantically wrong queries. The paper's conclusions about text-to-SQL capability cannot be reliably drawn from these metrics alone.

- **Incomplete ablation — β (table-table relevance) never isolated**: The entire motivating mechanism for MTR is the *joint* consideration of question-table relevance α and table-table relevance β (Eq. 1). The only ablation removes question decomposition (QD), which is a separate component. There is no experiment removing β alone, no experiment comparing to decomposition-only baseline, and no experiment testing whether the column-overlap heuristic matters vs. random linking. The claim that the joint α·β scoring drives gains is therefore entirely unsupported by evidence.

- **GPT-4-turbo as both question generator and answer evaluator is a circular setup**: GPT-4-turbo is used to generate the NL questions from SQL templates, and the same model family is used as the Partial Match (PM) evaluator for QA answers. This creates potential evaluator bias — GPT-4-based models may be systematically evaluated more favorably because the evaluation prompt echoes the generation style. No calibration against human judgments is provided for PM scores.

### Minor

- **Benchmark construction validation is absent**: There is no analysis of the distribution across 45 SQL templates (which templates are overrepresented?), no check for schema-name artifacts that could trivialize join discovery (e.g., foreign keys are obvious from column name overlap), and no study of whether the generated questions are natural beyond the 82–86% inter-annotator agreement on key annotations. The inter-human agreement reported is specifically on PK/FK annotation, not question naturalness or reasoning validity.

- **"5,000 samples" vs. "3,312 tables" discrepancy**: Section 3.1 states "we randomly select a total of 5,000 samples from Spider," but Table 1 and later text report 3,312 tables total (2,591 + 721). The unit of measurement (samples vs. table groups vs. questions) is never reconciled, making the benchmark's precise size and composition unclear.

- **"Reasoning steps" is undefined across datasets**: Figure 3 claims MMQA has 7.46 average reasoning steps vs. 3.15 for HybridQA. The method for computing this metric is not described in the main text, and it is unclear whether the same procedure is applied to all datasets or whether MMQA's template-based construction inflates this measure by design.

- **Temperature 0.7 for benchmarking proprietary models**: This is an unusual choice for evaluation tasks where determinism is preferred. The authors report averages of 3 runs, which partially mitigates this, but for key selection and exact match tasks, variance from temperature should be minimized rather than averaged out.

### Trivial

- **Human performance protocol is underspecified**: Tables 5–6 show human scores (89.8 PM, 82.7 for SQL, 96.5/95.3 PKS/FKS) but the description of who did it, how many annotators, and how conflicts were resolved is too brief to interpret or reproduce the baseline.

---

## Nice-to-Haves

- Adopt execution accuracy (EX) as the primary text-to-SQL metric, supplementing with partial credit metrics; this would make results comparable to Spider/BIRD community standards.
- Ablate the β term (table-table relevance) independently, and test whether column-name overlap vs. learned schema linking drives performance; this would considerably strengthen the methodological contribution.
- Add an end-to-end evaluation where MTR retrieved tables are piped directly into QA/SQL; disjoint evaluation obscures real error propagation.
- Provide a short error taxonomy (wrong join detection, wrong aggregation, value hallucination) to help characterize model failure modes.
- Replace the framing of "real-world complexity" with "synthetic stress test for compositional multi-table reasoning" — a more defensible and still compelling framing.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED — Scope creep] "The benchmark only covers Spider databases, which are clean and professionally curated; real-world tables are noisy."** Valid as a nice-to-have future direction, but a paper introducing a *benchmark* is not obligated to cover all sources of real-world noise in its first release. Evaluating against this standard is scope creep.

- **[REMOVED — Generic / applies to any benchmark paper] "The paper is well-written and the topic is important."** Removed as a non-specific strength.

- **[REMOVED — Reproducibility nitpick] "Complete prompts and hyperparameters not all disclosed."** The paper points to Appendix B for prompts. Requesting full disclosure of all hyperparameters for proprietary APIs is an impractical demand.

- **[REMOVED — Overclaim about "not yet released" models] "O1-preview does not correspond to currently available systems."** The paper cites O1-preview as a tested model; per hard rules, if it is cited it exists. This is a reviewer knowledge gap, not an author error.

- **[REMOVED — Generic weakness] "Requesting larger dataset when current size is sufficient."** The dataset size (3,312 table groups) is comparable to many accepted benchmark papers and is not a core weakness.

---

## Novel Insights

The most genuinely useful observation across all reviewers is the **elbow-point finding at 800 rows** (Figure 4): QA performance degrades gently below this threshold but drops sharply above it, while text-to-SQL shows a steady slow decline regardless of table length. This suggests LLMs process QA and SQL through qualitatively different mechanisms — QA requires reading table content and hits a retrieval/attention limit, while SQL generation primarily attends to schema headers and degrades linearly. If validated with more samples and statistical tests, this is an actionable finding for deployment and model development. However, the current evidence (50 samples per bucket, no error bars) is preliminary.

---

## Evaluation on Key Axes

- **Novelty**: Moderate. Multi-table benchmark with the specific 5-task combination is new; MTR is an incremental retrieval method combining existing components (decomposition + single-table retriever + column overlap). The novelty is primarily benchmark design, not method.
- **Technical soundness**: Below standard. The factual contradiction in Table 4's analysis, the absence of the β-term ablation, and the use of Rouge/BLEU for SQL evaluation are substantive technical problems.
- **Empirical support**: Moderate. Human–model gaps are clear and consistent. Retrieval results are meaningful. But the MTR mechanistic claim and the real-world representativeness claim are under-supported.
- **Significance**: Moderate. A multi-table benchmark is a useful community resource if valid; the 5-subtask framework is a genuine design contribution.
- **Clarity**: Weak in key places. The Table 4 contradiction, the "5,000 vs. 3,312" discrepancy, and the undefined reasoning-steps measure reduce trust in the exposition.

---

## Calibration

**Papers compared:**
1. **MDBench** (KNkalZnq3f, avg 4.0, Rejected): Also a synthetic benchmark with LLM-generated questions, similar overclaiming of "real-world" representativeness, similar lack of distribution validation. MMQA is slightly stronger due to having a proposed method (MTR) and a 5-task framework.
2. **iSTMsye6SD** (programmatic KG benchmark, avg 5.25, Rejected): Programmatically generated, limited real-world validation, presented to evaluate LLM reasoning. Similar profile to MMQA.
3. **DSBench** (DSsSPr0RZJ, avg 6.8, Accepted): Uses genuinely real-world data (Kaggle, ModelOff), no synthetic generation, comprehensive evaluation — significantly stronger than MMQA in dataset realism.
4. **LAIA-SQL** (WYdpjwKQma, avg 5.0, Rejected): Multi-table NL2SQL benchmark+method, similar evaluation issues.

MMQA sits between MDBench (too weak, 4.0) and iSTMsye6SD (5.25) on benchmark validity, but is weakened by the factual error in its own analysis and missing ablations. The paper deserves credit for having a cleaner multi-task framework than MDBench. I place it at **4.5**.

---

## Score and Decision

**Score: 4.5** — The paper addresses a real gap and contains useful benchmarking material, but the combination of (a) the internal contradiction in Table 4 analysis, (b) below-standard SQL evaluation metrics, (c) absent ablation of the MTR's core claimed mechanism, and (d) overclaiming benchmark realism without validation places this below the ICLR acceptance bar. A revision addressing these issues — especially adopting execution accuracy, adding the β-term ablation, and fixing the Table 4 analysis — could bring this to a borderline accept.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>