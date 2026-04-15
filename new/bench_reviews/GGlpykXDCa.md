Now let me look at the human reviews for calibration.Now let me do a targeted search to anchor the score better.Now I have enough information to write the final consolidated review. Let me synthesize all findings.

---

## Summary

This paper introduces MMQA, a multi-table multi-hop question answering benchmark built atop the Spider database, with questions generated via 45 hand-crafted SQL templates and GPT-4-turbo paraphrasing. The evaluation framework spans Multi-Table Retrieval, Text-to-SQL Generation, QA, Primary Key Selection, and Foreign Key Selection. The authors also propose MTR, an iterative multi-table retrieval method combining question decomposition with a table-table relevance score. Experiments across open-source and proprietary LLMs reveal a substantial gap behind human performance.

---

## Strengths

- **Multi-granularity evaluation design**: The paper's task suite spans table-level (retrieval), column-level (PK/FK selection), and cell-level (QA) evaluation. This is a concrete, non-trivial decomposition that enables fine-grained diagnosis of where LLMs fail—not just *that* they fail. Most existing tabular benchmarks (WikiTableQuestions, WikiSQL, Spider) do not offer this coverage.
- **Table length impact analysis**: The elbow-point finding in Figure 4—QA performance degrades sharply past 800 rows while Text-to-SQL declines gently—is an interesting and plausibly interpretable empirical finding. The explanation (SQL relies on schema headers rather than content) is testable and actionable.
- **Breadth of LLM evaluation**: The paper evaluates open-source and proprietary models across both 2-table and 3-table subsets, with three-run averages and standard deviations. The inclusion of O1-preview as the strongest system sets a useful upper bound.
- **Ablation confirming Question Decomposition (QD)**: Table 4's "w/o QD" variant demonstrates a meaningful contribution from the decomposition module (72.3 → 65.3 precision in Top-2 2-table), providing direct evidence for a design choice.

---

## Weaknesses

### Fatal

*(none that single-handedly invalidate all contributions, but the first major weakness below critically undermines a primary task of the framework)*

### Major

- **Text-to-SQL is evaluated with ROUGE/BLEU rather than execution accuracy — making all SQL conclusions invalid.** The paper explicitly uses Rouge-1, Rouge-L, and BLEU to evaluate SQL generation (Section 3.3: *"we utilize Rouge-1, Rouge-L (Lin, 2004), and BLEU (Papineni et al., 2002) to evaluate the LLM-generated SQL query quality"*). These lexical-overlap metrics are fundamentally inappropriate for SQL: semantically equivalent queries can differ entirely in surface form (aliases, ordering, join direction), while syntactically similar strings may be logically wrong. Conclusions in Section 4.3—e.g., "O1-preview's sophisticated language parsing and structured output generation capabilities"—are drawn from BLEU/ROUGE differentials and cannot be trusted. Execution accuracy (EX) or test-suite accuracy is the community standard for Spider and BIRD and its absence is a critical gap, not a minor oversight.

- **MTR's table-table relevance mechanism directly contradicts the paper's motivating claim about FK/PK-aware relational reasoning.** The paper opens with the importance of identifying primary and foreign keys for multi-table traversal. Yet the β score in Eq. (1) is a binary indicator based solely on column-name overlap (Section 3.2: *"If the retrieved tables overlap columns with tables retrieved in the previous round, then assign a score of 1 and 0"*). Column-name overlap is not a proxy for foreign-key semantics: common column names (e.g., "id", "name") create spurious links, while genuine FK joins with different header names (e.g., `stu_id` → `student_id`) are missed. The paper annotates PK/FK explicitly but does not use these annotations in MTR. This conceptual mismatch between the stated motivation and the actual implementation significantly weakens the core method claim.

- **The "reasoning steps" metric used to establish benchmark complexity is undefined and unvalidated.** Figure 3 presents MMQA as having 7.46 average reasoning steps versus 2.14–6.28 for other datasets, which is deployed as key evidence that MMQA is substantially harder. However, no protocol is given for how reasoning steps are counted or standardized across datasets. For MMQA, if steps are derived from SQL template structure, the counts may be an artifact of generation rather than a valid difficulty measure. No inter-annotator agreement for step counts is reported, and no correlation between step count and empirical model difficulty is shown.

- **The paper does not specify whether downstream QA and SQL tasks are evaluated on gold tables or retrieved tables.** This ambiguity is consequential: if downstream tasks receive gold tables, reported QA/SQL scores overstate model capability by bypassing retrieval errors; if retrieved tables are used, propagation of retrieval errors is not analyzed. The paper (Figure 2, Section 4.1) is silent on this point, leaving the two-step evaluation pipeline unspecified.

- **Human evaluation is underspecified and incomparable across metrics.** The abstract and conclusion strongly foreground the human/LLM gap, but Tables 5 and 6 show human results only for PM (not EM for QA), only one SQL metric, and omit the evaluation procedure entirely: number of annotators, interfaces provided, whether annotators had access to full schemas, and how the human EM would look. The claim that "LLMs leave significant performance room for improvement" is directionally plausible but not rigorously demonstrated at the metric level.

### Minor

- **Narrative in Section 4.2 is internally inconsistent with the data.** The text states: *"the MTR combined with TableLlama-7b while showing commendable recall with a score of 64.7%, lagged in precision."* But Table 4 shows MTR(TableLlama-7b) has Top-2 precision of 72.3—the highest value in the table. This appears to be an analysis copy-paste error.

- **GPT-4-turbo serves three conflicting roles: question generator, QD module in MTR, and answer evaluator for PM.** This creates a potential circularity loop (the generator may embed stylistic biases that inflate GPT-4-turbo downstream performance). No ablation with an alternative evaluator or human-calibration study for the PM metric is provided.

- **The table length analysis is underpowered.** Only 50 samples per bucket, no confidence intervals, and no control for confounding variables (question type, domain, or schema complexity) across length buckets. The "elbow point" at 800 rows is presented definitively but the statistical warrant is thin.

- **Algorithm 1 contains confusing notation.** γ is used both as an initialization and as a scalar in the update step `γ ← γ + α(q₀, table₀)` and then later `ArgSort(γ)` as if it is a vector. This makes the method unreproducible as described.

### Trivial

- The benchmark covers only 2- and 3-table scenarios. While a step beyond single-table, real-world schemata frequently involve larger join chains. The scope limitation itself is acceptable for a first benchmark but should be explicitly stated as a limitation rather than implicitly overstated as "multi-table."

---

## Nice-to-Haves

- Add execution accuracy or exact set match as the primary SQL metric; this alone would substantially improve the paper's credibility.
- Provide a full end-to-end evaluation (retrieved tables → QA/SQL) alongside the gold-table setting for a cleaner empirical story.
- Ablate table-table relevance via PK/FK annotations vs. column-name overlap to test whether relational structure matters beyond surface similarity.
- Validate PM scoring (GPT-4-turbo judge) against human agreement on a 100-sample subset.
- Include error analysis categorized by question type and failure mode (retrieval error, key mismatch, SQL error, etc.).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"SOTA claim is too strong"** (Harsh Critic, Claim 5): On the paper's own benchmark, claiming SOTA among evaluated baselines is standard practice. The claim is mildly overstated in framing but does not reflect a methodological error.
- **Reproducibility concerns about hyperparameters and training logs**: Removed per hard rule on trivial implementation details.
- **Missing related work on HybridQA/OTT-QA retrieval baselines** (Spark): Removed per hard rule—cannot verify existence of cited works independently.
- **Generic strength "addresses an important gap"**: Removed per hard rule—this is true of any benchmark paper.
- **Generic strength "the paper is well-written/comprehensive"**: Removed per hard rule.
- **Data contamination from Spider**: Flagged but removed as a primary weakness—this is a known limitation of any Spider-based benchmark and all similar benchmarks (HoloBench included) share it. It is a real concern but standard to the field and should be disclosed as a limitation, not a blocking flaw.

---

## Novel Insights

The most genuinely novel observation surfacing from this review is the elbow-point finding in Figure 4: QA performance degrades sharply past ~800 rows while Text-to-SQL declines gradually, suggesting that multi-table QA and SQL generation have qualitatively different failure modes under long-context conditions—cell retrieval versus schema-reading. This aligns with the intuition that SQL generation relies on header structure (which scales gracefully) while QA requires row-level lookup (which degrades with table length). This finding, if replicated with better-controlled experiments, would be a useful contribution independent of the benchmark construction choices.

---

## Suggestions

1. **Replace ROUGE/BLEU with execution accuracy for SQL**: This is the highest-priority fix; without it, the SQL subsection's findings cannot be trusted.
2. **Specify gold vs. retrieved tables in downstream evaluation**: Add a clear statement and ideally an end-to-end evaluation condition.
3. **Use annotated PK/FK links in MTR's β score**: The paper already has this annotation; integrating it into the method would close the gap between motivation and implementation.
4. **Report human EM for QA and human EM for SQL**, with a description of the annotation interface and sample size.
5. **Fix Algorithm 1 notation** to make γ clearly a vector accumulator, not a scalar.

---

## Score and Decision

**Calibration against retrieved papers:**

| Paper | Score | Decision | Key similarities |
|---|---|---|---|
| MDBench (KNkalZnq3f) | 3–5, avg ~4 | Rejected | Synthetic benchmark from existing database, unvalidated complexity claims |
| HoloBench (5LXcoDtNyq) | 5–8, avg ~6.25 | Accepted | Built on Spider, evaluates LLMs on database operations, more rigorous experimental design |
| DivKnowQA (9AnR2z7iNL) | 3–5, avg ~4.3 | Rejected | Machine-generated QA benchmark, naturalness concerns |
| LAIA-SQL (WYdpjwKQma) | 3–8, avg ~5 | Rejected | Multi-table QA task decomposition, flawed evaluation |

MMQA sits closer to MDBench/DivKnowQA than to HoloBench on evaluation rigor. The key differentiators: MMQA has more models, a multi-faceted framework, and a proposed retrieval method with ablations—which MDBench lacks. But unlike HoloBench, MMQA's primary SQL evaluation metric is fundamentally inappropriate, its core method claim (relational FK/PK reasoning) is not matched by the implementation, and its benchmark complexity argument rests on an undefined metric. The paper's benchmark idea has genuine merit but the execution falls short of the level of rigor expected at ICLR. A score of **4.0** is appropriate—below the ICLR mean (~5.12), consistent with similar synthetic benchmark papers that were rejected, and reflecting that the work requires substantial methodological corrections before the central claims can be substantiated.

**Novelty:** Moderate — addressing a real gap but built on existing Spider schemas with template-based synthesis.
**Technical soundness:** Low — invalid SQL metric, conceptual mismatch in MTR's core mechanism.
**Empirical support:** Partial — retrieval and QA results are credible; SQL conclusions are not.
**Significance:** Moderate potential if methodological issues are fixed.
**Clarity:** Fair, with one notable internal inconsistency (Section 4.2 narrative vs. Table 4).

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>