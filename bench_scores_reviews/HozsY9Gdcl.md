## Summary
Set-MI is a framework for Membership Inference (MI) in Language Models that improves upon individual-document MI by aggregating per-document scoring signals over groups of documents that share a natural membership property (the "set assumption": documents grouped by creation date, language, license, or instruction dataset source are either all in or all out of the training set). Built on top of four existing Individual-MI methods, Set-MI is evaluated on five newly constructed benchmarks spanning Wikipedia, Arxiv, language identity, licensing, and instruction-tuning data, reporting an average AUROC improvement of 0.14. Ablations study the role of model size, deduplication, document length, set size, and aggregation strategy under simulated label noise.

---

## Strengths

- **Novel and diverse benchmark construction.** The paper constructs five domain-spanning benchmarks (Wikipedia, Arxiv, language identity, license category, instruction-tuning datasets) that did not previously exist as set-structured MI testbeds. This is a tangible, reusable contribution beyond the method itself.

- **Practically informative ablations.** The set-size ablation (Figure 4 right) demonstrates that even 3 documents per set yields meaningful gains over individual inference — a specific, actionable result for practitioners. Similarly, the finding that document length saturates around 256–512 tokens gives concrete guidance.

- **Novel finding on deduplication.** The analysis in Section 5.3 reveals that deduplication in training data impacts Set-MI substantially more than Individual-MI, a finding that carries real implications for understanding the relationship between training data processing and memorization.

- **Robustness analysis with verified labels.** Section 6 uses 13-gram overlap to verify actual membership against the Pile, providing a ground-truth reference point that is stronger than the proxy labels used in the main experiments. The controlled noise simulation comparing MAX/MIN/FULL aggregation under different noise configurations is a useful practical guide.

- **Modular and lightweight design.** The framework is plug-and-play on top of any existing Individual-MI method and requires no retraining or model access beyond document-level loss — a practically attractive property.

---

## Weaknesses

### Fatal
None.

### Major

- **Temporal distributional shift confound for Wikipedia and Arxiv (unaddressed).** The set assumption for these two benchmarks is defined by document creation date, which is the *same* attribute used to assign ground-truth labels. Lower model loss on pre-cutoff documents could reflect temporal language/topic shift — documents from 2010 have different statistical properties from documents from 2022 regardless of memorization. The paper never controls for this: there is no experiment comparing a model trained on a different temporal slice of the same data, nor any analysis separating distribution shift from memorization. Since Wikipedia and Arxiv are the primary benchmarks (1,000 sets each, featuring prominently in the abstract's 0.14 average AUROC claim), an uncontrolled confound here substantially weakens the attribution of improvement to MI signal aggregation.

- **Missing metadata-only baseline.** If the set grouping attribute (date, language, license) itself is sufficiently predictive of training membership — independent of any model loss signal — then aggregating over sets could trivially improve AUROC by exploiting metadata structure rather than a better MI signal. The paper never tests a baseline that predicts membership purely from set metadata without querying the model. Without this control, it is impossible to distinguish genuine MI signal amplification from set-structure information leakage. This is a fundamental gap in experimental design.

- **Proxy membership labels in main experiments.** For Wikipedia and Arxiv, ground-truth membership is assigned by creation date relative to the Pile's collection date, not by verifying whether each document actually appears in the Pile. The clean Wikipedia experiment in Section 6 — using 13-gram overlap to verify membership — achieves AUROC near 1.0 at zero noise, substantially higher than the proxy-label Wikipedia results in Table 2 (e.g., 0.575 for Loss Attack). This gap suggests that proxy labels may introduce substantial noise that bounds headline performance, and that the main results reflect label quality as much as MI difficulty. The main experiments would be substantially more convincing if at least one model/domain setting used verified labels.

- **Inconsistencies between Table 1 and benchmark descriptions.** The text for Wikipedia states "We subsample 100 sets with 100 documents per set," but Table 1 reports 1,000 sets and 100,000 documents. Arxiv has the same discrepancy. For Language, the text says "resulting in 130 sets" whereas 20 languages × 10 subsets = 200 sets (matching Table 1). For License, the text again says "resulting in 130 sets" while Table 1 reports 190 sets. These are not trivial discrepancies — they concern the fundamental scale of evaluation and are directly relevant to reproducibility.

### Minor

- **Document-level AUROC conflates set-level decisions.** The paper assigns the aggregated set score to every document in the set and evaluates AUROC over documents. Since all documents in a set receive identical scores, the effective number of independent decision points is the number of sets (e.g., 1,000 for Wikipedia), not the number of documents (100,000). Reporting set-level AUROC alongside document-level AUROC is needed for proper statistical interpretation, particularly for smaller benchmarks (Languages: 200 sets, Instructions: 130 sets).

- **Narrow scope of key ablations.** The deduplication ablation (Section 5.3) uses only Loss Attack on Wikipedia; the document-length ablation (Section 5.4) uses only LiRA on Wikipedia. Broader coverage across at least two domains and two MI methods would be needed to assert these findings generalize. As stated, these are findings specific to one benchmark × one method combination.

- **Robustness analysis restricted to a single setting.** Section 6 uses only Pythia 2.8B dedup, Wikipedia, and Loss Attack. The robustness claims in the abstract ("robust under practical settings") generalize from a single controlled configuration.

### Tiny

- Notation in Section 3: The formal set partition writes "$S_i, S_j \in \mathcal{D}$" but sets are not elements of $\mathcal{D}$ (documents are). Lower-case $s_i$ is also used inconsistently with upper-case $S_i$ in the same block.

- The abstract says Set-MI "enhances prior MI methods"; a small qualification is needed given that zlib entropy on Instructions decreases from 0.458 to 0.429 (Table 2). The text of Section 5.1 handles this correctly but the abstract does not.

---

## Nice-to-Haves

- **Adaptive aggregation selection**: Since practitioners typically lack prior knowledge of which sets are noisier (member or non-member), an automatic or heuristic strategy for selecting MAX/MIN/FULL without oracle knowledge would significantly increase practical utility.

- **Embedding-based set construction**: When explicit metadata is unavailable, clustering documents by semantic similarity to infer sets is a natural extension. A small experiment or discussion of this would help practitioners facing unstructured corpora.

- **Stratified analysis by set distance from membership boundary**: For date-based benchmarks, sets close to the cutoff date are much harder than sets far from it. Reporting AUROC stratified by temporal distance from the cutoff would reveal whether gains concentrate on easily separable sets, which would qualify the practical utility of the headline improvement figure.

- **Set-level AUROC as a complementary metric**: Reporting both document-level and set-level AUROC throughout would make the statistical interpretation of all tables and figures cleaner.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

**Removed criticisms:**

- *LiRA implementation is incorrect*: The paper adapts LiRA to a simplified token-probability ratio form for the black-box loss-only setting. This is a deliberate design choice given the access assumptions, not an implementation error. The paper is internally consistent about what access is assumed.

- *The "any method" framing is fundamentally misleading*: One counterexample (zlib/Instructions) exists and is visible in Table 2. The paper discusses in Section 5.1 that poor Individual-MI can lead to worse Set-MI. This is not a fatal framing issue but a minor qualification already partially handled.

- *Larger dataset sizes / more models in benchmark*: The dataset sizes (100,000 docs for Wikipedia/Arxiv, 20,000 for Languages) and model zoo (Pythia family, GPT-Neo, BLOOM, SILO, Tulu) are adequate for the evaluation claims. Demanding more is generic and not specific to a flaw in this paper.

- *Closed/commercial model evaluation (GPT-4, Claude)*: Testing on models with unknown ground-truth training sets would require speculative membership labels, which could not validate claims rigorously. This is scope creep rather than a weakness.

- *Theoretical proofs of convergence or noise tolerance*: This is an empirical systems paper. Demanding theoretical analysis is not standard practice in this area and goes beyond the paper's stated scope.

- *Confidence intervals for every table*: For benchmarks with 1,000 sets (Wikipedia, Arxiv), single-run AUROC evaluation is standard. The smaller benchmarks (Languages, License, Instructions) do warrant careful interpretation, which is better addressed by the set-level AUROC point above.

- *Unfair baseline comparisons*: No cases of comparisons that are unfairly asymmetric in favor of the baseline were identified.

**Removed generic strengths:**

- "The paper is well-written and clearly structured" — applies to any competently written paper and is not specific to this contribution.

---

## Novel Insights

The most substantive insight from the combined reviews — not explicitly acknowledged in the paper itself — is the dual confound problem for date-based benchmarks: (1) temporal distributional shift (older text has different statistical properties than newer text, independent of memorization) and (2) metadata structure exploitation (aggregating over date buckets may partially reconstruct the membership signal from metadata alone, without the model's loss contributing anything). These two issues operate in the same direction: both would inflate the apparent AUROC improvement of Set-MI on Wikipedia and Arxiv. The paper's Section 6 experiment provides indirect reassurance — verified 13-gram labels with a known training corpus still show high AUROC, suggesting some genuine MI signal — but since Section 6 is not the main experimental setup, the confounds remain live concerns for the headline results. Resolving them would either strongly validate or meaningfully revise the core contribution.

---

## Suggestions

1. **Add a metadata-only baseline**: for each benchmark, report the AUROC of a classifier that predicts membership from set metadata alone (e.g., logistic regression on date, language label, license category) without any model queries. This single experiment would resolve the most important methodological ambiguity in the paper.

2. **Add a distributional shift control for Wikipedia/Arxiv**: evaluate a model whose training cutoff *differs* from the Pile cutoff (but trained on the same general data distribution) on the date-structured Wikipedia/Arxiv benchmarks. If Set-MI still correctly identifies the *actual* training cutoff, this rules out pure distributional shift as the explanation.

3. **Extend verified-label experiments to main results**: run the 13-gram-overlap membership verification (as in Section 6) for at least Pythia 12B on Wikipedia as a main result, not just a robustness check. This would directly address the proxy-label concern for the paper's most prominent benchmark.

4. **Reconcile Table 1 with the benchmark construction text**: clarify whether Wikipedia/Arxiv comprise 100 or 1,000 sets, and whether Language/License/Instructions comprise 130 or 200/190/130 sets.

5. **Report set-level AUROC alongside document-level AUROC** in all main result tables, to avoid conflating set-level statistical power with document-level sample size.

6. **Expand deduplication and document-length ablations** to at least one additional domain and one additional MI method to support the generality claims made in the abstract.

---

## Paper Quality Assessment

**Originality**: Moderate-to-high. The conceptual shift from individual-document MI to set-level aggregation is a clear and well-motivated contribution. Simple aggregation has appeared in clinical NLP contexts, but applying it to web-scale pretraining data using natural metadata is novel and the benchmark construction is a distinct contribution.

**Importance of research question**: High. Training data auditing, evaluation contamination, and copyright analysis are urgent practical problems for LLM deployment and governance.

**Whether claims are well supported**: Moderate. The empirical improvement in Table 2 is clearly demonstrated across diverse settings. However, the two unaddressed confounds (distributional shift, metadata leakage) and the proxy-label issue for main experiments mean the attribution of improvement to MI signal aggregation is not fully established. The headline number (0.14 AUROC average) is directionally reliable but its mechanistic interpretation is uncertain.

**Soundness of experiments**: Moderate. The benchmark variety, model coverage, and ablation breadth are commendable. The narrow scope of individual ablations, conflation of document- and set-level AUROC, and absence of the missing-baseline experiment are meaningful methodological gaps.

**Clarity of writing**: Mostly good, but the inconsistencies between Table 1 and section text are a notable lapse in precision for a paper whose contribution partly rests on benchmark construction.

**Value to the research community**: Moderate-to-high, conditional on benchmark release. The benchmarks and the practical ablation findings (set size, document length, deduplication impact) would be genuinely useful; the core Set-MI method is easy to adopt.

**Contextualization relative to prior work**: Adequate. The differentiation from Jagannatha et al. (2021) and the Individual-MI baselines are well-handled. The paper appropriately scopes to the black-box loss-only setting.