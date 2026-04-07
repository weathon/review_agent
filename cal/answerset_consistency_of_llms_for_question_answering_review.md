=== CALIBRATION EXAMPLE 48 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me compose my review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "Answer Set Consistency of LLMs for Question Answering" accurately reflects the contribution. The abstract correctly claims that the paper formalizes the problem, creates a benchmark of 600 quadruples, evaluates 18 LLMs, and proposes prompting mitigations. The claim that "pervasive inconsistency" is found and that prompting "leads to improved answer-set consistency" is well-supported by the experimental results. The scope is appropriately qualified in the limitations section.

---

### Introduction & Motivation

The problem is well-motivated and the running example (Table 1, the Madeira River tributaries) is crisp and effective. The five research questions are clearly enumerated and are all addressed in the paper. The contributions are stated concisely.

**Concern:** The note that "newer or bigger models do not universally outperform older or smaller variants" is prominent in the introduction summary (p. 2), but the correlation analysis in Table 4 shows that several metrics *do* significantly correlate with external benchmark rankings. The framing in the introduction is somewhat misleading about what was actually found.

---

### Formalization (Section 3.1)

The formalization is clean and necessary. The distinction between *answer-set inconsistency* (LLM answers don't satisfy a known relation) and *answer-set contradiction* (LLM answers don't satisfy the relation the LLM itself predicted) is well-drawn. The decision to relegate contradictions to Appendix F weakens the main paper, as this is conceptually the more informative of the two notions: it does not require ground-truth relations and provides a cleaner self-referential diagnostic.

**Concern:** The definition of *answer-set consistency* for equivalence checks whether [[Q₁]]_M = [[Q₂]]_M. In practice, this comparison is done over string-parsed answer lists. The paper does not explain *how* string normalization or entity resolution is performed. Appendix H acknowledges that "Spain" vs. "Kingdom of Spain" causes spurious inconsistency, but this is never quantified. Without knowing the extent of this issue, the reported consistency numbers could be substantially underestimating true set-level consistency. This is a significant methodological gap that needs to be addressed, ideally with an ablation using fuzzy matching or entity linking.

---

### Dataset Construction (Section 3.2 & Appendix B)

This is the paper's central empirical contribution, and the construction process is documented in commendable detail. The quadruple structure (Q₁, Q₂, Q₃, Q₄) is elegant, yielding 12 primary relations per quadruple from a single manual effort. The use of SPARQL queries to verify cardinality bounds before inclusion is a thoughtful design choice.

**Concern 1 — Circular use of GPT-4.1:** The dataset was constructed using GPT-4.1 (Appendix B.1: "The LLM employed for this task was GPT-4.1-2025-04-14"), and GPT-4.1 variants are also among the evaluated models (GPT-4.1, GPT-4.1-mini, GPT-4.1-nano appear in Table 3). The heavy manual curation partially mitigates data contamination concerns, but the authors should explicitly discuss whether GPT-4.1's involvement in dataset creation could introduce systematic bias in its favor on the consistency task — or conversely, whether it learned the style of questions in a way that affects its enumeration performance.

**Concern 2 — Ground-truth temporal validity:** The ground-truth relations are derived from Wikidata/DBpedia at dataset creation time. Facts change (e.g., EU membership). When evaluating LLMs with different training cutoffs, "correct" ground truth may diverge from the model's parametric knowledge. This is acknowledged in the limitations section but never quantified. Since 600 quadruples span diverse factual domains, at least some instances could be affected.

**Concern 3 — One language, one domain type:** The dataset is English-only and focuses on "crisp," static, factual enumeration questions. The claim in the limitations section that "600 quadruples is sufficient to derive conclusions with statistical significance" is justified for the five tested relations, but generalizability to other question types (temporally-dynamic, vague, non-English) is unclear. For ICLR, this is a reasonable scope, but should be stated more prominently as a boundary condition rather than just a limitation.

---

### Tasks and Metrics (Section 3.3 & 3.4)

The three-task design (Base, CtE, Oracle) forms a coherent experimental ladder. The McNemar test is appropriate for paired binary comparisons.

**Concern 1 — The IDK confound:** The %IDK values for some models under CtE are very high (e.g., GPT-4o: 66.66%; GPT-5: ~47.50%; GPT-5-mini: ~55%). Since IDK responses are *excluded* from consistency rate computations (Section 3.4), a model that refuses to answer most questions can achieve high *measured* consistency on the subset it does answer, since that subset may be systematically easier cases. The paper acknowledges this (p. 8: "LLMs tend to adopt a safer approach by answering 'idk' when uncertain"), but does not address it analytically. A consistency measure that *counts IDK as inconsistent* should be reported in addition to the current measure, to separate genuine consistency improvement from strategic abstention. This is a significant confound for the main claim that CtE improves consistency.

**Concern 2 — Jaccard for disjointness:** Using Jaccard similarity for disjointness (D_{3,4}) is correct in direction (lower is better), but Jaccard similarity is bounded at 0 by construction when the sets are empty — meaning models returning empty answers for both Q₃ and Q₄ would score perfectly. The paper excludes empty answers, but the interaction between the empty-rate exclusion and Jaccard for disjointness should be clarified.

**Concern 3 — Priority ordering in Task 1.1:** The classification task asks models to identify the "first" relation in a fixed ordering (equivalence → contained by → contains → disjointness → overlap). This ordering may inflate equivalence accuracy if models are inclined to default to equivalence. The choice of ordering is not justified, and no sensitivity analysis is reported.

---

### Results (Section 4)

**Section 4.1 — Classification:** There appears to be a textual inconsistency in Appendix D: the text states that "models perform best for disjointness (D_{3,4}) with ~60% accuracy" and that "N_{4,1} is most difficult with ~91% accuracy." But Table 5's average row shows D_{3,4}^ACC = 90.48 and N_{4,1}^ACC = 60.64 — these numbers are transposed in the text description. This should be corrected.

**Section 4.2 — CtE outperforming Oracle:** The paper reports (p. 8) that CtE "surprisingly even outperforms the Oracle in many cases." The explanation offered is that CtE forces reasoning before enumeration. But a simpler explanation — higher IDK rates under CtE self-select for easier or more confident cases — is not ruled out. Given GPT-4o's 66.66% IDK under CtE, any consistency measure computed only on the remaining ~33% of cases may reflect a heavily biased sample. This alternative explanation should be tested, for example by comparing consistency only on the subset of questions answered under all three strategies.

**Section 4.3 — Hypothesis H2:** The paper says H2 ("better general performance → more consistent responses") is "partially confirmed." However, Table 4 shows that among 9 metric–relation combinations, only 2 reach p < 0.05 (D_{3,4} and E_{4,1\3}). The remaining correlations are not statistically significant. A claim of "partial confirmation" for a hypothesis that tests 9 metrics, with only 2 reaching significance, is borderline. The effect of multiple comparison inflation (no Bonferroni correction applied) should be acknowledged.

**Missing ablation — Dataset source breakdown:** Results are presented only for the merged dataset. Individual breakdowns by source (LC-QuAD 2.0, QALD, QAWIKI, Synthetic) are on GitHub but not in the paper. Given that these sources differ substantially in how questions were constructed, this breakdown would be informative about where inconsistency is most severe and whether the results are stable across domains.

---

### Discussion & Limitations (Section 5)

The RQ answers are concise and match the evidence. The limitations section is honest and covers the main gaps (single-turn, English-only, static, scale). The suggestion to combine LLMs with formal query reasoning engines is a reasonable research direction.

**Concern:** The paper does not discuss negative societal implications. LLMs are increasingly used for information retrieval and enumeration tasks (e.g., "list all drugs that interact with X"). The finding that even CtE leaves high inconsistency in many relations — especially for containment and set-difference — has direct implications for high-stakes enumeration tasks. A sentence on this would strengthen the broader impact framing.

---

### Writing & Clarity

A few passages are genuinely confusing. In Section 3.4, the description of consistency rates and the McNemar test setup (p. 6) is dense and would benefit from a worked example. Section 4.2 contains a run-on sentence listing causes of inconsistency with an untidy set of inline citations that spans several lines and impedes reading. The main tables (Table 3) are very wide and the relation notations are overloaded — readers must hold multiple notation definitions in mind simultaneously.

---

## Overall Assessment

This paper addresses a well-defined and practically relevant problem: the self-contradictory behavior of LLMs when answering sets of logically-related enumeration questions. The formalism is clean, the benchmark is carefully constructed with substantial manual effort, and the empirical scope (18 models, 5 relation types, 3 mitigation strategies) is commendable. The core findings — that inconsistency is pervasive, that semantic misunderstanding (not just stochasticity) drives failures for containment and ternary relations, and that classify-first prompting helps significantly — are interesting and actionable. However, several issues must be addressed before this work is ready for ICLR. Most critically: (1) the string-matching methodology for consistency evaluation is not documented and its impact on measured inconsistency is unknown; (2) the high IDK rates under CtE introduce a major confound for the main positive claim about that mitigation strategy; and (3) H2 is presented as "partially confirmed" despite most Pearson correlations being non-significant. The answer-set contradiction analysis (Appendix F) is conceptually richer than the main inconsistency analysis and deserves more prominence. The paper makes a solid empirical contribution and the benchmark is a genuine asset to the community, but the analysis of the CtE results requires deeper treatment to be convincing at ICLR's standard.

# Neutral Reviewer
## Balanced Review

### Summary
This paper formalizes "answer-set consistency" to quantify self-contradictions in Large Language Models (LLMs) when answering enumeration questions that satisfy specific set-theoretic relations (e.g., equivalence, containment, disjointness). The authors introduce the Answer-Set Consistency Benchmark (ASCB), a dataset of 600 question quadruples, and evaluate 18 state-of-the-art models to demonstrate pervasive inconsistency. Furthermore, they propose and validate mitigation strategies, such as asking models to classify relations before enumerating answers (Classify-then-Enumerate), which significantly improves consistency rates across models.

### Strengths
1.  **Novel Formalization and Framework:** The paper provides a clear mathematical formalization of answer-set consistency, distinguishing it from existing boolean consistency works (Sec 3.1, Example 3.1). This moves beyond simple factual accuracy to evaluate internal logical coherence across related queries.
2.  **Comprehensive Benchmark Creation:** The construction of the ASCB dataset (2,400 questions) is transparently described (Sec 3.2, Appendix B). It fills a gap in existing QA benchmarks by including set-theoretic ground truth relations, enabling the specific evaluation of logical consistency that previous datasets do not support.
3.  **Rigorous Empirical Analysis:** The evaluation includes 18 diverse models (GPT, Llama, Gemini, etc.) across three tasks (Base, CtE, Oracle) using multiple metrics (Consistency Rate, Jaccard Similarity) and statistical significance testing (McNemar test) (Sec 3.4, Table 3, Table 6). The finding that the mitigation strategy (CtE) often significantly outperforms the baseline (p < 0.001) is statistically robust.
4.  **Meaningful Mitigation Insights:** The discovery that prompting models to reason about relations before answering (CtE) improves consistency, sometimes exceeding the "Oracle" baseline due to model caution (Sec 4.2), offers a practical and deployable solution for improving LLM reliability without fine-tuning.

### Weaknesses
1.  **Dataset Domain Diversity:** The benchmark relies heavily on Knowledge Graph Question Answering datasets (LC-QUAD, QALD) and synthetic generation, which focuses primarily on factual entity lists (e.g., rivers, countries, EU members) (Sec 3.2). This limits the generalizability of findings to more subjective or open-ended enumeration tasks (e.g., "list reasons why...").
2.  **Metric Interpretation for Disjointness:** The use of Jaccard similarity for Disjointness relations requires a score close to 0 for "good" performance, whereas equivalence requires a score close to 1 (Sec 3.4). While explained, this asymmetry in metric interpretation (high is good vs. low is good) increases the cognitive load for readers interpreting Table 3 and Figure 1.
3.  **Incomplete Analysis of 'Oracle' Performance:** The paper notes CtE sometimes outperforms Oracle (Sec 4.2), attributing this to models refusing to answer ("idk") more safely. However, this introduces a bias where "no answer" is preferable to "inconsistent answer." The trade-off between completeness and consistency is not fully quantified, particularly regarding the high "%IDK" rates observed in some mitigation strategies (e.g., ~32% for GPT-5 in Base, though lower in CtE, Table 3).
4.  **Lack of Ablation on Stochasticity:** While the paper identifies stochasticity vs. semantic misunderstanding as causes (Sec 4.2) and uses a control task ($E_{1,*}$) to estimate stochasticity, it does not provide a full breakdown of variance contributions for all models. Some smaller models show high inconsistency even with stochasticity controlled, suggesting architecture dependencies that are briefly mentioned but not deeply ablated.

### Novelty & Significance
*   **Novelty:** High. While consistency in LLMs is well-studied for boolean logic, the extension to **enumeration sets** with **set-theoretic relations** (containment, disjointness) is a distinct and necessary contribution. The ASCB dataset is specifically designed for this purpose and is not available in prior works.
*   **Significance:** High. The reliability of LLMs for information retrieval is a critical concern. By demonstrating that models frequently violate logical relations they can verbally identify, the paper highlights a specific "reasoning gap" in generative models. The practical mitigation strategies proposed are immediately useful for developers.
*   **Clarity:** Good, despite the provided text containing OCR/formatting artifacts (e.g., broken equations, garbled table headers). The core definitions and logic are clear.
*   **Reproducibility:** High. The authors commit to releasing code, data, and detailed prompt specifications on GitHub (anonymous link provided in text), and the methodology section is detailed enough to follow.

### Suggestions for Improvement
1.  **Expand Dataset Diversity:** Include enumeration tasks beyond standard Knowledge Graph facts (e.g., literature, creative domains) to ensure the consistency issues are not limited to rigid entity lists. This would strengthen the claim of general applicability.
2.  **Clarify Metric Reporting:** Consider using a normalized "Consistency Score" that ranges 0 to 1 for all metrics (e.g., inverting Jaccard for disjointness) or clearly distinct visualizations for "High is Good" vs "Low is Good" relations to aid interpretation.
3.  **Address Completeness vs. Consistency:** Provide a deeper analysis of the "Completeness-Consistency Pareto Frontier." For applications requiring exhaustive lists vs. high-fidelity lists, how should users balance the high "%IDK" rates seen in mitigation strategies (Task 2) against Base accuracy?
4.  **Strengthen Stochasticity Analysis:** Provide a variance analysis of $E_{1,*}$ consistency across repeated runs for specific models to quantify exactly how much inconsistency is due to decoding randomness versus semantic understanding, perhaps visualized as a bar chart of variance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a standard Chain-of-Thought (CoT) baseline comparison. Without comparing Classification-then-Enumeration (CtE) against generic CoT prompting, the claimed novelty of the mitigation strategy is unverified.
2. Verify all dataset set-theoretic relations via external knowledge base execution (e.g., SPARQL). Relying on LLM generation and manual inspection for ground-truth relations risks building the benchmark on incorrect logical premises.
3. Ablate the impact of answer normalization (synonyms, acronyms, ordering) on consistency scores. Exact string matching likely conflates lexical variation with logical inconsistency, artificially inflating error rates.
4. Test the mitigation strategies without rigid formatting constraints (e.g., '|' separators). The current results may reflect instruction-following failures rather than logical inconsistencies, limiting practical applicability.

### Deeper Analysis Needed (top 3-5 only)
1. Decouple factual accuracy from logical consistency in the results analysis. High inconsistency might simply reflect hallucination rather than a failure of logical reasoning over known facts, undermining the core claim.
2. Quantify formatting failure rates separately from logical errors. Instances where models omit separators or add text should not be penalized as answer-set inconsistencies in the primary metrics.
3. Analyze the correlation between model confidence (logprobs) and inconsistency events. This determines if inconsistencies arise from epistemic uncertainty or systematic reasoning flaws.
4. Investigate why CtE sometimes outperforms Oracle. Since Oracle provides the correct relation, CtE surpassing it suggests potential prompt interference or that self-generated reasoning offers distinct benefits that need explanation.

### Visualizations & Case Studies
1. Provide case studies where semantic normalization would resolve flagged inconsistencies. Visualizing false positives due to lexical variation (e.g., "USA" vs "United States") reveals whether the problem is semantic or superficial.
2. Plot consistency rates against factual accuracy for each model. This visualizes whether consistent models are also correct, or merely consistently hallucinating.
3. Include a heatmap of inconsistency rates by domain (e.g., geography vs. history). This exposes whether the issue is driven by domain-specific knowledge gaps rather than general logical failure.

### Obvious Next Steps
1. Implement semantic set similarity (e.g., embedding-based) instead of exact string matching for evaluation. This is necessary to trust the consistency metrics reported in the paper.
2. Expand the dataset beyond 600 quadruples to ensure statistical robustness across diverse domains. The current scale is borderline for the number of models evaluated (18) at ICLR standards.
3. Release the full evaluation pipeline and normalization code explicitly. Reproducibility of the set-extraction and matching logic is critical for validating the benchmark claims.
4. Validate the mitigation strategy on open-ended tasks without rigid formatting constraints. The current reliance on specific separators limits the generalizability of the proposed solution.

# Final Consolidated Review
## Summary
This paper formalizes "answer-set consistency" for LLMs answering enumeration questions whose answer sets should satisfy set-theoretic relations (equivalence, containment, disjointness). The authors construct a benchmark of 600 handcrafted question quadruples (2,400 questions) with known ground-truth relations, evaluate 18 state-of-the-art LLMs, and propose a Classification-then-Enumeration (CtE) prompting strategy that significantly improves consistency.

## Strengths
- **Novel formalization of an important problem:** The paper introduces a clean mathematical framework for answer-set consistency that extends prior work on boolean/factual consistency to enumeration questions with set-theoretic relations. The distinction between answer-set inconsistency (violating known relations) and answer-set contradiction (violating relations the model itself predicts) is conceptually useful.
- **Careful benchmark construction:** The ASCB dataset construction is transparently documented, with question quadruples designed to test 12 relations from each set of four questions. The use of SPARQL queries to verify cardinality bounds and substantial manual curation (three authors) contributes to dataset quality.
- **Comprehensive empirical scope:** Evaluating 18 models across five relation types and three prompting strategies (Base, CtE, Oracle) with statistical significance testing (McNemar test) provides actionable findings. The discovery that semantic misunderstanding (not just stochasticity) drives failures for containment and ternary relations is empirically supported.
- **Practical mitigation with significant improvements:** The CtE strategy yields statistically significant improvements across most models and relations (Table 6), and the analysis that stochasticity primarily affects equivalence relations while semantic misunderstanding dominates containment/disjointness provides useful diagnostic insight.

## Weaknesses
- **The IDK confound undermines the CtE improvement claim:** The high %IDK rates under CtE for some models (e.g., GPT-4o at 66.66%) mean consistency is computed only on the subset of questions the model chooses to answer. Since models may strategically refuse uncertain or difficult cases, the measured consistency improvement may reflect selection bias rather than genuine logical coherence gains. A more rigorous analysis would measure consistency on the *intersection* of questions answered across strategies, or report a metric that counts IDK as inconsistent.
- **String matching methodology is underdocumented:** The paper measures set equality via string parsing of LLM outputs (using '|' as separator), but does not specify how entity normalization is performed. Appendix H acknowledges issues like "Spain" vs. "Kingdom of Spain" but does not quantify their impact on measured inconsistency. Without this quantification or an ablation using fuzzy matching/embedding similarity, the reported consistency rates may conflate lexical variation with genuine logical inconsistency.
- **Potential data contamination from GPT-4.1:** The dataset was constructed using GPT-4.1-2025-04-14 (Appendix B.1), and GPT-4.1 variants appear among evaluated models. While heavy manual curation mitigates direct contamination, the authors should discuss whether GPT-4.1's familiarity with the question style or generation patterns could bias its performance. A brief discussion or ablation excluding GPT-4.1 variants would strengthen confidence in the findings.
- **Factual accuracy is not decoupled from logical consistency:** The consistency metrics do not distinguish between a model that enumerates incorrect entities (but respects set relations) versus one that violates relations. A model hallucinating entities could appear "consistent" if it hallucinates consistently across related questions. The paper would benefit from analyzing whether high-consistency models also achieve higher factual correctness, or whether consistency improvements come at the cost of accuracy.
- **H2 correlation analysis is overstated:** The hypothesis that "LLMs with better general performance produce more consistent responses" is described as "partially confirmed," yet Table 4 shows only 2 of 9 correlations reach statistical significance (p < 0.05), without correction for multiple comparisons. The claim should be tempered to reflect that the evidence is mixed rather than partially confirming.

## Nice-to-Haves
- Comparing CtE against a standard Chain-of-Thought baseline would help isolate whether the improvement comes from relation-specific reasoning or general chain-of-thought benefits.
- Analyzing consistency rates by dataset source (LC-QuAD, QALD, QAWIKI, Synthetic) would reveal whether results generalize across question construction methodologies.
- Reporting formatting failure rates separately from logical errors would clarify whether inconsistencies stem from instruction-following failures or reasoning deficits.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Newer/bigger models do not universally outperform" is misleading:** The paper's wording is accurate—the claim is not "no correlation" but "not universal," which Table 4 supports since only some metrics correlate with external rankings.
- **Single-turn evaluation is a limitation:** Stated explicitly in the limitations section; scope, not a weakness.
- **English-only dataset:** Also stated in limitations as a boundary condition; reasonable scope for an initial contribution.
- **Temporal validity of ground truth:** Acknowledged in limitations; the paper uses static factual domains specifically to minimize this issue.
- **No societal impact discussion:** This is an empirical methodology paper; such discussion is not expected at ICLR standards.
- **Answer-set contradiction analysis relegated to appendix:** This is a paper organization choice; Appendix F provides the analysis for interested readers.
- **Metric interpretation for disjointness is confusing:** The paper clearly states that for D_{3,4}, lower Jaccard is better; this is explained adequately.
- **Dataset scale is insufficient:** 600 quadruples with statistical significance testing is appropriate for the five relations tested.

## Novel Insights
The paper's most interesting finding is that the CtE strategy sometimes outperforms the Oracle (which provides correct relations). The authors suggest this is because CtE forces models to reason before enumerating, but an alternative explanation is that CtE's higher IDK rates self-select for easier cases. This paradox—where self-generated reasoning can outperform being given correct information—warrants further investigation. It suggests models may distrust their own classifications and refuse to enumerate when uncertain, which improves measured consistency on the subset they do answer. Understanding whether this is beneficial strategic abstention or undesirable avoidance behavior would advance the reliability literature.

## Suggestions
- Report consistency rates on the subset of questions answered under all three strategies (Base, CtE, Oracle) to control for the IDK selection effect, and/or report a unified metric that treats IDK as inconsistent.
- Quantify the impact of lexical variation (e.g., "Spain" vs. "Kingdom of Spain") on measured inconsistency by running an ablation with fuzzy string matching or embedding-based entity linking.
- Clarify how answer set parsing handles edge cases (missing separators, extra text, partial answers) and report the failure rate for the parsing step separately from logical consistency errors.
- Temper the H2 claim to acknowledge that only 2 of 9 correlations are significant at p < 0.05, and discuss why containment/ternary relations correlate with general performance while equivalence does not.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 6.0, 4.0]
Average score: 4.0
Binary outcome: Reject
