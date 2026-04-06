=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title clearly reflects the core contribution. The abstract succinctly states the problem (answer-set inconsistency), the solution (a benchmark and metrics), key findings (pervasive inconsistency, mitigation via prompting), and implications. All claims are supported by the paper's content.

**Introduction & Motivation:** The problem is well-motivated through a clear example (Table 1) and by distinguishing this work from prior studies on single-answer consistency. The five research questions (RQs) are precisely stated and directly guide the paper's structure. The contributions are listed clearly.

**Method / Approach (Section 3):**
*   **Definitions (3.1):** The definitions of *answer-set consistency* and *answer-set contradiction* are precise and form a solid foundation. The handling of empty answers and the open/closed world assumption is appropriately noted.
*   **Dataset Construction (3.2):** The construction of the ASCB dataset is described in substantial detail, including base sources (LC-QUAD 2.0, QALD, QAWIKI, SYNTHETIC) and the multi-stage, LLM-assisted but ultimately *handcrafted* process. This is a strength, as it ensures high-quality, relation-grounded quadruples. The manual curation and revision are appropriately emphasized, addressing potential concerns about LLM-generated data quality. The final size (600 quadruples, 2400 questions) is reasonable for a benchmark.
*   **Evaluation Tasks & Metrics (3.3, 3.4):** The three tasks (Base, Classification-then-Enumeration/CtE, Oracle) are well-designed to isolate the core issue and test mitigations. The selected relations (Table 2) cover the core set-theoretic concepts and include a sensible control (`E_1,*`). The metrics (Classification Accuracy, Consistency Rate, Jaccard Similarity, %IDK) are appropriate and complementary. The plan for statistical significance testing (McNemar) and the analysis separating stochasticity (`E_1,*`) from semantic misunderstanding is thoughtful.
*   **Potential Gaps/Questions:**
    1.  While the prompts are provided in Appendix A, the paper does not discuss potential sensitivity of results to the exact phrasing of these prompts (especially for the complex `E_4,1\3` relation). A brief discussion or minimal ablation on prompt design would strengthen reproducibility.
    2.  The evaluation of the ternary relation `E_4,1\3` is an interesting probe of complex reasoning. However, its operationalization in Task 1.1 (classification) is slightly ambiguous: the prompt asks for the relation between `s1` (answers to Q1 not in Q2) and `s2` (answers to Q3). The expected answer is "Equivalence", but this requires the model to *compute* a set difference. It's not fully clear if this is testing *recognition* of a stated relation or *derivation* of a new relation through computation. This should be clarified.

**Experiments & Results (Section 4 & Appendices):**
*   **Model Selection & Setup:** Testing 18 models across families and sizes is comprehensive. Using the lowest possible temperature is correct for isolating semantic inconsistency from sampling randomness.
*   **Classification Results (Appendix D/4.1):** The key finding—that large modern models can classify relations with >90% accuracy, while smaller models struggle—is clear and supports RQ2. The note that `N_4,1` (containment based on a negated restriction) is hardest is a useful insight.
*   **Consistency Results (Table 3, 4.2):** The central results are presented effectively. The high inconsistency even for the control relation `E_1,*` convincingly demonstrates the significant role of LLM stochasticity (addressing RQ4). The larger gaps for containment/ternary relations show the added challenge of semantic misunderstanding. The finding that CtE sometimes outperforms Oracle is intriguing and well-discussed (linked to increased %IDK, i.e., a "safer" approach).
*   **Hypothesis Testing (4.3, Appendix E):** Statistical significance (`H1`) is thoroughly validated. The correlation analysis for `H2` is presented, but the results are mixed: while `E_4,1\3` shows strong correlation with model score, `D_3,4` and `E_1,2` do not. The authors' conclusion that consistency is "more pronounced in the more complex answer-set tasks" is supported, but `H2` as stated ("LLMs with better general performance produce more consistent responses") is only partially confirmed and should be nuanced.
*   **Major Concerns:**
    1.  **Missing Ablation on CtE Mechanism:** The CtE strategy is a core contribution. However, the improvement could stem from (a) the act of reasoning about the relation, or (b) simply placing both questions in the same context window before enumeration. A critical ablation is missing: a "Joint-Context Enumeration" task where both questions are presented simultaneously *without* asking for classification, followed by instructions to enumerate both. This would disentangle the effect of shared context from explicit relational reasoning.
    2.  **Interpretation of Jaccard for Disjointness:** Using Jaccard Similarity (`D_3,4^SIM`) for disjointness is unconventional (lower is better). While workable, it can be misleading. If one answer set is empty and the other is not, Jaccard is 0 (perfect for disjointness), but this might be an error rather than true consistency. This edge case should be noted, and perhaps a secondary binary metric (strict disjointness satisfied yes/no) could be reported alongside.
    3.  **Presentation of Key Data:** Table 5 (classification accuracy) and the full results of the statistical significance tests are relegated to appendices. Given that classification accuracy is a key component of RQ2 and RQ3, a summary table of the most important models' accuracies should be in the main paper. Similarly, a concise summary of the McNemar test results (e.g., "CtE and Oracle led to significant improvement (p<0.05) in X out of Y model-relation pairs") would strengthen the main narrative.

**Discussion & Limitations (Section 5):** The RQs are answered directly and clearly. The discussion of stochasticity vs. semantic misunderstanding is good. The limitations section is somewhat brief. It correctly identifies avenues for future work (extending relations, multi-turn dialogue, scaling the dataset) but could more deeply engage with the core limitations of the *current* study:
1.  The benchmark is English-only and focused on "crisp" factual domains. The impact of this on generalizability should be discussed.
2.  The prompting mitigations, while effective, come at the cost of increased "idk" responses. This trade-off between consistency and coverage/helpfulness is a crucial practical limitation that deserves more emphasis.
3.  The analysis of *why* containment is harder, or why negation (`N_4,1`) is particularly challenging, remains somewhat surface-level. A deeper error analysis (beyond Appendix H) categorizing the nature of semantic misunderstandings (e.g., lexical ambiguity, failure of logical composition, scope neglect) would elevate the work.

**Writing & Clarity:** The paper is generally well-written and logically structured. Some parts of the methodology (Section 3.2) are dense but necessary. The frequent cross-referencing to appendices for figures and tables, while common, slightly disrupts the flow for a reader without the full document.

### Overall Assessment
This paper makes a valuable and novel contribution by formalizing, benchmarking, and analyzing a practically important form of LLM inconsistency—answer-set relations for enumeration questions. The ASCB dataset is a significant resource, constructed with commendable rigor. The empirical findings are robust, revealing pervasive inconsistency stemming from both stochasticity and semantic limitations, and showing that prompting-based mitigation is possible but imperfect. The main weaknesses are the lack of a critical ablation study for the CtE mitigation and a somewhat shallow analysis of the underlying causes of errors. Nevertheless, the core contribution is solid, the methodology is sound, and the work provides clear practical insights and a foundation for future research. It meets the empirical and clarity standards expected for ICLR, provided the authors can address the major concerns regarding the ablation study and deepen the discussion of limitations and error analysis.

# Neutral Reviewer
## Balanced Review

### Summary
This paper formalizes the problem of answer-set inconsistency in LLMs, where models generate contradictory sets of entities when answering related factual enumeration questions (e.g., questions with equivalence, containment, or disjointness relations). The authors create a benchmark dataset (ASCB) of 600 handcrafted question quadruples, propose evaluation metrics, and empirically evaluate 18 state-of-the-art LLMs, finding pervasive inconsistency. They also propose and evaluate simple prompting strategies (Classification-then-Enumeration and Oracle) that significantly improve consistency.

### Strengths
1. **Clear Problem Formalization**: The paper clearly defines answer-set consistency and contradiction, grounding the problem in set-theoretic relations and connecting it to database theory (query containment/equivalence). This provides a solid conceptual foundation.
2. **High-Quality Benchmark Dataset**: The ASCB dataset is carefully constructed, combining and curating questions from multiple sources (LC-QUAD 2.0, QALD, QAWIKI) and synthetic generation, with manual review to ensure logical relations are crisp and well-defined. The dataset is publicly released.
3. **Comprehensive Empirical Evaluation**: The paper evaluates 18 diverse LLMs across three tasks (Base, Classification-then-Enumeration, Oracle) using multiple metrics (consistency rate, Jaccard similarity, classification accuracy). Results are presented thoroughly, with statistical significance testing (McNemar test) and correlation analysis against external benchmarks.
4. **Actionable Insights and Mitigation Strategies**: The proposed prompting strategies (CtE, Oracle) are simple yet effective, showing statistically significant improvements. The analysis distinguishes between stochastic and semantic causes of inconsistency and includes a detailed error analysis (Appendix H).

### Weaknesses
1. **Limited Dataset Scale and Diversity**: The dataset (600 quadruples, 2400 questions) is modest for contemporary LLM evaluation. It is limited to English and static, factual domains, which may not fully capture the complexity and diversity of real-world enumeration questions.
2. **Incremental Mitigation Strategies**: The proposed prompting strategies, while effective, are relatively straightforward extensions of chain-of-thought/self-consistency ideas. The paper does not explore more advanced techniques (e.g., constrained decoding, fine-tuning, or neuro-symbolic integration) that might offer more fundamental improvements.
3. **Superficial Analysis of Model Architecture Factors**: While the paper notes that newer/bigger models do not universally outperform older/smaller ones, it does not deeply investigate why—e.g., the impact of model architecture (MoE vs. dense), training data, or alignment techniques on consistency.
4. **Incomplete Error Analysis**: The error analysis (Appendix H) lists categories (terminology, completeness, implicit logic) but lacks a quantitative breakdown of their prevalence or a root-cause analysis linking errors to specific model capabilities or dataset characteristics.

### Novelty & Significance
**Novelty**: The paper introduces a well-formalized, underexplored problem—consistency in multi-answer enumeration—distinct from prior work on single-answer factual consistency or logical entailment. The creation of the ASCB benchmark is a novel contribution.
**Significance**: The findings highlight a critical reliability issue in LLMs that affects their use as knowledge sources or reasoning engines. The benchmark and metrics provide a valuable tool for the community. The work is pragmatically significant, showing that simple prompting can improve consistency, but it also underscores that LLMs alone cannot guarantee formal consistency.

### Suggestions for Improvement
1. **Expand and Diversify the Benchmark**: Increase the dataset size (e.g., via semi-automated generation) and include multilingual examples, temporal/dynamic questions, and questions with fuzzy or graded set membership to test robustness.
2. **Deeper Architectural and Causal Analysis**: Investigate how consistency correlates with specific model attributes (training objectives, reinforcement learning from human feedback, retrieval augmentation). Conduct ablation studies to disentangle the effects of stochasticity vs. semantic misunderstanding.
3. **Explore More Advanced Mitigation Techniques**: Experiment with constrained decoding to enforce set relations, fine-tuning on consistency objectives, or hybrid neuro-symbolic approaches that offload relational reasoning to external solvers.
4. **Enhance Error Analysis**: Provide a quantitative distribution of error types across models and relations. Analyze whether errors stem from knowledge gaps, reasoning failures, or prompt sensitivity, and propose targeted fixes.
5. **Compare with Retrieval-Augmented Methods**: Evaluate whether augmenting LLMs with external knowledge bases (via retrieval) improves answer-set consistency, as this is a common deployment paradigm.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison to a non-LLM baseline (e.g., a rule-based system or database).** Without this, the severity of LLM inconsistency is not contextualized, undermining the claim that LLMs are uniquely problematic for enumeration tasks.
2. **No ablation on answer set cardinality.** The benchmark limits answers to 2-100, but results are not broken down by size. Inconsistency may correlate with cardinality, which is critical for understanding practical limitations.
3. **No systematic study of prompt variations.** Only one specific CtE prompt is tested. Ablations (e.g., chain-of-thought, few-shot) are needed to confirm the proposed strategy's efficacy and generalizability.
4. **No controlled study of temperature effects.** While temperature is minimized, its impact on inconsistency is not isolated. Sweeping temperature would disentangle stochasticity from systematic errors.

### Deeper Analysis Needed (top 3-5 only)
1. **Lack of error type breakdown.** The paper attributes inconsistency to stochasticity and semantic misunderstanding but does not categorize errors (e.g., terminological variation, omission, hallucination). This is essential for targeted mitigation.
2. **Insufficient analysis of the link between classification accuracy and consistency.** Although Appendix F touches on this, the main analysis does not quantify how often correct relation recognition leads to consistent enumeration—key for understanding self-contradiction.
3. **No analysis of model characteristics vs. inconsistency.** The paper tests 18 models but does not correlate consistency with model size, architecture, or training data composition. This misses an opportunity to explain what makes a model more consistent.

### Visualizations & Case Studies
1. **Missing concrete examples of inconsistent outputs.** The paper needs illustrative case studies (e.g., a few quadruples with model outputs) to show what inconsistency looks like, making the problem tangible and diagnosing failure modes.
2. **No visual representation of answer-set relationships.** Venn diagrams for sample quadruples would clearly show how model-generated sets violate expected relations, enhancing interpretability of the metrics.

### Obvious Next Steps
1. **Ablation of the CtE prompt components.** The paper should test which parts of the CtE strategy (e.g., relation classification format, ordering) are necessary, as the current prompt is a black box.
2. **Generalization test to other question types/languages.** The benchmark is English and factual. Testing on subjective or multilingual questions is needed to claim broader relevance of the findings.
3. **Exploration of fine-tuning for consistency.** Prompting is a zero-shot fix; fine-tuning on the benchmark (or similar data) is a logical next step that should be discussed and preliminarily tested.

# Final Consolidated Review
## Summary
This paper formalizes answer-set inconsistency in LLMs, where models generate contradictory sets of entities when answering related factual enumeration questions (e.g., questions with equivalence, containment, or disjointness relations). It introduces a handcrafted benchmark (ASCB) of 600 question quadruples, proposes evaluation metrics, and empirically shows pervasive inconsistency across 18 state-of-the-art LLMs. Simple prompting strategies like Classification-then-Enumeration significantly improve consistency, with statistical significance.

## Strengths
- **Clear problem formalization:** The definitions of answer-set consistency and contradiction are precise, grounded in set-theoretic relations, and distinguish the problem from prior work on single-answer consistency.
- **High-quality benchmark dataset:** The ASCB dataset is carefully constructed from multiple sources (LC-QUAD 2.0, QALD, QAWIKI, SYNTHETIC) with rigorous manual curation to ensure logical relations are crisp and well-defined, and it is publicly released.
- **Comprehensive evaluation:** The paper evaluates 18 diverse LLMs across three tasks (Base, Classification-then-Enumeration, Oracle) using multiple metrics (consistency rate, Jaccard similarity, classification accuracy), includes statistical significance testing, and analyzes causes like stochasticity vs. semantic misunderstanding.

## Weaknesses
- **Missing ablation for the CtE mechanism:** The improvement from the Classification-then-Enumeration strategy could stem from either explicit relational reasoning or merely from presenting both questions in the same context. Disentangling these effects is crucial for understanding and optimizing the mitigation.
- **Unconventional metric for disjointness:** Using Jaccard similarity to measure disjointness (where lower scores are better) is non-standard and may not fully capture strict disjointness, though empty answer sets are excluded from this metric.
- **Limited dataset generalizability:** The benchmark is English-only and focused on static, factual domains, which may restrict the applicability of findings to multilingual or dynamic contexts, though this limitation is acknowledged.

## Nice-to-Haves
- Ablation studies on prompt variations or components of the CtE strategy to identify which elements are most effective.
- Breakdown of results by answer set cardinality to examine if inconsistency correlates with the size of enumerated lists.
- Deeper investigation into how model architecture (e.g., MoE vs. dense) or training data characteristics relate to consistency performance.
- More visual examples or case studies illustrating specific inconsistencies to make the problem more tangible.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct an ablation experiment comparing CtE with a "Joint-Context Enumeration" task where both questions are presented together without explicit classification, to isolate the effect of relational reasoning from shared context.
- Augment the error analysis in Appendix H with a quantitative distribution of error types (e.g., terminological variation, omissions) across models and relations to better diagnose failure modes.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 6.0, 4.0]
Average score: 4.0
Binary outcome: Reject
