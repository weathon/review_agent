=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
##Summary

This paper formalizes "answer-set consistency" for LLM enumeration questions: given two questions whose answer sets should satisfy a set-theoretic relation (equivalence, containment, disjointness), does the model's generated answers respect that relation? The authors construct ASCB, a benchmark of 600 handcrafted question quadruples (2,400 questions), evaluate 18 LLMs, find pervasive inconsistency even when models correctly classify the relation, and propose a Classification-then-Enumeration (CtE) prompting strategy that improves consistency.

## Strengths

- **Precise formalization grounded in set-theoretic relations.** The paper introduces answer-set consistency and answer-set contradiction as well-defined notions (Section 3.1), distinguishing between consistency w.r.t. a gold-standard relation and internal self-contradiction. This framing is novel for LLM evaluation—prior work (Elazar et al., 2021; Ghosh et al., 2025) addresses boolean or single-answer consistency, not set-valued enumeration—and provides a clean analytical vocabulary for the community.

- **The E₁,∗ control disentangles stochasticity from semantic misunderstanding.** By testing the same question at different times in different contexts, the paper provides a principled baseline for how much inconsistency arises from model nondeterminism alone. The gap between E₁,∗ and other relations' consistency rates (Table 3) is a genuine methodological contribution that many consistency evaluations lack, enabling the key finding that equivalence inconsistency is largely stochastic while containment/disjointness inconsistency is largely semantic.

- **The "knowing vs. doing" gap is a substantive empirical finding.** The paper shows that large models classify relations at ~90%+ accuracy (Appendix D) yet still produce inconsistent answer sets (e.g., GPT-5: 57% average consistency). This dissociation between relation recognition and answer-set compliance is a non-obvious result with implications for how we think about LLM reasoning.

- **Broad empirical coverage with statistical testing.** 18 models from 6 families, evaluated across 6 relations with McNemar tests for significance (Appendix E). The consistent pattern of results across diverse model families strengthens the generality of the findings.

## Weaknesses

### Major:

- **The %IDK confound undermines consistency metric interpretation.** CtE shows higher consistency rates than Base, but also substantially higher %IDK rates (e.g., GPT-4o: 29.79% → 66.66%; GPT-5-mini: 47.17% → 55.08%). Since the consistency rate (CON) and Jaccard similarity (SIM) exclude empty and "idk" responses (Section 3.4), a model that strategically refuses to answer difficult questions will show inflated consistency on the remaining responses. The paper acknowledges this ("LLMs tend to adopt a safer approach by answering 'idk' when uncertain, which may explain why CtE outperforms the other two strategies," Section 4.2) but does not account for it in the metrics or provide a sensitivity analysis. Without knowing whether the excluded "idk" instances would have been consistent or inconsistent, the raw consistency gains from CtE are ambiguous. This is a central claim of the paper and the metric does not properly support it.

- **Insufficient validation of dataset quality and relation correctness.** The ASCB was constructed with heavy LLM assistance: Q₃ and Q₄ were suggested by GPT-4.1 (Section 3.2), the entire SYNTHETIC subset (150 of 600 quadruples) was LLM-generated (Appendix B.1), and the filtering pipeline uses a multi-agent LLM system (Appendix B.2). While the paper states that all questions were manually reviewed by three authors, there is no inter-annotator agreement, no external validation that the set-theoretic relations actually hold in the modified questions, and no verification that ground-truth answer sets satisfy the claimed relations after the "heavy modifications" described. For a benchmark paper, the absence of any quantitative quality validation beyond author review is a significant gap.

- **RQ4 (causes of inconsistency) is only coarsely addressed.** Section 4.2 attributes inconsistency to stochasticity vs. semantic misunderstanding based on the E₁,∗ gap, but the "semantic misunderstanding" category is treated as a monolith. Appendix H lists error patterns (terminology variation, completeness, implicit logic failures) but provides no quantification of their prevalence. The causal analysis in Appendix G lists four sources of nondeterminism without empirically disentangling their contributions. For a paper that explicitly poses "Which key factors cause answer-set inconsistency?" as a research question, the answer remains at the level of "it's some mix of stochasticity and semantic errors," which is too coarse to be actionable.

### Minor:

- **The CtE-outperforms-Oracle anomaly lacks explanation.** Table 3 shows CtE achieving higher consistency than Oracle for some model-relation pairs (e.g., GPT-4o on E₁,₂: 98.33% vs. 85.33%; Gemini-2.0-flash on N₃,₁: 94.50% vs. 92.00%). The paper calls this "surprising" and speculates it is due to "forcing the LLM itself to reason about the questions when classifying their relation," but provides no ablation or case analysis to support this. Given that the Oracle has ground-truth relation information, this result demands deeper investigation—it could indicate prompt design issues in the Oracle condition (Appendix A.3 uses adversarial phrasing: "you returned me different values") rather than genuine reasoning benefits.

- **Overlap is defined but never tested.** Section 3.1 defines overlap (⟦Q₁⟧ ∩ Q₂⟧ ≠ ∅) as one of the primary relations, but Table 2 and the experiments exclude it without explicit justification. Section 3.3 notes "many such relations are redundant" and that only primary relations are tested, but overlap is a distinct relation that is not entailed by the others in all cases, creating a gap between the formal framework and the experimental evaluation.

### Trivial:

- The "ternary relation" terminology for E₄,₁\₃ (Section 4.2) is slightly imprecise: it is a binary relation on a derived set involving three questions, not a ternary relation in the standard mathematical sense.

## Nice-to-Haves

- A weighted or adjusted consistency metric that penalizes high %IDK rates, enabling fair comparison between strategies that trade off coverage for consistency.
- Domain and answer-set-cardinality stratification of results (e.g., geography vs. politics; 2–10 entities vs. 50–100 entities) to identify whether certain relation difficulties are domain-specific.
- Concrete qualitative examples of inconsistent answer pairs in the main paper—metrics alone do not convey what inconsistency looks like in practice.
- Cost/latency analysis of CtE (which requires two model calls per question pair) to assess practical deployability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Jaccard similarity interpretation for disjointness is confusing** (Harsh Critic): The paper correctly states that low Jaccard similarity for disjointness is desirable (answer sets should not overlap). The Table 3 results are consistent with this interpretation—high D₃,₄^CON with low D₃,₄^SIM is exactly what one would expect. The reviewer's confusion was their own, not the paper's fault.

- **100% CtE consistency on N₃,₁ is suspicious** (Harsh Critic): The reviewer misidentified the column. The 100% values appear in E₁,₂^SIM (Jaccard similarity for equivalence under CtE), not N₃,₁^CON. High Jaccard similarity for equivalent questions when the model has just classified them as equivalent and enumerates both in the same context is expected behavior, not an anomaly.

- **Multiple comparison correction missing** (Harsh Critic): While 180 tests are performed, the reported p-values are overwhelmingly < 0.001, which would survive even the most conservative Bonferroni correction. This does not materially affect the conclusions.

- **Garbled text in Section 3.3** (Harsh Critic): Identified as a parser artifact per review instructions.

- **Comparison to RAG/tool-use/neuro-symbolic baselines** (Spark Finder): Outside the paper's stated scope of diagnosing and evaluating answer-set inconsistency. The paper's contribution is the benchmark and analysis, not a complete solution framework.

- **Downstream task impact** (Spark Finder): Outside stated scope. The paper evaluates consistency directly, not its effect on downstream tasks.

- **Abstract overclaims about bigger models** (Harsh Critic): The paper itself qualifies this claim with the GPT-5 %IDK caveat in Section 4.2, making the abstract statement partially addressed in the body.

- **Oracle prompt phrasing could confuse model** (Harsh Critic): While the adversarial phrasing ("you returned me different values") is suboptimal, this is a minor prompt design choice and does not invalidate the Oracle results, which serve as an upper bound.

## Novel Insights

The dissociation between relation classification accuracy and answer-set consistency is more than a curiosity—it suggests that LLMs encode sufficient knowledge to *recognize* structural relationships between questions but lack the computational mechanism to *enforce* these relationships during generation. This is analogous to the difference between knowing a constraint and being able to optimize under it, pointing toward neuro-symbolic integration (where a symbolic layer enforces constraints over LLM-generated candidates) as a more promising direction than pure prompting strategies. The E₁,∗ control result—that even at temperature=0, models produce inconsistent answers to the same question across runs—also underscores that answer-set inconsistency is not purely a semantic reasoning problem but is compounded by infrastructure-level nondeterminism that prompting alone cannot address.

## Suggestions

- **Report consistency metrics with %IDK sensitivity analysis:** Compute consistency rates under different treatments of "idk" responses (e.g., counting them as inconsistent, reporting coverage-normalized consistency), so that the trade-off between consistency and coverage is transparent.
- **Validate a subset of ASCB quadruples with external annotators:** Have non-author annotators verify that the set-theoretic relations hold for 50–100 quadruples and report agreement rates. This would substantially strengthen the benchmark's credibility.
- **Quantify the error taxonomy from Appendix H:** Report what percentage of remaining inconsistencies (after CtE) are attributable to terminology variation vs. completeness vs. logical failures. This would make the RQ4 contribution actionable.
- **Investigate the CtE > Oracle anomaly with ablation:** Test variants where the Oracle prompt is rephrased neutrally (without "you returned me different values") and where CtE is run without the classification step, to isolate whether the benefit comes from reasoning, context, or prompt design.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 6.0, 4.0]
Average score: 4.0
Binary outcome: Reject
