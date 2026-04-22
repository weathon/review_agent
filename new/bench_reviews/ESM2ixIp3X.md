Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

This paper proposes SBERTScore, a sentence-level extension of BERTScore for evaluating summary factual consistency. The key contributions are: (1) demonstrating that prior similarity metrics' poor performance on factuality evaluation stemmed from using reference texts rather than source documents for comparison, (2) extending BERTScore to operate at the sentence level using sentence-transformer embeddings, and (3) showing that combining different metrics via logical AND improves detection performance.

## Strengths

- **Clear, impactful demonstration that reference-vs-source comparison explains prior failures of similarity metrics.** Table 3 shows BERTScore jumps from 0.500 to 0.759 and SBERTScore from 0.499 to 0.779 when switching from reference-based to source-based comparison. All improvements are statistically significant (p < 0.05). This is the paper's strongest contribution—it reframes a prior conclusion about similarity metrics.

- **SBERTScore is competitive with trained metrics on CNN/DM in zero-shot settings.** Table 7a shows SBERTScore achieves 0.720 balanced accuracy on CNN/DM, outperforming trained metrics DAE (0.696) and SummaC_ZS (0.686), and second only to QAFactEval (0.757). This supports the claim that zero-shot similarity metrics are viable in this setting.

- **Thorough granularity and precision/recall/F1 analysis.** Table 4 systematically evaluates 9 granularity combinations and documents truncation problems (45.76% of sources truncated at Doc level). Table 2 shows precision significantly outperforms recall and F1, supporting the theoretical argument. This gives clear guidance to practitioners.

- **Honest discussion of negation limitations.** Table 5 transparently shows SBERTScore assigns similar scores to contradictory pairs (0.720) and topically related pairs (0.701), while acknowledging that future work is needed. The improvement over BERTScore's 0.984 on negation pairs is shown but the residual problem is not hidden.

- **Computational efficiency advantage.** Section 3.1 establishes O(N+M) complexity for SBERTScore vs. O(NM) for NLI-based metrics, with empirical validation showing 3× speedup over SummaC_ZS and 30× over QuestEval.

## Weaknesses

### Fatal
None.

### Major

- **The competitiveness claim is overclaimed for XSum.** The abstract states SBERTScore "can compete with existing NLI and QA-based factuality metrics on the benchmark," but on XSum (Table 7b), SBERTScore achieves only 0.605 balanced accuracy—lower than QAFactEval (0.705), QuestEval (0.665), and even BERTScore (0.695). Its ROC-AUC on XSum (0.653) is also worst among all metrics except SummaC_ZS. The paper's explanation (single-sentence summaries prevent averaging) is plausible but means the competitiveness claim should be scoped to non-extreme-compression settings, not stated blanketly in the abstract. — Overclaiming in the abstract undermines the paper's stated scope and could mislead readers about the method's general applicability.

- **Granularity contribution is conflated with backbone model contribution.** Table 4 shows SBERTScore at Word-Word granularity (0.767) already outperforms BERTScore at Word-Word (0.759), proving the sentence-transformer backbone accounts for part of the improvement. The Sent-Sent result (0.779) is only 1.2 points above Word-Word SBERTScore. The paper partially addresses this in Section 5.3 ("improvement is brought by both the architecture and the appropriate text granularity"), but the second contribution statement ("We develop the token-level similarity-based metric BERTScore into sentence-level SBERTScore and improve the performance") and abstract attribute the improvement primarily to the sentence-level design. Without a controlled ablation where both methods use the same backbone at the same granularity, the sentence-level design's independent contribution is unclear. — This matters because it affects whether the core contribution (sentence-level extension) is actually responsible for the performance gains claimed.

### Minor

- **The logical-AND combination analysis, while informative, provides limited insight beyond what is expected from the interaction of imbalanced classifiers.** AND of two high-FP classifiers will naturally improve balanced accuracy by reducing false positives. The paper does not report per-class precision/recall under AND, nor compare against calibrated combination methods. That DAE ∧ QAFactEval (0.828) only marginally improves over QAFactEval alone (0.817) suggests modest gains. The main value of Figure 1 is confirming complementary error types (supported by Table 8 and low κ), not the AND combination itself.

- **The max-over-source operation in Eq. 1 is a structural weakness for factuality detection.** A summary sentence partially entailed by different source sentences can receive a high score because each part matches a different source sentence maximally. This is inherited from BERTScore but compounded at sentence level where composition matters more. The paper does not discuss this limitation.

- **All datasets are English news articles from CNN/DM and XSum.** Generalizability to other domains, languages, or longer documents is unknown. This is a scope limitation acknowledged implicitly by the paper's experimental design.

### Trivial
None.

## Nice-to-Haves

- A controlled ablation using the same backbone (all-roberta-large-v1 or vanilla roberta-large) for both BERTScore and SBERTScore at the same granularity, isolating the sentence-level design's contribution from the backbone effect.
- Per-class precision/recall analysis under AND combination and comparison against a principled combination method (e.g., logistic regression stacking).
- Evaluation on non-news domains or longer documents where the truncation issue becomes more severe.

## Removed Points

- **"Unfair experimental settings" framing is a strong claim** — The harsh critic argued this overstates the case, as prior work evaluated BERTScore for its intended purpose. However, the paper's point is legitimate: prior work concluded similarity metrics were unsuitable for factuality evaluation *based on reference-based settings*. Whether to call this "unfair" or "suboptimal" is a framing choice, not a methodological flaw. Moved here as it is a minor wording concern. — *Treat with caution.*

- **SummaC_ZS "zero-shot" classification scrutiny** — The harsh critic questioned whether SummaC_ZS should be called zero-shot given its NLI fine-tuning. This is a genuine categorization nuance but the paper groups by whether methods use factuality-specific training data, which is a reasonable distinction. The paper is transparent about what each method requires. — *Treat with caution.*

- **High correct recall as a "strength" being reframed** — The harsh critic argued that high correct recall is just a property of high-specificity classifiers. The paper's framing ("if a summary is assigned with low SBERTScore, then it is very likely to be unfaithful") is factually supported by the data in Table 8 and is a useful practical observation, even if the underlying mechanism is not novel. — *Treat with caution.*

- **Demanding evaluation on non-news domains** — This is a standard scope extension request. The paper uses standard benchmarks in this area. Listed as a Nice-to-Have instead. — *Treat with caution.*

## Novel Insights

The paper's most significant insight—that the failure of similarity metrics for factuality evaluation was not inherent to similarity-based methods but rather a consequence of using reference summaries as the comparison target—is well-demonstrated and potentially impactful. However, the negation case study reveals a deeper structural issue: sentence embeddings that conflate contradiction with topical relatedness (scores of 0.720 vs. 0.701) suggest that the sentence-transformer's training objective (paraphrase/NLI data) encodes topical similarity more strongly than logical contradiction, which fundamentally limits any similarity-based approach to factuality detection. This is not just a bug in SBERTScore but reflects a deeper constraint of the embedding paradigm for this task.

## Suggestions

- Scope the abstract's competitiveness claim to CNN/DM and similar settings, and explicitly note XSum as a limitation rather than explaining it away in a footnote.
- Add a same-backbone ablation (e.g., SBERTScore using vanilla roberta-large embeddings at sentence level, or BERTScore computed with all-roberta-large-v1 at word level) to cleanly separate the granularity effect from the backbone effect.
- Replace the "unfair" characterization with "suboptimal" or "mismatched" to more precisely describe the difference between reference-based and source-based comparison settings.

## Calibration Papers Referenced

1. **BooookScore** (/home/wg25r/review_agent/human_reviews/7Ttk3RzDeu.md) — avg score 8.5, Accept (oral). A thorough systematic study of book-length summarization with a new automatic metric validated against human annotations. Far more novel and comprehensive than SBERTScore, which is a straightforward extension. The current paper is significantly weaker.

2. **CDM** (/home/wg25r/review_agent/human_reviews/rYyu3jpk8z.md) — avg score 4.8, Reject. A novel evaluation metric for text generation with similar domain overlap (factuality in summarization). Reviewers noted missing baselines and limited generalizability. The current paper has more thorough experiments but a less novel methodological contribution.

3. **DetEmbedMetrics** (/home/wg25r/review_agent/human_reviews/OdoS6cH8MP.md) — avg score 2.0, Withdrawn/Reject. An embedding-based metric with overclaimed results and only synthetic evaluation. The current paper is substantially stronger: real benchmarks, honest limitations discussion, and genuine insights (reference-vs-source finding).

4. **BERTScore for inconsistency** (/home/wg25r/review_agent/human_reviews/0pbxX2jatP.md) — avg score 4.33, Reject. Uses BERTScore for measurement with limited novelty. The current paper provides more analysis and a clearer contribution, but similar concerns about overclaiming apply.

The current paper sits above the low-scoring anchors (2-3) due to genuine empirical insights and thorough analysis, but below medium anchors like CDM (4.8) due to limited methodological novelty (SBERTScore is a straightforward extension of BERTScore) and overclaimed abstract language. The reference-vs-source finding is the most valuable contribution but is an observational insight rather than a methodological innovation.

## Score and Decision

**Originality**: The methodological contribution (sentence-level BERTScore) is incremental. The reference-vs-source finding is an important observation but is an empirical insight rather than a novel technique. The combination analysis is straightforward.

**Importance of research question**: Factuality evaluation for summarization is important and well-motivated.

**Claims support**: Core claims are partially supported. The competitiveness claim holds for CNN/DM but not XSum. The granularity contribution is conflated with the backbone effect.

**Experimental soundness**: Experiments are thorough across multiple datasets and metrics, with appropriate statistical testing. The negation case study and error analysis are informative.

**Clarity**: The paper is well-written and clearly structured.

**Community value**: The reference-vs-source insight and granularity analysis are useful for practitioners, and the negative findings (negation insensitivity) are valuable.

Score: 5.0. The paper has genuine contributions (reference-vs-source finding, granularity analysis, error type analysis) that are valuable but partially undermined by overclaimed abstract language and an inability to cleanly isolate the sentence-level design contribution. It falls in the borderline range—informative empirical observations packaged as a more novel methodological contribution than the evidence supports.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>