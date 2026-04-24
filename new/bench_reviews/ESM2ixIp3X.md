## Summary

This paper revisits similarity-based metrics for summary factual consistency evaluation, demonstrating that their prior poor performance was largely due to comparing summaries against reference texts rather than source documents. It proposes SBERTScore, a sentence-level embedding similarity metric that operates zero-shot. The paper shows that source-based comparison dramatically improves balanced accuracy (Table 3), and that sentence-sentence granularity outperforms word-level and document-level alternatives (Table 4). On the CNN/DM split, SBERTScore achieves competitive results against trained NLI and QA-based metrics while being computationally cheaper.

## Strengths

- **Controlled evidence that prior failures of similarity-based metrics were confounded by experimental settings**: Table 3 shows that when BERTScore and SBERTScore compare summaries to source documents rather than reference summaries, balanced accuracy jumps from near-random (~0.50) to 0.759 and 0.779 respectively, with all source-based results significantly higher (p < 0.05). This directly challenges the consensus from Maynez et al. (2020), Pagnoni et al. (2021), and Durmus et al. (2020).
- **Thorough granularity ablation**: Table 4 systematically tests eight granularity configurations, establishing that sentence-sentence matching (0.779) is optimal and significantly better than word-word (0.767), document-level (0.576–0.746), and mean-pooling variants (0.512–0.602). This provides an empirical justification that prior work lacked.
- **Strong CNNDM results and practical efficiency**: On the CNN/DM split (Table 7a), SBERTScore achieves 0.720 balanced accuracy and 0.804 ROC-AUC, surpassing zero-shot NLI methods and trained metrics such as QuestEval (0.670) and DAE (0.696). The paper also derives and reports that SBERTScore is 3× faster than NLI-based SummaC and 30× faster than QA-based QuestEval.
- **High recall on correct summaries**: Table 8a shows SBERTScore achieves the highest recall on correct CNN/DM summaries (0.522 vs. 0.436 for DAE and 0.401 for QAFactEval), indicating it is conservative and potentially useful as a low-false-positive filter.

## Weaknesses

### Fatal
None.

### Major

- **Combination analysis confounds dataset coverage**: DAE is explicitly excluded from XSum-origin datasets in Table 7b due to training-set overlap (Section 4.1), yet it appears in the combination matrix (Figure 1). The paper does not clarify whether all metric pairs in Figure 1 were evaluated on identical data distributions. For instance, DAE ∧ QAFactEval reports 0.828 while QAFactEval alone reports 0.817; however, the former may exclude XSum data whereas the latter is averaged over the full benchmark. Without evidence that the comparison is fair, the claim that a “simple combination outperforms the state of the art” (Introduction) is unsupported.
- **Abstract overstates generality of SBERTScore’s superiority over BERTScore**: The abstract claims SBERTScore “outperforms widely-used word-word metrics including BERTScore” without qualification. However, on the XSum split—where all summaries are single-sentence—SBERTScore drops to 0.605 balanced accuracy, trailing BERTScore’s 0.695 (Table 7b). While the body honestly acknowledges this limitation (Section 5.5: “it is not as good as BERTScore on the XSum split…leading to degeneration”), the abstract’s unqualified claim is misleading because single-sentence summarization is a standard and important setting.

### Minor

- **Shallow analysis of metric complementarity**: Section 5.6 shows that logical AND can improve balanced accuracy for some pairs, but most combinations do not improve over both parents, and the analysis does not establish whether gains stem from genuine complementary error coverage or simply from variance reduction via double-checking.
- **Negation case study is anecdotal**: The case study in Section 5.4 relies on four hand-crafted sentences (Table 5). While honestly reported, it offers limited generalizable insight into how the metric behaves on naturally occurring negation.

### Trivial
None.

## Nice-to-Haves

- A hybrid fallback that uses SBERTScore for multi-sentence summaries and BERTScore for single-sentence inputs would strengthen practical applicability and address the known XSum degeneration.
- Deeper analysis disentangling whether sentence-SBERT gains come from better semantic representations or from the statistical benefit of averaging over multiple sentence comparisons.
- Confidence intervals or bootstrap variances for the aggregated scores in Table 7.

## Removed Points
These points are flagged to be removed; treat them with caution.

- “The claim that SBERTScore can compete with NLI/QA metrics overstates evidence and conflates zero-shot with supervised settings” — The paper uses the vague phrase “can compete,” which is defensible given that SBERTScore beats some trained metrics on CNNDM and overall. The body honestly reports XSum gaps, so this criticism is overstated.
- “SBERTScore degenerates to near-random performance on XSum” — 0.605 balanced accuracy is 10.5 points above chance (0.5), so “near-random” is an exaggeration. The paper itself appropriately labels this as degeneration due to the single-sentence boundary condition.
- “A hybrid single-sentence fallback should be tested” and “variance/calibration should be reported” — These are reasonable suggestions but not requirements for the core contribution.
- Any formatting, grammar, or typo complaints — These are parser artifacts, not author errors.
- Any criticisms about missing appendices or proofs — The parser strips these sections; they exist in the original submission.

## Novel Insights

The paper’s central insight—that prior negative findings about similarity-based metrics were confounded by the choice of comparison text (reference vs. source)—is genuinely valuable and reframes a long-standing consensus in the summarization evaluation literature. This observation has implications beyond SBERTScore, suggesting that any embedding-similarity metric for factuality should be evaluated against source documents, not reference summaries. If the authors more carefully scope their claims to the multi-sentence regime and resolve the combination-analysis confound, this work could usefully redirect how the community deploys similarity-based factuality metrics.

## Suggestions

- Reframe SBERTScore in the abstract and conclusion as a multi-sentence metric that improves over BERTScore in that regime, and explicitly note the single-sentence limitation rather than presenting an unqualified superiority claim.
- Clarify in Figure 1 whether all combinations were evaluated on identical data distributions; if not, recompute combinations on the intersection of available data for each pair, or exclude DAE from the combination matrix to avoid coverage confounds.
- Add bootstrap confidence intervals to the aggregated scores in Table 7 to improve reproducibility and interpretability.

## Score and Decision

**Calibration comparison:**
- *High anchor*: `/home/wg25r/review_agent/human_reviews/7Ttk3RzDeu.md` (BooookScore, avg 8.50, Accept oral) — far above our paper in scale, human evaluation rigor, and novelty.
- *Medium-high anchor*: `/home/wg25r/review_agent/human_reviews/phBS6YpTzC.md` (GV-consistency, avg 6.67, Accept poster) — comparable empirical breadth and some framework issues, but with a clearer main claim and less overstatement.
- *Medium anchor*: `/home/wg25r/review_agent/human_reviews/1tBvzOYTLF.md` (RevisEval, avg 6.00, Accept poster) — proposes a new evaluation metric with similar experimental depth; accepted despite concerns about novelty and cost. Our paper has stronger core empirical findings but a more serious methodological issue in the combination analysis.
- *Medium-low anchor*: `/home/wg25r/review_agent/human_reviews/mIl15VP7vt.md` (IRT evaluation, avg 6.50, Reject) — larger-scale experiments but rejected for abstract overclaiming and missing adaptive-monitoring evidence. Our paper is smaller in scale but its core experiments are better executed.
- *Low anchor*: `/home/wg25r/review_agent/human_reviews/rYyu3jpk8z.md` (CDM, avg 4.80, Reject) — missing essential baselines and unclear applicability. Our paper is clearly better: it uses fair comparisons, proper baselines, and acknowledges limitations.

Our paper sits between the accepted poster band (~6.0) and the medium-low reject band (~5.0–6.0). The source-vs-reference finding (Table 3) and the granularity ablation (Table 4) are genuinely useful contributions supported by solid experiments. However, the abstract’s unqualified superiority claim and the combination-analysis confound are substantive issues that undermine two of the paper’s stated contributions. These flaws are addressable in revision but are significant enough to keep the paper below the current acceptance threshold.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>