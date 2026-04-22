Now I have a good understanding of the paper and the calibration anchors. Let me synthesize my review.

Let me verify some specific claims from the harsh critic against the paper text:

1. **Confounding backbone model**: The paper says in Section 4.3: "we use the pretrained RoBERTa-large as the backbone for BERTScore and all-roberta-large-v1 for SBERTScore. The two checkpoints have identical numbers of layers, and the only difference is that they are trained for different text embeddings." The paper does acknowledge this difference. Table 4 shows SBERTScore Word-Word (0.767) vs BERTScore Word-Word (0.759), which isolates granularity somewhat but still uses different checkpoints. The Sent-Sent improvement (0.779 vs 0.767) is the cleanest comparison within SBERTScore.

2. **XSum performance**: Table 7b confirms SBERTScore gets 0.605 vs BERTScore 0.695. The paper acknowledges this and speculates about single-sentence summaries.

3. **Negation insensitivity**: Table 5 confirms - SBERTScore gives 0.720 for contradictory pair (S1,S2) and 0.701 for neutral pair (S1,S4). The paper acknowledges this as a limitation.

4. **"Competitive" framing**: SBERTScore 0.720 on CNNDM vs QAFactEval 0.757, and on XSum SBERTScore 0.605 vs QAFactEval 0.705. The paper does say "can compete with" but also acknowledges QAFactEval outperforms on all splits.

5. **AND combination**: Figure 1 shows the results. The AND combination of DAE + QAFactEval gets 0.828 vs QAFactEval alone 0.817.

6. **Source vs reference**: The paper does credit Bao et al. (2023) for the idea of using source documents with BERTScore.

Now let me write the final review.</think>Now I have all the information I need. Let me synthesize the final review.

## Summary

The paper proposes SBERTScore, a sentence-level similarity-based metric for detecting factual inconsistencies in abstractive summaries. Rather than matching word embeddings (as in BERTScore), SBERTScore computes cosine similarity between sentence embeddings. The paper's main findings are: (1) similarity-based metrics previously failed primarily because they compared summaries to reference texts rather than source documents; (2) sentence-level granularity improves over word-level comparison; and (3) SBERTScore is competitive with trained NLI/QA-based metrics in zero-shot settings and excels at identifying correct summaries.

## Strengths

- **Clear empirical finding on comparison text selection**: Table 3 provides strong evidence that switching from reference to source comparison dramatically improves all similarity-based metrics (e.g., BERTScore: 0.500→0.759, ROUGE-2: 0.318→0.706). While Bao et al. (2023) pointed in this direction, this paper provides a thorough, multi-metric validation with comparison against other paradigms.

- **Thorough granularity ablation**: Table 4 systematically tests 9 granularity configurations (Sent-Sent, Doc-Sent, Sent-Doc, Doc-Doc, Mean-Sent, Sent-Mean, Mean-Mean, Word-Word for both BERTScore and SBERTScore), making the case for sentence-level processing empirically robust rather than assumed. The internal SBERTScore comparison (Word-Word 0.767 → Sent-Sent 0.779) cleanly isolates granularity within the same architecture.

- **Computational efficiency advantage is concrete**: Section 3.1 establishes O(N+M) vs O(NM) complexity for NLI-based methods, with empirical runtime measurements showing SBERTScore is 3× faster than SummaC and 30× faster than QuestEval.

- **High recall on correct summaries**: Table 8a shows SBERTScore achieves 0.522 recall on correct summaries in CNN/DM, notably higher than all other metrics (next best: DAE at 0.436). This is a distinctive and practically useful property — low SBERTScore scores are reliable indicators of factual inconsistency.

- **Honest analysis of limitations**: The negation case study (Table 5) transparently shows SBERTScore assigns 0.720 to a contradictory pair and 0.701 to a merely unrelated pair, and the paper explicitly states the XSum weakness and speculates about its cause.

## Weaknesses

### Fatal
None.

### Major

- **Confounded backbone comparison between SBERTScore and BERTScore**: The headline comparison relies on two different checkpoints — `roberta-large` for BERTScore vs. `all-roberta-large-v1` for SBERTScore — which differ in their training objectives (language modeling vs. sentence similarity). While Table 4's within-SBERTScore comparison (Word-Word 0.767 → Sent-Sent 0.779) partially isolates granularity, the main claim that "SBERTScore outperforms BERTScore" conflates architecture differences with embedding quality. The paper acknowledges both factors contribute (Section 5.3: "the improvement is brought by both the architecture and the appropriate text granularity") but does not attempt the straightforward control experiment of running BERTScore with `all-roberta-large-v1` or SBERTScore Word-Word with `roberta-large`, leaving the relative contribution of each factor unclear. This matters because the claimed contribution of "extending to sentence level" cannot be cleanly substantiated without it.

- **SBERTScore substantially underperforms on single-sentence summaries (XSum)**: On the XSum split (Table 7b), SBERTScore achieves balanced accuracy of 0.605 versus BERTScore's 0.695 — a 9-point gap — and ROC-AUC of 0.653 vs. 0.738. This is not a minor deficit; it means the core claim that "sentence-level similarity is better" holds only for multi-sentence summaries. The paper speculates this is because XSum summaries are single sentences, but offers no validation of this hypothesis (e.g., testing clause-level segmentation). The title and abstract make no such scope qualification.

- **Negation insensitivity is a fundamental limitation for a factuality metric**: Table 5 shows SBERTScore assigns nearly identical scores to a contradictory pair (0.720) and an unrelated pair (0.701). For a metric whose purpose is detecting factual inconsistency, the inability to distinguish contradiction from irrelevance undermines reliability in real-world deployment. The paper acknowledges this briefly but does not discuss whether this is an inherent limitation of cosine-similarity-over-embeddings approaches (which it likely is: sentence-transformer models trained on NLI/paraphrase data underrepresent negation).

### Minor

- **Logical AND combination is an unprincipled design with threshold-dependent artifacts**: The AND combination (Figure 1) uses independently selected thresholds for each metric rather than jointly optimizing. AND mechanically increases precision and decreases recall, so whether balanced accuracy improves depends on the operating point. The improvement of DAE AND QAFactEval (0.828) over QAFactEval alone (0.817) could simply reflect DAE acting as a precision booster at QAFactEval's threshold. No comparison against principled combination methods (e.g., score-level averaging, logistic regression) is provided, weakening the complementarity claim.

- **"Competitive with trained metrics" framing overstates results**: On CNN/DM, SBERTScore (0.720) is competitive with some trained NLI metrics (DAE 0.696, SummaC_ZS 0.686) but trails QAFactEval (0.757) by a meaningful margin. On XSum, SBERTScore (0.605) trails QAFactEval (0.705) by 10 points. SummaC_ZS is itself zero-shot but trained for an NLI task, creating a somewhat asymmetric comparison. The claim is not baseless but should be more carefully scoped.

- **No confidence intervals or per-dataset variance**: Given the diversity of the 9 sub-datasets (different sizes, annotator counts, error distributions), the absence of standard deviations or confidence intervals beyond significance t-tests makes it difficult to assess robustness of the reported differences.

### Trivial
None.

## Nice-to-Haves

- Running BERTScore with `all-roberta-large-v1` (or SBERTScore Word-Word with `roberta-large`) to cleanly isolate the granularity contribution from the backbone effect would significantly strengthen the paper's main claim.
- Testing clause-level or phrase-level segmentation on XSum summaries to validate the hypothesis about why SBERTScore underperforms on single-sentence inputs.
- Comparing against principled combination baselines (score averaging, logistic regression with jointly optimized thresholds) to better support the complementarity finding.

## Removed Points

- **"Formatting/style issues"**: Removed — these are parser artifacts, not real issues.
- **"Missing related work"**: Removed — per instructions, I cannot verify existence of uncited works.
- **"Reproducibility concerns about models/checkpoints"**: Removed — the cited models and checkpoints are assumed to exist as stated in the paper.
- **"Missing appendix/proofs"**: Removed — these sections were stripped by the parser and exist in the original submission.
- **"Bao et al. (2023) already proposed source-based comparison, so contribution is overstated"**: Partially removed — the paper does acknowledge Bao et al. (Section 2.3: "Bao et al. (2023) attempted to feed source document to BERTScore, but they did not compare its performance against metrics using other methodologies"). The contribution of showing this systematically across multiple metrics and paradigms is genuine, though the harsh critic's point that this partially diminishes novelty is noted.
- **"Missing confidence intervals as a fatal flaw"**: Downgraded from fatal to minor — in the benchmark evaluation community for this subfield, reporting balanced accuracy with t-tests is standard practice; confidence intervals would be nice but are not fatal.

## Novel Insights

The most striking insight, which the paper demonstrates but does not fully emphasize, is the asymmetry in SBERTScore's error profile: it achieves exceptional recall on *correct* summaries (0.522 vs. next-best 0.436 on CNN/DM) while being relatively weak at detecting actual errors (0.454/0.436 on intrinsic NP/Predicate errors in CNN/DM). This means SBERTScore functions essentially as a high-specificity screening tool — a low score is a reliable alarm for inconsistency — rather than a well-rounded factuality metric. This reframes its practical utility: it is best used as a first-pass filter rather than a standalone evaluator, which aligns with the AND-combination finding but suggests a more nuanced combination strategy (using SBERTScore as a recall-oriented pre-filter before applying more expensive NLI/QA metrics) might be more principled than threshold-dependent logical AND.

## Suggestions

- Add a controlled experiment comparing BERTScore using `all-roberta-large-v1` word embeddings to isolate the granularity effect; this is a straightforward experiment that would substantially strengthen the core contribution claim.
- Qualify the "SBERTScore outperforms BERTScore" claim to specify that it holds for multi-sentence summaries, and explicitly note the XSum limitation in the abstract.
- Replace or supplement the AND combination with at least one principled alternative (e.g., logistic regression on raw scores) to demonstrate that the complementarity finding is not an artifact of threshold selection.

---

**Originality**: The main insight (source > reference for similarity metrics) was partially anticipated by Bao et al. (2023), but this paper extends it to sentence-level granularity and provides comprehensive benchmark comparisons. The SBERTScore method itself is a straightforward extension (sentence embeddings vs. word embeddings in BERTScore's matching framework). **Moderate originality**.

**Importance of research question**: Summary factuality evaluation is an important and active area. Understanding when and why similarity-based metrics work (or fail) has practical implications. **Good importance**.

**Claims well supported**: The main claims are partially undermined by the confounded backbone comparison and the XSum failure. The source-vs-reference claim is well supported; the sentence-level improvement claim is partially confounded; the "competitive with trained metrics" claim is overgeneralized. **Partially supported**.

**Soundness of experiments**: The experimental setup is generally thorough, using a standard benchmark with multiple datasets. The granularity ablation is systematic. However, the lack of a controlled backbone experiment and absence of variance reporting are notable gaps. **Reasonably sound but with a meaningful confound**.

**Clarity**: The paper is well-structured and clearly written, with good separation of experimental questions. The negation case study is transparent. **Good clarity**.

**Value to community**: The finding that reference-based comparison was the primary failure mode of similarity metrics, and the error-profile characterization, provide practical value even with the methodology limitations. The computational efficiency argument is also useful for practitioners.

## Calibration Anchors

| Paper | Path | Avg Human Score | Comparison |
|-------|------|----------------|------------|
| CDM (Contrastive Distribution Methods) for text evaluation | rYyu3jpk8z.md | 4.80 (Reject) | Similar: proposes a new evaluation metric for text generation including factuality in summarization, with empirical validation. CDM had missing baselines and applicability limitations. This paper has a more thorough evaluation but similar scope limitations and confounded comparisons. Slightly stronger than CDM. |
| LLM Embedding Design study | CWAvMSNUqT.md | 4.50 (Reject) | Similar: controlled empirical study of design choices for embeddings, with confounded comparisons flagged by reviewers. This paper's confounded backbone comparison is analogous to their concern. However, this paper has clearer practical implications. |
| Beyond Correlation (human uncertainty in evaluation) | E8gYIrbP00.md | 6.75 (Accept Poster) | Higher quality: identifies a fundamental methodological issue (correlation masking confounds) with principled solutions. Our paper identifies a practical insight (source > reference) but with more confounds in its own experiments. |
| FreeLM (0.3B outperforms GPT-3 claims) | qgLyKwXVDs.md | 2.00 (Reject) | Much weaker: fundamentally confounded comparison (multitask-fine-tuned model vs. in-context-learning, no fair baselines). Our paper is far more honest and empirically grounded than this. |
| BERTScore as diversity metric | jvRCirB0Oq.md | 3.40 (Withdrawn/Reject) | Weaker: BERTScore shows minimal differentiation and overclaimed results. Our paper has more substantive findings. |
| Confounded evaluation metric (ACR for randomized smoothing) | KX5hd1RhYP.md | 4.67 (Reject) | Similar: demonstrates that a dominant metric is fundamentally confounded. Our paper has a confound in its own comparison but identifies a real insight about reference vs. source. |

Relative to these anchors: This paper is stronger than CDM (4.80), the embedding design study (4.50), and the ACR paper (4.67) because it has a clearer practical insight (source > reference) with comprehensive experiments across multiple datasets and metrics. It is weaker than "Beyond Correlation" (6.75) because that paper identifies a fundamental methodological issue and proposes principled solutions, while this paper has a meaningful confound in its own methodology (backbone comparison) and an important scope limitation (XSum). The paper falls in the 5-6 range:有价值贡献但有notable weaknesses that limit the strength of its claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>