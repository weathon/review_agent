Now I have enough calibration data. Let me synthesize the final review.

## Summary

CHECKEMBED proposes verifying LLM outputs on open-ended tasks by embedding whole answers with modern embedding models (e.g., GPT Text Embedding Large) and comparing cosine similarities between multiple sampled answers (and optionally ground truth). The paper argues this answer-level embedding approach is simpler, faster, and more accurate than token-/sentence-level methods like BERTScore and SelfCheckGPT for tasks such as legal term extraction and document summarization.

## Strengths

- **Clear and practical methodology**: The core idea—embed whole answers and compare via cosine similarity—is simple to implement and highly scalable. The pipeline (Figure 2) is well-structured and easy to follow, making it accessible for practitioners.

- **Strong runtime advantages convincingly demonstrated**: Section 3 provides a clear asymptotic analysis, and Figure 7 provides empirical confirmation of 30×–300× speedups over baselines. This is a genuine and well-supported practical contribution.

- **Competitive benchmark performance on WikiBio**: On the WikiBio hallucination benchmark (Table 1), CHECKEMBED achieves the best Spearman correlation (76.2) and competitive Pearson correlation (73.6 vs SelfCheckGPT-NLI's 74.1), while being much faster. This shows that coarse embedding-based scores can meaningfully track hallucination severity.

- **Effective distinction of semantically similar vs. different passages**: Figures 3a and 3b convincingly show that CHECKEMBED separates semantically similar and different replies with little overlap, where baselines like BERTScore show substantial overlap. This demonstrates the practical value of leveraging modern embedding models for semantic-level comparison.

- **Systematic ablation of embedding models and sample sizes**: Testing across 6+ embedding models (Section 4) and varying k (Figure 8) provides useful practical guidance on design choices.

## Weaknesses

### Fatal
None.

### Major

- **Core verification claim conflates semantic similarity with correctness.** The paper's central claim is that CHECKEMBED enables "verification" of LLM answers and "assessing truthfulness." However, answer-level embedding similarity fundamentally measures self-consistency and topical similarity, not factual correctness. A model that is confidently and consistently wrong will produce tight embedding clusters that CHECKEMBED would flag as "high quality" (Section 4.2: "whenever CHECKEMBED has very high confidence in its answer…there is high likelihood that these replies are close to the corresponding GT"). This is a structural misalignment between the metric and what it claims to measure. The paper should sharply restrict claims to "semantic similarity/self-consistency detection" rather than "verification" or "truthfulness assessment."

- **Evaluation on the primary claimed use cases is primarily qualitative and anecdotal.** The legal term extraction experiments (Section 4.2) show only two cherry-picked heatmaps (Figure 4) with post-hoc "high confidence" / "low confidence" labels. There are no accuracy metrics, no ROC curves, no calibration analysis, no statistics over many documents, and no error analysis of cases where CHECKEMBED fails. The proposed thresholds (e.g., "mean > 0.9, std < 0.05") are stated without any quantitative evaluation of their precision or recall. The paper's strongest claims—about being an "accurate verification" method for open-ended tasks—rest on these qualitative observations, which is insufficient evidence for the claim's magnitude.

- **Missing important baselines in the verification/hallucination detection space.** The paper compares primarily to BERTScore and SelfCheckGPT. It does not compare to LLM-as-a-judge methods (now standard for open-ended evaluation), Semantic Entropy (Kuhn et al., 2023), or INSIDE/EigenScore (which also uses embedding-space consistency for hallucination detection and was published at ICLR 2024). These are directly relevant contemporaneous methods that overlap significantly with CHECKEMBED's approach. The paper explicitly excludes them (Section 5) citing methodological distinctions, but this leaves a significant gap in establishing what CHECKEMBED adds beyond applying off-the-shelf embedding models.

### Minor

- **Fine-grained hallucination results undercut the paper's claims.** Section 4.4 and Figures 5–6 show that CHECKEMBED "maintains high scores even with errors" and that meaningful separation only begins beyond ~5 introduced errors. The paper acknowledges this but does not clearly scope its method to coarse-grained verification. The narrative tries to claim positive results ("it can also recognize hallucinations after introducing a single error") while the data shows heavy overlap between 0-error and 1–5 error distributions. This should be stated more honestly as a known limitation.

- **WikiBio performance is competitive but not clearly superior.** On Pearson correlation (Table 1), SelfCheckGPT-NLI (74.1) outperforms the best CHECKEMBED variant (73.6), yet the paper claims "significant improvements" and "robust performance." The wording should more accurately reflect that CHECKEMBED is competitive on Pearson and better on Spearman, with the advantage being primarily in speed.

- **Legal dataset details are underspecified.** The paper's main practical use case (Section 4.2) uses "real" data from an "in-house legal analytics project" without describing the dataset size, document count, or how ground truths were constructed. This limits evaluation of generality claims.

### Trivial

- Some terminology is inconsistent (e.g., "truthfulness" vs. "similarity" vs. "verification" used interchangeably for different concepts).

## Nice-to-Haves

- Comparison against LLM-as-a-judge and Semantic Entropy baselines, which would substantially strengthen the paper's positioning.
- A systematic threshold analysis (ROC/precision-recall curves) on WikiBio or another labeled benchmark to validate the claimed decision thresholds.
- An analysis of failure cases where CHECKEMBED gives high similarity to factually incorrect but semantically coherent answers.
- An explicit discussion of context window limitations for embedding very long documents.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claims about missing LLM-as-a-judge baselines from the Neutral Reviewer and Spark**: While LLM-as-a-judge is relevant, the paper scopes its comparison to embedding-based and token-level methods. This is a legitimate methodological choice; LLM-as-a-judge operates at a fundamentally different cost and latency point. However, the related work discussion should acknowledge this distinction. (Kept as a minor point noting the baselines gap, but removed the assertion that this invalidates all claims.)

- **Harsh Critic's claim that BERTScore/SelfCheckGPT comparison is unfair because those baselines are "not intended" for answer-level comparison**: This is partially valid—the paper argues this is precisely the point, that existing baselines fail on open-ended tasks. However, the paper should be clearer that it demonstrates these methods' limitations on a task type they weren't designed for, rather than claiming general superiority. This asymmetry actually favors the baselines (as per review rules), so it's acceptable to show BERTScore/SelfCheckGPT underperform on this task type.

- **Reproducibility concerns about the GPT Text Embedding Large model**: Per review rules, we do not question the existence or availability of cited models.

- **Missing confidence intervals / statistical tests**: Single-run evaluation on benchmarks is standard in this area. Requesting significance tests is a nice-to-have rather than a core flaw.

- **Formatting and notation consistency issues**: These are style nitpicks per the review rules.

- **Harsh Critic's claim about "no demonstration that practical configurations of SelfCheckGPT cannot be made tractable"**: The runtime advantage is empirically demonstrated (Fig. 7) and the asymptotic analysis supports it. The claim that practical configurations could be optimized is speculative without evidence.

- **Missing related work claims from the Human Finder about Semantic Entropy, INSIDE**: These are noted in the major weakness about missing baselines, but the Human Finder's claim that these are "one of the most established methods" requires external verification we cannot confirm. Kept as a recommendation to compare, not as a fatal omission.

- **Human Finder weakness #1 about anisotropic embedding spaces**: This is a valid theoretical concern but is generic to all embedding-based methods and not specific to CHECKEMBED's design. Moved to a minor concern.

- **Spark's suggestion about context window limitations**: This is a valid practical concern but not a demonstrated failure mode; moved to nice-to-have.

- **Harsh Critic's claim about "circular" evaluation in Section 4.2**: The observation that "high similarity among samples + to GT is taken as evidence" is not circular per se—it's correlational evidence. However, the limited sample size (2 examples) is a valid concern, kept as part of the anecdotal evaluation issue.

## Novel Insights

The paper's most interesting empirical finding is the clear separation in Figures 3a/3b showing that modern embedding models can distinguish semantically similar from semantically different text passages with very little score overlap, whereas BERTScore and SelfCheckGPT-BERT fail dramatically. This suggests that the bottleneck in previous verification methods was not the comparison methodology per se, but the granularity of representation: token-/sentence-level embeddings fail on open-ended tasks precisely because they cannot capture whole-answer semantics. However, this insight also highlights the key limitation: answer-level embeddings will by definition smooth over fine-grained factual errors, making CHECKEMBED complementary to (rather than a replacement for) sentence-level methods.

## Suggestions

1. **Reframe claims**: Replace "verification" and "truthfulness assessment" with "semantic similarity detection" and "self-consistency scoring" throughout. Clearly scope CHECKEMBED as a coarse-grained verification tool that excels at detecting answer-level disagreement, with known limitations for fine-grained factual errors.

2. **Add a failure mode analysis**: Explicitly show cases where CHECKEMBED gives high similarity scores to consistently incorrect answers, and discuss the conditions under which self-consistency does or does not correlate with correctness.

3. **Add Semantic Entropy and INSIDE/EigenScore as baselines** in the WikiBio evaluation to position CHECKEMBED relative to the most directly related concurrent work.

4. **Add quantitative evaluation on the legal dataset**: Report accuracy, AUROC, or threshold-based metrics across a representative set of documents, not just 2 case studies.

## Score and Decision

**Calibration comparison:**
- INSIDE/EigenScore (Zj12nzlQbz): Similar proposal (embedding-space consistency for hallucination detection), but with a more novel metric (EigenScore) and broader evaluation. Scored 8/6/6/6 → accepted as poster.
- Semantic Clustering for Hallucination Detection (GXzwq6waYb): Similar idea (sentence embedding clustering + semantic entropy), limited novelty. Scored 3/3/3/8 → rejected/withdrawn.
- Improving UQ via Semantic Embeddings (N4mb3MBV6J): Very similar core idea (pairwise cosine similarity of response embeddings for uncertainty), scored 6/5/6 → rejected.
- Beyond Cosine Similarity / USMB (EwRxk3Ho1V): Benchmarking embedding similarity metrics, limited novelty. Scored 3/6/3/5 → rejected.

CHECKEMBED is very close in core idea to the N4mb3MBV6J (SEU) paper, which also uses average pairwise cosine similarity of answer embeddings for uncertainty estimation and was rejected. CHECKEMBED has arguably better empirical coverage (WikiBio + document analysis tasks) and a more complete pipeline, but faces similar novelty concerns (embedding whole answers and computing cosine similarity is a straightforward application of existing embedding models) and has a more significant overclaiming problem (claiming "verification" and "truthfulness" when measuring consistency/similarity). Compared to INSIDE (accepted at ~6.5), CHECKEMBED lacks methodological novelty (EigenScore vs. simple cosine similarity) and has weaker evaluation (primarily qualitative on the main use cases).

The paper has a genuine practical contribution (scalability + simplicity) and reasonable benchmark performance, but overclaims its capabilities and has inadequate evaluation on its primary use case. The novelty is incremental given concurrent work like SEU and Semantic Entropy.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>