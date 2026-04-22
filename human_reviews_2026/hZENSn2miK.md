# Optimizing What Matters: AUC-Driven Learning for Robust Neural Retrieval

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Dual‑encoder retrievers depend on the principle that relevant documents should score higher than irrelevant ones for a given query. Yet the dominant Noise Contrastive Estimation (NCE) objective, which underpins Contrastive Loss, optimizes a softened ranking surrogate that we rigorously prove is fundamentally oblivious to score separation quality and unrelated to AUC. This mismatch leads to poor calibration and suboptimal performance in downstream tasks like retrieval‑augmented generation (RAG). To address this fundamental limitation, we introduce the MW loss, a new training objective that maximizes the Mann‑Whitney U statistic, which is mathematically equivalent to the Area under the ROC Curve (AUC). MW loss encourages each positive-negative pair to be correctly ranked by minimizing binary cross entropy over score differences.  We provide theoretical guarantees that MW loss directly upper-bounds the AoC, better aligning optimization with retrieval goals. We further promote ROC curves and AUC as natural threshold‑free diagnostics for evaluating retriever calibration and ranking quality. Empirically, retrievers trained with MW loss consistently outperform contrastive counterparts in AUC and standard retrieval metrics. Our experiments show that MW loss is an empirically superior alternative to Contrastive Loss, yielding better-calibrated and more discriminative retrievers for high-stakes applications like RAG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates the limitations of the Contrastive Loss (CL) for training dense neural retrievers and proposes a alternative called Mann-Whitney (MW) loss. The MW loss is a simple yet principled objective that directly maximizes the Area Under the ROC Curve (AUC) by minimizing binary cross-entropy over pairwise score differences. Theoretical guarantees and empirical evaluations demonstrate MW loss’s effectiveness in improving calibration and retrieval performance compared to CL.

### Strengths
**Novelty:** The paper addresses a critical issue in dense retriever training and proposes a theoretically sound and empirically effective solution. The focus on global score calibration and AUC maximization is well-justified and has the potential to improve the reliability and applicability of neural retrievers in various domains.

**Evaluation:** The authors conduct extensive experiments on various datasets and models, demonstrating the effectiveness of MW loss in both in-distribution and out-of-distribution scenarios. The results consistently show that MW loss outperforms CL in terms of AUC and standard retrieval metrics.

**Clarity:** The paper is well-written and organized, with clear explanations of the problem, proposed method, and experimental results.

### Weaknesses
**Generalization:** Exploring the performance of MW loss on various models would strengthen the paper’s claims and provide a more comprehensive understanding of its generalizability.

**Efficiency:** While the paper mentions the computational differences between MW loss and CL, a more detailed analysis of the computational cost and its impact on training efficiency would be beneficial.

**Other Methods:** While the paper focuses on comparing MW loss with CL, a comparison with other recent approaches for improving retrieval performance, such as margin-based losses or data augmentation techniques, would provide more of MW loss’s advantages or limitations.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose MW loss, which directly maximizes AUC by minimizing BCE over score differences between positives and negatives sampled from different queries. They prove MW upper-bounds AoC, show the computation reuses the same embedding/similarity passes as contrastive while doing more pairwise comparisons in the loss, and report consistent gains in AUC with competitive or better MRR/nDCG in-domain and on BEIR. Convergence is slower but argued to reflect the harder objective.

### Strengths
1. clear theoretical critique

2 embeddings/similarities are unchanged vs. contrastive; only the aggregation does B(B−1) pairwise comparisons in the loss, which should vectorize well.

3. results show broad AUC gains and frequent MRR/nDCG improvements across MiniLM/XLM-R-Base/Large, with plots of per-dataset gains.

### Weaknesses
1. The baseline design largely collapses to CL vs. MW; this isolates the loss but under-probes stronger retrievers or alternative AUC-aware objectives, leaving open whether MW is the best practical route to calibration and top-k retrieval. 

2. In-domain improvements on MRR/nDCG for small/base models are modest on average even as AUC rises, so the paper should clarify when AUC gains meaningfully translate to ranking quality. 

3. The compute section argues “same embedding/similarity cost,” but the extra B(B−1) BCE terms can still be a real bottleneck; wall-clock, peak memory, and throughput should be reported for typical B, H, corpus sizes. Convergence is slower; this is acknowledged but not quantified against a fixed training budget. 

4. Sensitivity to 100/1k, as well as to full-corpus vs. ANN sampling, would strengthen external validity.

### Questions
1. Eq. (2) and Lemma 2: clarify the role of temperature τ in ℓ_BCE vs. the τ used in contrastive; report sensitivity curves of AUC/MRR to τ and whether τ→∞ reduces to a hinge-like objective. 

2. the AUC protocol picks top-500 corpus negatives; add a sensitivity analysis for 100/500/1000 and a variant using random-plus-hard mixes to show robustness.

3. How stable is the global threshold learned by MW across domains—can a single score cut transfer from NLI-trained models to BEIR subsets without per-task tuning?

4. Do AUC gains predict top-k gains? On which datasets do we see high ΔAUC but flat ΔMRR/nDCG, and why? Any evidence of MW optimizing separation where intra-query ordering is less affected? 


5. Any failure modes with multi-relevant queries or table-heavy evidence where global calibration helps less than within-query ranking? Please include targeted analyses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
RAG pipelines typically use retrievers based on dual-encoder models. These retrievers are often trained using contrastive learning (e.g., with an InfoNCE loss). However, this type of training suffers from the fact that the resulting relevance scores are not directly comparable across queries — a phenomenon long known in the Information Retrieval community — which makes a simple, globally applied thresholding strategy unlikely to succeed.

To address this, the authors propose the MW loss, based on the Mann–Whitney U statistic (specifically, a differentiable proxy of this statistic). This loss serves as an upper bound on the AoC (Area over the Curve), thereby directly aligning training with a globally calibrated retrieval objective.

### Strengths
* The paper is well written, well structured, easy to read and understand.
* The underlying claims are sound, at least within the (limited) context the authors are considering.
* It indeed offers a new loss that ensures consistency of relevance across queries; this implies that complex query-dependent thresholding strategies are no longer needed.

### Weaknesses
(1) The main weakness of the paper is that it does not truly optimize what matters (contrary to what its title suggests).

In Information Retrieval (IR) and, in particular, in Retrieval-Augmented Generation (RAG), what ultimately matters is ranking quality, rather than classification accuracy with respect to an artificially defined binary relevance label. In this context, the AUC metric is rarely used to evaluate the quality of a ranking method, for several well-known reasons:

- It does not depend on the actual positions of items in the ranked list, effectively treating the problem as classification rather than ranking.

- It assumes binary relevance, whereas in practice relevance is often graded or continuous.

- It is pairwise rather than listwise, and it is well established that listwise approaches tend to be more effective.

(2) The choice of baselines for comparison is also quite weak: only a single baseline is considered. Since this baseline is not trained using the same loss function, the comparison—at least for the AUC criterion—appears either unfair or trivial.

See the next section (Questions) for suggestions of additional baselines and ablation studies.

### Questions
* How do you position your method with respect to the RankNet objective function, which is fairly standard in the (older) Information Retrieval (IR) literature and can be seen as a differentiable proxy of the AUC, albeit computed at the level of a single query?
* Score normalization and calibration methods, typically applied as post-processing steps, are widely used in the IR community to make relevance scores more consistent and comparable across queries. How does your approach relate to, or compare with, these existing techniques?
* Given that the choice of baselines is somewhat limited (only a single method is considered, and arguably not the most representative of current practice in the RAG community), it would have been valuable to include additional ablations or variants, such asvthe use of other differentiable surrogates of the indicator function 1{𝑧≤0} (e.g., hinge loss, etc.);
* the influence of the temperature parameter;
* a deeper analysis of the somewhat counterintuitive observation that the proposed method preserves or even improves within-query (fine-grained) ranking quality. This last point is not obvious, since AoC is not inherently a ranking loss but rather a classification-oriented objective. Ideally, it would have been interesting to explore multi-valued (non-binary) relevance scenarios to validate this claim.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
It introduces a Mann–Whitney (MW) loss and proves an upper bound on the Area-over-ROC, so minimizing MW encourages higher AUC

### Strengths
It introduces a MW loss, showing that minimizing MW tightens the AoC bound and encourages higher AUC.

### Weaknesses
While the paper positions MW loss as a principled AUC-aligned alternative to Contrastive Loss, it does not compare against other well-known pairwise and listwise ranking objectives (e.g., margin ranking loss, triplet loss, RankNet/LambdaRank...).

Some of these losses optimize s(q, p⁺) − s(q, p⁻) directly, which is structurally very close to the proposed MW loss. Without such comparisons, it is difficult to assess whether MW loss provides a fundamentally new advantage.

### Questions
While AUC is theoretically meaningful for score calibration, retrieval benchmarks are primarily evaluated using top-k metrics (nDCG, MRR, Recall@k). The gains in these metrics are sometimes small or even mixed. It remains unclear whether optimizing AUC is the right objective for retrieval effectiveness in practice?

### Soundness
2

### Presentation
3

### Contribution
3
