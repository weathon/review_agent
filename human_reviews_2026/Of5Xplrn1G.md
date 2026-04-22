# Learning from the Best, Differently: A Diversity-Driven Rethinking on Data Selection

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
High-quality pre-training data is a decisive factor for large language models, where quality captures factual reliability and semantic value, and diversity ensures broad coverage and distributional heterogeneity. Existing approaches typically rely on single or multiple-dimensional score-based selection. However, empirical studies have shown that directly selecting top-scored data often degrades downstream performance, and sampling from a broader range is required to recover results. The above non-monotonicity between the dataset scores and the downstream benchmark results reveals a fundamental bias: score-based methods collapse correlated dimensions, causing top-scored data to appear high-quality while systematically overlooking diversity. We argue that ensuring diversity requires decomposing correlated evaluation metrics into orthogonal feature dimensions, from which the top-scored data can be directly selected. To this end, we proposed the Orthogonal Diversity-Aware Selection (ODiS) algorithm, a method to preserve both quality and diversity during high-quality data selection. First, ODiS evaluates data from multiple dimensions, covering language quality, knowledge quality, and comprehension difficulty. The resulting multi-dimensional scores are then decorrelated via Principal Component Analysis (PCA), yielding orthogonal evaluation dimensions. For each dimension, a Roberta-based scorer is trained to regress the data onto PCA-projected scores, enabling scalable inference on large corpora. Finally, ODiS constructs the training dataset by selecting top-scored data within each orthogonal dimension, thereby ensuring both quality and diversity.  Empirical results show that ODiS-selected data exhibit less than 2\% inter-dimension overlap, confirming the orthogonality between dimensions. More importantly, models trained with ODiS-selected data significantly outperform other baselines on multiple downstream benchmarks, highlighting the necessity of orthogonal, diversity-aware data selection for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ODiS (Orthogonal Diversity-Aware Selection), a data selection method for LLM pre-training that addresses the observation that directly selecting top-scored data often degrades downstream performance. Authors identify neglected diversity as the fundamental cause of this non-monotonicity between dataset scores and model performance. Their approach evaluates data across 11 dimensions (covering language quality, knowledge quality, comprehension difficulty, and information quality), applies PCA to decorrelate these dimensions into orthogonal components, trains RoBERTa-based scorers to predict scores along each principal component, and constructs training datasets by selecting top-scored data from each orthogonal dimension. Experiments on a 1.5B parameter model trained with 100B tokens show ODiS outperforms baseline methods on five downstream benchmarks.

### Strengths
1. The paper provides valuable analysis showing that top-scored data consistently underperforms compared to sampling from broader score ranges (Figure 1), identifying neglected diversity as the root cause.

2. Using PCA to decorrelate evaluation dimensions is a systematic and well-motivated solution.

3. The paper convincingly demonstrates improved diversity through multiple analyses. UMAP visualizations showing broader distribution (Figure 2a, 4b), increased pairwise distances (Figure 2b, 4a), correlation analysis revealing dimension orthogonality (Figure 5), and minimal inter-dimension overlap (Figure 6b).

### Weaknesses
1.  All experiments use only the Nemotron-CC Chinese dataset with a single 1.5B model size and 100B token budget. No experiments on English data, different model scales (e.g., 400M, 7B, as conducted in existing papers on LLM data selection) or varying data budgets are provided. This raises serious questions about whether findings generalize across languages, scales, and domains. 

2. The paper mentions several recent multi-dimensional methods (Meta-Rater, QuadMix, SoftDedup) in related work but doesn't compare against them experimentally. The "PC Average" baseline is somewhat weak and weighted combinations of dimensions or other aggregation strategies aren't tested. 
I think critical missing baselines include: 
- (a) sampling uniformly from each of the 11 original dimensions without PCA.
- (b) simple clustering-based diversity methods.
- (c) domain mixing strategies.

3. The paper doesn't establish that orthogonalization specifically is the key factor vs. simply using multiple diverse metrics. The comparison with "PC Average" (Table 1) shows ODiS performing better, but this doesn't isolate orthogonalization's contribution. 

4. The 11 dimensions appear manually designed based on intuition and prior work, without principled justification for why these specific dimensions or this specific decomposition is optimal. Different domains (code, scientific text, dialogue) might require different dimensions.

5. Training 11-dimensional scorers with GPT API is expensive (460k examples in FineWeb-Edu mentioned for correlation analysis), plus training K RoBERTa models. These are missing: 
- total computational cost vs baselines
- cost-performance tradeoffs

### Questions
1. These analyses are beneficial: 
- (a) sensitivity to dimension definitions 
- (b) whether learned/discovered dimensions would work better
- (c) how to adapt dimensions for different domains 
- (d) whether all 11 dimensions are necessary (some may be redundant even before PCA)

2. Why is PCA specifically the right transformation? Other decorrelation methods aren't considered.

3. While Figure 6b shows 2% overlap after tokenization, this still represents millions of tokens of redundancy at 100B scale. This overlap helps or hurts?

### Soundness
3

### Presentation
3

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
This paper concerns the data selection problem in LLM pre-training. The main contribution is not yet another heuristic criterion for assessing corpus “quality,” but rather an insightful argument that correlations between these criteria hinder effective selection. Inspired by this, this work transforms these scores into an orthogonal space (using PCA) and selecting data with top scores in each principal component dimension and thus improves downstream task performance.

### Strengths
The logic is clear, and most arguments are supported by convincing empirical evidence:

- There are correlations between existing data selection metrics → Figure 5(a)
- PCA can remove these correlations → Figure 6(b)
- Selecting data according to the PC dimension scores (the first four in practice) improves performance → Table 1
- In contrast, selecting according to only one PC dimension compromises performance → Figure 3(a)

### Weaknesses
There are some typos or mistakes in the figures/tables:

- (W1) In Figure 1, the Arc-Easy results are lower than those of the more challenging Arc-Challenge.
- (W2) In Table 1, the results are reported as averages over five domains. However, the average scores appear to be incorrect. This applies to the proposed method (ODiS) as well as to the baselines PC Average-Sample and PC Average-Top, while the averages for DSIR and Random Selection seem correct.

Additionally, (W3) the performance of the RoBERTa-based scorer on the validation set is missing.

### Questions
I don’t have any substantive questions at this time. 

However, due to potential errors in the report results (see weaknesses), I am unable to provide a clear recommendation for now. I will update my score after the necessary corrections, justifications, and explanations are provided.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a limitation of score-based data selection for LLM pretraining. The authors propose ODiS (Orthogonal Diversity-Aware Selection), which evaluates data across quality-related metrics, applies PCA to decorrelate the dimensions, and trains RoBERTa regressors to predict principal-component scores on large corpora. High-scoring data along each PC dimension is selected to preserve both quality and diversity. Experiments show consistent improvements across multiple benchmarks, outperforming baselines. Additional analyses show improved diversity and reduced inter-dimension redundancy.

### Strengths
The proposed ODiS approach is conceptually simple and effective, showing that removing correlation among quality metrics is a useful insight and appropriately implemented via PCA. The experimental results demonstrate consistent gains over widely used baselines. The method seems to be model-agnostic, scalable, and practical. The study highlights a non-monotonic relationship between data quality scores and downstream performance, providing a compelling explanation grounded in diversity.

### Weaknesses
The method relies heavily on GPT-based scoring to obtain 11-dimensional metrics. This is a substantial concern, as it introduces bias and cost concerns. 

The explanation of how thresholds per PC are chosen is underspecified. 

Interpretation of principal components remains unclear, limiting insight into what semantic attributes each dimension captures. 

Experiments are limited to a single language and corpus (Chinese Nemotron-CC) and one model scale (1.5B), raising concerns about generality. More evaluation on larger models or multilingual datasets would strengthen conclusions.

### Questions
LLM Scoring Bias: Since GPT models produce the initial 11-dimensional scores, how robust is ODiS to inaccuracies or systematic biases in these annotations? Have you experimented with weak or noisy scoring sources?

Generalization: Do improvements persist for larger model sizes or multilingual corpora?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new data (pretraining) selection algorithm that (1) chooses a space of scores (in this case, scores by GPT, although the model is not mentioned), (2) performs PCA to identify independent directions, (3) trains a RoBERTa classifier to regress the PCA-transformed scores, (4) select a dataset based on the transformed scores, a joint score threshold and a total data budget.

### Strengths
- The paper is well written 
- The introduction is clear

### Weaknesses
1. I think a fundamental limitation of data selection research is that the scale with which it is conducted is too small. The difference between the best 100B tokens and the average 100B tokens is massive, and thus data selection methods can make a huge difference. But once one is forced to go to 50T-100T tokens, there simply aren’t 50T high quality tokens and 50T low quality tokens to separate. This is the point made by https://openaccess.thecvf.com/content/CVPR2024/html/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.html. Consequently, I am skeptical that this method will make a difference at larger token budgets.

2. Section 2 is supposed to serve as motivation for the proposed algorithm ODiS, but the section is poorly explained with many concerns, and the pointers to learn more are incorrect and duplicated ("Sections 2.3, 3.1, and 3.1"). Without the motivation, the paper loses wind in its sails. As best as I can tell, Figure 1 doesn't show what the text claims it shows, Figure 2a looks like low quality noise and Figure 2b lacks context to be understood.

3. Section 3's demonstration of ODiS's superiority comes across as weak evidence to me. There are no confidence intervals or error bars (over samples in the benchmarks) to identify whether any of these scores are meaningfully different. Additionally, ODiS has so many hyperparameters (e.g., $\tau$, the joint score threshold $t$, how annotations are obtained, how the RoBERTA-based scorers are trained) that there's simply no way for the reader to know whether these scores are genuine improvements or p-hacked.

### Questions
## Title

- The title is a bit generic. It doesn’t communicate ODiS as a name or the key idea of ODiS

## Introduction

- Straightforwardly written - nice!
- nit: Use \citet{} and \citep{} as appropriate. For example, line 54/55 should be \citep{zhuang2025…, liu2025…,bai2025…}. This avoids the double parentheses

## Section 2.1

- line 102: nit: “Sections 2.3, 3.1, and 3.1” repeat 3.1 twice
- Section 2.1: I am confused by the methodology here and what exact claim is being made. The text says to look at 2.3, 3.1 and 3.1 for details, but as best as I can tell, those sections do not provide details on Figure 1
- Section 2.1: The key claim in this section “From Figure 1, we can observe that the data with the highest score performs the worst, while sampling data from a broader range leads to improvement” does not seem to be substantiated in Figure 1 at all. As best as I can tell, Figure 1 doesn’t show scores. Instead, it shows that a larger dataset to search through likely yields better performance across 4/5 evals while the fifth (PIQA) is noise.
- Figure 1: Where are the confidence intervals / standard errors? Each score is averaged over many samples - plot the uncertainty so we can know which (if any) are meaningfully different.
- Figure 1: Add a more descriptive caption. Make it easy for a lazy reader. I shouldn’t have to hunt through the paper to understand what was done, how to interpret these results, what your point is.
- Figure 2a: I have no idea what this figure is meant to communicate. It looks like noise.
- Figure 2b: I again do not know how to interpret this figure. What does “Average-100B” mean in this case? Are we averaging over different subsets of 100B tokens? If so, how is that averaging done? Are you creating multiple histograms and then averaging each bucket height? 
- Line 140: Where is the citation for m3e?
- Line 121-122: “ whereas top-scored data is relatively homogeneous, which explains the performance degradation of the top-scored data.” Where is this shown? I am confused by where scores are being visualized?

## Section 2.3

- The criticism about directly optimizing for downstream tasks is valid, but I’m not sure I understand how creating 11 proxy dimensions is anything more than a less direct proxy.

## Section 3.1

- line 257: “It is a large-scale Chinese dataset” To the best of my knowledge, Nemotron CC is not a Chinese dataset?

## Section 3.2

- Table 1: Where are the confidence intervals over samples in the benchmarks? How can we say whether any of these scores are meaningfully different?

### Soundness
1

### Presentation
2

### Contribution
1
