# Xihe: Scalable Zero-shot Time Series Learner via Hierarchical Interleaved Block Attention

- Decision: Reject
- Scores: 6, 2, 2

## Abstract
The rapid advancement of time series foundation models (TSFMs) has been propelled by migrating architectures from language. While existing TSFMs demonstrate impressive performance, their direct adoption of cross-domain architectures constrains effective capture multiscale temporal dependencies inherent to time series data. This limitation becomes particularly pronounced during zero-shot transfer across datasets with divergent underlying patterns and sampling strategies.To address these challenges, we propose Hierarchical Interleaved Block Attention (HIBA) which employs hierarchical inter- and intra-block sparse attention to effectively capture multi-scale dependencies. Intra-block attention facilitates local information exchange, and inter-block attention operates across blocks to capture global temporal pattern interaction and dynamic evolution. Leveraging the HIBA architecture, we introduce Xihe, a scalable TSFM family spanning from an ultra-efficient 9.5M parameter configuration to high-capacity 1.5B variant. Evaluated on the comprehensive GIFT-Eval benchmark, our most compact Xihe-tiny model (9.5M) surpasses the majority of contemporary TSFMs, demonstrating remarkable paramater efficiency. More impressively, Xihe-max (1.5B) establishes new sate-of-the-art zero-shot performance, surpassing previous best results by a substantial margin. This consistent performance excellence across the entire parameter spectrum provides compelling evidence for the exceptional generalization capabilities and architectural superiority of HIBA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Xihe, a new family of Time Series Foundation Models  designed for zero-shot forecasting. The central innovation is the Hierarchical Interleaved Block Attention  architecture. The authors argue that existing TSFMs, often adapted from NLP, fail to adequately capture the multi-scale temporal dependencies inherent in time series data. HIBA addresses this by partitioning sequences into hierarchical blocks and alternating between intra-block attention and inter-block attention. The authors present a scalable family of models  trained on a large corpus of 325B time points. Experimental results on the GIFT-Eval benchmark show that Xihe models achieve state-of-the-art zero-shot performance, with the largest model  setting a new SOTA and the smallest outperforming most contemporary TSFMs, demonstrating significant parameter efficiency.

### Strengths
The Xihe family demonstrates state-of-the-art zero-shot performance on the comprehensive GIFT-Eval benchmark. The fact that the smallest 9.5M parameter model  outperforms the majority of existing TSFMs is a very strong result, highlighting the architectural efficiency of HIBA.

The paper successfully trains and evaluates a family of models ranging from 9.5M to 1.5B parameters. The results show a clear scaling trend where performance on both CRPS and MASE metrics improves monotonically with model size, confirming the architecture's scalability.

### Weaknesses
W1: The pre-training dataset is a mix of public datasets and synthetic data. While large, the contribution of the data-quality-aware mixing strategy  versus the architectural improvements is not explicitly disentangled. It's unclear how much of the performance gain comes from this curated data mix.

W2: The implementation details of Hierarchical Block Size is ambiguous. Though authors provide information about the block size in appendix, it is not clear how to configure the block size for intra-block and inter-block attention.

W3: The design of Non-causal Intra-block Attention is not well-motivated. Since the ablation study shows that the causal intra-block attention is only slightly worse than the non-causal one, and the casual attention naturally fits the time series data and is more efficient for inference.

W4: There is clear typo in equation 5, the definition of b, m, B, M is ambiguous.

### Questions
listed in weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a time series foundation model (TSFM) architecture, that is claimed to establish new state-of-the-art in the popular GIFTEval benchmark. Specifically, it proposes Hierarchical Interleaved Block Attention (HIBA) in the Transformer architecture, which employs intra- and inter-block attention to capture multi-scale dependencies more effectively. The paper attempts to improve the state-of-the-art in ZeroShot (ZS) TS forecasting, which is an important problem in the time series domain.

### Strengths
The paper has the following strengths:
1. The presentation is clear, and understandable.
2. The experimental evaluation is performed on a well-established leaderboard (GIFT) which comprises of a significant amount of univariate time series evaluation data.
3. Ablations studies are reported.

### Weaknesses
However, the paper has the following weaknesses:
1. The concept of intra- and inter-block attention is not new. In the TSFM literature, similar concepts were proposed in the TSMixer[1] and TTM[2] papers, where they employed mixing instead of attention. How are intra- and inter-block attention conceptually different from intra- and inter-patch mixing?
2. The concept of varying the block length is not novel as well. The TTM paper proposed something which authors called as "Adaptive Patching". How is that different from the proposed variable block length?
3. Can the proposed model handle multivariate data, and model inter-channel correlations?
4. Can the authors benchmark the model on multivariate benchmarks such as the FEV leaderboard?
5. The authors claim to have established a SOTA score in GIFT, however, the leaderboard has much better SOTA numbers than the reported ones. Hence, this claim should be modified.
6. What the values of the multi-horizons? Only 96 and 768? Why not more?
7. The author(s) have claimed several times that " (TSFMs) has been propelled bymigrating architectures from language models". This statement is not entirely true. While there are some models which are adopted from language, there are many more novel models with significant modifications to handle multivariate time series data.
8. What are some of the weaknesses of the proposed model? When does it fail?
9. Were the entire 300+ Billion time points used in the pre-training? How did the authors mitigate extreme bias in the LOTSA dataset? Some of the LOTSA datasets have 15B time points, and some have 1000 time points.
10. "subsets of the training datasets from Chronos (Ansari et al., 2024a)" -- which subset?
11. "the LOTSA datasets from Moirai" -- doesn't it have GIFTEval data in it?
12. "synthetic time series generated using a procedure inspired by KernelSynth" -- what do the authors mean by inspired by KernelSynth? Is it a modified KernelSynth? The details are not provided, e.g., How many kernels? What kernels?

### Questions
See weakness.

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
This paper introduces Xihe, a family of Time Series Foundation Models (TSFMs), for the task of zero-shot forecasting. Their motivation is that existing TSFMs are often adapted directly from NLP/CV and therefore struggle to effectively capture the inherent multi-scale temporal dependencies, which are important for time series. To address this, the paper proposes the Hierarchical Interleaved Block Attention (HIBA) mechanism. HIBA hierarchically partitions the input sequence into blocks of varying granularity across different layers. It alternates between two types of attention: 1) intra-block attention, which models local dependencies within each block, and 2) inter-block attention, which models global dependencies across all patches. The resulting models ranges from 9.5M to 1.5B parameters, which are pretrained on a mixture of public data and synthetic data from existing TSFMs. The results on the GIFT-Eval benchmark show an outstanding performance. The paper also demonstrates that the proposed family of models follows clear scaling laws.

### Strengths
1. The paper presents the results on the GIFT-Eval benchmark, instead of only using the seven small LTF datasets.
2. The writing is very good and easy to follow. However, the authors tend to use ";" too often. For example, “strong zero-shot capabilities; Although Moirai”, you should replace “;” with “.”. Some other grammar errors:
- Some baseline methods are misspelled: “Dlinear” → “DLinear”, “PatchTsT” → “PatchTST”
- Line 84: "combining public available datasets" -> "combining publicly available datasets"
- Line 96: "while remaining efficiency suitable" -> "while remaining efficient and suitable"
- LIne 182: "The detailed design... are presented" -> "The detailed design... is presented".

### Weaknesses
1. Not enough ablations on the hierarchical structure.
2. Not enough ablations on the K prediction heads, and this part does not seem very novel/scalable.
3. Lack of pretraining details, such as dataset mixing strategies.

I feel like the paper in the current form is not ready for acceptance. My main concern is the current ablation studies are not comprehensive enough to highlight the main contribution of this paper, which is to use a hierarchical structure of interleaved intra-block and inter-block attention. However, I am willing to raise my score if you can address my concerns.

### Questions
1. How do the different block sizes and number of layers work? For example, when B=(3,7,21) and there are 24 layers, does that mean in each layer, I sequentially process the input with 3, then 7, and finally 21? 
2. Which prediction heads do you use? You only mentioned a total of K prediction heads. From what I understand, if the task is to predict 96 steps, then you only use the predictions from the head that predicts 96 steps, and ignore the predictions from all the other heads. In this case, how well does the model scale to a different forecast horizon not supported by any of the heads? Can you show some sensitivity analysis on whether having too many prediction heads will negatively affect the model performance?
3. Can you add some sensitivity analysis on the block sizes? It is always (3, 7, 21). Have you tried more than three block sizes and also different numbers? Do these numbers need to be in an increasing order? In the ablation study, does “w/o Hierarchy (B=3)” mean that you only use a single block of block size 3? Have you tried the same block size (e.g., (3, 3, 3) or (7, 7, 7))? For the “standard attn” ablation, do you use the same number of layers or twice the number of layers (since each layer has both intra-block attention and inter-block attention)?
4. In Table 1 left, the magnitude of differences seems to be quite small. Can you run the same model with different seeds and report the confidence intervals? Is the observation consistent across different model scales?
5. Why do tiny, lite and flash all have 24 layers? The number of layers seems to be noticeably higher than other transformer-based foundation models.
6. Line 280, “we adopt a data-quality–aware mixing strategy instead of the uniform mixing commonly used in prior TSFMs”. How exactly does this work?
7. In Line 264, why is there information leakage? If you apply left padding for the context, the context will be divisible by the patch size and block size. Why would there still be information leakage from the forecast horizon? My understanding is that the acausal signals learned during the non-causal intra-block attention will impact the subsequent causal inter-block attention, so there will be some information leakage during training, but not testing.
8. Have you tried using a student-t distribution (https://arxiv.org/pdf/2410.12360, https://arxiv.org/pdf/2402.02592) instead of quantile loss as the loss function?

### Soundness
2

### Presentation
4

### Contribution
2
