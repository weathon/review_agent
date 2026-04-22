# FREQMIXATTNET: CONTRASTIVELY SUPERVISED FREQUENCY-MIXING ATTENTION FOR TIME-SERIES FORECASTING

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Time series forecasting has gained significant attention due to its wide applicability in domains such as traffic prediction and weather monitoring. However, it remains a challenging task because of complex temporal patterns, such as multi-scale periodicities and dynamic fluctuations. Existing methods often focus on either time-domain decomposition or frequency-domain analysis, but rarely integrate both effectively.In this paper, we propose FreqMixAttNet, a novel cross-domain forecasting framework that mixes time and frequency representations via a cross-domain attention mechanism. We first introduce an adaptive convolutional wavelet decomposition to model and separate trend and seasonal components more efficiently. The seasonal part is dual-encoded in both time and frequency domains, which are treated as distinct modalities and fused through a cross-transform attention module. Meanwhile, the trend component is captured by a simple multi-scale MLP in the time domain.To further enhance robustness without pretraining, we incorporate a contrastive auxiliary loss. The combination of adaptive convolution, cross-domain mixing attention, and contrastive learning contributes to the superior performance of our method. Extensive experiments on multiple real-world benchmarks show that FreqMixAttNet consistently outperforms prior state-of-the-art methods, demonstrating the effectiveness of our unified cross-domain design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FREQMIXATTNET, a frequency-mixing attention network that combines multi-frequency representation with contrastive learning to improve cross-domain image classification. The framework appears reasonable. However, several key aspects need clarification and stronger experimental evidence.

### Strengths
1. Novel combination of frequency mixing and attention mechanisms for cross-domain feature learning.

2. The use of contrastive loss to enhance domain-invariant representation is well motivated.

3. The proposed model shows consistent improvement on several benchmark datasets.

4. The structure of the paper is mostly clear, and the technical formulation is generally easy to follow.

### Weaknesses
1. Lack of theoretical justification:
The frequency-mixing operation is described empirically, but the paper does not clearly explain why combining frequency bands enhances domain generalization. A mathematical or intuitive explanation of the mechanism would strengthen the contribution.

2. Ablation analysis is insufficient:
There is no clear separation of the contributions from the frequency-mixing module, attention mechanism, and contrastive loss. An ablation study quantifying the improvement of each component is essential.

3. Limited comparison to recent baselines:
The paper mainly compares with classic methods but omits some strong recent works (e.g., transformers-based DG models, diffusion-based domain adaptation). Including these would make the evaluation more convincing.

4. Experimental details missing:
Key training details (e.g., batch size, optimizer, learning rate schedule, number of epochs) are not fully provided, making reproducibility difficult.

5. Visualization and qualitative results:
The paper would benefit from t-SNE plots or attention heatmaps to demonstrate that the model learns domain-invariant features.

6. Contrastive loss formulation:
The contrastive learning section lacks details about positive/negative pair sampling, temperature parameter, and how it interacts with cross-domain samples.

7. English writing and presentation:
Several grammatical and formatting issues exist (e.g., inconsistent figure captions, equation numbering). The introduction and conclusion sections could be refined for clarity and conciseness.

### Questions
1. Provide a deeper theoretical or intuitive analysis of frequency mixing and its link to domain invariance.

2. Add a comprehensive ablation study (baseline vs. +FreqMix, +Attention, +CL).

3. Include more recent comparison methods and report statistical significance.

4. Provide visual evidence of learned representations (feature maps, t-SNE, Grad-CAM).

5. Refine language and structure — a professional proofreading pass is recommended.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a forecasting framework that fuses time-domain and frequency-domain representations via cross-attention, with an adaptive convolutional wavelet decomposition and contrastive auxiliary loss. Experiments across several standard long-horizon benchmarks and comprehensive ablations show the effectiveness of the proposed method.

### Strengths
1. The framework design is clear and well validated by extensive ablation studies, such as removing adaptive conv, removing wavelet decomposition, disabling cross-domain attention, dropping the contrastive loss, etc. It's also new to use wavelet decomposition for trend and seasonality decomposition. 

2. Extensive experiments validate the effectiveness of the proposed approach. The paper also provides detailed results on sensitivity analysis which improves the transparency.

### Weaknesses
1. The novelty is limited as there are many existing works on combining time domain and frequency domain analysis with decomposition [1]. The paper should more precisely explain what is fundamentally new about the proposed method compared to existing time and frequency domain approaches and why/how such design helps.

2. It is already known that different baselines perform the best under different lookback windows [2], so it is a bit unfair to compare with a unified lookback window for all baselines.

3. The writings look repetitive. For example, Table 2 vs Table 6, Table 3 vs Table 7, Table 4 vs Table 8/9/10 are very redundant. The paper would benefit from merging redundant tables and making the writings more concise.

[1] First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting

[2] Scaling Law for Time Series Forecasting

### Questions
1. What is the definition of forecastability in Table 1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors aim to improve time-series forecasting by jointly modeling temporal and frequency representations. To achieve this goal, they propose FreqMixAttNet, a unified cross-domain framework that integrates information from both the time and frequency domains. Specifically, the model consists of three key components: (1) a Patch-Contextual Adaptive Convolution module for context-aware feature extraction, (2) a Cross-Domain Mixing Attention mechanism to enable interaction between time- and frequency-domain features, and (3) a Contrastive Auxiliary Learning strategy to enhance robustness and generalization. In the experiments, the authors evaluate the proposed method on six benchmark datasets and compare it with several state-of-the-art baselines, demonstrating consistent improvements in forecasting accuracy.

### Strengths
1. The authors proposed a cross-domain module to integrate time and frequency domain representations.
2. The organization and writing are easy to follow.

### Weaknesses
1. The paper makes an abrupt transition from discussing how previous methods separately handle time and frequency domains to emphasizing robustness improvement. The connection between these two aspects is unclear, and this logical discontinuity weakens the overall coherence and persuasiveness of the paper’s argumentation.
2. The authors do not provide a clear explanation of why the proposed cross-domain attention mechanism can improve forecasting accuracy and lacks interpretability analysis to support this claim. It is also worth questioning whether the interaction between the two modalities could introduce redundant or interfering information that might negatively affect prediction performance.
3. The paper lacks an analysis of computational complexity, including comparisons of training time, inference efficiency, and parameter counts with baseline models.
4. In the paper, there are several hyperparameters are introduced, but only the impact of $$\beta_1$$is analyzed. The paper does not examine how different loss weights influence the results, nor does it clarify how the weights of multiple loss terms are designed. And another question is whether they sum to a fixed value or are tuned independently. A more  discussion of these aspects is needed.
5. There are some small typos and inconsistencies. For example, there are missing spaces between some sentences.  Besides,“hyperparamter” should be “hyperparameter”, and “analyst” should be “analyze”, among similar minor spelling errors.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
