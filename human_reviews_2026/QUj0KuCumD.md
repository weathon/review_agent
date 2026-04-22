# MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with $0.1K$ Parameters

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Recently, there has been a growing interest in Long-term Time Series Forecasting (LTSF), which involves predicting long-term future values by analyzing a large amount of historical time-series data to identify patterns and trends. Significant challenges exist in LTSF due to its complex temporal dependencies and high computational demands. Although Transformer-based models offer high forecasting accuracy, they are often too compute-intensive to be deployed on devices with hardware constraints. 
In this paper, we propose MixLinear, which synergistically combines orthogonal segment-based trend extraction in the time domain with adaptive low-rank spectral filtering in the frequency domain. Our approach exploits the complementary structural sparsity of time series: local temporal patterns are efficiently captured through mathematically linear transformations that separate intra-segment and inter-segment correlations, while global trends are compressed into an ultra-low-dimensional frequency latent space through learnable rank-constrained filters. By reducing the parameter scale of a downsampled $n$-length input/output one-layer linear model from $O(n^2)$ to $O(n)$, MixLinear achieves efficient computation without sacrificing accuracy.
Extensive evaluations show that MixLinear achieves forecasting performance comparable to, or surpasses, state-of-the-art models with significantly fewer parameters ($0.1K$), which makes it well suited for deployment on devices with limited computational capacity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MixLinear, a novel model for time series forecasting. The central contribution is a dual-domain architecture that processes time series in parallel: (1) a segment-based, factorized linear pathway in the time domain to capture local patterns, and (2) an adaptive low-rank spectral filtering pathway in the frequency domain to model global trends. The authors claim that this approach achieves SOTA-comparable forecasting accuracy while using an extremely small parameter budget of only 0.1K. The model is reported to be significantly more efficient in terms of parameters and inference speed compared to baselines.

### Strengths
1.Efficiency: the ultra-lightweight (0.1K params) model that matches SOTA performance and the reported gains in inference speed are impressive.

2.Design: The core design principle—exploiting complementary structural sparsity by separating local time-domain patterns (via factorized linear ops) from global frequency-domain patterns (via low-rank filtering)—is intuitive and elegant.

### Weaknesses
1.Marginal Accuracy Gain: The paper's contribution to forecasting accuracy is marginal, it places the entire burden of the paper's contribution on the efficiency claim.

2.Ambiguous Core Mechanisms: The paper lack of important details. The "learnable upsampling" needs to be explained.How is it implemented?

### Questions
1.Downsampling Factor: Could the authors provide a sensitivity analysis for the downsampling factor 
? How does the model's performance change with different values of 
 (e.g., 2, 4, 8, 16)?

2.The ablation study suggests the time-domain path is better for low-dim data and the freq-domain path is better for high-dim data . Does this not undermine the universality of the core assumption? How does the model perform on data that is known to be highly non-linear or non-stationary, which might not fit either the factorized-linear or low-rank-spectral assumptions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MixLinear, a novel and extremely lightweight model for long-term time series forecasting (LTSF). The core idea is to tackle the "parameter explosion" problem in current deep learning models by adopting a dual-domain architecture that processes time series patterns in their most natural domains. Specifically, MixLinear uses a segment-based pathway with factorized linear transformations in the time domain to capture local temporal patterns, and a frequency domain pathway that uses an adaptive low-rank spectral filter to compress and model global trends. The authors claim this hybrid approach reduces the parameter scale of a linear model from O(n^2) to O(n). The main contribution is a model that achieves forecasting performance competitive with or even surpassing some state-of-the-art methods, while using only ~0.1K parameters, making it exceptionally well-suited for resource-constrained environments.

### Strengths
1) The most prominent strength is the model's minuscule size (~0.1K parameters) and low computational cost (MACs). An 81% parameter reduction compared to the next-lightest model (SparseTSF) is a remarkable achievement. This has profound practical significance for deployment on resource-constrained hardware, which is a major bottleneck for many modern deep learning models.

2) The dual-domain design is well-motivated by the "Spectral-Temporal Decomposition Principle." The idea of processing local patterns in the time domain and global patterns in the frequency domain is intuitive and elegant. The ablation study provides strong empirical evidence that this separation of concerns is effective and that the two pathways are complementary.

3) The paper is backed by a thorough experimental evaluation on eight benchmark datasets. The authors compare against a strong and diverse set of baselines. The inclusion of detailed ablation and hyperparameter sensitivity studies adds significant credibility to the design choices and validates the core hypotheses, especially the finding that a spectral rank as low as 2 is often sufficient.

### Weaknesses
1) The paper's primary weakness is that the individual components of the architecture are not conceptually new. Time series segmentation (PatchTST), linear forecasting models (DLinear), and frequency-domain analysis (FEDformer, FITS) are all established techniques. The novelty lies in the specific integration and extreme simplification of these ideas. While the final result is impressive, the contribution is more of an engineering and simplification achievement than a fundamental theoretical breakthrough. To strengthen the paper, the authors could be more explicit about this, framing the contribution as a novel synthesis that achieves an unprecedented operating point on the efficiency-accuracy curve.

2) The abstract and main results sometimes use strong phrasing like "surpasses, state-of-the-art models." While MixLinear does outperform SOTA on some dataset/horizon combinations (e.g., Exchange), Table 1 and Table 5 show that it is more accurately described as being highly "competitive" or "comparable." For instance, on several datasets, FITS or even DLinear achieve slightly better or similar MSE. A more nuanced description of the performance would strengthen the paper's credibility, for example by emphasizing that it achieves this competitiveness with orders of magnitude fewer parameters.

3) The explanation of how the factorized linear projections in Section 2.3 disentangle "intra-segment" and "inter-segment" correlations is slightly imprecise. The mathematical formulation in the appendix ($W^T_2$ ($W_1 X_seg)^T$) suggests a token-mixing and channel-mixing mechanism on a reshaped tensor, akin to MLP-Mixer, rather than a hierarchical processing of segment embeddings. It's unclear how one linear layer acts on "intra-segment" features and the other on "inter-segment" features when they are applied in this manner. Clarifying this mechanism and its connection to the intuition would improve the paper.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
To bridge the gap in the design of scalable and lightweight forecasting models in the frequency domain, this work present MixLinear, a dual-domain framework that achieves competitive long-term time series forecasting performance with only 0.1K parameters. The extreme parameter reduction enables deployment on resource-constrained devices, opening new possibilities for real-time forecasting applications in IoT environments and edge computing.

### Strengths
1. The research direction of this work is very interesting and has practical significance.

2. This work is a highly innovative work. A large amount of evidence shows that it is different from the existing works.

3. The experiments in this work are thorough.

### Weaknesses
The research presented in this manuscript is highly intriguing, and I have personally found it to be quite rewarding. However, as I am not an expert in this particular domain, I would like to raise two minor questions:

1. While I appreciate the overall design of this work, the main text reads more like a technical report than a research paper, it lacks analysis or critical discussion to guide the reader toward a deeper understanding.

2. Can more interpretable evidence be derived from lightweight research to bolster its applicability across diverse domains?

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
MixLinear proposes an extreme low-parameter (0.1K) dual-domain framework for multivariate time series forecasting. It processes local trends via segment-based linear decomposition in the time domain (O(n) complexity) and global patterns via adaptive low-rank spectral filtering in the frequency domain. Experiments show competitive accuracy with 3.2× inference speedup and 16.2% error reduction versus baselines while enabling deployment on resource-constrained devices.

### Strengths
- Achieves SOTA-comparable results with only 0.1K parameters.
- Linear memory complexity (O(n)) enables edge/IoT deployment.

### Weaknesses
- Only MACs are compared, authors should provide the comparison of number of parameters.
- While authors claim that the number of parameter is reduced down to $0.1k$, the number of MACs of the proposed MixLinear is with similar scale compared to SparseTSF and FITS.
- The core idea of this paper is inherited from FITS. Good but not that impressive as FITS and SparseTSF.

### Questions
- Why does the frequency pathway use per-segment FFT instead of global FFT given the focus on global trends?

### Soundness
3

### Presentation
3

### Contribution
2
