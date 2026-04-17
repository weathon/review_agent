# Local Geometry Attention for Time Series Forecasting under Realistic Corruptions

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Transformers have demonstrated strong performance in time series forecasting, yet they often fail to capture the intrinsic structure of temporal data, making them susceptible to real-world noise and anomalies. Unlike in vision or language, the local geometry of temporal patterns is a critical feature in time series forecasting, but it is frequently disrupted by corruptions.
In this work, we address this gap with two key contributions. First, we propose Local Geometry Attention (LGA), a novel attention mechanism theoretically grounded in local Gaussian process theory. LGA adapts to the intrinsic data geometry by learning query-specific distance metrics, enabling it to model complex temporal dependencies and enhance resilience to noise. Second, we introduce TSRBench, the first comprehensive benchmark for evaluating forecasting robustness under realistic, statistically-grounded corruptions.
Experiments on TSRBench show that LGA significantly reduces performance degradation, consistently outperforming both Transformer and linear model. These results establish a foundation for developing robust time series models that can be deployed in real-world applications where data quality is not guaranteed. Our code is available at: https://github.com/dongbeank/LGA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new attention mechanism for time series forecasting, which is termed Local Geometry Attention. The designed LGA is inspired by local Gaussian process theory and aims to learning query-specific distance metrics and enable the model to learn complex temporal dependencies and enhance resilience to noise. Also, TSRBench, a comprehensive benchmark for evaluating time series forecasting robustness, is proposed.

### Strengths
1. The motivation seems clear with intuitive figures (Fig.1 and 2)

2. Clearly written and well-presented with source codes.

### Weaknesses
1. The technical details could be improved in a clearer way.

2. According to the performance reported in Table 2, the improvements led by the proposed method seem to be marginal.

3. the motivation of TSRBench is not clear. And more comprehensive datasets and baselines should be involved.

### Questions
Please refer to the weakness.

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
4

### Summary
The contribution of this paper is twofold; it presents a method called local geometry attention designed (LGA) to model local structures of time series data in transformer neural networks, and it introduces TSRBench a corruption suite for evaluating robustness of time series models. In the empirical evaluation part of the submission, the LGA models are evaluated using TSRBench.

More specifically, LGA uses local Gaussian process regression to model the data/representation manifold near the query points of a an attention module. Since, computing the local geometry aware score of the model is computationally prohibitive, a second neural network is trained that predicts the metric tensor.  
The TSRBench consists of a method to insert shifts and spikes into time series at 5 different severity levels.
In the empirical evaluation, LGA is inserted into different transformer models (with PatchTST being the default choice) to assess robustness of long-term time series forecasts.

### Strengths
- The LGA technique is universal in  the sense that it can easily be integrated into different transformer architectures and likely also in other networks that use attention modules.
- On the six corrupted time series datasets that were used, LGA (+PatchTST) consistently achieves the best forecasting accuracy especially at large severity levels, indicating that LGA adds robustness wrt shift and spike corruptions.
- The benchmarking tool comes with predefined and carefully calibrated severity levels which simplify its use.
- The paper is very well written, and therefore easy to read and understand. (But certain parts need more explanation, see questions)
- The paper is accompanied by an extensive supplementary material which provides the mathematical background, implementation details and further experiments.

### Weaknesses
- The LGA approach is only evaluated with respect to the (synthetic) TSRBench corruptions, i.e. shifts and spikes. It is not evaluated on other types of corruptions, for example on the anomaly types that are used in (Cheng et al, RobustTSF: ..., ICLR2024) or the synthesized outliers from (Lai et al., Revisiting Time Series Outlier Detection: Definitions and Benchmarks, NeurIPS 2021 Dataset and Benchmark track). One could therefore think that the benchmarks are designed to favor LGA.
- The LGA approach is only evaluated on forecasting tasks. 
- It seems that the severity levels of TSRBench need to be calibrated for every dataset individually, and the guidelines are quite vague, i.e. "we ensured that the performance differences between severity levels were neither too large nor too small, and that the degree of noise strictly increased with each level".   
Moreover, the resulting hyperparameter choices are model dependent and require training one (or multiple) neural networks per parameter choice. This hinders extending the benchmark to new datasets in a standardized manner, one of the key goals of this work ("Despite the need, a standardized benchmark for such realistic corruptions in time series remains a significant gap")
- The paper makes several implementation choices whose effects are not ablated empirically. This includes learning the metric tensor with an additional neural network (instead of computing it for every query) and using only a diagonal metric. Moreover, while it is stated that both are done because of computational costs, no runtime or complexity analysis of them are presented. 
- For the datasets that are used, the manuscript refers to the Autoformer paper (Wu et al., NeurIPS 2021), but the actual originators of the datasets are not credited.

### Questions
- Is the robustness of LGA by design, or did it surprise you? If by design, what is the reason or intuition?
- Would you explain in more detail how the network that predicts the metric tensor is trained. Is this done simultaneously to training the transformer, afterwards, or  alternately? Considering the former, could you explain again, why training such a network is more efficient than computing the metric, or if it only shift compute from inference to training?
- Would you explain the connection to Riemannian geometry and what the point of the detour to the geodesic distance was. Is the connection only that the score is determined by a bilinear form that depends on the query point?
- Would you compare TSRBench with this very recent robustness benchmark ( [Janßen et al., arxiv, Oct 2025](https://arxiv.org/pdf/2510.04900) )

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the robustness of time series forecasting transformer models under realistic corruptions. It first propose a novel attention mecanism derived from local Gaussian process theory called Local Geometry Attention (LGA). It takes into account the geometry of data to be more robust to corruptions in the time series data. It can be implemented by learning a neural network over the query vectors seen during training. Then, the paper introduces TSRBench, a benchmark with realistic corruptions (spikes and level shift) to estimate the robustness of models for forecasting. Experiments over 6 common forecasting benchmarks (with their corrupted versions) and 3 baselines show the performance benefits of the LGA attention incorporated into a PatchTST model.

### Strengths
- The paper is well-written and the proposed approach well motivated
- The proposed benchmark is very interesting with sounded corruptions (inspired from ImageNet-C benchmark)
- The experiments are convincing to show the performance improvement brought by LGA

### Weaknesses
- While the proposed method is sound, it requires training neural networks to compute the Local Geometry Attention. The current submission is missing a computational cost comparison when compared to self-attention
- Since the main contribution is a novel attention mecanism, it would be interesting to see the comparison to other types of attention in addition to the traditional one (e.g., channel-wise attention [1, 2])
- Connected to the previous weakness, it would be interesting to compare to more models such as iTransformer [1] or SAMformer [2] that reported robust performance with temporal/spatial and spatial attention respectively.
- While the benchmarks is interesting, I believe the current proposal would be strengthen with additional models and datasets to clearly show the failure of other models under realistic corruptions
- It would be interesting to see how LGA behaves when integrated into other models than PatchTST, for instance could it be applied on SAMformer with channel-wise attention or i-transformer with channel and temporal wise attention?

Overall, I find the submission interesting with a well-motivated LGA and robustness benchmark. However, it would be strengthen with additional methods and computational cost comparison.

*References*

[1] Liu et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR 2024

[2] Ilbert et al. SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel Wise Attention. ICML 2024

### Questions
- The LGA is computed by training a neural network on query vectors seen during training. Does it mean that there are two interconnected training loops: one for the model, one for the LGA module? If it is the case, what is the additional computational cost and does it vary with the number of training steps? Could the authors clarify that please? 
- Were the experiments run over several seeds? If yes, could the authors make the standard deviation appear? If no, could the authors please conduct the experiments with 2 additional seeds to have an idea of the significance of the performance improvement?

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
2

### Summary
The manuscript introduces Local Geometry Attention, an attention mechanism for time-series forecasting that replaces dot-product similarity with a query-specific metric estimated from local Gaussian-process theory, aiming to make attention scores geometry-aware and less sensitive to noisy or anomalous keys. It also proposes TSRBench, a robustness benchmark that injects spike and level-shift corruptions using statistically grounded processes and calibrated severities.

### Strengths
LGA is designed to improve robustness of attention to local anomalies and noise.

TSRBench is a potentially useful benchmarking tool for researchers to evaluate robustness under controlled corruptions.

### Weaknesses
The description of LGA is not detailed enough. A step-by-step derivation of the attention score and kernel construction is needed to enhance the clarity. Please also provide the tensor shapes at each stage.

The GP motivation is under-explained: it is unclear how local GP assumptions translate into better attention weights or how the geometric structure is actually learned from data.

The set of baselines is limited, making it hard to judge whether robustness gains hold against recent strong time-series Transformers.

### Questions
(a) Please expand the discussion of non-uniform, locally clustered time-series observations, since this is the core motivation for introducing LGA.

(b) Explain in detail how queries and keys are embedded in the 2D panel in Figure 2 and how this visualization demonstrates a more effective attention mechanism.

(c) Show how the proposed local kernel–covariance compares to vanilla dot-product attention under the same setup, and quantify robustness gains.

(d) In Figure 5, replace training time with FLOPs to make comparisons hardware-agnostic.

(e) Add more recent baselines in Table 2 to better position LGA and TSRBench.

### Soundness
3

### Presentation
2

### Contribution
2
