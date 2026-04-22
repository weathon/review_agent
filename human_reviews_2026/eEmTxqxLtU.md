# Flames: Multi-Scale Mamba with Adaptive Fourier Filters and Laplace Transform for Time Series Forecasting

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Time series data usually exhibit intricate characteristics such as non-stationarity, noise, multi-scale periodicity, and transient dynamics, posing significant challenges to long-term time series forecasting (LTSF). While transformer-based models effectively capture long-range dependencies, their practical applications are hindered by high computational cost with quadratic complexity, noise sensitivity, and overfitting on small datasets. Moreover, time series present distinct patterns at different temporal resolutions, containing both fine-grained (micro) and coarse-grained (macro) information. To address these issues, we propose a novel framework, Flames (multi-scale Fourier Filter Mamba with Laplace), designed for efficient and robust LTSF. Specifically: (i) We introduce an adaptive Fourier filter with a selection module embedded into Mamba. At each scale, the neural operator uses Fourier analysis to refine feature representations, applies learnable thresholds for noise reduction, and captures inter-frequency interactions via global-local semantic filters through element multiplication. (ii) We incorporate the Laplace transform to capture transient dynamics. Extensive experiments on multiple benchmarks demonstrate that Flames consistently outperforms SOTA methods, achieving superior accuracy–efficiency trade-offs. Results highlight its strong robustness and scalability, particularly in noisy or transient settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel multi-scale Mamba framework that incorporates adaptive Fourier filters to cover all frequencies at each scale and a Laplace-transform module to capture short-term fluctuations.

### Strengths
1. This paper proposes an Adaptive Fourier Filter Module (AFFM) encoder, which employs adaptive Fourier filtering to capture multi-scale periodic patterns and to reduce noise.
2. This paper applies the inverse Laplace transform to extract short-term dynamics.

### Weaknesses
1. The paper is poorly organized and lacks clarity in writing, making it difficult to follow. In addition, multiple inconsistencies in symbol usage should be carefully reviewed, for example, the one observed on Line 232.  
2. The motivation of the paper is not clear. In the Introduction, the authors argue that using Mamba as the backbone faces three key challenges: (i) multiscale periodicity, (ii) data noise, and (iii) transient dynamics. However, no strong prior studies or preliminary experiments are provided to substantiate these claims. Other important assertions, such as “Linear struggles with noisy data and fails to capture long-term dependencies effectively” are likewise unsupported. In addition, the manuscript does not offer a substantive discussion of whether alternative backbones (e.g., Transformer, MLP) can address these three challenges, nor does it provide comparative analysis to justify the choice of Mamba.  
3. It would strengthen the work to include additional frequency filter-based baselines, such as FilterNet [1], TSLANet [2], which are closely related to the proposed approach.  

[1] FilterNet: Harnessing Frequency Filters for Time Series Forecasting. NeurIPS, 2024.  
[2] TSLANet: Rethinking Transformers for Time Series Representation Learning. ICML, 2024.

### Questions
pls refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposed a framework for long-term time series forecasting. The model enhances the Mamba state space model by integrating three components. Experimental results on eight benchmark datasets show that Flames achieves good performance, outperforming Transformer- and MLP-based baselines with better noise robustness, scalability, and computational efficiency.

### Strengths
- Combines Fourier and Laplace transforms with state space modeling for a unique approach to capturing multi-scale and transient temporal patterns.
- Demonstrates consistent improvement over leading baselines across multiple datasets and noise levels, with extensive ablation and robustness studies.
- Achieves linear computational complexity, lower parameter counts, and faster inference while maintaining high accuracy.

### Weaknesses
- The combination of multi-scale processing, Mamba, Adaptive Fourier Filters, and the Inverse Laplace Transform makes the overall model architecture intricate and potentially complex to implement and fine-tune compared to simpler linear or pure Mamba models The formulation, particularly for the Laplace transform integration, is dense.
- While claims about interpretability are made, there is little empirical validation of how the model captures transient dynamics or multi-scale features in practice. 
- The paper notes that the optimal number of scales varies with the prediction length, suggesting that a fixed choice (like M=3 for efficiency) may not be universally optimal and would require a trade-off analysis for new datasets or forecasting horizons.

### Questions
- How to simplify or modularize the integration of multi-scale processing, Adaptive Fourier Filters, and the Inverse Laplace Transform to make the model easier to implement and fine-tune?
- What experimental approaches or visual analyses could be added to empirically validate how the model captures transient dynamics and multi-scale temporal features?
- How can the model adaptively determine or learn the optimal number of scales for different datasets or forecasting horizons wthout manual tuning?

### Soundness
2

### Presentation
3

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
The paper proposes FLAMES, a long-term time-series forecasting model. The FLAMES augments Mamba with three components. (i) multi-scale feature extraction, (ii) Adaptive Fourier Filter Mamba that performs FFT-domain masking and global/local learnable frequency mixing to denoise and enhance periodic structure, and (iii) Laplace transform to capture transient dynamics. Experiments on multiple benchmarks report consistent gains over baselines. The paper also conducts ablations for each component, robustness to synthetic noise, and a scalability analysis to show the superiority of FLAMES.

### Strengths
+ The time series forecasting, particularly long-term time series forecasting is an important and practical problem. 

+ The evaluation is comprehensive. The experimental results comprehensively cover multiple datasets and horizons with ablations, look-back sensitivity, robustness to noise, and scalability analysis.

### Weaknesses
- The method is a careful combination of known tools (such as Mamba, Fourier filter, and Laplace transform) and rather than a fundamentally new learning principle. The paper would benefit from a more formal analysis combing these modules together.

- The paper does not compare many Mamba-based time series forecaster despite mainly built on Mamba backbone. The only relevant baseline is DTMamba. It is not clear why the authors ignore many other Mamba-based time series forecasting papers, even if they are earlier or have been accepted by decent conferences. Just list some examples as below:

[1] Ahamed et al. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. ECAI. 2024

[2] Xu et al. SST: Multi-Scale Hybrid Mamba-Transformer Experts for Long-Short Range Time Series Forecasting. CIKM. 2025

[3] Patro et al. SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series. arxiv. 2024.

[4] Hu et al. Time-SSM: Simplifying and Unifying State Space Models for Time Series Forecasting. arxiv, 2024.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
