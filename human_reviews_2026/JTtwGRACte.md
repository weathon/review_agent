# Characteristic Root Analysis and Regularization for Linear Time Series Forecasting

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
Time series forecasting remains a critical challenge across numerous domains, yet the effectiveness of complex models often varies unpredictably across datasets. Recent studies highlight the surprising competitiveness of simple linear models,  suggesting that their robustness and interpretability warrant deeper theoretical investigation. This paper presents a systematic study of linear models for time series forecasting, with a focus on the role of characteristic roots in temporal dynamics. We begin by analyzing the noise-free setting, where we show that characteristic roots govern long-term behavior and explain how design choices such as instance normalization and channel independence affect model capabilities. We then extend our analysis to the noisy regime, revealing that models tend to produce spurious roots. This leads to the identification of a key data-scaling property: mitigating the influence of noise requires disproportionately large training data, highlighting the need for structural regularization. To address these challenges, we propose two complementary strategies for robust root restructuring. The first uses rank reduction techniques, including Reduced-Rank Regression (RRR) and Direct Weight Rank Reduction (DWRR), to recover the low-dimensional latent dynamics. The second, a novel adaptive method called Root Purge, encourages the model to learn a noise-suppressing null space during training. Extensive experiments on standard benchmarks demonstrate the effectiveness of both approaches, validating our theoretical insights and achieving state-of-the-art results in several settings. Our findings underscore the potential of integrating classical theories for linear systems with modern learning techniques to build robust, interpretable, and data-efficient forecasting models. The code is publicly available at: https://github.com/Wangzzzzzzzz/RootPurge.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper advances a linear time-series forecasting framework centered on characteristic roots: in the noise-free regime, it shows that characteristic roots govern long-term dynamics; in noisy settings, it exposes the MSE training data-scaling law and the risk of spurious roots, indicating that simply enlarging the sample size is insufficient for robust generalization. Accordingly, the authors introduce two complementary structural regularization strategies: (i) low-rank constraints on the parameter matrix via Reduced-Rank Regression (RRR) and Direct Weight Rank Reduction (DWRR) to recover low-dimensional latent dynamics; and (ii) Root Purge, an adaptive loss that learns a noise null space and performs online root restructuring, with training applicable in either the time or frequency domain. Evaluated on benchmarks including Traffic, Electricity, Weather, Exchange, and ETT, the methods deliver strong results.

### Strengths
1. According to the reported results, the proposed methods achieve strong empirical performance across multiple datasets, which is particularly notable given the simplicity of the designs.  

2. The paper provides substantial theoretical analysis with clear exposition to clarify the core concepts.  

3. A clear roadmap is presented, and overall readability is high.

### Weaknesses
1. The core insight “preserving dominant characteristic roots to suppress noise” is not novel and aligns with fundamental conclusions from classical Singular Spectrum Analysis (SSA).  

2. The technical contributions of DWRR and Root Purge appear simple.  

3. The Root Purge objective may induce model degeneration: the model could prefer mapping any input to zero, and the paper offers no theoretical guarantees to prevent or mitigate this failure mode.  

4. The theoretical analysis depends on i.i.d. Gaussian noise and stationarity assumptions to justify data efficiency and the regularizer; these assumptions often do not hold in practice, and no generalization beyond them is provided.

### Questions
1. Under complex, non-stationary noise typical of real-world series, do the proposed methods retain theoretical advantages?  

2. Can the approach be extended to more complex deep learning architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work presents a comprehensive study of linear models for time series forecasting, focusing on the role of characteristic roots in shaping model expressivity. Experiments validate our theoretical claims and demonstrate the effectiveness of both methods across a range of forecasting tasks.

### Strengths
1. The research motivation of this manuscript is clear and its content is very rich。

2. This manuscript is theoretical.

3. Sufficient experiments have verified the effectiveness of the proposed method.

### Weaknesses
1. Theoretical analysis is an advantage of this work, but it can easily make it difficult for readers to understand. It is necessary to provide intuitive explanations in the key derivation process

2. The manuscript lacks a comprehensive discussion of related work, which hampers readers, particularly those less familiar with the domain. Thus, it remains unclear how the proposed approach advances the state of the art or distinguishes itself from existing methods.

### Questions
1. What is the transferability of the method?Can linear models be used in other areas than time series learning?

2. Why is the frequency-domain linear layer necessary for Root-Purge instead of a mere implementation detail?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes linear time series forecasting via characteristic roots , identifying that noise creates "spurious roots" and a "data scaling law" that demands excessive data to mitigate this noise. To improve robustness and data efficiency, it proposes two structural regularization strategies: (1) post-hoc Rank Reduction (RRR) and (2) a novel "Root Purge" method. Root Purge is an adaptive training regularizer that encourages the model to learn a null space for the estimated noise (residuals).

### Strengths
1.Originality and Clarity: The "Root Purge" loss function is a novel, elegant, and intuitive regularization technique.

2.Quality and Significance: The paper provides a valuable insight into why simple linear models fail in noisy, low-data regimes.

### Weaknesses
1.Hyperparameter Sensitivity: The methods' performance relies heavily on tuning $\rho$ or $\lambda$. This data-specific tuning is difficult to estimate a priori and undermines the claim of robustness.

2.Contingent on Low-Rank Assumption: The methods are motivated by an assumption that the true signal is low-rank. However, on large, complex datasets, the paper's results show the optimal rank is often full , and the proposed methods offer minimal benefit . This limits the methods' generality.

3.Theory-Practice Gap (Trends): The theoretical analysis assumes homogeneous equations (constant/zero bias) . This is a poor fit for real-world data with non-stationary, dynamic trends.

### Questions
1. Does the Low-Rank Assumption limit the generality of method, making it effective only for a specific subset of (simpler) time series datasets? 

2.If we encounter a complex trend in the future that Instance Normalization can't flatten, does your Root Purge method just fail? How can you convince us that it's still robust in that scenario?

### Soundness
2

### Presentation
2

### Contribution
3
