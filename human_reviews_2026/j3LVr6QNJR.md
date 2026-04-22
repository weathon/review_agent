# Hopformer: Homogeneity-Pursuit Transformer for Time Series Forecasting

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 8

## Abstract
Forecasting multiple time-series with high-dimensional covariates presents a core challenge: unifying common temporal patterns while retaining meaningful series-specific information. We introduce Hopformer (Homogeneity-Pursuit Transformer), a two-stage forecasting framework that addresses this challenge. In the first stage, our novel Sparsity Pattern Aggregation (SPA) scheme extracts a common, low-variance trend incorporating the covariates. This acts as a homogenization layer. In the second stage, a LoRA-fine-tuned Transformer models the remaining complex dependencies in the residual signals. Our method is theoretically grounded. We prove that SPA achieves a near-optimal bias-variance trade-off via an oracle inequality. We also provide generalization bounds for the second stage under dependent time series via an information-theoretic analysis. Hopformer sets a new state of the art, improving the MASE by an average of 6.56\% on both synthetic and real-world benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Hopformer, a two-stage forecasting framework designed for modeling multiple time series with high-dimensional covariates. In the first stage, a Sparsity Pattern Aggregation module combines predictions from a pool of regression experts to extract a shared low-variance trend across series. In the second stage, the residual signals are modeled using a LoRA-fine-tuned Transformer, enabling efficient adaptation of a pre-trained backbone to capture nonlinear temporal dependencies. The authors provide theoretical guarantees, including an oracle inequality showing near-optimality of SPA and an information-theoretic generalization bound for LoRA under dependent time series. Experiments on six datasets demonstrate that Hopformer achieves up to 6.56% improvement in MASE over strong baselines, while maintaining parameter efficiency. The framework aims to integrate traditional regression aggregation and modern foundation models for scalable, covariate-aware forecasting.

### Strengths
1. The paper addresses the challenge of forecasting with high-dimensional covariates through a structured two-stage framework. The decomposition into a trend extraction stage and a residual modeling stage is conceptually sound and makes the overall workflow easier to follow.

2. The manuscript includes theoretical analysis for both components, which, despite being based on standard assumptions, provides some formal grounding to the proposed approach. This effort adds a degree of rigor and may help readers understand the model’s intended generalization behavior.

### Weaknesses
1. The motivation for leveraging a pre-trained backbone remains unclear. The authors should articulate why fine-tuning a pre-trained model is preferable to training a new model from scratch, and what concrete benefits this strategy brings in terms of generalization, efficiency, or convergence.

2. The proposed divide-and-conquer pipeline has been extensively adopted for modeling individual time series dynamics. The manuscript does not clearly explain what unique challenges arise when extending this idea to capture cross-series shared components. Without a precise formulation of these challenges, the contribution risks appearing incremental. Related works addressing multi-series or cross-sectional modeling should be discussed to better contextualize this study.
[1] [1] Wang, Shiyu, et al. "Timemixer: Decomposable multiscale mixing for time series forecasting." arXiv preprint arXiv:2405.14616 (2024). [2] Deng, Jinliang, et al. "Parsimony or capability? decomposition delivers both in long-term time series forecasting." Advances in Neural Information Processing Systems 37 (2024): 66687-66712. [3] Hu, Yifan, et al. "Adaptive multi-scale decomposition framework for time series forecasting." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 16. 2025.

3. The paper claims that the proposed model is efficient and scalable; however, no empirical evidence or complexity analysis is provided to substantiate these claims. A quantitative comparison of computational cost, memory usage, and scalability would strengthen the argument.

4. The paper’s layout does not fully comply with the official ICLR template. The authors should ensure consistency with the required style and formatting standards.

5. Several statements require clarification or stronger justification. Emerging studies have shown that vanilla Transformers often struggle in time series forecasting, which contradicts the manuscript’s assertion of their strong capability. The theoretical analysis also appears overly general and does not incorporate key properties of time series data—such as temporal dependence and autocorrelation. In particular, Theorem 2 assumes the residual series to be stationary, an assumption that is rarely satisfied in real-world time series. The authors should discuss the practical implications of this assumption and whether their method remains valid when it is violated.

6. Timexer addresses a highly similar problem setup but is not included in the comparative analysis. The paper should clarify the differences and advantages of the proposed model over Timexer and other contemporary baselines.

### Questions
Please refer to the weaknesses

### Soundness
3

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
3

### Summary
The paper introduces HopFormer, a novel framework for multivariate time-series forecasting that explicitly captures multi-scale temporal patterns and cross-frequency interactions. The model first decomposes input sequences into multiple temporal scales, each representing a specific frequency range, and processes them independently through Transformer blocks. In the frequency domain, it employs a Cross-Frequency Transformer Block to model dependencies between different frequency components, addressing the common problem of frequency aliasing.

### Strengths
1. The proposed method explicitly models interactions among frequency components, which is theoretically sound and empirically effective.
2. The paper is well-structured and easy to follow.

### Weaknesses
1. All benchmarks are standard and there is no test on ultra-long or high-frequency financial data, where frequency interactions could be more significant.
2. Although theoretical proof and quantitative evaluation are shown, there is no spectral visualizations of how cross-frequency attention contributes to interpretability.

### Questions
1. Since the proposed method is conducted on frequency domain, it's common to bear high computational cost, which makes the model far less efficient. Is it the case in Hopformer? What is the memory and runtime scaling with respect to the number of scales and sequence length?
2. Does the cross-frequency attention operate on raw FFT bins or learned spectral embeddings?
3. Have you tested whether the learned frequency dependencies transfer across datasets (e.g., pretrain on ETTm1, finetune on Weather)?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the fine-tuning strategy in the universal time-series forecasting in the presence of a covariate variable. The strategy is twofold: first, extract the trend pattern via sparsity pattern aggregation (SPA), which extracts low-variance information from the covariates. Second, compute the residual information from the original data against the trend data and fine-tune the LoRA model to predict the future residual information. The method is validated in various time-series forecasting datasets with covariate information available.

### Strengths
1) The formulation of multi-source time series makes sense. Integrating exogenous variables for forecasting is a timely subject.

2) It is not hard to understand the formulation and the analysis of the paper.

3) The paper further incorporates a theoretical analysis of the SPA estimator and the loss function.

### Weaknesses
1. I think the work misses several concurrent works [1,2] that also assume covariate or exogenous variables. It would be better to evaluate the gain of the proposed method against these methods.

2. The SPA module can be inefficient since it requires the Metropolis algorithm to perform the random walk. It can be inefficient when the number of covariates becomes large, which can be detrimental in the inference stage.

3. This is relatively minor, but I believe the margin of the paper should be corrected in a more readable form.


***References***

[1] ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables, AISTATS 2025

[2] CITRAS: Covariate-Informed Transformer for Time Series Forecasting, ArXiv 2025

### Questions
1. How does the method scale with larger prediction length? I believe standard benchmarks for multivariate time-series forecasting are tested on longer time horizons (e.g., 336).

2. Following W2, I'd like to know the scalability of the method in both time complexity and actual runtime. Would this method be relevant if there is re large amount of covariates?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces Hopformer, a two-stage framework for multi-series time-series forecasting with high-dimensional covariates. Stage 1 extracts a shared low-variance trend via Sparsity Pattern Aggregation (SPA) over a pool of cross-sectional regressors. Stage 2 fine-tunes a pretrained Transformer on the residuals using LoRA. The authors provide (1) an oracle inequality for SPA that relates risk to the best sparse expert subset, and (2) an generalization bound for LoRA under stationary, $\beta$-mixing residuals.

### Strengths
- **Clear architectural motivation**: The two-stage design, i.e., trend homogenization followed by residual refinement, offers a principled way to separate shared structure from idiosyncratic variation in multi-series forecasting.
- **Theoretical grounding**: SPA’s oracle inequality and a LoRA MI-based generalization bound provide clear, stage-specific justification.
- **Scalability**: The framework flexibly integrates various expert pools and Transformer backbones, facilitating adaptation to diverse forecasting scenarios.

### Weaknesses
1. **Potential error accumulation**: Since trend–residual decomposition is inherently non-identifiable, bias in Stage 1 may affect residual distributions. The discussion in Appendix C remains qualitative.
2. **Synthetic-data bias**: Appendix B’s generators assume known future covariates, potentially favoring Stage 1. Controls for leakage/endogeneity or shared latent confounders are not clearly described.
3. **Narrow evaluation metrics**: Results focus primarily on MASE/MAPE point accuracy. Other aspects such as structural fidelity and robustness are underexplored.

### Questions
1. Could the authors quantify Stage 2 degradation when Stage 1 is perturbed (e.g., through missingness or adversarial noise) to characterize error propagation?
2. In the synthetic setup, what steps prevent feature leakage and manage shared latent factors that couple covariates and targets?
3. As noted in Appendix C, how does the method behave when future covariates are unavailable or partially observed, which is common in practice? Would replacing Stage 1 with self-supervised representations adapt well?

### Soundness
4

### Presentation
3

### Contribution
3
