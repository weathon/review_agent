# DistDF: Time-series Forecasting Needs Joint-distribution Wasserstein Alignment

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Training time-series forecast models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach seeks to minimize the conditional negative log-likelihood  of the label sequence, typically estimated using the mean squared error. However, this estimation proves to be biased in the presence of label autocorrelation.  In this paper, we propose DistDF, which achieves alignment by alternatively minimizing a discrepancy between the conditional forecast and label distributions. Because conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a newly proposed joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. This discrepancy admits tractable, differentiable estimation from empirical samples and integrates seamlessly with gradient-based training. Extensive experiments show that DistDF improves the performance diverse forecast models and achieves the state-of-the-art forecasting performance. Code is available at https://anonymous.4open.science/r/DistDF-F66B.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Distribution-aware Direct Forecast (DistDF), which achieves alignment by minimizing joint-distribution Wasserstein discrepancy between conditional forecast and label distributions to enhance forecast accuracy.

### Strengths
1. This paper is well written and polished. Notations and equations are clearly presented and explained.
2. This paper is well-motivated and offers an extremely thorough explanation.
3. Experiments are comprehensive.

### Weaknesses
1. Experimental comparison (Tab. 2) lacks some most recent works, e.g., [*1]. The proposed method might not outperform these new works. TQNet [*1] achieves **0.377** MSE, **0.393** MAE on ETTm1.
2. Experimental results could not fully support the significance of the method. The improvement is marginal when compared to prior art, e.g., TimeBridge, Time-o1, and TQNet [*1]. 
3. Improvement on presentation:
 - For results in the table, should not use **Bold** and $\underline{\text{Underline}}$ when two numbers are the same, use Bold for both.
 - (minor) In Section 4.3, the reference to Table 4 should be changed to Table 2.
 - (minor) Use consistent table style. Use \toprule for Tab. 5 & 6


[*1] Lin, Shengsheng, et al. "Temporal Query Network for Efficient Multivariate Time Series Forecasting." Forty-second International Conference on Machine Learning.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DistDF, a new training objective for time-series forecasting that aims to align the conditional distributions of forecasts and labels, rather than relying on point-wise MSE. Since conditional discrepancies are difficult to estimate from limited data, the authors introduce a joint-distribution Wasserstein discrepancy, optimized between the distributions of (history, labels) and (history, predictions). The method is model-agnostic and can be plugged into existing forecasting models. Experiments show performance improvements on multiple benchmarks.

### Strengths
- Strong and clearly articulated motivation regarding autocorrelation bias in likelihood-based objectives.
- Solid theoretical foundation, including alignment guarantees and non-negativity properties of the objective.
- Method is architecture-agnostic, enabling integration with a broad range of forecasting models.
- Extensive benchmarking shows consistent improvements, supported by ablation studies demonstrating contribution of components.
- Generally clear writing and clean presentation of the mathematical formulation.

### Weaknesses
- A key limitation is that the proposed discrepancy objective lacks guaranteed convergence or clear interpretability during training, making its practical effect on conditional alignment somewhat uncertain. Because the loss must be combined with MSE, the discrepancy may act more like a regularizer than a principled stand-alone objective. Additional empirical analysis of its optimization dynamics and correlation with performance would strengthen the claims.
- More comprehensive experiments are needed to isolate the contribution of the proposed objective. Given that the method relies on a weighted combination with MSE, it should be compared not only against plain MSE training but also against other established time-series learning objectives (e.g., Dilate, Soft-DTW) when similarly combined with MSE. Such comparisons would help determine whether the observed gains stem from the specific discrepancy formulation or simply from augmenting the loss with an auxiliary term.
- Evaluation is restricted to direct forecasting, limiting evidence of robustness across different training paradigms. Additional experiments under an autoregressive setting would be valuable to validate whether the proposed objective is broadly applicable across different forecasting architectures and training pipelines.
- In Table 1, it is unclear which underlying model architectures DistDF is applied to. Since DistDF is a learning objective rather than a new architecture, and the table compares against architectural baselines, the presentation may confuse readers regarding what is being evaluated. Clarifying the base model used for each dataset would improve readability. Explicitly specifying the base architecture for each dataset (e.g., as done in Scaleformer, ICLR 2023) would improve clarity and ensure a fair interpretation of the reported gains.
ref. Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting, ICLR 2023

### Questions
- Since the proposed objective must be combined with MSE for stable training, can the authors provide evidence that the improvement does not simply arise from a regularization effect? For example, how does the discrepancy term alone behave, and how strongly does its reduction correlate with forecasting accuracy?
- The distinction between DistDF and existing learning-objective methods such as Time-o1, FreDF, Koopman-based losses, and Soft-DTW remains somewhat unclear. Can the authors more explicitly highlight the conceptual and practical differences, particularly regarding theoretical guarantees and optimization behavior?
- The discussion of likelihood bias focuses primarily on MSE. Do similar issues arise in probabilistic forecasting frameworks using alternative objectives (e.g., quantile loss, CRPS)? If so, is DistDF compatible with or beneficial under such setups?
- How well does DistDF extend to multivariate forecasting, probabilistic output formulations, or multi-scale architectures? Providing results or analysis in these more general settings would help verify that the proposed approach is broadly applicable beyond the current scope.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a Wasserstein-based discrepancy measure for time series that captures label autocorrelation and demonstrates the benefits of using it for time series alignment compared to established methods. Several experiments were conducted to support this claim.

### Strengths
The presented Wasserstein discrepancy seems original and effective. The experiments seem comprehensive and well carried out.

### Weaknesses
I think the paper should discuss the assumption of Gaussian distributed data more. It seems absolutely necessary to derive the discrepancy measure and yet I suppose the benchmark datasets do not satisfy this property.

I consider this a mild weakness but the theory regarding the general Wasserstein metric is presented mostly for discrete measures. Given that a Gaussian data distribution is assumed, it could be discussed how the presented results for empirical measures relate to the original Gaussian data distribution.

Minor
-------
The Bures-Wasserstein discrepancy is spelled as “Bruce-Wasserstein” in Lemma 3.5. Also in this Lemma, the equality to the W_2 metric should be made clear.

### Questions
Perhaps I am missing something, but Table 1 seems confusing. DistDF, which is a discrepancy measure, is compared to other models. It should be pointed out which model was used with DistDF loss.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses time series forecasting and proposes aligning the predictive conditional distribution with the true conditional distribution by minimizing the joint-distribution Wasserstein discrepancy. This approach mitigates the bias introduced by autocorrelation when using maximum log-likelihood objectives to train forecasting models.

### Strengths
* Good observation that both frequency and PCA components exhibit autocorrelation, which affects their learning bias; this provides a well-motivated basis for applying optimal transport theory.

* The incorporation of DistDF in existing frameworks is straightforward

* Comprehensive experiments; I appreciate the effort to compare with other distributional discrepancies and the application of DistDF to other approaches

### Weaknesses
* The central hypothesis of this work is that aligning conditional distributions is beneficial, and the authors provide theoretical justifications along with empirical evidence through forecasting error metrics. However, it is unclear whether the conditional distributions actually align for the best alpha values reported in Tables 5 and 6. In other words, the hypothesis is not directly evaluated in the experiments through distributional discrepancy, but rather indirectly through forecasting performance.

* Improvements wrt to existing SOTA methods look rather small; however, they are consistent across models and datasets

### Questions
Please address my first point in the weaknesses. If no distributional discrepancy needs to be shown in the experiments, please elaborate why.

### Soundness
3

### Presentation
4

### Contribution
3
