# Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Probabilistic forecasting of multivariate time series is challenging due to non-stationarity, inter-variable dependencies, and distribution shifts. While recent diffusion and flow matching models have shown promise, they often ignore informative priors such as conditional means and covariances. In this work, we propose Conditionally Whitened Generative Models (CW-Gen), a framework that incorporates prior information through conditional whitening. Theoretically, we establish sufficient conditions under which replacing the traditional terminal distribution of diffusion models, namely the standard multivariate normal, with a multivariate normal distribution parameterized by estimators of the conditional mean and covariance improves sample quality. Guided by this analysis, we design a novel Joint Mean-Covariance Estimator (JMCE) that simultaneously learns the conditional mean and sliding-window covariance. Building on JMCE, we introduce Conditionally Whitened Diffusion Models (CW-Diff) and extend them to Conditionally Whitened Flow Matching (CW-Flow). Experiments on five real-world datasets with six state-of-the-art generative models demonstrate that CW-Gen consistently enhances predictive performance, capturing non-stationary dynamics and inter-variable correlations more effectively than prior-free approaches. Empirical results further demonstrate that CW-Gen can effectively mitigate the effects of distribution shift.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose to use a separate model to predict sliding-window means and covariances and subsequently use those to whiten to prediction targets of generative forecasting models.
Their method improves forecasting results for a broad selection of generative forecasting models on several datasets.

### Strengths
1. Consistent improvements in experimental results
1. Clear graphical illustration of method

### Weaknesses
1. As far as I understand from the algorithms in the appendix, conditional whitening with a JMCE is a wrapper or pre/postprocessing around a generative forecasting model. While this can be interwoven with the diffusion/flow matching dynamics in Section 4, this does not seem essential. If I am correct, I think it would be an advantage to highlight the simplicity of the final method.

### Questions
1. Do you train the JMCE model jointly with the generative model?
1. Line 313: How can this be done more efficiently exactly?
1. Are you modifying the generative models themselves? Or do you train the models on the conditionally whitened data as they are?
1. Have you ablated the different components of your JMCE loss?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces conditionally whitened generative models that incorporate information in the noising process. The authors tailor diffusion and flow matching processes by including conditional information obtained via a mean and covariance estimator (termed JMCE). Their approach is theoretically motivated and empirically demonstrated to improve generative performance.

### Strengths
- The methodology is theoretically justified, and the authors show under which conditions the distance between the prior and conditional distribution is minimized.
- A simple mean and covariance estimator is introduced and well-motivated to parametrize the generative process.
- The proposed conditional whitening leads to empirical performance improvements.
- The framework includes diffusion and flow matching.

### Weaknesses
- The model requires a two-stage process now. First fit JMCE, then train the generative model.
- The derived diffusion process requires the inversion of the covariance matrix, resulting in a higher runtime complexity compared to standard diffusion models. A runtime comparison would aid the comparison.
- The limitations of the method should be discussed more thoroughly. Furthermore, I recommend separating the related work section from the introduction.

Minors:

- L260: Sentence incomplete

### Questions
- Can the model be trained in an end-to-end fashion or does it require a two-stage process?
- Did you try diagonal covariance parametrizations to reduce the runtime complexity?
- Can you elaborate more on the performance of the JMCE model itself? How does it compare in forecasting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CW-Gen, which whitens data using learned conditional means and covariances from a Joint Mean-Covariance Estimator before training generative models. The idea to align the model priors with data statistics is sound and yields consistent empirical gains across datasets.

### Strengths
- Tackle a challenging, important and underrepresented aspect for time-series modelling: multivariate time-series.
- Clear motivation: Proposes a method that allows for a new prior, that is closer to the data-distribution.
- Well written and theoretically founded.
- Extensive evaluation, which show that the proposed prior actually improves state-of-the art models.

### Weaknesses
- Unclear connection of DKL reduction in Theorem 1 to actual practical guarantees: Can you (a) clarify the tightness of the bound; (b) give intuition / toy examples showing when the condition is achievable (or not); (c) explicitly discuss regimes where the condition can fail
- Unclear estimator quality: Covariance targets are noisy; stability and regularization not analyzed.
- The paper notes that CW requires eigen-decomposition per sample / per time step but then glosses over practical limitations. More precise complexity analysis and discussion of remedies (approximate eigen/svd, diagonal + low-rank approximation, block-diagonalization, randomized SVD or factor models) could benefit the paper.
- The paper argues joint mean+full covariance is important. But it’s not fully convincing which component drives gains. There are some ablations (backbone/wEigen) but I don’t see a simple controlled ablation that compares: (a) subtract mean only, (b) subtract mean and scale by diagonal variance (NsDiff style), (c) full covariance JMCE. Can you add a direct ablation and show where full covariance helps.

### Questions
- Can you be very explicit on how your proposed method compare to other (univariate) flow-based and diffusion-based methods that introduced flexible priors, e.g., [1,2]
- Can you add a short experiment showing numeric values of both sides of Equation (3) on training/validation examples so readers can see how far the estimators are from satisfying the bound in practice.
- Choice of sliding window length (95 for most datasets, 15 for ILI) needs justification. Why 95? How sensitive are results to this hyperparameter?


1. Modeling temporal data as continuous functions with stochastic process diffusion, ICLR 2023
2. Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting, ICLR 2025

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CW-Gen, including CW-Diff and CW-Flow. It proposes a novel JMCE to simultaneously estimate the conditional mean and sliding-window covariance of future time series, which guides the whitening process. The authors provide theoretical guarantees showing conditions under which their approach improves the generative model's sample quality by reducing the KL divergence between the conditional distribution and the model's terminal distribution. Experimental evaluations demonstrate improvements in probabilistic forecasting accuracy, capturing non-stationary dynamics and inter-variable correlations better than baselines.

### Strengths
1. The paper establishes conditions that justify why replacing the traditional terminal Gaussian distribution with one parameterized by estimated conditional mean and covariance improves sample quality.
2. The JMCE simultaneously learns accurate conditional means and sliding-window covariances with eigenvalue control to ensure stability and robustness—this nuanced approach effectively addresses non-stationarity and heteroscedasticity.
3. Experiments show the outperformed performance over baselines.
4. The algorithms and theoretical proofs are clearly detailed, and code for experiment reproduction is available.

### Weaknesses
1. The approach involves computationally expensive operations, particularly eigen-decomposition for whitening covariance matrices, which scales cubically with dimensionality. For very high-dimensional datasets, CW-Gen can become quite slow, limiting real-time deployment scenarios.
2. The framework's reliance on complex joint estimators and whitening transformations may impose a higher barrier to adoption compared to simpler baseline models.
3. The paper ablates hyperparameters but does not isolate the individual contributions of conditional mean estimation versus covariance estimation. Which component drives most improvements?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
