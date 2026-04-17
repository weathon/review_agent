# SPATIAL CONFORMAL INFERENCE THROUGH LOCALIZED QUANTILE REGRESSION

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Reliable uncertainty quantification at unobserved spatial locations, particularly for complex and heterogeneous datasets, is a key challenge in spatial statistics. Traditional methods like Kriging rely on strong distributional assumptions such as normality, which often fail in large-scale datasets, leading to unreliable intervals. Conformal prediction offers distribution-free coverage, yet most implementations rely on i.i.d. data and ignore spatial dependence. We propose Localized Spatial Conformal Prediction (LSCP), a model-agnostic framework that couples local quantile regression with conformal calibration to produce spatially adaptive prediction intervals. LSCP conditions on neighborhoods in space to capture local heterogeneity and relaxes i.i.d. requirements: it retains finite-sample marginal coverage under exchangeability and, under stationarity and spatial mixing, attains asymptotic conditional coverage. Across synthetic and real datasets, LSCP consistently achieves near-nominal coverage with tighter and more stable intervals than other existing CP methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses uncertainty quantification for spatial prediction, where traditional methods like Kriging require strong parametric assumptions that often fail in complex datasets, while standard conformal prediction ignores spatial dependence. The authors propose Localized Spatial Conformal Prediction (LSCP), a model-agnostic framework that learns data-adaptive weights through quantile regression on neighborhood residuals to construct spatially adaptive prediction intervals. LSCP automatically learns optimal local weights rather than relying on hand-tuned kernels, capturing complex spatial dependencies. Theoretically, the paper establishes finite-sample marginal coverage under exchangeability and proves asymptotic conditional coverage under the weaker assumptions of stationarity and spatial mixing, with explicit convergence rates. Empirically, experiments on synthetic and real datasets demonstrate that LSCP achieves near-nominal coverage with tighter and more stable intervals than existing conformal prediction methods.

### Strengths
1. The paper demonstrates that local residual patterns contain rich information about spatial uncertainty structure, which can be learned rather than manually specified. This shift from user-designed kernels to data-driven weight learning represents a paradigm change applicable beyond spatial settings.

2. Separating the base predictor from the uncertainty quantifier makes the method model-agnostic and easily deployable.

### Weaknesses
1. While the authors mention spatio-temporal extensions, the current framework is static. Many spatial applications (climate modeling, traffic prediction) have inherent temporal dependencies that aren't captured.

2. The paper doesn't address computational complexity for massive datasets. The QRF-based approach requires fitting models on neighborhood residuals, which could be prohibitive for millions of spatial locations.

3. The core idea of using data-adaptive weights isn't entirely novel: it extends existing weighted CP frameworks (Tibshirani et al., 2019; Barber et al., 2023) by learning weights through QRF rather than specifying them. The "localization" concept has also been introduced, the authors seem to miss another potential baseline here: Han et al., (2022) Split Localized Conformal Prediction.

4. The proof techniques largely follow standard approaches from the time series CP literature, adapting them to spatial settings through mixing conditions.

5. Limited diversity in real datasets, where only mobile signal data is tested.

### Questions
The k-nearest neighbor approach can produce artifacts at spatial boundaries or in regions with highly irregular sampling density, is this ever observed in the experiment and how it might affect the LSCP results?

### Soundness
3

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
This paper proposes LSCP, a novel framework for producing  reliable prediction intervals for complex and heterogeneous spatial data. LSCP combines the flexibility of local quantile regression with a new spatially-weighted conformal calibration procedure. This allows the method to produce prediction intervals that adapt to local data heterogeneity. The key contribution is the theory that LSCP achieves asymptotic conditional coverage, replacing the i.i.d. assumption with spatial mixing and local stationarity.

### Strengths
1. The problem is has broad practical applications
2. The paper provides strong results
3. The localization via spatial weighting is an intuitive and sensible adaptation of CP for non iid scenarios.
4. The authors present a novel theoretical claim demonstrating that LSCP achieves asymptotic conditional coverage. The theorem works under specific, realistic assumptions for spatial data (spatial mixing and local stationarity) rather than the iid assumption.

### Weaknesses
1. The preliminary section is hard to follow. It requires more elaboration on notation and intuition.
2. Experiments are extremely thin. More real-life data experiments would significantly enhance the quality of the paper.
3. Real-life experiments would benefit from better visualization of the spatial heatmap in the main text.
4. There is a lack of analysis on how coverage and efficiency change as the bandwidth is varied, as well as on how to choose the hyperparameters. A sensitivity analysis/hyperparameter sweep showing how coverage and interval width change would be beneficial for the reader and significantly improve the paper's quality. For example, how does the number of neighbors affect the efficiency? Is there an intuition of what hyperparameter should be used?

### Questions
1. What is the size of the dataset needed in order for this framework to be practical? The dataset sizes used in the experiments are extremely large.
2. Is the cross-validation used to select the hyperparameters done on the training or calibration dataset? This needs to be clarified in the manuscript.
3. Assumption 4.1 - the weights produced by the Quantile Random Forest (QRF) decay at a specific rate. Is this assumption always true for QRF, or is it just a condition required for the proof to work?
4. Assumption 4.2: From my understanding, the model must be "good enough" and bounded in a specific way. This is a major departure from standard CP. A key appeal of CP is that its guarantee holds even if the underlying model is not good. This paper's guarantee explicitly depends on the quality of the base model. Is there practical insight on when this assumption holds true?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Localized Spatial Conformal Prediction (LSCP), a model-agnostic method to form spatially adaptive prediction intervals by combining localization with learned weighted quantiles. Given residuals from any base predictor, LSCP trains a quantile regression model (implemented with Quantile Random Forests) that, for a new location, estimates lower/upper residual quantiles from the residuals of its $k$ nearest neighbors. It then chooses a tail-splitting parameter $\beat \in [0, \alpha]$ to minimize interval width. Authors provide several theoretical results about the proposed method, including marginal coverage under iid locations and asymptotic conditional coverage. Experimental evaluation includes three synthetic and two real datasets.

### Strengths
1. The method is easy to implement and model‑agnostic. The empirical study spans heterogeneous synthetic settings (stationary, heteroskedastic, and non‑stationary) and two real states, with consistent improvements in width at near‑nominal coverage.
2. The method is clear and easy to implement.
3. Authors provide a theoretical analysis of the coverage for the provided prediction intervals.

### Weaknesses
Learning data‑adaptive local weights via quantile regression on neighborhood residuals is a simple, appealing idea which was already explored in the recent literature. Improving conditional coverage of conformal prediction based methods by studying the score distribution has been explored extensively in recent years. Apart from LCP (Guan, 2023), multiple other methods were proposed, including the methods of Han (2022), Colombo (2024), Gibbs (2023), Plassier (2024). For example, the earlier SLCP (Han, 2022) approach proposes to estimate the CDF of the residuals on a portion of the calibration set with a kernel-based method. While the mentioned works do not explicitly target spatial setting, they can be readily applied to the authors’ setup by merging feature and location vectors. In this light I consider revising the exposition in light of this previous work to be worthwhile for improving the paper.

Particular technical weaknesses can be formalized as follows
1. Mismatch between theory and implementation: the weight‑decay assumption (Assumption 4.1) appears incompatible with the fixed $k$ neighborhoods used in practice (Appendix C.4).
2. Appendix A.2 equation 8 incorrectly equates the noise with the prediction error.
3. Data‑dependent weights learned on the same calibration set that the empirical weighted CDF is estimated.
4. Inconsistent experiment setup: Section 5 claims $k$ is cross‑validated, while Appendix C.4 fixes $k=50$.

References
1. Split Localized Conformal Prediction (Han, 2022), 
2. Normalizing Flows for Conformal Regression (Colombo, 2024), 
3. Conformal Prediction With Conditional Guarantees (Gibbs, 2023), 
4. Probabilistic Conformal Prediction with Approximate Conditional Validity (Plassier, 2024)

### Questions
1. Assumption 4.1 or fixed $k$: can you please clarify whether your assumption on $M_n$ is satisfied in the implementation? How sensitive are the empirical results to increasing $k$ with $n$? Can the theoretical results be adjusted to incorporate fixed $k$?
2. How reasonable is Assumption 4.3 for the datasets that you consider?
2. Appendix A.2 equation 8 looks incorrect. Does any step in Lemma A.1 or Theorem 4.6 use this identity critically? If so, can you correct the proof?
3. Have you tried to learn data‑dependent weights on a held-out part of the calibration set, not used for conformal prediction?
4. In Algorithm 1, $\tilde{X}(s_i)$ is the vector of neighbor residuals. How do you order these residuals (by distance, by index)? Is the QRF invariant to permutations?

### Soundness
2

### Presentation
2

### Contribution
2
