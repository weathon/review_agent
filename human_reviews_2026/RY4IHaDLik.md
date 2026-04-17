# CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 6

## Abstract
Accurate uncertainty quantification is critical for reliable predictive modeling. Existing methods typically address either aleatoric uncertainty due to measurement noise or epistemic uncertainty resulting from limited data, but not both in a balanced manner. We propose CLEAR, a calibration method with two distinct parameters, $\gamma_1$ and $\gamma_2$, to combine the two uncertainty components and improve the conditional coverage of predictive intervals for regression tasks. CLEAR is compatible with any pair of aleatoric and epistemic estimators; we show how it can be used with (i) quantile regression for aleatoric uncertainty and (ii) ensembles drawn from the Predictability–Computability–Stability (PCS) framework for epistemic uncertainty. Across 17 diverse real-world datasets, CLEAR achieves an average improvement of 28.3\% and 17.5\% in the interval width compared to the two individually calibrated baselines while maintaining nominal coverage. Similar improvements are observed when applying CLEAR to Deep Ensembles (epistemic) and Simultaneous Quantile Regression (aleatoric). The benefits are especially evident in scenarios dominated by high aleatoric or epistemic uncertainty. Project page: https://unco3892.github.io/clear/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes CLEAR, a method for constructing calibrated prediction intervals in regression by explicitly combining aleatoric and epistemic uncertainties. It estimates aleatoric uncertainty through quantile regression on residuals and epistemic uncertainty via bootstrapped ensemble variation, then forms intervals as a linear combination of both components.

### Strengths
The strengths of this paper are:
- The method builds on an intuitive and practically relevant distinction between aleatoric and epistemic uncertainty, and operationalises it in a simple, interpretable way through a weighted combination of the two components.
- I usually dislike arbitrary \lambda weight params, but in this paper, it is justified in the method and offers an interpretable measure of how epistemic and aleatoric components contribute to overall predictive uncertainty.

### Weaknesses
The weaknesses of this paper are:
- CLEAR builds directly on existing CQR-derived approaches that incorporate uncertainty decomposition, particularly Uncertainty-Aware CQR (UACQR; Rossellini et al., 2024). As acknowledged by the authors themselves, UACQR can be viewed as a special case of CLEAR with \gamma = 1. CLEAR’s main extension is to calibrate both parameters, allowing it to adjust for miscalibration in the aleatoric component. While this generalisation is reasonable and practically useful, it represents an incremental refinement. Thus, novelity is very low in my opinion.
- Correct me if I am wrong, but in the experimental setup, the same data is used to tune \lambda and calibrate \gamma, which should break the independence assumption required for coverage guarantees.
- Since this is derivative to CQR and other prior methods, not comparing against interval-creation baselines in the main paper seems like a missed step. Why leave competitive results for Appendix D? This has left me confused.
- After spending much time reading the appendix, I am left confused by the structure of the paper. It seems like the strong literature comparisons, justifications, good results are all in the Appendix and the main body of the paper is given less thought. This seems backwards to me?

### Questions
- In the default implementation, the same dataset is used both for tuning \lambda and conformal calibration. How do the authors reconcile this with the independence requirement of split-conformal prediction? Can they provide empirical or theoretical evidence that nominal coverage is still preserved under this data reuse?
- Since CLEAR is presented as an evolution of CQR-derived conformal methods, why are direct comparisons against UACQR and related interval-construction baselines relegated to Appendix D rather than integrated into the main results? Would including these baselines in the principal tables change the strength of the empirical conclusions?
- Much of the substantive discussion—literature positioning, theoretical justification, and broader comparisons—appears only in the appendix. Could the authors explain the rationale for this organisation and whether key arguments could be moved into the main text to improve clarity and self-containment?

### Soundness
2

### Presentation
2

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
This paper introduces CLEAR (Calibrated Learning for Epistemic and Aleatoric Risk), a novel framework for constructing regression prediction intervals by adaptively balancing epistemic uncertainty and aleatoric uncertainty. Theoretically, it guarantees asymptotic conditional validity; empirically, on 17 real-world datasets and synthetic data with distribution shifts, CLEAR outperforms baselines by reducing interval width and quantile loss while maintaining nominal coverage.

### Strengths
1. Unlike methods that use a fixed ratio to combine the two uncertainties, CLEAR selects the balance parameter based on data characteristics. For example, it emphasizes aleatoric uncertainty when working with data with few features and epistemic uncertainty when using data with many features, making it flexible across different scenarios.
2. It works with various uncertainty estimation methods (including tree-based PCS and deep learning-based Deep Ensembles or Simultaneous Quantile Regression) and maintains reliable coverage even for data points outside the training data distribution or in extrapolation regions—areas where many baseline methods struggle.

### Weaknesses
1. The proof requires that "at least k base models in the PCS ensemble are consistent with the true function," but it does not define specific criteria for determining consistency (such as error convergence thresholds) nor explain the basis for selecting k. In practical experiments, the consistency of different models varies significantly, yet the paper fails to analyze the risk of theoretical guarantees failing in such scenarios.

2. The additivity of the two types of uncertainty has not been proven — epistemic uncertainty and aleatoric uncertainty essentially belong to risks of different dimensions. The additive combination implies the assumption that the two can be directly superimposed on the numerical scale, but the paper does not verify this assumption. For example, it does not compare the performance differences between additive, multiplicative, and nonlinear combinations.

3. In some datasets, there is a significant correlation between the two types of uncertainties. The additive combination may amplify the uncertainty superposition effect, leading to overly wide intervals. In low-correlation datasets, however, the additive combination may underestimate risks due to improper weight allocation. The paper does not analyze the impact of uncertainty correlation on the combination structure, which limits the generality of the method.

### Questions
1. Regarding computational efficiency, CLEAR’s grid search for lambda (over 4000 points) consumes significant resources, especially for large datasets. Have you explored adaptive search strategies? If yes, what was the reduction in computational time while maintaining performance?
2. This paper verifies CLEAR’s performance on 17 regression datasets, but most of these datasets have relatively balanced feature distributions. For high-dimensional sparse datasets (e.g., tabular data with hundreds of features where most are irrelevant), how does CLEAR’s performance change?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose the CLEAR algorithm for UQ in regression, with the following steps.
1. An epistemic model is built to learn the mean and the boundaries of a confidence interval.
2. The predicted mean is subtracted from the target and an aleatoric model is trained on the residual. The boundaries of the confidence interval is kept.
3. Two coefficients $\gamma_1$ and $\lambda$ are determined, from which $\gamma_2 = \lambda * \gamma_1$. For all possible $\lambda$, separately $\gamma_1$ are computed by calibration, and $\lambda$ that optimizes the evaluation metric is selected.
4. The aleatoric/epistemic intervals around mean are scaled symmetrically with the coefficients.
I find the contribution is the combination of known methods with a grid search, with limited theoretical justification and experimentation.

### Strengths
+ Learning the combination of aleatoric and epistrmic uncretainties in one model is an important practical question
+ Reproducibility in the supplementary material
+ Improvement in experiments

### Weaknesses
- Contribution is the combination of known methods via grid search 
- If the distribution is non-Gaussian, the proposed method is limited due to its strong focus on the mean, and its symmetry. Why not median or another statistic? How about skewed distributions? Bimodal?
-  Theoretical justification is limited

### Questions
How can you handle the following limitations of the proposed method:
 - confidence interval is built around the mean $\hat{f}$, which is subtracted from the target for training the aleatoric model. Why not, for example, the median?
- Confidence intervals are treated symmetrically, how bout skewed distributions, or bounded target domain?
- How about shape parameters beyond mean and confidence intervals?
 - It is assumed that if you subtract mean from the target, what is left is aleatoric uncertainty. How about e.g. bimodal?

What is the relation of the proposed method to  https://en.wikipedia.org/wiki/Nonhomogeneous_Gaussian_regression ?  There, also an ensemble is used to calibrate the prediction for the mean (deep ensemble as an epistemic model in the paper). Gaussian variance corresponds to the aleatoric distribution, which actually needs to be assumed at several places in the proposed method.

How do you measure whether the two types of uncertainties are properly balanced? Is there ground truth?

The introduction claims "However, they may suffer from poor conditional coverage, meaning well-calibrated coverage at the individual or subgroup level", i.e. in the literature, marginal calibration is insufficiently solved. How does the proposed method solve the question?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing methods typically address either aleatoric uncertainty due to measurement noise or epistemic uncertainty resulting from limited data, but not both in a balanced manner. In this manuscript, however, the authors propose a calibration method that combines both aleatoric (data noise) and epistemic (model/data limitation) uncertainties to improve the conditional coverage of predictive intervals for regression tasks. Dubbed CLEAR, the framework uses two learnable calibration parameters, $(\gamma_1, \gamma_2)$, to combine the two uncertainty components. CLEAR is compatible with any pair of aleatoric and epistemic estimators, enabling adaptive weighting based on data characteristics; unlike prior methods that fix their ratio (e.g., $\gamma = 1).

### Strengths
CLEAR presents a principled, practical, and empirically effective framework for calibrated uncertainty quantification by adaptively fusing epistemic and aleatoric components. Its main innovation lies in the dual-parameter calibration, which yields sharper, better-calibrated intervals without sacrificing coverage. In particular, its key strengths are:

**Strengths:**

1. **Balanced Uncertainty Integration**: CLEAR uniquely combines both aleatoric (data noise) and epistemic (model/data limitation) uncertainties using two learnable calibration parameters $\gamma_1 \text{ and } \gamma_2, \text{ with } \lambda = \frac{\gamma_2}{\gamma_1}$, enabling adaptive weighting based on data characteristics; unlike prior methods that fix their ratio (e.g., $\lambda = 1$).

2. **Improved Performance**: Across 17 real-world regression datasets, CLEAR consistently achieves narrower prediction intervals (e.g., 28.2% and 17.4% average width reduction vs. baselines) **while maintaining nominal 95% coverage**, outperforming its components (CQR and PCS ensembles) and other strong baselines.

3. **Flexibility & Generality**: CLEAR is model-agnostic. It works with various uncertainty estimators (e.g., PCS ensembles + quantile regression on residuals, or Deep Ensembles + Simultaneous Quantile Regression), demonstrating broad applicability.

4. **Theoretical Justification**: The paper provides asymptotic conditional coverage guarantees under mild consistency assumptions (Lemma 2.1), addressing a known weakness of standard conformal methods like CQR, which often under-cover in low-density or extrapolation regions.

5. **Practical Design Choices**: Estimating aleatoric uncertainty on residuals (rather than raw targets) improves stability; using quantile loss for $\lambda$ selection ensures proper scoring and incentivizes conditional calibration.

6. **Interpretability**: The learned $\lambda$ offers insight into whether aleatoric or epistemic uncertainty dominates in a given problem (e.g., $\lambda \approx$ 0.6 vs. 14.5 in the Ames Housing case study with 2 vs. 80 features).

### Weaknesses
Although CLEAR enjoys some key benefits over existing methods, it does have some limitations. In particular, its key weaknesses are:

**Weaknesses:**

1. **Dependence on Base Estimators**: CLEAR’s performance hinges on the quality of the underlying aleatoric and epistemic estimators. Poor base models may limit gains or require careful tuning.

2. **Calibration Data Requirements**: The dual-parameter calibration ($\gamma_1, \lambda$) uses the validation set for both model selection and calibration. While empirically effective, this lacks finite-sample marginal coverage guarantees unless a separate calibration split is used (as shown in Appendix G).

3. **Computational Overhead**: While the grid search for $\lambda$ is fast, the full pipeline requires training ensembles (e.g., 100 bootstraps in PCS), which can be expensive on large datasets, although the authors note this is modular and parallelizable.

4. **Regression-Only Focus**: The method is developed and evaluated only for regression; extension to classification or structured prediction is left for future work.

5. **Limited Theoretical Scope**: The asymptotic guarantees assume i.i.d. data and consistent estimators—conditions that may not hold in complex real-world settings with distribution shifts or model misspecification.

6. **Empirical Results Interpretation**: While CLEAR achieved improved results over existing methods, the paper failed to properly discuss what those improvements concretely mean in terms of choosing CLEAR over existing methods, and if such improvements can translate to more complex regression tasks.

### Questions
1. Would it possible to test CLEAR on toy classification problems?
2. Could the authors provide some asymptotic compute cost for larger datasets? Put differently, how do the authors see CLEAR scale for larger datasets?
3. The paper focuses mainly on I.I.D datasets, without providing clear evidence on or discussing how CLEAR would perform on non-i.i.d data. Could the authors provide some insights for non i.i.d data?

### Soundness
3

### Presentation
3

### Contribution
3
