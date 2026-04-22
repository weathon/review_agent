# CNB: A Bayesian Nonparametric Approach to Optimal Conformal Prediction

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Conformal Bayes has been shown to yield the optimal (i.e., smallest expected volume) prediction sets among all prediction sets with a $(1-\alpha)$ coverage guarantee if the model is correctly specified. However, a critical issue arises when the model is misspecified: the resulting prediction sets, while still satisfying the frequentist coverage guarantee, can become inefficient and suboptimal. To address this limitation, we propose a conformal nonparametric Bayes (CNB) prediction approach, an innovative solution that incorporates Bayesian nonparametric procedures within conformal prediction. This hybrid offers significant improvements over existing methods in three key aspects: (i) it retains the strengths of the full conformal Bayes, (ii) the Bayesian nonparametric layer  enhances robustness under model uncertainty and induces endogenous clustering in the data, (iii) model complexity adapts to the data.  Theoretically, we show that the resulting CNB prediction sets are valid and will converge to the optimal level of efficiency. The proposed CNB prediction approach provides significant improvements over existing methods while ensuring optimality and precise uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies constructing the optimal conformal prediction set under Bayesian modeling.
In particular, it proposes using a non-paramtric modeling approach to approximate the 
true Bayesian model, thereby approximating the optimal prediction set under the (unknown) true Bayesian model.
The proposed method has the frequentist guarantee of conformal inference, and asymptotic Bayesian optimality under 
certain assumptions. The proposed method is evaluated on synthetic and real dataset.

### Strengths
1. The proposed method appears to combine the benefit of Bayesian modeling with the distribution-free guarantee 
offered by conformal inference.
2. The empirical results demonstrate the preliminary benefits of the proposed approach.

### Weaknesses
1. The exposition of the methods and theories lacks sufficient rigor (to be specified in the question section),
making it hard to properly evaluate/interpret the results.
2.The numerical comparisons are limited, and the reported advantages are not particularly convincing—for example, the proposed prediction interval is substantially shorter than those of competing methods only when the sample size is very large.
3. The methodological innovation seems to be limited.

### Questions
1. On page 5, equation (11), I am confused by the use of the running variable $i$ on the right-hand side, and by whether it is correct that it sums over $(x_i,y_i)$ for $i \in\{1,2,\ldots\}$.
This is confusing since (1) $(x_i,y_i)$ is not defined for $i>n$ and (2) in the case of truncation with $K_n<n$, then obviously $\sigma_i$ is no longer invariant to the permutation of $Z_{i=1}^{n+1}$.
2. Theorem 1 claims that the proposed method achieves finite-sample coverage. If I understand correctly, the result only applies to the hypothetical set by enumerating all the 
values of $y \in \mathbb{R}$; what guarantees are provided for the approximation via the AOI strategy, which is also the actual computable prediction set used in the numerical experiments?
3. In Theorem 2, what does "optimality" refer to exactly? Is it the optimal prediction set with some full Bayesian model? Also, I wonder how the convergence depends on the modeling choice 
of $f(y \mid x;\theta)$ and the dimension of $X$.

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
The paper proposes Conformalized Nonparametric Bayes (CNB), a framework that combines Bayesian predictive modeling with distribution-free conformal calibration to construct prediction sets that remain valid and efficient even under model misspecification (addressing a key limitation of existing conformal prediction methods). CNB leverages Dirichlet Process Mixtures Models (DPMM) to model conditional densities and define conformity scores that account for model uncertainty. Theoretical results guarantee finite-sample validity and show convergence to optimal efficiency, while empirical results on synthetic and real data demonstrate that CNB produces smaller, more adaptive prediction sets that maintain target coverage, whereas other methods perform poorly when the underlying Bayesian model is mis specified.

### Strengths
The paper addresses a limitation of conformal prediction, inefficiency under model misspecification, by integrating it with Bayesian nonparametric inference. The formulation is principled, well-motivated, and clearly presented.

Theoretical results guarantee finite-sample validity and asymptotic optimal efficiency, showing that CNB preserves the coverage properties of conformal prediction while converging to the smallest valid prediction sets under the true data-generating model.

The use of DPMMs allows the method to capture complex, multimodal conditional densities, providing adaptability beyond parametric Bayesian models and accounting for model misspecification.

The framework seems to be agnostic to the exact Bayesian model used, it can integrate different predictive models such as DPMMs with Gaussians or DPMMs with regressions.

Experimental results on synthetic and real datasets demonstrate that CNB consistently produces competitive and efficient prediction intervals across a range of sample sizes, empirically supporting its theoretical claims.

### Weaknesses
1. The experimental evaluation is relatively narrow, relying mainly on small synthetic settings and a single real dataset. 

2. Experiments are conducted at a fixed error rate of 0.2, without discussion of why this value was chosen or how performance might vary with different target coverages.

3. It would be valuable to include experiments or discuss the integration of CNB with pretrained or foundation models (e.g., tabular or time-series models) to show how the framework performs in large-scale or representation-rich settings.

4. The paper does not include comparisons or discussion of adaptive conformal prediction approaches designed to handle heteroscedastic or distribution-shifted uncertainty, such as:

   •Romano, Patterson, and Candès (2019), Conformalized Quantile Regression;

   •Gibbs and Candès (2021), Adaptive Conformal Inference Under Distribution Shift;

   •Gibbs, Cherian, and Candès (2024), Conformal Prediction with Conditional Guarantees;

   •Guan (2023), Localized Conformal Prediction;

   •Amoukou and Brunel (2023), Adaptive Conformal Prediction by Reweighting Nonconformity Scores.

Including comparisons or at least a discussion relative to these adaptive methods would help clarify CNB’s practical advantages and positioning within the broader conformal prediction landscape.

### Questions
See weaknessess

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
The present paper considers the application of Dirichlet Process Mixture Models in Bayesian inference coupled with conformal prediction. The authors derive the corresponding conference score, prove marginal finite-sample validity of the method, and draft the proof of its asymptotic optimality. Some experiments on synthetic and real-world data are provided.

### Strengths
- Conformalization of Bayesian methods is an interesting and currently underexplored area of research.
- Potentially, flexible nonparametric procedures can be a powerful tool for inference in real-world applications.

### Weaknesses
- The authors don't clearly convey the benefits of the proposed method compared to more standard Bayesian conformal prediction approaches. The resulting score is still a combination of function values over some samples from the posterior. Even if formally posterior is defined in a different way, one needs to justify possible benefits for practical applications. Some examples could help.

- The theoretical part doesn't contain full proof of the results. In particular, proof of Theorem 2 is not more than a sketch.

- The experimental part is extremely lightweight, with only one real-world dataset considered. One would need more detailed experimental study to comprehensively show the usefullness of the method.

### Questions
1. What does $\mathcal{V}$ mean in Theorem 2? This notation seems not to have been introduced.

2. What is the use of Theorem 3? Isn't it a standard result in Bayesina nonparametrics not directly related to conformal inference?

### Soundness
1

### Presentation
1

### Contribution
2
