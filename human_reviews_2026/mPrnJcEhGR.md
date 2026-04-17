# Direct Bias-Correction Term Estimation for Average Treatment Effect Estimation

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
This study considers the estimation of the direct bias-correction term for estimating the average treatment effect (ATE). Let $\{(X_i, D_i, Y_i)\}_{i=1}^{n}$ be the observations, where $X_i \in \mathbb{R}^K$ denotes $K$-dimensional covariates, $D_i \in \{0, 1\}$ denotes a binary treatment assignment indicator, and $Y_i \in \mathbb{R}$ denotes an outcome. In ATE estimation, $h_0(D_i, X_i) \coloneqq \frac{1[D_i = 1]}{e_0(X_i)} - \frac{1[D_i = 0]}{1 - e_0(X_i)}$ is called the bias-correction term, where $e_0(X_i)$ is the propensity score. The bias-correction term is also referred to as the Riesz representer or clever covariates, depending on the literature, and plays an important role in construction of efficient ATE estimators. In this study, we propose estimating $h_0$ by directly minimizing the Bregman divergence between its model and $h_0$, which includes squared error and Kullback--Leibler divergence as special cases. Our proposed method is inspired by direct density ratio estimation methods and generalizes existing bias-correction term estimation methods, such as covariate balancing weights, Riesz regression, and nearest neighbor matching. Importantly, under specific choices of bias-correction term models and Bregman divergence, we can automatically ensure the covariate balancing property. Thus, our study provides a practical modeling and estimation approach through a generalization of existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work presents a way to estimate the Riesz representer of a well-studied quantity, the average treatment effect (ATE). In Section 1.2, the authors claim the following contributions:

1. The proposal of a framework for direct bias-correction term estimation.
2. The theoretical analysis of the estimator obtained via direct bias-correction term estimation.
3. Generalizing this framework from L2 losses to Bregman losses.
4. Unifying the existing literatures between Riesz regression and covariate balancing.

I respectfully disagree with the authors that #1, #2, and #4 are novel contributions. See my answer in **Weaknesses** for more detail.

### Strengths
To the best of my knowledge, contribution #3 is novel.

### Weaknesses
As the authors acknowledge later (beginning of Section 5.1), the framework in #1 is equivalent to Riesz regression, an existing framework that has already been used to estimate ATEs and linear regression functionals more generally. As a result, the extensive theoretical analyses of Riesz regression already apply to this method and the resulting estimator, e.g., from these works:

https://arxiv.org/pdf/2104.14737
https://arxiv.org/pdf/1809.05224
https://arxiv.org/pdf/2110.03031

#4 is discussed in this paper (e.g., Section 2.2):

https://arxiv.org/pdf/2304.14545

Finally, while #3 may be novel, the authors don't make a compelling case for why it's worth optimizing a Bregman divergence instead of just an L2 divergence. Making such a case is important since, when Cauchy-Schwarz is applied to the von Mises expansion, the L2 distance between the estimated and true Riesz representer emerges naturally in an upper bound. Hence, clearly making the L2 distance small will also reduce the bias. A clear case would be needed to explain why some other Bregman divergence would be preferred. While making such contributions would be an interesting direction for improvement, I'm skeptical that the improvement to the manuscript would be sufficient to outweigh my other novelty concerns.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel approach to estimating the Average Treatment Effect (ATE) by directly estimating the bias-correction term $h_0$, based on the key insight that the objective function can be reformulated without requiring prior knowledge of the true $h_0$​. The authors first present a basic least squares formulation and then extend the method using Bregman divergence minimization. This general framework unifies and broadens existing approaches such as Riesz regression and covariate balancing.

### Strengths
This paper proposes a general framework for directly minimizing the error between the true bias-correction term $h_0$​ and its estimate $\hat h$, and establishes connections to several existing estimators, including Riesz regression and covariate balancing methods.

### Weaknesses
1. Although the paper emphasizes that the goal is not to estimate the function $r$ directly, the theoretical analysis still appears to rely on its convergence. In particular, the root-n convergence of the ATE estimator is established through the joint convergence of both $r$ and the outcome model $\mu$, following the standard ATE analysis framework.  Since $h_0$ is directly estimated, I am wondering if there are theoretical advantages that can be demonstrated? For example, in the covariate balancing paper "Kernel-based Covariate Functional Balancing for Observational Studies", it can be shown that root-n convergence for ATE is achieved without any modeling assumptions on the weights and without requiring estimation of the regression function.

2. For the comparison with other ATE estimators, it would be beneficial to include a broader set of benchmarks, such as augmented covariate balancing estimators and various weighted estimators commonly used for ATE. Additionally, presenting inference results—such as coverage probabilities of confidence intervals—would strengthen the empirical evaluation. Finally, incorporating a real data application would provide valuable insight into the practical performance and robustness of the proposed estimator.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the estimation of the average treatment effect (ATE) in observational studies. Instead of using the standard approach which estimates the propensity score and then inverts it, the authors propose to estimate the inverse of the propensity score directly. They develop an estimator under this framework and analyze its theoretical properties.

### Strengths
The paper tackles an important and classical problem in causal inference -- ATE estimation under confounding.

The authors provide a theoretical analysis of their proposed estimator, which adds rigor to their contribution.

### Weaknesses
The core idea appears to overlap substantially with the extensive literature on balancing weights, which already focuses on constructing weights that directly achieve covariate balance without explicitly estimating the propensity score. Relevant prior works include Imai & Ratkovic (2014), Zubizarreta (2015), Chan et al. (2016), Zhao & Percival (2016), Fan et al. (2016), Wong & Chan (2018), Zhao (2019), and Wang & Zubizarreta (2020). The paper does not clearly distinguish itself from these studies or state what is fundamentally new.

Estimating the inverse propensity score remains an intermediate step toward estimating the treatment effect. It is not evident what benefit is gained from focusing on this quantity rather than directly estimating the treatment effect.

The authors do not adequately discuss how their method compares in performance or robustness to existing approaches, particularly under nonsmooth or misspecified settings, where other methods (e.g., Robins et al., 2008, 2009, 2017; Yu & Wang, 2024) have explored efficiency improvements.

### Questions
How does the proposed method differ conceptually and practically from balancing-weight approaches such as entropy balancing or covariate balancing propensity scores?

What is the main benefit (either theoretical or empirical) of estimating the inverse of the propensity score directly rather than estimating the treatment effect directly?

### Soundness
3

### Presentation
2

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
This paper proposes a novel framework for direct bias-correction term estimation in the context of average treatment effect (ATE) estimation. Instead of estimating the propensity score as an intermediate step, the authors directly target the bias-correction term $h_0(D, X)$, which appears in inverseprobability weighting (IPW) and doubly robust (AIPW) estimators.
The paper shows that this term can be estimated without knowing the true propensity score by minimizing an equivalent empirical risk, inspired by direct density-ratio estimation (DRE). The authors provide theoretical guarantees under linear, RKHS, and neural network settings, demonstrate asymptotic normality for AIPW estimators incorporating their method, and extend the framework using Bregman divergence minimization, unifying existing approaches such as Riesz regression and covariate balancing.
Simulation studies compare the proposed method with Logistic regression, CBPS, and RieszNet, showing comparable or superior ATE estimation accuracy in moderate-dimensional settings.

### Strengths
1. Conceptual originality: The direct estimation of the bias-correction term without explicit propensity score estimation is novel and appealing, consistent with the Vapnik principle.
2. Solid theoretical backing: Provides consistency and asymptotic normality results under multiple model classes (linear, RKHS, neural networks).
3. Unified framework: The generalization via Bregman divergence elegantly connects DRE, Riesz regression, and covariate balancing.
4. Empirical validation: The proposed approach achieves competitive ATE estimation accuracy relative to existing methods, confirming its practical viability.
5. Clarity and completeness: The proofs and references are comprehensive, and the relation to prior work is clearly articulated.

### Weaknesses
1. Limited empirical scope: Experiments are restricted to synthetic data. Evaluation on semi-synthetic or real-world causal inference datasets (e.g., IHDP, ACIC) would strengthen the empirical claims.
2. Marginal practical improvement: While theoretically elegant, the empirical gains over existing baselines such as CBPS or RieszNet are small.
3. Computational considerations: The method's scalability and hyperparameter sensitivity (e.g., regularization in RKHS or neural networks) are not discussed.
4. Clarity on assumptions: Some theoretical results assume boundedness or Donsker class conditions that may not hold for modern deep models; discussion of these limitations could be expanded.
5. Connection to efficiency: Although asymptotic efficiency is claimed, empirical efficiency gains are not clearly demonstrated.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
4

### Contribution
3
