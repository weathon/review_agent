# Sample-efficient evidence estimation of score based priors for model selection

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
The choice of prior is central to solving ill-posed imaging inverse problems, making it essential to select one consistent with the measurements $y$ to avoid severe bias. In Bayesian inverse problems, this could be achieved by evaluating the model evidence $p(y \mid M)$ under different models $M$ that specify the prior and then selecting the one with the highest value. Diffusion models are the state-of-the-art approach to solving inverse problems with a data-driven prior; however, directly computing the model evidence with respect to a diffusion prior is intractable. Furthermore, most existing model evidence estimators require either many pointwise evaluations of the unnormalized prior density or an accurate clean prior score. We propose DiME, an estimator of the model evidence under a diffusion prior by integrating over the time-marginals of posterior sampling methods. Our method leverages the large amount of intermediate samples that are naturally obtained during the reverse diffusion sampling process to obtain an accurate estimation of the model evidence using only a handful of posterior samples (e.g., 20). We demonstrate how to implement our estimator in tandem with recent diffusion posterior sampling methods. Empirically, our estimator matches the model evidence when it can be computed analytically, and it is able to both select the correct diffusion model prior and diagnose prior misfit under different highly ill-conditioned, non-linear inverse problems, including a real-world black hole imaging problem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The choice of the prior distribution plays a central role in Bayesian inverse problems and a common way to select the prior is to use model evidence estimation. In recent years, diffusion models have emerged as powerful tools for constructing expressive priors across a range of challenging tasks. However, the corresponding model evidence is intractable in this framework as all marginal probability density functions are not available explicitly.

To address this limitation, the authors introduce DIME, a novel method for estimating the log-evidence of diffusion-based priors using only a limited number of samples. The proposed approach relies on samples drawn from the marginal posterior distributions of the diffusion process using a procedure inspired by Decoupled Annealing Posterior Sampling (DAPS). Furthermore, the authors enhance the Gaussian approximation of the denoiser by introducing a new covariance estimator, derived from a Gaussian approximation of the true underlying distribution.

The proposed estimator is evaluated using five baselines including sequential Monte Carlo methods across several scenarios: a high-dimensional Gaussian target with known ground truth, two non-convex inverse problems, and a model selection and validation task based on real observations of the M87* black hole. Empirical results demonstrate that DIME consistently outperforms all baseline approaches and effectively applies to real-world black hole imaging problems.

### Strengths
- The proposed method is derived using only a few posterior samples and the squared likelihood score can be estimated with state-of-the -art methods such as Decoupled Annealing Posterior Sampling.
- The algorithm is compared to various baselines in several challenging tasks and consistently outperforms the alternatives.
-  The model selection results for the real M87* data and the posterior samples with the lowest and highest evidence highlights that the method can be used in real data settings, using numerous potential priors.

### Weaknesses
- The proposed algorithm is well motivated and easy to derive from pretrained denoisers. However there are no theoretical guarantees or no intuitions on the data distribution for which the logevidence estimator is good. 
- The algorithm uses DAPS whose annealing process reduces the dependency between samples at consecutive time steps,and allows to obtain marginal posterior samples for any noise level starting from samples from any other noise level. However, the reasearch on posterior sampling using pretrained score-based models has been very active for the past few years and alternatives could be considered. As there is no theoretical guarantee on the proposed estimate, exploring other samplers would support the choice made by the authors.
- The proposed estimate of the posterior covariance is based on a Gaussian approximation of the data distribution using training data. Although it provides a very easy to compute estimate, the assumption that the data distribution is close to a Gaussian is strong in settings where this distribution is likely to be multimodal.

### Questions
The covariance estimate proposed by Lemma 1 used a Gaussian assumption. Would you have insights on how to propose an estimate in a more general setting ? Or to provide guarantees that the proposed estimates is good even in non Gaussian settings.

The literature providing theoretical guarantees for score-based diffusion models is very rich for unconditional and conditional sampling. Could this theory be used to provide guarantees on the proposed logevidence estimator ? 

Did you try score-based alternatives to DAPS to obtain the samples to support that this choice outperforms other posterior sampling algorithms based on pretrained diffusions ?

Could you include computational time to your experiments to understand the trade-off between number of samples from the diffusion and computational complexity ?

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
4

### Summary
This paper introduces DiME, an estimator for the  marginal likelihood $\log p(y)$ when the prior is a pre-trained diffusion model. The key contribution is the estimation of a KL path integral representation of the evidence, leading to a practical estimator that leverages samples from the diffusion posterior sampling algorithm DAPS.

### Strengths
Through compelling applications with extensive experiments, the proposed method seems to exhibit strong performance. The authors have tested the method on:
- Controlled Gaussian prior benchmark:  here the method matches the performance of the SMC baseline. 
- Phase retrieval on MNIST: DiME correctly identifies the generating prior for most digits; confusions appear only for symmetric cases and the method clearly outperforms SMC. 
- Black hole imaging: Here the method provides results that align with physical evidence.

These results show that the method is in itself useful and could be interesting for the ICLR community.

### Weaknesses
I only see one weakness of this work: 

- One of the contributions of the paper is the improved approximation for the posterior covariance but this approximation is already leveraged in a previous work that the paper doesn't cite [1]; see last paragraph of section 3 right before the algorithms. This is the same approximation as the one proposed in this work. 

[1] Linhart, J., Cardoso, G.V., Gramfort, A., Corff, S.L. and Rodrigues, P.L., 2024. Diffusion posterior sampling for simulation-based inference in tall data settings.

### Questions
I am not sure to understand what's the difference between analytic score and clean score; cf line 309: the clean score is never used

### Soundness
3

### Presentation
3

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
This paper provides an efficient method for estimating normalising constants (model evidence) which can be utilized for model selection.

### Strengths
The paper presents the methodology cleanly and displays interesting empirical results, especially presenting an example on a GMM case as well as a practically interesting case of choosing priors for scientific datasets.

### Weaknesses
The paper has multiple weaknesses - ranging from the formulation and empirical results. Mainly

- Some of the theoretical results (see my comments) are well known and not novel
- Empirical comparisons miss some relevant work that also consider full covariance-based approximations of $p(x_0 | x_t)$

In general, it feels like the methodological contribution is limited, although I think experimental section is nicely done (see strengths)

### Questions
- The literature on alternative posterior covariance approximations were not surveyed properly, there are some relevant methods authors should discuss whether they can utilize it.

> Boys, B., Girolami, M., Pidstrigach, J., Reich, S., Mosca, A., & Akyildiz, O. D. Tweedie Moment Projected Diffusions for Inverse Problems. Transactions on Machine Learning Research, 2024.

Please discuss the approximation provided above in the light of Lemma 1. Would the use of approximations given above (or others in the literature) improve your method?

- Proposition 1 is written as a result of this paper, however, the formula for $\log p(y)$ is quite standard (just an application of Bayes rule) and the approximation of the KL divergence also follows from the definition of the diffusion model. I think this is misleading - authors should clarify that this is a standard and well-known consequence of Bayes rule.

- Can you explain the robustness properties of your method? If the data has outliers, how badly does it affect the results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper addresses the problem of selecting an appropriate prior model in inverse problems by proposing an estimator of the model evidence
- The estimator leverages samples generated during the diffusion reverse process and decomposes the evidence computation into two components: the expected log-likelihood and KL divergence whose considered approximation involves the norm of the gradient of the intermediate log-likelihood
- The intermediate likelihoods are approximated using two estimators depending on low/high noise setting.
- As an example, the authors apply the approach to the DAPS and PnP-DM samplers 
- for DAPS, they also introduce a new approximation of the covariance $X_0 | X_t$, estimated empirically from the training dataset through the covariance of the prior $p_0$

### Strengths
- The introduced procedure to estimate the model evidence is practical and may enable informed model selection in inverse problems.
- the proposed approximation of the conditional covariance $X_0 | X_t$ within the DAPS framework is novel

### Weaknesses
**Methodological issues**
- The authors criticize alternative methods for model evidence approximation for being inaccurate and biased (Line 52-53 and Line 64-65). this claim is misleading as the proposed approach relies heavily on Gaussian approximations; and more precisely, the authors don't quantify the impact of such assumption on the approximation
- In Lemma 2, the variance computation of both estimators is incorrect. In particular, equation (12) incorrectly states that $Var(\Theta_{\text{high}}) = O(1/\sigma_t^2)$; however, this ignores the dependence of $\Sigma_{0|t}$ on $\sigma_t^2$, which would yield, if considered, a variance $O(1)$. Consequently, the practical criterion for choosing between low- and high-noise estimators is based on a flawed derivation. In addition, in practice, the authors choose the variant of the estimator with the lowest variance, but this irrelevant as the variance of the estimator are computed with different asymptotic assumptions
- It is concerning that the high-noise estimator in equation (11) does not depend on the observation $y$, but may still be used during the algorithm
- The proposed approximation of $Cov(X_0 \mid X_t)$ is impractical. the authors propose to estimate the empirical covariance $\Sigma_0$ of $p(x_0)$ from training data but this computationally expensive (quadratic memory cost); in addition, this covariance $\Sigma_0$ is involved in matrix-matrix operations and must also be inverted

**Typos and technical inaccuracies**
- Section 2.2: The categorization of posterior sampling methods is irrelevant and doesn't reflect state-of-the-art. Methods such as TDS [2] and MCGDiff [3] are based on SMC and evolve particle ensembles to solve inverse problems; MGPS [4] uses midpoint guidance for estimating transitions; and RedDiff [5] is a fully variational approach.
- Line 269: The equation is mathematically incorrect: the square of $\Theta$ is computed but $\Theta$ is a vector.
- Lines 60–63: The claim that computing the log-density of a prior sample requires integrating the Jacobian of the score overlooks existing work (see Eq. (13) in [1]), where this can be done without vector jacobian product over the score.
- The statement in Lines 207–209 is ambiguous, at low noise, $p(x_t \mid x_0)$ concentrates near $x_0$, so saying it "dominates over the Gaussian approximation of $p(x_0)$" lacks clear meaning.

**Limitations**
The method is applicable in settings where the posterior sampler follows a noise–then-denoise procedure.
This restriction should be explicitly stated. Furthermore, the paper does not discuss how $\Sigma_{0|t}$ is handled in practice. Since the space complexity of thus matrix scales quadratically with the dimension if the problem and appears in the estimator’s computation, it poses computational/memory challenges


---

.. [1] Skreta, Marta, et al. "The Superposition of Diffusion Models Using the It\^ o Density Estimator." arXiv preprint arXiv:2412.17762 (2024).

.. [2] Wu, Luhuan, et al. "Practical and asymptotically exact conditional sampling in diffusion models." Advances in Neural Information Processing Systems 36 (2023): 31372-31403.

.. [3] Cardoso, Gabriel, et al. "Monte Carlo guided diffusion for Bayesian linear inverse problems." arXiv preprint arXiv:2308.07983 (2023).

.. [4] Moufad, Badr, et al. "Variational diffusion posterior sampling with midpoint guidance." arXiv preprint arXiv:2410.09945 (2024).

.. [5] Mardani, Morteza, et al. "A variational perspective on solving inverse problems with diffusion models." arXiv preprint arXiv:2305.04391 (2023).

### Questions
- Since Equation 9 intervenes computation of quantities under different the expectation of different marginals: how many sample paths are used to evaluate them in Algorithm 1 ?
- Can the authors provide the runtime of the algorithms (For DAPS and PnP-DM)?

### Soundness
2

### Presentation
2

### Contribution
2
