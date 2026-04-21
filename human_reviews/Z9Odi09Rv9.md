# Fast and Noise-Robust Diffusion Solvers for Inverse Problems: A Frequentist Approach

- Avg Score: 4.75
- Decision: Reject
- Scores: 8, 5, 3, 3

## Abstract
Diffusion models have been firmly established as principled zero-shot solvers for linear and nonlinear inverse problems, owing to their powerful image prior and ease of formulation as Bayesian posterior samplers. However, many existing solvers struggle in the noisy measurement regime, either overfitting or underfitting to the measurement constraint, resulting in poor sample quality and inconsistent performance across noise levels. Moreover, existing solvers rely on approximating $x_0$ via Tweedie's formula, where an intractable \textit{conditional} score is replaced by an \textit{unconditional} score network, introducing a fundamental source of error in the resulting solution. In this work, we propose a novel frequentist's approach to diffusion-based inverse solvers, where each diffusion step can be seen as the maximum likelihood solution to a simple single-parameter conditional likelihood model, derived by an adjusted application of Tweedie's formula to the forward measurement model. We demonstrate that this perspective is not only scalable and fast, but also allows for a noise-aware maximization scheme with a likelihood-based stopping criterion that promotes the proper noise-adapted fit given knowledge of the measurement noise $\sigma_\mathbf{y}$. Finally, we demonstrate comparable or improved performance against a wide selection of contemporary inverse solvers across multiple datasets, tasks, and noise levels.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates solving inverse problems with score-based diffusion models. Particularly, they address the problem that comes with noisy data and the inconsistency between generated image and noisy measurement common algorithms encounter. To this end, the authors propose a novel algorithm that overcomes this problem. The algorithm is theoretically analyzed and extensively investigated numerically on various tasks and noise-levels.

### Strengths
I really like this paper. The idea is to the best of my knowledge novel and solves a very important problem in the field. The clarity of the argument is excellent as is the presentation. The numerical results are very comprehensive giving a transparent assessment of the potential of the method for common computer vision tasks. I also really like that they test their method on high-noise levels showing that it may indeed have practical relevance, even for severely ill-posed inverse problems (which they haven't tested on).

### Weaknesses
In my view there are no important weaknesses.

The writing is at times a little bit off:
- Introduction: "Generally, A is assumed to be non-invertible, meaning that any solution x satisfying A(x) = y ...". This statement is false. Many inverse problems are invertible: Gaussian and motion deblurring, X-ray tomography. The former is even considered in this paper. This then has consequences for the argument that follows.
- line 50: they refer to "smoothness" as a function, similar to total variation. I guess they mean the squared H1-seminorm ||\grad u||_2^2?
- line 68: "While already effective, this approach suffers from a unique problem where the explicit form of the consistency error ||A(x) − y|| only exists for x = x0 (Chung et al., 2022a)." The error of course exists everywhere but cannot be easily or readily evaluated at the correct location.
- line 214: "Of course, this strategy is only correct when two conditions simultaneously hold true: (1) the measurement operator A is linear, and (2) the inverse problem is noiseless, i.e, η is identically 0." I don't follow their argument. I would be happy to say that (1) and (2) are sufficient for this to be correct (but may not be necessary in general).
- line 260: "Inverse problem solvers (Section 2.2) face a fundamental ..." This of course is only true for score-based diffusion method. Most algorithms for inverse problems do not have this problem.

### Questions
I am fully sold on this paper. Of course the authors may want to reply to the comments as described above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a novel frequentist's approach to score-based diffusion inverse solvers by directly sampling with a data-conditional score. Each diffusion step can be seen as the maximum likelihood solution to a single-parameter conditional likelihood model. This model is derived through an adjusted application of Tweedie’s formula to the forward measurement model. It allows for a noise-aware maximization scheme.

### Strengths
The approach is scalable, and allows for a noise-aware maximization scheme with a likelihood-based stopping criterion that promotes the noise-adapted fit given knowledge of the measurement noise. The authors extensively demonstrate the performance on a variety of tasks from inverse problems.

### Weaknesses
1. The paper makes strong assumptions about the measurement operator $A$. Theorem A.3 assumes $A$ can be decomposed into a linear projection and a surjective function, which, while covering quite many operators, is still restrictive.

2. Although Diffusion Conditional Sampling (DCS) claims computational efficiency by avoiding score function gradients, it depends on a noise-aware maximization (NAM) scheme that adds its own complexity. The paper lacks a thorough analysis of NAM’s computational complexity.

3. The ablation study in Section 5.2 indicates that DCS’s performance depends on the optimizer used in NAM, introducing a sensitive hyperparameter. This sensitivity may reduce the method's computational advantages.

4. Despite recognizing limitations in Tweedie’s formula, the paper relies on it to estimate $x_0$, arguing that the Gaussian assumption in the reverse process justifies this. However, this justification hinges on the accuracy of the score estimate from NAM; any inaccuracies could propagate through Tweedie’s formula and affect reconstruction quality.

5. While Theorem A.3 affirms the sufficiency of the score estimate, I am not sure if the paper provides theoretical guarantees on DCS’s convergence or overall accuracy.

6. Qualitative comparisons in Figures 10-19 reveal subtle differences between DCS and other methods, often with minimal resemblance to the ground truth. This makes it hard to assert DCS’s advantage in image quality or fidelity.

7. DCS’s frequentist approach may limit its use in applications needing uncertainty quantification. Figures 10-19 show none of the methods fully recover the ground truth, but some existing methods enable posterior sampling to quantify uncertainty - a capability that DCS seems to lack.

8. It would have been beneficial to elaborate on a comparison with Y. Sun, Z. Wu, Y. Chen, B. T. Feng, and K. L. Bouman. Provable probabilistic imaging using score-based generative priors.

### Questions
Could the authors clarify Theorem A.3 and its proof? for example, concerning guarantees on the DCS's convergence.

Could the authors comment on point 4. under perceived Weaknesses?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper presents a likelihood-based algorithm for solving inverse problems using diffusion models as regularizers.
The approach involves solving a sequence of maximum likelihood problems with an adaptive early stopping criterion, followed by a backward diffusion step.
In the maximum likelihood stage, the authors apply Tweedie’s formula to estimate a clean sample and then regress over its corresponding residual.
To prevent overfitting the noisy measurement, the authors employ an early stopping criterion based on hypothesis testing.
The authors validate their method through extensive experiments on linear inverse problems with both pixel-space and latent-space diffusion models across different noise levels.
Also, they conduct ablation studies to assess the algorithm's sensitivity to hyperparameter settings.

### Strengths
- Extensive experimental suite
- Discussion of the disparity between the paper and the released code in ReSample algorithm

### Weaknesses
**Technical concerns**

- The paper present problematic discussion of Tweedie's formula, particularly in Lemma 3.1 and Theorem 3.2.
In diffusion models, Tweedie’s formula is valid because the transition kernels are Gaussian; see [1], Section 2.3, for a detailed proof of

$$
E(X_0 | X_t) = \frac{1}{\sqrt{\alpha_t}} (x_t + (1 - \alpha_t) \nabla \log p_t(X_t))
$$

Hence conditioning on $x_0$ and restating Tweedie's formula is irrelevant.

- Equation (17) raises concerns as $x_0$ appears on both sides of the equation.
As $x_0$ is inferred from $x_t$, it can never be accessed directly; only an estimate of its expected value is obtainable.

- The claim in lines 303–304, is problematic. The transition kernel $p_{t|0}(x_t | x_0)$ is an isotropic Gaussian for any $t$, namely

$$
p_{t|0}(x_t | x_0) = N(\sqrt{\alpha}_t x_0, (1 - \alpha_t) I)
$$

hence when t is very close to zero, the kernel is almost a dirac around $x_0$

**Mistakes**

The paper contains several substantial errors that affect the technical clarity and accuracy of the proposed methodology:

- In Equations (5) and (7), $x_t$ is omitted from the drift term of the SDE, which is necessary for a correct formulation of Diffusion Models; see Equation 11 in [2]
- In line 106 (footnote), there is an inconsistency: the score and epsilon terms are swapped. The correct expression should be $\epsilon_\theta = -\sigma_t s_\theta$
- Equation (9) does not align with the sampling scheme proposed in DDPM [3]. In DDPM, sampling is performed by recursively applying the bridge kernel $q(x_{t-1} | x_t, x_0)$; however, here the authors apply the forward kernel $p_{t|0}(x_t | x_0)$ which differs fundamentally.
- In Line 5 of Algorithm 2, the gradient should be taken with respect to $\log p_t$
- In Line 400, the correct term should refer to the Jacobian of the score, not the gradient of the log of the score.


**Irrelevant comparisons in experiments**

The experimental comparisons presented in the paper has inconsistences leading to potentially misleading conclusions:

- In Table 1, the authors compare multiple algorithms that utilize different types of priors (pixel space models and latent space models). However, changing the prior (the regularizer) change the problem being solved. Specifically, Latent DPS, PSLD, and ReSample employ latent diffusion priors, while other algorithms use pixel-space models.
Noteworthy, the use of latent diffusion models introduces significant nonlinearities into the inverse problem due to the auto-encoder.
- In the experiments, it is unclear whether the results for the authors' algorithm are based on a latent-space or pixel-space diffusion model.
This ambiguity also applies to the results reported in Tables 3 and 4 in the appendix

---

.. [1] Meng, C., Song, Y., Li, W., & Ermon, S. (2021). Estimating high order gradients of the data distribution by denoising. Advances in Neural Information Processing Systems, 34, 25359-25369.

.. [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

.. [3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.

### Questions
I find using "hypothesis testing" for the early stopping criterion misleading.
Hypothesis testing traditionally assesses the statistical significance of a hypothesis based on multiple samples, whereas, in the current setup, only one sample of the residual is available at each iteration of the likelihood minimization.
This make hypothesis testing less irrelevant. 
Additionally, setting $\sigma_t$ as the critical probability $p_{critic}$ lacks sufficient justification.
While $\sigma_t$ values remain in [0, 1] in this setup and can thus be interpreted as probabilities, this approach may not extend to Variance Exploding (VE) diffusion models, where $\sigma$ takes values beyond the interval [0, 1].

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposed a diffusion posterior sampling method which made a correction to the score function by a Maximum Likelihood Estimation (MLE). While the MLE is usually ill-defined since in many cases the measurement operator is underdetermined, this paper proposed a remedy by running gradient descent from a good initialization and introducing an early stopping criterion to compute MLE.

### Strengths
The proposed algorithm is simple and has a better performance than DPS.

### Weaknesses
1. The experimental results are confusing. It is possible that some of the baselines were not implemented correctly. 
- In the experiments, many of the more recent algorithms, including PSLD, ReSample, had worse performance than DPS in most settings. This is apparently different than what was reported in the literature. I also have some personal experience of implementing these algorithms, and they all demonstrated clear advantage over DPS in my setting. 
- The paper used Stable Diffusion v1.5 for some of the baselines while using a specialized ImageNet score network for the proposed algorithm. This is clearly unfair. 
- Appendix C.4 seems to have some serious concerns about ReSample. However, not enough evidence was provided to support their arguments. 

2. While one of the major advantages of the paper was claimed to be less computation as it did not require computing Jacobian, this did not take into account possible variants of previous algorithms like DPS, LGD-MC. There are easy ways to get rid of Jacobian in these algorithms, and there should be thorough comparison with them.

### Questions
Please refer to the Weaknesses part.

### Soundness
1

### Presentation
3

### Contribution
2
