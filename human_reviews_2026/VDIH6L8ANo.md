# Convergence Dynamics of Over-Parameterized Score Matching for a Single Gaussian

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Score matching has become a central training objective in modern generative modeling, particularly in diffusion models, where it is used to learn high-dimensional data distributions through the estimation of score functions. Despite its empirical success, the theoretical understanding of the optimization behavior of score matching, particularly in over-parameterized regimes, remains limited.
  In this work, we study gradient descent for training over-parameterized models to learn a single Gaussian distribution. Specifically, we use a student model with $n$ learnable parameters, motivated by the structure of a Gaussian mixture model, and train it on data generated from a single ground-truth Gaussian using the population score matching objective.
  We analyze the optimization dynamics under multiple regimes. When the noise scale is sufficiently large, we prove a global convergence result for gradient descent, which resembles the known behavior of gradient EM in over-parameterized settings.
  In the low-noise regime, we identify the existence of a stationary point, highlighting the difficulty of proving global convergence in this case. 
  Nevertheless, we show convergence under certain initialization conditions: when the parameters are initialized to be exponentially small, gradient descent ensures convergence of all parameters to the ground truth. 
  We further give an example where, without the exponentially small initialization, the parameters may not converge to the ground truth.
  Finally, we consider the case of random initialization, where parameters are sampled from a Gaussian distribution far from the ground truth. We prove that, with high probability, only one parameter converges while the others diverge to infinity, yet the loss still converges to zero with a $1/\tau$ rate, where $\tau$ is the number of iterations. We also establish a nearly matching lower bound on the convergence rate in this regime.
  This is the first work to establish global convergence guarantees for Gaussian mixtures with at least three components under the score matching framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Diffusion models have become go-to approaches in generative modeling, achieving state-of-the-art performance across numerous visual computing tasks. Recent research has extensively analyzed their convergence behavior through metrics such as the Kullback–Leibler divergence and the Wasserstein distance under various assumptions on the true data distribution.

This paper focuses on a simplified theoretical setting in which the ground truth distribution is Gaussian and the score network is parameterized by n learnable parameters. The authors establish, in Theorem 2.1, the convergence of gradient descent applied to the score matching objective,  extending the known relationship between score-based diffusion models and the Expectation–Maximization (EM) algorithm. They prove that, under an appropriate choice of step size, the convergence rate scales with the square root of the number of iterations.

Moreover, Theorem 3.1 demonstrates that for small diffusion times t, global convergence cannot generally be guaranteed; however, convergence to the ground truth can still be achieved under a suitable exponentially small initialization of the parameters. 

Finally, a concise empirical experiment is provided to validate and illustrate the theoretical findings in a controlled setting.

### Strengths
- The paper provides a detailed convergence analysis of the gradient descent optimization in the setting where the true data distribution is Gaussian. It allows to display settings in which convergence is guaranteed with a quantified rate (large t) and settings where convergence depends on the initialization of the parameters (small t). The authors also highlight a behavior where loss convergence may still lead to non converging behavior of the parameter.

-  Even if the chosen true data distribution is really simple, the authors highlight a complex (and technical) analysis which has not been considered in the literature.

- The experiments are simple but support well the theoretical guarantees.  They illustrate both behaviors, the case where all parameters are estimated correctly and the behavior under the other regime, where only one parameter converges and the others diverge.

### Weaknesses
- Of course restricting to a Gaussian case allows explicit computations which are interesting to understand deeply the behavior of the optimization process. Still, most theoretical papers have obtained theoretical guarantees in terms of KL or Wasserstein distance for strongly log concave distributions (even if this assumption has been relaxed recently). Extending to this setting or discussing the difficulty to extend the results, even if strong log concavity is still a strong assumption, would greatly improves the impact of the paper with respect to the literature.

### Questions
- Would it be possible to extend the analysis in the strong logconcave setting ? And if not, the paper being already rather technical, could you discuss the most limiting points where strong log concavity is not enough and Gaussianity really matters to obtain the convergence results ?

- Even if the paper's contribution is mostly theoretical, as the true data distribution is Gaussian, it would be interesting to see the empirical impact (i) of the dimension and (ii) of the structure of the covariance matrix of the true distribution. 

- The main results focus on the classical gradient descent algorithm.  However, score-based models are trained using stochastic gradient descent or adaptive gradient descent. Could you discuss the extension of your analysis for a stochastic gradient optimization ?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the training dynamics of Gaussian mixture models of the score matching objective at a fixed point t. The analysis is conducted under the assumption that the underlying data follows a simple Gaussian distribution with an identity covariance matrix. The “overparametrization” comes from the fact that the class of models is that of Gaussian mixture models with $n$ modes while the underlying distribution is just a Gaussian distribution (i.e. Gaussian mixture with $m=1$ mode). In this work, the authors also show a surprising phenomenon that when the initialization of the weights $\mu_i^{(0)}$ is far away from the target $\mu^*$, the weights converge whilst a slight perturbation in the initialization may lead to divergence of the weight. It is worth pointing out that, although the individual parameters may not converge, the score function still converges to the target score function.

### Strengths
This paper builds on and simplifies the model introduced in Buchanan et al., “On the Edge of Memorization in Diffusion Models”. Its main contribution lies in the analysis of the training dynamics of such models, which provides a complementary perspective to Buchanan et al.’s work. The paper also offers interesting insights into the sensitivity of these dynamics: it demonstrates that even small changes in initialization can shift the system from convergence to divergence, revealing a subtle transition phenomenon.

### Weaknesses
1. I am not completely convinced that this overparametrized setting, even if it can outline interesting phenomena, is the one to be considered. 
2. The authors claimed that it is important in practice to include the score matching in the large noise regime. However, the value of the loss function is scaled by $\exp(-t)$, which should be negligible.
3. It may be better to merge the two separate regimes (i.e. $t$ large and $t$ small), and discuss the effect of initialization, which in my opinion gives a clearer picture of the problem.
4. The problem studied is actually equivalent to approximating the standard Gaussian from different initialization; the multiple reparametrizations (setting $\mu=0$ and $t=0$) are a bit hard to parse.

### Questions
What happens when two or more parameters are closer to the target $\mu^*$ and the others are further? I would expect that the ones that are closer to converge to the target while the others diverge to infinity. A deeper discussion on this would be welcome.

### Soundness
3

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
The paper studies the convergence dynamics of score matching in an over-parameterized teacher-student setting, where the ground truth is a single Gaussian with identity covariance, and the score network is parameterized as a mixture of $n$ Gaussians with identity covariance. When the noise level $t$ is large, it is shown that gradient descent enjoys global convergence at a sub-linear rate. When the noise scale is small, it is shown that there are spurious stationary points, and thus the training is sensitive to initialization. When the parameters are initialized exponentially close to zero, global convergence is established, while under random Gaussian initialization, it is shown that the loss converges to zero at a linear rate, but only one parameter converges while the others diverge to infinity.

### Strengths
The paper makes a significant contribution towards understanding the optimization dynamics of score matching, moving beyond the typical student-teacher setting. The paper identifies a nice set-up which captures some key elements of over-parameterized deep learning while still being mathematically tractable. The results shed light on the qualitative effects of over-parameterization for score matching, in particular the sensitivity of initialization in the low-noise regime.

Additionally, it is a very well-written paper. I did not have time to read proofs in the appendix carefully, but the main paper does a nice job of illustrating the central ideas. Their analysis of the gradient dynamics in the small noise regime requires novel techniques.

### Weaknesses
The only weakness I can think of is that the paper analyzes the gradient dynamics of the score matching loss at a fixed time, whereas in practice, one usually trains a single, time-averaged score matching loss. I don't see this as a major flaw of the paper, but I am curious whether the results could be extended to this setting.

### Questions
See weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
4
