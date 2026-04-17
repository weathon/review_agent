# Efficient Approximate Posterior Sampling with Annealed Langevin Monte Carlo

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
We study the problem of posterior sampling in the context of score based generative models. We have a trained score network for a prior $p(x)$, a measurement model $p(y|x)$, and are tasked with sampling from the posterior $p(x|y)$. Prior work has shown this to be intractable in KL (in the worst case) under well-accepted computational hardness assumptions. Despite this, popular algorithms for tasks such as image super-resolution, stylization, and reconstruction enjoy empirical success. Rather than establishing distributional assumptions or restricted settings under which exact posterior sampling is tractable, we view this as a more general "tilting" problem of biasing a distribution towards a measurement. Under minimal assumptions, we show that one can tractably sample from a distribution that is simultaneously close to the posterior of a noised prior in KL divergence and the true posterior in Fisher divergence. Intuitively, this combination ensures that the resulting sample is consistent with both the measurement and the prior. To the best of our knowledge these are the first formal results for (approximate) posterior sampling in polynomial time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an algorithm for posterior sampling under oracle access to the score function of the prior distribution along the OU process. Their key insight is using annealed Langevin dynamics to represent the particle-level dynamics of the curve of measures $(\mu_t \propto p_t e^{-R_y})_{t \in [0,1]},$ where $p_t$ is the annealed prior and $R$ is the likelihood, and a rate parameter $\kappa$ is introduced to control the speed at which this path is traversed. Under fairly standard assumptions, they bound the Fisher information between the generated distribution and the true posterior, as well as the KL divergence between the generated distribution and a noisy version of the prior. They demonstrate in a specific example that, due to the KL divergence guarantee, their algorithm does not suffer from mode collapse issues that Fisher information guarantees are susceptible to, and they argue that sampling from a noisy posterior is the best that can be hoped for in the worst case.

### Strengths
- To the best of my knowledge, this paper is the first to provably sample from the (approximate) posterior in polynomial time at this level of generality. The proof is interesting and a nice synthesis of different ideas.
- The main text of the paper is well-written and the authors made a good effort to explain the high-level ideas of their algorithm.

### Weaknesses
- It is not clear whether the rate achieved is optimal (I would guess not), and the iteration complexity is not carefully tracked.
- The proofs in the appendix have many typos/errors and many details are not completely filled in. This makes it difficult to assess the contribution.
- Even though the primary contribution of the paper is the theory, I still would have liked to have seen a couple more experiments to demonstrate the algorithm.

### Questions
- I have some questions about the example in Section 4.1 involving the Gaussian mixture prior and linear inverse problem. When describing the failure of the Fisher information, you take $\lambda \rightarrow \infty$, but when describing the ability of your algorithm to overcome mode collapse, you fix $\lambda = 1/\eta.$ Is this choice without loss of generality? Also, it seems that in this example, the posteriors $\mu_t$ converge to $\mu_0$ in KL divergence, which will not always be the case. In these more general cases, do you still expect your algorithm to overcome the mode collapse issue?
- I don't quite understand precisely how you apply Girsanov to prove Theorem 4.4, could you clarify that in more detail? From the definition of annealed LD, I don't see where the velocity field $v_t$ shows up. I assumed that this followed from rewriting the continuity equation to include a diffusive term, but I am not completely sure. 
- In the proof of Lemma B.2, it seems the parameters $\alpha$ and $\beta$ are chosen sub-optimally (if I understand correctly, $\alpha$ is the convexity parameter of the potential and $\beta$ is the Lipschitz constant of the score). Since the likelihood is convex, can't you simply take $\alpha = 1$? Also, it seems $\beta$ can be taken to be $1 + R$ rather than $\sqrt{d} + R$, since you only need an operator norm bound on the Hessian.

Other questions/typos:
- In Assumption 4.1, you have items i) and iii). Perhaps the first assumption is supposed to be broken into two parts, with the second being the Lipschitz score assumption?
- In Step 3 of the proof of Lemma B.2, starting in the 4th equality, it seems that you drop the term $\textrm{KL}(\mu_{T_{ws}} \parallel \mu_{\infty})$, but shouldn’t this term appear in the final bound? I don’t think it changes much for the final bound, since it only contributes a factor of $\epsilon^2$ under your choice of $T_{ws}$.
- In Step 3 of the proof of Lemma B.2, I think the 5th equality (taking the sup over $x$) should be an inequality.
- In line 888, “We have $\lim_{\delta \rightarrow 0} \int f d(\mu_t - \mu_{t-\delta}) =$...” I think you are missing a factor of $1/\delta$. Also, here you should justify exchanging the limit and supremum.
- In the proof of Lemma B.3, I think you should have $f \leq f(0) + \|x\|$ instead of $f \leq \|x\|$. Also, I don't understand the reduction to non-negative $f$, because the class of Lipschitz functions includes unbounded functions (so you cannot simply subtract off a constant). It doesn't even seem like you need $f$ to be positive in the proof.
- In the proof of Theorem B.4, line 906, I think there is a typo in the integral $\int_{i \delta}^{i \delta + \delta} \textrm{FI}(\rho_{I\delta + \delta} \parallel \mu_{I\delta + \kappa})$ (what variable is being integrated?).
- Typo at the top of page 19, $L^2$ should be a script 'L'.
- On page 19, it seems like there is some text missing between the inequalities in lines 978 and 982, please fill those details in.
- At the end of the proof of Theorem 4.5 on page 19, you have a bound on the Fisher information between the averaged $\rho_t$ and $\mu_0$. How do you translate this into a bound on a single iterate? You mention earlier about using convexity of the Fisher information in the first argument, but this gives you an inequality in the wrong direction.
- In the proof of Theorem 4.4, starting at line 1008, I think $\nabla \mu_{k \delta}$ is supposed to be $\nabla \log \mu_{k \delta}$.
- On page 20, in the proof of Theorem 4.5, you eventually throw away the $O(\delta)$ term in the bound, but shouldn't this term dominate the $O(d \delta^2)$ term?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a novel approximate posterior sampling framework for diffusion models, aimed at improving inference efficiency while preserving sample quality. The method reformulates the posterior inference as an optimization problem and derives an approximate sampler that leverages diffusion score information with reduced computational cost. Theoretical analysis supports the validity of the approximation, and experiments show strong empirical performance on benchmark datasets.

### Strengths
This is a well-written and technically sound paper with clear motivation, strong theoretical grounding, and solid empirical evidence. The methodology is both elegant and practical, providing a meaningful contribution to improving inference efficiency in diffusion models. The paper is well-organized and easy to follow, with a consistent logical flow from theoretical derivation to experimental validation.

### Weaknesses
I have only two minor concerns that do not undermine the overall quality of the paper. 

1. At line 159, the authors claim that two processes have the same joint distribution. However, this statement may not be strictly accurate: one process is a Markov process while the other is an 'inverse' Markov process, and although they share the same marginal distribution at each fixed time $t$, their joint distributions are not identical. This point should be clarified for mathematical precision.

2. At line 235, two distinct notations for score functions are introduced without explicit definitions. Although the meaning becomes clear later in the text, it would improve readability and self-containment to define them formally when first introduced.

### Questions
NA. See the weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies approximate posterior sampling using a pre-trained score model for the prior. The authors propose an algorithm based on Annealed Langevin Monte Carlo (ALMC) and analyze its convergence. The main contribution is a theoretical guarantee that the algorithm can, in polynomial time, produce samples from a distribution that is close to the posterior of a noised prior in KL divergence and close to the true posterior in Fisher Divergence. This result is presented as a way to circumvent the known computational hardness of exact posterior sampling.

### Strengths
1. The authors provide a detailed and mathematically rigorous analysis for approximate posterior sampling using a pre-trained score model.
2. The paper does a good job of positioning itself relative to recent negative results (Gupta et al., 2024) and explaining the limitations of existing approaches.
3. The problem of developing provable methods for posterior sampling with diffusion models is of high importance to the community.

### Weaknesses
1.  The paper makes no attempt to bridge the gap between its theoretical findings and practical applications. There are no experiments to illustrate the behavior of the algorithm or the meaning of the theoretical bounds. The claim that the method solves a practical problem is therefore not empirically supported.

2. The practical relevance of the results is limited by strong assumptions. For example, the requirement that the log-likelihood `R(x)` be convex is a major restriction that excludes many inverse problems of practical interest (e.g. phase retrieval). The paper doesn't sufficiently discuss the implications of this assumption or how the guarantees might change without it.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles the problem of posterior sampling: given a pretrained diffusion model that allows you to sample from a prior p(x), and given a negative log-likelihood function R(x), how does one sample from p e^-R?

Doing Langevin Monte Carlo directly on the posterior runs into well-known problems with sampling from multi-modal distributions. It's well-known that in this setting, LMC will have quick convergence in FI but can have arbitrarily slow convergence in KL (e.g. due to well-separated modes). 

This paper's contribution is proposing an annealed Langevin sampling algorithm that has both guarantees in FI with respect to the true posterior as well as, more significantly, guarantees in the KL with respect to a tilted version of the noised prior. It does so by separating the sampling process into two phases: a first phase where you create a warm start by generating a sample from \gamma e^{-R} (standard Gaussian tilted by R). And then finally, a phase where you do annealed Langevin sampling: at each time step, you change the distribution gradually from \gamma e^{-R} to p e^{-R}. Using standard discretization and telegraphing techniques, you can bound the KL.

### Strengths
* The presentation is clear and the overview of the background is well-done. They do a good job of stating clearly the main contribution of the paper (dual guarantees of FI and KL)

* The approach is novel. It seems more normal to start with the prior and incorporate likelihood information (as with Bayesian updating), or to incorporate prior and likelihood information simultaneously (as with standard classifier-free guidance). But incorporating the likelihood information first and then incorporate the prior is different.

* They managed to get a KL guarantee in a setting where KL guarantees have been elusive.

### Weaknesses
* Lack of empirical validiation. Specifically, they have a guarantee with respect to a noised version of the prior, but it's unclear how significant that noising is for degrading the quality of the sample (e.g. the perceptual quality in image diffusion models)

* Related to the above, but while there are polynomial guarantees in d and 1/\epsilon, the possiblity of large hidden constants could be relevant to applications.

### Questions
Could the authors expand on the nature and role of \tau? (The noise level of the prior which gets the KL guarantees.)

How crucial is the convexity assumption on R(x) for the analysis?

Any insight on the empirical performance of this algorithm in simple toy model settings?

### Soundness
3

### Presentation
3

### Contribution
3
