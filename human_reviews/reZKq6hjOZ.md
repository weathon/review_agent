# Broadening Target Distributions for Accelerated Diffusion Models via a Novel Analysis Approach

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Accelerated diffusion models hold the potential to significantly enhance the efficiency of standard diffusion processes. Theoretically, these models have been shown to achieve faster convergence rates than the standard $\mathcal O(1/\epsilon^2)$ rate of vanilla diffusion models, where $\epsilon$ denotes the target accuracy. However, current theoretical studies have established the acceleration advantage only for restrictive target distribution classes, such as those with smoothness conditions imposed along the entire sampling path or with bounded support. In this work, we significantly broaden the target distribution classes with a new accelerated stochastic DDPM sampler. In particular, we show that it achieves accelerated performance for three broad distribution classes not considered before. Our first class relies on the smoothness condition posed only to the target density $q_0$, which is far more relaxed than the existing smoothness conditions posed to all $q_t$ along the entire sampling path. Our second class requires only a finite second moment condition, allowing for a much wider class of target distributions than the existing finite-support condition. Our third class is Gaussian mixture, for which our result establishes the first acceleration guarantee. Moreover, among accelerated DDPM type samplers, our results specialized for bounded-support distributions show an improved dependency on the data dimension $d$. Our analysis introduces a novel technique for establishing performance guarantees via constructing a tilting factor representation of the convergence error and utilizing Tweedie's formula to handle Taylor expansion terms. This new analytical framework may be of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper continues the study of proving convergence bounds for diffusion models under the assumption of accurate score estimation, with a focus on improved dependence on the target error. The "standard" rate achieved by DDPMs is an iteration complexity of $O(d/\epsilon^2)$ for sampling any distribution with finite variance to $\epsilon$ error in TV (after it has been perturbed with a small amount of noise), due to Benton et al. '23 and Conforti et al. '23. A number of recent works have tried to improve the $\epsilon$ dependence to $1/\epsilon$, but only in more specialized settings (e.g. assuming smoothness or bounded support). One of the main results of this work is to show such a result under only the assumption of finite variance. They achieve an iteration complexity of $O(d^{3/2}/\epsilon)$ using a diffusion-based sampler that has approximate access to both the score function *and its Jacobian*. In the smooth setting, they obtain samplers with similar scaling under one of two assumptions: 1) second-order of the smoothness, and 2) Gaussian mixture.

### Strengths
The result in the finite variance setting (Theorem 3) improves upon prior work in a few directions. Firstly, among works that achieve scaling $O(1/\epsilon)$ or better, theirs achieves the best dimension dependence. Secondly, unlike the most relevant preceding work of Li et al. '24c, the present work can tolerate score error. Finally, compared to those prior works, the present work applies in a more general setting where the distribution only has finite variance rather than bounded support.

The Gaussian mixture result is interesting as it requires proving new bounds on the higher-order derivatives of the score function for Gaussian mixtures, which are of independent interest.

### Weaknesses
One weakness is that the sampler requires approximate access to *higher-order* score functions (namely, the Jacobian of the score). While there are examples of diffusion generative models in practice that use higher-order scores essentially by applying automatic differentiation to the first-order score (e.g. GENIE), this is not standard and introduces significant computational overhead. Additionally, these models in practice are not actually trained according to some kind of explicit Hessian matching loss, so it is unclear why one should expect the Jacobian of the score estimates to be close to the Jacobian of the scores. That said, the prior work of Li et al. on acceleration, which appeared in a previous ICLR, also works with second-order access, so this weakness is somewhat justifiable.

Also, not a weakness per se, but note that the result for second-order smoothness (Theorem 4) is technically incomparable to the other previous works getting $1/\epsilon$ scaling because while it does not assume smoothness along the trajectory, it needs to assume that the *Hessian* of the log-density is Lipschitz. And as mentioned above, this result requires second-order scores whereas the probability flow ODE paper of Chen et al. only uses first-order scores.

Another note: while the authors are careful to clarify the chronology of the various related works, it is worth pointing out that in the last month or so (and before the ICLR deadline), there have been important developments in this literature that supersede some of the main results of this submission. For instance, [Li & Yan](https://arxiv.org/pdf/2409.18959) showed that even for the standard SDE-based sampler (without second-order score), one can achieve an accelerated rate of $O(d/\epsilon)$, in fact under only a *first* moment bound.

### Questions
- A recent work of [Gatmiry-Kelner-Lee](https://arxiv.org/abs/2404.18869) also established bounds on higher-order scores of Gaussian mixtures, though specifically in the context of identity-covariance components. How do their bounds compare to the ones proved in this submission?
- To get intuition for the assumption for Theorem 4: Is there an example of a second-order smooth data distribution for which the scores along the trajectory are not all first-order smooth?
- Perhaps it is worth including in the related work some discussion on the use of higher-order scores for diffusion models in practice like GENIE?

### Soundness
4

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
The paper proposes a new approach to diffusion-based generative modeling that leverages second-order derivatives of the log-density function to achieve accelerated convergence. The framework assumes that the underlying distribution has a finite second-order moment and an infinitely smooth density with respect to the Lebesgue measure. For cases where the distribution lacks infinite smoothness, the algorithm is shown to converge to a convolution of the original distribution with a Gaussian distribution.

The main results provide an upper bound on the KL-divergence between the learned and target distributions. This bound depends on the number of iterations, the error in estimating the gradient and Hessian of the log-density, and a remainder term involving higher-order derivatives of the smoothed density function. The paper also examines Gaussian mixtures and distributions with Lipschitz-continuous log-densities in detail, explicitly characterizing the dependence of the remainder term on the dimension in these cases.

### Strengths
1. The paper deals with an important and timely problem.
2. The idea of considering the second-order derivatives to accelerate the convergence is interesting and the results presented in the paper show that it leads to some improvements.
3. The authors did a great job in discussing the prior work and in comparing their results to previous ones.

### Weaknesses
### **Main concern**
My primary concern with this paper is the quality of the writing, which requires significant polishing to reach a publishable standard.

The main contributions of this work are clearly mathematical and reside in the theorems presented. The proofs are highly technical, involving lengthy and intricate computations. However, the current presentation makes the proofs challenging to follow and verify. This poses a problem, as reviewers often lack the time to check such detailed proofs thoroughly. If an error were to be found in a proof, it would substantially diminish the paper’s value.

I strongly recommend that the authors dedicate additional time and effort to revising the manuscript for clarity, making the technical sections easier to parse and verify.

### **Specific comments/typos**
- Lines 176-177: Avoid writing expressions such as "for all $1 \leqslant t$" or "$\forall 1 \leqslant t$", as these read ambiguously as "for all 1 that are less than $t$," which is clearly not the intended meaning. Instead, use "for all integers $t \in [0, T]$" or "for all $t \in \\{1, \ldots, T\\}$" to convey the intended range of values.
- line 215: I find it worrisome that the total variation is defined as the supremum over **all** subsets $A$ of $\mathbb R^d$. This should be replaced by all the measurable subsets of $\mathbb R^d$. 
- Lines 213–218: I do not see any reason for using the notation $\hat P'_0$ in this paragraph. It would be preferable to maintain consistency with the next paragraph by defining the KL-divergence between two measures $Q$ and $P$ instead.
- Line 220: Instead of 'when $q_0$ exists', it would be more accurate to write "when $Q_0$ is absolutely continuous with respect to $\hat P_0'$." The existence of $q_0$ alone does not ensure that the KL-divergence is finite.
- Line 243: For the Ozaki discretization, it would be more suitable to cite the papers
  1.  T. Ozaki. A bridge between nonlinear time series models and nonlinear stochastic dynamical systems: a local linearization approach. Statistica Sinica, 2(1):113–135, 1992.
  2. O. Stramer and R. L. Tweedie. Langevin-type models. II. Self-targeting candidates for MCMC algorithms. Methodol. Comput. Appl. Probab., 1(3):307–328, 1999b
- Line 244: "estimate" -> "estimates"
- Eq. (6): There is no guarantee that this matrix is positive semi-definite and, therefore, can be used as a covariance matrix. I agree that for most $x_t$ this will be the case for large $T$, but it is necessary here to have a definition that works for **every** $x_t$. 
- Line 251: "denote as Hessian matching" -> "refer to as Hessian matching".
- Line 262: It is better to remove the second equality here and to keep it in the claim of Lemma 1 only.
- Line 269 (and in several places in the proof): instead of writing the marginal distributions of $X_0$ and $\bar W_t$, it is necessary to indicate the joint distribution of these random variables. It suffices to write $(X_0,\bar W_t)\sim Q_0\otimes \mathcal N(0,I_d)$. 
- line 280: "satisfy" -> "satisfies"
- lines 277-289: It is generally not recommended to refer to material, such as an assumption, before it has been introduced. In this case, Assumption 3 is mentioned before it is properly stated. Similarly, within the text of Assumption 3, Assumption 4 is referenced prior to its statement. I strongly recommend that the authors adjust the flow of the text to avoid such inversions.
- Lemma 2: In the statement of this lemma and in other places, the meaning of the Landau notation $O(\cdot)$ is unclear. It should be specified that this notation is understood in the asymptotic limit $T \to \infty$. Additionally, in Eq. (10), there should be no term depending on $t$ on the right-hand side. Either $\alpha_t$ should be replaced with $\alpha_T$, or, preferably, the left-hand side should be divided by $(1 - \alpha_t)^2$ within the maximum. **My recommendation would be to avoid these notation and to prefer more conventional writing** ``there exist constants $C_1$ and $C_2$ such that ... for all $T\in\mathbb N$''. 
- Line 281: $\alpha_t$ is not defined in (1). I guess you wanted to write Definition 1 instead of (1).
- Lines 287–288: Since $H_t$ first appears on line 287, it would be clearer to move the indication "defined by (8)" forward to accompany its first mention.
- Line 287: It is unclear why the factor $(1 - \alpha_t)$ is retained on the left-hand side while $O(1 - \alpha_t)$ is used on the right-hand side. Rather than writing $(1 - \alpha_t) g_t = O(1 - \alpha_t)$, wouldn’t it be clearer to simply write $g_t = O(1)$? Additionally, should this relation hold uniformly for $\ell \geqslant 1$?
- Line 300: What is the precise meaning of *analytic* in this context? I was unable to find a definition in the text, either in the main body or in the appendix.
- Assumption 4: Same remark as above, I guess it is possible to suppress the factors $(1-\alpha_t)^{p\ell}/2$ on both sides of the equalities.
- Assumption 4: What is $p$ in this assumption? Should the boundedness of $p$-th order derivatives hold for every $p$? Should the upper bound be independent of $p$?
- Definition 1: This should be stated as a condition rather than a definition.
- Line 342: I did not find the definition of $\widehat P_0'$. 
- Line 376: "titling" -> "tilting"
- Line 389: "rest two terms" -> "remaining two terms".
 
- I guess in the claim of Theorem 1 it is also necessary to assume that $q_0(x)>0$ for all $x$. Indeed, if this condition is not satisfied, the term $\log q_0(\mu_1(X_1))$ appearing in the right-hand side of the equation on line 344 is meaningless.

### Questions
I have several questions regarding Theorem 1 and Corollary 1, which form the central results of the paper. 

In Theorem 1, constants are encapsulated within the notation $\lesssim$. I assume these constants depend on the distribution $Q_0$. It would be valuable to clarify which specific characteristics of $Q_0$ influence these constants. In particular, given that the theorem assumes the density $q_0$ of $Q_0$ is analytic, it would be helpful to know whether the constants depend on the derivatives of $q_0$.

If these constants are independent of $q_0$ and its derivatives, this would imply that the claim of Theorem 1 might not require the assumption that $q_0$ is infinitely smooth. Conversely, if the constants do depend on $q_0$ or its derivatives, then it raises the possibility that Theorem 1's claim may not be applicable in proving Corollary 1.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel accelerated stochastic DDPM sampler that leverages the tilting factor representation and Tweedie's formula. This approach is designed to accommodate a broader class of target distributions, requiring weaker conditions than previous methods.

### Strengths
The paper introduces a novel accelerated stochastic DDPM sampler and demonstrates its applicability to various distribution classes while relying on weaker smoothness conditions and the bounded second moment condition. Additionally, it establishes the first acceleration guarantees for Gaussian mixture models.

### Weaknesses
Assumption 4 plays a crucial role in the construction of the tilting factor, requiring that the score function has derivatives of all orders and that the $p$-th moment is bounded $p \ge 1$. In my view, this condition is not as "rather soft" as suggested in the submission. The requirement for all derivatives to exist means that the score function is not merely smooth but infinitely differentiable, a level of regularity that is often absent in practical score functions, particularly for complex or non-smooth distributions.
Furthermore, the requirement that the $p$-th moment of the derivatives is bounded indicates that the derivatives exist and do not grow excessively. This can be quite a restrictive requirement, especially for distributions with heavy tails or irregularities. While this assumption is essential for maintaining theoretical rigor, it may limit the applicability of the models to a narrower range of problems.

### Questions
The proof is quite lengthy, and I did not have the opportunity to examine all the details. It would be beneficial if the author could elaborate on how the accelerated convergence is achieved using the proposed novel proof technique (specifically, in Step 2 regarding the tilting factor). Additionally, restating Tweedie's formula and clarifying its extension in this context would help enhance the clarity of the proof for readers.

### Soundness
3

### Presentation
3

### Contribution
3
