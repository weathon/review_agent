# High-accuracy and dimension-free sampling with diffusions

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Diffusion models have shown remarkable empirical success in sampling from rich multi-modal distributions. Their inference relies on solving a certain differential equation initialized at pure noise. However, this differential equation cannot be solved in closed form, and its resolution via discretization typically requires many small iterations to produce \emph{high-quality} samples. More precisely, prior works have shown that the iteration complexity of discretization methods for diffusion models scales polynomially in the ambient dimension and the inverse accuracy $1/\varepsilon$. In this work, we propose a new solver for diffusion models relying on the collocation method~\cite{lee2018algorithmic}, and we prove that its iteration complexity scales \emph{logarithmically} in $1/\varepsilon$, and does not depend explicitly on the ambient dimension. More precisely, the dimension affects the complexity of our solver through the \emph{effective radius} of the support of the target distribution only. Our solver constitutes the first "high-accuracy" diffusion-based sampler that only uses approximate access to the scores of the data distribution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new diffusion-based sampling algorithm that achieves high accuracy (logarithmic dependence on error tolerance ($1/\epsilon$)). 
The authors adapt the collocation method to diffusion models, showing that the score function’s time evolution along the probability flow ODE can be approximated by a low-degree polynomial. This enables long, stable integration steps without dimension-dependent error growth.
Under mild assumptions (bounded-support distribution convolved with Gaussian noise), they prove that their sampler reaches total variation distance $\epsilon$ from the target in
$(R/\sigma)^2 \log(1/\epsilon)$
iterations, where $R$ is the effective radius and $\sigma$ the noise level.
The authors claim that this is the first diffusion sampler with polylogarithmic dependence on ($1/\epsilon$) and dimension-free complexity. Theoretical guarantees are supported by rigorous convergence proofs and extensions to Wasserstein and TV metrics.

### Strengths
* **Theoretical novelty:** Provides the first high-accuracy, dimension-free complexity bound for diffusion sampling.
* **Elegant mathematical analysis:** Derives explicit bounds on higher-order time derivatives of the score, enabling low-degree polynomial approximation.
* **Clarity of assumptions:** The “bounded plus noise” assumption is realistic and aligns with practical diffusion model setups.
* **Practical relevance:** Insights into discretization stability and polynomial convergence are directly relevant to improving real-world diffusion solvers.

### Weaknesses
* **Strong assumptions:** Requires sub-exponential tails for score estimation error.
* **No empirical validation:** Results are purely theoretical; no experiments confirm practical speedups or quality gains.
* **Restricted generality:** The framework may not apply cleanly to unbounded or heavy-tailed data distributions common in real applications.
* **Method complexity:** Implementation of the collocation solver may be nontrivial compared to standard discretization schemes.

### Questions
- How sensitive is the convergence rate to violations of the sub-exponential error assumption in practice?
- Does the method maintain stability when applied to very high-dimensional or multimodal image data distributions?
- Could similar dimension-free guarantees hold under weaker moment conditions rather than bounded support?
- Are there empirical results planned to validate these theoretical claims on benchmark diffusion models?

### Soundness
3

### Presentation
3

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
This work introduces a new diffusion model solver based on the collocation method. The proposed solver achieves logarithmic scaling in inverse accuracy and avoids explicit dependence on dimensionality. It is the first high-accuracy diffusion-based sampler that operates with only approximate score information.

### Strengths
(1) The paper proposes a new solver based on the collocation method for diffusion models. A delicate design will lead to high-accuracy sampling.

(2) It proves that a polynomial iteration complexity on $1/\epsilon$, where $\epsilon$ is the sampling accuracy.

(3) The paper provides a detailed theoretical analysis.

### Weaknesses
(1) The introduction of collocation methods contains much ambiguity. For example, “ by polynomial interpolation”, in my understanding, polynomial interpolation first determines where it takes values ($c_i$), then we can find corresponding polynomials. The argument first finding a polynomial basis, then selecting nodes satisfying the equations, seems weird.

(2) Many definitions of norms are not specified in this paper. For example, Lemma 8, did I miss where the $||\cdot||_ {p,\infty}$ is defined? In lemma 9, $||y-\tilde{y} ||$, what is the norm here? Then in Theorem 10, the norm becomes $||\cdot ||_{\infty}$. Should it be the same as that in Lemma 8? I recommend that the readers write a section in the main text, including all the notations used.

(3) Theorem 10: the result mixes different $\delta$, one is the error of initialization distribution, another is the probability (at least $1-\delta$).

(4) The proof of the paper is very hard to read. The technical lemmas, for example, Lemma 6, look horrible. Could you provide a proof sketch of the main theorem? It should discuss how the technical lemmas are applied and why they are necessary. 

(5) What is $\epsilon_{sc}$ in Cor 14? In Theorem 14, why you require $\epsilon_{err} \ge \epsilon$, instead of $<$?

(6) Line 1257, Lemma ??? is not exhibited.

(7) The paper misses some important related work, e.g., [1][2].

[1] Unified Convergence Analysis for Score-Based Diffusion Models with Deterministic Samplers ICLR Li et al.2024

[2] Convergence analysis of probability flow ode for score-based generative models Huang et al. 2024

### Questions
The contribution of this paper is more to apply the collocation method (a numerical method) to the ODE analysis of diffusion models, rather than proposing something specific to diffusion. Do you try to implement the algorithm in practice? How will it behave?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors introduce a new parallel sampler based on the colocation method [1] for diffusion models on Euclidean state spaces. The main contribution of the paper is to show a convergence bound that the iteration complexity scales logarithmically with the inverse accuracy (compared to polynomial for non-parallel samplers) and does not depend on the ambient dimension for a class of target distributions. The interest of the authors for the polylogarithmic dependency with respect to the inverse accuracy stems from the similar picture which exists for sampling where Metropolis-based correction can achieve poly-logarithmic (wrt the inverse accuracy) iteration complexity.

[1] Lee et al. -- Algorithmic theory of ODEs and sampling from well-conditioned logconcave densities

### Strengths
* The paper is well-written. I think the discussion and clarification regarding the high-accuracy situation in classical statistical sampling is useful for the readers. I could follow the paper with ease. 

* The results are new and as far as I know the use of the colocation method [1] in diffusion models is novel. 

[1] Lee et al. -- Algorithmic theory of ODEs and sampling from well-conditioned logconcave densities

### Weaknesses
* Assumption 1 seems extremely strong. I think the authors should discuss more the impact of this assumption. Especially with regards to the works of [1,2] which have weaker assumptions on the target distribution (I do understand that the samplers and results are different in those papers but I think it is important to consider the potential limitations of Assumption 1). It seems that it would be well discussed in the context of the manifold hypothesis. 

* I believe that the related work section is a bit misleading. In particular I would have expected more discussion around the work of [5] which is a parallel method also exhibiting iteration complexity with poly-logarithmic dependency in the inverse of the accuracy. It also heavily relies on a parallel sampler. I am aware of the comment of the authors "Finally, we note that the collocation method (see Section 2.3) has been studied in the context of diffusions, but primarily as a way to parallelize the steps of the sampler Anari et al. (2023); Gupta
et al. (2024); Chen et al. (2024a), but not using low-degree polynomial approximation." This comment in my opinion tames the claims made in the introduction. Further clarification and comparison is required.

* There exists further work improving the dependency of the iteration complexity from linear with respect to the inverse accuracy. In particular, in [6] for instance the authors show bounds of the order of $O(1/\varepsilon^{1/K})$ where $K$ is the order of the samplers (see also the references therein). 

* I do understand that this is a theoretical paper but given that the authors introduce a new algorithm it would be great if they could illustrate the methodology in low dimensional settings at least. 

[1] Conforti et al. (2023) -- KL Convergence Guarantees for Score diffusion models under minimal data assumptions

[2] Benton et al. (2023) -- Nearly d-Linear Convergence Bounds for Diffusion Models via Stochastic Localization

[3] De Bortoli et al. (2022) -- Convergence of denoising diffusion models under the manifold hypothesis

[4] Azangulov et al. (2024) -- Convergence of diffusion models under the manifold hypothesis in high-dimensions

[5] Gupta et al. (2024) -- Faster Diffusion Sampling with Randomized Midpoints: Sequential and Parallel

[6] Li et al. (2025) -- Faster Diffusion Models via Higher-Order Approximation

### Questions
* The authors only discuss the ODE case, what about the SDE one? The introduction of stochasticity is very useful in practice. 

* Assumption 4 is very strong. Could you discuss this assumption and refer to other papers which might use a similar assumptions (if they exist?). 

* I haven't put this concern in the Weaknesses section because it might simply be that I misunderstood part of the paper. Something that is unclear to me is the number of calls required to the model. It seems to me that the number of calls to the model in the Picard iteration is $DN$. Then in the whole diffusion sampler we use $n$ iterations resulting in $NDn$ (by the way there is a typo in Algorithm 2, line2, $k=q$ should be $k=0$). Is $D=d$? I am having trouble understanding the difference between these two quantities. If that is indeed the case then if we require $d$ calls to the model then the method is **highly** impractical in high dimension.

* I am a bit confused as to what is relevant to the "parallel" part of the sampler. In [1] for instance multiple steps are denoised during the Picard iterations. It doesn't seem to be the case here for me. Could you clarify? 

* Could the method be applied to provide convergence bounds  for algorithms leveraging other parallel procedures such as speculative sampling [2]?

* Some lemmas do not compile in the supplementary material

[1] Shih et al. (2023) -- Parallel Sampling of Diffusion Models

[2] De Bortoli et al. (2025) -- Accelerated Diffusion Models via Speculative Sampling

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new solver of diffusion models based on the collocation method to solve ODE, and proves that its sampling complexity is logarithmically in $1/\epsilon$ for $\epsilon$ error. This complexity is improved over  that of the classical diffusion samplers. The paper provides theoretical construction of the proposed sampler and relevant proofs. The key assumption relies on some smoothness property of the score function, which allows a low-degree polynomial approximation. This approximation motivates the use of collocation ODE solver, which has exponential convergence in small time windows. Overall, the sampler is proven to have polylogarithmically complexity.

### Strengths
- The paper is well-written and easy-to-follow.

- This work seems to be the first analysis to give a polylog-complexity diffusion sampler.

### Weaknesses
The contribution of this work is primarily theoretical and the practical implication is not clear. The proposed sampler appears to be a mathematical construction rather than a practical algorithm, as it relies on a polynomial approximation of the score function that is not tractable in real diffusion inference settings. In this sense, the current title and abstract may be somewhat misleading, as they somewhat suggest the existence of an actual sampler rather than a theoretical construction.

### Questions
- What is the “aforementioned Gaussian mixture setting” mentioned in the last paragraph of page 2?

- Algorithm 2 line 2, "for k=q,...,n-1,n", what is q?

### Soundness
3

### Presentation
3

### Contribution
2
