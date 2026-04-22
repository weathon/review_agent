# When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
Score-based methods, such as diffusion models and Bayesian inverse problems, are often interpreted as learning the data distribution in the low-noise limit ($\sigma \to 0$). In this work, we propose an alternative perspective: their success arises from implicitly learning the data manifold rather than the full distribution. Our claim is based on a novel analysis of scores in the small-$\sigma$ regime that reveals a sharp separation of scales: information about the data manifold is $\Theta(\sigma^{-2})$ stronger than information about the distribution. We argue that this insight suggests a paradigm shift from the less practical goal of distributional learning to the more attainable task of geometric learning, which provably tolerates $O(\sigma^{-2})$ larger errors in score approximation. We illustrate this perspective through three consequences: i) in diffusion models, concentration on data support can be achieved with a score error of $o(\sigma^{-2})$, whereas recovering the specific data distribution requires a much stricter $o(1)$ error; ii) more surprisingly, learning the uniform distribution on the manifold—an especially structured and useful object—is also $O(\sigma^{-2})$ easier; and iii) in Bayesian inverse problems, the maximum entropy prior is $O(\sigma^{-2})$ more robust to score errors than generic priors. Finally, we validate our theoretical findings with preliminary experiments on large-scale models, including Stable Diffusion.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies diffusion models and argues that their success is due to learning the manifold supporting the data distribution rather than finely learning the data distribution itself. The authors support their claim by showing that, in the small noise limit, the dominating term in the noised log density is determined purely by the manifold and does not involve the data distribution. The authors go on to argue that from a methodological perspective, the aim should be to estimate the manifold rather than the distribution since this task can be done successfully even with larger errors in the score estimation; in this sense, geometric learning is easier than distribution learning. Additionally, the authors propose a sampling algorithm to sample from the uniform distribution on the manifold given a score function estimator. Finally, the authors present experimental results to support their claims.

### Strengths
The paper makes a good case for switching paradigms to geometric learning (i.e. learning the uniform distribution on the manifold) as the score error requirements are much less stringent than that required for full distribution learning. It also seems natural to me from a generalization perspective that the manifold ought to be the target. It is also quite nice that a simple (one-line!) modification to the sampling dynamics can tolerate higher score errors and give uniformly distributed samples on the manifold. The paper is written very clearly and the authors provide good exposition. The experiments indeed show better diversity as one expects from targeting the manifold rather than the data generating distribution.

### Weaknesses
1) The result of Theorem 4.1 requires an $L^\infty$ bound on the score estimation error. This is quite stringent and undesirable, especially since diffusion models are typically trained via score matching which is an $L^2$ loss. Furthermore, there is some empirical evidence that $L^\infty$ assumptions are not typically satisfied in practice (e.g. see Section 3.1 of "Fast Sampling of Diffusion Models with Exponential Integrator", Zhang & Chen 2023). The authors do note this in their Limitations section, but it is a weakness nonetheless. 

2) The results in Theorem 3.1 are somewhat qualitative (i.e. consistency results and weak convergence), and it would be a stronger paper if quantitative rates in some distance (e.g. Wasserstein distance) were available. For example, if E(\sigma) = o(1), is it meaningfully easier to estimate the uniform distribution on $M$ rather than the true $p_{\text{data}}$? If it is not, then perhaps the justification for geometric learning becomes weaker. 

3) I'm confused about the justification for learning the uniform distribution from the Bayesian Inverse problem perspective. I agree that choosing it as the uniform yields a weaker score estimation condition for correct posterior sampling. But I'm not convinced this is by itself a good enough reason. There is strong information that is being lost by switching to the uniform distribution which might have resulted in a "better" posterior distribution but harder to sample from. It's not obviously clear why this is less desirable than a less informed prior with easier sampling requirements.

### Questions
1) Can the authors discuss regularity conditions on $p_{\text{data}}$? It seems that learning the uniform distribution on the manifold can get quite difficult if there are meaningful regions on $M$ where $p_{\text{data}}$ is very small. But this doesn't seem to be reflected in any of the results. Is this due to the qualitative nature of Theorem 3.1? It would be very clarifying if the authors could discuss this point in some detail.

2) Can the authors provide some high-level discussion on sample complexity? With finite data, is it the case that learning the uniform distribution on $M$ is only going to be feasible when $p_{\text{data}}$ is bounded above and below by a constant on its support (i.e. there's enough mass everywhere on the manifold)?

3) What are the difficulties in getting a quantitative version of Theorem 3.1 (say in Wasserstein distance or something else)? It would be nice to add a bit of discussion about this in the paper.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors studies a rate separation when score-based methods learn manifold data, namely that recovering the data manifold is easier than recovering the exactly underlying distribution on the manifold, where the difficulty is measured by the error tolerance for score approximation. The authors thus argues for a paradigm shift from distribution learning to geometry learning - more specifically, targeting uniform distributions on manifolds - and introduces the Tampered Score (TS) dynamics to generate distributions on manifolds that are close to uniform.

### Strengths
1. The authors' main hypothesis - that recovering manifolds is easier than recovering the exact distributions on manifolds for Langevin-type dynamics with approximate score functions - is novel to my knowledge. If fully validated on score-based generative models, it can potentially shape our understanding of how such models fundamentally work.

2. The authors derive precise theorems to quantitatively characterize the hypothesis, enabled by a rigorous framework based on differential geometry and impressive mathematical techniques (not fully checked).

3. Although the idea of tempering the drift term in Langevin dynamics is not entirely new, the proposal to use it for learning uniform distributions on sub-manifolds is novel to the best of my knowledge, and also as a modification to the corrector step in predictor-corrector sampling.

4. The presentation is clear overall.

### Weaknesses
1. The analysis is performed towards the stationary distributions of the various score function, rather than the distributions obtained from the denoising dynamics (i.e. reverse ODE / SDE) as in score-based generative models. Hence it is unclear to what extent the theoretical insights derived for the former idealized setting also hold for the latter (even in the continuous-time setup). In the latter case, for example, even small deviations in the score function can cause the final distribution to be supported beyond the manifold.

2. In particular, I wonder whether it tends to take longer for the TS dynamics to converge to the stationary distribution than the (untempered) Langevin dynamics. Intuitively I suspect this is the case since the drift is reduced with $\alpha > 0$. As the theoretical analysis is concerned only with the stationary distributions, the results are unable to address questions regarding the rate of convergence, which are nevertheless quite relevant for practice.

3. In the paragraph after Theorem 5.2, the authors claim that the TS scheme helps with recovering "the uniform distribution on the data manifold from samples of $p_{data}$", but I don't see a justification of this by the theoretical results. In fact, all the main theoretical results require the regularity assumption (Assumption 2.2) that the noiseless distribution $p_{data}$ has a continuously differentiable density on the manifold, which does not hold in the case of empirical distributions of finite samples.

4. In Remark 4.1 on Page 5, the authors claim that the assumption of a compact support of the limiting distribution is reasonable because "many diffusion models apply clipping to generated samples". I think this is a bit misleading - applying clipping as a post-processing step after sampling doesn't affect the properties of the stationary distributions of the score functions, about which the assumptions are stated.

Typos:
1. Line 202, "difficult" -> "difficulty"
2. Line 336, "satisfie" -> "satisfy"

### Questions
See the "Weaknesses" section above.

### Soundness
2

### Presentation
3

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
While score-based models such as diffusion models are typically thought as approximating the data distribution, this paper hypothesizes that---assuming the popular manifold hypothesis---their success arises from implicitly learning the manifold instead ("geometry learning").
The authors provide theoretical results that show that (1) recovering the data distribution (as the noise level $\sigma \to 0$) is *difficult* as it requires a strict $o(1)$ error on the learned scores, while (2) mere *concentration* on the data support (i.e., the manifold) can be achieved with a much larger score error of $o(\sigma^{-2})$. However, in the latter case, it is shown that the learned distribution supported on the manifold can be arbitrary. To this end, the paper proposes a simple modification to Langevin-based samplers that draws from the *uniform distribution* on the manifold, which still only requires $o(\sigma^{-2})$ score errors. Similarly, the authors show that in Bayesian inverse problems, sampling from the posterior induced by a prior that is uniform on the manifold also tolerates $o(\sigma^{-2})$ score error, while $o(1)$ error must be assumed when picking the (Gaussian smoothed) data distribution as the prior.
The author's theoretical results are empirically validated using low-dimensional synthetic data, as well as pre-trained large-scale diffusion models.

### Strengths
+ **Motivation and Relevance.** The work is clearly motivated: Understanding learning dynamics in score-based models is highly relevant for contemporary generative modeling and Bayesian inference, but also for potential downstream tasks such as using the learned density for out-of-distribution detection etc.
+ **Presentation.** The theoretical results are presented well. The Taylor expansion in Theorem 3.1 gives a good intuition for the results, while being less technical.
+ **Novel, important theoretical results.** The rate separation results (Theorem 4.1) are novel and important for this line of research. Under standard assumptions, the results give new insight into why learning geometry information is "easier" than distribution learning in score-based models.
+ **Uniform-on-manifold sampling.** Tampered Score (TS) Langevin dynamics is a surprisingly straightforward adaptation that comes with relatively strong theoretical guarantees regarding convergence to the uniform distribution on the manifold. Specifically, the guarantees also hold in the practical case where $s(\cdot, \sigma)$ is a non-conservative vector field (as in most diffusion models that directly output the score).

### Weaknesses
+ **Empirical Validation.** 
	+ The empirical results are limited in scope, and do not directly support the main rate separation results in Theorem 4.1. The paper would benefit from such experiments (e.g., synthetic data with known manifold and ground truth scores, controlled injection of score error, systematic analysis on how manifold concentration and distribution recovery behaves). 
	+ The experiment in Section 7.1 would benefit from a quantitative evaluation of the qualitative results in Figure 2. Moreover, higher dimensional synthetic experiments (with known manifolds) would shed light onto more practical settings, as score errors can be very different in 2D when compared to high-dimensional problems.
	+ The experiment in Section 7.2 seems weak as it shows only slight quantitative improvements on merely three prompts. No error bars are provided, and it is unclear if the results are significant. While in Table 2, $\alpha=1$ is fixed, in Table 1, TS was tuned on both the number of correction steps *and* $\alpha$ , while PC was only tuned on the former, which makes the tuning budget unbalanced.
	
+ **Discretization Error.** All theoretical analysis assume continuous time models, while discretization error is disregarded. However, in practice, discretization error has an empirically large influence on the performance. Extending the theory in this direction would make it more practically applicable.

### Questions
+ How sensible are the results in Section 7 to the choice of $\alpha$? 
+ A possible relaxation of the $L^{\infty}$ error assumption is mentioned in the limitations. Can you elaborate on the importance of this relaxation? How could the presented results be related to e.g. the denoising score matching loss (Fisher divergence) of trained models?

### Soundness
3

### Presentation
3

### Contribution
3
