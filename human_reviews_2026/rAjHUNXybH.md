# Computational Bottlenecks for Denoising Diffusions

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 8, 6

## Abstract
Denoising diffusions sample from a probability distribution $\mu$ in $\mathbb{R}^d$ by constructing a stochastic process $(\hat{\mathbf{x}}_t:t\ge 0)$ in $\mathbb{R}^d$ such that  $\hat{\mathbf{x}}_0$ is easy to sample, but the distribution of $\hat{\mathbf{x}}_T$ at large $T$ approximates $\mu$. The drift $\mathbf{m}:\mathbb{R}^{d}\times\mathbb{R}\to\mathbb{R}^d$ of this diffusion process is learned by minimizing a score-matching objective. 

Is every probability distribution $\mu$, for which sampling is tractable, also amenable to sampling via diffusions? We address this question by studying its relation to information-computation gaps in statistical estimation. Earlier work in this area constructs broad families of distributions $\mu$ for which sampling is easy, but approximating the drift $\mathbf{m}(\mathbf{y},t)$ is conjectured to be intractable, and provides rigorous evidence for intractability.

We prove that this implies a failure of sampling via diffusions. First, there exist drifts whose score matching objective is superpolynomially close to the optimum value (among polynomial time drifts) and yet yield samples with distribution that is very far from the target one. Second, any polynomial-time drift that is also Lipschitz continuous results in equally incorrect sampling.

We instantiate our results on the toy problem of sampling a sparse low-rank matrix, and further demonstrate empirically the failure of diffusion-based sampling. Our work implies that caution should be used in adopting diffusion sampling when other approaches are available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a very interesting question: Can denoising diffusions efficiently sample from any distribution $\mu$ for which sampling is otherwise tractable?

The paper presents a rigorous negative answer by showing that diffusion samplers inherit the computational hardness of the underlying denoising task. The core mechanism of diffusion relies on learning the drift term $m(y,t)$, i.e. the Bayes-optimal denoiser. If this denoising problem exhibits an Information-Computation Gap (ICG), a regime where the statistically optimal error is low, but no polynomial-time algorithm can achieve it, then the authors show that diffusion sampling based on score-matching must fail.
The paper includes several key theoretical results.
Theorems 2 \& 5): Efficient diffusion sampling implies efficient near-Bayes-optimal estimation. The contrapositive confirms the central thesis: computational hardness in denoising implies diffusion sampling fails.
Theorem 1: The paper proves the existence of polynomial-time drifts that are super-polynomially close to the optimum (among poly-time algorithms) in the score-matching objective, yet produce samples maximally far from the target $\mu$ (in Wasserstein distance). So the standard training objective is insufficient and potentially misleading when an ICG exists.
Theorem 3: The paper shows that any poly-time, near-optimal drift satisfying a mild any \emph{Lipschitz} near-optimizer fails to sample correctly. Furthermore, \emph{Corollary~5.1} provides a concrete sufficient condition: a $C/t$-Lipschitz drift fails when $t \ge (1+\delta)\,t_{\mathrm{alg}}$.

These results are illustrated using the sparse PCA model where sampling is trivial but denoising is conjectured to be hard below a threshold $t_{\mathrm{alg}}$. Empirical results show that the diffusion process produces degenerate samples.

This work is very significant. It goes beyond analyzing statistical generalization or discretization errors, identifying fundamental limits of denoising diffusions.

### Strengths
Originality. The paper goes beyond the previous analysis of diffusion samplers devoted to KL/Wasserstein bounds or generalization arguments. It provides an interesting computational perspective. In particular, it identifies the information-computation gap as the key problem for correct diffusion sampling; it shows that even near-optimal score-matching drifts (among poly-time methods) can sample the wrong distribution and it establishes reductions from diffusion sampling to denoising.

Quality: Theorems 1--3 are stated with clear hypotheses, and Appendix proofs are thorough. The proofs are relatively easy to follow given the technicality of arguments.

Clarity: The paper is well-written. Assumptions specifying the ICG gap are stated clearly. The paper separates neatly the main conceptual arguments in the main text from the proofs. The figures nicely illustrate the theoretical claims.

Significance: This is a very interesting paper explaining when diffusion samplers fail for computational reasons—even when sampling from $\mu$ is itself easy. They also specify when positive results (which rely on accurate scores or favorable distribution classes) do not apply.

### Weaknesses
Dependence on assumptions: The main results depend on Assumptions 1 and 2, which do not hold unless an ICG is present. This aligns with predictions for models like Sparse PCA, the underlying hardness remains conjectural in the general case.
It would help to map Assumptions 1 and 2 to specific hardness conjectures in the literature.

Robustness of Theorem 1: Th 1 shows that sampling can fail when the drift is $O(n^{-D})$ close to the optimum among poly-time algorithms. This is a fairly stringent definition of near-optimality. How robust are the results to this assumption? even an heuristic argument would be good. 

Construction of the Drift (Theorem 1) The drift $\hat{m}$ constructed to prove Theorem 1 is somewhat artificial. While this demonstrates the insufficiency of the score matching objective, it does not demonstrate that such deceptive drifts arise naturally during standard training procedures (e.g., gradient descent on neural networks).
        \item \emph{Actionable Insight:} Discuss whether standard optimization dynamics are likely to find these specific deceptive local minima, or if this is primarily an existential proof. Are there architectural choices or regularization techniques that might steer optimization away from these solutions?
    \end{itemize}

Lipschitz assumption. Th 3 establishes a negative result for Lipschitz drifts, specifically requiring a $C/t$-Lipschitz condition for $t \ge (1+\delta)t_{\mathrm{alg}}$. However, non-Lipschitz drifts might succeed.
 It would be good to clarify whether this Lipschitz condition is key to the failure of just a technical requirement. Are the typical architectures used in practice (e.g. your GNN in section 3) satisfy this specific $C/t$-Lipschitz property?
    
Applicability of the results. The analysis focuses on distributions where the ICG is well understood, e.g. Sparse PCA. How are these findings translate to standard distributions meet in physical systems; e.g. Lennard-Jones or \phi^4? 
It would be beneficial if the authors could discuss even informally the characteristics of  distributions that might lead to a ICG. 

Minor point: The authors follow a non-standard presentation of diffusion models (the one originating from stochastic localization). I know this is equivalent but i think it hurts the readability of the paper.

### Questions
Alternative Objectives: Proposition 2.1 suggests that score matching is the root cause of the failure when ICGs are present. Does this motivate the development of alternative training objectives for diffusion models that do not strictly enforce denoising optimality, particularly at low SNR (small $t$)?

Optimization: Theorem 1 implies that the score matching loss landscape possessing near-optimal solutions (in the poly-time class) that are bad for sampling.  Do we know anything about whether standard optimization methods are likely to find these bad near-optimizers rather than potential good ones?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In this paper, the authors study the feasibility of sampling with diffusion models. More precisely, they are interested in answering the following question: "Are there distributions such that, there exist almost optimal denoisers that can be computed in polynomial time and for which diffusion sampling fail?" They answer this question by the affirmative. The paper proceeds as follows:

* In  Theorem 1, they show that for any arbitrary polynomial time denoiser which satisfies two generic assumptions, there exists a modified denoiser for which the diffusion sampling fails. In Corollary 3.1, they apply Theorem 1 to the case of sparse random matrix measures. Other sparse examples are considered 

*  While Theorem 1, shows that "good denoiser does not imply good sampling", in Theorem 2 they show that "good sampling implies good denoiser".

* In Theorem 3, they show that under an additional Lipschitz condition, the results can be strengthen to show that *any* polynomial time algorithm which is nearly optimal for the denoising task will lead to poor sampling quality.

The authors conclude the papers with toy numerical experiments.

### Strengths
* The paper is theoretically strong. I have to command the quality of the writing (apart from the introduction of diffusion models and some notation choice discussed in the "Weaknesses" section). Even though I am not a domain expert on stochastic localization and random matrix theory, the paper was pleasant to read and the ideas clearly exposed. 

* The results presented in the papers will be of great interest for the theoretical diffusion model community. Those results indeed fill a gap in the literature regarding what can be achieved with diffusion models.

### Weaknesses
* One of my main pain point with the paper is the choice of following the notation and convention of [1]. While I understand that it might be easier for the authors who seem to familiar with the techniques developed in [1] to build their theory on top of this framework it severely limits the adoption of the work, as it has to be translated back into "classical" diffusion model framework (see [2, 3, 4] for a (non-exhaustive) list of papers which adopt the classical diffusion model convention). The burden of the explanation and communication should be on the authors and not on the readers.

* In my opinion, the strongest result is Theorem 3 but there is little discussion on how realistic is the Lipschitz assumption the authors consider. I found Corollary 5.1. to be very hard to read. To come back to the Lipschitz justification I am not sure I understand the following sentence "We assume the Lipschitz constant to be C/t, because the input of the denoiser is yt = tx + Wt, and hence the two t-dependent factors cancel.)".

* Something that is unclear to me is the generality of the results presented in the paper. While Theorem 1 is very general it is not quite clear to me how to apply them beyond the sparse random matrix measure studied by the authors. In particular, I am wondering how specific those results are. To further specify my question: is (one of) the contribution(s) of the paper 1) the identification of a set of measures under which the assumptions stated in the general Theorems are valid thereby showing some limitations of diffusion models 2) the statement that most data distributions (even the ones considered in practice) fall under the category of the general Theorems thereby showing the much more stronger statement that diffusion models are flawed. It would be good to clarify  the scope of the paper in that respect. 

[1] Montanari (2023) -- Sampling, diffusions, and stochastic localization

[2] Conforti et al. (2024) -- KL Convergence Guarantees for Score diffusion models under minimal data assumptions

[3] Li et al. (2024) -- O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions

[4] Chen et al. (2022) -- Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions

### Questions
* "l. However if such drifts exist, our results suggest that minimizing the score matching objective is not the right approach to find them (since the difference in value with bad drifts will be superpolynomially small)." (l.186) This is an interesting remark. There is some evidence that diffusion models fail at approximating the score. First, they are trained on the empirical measure hence the target distribution is the "optimal" target gives a fully memorizing denoiser.  Second, due to neural network approximations they often fail to correctly approximate the true denoiser. I would be keen to understand how the authors deal with that intrisic limitation of diffusion models and how this would affect their analysis. 

* In Proposition 2.1., I do not understand why we can have equality to 0 in (ii).

* Could you provide more details regarding the construction of the drift in Proposition 2.1? (the construction of a drift which is a bad approximation to the true denoiser but gives good diffusion sampling results). 

* Before Theorem 1 I think it would be worth providing just a few sentences regarding the validity of Assumption 2 and Assumption 1. At this stage it is very much unclear for the reader if those assumptions are easy or not to satisfy.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper's primary contribution is to rigorously prove that a computational bottleneck, not just a statistical one, limits the power of denoising diffusion models. The authors prove that not all easily-sampled probability distributions can be efficiently learned by these models. They construct a specific counterexample based on the sparse submatrix estimation problem, which is conjectured to be hard. This shows that even a nearly-perfect, efficiently-computed drift function will fail to generate accurate samples, thus identifying a fundamental gap between what is statistically possible and what is computationally feasible for this class of generative models.

### Strengths
1. This paper's core strength is its novel perspective, shifting the analysis of diffusion models from standard statistical limits to the overlooked statistical-computational gap.
2. The authors construct an excellent counterexample using exactly the sparse submatrix estimation to provide a distribution that is easy to sample from directly but computationally intractable for a diffusion model to learn.
3. Supportive simulations clearly validate the theory, showing that a computationally-bounded diffusion model fails to generate accurate samples from this "hard" distribution, just as the authors predict.
4. The paper's argument is mathematically rigorous, employing formal, well-established proofs.

### Weaknesses
1. The paper's organization is not straightforward; the structure is relatively loose, which makes the argument hard to follow.
2. The planted clique counterexample feels specific, which makes it difficult to connect this theoretical computational bottleneck to the broad success of diffusion models on real-world data.

### Questions
1. Would considering the 'average complexity' via diffusion over a class of easily-sampled distributions be more meaningful than focusing on 'bad' extreme counterexamples like the sparse submatrix prior?
2. Do your have conclusions about the training procedure in this paper, or does it imply that for the provided counterexample, the training of diffusion models will always fail due to its inherent computational hardness?
3. Is it possible this specific computational bottleneck is just an artifact of this particular formulation of diffusion sampling, or does it represent a fundamental hardness that alternative generative models like flow-matching models could not bypass?

### Soundness
4

### Presentation
3

### Contribution
3
