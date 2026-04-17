# Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
In this paper, we study Riemannian zeroth-order optimization in settings where the underlying Riemannian metric $g$ is geodesically incomplete, and the goal is to approximate stationary points with respect to this incomplete metric. To address this challenge, we construct structure-preserving metrics that are geodesically complete while ensuring that every stationary point under the new metric remains stationary under the original one. Building on this foundation, we revisit the classical symmetric two-point zeroth-order estimator and analyze its mean-squared error from an intrinsic perspective, depending only on the manifold’s geometry rather than any ambient embedding. Leveraging this intrinsic analysis, we establish convergence guarantees for stochastic gradient descent (SGD) with this intrinsic estimator. Under additional suitable conditions, an $\epsilon$-stationary point under the constructed metric $g'$ also corresponds to an $\epsilon$-stationary point under the original metric $g$, thereby matching the best-known complexity in the geodesically complete setting.  Empirical studies on synthetic problems confirm our theoretical findings, and experiments on a practical mesh optimization task demonstrate that our framework maintains stable convergence even in the absence of geodesic completeness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors consider the minimization of a real-valued smooth function on a manifold, using only function evaluations. Contributions are as follows:
- To point an issue with existing zero-th order optimization methods, as follows. The exponential mapping is often defined on an open subset of the tangent space that contains the zero vector. There exist algorithms that rely on the exponentail mapping, but do not account for this fact. Unless the manifold is proved to be geodesically complete, that is the exponential mapping is defined for any tangent vector, such algorithms are ill-defined (theoretical guarantees are not applicable, possible breakdown in practice).
- A theoretical guarantee that the metric of a geodesically incomplete manifold may modified such that it becomes geodesically complete, while preserving the stationary points of any function to be minimized (Th. 2.2).
- a zero-th order gradient estimator that relies on two function evaluations, two geodesic computation, and a sampling routine on the tangent space. A mean-squared error analysis is provided (Th.  2.3).
- an  analysis of the complexity of SGD with zero-th order gradient estimator for finding $\epsilon$-stationary points (Th. 2.5),  and an analysis of the complexity of SGD and mapping from one metric to a geodesically complete metric, and back to the original metric (Coro 2.6).

### Strengths
- the paper identifies a reasonable issue in existing works,
- the result Th. 2.3 account for manifolds curvature, a difficult topic, in a reasonably clear way,
- one experiment (sec. 3.3) stems from a real-world application.

### Weaknesses
The following aspects motivate my current assessment of the paper.
- The scheme of (eq. 2, $f(Exp(\mu v) - f(Exp(-\mu v)))$) differs from the two references (l. 46 -  $f(Exp(\mu v) - f(x))$). Yet, this difference is not acknowledged, motivated, nor discussed.
- I fail to see why does working with a non-euclidean metric precludes the use of previous estimators, such as in Li et al, 2023b.
- The work focuses on situations where the manifold and its metric are not geodesically complete. However, the introduction does not provide any practical situations where this situation occurs. The only example in the paper is optimization on the simplex (fig. 1, and experiments), or a union thereof. I am not convinced this example alone is sufficient to generate interest from the iclr optimization community.
- l. 198-202: I fail to see why existing results in Riemannian optimization may not be applied with arbitrary metrics, different from the ones of the ambiant space.
- The only example with relevant application of a manifold that is not geodesically complete is the unit simplex (Fig 1), or a collection thereof (numerical experiments).
- It is not clear to me that the proposed procedure to sample uniformly on an ellipse is novel (a contribution of this work) or classical. I am not an expert of distribution sampling, but I suspect the rejection sampling on an ellipse is classical.
- The same notation, $\hat{\nabla} f(p)$, is used for the gradient estimator when sampling with exponential mapping and retraction (eq. 2 and 5). It is thus not clear to which estimator Theorems 2.3 and 2.5 apply.
- As far as I understand, while Th. 2.2 guarantees existence of a structure preserving metric, this proof does not help in designing the structure preserving metric in a practical situation, as for instance the example of fig 1 or the experiments. The applicability of the method thus fully relies on the practitionner.
- Experiments, sec 3.1: authors compare SGD with rescale sampling and rejection sampling in terms of SGD iterations. This does not account for the additional complexity of the rejection sampling, relative to the rescale sampling. As such, the plots does not inform on the performance of the complete method (SGD + gradient sampling scheme).
- Experiments, sec 3.3: authors compare variants of SGD relative to the number of iterations, with four different strategies for the update. Besides, there is one trajectory for each method, which does not provide any information on the variance of the methods.  Again, I believe this does not inform on the performance of the overall methods, which I believe is ulitmately the metric of interest.
- Appendix C (proofs) shows several serious issues:
  - Lemma C.5: in view of $f*$ definition, there holds, by definition, $B = 0$.
  - Lemma C.5, l. 1005-1008: the proof is not detailled enough to be checked.
  - the lemmas C.9, C.10 are not referenced anywhere. Lemmas C.9, C.10, and C.13 are not used anywhere as far as I can see.
  - l. 1072: introduces $g$\# but does not uses it.
  - l. 1106: $n$ is used but not defined there, nor in the referenced Lemma C.8.
  - l. 1107: the Ricci tensor is nowhere defined, nor discussed except in this lemma. This lemma is not used is the appendix as far as I can see.

Minor points
- l. 42: what is nondifferentiable modules?
- l. 90: writing issue "while maintain"
- l. 127: syntax issue "present main"
- l. 134: syntax issue "we establishes"
- l.155-160 : ($\epsilon$)stationary point not defined
- l. 255: syntax issue in "metric $g$ the choice"
- l. 303: I would appreciate that the theorems assumptions are stated in the main body rather than supplementary
- l. 305: the SGD dynamics do not solve (1) (i.e. converge to the global minimizers of $f$ on M)
- l. 372: $g_A$ not defined
- l. 408: syntax issue "aligns our"
- l. 939, last sentence of section: syntax issue.
- l. 1015, lemma C.6: syntex "there exists a function [...] is proper"

### Questions
- Why does the scheme of (eq. 2, $f(Exp(\mu v) - f(Exp(-\mu v)))$) differs from the two references (l. 46 -  $f(Exp(\mu v) - f(x))$)?
- Why does working with a non-euclidean metric precludes the use of previous estimators, such as in Li et al, 2023b?
- What are situations where the manifold and its metric are not geodesically complete?
- l. 198-202: why do existing results in Riemannian optimization may not be applied with arbitrary metrics, different from the ones of the ambiant space?
- What are examples with relevant applications of manifold that are not geodesically complete?
- The same notation, $\hat{\nabla} f(p)$, is used for the gradient estimator when sampling with exponential mapping and retraction (eq. 2 and 5). To which estimator do Theorems 2.3 and 2.5 apply.
- As far as I understand, while Th. 2.2 guarantees existence of a structure preserving metric, this proof does not help in designing the structure preserving metric in a practical situation, as for instance the example of fig 1 or the experiments. Am I correct in this? The applicability of the method thus fully relies on the practitionner. Can this be addressed by your theory?
- Experiments, sec 3.1 and 3.3. Can you report experiments relative to time, and with variance indicators for 3.3?
- Appendix C (proofs):
  - Lemma C.5, can you review the proof with more details?
  - What is the use of lemmas C.9, C.10, C.13?
  - l. 1072: why introduce $g$\# but not use it?
  - l. 1106: what is $n$?
  - l. 1107: the Ricci tensor is nowhere defined, nor discussed except in this lemma. This lemma is not used is the appendix as far as I can see. Why is that?

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
2

### Summary
This paper studies Riemannian zeroth-order optimization on geodesically incomplete manifolds. As the main contribution, It constructs structure-preserving metrics that are geodesically complete, conformally equivalent to the original one, while ensuring that every stationary point under the new metric remains stationary under the original metric.  A symmetric two-point zeroth-order estimator was developed with MSE analyzed. The paper then establishes convergence guarantees for SGD with this intrinsic estimator.  The proposed theory and methods are validated via synthetic experiments, and a practical mesh optimization task as well.

### Strengths
1. Connecting the geometric issue of **metric incompleteness** with Riemannian zeroth-order optimization is a worthwhile direction.  

2. The intrinsic-view MSE analysis and the unified sampling perspective are promising.  

3. The paper is generally clearly presented adn the theoretical development is solid.

### Weaknesses
## Major comments

1. The Step 1 of Algorithm 1 requires eigen-decomposition of the metric matrix $A = Q\Lambda Q^\top$, and every sample requires generating Gaussians and computing the acceptance probability $\sqrt{\dfrac{v^\top A^2 v}{\lambda_{\max}}}$. Eigen-decomposition is computationally expensive in high dimensions, and I guess this is why the experiments focus on relatively simple tasks. 

2. Assumption C.3 requires the objective function to have bounded third- and fourth-order derivatives which is a strong condition. As acknowledged by the authors, such a condition is less common in the literature.  The authors emphasized in the appendix that this assumption has also been used by Alimisis et al. (2021, Assumption 1), I fail to identify an equivalent assumption in that reference.

## Minor comments
1. It is suggested to add a short section in the appendix that lists and explains the frequently used notation. Although most symbols are defined at first use, having a centralized symbol table would greatly ease reading when symbols reappear across the paper.  

2. In the Introduction section, it is desirable to provide some examples to help readers understand in which scenarios Riemannian zeroth-order optimization is preferable to conventional (Euclidean) zeroth-order optimization.
 
3.  Concerning experiments, some more details about the settings of hyper-parameters should be provided for the sake of reproducibility.

4. In Theorem 2.3, the term $\dfrac{6}{d} + \dfrac{8}{d}$ looks a bit weird — is there any typo here?

### Questions
1. The paper states that “the rescaling method even leads to divergence for the logistic loss objective.” Is this divergence inevitable (i.e., an inherent failure mode of the rescaling method), or could it be due to unlucky hyperparameter choices?

2. In the geodesically complete case, can the more general framework presented in this paper imply the known optimal bound?

3. Does there exist a class of "structure-preserving" transformations more general than conformal scaling that can simultaneously guarantee geodesic completeness and be more compatible with gradient / estimator perturbations; Is it feasible to use the "structure-preserving" idea as a preprocessing / transform to "straighten" a difficult-to-optimize manifold into an (computationally) equivalent problem that is more favorable for zeroth-order methods?

4. Theorem 2.2 provides an existence result and a constructive proof for the conformal coefficient $h$. How can this construction be implemented in practice?

### Soundness
3

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
This paper considers zeroth-order optimization over Riemannian manifold that are embedded in Euclidean space. Existing notions of zeroth-order derivatives on Riemannian manifold critically rely on the exponential map. Nevertheless, this might be ill-defined when the inherited Euclidean metric is not complete. To address this issue, the authors show that there always exists a different non-Euclidean and structure-preserving metric. Using the new metric, they construct a sampling-based zeroth-order gradient estimator and establish theoretical convergence guarantee for this algorithm.

### Strengths
This paper has a clear structure is quite well-written. From a contribition perspcyive, the idea of using a structure-preserving metric is novel and mathematically elegant when the inherited Euclidean metric is not complete. The following theoretical analysis appears sound and follows standard arguments.

### Weaknesses
However, my major concern lies in the contribution aspect. Although the paper provides an interesting solution, the issue of metric incompleteness is quite specialized and rarely arises in practical Riemannian optimization scenarios. Researchers in this area are typically more interested in improved convergence rates or more efficient algorithms, which this work does not provide. Therefore, I tend to reject the paper due to the limited significance of its contribution.

### Questions
No further questions.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies Riemannian zeroth-order optimization under geodesically incomplete metrics—a setting often overlooked in existing analyses that assume global geodesic completeness. The authors introduce a novel concept of structure-preserving metrics and develops an intrinsic two-point zeroth-order gradient estimator to resolve this problem. Convergence guarantees for Riemannian SGD with this estimator are provided, achieving rates comparable to those in geodesically complete settings.Empirical results on synthetic and mesh optimization tasks validate the theoretical findings.

### Strengths
1. This paper investigated a novel problem setting—— geodesically incomplete.
2. This paper introduces some relative novel conceptions,especially structure-preserving metric and has elegant theoretical contribution.
3. This paper has a great presentation with well writing.

### Weaknesses
It seems that Theorem 2.2 only provide the existence of structure-preserving metric. Can the authors give a guidance to construct such structure-preserving metrics?

### Questions
See "weakness" part.

### Soundness
3

### Presentation
3

### Contribution
3
