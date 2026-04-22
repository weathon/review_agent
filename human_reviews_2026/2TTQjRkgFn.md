# Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6, 8

## Abstract
We propose RAdaGD, a novel family of adaptive gradient descent methods on general Riemannian manifolds. RAdaGD adapts the step size parameter without line search, and includes instances that achieve a non-ergodic convergence guarantee, $f(x_k) - f(x_\star) \le \mathcal{O}(1/k)$, under local geodesic smoothness and generalized geodesic convexity. A core application of RAdaGD is Gaussian Variational Inference, where our method provides the first convergence guarantee in the absence of $L$-smoothness of the target log-density, under additional technical assumptions. We also investigate the empirical performance of RAdaGD in numerical simulations and demonstrate its competitiveness in comparison to existing algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes RAdaGD, a family of adaptive gradient descent algorithms on Riemannian manifolds. The method generalizes Euclidean adaptive step-size rules to the manifold setting, avoiding the need for a known smoothness constant or line search. The authors prove non-ergodic O(1/k) convergence under local geodesic smoothness and generalized geodesic convexity, relaxing the assumption of global smoothness. They apply RAdaGD to Gaussian Variational Inference (GVI) and provide convergence guarantees even when the target log-density is not globally smooth.

### Strengths
The paper makes a novel theoretical contribution by defining local geodesic smoothness and proving that it holds for all twice-differentiable functions on complete manifolds, thereby relaxing the need for global smoothness assumptions. It introduces generalized geodesic convexity as a principled framework for analyzing adaptive Riemannian optimization methods and provides the first adaptive Riemannian gradient algorithm with non-ergodic O(1/k) convergence rates. The application to Gaussian Variational Inference is well motivated and theoretically sound, extending results from the Euclidean to the non-Euclidean setting. Overall, the theoretical analysis is rigorous and carefully adapts adaptive step-size techniques to curved spaces.

### Weaknesses
The paper’s main weaknesses lie in its presentation and empirical scope. The exposition is mathematically dense and at times difficult to follow, with heavy notation and definitions that could be streamlined for clarity. Empirical validation is limited to a single Poisson regression experiment, which does not fully demonstrate the algorithm’s practical benefits or robustness across different manifolds. Comparisons with other Riemannian adaptive or stochastic methods, such as Riemannian Adam, are missing, leaving open questions about relative performance. Some theoretical assumptions, including bounded curvature and known constants, may be hard to verify or implement in practice, and the computational cost of adaptive step-size evaluations is not thoroughly discussed. Overall, while the theory is strong, the work would benefit from clearer exposition and more extensive empirical evidence.

### Questions
1. Can the authors clarify how restrictive the local geodesic smoothness assumption is in practice? For example, are there realistic manifolds or objective functions where it might fail to hold, and how would RAdaGD behave in such cases?

2. As I've understood it, the experiments are limited to Poisson regression on the Bures–Wasserstein manifold. Could the authors provide results or discussion on how RAdaGD performs on other manifolds (e.g., SPD or hyperbolic spaces) or compare it directly to Riemannian Adam or stochastic methods?

3. How sensitive is RAdaGD to the choice of initial step size or curvature estimates in practice, and what is the computational overhead of computing the adaptive step-size parameters compared to standard Riemannian gradient descent?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a adaptive optimization method, RAdaGD, for smooth optimization problems defined on Riemannian manifolds. The authors extend adaptive gradient methods to the non-Euclidean setting under relaxed smoothness and convexity assumptions—specifically, local geodesic smoothness and generalized geodesic convexity—and provide a non-ergodic $\mathcal{O}(1/k)$ convergence rate. The approach is further applied to Gaussian Variational Inference (GVI) problems formulated on the Bures–Wasserstein manifold of Gaussian distributions. However, this paper heavily relies on the algorithms and theories presented in Suh & Ma (2025). Furthermore, several theoretical assumptions and proof steps require clarification or stronger justification.

### Strengths
1.	The authors propose a family of Riemannian adaptive gradient descent algorithms, RAdaGD, which automatically adjusts the step size by approximating the local smoothness parameters. 
2.	The introduction of local geodesic smoothness and generalized geodesic convexity provides a new theoretical framework that generalizes existing smoothness/convexity assumptions.
3.	The application to Gaussian Variational Inference under the Bures–Wasserstein geometry is well motivated and could be useful in practice.

### Weaknesses
1.	The main theoretical results (e.g., Theorem 4.1) rely on an upper bound $L \geq L_k$ for $k \geq 0$. While the authors address the existence of such a bound, this assumption obscures the distinction between global and local smoothness, thereby reducing the theoretical appeal of the proposed framework. 
2.	The constants in the $\mathcal O{(1/k)}$ rate (e.g., dependence on $r, L, R, \bar \zeta$) should be discussed. If these constants can be large in curved manifolds, the theoretical rate may not be meaningful in practice.
3.	Theorem 3.3 claims that all $C^2$ functions are locally geodesically smooth. While intuitively true, the proof relies on the boundedness of the Riemannian Hessian. The manuscript should more explicitly state the curvature-dependence of this claim.
4.	The authors mentioned another adaptive gradient descent algorithm (Ansari-O¨ nnestam & Malitsky (2025)) on a Riemannian manifold in the previous text , but did not conduct a comparison in the experiments. The authors claim that the paper was withdrawn from arXiv on September 15, 2025, yet its V1 remains accessible for viewing.
5.	In Corollary 5.4, the author claims that $\bar \zeta =1$, based on the assumption in Corollary 4.6 that $\bar \zeta =sup_{x \in N} \zeta(d(x,x_*))$, but I have not found a proof for $\zeta = 1$.

### Questions
1.	The main theoretical results (e.g., Theorem 4.1) rely on an upper bound $L \geq L_k$ for $k \geq 0$. While the authors address the existence of such a bound, this assumption obscures the distinction between global and local smoothness, thereby reducing the theoretical appeal of the proposed framework. 
2.	The constants in the $\mathcal O{(1/k)}$ rate (e.g., dependence on $r, L, R, \bar \zeta$) should be discussed. If these constants can be large in curved manifolds, the theoretical rate may not be meaningful in practice.
3.	Theorem 3.3 claims that all $C^2$ functions are locally geodesically smooth. While intuitively true, the proof relies on the boundedness of the Riemannian Hessian. The manuscript should more explicitly state the curvature-dependence of this claim.
4.	The authors mentioned another adaptive gradient descent algorithm (Ansari-O¨ nnestam & Malitsky (2025)) on a Riemannian manifold in the previous text , but did not conduct a comparison in the experiments. The authors claim that the paper was withdrawn from arXiv on September 15, 2025, yet its V1 remains accessible for viewing.
5.	In Corollary 5.4, the author claims that $\bar \zeta =1$, based on the assumption in Corollary 4.6 that $\bar \zeta =sup_{x \in N} \zeta(d(x,x_*))$, but I have not found a proof for $\zeta = 1$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RAdaGD, a novel family of adaptive gradient descent methods on Riemannian manifolds. The method adapts the step size automatically—without explicit line search—by estimating a local smoothness parameter, achieving the first known non-ergodic convergence rate \mathcal{O}(1/k) under local geodesic smoothness and generalized geodesic convexity. The authors extend adaptive optimization concepts from the Euclidean setting (e.g., Suh & Ma, 2025) to Riemannian manifolds via nontrivial generalizations of Lyapunov analyses, parallel transport arguments, and curvature-dependent scaling factors.
A notable application is given to Gaussian Variational Inference (GVI), where the method provides the first convergence guarantee without assuming global L-smoothness of the target log-density. Empirical experiments confirm the theoretical results and show competitive or superior performance compared to existing algorithms.

### Strengths
The paper is rigorous, technically elegant, and addresses a clear theoretical gap in Riemannian optimization—namely, the lack of adaptive step-size methods with formal convergence guarantees. The introduction of local geodesic smoothness as a relaxation of traditional L-smoothness is both natural and powerful, broadening the class of admissible objective functions while preserving tractability.
The convergence proofs are carefully executed and extend nontrivial elements from Euclidean adaptive methods to manifold geometry (e.g., dealing with exponential/log maps, curvature terms, and parallel transport). The theoretical guarantees (non-ergodic \mathcal{O}(1/k) rate) are significant, and the paper connects the framework elegantly to practical applications in Bayesian inference. The exposition is also exemplary—clear definitions, consistent notation, and intuitive motivation for the introduced assumptions.

### Weaknesses
While the paper fills a critical theoretical gap, I have a few questions regarding its positioning and broader implications:
	1.	Relation to recent Riemannian adaptive methods.
The paper positions itself as the first of its kind for adaptive algorithms on manifolds. However, a recent arXiv preprint (link https://arxiv.org/abs/2306.16617) also studies adaptive Riemannian optimization with curvature-aware preconditioning and adaptive metric scaling. Could the authors clarify whether RAdaGD fundamentally differs from these approaches (e.g., in the use of local smoothness estimation versus curvature-adapted preconditioning)? Are the convergence guarantees directly comparable, or do they apply to distinct settings (e.g., deterministic vs. stochastic, local vs. global smoothness)?
	2.	Tightness of assumptions.
The local geodesic smoothness assumption, while weaker than L-smoothness, still presupposes control of curvature-dependent Lipschitz constants L_K. Is it possible to relax this further, for example to Hölder-smooth functions, or to obtain adaptive control of curvature-dependent constants during optimization?
	3.	Empirical scope.
The experiments are clean but limited to Gaussian Variational Inference tasks. Given the generality of the method, could the framework extend naturally to Riemannian neural optimization or hyperbolic embeddings (where curvature varies strongly)? A short experiment in such a context could enhance the impact.
	4.	Adaptivity mechanism.
The paper introduces adaptive sequences A_k, B_k, \tilde{B}_k, but it would be helpful to clarify whether the adaptivity primarily tracks local smoothness, curvature, or both. In other words, is the method “geometry-aware” in practice, or purely driven by gradient variance?

Overall, these are minor concerns—the theoretical contribution is strong and original.

### Questions
Please comment about the weaknesses

### Soundness
2

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
3

### Summary
The paper proposes RAdaGD, a line-search-free family of adaptive gradient descent methods on general Riemannian manifolds. The method adapts the stepsize by estimating a local smoothness quantity from two consecutive iterates and gradients, and it is analyzed under local geodesic smoothness and generalized geodesic convexity, which are weaker than the usual global geodesic $L$-smoothness. The authors prove a non-ergodic convergence rate
$$
f(x_k)-f(x^\star) \le O(1/k)
$$
matching classical Riemannian gradient descent with known $L$, and also give bounds on $|\operatorname{Grad} f(x_k)|^2$. A technically involved part of the analysis is the treatment of curvature via the parameter $\zeta(\cdot)$ and the triad $(A_k,B_k,\tilde B_k)$ in the Lyapunov function. As a main application, the paper shows how to instantiate RAdaGD for Gaussian variational inference on the Bures–Wasserstein manifold, obtaining (to the authors’ knowledge) the first convergence guarantee for GVI without assuming global $L$-smoothness of the log-density, under an additional eigenvalue-boundedness condition. Experiments on Poisson regression GVI and negatively curved spaces demonstrate competitiveness with prior Riemannian and BW-based optimizers.

### Strengths
* A concrete, implementable Riemannian adaptive gradient method (RAdaGD) that removes line search and still attains $O(1/k)$ in function value.
* Relaxation from global geodesic $L$-smoothness to local geodesic smoothness, supported by the statement that every $C^2$ function on a complete Riemannian manifold is locally geodesically smooth; this enlarges the class of admissible objectives.
* Careful curvature-aware analysis via the $\zeta(\cdot)$ parameter and the triplet $(A_k,B_k,\tilde B_k)$, including extensions to unbounded or unknown curvature.
* Application to Gaussian variational inference on the Bures–Wasserstein manifold, yielding (under an eigenvalue assumption) what appears to be the first convergence guarantee for non-$L$-smooth GVI.
* Empirical results on Poisson regression GVI that illustrate the claimed setting where prior methods lack guarantees but RAdaGD still converges.

### Weaknesses
* The core convergence theorems require generalized geodesic convexity, which is stronger than plain geodesic convexity and may limit applicability on general manifolds or nonconvex statistical objectives.
* The method still assumes access to exponential maps and parallel transport on the manifold; this is reasonable for BW$(\mathbb R^n)$ but can be a practical limitation on more general manifolds where only retractions and vector transports are available.
* The step-size rule in Algorithm 1 depends on quantities that must be computed from two consecutive gradients and parallel transports; while explicit, it is more involved than standard RGD and the paper could more clearly quantify this overhead.
* Experiments are focused on a few GVI scenarios and do not yet explore larger-scale or stochastic-gradient settings, which are natural for VI in practice.
* The GVI convergence result needs an additional uniform lower bound on the covariance eigenvalues, which is plausible but not guaranteed by the algorithm itself.

### Questions
1. Can the authors clarify whether the adaptive rule for $s_{k+1}$ can be made compatible with retractions/vector transports (as opposed to exact $\exp$ and parallel transport) while preserving the rates, possibly with higher-order error terms?
2. In the BW-GVI application, the extra step-size clipping in equation (17) depends on the largest eigenvalue of an expectation involving $\nabla^2 V(X)$. How is this implemented in practice when only Monte Carlo estimates of expectations are available, and how sensitive is convergence to this clipping?
3. The convergence proof in Section 4 relies on the compactness/boundedness of the iterates (via the Lyapunov argument). Is there an example of a nonnegatively curved manifold and locally geodesically smooth, generalized geodesically convex function where RAdaGD would fail without this boundedness step?
4. For the Poisson regression GVI example, can the authors provide a small ablation showing RAdaGD vs. a Riemannian method with backtracking line search, to separate the benefit of adaptivity from the benefit of avoiding line search?
5. The paper mentions possible stochastic variants. Are there obstacles in controlling the estimate of $L_{k+1}$ in (5) under stochastic gradients, or is it mainly a matter of concentration bounds?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose adaptive gradient descent methods on Riemannian manifolds. Following the approach introduced in Suh and Ma (2025), they show that the iterates of their algorithm converge in objective value at a rate of $\mathcal{O}(1/k)$ when the potential function $V$ is locally geodesically smooth and generalized geodesically convex. They apply this framework to Gaussian Variational Inference (GVI), establishing provable computational guarantees while relaxing the standard $L$-smoothness assumption.

### Strengths
Overall, the paper is carefully written and mathematically rigorous. It extends the work of Suh and Ma (2025) to the setting of optimization on Riemannian manifolds and establishes convergence guarantees for the proposed algorithm. In particular, this work provides a algorithm with convergence results for Gaussian Variational Inference (GVI) even when the target distribution is not log-smooth.

### Weaknesses
The main idea appears to be a rather direct extension of Suh and Ma (2025) to the Riemannian setting, with additional technical complications arising from the non-Euclidean geometry. In particular, Algorithm 1 in the paper is largely a manifold-based translation of Algorithm 2 in Suh and Ma (2025). The comparison with the reference paper is somewhat limited and does not sufficiently emphasize the novel challenges or technical difficulties encountered in extending the analysis to the manifold case.
In Section 4.1, the authors mention that the constants $\tilde B_k$ are introduced to handle the terms $\zeta_k$, which are specific to the manifold setting (since $\zeta_k \equiv 1$ in Euclidean geometry). A more detailed explanation and comparison in this direction would be appreciated—especially to clarify how these constants affect the algorithm’s behavior and the convergence analysis.

### Questions
1. The citation to \emph{Ansari-Onnestam \& Malitsky (2025)} should be removed, as that paper has been withdrawn.  

2. Some discussion should be included for non-convex objectives, clarifying whether the proposed methods or convergence results extend in any way beyond the geodesically convex case.  

3. Since only geodesic convexity is assumed, there may exist multiple minimizers. It would be helpful to clarify whether the choice of $x^*$ in the statements of results (e.g., Theorem~4.1) affects the conclusions or constants.  

4. In Section 4.2, the authors address the case of manifolds with negative sectional curvature and unbounded $\bar{\zeta}$ by assuming that $d(x_0, x_*)$ is bounded above (in Corollary 4.7). This condition appears vacuously true, since otherwise the constant $R$ defined in Theorem 4.1 would be infinite, implying that the smoothness parameter $L$ should be infinite as well. It is unclear what substantive difference is between Corollary 4.7 here and Corollary 4.6 in the previous section. In particular, as suggested by Corollary 4.7, what happens if we set $\tilde B_k = B_k + \bar{\zeta}_0$?  

5. Although the proposed method is a direct extension of Suh and Ma (2025)  to the manifold case, more intuition and explanation should be provided regarding Algorithm~1. For instance, what are the respective roles of $A_k$, $B_k$, and $\tilde B_k$ in the algorithm? Is there an intuitive interpretation of these quantities or of the update mechanism?  

6. In line 326, the sentence  
    ``The following corollary states that, as a trade-off, if $\zeta$ is known to be bounded, we can still achieve the same asymptotic convergence rate without any prior knowledge of $\zeta$.''
is unclear. What does $\zeta$ refer to here? Is it the function defined in equation~(2), or is this a typo?  

7. In Section E.1.3, the proofs of parts (iii) and (iv) are provided, although these statements do not appear in Proposition 5.3. It would be clearer to reorganize the proof into Steps 1--4 instead. In Lemma E.1 (line 1832), the phrase should read:  
    ``is a compact set in ${\rm BW}(\mathbb{R}^n)$.''

### Soundness
4

### Presentation
3

### Contribution
3
