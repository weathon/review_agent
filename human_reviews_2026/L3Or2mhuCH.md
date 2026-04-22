# A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Nonsmooth composite optimization with orthogonality constraints has a wide range of applications in statistical learning and data science. However, this problem is challenging due to its nonsmooth objective and computationally expensive, non-convex constraints. In this paper, we propose a new approach called \textbf{OBCD}, which leverages Block Coordinate Descent to address these challenges. \textbf{OBCD} is a feasible method with a small computational footprint. In each iteration, it updates $k$ rows of the solution matrix, where $k \geq 2$, by globally solving a small nonsmooth optimization problem under orthogonality constraints. We prove that the limiting points of \textbf{OBCD}, referred to as (global) block-$k$ stationary points, offer stronger optimality than standard critical points. Furthermore, we show that \textbf{OBCD} converges to $\epsilon$-block-$k$ stationary points with an iteration complexity of $\mathcal{O}(1/\epsilon)$. Additionally, under the Kurdyka-Lojasiewicz (KL) inequality, we establish the non-ergodic convergence rate of \textbf{OBCD}. We also demonstrate how novel breakpoint search methods can be used to solve the subproblem in \textbf{OBCD}. Empirical results show that our approach consistently outperforms existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
A new approach called OBCD is proposed for minimizing the sum of a smooth and a nonsmooth convex functions over the nonconvex set of orthogonal matrices.

### Strengths
Optimization under orthogonality constraints is an important topic with many applications. The paper is well written. My expertise of the topic is limited, though. I know about projection methods, as there is a surge of interest for orthogonalization recently in the context of the Muon optimizer, see for instance Grishina et al. "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials". The proposed method maintains feasibility throughout the process, which is a different approach than relying on approximate orthogonalization.

### Weaknesses
I don't see limitations.

### Questions
N/A

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
The authors propose an algorithm for minimizing the sum of a smooth and nonsmooth function on the Stiefel manifold.
Relevant applications include sparse Principal Component Analysis (PCA) with l1 norm and l0 norm regularization.
The algorithm is a descent method, and features a block coordinate update, which requires computing the corresponding coordinate block of gradient, as opposed to the full gradient.
In theory, the algorithm converges in expectation to a relevant point with classical complexity (Th. 4.2), and under additional assumptions (including continuity of $h$, which rules the important l0 norm example and leaves l1 norm) the last iterate converges to a relevant point with classical complexity. We note that the notion of "relevant point" is slightly stronger than the usual notion of critical point (0 in limiting subdifferential).
In practice, the algorithm performs well on PCA with l0 regularization.

### Strengths
- the submission is clearly written,
- the problem is motivated by a relevant application (sparse PCA with l0 and l1 norm regularization),
- the algorithm is derived and formulated in a rigorous and clear way (up to one detail, discussed below),
- the theoretical analysis presents a notion of stationary points (BS$k$-points) that is stronger than the customary critical point (zero in subdifferential), and a proof that the algorithm converges to such points on average. I am quite familiar with composite optimization, but less so on block coordinate schemes so this idea may be common in that field.
- the algorithm combines known ideas (block coordinate descent, majorization minimization) suitably, and successfully applies these ideas to minimization with l0 norm regularization. As far as I know, this task is highly challenging, and few methods for it are available even in the unconstrained case.

### Weaknesses
Here is a list of points that prevent me from providing a better assessment to the submission.
- The nonsmooth function $h$ has strong, restrictive assumptions. This severely limits the applicability of the method.
  + Assumption (ii): $h$ is assumed to be coordinate separable, with the same expression for each coordinate.
  + Assumption (iii) on function $h$ is shown to be tractable in theory for three values of $h$ only (the $\ell1$ norm, the indicator of a polyhedron, and the $\ell0$ norm when $k=2$).
- l. 122-128: I disagree with part of the positioning relative with the literature:
  + contrary to what point (ii) implies, I think that the proposed method is not applicable to general nonsmooth composite problems. Indeed it relies on an assumption on the nonsmooth function, which is arguably restrictive (see point below) but certainly not valid for "general nonsmooth composite problems".
  + point (iii) asserts that it is a "limitation" for methods to be "infeasible", that is to generate iterates that are only asymptotically feasible. Yet, infeasible methods are accepted for solving large-scale optimization problems under general constraints (e.g. with the Augmented Lagrangian method), and Stiefel manifold constraints specifically (e.g. with the recent Landing methods, see next point).
  + point (iv), "they often lack rigorous convergence guarantees" should be more precise on the type of convergence guarantees that previous methods fail (for instance: convergence of average, last iterate, complexity guarantee).
- l. 78-107: Literature review misses one recent line of work of so-called "landing methods" for optimization on Stiefel manifolds: [1-4]. These methods also target large-scale minimization on the Stiefel manifold; the paper should discuss these methods.
- Algorithm 1: The algorithm seems to depend on a parameter $\alpha$ (l. 194, Lemma 2.3, Th. 4.2, among other occurrences), yet this parameter does not appear in the algorithm statement, and it is unclear what condition $\alpha$ should meet so that the convergence guarantees hold, and particularly so the sufficient decrease condition of Theorem 4.2(a).
- l. 374: the condition $\| \partial h(X) \|_{sp}<l_h$ is ill-posed: the "sp" norm of a set is not defined. Besides, this condition may appear more natural if it were implied by Lipschitz continuity of $h$.
- l. 392: the assumption that $F_i$ is a KL function is not discussed at all. It is thus unclear whether it holds for any of the application cases, and the three discussed $h$ functions.
- l. 395: Proposition 4.8, from previous work, is stated without providing a precise reference, that is, both the paper and statement references.
- l. 404 & 412: "the continuity assumption made in lemma 4.4" is unclear. To what condition does this sentence refer to exactly? Maybe writing an additional assumption for this would clarify the situation.
- Experiments: it is unclear whether the three baselines generate feasible iterates. In addition, these three baselines involve operator splitting. Such methods usually involve a reformulation of the problem with additional constraints, which are satisfied asymptotically only. Finally, it is not clear whether the three baselines have convergence guarantees on  the problem with $h = \| \cdot \|_{0}$, which is discontinuous; this aspect conditions the interpretation of the whole experimental section.

References:
- [1] Goyens, Absil & Feppon (2026) Geometric Design of the Tangent Term in Landing Algorithms for Orthogonality Constraints, Springer Nature Switzerland.
- [2] Ablin & Peyré (2022) Fast and Accurate Optimization on the Orthogonal Manifold without Retraction, PMLR.
- [3] Vary, Ablin & Gao et al. (2024) Optimization without Retraction on the Random Generalized Stiefel Manifold, PMLR.
- [4] Gao, Vary & Ablin et al. (2022) Optimization Flows Landing on the Stiefel Manifold$\star$, IFAC-PapersOnLine.


Minor points, that do not impact the assessment of the paper:
- l. 35: clarity would improve by mentioning explicitly that $f$ is assumed to be differentiable and $H$-smooth
- l. 39, footnote: Is $f$ assumed to have expression $1/2 \|X\|_{H}^{2}$ on the whole paper?
- l. 40 & 296: what are the definitions of $F$ "closed", and $F$ lower semicontinuous? How do they differ?
- l. 230: is the
- l. 305: The writing of Definition 3.3 is confusing: the sentence "Furthermore $\lambda \in [\partial F(X)]^\top X$" can read as an additional condition for a point $X$ to be critical, but the notion  of criticality does not involve $\lambda$.
- l. 313-318: what is the purpose of this paragraph?

### Questions
Any comment and detail on each of the listed weak points is welcome.

Below are some suggestions in the form of questions; any answer on the following questions is welcome, but no answer is also fine.
- connection to (variable metric) proximal gradient method? That would connect assumption (iii) to the notion of "prox-friendly" nonsmooth function $h$, standard in nonsmooth optimization. That may also help and connect
- Th. 3.6: globally optimal points are BS2-points. Are they also BSk points, for $2 \le k \le n-1$?
- Assumption (iii) is reminiscent of the  might appear more natural if connected to the proximal gradient operator.
- optimization methods on problems with l0-norm regularization usually face the combinatorial difficulty of the l0 norm in some way. I am surprised that it doesn't show in your analysis. Do you have some intuition to share on this? I am also surprised that the complexity of OBCD with parameter $k$ in Theorem 4.2 does not depend on $k$. Again, do you have some intuition to share on this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study the problem of minimizing a nonconvex nonsmooth function $F(X)$ over the space of $n \times r$ orthogonal matrices (the Stiefel manifold $St(n,r)$). This is a challenging class of problems with wide applications. Common approaches for such problems include projection-based methods, Riemannian methods using tangent-space surrogates and retractions, or Block Majorization-Minimization (BMM) methods that iteratively minimize a surrogate directly on the manifold.

The authors propose an approach, **OBCD**, which falls under the BCD/BMM umbrella but with a novel and distinct design. Instead of updating a manifold block $X_k$ directly, the method subsamples $k$ rows and finds a small $k \times k$ orthogonal matrix $V$ that *transforms* this block. This "row-wise" update ($X^{t+1}(\mathcal{B},:) \leftarrow \overline{V}^{t}X^{t}(\mathcal{B},:)$) is a key contribution, as it is inherently feasible and avoids the standard tangent-space/retraction machinery.

While the general BCD/BMM idea is known, this paper's novelty lies in the specifics of its framework for the Stiefel manifold:

1.  It defines a new optimality condition, the **"block-k stationary point" ($BS_k$-point)**.  Theorem 3.6 shows that this condition is **stronger** than the standard critical point condition. The authors justify this by showing their $k=2$ solver uses both rotations and reflections, allowing it to escape suboptimal points.

2.  It provides a **constructive and exact solver** for its nonsmooth subproblem. Appendix B introduces a novel "Breakpoint Searching Method (BSM)" that finds the *exact global solution* for the $k=2$ subproblem with $l_0$, $l_1$, or non-negativity regularizers. This is a non-trivial technical result that provides a solid foundation for the algorithm.

On this theoretical foundation, the authors derive a comprehensive convergence analysis, including an $\mathcal{O}(1/\epsilon)$ iteration complexity for an $\epsilon$-$BS_k$-point (Theorem 4.2) and a full non-ergodic (last-iterate) convergence analysis under the KL property (Theorem 4.10). The presentation is clear, with assumptions formally stated, the algorithm well-defined, and the theoretical claims rigorously established.

### Strengths
The authors study the problem of minimizing a nonconvex nonsmooth function $F(X)$ over the space of $n \times r$ orthogonal matrices (the Stiefel manifold $St(n,r)$). This is a challenging class of problems with wide applications. Common approaches for such problems include projection-based methods, Riemannian methods using tangent-space surrogates and retractions, or Block Majorization-Minimization (BMM) methods that iteratively minimize a surrogate directly on the manifold.

The authors propose an approach, **OBCD**, which falls under the BCD/BMM umbrella but with a novel and distinct design. Instead of updating a manifold block $X_k$ directly, the method subsamples $k$ rows and finds a small $k \times k$ orthogonal matrix $V$ that *transforms* this block. This "row-wise" update ($X^{t+1}(\mathcal{B},:) \leftarrow \overline{V}^{t}X^{t}(\mathcal{B},:)$) is a key contribution, as it is inherently feasible and avoids the standard tangent-space/retraction machinery.

While the general BCD/BMM idea is known, this paper's novelty lies in the specifics of its framework for the Stiefel manifold:

1.  It defines a new optimality condition, the **"block-k stationary point" ($BS_k$-point)**. This is a significant contribution, as Theorem 3.6 proves this condition is **provably stronger** than the standard critical point condition. The authors justify this by showing their $k=2$ solver uses both rotations and reflections, allowing it to escape suboptimal points.

2.  It provides a **constructive and exact solver** for its nonsmooth subproblem. Appendix B introduces a novel "Breakpoint Searching Method (BSM)" that finds the *exact global solution* for the $k=2$ subproblem with $l_0$, $l_1$, or non-negativity regularizers. This is a non-trivial technical result that provides a solid foundation for the algorithm.

On this theoretical foundation, the authors derive a comprehensive convergence analysis, including an $\mathcal{O}(1/\epsilon)$ iteration complexity for an $\epsilon$-$BS_k$-point (Theorem 4.2) and a full non-ergodic (last-iterate) convergence analysis under the KL property (Theorem 4.10). The presentation is clear, with assumptions formally stated, the algorithm well-defined, and the theoretical claims rigorously established.

### Weaknesses
### 1. Unclear Practicality for the `k > 2` Case

A primary weakness is the gap between the well-analyzed $k=2$ case and the general $k > 2$ case. The paper's core assumption (Asm-iii) is that the $k \times k$ nonsmooth subproblem can be solved "exactly and efficiently." The authors provide an impressive, detailed proof of this for $k=2$ using their novel Breakpoint Searching Method (Appendix B).

However, for $k > 2$, this assumption is highly questionable. Solving a general $k \times k$ nonsmooth composite problem over the Stiefel manifold $St(k,k)$ is not a trivial task, and the paper provides no algorithm or justification for it.

The paper *does* offer a fallback in Algorithm 1, suggesting one can "Alternatively, find a local solution $\overline{V}^t$ such that $\mathcal{K}(\overline{V}^{t};X^{t},B)\le\mathcal{K}(I_{k};X^{t},B)$". But the paper provides no discussion on *how* to find such a local solution. For a general nonsmooth, nonconvex subproblem, even finding a point that guarantees this simple descent from the identity matrix ($I_k$) is a non-trivial problem in itself. Without a proposed method, the practical application of OBCD for block sizes $k > 2$ remains unclear.

### 2. Overstated Novelty Claim

In the "Summary" of the related work (Section 1.2), the paper claims: "To our knowledge, this represents the first application of BCD methods to solve nonsmooth composite optimization problems under orthogonality constraints...". This claim appears to be incorrect. The paper's own literature review (Section 1.2, under "Minimizing Nonsmooth Functions...") cites existing work on "Block Majorization Minimization (BMM) on Riemannian manifolds," which directly addresses nonsmooth problems on manifolds, including the Stiefel manifold. This contradiction in the paper's own text weakens the positioning of its contribution. Rather than claiming to be the first of its kind, the authors could emphasize the tailored approach for solving problems on Stiefel manifold efficiently. 

### 3. Mismatch Between Theory and Experiments

There is a disconnect between the theoretical contributions and the experimental validation. The paper provides a strong theoretical justification (in Appendix B) for handling $l_0$, $l_1$, and non-negativity constraints. However, the experiments in Section 5 are conducted *only* on $L_0$-norm-based SPCA. This is a missed opportunity to not validate the new, non-trivial solvers for the $l_1$ and non-negativity constraint problems, which would have made the experimental section much more comprehensive and demonstrated the full power of the proposed subproblem solvers.

### Minor comments

While the paper's literature review is adequate, the authors may consider including the following recent paper on Euclidean BMM as part of the BCD section: 

Hanbaek Lyu and Yuchen Li, “Block majorization-minimization with diminishing radius for constrained nonconvex optimization.”  SIAM Journal on Optimization, Vol. 35, Iss. 2 (2025)

### Questions
1. The authors mention in L120 that [Gao et al. 2019] studies a similar problem with columewise updates, whereas the proposed method is rowwise. Besides these algorithmic design choices, what are the important differences? Pros/cons on the computational cost? Theoretical property? 

2.  The definition of the "block-k stationary point" ($BS_k$-point) (Definition 3.5) is based on $I_k$ being the *global minimizer* of the subproblem. If one only finds a local solution $\overline{V}^t \neq I_k$ that satisfies the descent condition (as suggested in Algorithm 1) and the algorithm converges, does the limit point have any meaningful theoretical properties? Does the hierarchy in Theorem 3.6 still hold, or is the $BS_k$-point definition fundamentally tied to the (impractical) global solution of the subproblem for $k > 2$?

### Soundness
3

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
4

### Summary
The paper proposes a block coordinate descent (OBCD) algorithm for nonsmooth composite optimization on the Stiefel manifold. The method updates $k$ blocks at each iteration while preserving orthogonality, and achieves convergence to a stationary point under suitable assumptions. The authors also design an exact solver for the $k=2$ subproblem when the nonsmooth term $h$ is coordinate-wise separable, by reducing it to a one-dimensional breakpoint-search problem. Theoretical results include global convergence and iteration complexity bounds, and experiments on sparse PCA show promising performance.

### Strengths
* They propose a first BCD method for solving nonsmooth composite optimization problems under orthogonality constraints.
* The breakpoint-search solver for the $k=2$ subproblem is elegant and provides an exact and efficient solution when $h$ is separable.
* The experimental results on sparse PCA are convincing.

### Weaknesses
* (Assumption 3 (exact subproblem solution).) This assumption is quite restrictive, as exact solutions are only possible when $h$ is coordinate-wise separable (e.g., $\ell_0$ or $\ell_1$ penalties). For general non-separable regularizers such as the nuclear norm, this assumption may not hold. Could the authors discuss (i) whether the framework can be extended to non-separable $h$, and (ii) whether their convergence analysis remains valid if each subproblem is solved only approximately or to a stationary point? A relaxed ``inexact subproblem'' assumption might make the method more generally applicable.
* (Bounded subgradient assumption (Lemma 4.4).) I think the assumption $||\partial h(X)||_{sp} \le \ell_h$ appears problematic for $h(X)=||X||_0$ (please correct me if I am wrong). However, you use the $\ell_0$-norm in your experiments. It would be better justify this assumption. 
* (Relation to prior work ([1]).) I think problem (1) is a special case of [1] when $H=L_f \mathcal{I}_{nr}$. It would strengthen the contribution to explain clearly how this work differs from and improves upon [1].
* (Experiment) Would the authors also include experimental comparisons with the method proposed in [1]? Such a comparison would better demonstrate the advantages of the proposed OBCD algorithm.

[1] Cheung, Andy Yat-Ming, et al. "Randomized Submanifold Subgradient Method for Optimization over Stiefel Manifolds." arXiv preprint arXiv:2409.01770 (2024).

### Questions
Please see above.

### Soundness
3

### Presentation
2

### Contribution
3
