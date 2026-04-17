# A Practical Descent Method for Singular Value Decomposition

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Singular Value Decomposition (SVD) is a long-established technique, with most existing methods relying on matrix-based formulations. However, matrix operations are relatively less friendly to parallelization and distributed computation compared to descent-based methods, motivating the need for alternative approaches. Descent-based methods offer a promising direction, yet existing ones such as Riemannian gradient descent suffer from inefficiency due to the need for repeated projections onto nonlinear manifolds.
In this work, we introduce a novel descent method for SVD grounded in a primal–dual reformulation. Specifically, we construct a least-squares primal problem whose dual corresponds to the SVD. We show that (i) the non-zero KKT solutions of the primal problem yield the singular vectors of the matrix, and (ii) inexact singular value estimation still ensures bounded reconstruction error. Building on these results, we propose an iterative descent-based algorithm, Des-SVD, along with scalable variants leveraging random sampling and parallelization.
Extensive experiments demonstrate that Des-SVD achieves significantly higher computational efficiency compared to prior descent methods, while remaining competitive with matrix-based algorithms. Our implementation is publicly available at https://anonymous.4open.science/r/Descent-SVD-method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a practical descent-based method for computing the Singular Value Decomposition (SVD) that avoids traditional matrix-centric pipelines and costly manifold projections used by Riemannian methods. Building on Suykens (2016), the authors formulate a primal least-squares problem whose dual corresponds to SVD and derive KKT conditions that bridge the two spaces. The key contributions are: 1) Showing that when the regularization $\gamma$ equals the reciprocal of a singular value $s$, any non-zero KKT solution can be normalized to yield the corresponding singular vectors, with orthogonality arising naturally. 2) Proving that if $\gamma$ is incorrect (i.e., not $1/s$), the KKT matrix is full-rank so the only feasible descent direction at initialization leads to a zero solution—highlighting the need for accurate or sufficiently accurate singular value estimates. 3) Establishing that inexact singular value estimates suffice under explicit reconstruction tolerances $\epsilon_1$, $\epsilon_2$, yielding bounds on allowable $\gamma$-error ($T_{\mathrm{err}}$). This enables the use of fast eigenvalue estimators (Rayleigh quotient iteration). 4) Designing Des-SVD: a Newton-based descent algorithm that (i) estimates singular values, (ii) solves the primal KKT system to obtain dual variables, and (iii) normalizes them to get singular vectors. The method supports parallelization (each singular value independently) and randomized subspace reduction, giving overall complexity comparable in order to Lanczos ($O(k^3)$ in the reduced $k$-space).

Empirically, Des-SVD is much faster than Riemannian gradient descent and competitive with matrix-based methods (Lanczos, randomized + Jacobi) on low-rank matrices, grayscale/RGB images, random large matrices, and challenging spectra (tight gaps, large condition numbers). The authors report stable orthogonality and reconstruction accuracy across settings.

### Strengths
Originality: 1) A clear primal–dual route to SVD that avoids manifold projections, built upon but extending Suykens (2016) with a practical path from primal KKT solutions to SVD factors. 2) Theoretical identification of when the KKT system admits non-zero solutions ($\gamma=1/s$) and when it collapses (incorrect $\gamma$), plus a tolerance-bound analysis that legitimizes inexact singular value estimation. This is a neat, useful bridge between optimization feasibility and spectral approximation. 3) The integration of randomized subspace methods and process-level parallelization into a descent framework is well motivated and practically relevant.

Quality: 1) Theoretical results are clear and aligned with algorithmic design: (i) non-zero KKT solutions imply orthogonal singular vectors after normalization; (ii) rank analysis of the KKT matrix; (iii) error tolerance $T_{\mathrm{err}}$ tied to reconstruction budgets. The proofs are outlined in the main text with appendices for details. 2) Algorithmic choices are coherent: Newton’s method for the small KKT systems after sampling; Rayleigh quotient iteration for singular value estimates; per-singular-value parallelization. 3) Empirical evaluation covers low-rank matrices, images, random large matrices, and special spectra, consistently comparing against representative baselines (Riemannian GD, Lanczos, randomized SVD + Jacobi).

Clarity: 1) The narrative clearly states the obstacle with non-convexity and how the proposed normalization and $\gamma$-selection resolve it. The role of $\gamma$ is emphasized repeatedly with proofs to support intuition. 2) Pseudocode for core routines (Algorithm 1–3) and method variants (refined, parallel, randomized) helps reproducibility.

Significance: 1) Descent-based SVD that is competitive with classical matrix methods—while being naturally parallel and sampling-friendly—addresses an important gap for large-scale, distributed, or privacy-conscious settings where matrix factorization pipelines are awkward. 2) The tolerance-based theory provides a practical recipe for integrating fast spectral estimation within a descent solver, which could influence future work in primal–dual spectral algorithms and distributed SVD/EVD.

### Weaknesses
Dependence on singular value estimation: While Theorem 3.2 justifies inexact $\gamma$ within $T_{\mathrm{err}}$, the practical tightness of $T_{\mathrm{err}}$ and its dependence on data norms $(\sum_i\|D^{T}x_i\|^2, \sum_{j}\|y_j\|^2)$ may be conservative. The paper does not quantify how often Rayleigh iteration meets this tolerance in practice, how many iterations it requires for various spectra, or how sensitive overall runtime is to this stage.

Role and construction of ${D}$ (feature mapping): The feature map uses a matrix ${D}$ satisfying ${ADA} = {A}$. The paper does not discuss concrete choices for $D$, their computational cost, or sensitivity. If $D\approx I$ is intended in many cases, clarify; if not, provide guidance on constructing $D$ efficiently and its effect on $\Phi, \Psi$ conditioning and $T_{\mathrm{err}}$.

Complexity and scaling details: The stated $O((6k)^3)$ complexity after reduction is appealing, but a more granular cost breakdown would help: cost of randomized subspace iteration, Rayleigh quotient iterations, number of Newton steps per singular vector, linear solver costs for the KKT system, and communication in parallel execution. 

Numerical robustness and stopping criteria: The descent loop stops when $\|J\|^2<\epsilon$, but there is limited discussion on i) Conditioning of the KKT system and necessity of regularization or preconditioning; ii) Line search strategy specifics and safeguards against stagnation; iii) How normalization interacts with Newton steps across iterations (e.g., re-scaling $e$ and $v$ each iteration can alter curvature; is this accounted for in convergence guarantees?).

Comparisons and breadth: 1) The baseline “randomized + Jacobi” is reasonable, but state-of-the-art randomized SVD implementations often avoid full Jacobi at the end. Including a power-iteration or subspace-iteration-based RSVD with small dense SVD on the core would be informative. 2) GPU or multi-node settings are hinted as strengths; however, all experiments are CPU-only. A small-scale demonstration of parallel speedups or scalability with increasing number of cores/nodes would better support the parallelization claims.

Theoretical scope: 1) The non-convex landscape: the paper shows that correct $\gamma$ enables non-zero KKT solutions aligned with singular vectors, but it does not analyze basin of attraction, global convergence, or robustness to noise/perturbations beyond $T_{\mathrm{err}}$. A brief discussion of potential spurious stationary points and how the algorithm avoids them (beyond $\gamma$ correctness) would help.

### Questions
Singular value estimation and $T_{\mathrm{err}}$: 1.  How tight is $T_{\mathrm{err}}$ in practice? Can you report empirical $T_{\mathrm{err}}$ values alongside achieved $\gamma$-errors from Rayleigh iteration across datasets, and the number of iterations needed to meet $T_{\mathrm{err}}$? 2. Does the algorithm have an adaptive mechanism to refine $\gamma$ if the KKT system indicates near-full-rank behavior (e.g., small but nonzero residuals), instead of collapsing to zero on the first step? For instance, can you detect ill-conditioning of $\kappa$ and trigger another Rayleigh update?

Choice and construction of $D$: 1. What concrete $D$ do you use in experiments? Explain the construction and its cost. 2. How does $D$ affect $T_{\mathrm{err}}$ and the conditioning of $\Phi, \Psi$? Could $D$ be tuned to improve numerical stability or accelerate convergence?

Newton solver and numerical stability: 1. Which linear solver and preconditioning strategy do you use for the KKT system? Any regularization applied to handle near-singularity when $\gamma$ is close but not exact? 2. Can you provide ablations on the number of Newton iterations ($n_{\mathrm{max}}$), line search parameters, and their impact on both accuracy and runtime?

Extensions and limitations: 1. Could the framework extend to computing singular subspaces directly (block methods), potentially improving robustness for clustered singular values? 2. What are the failure modes you observed (e.g., near-identical singular values with noise), and can you propose diagnostics or automatic restarts?

Theoretical clarifications:  1. In Theorem 3.1, you show full-rank $K$ when $\gamma$ is incorrect. Is there a quantitative condition number characterization as $\gamma$ approaches $1/s$, to explain practical behavior before collapse? That could guide adaptive $\gamma$ updates. 2. In Theorem 3.2, the bound uses Cauchy–Schwarz over sums of features. Are there tighter, data-dependent bounds (e.g., using coherence or leverage scores) that give less conservative $T_{\mathrm{err}}$ in practice?

Reproducibility and code: Please clarify the exact randomized subspace iteration routine and parameters (power iters, oversampling $p$). Also include the specific Rayleigh quotient iteration settings (niter, tolerance), and the line search strategy details.

Overall suggestion: Add a section with (i) an adaptive $\gamma$-refinement loop informed by KKT residuals/conditioning, (ii) more thorough ablations on $D$, Newton iterations, and RSVD parameters, and (iii) scaling experiments that demonstrate parallel gains. These would substantially strengthen the practical case for Des-SVD.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
### Summary

This paper introduces "Des-SVD," a new algorithm for Singular Value Decomposition (SVD), motivated by the need for methods that are more amenable to parallelization than traditional matrix-based algorithms and more efficient than existing descent-based approaches (like Riemannian gradient descent).

The proposed solution is based on a primal-dual reformulation (first proposed by Suykens, 2016) that connects SVD to the KKT conditions of a non-convex least-squares problem.

The paper's core contributions are:
1.  **Theoretical Analysis (Section 3):** A key insight (Theorem 3.1) that for this non-convex problem, non-zero KKT solutions—which can be normalized to yield singular vectors—*only* exist when the correct singular value $s$ is used as a parameter. An incorrect $s$ leads to a trivial zero solution.
2.  **A New Algorithm (Des-SVD):** A practical, multi-stage algorithm that first estimates singular values (e.g., via Rayleigh-Quotient Iteration) and then uses this theoretical insight to solve for the singular vectors using an iterative method (Algorithm 1), which is shown to be Newton's method on the KKT system.
3.  **Empirical Results:** The paper demonstrates that Des-SVD is "significantly higher" in efficiency than Riemannian SVD and "competitive with" (or even faster than) standard matrix-based methods like Jacobi-SVD and Lanczos-SVD.

### Strengths
1.  **Novel Theoretical Insight:** The paper's core theoretical contribution (Theorem 3.1), which analyzes the KKT conditions of the (Suykens, 2016) formulation, is a valuable and novel insight. It provides a clear theoretical justification for *why* a non-convex least-squares problem can be used to find the SVD, showing that non-trivial solutions are only possible when the correct singular value is used as a parameter.
2.  **Strong Empirical Performance:** The proposed Des-SVD algorithm demonstrates impressive empirical results. It is shown to be significantly more efficient than the primary descent-based alternative (Rie-SVD) and performs competitively against, or even faster than, standard matrix-based methods like Lanczos-SVD on several benchmarks.
3.  **Valid Problem Motivation:** The paper correctly identifies a valid gap in the literature: the desire for SVD algorithms that are better suited for modern, large-scale, and potentially distributed/parallel computing paradigms, where traditional centralized matrix methods can be bottlenecks.

### Weaknesses
### 1. Misleading Theoretical Framing: "Descent Method" vs. "Matrix-Based Method"

The paper's entire narrative is built on a false dichotomy: "matrix-based methods" vs. "descent methods." It then frames Des-SVD as a "descent method." This is fundamentally incorrect. The core of the "descent method" (Algorithm 1) is **Newton's method**, which requires solving a large, dense KKT linear system in each iteration. This is a "matrix-based operation" by any definition. The paper's own complexity analysis (Section 4.2) states this KKT system is $(6k \times 6k)$ and solving it is an $O((6k)^3)$ operation. The paper simply **replaces one set of matrix operations** (e.g., Lanczos) with **another** (Newton on a KKT system).

### 2. The Core SVD Problem is Assumed Solved

The paper's "novel" part, Algorithm 1, is titled "Descent method for calculating the singular vectors **from a given singular value**." This framing reveals the core weakness: the method *presupposes* that the singular values ($s$) are already known. Algorithm 2 (the "refined" version) is a **multi-stage, hybrid, matrix-based algorithm** that uses classical matrix methods (like Randomized-Subspace-Iteration and Rayleigh-Quotient Iteration) to find $s$ *before* the novel part of the algorithm ever runs.

### 3. Lack of Robustness Analysis and Convergence Guarantees

The algorithm's structure introduces new questions that the paper leaves unanswered.
* **Missing Robustness Analysis:** The algorithm's success hinges on an accurate estimate of $s$. While a theoretical bound ($T_{err}$) is provided (Theorem 3.2), the **empirical impact of a misspecified $s$ is never shown**.
* **Missing Convergence Guarantees:** For a new numerical algorithm, the paper provides **no explicit convergence guarantees** for the Newton solve in Algorithm 1.

### 4. Unsubstantiated Claims of Parallelization

The paper's primary motivation is parallelization, but its claims are weak and unproven.
* The claim that matrix-based methods are "unfriendly to parallelization" is a strong and largely inaccurate generalization.
* The paper's "parallelization" (Algorithm 2, Step 11) is an "embarrassingly parallel" loop over $k$ values, which is only possible *after* a **serial, matrix-based pre-computation** (Step 8) has found all $k$ singular values.
* The paper provides **no empirical evidence** (e.g., a scaling plot) to quantify its parallel performance.

### 5. Incremental Novelty

The core primal-dual formulation (Equation 4) is explicitly attributed to **Suykens (2016)**. The paper's *true* theoretical contribution is the *analysis* of this existing formulation, making the novelty more incremental than presented.

### Questions
I may have missed this but what is p in table 2?

### Soundness
2

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
3

### Summary
A descent method is developed based on the LS formulation and its dual. Basic properties of the formulation are briefly discussed.

### Strengths
A descent method based on the LS formulation, and the discussion of the basic properties. The idea is simple and interesting.

### Weaknesses
I am not sure whether the method is really practical or not. Indeed, it is not compared with the true SOTA solvers for SVD. For dense matrices, the QR algorithm and the divide and conquer algorithm are more efficient. For sparse matrices, there also exists iterative optimization solver such as LOBPCG which is well developed. In addition, the sizes of some test problems are relatively small.

### Questions
See weakness.

### Soundness
2

### Presentation
3

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
The work revisits the classical singular value decomposition problem and aims to propose a computationally efficient algorithm based on gradient descent. The authors build their work on the result that nonzero KKT solutions of a least squares primal problem give the singular vectors under certain conditions. The method can support parallelization unlike the classical approaches and hence can reduce the computational time compared to projection-based approaches like Riemann SVD. The experiments are presented to showcase the effectiveness of the approach.

### Strengths
Strengths:

1.	The paper is written in an organized, well-articulated manner. It is easy to follow (in most parts) and the theoretical results are discussed in sufficient detail.

2.	The problem statement is also relevant and timely as decomposition methods are recently getting a lot of attention in AI-based efficient/interpretable models including in the domain of LLMs

### Weaknesses
Weaknesses/Questions:

1.	Problem (4) seems like the primal problem. It is mentioned before 4 that it is the dual formulation.

2.	The dual solutions $\alpha$ and $\beta$ are introduced and discussed before stating the dual problem which brings some confusion and less clarity even in the notations itself.

3.	In Section 3.2, it is intended to show that when the singular values are not known to set the regularization coefficient, we get zero solutions. It is not clear how does the KKT matrix is shown to be full rank. It would be better to give some insights of this proof step since it is very important in the message conveyed in Section 3.2. 

4.	In Section 3.2 itself, $\bm v$ is mentioned as the Lagrange operator initialized as 0 and also the primal variables. So it is unclear how you obtained Eq. 17 from Eq. 16. There is some lack of clarity in these steps.

5.	I think, the proof does not show there exists no other nonzero KKT solutions that are not singular vectors. 

6.	In the practical implementation of the algorithm, the inputs require C matrix which needs the mapping functions to be computed that involves an estimation of D and also the prior estimation of singular values. Do you also consider these computational costs when comparing them with other algorithms? If not, I highly suggest the comparison has to be with overall cost of obtained SVD starting from the input matrix A without assuming any other prior info.

7.	It is also mentioned that the normalization of the dual variables are done in each iteration. But in pseudo code, it is performed only once after all the iterations. Why is such a design choice?

8. Also, in the pseudo code, a check to see if the solutions are zero is missing. If the singular values are not specified exactly, how do we inspect and detect the failure scenarios?

9.	In experiments, there should also be some experiments to validate Theorem 3.2 to see the effect of misspecification of singular values.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
