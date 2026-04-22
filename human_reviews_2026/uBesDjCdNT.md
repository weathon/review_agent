# Sequential Least-Squares Estimators with Fast Randomized Sketching for Linear Statistical Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
We propose a novel randomized framework for the estimation problem of large-scale linear statistical
models, namely Sequential Least-Squares Estimators with Fast Randomized Sketching (SLSE-FRS),
which integrates Sketch-and-Solve and Iterative-Sketching methods for the first time. By iteratively
constructing and solving sketched least-squares (LS) subproblems with increasing sketch sizes to
achieve better precisions, SLSE-FRS gradually refines the estimators of the true parameter vector,
ultimately producing high-precision estimators. We analyze the convergence properties of SLSE-FRS,
and provide its efficient implementation. Numerical experiments show that SLSE-FRS outperforms
the state-of-the-art methods, namely the Preconditioned Conjugate Gradient (PCG) method, and the
Iterative Double Sketching (IDS) method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Sequential Least-Squares Estimators with Fast Randomized Sketching (SLSE-FRS) for large-scale linear statistical models. The method appears to be the first to unify Sketch-and-Solve with Iterative-Sketching. It enlarges the sketch size progressively and solves a sequence of sketched least-squares subproblems, which incrementally refines the estimate of the true parameter vector. The authors also provide a systematic treatment of convergence behavior and computational complexity.

### Strengths
* The main contribution is the novel integration of Sketch-and-Solve and Iterative-Sketching within a single framework. 
* The analysis is comprehensive:
   1)  it explains how to construct the sequence of sketched subproblems to balance estimation accuracy with computational cost;
   2) it proposes a theoretically sound and computationally tractable stopping criterion to achieve optimal iterative accuracy;
  3) it establishes, both in theory and in experiments, that the estimator attains noise-level accuracy comparable to ordinary least squares (OLS).

### Weaknesses
1) The *o(1)* term in equation (11) of Theorem 4.2 is not quantified. Moreover, letting $M \to \infty$ appears at odds with the paper’s non-asymptotic stance. In line with IDS-style results, please specify the scale of $M$ (e.g., explicit bounds or rates). Without such quantification, the result is not fully convincing.

2) Aside from the added momentum component, the proposed algorithm is essentially equivalent to the two-stage process used in IDS. Could the authors explicitly and theoretically demonstrate how much improvement their method achieves over IDS?

3) Theorem 4.4 provides a complexity analysis but does not present an explicit convergence rate. As a result, it is not immediately clear how the proposed algorithm achieves the claimed convergence–complexity trade-off.

### Questions
1) It is recommended that the authors cite the following works for completeness:
Derezinski, Michal, et al. “Newton-LESS: Sparsification without trade-offs for the sketched Newton update.” Advances in Neural Information Processing Systems, 34 (2021): 2835–2847.
Garg, Sachin, Kevin Tan, and Michał Dereziński. “Distributed least squares in small space via sketching and bias reduction.” Advances in Neural Information Processing Systems, 37 (2024): 73745–73782.
These works emphasize that bias removal can significantly improve estimation accuracy and convergence performance.

2) The paper compares the proposed method with M-IHS and shows that the second-stage procedure improves performance over M-IHS. However, it lacks an ablation comparison between SLSE-FRS and the estimator (\boldsymbol{\beta}_T) obtained using only the first-stage iterative form (7). It is recommended to include such an ablation study to better isolate and demonstrate the contribution of the second stage.

3) Could the authors clarify the purpose of the condition number  $ \kappa $ used in the experiments? Specifically, how is it defined and what role does it play in the analysis or performance evaluation?

4) The paper presents numerical experiments showing that the proposed SLSE-FRS method outperforms IDS in terms of the convergence–complexity trade-off. However, it lacks an explicit, formula-based comparison of the computational complexities of SLSE-FRS and IDS. Providing such a comparison would make the claimed improvement more transparent and convincing.

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
This paper introduces a two-stage framework that addresses a sequence of increasingly large sketched least-squares subproblems, followed by a few full LS iterations to achieve OLS-level accuracy. The authors provide a contraction guarantee (under specific SRHT-based conditions and momentum selection) and an implementation using M-IHS. Experiments demonstrate faster convergence than traditional random sketched least square methods.

### Strengths
$\textbf{Clear two-stage recipe.}$ Stage-1 builds $K$ sketched LS problems with growing sketch sizes and uses those solutions to warm-start Stage-2 on full data; the framework is solver-agnostic and explains the cost savings intuitively. 

 $\textbf{Convergence statement and practical parameters.}$ Theorems provide an exponential decay bound and a global contraction rate for suitable $(\mu,\eta)$, with a data-driven surrogate stopping rule (via $\omega$) and a lower bound on the per-subproblem iteration count $a_i$. 

$\textbf{Empirical speedups.}$ Plots/tables indicate faster time-to-accuracy than IDS and PCG on synthetic settings; M-IHS comparisons are also included.

### Weaknesses
$\textbf{Scope and assumptions feel narrow.}$ The method is largely a combination of standard components—Sketch-and-Solve least squares, iterative Hessian sketching (IHS), SRHT/CountSketch embeddings, and preconditioned Richardson/gradient iterations. The contribution appears only to lie in how these blocks are combined.

$\textbf{Literature position.}$ The authors need to clarify whether similar “sketch-warmstart + full-data polish” patterns have ever been explored, and articulate the specific differences with this literature.

$\textbf{Ablations.}$ Due to the combination nature of the paper, I expect more ablation studies to be performed to determine which components are the key to the success of the performance. See questions.

$\textbf{Unclear synthetic settings.}$ Experiments have some unclear and unexplained settings, see questions..

### Questions
1. To better evaluate the paper's “integration-and-scheduling” perspective is reasonable, but novelty claims should be framed accordingly. Clear positioning is recommended by clearly stating that each component is established.

2. Why do the simulation settings have unrealistically small noise level $\sigma^2=1e-8$? This narrows the problem, only placing the problem in an ultra–high-SNR regime. Adding a noise sweep is necessary.

3. The synthetic data description is under-specified: while $X$ is said to be i.i.d. Gaussian with an “artificially adjusted” condition number and 
$\beta$ is Gaussian, the paper does not describe the actual procedure to impose a target condition number $\kappa$ on $X$ (e.g., singular-value planting vs. diagonal scaling), nor whether columns are standardized, whether an intercept is used, or how 
$X$ is normalized before sketching.

4. Additional ablations are necessary. e.g., are there differences between one-shot big sketch vs. the schedule? How about the comparison between fixed estimated Hessian vs. updating after several iterations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors provide a randomized algorithm for the least squares problem. The algorithm is simple, and amounts to repeatedly performing a sketch-and-solve procedure with a doubling sketch size until some tolerance is achieved. In practice, this is done with a sketched Hessian + sketched gradient update. After which, the authors perform sketched Hessian updates on the full gradient. The authors show a high-probability guarantee for the first stage, and an asymptotic guarantee for the second. The number of iterations of the first stage is lower bounded by a logarithmic term in the reciprocal of the tolerance.

### Strengths
The idea of likening a series of sketch-and-solve solutions to an iterative sketched Hessian + sketched gradient method for computing an approximate least squares solution is nice and may be original. The proofs are rigorous. The writing within the paper is above average in quality. If the algorithm works well in practice, it could improve the performance of libraries for randomized methods for least squares by improving the warm-starts given to sketch-and-precondition solvers.

### Weaknesses
1. The algorithm is very simple, and amounts to using the doubling trick to determine the sketch size that should be employed when performing a sketch-and-solve operation. The authors solve each sketch-and-solve problem along the way efficiently via sketching the data, computing the sketched gradient, and then performing a sketched Hessian update on the coefficient obtained from the last iteration with the sketched gradient. Once the optimal sketch size is found, the authors then employ a sketched Hessian update on the full gradient. 
- The problem with this is that the authors ultimately still need a sketch size $r$ on the order of $d/\epsilon^2$, as seen in Theorems 4.1 and 4.2. In a sense, we already know the necessary sketch size, or at least its order. The doubling trick allows for only a logarithmic inflation, but it is difficult to see this as anything more than an engineering tweak. 
- After this, a sketched Hessian update is employed, that obtains an asymptotically linear convergence rate. I say asymptotically because the second term only goes to zero asymptotically at an unknown rate, and it is unclear how large the "constant" $M$ should be. This guarantee is not very sophisticated, and I believe the proof of this is folklore. 

2. The paper does not compare the runtime of its method to other papers within the literature that solve the least squares problem with sparse sketches, e.g. Garg et al. (2024), Chenakkod et al. (2024), Anari et al. (2022), etc. One can generally expect to solve the problem (of the first stage) in $\text{nnz}(A) + O(d^2/ / \epsilon)$ time with sparse sketches. It is odd that the procedure does not yield an improvement from $O(Nd)$ to $\text{nnz}(A)$. I suspect that this is an issue with the analysis. 
- In Theorem 4.2 the authors rely on a large constant $M$ and ultimately employ an asymptotic analysis of the runtime, in contrast to the non-asymptotic analyses common in the literature. 
- Accordingly, the benefit of this rather complicated first stage is not evident in the guarantee for the second stage. If these guarantees are tight (I do not think they are), then there is no benefit to performing the first stage over simply initializing the sketch-and-precondition second stage with a sketch-and-solve solution. 
- As such, the numerical experiments are somewhat unfair to IDS and PCG -- they are not initialized with a sketch-and-solve solution, while SLSE-FRS almost is. 


### References
- Garg et al. (2024), Distributed Least Squares in Small Space via Sketching and Bias Reduction
- Chenakkod et al. (2024), Optimal Embedding Dimension for Sparse Subspace Embeddings
- Anari et al. (2022), Optimal sublinear sampling of spanning trees and determinantal point processes via average-case entropic independence.

### Questions
See weaknesses. This score is somewhat harsh. At the moment, I am on the fence between a 2 and a 4, and am willing to increase my score if I am proven wrong or my concerns are addressed.

### Soundness
3

### Presentation
2

### Contribution
2
