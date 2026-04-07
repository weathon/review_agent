=== CALIBRATION EXAMPLE 2 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title ("A New Efficient Method for Combining Gradients of Different Orders") is imprecise. The method combines gradient-related quantities weighted by powers of the Hessian, but calling these "gradients of different orders" is non-standard terminology that obscures what is actually being done. More accurate would be "polynomial acceleration via Hessian-vector products."

The abstract is severely deficient for ICLR. It contains obvious typos ("recipprocal," "steplenth," "Whave"), but more substantively: the claims are vague ("faster convergence rates"), no theoretical guarantees are stated, and no quantitative improvement is reported. The abstract does not tell the reader what problem is solved, in what setting, at what computational cost, or with what proven guarantees. For ICLR this is well below the bar.

---

### Introduction & Motivation

The introduction restricts attention to **unconstrained convex quadratic minimization** (Eq. 1). This is an extremely narrow problem class. For ICLR — a venue focused on representation learning and modern machine learning — there is no motivation given for why this problem class is the right target, no connection to neural network training, and no discussion of non-convex or stochastic settings. The paper never attempts to argue relevance beyond this narrow class.

The related-work coverage is thin even for the quadratic setting. Conspicuously absent are:
- **Conjugate gradient (CG) methods**, which solve n-dimensional quadratics in at most n steps — the obvious baseline for this problem.
- **Chebyshev polynomial acceleration**, which is conceptually very close to what GOC does (polynomial filtering of eigenvalues).
- **Heavy ball / Polyak momentum**, which is another classical second-order acceleration of gradient descent.
- Any connection to **Krylov subspace methods**.

The omission of CG alone is a serious flaw: CG is the standard solver for the exact problem the paper addresses, and an optimizer targeting convex quadratics that does not compare to CG is hard to take seriously.

---

### Section 2 — Analysis of SD and CBB

The geometric analysis interpreting CBB as a "reflection" of the steepest-descent point through an axis defined by Ag₀ is interesting and provides some intuition. The derivation of Eq. (18), showing x₁⁽ⁱ⁾ = x₀⁽ⁱ⁾ µ₀⁽ⁱ⁾², is correct for the diagonal case and is a known reformulation of CBB.

However, the leap to calling SD "first-order" and CBB "second-order" (with respect to powers of µ) is informal. The parameter m appearing in Eq. (21) is introduced without prior definition; it seems to track consecutive iterations with the same step size, but this is not clearly stated. The analysis in cases 1–3 (large-eigenvalue–dominated, small-eigenvalue–dominated, and general) is qualitative and never supported by formal lemmas or theorems. The convergence factor referenced at the end of Section 3 (1 − θ = (λₘₐₓ − λₘᵢₙ)/λₘₐₓ) is stated without citation specifics and does not appear in the main narrative in a way that connects to the proposed method.

---

### Section 3 — GOC Method

This is the core technical section. The central idea is:

- Interpret m consecutive SD steps as applying (1 − a⁽ⁱ⁾/r)ᵐ to each eigenvector component.
- Expand this binomial: for m=3, (1 − µ)³ = 1 − 3µ + 3µ² − µ³.
- Implement the binomial coefficients by computing Agₖ and A²gₖ via finite differences of the gradient (Eq. 24 and Algorithm 1).

**Conceptual connection not made:** This is precisely the idea behind **Chebyshev polynomial acceleration** and **polynomial preconditioning** of Krylov methods, where one applies a degree-m polynomial to the spectrum to accelerate convergence. The authors do not acknowledge this connection, which is a significant omission.

**Key technical gap — auxiliary step size d:** Algorithm 1 depends critically on a user-chosen step size d to compute finite differences (dAgₖ ≈ gₖ − gₖ¹, d²A²gₖ ≈ gₖ − gₖ²). The paper never discusses: (i) how to choose d, (ii) sensitivity to d, (iii) numerical stability when d is too small (cancellation) or too large (nonlinearity corrupts the approximation). For a quadratic objective, Agₖ can be computed exactly as Agₖ = (∇f(xₖ + d·gₖ) − ∇f(xₖ))/d with exactness regardless of d (since the Hessian is constant). This is never noted, and whether or not the authors exploit this exactness for quadratics is unclear.

**Per-iteration cost not accounted for:** GOC requires **3 gradient evaluations per iteration** (gₖ, gₖ¹, gₖ²) versus 1 for SD/CBB. The entire experimental comparison reports raw iteration counts, not gradient evaluation counts. This makes the comparison unfair: GOC is compared to BB/CBB on "iterations," but each GOC iteration costs 3× as much. When the cost is normalized to gradient evaluations, the advantage may disappear or reverse.

**No convergence theory:** There is no theorem, lemma, or even informal convergence argument for the GOC method. The paper claims "faster convergence rates" (abstract) and "faster rate of descent" (conclusion) without any theoretical support. Even a simple analysis on the two-dimensional quadratic case would substantially strengthen the paper.

**The step rₖ:** Equation (8) defines rₖ = gₖᵀAgₖ / (2 gₖᵀgₖ), essentially half the Rayleigh quotient of A with respect to gₖ. This is used as the normalization in the update. The choice to define r as 1/(2α) and use it throughout is a clean re-parameterization, but the paper does not discuss what happens when rₖ is not a good approximation to the true eigenvalue structure — i.e., in ill-conditioned or non-quadratic settings.

---

### Section 4 — Numerical Experiments

The experiments test one problem (Eq. 25): a 100,000-dimensional **diagonal quadratic** with eigenvalues a⁽ⁱ⁾ forming an arithmetic progression in [0.001, 10000].

**Critical deficiencies:**

1. **Only one problem.** A single diagonal quadratic tells us almost nothing about general performance. Real test suites for gradient methods include problems like those in CUTEst or at minimum multiple ill-conditioned quadratics with varying condition numbers.

2. **No CG baseline.** For a 100,000-dimensional diagonal quadratic, CG solves the problem **exactly in at most 100,000 steps**, and typically much fewer. BB and CBB are gradient methods, not Krylov methods, and are not the state-of-the-art for this problem class.

3. **Unfair iteration comparison.** As noted above, GOC uses 3× more gradient evaluations. In Figure 3, BB/CBB/GOC are compared per "iteration." No gradient-evaluation-normalized plot is provided.

4. **Ambiguous stopping metric.** The paper reports "number of times the method satisfies the stopping condition" out of 5000 iterations (e.g., BB satisfies 4930 times, GOC 1864 times). This is an unusual metric. If smaller counts indicate fewer stopping events, the interpretation is that GOC converges to tolerance less often — which would be a disadvantage, not an advantage. This needs clarification.

5. **"BB could not satisfy the stop condition" in Figure 3b** is stated without explanation. This is suspicious and should be investigated and reported carefully, especially since BB is reported as satisfying the condition 4930 times in the fixed-initial-point experiment.

6. **No non-quadratic test.** The entire motivation breaks down outside quadratics (where A is the exact Hessian), yet no non-quadratic problem is tested.

7. **No statistical reporting.** Figure 3b involves random initial points, but no variance, confidence intervals, or multiple seeds are reported.

---

### Writing & Clarity

Beyond typographic issues (which are noted as possibly parser artifacts), there are genuine structural problems: Section 4 text appears **after** the References in the extracted PDF, the algorithm box content is interspersed with experimental text, and Equation (5) (the SD convergence rate bound) appears after Equations (6) and (7) (the BB steps). Even accounting for parser artifacts, the paper's logical flow is fragmented and hard to follow. A reader cannot extract the method clearly without significant effort.

---

### Limitations & Broader Impact

The paper has no limitations section. The key unacknowledged limitations include:

- **Restricted to quadratics:** The method is defined in terms of r = gᵀAg/(2gᵀg), which requires knowledge of A (the Hessian). In non-quadratic problems, A is not constant, and the entire derivation based on fixed eigenvalues collapses.
- **No stochastic version:** Modern ML optimization is almost entirely stochastic (mini-batch gradients). There is no discussion of how GOC would behave under gradient noise.
- **Cost per iteration:** 3 gradient evaluations per step is not free, especially in deep learning where gradient computation dominates runtime.
- **No comparison to CG or momentum methods:** These are the natural baselines and are absent.

---

### Overall Assessment

This paper presents an idea — interpreting steepest descent and CBB as first- and second-order polynomial approximations in the eigenspectrum, then extending to third order via finite-difference Hessian-vector products — that has some conceptual interest. However, the paper is far below the ICLR acceptance bar on every relevant dimension. There is no convergence theory. The single numerical experiment is on a trivial diagonal quadratic and omits the most important baseline (conjugate gradient). The per-iteration cost of GOC (3 gradient evaluations) is never accounted for in comparisons. The method's connection to well-established Chebyshev/polynomial acceleration is not acknowledged. The paper applies only to convex quadratics and makes no connection to machine learning or neural network training. Even as a pure optimization paper at a specialized venue, the contribution would require substantial theoretical and experimental development. As submitted to ICLR, the paper is not ready for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes GOC (Gradient Order Combination), a new optimization method for unconstrained problems that combines gradient information with products of Hessian matrices of varying orders. The authors argue that Stochastic Descent (SD) and CBB methods correspond to 1st and 2nd-order methods respectively within a specific parameter framework, and generalize this to a higher-order update scheme using finite differences to approximate Hessian-vector products. The method is evaluated on convex quadratic functions, showing fewer iterations to convergence compared to BB and CBB baselines.

### Strengths
1.  **Theoretical Derivation:** The paper provides a systematic mathematical analysis linking the parameter $r_k$ (reciprocal of step size) to the convergence behavior of Steepest Descent and CBB methods on quadratic forms (Section 2). This offers a unified perspective on existing gradient methods.
2.  **Algorithm Simplicity:** Algorithm 1 describes a relatively straightforward update rule that utilizes finite differences ($Ag \approx (g_k - g_{k+1})/d$) to approximate Hessian-vector products without explicitly forming the Hessian matrix.
3.  **Clear Baseline Comparison:** The numerical experiments (Section 4) clearly demonstrate the iteration counts required for GOC to converge relative to BB and CBB on specific quadratic landscapes, allowing for direct comparison of the proposed method against known methods.

### Weaknesses
1.  **Lack of Relevance to ICLR Scope:** ICLR focuses on Learning Representations and deep learning applications. The experiments are restricted strictly to convex quadratic functions with no connection to neural networks, stochastic gradients, or non-convex optimization landscapes typical in ML (Section 4). This severely limits the paper's applicability to the conference audience.
2.  **Limited Experimental Validation:** The method is only tested on high-dimensional convex quadratics. There is no comparison against modern adaptive optimizers (e.g., Adam, L-BFGS-B) or on real-world datasets (e.g., ImageNet, MNIST, BERT training), making claims of "efficiency" difficult to contextualize in the broader ML field.
3.  **Unconventional Terminology and Claims:** The classification of SD as "first-order" and CBB as "second-order" based on the $r$ parameter behavior (Section 3) diverges from standard complexity and derivative definitions. While heuristic, this framing lacks rigorous proof regarding the actual order of convergence (e.g., Q-linear vs. superlinear).
4.  **Approximation Sensitivity:** The method relies on finite difference approximations for Hessian-vector products (e.g., $Ag \approx (g_k - g_{k+1})/d$). In practice, the small step size $d$ required for accuracy could introduce numerical instability or noise, particularly in stochastic settings, which is not addressed.

### Novelty & Significance
**Novelty:** The proposed GOC method represents an incremental variation of spectral gradient methods (like BB) using finite differences. While the specific combination formula is novel, the concept of using high-order derivative information (approximated via finite differences) for acceleration exists in literature (e.g., truncated Newton, L-BFGS). The derivation is solid within the context of deterministic quadratic optimization but offers limited novelty for representation learning.

**Significance:** The significance for ICLR is low because the optimization challenges addressed (convex quadratic minimization) are solvable with existing robust methods (Conjugate Gradient, L-BFGS) without the added complexity or assumptions of GOC. Without empirical evidence on non-convex or stochastic problems, the method's significance in training modern models remains unproven.

### Suggestions for Improvement
1.  **Add Machine Learning Experiments:** To meet ICLR standards, include experiments on non-convex loss landscapes, ideally training a neural network (e.g., MLP or CNN) on standard datasets to compare iteration counts and wall-clock time against Adam or SGD+Momentum.
2.  **Clarify Convergence Theory:** Provide rigorous proof of the convergence rate for general non-quadratic $f(x)$ (e.g., using Lipschitz continuity and Polyak-Lojasiewicz conditions), rather than relying on the heuristic $r$ parameter analysis.
3.  **Address Stochastic Noise:** Discuss how the finite difference approximation of $Ag$ performs when gradients are noisy. The proposed method assumes deterministic gradients for high-order accuracy; clarification on stochastic variants is necessary.
4.  **Comparison with L-BFGS:** Add comparisons with L-BFGS, which is the standard "memory-based" optimization method for high-dimensional problems, to better position GOC within the current landscape of scalable optimizers.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Standard ML Benchmarks:** The paper only evaluates on a synthetic quadratic function. You must validate on standard ICLR datasets (e.g., logistic regression on LIBSVM, training ResNet on CIFAR-10) to prove utility beyond toy problems.
2. **Computational Cost Metrics:** You report iteration counts, but GOC requires 3+ gradient evaluations per step to approximate $Ag$ and $A^2g$. You must report wall-clock time and total gradient evaluations, otherwise the speedup is misleading.
3. **Stronger Baselines:** Comparisons are limited to BB and CBB. You must compare against Conjugate Gradient (CG) for quadratics and Adam/SGD for general optimization, as these are the standard benchmarks for efficiency.
4. **Non-Convex Validation:** ICLR focuses on deep learning. You must demonstrate efficacy on non-convex objectives (neural networks), as convergence on convex quadratics does not guarantee performance on loss landscapes with saddle points.

### Deeper Analysis Needed (top 3-5 only)
1. **Per-Iteration Complexity:** There is no analysis of the computational overhead per step. You need to quantify the FLOPs required to estimate Hessian-vector products via finite differences compared to a standard gradient step.
2. **General Convergence Proof:** The theoretical claims rely on heuristics for quadratic forms. You need a formal convergence theorem for general smooth non-convex functions to support the claim of "higher-order" efficiency.
3. **Numerical Stability:** The method approximates $Ag$ using $g_k - g_{k+1}$ with step size $d$. You must analyze how sensitive the method is to $d$, as finite-difference Hessian approximations are notoriously unstable for small steps.

### Visualizations & Case Studies
1. **Loss vs. Wall-Clock Time:** Plot objective value against actual training time, not just iterations. This will reveal if the extra computation per GOC step negates the reduction in iteration count.
2. **Hyperparameter Sensitivity:** Visualize performance across a range of the step size $d$ used for Hessian approximation. If the method fails outside a narrow range, it is impractical for general use.
3. **Condition Number Scaling:** Show how convergence rates degrade as the condition number of the problem increases compared to CG. This validates the claim that GOC better handles ill-conditioning.

### Obvious Next Steps
1. **Deep Learning Application:** Implement GOC as an optimizer for a standard neural network architecture to verify compatibility with backpropagation and auto-differentiation frameworks.
2. **Adam/SGD Comparison:** Run head-to-head trials against Adam and SGD on generalization metrics (test accuracy), not just training loss, to establish relevance for the ICLR community.
3. **Verify Hessian Approximation:** Explicitly measure the error between the finite-difference approximation ($g_k - g_{k+1}$) and the true Hessian-vector product to validate the core mathematical mechanism.

# Final Consolidated Review
## Summary

The paper proposes GOC (Gradient Order Combination), a new optimization method for unconstrained convex quadratic problems. The key insight is interpreting steepest descent (SD) and CBB as "first-order" and "second-order" methods respectively based on how they apply polynomial coefficients to eigenvalue components, then extending this to higher orders (m=3) by computing Hessian-vector products via finite differences of gradients. The method requires three gradient evaluations per iteration to approximate Ag and A²g.

## Strengths

- **Geometric interpretation of CBB**: The paper provides an interesting geometric analysis (Section 2) showing that CBB can be viewed as a reflection of the steepest-descent point through an axis defined by Ag, and derives the relationship x₁⁽ⁱ⁾ = x₀⁽ⁱ⁾ μ₀⁽ⁱ⁾². This offers a unified perspective on existing gradient methods.
- **Conceptual framework for polynomial filtering**: The binomial expansion approach—(1−μ)³ = 1−3μ+3μ²−μ³ for m=3—connects to the idea of applying degree-m polynomials to the eigenvalue spectrum to accelerate convergence. While the connection to Chebyshev polynomial acceleration is not acknowledged, the derivation is mathematically coherent for the quadratic case.

## Weaknesses

- **No convergence theory**: The paper claims "faster convergence rates" (abstract) and "faster rate of descent" (conclusion) without any theorem, lemma, or formal convergence analysis. Even a simple proof for the two-dimensional quadratic case is absent. This is a critical gap for an optimization paper.
- **Per-iteration computational cost ignored**: GOC requires 3 gradient evaluations per iteration (computing gₖ, gₖ¹, gₖ²) versus 1 for SD/BB/CBB. The experiments report iteration counts without normalizing for gradient evaluations. When cost is measured in gradient evaluations—the standard metric—GOC's apparent advantage may disappear or reverse. This makes the comparison misleading.
- **Extremely limited experimental validation**: The paper tests only a single diagonal quadratic function (Eq. 25). There is no comparison to conjugate gradient (CG)—the standard solver for this exact problem class, which would solve a 100,000-dimensional quadratic in at most n iterations. There is no testing across varying condition numbers, non-diagonal matrices, or standard optimization test suites (e.g., CUTEst).
- **No connection to machine learning**: The paper addresses only deterministic convex quadratic minimization with constant Hessian A. There is no discussion of non-convex optimization, stochastic gradients, or neural network training. The method depends on rₖ = gₖᵀAgₖ/(gₖᵀgₖ), which exploits the exact Hessian structure of quadratics. How the method would behave when Hessian information is approximated or when the objective is non-quadratic is not addressed.
- **Ambiguous experimental metric**: The paper reports "number of times the method satisfies the stopping condition" out of 5000 iterations (e.g., BB: 4930, GOC: 1864 in Figure 3a). This is an unusual and unclear metric. It is not explained what this count means or why fewer counts (GOC) should indicate better performance.
- **Numerical stability of step size d unanalyzed**: Algorithm 1 depends on a user-chosen finite-difference step size d to approximate Hessian-vector products. The paper never discusses how to choose d, sensitivity to d, or numerical stability when d is too small (cancellation errors) or too large. For a method whose core mechanism relies on these approximations, this is a significant omission.

## Nice-to-Haves

- **Comparison with conjugate gradient**: For convex quadratic optimization, CG is the natural baseline. Demonstrating that GOC offers advantages over CG—even in specific regimes—would substantially strengthen the paper.
- **Experiments on non-quadratic objectives**: Testing on non-convex problems or standard ML benchmarks would establish relevance beyond toy quadratics.
- **Gradient-normalized performance**: Reporting performance in terms of total gradient evaluations rather than iterations would provide an honest comparison.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that Section 4 appears after References**: This is a parser artifact, not a paper issue.
- **Generic strengths**: Claims like "the topic is important" or "experiments are extensive" have been removed—experiments are in fact extremely limited.
- **Missing citations (Chebyshev, heavy ball, Krylov methods)**: Per instructions, I do not have external sources to confirm these exist, so cannot claim they are missing.
- **Demand for confidence intervals on single-run experiments**: For deterministic optimization on convex quadratics, single-run results are the norm.

## Novel Insights

The geometric interpretation of CBB as "reflecting" the steepest-descent point through an Ag-defined axis is a genuine insight that could help practitioners understand why CBB sometimes outperforms SD. The polynomial-order framework for viewing SD (m=1), CBB (m=2), and GOC (m=3) as points on a spectrum is conceptually clean, though the connection to Chebyshev acceleration—where similar polynomial filtering ideas have been extensively studied—should be acknowledged and differentiated.

## Suggestions

1. **Provide formal convergence analysis**: At minimum, prove convergence for the quadratic case and establish the convergence rate. Without this, the claims of "faster" convergence are unsupported.
2. **Normalize by computational cost**: Report gradient evaluations or wall-clock time, not just iterations. A method requiring 3 gradient evaluations per step must show 3× improvement to claim any efficiency gain.
3. **Add CG baseline**: For convex quadratics, CG is the standard. Compare against it.
4. **Clarify the experimental metric**: Explain what "number of times satisfying stopping condition" means, or use a standard metric like iterations to reach ε-tolerance.
5. **Analyze step size d**: Provide guidance on choosing d and analyze sensitivity to this hyperparameter.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject
