# A Proximal-Sinkhorn-Newton Method for Entropic Optimal Transport

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4, 4, 4

## Abstract
Entropic optimal transport (OT) enables efficient distribution alignment through the Sinkhorn method. However, it suffers from numerical instability and slow convergence under weak entropic regularization. We propose a two-stage framework that establishes an inexact-to-exact paradigm to address these challenges. The first stage employs an inexact proximal point method to decompose the entropic OT into simpler subproblems, yielding approximate solutions with superior numerical stability. The second stage employs a sparse Newton method with global convergence and a locally quadratic rate to refine the approximate solutions. Compared to previous Newton-based algorithms, it accelerates updates and prevents the objective score from plateauing during optimization. With numerical instability handled in the first stage,  Sinkhorn iterations can provide an alternative to the Newton method under relatively heavy entropic regularization. The yielding Proximal-Sinkhorn-Newton method enjoys the strengths of three approaches and outperforms the baselines across various regularizations and error tolerances.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a hybrid algorithm, the Proximal-Sinkhorn-Newton (PSN) method, combining an inexact proximal point step for stability with a sparse Newton refinement for efficiency in solving entropic optimal transport (EOT) problems. The authors establish local quadratic and global convergence results and report empirical speedups over existing Sinkhorn and Newton-type solvers.

While the paper is well structured and presents a coherent algorithmic framework, its technical and empirical contributions are incremental relative to existing Newton-based OT solvers such as “A Truncated Newton Method for Optimal Transport” (Kemertas et al., 2025).

### Strengths
(1) A clear integration of proximal regularization with sparse Newton iterations and with theoretical justification, including convergence proofs and use of incomplete Cholesky preconditioning.
(2) A unified “inexact-to-exact” interpretation that is conceptually appealing.

### Weaknesses
(1) Limited test scale. Numerical tests are not compelling—datasets have at most n = 5000, far below what recent OT solvers handle. To support the claimed “20× speedup,” larger-scale experiments (≥ 10⁵ samples) or GPU benchmarks would be necessary.
(2) Analytical specificity. The spectral estimates (e.g., Eqs. (54)–(56), (80)) largely follows standard proofs. The claimed local quadratic rate and global convergence are textbook results; a more informative contribution would involve explicit spectral estimates for the Hessian H in terms of sample size n, feature dimension d, and the EOT parameter \beta.
(3) Complexity discussion. The runtime analysis is qualitative. It would strengthen the paper to derive complexity bounds tied to the annealing schedule of \Delta \beta, and to compare analytically (or experimentally) how different \Delta\beta choices affect convergence.
(3) Comparison depth. The comparison to prior Newton-type methods (e.g., SNS, SSNS, Truncated Newton) lacks sufficient quantitative context—please include wall-clock and iteration counts over a wider range of \beta and n.

### Questions
(1) I wonder if the authors can provide more explicit spectral estimates for the Hessian H in terms of sample size n, feature dimension d, and the EOT parameter \beta, at least numerically. 
(2) I wonder if the authors can derive complexity bounds tied to the annealing schedule of \Delta \beta, and to compare analytically (or numerically) how different \Delta\beta choices affect convergence.
(3) The comparison to prior Newton-type methods (e.g., SNS, SSNS, Truncated Newton) lacks sufficient quantitative context—please include more comparison such as the wall-clock and iteration counts over a wider range of \beta and n.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper is dedicated to efficient computation of entropic optimal transport (OT) between discrete distributions. Achieving high precision with low regularization poses a major challenge for existing algorithms. For example, the famous Sinkhorn algorithm becomes unstable for such formulations. Proximal point methods relax these problems into a series of subproblems with stronger regularization, which improves stability but requires careful tuning of the regularization parameters. Newton methods require proper initialization and have a prohibitively large iteration cost, while their sparse variants suffer from certain stagnation issues. To overcome these difficulties, the authors propose versions of inexact proximal point and sparse Newton methods, as well as a hybrid solver. The experiments showcase the fast convergence of the proposed methods.

### Strengths
1. The paper presents a theoretical result on the proximal point method for entropic OT, providing a simple way to choose the regularization parameters for its subproblems; this provides a practical guideline for parameter selection.
2. The authors propose a new efficient Hessian sparsification procedure tailored to OT problems. Two convergence theorems for the related procedures are provided.
3. The experiments demonstrate that the proposed hybrid method is efficient across several different scenarios.

### Weaknesses
1. In some parts of the paper, the presentation is informal. For example, the statement of Lemma 2 does not specify what is meant by the word "approximation". Furthermore, the fast positive-definite-preserving sparse scheme should be formally defined. Theorems 2 and 3 should clearly state the assumptions on the parameters of the procedures. The term "objective score" should be defined.
2. The hybrid method lacks convergence guarantees.
3. The paper contains a lot of typos and small errors.

### Questions
1. Where appropriate, could you perform multiple runs of experiments with different realizations of randomness, and depict the variance in the plots?
2. Is it possible to formulate some convergence statement regarding Algorithm 3?
3. Could you please provide a zoom-in for the plots where some curves almost overlap, e.g., Figure 5(a).

Typos and minor errors:
- Line 107: "it can serve**s**"
- Theorem 1 (and some of the other statements): "Let ... denote**s**"
- Equations (5), (7): did you accidentally use elementwise product instead of the Frobenius inner product? Also, please introduce the notation for the elementwise product before using it.
- Equations (9): notation for an approximate KL-projection is used here, but introduced only after equation (10)
- Algorithm 2, line 5: is PCG defined in the text?
- Line 279: "grantees"
- Line 313: hat should be above H
- Line 323: missing space
- Line 363: "entries **[are?]** sampled"
- Line 380: missing space

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
4

### Summary
The paper proposes the Proximal-Sinkhorn-Newton (PSN) method, a hybrid algorithm for solving entropic optimal transport problems. The method combines an inexact proximal point method (IP-EOT) to generate a stable initial solution, which is then refined by either Sinkhorn iterations or a sparse Newton method. The authors provide theoretical analysis for their framework and present experimental results showing that PSN is faster than several baseline methods on the tested problems.
The paper makes a solid contribution by demonstrating how three well-known optimization techniques can be integrated to address the practical challenges of entropic OT. The approach is logical and the empirical results are positive. However, the theoretical guarantees follow standard forms and have practical limitations, and the introduction of new hyperparameters adds to tuning complexity. On balance, the paper's contributions warrant acceptance.

### Strengths
1. A key contribution is Theorem 1, which formalizes the connection between the iterates of the IP-EOT method and the entropic OT solution for a linearly increasing regularization parameter β. This provides a clear justification for using the IP-EOT output as a warm-start for the refinement stage.
2. The paper's main strength is the design of a multi-stage algorithm that leverages the advantages of different optimization methods. Using a proximal point method to ensure numerical stability before switching to a faster second-order method is a sound and logical strategy. The adaptive switch between Sinkhorn and Newton based on Hessian sparsity is a thoughtful addition for improving robustness.

### Weaknesses
1. The framework introduces new, important hyperparameters, namely the proximal step-size $\Delta \beta$ and the sparsity threshold $\lambda$. The performance of the algorithm appears to depend on these settings, but the paper lacks a sensitivity analysis or ablation study. This makes it difficult for a practitioner to understand how to tune these parameters for new problems, and whether the reported performance is robust to these choices.
2. The global convergence proof establishes asymptotic convergence but does not characterize the constants that govern the actual speed.

### Questions
1. Could you provide some intuition on how a user should set the $\Delta \beta$ and $\lambda$  parameters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PSN, a hybrid solver for entropic optimal transport (EOT) that combines (1) a proximal homotopy scheme (IP-EOT), (2) a sparse preconditioned Newton refinement, and (3) an adaptive switching rule. It targets the regime of large $\beta$ (weak regularization), where classical Sinkhorn methods become numerically unstable. Experiments show that the method works well on small scale problems.

### Strengths
- The paper directly addresses a known limitation of the Sinkhorn algorithm: numerical instability and slow convergence when $\beta$ is large. By introducing a proximal continuation scheme (IP-EOT, Algorithm 1), the authors ensure stable iterations even under weak regularization. This makes the contribution both timely and practically relevant for modern OT solvers.
- The convergence of the proposed algorithms are provided by the solid theoretical results.
- Experiments on small scale problems show the effectivity.

### Weaknesses
- The method’s performance depends on several hyperparameters such as the proximal step size $\Delta \beta$, sparsification threshold $\tau$, and switching parameter $\lambda$ (Sec. 4–5). These are chosen heuristically, with no ablation or sensitivity analysis provided. As a result, it is unclear how robust PSN remains under different settings or across diverse cost structures.
- All reported experiments are conducted on moderate-sized problems (n ≤ 4096), while modern OT applications often involve tens or hundreds of thousands of samples. Without results on larger datasets, the scalability and numerical stability of the proposed solver remain uncertain, especially considering the overhead of Hessian construction and preconditioning.
- The empirical study focuses mainly on synthetic linear assignment and MNIST benchmarks. No results are shown on more complex or real-world machine-learning applications such as domain adaptation, graph matching, or OT-based generative models. This limits the evidence that PSN can generalize beyond controlled test cases.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel **Proximal-Sinkhorn-Newton** (PSN) method for entropic optimal transport, designed to address the challenges of numerical instability and slow convergence. The approach employs a two-stage framework that integrates an inexact proximal point method with (optionally) a sparse Newton refinement step.
The algorithm can be summarized as follows:
- For a given regularization parameter $\beta$, construct a sequence of intermediate regularization values that increase linearly up to $\beta$, and obtain an approximate transport plan using the proposed IP-EOT method.
- Based on the characteristics of this approximate solution, refine it by applying either the sparse Newton method or exact Sinkhorn iterations.

### Strengths
**Linking the “Inexact” and “Exact” paradigms**: Theorem 1 provided a link between the result of IP-EOT and the exact EOT solution, which may benefit future EOT methods using inexact initialization.

**Verifiable numerical stability and efficiency**: Through numerical experiments, it is demonstrated that PSN can avoid underflow issues common in Sinkhorn under large $\beta$. Without the heavy overhead of log-domain and/or using exact Hessian, PSN can also outperform other 1st-order methods significantly.

### Weaknesses
**Limited comparison with existing methods**: Solving entropic OT with varying regularization parameters is not a new concept. For instance, [1] proposed a log-stabilized approach with exponentially scaled $\beta$, which is implemented as `sinkhorn_epsilon_scaling()` in the Python POT package and is easily accessible to practitioners. For a fair evaluation, the authors should include this method in their experiments.

Furthermore, as an ablation study, the authors could consider comparing IP-EOT with other approximate entropic OT methods, such as Screenkhorn [2].

**Lack of discussion on hyperparameters**: The PSN algorithm introduces several hyperparameters, yet the paper provides little guidance on how to select them. It would be helpful to include a discussion addressing questions such as: (i) how sensitive are IP-EOT and PSN to the choice of $\Delta\beta$? (ii) what is the impact of using exponential rather than linear scaling in IP-EOT? (iii) how should the truncation cap $\tau$ be chosen to avoid excessive sparsification? and (iv) how does excessive sparsification (Appendix H) affect the final solution?

**Unclear graph**: Figure 4(a) is intended to illustrate that IC preconditioners produce a more concentrated eigenvalue distribution, yet the plot appears to suggest the opposite. Only upon very close inspection can one notice the red points beneath the blue ones. This detail is easily overlooked and will be lost in print. The authors should use a clearer visualization method, such as histograms, to better convey their message.

**Minor issues/typos**:
- Line 380: missing a space between “log-Sinkhorn” and the reference after.
- Line 986: “Futhermore” should be “Furthermore”.
- Line 1133: “.com” in the footnote comes with an extra “m”.

**References**:

[1]: Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv:1610.06519.

[2]: Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019). Screening Sinkhorn Algorithm for Regularized Optimal Transport. arXiv:1906.08540

### Questions
The article demonstrates that IC preconditioners can substantially enhance the performance of Newton methods. I am interested in understanding under what conditions can such improvements be expected, and whether there is anything specific about the Lagrange function in equation (12) or the problem setup (e.g. assignment vs $L_1$ vs $L_2$) that facilitates these conditions. I understand these questions may be somewhat broad, so even general intuition or insights would be greatly appreciated.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces inexact proximal point methods(IP-EOT) and fast sparse newton method, and its integrated framework proximal sinkhorn newton(PSN) method to target entropic optimal transport(EOT). IP-EOT constructs numerically stable warm starts by increasing the regularization in small increments, mitigating underflow. The sparse Newton phase achieves global convergence via backtracking line search with Sinkhorn fallback, and enjoys fast local (near-quadratic) convergence near the solution. Each method shows low computational complexity ($O(n^2)$ per iteration) compared to other methods for EOT problem.  Empirically, PSN delivers faster and more stable convergence than competing EOT solvers across regularization regimes.

### Strengths
1. The paper introduces a proximal scheme for entropic OT with an **Inexact→Exact transition** (Theorem 1), splitting $\beta$ into $l$ small steps to avoid underflow/instability and warm-start refinement.
2. The Fast Sparse Newton combines sparsification that preserve positive definiteness, and a line-search with Sinkhorn fallback, yielding global convergence of Algorithm 2.
3. The numerical studies contain numerous simulations demonstrating its better convergence over other existing methodologies.

### Weaknesses
* Theorem 3 (Global Convergence) doesn’t specify its global convergence rate.
* There’s no theorem for the full PSN pipeline (Algorithm 3) that covers the switching logic (IP-EOT → Sinkhorn vs Newton). Theorem 3 is for Algorithm 2 only. 
* Experiments could better evidence applicability through various examples (e.g., synthetic cases with known ground truth like Gaussians/uniforms).
* Compared to unregularized optimal transport problem [1], entropic optimal transport suffers from high space complexity as well as high time complexity and its representation is blurrier due to regularization term. This framework is also expected to suffer this issue.

Other issues
* The choice of numerical setting of the main hyperparameters (eg. total regularization $\beta$) is not discussed.
* It would be better if the experiment includes evidences showing its excellence in time complexity over other EOT methods( eg. time per iteration).

[1] Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization. Kim et al, 2025

### Questions
* In section 3, I would be glad if you clarify how using $\Delta\beta = \beta/l$ leads to numerical stability.
* Does this have a potential to be used for solving Wasserstein barycenter problem, as was in IPOT [2]?
* In section 4, I would appreciate if you literally explain how adjustments affect the global convergence. And can you specify the convergence rate for theorem3?
* Is this method scalable to high dimensional settings such as multivariate distributions?

[2]  A Fast Proximal Point Method for Computing Exact Wasserstein Distance. Xie et al. 2020.

### Soundness
3

### Presentation
3

### Contribution
2
