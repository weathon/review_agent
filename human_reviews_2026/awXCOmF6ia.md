# Zeroth-Order Methods for Stochastic Nonconvex Nonsmooth Composite Optimization

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
This work aims to solve a stochastic nonconvex nonsmooth composite optimization problem. Previous works on composite optimization problem requires the differentiable part to satisfy Lipschitz smoothness or some relaxed smoothness conditions, which excludes some machine learning examples such as regularized ReLU network and sparse support matrix machine. In this work, we focus on stochastic nonconvex composite optimization problem without any smoothness assumptions. In particular, we propose two new notions of approximate stationary points for such optimization problem (one stronger than the other) and obtain finite-time convergence results of two zeroth-order algorithms to these two approximate stationary points respectively. Finally, we demonstrate that these algorithms are effective using numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes new approximate stationary points and zeroth-order stochastic algorithms for solving the stochastic nonconvex nonsmooth composite optimization problem.

### Strengths
1. This paper introduces novel approximate stationary points by utilizing the Goldstein $\delta$-subdifferential for the nonsmooth stochastic composite optimization problem.

2. This paper proposes novel zeroth-order stochastic methods with an improved convergence rate than existing work [1].

**References**

[1] Liu, Z., Chen, C., Luo, L., & Low, B. K. H. (2024, July). Zeroth-order methods for constrained nonconvex nonsmooth stochastic optimization. In Forty-first International Conference on Machine Learning.

### Weaknesses
1. My primary concern regarding this work is that it appears to be a straightforward extension of the previous study [1], encompassing approximate stationary points and stochastic algorithms. As a result, the contribution of this work seems incremental, and its novelty is limited.

2. Since the objective function $\phi(x)$ is the sum of two nonsmooth functions, can I just apply the zeroth-order unconstrained stochastic method introduced in [2] on $\phi(x)$ to achieve a convergence rate of $\mathcal{O}(d \delta^{-1} \epsilon^{-3})$.  

**References**

[1] Liu, Z., Chen, C., Luo, L., & Low, B. K. H. (2024, July). Zeroth-order methods for constrained nonconvex nonsmooth stochastic optimization. In Forty-first International Conference on Machine Learning.

[2] Kornowski, G., & Shamir, O. (2024). An algorithm with optimal dimension-dependence for zero-order nonsmooth nonconvex stochastic optimization. Journal of Machine Learning Research, 25(122), 1-14.

### Questions
See weakness 2.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies stochastic nonconvex nonsmooth composite optimization problems. The key contributions are: (i) Two new notions of approximate stationarity: ($\gamma$, $\delta$, $\epsilon$)-proximal Goldstein stationary points (PGSP) and ($\delta$, $\epsilon$)-conditional gradient Goldstein stationary points (CGGSP) which generalize Goldstein stationary points to composite objectives. (ii) Two zeroth-order algorithms: zeroth-order proximal gradient descent and zeroth-order generalized conditional gradient methods. Both methods achieve finite-time convergence guarantees to the above approximate stationary points, with and without variance reduction. (iii) Improved complexity bounds upon prior work such as Liu et al. (2024). (iv) Empirical validation on a regularized ReLU network illustrating practical convergence of both methods.

### Strengths
1. The definitions of PGSP and CGGSP extend Goldstein-type stationarity to nonsmooth composite settings, which had no tractable finite-time criteria before.

2. The convergence and complexity proofs are rigorous, connecting zeroth-order smoothing and nonsmooth analysis.

3. The paper clearly relates its framework to proximal methods, conditional gradient methods, and previous Goldstein-stationary notions.

4. Although small-scale, the experiments demonstrate that the methods work as claimed and variance reduction indeed accelerates convergence.

5. The exposition is organized and self-contained, with detailed assumptions, propositions, and proofs.

### Weaknesses
1. Only a toy ReLU network example ($d=34$) is shown. There is no comparison with baselines (e.g., stochastic subgradient, first-order PGD, or other zeroth-order methods).

2. While the theory is clean, it is not obvious how these algorithms perform in high-dimensional machine-learning applications.

3. The paper could benefit from clearer motivation and intuition before diving into technicalities.

4. The comparison to contemporary zeroth-order nonconvex optimization papers (e.g., Cutkosky 2023) could be deepened.

### Questions
1. Could the proposed stationarity notions be extended to settings where $h$ is nonconvex but prox-friendly?
2. How sensitive are the algorithms to the smoothing radius $\delta$ in practice?
3. Is there any connection between PGSP and the weak subgradient mappings used in Clarke’s generalized gradients?
4. Have the authors tried larger-scale tasks (e.g., CIFAR or low-rank matrix problems) to test scalability?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies zeroth-order methods for stochastic nonconvex nonsmooth composite optimization, proposes PGSP and CGGSP as approximate stationary notions, and analyzes 0-PGD and 0-GCG algorithms with minibatch and variance-reduction gradient estimators, giving finite-time complexity bounds. Experiments on synthetic two-layer ReLU and ResNet-20 validate algorithmic behavior.

### Strengths
- The authors extend the notion of constrained stationarity to general nonsmooth composite settings by replacing the projection operator with a proximal mapping. This yields a unified definition applicable to many practical problems. Two zeroth-order algorithms are proposed to handle different oracle settings (proximal vs LMO). The convergence analyses and intractability results are presented clearly.
- The paper is generally well-written, logically organized, and easy to follow. Definitions and assumptions are stated clearly, and the appendix provides detailed proofs.

### Weaknesses
-  While the definition of PGSP extends previous constrained stationary notions to the nonsmooth setting, this extension is **rather straightforward** — essentially replacing the projection operator in the constrained case by a proximal operator. Similarly, the algorithmic framework closely follows that of **Liu et al. (2024)** for the constrained Lipschitz case, with minor modifications. 
- Although the paper claims an improvement in complexity bounds, the improvement is only in the **parameter dependence** , which is mainly due to the tight bound for $F_\delta(x_0) - F_\delta(x_T)$ (as mentioned in **Comparison with Constrained Optimization**), while the overall order of complexity remains identical.  For the tight bound, I think the new term $\psi - \psi_*$ introduced in the analysis may weaken the claimed improvement, and it is unclear whether this scaling is indeed tight or essential. The resulting theory, though consistent, does not introduce fundamentally new mathematical tools or algorithmic ideas. A detailed side-by-side comparison of assumptions, complexities, and definitions would strengthen the contribution.
-  The complexity bounds in Table 1 do not explicitly include $\gamma$. If the complexity is indeed independent of $\gamma$, does that imply the same rate holds for any $\gamma$? This point needs further explanation, as $\gamma$ appears both in the proximal mapping and smoothing radius, and typically affects the variance–bias trade-off.

### Questions
Please clarify whether the derived iteration and query complexities depend on $\gamma$.
 If not, why does $\gamma$ appear in the proximal update?
 Intuitively, the step size affects both convergence and stationarity precision — this should be explicitly reflected in the bounds.

$\delta$ appears both in the smoothing process and the stationarity definition. How should δ be chosen in practice? What happens when δ is too small — does the variance term blow up?

### Soundness
3

### Presentation
3

### Contribution
2
