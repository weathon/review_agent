# Scalable Second-order Riemannian Optimization for $K$-means Clustering

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Clustering is a hard discrete optimization problem. Nonconvex approaches such as low-rank semidefinite programming (SDP) have recently demonstrated promising statistical and local algorithmic guarantees for cluster recovery. Due to the combinatorial structure of the $K$-means clustering problem, current relaxation algorithms struggle to balance their constraint feasibility and objective optimality, presenting tremendous challenges in computing the second-order critical points with rigorous guarantees. In this paper, we provide a new formulation of the $K$-means problem as a smooth unconstrained optimization over a submanifold and characterize its Riemannian structures to allow it to be solved using a second-order cubic-regularized Riemannian Newton algorithm. By factorizing the $K$-means manifold into a product manifold, we show how each Newton subproblem can be solved in linear time. Our numerical experiments show that the proposed method converges significantly faster than the state-of-the-art first-order nonnegative low-rank factorization method, while achieving similarly optimal statistical accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the computational difficulty of solving the nonconvex, constrained formulation of K-means clustering. It reformulates the low-rank SDP relaxation as a smooth optimization over a Riemannian manifold, so that constraint feasibility is handled intrinsically. This allows the use of a cubic-regularized second-order Riemannian Newton method that guarantees convergence to second-order critical points—empirically aligned with global optima in K-means.

To make the approach scalable, the authors introduce a product-manifold factorization and exploit the Hessian’s structure to achieve linear-time per-iteration complexity. Experiments on Gaussian mixture and real-world datasets show faster convergence and lower mis-clustering rates than state-of-the-art first-order methods, demonstrating both theoretical soundness and strong empirical performance.

### Strengths
1. The paper successfully integrates second-order Riemannian methods, specifically the cubic-regularized Newton approach, into the K-means SDP framework. Moreover, The proposed product-manifold factorization resolves the computational bottleneck of expensive retraction operators in previous works.

2. Demonstrates consistent improvements in convergence speed and clustering accuracy over leading baselines, including NLR, RTR.

3. Sections 2 and 3 provides a concise review of SDP relaxations and information-theoretic thresholds. Theorem 1 and Theorem 2 are well-chosen to connect existing Riemannian convergence theory to this specific manifold formulation.

4. Reduces per-iteration complexity to linear in n, making second-order methods computationally competitive. The approach can be generalized to other manifold-constrained ML problems.

### Weaknesses
1. Assumption 1 (Benign Nonconvexity) remains unproven. The paper heavily relies on this assumption for its theoretical motivation but provides only empirical justification. A rigorous explanations or partial theoretical evidence would substantially strengthen the work.

2. In the numerical experiments, only a single real-world dataset (CyTOF) is used. Additionally, there is no ablation study on manifold dimension, sensitivity to initialization, or robustness to cluster imbalance.

3. The main theorems of complexity are adaptations of known Riemannian results rather than new convergence analyses specific to K-means.

4. The paper’s contribution seems somewhat incremental. It reformulates the K-means problem and employs a standard cubic-regularized Newton method, with the primary advancement lying in a more efficient computation of the Newton step by exploiting its structure.

### Questions
1. In line 295, how sensitive is the algorithm to the initial point $(V_0,Q_0)$?

2. Could this approach generalize to kernelized or probabilistic K-means variants? If so, how would the manifold and retraction structures adapt?

3. How is the second-order condition in Eq.(10) verified numerically? Does this influence empirical runtime?

4. As discussed in Appendix C.1, the Riemannian cubic-regularized Newton is equivalent to certain SQP methods. What are the computational advantages compared to projected Newton or augmented Lagrangian solvers?

5. In the proof of Lemma 4, the fourth equality is incorrect, but I believe it's just a typo and not a significant issue.

6. In Eq. (11), is the efficiency of the Hessian 1/2?

7. Typo on line 1127: this is not a quintic equation, but a quartic one.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new smooth, unconstrained Riemannian reformulation of a nonnegative low-rank factorization of the K-means SDP, enabling second-order optimization with feasibility preserved via retractions. The key idea is to map the original constraint set $\mathcal{M}$ onto a product manifold $\tilde{\mathcal{M}} = \mathcal{V}\times \mathrm{Orth}(r)$ via a submersion, which yields simple $O(nr + r^3)$ retractions and efficient expressions for Riemannian gradients/Hessians.

Building on this geometry, the authors design a cubic-regularized Riemannian Newton method that solves each Newton subproblem in $n\cdot\mathrm{poly}(r,d)$ time by exploiting a block-diagonal-plus-low-rank structure and a bisection scheme on the regularization parameter. The overall algorithm finds $\epsilon$–second-order points in $O(n\cdot\epsilon^{−3/2} \cdot\mathrm{poly}(r, d))$ time.

The work hinges on an empirical "benign nonconvexity" assumption: in regimes where the convex $K$-means SDP recovers ground truth, all approximate second-order critical points of the nonnegative low-rank model are near-global optima. The authors provide extensive empirical evidence for this behavior.

Experiments on synthetic GMMs and CyTOF data show that the proposed second-order method converges in hundreds of iterations, achieves similar or better clustering accuracy than the strongest first-order baseline (NLR), and substantially reduces time despite costlier iterations. It also outperforms prior Riemannian $K$-means methods and classical RTR/CG solvers that struggle with the log-barrier’s ill-conditioning.

The paper supplies detailed derivations: manifold geometry, LICQ verification, submersion proof, closed-form Lagrange multipliers for projections, efficient Hessian-vector products, feasible initialization (and necessity of $r > K$), and implementation details for the linear-time inner solves.

### Strengths
Conceptual advance: The submersion to a product manifold with simple retractions is elegant and practically impactful, removing the main bottleneck that hindered prior Riemannian approaches to $K$-means (expensive retractions and feasibility maintenance).

Algorithmic engineering: The cubic-regularized Riemannian Newton solver is carefully tailored—analytical gradients/Hessians, efficient tangent projections, and a block-diagonal-plus-low-rank exploitation for the inner linear systems. The bisection-based solver for the regularization parameter is simple and reliable.

Clear bridge to theory: The paper situates the contribution within the exact-recovery phase transition for the $K$-means SDP and explains how second-order points suffice under the benign nonconvexity hypothesis. The LICQ and manifold calculus are handled rigorously; the smoothing argument for the log penalty clarifies the use of second-order guarantees.

Strong empirical evidence: On both synthetic and real data, the method exhibits rapid, stable convergence to second-order points with near-zero mis-clustering; it consistently reduces iteration counts by orders of magnitude compared to NLR and RTR, translating to 2-4x faster runtimes despite more expensive steps.

Reproducibility and completeness: The paper provides explicit formulas, complexity accounting, initialization constructions, hyperparameter guidance, and ablations (rank $r$, penalty $\mu$, comparisons with several baselines), which make the contribution actionable for practitioners.

### Weaknesses
1. Assumption 1 (benign nonconvexity) is purely empirical in this work. While the authors motivate it with analogies to Burer–Monteiro and back it with experiments, the lack of any partial theoretical characterization (e.g., under separation/noise conditions and mild overparameterization $r > K$) limits the scope of the main claim in average-case regimes.

2. Dependence on the log-barrier: Although handled well algorithmically, the severe ill-conditioning induced by the log term drives both the design choices and some limitations (e.g., RTR/CG underperform). A discussion or experiment on alternative barrier/penalty designs (e.g., smooth-plus hard positivity via projections on $U$) could strengthen the case or show robustness.

3. Sensitivity to $\mu$ and feasibility interior: The method requires strictly positive $U$ and shows a phase transition when $\mu$ is too large. While the paper provides heuristics, a more systematic procedure (or adaptive schedule with safeguards) would make the solver more turnkey across datasets; also, the necessity of $r > K$ to ensure interior feasibility may be restrictive in memory-limited settings.

4. Scalability constants: The per-iteration complexity is linear in $n$ with $\mathrm{poly}(r, d)$ factors; however, the inner solves involve $r^3$ and $d$-dependent Schur complements. Scaling with varying $r$ and $d$ would help practitioners understand the limits and guide parameter choices.

5. Generality beyond GMM: Although the manifold formulation applies to kernelized $K$-means, empirical validation is focused on GMMs and one CyTOF setup. Broader tests (imbalanced/many clusters, higher $d$, other real datasets, kernels) would better support claims of robustness.

### Questions
1. Can you provide partial theory toward Assumption 1? For example, under the separation in Chen & Yang (2021b), and mild overparameterization $r = K + O(1)$, can you show absence of spurious second-order points in a neighborhood of the ground-truth factor, or establish a strict saddle property?

2. How robust is the method to mis-specified $K$ and to cluster imbalance? Could you include experiments where $K$ is under/over-estimated, and where cluster sizes vary significantly, and report both accuracy and convergence behavior?

3. Could you explore alternative penalties that retain positivity while improving conditioning (e.g., softplus smoothing, additive offsets, or barrier homotopy/continuation schedules), and compare convergence to your log-barrier?

4. What is the practical guidance for selecting $r$ beyond $r = K + 1$? Are there cases where larger $r$ improves optimization (escaping poor basins) or statistical robustness, and what is the runtime trade-off empirically?

5. Can you extend experiments to kernel $K$-means and non-Gaussian mixtures, or real vision/NLP datasets where the Gram matrix is built from learned features? This would help demonstrate the generality implied in Appendix A and the stated manifold framework.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the nonconvex low-rank factorization of SDP relaxation of K-means clustering. By penalizing the nonnegative constraints, the authors view the resulting problem as a manifold optimization problem and apply the Riemannian cubic regularized Newton to obtain a 2nd-order solution.

### Strengths
1. Develop a fast algorithm from manifold optimization viewpoint (Theorem 2), which is novel at least to me.
2. The subproblems of Riemannian cubic regularized Newton can be solved efficiently.
3. Empirically, 2nd-order critical points are globally optimal.

### Weaknesses
1. This paper only provides sublinear convergence rate. Is it possible to prove stronger local convergence rate results? For example, quadratic/superlinear convergence rate. Fig1 shows fast convergence rate in practice.
2. The benign nonconvexity is described in Assumption 1. Is it possible to say something about benign landscape? For example, related work of benign nonconvexity results.

### Questions
1. Is the part of lines 311-355 new result? If this is not new, it would be better to cite some references.
2. Could the authors explicitly explain why Riemannian cubic regularized Newton can overcome the ill-conditioning and RGD/RTR cannot?

### Soundness
3

### Presentation
4

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
This paper proposes a scalable second-order Riemannian optimization algorithm for solving the K-means clustering problem through a low-rank semidefinite relaxation. The authors reformulate the nonconvex problem as a smooth unconstrained optimization problem over a product manifold and apply a cubic-regularized Riemannian Newton method. The algorithm is claimed to achieve linear per-iteration complexity with provable convergence to second-order critical points.  

However, several issues remain. The key assumption (Assumption 1) seems inconsistent and lacks justification; the algorithmic novelty is limited.   If the authors can address my concern, I would raise my score.

### Strengths
- The mapping from the constraint manifold $\mathcal{M}$ to the product manifold provides a principled and novel way to represent K-means as a smooth manifold problem.   This reformulation reduces the projection cost from $O(n^2)$ to $O(nr + r^3)$.  
- By exploiting block-diagonal-plus-low-rank Hessian structure (Appendix E), each iteration scales linearly in \(n\) in the cubic-regularized Newton approach.  
- Use bisection search to solve the subproblem in the cubic-regularized Newton approach.

### Weaknesses
- Assumption 1 is difficult to interpret. The problem (1)  does not require \(U > 0\),  (U > 0\) enforced in problem (3), which is only a sufficient but not necessary condition for $Z = UU^\top > 0$. Hence, the assumption is logically inconsistent with the model definition.  There exists the case that the optimal solution $U_{ij}<0$ .  In addition, this assumption is verified only empirically (Fig. 1), with no analytical justification or example provided.  Since all global optimality claims rely on this assumption, its ambiguity weakens the theoretical contribution.

- The method essentially applies a standard cubic-regularized Newton algorithm to a reformulated manifold problem, which not  introduce new algorithm, or adaptive strategies, or theoretical improvements.

- Although the reformulation simplifies the constraint structure, computation of gradients and Hessians remains costly.  I recommend the authors discuss hybrid methods that separate simple manifold constraints (handled via Riemannian optimization) and other remained constraints (handled via augmented Lagrangian function).  Related work worth discussing includes:  

  - Wang, J., & Hu, L. (2025). Solving low-rank semidefinite programs via manifold optimization. Journal of Scientific Computing, 104(1), 33. 

  - Monteiro, R. D., Sujanani, A., & Cifuentes, D. (2024). A low-rank augmented Lagrangian method for large-scale semidefinite programming based on a hybrid convex-nonconvex approach. arXiv preprint arXiv:2401.12490. 
  -  Hou, D., Tang, T., & Toh, K. C. (2025). A low-rank augmented Lagrangian method for doubly nonnegative relaxations of mixed-binary quadratic programs. Operations Research.  
  -  Wang, Y., Deng, K., Liu, H., & Wen, Z. (2023). A decomposition augmented lagrangian method for low-rank semidefinite programming. SIAM Journal on Optimization, 33(3), 1361-1390.

- Important components—algorithm pseudocode, subproblem solvers, and retraction details—are placed only in the appendix.

### Questions
Why is the cubic-regularized Newton framework preferred? The original method (Agarwal et al., 2021) solves subproblems via *Lanczos iterations*, whereas this paper adopts bisection search. The authors should discuss those two approaches.

### Soundness
3

### Presentation
3

### Contribution
2
