# Closing the Performance Gap in Neural Conjugate Gradient Method: A Hybrid Multigrid Preconditioning Approach

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent studies on neural preconditioners for the conjugate gradient (CG) methods have shown promise but sometimes over-optimistic: methods targeting small-scale problems are compared with decomposition-based preconditioners but do not scale; those aimed at moderate- to large-scale problems are typically 5$\times$ slower in wall-clock time than state-of-the-art multigrid (MG) preconditioners, with worse iteration counts and higher model complexity. To address this gap, we revisit the designs of neural preconditioners and identify the key trade-off: methods tailored for scalability lack the expressiveness to emulate effective smoothers, whereas highly expressive designs struggle to scale. Building on these insights, we introduce a dual-channel neural multigrid preconditioner that couples a classical smoothing path with a lightweight neural convolutional path. This architecture preserves the minimal expressiveness and symmetric positive definite property while injecting data-driven adaptability. Our method demonstrates, for the first time, that neural preconditioners can surpass SOTA MG preconditioners on large-scale problems, achieving a 1.03-1.26$\times$ speedup on Poisson equations and 2-3$\times$ acceleration on other second-order PDEs involving up to 64 million unknowns, while also delivering 5-10$\times$ improvements over existing neural methods. These results establish a new benchmark for neural preconditioning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents an efficient neural preconditioner that outperforms existing SOTA neural solvers as well as geometric and algebraic multigrid preconditioned solvers on large scale linear systems arising from different physics problems. This so-called dual-channel neural preconditioner solver preserves symmetry and positive definiteness of conjugate gradient solver. The model is validated on Poisson equations, heat equations and Helmholtz equations.

### Strengths
- The results are very impressive, considering geometric multigrid as a strong baseline, and the speedup is demonstrated through extensive experiments. 
- This paper addresses the drawback of most neural solvers: unpredictable convergence rate. Since its framework is built upon a traditional multigrid V-cycle, it preserves the SPD of the V-cycle, and so convergence rate is guaranteed.
- It scales very well to large problems in various PDE problems.

### Weaknesses
- Since the neural operator is CNN-based, it does not take account of *sparsity* of a linear system. This model likely would not offer much benefit on large *sparse* linear systems. 
- This model is limited to first-order grid-cell discretization. It does not address other discretization schemes or non-structured grids. 
- This problem probably comes with any solver that involves down- and up-sampling: since coarsening halves size of the grid on each dimension, what would you do for a domain with odd dimension, eg, $511^3$.

### Questions
- Did you experiment with different kernel sizes for the convolution operators?
- What is your intuition on choosing *k-iteration* residual as your RHS and loss function to train your network? For one matrix, do you use a fixed k-iteration (hence one RHS vector) or various iterations (more than one RHS)?
- What is your definition of *expressiveness* for neural solvers?
- I need help understand why the neural smoothing convolution operator is *SPD*, especially why it is positive definite?
- What is the stop condition for training?
- Do you have a report of GPU usage when training your models? Does the model of $512^3$ fit on a single GPU?
- Is there any evidence that the proposed model actually improved *condition number* of the linear systems?
- Did you train a separate model for each scene? Does the model generalize to differene scenes?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors consider a problem of learning neural preconditioner for conjugate gradient method. The proposed neural preconditioner is a multigrid method with learnable smoother parametrising with the neural network in such a way that the resulting operator remains a linear iterative method.

The resulting method is successfully applied to Poisson, heat and Helmholtz equations.

### Strengths
Authors perform a set of large scale experiments in $D=3$. The number of degrees of freedom significantly exceeds typical experiments on learning and adaptation of multigrid methods making results more convincing and more realistic.

### Weaknesses
Many similar results are available in literature, so novelty of the proposed approach (besides the scale of experiment) is questionable. Besides that, there may be issues with diversity of train and test data.

### Questions
**Novelty**

Authors describe their research as completely novel: "the first demonstration that a neural preconditioner can surpass highly optimized MG methods on systems with tens of millions of variables", "we have shown that the carefully tuned geometric MG preconditioners
remain strong baselines, but our method surpasses them for the first time", etc. There are dozens of related works that target similar problem--improvement of multigrid components given some family of equations:
1. Multigrid acceleration techniques and applications to the numerical solution of partial differential equations, J Zhang. Application of adaptive multigrid that computes relaxation parameters based on the performance of the method.
2. Automating the Design of Multigrid Methods with Evolutionary Program Synthesis, https://arxiv.org/abs/2312.14875. Design of multigrid algorithms (including preconditioners) with evolutionary algorithms.
3. Learning to Optimize Multigrid PDE Solvers, https://arxiv.org/abs/1902.10248. Direct learning of restriction and prolongation operators of geometric multigrid for parametric PDE problems.
4. **Mentioned by authors.** Learning algebraic multigrid using graph neural networks, https://arxiv.org/abs/2003.05744. Work by the same group but this time restriction and prolongation operators are learned from algebraic multigrid.
5. Neural Multigrid Architectures, https://arxiv.org/abs/2402.05563. Learning restriction, prolongation operators and parameters of smoothers for geometric multigrid.
6. Deep Multigrid: learning prolongation and restriction matrices, https://arxiv.org/abs/1711.03825. Optimisation of restriction and prolongation matrices.
7. **Mentioned by authors.** Learning optimal multigrid smoothers via neural networks, https://arxiv.org/abs/2102.12071.

Authors downplay several contributions mentioned above based on the small scale problems ($\simeq 10^{5}$ variables) considered in these publications. I find this framing unfair: (i) in several publications, e.g., [3], [4], [5] authors train on problems of specific size and confirm that the method performs well for problems of larger size; (ii) evaluation on small size still enough as a proof of concept.

Can the authors explain why they believe their experiments are sufficiently novel and interesting when many very similar contributions are available in the literature?

**Training details**

In Appendix E authors summarise training details. The description here is fragmentary, so I have many questions.

*Details on equations, discretisations, etc*

1. What is the pressure Poisson problem precisely? Can the authors provide details on boundary conditions, discretisation, integrators they use for Navier-Stokes or other equations they used to model fluid flow?
2. Similarly, it is not clear how the heat equation is discretised, what thermal conductivity used, etc. Can the authors provide more details?
3. The same goes for the Helmholtz equation. Please, specify boundary conditions, discretisation, how wave numbers were selected, etc.

*The data on the number of train and test instances is unusually small.*

The number of train and test data is reported in appendix and in the main text

| Equation  | Dataset size                              | Training time |
| --------- | ----------------------------------------- | ------------- |
| Poisson   | 2000 equations, 50 for train              | 2 days        |
| Heat      | 300 equations, 10 for train, 290 for test | 10 hours      |
| Helmholtz | 300 equations, 10 for train, 290 for test | 2 hours       |

1. It is not entirely clear whether authors report the number of equations of the form $Ax = b$, or the number of rollouts of numerical integrators for evolution PDE. Can the authors clarify this? In addition if the later or rollouts is reported, can the authors provide the number of linear systems $Ax = b$ the train on?
2. The dataset sizes are unusually small, and it makes it unclear how well neural networks trained on such problems are going to generalise. I kindly ask the authors to explain why the training data is sufficient and what is a practical significance of selected training setup.
3. Training time is clearly not great, likely owing to the unusual gradient-free optimisation strategy. It is clear that overall the approach is not going to scale well, when more diverse datasets are considered. Can the authors comment on that?

**Helmholtz equation**

Vanilla multigrid is not efficient as a preconditioner for the Helmholtz problem. The search for good preconditioners is still active for the Helmholtz equation with many different options proposed in the last 30 years. A notable strategy is to apply complex shift to the Helmholtz operator:
1. On a class of preconditioners for solving the Helmholtz equation YA Erlangga, C Vuik, CW Oosterlee
2. How large a shift is needed in the shifted Helmholtz preconditioner for its effective inversion by multigrid? PH Cocquet, MJ Gander
3. Preconditioning Helmholtz linear systems, D Osei-Kuffuor, Y Saad
There are also other strategies, e.g., sweeping preconditioner, perfectly matched layers. Did the authors consider preconditioners specialised for the Helmholtz equation? Why did authors decide to compare with multigrid, which is known to perform poorly for this particular problem?

**Smoothers and parallelisation**

Authors claim that GS smoother is highly parallel. This is not the case. In fact, GS smoother is hard to parallelise and because of that polynomial smoothers (Chebyshev) are often preferable, see Parallel multigrid smoothing: polynomial versus Gauss–Seidel, https://www.sciencedirect.com/science/article/abs/pii/S0021999103001943. It may be the case that authors mean a specific read-black Gauss-Seidel smoother, which is actually a Jacobi method applied on a staggered grid.

Polynomial smoothers also potentially form a stronger baseline for geometric multigrid method, see Local Fourier Analysis of Multigrid Methods with Polynomial Smoothers and Aggressive coarsening, https://arxiv.org/abs/1310.8385.

It would be helpful if authors could report the performance of polynomial smoothers (with appropriate prolongation and restriction) for spd matrices in their benchmarks.

**Minor points**

1. Authors reference (Nocedal & Wright, 2006) for CG and (Evans, 2022) for PDE discretisation. I find both these references to be not entirely relevant: (i) for CG it is easy to find original paper where the algorithm was introduced; (ii) the book of Evans is not on numerical solution of PDEs.
2. Algorithm 2 in Appendix D.
   a. Why does post-smoothing directly follow pre-smoothing? In the multigrid scheme it is applied after coarse grid correction.
   b. Why is there no computation of residuals in the whole algorithm? In the current version $x^{(l)}$ is computed from smoother and never used after that.

### Soundness
3

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
The paper introduces a dual-channel neural multigrid preconditioner. This hybrid approach is designed around a dual-channel expression of the V-cycle. The authors propose to integrate a classical smoothing path with a lightweight, data-driven neural convolutional path. The proposed approach achieves a wall-clock time speedup of $1.03 - 1.26 \times$ over the GMG baseline for Poisson equations and a $2 - 3 \times$ acceleration on second-order PDEs (Heat and Helmholtz equations). It demonstrates a $5 - 10 \times$ improvement over MLPCG. The authors construct a large-scale benchmark (linear SPD systems up to $64$M unknowns) to fairly evaluate methods.

### Strengths
1. The authors propose a dual-channel neural multigrid preconditioner that reframes the standard Multigrid V-cycle to inject a lightweight neural convolutional path while strictly preserving the necessary SPD property.
2. The idea of rigorously benchmarking against a GMG solver is a significant original step in execution.
3. The experiments are conducted on a new large-scale dataset (linear SPD systems up to $64$M unknowns), demonstrating the method's ability to scale effectively to real-world scientific problems. The results are presented clearly, allowing for direct comparison to the strong GMG baselines.

### Weaknesses
1. The use of the CMA-ES optimizer is a significant practical drawback. The paper cites the reason: "challenging computational graphs for auto-differentiation tools," which suggests a fundamental design limitation. Gradient-free methods typically scale poorly to the high-dimensional parameter spaces common in deep learning, and they limit the ability to leverage standard, highly-optimized deep learning frameworks (e.g., PyTorch, TensorFlow).
2. The article does not answer the fundamental question: why is the computational graph intractable for standard backpropagation?
3. The performance gains on the second-order PDEs (Heat and Helmholtz equations), which feature more complex physics and operators, are less discussed in depth compared to the Poisson speedup. Helmholtz equations, in particular, are difficult for classical Multigrid due to their frequency content.

### Questions
See the Weaknesses

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
4

### Summary
The paper proposes a hybrid multigrid preconditioner for PCG that runs a lightweight learned path (H) alongside a classical smoothing path (S). Using an equivalent V-cycle formulation, the authors argue the overall preconditioner remains SPD, preserving PCG guarantees. Training minimizes the k-step CG residual (self-supervised) with a derivative-free optimizer. On large regular-grid Poisson/Heat/(SPD) Helmholtz problems, the method reports GMG-like per-iteration cost with fewer iterations, outperforming prior neural preconditioners.

### Strengths
Consistent preservation of SPD guarantees.
By formulating the preconditioner as an equivalent V-cycle with parallel S/H paths, the overall operator is kept symmetric positive definite, thereby preserving PCG’s convergence guarantees while introducing learnable components. This is a practically critical property in this domain.

Well-tuned GMG as the primary baseline.
The comparisons are made against a carefully engineered geometric multigrid (GMG) rather than a textbook/default setup, so the results are meaningful regardless of win/loss and the risk of overstating gains is reduced.

Self-supervised objective (no labels required). By optimizing the k-step residual as the training objective, the method avoids ground-truth solutions and other costly supervision. This is practical for physics applications where data preparation is often the bottleneck.

Classical S-path as a safety net (robustness foundation). When the learned H-path underperforms out of distribution, the classical MG S-path still ensures baseline convergence. This delivers a clear risk profile—learning provides upside; classical MG guarantees the floor—which is appealing for real deployments.

### Weaknesses
1. Scope limitation. Experiments focus on regular-grid SPD elliptic problems (Poisson/Heat/(SPD) Helmholtz) whose spectra are closely related; effectiveness is shown within this family, not beyond it.
2. Out-of-distribution (zero-shot) is thin. Systematic tests where RHS/coefficients/BCs deviate substantially from the training distribution are limited. Results on an unseen scene (e.g., Worm) remain in-family and do not establish cross-family generalization.
3.Boundary-dominated settings (still SPD) are untested. No evaluation with obstacles/rotating bodies/cut-cell or immersed-boundary style setups where irregular boundary stencils dominate; robustness near boundaries is unclear.
4. Resolution generalization is limited. There is an extrapolation 256³→512³ for Poisson, but no systematic sweeps for Heat/(SPD) Helmholtz, nor train@low → test@high cross-resolution tests.
5. Comparison protocol gap. Prior work shows “train without objects → zero-shot to scenes with objects.” In contrast, this paper does not provide a comparable boundary-dominated protocol, weakening claims of geometric/BC diversity.

### Questions
1. Training cost & amortization. Please report CMA-ES trial counts, per-trial wall-clock, hardware, and an estimate of train-once-solve-many payback.
2. Boundary-dominated (SPD) tests. On obstacle Poisson and rotating-body Poisson (3D 256³), can you report iterations, wall-clock, peak memory, boundary-zone (≤2 cells) residual decay, and preconditioned condition number?
3. Cross-resolution. What are the degradation ratios for train@256³ → test@512³/384³ (iterations, time, condition number)? Is retraining necessary?

### Soundness
4

### Presentation
4

### Contribution
2
