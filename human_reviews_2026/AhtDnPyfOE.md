# (U)NFV: (Un)Supervised Neural Finite Volume Methods for Solving Hyperbolic PDEs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We introduce (U)NFV, a modular neural network architecture that generalizes classical finite volume (FV) methods for solving hyperbolic conservation laws. Hyperbolic partial differential equations (PDEs) are challenging to solve, particularly conservation laws whose physically relevant solutions contain shocks and discontinuities. FV methods are widely used for their mathematical properties: convergence to entropy solutions, flow conservation, or total variation diminishing, but often lack accuracy and flexibility in complex settings. Neural Finite Volume addresses these limitations by learning update rules over extended spatial and temporal stencils while preserving conservation structure. It supports both supervised training on solution data (NFV) and unsupervised training via weak-form residual loss (UNFV). Applied to first-order conservation laws, (U)NFV achieves up to 10x lower error than Godunov's method, outperforms ENO/WENO, and rivals discontinuous Galerkin solvers with lower implementation burden. On traffic modeling problems, both from PDEs and from experimental highway data, (U)NFV captures nonlinear wave dynamics with significantly higher fidelity and scalability than traditional FV approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript proposes the U-NFV framework, which employs neural fields to solve PDEs under both supervised and unsupervised settings. Existing neural fields depend on labeled data (supervised) or require heavy reliance on physics-informed losses (unsupervised). U-NFV introduces a variational loss functional to allow for both training settings, a neural functional operator that maps input coordinates to PDE residuals and boundary losses, and a hybrid optimization strategy which combines direct residual minimization (unsupervised) with data-driven regression (supervised). Experimental results show that U-NFV achieves SOTA accuracy across both data-rich and data-sparse settings, and exhibits continuous resolution generalization.

### Strengths
- **Technical novelty**: The proposed general neural variational operator is compatible with both data-driven and physics-based objectives.
- **Extensive experiments**: The proposed method is tested across both linear and nonlinear PDEs, multiple spatial dimensions, and diverse boundary conditions. The experimental results show SOTA-comparable accuracy. 
- **Resolution generalization**: Neural field representations enable solving PDEs at unseen spatial resolutions without retraining.
- **Mathematical backbone**: The proposed method links to the Ritz-Galerkin method and classical variational formulations.

### Weaknesses
- **No efficiency comparisons**: Metrics like training time, inference time, and memory consumption are missing. 
- **Scalability**: Experiments are limited to PDEs with moderate grid sizes. The framework’s computational feasibility for large-scale 3D problems is not discussed.
- **Uniform sampling**: Uniform point sampling is used for both loss variants. Whether the method allows for adaptive residual sampling is unclear.

### Questions
- Please refer to the "Weaknesses" section above.

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
This paper introduces Neural Finite Volume, a neural network architecture that generalizes classical finite volume methods for solving hyperbolic conservation laws. The key innovation is learning numerical flux approximations over extended spatiotemporal stencils while preserving the conservation structure of traditional FV methods. The authors develop both supervised and unsupervised variants, demonstrating up to 10× improvement over Godunov's method on benchmark PDEs and showing practical applicability to real highway traffic data from the I-24 MOTION dataset.

### Strengths
1. The proposed NFV framework provides a method to incorporate neural networks into FV methods while maintaining conservation.
2. The application to real traffic data seems valuable and shows practicality.
3. The comparisons with classical methods provide useful context
4. Table 1 shows consistent improvements across multiple PDEs

### Weaknesses
1. the method is restricted to 1D conservation laws. Although the authors claim that "NFV architecture is, in principle, extendable to higher dimensions", it has not been demonstrated in the paper and may introduce significant challenges.
2. The comparisons are limited to classical numerical methods. It would be significantly beneficial if the authors could compare their methods to some of the recently proposed neural network based method, (e.g. https://proceedings.mlr.press/v235/chen24ad.html, https://arxiv.org/abs/2410.22193, https://proceedings.mlr.press/v202/muller23b.html). If the comparison is difficult, maybe the authors can comment more on how their method differs from these methods and how their method may be advantageous?
3. When does NFV fail? The paper shows it works well on the tested cases but doesn't explore boundary conditions beyond periodic, complex geometries, or cases with strong shocks.
4. While Table 3 shows NFV variants with increasing complexity, the paper doesn't discuss the training time or data requirements. How many Riemann problems are needed? How long does training take compared to one-time use of a classical method?

### Questions
1. How would NFV handle non-periodic boundary conditions (Dirichlet, Neumann, open boundaries)?
2. How does the method compare to some of the other NN-based methods as mentioned above?
3. Is it possible to conduct ablation studies on architecture choice, stencil size selection, or the training data size requirements?

### Soundness
3

### Presentation
3

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
The paper proposes (U)NFV, a neural finite volume (FV) framework for 1D hyperbolic conservation laws, that learns numerical fluxes on extended space-time stencils, while keeping the classic FV conservation update, so mass is conserved by construction. It comes in two flavors: NFV (supervised) trained on solution data, and UNFV (unsupervised), trained with a weak form residual loss, targeting entropy solutions without labeled data. The main contributions are: (1)  (U)NFV is a conservation preserving neural FV generalization, (2) supervised and weak form unsupervised training options, (3) strong accuracy vs. FV/ENO/WENO and DG-level performance with far simpler implementation, (4) successful application to noisy, non-ideal field data, and (5) empirical convergence with notes on parallel theoretical guarantees to appear. In summary, it is a practical, accurate, and physics respecting route to data driven solvers for hyperbolic conservation law, which is useful in diverse areas, from scientific computing to real world systems like traffic.

### Strengths
(i) The method keeps the classic FV update, so mass conservation is guaranteed, while learning the numerical flux on extended space–time stencils, (ii) It supports both supervised NFV and unsupervised UNFV, (iii) (U)NFV reports up to 10× lower error than classical FV, and matches DG level accuracy without DG’s complexity. On I-24 MOTION field data, NFV outperforms calibrated Godunov baselines, and handles noisy conditions. Empirical refinement studies suggest convergence, and models trained on simple Riemann problems generalize to more complex initial conditions. A small CNN makes large stencils practical and fast, avoiding the hand-crafted complexity of high-order schemes.

### Weaknesses
(i) All experiments are on 1D, first-order scalar conservation laws. Multi-dimensional problems and systems, are left for future work, and acknowledged to bring extra stability/complexity challenges, (ii) The UNFV weak form loss does not guarantee convergence to the entropy solution, and the success is reported empirically only. Formal guarantees are deferred, and are not contained here, (iii) Training/rollouts are most reliable at CFL ≈ 0.5; higher CFLs sometimes work but are less reliable, and CFL=1.0 degrades on Triangular flux, which is stricter than typical Godunov stability limits.  NFV solutions can show spurious oscillations, even when Godunov is monotone/TVD, both on synthetic PDE tests and in field-data rollouts.  Supervised models are trained predominantly on Riemann problems, which is a strong but narrow distributional assumption. The weak loss uses 250 random polynomials of degree 50, and performance may be sensitive to such design choices.

### Questions
Suggestions: (i) State, or at least sketch, the promised convergence guarantees for (U)NFV. Add an entropy consistency discussion and a discrete mass/entropy error monitor across rollouts, (ii) For UNFV, analyze identifiability of the weak loss, and report sensitivity to the number/degree/family of test functions.
Run a CFL sweep and report stability/accuracy curves (CFL ∈ [0.2,1.2]) across fluxes, and quantify failure modes. (iii) Add a 2D case,  as a short appendix experiment, to show that the method survives beyond 1D scalars.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a neural method to solve hyperbolic PDEs. The core method is called neural finite volume (NFV), and is inspired by the finite volume methods that attempt to satisfy local conservation of the appropriate variables. The unsupervised variant of this method uses the basic concept of the update rule of a generic finite volume method. Thus, the loss function is the squared norm of the residual that is obtained from this update rule. This loss is then minimized to train the network. The method is applied on two benchmark problems, namely, the traffic flow equation (many variants) and the Burgers' equation, with competitive results against classical finite volume methods. Finally, a supervised variant of NFV is also presented, trained on a real dataset.

### Strengths
* The paper is well-written, with ideas clearly conveyed.

* The method is shown to be yielding lower errors compared to some of the classical methods. The chosen equations are in one dimension, and have one unknown. But they are usually the standard places to start.

* Numerical convergence results are presented, showing the behavior of the errors with respect to the time-step size.

### Weaknesses
* One of the flaws in this paper is that the neural network design is not discussed enough, or relegated to the appendix. While it is understandable that, a significant portion of this design is heuristic, it is still important to make an attempt to synthesize that knowledge, especially when one is trying to solve scientific computing problems.


* While the comparisons with the classical methods are essential, it is also very important to compare a new method with existing methods that are not classical. Currently, no comparisons are made with other existing neural network based methods, especially in the unsupervised regime.

* The paper does not flesh out the limitations of this method. What happens when a rarefaction happens? Does the method select the correct weak solution? The authors do mention that imposing the $L_2$ norm loss does not guarantee the entropy solution, but remark that most of the numerical experiments show that the method selects of the entropy solution. Such a remark only pertains to the few cases considered in the current work. 

$\textbf{Minor}$

* No equation number on some equations, e.g., line 236 ($L_w$).

### Questions
* My understanding is that the proposed method is intended to solve only one instance of an equation, correct?

* It seems to me that, we need to solve an optimization problem at every time step. Is that correct? In that case, this method is not very competitive in terms of time to solve. What do the authors think?

* My understanding is that, to reach $u^{\Delta t}$ from $u^0$, one uses the neural network to predict the fluxes, and then update the solution, and iteratively solve the optimization for this one time step, and obtains a good approximation of $u^{\Delta t}$ at the end. And then, one proceeds to calculate $u^{2\Delta t}$. Is that correct? I think, a flow chart would be a good addition.

* The authors remark that it is very easy to implement this method compared to the classical methods. Could the authors elaborate on this? How easy is this method? In what way? Usually, in a conventional code, the important pieces are the correct differentiation rules / stencils, communication between values in the volume and the values on the facets, and implementing the special adjustments at the boundaries, among many other aspects. How does NFV fare on these considerations?

* Also, by the virtue of choice of the example PDEs, I am assuming that the authors did not need to calculate the spatial derivatives of the primal variables during the training process. That is, both the $t-$ and the $x-$ derivatives are integrated away, and the loss functional $L_w$ does not contain any spatial derivatives of the function $u$. Is that correct?

### Soundness
2

### Presentation
2

### Contribution
2
