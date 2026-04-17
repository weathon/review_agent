# Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 6

## Abstract
We introduce an output layer for neural networks that ensures satisfaction of convex constraints. Our approach, $\Pi$net, leverages operator splitting for rapid and reliable projections in the forward pass, and the implicit function theorem for backpropagation. We deploy $\Pi$net as a feasible-by-design optimization proxy for parametric constrained optimization problems and obtain modest-accuracy
solutions faster than traditional solvers when solving a single problem, and significantly faster for a batch of problems. 
We surpass state-of-the-art learning approaches by orders of magnitude in terms of training time, solution quality, and robustness to hyperparameter tuning, while maintaining similar inference times. Finally, we tackle multi-vehicle motion planning with non-convex trajectory preferences and provide $\Pi$net as a GPU-ready package implemented in JAX.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces \PiNet, a neural network architecture that guarantees hard constraint satisfaction by projecting the network’s output onto a convex feasible set using operator splitting (Douglas–Rachford iterations) in the forward pass and applying the implicit function theorem (IFT) for efficient differentiation in the backward pass. ΠNet acts as a feasible-by-design implicit layer, enabling neural networks to directly solve parametric constrained optimization problems. The method is further applied to multi-vehicle motion planning, showcasing constraint satisfaction, parallelizability, and flexibility to optimize arbitrary differentiable objectives. A JAX implementation is provided for GPU-ready deployment.

### Strengths
ΠNet introduces a principled mechanism for embedding orthogonal projection layers into neural networks, rigorously ensuring constraint satisfaction during both training and inference.

The layer can be seamlessly attached to any backbone network, transforming unconstrained predictors into feasible solution generators for convex programs, PDEs, and motion-planning tasks.

### Weaknesses
ΠNet cannot directly handle non-convex or mixed-integer constraints; while the authors mention sequential convexification as future work, this remains a strong limitation for broader adoption.

While convergence of the projection layer is discussed, formal proofs of differentiability stability and error propagation across finite iterations are missing.

The multi-vehicle experiment is insightful but limited. No comparison is made with structured planners (e.g., QP-based MPC, or differentiable MPC variants).

### Questions
How does ΠNet perform relative to OptNet or Theseus, which also differentiate through optimization problems?

How many Douglas–Rachford iterations are typically needed for practical convergence? Is there a risk of slow progress or oscillation for poorly conditioned problems?

In multi-vehicle planning, each vehicle’s constraints are decoupled. How would ΠNet behave when inter-agent constraints (e.g., collision avoidance) are included?

The paper claims strong insensitivity to tuning—could you provide more quantitative evidence or stress tests across random seeds and scaling factors?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Πnet, a neural network architecture that generates provably feasible solutions for parametric constrained optimization problems. The core innovation is a projection layer that maps a backbone network's raw output onto a convex feasible set using the Douglas-Rachford algorithm, guaranteeing hard constraint satisfaction by design. A key advantage is its efficient training via the implicit function theorem for backpropagation, avoiding the computational cost of unrolling the optimization.

### Strengths
1)  А mathematically sound appendable layer with a robust Forward pass formulation and locally correct and heuristically consistent Backward pass. 
2)  Empirical results are meaningful and appear to be reproducible. The ablation study is sufficient. 
3)  Even with approximate or subgradient differentiation, the forward pass is a valid projected inference scheme. Douglas–Rachford ensures convergence under mild conditions.
4)  The backward pass gives a practical, stable gradient proxy. Empirically, autodiff through a few DR steps often works and trains faster than unrolled solvers.
5)  The extra projection layer adds no extra parameters, and only a moderate computational overhead—roughly 2–3x vs an unconstrained MLP, far less than unrolled solvers or cvxpylayers.

### Weaknesses
My main concern is the differentiability during training passes, which may make the Pinet functional only locally.  Please review the mathematical assumptions required for the method to work:
-- C(x) convex, closed, and with a fixed active constraint set around each training point (no switching during differentiation);
-- A(x) full row rank;
-- Φ(s,y) differentiable in a neighborhood of the fixed point (piecewise but locally nonsingular (local contraction);
It is my understanding that only then the implicit-function theorem can apply locally, and the gradient formula (Eq. 8) would become correct within that region. 
Likewise, in appendix, no derivations of the explicit Jacobians are provided. For the IFT to apply, the map $\Phi(s, y_{\text{raw}})$ must be differentiable. The authors never show $\partial_s \Pi_A$ or $\partial_s \Pi_K$.

The authors also start from $s_\infty = \Phi(s_\infty, y_{\text{raw}})$ and derive $$(I - \partial_s \Phi)^\top \xi = v, \quad v^\top \partial_{y_{\text{raw}}} \Phi$$, which is algebraically correct; however, the solution might diverge if $I - \partial_s \Phi$ is singular (non-invertible), and the numerical solution via Bi-CGSTAB may be wrong. Also, if $\Phi$ is built from non-smooth proximal operators it is obviously not $C^1$. Please clarify whether the solution is stable and unique. 

Lines 161–163: The projection map is non-differentiable at points where the active constraint set changes. The paper later differentiates this function as if it were $C^1$. So, one needs to clarify that it is Lipschitz but not differentiable everywhere; and the gradients are defined only almost everywhere or as subgradients.

Lines 173–179: Projection commutativity is assumed without proof. Derive or restrict A,K so that projection decomposition preserves convex geometry; otherwise (2) does not yield the same minimizer as (1).

Eqs. (3a–3c): replace “strict feasibility” with “non-empty intersection and standard monotonicity assumptions.”, because DR requires only non-empty intersection of closed convex sets. Also, include the σ scaling in prox definitions (currently inconsistent with (3a, 3b)). 

Lines 218–228: IFT application will be valid only where projection is differentiable (interior or non-degenerate active set). This assumption is implicit and needs to be discussed.

Recommendations: 

•  Replace global claims (“guaranteed differentiability”) with local or empirical ones. Framing results as empirical efficiency rather than mathematical guarantee is advised.
•  Include a small proof or citation showing IFT validity for smooth convex sets with full-rank equality constraints.

### Questions
Given the significant challenge of enforcing the strict constraints in large-scale or parametric Optimal Transport (OT) problems, how does the Πnet architecture with its hard-constrained compare to state-of-the-art OT solvers like the Sinkhorn algorithm and Adaptive Primal-Dual Accelerated Gradient Descent (APDAGD)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a differentiable projection network (ΠNet) for handling hard constraints within end-to-end neural network training. The key idea is to embed a projection operator directly as a network layer, ensuring that outputs always satisfy predefined linear equality and inequality constraints. The projection is computed via a Douglas–Rachford splitting scheme, which alternates between projections onto an affine set and a box set until convergence to their intersection. For backpropagation, ΠNet avoids differentiating through the iterative steps. Instead, it applies the Implicit Function Theorem (IFT) at the fixed point of the DR operator, computing the vector–Jacobian product by solving a linear system. Overall, the method provides a principled, computationally efficient, and differentiable mechanism for enforcing hard linear constraints in neural networks.

### Strengths
The paper is well organized and clearly written, with consistent notation and a coherent presentation.

It introduces a conceptually clean and mathematically principled approach to making projection-based constraint enforcement differentiable. By combining the Douglas–Rachford splitting scheme with the Implicit Function Theorem at the fixed point, the authors design a projection layer that is both efficient and end-to-end trainable. This integration of classical operator-splitting methods with modern differentiable programming is novel and results in a solver-free alternative to existing differentiable optimization layers such as cvxpylayers, DC3, and JAXopt.

The technical development is sound and internally consistent. The backward pass through the fixed-point system is derived correctly, and the implicit differentiation is implemented carefully via the solution of a linear system. The experimental evaluation is thorough: benchmarks include both convex and nonconvex problems across multiple scales, and comparisons are made against strong baselines. ΠNet achieves comparable or superior constraint satisfaction while reducing computation time by one to two orders of magnitude, demonstrating clear numerical advantages.

ΠNet provides a practical and general framework for incorporating hard constraints into neural networks without compromising differentiability. Its efficiency and stability make it particularly relevant for constrained learning tasks in control, trajectory optimization, and physics-informed modeling, representing a meaningful contribution to the literature.

### Weaknesses
While the paper is technically solid, several aspects could be clarified or strengthened.

First, the claim of handling convex constraint sets is somewhat overstated. All derivations and experiments are limited to linear (affine) equality and inequality constraints. Although these are convex in the geometric sense, the current method does not address more general convex sets such as norm balls, SOCs, or PSD cones. The projections and corresponding Jacobians for these sets are nontrivial, and it remains unclear whether the Douglas–Rachford formulation or the implicit differentiation strategy would remain computationally tractable in those cases. Clarifying this limitation and outlining potential extensions would make the contribution more precise and complete.

In addition, the experimental comparisons, though comprehensive, could be made fairer and more reproducible. JAXopt and DC3 are trained under different budgets and tolerances. Reporting results under equal wall-clock time or identical training epochs would offer a more balanced assessment.

Finally, although the backward pass based on the Implicit Function Theorem is elegant, the paper does not discuss the conditions ensuring the stability or invertibility of the associated linear system. A short clarification or empirical remark on this aspect would strengthen the completeness of the work.

### Questions
1. The choice of $\omega = 1.7$ is mentioned as default but not justified. Did the authors observe sensitivity in convergence or gradient stability with different values of $\omega$? Providing even brief empirical guidance would be useful for practitioners.
2. The method uses a fixed number of Douglas–Rachford iterations at inference equal to that during training. Have the authors verified that performance remains stable if more (or fewer) iterations or early-stopping are used at test time? This could reveal whether ΠNet truly learns a stable fixed-point operator or overfits to the training iteration budget.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PiNet, a differentiable layer for solving projection problems over convex constraints to ensure neural network output feasibility. The key contribution is a decomposition approach for certain constraints that can be represented as the intersection of hyperplanes and boxes, where each component admits efficient projection. The method applies operator splitting algorithms to the projection problem in this decomposed form, with the backward pass implemented via implicit gradients. Experiments demonstrate improved efficiency and reduced optimality gaps compared to baseline methods on benchmark problems and multi-vehicle motion planning tasks.

### Strengths
1. Ensuring neural network output feasibility is important for real-world applications, and this work addresses a practical and relevant problem.
2. The combination of constraint decomposition and operator splitting algorithms appears to be novel, and significantly improves projection efficiency for certain types of constraints.

### Weaknesses
1. The range of constraints that admit the decomposition form of hyperplanes and boxes is unclear. Besides the examples provided in the appendix, can the authors provide a clear definition or characterization of the types of constraints that fall within this framework?
2. The experiments mainly focus on linear constraints. Evaluation of non-linear constraints would further strengthen the contribution and demonstrate the broader applicability of this work.
3. The authors mention that Min et al. (2024) propose a closed-form expression to recover feasibility for polyhedral constraints. It would be valuable to include a comparison with this recent work to better position the proposed approach.
4. The related work section on hard-constrained neural networks could be strengthened by including other approaches, such as reparameterization approaches [1-2] and sampling-based approaches [3].

- [1] Tabas D, Zhang B. Safe and efficient model predictive control using neural networks: An interior point approach. CDC 2022.
- [2] Liang E, Chen M, Low SH. Homeomorphic projection to ensure neural-network solution feasibility for constrained optimization. JMLR 2024.
- [3] Kratsios A, Zamanlooy B, Liu T, Dokmanić I. Universal approximation under constraints is possible with transformers. ICLR 2022.

### Questions
Please refer to the weakness section. 

Overall, this work is well-written. I will adjust my rating upon clarification of the above points and possible additional experiments addressing the weaknesses mentioned.

### Soundness
3

### Presentation
3

### Contribution
2
