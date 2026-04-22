# Descent-Net: Learning Descent Directions for Constrained Optimization

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 8, 2, 6

## Abstract
Deep learning approaches, known for their ability to model complex relationships and fast execution, are increasingly being applied to solve large optimization problems. However, existing methods often face challenges in simultaneously ensuring feasibility and achieving an optimal objective value. To address this issue, we propose Descent-Net, a neural network designed to learn an effective descent direction from a feasible solution. By updating the solution along this learned direction, Descent-Net improves the objective value while preserving feasibility. Our method demonstrates strong performance on both synthetic optimization tasks and the real-world AC optimal power flow problem.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a machine learning architecture for constrained optimization learning that approximates an iterative descent algorithm. The proposed approach integrates an active set strategy, an approximate descent direction computation, and a projection operator to ensure equality constraint feasibility. The approach is evaluated on small nonlinear problems (synthetic convex QPs, a mildly nonlinear variant of these, and small AC Optimal Power Flow instances).

Overall, I have several reservations regarding the validity of the proposed methodology, stemming from limited assumptions (e.g. linear equality constraints) and unpractical existential results (universal approximation theorem). In particular, as far as I can tell, the proposed scheme, in general, is not guaranteed to converge to a feasible solution. Furthermore, numerical experiments are conducted on small instances that are either synthetic or orders of magnitude smaller than real-life instances.

### Strengths
* The paper considers a constrained learning task (where the output of a machine learning model should satisfy constraints), which has received less attention in the literature (although there is a growing body of works in that field)
* The paper leverages insights from existing optimization algorithms to inform the design of their ML framework

### Weaknesses
* The proposed method relies on inverting the matrix $H^{\top}H$, where $H$ is the Jacobian of equality constraints at the current iterate. This is a large obstacle to scalability, as i) this matrix may become dense and numerically ill-conditioned (especially close to the optimum), and (ii) forming and inverting this matrix will become expensive for larger instances.

* The convergence result of Theorem 4.2 relies on the universal approximation theorem of ReLU networks. It is an existential result, which only states that there exists an architecture and parameter assignment such that the returned point is a KKT point. It does not provide any practical guarantee because one can never know whether the network architecture is large enough. 

* In addition, the universal approximation theorem is only valid for functions with bounded Lipschitz constant, which is not the case for the case at hand: the problem setup only assumes that objective and constraint functions $f_{x}(y), g_{x}(y), h_{x}(y)$ are smooth functions of $y$ given $x$. Additional assumptions would be needed regarding the domain of $x$ and the smoothness of $f, g, h$ w.r.t $x$.
    
    (note that even with those additional assumptions, the result of Theorem 4.2 remains impractical)

* The step size rule makes the following restrictive assumption (Section 4.3): "We assume that all equality constraints are linear."

* The step size rule relies on a local linear approximation of the inequality constraints using a first-order Taylor expansion. The proposed rule for controlling the step size also relies on that linear approximation, and may therefore fail to ensure feasibility in general.

* Numerical experiments are reported on small synthetic instances (which are not representative of real-life problems) and small AC-OPF instances. Regarding the latter, results should be reported on AC-OPF instances with several thousands of buses. There are several publicly-available datasets for this, e.g., [PGLearn](https://huggingface.co/PGLearn), which has data for systems with up to 24,000 buses

### Questions
* Problem MFD is formulated with an $\ell_{\infty}$ constraint on $d$, problem UFD is formulated with an $\ell_{1}$ box constraint on $d$, and problem (3) is formulated with an $\ell_{2}$ box constraint on $d$. Can you please explain why the different treatment in Eq. (3), and the potential impact of using an $\ell_{2}$ box instead of $\ell_{1}$ of $\ell_{\infty}$?
* Please comment on the validity of the linear approximation in the step size computation of Sec. 4.3, especially regarding feasibility guarantees
* I am very surprised by the computing times reported for ACOPF instances. According to [this benchmark](https://discourse.julialang.org/t/ac-optimal-power-flow-in-various-nonlinear-optimization-frameworks/78486/98), solving the AC-OPF takes about 0.1s and 0.3s for the 30 and 118 bus systems, respectively.
  I understand that Table 3 reports average solution times, but taking into account that i) these cases can be solved 4x faster than reported and ii) one GPU costs about 100x more than a CPU, the resulting speedup is actually quite modest compared to CPU-only solvers

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes DescentNet, an unrolled optimization module for neural networks based on the method of feasible directions framework, which takes a feasible solution to an optimization problem and aims to iteratively refine it into one that is still feasible but has improved optimality. The paper:
* Designs a Descent-Module that unrolls projected (sub)gradient descent on a penalty version of the uniformly feasible direction (UFD) formulation. This module includes the application of a learnable module to the subgradient term, and a step size-sizing strategy with a learnable parameter to ensure feasibility retention and optimality improvement.
* Provides theoretical guarantees about the existence of a Descent-Net instantiation that provides an optimal solution (assuming linear equality constraints).
* Provides experiments on convex QPs, a simple non-convex problem, and ACOPF, where Descent-Modules are appended to DC3. For the convex QP and simple non-convex settings, solutions obtain close-to-optimal solutions (in contrast to DC3) while solving 1-2 orders of magnitude faster than standard optimization solvers. For ACOPF (which has nonlinear equality constraints), "the relative error of the solution obtained by Descent-Net decreases only marginally compared to the initial point," and the solution time is about 50% faster than a traditional optimization solver.

### Strengths
* The submission is an interesting and thoughtful application of the Uniformly Feasible Direction Subproblem within neural networks.
* The Descent-Net method convincingly improves upon the optimality performance of DC3 (which provides feasibility but sometimes struggles with optimality) on convex QPs, the simple nonconvex problem class, and the ACOPF 30-bus system. It also improves, albeit more marginally, upon optimality performance for ACOPF 118-bus. This is a significant contribution, as SOTA methods for feasibility enforcement in amortized optimization can sometimes provide suboptimal performance with respect to objective value, due to the difficulty of neural network optimization for these methods -- Descent-Net addresses this via post-hoc iterative refinement of feasible solutions.
* The method also incurs high speedups relative to the traditional solvers for the convex QPs and simple nonconvex problem class, and does not incur significant additional computational burden relative to DC3. In other words, the Descent-Net updates are relatively lightweight.
* The paper is clearly written.

### Weaknesses
* There is a major missing ablation: What happens if the learnable modules $T^k$ are not applied in the descent module, i.e., what if only the original projected subgradient updates are applied? In general, what about other post-hoc iterative refinement strategies (with or without learnable parameters)?
* The timing improvements for ACOPF 30-bus and ACOPF 118-bus are more marginal (only 2x faster than traditional solver for ACOPF 30-bus, and only 30% faster for ACOF 118-bus). In addition, for ACOPF 118-bus, the updates actually add 1/4 - 1/3 more computational time compared to the base method, while not offering much optimality improvement. It would have been nice to see realistic/large-scale experiments where the improvement is more pronounced.
* The ACOPF experiments are missing some of the baselines provided in the other settings, making the comparisons slightly incomplete.

### Questions
* What happens if the learnable modules $T^k$ are not applied in the descent module, i.e., what if only the original projected subgradient updates are applied?
* Are there realistic settings where the improvements can be shown to be more pronounced (either via experiments or good theoretical arguments about classes of settings)?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Descent-Net, a learn-to-optimize framework for solving constrained optimization problems. The authors prove that Descent-Net achieves global convergence to a KKT point when both the inequality and equality constraints are linear. The proposed Descent-Net mimics the projected gradient descent algorithm but incorporates a nonlinear preconditioner. Specifically, each layer of Descent-Net applies a nonlinear transformation (implemented as a trainable two-layer ReLU network) to a descent direction computed from the penalized Topkis-Veinott uniformly feasible direction. The layer input is then updated along this descent direction using a trainable step size, followed by a projection onto the tangent space of the equality constraints to ensure the output remains feasible.

The authors provide simulation results for convex QPs, nonconvex problem (by replacing $y$ with $\mathrm{sin}(y)$ in the objective of convex QPs), and ACOPF problems, demonstrating that the proposed Descent-Net efficiently achieves an approximate KKT solution.

### Strengths
1 The paper is well-written and well-organized.

### Weaknesses
1. The experimental results are weak, as the proposed method shows only marginal improvements over some of the compared approaches. Moreover, it appears that the baseline methods used for comparison are not state-of-the-art for the problems considered in the experimental section. For example, general-purpose solvers based on trust-region methods, such as Fmincon and Knitro, should also be included in the experiments. For problems with smooth objectives and satisfying the LICQ condition, interior-point methods are known to achieve superlinear to quadratic convergence rates.
2. The proposed method requires computing the inverse of an $m \times m$ matrix $H^T H$ in the projection step at each iteration, which can become computationally prohibitive for large-scale problems due to its $O(m^3)$ time complexity. Consequently, the per-iteration cost of the proposed method is only marginally better than the $O((n + m)^3)$ per-iteration cost of trust-region methods.
3. The scale of the problems considered in the experiments is too small to justify the development of new methods for solving them. The authors should also include large-scale experimental results. Furthermore, the first two problems are overly simplistic since $Q$ is merely a diagonal matrix; in this case, the per-iteration cost of trust-region methods is significantly reduced because the corresponding KKT system is highly sparse. In addition, for ACOPF problems, Knitro includes specialized QCQP solvers that are widely used in both academic and industrial settings. The authors should also benchmark their method against the Knitro QCQP solver; otherwise, it is difficult to justify the need for developing new methods for solving ACOPF.
4. Based on the points discussed above, the proposed methods do not appear to offer any significant advantages over classical approaches for solving constrained optimization problems.
5. Finally, the proposed method also requires a feasible initial point, which is highly impractical in most real-world scenarios.

### Questions
Please refer to the weakness section for detailed comments. I have no further questions or suggestions.

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
5

### Summary
This paper proposes Descent-Net, a neural network-based framework for constrained optimization, which refines projected gradient descent (PGD) directions using learned operators. The method aims to handle general equality and inequality constraints by integrating projection steps and neural descent modules.

### Strengths
1. The paper studies an important and timely topic, addressing constrained optimization with neural networks.
2. The proposed pipeline is clear and closely follows the standard PGD approach, making it easy to understand the overall structure.

### Weaknesses
1. The contribution of the paper is vague and not clearly distinguished from prior work.
2. The technical approach is not fully rational or convincing, as the feasibility guarantee for nonlinear, non-convex constraints is not theoretically ensured.
3. The paper is difficult to follow; it would benefit significantly from more illustrative figures and a clearer pipeline description.

### Questions
1. What if the equality constraints are not linear? After moving along the proposed direction, how do you guarantee that the new point satisfies the equality constraints?
2. What is the fundamental contribution compared with existing papers? Please clarify the novel aspects over prior neural optimization and PGD-based methods.
3. How does the method scale with the number of constraints, especially if the projection step becomes computationally expensive?
4. How robust is the learned descent direction to changes in the constraint set or problem structure? Is retraining required for different constraint types?
5. Compared with the end to end papers with theoretical feasibility guarantee, what is the core advantage of the paper? As the projection itself can be time-comsuming

### Soundness
3

### Presentation
3

### Contribution
3
