# Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding

- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
We propose FlexQP, an always-feasible convex quadratic programming (QP) solver based on an $\ell_1$ elastic relaxation of the QP constraints. If the original constraints are feasible, FlexQP provably recovers the optimal solution. If the constraints are infeasible, FlexQP identifies a solution that minimizes the constraint violation while keeping the number of violated constraints sparse. Such infeasibilities arise naturally in sequential quadratic programming (SQP) subproblems due to the linearization of the constraints. We prove the convergence of FlexQP under mild coercivity assumptions, making it robust to both feasible and infeasible QPs. We then apply deep unfolding to learn LSTM-based, dimension-agnostic feedback policies for the algorithm parameters, yielding an accelerated Deep FlexQP. To preserve the exactness guarantees of the relaxation, we propose a normalized training loss that incorporates the Lagrange multipliers. We additionally design a log-scaled loss for PAC-Bayes generalization bounds that yields substantially tighter performance certificates, which we use to construct an accelerated SQP solver with guaranteed QP subproblem performance. Deep FlexQP outperforms state-of-the-art learned QP solvers on a suite of benchmarks including portfolio optimization, classification, and regression problems, and scales to dense QPs with over 10k variables and constraints via fine-tuning. When deployed within SQP, our approach solves nonlinear trajectory optimization problems 4-16x faster than SQP with OSQP while substantially improving success rates. On predictive safety filter problems, Deep FlexQP reduces safety violations by over 70\% and increases task completion by 43\% compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FlexQP, an ADMM-based $l_1$-penalizing formulation for quadratic programs; the authors claim it produces feasible iterates, recovers the original optimum when feasible, and otherwise minimizes constraint violations. They further unroll the solver into Deep FlexQP with LSTM-based parameter policies, present PAC-Bayes generalization bounds for the learned optimizer, and integrate it into SQP solver for nonlinear control and safety filtering.

### Strengths
1. The idea of using a uniformed penalty formulation to treat both feasible and infeasible points within the same objective is novel as it yields a single ADMM-based procedure.
2. Unfolding learns LSTM-based parameter policies while retaining the structure of the original solver, enabling accelerations to the original approach without discarding the algorithmic backbone.
3. The author provides theoretical support, including convergence characterizations of the penalty/ADMM scheme and PAC-Bayes generalization bounds.

### Weaknesses
1. The motivation behind and the advantages of using a $l_1$ penalty is not clear. The theory part claims properties of points that solve the  problem, but it does not directly establish a guarantee on whether Algorithm 1 and Deep-FlexQP can converge to those feasible/optimal solutions. A detailed explanation would be helpful.
2. The significance of the reported acceleration is unclear. As noted, the dominant cost remains the first ADMM block update, and in some cases Deep FlexQP does not surpass Deep OSQP. Please add a detailed discussion on where the method is expected to help or not. Besides, it would be interesting to see how Deep FlexQP's predictions differ from those values of the original FlexQP.

### Questions
Please refer to the weakness section.

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
4

### Summary
This paper presents a learning-based optimization method rooted in the Alternating Direction Method of Multipliers (ADMM) for solving convex quadratic programming problems. The proposed approach begins by introducing slack variables to transform inequality constraints into equality constraints, and subsequently incorporates all equality constraints into the objective function using an ℓ₁-norm penalty. A resulting ADMM-type solver—referred to as FlexQP—is then derived following an update scheme analogous to that of OSQP. To further accelerate convergence, this paper employs an LSTM network to generate all hyperparameters originally required by the algorithm, and train the model in a supervised learning framework. Experimental results demonstrate that the proposed method achieves a faster convergence rate in terms of optimality gap under a fixed iteration budget compared to several baseline methods.

### Strengths
This paper presents a novel learning-enhanced ADMM framework, supported by theoretical analysis and demonstrated to achieve faster convergence rates compared to established baselines across diverse datasets.

### Weaknesses
1. The motivation for introducing slack variables and an ℓ₁-penalty term appears insufficiently justified. Since the ADMM-based solver OSQP can directly solve Problem (1), why not simply accelerate that algorithm using a neural network? Is the intention to use $z_I$ and $z_E$ to determine the feasibility of the original problem? However, in practice, $z_I$ and $z_E$ are unlikely to be exactly zero during iterations, as their values are strongly influenced by how well constraint (4b) can be satisfied. Moreover, if the original problem is infeasible, what is the practical value of providing a solution that only "minimizes the constraint violations"?
2. While Figure 5 indicates superior convergence behavior of Deep FlexQP on all nine datasets, the actual computation time is missing. A clear description of how the runtime was measured should also be provided.
3. The scale of the datasets used for solving the problems remains relatively limited. Could results on larger and more challenging problem instances be provided?
4. The manuscript contains several instances of non-standard or undefined notation. For example, the abbreviation "SQP" in line 26 and "S$\ell_1$QP" in line 145 lack clear definitions. In Theorem 3.1, the variables $y^{\star}_I$ and $y^{\star}_E$ are introduced without explanation. Additionally, the expression “$\mu_i \geq y_i$” in line 161 is ambiguous: if $\mu_i$ is a scalar, should $y_i$ not also be a scalar rather than a vector? Similarly, in lines 158 and 164, it is unclear whether $y_i$ should carry an absolute value and whether it refers to an element of the dual variable $y$. We recommend a thorough review and clarification of notation throughout the text.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

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
The paper tackles fast solving for structured optimization by combining:
a learned generator (diffusion) that predicts primal–dual variables,
a KKT-aware loss (feasibility and stationarity residuals), and
a post-refinement stage using classical primal–dual updates.
The aim is to obtain near-feasible, near-optimal solutions in very low inference time, with a few corrective iterations closing any remaining gaps.

### Strengths
1. Accelerating constrained optimization with learning is timely and useful.

2. Primal–dual parameterization + KKT residuals makes the supervision meaningful; the post-refinement stage is practical for polishing errors.

3. On synthetic QP-style tasks, the one-step (or few-step) approach achieves competitive gaps/residuals with favorable wall-clock times.

### Weaknesses
1. Limited novelty in core ingredients. Diffusion generation, GNN message passing over factor graphs, and KKT-residual losses are all known; the paper reads as a careful composition/tuning rather than a new algorithmic principle or theory.

2. The paper lacks component-wise ablations that isolate the value of diffusion vs. a non-diffusive predictor, GNN vs. MLP, and KKT loss vs. plain supervised losses, as well as sensitivity to refinement steps and guidance scales.

3. Under what assumptions (e.g., strong convexity, Slater/LICQ) does your KKT-guided sampling guarantee monotone decrease of a KKT energy or local convergence? Please state the step-size / guidance strength conditions.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2
