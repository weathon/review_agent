# Constrained Stochastic Multi-Objective Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
This paper aims to address the constrained stochastic multi-objective optimization (CSMOO) problem, where both objectives and constraints involve expectations over random variables. Firstly, to tackle the computational challenge of exact expectation evaluations, we propose two approximation schemes: stochastic approximation, which updates the entire problem using new samples at each iteration, and block stochastic approximation, which updates only subsets of variables iteratively. Secondly, to handle potential infeasibility in the surrogate problems, we develop two strategies: a feasible update reformulation and a rigorously justified penalty scheme equivalent to the original problem. Our framework provides asymptotic convergence guarantees to stationary points that satisfy Fritz John conditions. Experiments on synthetic and real-world wireless communication benchmarks demonstrate superior convergence, stability, and constraint satisfaction over state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work studies constrained stochastic multi-objective optimization. To handle the randomness in the objective/constraints, the authors introduce different asymptotically consistent estimators. Based on them, the authors propose two algorithms, CSMOO-1 and CSMOO-2, which also adopt the surrogate quadratic program. Lastly, numerical experiments demonstrate the effectiveness of the new algorithms.

### Strengths
1. The paper is easy to follow.

1. The experiments are detailed.

### Weaknesses
1. Please add a reference for Definition 3.

1. The notion $\xi_t$ in $\mathbb{E}\_{\xi_t}[\cdot\mid\mathcal{F}_t]$ is redundant.

1. In Line 116, the authors use $d$ to denote the dimension of the variable $x$. However, in many other places (e.g., Line 212), the authors change the notation to $k$. Please unify it.

1. In Line 222, the authors use $p_r^t$ to denote the probability of selecting the $r$-th coordinate at the $t$-th iteration. However, in the proof of Theorem 1, it changed to $p_{r,t}$. Please unify it.

1. I cannot find any convergence results for the two newly proposed algorithms. Could the authors provide/say any of them?

1. For Theorem 3, from the current proof, the authors only show that the Fritz John condition is satisfied. However, it is only a necessary condition for a point to be weakly Pareto optimal, but not sufficient. I don't understand why the authors can claim $x^*$ is weakly Pareto optimal.

1. For both algorithms, according to Line 160, I assume the authors default to $z_i=\inf_{x\in\mathcal{X}}f_i(x)$. However, finding $z_i$ may not be easy in many cases. This largely limits the practicality of both algorithms.

1. For CSMOO-2, how does penalty parameter $\beta$ affect the algorithm? Adding more discussions on it will benefit the work.

1. In Lemma 2, what are the second order sufficiency conditions? In the proof of Theorem 2, does the Lagrange multiplier $\eta$ satisfy them?

1. In many places in the proof of Theorem 1 (e.g., Line 684), $\gamma_t^2$ is redundant, and $\bar{x}_t$ should be $x_t$.

1. Line 755, it should be $t=\ell k+r-1$ according to the definition in equation (6).

1. Lines 750 and 812, $f_{i,t}$ should be $\hat{\nabla}f_i(x_t)$.

1. Lines 797 to 802, these steps only hold for the specified algorithm but not in general, meaning that they are not true under the current statement of Theorem 1. Please revise them.

### Questions
See **Weaknesses**.

### Soundness
2

### Presentation
2

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
In this paper, the authors studied CSMOO problems and proposed stochastic and block stochastic approximation schemes to efficiently approximate the original formulation. To handle potential infeasibility in the surrogate problems, they further introduced feasible-update reformulations and a penalty-based strategy with theoretical guarantees. Experiments have been conducted for testing their proposed algorithms.

### Strengths
1. The paper effectively mitigates the computational burden of exact expectation evaluations by introducing two approximation schemes: (i) a stochastic approximation method that updates all variables with fresh samples each iteration, and (ii) a block stochastic approximation method that updates variable subsets iteratively. The distinction between these schemes makes them suitable for variables of different dimensions.
2. The paper also addresses potential infeasibility in surrogate problems through two well-motivated strategies: a feasible-update reformulation and a rigorously justified penalty scheme that is theoretically equivalent to the original formulation.

### Weaknesses
1. The literature review is not comprehensive. A more thorough discussion of prior work on CSMOO is needed, including existing stationarity conditions and algorithms for solving such problems. The current related-work section focuses primarily on deterministic settings.
2. The metric, FJ condition, is relatively weak. Are there stronger stationarity guarantees applicable to this class of problems? For example, could the proposed method be shown to converge to KKT points instead?
3. The presentation of the experimental results lacks clarity. For instance, in Figure 1, the three curves are heavily overlapped—what is the intended takeaway from this plot? Moreover, additional baselines beyond projected SGD would help strengthen the empirical evaluation.

### Questions
1. How is subproblem (14) solved in Algorithm 2? Please clarify the exact solution procedure or any approximations used.
2. What are the advantages of using block stochastic approximation? Does it lead to improved convergence rates or better sample complexity compared to the standard stochastic approximation?
3. How do you choose $\rho_t, \gamma_t, \beta$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the **constrained stochastic multi-objective optimization (CSMOO)** problem, where both objectives and constraints are expectation-valued. The authors introduce a general algorithmic framework based on **Tchebycheff scalarization**, **stochastic (and block-stochastic) approximations**, and two strategies to handle infeasibility of surrogate problems:

-   **CSMOO-1:** A feasible-update formulation that ensures progress even under inaccurate estimates.
    
-   **CSMOO-2:** A penalty-based reformulation proven equivalent to the original problem and always feasible.
    

They provide asymptotic consistency results for both function and gradient estimates and show that their algorithms converge to stationary points satisfying the **Fritz John** conditions. Empirical validation includes synthetic experiments and a **wireless communication (physical-layer security)** benchmark, demonstrating better constraint satisfaction and convergence versus baselines.

### Strengths
1. CSMOO sits at the intersection of constrained optimization and stochastic multi-objective learning — a setting with growing importance in ML and communications. The formulation is rigorous and well-motivated.
    
    
2. The two proposed methods (feasible-update vs. penalty-based) address key implementation bottlenecks — infeasibility and non-differentiability — in a principled way.
    
3. Synthetic and real-world wireless tasks show clear convergence and feasibility improvements. Visualization of Pareto fronts and constraint violation trends are helpful.
    
4. The paper is clearly organized, with detailed notation, assumptions, and comparison to prior MOO and CMOO work.

### Weaknesses
1. Both algorithms need the **z** vector which is the set of optimal values of the multiple objective functions, which requires minimizing all the m objective functions in advance.
    
2. The assumption 1 is somewhat strong since it requires the Lipschitz continuity and smoothness of stochastic function $f_i(\cdot, \xi)$ and $g_j(\cdot, \xi)$.

   
3. Provide only asymptotic convergence guarantee. A non-asymptotic analysis would be better.

4. Lack of novelty. Both the moving average estimator for function values and gradients and the quadratic surrogate functions are well-known ideas in stochastic optimization. And I believe that the asymptotic consistency(theorem 1) for moving average estimator is also a well-known result.

5. Evaluation is restricted to low-dimensional synthetic and a single wireless-security case. No large-scale or ML-relevant benchmarks are tested, limiting generality.
 
 
6. The baseline in wireless communication experiments is a simple projected SGD. Missing comparisons to modern stochastic MOO methods (e.g., stochastic MGDA variants) weakens empirical claims.

### Questions
1.  Regarding weakness 2, can you relax assumption 1 to the Lipschitz continuity and smoothness of the expected function $f_i(\cdot)$ and $g_j(\cdot)$?

2.  In definition 3 you mention that Fritz John condition is only a **necessary** condition to weak pareto optimality, then how do you conclude in the proof of theorem 3 that $x^*$ is weak Pareto solution by only verifying the Fritz John condition?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses constrained stochastic multi-objective optimization (CSMOO) problems where both objectives and constraints involve expectations over random variables. The authors propose two approximation schemes (stochastic approximation and block stochastic approximation) to handle the computational challenge of exact expectation evaluation, and two strategies (feasible update reformulation and penalty scheme) to handle infeasibility in surrogate problems. Theoretical analysis provides asymptotic convergence guarantees to Fritz John stationary points. Experiments on synthetic and wireless communication benchmarks are presented.

### Strengths
- The paper focuses on constrained stochastic multi-objective optimization which has applications in many domains including wireless communications and industrial design.
- The paper proposes two complementary strategies (CSMOO-1 and CSMOO-2) to handle infeasibility issues that arise from inaccurate expectation estimation, where CSMOO-2 overcomes the increased computational overhead and potentially slower convergence of CSMOO-1.
- Theoretical analysis are provided to show convergence of proposed methods and the equivalence between the reformulated problem and original problem.

### Weaknesses
- The novelty in the paper seems to be limited at the reformulation from original MOO formulation into the constrained stochastic optimization problem. The techniques used after that are rather standard including the surrogate function approximation and gradient estimators. Also, the theoretical results look similar to results in [1] for CSMOO-1 as well.
- The numerical experiments are not illustrating the complexity of MOO problems as both examples only contain 2 objectives. Overall they do not show how the proposed method can scale up to more complex settings.
- It looks like $\beta$ is an important parameter for CSMOO-2 but I do not see discussion on how to specify it. Also, it would be great to have ablation study on how $\beta$ affects the performance of CSMOO-2.
- There is only a gradient-based baseline in the experiments and no discussion on why not including other related methods on MOO, e.g. preference-based methods...
- There are no information provided on how to solve (11), (13) or (14). Even though it might be standard, it would benefit the readers if the authors provide the full details.

[1] Liu, An, Vincent KN Lau, and Borna Kananian. "Stochastic successive convex approximation for non-convex constrained stochastic optimization." IEEE Transactions on Signal Processing 67.16 (2019): 4189-4203.

### Questions
- In the reformulated problem such as (9) or (13), should we consider individual slacks for each constraints instead of having only one $y$ or $delta$?
- What is the complexity of solving problems (11), (13), or (14)?

### Soundness
3

### Presentation
2

### Contribution
2
