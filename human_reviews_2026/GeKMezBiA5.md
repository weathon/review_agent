# Lower-level Duality Based Penalty Methods for Nonsmooth Bilevel Hyperparameter Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Hyperparameter optimization (HO) is a critical task in machine learning and can be naturally formulated as bilevel optimization (BLO) with nonsmooth lower-level (LL) problems. However, many existing approaches rely on smoothing strategies or sequential subproblem solvers, both of which introduce significant computational overhead. To address these challenges, we develop a penalization framework that exploits strong duality of the LL problem and its dual. Building on this, we design first-order single-loop projection-based algorithms to solve the penalized problems efficiently. Our methods avoid smoothing and off-the-shelf solvers, thereby greatly reducing per-iteration complexity and overall runtime. We provide rigorous convergence guarantees and analyze the stationary conditions of BLO with nonsmooth LL problems under penalty perspective. Through extensive numerical experiments on a variety of benchmark and real-world tasks, we demonstrate the efficiency, scalability and superiority of our method over existing BLO algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel duality-based penalization method for bilevel optimization with nonsmooth lower-level problems. The key idea is to reformulate the LL problem via its dual, keep conic/epigraph constraints explicit, and penalize the rest. This leads to single-loop, first-order algorithms that are both projection-based and solver-free. Experiments on high-dimensional problems show the method achieves competitive test accuracy with lower compute time than several BLO baselines.

### Strengths
1. The framework broadens the class of tractable LL problems by only requiring the base potential $\varphi$ to be convex and L-smooth, effectively handling non-strongly convex losses.

2.  The single-loop, projection-based design avoids the computational overhead of nested loops and external solvers, making it highly efficient per iteration.

3. Provides non-asymptotic convergence guarantees for the proposed methods under standard and mild assumptions.

### Weaknesses
1. The numerical evaluation focuses on validation/test error but omits a key metric for the presented tasks: solution sparsity. For hyperparameter selection promoting sparsity, analyzing the sparsity of the obtained solutions is crucial for a complete comparison.

2. The labels and text in Figure 1 are too small, hindering readability. A version with larger fonts is needed.

### Questions
1. In Section 3.2, the authors use ADMM to handle projections onto the intersection of multiple cones, noting that alternating projections could be used. Could the authors comment on why they chose ADMM over alternating projections? While ADMM is efficient, its convergence in nonconvex settings requires restrictive assumptions (like Assumption 3.6). What are the trade-offs between these two approaches in this specific context?

2.  Assumption 3.2 could be stated more succinctly by directly requiring that $\varphi$ is L-smooth and μ-strongly convex.

3. When first introducing equation (19), it would be helpful to explicitly define the penalty parameter $\gamma$ and its admissible range (e.g., $\gamma>0$).

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
This paper investigates nonsmooth bilevel hyperparameter optimization and develops a lower-level duality-based penalty framework that transforms the bilevel problem into a single-level form. The method leverages strong duality in the lower-level convex subproblem to design penalty-based algorithms that eliminate the need for explicit smoothing or nested differentiation. Two first-order schemes, LDP-PGD and LDP-ADMM, are proposed and analyzed. Theoretical results establish convergence guarantees under standard convexity and regularity assumptions. Experimental validation is conducted on small-scale convex and sparse logistic regression tasks.

### Strengths
The paper presents a rigorous and principled treatment of nonsmooth bilevel optimization under convexity assumptions. The use of lower-level duality to design a penalty reformulation is elegant and mathematically sound. The resulting single-loop algorithms are computationally efficient, Hessian-free, and accompanied by formal convergence proofs. The theoretical exposition is detailed, and the proofs appear correct under the given assumptions.

### Weaknesses
The method relies on convexity and strong duality of the lower-level problem, limiting its applicability to simple bilevel setups and excluding nonconvex inner problems common in modern hyperparameter optimization.

The experiments are small-scale and confined to convex tasks such as sparse logistic regression, providing limited evidence of scalability or robustness.

There is no comparison with recent single-loop or envelope-based approaches, and no sensitivity analysis of the penalty parameter or dual variable errors, both of which can strongly affect convergence and stability.

### Questions
1. How are the penalty multipliers $\lambda_i$ chosen or updated, and does convergence depend on their initialization? 
2. Is there a bound on the stability or growth of the dual variables $(\xi, \rho)$ during optimization? 
3. What is the per-iteration computational complexity of LDP-PGD relative to existing single-loop algorithms? 
4. How does the penalty-induced bias compare analytically to Moreau-envelope smoothing methods? 
5. Was the duality gap empirically verified to remain small in numerical experiments?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles hyperparameter optimization via Bilevel Optimization. They propose a very elegant reformulation of the problem which exploits convex conjugation, and finally, they propose a practical single-loop Hessian-free optimization algorithm which achieves strong guarantees in the separable case and convincing practical performance. Unfortunately, the guarantees are way weaker in the non-separable case.

### Strengths
The reformulation of the problem exploiting convex conjugates is very elegant.

The experiments are convincing.

### Weaknesses
Better to explicitly introduce the data matrix $A_t$ and the labels $b_t$. In the smoothed Hinge loss line on page 2 the weights are denoted via $w$ rather than $x$ (as in the rest of the paper).

Theorem 3.5 does not indicate which is the optimal choice of $\underline{\beta}$. Moreover, I think that there is an abuse of big-oh notation. By this I mean that beyond the dependence on $K$, the authors should indicate how the rate depends on $\alpha_L$, $\alpha_P$, and the dimension of $x$.

Assumption 3.6 seems unnatural and should be guaranteed by the update rule itself.

The statement of Theorem 3.7 is rather weird because the constant M_e contains the learning rate $\gamma$. There is no guidance on how $\gamma$ should be chosen. Moreover, no finite time rates are provided.

Also, the assumption that $F_k$ remains bounded should be avoided.

### Questions
Can you provide the dependence on $\alpha_L$, $\alpha_P$, and dimension in Theorem 3.5? Moreover, indicate which is the best choice of $\underline{\beta}$. In particular, you should be able to characterize how large $\underline{\beta}$ should be taken to ensure that small merit functions also imply that the algorithm output is a solution of the problem of interest, which is (6).

Can Assumption 3.6 be avoided?

Is the boundness of $F_k$ necessary?

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
This paper addresses the bilevel hyper-parameter (for norm-based regularizers) optimization problem. It first proposes a series of penalty reformulation of the original bilevel problem. Under certain convexity and other regularity assumptions, the paper shows that the penalty reformulations lead to the solutions of the original problem. To solve for the reformulated problems, the paper considers the gradient descent approach with epigraphic projections. The proposed method is supported with a convergence guarantee and some experimental results.

### Strengths
1. This work is highly relevant, as it studies the important problem of optimizing hyper-parameters.
2. The paper is overall clear and well structured. The key theoretical points are made clear and well supported.
3. The proposed algorithm is single-loop, which is more suited for computationally expensive machine learning applications as compared to a double-loop structure.

### Weaknesses
My major concern on this work is its general effectiveness:

1. The problem setup, and the key assumptions needed for the algorithm to work, or for the theories to hold, is somewhat restrictive to me. The paper considers a problem in (4), which assumes a linearly structured data and a restricted set of nonlinear loss which excludes some most frequently used loss like cross-entropy. The proposed algorithm is tightly built on this set of assumptions and the theoretical result has a relatively strong dependence on them. 

2. If the theoretical results are hard to generalize to broader settings, it is then natural to look for empirical evidence for the algorithm's effectiveness beyond the made assumptions. However, it seems the paper does not provide ample evidence for this. It might be beneficial to test the algorithm in more large-scaled problems that do not admit a convex/linear structure.

3. Missing baselines in experiments: In section 4.2, the experiments miss arguably the most important baseline that is the random and the grid-search-based hyper-parameter optimization. Since those methods are the most commonly adopted approach in practice, it might be beneficial to compare with them.

### Questions
In addition to the questions posed in the weakness section, I also have the following ones:
1. What is the overall computational complexity of this algorithm? How does it compare to other penalty-based BLO methods?
2. The proposed algorithm directly optimizes on the validation set, which is usually prohibited in machine learning practice since the validation set is supposed to be independent of the training process as a generalization check. Directly putting optimization pressure on the validation set makes it fail its role as the generalization check.

### Soundness
2

### Presentation
2

### Contribution
2
