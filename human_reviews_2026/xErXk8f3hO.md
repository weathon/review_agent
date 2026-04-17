# Implicit bias of Hessian Approximation in Regularized Randomized SR1 Method

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Quasi-Newton methods have recently been shown to demonstrate dimension-independent convergence rate outperforming vanilla gradient descent (GD) in modern high-dimensional problems. By examining the spectrum of the Hessian approximation throughout the iterative process, we analyze a regularized quasi-Newton algorithm based on the standard randomized symmetric rank-one (SR1) update. The evolution of the spectrum reveals an implicit bias introduced by the Hessian learning, which promotes a preferential reduction of certain eigenvalues. This observation precisely captures the quality of Hessian approximation. Incorporating the implicit effect of Hessian update, we show that the regularized randomized SR1 method achieves a convergence rate of $\tilde{\mathcal O}\left(\frac{d_{\operatorname{eff}}^2}{k^2}\right)$ for standard self-concordant objective functions, where $d_{\operatorname{eff}}$ is the effective dimension of Hessian. In specific high-dimensional settings, which are common in practice, this method preserves convergence speeds comparable to accelerated gradient descent (AGD)  while maintaining similar computational complexity per iteration. This work highlights the impact of implicit bias and offers a new perspective on the efficiency of quasi-Newton methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a regularized quasi-Newton method based on the randomized SR1 method, and it achieves a sublinear convergence rate of $\mathcal{O}(d _{eff} ^2 / \epsilon ^{-2})$ where $d _{eff}$ is the effective dimension of the problem for self-concordant functions.

### Strengths
1. This paper combines several ingredients, including the randomized SR1 method, $\ell _2$ regularization, Hessian correction term, and lazy Hessian strategy to propose a regularized SR1 method.

2. It provides both theoretical analysis and empirical experiments.

### Weaknesses
1. Assumption 3 does not correspond to the standard self-concordance assumption commonly adopted in existing works. Rather, it is a lemma derived under the assumption that the function is strongly self-concordant (see Lemma 4.2 in [1]). The class of strongly self-concordant functions includes $\mu$-strongly convex functions with Lipschitz continuous Hessians. For such functions, existing studies [2] have established a global linear convergence rate that is independent of the dimension $d$, as well as a local superlinear convergence rate (see Table 1 in [2]). In contrast, the proposed regularized SR1 method attains only a dimension-dependent sublinear convergence rate, which is significantly weaker than the state of the art.

2. For the quadratic function case, since adding an $\ell_2$ regularizer makes the objective function strongly convex, the proposed method only achieves a sublinear rate, which is weaker than the state of the art [1,3]. 


**References**

[1] Rodomanov, A., & Nesterov, Y. (2021). Greedy quasi-Newton methods with explicit superlinear convergence. SIAM Journal on Optimization, 31(1), 785-811.

[2] Jin, Q., Jiang, R., & Mokhtari, A. (2024). Non-asymptotic global convergence analysis of BFGS with the Armijo-Wolfe line search. Advances in Neural Information Processing Systems, 37, 16810-16851.

[3] Rodomanov, A., & Nesterov, Y. (2022). Rates of superlinear convergence for classical quasi-Newton methods. Mathematical Programming, 194(1), 159-190.

### Questions
See the weakness.

### Soundness
2

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
4

### Summary
The authors introduce a regularized randomized SR1 algorithm, where the approximate Hessian is updated using randomly sampled vectors. The method integrates several techniques including $l_2$ regularization, Hessian correction, and a lazy update strategy.
The paper further investigates the spectrum of the approximate Hessian to reveal the impact of implicit bias toward capturing components associated with larger eigenvalues over smaller ones. Through comprehensive theoretical analysis, the paper establishes a non-asymptotic convergence rate and provides a complexity analysis, showing performance on par with AGD method in high-dimensional settings.

### Strengths
1. The authors provide an impressive and insightful analysis of the implicit bias in the spectrum of the Hessian approximation. Their theoretical analysis shows that SR1 tends to capture the Hessian components associated with the larger eigenvalues.

2. The paper offers a solid convergence analysis where the proposed RSR1 method achieves a convergence rate of $\tilde{\mathcal{O}}(d_{eff}^2/k^2)$ for standard self-concordant objective functions, matching the optimal rate of AGD.

### Weaknesses
1. The datasets "w8a" and "a9a" are too small to support the paper’s conclusions about high-dimensional optimization. The authors should include problems with much larger dimensions (for example, $d \geq 10,000$).

2. The authors should conduct longer experiments (> 1,000 steps) on larger datasets and more complex problems to illustrate the non-asymptotic convergence rate both at the start and near the solution.
Also, in Figure 3, CBFGS and CSR1 consistently outperform SSR1 and RSR1. Although these results tend to show the superlinear convergence of SSR1 and RSR1, the wall-clock time should be reported to demonstrate any advantages over classical quasi-Newton methods with line search.

3. Missing reference: Wang, Shida, et al. (2024) "Global non-asymptotic super-linear convergence rates of regularized proximal quasi-Newton methods on non-smooth composite problems."

4. Minor notation issue: both equation/figure references and bullet points use the same notation type (e.g., "(1)"). This can be confusing when they appear in the same paragraph (e.g., line 204). I suggest using different notations (e.g. "Eq. (3)" for equations).

### Questions
See weakness above.

### Soundness
2

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
This paper analyzes the **regularized randomized symmetric rank-one (SR1) method**, a type of quasi-Newton algorithm. It aims to explain why quasi-Newton methods often outperform gradient descent (GD) in high-dimensional problems, even when theory suggests their performance should depend heavily on the problem's dimension ($d$).

### Key Contributions

1.  **Identifies Implicit Bias:** The core idea is that the SR1 Hessian approximation process has an "implicit bias". It preferentially learns and reduces errors in the directions of the Hessian's largest eigenvalues first, rather than approximating all dimensions uniformly.

2.  **Analyzes Hessian Learning:** The paper first analyzes the SR1 update (Algorithm 1) as a matrix-learning process, separate from the optimization. It proves that this process can quickly approximate the "top" part of the Hessian's spectrum, with the error $||A-B_{K}||_{2}$ decreasing as $\tilde{\mathcal{O}}(Tr(A)/k)$. 

3.  **Proves an Accelerated Global Rate:** By combining this analysis with a regularized Newton-type framework (Algorithm 2), the paper establishes a global convergence rate of **$\tilde{\mathcal{O}}(d_{eff}^{2}/k^{2})$** for standard self-concordant functions.

### Main Takeaway

The paper's main finding is that for high-dimensional problems where the "effective dimension" ($d_{eff}$) is small—a common scenario in machine learning—this regularized SR1 method can achieve an accelerated convergence rate that is independent of the full dimension $d$. This $\tilde{\mathcal{O}}(1/k^2)$ rate matches the speed of Accelerated Gradient Descent (AGD) while maintaining a comparable computational cost per iteration, providing a strong theoretical justification for using quasi-Newton methods in modern, high-dimensional settings.

### Strengths
The paper presents several significant strengths, primarily in its novel theoretical analysis of the randomized SR1 method.

1.  **Novel Conceptual Insight on "Implicit Bias":** The paper's core strength is the identification and formalization of an "implicit bias" in the SR1 Hessian learning process. It moves beyond simply using SR1 as a black-box approximation and asks *how* it approximates the target Hessian. The insight that the update preferentially reduces error in the directions of the largest eigenvalues provides a compelling theoretical explanation for a well-known empirical phenomenon: that quasi-Newton methods often outperform first-order methods in high-dimensional settings where a full-rank approximation seems computationally infeasible.

2.  **Rigorous Analysis of SR1 Dynamics:** The paper's most significant technical contribution is the standalone analysis of the SR1 matrix-learning process in Section 4 (and Appendix B). This section is very well-developed and insightful.
    * It correctly identifies the main challenge in the analysis: the difficulty of proving a uniform $l_2$-norm bound, especially when multiple large eigenvalues are present.
    * The two-stage proof sketch (decomposing the process into a "Dispersion" stage to create eigengaps and a "Normalization" stage to reduce the top eigenvalue) is an elegant and powerful analytical technique.
    * This analysis culminates in Theorem 1, which provides a concrete, non-asymptotic, high-probability bound on the approximation error ($||A-B_K||_2 = \tilde{\mathcal{O}}(Tr(A)/k)$). This dimension-independent result is the key theoretical tool that the rest of the paper builds on.

3.  **Connection to Accelerated Global Convergence:** The paper successfully leverages this matrix-learning analysis to provide an end-to-end guarantee for an optimization algorithm. By (theoretically) using the SR1 learning process as a "warm-up" phase, it justifies the good initial approximation needed for its main optimization algorithm. This connection allows the paper to establish a final $\tilde{\mathcal{O}}(d_{eff}^{2}/k^{2})$ global convergence rate, which provides a strong theoretical reason for why a quasi-Newton method can match the performance of Accelerated Gradient Descent in practice.

4.  **Tackles an Important Problem:** The paper addresses a highly relevant and important question: why do quasi-Newton methods work so well in high-dimensional ML? By focusing on the *effective dimension* ($d_{eff}$) rather than the ambient dimension ($d$), the paper aligns its theory with the realities of modern machine learning, where data often possesses low-rank or rapid spectral-decay properties.

### Weaknesses
## 1. Limited Practicality of the Main Algorithm

The primary weakness of the paper is the significant gap between the theoretical algorithm and a practical, usable method. The main convergence guarantee (Theorem 2) relies on a complex, 3-phase algorithm that is only fully detailed in the appendix. 

This theoretical algorithm is not practical for two key reasons:

* **Three-Phase Structure:** The algorithm is not a simple, single-loop procedure. It requires a specific sequence of (1) a gradient descent phase, (2) a dedicated Hessian-learning phase (where optimization is paused), and (3) a final quasi-Newton phase. This is far more complex than a standard optimizer.

* **Unusable "Explicit" Parameters:** The paper claims to provide an "explicit choice of parameters", but the formulas provided in the appendix (Tables 1 and 2) are not practically implementable. They depend on *a priori* knowledge of global, problem-specific constants, including the level-set diameter $D$ (Assumption 1), the Lipschitz constant $L$ (Assumption 2), the self-concordancy constant $M$ (Assumption 3), and the effective dimension $d_{eff}$. A user has no way to know these values in advance, making it impossible to set the required schedule for regularization, step sizes, and even the lengths of the three phases. The paper's own experiments use a grid search, not this theoretical schedule.


## 2. Restrictive Theoretical Assumptions

The theoretical guarantees of the main algorithm (Theorem 2) are built on **Assumption 3 (Self-concordancy)**. This is a very strong condition that is far more restrictive than the standard $L_2$-Lipschitz Hessian assumption used in much of the optimization literature. This assumption limits the applicability of the paper's results to a specific, non-standard class of functions. It is not clear if the analysis holds for general convex problems (like the logistic regression used in the experiments, which is not typically self-concordant).



## 3. Unclear Presentation of the Main Algorithm

There is a significant disconnect between the algorithm presented in the main text (Algorithm 2) and the algorithm actually analyzed in the proof. Algorithm 2 appears to be a standard, single-loop iterative method. However, the proof of Theorem 2 relies on the complex 3-phase schedule (GD, Hessian learning, QN) that is only introduced in **Appendix C.2**. This makes it very difficult for the reader to understand what algorithm actually achieves the claimed $\tilde{\mathcal{O}}(1/k^2)$ rate, as the algorithm in the main text does not appear to follow this structure.



## 4. Misleading Terminology and Computational Cost

The paper repeatedly uses the term "quasi-Newton method," which typically implies a method that avoids direct Hessian information and uses only gradient differences (e.g., BFGS). The method analyzed here is fundamentally different. The randomized SR1 update (Equation 3) explicitly requires the target Hessian $A = \nabla^2 f(x_{n_k})$.

This means that at each step of the SR1 update, the algorithm must be able to compute a **Hessian-vector product** ($As_k = \nabla^2 f(x_{n_k}) s_k$). This is the oracle of a "Hessian-free" or "Inexact Newton" method, not a classical quasi-Newton method. This oracle is computationally more expensive and may not be available in all settings where classical QN methods are used.

### Questions
## Questions for the Authors

A major point of confusion is the 3-phase structure of the algorithm analyzed in the appendix (Appendix C.2), which is required to achieve the main convergence result (Theorem 2) but is not clearly presented in the main text. This theoretical algorithm, which involves separate, pre-scheduled phases for gradient descent, Hessian learning, and quasi-Newton steps, seems to have limited practicality.

This raises a significant question about alternative, more practical analyses:

My main question is whether a *true single-loop* algorithm could be analyzed using the paper's core technical contribution (Theorem 1). Specifically, what if at each optimization step $k$, one computes the Hessian approximation $B_k$ by running a fixed or variable number of randomized SR1 iterations (i.e., using Algorithm 1 as an inner sub-routine) targeting the current Hessian $\nabla^2 f(x_k)$?

This single-loop structure (an outer optimization loop with an inner approximation sub-routine) is common in other Inexact Newton methods. For example, the "Regularized Overestimated Newton with RPCholesky" (Duan and Lyu, arXiv:2509.21684) uses a Nyström-based approximation sub-routine (RPC) in this exact manner. That method, much like the SR1 analysis here, builds a low-rank approximation based on Hessian-vector products.

1.  Could the authors' analysis from Section 4 be adapted to this practical single-loop setting?
2.  For instance, could one use Theorem 1 to justify running $m_k = \tilde{\mathcal{O}}(d_{eff})$ SR1 steps *at each* iteration $k$ to achieve a high-quality Hessian approximation? What would the resulting *total* complexity be?
3.  How would such an approach compare to the 3-phase algorithm? While the downside of an algorithm like RPC is that the rank parameter must be globally large enough, this paper's SR1 analysis seems to guarantee a good $l_2$ approximation. It seems a single-loop version would avoid the need for the impractical, pre-defined parameter schedule of Tables 1 & 2, at the potential cost of more computation (running $m_k$ SR1 steps instead of 1) at each iteration.

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
In this paper, the authors analyze the randomized SR1 quasi-Newton method for minimizing quadratic objectives and, more generally, self-concordant functions. They start by considering a randomized SR1 update for approximating a fixed positive semidefinite matrix. By analyzing the eigenvalue dynamics of the resulting error matrix, they obtain a high-probability error bound in terms of the operator norm. Building on this result, they introduce a regularized version of the SR1 method and establish a convergence rate of $\tilde{O}(\mathrm{Tr}(A)/k^2)$ for quadratic objectives and $\tilde{O}(d_{\mathrm{eff}}^2 / k^2)$ for self-concordant minimization.

### Strengths
- Prior complexity analyses of quasi-Newton methods, including SR1, typically rely on a potential function, such as the trace or the Frobenius norm, and relate the convergence rate to that potential. In contrast, this paper adopts a different approach and performs a more fine-grained spectral analysis of the Hessian approximation error. The proof of the core result, Theorem 1, combines concentration inequalities, a construction of a rational function for bounding the eigenvalues, and an induction argument, which I find technically interesting.
- The authors present a global convergence rate for their proposed SR1 update, whereas most existing SR analyses are limited to local convergence or a quadratic objective. In addition, their rate depends on the Hessian’s “effective dimension,” which can be significantly smaller than the ambient dimension $d$.

### Weaknesses
- The final convergence guarantees feel somewhat unsatisfying. Compared to standard first-order methods such as accelerated gradient descent, the rate here is worse due to the additional dimensional factor $d_{\text{eff}}^2$. Compared to existing analyses of quasi-Newton methods, it only provides a slower sublinear convergence rate instead of a superlinear rate. Hence, the theoretical result does not justify the practical advantages of quasi-Newton methods over gradient-descent-based methods.  
- The regularized SR1 method in Algorithm 2 deviates from the common practice. From my understanding of the proof in Appendix C, the proposed method appears to first perform gradient descent, then estimate the Hessian using the SR1 update, and finally doing quasi-Newton updates. In effect, the method relies on gradient descent to reach a local neighborhood of the solution and then performs a local analysis of the SR1-based preconditioner. Moreover, in typical implementations, the Hessian approximation matrix is updated continuously rather than split into two separate phases.

### Questions
- I find the use of the term “implicit bias” somewhat confusing in this context. As the authors themselves note, implicit bias typically refers to the phenomenon where an algorithm, even without explicit regularization, converges toward particular solutions or optimization trajectories. In contrast, the SR1 update does not appear to be “biased” toward any specific solution in this sense, so the connection to the established implicit-bias literature is not immediately clear.
- In Section 4.1, the authors introduce a deterministic dynamical system to motivate the later analysis. However, the actual proof of Theorem 1 seems to follow a different line of argument, and it is unclear how (or whether) the deterministic dynamics in Section 4.1 meaningfully parallel the steps used in the main proof.

### Soundness
3

### Presentation
2

### Contribution
2
