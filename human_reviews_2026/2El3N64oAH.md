# Matrix-Free Least Squares Solvers: Values, Gradients, and What to Do With Them

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
This paper argues that the method of least squares has significant unfulfilled potential in modern machine learning, far beyond merely being a tool for fitting linear models. To release its potential, we derive custom gradients that transform the solver into a differentiable operator, like a neural network layer, enabling many diverse applications. Empirically, we demonstrate: (i) scalability by enforcing weight sparsity on a 50 million parameter model; (ii) imposing conservativeness constraints in score-based generative models; and (iii) hyperparameter tuning of Gaussian processes based on predictive performance. By doing this, our work represents the next iteration in developing differentiable linear-algebra tools and making them widely accessible to machine learning practitioners.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an efficient VJP method to backpropagate through the solution of a least square operation, thus
enabling it as a differentiable operation in any ML workflow.

### Strengths
* Tacking an important linear algebra primitive such as LstSq is potentially of high impact.
* Framing the LstSq operation as a layer is novel and though provoking. 
* The VJP implementation that avoid backpropagating through iterative algorithms is theoretically well motivated,
  efficiently implemented and functional.

### Weaknesses
* Some of the applications do not show the full potential of the method. In particular the Gaussian Process example  is
  small in terms of the features of the data, the number of hyperparameters and the use of random Fourier features
  (which is not the common practice)
* The method just applies to full rank matrices.

### Questions
* Line 093: What does it mean to require "twice the precision". You mean potentially twice the number of iterations?
  Precision could also suggest going from float32 to float64 or float128 but I doubt that you are referring to this.
  If you are referring to numerical precision. What precision do you use in your experiments? The standard float32 in
  deep learning?
* Are you familiar with [1]? The authors also get transposition automatically as a VJP and offer a series of matrix
  operations with differentiation both in JAX and PyTorch. I believe this work should be discussed in your related work
  section.

[1] Potapczynski et al 2023. CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper treats the least-squares solver as an operator and derives its vector–Jacobian product when used as the input to a scalar function.
- Building on the null-space method, the authors reformulate constrained optimization problems into unconstrained forms that incorporate the least-squares operator.
- The authors validate experimentally the two procedures

### Strengths
- The paper is well-written with concise explanations and helpful supplementary material in the appendix.
- The proposed VJP procedure demonstrates strong computational efficiency, as supported by timing comparisons.
- The introduced framework and accompanying library provide a practical and flexible means to incorporate constraints optimization setups.

### Weaknesses
**Nuancing statements**
In Section 2.1, matrix-free methods are presented as ideal, yet it must be must highlighted that they also require access to a procedure to evaluate $A^\top u$.
When such access is unavailable, matrix-free approaches lose their advantage and can become as expensive as direct linear-system methods.
Similarly, Equation (4), which deduce $A^\top u$ via a vjp computation, should be tempered: vjp can introduce non-negligible computational overhead; see [2, Table 7.1].

**Ambiguity between Sections 2.2 and 2.3**
The connection between Sections 2.2 and 2.3 is unclear. Section 2.3, on constrained optimization via least squares, appears largely independent of the vjp derivations in 2.2. Since the results of 2.2 are not required for the developments in 2.3, the transition feels abrupt and breaks the logical flow of the paper. The authors should clarify how these sections relate conceptually or reorganize the exposition to improve coherence.

**Positioning within the literature and experimental context**
The discussion of prior work on vjp computation is limited. The manuscript should better position its contribution relative to existing approaches, particularly those addressing implicit differentiation (e.g., [1]) and direct derivations of the VJP for the least-squares operation (e.g., [3]).
In particular, it is not clear what are the pros and cons of the proposed derivation relative to that.
This incompleteness carries into the experiments, where the proposed method is evaluated against few baselines, for instance two bases for runtime comparison 


**Typos/minor issues**

* Line 94: missing $\theta$ in front of $A$.
* Lines 110–111: replace "By reducing the likelihood of errors" ---> “By reducing sources of errors.”
* Line 219: define the abbreviation PGD (Projected Gradient Descent) as it is used right after in Line 221-222
* Section 2.3: explicitly state that the number of constraints is smaller than the parameter dimension, as assumed in Theorem 3


---

.. [1] Blondel, Mathieu, et al. "Efficient and modular implicit differentiation." Advances in neural information processing systems 35 (2022): 5230-5242.

.. [2] Blondel, Mathieu, and Vincent Roulet. "The elements of differentiable programming." arXiv preprint arXiv:2403.14606 (2024).

.. [3] Wan, Zhou-Quan, and Shi-Xin Zhang. "Automatic differentiation for complex valued SVD." arXiv preprint arXiv:1909.02659 (2019).

### Questions
- Line 243-245: in the experiment design, can you clarify "increasing number of rows and columns"?
- Full-rank assumption: The method relies on the matrix being full-rank. How realistic is this assumption in practical applications, especially when dealing with ill-conditioned or rank-deficient systems, e.g. in inverse problems?

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
4

### Summary
This paper introduces a differentiable, matrix-free least-squares operator that turns classical solvers like LSMR into drop-in neural network layers with custom gradients. The authors derive reverse-mode derivatives for tall and wide matrices (Theorem 1) and provide a JAX implementation that automatically constructs $A^\top$ via vector-Jacobian products. They further reinterpret the null-space method for constrained optimization as a least-squares projection, enabling the use of standard Optax optimizers for constraint enforcement. Applications include Gaussian process calibration via differentiable least squares, symmetry and sparsity constraints in deep nets, and conservative score-based generative models. Empirically, custom gradients yield 5–10× runtime improvement over solver unrolling, and the null-space formulation produces strong results on large models with minimal code overhead.

### Strengths
The paper has the following strengths:

- **Originality:** Theorem 1 extends beyond prior work by Golub & Pereyra (1973) and Krämer et al. (2024) by providing reverse-mode derivatives for adaptive least-squares solvers with regularization terms, handling both tall and wide cases in a unified framework. The reformulation of Yamashita (1980)'s null-space algorithm as a least-squares projection is particularly creative, enabling practical large-scale constrained optimization on models with up to 50 million parameters. 

- **Quality:** The technical quality is good, with rigorous mathematical foundations throughout. The proofs in Appendix C are complete and correct, and the distinction between tall and wide matrices with their limiting behaviors is thoroughly analyzed. The experimental validation is comprehensive, spanning five diverse applications that demonstrate broad applicability. 

- **Clarity:** The paper writing is smooth, which progresses logically from fundamentals to gradients to applications, and the appendices effectively organize background material and detailed proofs. 

- **Significance:** The work addresses a significant gap in the machine learning toolkit by making numerical least-squares solvers more accessible in differentiable programming frameworks. The application of Gaussian process calibration is particularly noteworthy, as directly optimizing predictive fit is simpler and more scalable than marginal-likelihood optimization, yet surprisingly underexplored in the literature.

### Weaknesses
- **Full Rank Assumption**: The most critical limitation is the full-rank assumption that pervades the entire paper. The paper does not guide diagnosing rank deficiency, choosing regularization λ to stabilize near-singular cases, or understanding what happens numerically when the assumption is violated. Even when matrices are technically full rank, severe ill-conditioning can produce similar numerical failures. Moreover, no experiments report the numerical rank or condition number of $J_c$, leaving it unclear whether the demonstrated applications actually satisfy the full-rank requirement or whether the method is more robust than the theory suggests.

- **Practical Applicability**: The paper provides no guidance on how practitioners should diagnose rank deficiency before applying the method, what happens numerically when assumptions are violated, when or how to choose regularization λ to stabilize near-singular cases, or whether the technique degrades gracefully or fails catastrophically. This limitation is not mentioned in the abstract and is only briefly acknowledged in the conclusion. Given that the title promises "Matrix-Free Least Squares Solvers" for "modern machine learning," the full-rank restriction substantially narrows the scope. The paper should prominently scope the work to full-rank problems with a clear justification of their practical relevance compared to non-full-rank matrices. 

- **Insufficient Numerical Analysis**: The paper uses a tolerance of $10^{-6}$ for LSMR but does not investigate how backward-pass tolerances affect gradient accuracy. While Example 2 shows that LSMR handles condition numbers around $10^7$ well for the forward pass, the backward pass requires solving two additional least-squares problems, as stated in Theorem 1. No experiments report condition numbers, ranks, or numerical diagnostics for the constraint Jacobians. 

- **Limited discussion on preconditioning**: Preconditioning is essential for practical large-scale problems, yet readers have no guidance on how custom gradients interact with preconditioners or whether users can supply preconditioners to the LSMR implementation.

### Questions
1. Do backward passes require tighter tolerances than forward passes? What is the relationship between solver tolerance and gradient error? 
2. Have the authors validated the custom gradients against finite differences or double-precision automatic differentiation? How sensitive are the gradients to LSMR tolerance: should backward passes use tighter tolerances than forward passes?
3. What is the effective rank and conditioning of $J_c$ in the equivariance experiments, and for SWIN-S with 50M parameters? What is the rank and conditioning of the constraint matrix? 
4. Based on the response to Q3, can the authors comment on the generality of the proposed matrix-free least square solver?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Authors propose a method to differentiate efficiently through the solution of a linear system, e.g., if one defines $x^{\star}(A, b)$ as the solution of $Ax^{\star} = b$, then one can differentiate $x^*$ with respect to $A$ or $b$. Authors have experiments to demonstrate that the proposed gradient computation method is faster then other methods, and show that their method can be useful for Gaussian process calibration and constrained optimization.

### Strengths
The topic of differentiating through optimization problem is interesting, especially since the resurgence of implicit differentiation-based approach like "Tiny Recursive Models".

### Weaknesses
Weaknesses:
- Clarity:
    - I did not understand the proposed approach, do authors use a different solver than conjugate gradient? and then implicit differentiation?
    - What does the costs in Table 1 represent? To my knowledge, direct methods are $O(n^3)$. Is this the memory cost? Is this the computer cost of "one iteration" of each method?
    - What exactly is the proposed approach? Is it equation 10b?

- Novelty:
    - I do not understand the novelty, especially since authors use an iterative solver: solving a linear system is equivalent to solve an optimization problem (as authors themselves state before equation 1), that is solve using well-known iterative methods. I.e., differentiating through the solution of linear system is equivalent to differentiate through the solution of an optimization problem.
    - I know that authors mentioned [1] in their related work, but could authors extend on the difference between their work and [1]? It would really help me understand the contribution of he paper. What is the difference with approaches like [2]?

- Experiments
    - IMO experiment in 3.1/Figure 3 does not meet top-conference standard requirements, and can be prone to misunderstanding: authors display the wall time as a function of the size of the problem. However, all the displayed methods are iterative, how was the number of iterations of each method was chosen/fixed? If the number of iteration is  not prefixed, what is the stopping criterion? Usually, methods all have different stopping criterion, which make them harder to compare. This details are not mentioned, nor in the main paper, nor in the appendix.  IMO, not mentioning this crucial details make this experiment inconclusive.
    - What are the baseline in Figure 3? I cannot find any mention of "AD (chekcpointed)" in the manuscript. Could authors compare to standard implicit differentiation [2]? with multiple inner solver, like conjugate gradient and/or the suggested one?
    - I do not understand how equation 12 differs from the setting in [2]

[1] Brandon Amos and J Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks

[2] Pedregosa, Fabian. "Hyperparameter optimization with approximate gradient." International conference on machine learning. PMLR, 2016.

### Questions
cf weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
