# Optimization without retraction on the random generalized Stiefel manifold for canonical correlation analysis

- Avg Score: 6.50
- Decision: Reject
- Scores: 10, 5, 5, 6

## Abstract
Optimization over the set of matrices that satisfy $X^\top B X = I_p$, referred to as the generalized Stiefel manifold, appears in many applications such as canonical correlation analysis (CCA) and the generalized eigenvalue problem. Solving these problems for large-scale datasets is computationally expensive and is typically done by either computing the closed-form solution with subsampled data or by iterative methods such as Riemannian approaches. Building on the work of Ablin \& Peyré (2022), we propose an inexpensive iterative method that does not enforce the constraint in every iteration exactly, but instead it produces iterations that converge to the generalized Stiefel manifold. We also tackle the random case, where the matrix $B$ is an expectation. Our method requires only efficient matrix multiplications, and has the same sublinear convergence rate as its Riemannian counterpart. Experiments demonstrate its effectiveness in various machine learning applications involving generalized orthogonality constraints, including CCA for measuring model representation similarity.

## Human Reviews

## Human Reviewer 1

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a landing method for optimization over the generalized Stiefel manifold. The proposed method is inexpensive compared to classical optimization on Riemanian manifold. Furthermore the authors propose a stochastic version of the algorithm leading to an inexpensive and scalable method. The experiments show the good quality of the framework for different machine learning tasks.

### Strengths
This paper presents a general framework for optimization over the generalized Stiefel manifold. Such framework have many possible applications in the machine learning world (e.g. CCA, SVD...). The great strength of the method is its computational cost compared to classical framework for such optimization. The stochastic version is a great addition for scalability.

### Weaknesses
The framework has few weakness. The sublinear convergence shows that getting a suitable estimate may ask for many iterations as shown in the experiments. If the quality of the estimation is crucial the computation, even if inexpensive, can be long.

### Questions
I have one minor remark, on figure 3(b) of the left graphics the objective value followed the time when all others follow the iterations.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Optimization under generalized orthogonality constraints captures many applications in machine learning. This paper extends Ablin & Peyré's landing method, adapting it for stochastic settings to address problems involving generalized orthogonality constraints. 

The proposed method does not rigorously impose the constraint in each iteration but rather produces a series of iterations that gradually conform to the generalized Stiefel manifold. Therefore, it can avoid the retraction step. They show that the proposed method is convergent, and any clustering point is a critical point. The authors have conducted experiments to validate the efficacy of the proposed methods.

### Strengths
S1. A retraction-free algorithm is introduced to solve the optimization problem involving generalized orthogonality constraints.

S2. The authors demonstrate that the proposed method converges to a critical point of Equation (1) in both deterministic and stochastic cases.

S3. They provide numerical evidence of the proposed method's efficiency through the deterministic generalized eigenvalue problem and through the stochastic CCA.

### Weaknesses
W1. The adaptation of Ablin & Peyré's landing method to accommodate the general setting with $\mathbf{B}\neq \mathbf{I}$ and the stochastic context is straightforward.

W2. The convergence results are relatively weak. These results could be readily derived using the sufficient descent condition.

W3. The time complexity comparison in Table 1 is biased, and the motivation for the retraction-free method is not strong. The authors argue that the polar method has a time complexity of $O(n^2 p)$, while the proposed method can leverage the rank-r structure of matrix $\mathbf{B}$ and achieve a time complexity of $O(\min(n^2 p, nrp))$. However, if the polar method takes advantage of the rank-r structure of matrix $\mathbf{B}$ for matrix multiplication, its complexity could also be greatly reduced when employing the Woodbury matrix identity.

W4. The paper lacks comparisons with the Multiplier Correction Methods (Gao et al., 2019a, 2022a). The authors only briefly mention that the Multiplier Correction Methods are sensitive to the appropriate selection of the penalty parameter, but a direct comparison with these methods is necessary.

W5. The authors should compare their method against the modified Gram-Schmidt-based generalized polar decomposition method (e.g., "A practical Riemannian algorithm for computing dominant generalized Eigenspace, Z Xu, P Li, UAI 2020").

W6. The parameter $\omega$ is crucial for the convergence of the algorithm. The authors should demonstrate how this parameter's sensitivity affects the performance of the algorithm.

### Questions
Is the inequality in Equation (11) tight? What are the values for the upper bound (namely $\eta(x)$) in practice? I suspect that restricting the step size could greatly limit the practical efficacy of the proposed algorithm.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper delves into the optimization problem associated with the generalized Stiefel manifold, employing methods that do not involve retraction. The authors introduce a novel 'landing' algorithm, extending prior work by Ablin and Peyre (2022). Notably, the landing algorithm offers the advantage of efficient iteration computation complexity. The study investigates the convergence properties of both deterministic and stochastic algorithms when applied to the generalized Stiefel manifold.

### Strengths
This paper generalizes previous landing algorithms to stochastic setting for both constraint and objective. The convergence results are given for deterministic and stochastic cases.

### Weaknesses
1. The paper lacks a clear explanation regarding the rationale behind the use of two random variables, $\zeta$ and $\zeta'$, as mentioned in equation (3). While the paper briefly mentions that noise in the tangent space is allowed and that the relative gradient has an unbiased estimate, it fails to adequately address how this difficulty is overcome in the main context of the research. It would be beneficial to provide more clarity on this aspect in the main body of the paper.

2. Table 2 reveals that previous works have enjoyed a linear convergence rate, whereas this work achieves a sublinear rate. It is essential to acknowledge this difference in convergence rates. Furthermore, the paper's proved results appear to be more favorable when the condition number of matrix $B$ is low, and this should be explicitly mentioned.

### Questions
1. In the introduction, the paper stipulates that $B_{\zeta}$ must be positive definite. However, this requirement implies that the sample size should exceed the dimensionality $n$. Could the authors clarify why this is a necessary condition and how it relates to the sample size?
2.  Could the paper provide insight into the derivation of the distinct formulas for $\Phi_B(X)$ and $\Phi_B^R(x)$ as presented in Proposition 3.2? Are these formulas related to different metrics? Additionally, in Figure 2, the term 'landing (Riem. gradient)' is used. It seems that "Riemannian steepest descent with QR-based Cholesky retraction (Sato & Aihara, 2019)" is a retraction-based method. Could the paper explain why the distance $\mathcal{N}(x)$ is not zero in this context?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper the authors propose an optimization method in the (random) generalized Stiefel manifold, which doesn't require a retraction to the manifold. This work follows the line of some recent papers, which started by optimizing on the orthogonal group (Ablin and Peyré). All these formulations are based on a flow with two terms, one trying to minimize the function (or the norm of the gradient), and the other one minimizing the distance to the manifold. These terms are orthogonal to each other, under certain smart choices.

This paper contributes to the field in three ways: by considering a more general case for directions in the flow, by explicitly proposing a method for the generalized Stiefel manifold, and by addressing the random case.

The paper is well written in general. The theoretical part is sound. However, the experimental section is not convincing.

### Strengths
Out of the three contributions (more general descent directions, general Stiefel manifold, and random $B$), I think that the first one and the third one are the most important contributions.

The theoretical results are important and non-trivial.

### Weaknesses
To me, one of the strenghts is the application to the stochastic framework. However, the experimental section is particularly weak in that point. I don't know what to look for in Figure 3 (by the way, one label says "time", I assume incorrectly, and I don't know what "5 epochs" means in this context). The description in the text doesn't help much.

### Questions
In the paper there are results concerning feasible steps, bounds, and the existance of useful positive steps. How is the stepsize chosen in practice? At least in the experiments.

The choice $\Psi_B(X)$ seems to do a better job than $\Psi_B^R(X)$. Do you have any intuition about why that is?

I assume that the need for an inverse computation in $\Psi_B^R(X)$ makes this choice slower. What's the comparison in execution time?

Minor comments:

 - In Lemma 2.4, it reads "where $C_h$ is from Assumption 2.2", but $C_h$ is not present in the inequality. I think it should be $L_N$.

 - In the supplementary material, between equations (74) and (75), starting with the phrase "For the other choice of relative gradient ..." and to the end of the section, I think it would be better to use the same notation as in the paper, namely, $\Psi_B^R(X)$.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
