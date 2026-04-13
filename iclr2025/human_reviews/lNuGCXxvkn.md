## Human Reviewer 1

### Summary
This paper examines the statistical performance of kernel-based methods for linear elliptic Partial Differential Equation (PDE) inverse problems. For the regularized least squares estimator with a reproducing kernel Sobolev space (RKSS) norm as the penalty term, and the minimum norm interpolation estimator, this paper derives learning curves in terms of the RKSS norm for both estimators. Utilizing the bias-variance decomposition from an explicit representation theorem of the estimators, the paper demonstrates that elliptic operators, under certain capacity assumptions, can exhibit benign-overfitting behavior for fixed-dimensional problems. Furthermore, the results indicate that all regularized least squares estimators, with appropriately chosen regularization parameters, can achieve an optimal convergence rate. This convergence rate is independent of the choice of inductive bias, provided that the inductive bias is sufficiently smooth.

### Strengths
This paper introduces novel methods to search for solutions of PDE problems. Distinct from prior works, this paper uses RKSS norm as penalty term instead of RKHS norm, and demonstrate firm proofs for the convergence rate of the models. Impressively, the convergence rate is actually optimal in certain circumstances. The result of independence of convergence rate on inductive bias also provides guidence to determine suitable kernel functions, inductive bias and regularization parameters.

### Weaknesses
The assumptions adopted in this paper are strong, especially the capacity assumption (Assumption 2.2(d)). This paper needs to demonstrate more detailed and nontrivial examples where the assumptions are satisfied and the models are computed explicitly. 
Besides, both the regularized least square loss function in the problem settings (1)(2) and the representer provided in Lemma 3.1 are involved with the inductive bias and the RKSS norm, hence the numerical computations are neither trivial nor straightforward. This paper lacks numerical experiments to demonstrate the practical performance of the algorithms mentioned.

### Questions
First, I prefer to call the phenomena shown in the main theorems (Theorem 4.1 and Theorem 4.2) ‘generalization ability’ of the models, and I am confused with the term ‘benigning overfitting’ in the title. Can you explain more how the models are overfitting? 
Besides, the capacity condition on elliptic operators (Assumption 2.2(d)) seems too strong, making the problem setting of regularized least square (1) very similar with the setting of kernel ridge regression. Except the RKSS norm penalty instead of RKHS norm penalty, is there any other essential difference between kernel ridge regression and the regulized least square model in this paper?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
The authors show that certain design of norms (called Kernel Soboleve norm) can lead to more stability of the variance and even benign overfitting for fixed-dimension problems. (Previous work suggests that common kernels might need high dimensionality to do that) Furthermore, they show that inductive biases don't affect convergence rate, with optimal achieved with appropriate parameter chosen. Also some surprising connection in Bayesian settings are discovered.

### Strengths
Paper solid. Handles sophisticated mathematical theories.

The authors show a refreshing story that even for fixed dimension one can tune the kernel to achieve benign overfitting. This complements the storyline of kernel ridgeless ridge very well.

The authors present rich results with depth on the subject.

### Weaknesses
The title "Physics-Informed" is not 100% appropriate. Physics seems to be indirectly connected to the theme of the paper. Physics -> PDE -> Sobolev norm -> Kernel Sobolev norm. Sobolev is quite a general notion, a natural choice for quantifying smoothness. Many different settings could lead to Sobolev, not just Physics. It would have been more justified, if many empirical experiments related to Physics were done in the paper. For me personally, "Physics-informed" is misleading and gave the wrong impression before I fully read the paper.

The essence of the proof and theory is not explained clearly. Essence is the core mechanism that makes a difference, an intuitive explanation of which makes it easy to understand why things work before eating all technical details and to credit the paper for its theoretical contribution. The authors show a phenomenon very different from previous work on the subject. Why the kernel proposed in the paper can work for benign overfitters even for fix dimensionality? My understanding is that, the kernels in the paper have more flexible spectrums, making it possible to simultaneously making both biases and variances small.

The authors show that "the choice of smooth enough inductive bias also does not affect convergence speed". To make this claim sounder, some empirical experiments are needed. Otherwise, it looks like it might just be theories are too intractable to lead to a difference.

Some statements are confusing. It's said with emphasis that "Our results show that the PDE operators in inverse problems possess the capability to stabilize variance and remarkably behave benign overfitting ...". However Gaussian kernels are the inverse of laplacian operators and previous work has proven Gaussian kernels for any bandwidth has a nonzero lower bound on error rate for fixed dimension. And it's natural to think Laplacian operators as suitable for inverse problems (although I'm not familiar with that). It would be better if there are more explanation on the difference between these kernels and previous ones.

### Questions
Included in weakness.

### Soundness
4

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper studies performance of solving inverse problems using kernel based methods. Considering the least square problem regularized by kernel Sobolev norm or the interpolator, under certain assumptions, this paper shows the generalization error in terms of bias-variance decomposition. This paper further discusses the implications of the results that how the inductive bias from the chosen kernel Sobolev space can affect the generalization.

### Strengths
- This paper is well-organized and accessible, with clear presentation of the main results even for those unfamiliar with complex notations and quantities.
- This paper studies the solution of inverse problems using least squares regression in RKHS, establishing upper bounds. This contributes to the theory of physics-informed machine learning.

- The finding that "the PDE operator in the inverse problem enables benign overfitting even in fixed-dimensional settings" is interesting and provides new insights into physics-informed machine learning.

### Weaknesses
- Problem Setting: The paper does not consider boundary conditions, which are essential for inverse problems, which limits the applicability and the implications of the results.

- Strong Assumptions: Assumption 2.2 (d) is particularly strong, requiring the kernel's orthonormal basis to diagonalize the operator.  Additionally, with operator  diagonalized by the kernel's basis and without boundary conditions, the analysis closely follows that of kernel ridge regression as presented in Barzilai & Shamir (2023), making the techniques used less novel.

- Lack of Empirical Validation: There is no empirical validation to verify the effectiveness of the method. Experiments are necessary to convincingly demonstrate the practical utility of the proposed approach across different types of inverse problems, including those with boundary conditions or in high-dimensional settings.

### Questions
1. In Theorem 4.2, there is a term $O(1/\log n)$ in the probability bound, making it almost vanishing. Can this bound be improved?

2. I would like the authors to discuss in detail how $ \rho_{k,n}$ behaves in concrete and realistic settings. It seems that I cannot find a thorough analysis in the appendix as suggested in Remark 6.

3. For real-world applications, how can we choose the hyperparameters in the proposed method empirically?

4. How does the dimensionality of $\mathcal{X}$ affect the results, particularly in terms of generalization and variance stabilization?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces an approach for the solution of inverse problems, where the forward problem is assumed to be given by a linear, self adjoint elliptic operator,  via a kernel-based fitting (least-squares of interpolation) regularized with a kernel Sobolev $\beta$ norm. 
The focus is particularly on the role of the parameter $\beta$ in terms of the existence of a non-asymptotic 
benign overfitting regime.

### Strengths
- The topic is of current interest, and the results are clearly presented, discussed, and put into perspective in the existing literature. 
- The new results are interesting and informative w.r.t. the role of the various parameters.

### Weaknesses
- The results are presented and discussed in terms of general machine learning techniques (see e.g. the mention of PINN in "Takeaway to Practitioners" in Section 4.3), but it's not directly clear how the result of the paper can be applied beyond kernel-based methods.

- The results rely on a number of assumptions (Assumption 2.2) that are a bit hard to transfer to practical cases. For example, Assumption 2.2, point (d): The co-diagonalization assumption is quite restrictive, and does not generally hold if A is PDE operator and K is chosen freely. 

- Similarly, Assumption 3.4 is hard to relate to actual kernels. In particular, although the examples in Remark 3 are correct, $\beta_k$ in Assumption 3.4 is usually $\infty$, as the first sum in the maximum can be diverging, see e.g. [2]. This is the case with many commonly used kernels. Especially, I believe the assumption $\beta_k=\Theta(1)$ is generally too restrictive.

[2] Ding-Xuan Zhou, The Covering Number in Learning Theory, Journal of Complexity (2002)

### Questions
Regarding the three points discussed above, I would suggest the following:
-  Discuss how the results could potentially be extended or applied to non-kernel methods.
-  Provide practical caes of PDEs and kernels that fit into Assumptions 2.2, or otherwise discuss how the results might be extended if the co-diagonalization assumption is relaxed. 
- Provide practical examples of kernels (beyond Remark 3) that fit into Assumption 3.4, or discuss how the results might be extended if these hypotheses are relaxed. 

Apart from the points discussed above, there are the following minor points: 
 
- Example 2.3: I believe Bohner should be Bochner.
- in "Decomposition of signals", end of page 5: Is the map $\phi_{>k}^*$ well defined for any $\theta\in\mathbb{R}^\infty$?
- Assumption 3.5 is cited in Theorem 3.2, before being introduced.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3