# High Probability Bounds for Non-Convex Stochastic Optimization with Momentum

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Stochastic gradient descent with momentum (SGDM) is widely used in machine learning, yet high-probability learning bounds for SGDM in non-convex settings remain scarce. In this paper, we provide high-probability convergence bounds and generalization bounds for SGDM. First, we establish such bounds for the gradient norm in the general non-convex case. The resulting convergence bounds are tighter than existing theoretical results, and the obtained generalization bounds seem to be the first for SGDM. Next, under the Polyak-{\L}ojasiewicz condition, we derive bounds for the function-value error instead of the gradient norm, and the corresponding learning rates are faster than in the general non-convex case. Finally, by additionally assuming a mild Bernstein condition on the gradient, we obtain even sharper generalization bounds whose learning rates can reach $\widetilde{\mathcal{O}}(1/n^2)$ in the low-noise regime, where $n$ is the sample size. Overall, we provide a systematic study of high-probability learning bounds for non-convex SGDM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper establishes high probability convergence and generalization upper bounds for non-convex SGDM. Specifically, it first considers the general non-convex case. The convergence results are better than previous works and the generalization bounds are the first ones for SGDM. Then, when PL condition holds, faster rates are achieved than the general non-convex case. Finally, further assuming Bernstein condition brings sharper generalization bounds in the case of low noise.

### Strengths
1.This paper utilizes some special properties of a heavy-tailed distribution, Sub-Weibull distribution, to develop the theoretical analysis of SGDM.

2.The whole paper is written clearly and easy to be understood.

### Weaknesses
1.The inequality in line 1240 may not be true, considering that $(1-\gamma^{t-i+1})$ is the coefficient of the gradient $\nabla F_S(x_i)$. We can not ensure the norm with $(1-\gamma^t)$ is larger than the one with $(1-\gamma^{t-i+1})$. Therefore, it may greatly affect the subsequent proof.

2.It is better to make a table to list all convergence and generalization results and all previous results, which facilitates readers' understanding of the advantages of their results over previous works.

### Questions
1.Why do the authors use two symbols $f$ and $g$ in Assumption 2.1?

2.There are some typos, such as “smothness”in the last line of page 3 and $k=2,...，$ in Remark 2.3.

3.The whole analysis framework is very similar to [1] except for the lemma which interchanges the order of summation (Lemma C.6) to enable the analysis of [1] in the SGDM case. Due to this reason, all results are the same as those of [1]. Is my understanding right? If so, do these results give us any insights?

Note: From my perspective, the major innovation of proof is mainly presented in Theorem 3.1 and Theorem 3.3, which is listed in **Q3**. If any other innovations exist in the proofs of other results compared with [1], please make them clear. If so, I will improve my score.

### Soundness
2

### Presentation
3

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
Strengths

1. First high-probability analysis of SGDM in non-convex settings: The paper provides the first comprehensive high-probability convergence and generalization guarantees for stochastic gradient methods with momentum under nonconvex objectives, bridging an important theoretical gap in stochastic optimization.

2. Sharp convergence rates with structural conditions: The results establish an \tilde{O}(1/\sqrt{T}) high-probability convergence rate in the general nonconvex case, improving to \tilde{O}(1/T) under the Polyak–Łojasiewicz condition and even \tilde{O}(1/n^2 + F^*/n) under a Bernstein assumption—showing remarkable theoretical depth.

3. Empirical results align with the theory: Experiments on multiple LIBSVM datasets systematically verify the predicted influence of the tail parameter $\theta$ on convergence speed, offering intuitive visual confirmation of the theoretical high-probability trends.

Weaknesses

1. Limited experimental scope and scalability: Experiments are confined to small-scale logistic regression tasks. The absence of results on large-scale or deep learning benchmarks makes it unclear how well the high-probability bounds manifest in practical training.

2. Lack of practical runtime validation: Theoretical IFO complexity and convergence bounds are not accompanied by empirical runtime or gradient-call comparisons, leaving efficiency gains largely unquantified.

### Strengths
see Summary

### Weaknesses
see Summary

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies high-probability convergence and generalization bounds for SGDM in non-convex settings, achieving faster rates under the Polyak-Łojasiewicz (PL) and a Bernstein-type gradient condition.  
Specifically, assuming that the stochastic noise follows a sub-Weibull distribution parameterized by $\theta$, the authors establish a slightly tighter bound at $\theta = 1/2$ (improving the logarithmic factor from $\log(T/\delta)$ to $\log(1/\delta)$) and extend the analysis to the case $\theta > 1$, compared with prior work.

### Strengths
1. The paper provides convergence and generalization bounds for SGDM under arbitrary values of $\theta$, offering a unified treatment that covers a wide range of noise distributions from sub-Gaussian to heavy-tailed cases.

2. The focus on SGDM rather than plain SGD is interesting, as understanding how momentum affects convergence and generalization remains an important open question in stochastic optimization.

### Weaknesses
1. For the case $\theta = 1/2$, the improvement over related work is essentially from $\log(T/\delta)$ to $\log(1/\delta)$ while keeping the same leading-order rate. This represents a mild tightening rather than a substantive advance.

2. The extension to the case $\theta > 1$ relies on general concentration inequalities applicable to arbitrary $\theta$ (Appendix Lemmas C.2–C.4). It seems that the main change lies in using a more general tail inequality, which does not introduce substantial technical difficulty or genuine novelty.

3. The paper is difficult to follow, with many typos and inconsistencies. Examples include:

   Line 161: “smothness” → “smoothness”.  
   Assumption 2.1: The definition of smoothness should not be embedded in the assumption itself.  
   Theorems: All numbering “(1.) (2.) (3.) (4.)” should be corrected to “(1). (2). (3). (4).”. 

   Line 286: “study” should be “studied”.  
   Theorem 5: The term $\mu(S)$ appears before being defined.  
   Line 373: “theFS” should be “the FS”.  
   Table 1: The first two rows list identical assumptions but yield different error bounds—this should be clarified.


To enhance coherence and credibility, I recommend that the authors thoroughly revise the paper for both clarity and presentation quality.

### Questions
1. Under the same assumptions as in the main theorems, are there corresponding high-probability convergence and generalization results for SGD (without momentum)? A direct comparison would help clarify whether momentum offers any provable benefit in this setting.

2. Which concrete model classes (e.g., deep neural networks) are known to satisfy the PL condition?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors have obtained high probability gradient norm bounds in a general non-convex problem for stochastic gradient descent with momentum (SGDM). Then under PL condition of loss, they show high probability bounds on the function values and achieve faster rates under Bernstein condition for gradients.

### Strengths
The authors have done a very good job in terms of literature review. The paper is overall very well written. The assumptions are clearly provided and the theoretical results are well stated. Establishing both convergence and generalization bounds in high probability with clear description is very interesting.

### Weaknesses
This is a nice submission. While I note that the authors have focused on theory, providing a toy experiment in Appendix A will be the main weakness of this submission. I think the paper will be significantly improved by providing more relevant experiments with clear connection with the developed rates under various assumptions and potentially in a more practical setting.

### Questions
Please check the weaknesses

### Soundness
4

### Presentation
3

### Contribution
3
