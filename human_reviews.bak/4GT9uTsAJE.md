# AdaGrad under Anisotropic Smoothness

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Adaptive gradient methods have been widely adopted in training large-scale deep neural networks, especially large foundation models. Despite the huge success in practice, their theoretical advantages over classical gradient methods with uniform step sizes across all coordinates (e.g. SGD) have not been fully understood, especially in the large batch-size setting commonly used in practice. This is because the only theoretical result that can demonstrate this benefit was obtained in the original paper of Adagrad for convex nonsmooth objective functions, which is insufficient for large batch algorithms. In this work, we attempt to resolve this gap between theory and practice by proposing a novel anisotropic generalized smoothness assumption and providing corresponding analysis of Adagrad. It is shown that under anisotropic smoothness and noise conditions, AdaGrad can achieve faster convergence guarantees in terms of better dimensional dependence than algorithms with uniform step sizes across all coordinates. Experiments in logistic regression and instruction following fine-tuning tasks provide strong evidence to support our novel assumption and theoretical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzed the convergence rate of AdaGrad, showing that AdaGrad converges faster than SGD with respect to the dimension of parameters.

### Strengths
1. This paper analyzed the convergence rate of AdaGrad. Then, it showed that the convergence rate of SGD depends on $D_{\infty}$, while the rate of AdaGrad depends on $D_2$. $D_2$ depends on the dimension of parameters, while $D_{\infty}$ does not. Thus, this paper claimed that AdaGrad converges faster than SGD.

### Weaknesses
1. The authors compared the convergence rate of AdaGrad in Theorem 4.1 and the convergence rate of SGD, Eq. (6). The convergence rate in Eq. (6) depends on $D_2$, while the tighter convergence rate that depends on only $\| x_0 - x^\star\|$ was more common. 

2. $D_\infty$ does not depend on the dimension of the parameter. However, it is unclear whether $D_\infty$ is smaller than  $\| x_0 - x^\star\|$.

2. Theorem 4.1 assumes that $L_1 = 0$, which sounds a bit strong assumption. The reviewer feels that it would be better to provide the intuition why this assumption is necessary, at least in the Appendix.

2. It was confusing which term corresponds to "bias term" in line 272.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies the convergence of Adagrad under anisotropic smoothness conditions. The main contributions are:
1. Defining the anisotropic smoothness condition.
2. Studying the convergence of Adagrad for convex and nonconvex problem under the anisotropic condition.
3. Further nonconvex results for relaxed smoothness conditions. 

Strength:

I think the paper is presented in a very clean and organized manner. The key results and discussions are clear. 

Up to my knowledge, although anisotropic smoothness were hinted across different setups, there is no very systematic study prior to this work. Therefore, I think the results here can be a valid contribution to optimization theory.

Weakness:

-The results are not surprising, and hence I didn't find the analysis / statements to be novel.


-In addition to reviewing adagrad analyses, it would be helpful to review anisotropic analysis. Several related works that I could think of : analysis on coordinate descent;  Nesterov's study on zero-order methods; adagrad's advantage on sparse gradients; Adam's convergence on infinity norm rather than l2 norm of gradients, etc.

Although, the above results probably are not directly comparable, it would be good to summarize and discuss the differences.

Some results that can make the work more impressive are listed below:

-Lower bounds to justify when and why adaptive step,  diagonal / full matrix adaptivity are sufficient / insufficient would be very interesting. 

-Given the analysis, can faster methods be proposed for neural nets?

### Strengths
See summary

### Weaknesses
See summary

### Questions
See summary

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work provide analysis result for the convergence of Adagrad for training of machine learning models with large batch size, emphasizing the effects of anisotropic smoothness. The authors then compare the results with similar results for SGD and Adagrad-norm and point out the potential of Adagrad. In general, the work can be helpful to under stand the training process.

### Strengths
The work provides an analysis result which may be the first one for Adagrad. This can be helpful for others to understand the potential of Adagrad and select optimizers for training tasks.

### Weaknesses
The numerical results are not sufficient to verify the assumptions and analytic results.

### Questions
1. Convexity is used in a large part of the paper. In many machine learning models, there are more parameters than data. In this case, local minima w* may not be isolated points. Instead, it can be a manifold. What is the impact of overparametrization to the results in this work?

2. In Table 2, the authors list the coefficients and norms in the analytical results. It is also important to see how well the convergence of the loss (or gradients) are controlled by these coefficients and norms.

3. For Table 4, since the work mainly discusses the convergence rate of Adagrad, it is better to show how the loss converges and how do the  authors select hyperparameters.

### Soundness
4

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
3

### Summary
This paper provides a detailed analysis of the AdaGrad optimization algorithm under anisotropic smoothness assumptions, addressing gaps in theoretical convergence for large-scale tasks. It introduces a new anisotropic smoothness framework that better explains AdaGrad’s convergence speed, especially for large-batch training. Experiments on logistic regression and GPT-2 fine-tuning support these theoretical claims, showing AdaGrad’s improved performance over SGD.

### Strengths
The main strength lies in its novel anisotropic assumptions, which align well with AdaGrad’s observed performance in high-dimensional settings. The experiments effectively validate the theoretical benefits, highlighting AdaGrad’s adaptability to large batch sizes and diverse data structures. For the rest it is a standard optimization analysis.

### Weaknesses
This kind of work always relies on assumptions which limits their applicability to the setting of interests, as neural networks. However, this is common and not really an issue. See also questions.

### Questions
Can please you better compare to Convergence Analysis of Adaptive Gradient Methods under Refined Smoothness and Noise Assumptions - D Maladkar, R Jiang, A Mokhtari - arXiv preprint arXiv:2406.04592, 2024?
Also, is much work required to generalize to Adam?

### Soundness
3

### Presentation
3

### Contribution
3
