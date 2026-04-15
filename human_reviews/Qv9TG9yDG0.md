# Error Feedback for Smooth and Nonsmooth Convex Optimization with Constant, Decreasing and Polyak Stepsizes

- Decision: Reject
- Scores: 3, 8, 3, 5

## Abstract
Error feedback, originally proposed a decade ago by Seide et al (2014), is an immensely popular strategy for stabilizing the convergence behavior of distributed algorithms employing communication compression via the application of contractive compression operators, such as greedy and random sparsification, quantization, and low-rank approximation. While our algorithmic and theoretical understanding of error feedback has grown immensely over the years, several important considerations remained elusive. For example, the theory of error feedback is fully focused on the smooth convex and nonconvex regimes, and results in the nonsmooth convex setting are limited. This is not a coincidence: Error feedback works when the gradients converge, and this is not necessarily the case in the nonsmooth setting. Further, existing stepsize rules for error feedback are limited to constant schedules; a by-product of the current theoretical approach to analyzing error feedback. By modifying the algorithmic design of error feedback, we are able to resolve these issues. In particular, we provide a comprehensive analysis covering both the smooth and nonsmooth convex regimes, and give support for constant, decreasing and adaptive (Polyak-type) stepsizes. This is  the first time such results are obtained. In particular, this is the first time adaptive stepsizes have successfully been combined with compression mechanisms. Our theoretical results are corroborated with suitable numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this work, the authors analyzed the convergence rate of the error feedback algorithm EF21-P. The authors focused on the convex setting when the objective function can be a smooth or non-smooth convex function. Three different stepsize schema, constant, decreasing and Polyak, are considered and their convergence rates are proved. In addition, the authors provided the convergence rate when the stochastic EF21-P algorithm is applied and the objective function is a finite-sum loss function with the interpolation condition. Numerical examples are provided to support the theoretical findings.

### Strengths
The paper has is well-structured and easy to follow. The intuition behind the design and the theory of the new stepsize schema is clear and easy to understand. The results under the setting of distributed convex optimization should be novel (although a similar problem may have been studied under a different setting) and the proofs should be correct.

### Weaknesses
The major weakness of the paper is that the theory in this paper seems very preliminary. For example, in Theorem 2, the decreasing stepsize has a worse convergence rate than the constant stepsize. It would be curious if this is also the case for other schema of decreasing stepsize (e.g., $\gamma_k = O(k^{-1/3})$). Considering one specific decreasing scheme makes the theoretical contributions very limited. In addition, the authors defined Assumption 3 with a general $\eta$ but focused on two special choices, $\eta=0$ and $\eta=1$. This also seems too limited to me and may be generalized to other values of $\eta$. It would be helpful if the authors could generalize the results to other choices of decreasing stepsizes and other values of $\eta\in[0,1]$.

The authors claimed that the advantage of the Polyak stepsize is that it does not require estimating the Lipschitz constant and the optimal objective function value is easier to estimate. However, I feel that the authors did not provide convincing evidence that the optimal objective function value is easier to estimate than the Lipschitz constant, although I agree that in certain problems the optimal objective function value is exactly zero. I would suggest the authors provide more supporting evidences for this claim. For example, the authors may provide references or practical examples where the optimal value can be estimated a priori. This would help substantiate the claim about the advantages of the Polyak stepsize.

Finally, I feel that similar problem may have been studied by splitting-operator algorithms, where the compression operator $C$ can be the operator induced by the splitting operator (e.g., the Douglas–Rachford splitting or the Peaceman-Rachford splitting). It would be helpful if the authors could discuss the similarities and differences between the compression operator $C$ and a split operator. This could provide valuable context on the novelty and positioning of this work.

Overall, I feel that this paper presented some novel and interesting results, but the theory of the paper may have focused on a few very specific settings and this will limit the applicability of the results in this work.

### Questions
I have a few other minor comments for the authors to consider:

- The theory in Theorem 3 cannot capture the better empirical performance of Polyak stepsize than the constant stepsize. Maybe the authors could consider improving the constant before $1/\sqrt{K}$ or $1/K$.

- Lines 122 and 256: If EF21-P is equivalent to EF14, maybe the authors can choose to replace EF14 with EF21-P throughout the paper to avoid confusions.

- Line 270: $\alpha=1$ should be $\eta=1$. Also, in Theorem 1, $\alpha=0$ should be $\eta=0$.

- Assumption 3: I wonder if the assumption applies to the case when the subgradient is not unique at some points. In the case when the subgradient may not be unique, how do we choose the function $g(x)$ to guarantee the Holder continuity and the convergence? For example, in the SVM example, why did the authors choose the subgradient at line 499 when $b_{i}\langle a_{i}, x\rangle = 1$.

- Line 312: the current choice of $\lambda$ and $\theta$ only considers constant B in the inequality. This choice ignores their appearance in the Lyapunov function. I wonder if it is possible to derive the convergence rate with general $\lambda$ and $\theta$ and then choose the optimal value based on the convergence rate?

- Line 368, it should be for any initial iterates $x_0,w_0$. There are many other typos in the paper. I would suggest the authors check the paper carefully and fix the typos.

- Lines 373 and 377: $\hat{x}_{K}$ is not defined.

- Line 452: I cannot see why the interpolation condition implies that $f(x_*)=0$.

- In Figure 1, it would be better if the authors could show the performance of the constant/decreasing stepsize when they converge. It is hard to check if the current stepsize schema lead to global convergence, since it is possible that the current stepsize is too large or too small.

- In the main paper, the authors focused on the SVM example. However, the results of SVM are not presented in the main paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents new analysis for compressed (sub-) gradient descent with error feedback. The contributions include analysis for full batch GD for convex functions in the non-smooth case and analysis with decreasing and Polyak step-sizes. SGD with compression in the interpolation regime is also analyzed using a similar Lyapunov function as in the full batch case.

### Strengths
The paper is very well-written and easy to read. The authors invest substantial effort to explain their contribution compared to related work. This contribution, while not too original, is fair in the sense that presents analysis of a popular algorithm (error feedback) with popular choice of step-sizes apart from the constant one for which analysis already exists. In addition, there is analysis for non-smooth optimization problems (like support vector machines), which is not really surprising, but still a good to know result.

### Weaknesses
As discussed previously, my only issue with the paper is that it is blend of very-well known ideas, like error-feedback, adaptive step-sizes for GD, SGD in the interpolation regime etc.

### Questions
1. Is the Lyapunov function presented in Lemma 1 completely original, or there is some version of it in previous work?

2. It is repeatedly written over the results about Assumption 3 holding with $\alpha=0$ or $\alpha=1$, but $\alpha$ measures the quality of the compressor and Assumption 3 is about Holder continuity.

3. Why some to use diminishing step-sizes? The rate is worse and if I'm not mistaken one needs again to compute the generally unknown hyperparameters $L_0,L_1$.

4. If $f$ is not differentiable the subgradient can have many elements. What does Assumption 3 mean in this case?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies the error feedback mechanism under non-constant step sizes. In particular, with decreasing step sizes and Polyak step sizes. The author corroborated their theory with some numerical experiments.

### Strengths
The majority of the existing works on the Error Feedback mechanism focused on well-tuned constant-step sizes. This paper instead studies the case where the step sizes are changing. It also investigates the combination of EF with adaptive step sizes which seems to be the first in the field.

### Weaknesses
1. Seide's formulation of the Error Feedback mechanism (as is adopted in most existing works, e.g. [1]) is well-established in the literature. The authors refers to this formulation as EF14. I wonder why does the author, instead of just using this well-established formulation, use what they called "EF21-P" formulation? It seems to me that this EF21-P formulation is just writing the "virtual iteration" (used in the analysis of Seide's EF, see e.g. [1]) explicitly into the algorithm, which is a reasonable reformulation of Seide's EF but does not seem to bring any new benefit in the context of this paper. I would suggest the authors to just stick to the well-established formulation instead of rebranding it.

2. I do not understand the necessity of introducing the Hölder's smoothness condition. The paper focuses on two cases: 1) Lipschitz gradient and 2) Lipschitz continuity. Both cases are standard and extensively studied in the optimization literature. My understanding is that, when you bring up the Hölder's smoothness condition, you typically deal with "Universal methods" that automatically adapt to the best $\eta$, which is clearly not the case here. Since the theorems are stated separately for 1) and 2) anyway, I would suggest the authors to remove Assumption 3 completely to avoid confusions.

3. Regarding Lemma 1 and Lemma 2: The typical workflow of the analysis of variants of Seide's EF is the following: First you derive a descent lemma for the virtual iteration (which is the $x_t$ in this paper), and then you derive a descent lemma for the error term (which is the $w_t-x_t$ term in this paper), and then you combine the two into a Lyapunov function and proceed with the complexity analysis. Different regularity assumptions (e.g. convexity, smoothness, Lipschitzness) might affect the specific details of the derivations of these descent lemmas, but typically the overall template is not affected. The Lemma 1 and Lemma 2, though have been given strong emphasis in the main text (spanning 2 sections), seem to be a simple summary of the workflow I just described above. If these two lemmas somehow carries more insights, the authors should highlight it more clearly.

4. Regarding Section 5.1: These results seem to be well-known in the literature (see e.g. [1][2]). Is there any reason why they have to be there?

5. Regarding the changing step sizes: Typically the problem with changing stepsizes is extracted out and dealt with separately. In the case of decreasing step sizes, it seems to me that the backbone of the analysis is completely intact and one only has to do a different "summation lemma" for a complexity estimation (see, e.g. section 3.5 of [1]). Is it also the case in this paper? The case of Polyak stepsizes seems more interesting, but I would like to point out that, even though the authors claim in Table 2 that  for the Polyak step sizes the only required parameter is B, it is a bit disingenuous because you would also have to know $f^\star$.

[1] Stich, Sebastian U., and Sai Praneeth Karimireddy. "The error-feedback framework: SGD with delayed gradients." Journal of Machine Learning Research 21.237 (2020): 1-36.
[2] Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes SGD and other gradient compression schemes. In International Conference on Machine Learning, pp. 3252–3261. PMLR, 2019.

### Questions
see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper studies the convergence behaviors of EF21-P when using different kind of stepsizes (fixed, decreasing, polyak) under different cases (deterministic and stochastic, convex and non-convex).

### Strengths
This paper presents a comprehensive study on the stepsize scheduler of EF21-P. The work contains the convergence rates of EF21-P under deterministic convex case when using constant, decreasing, and Polyak stepsize, as well as convex stochastic under an interpolation condition.

### Weaknesses
I find the following weaknesses in the paper and hope the author can address them:

1. The paper only focuses on the stepsize scheduler and the related convergence rates on EF-21P, however, they do not compare their convergence rates with other distributed convex optimization methods. Hence, it is quite difficult to check their theoretical contributions.

2. For the stochastic cases, the authors consider an interpolation condition which is too strong in distributed optimization, which means the optimal point of $f$ is also the optimal point of every $f_i$.


Some typos:

line 270 $\alpha=1$ should be $\eta=1$ and all left theorems use $\alpha$, which should be replaced by $\eta$.

### Questions
Please check the weakness parts. Also with the following questions:

1. What is challenge of dealing with the non-smoothness in analysis? It seems that doing analysis on Assumption 3, one can directly obtain a general result that cover the case of smooth $\eta=1$ and non-smooth ($\eta=0$).

I am not an expert in this field and I shall read the comments from other reviewers and the rebuttal from authors and may change my score accordingly.

### Soundness
3

### Presentation
2

### Contribution
2
