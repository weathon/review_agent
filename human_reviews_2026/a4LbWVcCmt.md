# Generalized Smoothness in Stochastic Convex Optimization: First- and Zero-Order Methods

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
This paper is devoted to the study of stochastic optimization problems under the generalized smoothness assumption. By considering the unbiased gradient oracle in _Stochastic Gradient Descent_, we provide strategies to achieve in bounds the summands with exponential objective decrease. In particular, in the case $L_0 = 0$, we obtain in the **convex setup** the iteration complexity: $N = \mathcal{O} \left( L_1R \log\frac{1}{\varepsilon} + \frac{L_1 c R^2}{\varepsilon}\right)$ for _Clipped Stochastic Gradient Descent_ and $N =  \mathcal{O} \left(L_1R \log\frac{1}{\varepsilon}\right)$ for _Normalized Stochastic Gradient Descent_. Furthermore, we generalize the convergence results to the case with a biased gradient oracle, and show that the power of $(L_0,L_1)$-smoothness extends to _zero-order algorithms_. Finally, we demonstrate the possibility of the zero-order algorithm outperforming the first-order algorithm in the convex setup through numerical experimentation, which has aroused some interest in the machine learning community -- logistic regression problem.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies both first-order and zero-order algorithms for stochastic convex optimization under generalized smoothness. The key result is the newly discovered linear term in the convergence rate.

### Strengths
The paper is easy to follow.

### Weaknesses
This paper has many undesirable points from various perspectives.

**Format**

1. As far as I can see, all citation is based on \citet. However, many places should be changed to \citep.

1. In many places (e.g., Line 038), including the appendix, "equation x" should be either "equation (x)" or "(equation (x))".

1. In numerous places (including the appendix), the articles are missing. For example, line 048 should read "an unbiased gradient oracle". The authors should polish the writing more carefully.

1. Line 294, Anonymous (2025) is not a proper citation since the paper is non-anonymous.

**Theory/Proof**

1. Line 120, assuming a uniform upper bound $M$ on the gradient is too strong, especially when considering a (generalized) smooth problem on the domain $\mathbb{R}^d$. For example, no quadratic optimization can satisfy this assumption.

1. Line 823, $\nabla f(x^{k+1})$ should be $f(x^{k+1})$.

1. The authors use the inequlaity $\\\|x^k-x^* \\\| \leq\\\|x^0-x^*\\\|$ many times (e.g., Line 840). However, I cannot see why.

1. Line 854, I cannot see any formal reason for this recursion step, since Line 852 only holds conditionally. Note that similar steps also has been used many times (e.g., Line 908).

1. Line 1062, I cannot see why this step holds given the previous derivations (especially, considering they contain some errors).

1. Line 1239, it seems $\frac{\eta}{2}$ should be $\frac{\\\|\nabla f(x^k)\\\|\eta}{2}$.

1. Line 1661, $\frac{c}{2}$ should be $\frac{2c}{3}$. Moreover, why can it be assumed $\zeta\leq \frac{c}{3}$?

1. Line 1775, $x^k$ should be $x$. This step also requires $\gamma\leq 1/L_1$.

**Experiments**

1. Line 439, $M$ has been used to denote the uniform upper bound of the gradient norm. Please consider changing it to another letter.

1. Line 444, $L$ should be $\frac{1}{4M}\lambda_{\max}(A^TA)$.

1. Line 449, I think $h$ should be $\gamma$.

1. Line 456, why does $\eta=1/(c\\\|A\\\|_1)$ correspond to your step size? Do the authors mean the upper bound on the step size in Theorem 3.1? If so, please provide the formal calculation. In addition, I cannot find the definition of the $1$-norm $\\\|A\\\|_1$ for a matrix.

1. How do the authors set all hyperparameters for all four of these algorithms? The authors only state partial of them, e.g., the stepsize of ClipSGD.

1. I cannot find how the starting point is initialized.

1. People always preprocess data to make every line of $A$ have a unit length. Did the authors do so? If not, I wonder what will happen after this operation.

1. The experiments also lack error bars. This is a very important index for understanding the performance of any stochastic optimization algorithm.


**Others**

1. Line 047, as far as I know, Gorbunov et al. (2020) didn't contain any experiments about deep learning models. Therefore, placing it here is not proper.

1. The term $(L_0, L_1)$-smoothness first appeared in Line 058 without any reference. Please add a proper reference.

1. For Figure 1, please provide the concrete value of $(L_0, L_1)$ for $\\\|x\\\|^{2n}$.

1. In Table 1, the term ''Maximum Noise Level'' is used without any reference/definition. It is not explained until Section 5. Please explain it earlier or point the reader to Section 5.

1. In Table 1, the notion $\tilde{\sigma}$ is also used without any definition until Theorem 5.2.

1. Line 122, $x^*$ is used without any definition.

1. The authors emphasize the case of $L_0=0$ many times. However, as the authors mentioned in Remark 1.3, the possible function class is very limited. Could the authors provide extra practical examples beyond the exponential and logistic loss?

1. Line 157, to the best of my knowledge, people rarely call the bounded variance condition itself heavy-tailed noise. Instead, heavy-tailed noise often refers to a bounded $\alpha$-th central moment on the noisy gradient for $\alpha\in(1,2)$ (sometimes including 2).

1. The discussion under Theorem 3.1 related to Gaash et al. (2025) is not convincing. Please note that the result of Gaash et al. (2025) is obtained under the existence of an optimal solution in $\mathbb{R}^d$, meaning that one cannot take $L_0=0$.

1. Linie 226, the sentence "The summand..." is not well-written. What does it mean that "The summand" is "a typical SGD"? Please rephrase it.

1. Linies 227 and 232, it should be $\eta=[4(L_0+L_1c)]^{-1}$ according to theory.

1. In Algorithms 2, 3, and 4, $x_0$ should be $x^0$.

1. Line 285, please note that Cutkosky & Mehta (2020) studied non-convex problems and required a large batch size for NSGD. However, this work focused on convex optimization. Therefore, whether a large batch size is indeed required is unclear.

1. Line 721, it is not necessary to emphasize $n\in\\\{2,3\\\}$. The subscript $2$ in $\\\|\\\|_2$ is also redundant.

1. Line 733, the third $\mathbb{E}$ on the R.H.S. is redundant.

1. Line 2304, $R=\mathrm{arginf} f(x)=\infty$ doesn't make sense.

### Questions
Please refer to **Weaknesses**.

### Soundness
1

### Presentation
1

### Contribution
1

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
1. Comprehensive generalization under (L_0,L_1)-smoothness: The paper systematically studies several first- and zero-order stochastic optimization methods under the generalized smoothness assumption, offering a unified and well-structured theoretical framework.

2. Novel exponential decrease in convex settings: Demonstrating exponential objective descent when L_0 = 0 is theoretically significant—it challenges the conventional boundary of sublinear convergence in convex optimization.

3. Complete theoretical extension to zero-order methods: Extending the (L_0,L_1)-smooth framework to zero-order algorithms and providing rigorous bounds represents a clear theoretical advancement.

4. Experimental validation supports theory: Logistic-regression experiments are consistent with the theoretical claims and visually confirm the predicted convergence patterns.

Weaknesses
1. Derivations lack detailed intermediate steps: Theorems 3.1, 4.1 omit crucial reasoning chains connecting (L_0,L_1)-smooth conditions with convergence bounds.

2. Batch size assumptions are unrealistic: The required batch size (e.g.,B=O(σ²MR³/ε³) or O(d σ² MR³/ε³)) is computationally infeasible in practical large-scale training.

3. Limited experimental diversity: The evaluation focuses on logistic regression only. Testing on more benchmark functions (e.g., quadratic, hinge loss, or non-smooth proxies) would offer stronger empirical support for the generalized theory.

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
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies stochastic convex optimization under the **(L₀, L₁)-smoothness** assumption, a generalization of standard Lipschitz smoothness. It analyzes first-order (ClipSGD, NSGD) and zero-order (ZO-ClipSGD, ZO-NSGD) algorithms and claims **exponential objective decrease (linear convergence)** under certain conditions.  

The authors provide convergence bounds for both unbiased and biased gradient oracles and extend these results to zero-order methods using gradient approximations. The paper also includes a small experiment on logistic regression (w1a dataset), showing that ZO-NSGD can outperform ClipSGD.  

However, the claimed “linear convergence” is **only in iteration count**. Once batch size scaling (e.g., \(B = O(\varepsilon^{-3})\)) is considered, the **total oracle or sample complexity remains sublinear**. Thus, the main result does not represent a true acceleration in computational terms, and the contribution is incremental.

### Strengths
- Provides a **unified theoretical analysis** covering both biased and unbiased gradient oracles.  
- Improves iteration complexity bounds over prior works (e.g., Gaash et al., 2025) (Theorem 1)  
- Clearly structured derivations and comparison table aid understanding.

### Weaknesses
- **Misleading main claim:** Linear convergence holds only in terms of iteration count; total computational complexity remains **sublinear** once batch scaling is considered.  
- **Marginal novelty:** Extends deterministic analyses (e.g., Lobanov et al., 2024b) to stochastic and zero-order cases without new techniques.  
- **Restrictive assumptions:** The \(L₀ = 0\) regime is narrow and not representative of most convex ML problems.  
- **Minimal empirical validation:** Only one logistic regression experiment is provided, with no ablations, significance analysis, or comparison to stronger baselines.

### Questions
1. When batch size scaling is included, do your total oracle or sample complexities remain linear, or do they revert to sublinear rates?  
2. What realistic convex functions beyond logistic regression actually satisfy \(L₀ = 0\)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studied the convergence rates of ClipSGD and Normalized-SGD for $(L_0, L_1)$-smooth optimization problems, and further studied their zeroth-order variants. The results exhibit two-phase behavior with an exponential decrease followed by a sublinear convergence.

### Strengths
1. Generalized smoothness is an useful and important new concept in optimization community, this work devoted efforts into understanding the condition, which should be interesting to the optimization and ML community.
2. The two-phase behavior is interesting, and the theory proposed can match the phenomena, which improves our understanding on the algorithm design.
3. The flow of the work is easy to follow.

### Weaknesses
1. But the claimed linear convergence result only arises when $L_0=0$, even though Remark 1.3 mentioned some toy examples fulfilling the assumption, but still it is too restricted (because the condition is proposed for DL applications) in my opinion. It is not clear whether it is still hold for general $L_0$.
2. Line 235, "This iteration complexity significantly outperforms standard results in the L-smoothness setting (Assumption 1.1)", however these two complexities comes from two different scenarios ($L$-smooth and $(0, L_1)$-smooth), therefore the comparison requires additional justification.
3. Finally, in the abstract, "we demonstrate the possibility of the zero-order algorithm outperforming the first-order algorithm in the convex setup through numerical experimentation", but in Section 7, ZO-NSGD (see orange line), ZO-ClipSGD (see blue line) never outperform ClipSGD (see green line) if I understand the figure correctly. Also you said "(ZO-NSGD) outperforming the first-order ClipSGD algorithm after 55000 iterations.", but this crossover is not visible in the figure.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
