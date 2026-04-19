# Provable Faster Zeroth-order Method for Bilevel Optimization with Optimal Dependency on Error and Dimension

- Decision: Reject
- Scores: 6, 6, 3, 5

## Abstract
In this paper, we study and analyze zeroth-order stochastic approximation algorithms for solving black-box bilevel optimization problems, where only the upper and lower function values can be obtained. \citep{Saeed2024} proposed the first full zeroth-order bilevel method that utilizes Gaussian smoothing to estimate the first- and second-order partial derivatives of functions with two independent blocks of variables. However, this method suffers from a high dimensional dependency of $\mathcal{O}((d_{1}+d_{2})^{4})$, where $d_{1}$ and $d_{2}$ are the dimensions of the outer and inner problems, respectively. They left an open question: can this dimension dependency be improved? To answer this question, we propose a single-loop accelerated zeroth-order bilevel algorithm, which achieves a dimension dependency of $\mathcal{O}(d_{1}+d_{2})$ by incorporating coordinate-wise smoothing gradient estimators (coord). 
	We develop a new theoretical analysis for the proposed algorithm, which converges to a stationary point of $\Phi(x)$ with a complexity of $\mathcal{O}((d_{1}+d_{2})\epsilon^{-3})$ in expectation settings and $\mathcal{O}((d_{1}+d_{2})\sqrt{n}\epsilon^{-2})$ in finite sum settings. These complexities are both best-known with respect to dimension and error $\epsilon$. We also provide  experiment to validate the effectiveness of the proposed algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper considers bilevel optimization. In particular, the authors aim at solving the stochastic bilevel optimization via zeroth-order methods. Motivated by the fact that a recent work has high dimensional dependency in the complexity bound, the authors propose VRZSBO, and prove that it requires less oracle complexity than the existing work, in the context of zeroth-order bilevel optimization algorithms. Numerical simulations are provided to validate the performance of their algorithm under different settings and their better convergence as compared to existing algorithms.

### Strengths
1. The authors provide a better upper bound for the oracle complexity of zeroth-order bilevel optimization, without the need of additional assumptions.

2. The technical claims are in general sound.

### Weaknesses
1. Incomplete related work. Some important bilevel optimization algorithms are missing, even if they are not based on zeroth-order methods. It would be good if the authors could include a more comprehensive table containing existing algorithms with comparisons of assumptions, complexities, and types of oracles needed. See, for example, the related works in table 1 in [1].

2. Presentation. 
    1. the authors claim one contribution is that they do not have additional assumptions like 3rd Lipschitz. However, this is not well reflected in Table 1. 
    2. It seems that Section 1.1 can be moved to other sections for better presentation — for those who are familiar with bilevel optimization this section seems unnecessary and for those who are unfamiliar with bilevel optimization this section is full of technical details and hard to follow.
    3. It would be better if the authors could provide a well-organized Section 3 — the details of analysis described in 3.5 - 3.7 can be moved to the appendix to make the main context more readable. For example, proof sketch in section 3.6 is fairly standard and unnecessary to be included in the main body of the paper.

3. There are many important baselines missing in the experiments. 

4. Overfull box in lines: 257 - 262, 409, 453, etc.

Reference

[1] BILEVEL OPTIMIZATION UNDER UNBOUNDED SMOOTHNESS: A NEW ALGORITHM AND CONVERGENCE ANALYSIS

### Questions
1. Can the authors explain the motivation of studying zeroth-order bilevel algorithms? The experiments can not reflect if VRZSBO can still be a better choice in large-scale settings.

2. Can the authors explain why (d_1 + d_2) dependency is optimal? For example, is there a reference for the lower bound of the dimension dependency.

3. The oracle cost mentioned in Remark 2 is in expectation (see (14) - (16)). However, the cost in (Aghasi and Ghadimi, 2024) is not in expectation, as the estimator therein does not require an additional sampling step (i.e., (14) - (16)). I am wondering if the authors could provide more context on this unfair comparison.

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
3

### Summary
Using finite-difference approximations and the PAGE method for nonconvex optimization, this work introduces a novel zeroth-order algorithm, VRZSBO, for solving bilevel optimization problems with a strongly convex lower-level problem. The authors show that VRZSBO converges to an $\epsilon$-stationary point with a function query complexity of $\mathcal{O}((d_1 + d_2)\sqrt{n} \epsilon^{-2})$ in the finite-sum setting and $\mathcal{O}((d_1 + d_2) \epsilon^{-3})$ in the expectation setting, where $d_1$ and $d_2$ denote the dimensions of the upper- and lower-level variables, respectively. Experimental results on hyper-representation tasks using linear and two-layer network embedding models further validate the algorithm’s effectiveness.

### Strengths
$\textbf{S1:}$ The paper is easy to follow. Zeroth-order algorithms for bilevel optimization remain underexplored, even in the nonconvex-strongly-convex setting, making the introduction of the novel VRZSBO algorithm a valuable algorithmic contribution to this area. 

$\textbf{S2:}$ The proposed VRZSBO algorithm achieves a significant  improvement over the best-known complexity of ZDSBA (Aghasi and Ghadimi, 2024), reducing the query complexity from $\mathcal{O}((d_1 + d_2)^4 \epsilon^{-3})$ to $\mathcal{O}((d_1 + d_2) \epsilon^{-3})$ in the expectation setting. Additionally, this work provides a complexity analysis for the finite-sum setting.

### Weaknesses
$\textbf{W1:}$ It is well-known that the Hessian-vector product can be implemented through finite-difference approximations of the gradient along a given direction. Given this, the novelty of the proposed results seems limited compared to existing methods. 

Moreover, as noted in Chapter 9 of Numerical Optimization by Jorge Nocedal and Stephen J. Wright, finite-difference approaches are often impractical due to potential inaccuracies in function evaluations (e.g., rounding errors), which prevent the distance of finite differences from being arbitrarily small.

$\textbf{W2:}$ The experiments could be more comprehensive. Incorporating updated first-order algorithms, such as F2SA (Kwon et al., ICML 2023), would provide a more thorough comparison. Furthermore, including commonly used bilevel optimization tasks, such as hyperparameter selection and data hyper-cleaning, would further validate VRZSBO’s effectiveness, showcasing its reliability across various bilevel optimization problems. 

I would be willing to increase the score based on the rebuttal.

### Questions
Apart from the questions raised in the Weaknesses section, some additional questions are as follows:

$\textbf{Q1:}$ Why does the proposed algorithm achieve an optimal result with respect to dependence on dimension? A bit more discussion is needed.

$\textbf{Q2:}$ In Assumption 2, why is there no stochastic assumption on $\nabla_{xy}^2 g$? 

$\textbf{Q3:}$ Based on the proofs of Theorems 1 and 2, the iteration count $T$ in these theorems depends on $\psi_0$, which may in turn depend on $n$ (in the finite-sum case) and $\epsilon$ (in the expectation case). Therefore, to derive the complexity analysis results (i.e., the total oracle cost), does this require certain conditions on the initial points? Additionally, does it require an assumption of a lower bound for $\Phi(x)$? A bit more discussion on these points would be helpful.

$\textbf{Q4:}$ In Theorems 1 and 2, the zeroth-order tuning parameter $h$ does not appear—why is this the case? Is there a relationship between $v$ and $h$, potentially stemming from Corollary 1 and Lemma 3 through an intermediate parameter $\delta$? Additionally, in Figure 1, ablation studies examine the effects of different choices of $h$ and $v$ on the zeroth-order estimator. Were these experiments conducted with consideration of the relationship between $v$ and $h$?

$\textbf{Q5:}$ The theoretical results in this work suggest that smaller zeroth-order tuning parameters $v$ and $h$ yield better performance, yet the experimental results in Figure 1 do not support this conclusion. Could the authors explain this discrepancy?
 

$\textbf{Q6:}$ In the experiments, why was there no comparison with ZDSBA from Aghasi and Ghadimi (2024)?

$\textbf{Suggestions for improvement that did not affect the score:}$

In the convergence analysis, for example, in Theorem 1, it is stated that $\psi_{t+1}-\psi_{t} \leq -\frac{\eta_x}{2} ( | \nabla \Phi(x_t) |^2 + \epsilon^2)$. This implies that the Lyapunov function $\psi_t$ is decreasing. However, how is this possible when $\Phi(x)$ is nonconvex? It seems that the sign before $\epsilon^2$ may be incorrect.

In Theorem 1, why is there no $m$ in the finite-sum case? Is there an assumption that $m=\mathcal{O}(n)$?

Page 1, Line 045: In $g(x,y)$, $n$ should be $m$. 

Page 2, Lines 102-103: Verify the form of $\bar{\nabla}\Phi(x)$.  

Page 3, Line 145: Replace “$y(x)$, $y_\lambda(x)$, $\nabla \mathcal{L}\lambda(x)$” with “$y^*(x)$, $y\lambda^(x)$, $\nabla \mathcal{L}_\lambda^(x)$”. 

Page 5: Line 228: The referenced equation (2) may not be appropriate. 

Page 5: Lines 230-231: $R(x,y,z)$ does not include an inverse matrix operator.

Page 5, Lemma 1: It would be clearer to replace $y^$ with $y^(x_t)$ since $y^*$ is a function.

Page 5, Definition 1: In $\hat{\nabla}_x g(x,y)$, replace $d_2$ with $d_1$, and in $\hat{\nabla}_y f(x,y)$, replace $d_1$ with $d_2$.  

Page 6, Lemma 3: Clarify the meaning of $r_z$ in the definition of $h$. 

Page 7, Line 327: “Section 3.1” should be changed to “Sections 3.2–3.3”.  

Page 7, Equations (14), (16): Keep the notations consistent with those in (10)–(11).  

Page 7, Line 369: Should $-p$ replace $-\frac{1}{p}$ according to Lemma 9? 

Page 7, Line 372:  Clarify what $\eta$ represents in the definition of $\Psi_t$. 

Page 8, Lines 383-387: Should $-p$ replace $-\frac{1}{p}$ based on Lemmas 6–7? 

Page 8, Line 398: Replace $\eta$ with $\eta_x$.  

Page 8, Lemmas 4-5: Define $\ell_{Z^}$ and $\ell_{Y^}$.   

Page 10, Figure 1: Correct typos, particularly in the caption.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, the authors propose a zeroth-order method for solving stochastic bilevel optimization problems, where the lower-level problem is strongly convex and only function values are accessible for both levels. Their method can be regarded as a zeroth-order approximation of the SPABA method proposed by (Chu et al., 2024), where the gradients and Hessian-vector products are replaced by their zeroth-order estimators. Specifically, the authors employ a coordinate-wise zeroth-order estimator, approximating each coordinate of the gradient by finite differences. They analyze their proposed algorithm under both general stochastic and finite-sum settings, demonstrating that the function value query complexity scales linearly with the problem’s dimension and has optimal dependence on the target accuracy.

### Strengths
In the context of zeroth-order algorithms for bilevel optimization, the proposed algorithm improves dimensional dependence compared to prior work and introduces a simpler single-loop update format.

### Weaknesses
- My major concern is the use of the coordinate-wise zeroth-order estimator in Definition 1. As noted by the authors, this estimator requires $\mathcal{O}(d_1+d_2)$ function value queries in total, which may be as computationally expensive as obtaining the exact gradient by backpropagation. While it could be argued that only function values are accessible in certain applications, this does not hold for most common bilevel optimization applicationss, including the hyper-representation problem used in the experiments. 
- In addition, the novelty of the paper appears somewhat limited. If my understanding is correct, the algorithmic framework closely follows the SPABA method by (Chu et al., 2024) such as the use of the PAGE variance reduction technique, and the main modidfication is to substitute the exact gradients and Hessian-vector products by their zeroth-order estimators. However, since the coordinate-wise zeroth-order estimator is employed, the gradient estimator error can theoretically be made arbitrarily small by selecting a sufficiently small $v$ (see Lemma 2). Indeed, the authors choose $v = \mathcal{O}(\epsilon^2)$ in Theorems 1 and 2, which means that the gradient error is also on the order of $\mathcal{O}(\epsilon^2)$. Consequently, the convergence analysis may not significantly differ from that of the original SPABA method.

----
Tianshu Chu, Dachuan Xu, Wei Yao, and Jin Zhang. SPABA: A single-loop and probabilistic stochastic bilevel algorithm achieving optimal sample complexity. ICML 2024

### Questions
- The author claims that the dependence on the dimension is optimal. While this appears reasonable, a rigorous lower bound is necessary to support this claim. 
- The experiments should include the SPABA method as a baseline, as it is the most relevant algorithm for comparison with the proposed method. Additionally, comparisons across different algorithms should be based on runtime, given the potentially significant differences in per-iteration costs.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores zeroth-order algorithms for addressing stochastic bilevel optimization problems. Expanding on earlier work that introduced a zeroth-order bilevel method using Gaussian smoothing, the authors enhance the dimensional dependency in sample complexity from $(d_1 + d_2)^4$ to $(d_1 + d_2)$. They also utilize variance reduction techniques to optimize the epsilon dependency in sample complexity from $\epsilon^{-4}$ to $\epsilon^{-3}$, achieving optimal bounds. The new algorithm undergoes theoretical analysis, demonstrating convergence to a stationary point with optimal complexity in both expectation and finite sum setting. Additionally, experiments are conducted to validate the algorithm's effectiveness.

### Strengths
The paper presents several key strengths that enhance its contribution to the field of stochastic bilevel optimization. Notably, it significantly improves the dimensional dependency in sample complexity from $(d_1 + d_2)^4$ to $(d_1 + d_2)$ and employs variance reduction techniques to optimize epsilon dependency, achieving optimal bounds. The thorough literature review effectively situates the work within existing research, underscoring its relevance. Additionally, the clear and structured presentation of the methodology supports reproducibility, while the empirical validation through experiments demonstrates the algorithm's practical effectiveness.

### Weaknesses
Lack of Novelty: The contributions of this paper build upon established concepts, integrating ideas from quadratic auxiliary functions, PAGE-type variance reduction, and zeroth-order gradient estimation. While these improvements are valuable, they may not be unexpected, in my humble opinion. For instance, ZDSBA built on previous work as a double-looped algorithm utilizing Hessian inverse approximation, so it is understandable that the incorporation of auxiliary quadratic functions would enhance dimension dependency. Additionally, the introduction of various variance reduction techniques is likely to improve epsilon dependency. Although I recognize the meaningful nature of these contributions, I feel that the paper does not fully meet the acceptance criteria for ICLR. Therefore, I will reserve my opinion on acceptance until after the rebuttal stage.

### Questions
Gaussian Smoothing vs. Coordinate-wise Smoothing: Based on my previous experiences, I have not observed a significant difference in the theoretical bounds between these two smoothing techniques. Consequently, I believe the improvements presented in the paper may not be attributed to coordinate-wise smoothing. The paper does not address the rationale behind choosing coordinate-wise smoothing over Gaussian smoothing. Could the authors please clarify the reasoning for this choice?

### Soundness
3

### Presentation
2

### Contribution
2
