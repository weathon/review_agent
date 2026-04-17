# Exponential Objective Decrease in Convex Setup is Possible! Gradient Descent Method Variants under $(L_0,L_1)$-Smoothness

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The gradient descent (GD) method -- is a fundamental and likely the most popular optimization algorithm in machine learning (ML), with a history traced back to a paper in 1847 (Cauchy, 1847). It was studied under various assumptions, including so-called $(L_0,L_1)$-smoothness, which received noticeable attention in the ML community recently. In this paper, we provide a refined convergence analysis of gradient descent and its variants, assuming generalized smoothness. In particular, we show that $(L_0,L_1)$-GD has the following behavior in the _convex setup_: as long as $||\nabla f(x^k)|| \geq \frac{L_0}{L_1}$ the algorithm shows _exponential objective decrease_, and when $||\nabla f(x^k)|| < \frac{L_0}{L_1}$ is satisfied, $(L_0,L_1)$-GD has standard sublinear rate. Moreover, we also show that this behavior is common for its variants with different types of oracle: _Normalized Gradient Descent_ as well as _Clipped Gradient Descent_ (the case when the full gradient $\nabla f(x)$ is available); _Random Coordinate Descent_ (when the gradient component $\nabla_{i} f(x)$ is available); _Random Coordinate Descent with Order Oracle_ (when only $\text{sign} [f(y) - f(x)]$ is available). In addition, we also extend our analysis of $(L_0,L_1)$-GD to the strongly convex case. We explicitly confirm our theoretical results through numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies gradient-based methods (GD, NGD, Clip-GD, RCD, and RCD with order oracle) for convex objectives under \((L_0,L_1)\)-smoothness. It establishes a clean two-phase behavior: when $\|\nabla f(x_k)\|\ge L_0/L_1$, the objective decreases exponentially; once below this threshold, the rate becomes sublinear. The story is extended from full-gradient methods to normalized/clipped variants and coordinate-oracle settings, with remarks for the strongly-convex case and small illustrative experiments.

### Strengths
- Clear regime characterization. With a stepsize like $\eta_k=(L_0+L_1\|\nabla f(x_k)\|)^{-1}$, the paper gives an interpretable ``fast-then-slow'' picture: exponential decrease above the $L_0/L_1$ threshold and standard $O(L_0R^2/N)$ behavior below it.

- Coverage across oracles. The analysis is propagated from full-gradient methods to NGD/Clip-GD and to RCD/OrderRCD, providing a unified treatment under $(L_0,L_1)$-smoothness.

- Tidy write-up. Assumptions and theorem statements are organized; examples help illustrate the phase transition.

### Weaknesses
- Topic saturation / mature techniques. Both the $L_0,L_1$-smoothness line and clipping/normalization ideas have been heavily studied; the contribution reads as a careful consolidation rather than a new mechanism.

- Incremental advance. The work largely systematizes known ingredients with refined constants and extensions across oracles; conceptual novelty is limited.

- Dominant sublinear regime near optimum. Even with an initial exponential phase, the practically dominant regime remains sublinear unless \(L_0\) is very small, tempering impact on broad convex problems.

- Dependence on thresholds/unknown constants. The results hinge on the $\|\nabla f(x_k)\|\ge L_0/L_1$ threshold and schedules that presume knowledge (or good estimates) of $L_0,L_1$; robustness when these are unknown is not fully addressed.

### Questions
I think the claimed geometric (linear) decrease in the $L_1$-dominated regime is not novel—it follows routinely from the standard $(L_0,L_1)$ descent calculus with gradient-normalized steps; if this assessment is mistaken, I welcome a correction.




In the $L_1$-dominated regime—either $L_0=0$ or whenever $\|\nabla f(x_k)\|\ge L_0/L_1$—the standard $(L_0,L_1)$ descent with the normalized stepsize $\eta_k=(L_0+L_1\|\nabla f(x_k)\|)^{-1}$ gives
$f(x_{k+1})-f^\star \le (1-\rho)[f(x_k)-f^\star]$ for some $\rho\in(0,1)$,
since $\frac{\|\nabla f\|^2}{L_0+L_1\|\nabla f\|}\ge c\,\|\nabla f\|$ once $\|\nabla f\|\ge L_0/L_1$.
Thus “geometric descent” above the threshold is a routine, well-known consequence of the $(L_0,L_1)$ calculus with gradient-normalized steps, and is not conceptually new; the only delicate part is the $L_0$-dominated small-gradient phase, where one typically recovers at best sublinear progress.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies gradient descent (GD) and its variants (Normalized GD, Clipped GD, and Random Coordinate Descent) under the recently popularized $(L\_0, L\_1)$-smoothness assumption. The authors show that under this generalized smoothness, GD and its variants exhibit exponential objective decrease as long as the gradient norm is above a threshold ($\\|\nabla f(x)\\| \ge \frac{L\_0\}{L\_1}$), and revert to sublinear convergence afterwards.

### Strengths
- The result that GD can exhibit exponential objective decay in the convex setup under $(L\_0, L\_1)$-smoothness, particularly when $L\_0 = 0$, is conceptually interesting and highlights a nuanced behavior of gradient descent.

- The proofs and arguments are concise and technically sound.

- The exposition is organized and clear in most places, making it easy for the readers to follow the storyline of the paper.

### Weaknesses
1. *Significance of the results.* While the exponential objective value decay is mathematically neat, it only holds when the gradient norm is large. Once the iterates enter the small-gradient regime, the convergence becomes sublinear, which dominates asymptotically. In the end, the overall improvement is not substantial.
The extensions to Clipped GD and Random Coordinate Descent are provided, but they do not seem to constitute conceptually new insights beyond their analysis of full gradient descent.

2. *Lacking interpretation of the $L\_0 = 0$ case.*
The case of global linear convergence when $L\_0 = 0$ is emphasized as a key finding that distinguishes the result of this paper with similar existing works. But in that regime, the assumption $\\|\nabla f(x) - \nabla f(y)\\| \le L_1 \\|\nabla f(x)\\| \\|x - y\\|$ actually imposes a stronger constraint than standard Lipschitz smoothness near optima. Thus, the improvement stems from stronger assumptions rather than a fundamentally sharper analysis. I am not sure if such a setting is practically relevant.

### Questions
1. The standard $L$-smooth convex setting also satisfies $(L\_0, L\_1)$-smoothness if one sets $L\_0 = L$ and $L\_1 > 0$ arbitrarily. In that case, your results seem to suggest exponential decay in $f(x^k) - f^*$ with step size smaller than $1/L$, even though GD with step size less than $1/L$ is known to be sublinear (with a complexity lower bound). Could you clarify where this apparent contradiction is resolved? Can it be explained based on the structure of the worst case function for GD, or initialization?

2. I believe that a more proper way to state Theorem 3.1 is to first state a short preliminary lemma about monotonicity of $\\|\nabla f(x^k)\\|$ and then defining the transition point $T$ before or earlier within the theorem, rather than relegating them to the prose explanation following it. At the moment, it is not accurately stated. For example, the condition $\\|\nabla f(x^{N-1})\\| < \frac{L\_0}{L\_1}$ for the sublinear convergence part does not seem to reflect what the theorem actually tries to state. (Perhaps it was a typo for $\\|\nabla f(x^{0})\\| < \frac{L\_0}{L\_1}$.)

3. In Lines 79–86, $\lambda\_k$ is introduced without definition or context, making the introduction section non-self-contained for first time readers.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies gradient-based optimization methods (GD, NGD, Clip-GD, RCD, and OrderRCD) under the (L₀, L₁)-smoothness assumption, a generalization of Lipschitz smoothness where the smoothness constant depends linearly on the gradient norm. The key result is that exponential (linear) convergence rates can occur in convex problems when the gradient norm satisfies $\|f(x_k)\|\geq L_0/L_1$ even without assuming strong convexity.
The paper systematically analyzes GD, normalized GD, clipped GD, random coordinate descent, OrderRCD, and extend their results to the strongly convex case. For GD they show that their method achieves an exponential decrease when above the threshold. For normalized GD they show a similar rate. For random coordinate descent and OrderRCD they give the first analysis for these families of functions. They support their theoretical results with experiments.

### Strengths
1. The discovery and proof that exponential convergence can arise in convex settings under generalized smoothness is both non-trivial and potentially impactful.
2. Provides first convergence analyses under this generalized smoothness assumption, for RCD and OrderRCD
3. Shows how to get exponential rates of convergence for non strongly convex functions.

### Weaknesses
1. While rigorous, the results could use more intuitive explanations of why (L₀, L₁)-smoothness leads to exponential decrease.
2. If the gradient norm is large, isn’t gradient descent always going to make a large progress? I am not sure why this is new.
3. When was this kind of smoothness introduced? Based on the related works it seems to be quite recent.
4. Are these methods useful in practice over SGD in large data sets?
5. How would one estimate L_0 and L_1 in practice as they are needed in the step sizes?

### Questions
See weaknesses!

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes (unconstrained) convex optimization under the recently popularized $(L_0,L_1)$-smoothness assumption. For convex functions, the paper demonstrates a two-phase behavior for simple first-order methods. In particular, when the gradient norm is above $L_0/L_1$, gradient descent can achieve exponential (linear) decrease of the objective even for the (non-strongly) convex setup. Below that threshold, the rate reverts to the usual sublinear $O(1/N)$ regime. The same phenomenon is established for other first-order methods of normalized and clipped gradient descent, refining previous analysis on $(L_0,L_1)$-smooth functions. The paper additionally shows similar results for random coordinate descent and a sign-oracle variant (OrderRCD) which is new for $(L_0,L_1)$-smooth functions, includes results for strongly convex functions (which seems like a three-phase linear convergence rate) and a simple numerical experiment.

### Strengths
- The work extensively covers various types of first-order algorithms including classical methods, refined methods designed for or closely related to $(L_0,L_1)$-smooth functions (clipGD, etc.) and coordinate descent methods.
- Comparison between the new results and previous work seem to be clearly presented.

### Weaknesses
See **Questions.**

### Questions
- As the main result of this paper is the discovery of the two-phase analysis of first-order methods, can the authors explain a bit more on what would be the takeaway or benefits of having linear convergence in a region far from the optimum?
    - It’s of course nice to know more details about the convergence dynamics and I am aware that the paper’s focus is on the theory side, but I can’t see how this can be an “extremely” interesting result yet (compared to previous $(L_0, L_1)$-smooth convergence results). I also guess that we won’t be able to see similar phenomena for more realistic nonconvex functions (or if so, please tell me).
    - The two-phase analysis might turn out to be something interesting if this theory either better explains the actual dynamics instantiated by first-order algorithms for general $(L_0, L_1)$-smooth functions other than the simple example in Section 7, or motivates ways to accelerate in the linear convergence regime to enter the sublinear neighborhood faster, both of which I think are possible to check. Can the authors comment on these aspects?
- For coordinate descent, is it correct that we cannot expect to extend these to general stochastic gradient methods? While extending to stochastic first-order methods could be an interesting future direction, but the proof technique seems to be specific to coordinate descent methods. (I don’t really consider this as a weakness, I am just wondering if there is a possibility to manipulate terms so that we get a similar descent inequality with additional noise terms for stochastic oracles, but it seems hard and I wanted to check if this guess is right.)
- Minor: What is $N_3$ in Theorem 5.1? I might have missed something, but I can’t see how this is used in the theorem statement.

### Soundness
3

### Presentation
3

### Contribution
3
