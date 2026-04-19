# Local Curvature Descent: Squeezing More Curvature out of Standard and Polyak Gradient Descent

- Decision: Reject
- Scores: 5, 3, 3, 6

## Abstract
We contribute to the growing body of knowledge on more powerful and adaptive stepsizes for convex optimization, empowered by local curvature information. We do not go the route of fully-fledged second-order methods which require the expensive computation of the Hessian. Instead, our key observation is that, for some problems (e.g., when minimizing the sum of squares of absolutely convex functions), local curvature information is readily available, and can be used to obtain surprisingly powerful matrix-valued stepsizes, and meaningful theory. In particular, we develop three new methods — LCD1, LCD2 and LCD3 — where the abbreviation stands for local curvature descent. While LCD1 generalizes gradient descent with fixed stepsize, LCD2 generalizes gradient descent with Polyak stepsize. Our methods enhance these classical gradient descent baselines with local curvature information, and our theory recovers the known rates in the special case when no curvature information is used. Our last method, LCD3, is a variable-metric version of LCD2; this feature leads to a closed-form expression for the iterates. Our empirical results are encouraging, and show that the local curvature descent improves upon gradient descent.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper explores methods for enhancing gradient descent through adaptive step sizes. The basic idea is to exploit "local curvature" without resorting to fully second-order methods, which require computationally expensive Hessians, i.e., using extra knowledge the idea is to interpolate between pure first order and second order. To this end, the authors introduce three algorithms or more precisely step-size estimation strategies: LCD1, LCD2, and LCD3. Each one optimizes step size based on curvature properties of the objective function. 

The authors establish theoretical convergence rates for LCD1 and LCD2, aligning with those for standard and Polyak step size gradient descent methods, which they intend to extend. The authors also claim improved empirical performance.

### Strengths
The paper expands the class of objective functions for which we can obtain solid convergence guarantees. Moreover, the step size strategies (LCD 2 and LCD 3 which generalize Polyak steps) can provide some form of adaptivity. The computational results indicate that there might be empirical performance improvements as well.

### Weaknesses
I have some concern regarding the novelty of the proposed method/approaches. In particular, the proposed concept of local curvature and the analysis seems to be very similar to "relative smoothness" and the associated analysis, see e.g., 

https://pubsonline.informs.org/doi/10.1287/moor.2016.0817 

and 

https://arxiv.org/pdf/1911.08510

(just to name two examples - there is much more)

Given these similarities the authors should discuss and relate their work to that of relative smoothness and also clearly delineate their own contribution from what is already known or implied by relative smoothness. I would like to ask the authors to clarify this in the rebuttal:

1. How does the local curvature assumption differ from or extend relative smoothness?

2. Do the convergence guarantees for LCD1 and LCD2 improve upon what could be derived using relative smoothness?

3. Are there problem settings where the local curvature approach provides benefits over relative smoothness?


Independent of the above, the paper feels a little incremental, technical (the paper is very long), and niche in the sense that it provides "yet another" step size strategy for a broader class of objectives. The results are nice but not very surprising.

### Questions
see above -

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes three variants of descent methods for solving convex optimization problems that satisfy Assumptions 2.1, specifically (5) and (6). Conditions (5) and (6) define a class of convex problems with a curvature matrix C(y) and a smoothness parameter Lc​. These parameters, C(y) and Lc​, are used to compute the step sizes for the three methods. The first method generalizes gradient descent with a fixed step size; the second method generalizes gradient descent with the Polyak step size; and the third method is a heuristic variant of the second method. Both the first and second methods are proved to converge at a rate of O(1/k). Experimental results demonstrate the convergence of the methods, with Methods 2 and 3 notably outperforming the Polyak step-size approach, highlighting the benefits of incorporating curvature information in algorithm design. The paper also provides example problems for which the curvature matrix C(y) and Lc​ are computable.

### Strengths
- The methods illustrate how readily available curvature information for certain problems can be used to design algorithms that improve performance.
- Sections 6 and 7 provide problem examples and proofs, which help identify the class of problems for which these algorithms are applicable.

### Weaknesses
- The setting is very limited, namely, convex optimization problems for which the scaling matrix C(y) is known and available.
- It is claimed that Assumption 2.1 defines a new class of functions, but that is not true.  These ideas are all related to notions of generalized convexity and generalized distance functions.  In this respect, the extensions of the analyses of gradient descent, etc. are relatively straightforward.

### Questions
Prior to equation (22), what is Step 3 of LCD2?  There does not appear to be a "step 3"?  This seems to follow easily from the projection theorem onto closed and convex sets.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Convexity and L-smoothness are standard assumptions in optimization literature which are useful for easier analysis of optimization algorithms and determining the right algorithm parameters. These global conditions may not always take local differences in the curvature into account. This paper proposes new analogues of the assumptions to incorporate certain kinds of local curvature information. The paper also proposes modifications of gradient descent using matrix valued step sizes to take advantage of the modified assumptions.

### Strengths
The paper is clearly written and is easy to read. It is interesting to see that Newton's method and gradient descent can be seen as special cases of the same framework under the assumption proposed by the authors.

### Weaknesses
The first-order convexity condition and L-smoothness imply that $$ f(y) + \langle \nabla f(y), x-y\rangle \leq f(x) \leq f(y) + \langle \nabla f(y), x-y\rangle  + \frac L2 || x-y||^2.$$ The authors' assumption 2.1 modifies these inequalities by adding the term $\frac12 || x-y||^2_{\mathbf C}$ to the lower bound and upper bound for $f(x)$ provided by these inequalities (since $\frac12 || x-y||^2_{\mathbf C+L\mathbf I} = \frac12 || x-y||^2_{\mathbf C} +  \frac L2 || x-y||^2$). Thus, assumption 2.1 seems to be more general than L-smoothness but it is less general than convexity.

In the standard analyses of GD, L-smoothness guarantees a sufficient decrease with each step (with the right step size) and convexity ensures that that decrease pushes us towards the minimizer. The two inequalities balance each other in a crucial way. This work exploits that tradeoff. However, it is not surprising that if the same term is added to both the lower bound and upper bound of $f(x)$ then they will cancel each other out and the standard convergence proofs will still go through. 

Furthermore, if $f$ is twice differentiable, assumption 2.1 is actually equivalent to
$$\mathbf C(x) \preceq \nabla^2 f(x) \preceq \mathbf C(x) + L_C \mathbf I.$$ This follows from the same kind of standard arguments used to show that convex functions have positive semidefinite Hessians. Having observed this second order condition, many of the remarks that the authors make follow directly. The case when $\mathbf C = 0$ is the standard case with convexity and L-smoothness, and the case when $L_C=0$ is the realm of second order methods like Newton's method. My impression is that to give any useful advantage over standard GD, the map $\mathbf C(x)$ will have to approximate the Hessian $\nabla^2 f(x)$. But then the algorithms provided do not seem to be very useful unless there is a good way to approximate the Hessian.

LCD1 actually just seems to be a version of Newton's method where the Hessian is overestimated by its upper bound $\mathbf C(x) + L_C \mathbf I$ to make it more stable. The convergence rate provided for LCD1 is the same as GD (unless $L_C = 0$, in which case it is pure Newton's method anyway), which does not provide any new insights either. The authors present LCD2 as a generalization of Polyak's step size, but the step size $\beta_k$ in that case does not even have a closed form. Computing $\beta_k$ itself requires an optimization problem to be solved at each step, and the benefits of doing that are not clear. LCD3 has a closed form step-size but there are no convergence results provided for it, so it's not clear how well it performs.

The assumption could still have been justified with examples of interesting functions that satisfy assumption 2.1 in non-trivial ways. Unfortunately, that does not seem to be the case. One of the curvature matrices specified for each of the examples 6.1-6.4 is just the Hessian. For examples 6.2 and 6.4, $\nabla f(x) \nabla f(x)^\top$ is proposed as another candidate for the curvature matrix, but these kinds of approximations of the Hessian are already covered by quasi-Newton methods like Berndt–Hall–Hall–Hausman algorithm.

The experiments are also only performed on these trivial examples and authors compare their proposed step sizes only against Polyak step size. The first experiment is on a strongly convex and L-smooth function, which is covered by the classical assumptions, and the optimal method for which would have been a momentum-based algorithm like Nesterov's accelerated gradient descent. The second experiment chooses the Hessian as the curvature matrix, reducing it to the case where second order methods would perform better.

Overall, the assumption proposed in the paper does to offer many new theoretical insights nor do the algorithms proposed offer practical advantages over existing algorithms.

### Questions
- Can the authors comment on the second order characterization of assumption 2.1? 
- What is the time complexity of LCD2 compared to Polyak step size? Specifically, how does the computation of $\beta_k$ affect the complexity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper introduces a new class of functions called convex and smooth with local structure. Many interesting examples are provided that satisfy the proposed assumption, and the calculus for the proposed class is presented. For this class, the authors proposed three new algorithms LCD1, LCD2, and LCD3. Theorem 4.1, 4.2 shows $\frac{L_C ||x_0 - x_*||^2}{2k}$ for LCD1 which is a generalization of gradient descent, and LCD2 which is a generalization of Polyak stepsizes. It is a good paper, it is well-written, easy to follow, and motivated by examples.

### Strengths
1. The proof of Theorems 4.1 and 4.2 is correct.
2.  A wide range of examples is provided alongside a general framework for constructing convex and smooth functions with local curvatures. 
3. The theoretical results are motivated by examples and supported by experiments.

### Weaknesses
1. All three methods require a knowledge of $C(x)$. In the examples, such $C(x)$ is provided; however, if one wants to run these methods, one must find such $C(x)$, which can be a non-trivial task. Meanwhile, if $C(x)$ is not known in advance, one can still use GD  with line-search or Polyak stepsizes, or normalized gradient descent. Meaning, to use LCD1, LCD2, and LCD3, the knowledge of $C(x)$ is necessary.

### Questions
1. Is there any relation between convex and smooth local curvature functions with functions defined by Bregman divergence? If yes, is there a connection between Mirror Descent and LCD1?
2. GD with Polyak stepsizes also converge for the problem (1) under the proposed assumption, as well as the normalized gradient method, however, with a slower convergence rate. It would be beneficial to add a comparison of existing methods not only in experiments but also in the main section.
3. How the choice of $C(x)$ in experiments affects the performancece if the methods?
4. In the first experiment with $L_2$ regularization, GD also can be used; it would be interesting to see if LCD1 outperforms GD.

Typos.
- should it be $\mathbb{R}^d \rightarrow \mathbb{R}^d$ ?
- there is a problem with reference on lines 550-552
- in the proof of Lemma B.1, the last two lines of the first equation are repeated
- lines 756-757 ...the term $||x_k - x_*||^2_{C(x_k)}$ ... "positive" is missing.
- lines 1026-1027 remove "any".

### Soundness
3

### Presentation
4

### Contribution
3
