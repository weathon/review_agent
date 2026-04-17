# Loss Transformation Invariance of the Damped Newton Methods

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
The Newton method is one of the most widely used second-order optimization techniques, valued for its conceptual simplicity and extremely fast local convergence. A key advantage is its invariance under affine transformations (e.g., choice of coordinate basis), which greatly facilitates implementation. However, the classical Newton method fails to converge when initialized far from the solution, motivating the development of various globalization techniques.
In this work, we focus on step size damping, which, when appropriately scheduled, ensures fast global convergence while preserving both affine-invariance and superlinear local rates. Although highly effective in convex settings, existing algorithms offer limited guarantees for problems that are only nearly convex. To address this, we investigate loss transformations that convexify the objective. We show that Newton step size schedules are invariant under such transformations and that stepsize scheduling implicitly searches over the space of objective transformations. Our theoretical findings are further supported by comprehensive experimental validation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates loss transformation invariance in the damped Newton method, showing that the Newton stepsize schedules are invariant under monotonic loss transformations. 
The authors prove that, by carefully designing such transformations, applying the damped Newton method to the original objective function $f(x)$ is equivalent to applying the vanilla Newton method to a transformed objective function $L(x)$, extending its applicability to certain nonconvex problems. 
The paper also offers insights for why stepsizes greater than one and negative stepsizes can still be effective in second-order methods.

### Strengths
1. The main theory results are innovative, which provide explanations for several empirical observations such as why stepsizes larger than one and negative stepsizes can be effective for second-order methods.

2. The motivations for using loss transformation and stepsize scheduling are clear, and the writing flows smoothly, from the discussion on convex and pseudoconvex problems to the convexification of a broad class of nonconvex problems.

### Weaknesses
1. Such loss transformation assumes the objective function is pseudoconvex (sufficient) or its sublevel sets are convex (necessary), which are quite strong for real-world problems and difficult to verify.

2. All experiments are conducted on 1D and 2D synthetic problems which are insufficient. Larger and higher-dimensional problems should be included. Moreover, since Theorems 3 and 4 are stated for $x \in \mathbb{R}^d$, it is necessary to include higher-dimensional experiments to validate these claims.

3. Only the vanilla Newton method and the damped Newton method are compared. Other second-order baselines, such as quasi-Newton, cubic regularization, trust region, and Newton-Krylov methods, should also be evaluated on large, real-world nonconvex problems. It would be helpful to show whether the damped Newton method achieves better robustness, or whether the stepsize scheduling achieves better efficiency (in time) than traditional strategies like line search.

4. Some minor typos:
    - lines 41 - 42: should be $O(1/k^2)$ instead of $O(1/k^-2)$, and similar elsewhere.
    - line 62: "than" -> "then"
    - line 274: clarify the phrasing "strict pseudoconvexity is sufficient properties are sufficient"
    - line 278: should be $\phi \circ f$ instead of $f \circ \phi$

### Questions
1. Theorem 3 assumes a global upper bound. How can such a bound be defined, or what are the consequences if it is not tight?

### Soundness
1

### Presentation
4

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
This paper extends the convergence analysis of Newton’s method to pseudoconvex functions (possibly nonconvex) by composing the function $f$ with a ‘convexifier’ $\phi$ so that $\phi \circ f$ is convex. Applying Newton’s method for the composed, convex function corresponds to Newton’s method for the original nonconvex function with an adjusted step size, i.e., we can leverage the theoretical guarantees for convex functions to the nonconvex case, and such a convexifier always exists for strictly pseudoconvex functions.

### Strengths
- The writing is clear and easy to read overall.
- The paper includes numerical experiments that support the theoretical results.

### Weaknesses
See **Questions.**

### Questions
- To run the proposed algorithm in practical situations, the only constructive way suggested here seems to be finding the function $h$ and then using some $\phi$ that satisfies the inequality in Theorem 3. Can you explain a bit more detail on how this could be done given an arbitrary pseudoconvex functions? I would like to know how much computation will the process of computing $r(x)$ and then $\phi$ via Theorem 2 and 3 require, and whether it is negligible enough to consider doing all this to find the right step size schedules to run Newton.
- Is there a possibility that we might have slow convergence guarantees (compared to real performance) in cases when the inequality in Theorem 3 is loose? (I understand that the focus of this paper is that the convergence guarantees for pseudoconvex functions are *new*, but I am wondering if there could be cases where a better choice of $\phi$ can better capture the real dynamics or lead to a better step-size scheduling.)
- For the star-convex case, is it also possible to convert cubic Newton for the star-convexified  loss into a step-size scheduled version of cubic Newton for the original function? Or are there technical difficulties in doing the same transformation as classical Newton here?

**Typos?**

Page 4. $f \circ \phi \rightarrow \phi \circ f$

Theorem 4. I think $g$ is supposed to be star-convex (instead of convex).

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
2

### Summary
This paper studies convergent stepsizes for the Newton method through loss transformations. First, it shows that running Newton on a transformed loss is equivalent to running Newton on the original loss with a rescaled stepsize, and for convexification and star-convexification, provides necessary and sufficient conditions for such transformations to exist. In doing so, this work extends the convergence of Newton’s method to certain nonconvex functions through rescaled stepsize scheduling. Experiments visualize regions where the scaling factor becomes negative and demonstrate expanded convergence neighborhoods on test functions.

### Strengths
This paper offers an interesting approach to studying the Newton method’s stepsize via loss transformations, leading to nontrivial theoretical results not covered by classical analyses. It theoretically argued for a rescaled stepsize and the existence of suitable transformations for convexification, and the experiments are appropriately conducted to study the theoretical claims and practical effect.

### Weaknesses
Since the theoretical results rely on pseudoconvexity and on a compact set, truly nonconvex is not covered. Also, the rescaled-stepsize theorems require knowledge minimum and the minimizer, which may be hard to compute in practice.

### Questions
There is a typo on line 134.

Regarding the rescaled step size in Theorems, the current formulation requires knowledge of the minimizer and the minimum value. Could this be improved using adaptive step-size strategy?

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
4

### Summary
This paper studies the damped Newton's method with the goal of extending its global convergence to a class of mildly non-convex objectives via loss function transformations. Leveraging the observation that function range transformations provide a one-to-one correspondence between a Newton step on the original objective and one on the transformed objective with a modified stepsize, the authors identify a class of functions and corresponding transformations where global convergence is ensured on the new objective. Experiments are provided in support of the theory.

### Strengths
The topic is relevant to the research community, and the results are meaningful for computational science and engineering domains. The loss function transformation idea, while not novel in itself, is nicely leveraged for this class of problems/methods.

### Weaknesses
Unfortunately, the paper has quite a few shortcomings. Broadly speaking, there's missing related literature, the presentation lacks conciseness and order, the technical part contains mistakes, and experiments are not well explained. In particular:

1. **Related work**
 	* This work falls within the broader class of approaches leveraging "hidden convexity" to establish convergence guarantees. There is a broad literature here dating from the 80's and 90's and a placement within and comparison with those approaches are missing (e.g., why/why not domain transformations, rather than range are suitable for your scenario, which other settings are served by range transformations, etc.). See [2] for an old survey, and references within [3] and [4] for more recent works.  
 	* In addition, work [1] leverages both domain and range transformations to transfer convergence guarantees of Newton's method from one problem to the other. Please also see references therein.


2. **Presentation**
	* The writing is carried out negligently, with 
         * many missing, inappropriate, extra or mismatched articles (e.g., lines 029, 038, 040, 042, and many others); 
         * typos, mathematical or otherwise (e.g., convergence rates in lines 041, 043 should be O(k^-2) and O(k^-3); line 107, dual norm argument is $g$, not $h$; the optimum is denoted both $x^{\*}$ and $x_{\*}$ e.g., in lines 316 and 320 and others; in line 053 " not necessarily" and many others); 
         * verbosity and gauche phrasing (examples throughout the paper, e.g. 087 --- "Throughout the transformation invariance [...]"; subtitle 2.1); 
         * missing structural components like a conclusion-type section. 
       
      A vocabulary and grammar correction tool (e.g., Grammarly) could greatly improve the presentation. 
	* Lemma 2 is stated without a concise, mathematical description of the correspondence between the Levenberg–Marquardt hessian regularization and the stepsized Newton method. As such, this result's relevance to the topic seems tangential. What is the implication of the result? Why is it important within the present work? 
	* Technical assumptions are spread throughout the text, rather than being collected in statements marked explicitly as "Assumption". E.g., a central assumption about $\phi$ being strictly increasing is stated in a **footnote** on page 3. 
	* Theorem 3 and its corollary are proven as "Lemma 5" in the Appendix, which is confusing. Also, the statement proven in the appendix is an incomplete version of the statement of Theorem 3 (though the result of Theorem 3 can come out of it).


3. **Technical details**
	* The current proof of Theorem 1 is incorrect. The authors use the pseudoinverse as if it were the inverse, which leads to mistakes, since the former has different properties. Specifically, the errors are 
         * Going from line 686 to 688, it seems the authors use $\mathbf{H}^\dagger \mathbf{H} = \mathbf{I}$, which does not necessarily hold for pseudoinverses. Moreover, the authors seem to also use $(\mathbf{A}\mathbf{B})^{\dagger} = \mathbf{B}^\dagger\mathbf{A}^\dagger$ which, again does not hold in general. 
         * For the same reason as above, line 691 does not imply line 694
	* computation of $g$ in theorem 4 depends on the unique optimum $x^*$, and there is no discussion about this limitation.

4. **Experiments**
	* Plots in sections 4.1 and 4.2 are not explained, and one has to guess what is illustrated: what do the colors code for? Does the method converge on the modified loss? The chosen test functions are not pseudoconvex (e.g., the Goldstein-Price function has several local minima), so the theory does not apply to this experiment. To my understanding, the question was whether convergence is aided by negative stepsizes, so experiments should seek settings with convergence guarantees.   
	* What is the transformation $\phi$ used for the experiments in Fig. 1? Only $f$ and $L$ are stated.


[1] Izmailov, Alexey F., and Mikhail V. Solodov. "TRANSFORMATIONS OF VARIABLES AND TRANSFORMATIONS OF EQUATIONS VIA THE PERTURBED NEWTON METHOD FRAMEWORK." (2025).

[2] Horst, Reiner. "On the convexification of nonlinear programming problems: An applications-oriented survey." European Journal of Operational Research 15.3 (1984): 382-392.

[3] Fatkhullin, Ilyas, Niao He, and Yifan Hu. "Stochastic optimization under hidden convexity." arXiv preprint arXiv:2401.00108 (2023).

[4] Xia, Yong. "A survey of hidden convex optimization." Journal of the Operations Research Society of China 8.1 (2020): 1-28.

### Questions
* Could you please provide examples of real-world applications where pseudoconvex functions emerge?
* Could you please discuss the possible division by zero within the modified stepsize (this is in relation to the factor $\left[1 + \frac{\phi^{\prime \prime}(f(x))}{\phi^{\prime}(f(x))} (\\|\nabla f(x)\\|_x^{*})^2 \right]^{-1}$ when $\phi^{\prime \prime}(f(x))$ is negative)?
* Could you please provide a comparison with the missing related literature mentioned above?
* Could you please motivate your choice of test functions in the experiments of section 4? Also, please explain the color coding.

### Soundness
2

### Presentation
1

### Contribution
2
