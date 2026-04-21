# Sketching for Convex and Nonconvex Regularized Least Squares with Sharp Guarantees

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Randomized algorithms play a crucial role in efficiently solving large-scale optimization problems. In this paper, we introduce Sketching for Regularized Optimization (SRO), a fast sketching algorithm designed for least squares problems with convex or nonconvex regularization. SRO operates by first creating a sketch of the original data matrix and then solving the sketched problem. We establish minimax optimal rates for sparse signal estimation by addressing the sketched sparse convex and nonconvex learning problems. Furthermore, we propose a novel Iterative SRO algorithm, which reduces the approximation error geometrically for sketched convex regularized problems. To the best of our knowledge, this work is among the first to provide a unified theoretical framework demonstrating minimax rates for convex and nonconvex sparse learning problems via sketching. Experimental results validate the efficiency and effectiveness of both the SRO and Iterative SRO algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a sketching method for convex and nonconvex regularized least squares and sharp theoretical guarantees are provided for the algorithms.

### Strengths
1. The introduction of the method is clear and easy to follow.
2. The motivation is clearly conveyed.

### Weaknesses
Please see following questions.

### Questions
1. This work appears to be a straightforward extension of (Pilanci & Wainwright, 2016) with the addition of a regularization term. Authors should clarify the unique difficulties and challenges introduced by this regularization term. It does mention some differences like sampling the projection matrix only once. However, can this be easily adjusted in (Pilanci & Wainwright, 2016)? In addition, what are the primary obstacles to applying traditional analyses of convex and nonconvex regularizations in a least squares setting without sketching? What are the difficulties by extending the analysis in (Pilanci & Wainwright, 2016)? What are the new techniques introduced in this paper to study the error bound? 

2. The notations are not very clear. In the paper, $\tilde{\beta}^*$ represents the critical point of the objective function (2). When studying the theoretical properties, is it the solution of SRO algorithm or the solution to iterative SRO algorithm? What are the errors bound for solution of SRO and what are the error bounds for iterative SRO? 

3. In Section 3 about iterative SRO, it only concerns the convex regularization for $h_\lambda(\beta)$. Is there any particular reason that it cannot be applied to nonvex regularization? How about the error bound for the nonconvex regularization with iterative SRO?

4. The organization of the theoretical guarantees and proof sketch is not very clear. For instance, Corollary 4.2 is introduced before introducing Theorem 5.2, while Corollary 4.2 needs the conditions in Theorem 5.2. 

5. In Table 1, why the running time for SRO is longer than those iterative algorithm? The first step of iterative SRO is somehow equivalent to SRO?

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
4

### Summary
The article is concerned with the solution of sketch-based solution of regularized least squares problems, i.e. problems of the form $\min 1/2 \Vert X\beta-y \Vert^2 + f(\beta)$, where $X$ is a matrix and $f$ is a regularization term. If $X$ is of low rank, it can be well approximated via a sketch, i.e a matrix $\widetilde{X}=PX$, where $P$ is a random projector. The idea of sketching is to use this fact to reduce the complexity of the optimization problems.

Concretely, the authors propose the iterative SRO method. In SRO, the quadratic term $\langle X\beta, X\beta\rangle$ of the loss function is exchanged with a term based on a sketch, i.e $\langle \widetilde{X}\beta, \widetilde{X}\beta\rangle$. Since $\widetilde{X}$ is a matrix of smaller dimension than $X$, the latter problem is less expensice to solve. In iterative SRO, this procedure is repeated to obtain a better and better estimate of the true solution $\beta_*$.

The main result of the paper says that as soon as $P$ is an $\ell_2$-subspace embedding for the random matrix $X$, the iterative SRO will converge at a linear rate towards the true solution $\beta_*$. The authors also prove a result about applying the method to do sparse regularization - they show that with a moderate amount of iterations, the iterative SRO achieves the minimax sampling rate when solving the LASSO problem.

### Strengths
This article has many positive sides. The research questions of this article are sound, as is the algorithm it proposes. Its results are relevant and the proofs are modulo typos correct. The experimental verification is also reasonable. 
I would like to highlight the generality of the results (there is much freedom to choose both the regularization term and the sketching matrix $P$). I am also fond of the fact that the authors take the time to prove that there indeed are matrices $X$ that are both of low rank (making their sketching approach viable) but still has the RIP.

### Weaknesses
A big problem with this article is its presentation. Let me address some of the problems, in order of appearance.
- It takes some effort to sort out the relations between the problems (2), (5) and (6) in Section 3 (they are all equivalent - but it is non trivial to realize this due to the notation). Some ironing out of this would be good, possibly with an explicit calculation in an appendix.
- Many of the arguments in the beginning of proof of Theorem 5.2 are written down in an unnecessarily complicated: Since the vector in (25) is zero, there is no need to apply the Cauchy-Schwarz inequality to arrive at the inequality (26)
- In equation 35, $\kappa$ is still present although $\kappa$ has been set to 0 just before.
- There are many steps in the proof of the main result that are only sketched, in particular for $L_h$-smooth $h$ -- I think that I can reproduce the steps, but I really think that the proof of the main result of the paper deserves to have this step written out. *This is in my opinion the most pressing issue*.
- The sentence "We show that RIP($\delta,s$)" has appearantly lost its ending.
- Reference to equation (45) in the proof of Theorem 4.1 seems to be spurious.
- The term JLT in Theorem D.4 is undefined - I suppose it has something to do with a Johnson Lindenstrauss embedding. The meaning of $f_0$ is also unclear.
- Theorem D.6 has no statement.

Another weakness of the paper, in my opinion, is Theorem D.2. Due to the appearance of the term $\alpha_0$ in its current form, one needs to use $n\geq C(s\log(d))^{1/\alpha_0}$ to get a sparse recovery, which, due to the unknown size of $\alpha_0$, could be arbitrarily far away from the sample optimal $s\log(d)$. This question should at least be discussed.

However, I suspect that theorem D.2. can be made stronger. Would it not suffice to just set $X= UA$ where $U\in \mathbb{R}^{n,m}$ is in the Stiefel manifold (with $m=cn$) and $A\in \mathbb{R}^{m,d}$ Gaussian with $m\sim s\log(d)$?

### Questions
- In the beginning of section D.2, is $O(s\log(d)/n)$ a typo (should there not be a square root there)?
- Can the matrix in Theorem D.2 be constructed as I have outlined above, i.e. $X=UA$ with $U$ in the Stiefel manifold (and hence isometric) and $A$ a standard RIP matrix?
- In the formulation of Theorem 3.2, it is stated that $n\ll d$ and $\log(d) \ll n^\alpha_0$ -- it is unclear what this means and how it is used later in the proof.
- Can the authors provide some more details in their proof of their main theorem?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents an approach called Sketching for Regularized Optimization (SRO), which can tackle a broad class of regularized optimization problems by employing sketching techniques. This method is applicable to both convex and nonconvex forms of regularization. To address potential approximation errors in SRO, this paper introduces an enhanced variant, that is, Iterative SRO, which systematically reduces these errors at a geometric rate.  A unified theoretical framework is also proposed to establish the minimax rates for sparse signal estimation in both convex and nonconvex settings.

### Strengths
1. This work addresses a wide range of optimization problems using sketching techniques, effectively handling both convex and nonconvex regularizations.
2. The introduction of SRO and its refined version demonstrate a well-thought-out approach that balances efficiency and accuracy in solving complex optimization tasks.
3. This work presents a unified framework that derives minimax rates for sparse signal estimations, solidifying the method's effectiveness for both convex and nonconvex settings.

### Weaknesses
1. This work appears to be a specific case of the work proposed by Yang and Li, ISIT (2021). The similarities include the theoretical framework (such as the general bound in Theorem 5.2),  the algorithmic approach (the SRO and Iterative SRO), experimental design and results (the findings presented in Figure 1 is exactly the same), and even the writing style. Please clarify the differences between these two works.
2. Please provide more explanation regarding Definition 5.2. For convex functions, the value of theta is less than or equal to 0. So what about the impact of nonconvex functions on this value? For example, what range might it fall within? Additionally, could you provide some examples of nonconvex functions?
3. It seems that the assumption of full column rank in the explanation below Theorem 3.1 is not realistic for practical applications. In other words, if the smallest singular value is relatively small, would the smoothness constant become very small or even approach zero?
4. Some numerical experiments on nonconvex regularization should be conducted.

References
[1] Yang, Y., & Li, P.  FROS: Fast Regularized Optimization by Sketching. In 2021 IEEE International Symposium on Information Theory (ISIT), pp. 2780-2785.

### Questions
Please see the weeknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces an extension of the Iterative Hessian Sketch from Pilanci and Wainwright to the regularized setting. The authors first show convergence of the algorithm for sparse signal estimation (i.e. the LASSO) for both the traditional l1 norm as well as for a non convex penalty. In both cases, under various assumptions,  they are able to derive a minimax error  bound on the deviation between the N^th iterate of the single step algorithm (which they call SRO and which amounts to a minimization of the sketched formulation) and the groundtruth. They then extend their result to a general convex setting and show how to improve convergence by means of an iterative version of the SRO algorithm which is essentially an extension of the IHS algorithm from P and W to regularized formulations.

### Strengths
The paper is well written on the whole. The beginning of the paper remains very close to the paper “Iterative Hessian Sketch: Fast and Accurate Solution Approximation for Constrained Least-Squares” from Pilanci and Wainwright. In particular compare Equation (5) with Equation (25) in this last paper, or (24) with (7).  From what I understand, the main difference lies in the use of a regularizer. An incremental result is not necessarily a bad result but I’m still not fully decided about whether it is sufficiently original to be accepted.

### Weaknesses
The paper is not bad but it should clearly be reorganized. The distinction between the solution of the sketched program and the final iterate of the SRO/iterative SRO algorithms is not very clear. Does \tilde{\beta} represent a critical point of (2) or the N^th iterate of Iterative SRO ? This is especially unclear for section 4.2. where the meaning of \tilde{\beta} is not clarified in the statements of Corollary 4.2., and Theorem 4.5. but is mentioned as a critical point in the statement of Theorem 5.2.

Also your key contribution is the sketching step so the \tilde{n} should appear more clearly in all your statements (see my comments below)

### Questions
- line 30 “Sketching algorithms has been used to approximately” —>  “Sketching algorithms have been used to approximately”
- line 62, Equation (30) the lines 129-131 should appear way earlier. In particular, you want to better motivate the fact that the sketching is restricted to the quadratic term. I understand that restricting the sketching to the quadratic term might improve convergence but is that really necessary? If you still need to compute the product y^TX when do you the gradient descent, isn’t that equivalent to the X^TX*\beta ? I.e. in the latter you just have 2 more applications of a vector to the matrix X.
- line 68, you give the definition of the semi-norm ||u||_X twice 
- line 116 I would use the cardinality of the support to denote the number of non zero elements of a matrix X instead of introducing the notation nnz. I.e. |supp(X)|. It is a detail though
- line 123: “It is comprised of ” —> “It consist of”
- lines 125-127 are quite obvious to me. I think you could just remove them
- line 136-140, in the Definition 2.1. is there any reason you use \ell^2 instead of \ell_2 ?
- line 142, Definition 2.2. looks more like a proposition (Lemma  perhaps) to me. 
- line 151 : “Taking derivative with respect to” —> “taking the derivative with respect to”?
- line 189-190 “is a more accuracy” —> “is a more accurate”
- line 170 “consectively” —> “consecutively” or “iteratively/alternatively”?
- The structure of the proof sketch for Theorem 3.1. is good but your exposition of formulations (5) (6) and (7) is unclear. You should either keep the formulation from (6) to (7) (aside from the sketching) or provide at least a to (short) sentence indicating that (7) is obtained by moving the \beta^{(t-1)} out of the quadratic term to the linear term.
- Lines 295-296, Statement of Assumption 3, is the concavity parameter \zeta_- that you introduce in Assumption 3 the same as the zeta_- that you use in Assumption 2? if so it should be clarified in Assumption 2 (i.e. you could say something like “where zeta_- is known as the concavity parameter” or slightly modify Assumption 3 by saying “The concavity parameter \zeta_- introduced in Assumption 2…”)
- In the statements of Theorem 4.1, Corollary 4.2, Theorem 4.5 (this is especially true for Theorem 4.5), I would remove the explicit constants. The relation between the values you get for \tilde{n} and the constants is not clear anyways so those should go (keep the \varepsilon/1-\varepsilon for Corollary 4.2). If you really want you can add a one line sentence for Corollary 4.2. indicating how they depend on \rho_{\mathcal{L}, +} and \rho_{\mathcal{L}, -}. 
- I recommend that you change the statement of Theorems 4.1, 4.5 and 5.2. so that they look more like the statement of Corollary 4.2. in terms of how you introduce the \tilde{n}. I.e. the lower bounds on \tilde{n} should not be introduced as a side comment. \tilde{n} should appear before you introduce \tilde{\beta}. E.g. Let \tilde{\beta} denote the solution to (5) with a sketching operator of size \tilde{n}\times n (or even better “for the sketching parameter \tilde{n}…”)
- I’m not sure Figure 1 is really meaningful. It only indicates a constant reduction in the relative error. What is more interesting is a bound such as (8)
- Lines 403-407, Definition 5.2. From what I understand, the degree of non convexity for a particular value of kappa gives you a measure of whether the function is more convex than the function \kappa \|x\|^2 ? I.e \theta_h will give you a negative value if the Hessian of the function is more psd than -\kappa I and positive otherwise? Remark 5.1. could be clearer
- Lines 416-420, in the statement of Theorem 5.2., what do you mean by an “oblivious” \ell^2 subspace embedding? Are you referring to an embedding that satisfies Definition 2.1. ? Then why not just say it like that? If you use space to introduce Definition 2.1, leverage it. 
- line 424 - 426 : “by arbitrary critical point ” —> “by any arbitrary critical point”
- lines 428-431, from what I understand \theta_{h_\lambda} is lower bounded by \kappa? it might be good to add even a short clarifying sentence
- The sentence on line 430 does not make sense. Do you mean that “if the if the subdifferential is Lipschitz or if \kappa can be set to 0 then you get an admissible error bound?” If so, this is not what Theorem 5.4. states.  
- Statement of Theorem 5.4., I would remove the sentence following Equation (17) as it refers to a result that does not appear? 
- I think I would reorganize sections 5.1. and 5.2 . At the end of the day, the two Theorems 5.2. and 5.4 are really steps towards the proof of Theorem 3.1. I would add a one paragraph below this last theorem explaining how you use the degree of non convexity to prove it. This would make it possible to remove the statements of Theorems 5.1 and 5.2
- general question: wouldn’t it be more efficient to draw a new sketch at each iteration in the iterative SRO algorithm ?

### Soundness
3

### Presentation
2

### Contribution
3
