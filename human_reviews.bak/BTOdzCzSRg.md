# qNBO: quasi-Newton Meets Bilevel Optimization

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 8, 5

## Abstract
Bilevel optimization, which addresses challenges in hierarchical learning tasks, has gained significant interest in machine learning. Implementing gradient descent for bilevel optimization presents computational hurdles, notably the need to compute the exact lower-level solution and the inverse Hessian of the lower-level objective. While these two aspects are inherently connected, existing methods typically handle them separately by solving the lower-level problem and a linear system for the inverse Hessian-vector product. In this paper, we introduce a general framework to tackle these computational challenges in a coordinated manner. Specifically, we leverage quasi-Newton algorithms to accelerate the solution of the lower-level problem while efficiently approximating the inverse Hessian-vector product. Furthermore, by leveraging the superlinear convergence properties of BFGS, we establish a non-asymptotic convergence analysis for the BFGS adaptation within our framework. Numerical experiments demonstrate the comparable or superior performance of our proposed algorithms in real-world learning tasks, including hyperparameter optimization, data hyper-cleaning, and few-shot meta-learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a bilevel optimization framework, QNBO, which utilizes quasi-Newton methods to address the computational challenges associated with hierarchical learning tasks. The authors integrate quasi-Newton recursion schemes for efficiently approximating inverse Hessian-vector products, specifically using BFGS and SR1 methods. The authors argue that QNBO offers superior convergence and computational efficiency, substantiated through theoretical proofs and numerical experiments.

### Strengths
1. The QNBO framework presents an innovative approach by combining bilevel optimization with quasi-Newton methods to improve hypergradient approximation accuracy.

2. The paper provides a non-asymptotic convergence analysis for QNBO, especially for BFGS, detailing convergence under both quadratic and general settings. 

3. The authors validate QNBO on real-world tasks such as hyperparameter optimization, meta-learning, and data hyper-cleaning. QNBO’s performance is benchmarked against established algorithms, including SHINE and PZOBO, with performance gains observed in multiple scenarios.

4. I did not notice any major issues with the clarity of the presentation.

### Weaknesses
1. The SR1 method, despite its potential in quasi-Newton methods, is noted for its numerical instability in general functions without correction strategies.

2. QNBO offers a flexible choice between subroutine A and B based on performance and computational cost. However, the conditions for selecting one over the other are not rigorously outlined.

### Questions
1. Given the success of stochastic adaptations in other optimization frameworks, could the authors discuss potential challenges or advantages of incorporating stochastic quasi-Newton methods into QNBO? Specifically, how might this adaptation impact convergence rates or computational efficiency in practical implementations?

2. The initial matrix $H_0$ plays a significant role in quasi-Newton methods. Here, a scalar multiple of the identity matrix is used. Could the authors provide specific results or theoretical analysis on how different choices of $H_0$ might affect QNBO’s convergence rate and computational efficiency?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work studies the bilevel optimization problem, where the lower-level problem is strongly convex and the upper-level objective function is nonconvex. The authors introduce a novel framework called qNBO, which approximates the hypergradient by employing quasi-Newton methods for both evaluating the lower-level solution and estimating the inverse Hessian-vector product. When the quasi-Newton method used is BFGS, they establish a non-asymptotic convergence for the proposed algorithm. Finally, empirical experiments demonstrate the superior performance of qNBO compared to other bilevel optimization algorithms.

### Strengths
1.The new bilevel algorithms are well motivated. By employing quasi-Newton recursion schemes, they incorporate two subroutines to accelerate hypergradient approximation and avoid incorrect inversion in solving the bilevel optimization problem.

2.Several machine learning tasks, including hyperparameter optimization in logistic regression, data hyper-cleaning, and few-shot meta-learning, are evaluated, demonstrating the superior performance of the proposed methods. Additionally, various ablation studies are provided.

3.A convergence rate guarantee is established when the quasi-Newton method used is BFGS.

### Weaknesses
My main concern is that: 

Although quasi-Newton methods are used to estimate the inverse Hessian-vector product, the proposed algorithms still require computing Jacobian-vector products, which can be computationally expensive in large-scale cases. Could this computation pose a potential problem?

### Questions
For both the theory and experiments, here are a few things that need to be addressed.

1.In Remark 3.4, the authors stated that the qNBO (SR1) algorithm lacks convergence guarantees without specific corrections used to achieve numerical stability in the general case. It was also mentioned that the performance of qNBO (SR1) on the 20News dataset was omitted due to its ineffectiveness in the experiment in Section 4.2. What are the underlying reasons for these issues? A bit more discussion on these points would be helpful.

2.From the first terms on the right hand sides of both (6) and (7), it seems that Theorems 3.3 and 3.6 require a lower bound assumption for $\Phi(x)$. Is this correct? Additionally, in the proof sketch in Appendix D.2, $\tilde{\nabla}\Phi$ is mentioned but not defined. It would be helpful to specify its definition. 

3.In the experiments, how to select the step sizes and the number of inner iterations for the proposed algorithms?

4.In Table 1, why does qNBO (BFGS) save so much time compared to SHINE for data hyper-cleaning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Summary:

The paper considers bilevel optimization problems with strongly convex lower-level objectives and proposes a framework for performing hypergradient approximation by using recursive quasi-Newton schemes to avoid issues in Hessian inversion arising in common BO algorithm. 
The proposed algorithms, qNBO (BFGS) and qNBO (SR1), take inspiration from an existing algorithm 'SHINE' (Ramzi et al 2022). SHINE uses quasi-newton for solving the lower-level and reuses the approximate inverse hessian to perform hypergradient computation. 
However, such approximate hessian is only accurate when multiplied with a specific vector: the gradient of the inner objective. 
Hence, it might yield inaccurate solutions when multiplied by the gradient of the outer objective, as required by implicit differentiation. Although SHINE propose a way to steer the approximation towards the upper-level, it is unlear to what extend the method enjoys good quantitive convergence rates. This paper addresses such limitation by introducing a separate quasi-newton based estimation of the inverse-hessian along the gradient of the outer objective. This results, in principle, in a more accurate estimation of the hypergradient. 
The paper then provides finite time convergence guarantees and illustrate the method on a toy example, a regression task and a meta-learning task.

### Strengths
- The motivation and approach in the paper are quite interesting, it makes sense to accelerate convergence of the inner problem using quasi-newton approaches. It is also nice that the method provide convergence rates for their algorithm, although I think the analysis do not seem to capture the benefits of qn for computing the hyper-gradient (when computing  the iterates u_k). 
- The paper is overall clearly written and well explained, which makes it easy to read. 

- The method seems to give quite an improvement compared to SHINE in terms of convergence speed (no full finite time convergence was provided for SHINE). While the experiments seem in favor of the proposed method, I think the implementation of SHINE contains some errors after checking the code (see weakness section).

- I appreciate that the authors provided the code to reproduce the results, this allows for a better assessment for the results. Although there were bugs in it, that unfortunately change some conclusions (see the weakness section), I think this should be seen as an opportunity to improve the experimental study.

### Weaknesses
- Experiments: The results in the toy experiment looked suspicious to me: AID-TN, AID-BIO and AmigoCG seemed to work unreasonably worse on a quite simple example where they are supposed to perform quite well. I decided to check the implementation provided in the supplementary and found a number of bugs that explain these results. Please refer to the questions section for the details of these bugs. 
I think these can be easily fixed. However, after fixing these bugs, the results do not exactly match the conclusions of the current paper: it is possible to solve the toy problem faster/more accurately using AID-BIO/AmigoCG using a single step of CG (instead of 10). More precisely, after fixing the bugs: AID-BIO/AmigoCG reach 10^-4 relative error on the upper variable (instead the numbers reported in the paper: 10^-2 for AID-BIO and 10^0 for AmigoCG), and further go below 10^-5 (better than qNBO) when using only a single CG step and single gradient step. AID-TN reaches 10^-1, instead of 10^0. I do not exclude other bugs on other methods (the implementation of SHINE already looks incorrect (see questions section)), so I strongly encourage the authors to test the implementation with care. 




- Convergence results:  
While the convergence analysis exploits the super-linear convergence of quasi-newton for the inner-level iterates, the iterates used for computing the hyper-gradient do not seem to benefit from such super-linear rate. In fact, the convergence of these iterates is sub-linear: 1/Q where Q is the number of quasi-Newton iterates.  This might explain the worse dependence of the algorithm on the number of gradient queries on the lower loss epsilon^{-2} compared to AID-BIO epsilon^{-1}. Still there is an improved dependence on the conditioning for the vector-jacobian complexity (kappa^3) vs AID-BIO (kappa^{3.5}). However, it is unclear how this compensates the increased complexity for the gradient, especially given the results of the toy experiments (without the bugs). 
On a positive note, the result is an improvement over the guarantees provided for SHINE as they are quantitative.  

Ablation studies: methods such as AID-BIO do not exclude using other solvers for the lower-level problem. One could then use quasi-newton methods and still apply CG for the conjugate gradient. How do you think this would perform? 


Although I think the proposed direction is interesting, there is a number of problems with the current version of the paper that need to be addressed. I have many reasons not to trust the empirical results and that would require checking all the details of the code for all experiments. It is unclear to me if the rebuttal period would be enough for that.

### Questions
Some of the figures in the appendix seem to be consitent with what the python file toy3.py produces. However, it contains a number of bugs for various implementations: 
 1- AID-TN:  a) the implementation of the function TN is wrong, it should be:
	 	v = v - inner_lr * output
		return inner_lr * p
	instead of :
		v = v + inner_lr * output
		return p

	     b) the function TN (truncated-newton) should be called to compute v before computing the cross derivative f_xy, otherwise, one uses an outdated version of v for computing the hyper-gradient i.e.:  
		v = TN(fyy,Fy)
            	fyx=f_xy(x,y,v)
	instead of:
		fyx=f_xy(x,y,v)
		v = TN(fyy,Fy)
		
 2- AID-BIO:  Same issue as in AID-TN: Conjugate gradient should be called before computing the cross-derivative for the same reasons, otherwise this results in unnecessarily inaccurate hyper-gradients which slow performance:
		v = cg(fyy,Fy,v,10)
		fyx=f_xy(x,y,v)
	instead of:
		fyx=f_xy(x,y,v)
		v = cg(fyy,Fy,v,10)
 3- AmigoCG: In the deterministic case, this corresponds exactly to AID-BIO algorithm, so they should give the exact outputs. Similarly to AID-BIO, cg should be called before the cross-derivative. Additionally, there is no need to evalulate f_x(x,y) and f_y(x,y) for hyper-gradient computation as this unnecessarily slows the method. More importantly, there should be a minus sign instead of a + in front of the implicit gradient:
		x_grad=Fx-w
	instead of:
		x_grad=Fx+w

 4- Shine: The implementation is also incorrect though I did not have time to fix it myself. The OPA correction involves only the cross derivatives of the inner-level function not the outer level function:
		ogrady = f_xy(x,y)
	instead of:
		ogrady = F_y(x,y)

### Soundness
1

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
4

### Summary
This paper proposes an algorithmic framework, qNBO, which leverages quasi-Newton methods to solve bilevel optimization problems in the nonconvex-strongly-convex setting. Some practical implementations are also provided under the theoretical framework. It establishes non-asymptotic convergence for the BFGS adaptation within the framework and provides numerical experiments demonstrating the superior performance of the proposed algorithms across various machine-learning applications.

### Strengths
1. The paper is well written and easy to follow. Quasi-Newton type methods for solving bilevel optimization have not been well studied, even in the nonconvex-strongly-convex setting. This work provides a quite general algorithmic framework, allowing any quasi-Newton method to be applied in qNBO.
2. A convergence rate and complexity analysis are provided for qNBO (BFGS). Technical derivations seem to be nontrivial. The authors incorporate the superlinear convergence of BFGS into the non-asymptotic convergence framework commonly used in bilevel optimization literature. 
3. Experiments validate the promise of the proposed methods.

### Weaknesses
1. For the theory, the results in Theorems 3.3 and 3.6 are limited to the setting where $Q_k=k+1$. As a result, qNBO is a double-loop algorithm. Is it possible to design a single-loop version? Recent progress has been made toward single-loop bilevel optimization algorithms, especially in the nonconvex-strongly-convex setting, by using a warm-start strategy. 
2. For the experiments, since the value of $Q_k$ affects the running time, it would be beneficial to empirically demonstrate how increasing $Q_k$ influences performance gains. Note that the numerical results in Fig. 1(d) illustrate only the impact of $Q_k$ as the outer iteration $k$ varies.
3. Page 6: Do Theorems 3.3 and 3.6 require the assumption that an optimal solution $x^*$ exists? Note that $\Phi(x^*)$ appears in both (6) and (7). 
4. Page 7: Why is it stated that qNBO is more efficient than AID-BIO? Note that the notation $\tilde{\mathcal{O}}$ omits the $\log\frac{1}{\epsilon}$ term. 
5. In Appendix C.5, the authors report various ablation studies on the parameters $T$ and $Q$, but they do not specify which experiment these ablation studies are based on.

### Questions
Please see the questions in the weakness part. Here, I want to add some minor questions:

1. Page 6: In Eq. (6) of Theorem 3.3, the constant $M_{f_{xy}}$ can be computed, as $f$ takes the quadratic form given in (5).
2. Page 8: In Eq. (9), ``${a_i}’$, ${b_i}’$” should be ``$a’_i$, $b’_i$”.

I think this paper provides a good attempt to use second-order optimization to accelerate bilevel optimization. Overall, I like the approach. I am raising my score to 8 if the questions and the typos can be addressed well in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper integrates a quasi-Newton approach into a bilevel optimization algorithm to approximate the inverse Hessian-vector product.

### Strengths
The proposed method avoids costly second-order computations for approximating the Hessian-vector product while achieving a comparable convergence rate.

### Weaknesses
The paper is challenging to follow, with a vague explanation of the quasi-Newton recursion scheme. The design of the proposed method appears overly complex compared to existing methods, and there is limited explanation regarding the validity of  $u_{k+1}$ as an accurate approximation for the Hessian-vector product.

### Questions
Can this method be effectively applied in a stochastic setting while maintaining comparable convergence rates and computational efficiency? The required adjustments in iteration rounds and step size suggest potential difficulties in achieving the same convergence speed as existing methods.

### Soundness
2

### Presentation
2

### Contribution
3
