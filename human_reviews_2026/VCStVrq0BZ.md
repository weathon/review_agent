# Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Stochastic interpolants offer a robust framework for continuously transforming samples between arbitrary data distributions via ordinary or stochastic differential equations (ODEs/SDEs), holding significant promise for generative modeling. While previous studies have analyzed the finite-time convergence rate of discrete-time implementations for SDEs, the ODE counterpart remains largely unexplored. In this work, we bridge this gap by presenting a rigorous finite-time convergence analysis of numerical implementations for ODEs in the framework of stochastic interpolants. We establish novel discrete-time total variation error bounds for two widely used numerical solvers: the first-order forward Euler method and the second-order Heun's method. Our analysis also yields optimized iteration complexity results and step size schedules that enhance computational efficiency. Notably, when specialized to the diffusion model setting, our theoretical guarantees for the second-order method improve upon prior results in terms of both smoothness requirements and dimensional dependence. Our theoretical findings are corroborated by numerical and image generation experiments, which validate the derived error bounds and complexity analyses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work provides the first finite-time convergence analysis for ODE-based stochastic interpolants, establishing total variation error bounds for the Euler and Heun methods. The results yield optimized iteration complexity and step-size schedules, offering improved smoothness and dimensional dependence compared to prior diffusion model analyses.

### Strengths
(1) The paper is the first to analyze the convergence of ODE-based stochastic interpolants. The analysis is rigorous, and the presentation is very clear. The convergence results show improvement compared with prior works.

(2) It conducts synthetic experiments to exhibit the TV distance.

### Weaknesses
(1) There are some typos and ambiguities in the proof.
Line 730: In the triangle inequality, the last term (term C) seems to be $b(t,X_t)$, instead of $b(t,X_{t_k})$. The remaining proof should be correct.

Line 792: It takes me some time to verify the correctness of these inequalities. However, the reasoning skips too much explanation of why they work. I suggest adding a detailed description of why we need these inequalities and how they are proved, including the calculation of Holder coefficients.

Line 927: The $G_{t_k \rightarrow t_{k+1}}$ should be $F$. Moreover, the definition of $F$ should be restated to make it clearer.

(2) At first glance, I feel that the proof technique is very similar to that in Li et al.2025(b) with the application of Lemma C.2. After a careful examination, I find that it contains some novel techniques that should be highlighted. Could you add an additional section (Maybe in the appendix) to discuss the novel techniques (not just bounds) different from those used in Li et al.2025(b), and why these calculations can lead to a better result? In my opinion, this might be very helpful to readers. Same as the analysis of Heun's method, for example, Huang et al. 2025.

### Questions
In my understanding, many techniques are closely related and can be applied to previous analysis of diffusion ODEs. Do you think there are any properties specific to the stochastic interpolant problem (I.e., starting from another distribution instead of Gaussian noise)? For instance, can you provide some concrete examples to show the difference between the stochastic interpolant and the original diffusion process?

### Soundness
3

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
This paper presents a theoretical study of iteration complexity for discretization schemes of stochastic-interpolant ODEs. As representative schemes, we analyze the first-order forward Euler method and the second-order Heun’s method. Under certain assumptions, complexity bounds for achieving $epsilon$-TV error for each scheme are derived. The analysis proceeds via carefully constructed continuous-time interpolants, which allow us to import continuous-time techniques to quantify discretization error and evaluate approximation accuracy.

### Strengths
**(S1) Important research theme.**
Investigating the iteration complexity of ODE solvers to ensure accurate generation with stochastic interpolants is an important research direction. Although diffusion models (in discretization form of neural ODE) are a major focus in machine learning, most evidence remains empirical. This work may provide the missing theoretical analyses underpinning diffusion models.

**(S2) Theoretical contributions.**
For general stochastic-interpolant ODEs, we derive the iteration complexity required to achieve $\epsilon$-TV error when using the first-order forward Euler method and the second-order Heun’s method.

### Weaknesses
**(W1) Positioning relative to prior work.**
In the Introduction (lines 048–049), you state that for ODE-based transformations, “the analysis has been limited to the continuous-time setting.” Meanwhile, Sec. 2.2 (lines 123–127) and Appendix B (Table 1) summarize prior results on the time needed for an ODE to reach $\epsilon$-TV error. Am I correct that all of these results are derived in continuous-time setting and do not account for discretization error? If YES, please revise Sec. 2.2 to make this explicit.

**(W2) Challenge level of the proposed analysis technique.**
My understanding is that the techniques in Sec. 4.1 (“Main ideas of the proof”)—which bridge the gap between a continuous-time process and a discrete-time estimator—are the key methodological contribution, enabling tools from continuous-time ODE analysis to be imported into the discretized setting. That said, it is somewhat difficult for me to gauge how technically challenging this bridge is. Could you additionally explain the main difficulties encountered in introducing this technique?

### Questions
(1) Definition of $d$ seems not appear in Introduction.

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
3

### Summary
This paper gives convergence analysis of ODE methods for stochastic interpolants under general conditions, in particular, the first-order (forward) Euler method and second-order Heun's method.

For Euler's method, the requirements are L^2 error bound on the drift $\epsilon_{drift}$, an expected Jacobian estimation error in Frobenius norm $\epsilon_{div}$, and smoothness bounds (Frobenius norm bound in the Jacobian & 2nd derivative of the score and Lipschitzness of divergence). The theorem gives $O(d^2/\epsilon)$ iteration complexity.

For Heun's method (with Jacobian estimation error in Frobenius norm squared), the theorem gives $O(d^{3/2}/\epsilon^{1/2})$ iteration complexity.

Experiments were run to obtain the empirical scaling of TV error with step size and dimension.

### Strengths
The paper is the first to give an analysis of the ODE discretization for stochastic interpolants under general conditions. This works for any interpolations with path $I$ and noise magnitude $\gamma(t)$ satisfying some smoothness conditions, and estimates satisfying the assumptions. Although the analysis techniques do not seem particularly novel (in being similar to previous ODE flow analyses), it is nevertheless valuable to work out the precise bounds in new setting under carefully detailed general assumptions, as in this paper. In particular, the theory shows the benefits of a higher-order ODE integrator and under what smoothness/error assumptions we can expect the improvement.

### Weaknesses
There is an additional factor of $d$ in Euler's method compared to diffusion models, and it is unknown whether this is a limitation of the analysis, or necessary in this general setting. (The experiments suggest the true scaling is linear.)

### Questions
It would be helpful to explain where the extra factor of $d$ come from compared to diffusion models. 

* p. 6: "Divergencce" -> "Divergence"
* p. 8: "folloiwing" -> "following"
* p. 8: Missing period after "unbounded"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates finite-time convergence of ODE samplers built from stochastic interpolants (SI). While previous studies have established finite-time bounds for SDEs, the ODE scenario has notably lacked discrete-time guarantees. The authors fill this gap by providing total variation (TV) distance error bounds and detailing the iteration complexity for both the forward Euler and Heun's methods under regularity and estimator-error assumptions.

### Strengths
- The paper addresses a significant theoretical gap for SI-based ODEs by providing finite-time TV bounds and iteration complexities for both Euler and Heun methods within the SI framework.
- Mathematical proofs are solid and clear. Discrete-to-continuous interpolation yields a piecewise ODE, enabling drift/divergence comparisons to bound TV.
- 2D tasks and Gaussian-mixture tests verify $O(h)$ (Euler) and $O(h^2)$ (Heun) discretization orders.

### Weaknesses
- The requirement of uniform Lipschitz on $\hat{b}$ and its divergence (Assumption 4.4) seems kind of idealized. Is it possible to provide an empirical verification?
- Experiment demos are limited to three 2D transformations and d-dim Gaussian mixtures; Is it possible to provide results on real-data benchmarks?
- The theory in this paper establishes error bounds that scale with $d^2$ and $d^3$. However, the empirical findings indicate a roughly linear correlation, suggesting room for further refining on the bounds.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
