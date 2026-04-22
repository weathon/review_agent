# Newton Method Revisited: Global Convergence Rates up to $O(1/k^3)$  for Stepsize Schedules and Linesearch Procedures

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 4

## Abstract
This paper investigates the global convergence of stepsized Newton methods for convex functions with Hölder continuous Hessians or third derivatives. We propose several simple stepsize schedules with fast global convergence guarantees, up to $\mathcal O( k^{-3} )$. For cases with multiple plausible smoothness parameterizations or an unknown smoothness constant, we introduce a stepsize linesearch and a backtracking procedure with provable convergence as if the optimal smoothness parameters were known in advance.
Additionally, we present strong convergence guarantees for the practically popular Newton method with exact linesearch.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents new convergence guarantees for stepsized Newton methods under Holder continuity assumptions on the Hessian or third derivatives. The authors reinterpret the classical Newton method as a third-order tensor method and propose a family of stepsize schedules (RN), as well as linesearch and backtracking variants (GRLS, UN). These achieve global rates up to O(k^-3), improving upon the best-known O(k^2) results. The paper also provides convergence results for the “Greedy Newton” linesearch, supported by clear theoretical analysis and numerical experiments.

### Strengths
1. The theoretical contributions are solid and clearly positioned relative to prior work (notably Hanzely et al., Doikov et al., and Nesterov’s tensor methods).

2. The paper successfully extends the analysis of Newton-type methods into the regime of Holder continuous third derivatives, showing rates that were previously thought unattainable for basic Newton variants.

3. The exposition is mathematically rigorous and detailed, with clear structure and self-contained proofs.

4. The proposed algorithms (RN, UN, GRLS) are simple, practical, and connect naturally to existing Newton variants.

5.  The paper propose variants with adaptive sptesizes, which is important since we cannot 

6. Experimental results are well designed and demonstrate that the proposed methods are competitive, often outperforming baselines.

### Weaknesses
1. The experimental section, though helpful, remains somewhat limited in scope (mostly synthetic or classical problems). Stronger empirical validation on large-scale or nonconvex deep learning tasks would better support the practical relevance.

2. The proposed linesearch requires evaluating the dual norm, which in turn involves an additional matrix inversion at each iteration. This cost is not discussed, and it can be significant in large-scale settings. 

3. The experimental section reports convergence in terms of iterations (steps) rather than the actual number of matrix inversions or Hessian factorizations, which makes the practical comparison somewhat misleading.

4. The paper’s tone occasionally overstates its novelty. Conceptually, the main contribution builds on a straightforward observation: by substituting the Newton step into the Hölder continuity bound and optimizing the resulting inequality, one obtains improved global rates. This is an elegant and technically solid development, but the underlying idea remains relatively simple rather than transformative. The paper's strength lies in its clarity and rigor, not in a fundamentally new algorithmic principle.

5. The claim that the method can be viewed as a “third-order” method feels overstated. The algorithm still uses only first and second derivatives; the “third-order” aspect comes only from the smoothness assumption and the resulting analysis. This is somewhat analogous to calling gradient descent a “second-order” method because it relies on a bound on the Hessian variation.

### Questions
See weaknesses.

### Soundness
4

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
The paper studies various choices of step-size for Newton's method and the convergence guarantees they provide. It generalizes recent ideas (notably from Hanzely et al. 2022) that exploits the link between specific step-size choices and third-order regularization techniques. The main generalization revolves around higher-order regularity assumptions on the objective functions, considering in particular third-order H√∂lder continuity.

The paper provides various algorithms, the main ones called RN, when the smoothness parameters of the function are known, and UN when they are unknown. The authors provide various convergence results for functions that have bounded level sets (which is equivalent to coercivity for convex functions). The results essentially extend known results for twice differentiable functions to the case of 3rd order differentiable functions and considering any order $\nu\in[0,1]$ of Holder continuity (when often only the continuous case $\nu=1$ is considered).

The contribution is good and interesting, yet the writing is, in my opinion, problematic at many points, which make the paper and the importance of various results unclear (see below), or sometimes misleading (for example I do not believe the paper improves the known results for the case $p=2, \nu=1$, yet this is implictely implied in Sec. 1.5, see hereafter). 
Additionnally, I have several important concerns regarding the practical usefulness of the main algorithms (RN, UN and GRLS) introduced in the paper, which I detail in the weaknesses section.


Overall, the paper has merits and the contribution is fair and of interest to the community. Yet I belive that the paper needs significant improvements the writing and to give stronger arguments to convince the reader on the  the interest and impact of the main algorithms and results. It seems to me that it is yet to demonstrate how useful they are in practice compared to existing Newton's methods.
I would welcome clarifications and feedback from the authors on these key points. I believe the paper is not ready for publication in its current state but I would increase my rating if the writing is improved to clarify several key points as recommended below.

### Strengths
* Strong theoretical results on several new variants of Newton's method.
* Thorough analysis discussing all the main problems with Newton-like methods (line search or not, unknown regularization, etc.).
* Generalization of previous results to higher-order smoothness and H√∂lder continuity.
* Clear explanation of prior work
* Overall rigor on the theoretical side (despite some clarifications needed in the writting, especially in the introduction section).
* The RN algorithm is interesting and rather simple to implement when the smoothness constant is known. I have concerns regarding the other methods but RN is promissing.

### Weaknesses
* The introduction would be difficult to follow for reader not familiar with Newton's methods are many things are discussed without being properly defined (or often defined a few pages later). For example, Newton's methods is introduced in equation (1) without a step-size, but this step-size is then extensively discussed. 
* The original motivation given by the authors in the introduction was to improve the known rate for Newton's methods in the case $p=2$ and to bring it closer to the optimal rate. Yet the main contribution is in the end for $p=3$, as in the case $p=2$ the results do not improve the $O(k^{-2})$ previously known rate. 
* Some results would benefit from being further discussed to assert their usefulness in practice, a few examples are given in the questions below.
* The GRLS method in Section 3 seems much more expensive that the Greedy Newton method, since it requires computing $\langle \nabla f(y),\left(\nabla^2 f(x_k)\right)^{-1}\nabla f(y)\rangle$, possibly for several $y$ at each iteration. 
The authors then explain the method empirically produces step-sizes close to those of Greedy Newton. Therefore I am not sure that the method has a practical interest due to its high computational cost compared to (GN)?    
* The Universal Newton (UN) method is meant to be used on function where the smoothness constant $M_q$ is unknown, but it requires the value of $\gamma$ which is also related to the smoothness of the problem and is likely to be unknown as well. Similarly (see the questions below), it is not clear how $\theta_{k,0}$ is initialized sufficiently small (and it is not stated in the algorithm), since the true value is unknown. Therefore I am not sure that UN is more easy to use in practice than RN.

### Questions
* In Section 1.1, what does "basic" methods means? Is there a rigorous definition of it?
* Could the author clarify the global assumptions made on $f$ throughout the paper? Indeed, in Sec. 1.2, $f$ is convex twice differentiable and has non-degenerate Hessian. Since Holder continuity implies continuity, we later guess that it is actually $C^2$, or is this not correct? 
* The assumption in Sec 2.2 is equivalent to coercivity (because $f$ is convex), so adding the $C^2$ property and non-degeneracy discussed above, it seems that $f$ is actually strongly convex. If this is correct this should be more straightforwardly stated. 
* Similarly, the authors denote $x^\star$ any element of $\argmin_x f(x)$ but is this argmin assumed to be non-empty? Could the author clarify this sentence?
* The rate obtained for $p=2$, $\nu=1$ (so Holder-continuity of 2nd derivative) is no better than $O(k^{-2})$, so the current work does not improve on previous work for this case. Improvement require further assumptions. Yet the contribution section 1.5 makes it sound as it the case. I recommend reformulating or clarifying if I missed something, in the current form the contribution statement is misleading. 
* In Theorem 3, is the factor of the superlinear local rate always smaller than $1$? That is, is it guaranteed that $\frac{2}{\gamma} (9M_q)^{1/q-1}<1$? I recommend further discussing this result and its consequences.
* In the case where smoothness parameters are unknown, the proposed Universal Newton (UN) method resorts to doing a line-search on $\theta_k$ to then use a step-size $\alpha_k = \frac{1}{1+\theta_k}$. Could the author clarify whether there are real advantages compared to  doing the line-search directly on $\alpha_k$ as usually done?
* In Section 4, regarding the UN algorithm, the authors say that the backtracking strategy consists in starting with an estimate of $\theta_k$ that is smaller than the true value. How is it possible to ensure this since the true value is unknown?

### Soundness
2

### Presentation
2

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
This paper investigates globalization strategies for the Newton method under Hölder-type smoothness assumptions, including Hölder continuity of Hessians and third-order derivatives. The authors make three key contributions: (i) reinterpreting stepsize-controlled Newton methods as higher-order (third-order) regularized methods; (ii) proposing a new class of stepsize schedules (Root Newton — RN) along with corresponding line-search and backtracking procedures (GRLS and UN) that do not require exact knowledge of smoothness constants; and (iii) establishing global convergence rates up to $\mathcal{O}(k^{-3})$ under appropriate Hölder conditions. The work also provides local superlinear and linear convergence guarantees under additional assumptions, supported by numerical experiments comparing RN/UN/GRLS against existing Newton globalization techniques. The manuscript positions its main contribution as improving the best-known global rate for Newton stepsize methods from $\mathcal{O}(k^{-2})$ to $\mathcal{O}(k^{-3})$ by leveraging Hölder continuity of higher-order derivatives.

### Strengths
1. The paper achieves an improved global convergence rate of $\mathcal{O}(k^{-3})$ under Hölder continuity of higher-order derivatives, advancing the state-of-the-art for stepsize-based Newton methods.
2. The proposed GRLS and UN procedures eliminate the need for known smoothness constants while retaining theoretical guarantees, enhancing practical applicability.
3. A comprehensive set of convergence results is established, including local superlinear, global linear (under additional assumptions), and global sublinear rates.
4. Experimental results validate the theoretical claims and demonstrate the effectiveness of the proposed methods.

### Weaknesses
1. On Page 3, Line 132, the authors claim that the RN stepsize schedule generalizes the AICN stepschedule when $p=2$, $\nu=1$ (i.e., $q=3$). However, substituting these parameters yields $$\alpha_k = \frac{1}{1 + (9L_{2,1})^{1/2} \|\nabla f(x^k)\|_{x^k}^{*\frac12}},$$

which differs structurally from the AICN stepsize $$\alpha_k = \frac{2}{1 + \sqrt{1 + 2\sigma \|\nabla f(x^k)\|_{x^k}^*}},$$ even up to a constant factor. This claim requires clarification or correction.

2. On Page 6, Line 320, the authors state that Assumption 1 has been used in [1], but this assumption does not appear explicitly in the referenced work.

3. On Page 3, Line 158, the authors describe equation (6) as a "third-order tensor method." It would be more accurate to characterize it as a second-order method with third-order regularization, since the subproblem in (6) involves only up to second-order derivatives.

4. Minor typo: On Page 26, Line 1352, in the first equation, $\nabla^2 f(x - \tau(y - x))$ should be corrected to $\nabla^2 f(x + \tau(y - x))$.

### Questions
See the weakness.

### Soundness
3

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
4

### Summary
This paper revisits the global convergence properties of **Newton-type methods** under general **Hölder continuity** assumptions on the Hessian (or higher derivatives).  
Building on the **AICN framework of Hanzely et al. (2022)**, the authors propose a family of **stepsize rules** derived from solving a polynomial equation that relates gradient and Hessian norms.  
This formulation can be interpreted as a *third-order extension* of damped or regularized Newton methods, leading to improved global convergence rates up to **O(k⁻³)** without requiring explicit Hölder constants.  
The paper also introduces adaptive variants (GRLS and UN) that use backtracking or linesearch strategies to estimate stepsizes in practice. Experiments on convex and mildly nonconvex problems support the theoretical findings.

### Strengths
- **Solid theoretical foundation:** The work extends and generalizes the ideas in AICN to broader Hölder-smooth settings.  
- **Interesting parametrization:** The introduction of the parameter \( \theta \) to implicitly define the stepsize \( \alpha \) is mathematically elegant.  
- **Adaptive schemes:** The GRLS and UN variants are practical contributions that make the theory more broadly applicable, removing dependence on unknown Hölder constants.

### Weaknesses
- **Dependence on prior work:** Much of the structure and analysis builds directly on **Hanzely et al. (2022)**, with limited new conceptual ideas beyond the general Hölder extension.  
- **Clarity issues:**  
  - The role of the **parametrization \($\theta$ \)** and how it determines the stepsize \( $\alpha$ \) is not clearly explained. This deserves more detailed discussion, since it is central to the contribution.  
  - The main problem is that $\theta$ depends on $\alpha$ but in the algorithm it does not.  
- **Adaptive method overhead:** The adaptive (backtracking) procedures may add nontrivial computational cost. The paper would benefit from an explicit discussion or empirical measurement of this overhead.  
- **Scope limitation:** The analysis and guarantees are developed specifically for **convex deterministic functions**, and it is unclear how (or whether) these results extend to **non-convex** or **stochastic** optimization settings.  
- **Experimental scope:** The experiments are relatively small and primarily illustrative. Additional comparisons with standard baselines such as L-BFGS or trust-region Newton methods would strengthen the empirical section.

### Questions
1. Could the authors clarify **how the dependence between \( \theta \) and \( \alpha \)** was handled in both analysis and implementation?  
   - Is \( \theta \) treated as an implicit function of \( \alpha \), or is it estimated independently?  
   - How sensitive is the method to errors in estimating or initializing \( \theta \)?  
2. The results are presented for **convex deterministic** objectives.  
   - How might the proposed analysis or algorithms extend to **non-convex** settings, where the Hessian might not be positive definite?  
   - Could the same framework apply to **stochastic** optimization, where gradients and Hessians are noisy?  
3. In adaptive variants (GRLS and UN), what is the **practical computational overhead** of the backtracking procedure, and how does it scale with problem dimension?  I believe you should be able to have a theoretical guarantee for the backtracking approach.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors study the damped Newton's method for convex functions whose Hessians or third derivatives are Hölder continuous. The key idea is to apply higher-order regularization to Newton's method in the local norm induced by the Hessian, which is shown to be equivalent to using a specific damped step size. Under the assumption of bounded Hessian variation, they establish a suplinear local rate and a global sublinear rate up to $O(k^{-3})$. Since the derived step size depends on the unknown Hölder continuity exponent and constant, they further propose both exact and backtracking line search schemes that achieve the same global rate without prior knowledge of these parameters.

### Strengths
The paper builds on Doikov et al. (2024), who proposed Newton's method with higher-order regularization in $\ell_2$-norm and proved an $O(k^{-3})$ rate when the third derivative is Lipschitz continuous. However, that approach sacrifices affine invariance due to the use of $\ell_2$-norm. To address this limitation, similar to Hanzely et al. (2022), the authors adopt regularization in the local norm, thereby restoring affine invariance while retaining the same accelerated rate under higher-order smoothness. The convergence analysis is thorough and includes line-search variants that eliminate the need for prior knowledge of problem parameters.

### Weaknesses
- My main concern lies in Assumption 1, which appears somewhat restrictive. 
   - The authors cite Hanzely et al. (2022) for this assumption. However, in my understanding, their global sublinear rate does not rely on such an assumption, and a similar condition is only used to establish local quadratic convergence near the solution. In contrast, both the global and local convergence results in this paper depend on this assumption.
  - The authors argued that this assumption is satisfied when the function is $L$-smooth and $\mu$-strongly convex, or when it has a stable Hessian. Yet, in both cases, one would typically expect a faster global linear convergence rate, rather than the slower sublinear rate obtained here. 
  - The authors also noted that this assumption holds for self-concordant functions when iterates are close to each other or in the neighborhood of the solution. However, it remains unclear whether this condition is applicable for global analysis, since the iterates may move far in a single step. 
- I think the prior work by Doikov et al. (2024) is highly relevant and deserves a more detailed discussion in the main text. Specifically, their regularized Newton method achieves a comparable rate of $O(k^{q-1})$ under (the more standard) smoothness assumption in the $\ell_2$-norm. It will also be helpful if the authors can elaborate on how their proof techniques differ from those in that work.

### Questions
I find the proofs in the appendix somewhat difficult to follow and suggest that the authors improve their presentation. For example, in the proof of Lemma 4 for the case $p=3$, it is unclear to me how the inequality in line 1527 is derived.  I assume Lemma 1 is used in this step, but additional clarification would be helpful.

### Soundness
3

### Presentation
2

### Contribution
2
