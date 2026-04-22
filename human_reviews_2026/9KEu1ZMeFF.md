# A Finite-Time Analysis of TD Learning with Linear Function Approximation without Projections or Strong Convexity

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 10, 0

## Abstract
We investigate the finite-time convergence properties of Temporal Difference (TD) learning with linear function approximation, a cornerstone algorithm in the field of reinforcement learning.
  We are interested in the so-called ``robust'' setting, where the convergence guarantee does not depend on the minimal curvature of the potential function.
  While prior work has established convergence guarantees in this setting, these results typically rely on the assumption that each iterate is projected onto a bounded set, a condition that is both artificial and does not match the current practice.
  In this paper, we challenge the necessity of such an assumption and present a refined analysis of TD learning. For the first time, we show that the simple projection-free variant converges with a rate of $\widetilde{\mathcal{O}}(\frac{||\theta^*||^2_2}{\sqrt{T}})$, even in the presence of Markovian noise. Our analysis reveals a novel self-bounding property of the TD updates and exploits it to guarantee bounded iterates.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper analyzes TD(0) with linear function approximation, proposing a "self-bounding" mechanism with stepsize $\eta_t = \frac{1}{c\phi_\infty^2 \log T \log(t+3)\sqrt{t+1}}$ (where $c \geq 281$) to achieve $\tilde{O}(\|\theta^*\|_2^2/\sqrt{T})$ convergence without explicit projection or dependence on the strong convexity modulus.


**Major concern:** Under Assumption 2 ($\Phi$ full column rank), the potential function $f(\theta) = (1-\gamma)\left\| V_\theta - V_{\theta^*}\right\|_D^2 + \gamma\left\|V_\theta - V_{\theta^*}\right\|_{Dir}^2$ is strongly convex. The Hessian satisfies:

$$\nabla^2 f(\theta) = 2\Phi^\top[(1-\gamma)D + \gamma L]\Phi \succeq 2(1-\gamma)\Phi^\top D \Phi \succ 0$$

This gives strong convexity modulus $\omega \geq 2(1-\gamma)\lambda_{\min}(\Phi^\top D \Phi) > 0$. Standard methods for strongly convex functions achieve $O(1/T)$ rates, asymptotically faster than the $\tilde{O}(1/\sqrt{T})$ rate obtained here. The paper provides no justification for avoiding this property.

**Technical issues:**
- **Lemma B.4:** The term $Z'_k$ is undefined, and the bound $8\ell_{k-k'}C\alpha^k$ lacks rigorous justification
- **Theorem 4.2(b):** Inconsistency in the average iterate definition vs. how Jensen's inequality is applied
- **Appendix D:** Key inductive proof too condensed for verification

Generally well-organized, but the distinction between "without strong convexity" and "without explicit $\omega$-dependence" needs clarification. The "universal" step size claim (Table 1) overstates advantages since $\eta_t$ explicitly depends on $T$ via $\log T$ and implicitly on mixing properties.

The paper does not explain when $\tilde{O}(\|\theta^*\|_2^2/\sqrt{T})$ would be preferred over $O(\kappa/T)$ bounds from strongly convex methods (where $\kappa \sim 1/\omega$). For any fixed $\kappa$, the $O(1/T)$ rate eventually dominates. The motivation for avoiding strong convexity when the function possesses this property is absent.

### Strengths
1. Novel self-bounding analytical approach
2. Experimental validation of the theoretical threshold $c \geq 281$

### Weaknesses
1. **Fundamental premise flawed:** The function is strongly convex under stated assumptions, yet the paper deliberately avoids exploiting this to achieve a slower convergence rate
2. **No comparative analysis:** Missing quantitative comparison showing when this approach excels over strongly convex methods
3. **Misleading claims:** "Universal" stepsize still depends on $T$ and mixing properties; "without strong convexity" is inaccurate
4. **Technical gaps:** Undefined terms, unjustified bounds, and condensed proofs undermine confidence
5. **Practical limitations:** Requires $c \geq 281$ (very small stepsizes) and advance knowledge of $T$

**Core issue:** The paper's premise is fundamentally flawed. Under stated assumptions, the objective is strongly convex with positive modulus $\omega = 2(1-\gamma)\lambda_{\min}(\Phi^\top D \Phi)$. The authors develop an elaborate analysis avoiding this property to obtain $\tilde{O}(1/\sqrt{T})$ convergence, which is asymptotically worse than the $O(1/T)$ rates standard methods achieve for strongly convex functions. No justification or regime analysis is provided for when this approach would be preferable.

**Additional concerns:** Technical issues (undefined terms, unjustified bounds, condensed proofs), misleading characterizations ("universal" stepsize, "without strong convexity"), and practical limitations ($c \geq 281$, requiring advance knowledge of $T$) further undermine the contribution.

This requires complete reconceptualization: either explicitly acknowledge strong convexity and justify the approach, or demonstrate specific regimes where avoiding $\omega$-dependence provides meaningful advantages. The current framing suggests solving the wrong problem with the wrong tools.

### Questions
1. Under Assumption 2, is $f(\theta)$ strongly convex with $\omega = 2(1-\gamma)\lambda_{\min}(\Phi^\top D \Phi) > 0$? If so, why not exploit this?
2. When does your bound outperform $O(\kappa/T)$ achievable with strongly convex analysis?
3. In what sense is your stepsize "universal" given its explicit dependence on $T$ via $\log T$?
4. Can you provide complete derivations for Lemma B.4 (defining $Z'_k$, justifying bounds) and clarify Theorem 4.2(b)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission presents a finite-time convergence analysis for projection-free TD(0) learning algorithms with linear function approximation under Markovian observations. The authors prove a robust O(1/\sqrt{T}) rate independent of the curvature of the feature covariance matrix, eliminating the need for projection steps required in prior work to achieve this rate. The key idea is a novel self-bounding property that shows TD iterates remain bounded in expectation without explicit projections. The theoretical claims are validated by synthetic experiments.

### Strengths
+ improved convergence rate results for TD(0) algorithms with linear function approximation under Markovian noise
+ Interesting finding and use of the self-bounding property of TD(0) updates
+ Numerical validations of the theory results

### Weaknesses
- Alg. 1 may not seem practical since the stepsize \eta_t requires prior knowledge of the horizon T, the feature bound \phi_inf, and a very large constant c>=281. 
- Missing formal statement of the theorem 4.2 in the main paper. I do not believe it is right to have only an informal result in the main paper that will be publishd only while having the formal result in the supplementary material. 
- The paper compares its theoretical rate to prior results, but the experiments lack a crucial baseline: a projected TD method where the projection radius is set optimally \|\theta*\|. That will make clear whether the new analysis yield any advantafe over just using projected TD with a carefully chosen R?
- The paper points that the robust rate is preferrable when the condition number is bad. However, in many practical problems with favorable feature representations, the fast \tilde{O}(1/T) rate is more relevant. Anyways, one has the freedom to chose the feature functions. 
- Mising related references on finite-time analysis of TD(0) learning algorithms.

### Questions
- The constant c>281 is remarkly large. Is this a fundamental requirement for projection-free convergence, or is it an artifact of the proof? Can you provide any intuition for why such a large constant is needed, and whether it might be improved? In Appendix D, Theorem D.1, there seems a mistake "Note that \rho_c is decreasing in c, \rho_c\to..." to what? moreover, it is not obvious why \rho_c is decreasing in c over the entire horizon. The proof requires c>\frac{279+3\sqrt{8837}}{2} for the bound to hold. Since the constant on the right >281, I am not sure whether the bound will hold by relaxing the requirement c>\frac{279+3\sqrt{8837}}{2} to c>281?
- Theanalysis shows boundedness in expectation. Can you comment on the possibility of proving almost-sure boundedness or a high-probability bound? 
- The paper mentions extension to Q-learning and TD(\lambda) as straightforward. However, these algorithms have non-linearities (e.g., the max operator in Q-learning). It is not clear how the self-bounding technique will tackle these issues?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper investigates temporal-difference (TD) learning with linear function approximation without relying on projections or strong convexity. It removes a strong yet widely used assumption and, overall, makes a substantial contribution to the analysis of TD methods.

### Strengths
1. The authors establish non-asymptotic bounds for TD(0) with linear function approximation under Markov noise, matching practical projection-free  implementations.

2. The analysis shows the iterates remain bounded in expectation without projections or strong convexity, by leveraging a self-bounding structure in the recursion.

3. With a generic stepsize schedule, the method attains $ \tilde{O}\left(\frac{\\|\theta^\star\\|^2}{\sqrt{T}}\right)$,
  requiring no prior spectral/curvature information.

4. Using $f(\theta)= (1-\gamma)\left\lVert V_\theta - V_{\theta^\star}\right\rVert_{D}^{2}+ \gamma \left\lVert V_\theta - V_{\theta^\star}\right\rVert_{\mathrm{Dir}}^{2}$, which remains valid at $\gamma=1$ and aligns with the mean-path gradient, simplifies and strengthens the proof technique.

 5. The results remove reliance on projections/strong convexity and are supported by synthetic experiments that verify the existence of a stepsize threshold consistent with the theory.

### Weaknesses
The authors should provide intuition for their techniques—how they were derived and how they work in the proofs.

### Questions
1.  The authors should provide intuition for their techniques—how they were derived and how they work in the proofs.

2. Can your results be extended to adaptive settings (e.g., adaptive stepsizes) as in [1,2]?  

3. Some references on  TD  [1,2] appear to be missing;.

[1] Adaptive temporal difference learning with linear function approximation.
[2] Finite-Time Analysis of Adaptive Temporal Difference Learning with Deep Neural Networks.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Projection-free finite-time analysis of TD(0) with linear features under Markovian sampling. Uses a Liu–Olshevsky–style potential and a horizon-aware stepsize to prove self-bounded iterates and a robust $\tilde{\mathcal{O}}(1/\sqrt{T})$ rate; small synthetic experiments illustrate a stepsize threshold.

### Strengths
- First robust, **projection-free** guarantee for linear TD(0).
- Handles Markovian bias via mixing; clear comparison to prior work.

### Weaknesses
- **The paper clearly violates the ICLR format. Missing page numbers and the whole text has been shifted down and therefore should be desk rejected.**
- Experiments are narrow; few task details in main text.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
