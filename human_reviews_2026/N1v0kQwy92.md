# CaCuTe: Casual Cubic-Model Technique for Faster Optimization

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
We establish a local $\mathcal{O}(k^{-2})$ rate for the gradient update $x^{k+1}=x^k-\nabla f(x^k)/\sqrt{H\|\nabla f(x^k)\|}$ under a $2H$-Hessian--Lipschitz assumption. Regime detection relies on Hessian--vector products, avoiding Hessian formation or factorization.
Incorporating this certificate into cubic-regularized Newton (CRN) and an accelerated variant enables per-iterate switching between the cubic and gradient steps while preserving CRN’s global guarantees. The technique achieves the lowest wall-clock time among compared baselines in our experiments.
In the first-order setting, the technique yields a monotone, adaptive, parameter-free method that inherits the local $\mathcal{O}(k^{-2})$ rate. Despite backtracking, the method shows superior wall-clock performance. Additionally, we cover smoothness relaxations beyond classical gradient--Lipschitzness, enabling tighter bounds, including global $\mathcal{O}(k^{-2})$ rates. 
Finally, we generalize the technique to the stochastic setting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considered unconstrained minimization problem $\min f(x)$ where $f$ is twice-differentiable and the Hession of $f$ is $2H$-Liptchitz. The paper studies first-order methods in the form of 
$$
x^{k+1} = x^k - \frac{1}{\sqrt{H\|\nabla f(x^k)\|}} \nabla f(x^k).
$$
Combining with cubic regularized Newton steps, the casual cubic Newton method and its acclerated version are proposed. With specific pamameter choices, the casual cubic Newton method recovered the results of the cubic regularized Newton methods. Then under assumptions on bounded Hession eigenvalue in the graident direction, the paper poposed the casual cubic adaptive gradient descent method and extend to directional $L_0, L_1$- smoothness cases. Finally, a variant of methods under stochastic regime is studied.

### Strengths
The paper showed a potential framework to analyze cubid regularized Newton methods and gradient methods under various smoothness regimes.

### Weaknesses
While the core idea presented in this manuscript is interesting and has potential, the current presentation significantly limits its impact. The contribution of the work is not clearly articulated, and the results are not framed with the necessary rigor to be fully convincing. As a result, the contribution of the work remains unclear; see questions below regrading writing, paper structure and the correctness.

### Questions
- $\mathbf I$ is not used in the line 34.


- In Line 59, '...in the convex' seems complete; 'enhances' --> 'is enhanced'?

- In line 69, 'However  ... can be avoided while keeping ... without... under...' this sentence is strange. 

- What are the convergence results of the adaptive gradient methods in the related work? 


- The sentences and ambiguous words in Section 1.3 are confusing. 

    - The first long sentence in Line 89-91 is difficult to understand. I suggest to separate to several sentences. What does the 'corresponds coincides aligns' in the parenthese mean? The terminology 'certificate' is frequently used but I am not sure what method/technique they refer to.  

    - In the AccCaCuN paragraph, what does the 'same argument' refer to?

    - What does 'a method is monotone' stand for?

- In section 3,  $O(k^{-2})$ convergence is claimed for Cases I (cubic-model decrease) and $O(k^{-1})$ convergence is claimed for Case II (quadratic-model decrease). Are these claims existing results? If so, proper references should be listed; otherwise, rigorous statements and proofs should be given. 

- Why $L$-smoothness is mentioned in Line 186? Is it related to the smoothness assumption in the previous sentence. 

- **There is no clear direction linked the proofs in the appendix to the results (theorem, claims, or examples) in the main paper, making it difficult to verify the correctness of the results.**


- Below are questions regarding the results in section 4: 
    
    - Why the inequality in Line 203 holds? It seems inequality (6) is used but $M_k = (3/4)H$ may not satisfied (5).

    - What is $y^0$ in Line 208?

    - What is the gradient step that yields the inequality Line 208? I suggest to write out the first step explicitly.

    -  Section 4 seems to obtain convergence results by relating Algorithm 1 to the results in [Nesterov & Polyak, 2006]. But I am not sure what the 'original proof' refers to and most importantly why Algorithm 1 achieves global $O(k^{-2})$ convergence. The choice of $M_k$ in (5) switches to $M_k = (3/4)H$ and in Algorithm 1 there is no $M_k$. If Algorithm 1 and the proofs are fully covered by the results in [Nesterov & Polyak, 2006]. What is the point of Section 3 since (5) may not hold? Even though the proof is not new, I suggest to write a formal theorem as this is the first main result. The proof can be postponed to appendix. 


   - What do the 'same technique' and 'the original converge guarantees' in Section 4.2 refer to? 
   

   - In Algorithm 2, '$M_k \leq H, i.e. M_K = H$' does not make sense.

- The vanishing directional curvature is used in Section 1 and the title of Section 5.1 but never formally introduced. Is it defined as (10)?

- The proof for Theorem B.4 is not self-contained.

- The condition in Theorem 6.3 is inconsistent for the method and the result. 

- Can any global convergence rate be derived using the result of Theorem 6.3? How does it compared to other SGD methods?

### Soundness
2

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
4

### Summary
This paper presents **CaCuTe**, a family of algorithms that leverage **cubic regularization ideas** while avoiding the heavy cost of full cubic‐model solves.  
Starting from the **Lipschitz–Hessian (2H–smooth)** assumption used in classical cubic‐regularized Newton (CRN) methods, the authors derive a **Hessian–vector–based condition** that certifies when a **simple gradient step** will achieve the same decrease as a cubic step.  
This leads to the **CaCuN algorithm**, which dynamically switches between a cheap gradient step and a standard CRN step depending on the local curvature.  
The approach extends to an **accelerated variant (AccCaCuN)** that retains the accelerated \(O(k^{-3})\) global rate, as well as a **first-order adaptive method (CaCuAdGD)** that uses Hessian–vector products and backtracking to adapt \(M_k\) without explicit Hessians.  
A **stochastic version (CaCuSGD)** further generalizes the idea to noisy gradients.  
Overall, the paper provides a unified view of cubic-model techniques that reduce computational cost while maintaining theoretical guarantees.

### Strengths
- **Clear motivation:** Classical CRN guarantees are strong but expensive; CaCuTe identifies when cheaper first-order steps suffice.  
- **Theoretical soundness:** The paper rigorously derives one-step decrease bounds, global \(O(k^{-2})\) and accelerated \(O(k^{-3})\) rates, and expected stochastic analogues.  
- **Low overhead:** The curvature certificate requires only **one Hessian–vector product**, a modest cost compared to full second-order solves.  
- **Comprehensive framework:** Covers deterministic, accelerated, adaptive, and stochastic regimes under a single analytic template.  
- **Practical relevance:** Algorithms can exploit automatic-differentiation HVPs available in modern ML libraries.

### Weaknesses
- **Complex exposition:** The notation is dense and the main ideas are sometimes buried in algebraic detail; a higher-level intuition for why the curvature test works would help readers.  
- **Dependence on convexity:** The theoretical results assume convex objectives; it is unclear how the approach behaves on **non-convex** losses where Hessians may be indefinite.  
- **Parameter sensitivity:** The constants used in the curvature condition (e.g., the choice of \(H\) and \(\alpha\)) could affect practical performance, but this dependence is not discussed.    
- **Weak CaCuSGD results:** The empirical performance of **CaCuSGD** appears weak compared to standard **SGD**. It would be interesting to see whether adding **momentum or variance reduction** could mitigate stochastic noise.  
- **Reliance on differing assumptions:** The analysis invokes different assumptions (e.g., Eq. (10), Eq. (14)), making it difficult to compare results across variants. A **summary table** of assumptions and corresponding guarantees would greatly clarify the contributions.  
- **Quantification missing:** The paper does not quantify **how often the full cubic Newton step is avoided** in practice. Reporting this ratio would concretely demonstrate the computational advantage claimed.

### Questions
1. Could the same idea extend to **non-convex** settings?  
2. In practice, how often does CaCuN actually switch to the cubic step versus taking the cheaper gradient step?  
3. What is the **computational overhead** of the backtracking procedure in CaCuAdGD relative to standard gradient descent?

### Soundness
3

### Presentation
2

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
**Summary**

This paper establishes the convergence rate of a special stepsize rule for gradient descent under different settings. The stepsize rule is motivated by the optimality condition of the subproblem arising from the cubic regularized Newton's method. Some experiments validate the efficiency of the proposed method.

### Strengths
**Strength**

The paper is overall easy to follow.

### Weaknesses
**Weaknesses**

I have several concerns regarding both the theoretical and practical aspects of the paper.

1. Theoretical complexity

   In terms of the theoretical complexity, while the paper claims $O(1/K^2)$ result is achievable using first-order information, the analysis ends up relying on strong assumptions such as equation (10). In general, the first-order methods developed in the paper only guarantee an $O(1/K)$ convergence rate. I find the result relatively weak since it uses extra operations such as the Hessian vector product. Besides, some results (**Corollary 5.7**) in the paper contain strong assumptions on the trajectory, which I find unacceptable; some results (**Theorem 6.3**) are incomplete and do not give a convergence rate.

2. Practical implementation 

   The proposed algorithm is claimed to be parameter-free. However, in most cases, it still requires knowledge of the Hessian Lipschitz constant $H$. Even the adaptive variant still requires a line-search procedure and introduces an additional hyperparameter $\alpha$. Given that the algorithm essentially adopts the gradient direction, I don't think the algorithm design is justified.

3. Weak experiment evaluation

   The experiments in **Section 7**mseem cherry-picked. The results from the appendix suggest the proposed stepsize typically underperforms standard accelerated gradient descent.

Finally, I noticed a number of typos and notation inconsistencies throughout the paper. Overall, I don't think the paper can be published at ICLR in its current form.

### Questions
**Questions**

1. I don't feel I understand what "casual" in the title means. Does it come from the motivation on line 41?
2. Except for (AccCaCuN), the algorithms developed in the paper search along the gradient direction. Is there any intuition that the resulting algorithm can outperform vanilla gradient descent?

**Minor issues**

1. Line 11

   You mention "local" rate here, but the contributions claim global convergence.

2. Line 93

   "i.e. logistic regression" is unclear.

3. Line 130

   "is convex with, i.e." is unclear.

4. Line 132

   I don't think continuity is necessary here.

5. Line 160

   "Inequality (5) employs..." is unclear.

6. Line 186, 254, 284

   Gradient smoothness corresponds to Lipschitz Hessian; Hessian $\nabla^2 f$  corresponds to the Lipschitzness of the third-order derivative.

7. Line 356

   The assumption is not correct

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a ‘casual cubic Newton’ framework, where the algorithm decides whether to use second-order updates or cheaper normalized gradient updates based on a Hessian-vector-product (HVP) certificate, which allows algorithms inspired by the framework (CaCuN and AccCaCun) to enjoy similar convergence rates with cubic-regularized Newton, yet without forming or factorizing Hessians as often. Moreover, under a specific assumption, it is possible to construct an entirely HVP-based algorithm (CaCuAdGD), where the algorithm adapts to the local geometry via the HVP in the ‘certificate inequality’ and the backtracking of the constant $H$.

### Strengths
- The writing is clear and easy to read overall.
- The paper includes numerical experiments that support the theoretical results.

### Weaknesses
See **Questions.**

### Questions
- For CaCuN (and AccCaCuN) in Section 4, is it possible to quantify how much CRN steps we might expect to need? I think this should be a bit clearer that the algorithm will stay at the regime (with constants simplified to $c'$),
$$
f \left( y^k - \frac{\nabla f(y^k)}{\sqrt{c' \cdot H \| \nabla f(y^k) \|}} \right) \le f(y^k) - \frac{1}{c' \cdot \sqrt{H}} \| \nabla f(y^k) \|^{3/2}
$$
at least quite often to conclude that using CRN steps only outside this regime will really be computationally beneficial than vanilla methods and/or comparable with other efficient second-order methods like lazy Hessians (Doikov et al., 2023). Is it possible for the authors to quantify the Hessian (and gradient) oracle costs, at least on a high-level? Or could there be some specific cases where we can clearly expect fewer CRN steps?
    - In particular, when taking iterations other than CRN steps (i.e., excluding the Hessian computation steps), my intuition is that methods like lazy Hessians, which both preserves the Hessian structure itself but uses matrix-vector products for most steps, might be better than using a first-order method based on a upper-bound-ish scalar $H$, which is typically a bit more conservative. I didn’t check this in detail, could the authors elaborate on this?
    - I might be wrong, but isn’t this ‘HVP-based certificate’ is actually fully first-order computable if we take $H$ as constants, not matrices (which could actually be slightly better than Hessian-vector products)? This is a minor thing, but the term is a bit confusing to me as I can only see HVPs in Algorithm 3 (CaCuAdGD).
- For CaCuAdGD in Section 5, while it is nice that we can enjoy convergence only with first-order oracles and Hessian-vector products, it feels like that the assumption in $(10)$ is essentially just tailored to make the above regime to be true. As there are a few examples (cubic and logistic functions) mentioned that fall into this category, is there a better reason to consider this assumption other than merely a necessary condition for similar ideas as in the previous section to work? (This could also possibly be related to the first question, as we might be able to assert that CaCuN type algorithms will require very few Hessian oracles for the cubic and logistic functions.)

Doikov et al., 2023. Second-order optimization with lazy Hessians.

### Soundness
3

### Presentation
3

### Contribution
2
