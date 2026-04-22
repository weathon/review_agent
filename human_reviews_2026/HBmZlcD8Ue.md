# Beyond Short Steps in Frank-Wolfe Algorithms

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
We introduce novel techniques to enhance Frank-Wolfe algorithms by leveraging function smoothness beyond traditional short steps. Our study focuses on Frank-Wolfe algorithms with step sizes that incorporate primal-dual guarantees, offering practical stopping criteria. We present a new Frank-Wolfe algorithm utilizing an optimistic framework and provide a primal-dual convergence proof. Additionally, we propose a generalized short-step strategy aimed at optimizing a computable primal-dual gap. Interestingly, this new generalized short-step strategy is also applicable to gradient descent algorithms beyond Frank-Wolfe methods. Empirical results demonstrate that our optimistic algorithm outperforms existing methods, highlighting its practical advantages.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper revisits the Frank–Wolfe (FW) family of algorithms and develops two main innovations: (i) optimistic Frank–Wolfe (OFW): A new FW variant that incorporates optimism from online learning, predicting future gradients to adaptively refine updates. The algorithm enjoys a primal–dual convergence analysis and retains the rate under convex smooth settings. (ii) primal–dual short steps: A generalization of the classical “short-step” rule that maximizes primal–dual progress instead of just primal progress. This approach yields an interpretable stopping criterion based on a computable primal–dual gap and extends naturally to gradient descent.

The paper provides rigorous analyses, unifying online learning and Frank–Wolfe perspectives via the optimistic mirror descent/FTRL framework, and demonstrates the practical advantages of OFW through experiments on convex quadratic problems and polytopal constraints.

### Strengths
1. The incorporation of optimism into Frank–Wolfe algorithms is both original and theoretically well-motivated. It bridges FW with the online-to-batch conversion framework (Cutkosky, 2019), enriching the algorithmic toolbox for projection-free methods.

2. The proposed analysis elegantly derives both convergence guarantees and computable stopping criteria from a primal–dual perspective. The connection between FW, heavy-ball FW, and gradient descent is clarified through shared gap-based arguments.

3. The theoretical results are sound and internally consistent. The paper’s exposition (especially Section 3-4) carefully motivates each step-size rule and provides transparent algorithmic derivations.

4. Experiments on probability simplices and $k$-sparse polytopes convincingly show that the optimistic variant converges faster (both in iteration and wall-clock time) than standard and heavy-ball FW variants. The results are reproducible and coded in Julia.

### Weaknesses
1. The experiments are small-scale, focusing on synthetic convex problems. More realistic applications (e.g., structured prediction, optimal transport, or matrix completion) would better support the method’s practical relevance.

2. Despite being theoretically elegant, the primal–dual short-step variant performs comparably or slightly worse than vanilla FW with line search, suggesting limited practical value.

3. The optimistic connection could be explained more clearly and the connection to predictive gradient methods could be emphasized further.

4. The paper is dense with notations, which can make it difficult for non-specialists to follow. A clearer summary table or pseudocode for each variant would help.

### Questions
1. Can the optimism mechanism extend to nonconvex or stochastic FW settings?
2. How does the optimistic FW relate to momentum-based FW (e.g., Heavy-Ball FW) beyond shared lower bounds?
3. Could the primal–dual gap be efficiently estimated without full LMO recomputation at each iteration?
4. Do you foresee any extension to affine-invariant step-size strategies as in Pena (2023)?

### Soundness
4

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
This paper introduces new techniques for improving Frank-Wolfe (FW) algorithms by incorporating function smoothness beyond traditional short steps. The authors propose two main innovations. First it proposed the Optimistic Frank-Wolfe Algorithm (OFW), a novel method inspired by optimistic online learning, leveraging predicted gradients to improve convergence in smooth convex optimization. It adapts to changing curvature and achieves a provable 1/t^2 like primal-dual convergence rate. Second it introduced the Primal-Dual Short Steps (PDSS), a new class of step-size rules that optimize the primal-dual gap directly, leading to theoretically justified and generalizable improvements. The same principle extends to gradient descent. Experiments over standard convex benchmarks (simplex, k-sparse polytopes, portfolio optimization) show that the optimistic variant consistently outperforms classical and heavy-ball FW methods, both in iteration count and runtime.

### Strengths
This submission has conceptual innovation. The introduction of optimism into the FW setting is novel and well-motivated through online learning theory (OMD/FTRL frameworks). The primal-dual short-step idea elegantly unifies step-size selection with primal-dual analysis, offering tighter convergence and computable stopping criteria. The authors presented the rigorous theoretical framework. The analysis is carefully constructed through primal-dual gap bounds, clearly improving on classical FW and heavy-ball variants. Results from Theorems and Propositions provide clean, interpretable convergence bounds with constant factors comparable to or better than prior work. 

The authors also conduct the numerical experiments using standard benchmarks implemented in Frank Wolfe, ensuring reproducibility. The optimistic algorithm shows steeper log–log slopes (faster empirical order of convergence) and strong wall-clock performance improvements. The empirical results from the numerical experiments are consistent with the theoretical analysis.

### Weaknesses
There are no major weakness for this submission.

On minor weakness if that this submission has limited experimental scopes. Only convex smooth problems are tested; no constrained stochastic or non-convex scenarios. The proposed primal-dual short steps did not outperform standard heuristics in practice — acknowledged but not deeply analyzed. It is suggested to add more numerical experiments with more experimental scopes like including non-smooth and stochastic benchmarks to demonstrate generality and comparing against projection-free adaptive gradient methods or away-step FW variants to position contributions more competitively.

### Questions
There are no other questions.

### Soundness
3

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
The paper proposes an optimistic Frank-Wolfe algorithm that leverages optimistic online learning methods (FTRL, OMD). In particular, one can substitute the LMO oracle part into optimistic FTRL/OMD on linear losses to construct the proposed OFW. The OFW analysis establishes a classical $O(1/t)$ rate of convergence for the primal-dual gap via online-to-batch conversion and shows better performance in empirical studies, especially in terms of the dual gap. The paper also suggests a primal-dual short step schedule designed from a descent-lemma-like inequality tailored for the Frank-Wolfe (primal-dual) gap. This better suits FW type algorithms than classical step sizes that come from ‘primal’ descent-lemmas (i.e., smoothness) and provides a stopping criterion computable on the fly.

### Strengths
- The writing is clear and easy to read overall. The ideas of each sections are well presented through selected key inequalities from the proof, which makes a much better reading experience.
- The paper includes numerical experiments that support the theoretical results.

### Weaknesses
See **Questions.**

### Questions
- The second term in the proposed upper bound for optimistic FW (in Theorem 3.1) is $\frac{4LD^2}{t+1}$ which looks identical to that of vanilla FW, but in the experiments optimistic FW clearly looks faster. Could there be a way to theoretically show a strict improvement over FW/HB-FW in terms of either faster convergence under certain settings (sparse problems, etc) or guaranteed convergence in a broader setting with weaker assumptions or maybe something else? (In simple online learning settings, FTRL and OMD can have constant improvements in the regret upper bound compared to vanilla online GD in certain cases, e.g., the function $f$ has sparse gradients, if I am correct.)
- Is it a bad idea to choose $L_2$ or $L_1$ (or maybe combined) norm regularizers for optimistic FW, instead of the polytope indicator? I think this might sometimes better capture the benefits of using FTRL/OMD type updates, while using an indicator as the regularizer seems to make less distinction (though not exactly equivalent) with vanilla FW. What can you expect if we use different regularizers, or is there a reason why the paper does not consider these?
- What exactly are the benefits we get from using primal-dual step sizes? What I can see is that we can drop the constant $4$ from the upper bound of $G_t$ if we use GD with primal-dual step sizes (or line search), but is this the best we can get from this? Also, is there a way to find a ‘primal-dual-optimized’ $a_t$ for the optimistic FW versions? While merging the two ideas might be irrelevant or unbeneficial, I still think there should have been bits of explanation about why this is the case, and if these are two completely separate ideas, it seems unclear how primal-dual step sizes can be useful.

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
This paper studies an optimistic variant of the Frank-Wolfe method and establishes a worst-case convergence guarantee in terms of the primal dual gap $G_t = f(x_t) - L_t$, where $L_t$ is a computable lower bound on the optimal value $f(x_*)$. The analysis focuses on the primal-dual gap because it can be evaluated and used as a stopping criterion, unlike the primal gap $f(x_t) - f(x_*)$. In particular, the authors propose Algorithm 2, which uses the short step size in (5) to ensure a monotonic decrease of the primal-dual gap. The paper further demonstrates the generality of the approach by showing that it also applies to gradient descent.

### Strengths
Regardless of the weaknesses mentioned below, I believe this paper, by proposing a new optimistic variant of the Frank-Wolfe method with a convergence guarantee on the computable measure, has its own merit, and warrants further investigation in this direction.

### Weaknesses
- I agree that the primal-dual gap can serve as a practical stopping criterion, but I don't follow the authors' claim that this justifies the need for a method with a guaranteed convergence rate for that gap. First, the primal-dual gap is not a tight bound on the primal gap $f(x_t) - f(x_*)$, and it can be computed regardless of whether we have a theoretical guarantee on its decrease. Moreover, the paper does not show that existing methods do not efficiently decrease the primal-dual gap (in theory).

- I also find the repeated claim that the proposed optimistic approach better adapts to the environment unconvincing. I would appreciate a more precise and unambiguous explanation of what type of adaptivity is meant here.

- For the above reasons, I am not fully convinced by the motivation or the importance of the contribution.

### Questions
- Line 43: What do you mean by "corresponding convergence rate need to be heuristically estimated"?
- Line 46: What do you mean by "enhancing the convergence properties"? I don't see any results that supports this claim.
- Line 65: Could you justify how your method adapt effectively to varying conditions and provides a robust analysis?
- Line 78: As written, it seems to suggest that your method improves the theoretical constant, although you refer Appendix E. I recommend explicitly stating that this improvement is observed empirically.
- Lines 230-234 with line 7 in Algorithm 2: This part is particularly difficult to follow.
- Figure 1: Why does the optimistic method much more efficient per iteration compared to the plain one, given that Algorithm 2 is more complex? I also recommend plotting the primal gap $f(x_t) - f(x_*)$, rather than $f(x_t)$.

### Soundness
3

### Presentation
2

### Contribution
2
