# Distributionally Robust Bayesian Optimization: From Single to Multiple Objectives

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
In many real-world applications, systems are typically expensive to evaluate and influenced by contextual variables whose distributions may shift between training and deployment. While robust Bayesian optimization methods have been proposed for black-box functions under such conditions, most of them focus solely on single-objective settings. In practice, however, systems often need to be optimized across multiple criteria simultaneously, which is challenging since the same environment may affect different objectives in distinct ways. Although robustness against the contextual uncertainty has been investigated for single-objective problems, its extension to multi-objective optimization (MOO) problems remains limited, with existing works primarily addressing only input noise—a special case of the contextual uncertainty. To bridge this gap, in this work,  we propose the first Multi-objective Bayesian Optimization (MOBO) method for the general $\varphi$-divergence Distributionally Robust Optimization (DRO) problem with shared contexts, aiming to obtain *robust efficient*  solutions. Furthermore, a provable regret bound is provided, which is the first sublinear regret bound without requiring a decreasing radius of the DRO uncertainty set, even in comparison to existing works in the single-objective setting. Moreover, we provide numerical experiments to validate our theory and the empirical effectiveness of our proposed algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a distributionally robust multi-objective Bayesian optimization (BO). The authors claim that most of studies distributionally robust BO is single objective except for only a few studies. Based on robust efficiency, the authors develop a scalarization and upper confidence based acquisition function, for which a dual-problem based formulation is derived. Further, the regret bound is derived based on weaker assumptions than existing studies.

### Strengths
The topic (distributionally robust MOBO) has not been widely studied, but I think it is potentially quite important topic. Overall, the paper is well-written and well-organized. Although I'm not familiar with this specific topic, the paper provides a comprehensible introduction of the background, motivation, and existing studies, by which the novelty of the paper is quite clear (even in ICLR submissions, there are surprisingly few papers that properly achieve this). 

The theoretical analysis is seemingly reasonable and convincing (though I couldn't fully follow the entire proof).

### Weaknesses
A limitation by the linear scalarization is not clearly discussed.

The experiments are somewhat shallow and lack a baseline. I think that existing robust MOBO, even usual MOBO, and the random selection can be a baseline to demonstrate effectiveness of the proposed method. The goals of some of those methods  may be different from the proposed method, but for example, by comparing with usual MOBO methods, significance for considering a taylored approach for the target problem setting (distributionally robust MOBO) can be clarified. Without a baseline, efficiency is difficult to evaluate.

### Questions
The authors mention that the solution of linear scalarization is robust efficient. On the other hand, can any robust efficient solution be found by the optimizer of the linear scalarization (which is impossible in the case of the Pareto-frontier in usual multi-objective problems, i.e., so-called concave part of Pareto-frontier cannot be identified by linear scalarization)? 

If the answer to the above question is no, can it be resolved by changing the scalarization function? Further, in that case, the regret analysis can also be extended to those other scalarizations?

Inatsu et al. 2024 also provide a theoretical analysis by using a distance based criterion between Pareto-frontiers, but the relation with the proposed analysis is not fully clear to me. In Table 1, the authors compare existing studies based on 'sublinear regret', but as mentioned above, I am not fully sure if the current definition of the 'regret' can capture the true efficiency about the identification of all the efficient solutions. Based on this consideration, when looking at Table 1, I wonder whether it is fair to regard the regret evaluation, which might not be perfect (may overlook a concave front), as an advantage, while not mentioning theoretical analyses using (perhaps) more fundamental metrics (directly comparing distance among Pareto-frontiers instead of comparing scalarized quantities). However, I do not have a deep understanding of this matter. Could you elaborate on this perspective?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers a distributionally robust multi-objective optimization problem, where the objective functions are expensive to evaluate and the ambiguity set (a set of context distributions) is defined by $\varphi$ divergence. For this problem, this paper proposes a UCB-type acquisition function, develops its efficient computation approach, and derives regret upper bounds in the Bayesian setting in which multiple objective functions follow Gaussian processes. Finally, the authors provide the simple benchmark results.

### Strengths
This paper is clearly well-written and easy for me to follow.
This paper appears to employ solid definitions and techniques from the distributionally robust (white-box) multi-objective optimization, distributionally robust Bayesian optimization, and multi-objective Bayesian optimization literatures.
The theoretical results also appear rigorous.

### Weaknesses
On the other hand, I simultaneously feel that there are several severe limitations listed below:
- The experimental results are weak. Since there is no baseline method, we cannot verify the effectiveness of the proposed method. Even if there is no existing method tailored for this problem setup, I would strongly encourage comparing with some vanilla baselines. On the other hand, I conjecture that the method in [Inatsu et al, 2024, Daulton et al., 2022] can be compared in some setting of $\varphi$.

- Although this paper is based on the definition of regret from [Paria et al., 2020], its definition slightly changes. Although Paria et al. (2020) take the expectation with respect to $s_t$, this paper does not take it as in Eq. (11). The original definition guarantees the proximity of the recommended solutions and Pareto optimal solutions by showing the upper bound of the expectation where $s_t$ varies in some distribution (typically a uniform distribution). Therefore, this change of the definition affects the interpretation of the theoretical result. I think that the definition in Eq. (11) is not suitable for multi-objective optimization since it can approach zero even if $s_t$ is always the same vector.

- This paper claims that it relaxes the condition on the radius of the ambiguity set. However, instead of that, this paper requires the condition that $P\_t(c) > 0$ for all $c$ such that $Q(c)$ can be positive. Since this is the additional assumption from the existing studies, it should be more explicitly discussed.

- The final regret upper bound is $O(K)$. However, I conjecture that it can be tightened as $O(\sqrt{K})$ by applying Cauchy--Schwarts inequality to $\sum\_{t=1}^T s_t \sigma\_{t-1}(x\_t, c) \leq \sqrt{ \sum\_{t=1}^T \| s_t \|_2^2 \sum\_{t=1}^T \sum\_{i=1}^K  \sigma^{i2}\_{t-1}(x\_t, c)} \leq \sqrt{ T \sum\_{t=1}^T \sum\_{i=1}^K  \sigma^{i2}\_{t-1}(x\_t, c)} $.

- Although Table 1 shows that Inatsu et al. (2024) is N/A regarding the sublinear regret,  Inatsu et al. (2024) provided Theorems 4.1 and 4.2, which claim that the algorithm must achieve an $\epsilon$-accurate solution within finite iterations. This is essentially the same result as the sublinear regret upper bound.

Overall, I feel that the lack of experimental comparison is the most important weakness.

### Questions
Please answer the above comments.
In addition, does the theorem imply that the algorithm can enumerate the robust efficient input points?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a Distributionally Robust Multi-Objective Bayesian Optimization framework that extends robust Bayesian optimization to settings with multiple objectives and shared contextual uncertainty. It formulates robustness via a general $\omega$-divergence-based distributionally robust optimization model, encompassing well-known divergences such as KL, TV, and $\varepsilon^2$. The authors develop a random-scalarization algorithm that produces robust efficient (Pareto-optimal) solutions and establish a sublinear regret bound for this setting, without requiring the uncertainty-set radius to shrink over time. Theoretical guarantees are complemented by numerical experiments on synthetic benchmark functions showing consistent sublinear regret.

### Strengths
**Strengths:**

1) Multi-objective optimization problems appear in many real-world applications, yet existing distributionally robust Bayesian optimization methods focus almost exclusively on single-objective settings. This paper addresses this important gap by proposing a principled framework for robust multi-objective optimization. 

2) The positioning of the paper within the literature is clear and well justified. In particular, Table 1 provides a concise and informative comparison with related works, helping the reader to understand the novelty and scope of the proposed approach. 

3) The theoretical development is complemented by numerical experiments that effectively visualize the main results and confirm the sublinear regret behavior predicted by the analysis.

4) The framework is formulated for a general class of $\omega$-divergence DRO models, making it broadly applicable to various notions of robustness (e.g., TV, KL, $\varepsilon^2$, and Cressie--Read divergences).

### Weaknesses
**Weaknesses:**

1) The assumption of finitely supported context variables is highly restrictive. It remains unclear how the proposed approach scales to continuous or high-dimensional context spaces, which are the realistic cases in contextual Bayesian optimization.

2) The first six pages mainly summarize existing background on Gaussian processes and distributionally robust optimization. The main contributions appear only from Section 4 onward, and it is not clear what is novel in Section 3, as most of the material seems to restate known results.

3) Assumption 2 is difficult to interpret and practically verify. The paper should clarify under what conditions this assumption holds and how it could be checked or enforced in applications.

4) Theorem 1 is hard to follow. Important quantities such as the set $\mathcal{X}_t$ and the constant $d$ are not properly introduced, which makes it challenging to understand the result and its implications.

5) Algorithm 1 requires solving the optimization problem (10) in each iteration. The authors claim that this can be done efficiently using CVXPY, but no justification or computational complexity analysis is provided to support this claim.

### Questions
**Questions:**

1) You write "equation 10 can be efficiently computed using the CVXPY Python package (Diamond \& Boyd, 2016)." What exactly does "efficiently computed" mean in this context, and why is the problem computationally tractable?

2) What is the set $P_s$ in Algorithm~1? Is it related to the reference distribution $P_t$ (e.g., $P_s = P_t$ for $t = s$), or is it an independent sampling distribution for the scalarization weights?

3) The reference for the duality formula of the $\phi$-divergence DRO (Equation~9) is incorrect. This result originates from:  
\emph{A. Ben-Tal, D. Den Hertog, A. De Waegenaere, B. Melenberg, and G. Rennen, "Robust solutions of optimization problems affected by uncertain probabilities," Management Science, 59(2):341–357, 2013.}  
Please cite the original source.

4) Before Definition~1, are you not simply introducing the Minkowski sum? Please clarify this connection explicitly.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies $\varphi$-divergence distributionally robust multi-objective Bayesian optimization. Unlike prior work (Inatsu et al., 2024) that assumes independent context variables across objectives, this work considers shared context variables when constructing the ambiguity set for robustness. Building on the dual formulation of φ-divergence DRO, the paper extends this to the multi-objective setting via random scalarization (Paria et al., 2020). The authors derive sublinear cumulative regret bounds and validate the algorithm on synthetic benchmarks under TV and $\mathcal{X}^2$-divergence settings, empirically demonstrates consistent sublinear growth of cumulative regret under varies discretization fineity.

### Strengths
- The shared-context DRMO setting is realistic and practically important. 
- The proposed algorithm is in principle applicable to a broad class of $\varphi$-divergences (TV, KL, $\mathcal{X}^2$, CVaR), providing flexibility in modeling different robustness notions.
- The proposed algorithm comes with theoretical guarantee.
- Compared to Husain et al. 2023 and Tay et al. 2022, the analysis builds upon high probability bounds instead of the deterministic bounds on function values which is more sensible.

### Weaknesses
- As this paper mainly extended Huang et al., 2024 for the multi-objective setting in combination with Paria et al 2019, although spotting this gap is plausible, the approach is very straightforward. 
- Limited experimental scope: 
    - **Missing baselines**: While it is understandable that the problem setting is new in MOBO and there may not have any approaches that may be directly comparable, but some simple baselines (e.g., random sampling) could be considered. 
    - **Repetition**: Why there is no repetition variance provided.
    - **Real-word problem**: I think there are also synthetic real-life problem (e.g., crashworthiness) could be considered to fullfill the experiment except for the current well-studied synthetic problem.

I would be happy to reconsider my score if the author could provide more persuading experimental results.

### Questions
- **Extension to other divergences**: It is also mentioned that some other divergence measure like MMD and WD can be considered as future work, since the dual form is actually agnostic of the shift measure, how difficult the extension of this approach to this setting?
- **Thompson sampling**: Since Thompson sampling is also considered in Paria et al 2019, is there any reason that is not considered to be extended as well in this setting?
- **Chebyshev Saclarization**: Is there a reason of not considering Chebyshev sclarization?

### Soundness
3

### Presentation
3

### Contribution
2
