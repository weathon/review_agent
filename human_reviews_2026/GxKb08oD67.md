# Fully First-order Methods for Contextual Stochastic Bilevel Optimization

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Contextual stochastic bilevel optimization (CSBO) is a new paradigm for decision making under uncertainty that generalizes stochastic bilevel optimization (SBO) by integrating contextual information in the lower level optimization problem and thus offers a stronger modeling capability. Nevertheless, owing to its semi-infinite nature, CSBO is extremely challenging from a computational perspective, hindering its real-world applications. Indeed, many algorithms designed for SBO are not applicable to CSBO. In this paper, we devise a double-loop fully first-order algorithm for solving CSBO and prove that both sample and gradient complexities of the algorithm are $\widetilde{\mathcal{O}}(\epsilon^{-8})$. To tackle the increasing number of inner loop iterations, we further develop an accelerated version of our algorithm using the random truncated multilevel Monte Carlo technique. The accelerated algorithm enjoys the improved complexities of $\widetilde{\mathcal{O}}(\epsilon^{-6})$. Our algorithms are fully first-order in the sense that they do not rely on second-order information, and hence these complexities cannot be directly compared with those of Hessian-based methods. Numerical experiments on meta-learning with real datasets demonstrate the superiority of the proposed algorithms, especially the accelerated version, over existing Hessian-based method in terms of both speed and accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies contextual stochastic bilevel optimization (CSBO) and proposes a double-loop, fully first-order algorithm with sample/gradient complexities ($\tilde{\mathcal O}(\epsilon^{-8})$). An accelerated variant, built on random truncated multilevel Monte Carlo, improves this to $(\tilde{\mathcal O}(\epsilon^{-6}))$.

### Strengths
1.This paper proposes a fully first-order algorithm that does not rely on second-order information.

2.The numerical experiments demonstrate the empirical performance of the proposed methods.

3.This paper includes a helpful roadmap that improves readability.

### Weaknesses
1.Despite the authors’ emphasis that the contribution is fully first-order, the reported complexity results are not fully satisfactory. To my knowledge, for classical stochastic bilevel optimization, Chen et al. [1] proposed a near-optimal fully first-order method whose rate is within only a logarithmic factor of second-order methods; meanwhile, Bouscary et al. [2] established a connection between CSBO and SBO and—though using second-order information—achieved a sampling complexity of $(\tilde{\mathcal O}(\epsilon^{-3}))$. These comparisons may indicate that there could still be some room to further tighten the results.

2. The method adopts a double-loop architecture, which might introduce some implementation overhead in practice and could limit scalability and applicabilit.

3. The analysis requires a strongly convex lower-level problem.

[1]Chen L, Ma Y, Zhang J. Near-optimal nonconvex-strongly-convex bilevel optimization with fully first-order oracles[J]. Journal of Machine Learning Research, 2025, 26(109): 1-56.

[2]Bouscary M, Zhang J, Amin S. Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation[J]. arXiv preprint arXiv:2503.19991, 2025.

### Questions
1.In Algorithm 1, line 2, the stepsize choice for $(\alpha)$ includes $(\mathcal{O}(1))$. Is there a specific reason to emphasize $(\mathcal{O}(1)) $ here, or would a fixed constant suffice?

2.Related to Weakness 1: given the current results, it might be helpful to comment on whether sharper bounds could be attainable, or to offer some intuition on potential barriers to improving the rates under the present setting.

Minor typo: Line 444, “eary” --> “early”.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies contextual stochastic bilevel optimization (CSBO), where the lower-level problem depends on both the upper-level variable $x$ and a context $\xi$. The authors introduce the first fully first-order (Hessian-free) double-loop method for CSBO. The approach uses a penalty formulation $L(x, z, y, \lambda)$ (Eq. 3) and solves two inner stochastic problems $Q(x, y, \delta; \xi)$ (Eq. 4) at $\delta = 0$ and $\delta = 1/\lambda$. The resulting estimator avoids Hessian-inverse computations entirely.

Two algorithms are proposed: Algorithm 1: Basic first-order CSBO with $\tilde{O}(\varepsilon^{-8})$ complexity (Theorem 3.1). Algorithm 2: Accelerated variant leveraging Random Truncated Multi-Level Monte Carlo (RT-MLMC) and an adaptive step-size gate, achieving $\tilde{O}(\varepsilon^{-6})$ complexity (Theorem 3.4).

Experiments on meta-learning tasks constructed from TinyImageNet embeddings show that Algorithm 2 attains faster convergence than Hessian-based baselines and demonstrates robustness to variance growth caused by large $\lambda$ (Figs. 1–4).

### Strengths
1. Innovative first-order surrogate: A well-motivated penalty objective and twin inner SGD formulation (Eqs. 3–5, Alg. 1).

2. Variance-aware acceleration: RT-MLMC with adaptive gate stabilizes noisy gradients effectively (Alg. 2, Lemma 3.2, Theorem 3.4).

3. Transparent theory: Clear assumptions, proof structure, and complexity results.

4. Empirical validation: Algorithm 2 achieves faster wall-clock convergence than Hessian-based CSBO (Figs. 2–3).

### Weaknesses
1. Limited experimental breadth: Evaluations focus solely on meta-learning tasks using linear models and synthetic contexts.

2. Ablations missing: No analysis on $\lambda$ growth rate, truncation level $N$, or adaptive gate parameters $(c_0, a_1)$.

3. Cost per iteration under-reported: Lacks comparison of wall-clock time per gradient or memory usage between Algorithms 1–2 and Hessian-based methods.

4. Strong convexity requirement: May not hold for many practical CSBO tasks; extending to PL-condition or weak convexity would increase applicability.

5. Open gap to optimal rate: The variance bottleneck ($\tilde{O}(\varepsilon^{-2})$) prevents achieving $\tilde{O}(\varepsilon^{-4})$ complexity.

6. Incomplete statistical reporting: Missing standard deviations in all figures and convergence tables.

### Questions
1. How sensitive is performance to the $\lambda_k$ schedule? Could it be adaptively adjusted using inner residuals?

2. How were gate parameters $(c_0, a_1)$ selected in Algorithm 2? Is there a theoretical guideline or heuristic tuning rule?

3. Can the RT-MLMC estimator be further variance-reduced (e.g., via control variates) to reach $\tilde{O}(\varepsilon^{-4})$ complexity?

4. Would your theory still hold if $g(x, y; \xi)$ satisfies only the Polyak-Łojasiewicz (PL) condition instead of strong convexity?

5. How does Algorithm 1 perform when contexts $\xi$ are discrete and repetitive (e.g., finite meta-tasks)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studied contextual stochastic bilevel optimization and proposed a fully first order approach to solve it. To enhance the efficiency, they proposed an enhanced algorithm based on RT-MLMC technique which reduced the variance of gradient estimator by important sampling. Theoretical analysis is provided for both methods and numerical experiments validate the effectiveness of the proposed methods.

### Strengths
This is the first work proposing the first order method for contextual stochastic bilevel optimization with theoretical guarantee.

### Weaknesses
1. The major concern is that the theoretical analysis for the proposed methods seems not tight compared with either non-contextual stochastic bilevel optimization (${\cal O}(\epsilon^{-4})$ in [2]) or second-order contextual stochastic bilevel methods (${\cal O}(\epsilon^{-4})$ in [3]). Also, it is unclear whether the gap arises from the analysis or from inherent properties of the contextual setting. It would strengthen the paper to include a lower-bound analysis or to explain why the contextual formulation necessarily yields slower rates (e.g., due to additional variance terms, conditioning, or dependence on context complexity). 
2. Another concern is the limited experimental scale. This paper only tested one meta-learning application, which is insufficient to establish broader effectiveness. Moreover, a comparison to methods in the same setting is missing—such as [1], which is applicable when the contextual parameter space is finite. 

[1] Maxime Bouscary, Jiawei Zhang, and Saurabh Amin. Reducing contextual stochastic bilevel optimization via structured function approximation. arXiv preprint arXiv:2503.19991, 2025. 
[2] Lesi Chen, Jing Xu, and Jingzhao Zhang. On finding small hyper-gradients in bilevel optimization: Hardness results and improved analysis. CoLT, 2024. 
[3] Yifan Hu, Jie Wang, Yao Xie, Andreas Krause, and Daniel Kuhn. Contextual stochastic bilevel optimization. NeurIPS 2023.

### Questions
1. In theorem, Algorithm 2 has better convergence rate but in experiments (Figure 1), algorithm 1 is faster with respect to the outer iteration.   Could you get some insights from the theoretical analysis that in which setting, Algorithm 2 tends to outperform? 

2. I'm wondering the key reason for slower convergence rate. In non-contextual bilevel optimization, the rate of first order bilevel method matches that of second-order bilevel methods. 

3. In Remark 3.3, I did not understand why the variance will be ${\cal O}(\epsilon^{-2})$? When $N=c\log(\epsilon^{-1})$, it seems that $\mathcal{O}(2^{\frac{N}{2}})={\mathcal{O}}\left(2^{\log(\epsilon^{-c/2})}\right)=\widetilde{\mathcal{O}}\left(\epsilon^{-c/2}\right)$ which suggests the final complexity depends on the choice of ${\cal O}(1)$ constant c?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses contextual stochastic bilevel optimization, where the lower solution depends on both the upper variable and a context drawn from a possibly uncountable set. It proposes fully first-order algorithms based on a penalty reformulation and establishes sample and gradient complexities.

### Strengths
The motivation is clear since this is positioned as the first fully first-order treatment of general CSBO that does not rely on second-order oracles. Precise analysis is given under different conditions.

### Weaknesses
The proven complexities are weaker than accelerated Hessian based CSBO in terms of epsilon, and the paper notes that direct rate comparison is delicate due to per iteration costs.

### Questions
Do you believe that a fully first-order approach in contextual stochastic bilevel optimization can be competitive with second-order methods? I wonder if the proven complexities can be better or not.

On the other hand, if we compare that in wall-clock time, then is it possible to please specify regimes where a fully first-order approach is better by quantifying gradient and Hessian costs, problem dimension, and target accuracy, etc?

### Soundness
3

### Presentation
3

### Contribution
3
