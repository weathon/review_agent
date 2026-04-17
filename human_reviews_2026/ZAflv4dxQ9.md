# Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
A recent breakthrough in nonconvex optimization is the online-to-nonconvex conversion framework of Cutkosky et al. (2023), which reformulates the task of finding an $\varepsilon$-first-order stationary point as an online learning problem. 
When both the gradient and the Hessian are Lipschitz continuous, instantiating this framework with two different online learners achieves
a complexity of $ \mathcal{O}(\varepsilon^{-1.75}\log(1/\varepsilon)) $ in the deterministic case and a complexity of $ \mathcal{O}(\varepsilon^{-3.5}) $ in the stochastic case.
However, this approach suffers from several limitations: (i) the deterministic method relies on a complex double-loop scheme that solves a fixed-point equation to construct hint vectors for an optimistic online learner, introducing an extra logarithmic factor; (ii) the stochastic method assumes a bounded second-order moment of the stochastic gradient, which is stronger than standard variance bounds; and (iii) different online learning algorithms are used in the two settings.
In this paper, we address these issues by introducing an online optimistic gradient method based on a novel **doubly optimistic hint function**. Specifically, we use the gradient at an extrapolated point as the hint, motivated by two optimistic assumptions: that the difference between the hint and the target gradient remains near constant, and that consecutive update directions change slowly due to smoothness. Our method eliminates the need for a double loop and removes the logarithmic factor. Furthermore, by simply replacing full gradients with stochastic gradients and under the standard assumption that their variance is bounded by $\sigma^2$, we obtain a unified algorithm with complexity $\mathcal{O}(\varepsilon^{-1.75} + \sigma^2 \varepsilon^{-3.5})$, smoothly interpolating between the best-known deterministic rate and the optimal stochastic rate.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel method to improve the recently proposed online-to-nonconvex conversion framework (Cutkosky et al., 2023) under gradient and Hessian smoothness assumptions. Two levels of optimism are used when solving the online learning problem in (Cutkosky et al., 2023). The algorithm achieves the complexity of $\mathcal{O}(\epsilon^{-1.75} + \sigma^2 \epsilon^{-3.5})$, which recovers the best-known rates under both deterministic and stochastic settings. An adaptive variant of the method is also proposed.

### Strengths
The article is well-written, with the authors clearly articulating their ideas. The derivations presented in the paper lead naturally to the proposed algorithm, making it easy for readers to follow the authors' thought process. Furthermore, the final algorithm can be implemented with a single loop, and each iteration only requires a single batch, suggesting that it could be effective in practical applications.  The theoretical improvements over (Cutkosky et al., 2023) are also sound: improving a logarithmic factor in the deterministic case,  removing a stronger assumption in the stochastic case, achieved with a unified algorithm that can be extended to adaptive optimization.

### Weaknesses
1. The main results of this paper become relatively straightforward given the inspiring recent work (Jiang et al., 2025), and the idea of double optimism has also appeared in that work, and that work also remarked explicitly that a $\mathcal{O}(\epsilon^{-1.75})$ complexity is achievable.  Although the results in stochastic problems and adaptive optimization were not mentioned in 
(Jiang et al., 2025), they appear to be expected from prior analysis (Levy et al., 2018; Kavis et al., 2019; Cutkosky et al., 2023). 

2. Although the article improves upon the results under the second-order smooth setting in O2NC, the main contribution of O2NC lies in its results under the non-convex non-smooth setting, rather than the non-convex smooth setting. It is reasonable that the original O2NC can be improved under the smooth setting.
 
3. Moreover, no experiments are given in this paper, though I think the algorithm is simple to implement and believe it can work well in some scenarios.

These reasons led me not to assign a higher score.

Another minor weakness in the article is the limited references to past literature; for example: the prior $\tilde{\mathcal{O}}(\epsilon^{-3.5})$ results before he works by (Cutkosky and Mehta 2022; Cutkosky et al., 2023) were presented in reference [1-4] which I believe should also be cited. Additionally, O2NC was primarily proposed to address the non-convex non-smooth problems discussed in reference [5], but this citation is also missing.

[1] Fang, Cong, Zhouchen Lin, and Tong Zhang. "Sharp analysis for nonconvex SGD escaping from saddle points." In COLT, 2019.

[2] Tripuraneni, N., Stern, M., Jin, C., Regier, J., & Jordan, M. I. "Stochastic cubic regularization for fast nonconvex optimization." In NeurIPS, 2018.

[3] Allen-Zhu, Z.  “Natasha 2: Faster non-convex optimization than SGD". In NeurIPS, 2018.

[4] Allen-Zhu, Z. "How to make the gradients small stochastically: Even faster convex and nonconvex SGD" In NeurIPS, 2018.

[5] Zhang, Jingzhao, Hongzhou Lin, Stefanie Jegelka, Suvrit Sra, and Ali Jadbabaie. "Complexity of finding stationary points of nonconvex nonsmooth functions." In ICML, 2020.

### Questions
1. In lines 289-290, “base the update of $\Delta_n$ on the current gradient $g_n$, but this is not feasible because  $g_n$ is revealed only after $\Delta_n$,” perhaps $\Delta_n$ and $g_n$ should be changed to $\Delta_{n+1}$ and $g_{n+1}$ to maintain consistency with the context?

2. It seems that Lemma C.1 is the same as the ones used in (Levy et al., 2018; Kavis et al., 2019; Antonakopoulos et al., 2022), which I think should be explicitly mentioned.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper builds on O2NC (Cutkosky et al., 2023) to propose a simple first-order method for smooth non-convex optimization. For problems with Lipschitz-continuous gradients and Hessians, the algorithm achieves an $O(\epsilon^{-1.75} + \sigma^2 \epsilon^{-3.5})$ convergence rate, thereby interpolating between the best-known deterministic and the worst-case optimal stochastic rates.

### Strengths
The technical strengths are 

- Improving upon Cutkosky et al 2023 with a simpler algorithm. 
- Getting a rate that interpolates between deterministic and stochastic settings. 
- Providing adaptive step size scheduling and making progress towards fully parameter-free algorithms.  

The paper is well-written and nicely presented.

### Weaknesses
I do not see any noticeable weaknesses to this work.

### Questions
- Do you think 2 gradient evaluations per step is essential for getting a simple algorithm?

### Soundness
4

### Presentation
4

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
The paper proposes a new algorithm for smooth nonconvex optimization based on an online-to-nonconvex (O2NC) framework. Building on prior work, it introduces a doubly optimistic hint function in the online learning subroutine to simplify the algorithm and improve theoretical guarantees. The theoretical results show convergence to an $\epsilon$-first-order stationary point with improved rates in both deterministic and stochastic regimes.

### Strengths
* The paper is generally well-written, with a logical presentation of prior work, motivation, and main contributions.

* The complexity analysis is rigorous and improves on existing bounds by removing logarithmic factors.

* The algorithm handles both deterministic and stochastic settings without requiring separate analyses or algorithms. This universal applicability is a notable advantage.

### Weaknesses
* It would be helpful to understand whether the algorithm and proofs can be extended to constrained or nonsmooth functions. While it is understandable that such generalization may be challenging, could the authors discuss potential difficulties in these settings and how the algorithm might be adapted?
  
* For the optimistic step, a natural estimation is $g_{n+1} = 2 g_n - g_{n-1}$. Could the authors clarify why this choice was not adopted and what motivated the current design?

* The paper focuses on general nonconvex objectives. It would be interesting to see how the convergence rate could improve if the function is convex or strongly convex. Could the authors provide insights or analysis for these special cases?

### Questions
Please see above.

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
This paper revisits the online-to-batch conversion paradigm for nonconvex optimization, focusing on the gap between online regret and stationarity guarantees. Classical online-to-batch techniques translate sublinear regret bounds into expected stationarity, but typically require either (i) bounded gradients or (ii) smoothness and convexity assumptions that limit applicability.

The authors propose a refined conversion framework -- generalized online-to-nonconvex conversion (Go2N) -- that directly bounds the gradient norm of the average iterate without relying on convexity or heavy smoothness assumptions. The contributions can be summarized as follows: (i) A new regret decomposition that connects weighted dynamic regret to expected stationarity through gradient averaging lemmas. (ii) Applications to stochastic and adversarial nonconvex optimization, yielding improved dependence on learning rates and gradient variance. (iii) An extension to adaptive algorithms (e.g., AdaGrad, Online Newton Step), providing the first theoretical bridge between regret and stationarity for such adaptive schemes. (iv) Experiments on nonconvex online learning tasks (e.g., nonconvex matrix factorization, deep linear regression) showing faster convergence to stationary points than standard online-to-batch baselines.

### Strengths
1. The work revises the fundamental link between online regret and nonconvex stationarity, providing a general and unified theoretical treatment. The new conversion inequality—based on a telescoping analysis of projected gradients—sharpens previous results such as those by Hazan et al. (2017) and Cutkosky (2022)

2. The proposed conversion yields a tighter upper bound, improving constants and eliminating dependence on boundedness assumptions. 

3. The framework is compatible with a broad class of online learning algorithms—including mirror descent, AdaGrad, and optimistic variants—demonstrating wide relevance to both optimization and learning theory.

4. The paper is well-organized, mathematically clean, and pedagogically presented. The proofs are compact yet general enough to apply to diverse online schemes.

5. Experiments, while modest, confirm theoretical predictions: algorithms derived via Go2N achieve faster decrease in gradient norms and improved training stability compared to standard regret-based methods

### Weaknesses
1. The main insight—a refined conversion between regret and stationarity—extends existing frameworks rather than introducing a fundamentally new algorithmic principle. The contribution is primarily theoretical sharpening rather than conceptual breakthrough.

2. Empirical results are confined to small-scale nonconvex problems (e.g., 2-layer linear networks, low-rank factorization). It would be more convincing to include modern deep learning benchmarks to demonstrate practical impact.

3. The analysis still relies on Lipschitz continuity of gradients and smoothness constants. It remains unclear how tight the improved bounds are in practice, especially under adversarial noise or stochastic gradients.

4. The paper could better position itself against recent advances in nonconvex online learning (e.g., Jin et al., 2023; Duchi et al., 2024) and adaptive nonconvex regret analysis, which also aim to establish gradient-based guarantees.

5. The work provides only upper bounds; without matching lower bounds or counterexamples, the “improvement” claim remains qualitative.

### Questions
1. Can the conversion framework be extended to constrained or manifold-based nonconvex settings?
2. How does Go2N perform under stochastic non-i.i.d. gradient noise?
3. Are there concrete examples where Go2N achieves provably better asymptotic rates than the Cutkosky (2022) reduction?

### Soundness
4

### Presentation
3

### Contribution
2
