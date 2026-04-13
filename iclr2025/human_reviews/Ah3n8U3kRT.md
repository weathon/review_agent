## Human Reviewer 1

### Summary
This paper considers zeroth-order convex optimization problems with symmetric stochastic heavy-tailed noise. The gradient is estimated by the median in a batch, instead of using the empirical mean used in prior works. This technique is also applied to the linear bandits with heavy-tailed symmetric noise.

### Strengths
- With the additional symmetric noise assumption, this paper has improved the prior methods under the heavy-tailed noise case.
- The paper is relatively clear and easy to follow. Most concepts are well explained and the comparison of the final results with prior works is clear.

### Weaknesses
**Majors**:
- Under the additional symmetric noise assumption, this paper deals with the case when $\kappa\in(0,1)$, and improves the bound for $\kappa\in[1,2)$. From the algorithm design, Algorithm 1 and ZO-clipped-SSTM (Kornilov et al. (2023b)) largely follow the same procedure, expect that the estimate of the gradient changes from the randomly smoothed mean to the randomly smoothed median. Similarly, Algorithm 2 and ZO-Clip-SMD (Kornilov et al. (2023a)) are similar expect for the gradient estimate part. From this perspective, the main procedure required to deal with $\kappa\in(0,1)$ is the change of estimation statistics. It would be great if the authors can highlight the additional technical challenges and novelties. Otherwise, this paper looks incremental given the prior works.
- The motivation for the symmetric assumption is not well explained. As the symmetry in the noise is critical for the median estimate, a motivation is needed to support this assumption. 
- As this paper adopts bandits as an application and there is already some work in the bandits literature which considers heavy-tailed noise or robust bandits problems (e.g., [1]) without the symmetric noise assumption, it would be great if the authors can compare their results to them, in terms of the assumptions, methods, theoretical guarantees and efficiency of the algorithms.

**Minors**:
- Can the authors kindly modify Line 73? It can be misleading as Bubeck et al. (2013) does not make the symmetric noise assumption.

[1] Mathieu, T., Basu, D., & Maillard, O. A. (2024). Bandits Corrupted by Nature: Lower Bounds on Regret and Robust Optimistic Algorithms. _Transactions on Machine Learning Research Journal_.

### Questions
- For the stochastic MAB problem, as suggested by the authors that the regret upper bound of the proposed algorithm is $\tilde{O}(\sqrt{Td})$, which is suboptimal compared to the lower bound for MAB with bounded variance of losses. Is this because the proposed algorithm can deal with unbounded variance case (with the symmetric noise assumption)? If it is, how is the proposed upper bound in Theorem 3? Is there any lower bound?
- According to Theorem 1, 2, 3, the parameter $m$, which decides the batch size used to estimate the median, depends on the noise parameter $\kappa$. When the knowledge of $\kappa$ is not provided, how can the algorithm be implemented?

Please also kindly refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper considers the problem of non-smooth convex optimization with bandit and heavy tailed feedbacks, and introduces the technique of building unbiased median gradient estimate with bounded second moment. This technique are further applied into the MAB problems with heavy-tailed rewards. Several empirical investigations are conducted to verify their theoretical findings.

### Strengths
- This paper is clearly written and well-organized.
- The authors have established a series of theoretical results, and perform the corresponding numerical experiments.

### Weaknesses
- It seems that the proposed methods are simple combinations of existing algorithms with a median gradient estimate. For example, the proposed ZO-clipped-med-SSTM is similar to ZO-clipped-SSTM, with only a different gradient estimate. While I understand that even this combination presents certain analytical challenges, I encourage the authors to clearly describe these challenges
- Currently, the effectiveness of the method is verified on synthetic datasets. Could it also be validated on real-world datasets?
- Several crucial related works on bandits with heavy-tailed feedbacks [1, 2, 3] and  non-smooth optimization [4, 5, 6] are missing from the discussion of this paper. 


[1] Multi-armed bandit problems with heavy-tailed reward distributions. 2011.

[2] Almost Optimal Algorithms for Linear Stochastic Bandits with Heavy-Tailed Payoffs. 2018.

[3] Optimal Algorithms for Lipschitz Bandits with Heavy-tailed Rewards. 2019.

[4] Complexity of finding stationary points of nonconvex nonsmooth functions. 2020.

[5] No dimension-free deterministic algorithm computes approximate stationarities of Lipschitzians. 2024.

[6] High-Probability Bound for Non-Smooth Non-Convex Stochastic Optimization with Heavy Tails. 2024.

### Questions
See in Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This work makes the first step to study zeroth-order non-smooth convex optimization with heavy-tailed noises. To this end, the authors propose the ZO-clipped-med-SSTM estimator and apply it to solve zeroth-order non-smooth convex optimization with heavy-tailed noises in both the constrained and unconstrained cases. Also, the authors showcase the capability of their method in the case of multi-armed bandits (MABs) with heavy-tailed reward noises. Numerical experiments are provided.

### Strengths
1.	This work seems to be the first work to solve the zeroth-order non-smooth convex optimization with heavy-tailed noises.

### Weaknesses
1. **Technical Innovations**: Both the clipping operation, the construction of the batch gradient estimator, and the algorithmic designs of ZO-clipped-med-SSTM seem particularly similar to those in [1]. Also, Lemma 1, the key lemma in this work, is proven based on the 
Lemma 2.3 of [2], which guarantees the bounded second moment. I would suggest the authors give more detailed comparisons with previous works, especially on the key technical innovations that do not appear in previous works.
2. **Adversarial Case**: The proposed algorithm requires the construction of a gradient estimator in a batched manner, which makes it unable to deal with the adversarial MABs with heavy-tailed noises. This is a drawback compared with [3], which is unable to handle adversarial MABs with heavy-tailed noises.
3. **Known $\kappa$**: All the algorithms in this work seem to require the knowledge of $\kappa$ to tune the median size parameter $m$. However, the algorithm in [3] can work in the case with unknown $\kappa$.
4. **Presentation**: 
   * For Table 1, I think it would be better if there are some discussions about both the advantages and the disadvantages of the results of the proposed methods against the baseline methods. Currently, the authors just present the results without providing any discussions on them.
   * Assumption 3 seems to be quite different from the heavy-tailed assumption in [3]. I think it would be better if the authors could give detailed discussions on this.

[1] Kornilov et al. Accelerated Zeroth-order Method for Non-Smooth Stochastic Convex Optimization Problem with Infinite Variance. NeurIPS, 23.

[2] Kornilov et al. Gradient-free methods for non-smooth convex stochastic optimization with heavy-tailed noise on convex compact. Computational Management Science, 23.

[3] Huang et al. Adaptive Best-of-Both-Worlds Algorithm for Heavy-Tailed Multi-Armed
Bandits. ICML, 22.

____
**Post-rebuttal**: Sorry for the late replies. I thank the authors for the explanations. Some of my previous concerns regarding the presentation are addressed. However, my concern regarding whether the proposed algorithm is applicable in adversarial cases remains unsolved. In particular, the authors said that “algorithm will make a few shots with the same distribution to be able to construct a batch”. However, in adversarial cases, I think the key problem is exactly that the loss distribution may vary in each round and I believe it is impossible to construct a batch for the loss distribution in one round. For the technical novelty and my previous concerns regarding the advantages of the results against those in [1], it turns out that these are mainly due to the symmetric noise distribution assumption in Assumption 3 of this work. Such an assumption seems not very natural in practice and is also not required by [1]. As such, I am afraid that this work might not be ready for publication in ICLR and I have to maintain my current rating for this work.

[1] Huang et al. Adaptive Best-of-Both-Worlds Algorithm for Heavy-Tailed Multi-Armed Bandits. ICML, 22.

### Questions
1. What does “ZO-clipped-med-SSTM” denote for?
2. On Line 14-Line 15, the authors state that “Unlike the existing high-probability results requiring the noise to have bounded $\kappa$-th moment with $\kappa\in(1, 2]$..”. However, on Line 73-Line 74, it seems that this work actually needs the assumption of bounded $\kappa$-moment of losses ($\kappa\in(1, 2]$) in the case of heavy-tailed MABs. 
3. On Line 81, the authors state that the proposed method achieves high-probability optimal convergence rates. However, I do not seem to find any discussions about the lower bound of the convergence rate for the case of unconstrained optimization. How can we regard the upper bound as “optimal” if there are no lower bounds provided?
4. On Line 84-Line 85, the authors say that they achieve the $\widetilde{O}(\sqrt{dT})$ regret while also saying that their result is “suboptimal”. Why is this result “suboptimal”? To me, this is (nearly) optimal as it is known that the minimax-optimal regret lower bound for MABs is exactly of $\Omega(\sqrt{dT})$.
5. The algorithm in [3] can only obtain a regret of order $ O\left(\sigma K^{1-1 / \kappa} T^{1 / \kappa}+\sqrt{K T}\right) $. Intuitively, why the algorithm in this work can obtain the $\widetilde{O}(\sqrt{dT})$ regret without the dependence on $\kappa$ of the exponent of $T$? Is this because the algorithm explicitly leverages the knowledge of $\kappa$ to tune $m$?
6. The algorithm in [3] can only construct a biased estimator, which also results in their need to handle additional “skipping errors”. Intuitively, why the algorithm in this work can construct the unbiased estimator?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper consider non-smooth convex optimization with a zeroth-order oracle with heavy-tail symmetric stochastic noise. Algorithms are proposed for settings with heavier noise than what is often considered in the literature. The technique is further applied to stochastic multi-armed bandit problem.

### Strengths
In the convex optimization setting, convergence rate of the proposed algorithms are either the first result or matches the best known bounds in the literature (in the case of bounded variance).

Algorithms are demonstrated to perform well in comparison to previous SOTA, even in the bandit case.

### Weaknesses
The regret bound in the bandit case is suboptimal.

### Questions
The need for a two-point oracle seems like a limiting factor. Being able to obtain 2 samples with the same noise parameter does not seem trivial. In what real world settings do we have access to such oracles?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3