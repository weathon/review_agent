# Good Allocations from Bad Estimates

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Conditional average treatment effect (CATE) estimation is the de facto gold standard for targeting a treatment to a heterogeneous population. The method estimates treatment effects up to an error $\epsilon > 0$ in each of $M$ different strata of the population, targeting individuals in decreasing order of estimated treatment effect until the budget runs out. In general, this method requires $O(M/\epsilon^2)$ samples. This is best possible if the goal is to estimate all treatment effects up to an $\epsilon$ error. 
In this work, we show how to achieve the same total treatment effect as CATE with only $O(M/\epsilon)$ samples for natural distributions of treatment effects. The key insight is that coarse estimates suffice for near-optimal treatment allocations. In addition, we show that budget flexibility can further reduce the sample complexity of allocation.
Finally, we evaluate our algorithm on various real-world RCT datasets. In all cases, it finds nearly optimal treatment allocations with surprisingly few samples. Our work highlights the fundamental distinction between treatment effect estimation and treatment allocation: the latter requires far fewer samples.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate sample complexity bounds for estimating CATE across M different groups when trying to perform allocation to a K subset of them. Standard CATE analysis demonstrates a $\frac{M}{\epsilon^2}$ bound, as each group needs to learn a separate CATE, then the K-highest CATE values are selected. However, the authors argue for a $\frac{M}{\epsilon}$ bound by performing coarser estimation for treatments near the boundary. Essentially, the authors demonstrate that if CATE estimates can be learned within $\sqrt{\epsilon}$, then any estimation mistakes will not be very costly. Such learning is possible under certain assumptions on $\tau$; for example, that $\tau$ is smooth or is near-uniform. The authors conclude with experiments on real-world RCTs to demonstrate the efficacy of their experiments.

### Strengths
1. **Work tackles an important problem** - The authors tackle the important problem of $K$ selection from $M$ groups in a causal setting. Such a problem can be seen across a variety of real-word situations, and is especially prevalent in the world of policy. This can help more efficiently allocate resources and avoid unnecessary experiments. 
2. **Analysis is intuitive and clean** - The authors present a clean and intuitive reason why their proposed selector should outperform baselines. By avoiding the need to recompute CATE for each of the $M$ groups, the authors are able to achieve a better sample complexity, due to their ability to adaptively explore in some sense. 
3. **Characterizes when their new bound is possible** - The authors describe several scenarios where there newly proposed estimators reach the desired $\frac{1}{\epsilon}$ bound, and describe why such assumptions are not onerous. For example, the authors describe a class of smooth distributions for $\tau$ that allow for such bounds, and they express an if and only if condition based on the CDF of $\tau$. 
4. **Extension to Flexible Budget** - In Section 5, the authors include a discussion of flexible budgets, where an alternative budget $K'$ is used near $K$ that achieves better sample complexity performance. The authors sketch when such a method does and does not work, dependent on the distribution of $\tau$.

### Weaknesses
1. **Experiments are not Extensive** - The experiments in Section 6 are condensed to a half a page (with some extra material in the Appendix). As a result, it's hard to understand some of the results. For example, why is the failure percentage not monotonic in $\epsilon$; presumably with increasing $\epsilon$, it is less a stringent failure threshold, so it is surprising that this pattern is exhibited across datasets. Additionally, there is little comparison with the $\frac{1}{\epsilon^2}$ method. Finally, what does the actual distribution of $\tau$ look like in practice; are the assumptions validated?

### Questions
1. In practice, on the datasets listed in the experiments section, how does the sample complexity of the CATE-style selector ($\frac{1}{\epsilon^2}$ compare with the selector proposed

### Soundness
3

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
3

### Summary
The authors provide new theory showing that optimal treatment allocation can be achieved with fewer samples than classic predict-then-optimize approaches (FullCATE in the paper) would require. Their theory is built upon the insight that accurate estimates of CATE are only needed around the treatment allocation threshold.

### Strengths
- The authors provide interesting insights about sample size requirements for optimal treatment allocation.
- The authors substantiate their claims with extensive theoretical analysis

### Weaknesses
Some other related work exists that uses similar insights about the problem of optimally allocating treatment, though often without an extensive theoretical analysis. For example, some work has argued that when trying to find the optimal treatment allocation, accurate CATE estimation is not always the most effective [1,2]. 

While the authors provide a very extensive theoretical analysis, they only briefly explain the potential impact of their contributions. For example, I find it hard to understand what practitioners should do with this new information. It would be helpful if the authors could discuss this.

I wonder why the authors decided to put the individuals into different groups and do the analysis based on these groups. As the number of groups decreases, the number of samples needed also decreases, but the quality of the overall allocation will also go down (because as you make the groups more granular, you will find more heterogeneity between groups, allowing for better decision-making). How should you choose the number of groups in practice if you have very little a priori information about the CATE distribution?

Related to the previous point, I do not understand how the different groups are created in the experiments. To me, this seems like the most crucial part when evaluating treatment allocation quality.

I wonder why the authors did not use datasets that are often used in Uplift Modeling [1] and treatment effect estimation [3]. 

[1] Devriendt, F., Van Belle, J., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. IEEE Transactions on Knowledge and Data Engineering, 34(10), 4888-4904.

[2] Fernández-Loría, C., & Provost, F. (2022). Causal decision making and causal effect estimation are not the same… and why it matters. INFORMS Journal on Data Science, 1(1), 4-16.

[3] Curth, A., Svensson, D., Weatherall, J., & Van Der Schaar, M. (2021, August). Really doing great at estimating cate? a critical look at ml benchmarking practices in treatment effect estimation. In Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2).

### Questions
All the items listed in the Weaknesses section may be interpreted as questions by the authors.

Typo: line 353 "notion that requiring"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper fills the gap between estimating CATE and making decisions on the allocations. The authors show that while estimating all CATEs within $\epsilon$ accuracy requires $O(M/\epsilon^2)$ samples, achieving a near-optimal $(1-\epsilon)$ treatment allocation typically needs only $O(M/\epsilon)$ samples under mild distributional assumptions. In general, I personally see the results quite interesting and insightful.

### Strengths
1. The paper makes a clear and elegant theoretical distinction between estimation and allocation. The reduction of the sample complexity from $M/\epsilon^2$ to $M/\epsilon$ is insightful and exciting.
2. Practical relevance: Direct implications for RCT and policy design: significant reduction in sample cost.
3. The proofs are clean and well-structured and the theoretical results are rigor.

### Weaknesses
In general, I enjoy reading the paper a lot. I do not have major concerns.

1. Comparison to bandit best arm identification could be expanded. The link is conceptually strong. Particularly, recently, there are some works on good arm identification. Some ideas are very similar, although they are not is a causal inference setting.

2. Policy implication is strong (“RCTs underpowered for CATE estimation can still yield good allocations”), but guidance on how to detect $\rho$-regularity or compute sample sizes in practice is missing.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
4
