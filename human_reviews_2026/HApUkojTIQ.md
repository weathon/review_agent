# Label-consistent clustering for evolving data

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Data analysis often involves an iterative process, where solutions must be continuously refined in response to new data. Typically, as new data becomes available, an existing solution must be updated to incorporate the latest information. In addition to seeking a high-quality solution for the task at hand, it is also crucial to ensure consistency by minimizing drastic changes from previous solutions. Applying this approach across many iterations, ensures that the solution evolves gradually and smoothly. 

In this paper, we study the above problem in the context of clustering, specifically focusing on the k-center problem. More precisely, we study the following problem: Given a set of points X, parameters k and b, and a prior clustering solution H for X, our goal is to compute a new solution C for X, consisting of k centers, which minimizes the clustering cost while introducing at most b changes from H. We refer to this problem as label-consistent k-center, and we propose two constant-factor approximation algorithms for it. We complement our theoretical findings with an experimental evaluation demonstrating the effectiveness of our methods on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new problem formulation in the context of k-center (or in general center-based) clustering, named label-consistent clustering. In this problem, which is strongly related to the problem of minimizing recourse, one is given a previous solution (i.e. a set of centers) to the pointset at hand, as is tasked with finding a new solution that minimizes the objective function while introducing at most b changes to the assignment of points to their centers. The paper presents two constant-factor-approximation algorithms, provides a theoretical analysis of the algorithms, and provides some experimental evaluation of them.

### Strengths
- The definition of label-consistent clustering, an alternative definition of recourse, makes sense and could have practical implications. Therefore, the introduction of a new problem seems justified. 
- The two proposed algorithms seem to be a convincing contribution, as they provide a tradeoff between approximation quality and runtime. 
- The experimental evaluation in an interesting addition to the theoretical analysis.

### Weaknesses
- The definition of the problem is slightly unclear at times. The paper supposes an evolving pointset, akin to the fully dynamic setting. The task tries to minimize changes in the point assignments in the pointset $X_t$ given a solution $\mathcal H$ of the pointset $X_{t-1}$ at time $t-1$ (e.g. see lines 50-57). However, in Definition 3, it seems like $\mathcal H$ is already a solution for $X_t$. This might create some problems, as for example some points that were centers in $X_{t-1}$ might not be in in $X_t$. Therefore, $\mathcal H$ might not be a subset of $X$. Would the definition still be valid? And would the algorithms still work?
- Theorem 1 is hardly a contribution, as it follows directly from the hardness of k-Center. 
- Issues with the presentation of the algorithms:
    - In Section 4.2, the authors should explain that they take all distances, sort them and binary search the correct value for $ r^* $, as right now, for anyone outside the clustering community, it would be really hard to understand how the algorithm would have access to $ r^* $ without reading the appendix.
    - In line 7 of Algorithm 1, points are taken from $H \setminus C$, but $C$ should be empty. Should it be $H \setminus C_0$?
    - At first sight, it seems like $b$ is never used in the algorithm, which is puzzling. In fact, $b$ seems to be used to check if the solution is feasible when looking for the correct value of $r^*$. This should be explained clearly.
- The paper has a superficial literature review, and there seems to me some missing related work:
    - The paper only briefly discusses the literature on consistent clustering, which tries to control recourse. [Lattanzi and Vassilvitskii, 2017], [Lacki et al, 2024] and [Forster and Skarlatos, 2025] are cited, but not discussed in detail. The authors should clearly explain the differences between previous work and label-consistent clustering. [1] also deals with recourse, but was not cited.  
    - Consistent or low-recourse algorithms have been proposed for problems other than clustering, but have not been discussed. E.g. see [2], [3] and references therein. 
    - The paper does not discuss dynamic k-center algorithms (e.g. [4], [5], [6]), even though this framework seems to be the main motivation for this paper.
- There is no indication whether the gap between the lower bound of 2 and the upper bound of 3 can be closed with an algorithm that is polynomial time (i.e. not FPT) or not. Such a lower bound or a 2-approximation algorithm would have made the contribution much stronger. This is of course a *minor* weakness, I understand that achieving this might be really hard.  

----
References:

- [1] Fichtenberger et al. Consistent k-Clustering for General Metrics. SODA 2021.
- [2] Bhattacharya et al. Chasing Positive Bodies. FOCS 2023.
- [3] Bernstein et al. Online Bipartite Matching with Amortized O(log2n) Replacements. J. ACM 2019
- [4] Chan et al. Fully Dynamic k-Center Clustering. WWW 2018
- [5] Pellizzoni et al. Fully Dynamic Clustering and Diversity Maximization in Doubling Metrics. WADS 2023
- [6] Bateni et al. Optimal Fully Dynamic k-Center Clustering for Adaptive and Oblivious Adversaries. SODA 2023.

### Questions
- the choice to prove correctness of Algorithm 1 via Algorithm 2 seems odd. Do you think there is a simpler proof? 

----

Remark: 

Overall, I think the paper has some clarity issues, and the algorithmic contribution is not major. However, I am willing to raise my score if the clarity is improved.

### Soundness
3

### Presentation
2

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
This paper introduces and studies the problem of updating an existing historical $k$-center clustering solution $H$ when the dataset gets updated, under the constraint that only $b$ points are allowed to change their centers. They present a $2$ approximation (to the best possible $k$-center clustering under these constraints) running in FPT time in the number of centers, and a $3$ approximation running in polynomial time. They also extend their algorithms to generally work with evolving data by using the output of the previous time step as the historical solution to the current time step. Both their algorithms start by assuming they know the radius of an optimal label-consistent clustering, $r^*$, since they can obtain it by binary searching for the best value.

Their first contribution is an algorithm OverCover that gives a $2$ approximation. The algorithm works by guessing the set of centers in $H$ that do not change in a new optimal solution, called $H^*$, then uses the classic algorithm of Hochbaum and Shmoys on the remaining unclustered data to obtain a $2$ approximation. The guessing of centers causes the blow up in the terms that depend on $k$.

Their second contribution is a $3$-approximation algorithm that runs in polynomial time. In the first phase, the algorithm runs the classic algorithm above to obtain a set of centers. In the next phase, each center in this set is swapped with a center from its own cluster that is also a historical center. The historical center with the highest weight is chosen, which is the number of points currently within an $r^*$ radius of the center that also used to belong to its cluster in $H$. Finally, historical centers are added in decreasing order of their weights until the current solution reaches $k$ centers.

They also extend existing lower bounds to show that label-consistent $k$-center is NP hard to approximate within a less than 2 factor, and also W[2]-hard to approximate to the same factor when parameterized in the number of centers.

Finally, they evaluate slightly modified versions of their algorithms against the classical one by Hochbaum and Shmoys, the FFT algorithm that iteratively searches for the farthest point to explore, and the algorithm by Ahmadian et al. for resilient clustering.

Summary recommendation: I would weakly recommend rejecting the paper due to the above-mentioned doubts. Specifically, I do not immediately get the motivation for defining the model in this way as well, which, if the authors could talk about in their rebuttal, would be useful for situating their contributions more in context. The theoretical part and the experimental part also have coherent messages separately, but I could not see if the algorithms implemented had any of the guarantees that were proven earlier. A discussion about this (if it does satisfy some of the guarantees) would be useful.

### Strengths
- The paper presents algorithms that are theoretically neat and simple and give provable guarantees.
- The paper is generally written well and easy to read and follow.
- The experiments show that (a modified version of) their algorithms are better than the baselines.

### Weaknesses
- As I can see, there are two ways of asking for recourse bounds in the setting of evolving data: One is the method taken here, which is to consider the case where there already exists a historical solution (which does not arise from the algorithm itself). Another, which I've personally seen more of, is to ask that the algorithm maintains a good approximation to the overall best solution, but with low recourse. In the latter setting, the guarantee is measured with respect to all clusterings instead of just the ones within swap distance $b$ of the current clustering. Is there a reason for choosing the former (which also gives guarantees with respect to really bad historical clusterings) here? Are they equivalent in some way? Is there a reason for considering the setting where the algorithm might get bad historical solutions that it did not come up with but was given to it as a warm start? Are there bounds that say that any algorithm that maintains a close-to-optimal solution must have high recourse necessarily?
- The first algorithm requires iterating over all possible $H^*$s which is very costly and not part of the implemented version of the algorithm, which makes it practically infeasible. One can implement it without the iteration, but then does it have any theoretical guarantees?

### Questions
Typos:
line 045: re-assinged -> re-assigned
line 183: that present -> that are present
line 215: "a set S of k centers such that ... |S| \le k"
line 246: should the call to CARVE be r^* instead of 2r^*?
line 307: deferred in -> deferred to
line 322: algorithms -> algorithm
line 701: instanes -> instances
line 751: due because -> because?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the Label-consistent $k$-clustering problem, which aims to find a new clustering on an evolving dataset while keeping the labeling consistent with a given historical clustering. Specifically, the objective is to minimize the clustering cost subject to the constraint that the number of points whose cluster labels change does not exceed a given budget $b$. The authors propose a tight FPT 2-approximation and a polynomial-time 3-approximation algorithms, and provide experiments on both synthetic and real datasets to demonstrate the effectiveness of their approach.

### Strengths
1. The formulation captures an important setting in incremental or evolving data analysis, where stability across time steps is crucial for interpretability and system reliability.
2. The paper offers both provable guarantees and empirical validation, which strengthens its technical completeness.

### Weaknesses
1. While the introduction and experiments emphasize the incremental or evolving nature of the data (e.g., extending a historical clustering to a new dataset at time $t$), the formal definition (Definition 3) treats the instance as a static dataset $𝑋$. It would improve clarity if the authors explicitly defined how the dataset evolves, e.g., distinguishing between existing and newly arrived points, and how the label consistency constraint applies in that setting.

2. In the problem definition, the threshold $b$ bounds the allowed number of label changes, yet it is not clearly stated whether it is an input parameter. Since the algorithms and experiments rely on $b$, the paper would benefit from a short clarification of how $b$ is chosen or interpreted.

3. The paper mainly discusses data additions (incremental updates), but not the case where some previous data points are removed. A short remark on how the approach could be adapted to handle such deletions would make the formulation more complete.

### Questions
1. In the formal definition, the dataset $X$ appears static. How should readers interpret the **incremental** setting mentioned in the motivation. Does $X$ consist of both old and new data, or only newly added points? Could you make this distinction explicit to better reflect real incremental scenarios?

2. How would the proposed approach handle data deletions—i.e., when some previously clustered points are removed from the dataset? Would the label-consistency constraint or the budget $b$ need adjustment?

3. Do you think the same framework or algorithmic idea could be extended to other objective functions such as $k$-median or $k$-means? Since the definition of label consistency is independent of the specific cost function, a short discussion on this potential generalization would be valuable.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the "Label-Consistent $k$-Clustering" problem, aiming to compute a new $k$ cluster $C$ such that, while minimizing clustering costs (focusing on $k$-centers), the number of reassigned points does not exceed a budget $b$ compared to a given historical cluster $H$.

### Strengths
This paper introduces the problem of label-consistent k-clustering, a novel clustering formulation that aims to optimize clustering cost while ensuring a consistency constraint, in the form of a maximum number of data point re-labelings, from a historical clustering. Meanwhile, it presents two constant-factor approximation algorithms for the k-center variant of the proposed problem.

### Weaknesses
1. The innovation is insufficient. The label consistency constraint of historical cluster H is highly similar to the existing "elastic clustering" model, which lacks fundamental innovation and is more like an incremental adjustment of the existing model for a specific scenario.

2. The core theory and approximation algorithm in the paper are only for the k-center objective function, but the k-median and k-mean are more practical in the evolutionary data scenarios mentioned by the authors. This limitation greatly weakens the general value of the model.

3. The asymptotic complexity of the polynomial-time algorithm includes an $O(n^2)$ term, which is too inefficient for large datasets (where n is very large) common in evolutionary data streams, making it impractical.

4. The OVERCOVER algorithm, with the strongest theoretical guarantee (close 2-approximation), is exponentially time-consuming ($O(2^k)$) on $k$. The authors also acknowledge that it is impractical for large $k$ values, making the optimal theoretical result impractical.

5. The construction of the historical clustering $H$ in the experimental setup is too artificial and fails to effectively simulate real evolutionary scenarios; furthermore, the baseline algorithm ignores the crucial budget constraint $b$ in the comparison.

### Questions
See the Weakness.

### Soundness
2

### Presentation
2

### Contribution
2
