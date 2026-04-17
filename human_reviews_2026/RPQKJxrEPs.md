# A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4, 6

## Abstract
Let $(X,d)$ be a metric space and let $\mu,\nu$ be discrete probability distributions supported on finite point sets $A,B \subseteq X$.
For any $p \in [1,\infty]$, the {\it $W_p$-distance} between $\mu$ and $\nu$, $W_p(\mu, \nu)$, is defined as the $p$-th root of the minimum cost of transporting all the probability mass from $\mu$ to $\nu$, where moving a probability mass of $\delta$ from $a \in A$ to $b \in B$ incurs a cost of $\delta d(a,b)^p$.
We give a (Las Vegas) randomized algorithm that computes a $(4+\varepsilon)$-approximate $W_p$ optimal-transport (OT) plan in $O(n^2 + (n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ time with probability at least $1-1/n$, for all $p \in [1,\infty]$, where $\varepsilon > 0$ is an arbitrarily small constant and $\Delta$ is the ratio between the largest and smallest interpoint distances in $A\cup B$.
The previous best result achieved an $O(\log n)$-approximation in $O(pn^2)$ time, for constant values of $p$.
Our algorithm significantly improves the approximation factor and, importantly, is the first quadratic-time method that extends to the $W_\infty$-distance.
In contrast, additive approximation methods such as Sinkhorn are efficient only for constant $p$ and fail to handle $p=\infty$. \changed{Our algorithm also extends to a query model where, for any integer $k > 1$, we give an algorithm that preprocesses $X$ into clusters in $O(n^2+kn^{1+1/k}\log n\log\Delta)$ time, after which a $O(k)$-approximate $W_p$ distance between any two distributions $\mu$ and $\nu$ with $X$ as support can be computed in $(n^{1+1/k}\log n\log\Delta)^{1+o(1)}$ time with probability at most $1-1/n$.}
Finally, for $p=\infty$, we show that obtaining a relative approximation factor better than $2$ in $O(n^2)$ time would resolve the long-standing open problem of computing a perfect matching in an arbitrary bipartite graph in quadratic time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the problem of computing the $W_p$-distance, for $p \in [1, \infty]$, between two distributions $\mu$ and $\nu$ supported on finite point sets of a metric space. The authors give a $(4 + \epsilon)$-approximate algorithm for computing the optimal transport between $\mu$ and $\nu$ with expected runtime $O(n^2 + (n^{3/2} \epsilon^{-1} \log^2 \Delta \log U)^{1 + o(1)})$, where $\Delta$ is the ratio between the maximum and minimum pairwise distances of the support, and $U$ is the ratio between the maximum and minimum probabilities. Also, when $\mu$ and $\nu$ are uniform distributions supported on two sets of the same size, the authors give a $(4 + \epsilon)$-approximate combinatorial algorithm with expected runtime $O(n^2 \epsilon^{-2} \log^2 \Delta)$ and a $(8 + \epsilon)$-approximate combinatorial algorithm with expected runtime $O(n^2 + n^{5/3} \epsilon^{-2} \log^2 \Delta)$. Finally, the authors show that given an $O(n^2)$-time algorithm for $p = \infty$ that is either $(2 - \epsilon)$-multiplicative approximate or $(\Delta / 2 - \epsilon)$-approximate with constant $\epsilon$, then there exists an $O(n^2)$-time algorithm for computing a perfect matching in general graphs provided exists.

### Strengths
This paper studies an interesting and important problem. It is generally well-written, and the proofs are easy to follow. The technical contributions, despite mainly motivated by prior work and some classic techniques, are non-trivial and lead to improvement over prior work in terms of runtime.

### Weaknesses
Overall, I find the contributions of this paper marginal. It seems that the main technical contribution comes from the data structure given in Section 2, which at a high level is similar to Bourgain's multi-level sampling. Yet, the analysis of its guarantees follows quite straightforwardly from its construction. The remaining components of the algorithms then apply ideas from some classic algorithms.

Besides, the presentation of this paper requires considerable improvement. A related work section that places this paper into a broader context is missing. The proofs could be restructured following the theorem-lemma style instead of a plain description, and the guarantees of the data structure can be stated more explicitly.

I have some concerns regarding the correctness of the proofs (see Questions).

Detailed comments:
- Line 45-46: I believe that $W_p$ OT being a matching problem also requires $\mu$ and $\nu$ to have the supports of the same size.
- $\Delta$ is used to denote both the aspect ratio and the diameter, which might lead to confusion.
- Line 200: Should $C_y$ be $C_y[i]$? Same for $C_x$.
- Using Algorithm environments to formally describe algorithms presented in the paper would greatly improve the readability.

### Questions
- Line 62;64: Why is deciding the existence of a perfect matching in a dense graph is a simpler task?
- Is the approximation ratio $4 + \epsilon$ tight for the given algrorithm?
- Line 166: Would the degree of any particular point be as large as $nt$ instead of $n$?
- Line 178-180: Should the first term in the equation be $\Pr[p \in C_{w_j}[t] \mid w_j \in P_0 \setminus P_1]$? Also, should $j$ in the last term be $j-1$?
- Line 183-185: Should the term in the summation be $\Pr[w_s \in P_0 \setminus P_1] * Pr[p \in C_{w_s}[t] \mid w_s \in P_0 \setminus P_1]$? I believe the range of the second summation should be $s = 0, 1, \ldots, n - 1$. Finally, if the previous question is correct, should the last term be $O(1)$ now that $\Pr[w_s \in P_0 \setminus P_1] = 1 / \sqrt{n}$?
- Line 228-235: This paragraph seems confusing to me. Could you further justify its correctness and runtime? In my understanding here we should do the following: A heap for each $C$ containing $A \cap C$ is maintained, and for each $p$, a heap containing $X =  \{(a_C, C) \mid p \in C\}$ is maintained. So each update $p$ involves updating the heaps associated with $C$ that contains $p$, which might result in updates in the heaps associated with points in $C$. Please correct me if I'm wrong.
- At the end of Section 3.1, Lemma 2.3 is applied to conclude Theorem 1.1 for $p = \infty$. Yet it seems to me that Lemma 2.3 only works for $p < \infty$. Do I miss anything?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper deals with the $W_p$ optimal transport problem. In this problem we are given two node sets $A$ and $B$, a distance metric $d$ on $A \cup B$ and two probability distributions $\mu$ and $\nu$ defined on $A$ and $B$ respectively. The goal is now to move the probability mass of distribution $\mu$ from $A$ to $B$ such that $\nu$ is formed. Alternatively $\mu$ and $\nu$ could also be seen as node weights on $A$ and $B$ (each summing up to $1$) and we need to match the respective weights to each other. The respective movement or matching should then minimize the $W_p$ distance with respect to the distance metric $d$.

The authors point our that it is already known that the problem can be solved exactly with running time $O\left(n^{2+ o(1)}\right)$ using a min-cost flow algorithm. However, this algorithm is not feasible in practice and there has been significant research over the last years both trying to find practical algorithms as well as trying to obtain a theoretical running time in $O(n^2)$, even if this means that not the optimum solution is found.

The main contribution of the submission is a randomized $2$-layer clustering scheme that allows to create a weighted directed graph of expected size $O(n^{3/2})$ such that the shortest path distance between two nodes $a \in A$ and $b \in B$ in this graph is at least $d(a,b)$ and at most $(4 + \epsilon) d(a,b)$. Crucially all paths from $a$ to $b$ only contain a single weighted edge. As a result also the value $d(a,b)^p$ gets only distorted by a value of $(4+\epsilon)^p$ by this construction (for $p \in \mathbb{N}$). 

Afterwards the authors apply the min cost flow algorithm by Chen et al. using these new distances. Given that the cost of the algorithm depends on the number of edges in the respective graph which got reduced from $O(n^{2})$ to $O(n^{3/2})$, this improves the running time accordingly. The distance graph can be calculated in $O(n^2)$, and the author end up with an $(4+ \epsilon)$-approximation with expected running time $O(n^2)$ under reasonable assumptions. For the case that $p = \infty$ they provide an alternative approach with a similar running time.

Besides this, the submission provides an algorithm for the $W_p$ matching problem, which can be seen as the special case of the $W_p$ optimal transport problem where both $\mu$ and $\nu$ are uniform distributions. The advantage of this algorithm seems to be that it has a reasonable practical running time even though the theoretical running time is mostly comparable to the more general algorithm. The approximation ratio stays $(4 +\epsilon)$ (or $(8 + \epsilon)$ respectively). It is also shown that if one finds an $(2 - \epsilon)$-approximation algorithm for the $W_\infty$ matching problem in $O(n^2)$, this would imply that one could find a perfect matching in bipartite graphs in $O(n^2)$. This can be seen as a hardness result.

For their experiments the authors generated data sets using either a uniform or a truncated normal distribution in up to 10 dimensions. They do not evaluate the practical performance of their $W_p$ optimal-transport algorithm directly (probably due to the inefficient min-cost flow algorithm). Instead they only evaluated the quality of the $2$-layer clustering scheme. They showed that the size of the clustering fits the expected theoretical bounds and obtained that in their experiments the maximum distance distortion of this clustering often was between $3$ and $3.5$ while the average distortion was closer to $1.5$. They also provided some experiments for their $W_p$ matching algorithm with the approximation ratios seemingly approaching values between $2$ and $2.5$ with larger number of nodes (at most 8000). An exception seems to be the algorithm for $p = \infty$. Here the ratio reaches the value of $3$ and it is unclear if the value would increase even more for larger point sets. The algorithm also seems to be reasonably efficient.

### Strengths
The paper contains non-trivial theoretical results. At first glance the improvement from an exact $O(n^{2+ o(1)})$ algorithm to an $O(n^2)$-approximation algorithm (under certain assumptions) seems to be not that impressive. However, the authors provide a lot of literature that deal with this problem and their algorithm improves the best existing result significantly.

### Weaknesses
The fact that the new $W_p$ optimal-transport algorithm was not tested in the experiments seems to indicate that it is very inefficient in practice. It is unfortunate that the authors did not provide experimental results for existing algorithms for a better comparison.

### Questions
- The data structures for the proximity queries are rather involved and are not directly necessary for the optimal-transport algorithm. Maybe one could improve the readability of the paper by directly providing the algorithm after the introduction of the layer clustering and present the query structures afterwards. This would also be beneficial because the need to answer these queries only gets clear once the matching algorithm gets presented. Thus presenting this algorithm directly afterwards could be helpful.

- Given that the paper focuses a lot on improving the running time it could make sense to provide a summary of the analysis of the running time of the $W_p$ optimal transport algorithm somewhere in the paper or the Appendix (which could also discusses the running time for computing the clustering). 

- In the experimental section there is no comparison with existing algorithms which would have been very helpful to judge whether the new results also yield improvements in practice.

- In Theorem 1.3 the authors say that one could calculate the perfect matching for an arbitrary graphs (if it exists) but in the appendix they only prove that one could find the perfect matching in a bipartite graph.

- In line 471 the authors write that the $W_p$ matching algorithm performs 'near optimal' in practice. This seems to be a bit of a stretch for approximation ratios around $2$.

### Soundness
3

### Presentation
3

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
This paper studies the problem of computing the $p$-Wasserstein distance in general metric spaces for any $p\in[1,+\infty]$. The problem can be formulated as a minimum-cost flow problem and therefore can be solved exactly in $n^{2+o(1)}$ time using the sophisticated and impractical theoretical algorithm of Chen et al. (FOCS 2022). This paper considers approximation algorithms for the Wasserstein distance and obtains near-quadratic $\tilde{O}(n^2)$-time algorithms with $4+\epsilon$ approximation ratio, where $n$ is the support size of the two input distributions. This improves upon the previous $O(\log n)$-approximation in $\tilde{O}(n^2)$ time (Lahn et al., ICML 2025). The authors also present a simpler and more practical algorithm for the special case where the input distributions are uniform (i.e., the $W_p$ matching problem), as well as a nearly matching conditional lower bound.

Technically, both algorithms build on a clustering scheme obtained via a multi-level sampling procedure inspired by Bourgain (1985). The authors use this scheme to

- construct a directed spanner with $m = \tilde{O}(n^{3/2})$ edges that preserves the $p$-power metric $d(\cdot,\cdot)^p$, and by combining it with the $m^{1+o(1)}$-time algorithm for graphs (Chen et al., FOCS 2022), obtain a near-quadratic $\tilde{O}(n^2)$-time algorithm for $p$-Wasserstein distance; and

- design data structures for weighted nearest-neighbor search and dynamic bichromatic closest-pair queries, which are then used to speed up the Gabow–Tarjan bipartite matching algorithm, leading to a near-quadratic $\tilde{O}(n^2)$-time algorithm for $W_p$ matching.

### Strengths
- The paper addresses the efficiency challenge of the fundamental $p$-Wasserstein distance problem and achieves significant improvements (reducing the approximation ratio from $O(\log n)$ to  $O(1)$). Moreover, their algorithms are general and work for any $p \in [1,+\infty]$, whereas the previous $O(\log n)$-approximation only works for small $p$. 
- The algorithms are also practical, as demonstrated by the experimental results.
- The results are also complete, as the authors additionally provide a (conditional) lower bound.
- The paper is well-written and I can easily follow the presentation and understand the main idea of the algorithm and its analysis.

### Weaknesses
The only weakness in my view is that the technical contribution is not clearly explained. The paper’s main technique, a clustering scheme via multi-level sampling, seems largely based on prior work (e.g., Bourgain, 1985). The directed spanner construction and the data structures appear to follow naturally from this clustering scheme. It would be helpful if the authors could more explicitly highlight their technical novelty (for example, by clarifying how their multi-level sampling differs from prior work). 

In addition, I believe the experimental section could be further improved. In particular, it would be useful to compare against the previous $O(\log n)$-approximation algorithm of Lahn et al. (ICML 2025), and to evaluate the algorithm on some real-world datasets.

### Questions
In the “Algorithm efficiency” paragraph of the experiment section, the authors report the number of operations but not the actual running time. Could the authors clarify the motivation for this choice? 

Moreover, in Line 465 the paper states that *“Combined with the $O(n^2)$ per-query complexity … the algorithm runs in quadratic time ...”*. However, the results seem to indicate that there are at least $n^{3/2}$ queries, which would imply a total running time of $O(n^{3.5})$ rather than $O(n^2)$. Is this a typo, or am I misunderstanding something?

### Soundness
4

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
4

### Summary
The paper presents two constant factor approximation algorithms for optimal transport. The first one, relies on computing a min-cost flow, but does so on a smaller graph that approximates distances of the full graph. The second algorithm is for the restricted setting where the distributions are uniform.

### Strengths
Optimal transport is a heavily studied and new algorithmic insights should always be welcome.

### Weaknesses
The paper emphasizes the "scalable" nature of the algorithm, but the general algorithm still relies on Chen et al. for computing a min-cost flow. Moreover, I believe some more detail is needed to justify the expected running time. A graph is constructed, which has m edges in *expectation* and then the algorithm by Chen et al. is applied to that graph. However, as far as I can see, this does not immediately result in an algorithm that runs in time $m^{1+o(1)}$. To conclude this, some concentration bound on the number of edges or some suitable strict upper bound on the number of edges would have to be used. In any case, this would require a little more justification.

For dense graphs, there are also other nearly linear time algorithms available (van den Brand et al. STOC 2021). Maybe the current submission can claim to save polylogarithmic factors for maximally dense graphs? But the paper presents itself as aiming to be more practical and a result like that would not necessarily qualify for a practical improvement.

Not surprisingly, no experiments are included for the general optimal transport algorithm.

I am not claiming that the theoretical contribution is without merit (after adding an appropriate justification for the running time bound), but that there is a gap between the claimed practical focus and this result.

The algorithm for uniform distributions does not rely on Chen et al. and so the critique above does not apply.

### Questions
In line with the above, could please give a more detailed justification on the claimed running time?

Do you have experiments comparing your matching algorithms to others both in terms of efficiency and accuracy?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents the first scalable, constant-factor approximation algorithm for the $W_{p}$ optimal transport (OT) problem that runs in nearly quadratic time. This method significantly improves upon previous $O(\log n)$-approximation algorithms and is the first efficient, quadratic-time method to extend to the $W_{\infty}$ distance.

** Problem formulation.** 

The paper addresses the $W_{p}$ optimal transport problem for discrete probability distributions $\mu$ and $\nu$, supported on finite point sets $A$ and $B$ within a metric space $(X,d)$. The objective is to find a transport plan $\sigma: A \times B \rightarrow R_{\geq 0}$ that minimizes the $W_{p}$ cost, defined as $w_{p}(\sigma):=(\sum_{a\in A,b\in B}\sigma(a,b)\times d(a,b)^{p})^{1/p}$. This formulation also covers the $W_{\infty}$ cost, which is the limit as $p \rightarrow \infty$ and represents the maximum distance $d(a,b)$ for which $\sigma(a,b) > 0$.

** Main results ** 

The main result (Theorem 1.1) is a randomized algorithm that computes a $(4+\epsilon)$-approximate $W_{p}$ optimal transport plan in $O(n^{2}+(n^{3/2}\epsilon^{-1}\log^{2}\Delta~\log U)^{1+o(1)})$ expected time for any $p \in [1, \infty]$. Additionally, a simpler $\tilde{O}(n^{2})$ combinatorial algorithm is provided for the $W_p$ matching problem (when distributions are uniform).

** Technique/algorithm **

The core technique is a two-layered clustering scheme inspired by Bourgain's multi-level sampling, which approximates the cost function $d(\cdot,\cdot)^p$. The primary algorithm constructs a directed spanner graph based on these clusters and computes an approximate solution by running a minimum-cost max-flow algorithm on this graph. A second, simpler algorithm for the matching problem uses the same clustering to build efficient dynamic data structures (for bichromatic closest pair and weighted nearest neighbor queries) which are then used within a Gabow-Tarjan cost-scaling matching framework.

** Experiment sumamry **

The paper provides an empirical evaluation of the simpler combinatorial matching algorithm (from Section 3.2) on synthetic data. These experiments demonstrate that the algorithm's practical approximation ratio is consistently much better than the theoretical $(4+\epsilon)$ worst-case bound (typically around 1.5-2.0) and that the runtime scales quadratically, as predicted by the analysis.

### Strengths
The theoretical results are solid and also discusses its potential practical application

### Weaknesses
.

### Questions
.

### Soundness
2

### Presentation
2

### Contribution
2
