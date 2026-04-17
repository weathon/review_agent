# K^*-means: a parameter-free clustering algorithm

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 0

## Abstract
Clustering is a widely used and powerful machine learning technique, but its effectiveness is often limited by the need to specify the number of clusters, k, or by relying on thresholds that implicitly determine k. We introduce k*-means, a novel clustering algorithm that eliminates the need to set k or any other parameters. Instead, it uses the minimum description length principle to automatically determine the optimal number of clusters, k*, by splitting and merging clusters at the same time as optimizing the standard k-means objective. We prove that k*-means is guaranteed to converge and demonstrate experimentally that it significantly outperforms existing methods in scenarios where k is unknown. We also show that it is accurate in estimating k, and that empirically its runtime is competitive with existing methods, and scales well with dataset size.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes k*means, a parameter-free variant of k-means that automatically determines the number of clusters through the minimization of an objective (eq. 1) that involves a k-means loss term and an MDL term. The interesting aspect of the method is that it includes split and merge operations in addition to typical k-means operations that increase or decrease k always ensuring that the proposed objective is improved. Therefore, it is guaranteed that this dynamic method will converge to a minimum of the objective.
Actually, the method extends x-means which is an old incremental method (only cluster splitting takes place) based on BIC/MDL.

### Strengths
S1. The method automatically estimates the number of clusters k.
S2. It does not make use of any hyperparameter
S3. All operations ensure the minimization of the objective function.

### Weaknesses
W1. Although MDL is theoretically justified, it is not considered efficient or robust in practical finite-sample settings. For this reason the MDL-based x-means algorithm has been first replaced by a Gaussianity-based criterion (g-means algorithm) and later by a unimodality-based criterion (dip-means algorithm). 
W2. Related work omits significant contributions in number of clusters estimation, such as g-means, dip-means and  more recently the Uniforce and DipDeck algorithms to mention some of them.
W3. The experimental part ignores the majority of well-known methods that conduct automatic number of clusters estimation. 
W4. Since the method starts with a single cluster, the usefulness of the merging operation is questionable.

### Questions
Q1. Experimental comparison ignores several well-known or state-of-the-art methods for number of clusters estimation (see W2). Some methods are available in the ClustPy library.
Q2. The method should be tested on datasets with imbalanced clusters and/or not clearly separated clusters.
Q3. More complex datasets should be considered in the clustering comparison. In the present work dimensionality reduction takes place 
to tackle increased dimensionality, thus the clustered datasets are low-dimensional.
Q4. The method tends to underestimate the number of clusters. This is due to the fact that the MDL term is relatively big resulting in conservative estimations. This is in my opinion the main drawback of the proposed approach.
Q5. How often is merging actually triggered in the experiments? Could you provide ablation results (with vs. without merging) to demonstrate its impact?

### Soundness
3

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
This paper presents an extension of k-means that infers the appropriate number k of clusters. The algorithm is a heuristic method to minimize the description length of the input data.

### Strengths
The proposed approach is sensible and the algorithm works quite well in the experiments, which are designed reasonably well.

### Weaknesses
- This paper could have been written 20 years ago. It does not really engage with the current research in machine learning that is most relevant to ICLR. 

- The algorithm is fundamentally heuristic. The proof of convergence says only that the method reaches a local optimum after a finite number of steps. The bound on performance in Appendix D is potentially more interesting. However, the bound is not stated as a formal theorem, and the proof is long enough that this reviewer has not checked it. Lines 826 and 942 appear to say that the proof is merely about the initialization, and therefore not about the new algorithm in a substantive way. Equation 5 is complicated enough that it is not obvious when the RHS provides a trivial bound and when it is non-trivial.

- Since the algorithm is heuristic, the authors should describe how it is better than obvious variations on the same theme. In particular, it seems inefficient to consider splitting every center at every iteration. A different heuristic would be to consider splitting just the highest-cost center. However, something else missing in this submission is a discussion of time complexity (Table 3 is only empirical). At first sight, the "assign" step is much more expensive than the "update" step", which itself is much more expensive than the splitting and merging operations, because the latter involve computations on scalars only (not vectors of length d). So perhaps MDL would be minimized better and faster with more splitting and merging operations, and fewer "update" steps, and/or "update" steps that operate only on points with changed centers.

- The subroutines INITSUBCENTROIDS and KMEANSSTEP are described generally, but not given explicitly as pseudocode


Missing comparisons to related research:
- Unsupervised learning using MML by JJ Oliver, RA Baxter, CS Wallace, ICML, 1996 
- P.Kontkanen, P.Myllymaki, W.Buntine, J.Rissanen, H.Tirri, An MDL Framework for Data Clustering. In Advances in Minimum Description Length: Theory and Applications, edited by P. Gr ̈unwald, I.J. Myung and M. Pitt. MIT Press, 2005
- A novel split and merge EM algorithm for gaussian mixture model by Y Li, L Li, 2009 Fifth International Conference on Natural Computation

### Questions
125: This line is misleading because under certain conditions, BIC and MDL are equivalent.

161: Explain briefly why and how the Kraft-McMillan inequality is relevant and used.

162, 271: What does "noend" mean?

172: What does the notation [ ] mean in the RHS [{...}, {...}]?

186: What does max(X) mean given that X is a set of high-dimensional points, not of scalars?

203: How sensitive is the algorithm to the value of m? Why is this the best value of m?

336: Should be ImageNette and not ImageNet?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces a parameter-free clustering algorithm named k*means, which aims to eliminate the need for manual specification of the cluster number k and does not introduce any other parameters. The authors provide a theoretical proof of the algorithm's convergence within a finite time. Furthermore, through comprehensive experiments on diverse multimodal datasets, they demonstrate that k*means outperforms most established baseline methods.

### Strengths
They introduce k∗means, an entirely parameter-free clustering algorithm
For carefully constructed synthetic data, k∗means can infer the true number of clusters, and shows much higher accuracy than existing methods;

### Weaknesses
The performance advantage of the proposed algorithm over existing methods appears to be marginal. As shown in Table 3, the accuracy of 
k*means does not demonstrate a significant improvement across most datasets.

The synthetic experiments are limited to clusters generated from multivariate normal distributions with fixed variance. 
The algorithm's robustness remains unverified under more complex and challenging data scenarios, such as non-convex clusters (e.g., ring-shaped), uneven cluster densities, or datasets with a high proportion of noise (e.g., >20%).

The literature review primarily contrasts the method with traditional parameter-free algorithms. However, it omits comparisons with recent advancements in this direction (e.g., PFMVKM) as well as other relevant classical algorithms (e.g., DPC). This omission may lead to an incomplete and potentially less convincing performance evaluation.

A theoretical analysis of the algorithm's time and space complexity is absent, which is crucial for understanding its scalability and practical utility.

Minor Concerns:

Some algorithms included in the empirical comparison (e.g., DPMM in Table 3) are not introduced in the Related Work section.

The algorithm name "K-Means++" is incorrectly written as "k++means" in multiple instances throughout the manuscript and should be corrected.

Table 1: There is excessive blank space above the X-Means row.

Table 5: The table appears to be cut off and is not displayed fully.
The same issue occurs with the equation on Page 17.

### Questions
NaN

### Soundness
2

### Presentation
3

### Contribution
1
