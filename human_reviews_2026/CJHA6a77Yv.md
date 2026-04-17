# Riemannian Fuzzy K-Means on Product Manifolds

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
In this paper, we address an open problem: how to perform fast clustering on product manifolds.} With the increasing interest in non-Euclidean data representations, clustering such data has become an important problem. However, a naive extension of the classic K-Means algorithm to product manifolds requires $\mathcal{O}(\nu \omega)$ time, where $\omega$ is the number of alternating iterations and $\nu $ is the time complexity of each Riemannian optimization. Due to the need for numerous Riemannian optimizations, the naive Riemannian K-Means (NRK) is not suitable for large-scale data. To this end, we propose the Riemannian Fuzzy K-Means (RFK) algorithm for product manifolds, which reduces the time complexity to $\mathcal{O}(\nu )$. Importantly, RFK is not a straightforward extension of K-Means or Fuzzy K-Means to manifolds, it avoids the computation of the Fréchet mean and and achieve a true single-loop optimization. Furthermore, we introduce Radan to accelerate the optimization of RFK. We conduct extensive experiments. RFK and Radan outperform across nearly all metrics in almost every dataset, reaching an impressive level of performance. \textbf{RFK and Radan have been integrated into several non-Euclidean machine learning libraries, such as here.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a Riemannian Fuzzy k-means (RFK) algorithm for fast clustering on product manifolds, extending the conventional fuzzy k-means to Riemannian manifolds. This approach reduces the time complexity compared to the naïve Riemannian k-means (NRK) algorithm, by removing alternating iterations. In addition, the authors introduce Radan, an adaptive optimization algorithm on manifolds, as a generalization of Adan. Experimental results suggest that RFK achieves lower computational complexity than NRK, faster convergence than Riemannian Adam (Radam), and improved clustering performance on both synthetic and real-world datasets.

### Strengths
- The proposed algorithms seem novel and original, though they are relatively straightforward extensions of fuzzy k-means and Adan to the Riemannian setting.
- The computational efficiency of RFK relative to NRK is clearly established, with experimental results providing convincing evidence of its advantage.
- Most of the mathematical derivations seem correct.
- The paper is clearly written and generally well organized.

### Weaknesses
- **Limited comparison to Riemannian baselines:** It is unclear whether the compared algorithms are also formulated on Riemannian manifolds. The authors should explicitly state whether the baselines take into account the manifold geometry. To better demonstrate the advantage of RFK, additional comparisons with other Riemannian clustering algorithms beyond NRK are needed.

  To name a few, representative Riemannian clustering methods include Subbarao and Meer (2009), Nonlinear Mean Shift over Riemannian Manifolds (IJCV), Ashizawa et al. (2017), Least-squares Log-density Gradient Clustering for Riemannian Manifolds (AISTATS), and Zhao et al. (2016), Efficient Clustering on Riemannian Manifolds: A Kernelised Random Projection Approach (Pattern Recognition).
- **Marginal performance gains:**
In several cases, the improvement over NRK is limited, and it is not evident why RFK outperforms NRK given their similar objectives. No error bars are provided, and for some datasets (e.g., CiteSeer and Cora), RFK performs worse in terms of NMI and F1. Further analysis and discussion would help clarify these inconsistencies.
- **Unclear explanations and missing discussions:**
  - The reason why RFK shows faster convergence than Radam is not well explained.
  - The paper lacks a discussion of limitations, such as the sensitivity to the number of clusters or convergence to local minima.
  - Many results are marked as out-of-memory (OM), yet the experimental setup is not described in sufficient detail. It is unclear why NRK suffers from OM—perhaps a memory analysis or ablation would clarify this.
  - The description of parallel transport omits the specification of the curve along which vectors are transported.

### Questions
Please refer to the points raised in the weaknesses section.

In addition, would RFK be effective if the underlying manifold consists of SPD matrices or other manifolds without closed-form geodesics? Some remarks on these points would be helpful.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the open problem of efficient clustering on product manifolds by proposing Riemannian Fuzzy K-Means (RFK) and an accelerated optimizer Radan. By leveraging the fuzzy relaxation of cluster assignments, RFK eliminates the alternating updates required in Naive Riemannian K-Means (NRK), reducing time complexity from O(νω) to O(ν). Extensive experiments demonstrate significant speedups and superior clustering performance across diverse datasets.

### Strengths
1.It is practical to use fuzzy relaxation technology to transform the double-loop optimization of NRK into a single loop, reducing the time complexity from O(νω) to O(ν) while avoiding the problem of cluster centers exceeding the manifold.

2.The Adan optimizer is adapted to obtain Radan, which is suitable for product manifolds through strategies such as parallel transport and scalar second-moment maintenance. Regret bounds and convergence proofs are provided to ensure the theoretical reliability of the algorithm.

3.The paper includes extensive experiments across multiple datasets, demonstrating the superiority of RFK and Radan over existing clustering methods.

### Weaknesses
1.Lack of Comparison with Recent Clustering Algorithms:
Although the paper compares several methods, it appears to lack a comparison with the latest clustering algorithms specifically designed for non-Euclidean spaces.

2.Unclear Explanations in Several Sections:
Many parts of the paper lack clarity. For instance, in Chapter 2, since there are multiple isometric models of hyperbolic space, it is essential to explicitly specify which hyperbolic space model is being used. Additionally, in the regret bound proof of Theorem 3.1, the explicit form and range of values of the curvature functionζ(κ, c) are not sufficiently detailed.

### Questions
Convergence Proof of Radan (Theorem 3.2) and Fixed Hyperparameters:
The convergence proof for Radan (Theorem 3.2) assumes decaying hyperparameters (e.g., β3t= 1 - 1/t), but all experiments use fixed values (e.g., β3= 0.99). The authors claim this is “standard practice”, yet they provide no justification or reference to support that convergence holds under fixed hyperparameters, creating a gap between the theoretical analysis and practical implementation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
To address the high time complexity of Naive Riemannian K-Means (NRK), this paper proposes a Riemannian Fuzzy K-Means (RFK) method that relies solely on cluster centers for single-loop updates. This method avoids alternating updates between the membership matrix and cluster centers used in traditional approaches. This method reduces computational complexity. Meanwhile, the authors also introduce a Riemannian Adan (Radan) optimizer, which is designed for product manifolds to further accelerate the convergence of RFK.

### Strengths
1. The paper provides a clear derivation process from NRK to RFK, and also presents a convergence proof for the proposed Radan optimizer.
2. Experimental results demonstrate that the method indeed improves both clustering efficiency and accuracy.

### Weaknesses
1. Although the authors emphasize that RFK is not a simple extension of Fuzzy K-Means, fuzzy clustering in Euclidean space has been extensively studied.
2. There are some typos. For example, the reference “Lin et al.” on line 1449 is missing the publication year.
3. Some details should be provided. For example, is there any additional data pre-processing or pre-training before training? For another, how are the cluster centers initialized in RFK, and does this initialization affect the clustering results?

### Questions
1. The convergence proof relies on the convexity assumption on the manifold. Does this assumption still hold on non-convex manifolds?

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
4

### Summary
The authors present an application of the Fuzzy K-means algorithm using Riemanian distances. They use an indicator parameter that represents cluster belonging and is expressed in a closed form equation. By utilizing distances derived from several manifold representations, the Fuzzy K-means method is applied to the data. The authors presented results on several experiments with Gaussian synthetic data, graph data, and VAE embeddings of several benchmark datasets.

### Strengths
The combination of multiple geodesic distances, assuming Euclidean, Hyperspherical, and hyperbolic manifolds, showed improvements in clustering performance. The application of closed-form expressions for the geodesic distances and class belonging parameters is a main contribution of the paper.

### Weaknesses
However, there are several reasons why I cannot recommend acceptance of the proposed algorithm. The authors combined distances from different manifolds, assuming distinct manifold structures, but they did not provide any justification why it is valid to assume multiple manifold types for the same dataset. A more fundamental question is whether clustering performance can be improved even when ground-truth manifold distances are used. K-means can easily fail even with data having no manifold structure.

Can the authors relate their method to spectral clustering or a clustering in a feature space associated with some kernels? It is difficult to identify a significant contribution given that this is a well-studied problem, while the paper does not sufficiently engage with the existing literature.

### Questions
I did not examine all details in the Appendix, but it is unclear how the authors derived the closed-form expressions for the distances on each manifold. Did they first project the data onto the respective manifold and then computed the geodesic distances? 

The closed form solution for the cluster belonging parameter u appears to diverge with m=1. In fact, the hyperparameter m may not being meaningful, because any u^m satisfies the same condition as u, and u^m itself could serve as a new parameter u without the need for m.

### Soundness
1

### Presentation
1

### Contribution
2
