# Mixed-Curvature Tree-Sliced Wasserstein Distance

- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Mixed-curvature spaces have emerged as a powerful alternative to their Euclidean counterpart, enabling data representations better aligned with the intrinsic structure of complex datasets. However, comparing probability distributions in such spaces remains underdeveloped: existing measures such as KL divergence and Wasserstein either rely on strong assumptions on distributions or incur high computational costs. The Sliced-Wasserstein (SW) framework provides an alternative approach for constructing distributional distances; however, its reliance on one-dimensional projections limits its ability to capture the geometry of the ambient space. To address this limitation, the Tree-Sliced Wasserstein (TSW) framework employs tree structures as a richer projected space. Motivated by the intuition that such a space is particularly suitable for representing the geometric properties of mixed-curvature manifolds, we introduce the Mixed-Curvature Tree-Sliced Wasserstein (MC-TSW), a novel discrepancy measure that is computationally efficient while faithfully capturing both the topological and geometric structures of mixed-curvature spaces. Specifically, we introduce an adaptation of tree systems and Radon transform to mixed-curvature spaces, which yields a closed-form solution for the optimal transport problem on the tree system. We further provide theoretical analysis on the properties of the Radon transform and the MC-TSW distance. Experimental results demonstrate that MC-TSW improves distributional comparisons over product-space-based distance and line-based baselines, and that mixed-curvature representations consistently outperform constant-curvature alternatives, highlighting their importance for modeling complex datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an algorithm to construct a tree when the data lies in a mixed-curvature space. Then, using this tree metric, this paper proposed computing the tree-wasserstein distance efficiently.

### Strengths
* This paper proposed a novel method to construct the tree.
* The proposed methods can capture the structure of Muxture-curvature spaces, and the effectiveness of the proposed methods was demonstrated in several tasks: gradient flow and self-supervised learning.

### Weaknesses
* This paper did not evaluate the approximation error of the proposed method and the original Wasserstein distance. For instance, can the proposed method more accurately approximate the Wasserstein distance than the tree-sliced Wasserstein distance with the tree metric constructed by the clustering method [4]? This paper evaluated the performance of the proposed method in several tasks; however, these results are influenced by various factors. It would be good if this paper could demonstrate the effectiveness of the proposed method in the simplest task.
* Various methods have been proposed for constructing tree metrics for the tree-Wasserstein distance, e.g., [1,2,3,4], but this paper did not discuss the relationship between the proposed method and these existing approaches. Although the proposed method appears to differ from these existing approaches, it is necessary to discuss its relationship to them.
* In the experiments, what methods did this paper use for TSW to construct the tree metric?
* This paper did not evaluate the computational efficiency of the proposed method. In lines 378-388, this paper discusses the time complexity; however, can this paper also report the time required for training in the experiments?

## Reference
[1] Indyk et. al., Fast image retrieval via embeddings. In International Workshop on Statistical and Computational Theories of Vision 2003

[2] Takezawa et. al., Supervised tree-wasserstein distance. In International Conference on Machine Learning 2021.

[3] Lin et. al., Tree-wasserstein distance for high dimensional data with a latent feature hierarchy, In International Conference on Learning Representation 2021.

[4] Le et. al., Tree-sliced variants of wasserstein distances, In Neural Information Processing Systems 2019

### Questions
See the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
By utilizing mixed-curvature spaces that enable better data representations aligned with complex underlying structures—rather than relying solely on the Euclidean setting—this paper introduces the Mixed-Curvature Tree-Sliced Wasserstein, a computationally efficient discrepancy measure between distributions. The proposed technique builds upon the ideas behind the classical Sliced-Wasserstein (SW) framework, leading the authors to introduce a Radon-type transform. The paper demonstrates the effectiveness of this new tool through experiments involving generative flow models and variational autoencoders.

### Strengths
- The paper is very well organized.
- The paper is thorough and complete: it presents solid theoretical results, clearly introduces the necessary notions and new definitions, establishes key properties, and provides simplifications that enhance its applicability. It also includes extensive references to related work, provides time complexity analyses, and validates the proposed framework through different experiments.
- The idea is novel and has the potential to be applied to data analysis across various disciplines.

### Weaknesses
The only potential weakness is that the paper might appear overly technical at times; for example, it would benefit from including a brief explanation of how to obtain the closed-form expression in Equation (21).

### Questions
- Although I understand the general idea, I got lost and could not clearly identify where the stereographic projection $\rho_k$ is utilized in the methodology. Measures are assumed already in the hyperspher $F$ Poncare ball P
- Could the authors clarify what they mean by “consistency of operators across geometries” in line 155?
- Is there any intuition or guideline on how to choose the parameters $k$ and $m$?

Style Suggestions:
- The paper alternates between using MC-TSW and MCTSW. I suggest unifying the notation throughout the text.
- I would reserve the symbol $\rho$ exclusively for the projection in Equation (3). For instance, I recommend using a different symbol for the Sliced-Wasserstein operator in Equation (8), as well as later in Equation (18) and similar instances.

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
The central hypothesis is that the Tree-Sliced Wasserstein framework can be generalized to mixed curvature spaces (MCS). The core technical contributions include a mixed-curvature tree system $\mathcal{T}$ and a Radon Transform $\mathcal{R}^{\alpha}$ for MCS, which maps mass from the manifold $\mathcal{M}$ onto $\mathcal{T}$. The authors claim this construction yields a closed-form, computationally efficient solution for the 1-Wasserstein distance on the tree. They propose that this MC-TSW metric will more faithfully capture the joint geometric structure of MCS distributions than separable baselines (like product-space SW/TSW), leading to superior performance.

### Strengths
- The paper is nicely written and tackles a well-motivated and non-trivial problem. The authors correctly identifies the limitations of naive baselines. That is, ambient-space metrics ignore the geometry, while separable "product" metrics (like Prod-TSW) ignore joint structure across components. This work appears to be the first to build a TSW-like distance specifically for these joint spaces.


- Good experimental results. For example, 

  - the gradient flow experiment provides strong evidence for the paper’s central claim. In Table 1, the MC-TSW method successfully converges to a low $\log W_2$ error ($-3.65$). In stark contrast, the separable baselines, Prod-SW and Prod-TSW, fail entirely (positive $\log W_2$ errors). This result seems to suggest that the target distribution have cross-component correlations that only a joint metric like MC-TSW can capture.


  - the VAE and graph SSL experiments (Tables 2 & 3) further show that combining the proposed MC-TSW regularizer with an MCS latent space yields SOTA or highly competitive results, outperforming both KL-based regularizers and single-curvature (Euclidean, spherical, hyperbolic) models. This validates both the new metric and the underlying geometric hypothesis.

- The algorithm has the same asymptotic cost as TSW, parallelizes well on GPUs.

### Weaknesses
- The Radon transform (def 3.6) projects every point $z \in \mathcal{M}$ to the same coordinate $t = d_{\mathcal{M}}(x, z)$ on all $k$ rays of the tree. The only thing distinguishing the projection of $z$ onto different rays is the splitting map $\alpha(z, \mathcal{T})_i$. This spherical projection collapses all angular and component-specific directional information of $z$ relative to the root $x$, reducing it to a single scalar distance. This appears to be a geometrically lossy projection, and its trade-off is not discussed.

- Moreover, the paper's construction is limited to star-shaped trees at a single point $x$. The implementation in algo 1 simplifies this further by sampling rays that are axis-aligned, i.e., each ray extends in exactly one component $j$ of the product space. While the projection onto this tree (via the joint distance $d_{\mathcal{M}}$ and splitting map $\alpha$) is still joint, the tree structure itself is highly constrained and separable. This axis-aligned, star-shaped probe may not be the most effective geometry for capturing complex, non-axis-aligned correlations, even if it is computationally convenient.

- The $O(L n \log n)$ component of the complexity is a direct result of this information-collapsing projection. Because all points $a_j$ are projected to the *same* coordinate $c_j = d_{\mathcal{M}}(x, a_j)$ on every ray, a single sorting of the $n$ values $\{c_j\}$ is sufficient to compute the $W_1$ on the tree (Eq. 21). This tractability is thus achieved by collapsing all per-ray projection information into a single scalar distance.

### Questions
- The success in Table 1 suggests the joint projection (via $d_{\mathcal{M}}$ and $\alpha$) is doing all the work.  Does this imply the tree structure itself is less important than the projection function?

### Soundness
3

### Presentation
3

### Contribution
3
