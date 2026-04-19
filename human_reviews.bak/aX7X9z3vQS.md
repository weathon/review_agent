# Recovering Manifold Structure Using Ollivier Ricci Curvature

- Decision: Accept (Spotlight)
- Scores: 8, 6, 8

## Abstract
We introduce ORC-ManL, a new algorithm to prune spurious edges from nearest neighbor graphs using a criterion based on Ollivier-Ricci curvature and estimated metric distortion. Our motivation comes from manifold learning: we show that when the data generating the nearest-neighbor graph consists of noisy samples from a low-dimensional manifold, edges that shortcut through the ambient space have more negative Ollivier-Ricci curvature than edges that lie along the data manifold. We demonstrate that our method outperforms alternative pruning methods and that it significantly improves performance on many downstream geometric data analysis tasks that use nearest neighbor graphs as input. Specifically, we evaluate on manifold learning, persistent homology, dimension estimation, and others. We also show that ORC-ManL can be used to improve clustering and manifold learning of single-cell RNA sequencing data. Finally, we provide empirical convergence experiments that support our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors propose using the Ollivier-Ricci curvature, calculated for each edge of graph, to prune graphs in machine learning methods that construct local graphs from data points. They develop an algorithm that removes edges that, due to noise, create shortcuts connecting the low-dimensional manifold formed by the data. Using synthetic data where the shortcuts are known, they demonstrate the ability to identify these shortcuts. Furthermore, they experimentally show that this approach is useful for tasks such as preprocessing for manifold learning methods like ISOMAP and UMAP, computing persistent homology, and intrinsic dimension estimation.

### Strengths
Providing a method that can accurately prune shortcut edges—which significantly affect the performance of downstream tasks—using a relatively simple algorithm will have a major impact on many machine learning algorithms based on local graphs.

### Weaknesses
The presentation of the paper has considerable room for improvement.

Given the nature of the research, it's somewhat understandable that most of the content is described in the Appendix. However, it is undesirable that figures providing intuitive explanations of geometric concepts and constants (thresholds) that play important roles in the algorithm cannot be interpreted without consulting the Appendix. For example, rather than including numerous detailed experimental results, adding figures that intuitively explain the definition and properties of the minimum branch separation would contribute more to understanding after making rooms by moving one of two experimental results to the Appendix. Additionally, the fact that the basis for comparing with ORC, '-1 + 4(1 - δ)', is provided in Lemma A.1 hinders the understanding of the proposed algorithm.

The (weighted) shortest-path metrics \( d_G \) and \( d_{\mathcal{M}} \) are not formally defined. These are very important quantities, and the discussion should proceed after giving precise definitions of them.

In Figures 4, 5, and 6, some words (upper left part of Figures) are displayed overlapping, making it impossible to discern what is written.

It has been pointed out that intrinsic dimension estimation using the MLE by Levina & Bickel has bias.
https://www.inference.org.uk/mackay/dimension/
If the claim is that the proposed ORC-MANL preprocessing contributes to correct intrinsic dimension estimation, then a bias-reduced estimator should be used and discuss how their results might change with a different estimator.

### Questions
Major question:
Since the proposed ORC-based method is based on the idea of removing edges with highly negative curvature, wouldn't it perform poorly when the graph constructed using the epsilon-radius rule has an almost tree-like topology (i.e., graphs with very few cycles or triangles)? In other words, in graphs that are close to tree structures, a majority of edges may have negative curvature, and as a result, wouldn't the ORC-based method over-prune the graph?


Minor questions:
What does ORC-MANL stand for? While it's clear that ORC is Ollivier-Ricci curvature, there's no explanation of MANL. Since it is the proposed algorithm in this paper and an important term, it should be spelled out first before abbreviating.

Why is only Definition 2.1 enclosed in a box and emphasized in color? Other Propositions and Theorems are not treated this way; what is the reason for doing this only for this definition?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents ORC-MANL, an algorithm aimed at refining the structure of nearest neighbor graphs by eliminating spurious "shortcut" edges that disrupt accurate manifold representations. This method uses Ollivier-Ricci curvature (ORC) to detect and prune edges that likely bypass the true manifold structure. By removing these shortcuts, ORC-MANL improves manifold accuracy, enhancing performance in downstream tasks such as clustering, manifold learning, and persistent homology.

### Strengths
- ORC-MANL effectively removes shortcut edges, which often misrepresent relationships in the data by connecting distant points across the manifold. By pruning these edges, the algorithm preserves the true geometric and topological structure of the data, yielding a more accurate manifold representation.

- It is supported by a solid theoretical basis showing how Ollivier-Ricci curvature identifies shortcut edges. Empirical results on synthetic and real datasets confirm that ORC-MANL effectively prunes unnecessary edges while preserving important connections, making it a robust choice across a range of data types and applications.

- It operates with fixed parameters for curvature thresholds across different datasets, reducing the need for dataset-specific tuning. This consistency is an advantage over other methods that require careful parameter adjustment to perform optimally.

- ORC-MANL demonstrates versatility across both synthetic and real-world datasets, including complex biological data like high-dimensional single-cell RNA sequencing. This adaptability makes it suitable for diverse data shapes and manifold structures.

### Weaknesses
- Calculating ORC involves complex optimal transport calculations and distance checks, which can be computationally intensive. This added complexity may limit ORC-MANL's scalability for large graphs or high-dimensional data, making it less suitable for applications requiring quick processing.

- It seems to me that ORC-MANL’s performance depends on how the initial nearest neighbor graph is constructed, including choices like the number of neighbors k in k-NN graphs or the distance threshold ϵ in ϵ-radius graphs. Poorly chosen parameters can produce an inaccurate initial graph, which may reduce the effectiveness of ORC-MANL’s pruning. This sensitivity may require users to fine-tune parameters for each dataset to achieve the best results.

- While ORC-MANL generally operates effectively with fixed parameters, the initial nearest neighbor graph setup (e.g., choosing k for k-NN graphs) may still need adjustment across different data types. This reliance on specific curvature thresholds and distance validations can impose extra processing time, especially in cases where the default settings don’t align well with the data structure.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper focuses on characterizing ``shortcut'' edges as edges that are essentially unfaithful to the geometric curvature and then designs a polynomial time algorithm to detect these edges. They first define discrete Ricci curvature corresponding to each edge, where a negative curvature implies shortcut edges. To further confirm that these edges are indeed shortcut edges, they ensure that the shortest distance between the corresponding vertices is sufficiently large if the shortcut edges are absent. 

The algorithm is simple to understand, and the idea of negative Ricci curvature certainly shows promise in simulation settings when considering many different kinds of manifolds. The simulation results are quite rigorous, and the application to real-world data shows initial promise.

### Strengths
1) The algorithm aims to solve an important problem, pruning of ``bad edges'' in nearest-neighbor data embeddings with underlying manifold structures.

2) The theoretical results of the papers (especially Theorem 3.2 and 3.3) are both simple to understand and insightful.  

3) The algorithm performs very well on large simulation datasets. 

4) It also shows some promise in the real-world datasets (Improvement in the performance of ALM data for ISOMAP and reducing some inter-community edges in the PBMC dataset).

### Weaknesses
1) The primary weakness of the paper is that the application to real-world data is quite limited. The authors only consider $2$ real-world datasets. Experiments and verifications on more data (such as image datasets like Fashion-MNIST or K-MNIST) or embeddings of small document datasets would significantly increase confidence in the method. 

2) The robustness of the tolerance parameters $\lambda$ and $\delta$ are not well explored. This is especially important for the real-world datasets.

3) The time complexity of the algorithm seems quite bad $\mathcal{O}(|V|^3)$. This is a minor weakness, given the fact that the paper primarily focuses on a theoretically established idea.

### Questions
1) The authors assume that the data comes from a smooth manifold $\mathcal{M}$. However, real-world experiments are conducted on data with underlying clusters, where it makes more sense to assume that different clusters come from different manifolds (leading to multi-manifold structures [1] ). How do the theoretical guarantees of the paper hold in such a setting?

2) For the PBMC dataset, while there are now fewer edges between different separated components of the UMAP/TSNE pictures, in my understanding, the overall UMAP/TSE embedding quality remains unchanged for both pruned and un-pruned nearest neighbor graph distances. Would this be a correct inference? Also, have the authors considered using datasets like Fashion-MNIST (which are quite well-studied in manifold learning literature) to better understand the usability of the method?

3) How did the authors choose the parameters for the baseline algorithms? How did the authors choose the $\lambda$ and $\delta$ parameters for the real-world experiments?

4) The authors have not commented on the run time complexity of the algorithm (I understand that it is a theoretical paper). I believe it should be $\mathcal{O}(|V|^3)$. Is that correct? If so, have the authors attempted to develop faster algorithms via sampling/approximation style techniques?

[1] Trillos, Nicolas Garcia, Pengfei He, and Chenghui Li. "Large sample spectral analysis of graph-based multi-manifold clustering." Journal of Machine Learning Research 24.143 (2023): 1-71.

### Soundness
3

### Presentation
4

### Contribution
3
