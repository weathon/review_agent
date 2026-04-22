# Balanced Clustering in Reduced Dimensions

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
High-dimensional data often require embedding into lower-dimensional spaces to preserve essential features and structures, which is critical for large-scale analysis. Existing approaches typically treat embedding and clustering as joint optimization tasks but fail to integrate them within a unified framework, limiting clustering performance. Moreover, the interplay between labels and manifold structure is frequently overlooked. To address these challenges, we propose a low-dimensional manifold clustering method that integrates K-means with manifold learning. To mitigate inaccuracies in initial cluster labels, we introduce neighborhood constraints that promote intra-class compactness and inter-class separation, thereby improving label reliability. These refined labels are then used to construct a manifold representation, which in turn enhances clustering in a self-supervised loop that enforces consistency between structure and labels. Notably, we show that maximizing the Schatten-p norm naturally preserves class balance, and we provide a theoretical justification for this property. Extensive experiments on multiple datasets demonstrate the effectiveness and robustness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a novel approach to balanced clustering by integrating K-means with manifold learning, which addresses issues of label consistency and class balance in high-dimensional data. The authors propose a self-supervised framework that alternates between dimensionality reduction and clustering, ensuring consistency between labels and manifold structure. The proposed approach shows promising results on multiple datasets, with significant performance improvements over existing methods, particularly in handling cluster balance. However, several areas could be improved or clarified for better clarity and scientific contribution.

### Strengths
1. The self-supervised manifold clustering method proposed in this paper represents an intriguing innovation. By integrating K-means with manifold learning, the paper avoids the issue of separating dimension reduction and clustering in traditional approaches when handling high-dimensional data. This method effectively enhances clustering accuracy and stability by optimizing the consistency between labels and manifold structure.
2. A significant contribution of this paper is its natural preservation of category balance by maximizing the Schatten-p norm, which holds great significance for addressing category imbalance issues that may arise in practical applications. Many traditional clustering methods often overlook this crucial factor of category balance, and this innovation provides a novel approach to tackling clustering tasks.
3. Through theoretical analysis, the author demonstrates that maximizing the Schatten-p norm effectively achieves class balance. This provides a mathematical basis for the method's efficacy, enhancing its persuasiveness.

### Weaknesses
1. Although the authors conducted experiments across multiple datasets, the analysis of why certain datasets performed better or worse remains insufficient. Particularly for extreme or complex datasets, the adaptability and limitations of the method are not adequately discussed. Such in-depth analysis would facilitate a more comprehensive evaluation of the method's practical application scenarios.
2. Although the introduction section discusses various related techniques, it does not sufficiently emphasize the critical shortcomings of existing methods, particularly regarding label consistency and clustering balance. Further discussion of the limitations of existing approaches—especially their inadequacies concerning label consistency and category balance—would better clarify the paper's motivation and provide stronger context for its contributions.
3. The conclusion section can be expanded to include a discussion of the limitations of the current method and directions for future improvements. For example, how would this method handle larger datasets or more complex data structures? Briefly addressing these considerations will help situate the work within a broader research framework and provide direction for future exploration.
4. Not all datasets were utilized in the experiments, and the convergence analysis cited an insufficient number of iterations, potentially preventing the curve from converging to the optimal solution.

### Questions
1. I encourage the authors to supplement the experimental section with additional experiments, such as experiments on large-scale datasets, trade-off experiments between clustering quality and execution time, robustness tests against noise and outliers, and experiments addressing class imbalance.
2. I hope the authors will conduct experiments on all datasets in the experimental section and revise the convergence analysis iteration count.
3. I hope the author will emphasize the limitations of existing methods in the introduction.
4. The author is encouraged to expand the discussion on the limitations of the current method and propose directions for future improvements.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the separation of embedding and clustering  and the independence of manifold structure and clustering labels, this paper integrates k-means and low-dimensional graph embedding, and utilizes the neighborhood constraints to boost the intra-class compactness and inter-class separation. Further, it demonstrates that maximizing the Schatten-p norm naturally maintains cluster balance. A theoretical justification is developed for this property.

### Strengths
1. The comparative experiments are comprehensive, providing empirical support for the proposed method.

### Weaknesses
1. The writing would benefit from improved clarity and structure.

2. The designed model is complex, involving five parameters. Besides, combined with Figure 1 and Figure 2, we know the clustering performance is sensitive to these parameters. 

3. The use of limited-scale datasets may be insufficient to fully evaluate the robustness and generalizability of the proposed clustering algorithm.

4. There is a lack of relevant ablation experiments.

### Questions
1. According to Figure 4, it seems that the method is not convergent. So, how to determine the stopping condition? 
2. How about the running efficiency?

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
This paper proposes a low-dimensional balanced clustering method, Balanced Clustering in Reduced Dimensions (BCRD), which organically integrates K-means clustering with manifold learning within a unified framework. The method aims to jointly optimize embedding and clustering while maintaining class balance. To achieve this, the authors introduce a neighborhood constraint mechanism to enhance intra-class compactness and inter-class separability, thereby improving the reliability of initial labels. The refined labels are then used to reconstruct the manifold structure, forming a self-supervised iterative consistency mechanism between structure and labels. Additionally, the authors prove that maximizing the Schatten-$p$ norm naturally promotes class balance in clustering. Experiments on both balanced and imbalanced datasets demonstrate that BCRD outperforms K-means in both clustering accuracy and class balance.

### Strengths
1. The paper introduces Schatten-$p$ norm regularization, which promotes class balance and is supported by rigorous theoretical justification.

2. This paper builds a unique model which essentially encodes dimensionality reduction (DR) and K-means clustering (K) simultaneously, enhancing the reliability in real practice.

3. It constructs the manifold structure using the learned labels and neighborhood constraints to enhance intra-class compactness and inter-class separability, thereby establishing a self-supervised iterative consistency between labels and structure.

### Weaknesses
1. The paper does not clearly explain how the label matrix is initialized, which may affect the reproducibility and understanding of the method.

2. It is unclear why the Schatten-p norm is restricted to 1 ≤ p < 2, and what implications or challenges might arise if 0 < p < 1 were considered.

3. The manuscript lacks a visualization on a synthetic dataset to intuitively demonstrate how the proposed model performs clustering.

4. The parameter settings for each dataset are not provided, which hinders experimental reproducibility and fair comparison.

### Questions
1. How is the label matrix initialized?

2. Why is the Schatten-p norm restricted to 1 ≤ p < 2? What would happen if 0 < p < 1 were considered?

3. Could you provide a visualization on a synthetic dataset to more intuitively illustrate the model’s clustering performance?

4. Could you provide the parameter settings used for each dataset to facilitate experimental reproducibility?

If the authors can address my questions, I am willing to increase my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for joint dimensionality reduction and clustering in high-dimensional spaces. The authors claim to "structurally unify" manifold learning (specifically LPP) and K-means into a single self-supervised framework. The core ideas are: (1) to use cluster labels (from K-means) and a neighborhood constraint to construct a manifold graph, which is then (2) used to learn a projection (via LPP) that in turn improves the clustering, creating an iterative loop. (3) The paper's main theoretical claim is that maximizing the Schatten-p norm of the cluster assignment matrix, which they add as a regularizer, "naturally preserves class balance," and they provide a theoretical justification for this property. The method (BCRD) is evaluated on several small-scale image datasets, where it reportedly outperforms a range of K-means and two-step dimensionality reduction baselines.

### Strengths
1. The paper addresses the important problem of clustering high-dimensional data.

2. The idea of using cluster assignments to iteratively refine the manifold structure is a reasonable (though not novel) heuristic.

### Weaknesses
1. Fatal Contradiction: The paper's core claim about cluster balancing is contradictory. Theorem 1 proves the method enforces balance, while Section 4.3 claims it preserves imbalance. This is a critical failure of the paper's central premise.

2. Invalid SOTA Claims: The experimental evaluation is inadequate. The baselines are weak, and no modern SOTA methods (e.g., SSC, LRR, deep clustering) are included for comparison, rendering the results in Table 1 uninformative.

3. Overstated Novelty: The "structural unification" is just standard alternating optimization (Algorithm 2), and the "consistency loop" is a direct consequence of this, not a new framework.

4. High Parameter Sensitivity: The parameter analysis (Figures 1 and 2) shows that the method's performance is highly sensitive to the choice of k (neighbors) and m (dimensions), which undermines its practical utility and robustness.

5. Weak Datasets: The evaluation is performed exclusively on small, old-school face and digit datasets. There is no evaluation on any large-scale, high-dimensional modern benchmarks.

### Questions
1. Can the authors please clarify the fundamental contradiction in Section 4.3? How can their method, whose regularizer is proven in Theorem 1 to force a balanced clustering (n_j = N/c), be simultaneously claimed to preserve an imbalanced distribution on the ImB8 dataset? These two statements cannot both be true.

2. Why were no modern, state-of-the-art subspace clustering or manifold learning algorithms (e.g., Sparse Subspace Clustering, Low-Rank Representation, or any deep clustering methods) included as baselines in Table 1? The paper's SOTA claims are unsupported without them.

### Soundness
1

### Presentation
2

### Contribution
2
