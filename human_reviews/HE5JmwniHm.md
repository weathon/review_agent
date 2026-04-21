# DLEFT-MKC: Dynamic Late Fusion Multiple Kernel Clustering with Robust Tensor Learning via Min-Max Optimization

- Avg Score: 7.00
- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 6

## Abstract
Recent advancements in multiple kernel clustering (MKC) have highlighted the effectiveness of late fusion strategies, particularly in enhancing computational efficiency to near-linear complexity while achieving promising clustering performance. However, existing methods encounter three significant limitations: (1) reliance on fixed base partition matrices that do not adaptively optimize during the clustering process, thereby constraining their performance to the inherent representational capabilities of these matrices; (2) a focus on adjusting kernel weights to explore inter-view consistency and complementarity, which often neglects the intrinsic high-order correlations among views, thereby limiting the extraction of comprehensive multiple kernel information; (3) a lack of adaptive mechanisms to accommodate varying distributions within the data, which limits robustness and generalization. To address these challenges, this paper proposes a novel algorithm termed Dynamic Late Fusion Multiple Kernel Clustering with Robust {Tensor Learning via min-max optimization (DLEFT-MKC), which effectively overcomes the representational bottleneck of base partition matrices and facilitates the learning of meaningful high-order cross-view information. Specifically, it is the first to incorporate a min-max optimization paradigm into tensor-based MKC, enhancing algorithm robustness and generalization. Additionally, it dynamically reconstructs decision layers to enhance representation capabilities and subsequently stacks the reconstructed representations for tensor learning that promotes the capture of high-order associations and cluster structures across views, ultimately yielding consensus clustering partitions. To solve the resultant optimization problem, we innovatively design a strategy that combines reduced gradient descent with the alternating direction method of multipliers, ensuring convergence to local optima while maintaining high computational efficiency. Extensive experimental results across various benchmark datasets validate the superior effectiveness and efficiency of the proposed DLEFT-MKC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a novel algorithm, Dynamic Late Fusion Multiple Kernel Clustering with Robust Tensor Learning (DLEFT-MKC), which addresses key limitations in existing multiple kernel clustering methods. It dynamically optimizes base partition matrices, introduces min-max optimization for enhanced robustness and generalization, and reconstructs decision layers to capture high-order associations across views. These advancements improve computational efficiency and enable comprehensive extraction of multiple kernel information, leading to effective consensus clustering partitions.

### Strengths
1. The experimental results are comprehensive, covering datasets of varying scales, comparative methods, and diverse visualization experiments.

2. The overall writing is clear and easy to understand.

### Weaknesses
1. The motivation for this paper is unclear and lacks persuasiveness. For instance, the authors mention that one of the drawbacks of existing methods is their reliance on fixed initial base partitions, but they do not define what "initial base partitions" are. Additionally, while the authors highlight that current multiple kernel k-means methods often lack adaptive mechanisms for varying internal data distributions, the paper does not provide a solution to this issue.

2. A framework diagram of the proposed algorithm should be included in the manuscript to aid understanding.

3. The paper's novelty is insufficient, as it primarily integrates three related components from existing work, suggesting that it does not offer any new contributions.

4. The paper lacks a theoretical analysis of computational complexity.

5. Replacing Figure 4 with a table to present the comparison of runtime would provide a clearer and more intuitive representation.

6. The reference list contains inconsistencies, with some conference names abbreviated and others in full, and some entries including DOI information while others do not. These details should be made consistent.

### Questions
The authors claim that existing multiple kernel k-means methods often lack adaptive mechanisms for varying internal data distributions, stating that their proposed method addresses this issue. However, where is this adaptation reflected in the paper, and how does the method achieve it?

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
This paper proposes a novel DLEFT-MKC, which effectively overcomes the representational bottleneck of base partition matrices and facilitates the learning of meaningful high-order cross-view information. Specifically, it is the first to incorporate a min-max optimization paradigm into tensor-based MKC, enhancing algorithm robustness and generalization. Extensive experimental results across various benchmark datasets validate the superior effectiveness and efficiency of the proposed algorithm.

### Strengths
1. The paper is well-organized and written without technical errors.
2. The motivation behind the paper is clear. The proposed method is innovative as it is the first to incorporate a min-max optimization paradigm into tensor-based MKC and the first to dynamically reconstruct base partition matrices from late fusion multi-view clustering.
3. The experiments are comprehensive and solid, involving comparisons with over 20 competing algorithms. The proposed method consistently achieves the best performance, and the experiments have been evaluated from multiple perspectives. 
4. The experimental results demonstrate a substantial improvement in clustering performance, which is particularly exciting when coupled with a significant increase in computational efficiency.

### Weaknesses
1. The notation is somewhat difficult to follow. Some symbols are used without explanation. A table summarizing the notation would be beneficial.
2. The case where $\gamma_p = 0$ and $\Delta \leq 0$ is missing from the discussion in Eq.(14).
3. There is insufficient discussion on the proposed algorithm. The objective expression appears complex, and the max-min-max form raises questions about how it ensures rapid convergence. While the experimental results show a notable improvement in computational efficiency compared to most other algorithms, there is a lack of analysis on computational complexity to explain this phenomenon.
4. The cluster assignments in Fig.2 are not easily distinguishable, which is insufficient to validate the effectiveness of the learning process. The authors should provide additional visualizations to substantiate this contribution. For example, on the Reuters dataset, most algorithms achieve an average accuracy around 45%, whereas the proposed algorithm achieves an accuracy of 97%. Such a high accuracy rate raises concerns, and the authors need to provide an explanation.
5. The author should provide the code and sufficient information for readers to reproduce the experimental results.

### Questions
As mentioned above.

### Soundness
3

### Presentation
4

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
The authors present a novel method for multiple kernel clustering called DLEFT-MKC, which addresses the problem of representational bottleneck of base partition matrices. This paper proposes a min-max optimization paradigm in tensor-based MKC to improve both performance and robustness. In addition, the authors use tensor learning to capture the high-order correlations between kernels. Finally, the effectiveness of DLEFT-MKC is demonstrated through extensive experiments on benchmark datasets.

### Strengths
1) This study is the first to apply a min-max optimization paradigm to tensor-based multi-kernel clustering, pioneering a new approach to improve performance and robustness in clustering.
2）The overall structure of the paper is complete and well-organized. 
3) The formulas in the paper are written correctly and the derivations are correct.
4) Another bright point in this paper is that it combines the RGDE with ADMM to optimize their problem, this combination is a well-established technique to solve convex optimization problems.

### Weaknesses
1) The complexity of the proposed method is not analyzed. 
2) The base partition matrices directly capture data information from multiple kernels. Why reconstruct the dynamic partition matrices? It would be better if the authors clearly introduced this design.
3) Why are the standard deviation (std) values for your method so high on the Reuters and CCV datasets in Table 2?
4) As far as I know, datasets like Caltech101-7, Caltech101-20, and Caltech101-all are the most commonly used for multiple kernel clustering, and it would be more convincing to compare the experimental results on these datasets. Please show a comparative experiment of the proposed method on at least one of them.

### Questions
See the weakness.

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
The paper presents a dynamic late fusion multiple kernel clustering with robust tensor learning via min-max optimization, named DLEFT-MKC. The proposed method aims to address the robustness and generalization problem via a min-max objective. It utilizes t-TNN constraints to explore the high-order relationships and cluster structures between views. The unstable performance caused by the base partition matrices is mitigated by the dynamically adjusted partition matrices. Additionally, late fusion is introduced to reduce its computational complexity. These modules help enhance the clustering performance. The authors also conduct extensive experiments to validate the effectiveness and efficiency of DLEFT-MKC in terms of performance and time cost.

### Strengths
1.	The paper is well-written, and the motivations are clearly described;
2.	The idea that incorporates a min-max optimization paradigm into tensor-based MKC to enhance both performance and robustness in clustering tasks is novel to a certain degree.
3.	The paper conducts extensive experiments, and the results are adequately and logically discussed.

### Weaknesses
1.	For some justifications, more details should be provided.
2.	Leveraging t-SVD for tensor-based module optimization may recur high complexity, which is not consistent with the efficiency goal.
3.	The work seems to combine Wang’s work about late fusion with tensor learning. Leveraging tensor learning for multi-view clustering result fusion is not a novel idea, which has been utilized in many previous works. The main contribution of the work lies in the min-max optimization, which has also adopted by some previous works. Therefore, the novelty of the work might not be so convincing.

### Questions
1.	What does “robustness and generalization” refer to in this work? Does it mean less sensitivity to parameters?
2.	The underlying reason explaining the min-max optimization’s positive effect on robustness should be better explained.
3.	Eq. (4), $\gamma_p$ is used to balance the effect of each view, while in Eq. (6), $\gamma_p^2$ is adopted. We notice that the constraints on $\gamma_p$ in Eq. (4) and Eq. (6) remain the same, but why Eq. (6) adopts $\gamma_p^2$ instead?
4.	Though DLEFT-MKC is a late fusion approach, it is based on the third-order tensor and updating tensor requires t-SVD which is time-consuming. So why it has lower running on several datasets compared with other late fusion-based methods like OPLFMVC and LFMVC.

### Soundness
3

### Presentation
4

### Contribution
3
