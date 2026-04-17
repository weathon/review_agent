# GraphSpa: Self-supervised Graph Sparsification for robust generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Graph sparsification has emerged as a promising approach to improve efficiency and remove redundant or noisy edges in large-scale graphs. However, existing methods often rely on task-specific labels, limiting their applicability in label-scarce scenarios, and they rarely address the residual noise that remains after sparsification. To address this issue, we aim to jointly consider both sparsity and robustness. In this work, we present GRAPHSPA, a self-supervised graph sparsification framework that constructs compact yet informative subgraphs without requiring labels, while explicitly mitigating residual noise. We formulate sparsification as a constrained optimization problem in which flatness is incorporated as part of the objective. Specifically, we address this problem by leveraging an augmented Lagrangian scheme to  progressively satisfy the target sparsity. We also train the encoder to be robust to perturbations so that optimization is guided toward flatter regions of the loss landscape, reducing sensitivity to residual noise, and improving generalization. We theoretically demonstrate that this framework guarantees stable convergence while addressing both sparsity and robustness. Extensive experiments on benchmark datasets show that GRAPHSPA consistently outperforms baselines across various sparsity ratios and preserves cluster structures in t-SNE visualizations. Notably, it demonstrates strong and consistent performance on both large-scale and heterophilic datasets, validating its applicability in real-world scenarios. These results highlight GRAPHSPA as a principled and reliable framework for graph sparsification without labels and under residual noise.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies self-supervised sparsification of graphs for node classification. They first use Gumbel-softmax method to sample important edges in a differentiable way and then incorporate the robust training method of sharpness-aware minimization into the training process. Optimization process via the augmented Lagrangian relaxation. Experiments are conducted on classic GNNs with performance comparison with unsupervised baselines such as DropEdge.

### Strengths
Strengths:
1. The paper studies the interesting problem of constructing graph sparsifiers in a self-supervised way. It can be applied to label-free settings and might have wider applicability than supervised algorithms.
2. The paper is easy to follow and explains the key ideas and technical details clearly.

### Weaknesses
Weaknesses:
1. While there seems limited study on self-supervised graph sparsification, traditional graph sparsification in graph theory can be considered as unsupervised. They aim to preserve some important graph properties in the original graph, such as spectral properties, random walk properties, and shortest-path distances. The related work largely ignore these existing works, and some of them, e.g., [a,c], could be comparable to the current method. Some supervised sparsification methods, e.g., [b], are also missing. The authors may want to perform a more comprehensive review on existing work and better position the current paper.

References:

[a] Abd Errahmane Kiouche, Julien Baste, Mohammed Haddad, Hamida Seba, and Angela Bonifati. 2024. Neighborhood-Preserving Graph Sparsification. Proc. VLDB Endow. 17, 13 (September 2024), 4853–4866. https://doi.org/10.14778/3704965.3704988

[b] Cheng Zheng, Bo Zong, Wei Cheng, Dongjin Song, Jingchao Ni, Wenchao Yu, Haifeng Chen, and Wei Wang. 2020. Robust graph representation learning via neural sparsification. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 1062, 11458–11468.

[c] Sadhanala, V., Wang, Y.-X., and Tibshirani, R. J. Graph sparsification approaches for Laplacian smoothing. In Proceedings of AISTATS Conference, pp. 1250–1259, 2016.

2. The techniques used in this paper are quite typical and I am not feeling excited about the proposed method. The Gumbel-Softmax has been widely used in edge sampling, e.g., in [b] and the flatness-aware training [21] is also well-established. While the authors propose a complete framework, each part of the algorithm is based on a typical method, limiting the overall novelty of the technique.
3. It would be best to involve additional and diverse benchmark datasets in the experimental study.

### Questions
Does the method apply to graph classification and link prediction tasks and inductive learning setting? How about directed graphs?

### Soundness
2

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
2

### Summary
This paper proposes a self-supervised graph sparsification framework named GRAPHSPA, aiming to address two major challenges faced by existing sparsification methods: 1) dependence on task labels; 2) the negative impact of residual noise. Experiments were conducted on the Cora, Citeseer, and Pubmed datasets. The results show that at different sparsification rates, the accuracy of GRAPHSPA is superior to other baselines.

### Strengths
1. Addressing dual key issues: This paper simultaneously addresses two core challenges in graph sparsification: label dependency and residual noise. This is an important and practical contribution.  
2. Novel Framework Design: The paper ingeniously integrates three techniques (self-supervised mutual information, augmented Lagrangian constraint optimization, and SAM flatness-aware training) into a unified framework. This combination exhibits strong innovation, and the motivation behind each component is clear.

### Weaknesses
1. Outdated baseline: In the experimental section, the latest baseline, DropEdge, is from 2020, while the other two are from 2003 and 1999, respectively. These do not cover the latest achievements in this field. It is difficult to evaluate whether the experimental results achieve SOTA performance using graphspa.

2. High computational complexity: As shown in Appendix C, the asymptotic time complexity of GRAPHSPA, $O(TN^2d)$, is significantly high. Although the authors claim that the actual running time is comparable when T and d are small, this may limit its scalability on very large graphs.

### Questions
Why did the experimental part choose some old baselines? Is it because there have been no new models in this field for a long time, or did the author deliberately avoid them?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors proposes GRAPHSPA, a self-supervised framework for graph sparsification. The method is designed to be independent of downstream task labels and robust to the residual noise that can harm GNN performance on a simplified graph structure.

### Strengths
***S1:*** The paper's main strength is the integration of MI-based self-supervision, differentiable edge sampling, and SAM-based regularization. This isn't just a collection of methods; they work together to solve two distinct but related problems: label-free learning and noise robustness. The idea of using SAM to specifically counteract the effects of sparsification is insightful. 

***S2:*** Framing sparsification as a constrained optimization problem, using the augmented Lagrangian for stable convergence, and leveraging the established connection between flat minima and generalization are all marks of a methodologically sound approach. 

***S3:*** Most sparsification methods are either heuristic (e.g., based on degree), supervised, or they do not explicitly account for the fact that sparsification itself can make a GNN more vulnerable to the remaining noisy edges. By tackling both label-dependency and residual noise, this work addresses a practical gap in the literature.

### Weaknesses
***W1:*** The proposed framework is quite complex, involving nested optimization loops (for SAM) and multiple interacting components. This complexity leads to computational overhead. The experiments are limited to smaller benchmark datasets (Cora, Citeseer, Pubmed), leaving its scalability to very large graphs an open and important question.

***W2:*** While the paper shows the final model works well, there is limited analysis on the interplay between the key components. For instance, how does the SAM-induced flatness affect the MI maximization objective? Does one component dominate the other? A deeper ablation study could provide more insight into why the combination is so effective.

***W3:*** The choice of maximizing the mutual information between the original graph and the sparsified view is a standard approach in graph self-supervised learning. While effective, it feels like a generic "make the subgraph look like the full graph" objective. It is not entirely clear if this is the optimal self-supervised signal specifically for sparsification, which is fundamentally about identifying a critical subgraph, not just a similar one.

### Questions
Some more questions that could clarify my concerns and let me raise the paper's score:

***Regarding Scalability:*** Could you comment on the computational complexity of GRAPHSPA, particularly the overhead introduced by the SAM optimizer and the augmented Lagrangian scheme? Do you foresee any bottlenecks when applying this method to graphs with millions, or more, of nodes/edges?

***Regarding the MI Objective:*** Have you considered alternative self-supervised objectives tailored more specifically for identifying a graph's "backbone"? For example, objectives that prioritize preserving path-based information or specific topological properties, rather than representation similarity?

Also, you show robustness against injected noisy edges. Does the SAM component also make the sparsification process itself more robust? For example, is the final set of selected edges more consistent across different random initializations when SAM is used, compared to when it is not?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
GRAPHSPA learns to sparsify graphs without labels by maximizing mutual information between the original and sparsified graphs. Each edge is modeled as a differentiable Bernoulli variable with a Lagrangian budget constraint. To counter residual noise, the encoder is trained using Sharpness-Aware Minimization (SAM) to promote flat minima. The method is label-free and robust under injected noise.

### Strengths
1. Principled formulation: combines differentiable sparsification, MI-based learning, and noise-aware optimization.
2. Progressive edge pruning via an augmented Lagrangian ensures convergence stability.
3. Demonstrated robustness to post-sparsification noise and preservation of structural clusters (t-SNE).

### Weaknesses
1. Computationally heavier due to dual updates and SAM optimization.
2. Dependence on InfoNCE negative sampling temperature not fully analyzed.

### Questions
1. How does GRAPHSPA scale to large or dynamic graphs?
2. What is the impact of different noise distributions on stability?
3. Can this framework extend to node pruning or heterophilous graphs?

### Soundness
3

### Presentation
3

### Contribution
3
