# PRISM: Partial-label Relational Inference with Spatial and Spectral Cues

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
In many real-world scenarios, acquiring precise labels for graph-structured data is expensive or even infeasible, as reliable annotation often requires substantial expert knowledge or computational resources. As a result, graph labels are often noisy and ambiguous. This challenge motivates partial-label graph learning, where each graph is weakly annotated with a candidate label set containing the true label. However, such ambiguous supervision makes it hard to extract reliable graph semantics and increases the risk of overfitting to noisy candidate labels. To address these challenges, we propose a unified framework named PRISM that performs relational inference with spatial and spectral cues to alleviate the impact of label ambiguity. On the one hand, PRISM captures discriminative spatial cues by aligning prototype-guided substructures across graphs. On the other hand, it decomposes graph signals into multiple frequency bands and extracts global spectral cues with an attention mechanism, which preserve frequency-specific semantics. We integrate these complementary views into a hybrid relational graph and perform an iterative label propagation under candidate constraints. Extensive experiments on a range of well-known datasets demonstrate that PRISM consistently outperforms strong baselines under various noise settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PRISM, a framework for Partial-Label Graph Learning (PLGL). It resolves label ambiguity by integrating local spatial cues (substructure matching) with global spectral cues (frequency analysis). These views are used to construct a relational graph that guides an iterative label propagation process. Experiments show PRISM achieves state-of-the-art results, demonstrating strong robustness to label noise.

### Strengths
1. The manuscript tackles an important, practical task: learning from graphs that only provide a candidate label set, and it states the setup clearly for real applications.
2. It shows strong results with noticeable margins on several benchmarks and stays stable as label noise increases, which points to solid robustness.
3. The pipeline is modular and easy to adapt and swap encoders or spectrum modules without redesigning the whole method.

### Weaknesses
1. The theory relies on very strong assumptions and does not explain how the model actually learns under noisy supervision; it mostly describes behavior at an ideal fixed point.
2. The paper does not convincingly show that the two views truly help each other; ablations suggest the label-propagation step may account for most gains, not the new encoders, so the claimed synergy remains unclear.
3. Some key implementation details are missing, such as how confidence-based filtering is done, and the complexity claim skips the likely quadratic cost of cross-graph similarity and Top-k selection, which can mislead readers about scalability.

### Questions
1. Theorems 1/2 assume the GNN already maps samples near the correct prototypes and the classifier recognizes them perfectly; they describe an ideal fixed point rather than how the model reaches it under noisy supervision, offering little insight into the core challenge.
2. Is the claim that the model's complexity is O(|E| d) reasonable? Finding the top-k neighbors for all graphs in a batch seems like it would be at least an O(N^2) operation, which can be very slow. The paper doesn't seem to have accounted for this cost.
3. The paper claims the spatial and spectral views are complementary, but neither the theory nor the experiments really show how they help each other. Is there an example where one view gets it wrong, but the other one corrects it? Otherwise, it looks more like a simple combination of two methods.
4. Looking at your ablation study (Table 2), removing the "Relational Inference" step causes the biggest performance drop. This makes me wonder: is the final label propagation step the real key to the performance boost? It might mean that the spatial and spectral encoders you focus on are not the main reason for the model's great results.
5. In Section 3.1, you mentioned a key step called "confidence-based filtering" for updating the prototypes, but it's never explained how it works. Without it, it's hard to fully understand your method.

### Soundness
2

### Presentation
2

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
The paper tackles the practical and underexplored problem of Partial-label Graph Learning (PLGL), which reflects real-world situations where graph labels are incomplete or uncertain. The proposed method PRISM integrates three complementary components, i.e., prototype-guided substructure alignment, spectral modeling, and hybrid relational graph propagation, to address label ambiguity from spatial, spectral, and relational perspectives.

### Strengths
The approach is conceptually coherent and well-motivated, combining local structural cues with global spectral information for more reliable supervision. Experimental results on multiple benchmarks demonstrate clear performance improvements over existing weakly supervised and graph learning methods, supporting the effectiveness of the proposed framework.

### Weaknesses
1. Graphs differ significantly in size and structure, making cross-graph substructure alignment both conceptually unclear and potentially computationally expensive. The paper does not explain how this process is implemented or efficiently approximated, and when dealing with large graphs with many nodes and edges, it could still lead to severe computational bottlenecks.

2. Each graph has its own Laplacian basis, making frequency bands difficult to compare across graphs. The paper does not explain how spectral features are aligned or normalized, and when the structural differences between graphs are large, ensuring that their spectral representations are meaningfully aligned remains a significant challenge.

3. The theory assumes a perfectly trained classifier with one-hot outputs. In practice, momentum updates and label propagation interact, so this assumption may not hold.

4. Limited robustness evaluation. The work does not systematically evaluate different candidate set sizes, nor does it examine performance under varying noise levels or open-set scenarios.

### Questions
1. How is the cross-graph substructure alignment implemented in practice? Can the authors show that this step does not cause extra computational cost?
2. Since each graph has its own Laplacian basis, how does the method handle spectral inconsistency when using multi-band features?
3. The convergence proof assumes a well-trained classifier with correct one-hot outputs. What happens if this assumption does not hold? Any discussion or experiments?
4. Could the authors test the model under different candidate set sizes, noise levels, or open-set cases (where true labels are missing)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel graph classification method in partial-label setting, which captures relations between graphs from both spatial and spectral views.

### Strengths
- The proposed method designs a novel way to compute spatial and spectral relations between graphs.
- The method's training complexity is linear in the number of edges and consistent with standard GNN-based methods.
- The method applies a binary mask to enable partial supervision.

### Weaknesses
- The equations lack explanations. Why do you design the spatial relations by Eq.(5), which consists of two components. Why do you use Eq.(7)  to compute X^(p) . What is the deeper meaning behind these equations.
- How to decide the binary mask M, as it desides how many data are used for supervision. Does this method sensitive to M and the ratio of supervised data.

### Questions
- I am not quite understand Figure 3, please provide more explanations.
- For Eq.16 and Eq. 17, which one is the final loss function?
- Why using different q for different datasets?

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
The paper aims to solve a novel graph classification problem known as partial label graph learning where labels of each graph are ambiguous with a set of labels provided but not precise label. The paper proposed a novel framework that aims to establish both graph-graph relation from spatial view and spectral view and denoise the label set from the established graph pair relations. For spatial view, the author proposed to align subgraph structure with the clustered class-centered substructure prototype and for spectral view, the author aims to build band-frequency based alignment. Empirical results suggest a clear improvement of proposed method compared with various baselines from graph neural network based models and partial label learning from computer vision fields. Ablation results suggest each component's effectiveness. The author also show theoretical analysis on the convergence of the label confidence matrix and training loss.

### Strengths
1.The paper discusses a novel problem called partial label graph learning problem which would be practical in real application setting where labels are ambiguous for actual datasets.  
2. A novel framework is proposed that absorbs both spatial and spectral information for each graph to effectively capture useful information for denoise the noisy soft labels and train the classifier.
3. The theoretical analysis looks sound and comprehensive.
4. Empirical results suggest a clear improvements for proposed method compared with baselines from Graph neural network side and partial label learning benchmarks.

### Weaknesses
1. I think when it involves with the eigenvalue computation, especially for the dense and large-scale graph, the computation cost becomes expensive and prohibitive. This makes the spectral part of the framework not scalable. In the computation efficiency analysis, the author failed to discuss this important question and in my opinion falsely conclude the computation efficiency is comparable to standard GNN. For a dense and large-scale graph, the computation of eigenvalue and eigenvectors could be approaching O(n^2).
2. I find some parts of the paper lacks explanation of the symbols. For example, in the spectrum part, k seems to be an index, yet, in the computation analysis, k becomes the number of eigenvalues. M is referred to as the binary mask, but there is no definition on how it should be computed, which makes the paper hard to follow at some points.

### Questions
Please see weakness. I would like to know how the author considers their runtime on spectrum part and also how M is computed.

### Soundness
3

### Presentation
2

### Contribution
3
