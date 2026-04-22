# Which Eigenvectors Do Graph Transformers Need for Node Classification?

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Graph transformers have emerged as powerful tools for modeling complex graph-structured data, offering the ability to capture long-range dependencies beyond the graph adjacency. Yet their performance on node classification often lags behind that of message passing and spectral graph networks. Unlike these methods, graph transformers require additional mechanisms to inject structural information. In this work, we focus on Laplacian positional encodings, which use eigenvectors of the graph Laplacian to provide node-level positional information. Existing methods select eigenvectors using data-agnostic heuristics, assuming one-size-fits-all rules suffice.
In contrast, we show that the spectral distribution of class information is graph-specific. To address this, we introduce Broaden the Spectrum (BTS), a novel, intuitive, and data-driven algorithm for selecting subsets of Laplacian eigenvectors for node classification.
Our method is grounded in theory: we characterize the structure of optimal attention matrices for classification and show, in a simplified setting, how BTS naturally emerges as the eigenvector selection rule for achieving such attention matrices. When evaluated with standard graph transformer architectures, it delivers substantial performance gains across a wide range of node classification benchmarks. Our work shows that the performance of graph transformers on node classification has been held back by the choice of positional encodings and can be improved by employing a broader, well-chosen set of Laplacian eigenvectors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper examines the positional encoding effectiveness of Laplacian matrix eigenvectors in graph transformers, with a particular focus on node classification tasks. It introduces an Energy Spectral Density metric derived from class labels and uses this metric to identify the top-_k_ eigenvectors for encoding. The proposed approach is integrated into several existing graph transformer architectures, leading to consistent improvements in node classification performance across multiple datasets.

### Strengths
1. The proposed ESD metric and corresponding BTS method are simple, intuitive, and easily adaptable to a wide range of graph transformer models.
2. The paper offers a theoretical analysis of the rationale behind BTS, elucidating its effectiveness in the context of node classification tasks.
3. The experimental evaluation is thorough, including extensive ablation studies that validate the efficacy of the proposed BTS method.

### Weaknesses
1. The first one lies in BTS’s reliance on full eigen-decomposition, which incurs higher computational complexity compared to previous methods that select only the lowest or highest top-_k_ eigenvectors. This substantially restricts its scalability to large-scale graphs.
2. Both the BTS method and its theoretical analysis are limited to node classification task, posing considerable challenges when extending to other tasks such as link prediction or graph-level prediction.
3. The assumptions underlying the theoretical analysis are overly restrictive—particularly the formulation  $X = Y M_X + \sigma N $—which does not accurately reflect real-world conditions, where node features typically incorporate structural and neighborhood information.
4. The experimental section omits several important baselines, including PolyFormer [1] and SpecFormer [2].
5. Moreover, the Chameleon and Squirrel datasets used in the experiments exhibit substantial edge overlap, results should be reported on their filtered versions in [3].
6. (Minor) The graph should be denoted as "an undirected graph" in Definition 2.1.

[1] Ma J, He M, Wei Z. Polyformer: Scalable node-wise filters via polynomial graph transformer[C]//Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024: 2118-2129.

[2] Bo D, Shi C, Wang L, et al. Specformer: Spectral graph neural networks meet transformers[J]. arXiv preprint arXiv:2303.01028, 2023.

[3] Platonov O, Kuznedelev D, Diskin M, et al. A critical look at the evaluation of GNNs under heterophily: Are we really making progress?[J]. arXiv preprint arXiv:2302.11640, 2023.

### Questions
1. Please begin by responding to the Weaknesses part.
2. The baselines PolyFormer and SpecFormer should be included for comparison in the experimental evaluation.

### Soundness
2

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
3

### Summary
This paper introduces Broaden the Spectrum (BTS), a data-driven method for selecting Laplacian eigenvectors as positional encodings (PEs) in Graph Transformers (GTs) for node classification. Existing GTs underperform on node classification due to their reliance on data-agnostic heuristics which ignore the graph-specific spectral distribution of class information. The main contributions of this paper are: (1) BTS Algorithm: A lightweight, task-aware method that selects eigenvectors most aligned with class labels using Energy Spectral Density (ESD). (2) Theoretical Justification: Shows that the optimal attention matrix for classification has a class-wise block structure, and that BTS-selected eigenvectors best approximate this structure. (3) Empirical Validation: Extensive results demonstrate significant performance gains across homophilic, heterophilic, and long-range benchmarks using standard GT architectures.

### Strengths
(1)	Novel & Principled Method: BTS is intuitive and theoretically grounded, bridging graph signal processing and transformer architectures, and is a simple yet effective method for improving GTs. Moreover, it is a plug-in module and can easily be applied into different backbones.

(2)	Solid Theoretical Analysis: It proves that optimal attention matrices for classification should have class-block structure.

(3)	Strong Empirical Results: Extensive experiment result, including large gains on challenging datasets; consistent improvements across multiple architectures and task types (homophily, heterophily, long-range).

### Weaknesses
(1)	Moderate novelty: This work can be seen as an incremental work on traditional positional encoding methods by adding ESD before selection of eigenvectors.

(2)	Large experimental searching space: According to Tab. 12, hyper-parameter searching space seems to be extremely large, which diminishing reproducibility of the results.

(3)	Clarification issue of motivations: Statement of data-agnostic methods seems ambiguous. (see Q.1, Q. 2)

(4)	Some issues and concerns of experimental parts: See Q. 3.

### Questions
(1)	Author states that current positional encoding methods are data-agnostic and proposed BTS is a data-adaptive method. However, calculation of eigenvector is still required in Algorithm 1, which I think is still related to graph data (structure). Does data-agnostic mean feature-free? This part should be discussed or clarified to avoid misunderstanding. Moreover, now that your method is data-adaptive, it is suggested to add comparison with other learnable positional encoding methods like LSPE [1].

(2)	For definition of ESD, it seems that it can be regarded as a preprocessing procedure which is static in essence. Then what is the difference between BTS and static positional encoding methods? 

(3)	Please list computation cost and GPU consumption of BST and its comparison with LPE or other methods. It seems that ESD calculation is time-consuming, especially you claim that BST is “lightweight”. Moreover, what about its scalability on large-scale graphs?

[1] Dwivedi V P, Luu A T, Laurent T, et al. Graph neural networks with learnable structural and positional representations[J]. arXiv preprint arXiv:2110.07875, 2021.

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
This paper examines why graph transformers often underperform on node classification tasks, identifying the choice of eigenvectors for Laplacian positional encodings as a key factor. To address this, the authors introduce Broaden the Spectrum (BTS), a data-driven approach that selects eigenvectors based on their alignment with class label energy. Theoretical analysis explains how BTS promotes attention matrices with class-aligned block structures, and extensive experiments on homophilic, heterophilic, and long-range benchmarks demonstrate that BTS significantly outperforms common eigenvector selection heuristics.

### Strengths
1. Figure 1 offers a clear illustration of class-label energy distributions, highlighting the importance of adaptive spectrum selection.

2. The theoretical analysis helps to understand the method’s underlying principles.

3. Experiments across diverse graph types, particularly heterophilic and long-range datasets, demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. Additional related works [1-2] on adaptive or alternative positional encodings in graph transformers should be discussed to provide a more comprehensive context.

2. The proposed use of label-aligned spectral energy for positional encoding selection relies heavily on labeled data, which may not always be readily available.

3. While the paper focuses on node classification, it would be valuable to explore whether the proposed approach can generalize to graph-level classification tasks.

[1] Park, Wonpyo, et al. "Grpe: Relative positional encoding for graph transformer." arXiv preprint arXiv:2201.12787 (2022).

[2] Li, Chenyang, et al. "DAM-GT: Dual Positional Encoding-Based Attention Masking Graph Transformer for Node Classification." arXiv preprint arXiv:2505.17660 (2025).

### Questions
How does the proposed method perform under label-scarce settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper study how the selection of Laplacian eigenvectors as positional encodings influences graph transformer performance in node classification. The authors propose Broaden the Spectrum (BTS) to selects eigenvectors according to their class-label energy spectral density (ESD). They show that the optimal attention matrix has a class-wise block structure and that high-label-energy eigenvectors best approximate it. Experiments demonstrate consistent large performance gains on heterophilic and long-range benchmarks, across several graph transformer architectures.

### Strengths
1. Introducing adaptive frequency selection meaningfully advances positional encoding in graph transformers. 

2. The model is simple, effective, and supported by theoretical analysis.

### Weaknesses
1. While early graph transformers relied heavily on Laplacian eigenvectors to incorporate graph topology, recent work has demonstrated that structural biases can also be introduced through GNNs [1,2,3] and attention masks[4]. Thus, the statements "graph
transformers require explicit positional encodings to inject structural information" in Abstract and "transformers rely on positional encodings (PEs) to inject structural information" in Introduction may introduce some misleading understanding. 


2. While the proposed BTS is theoretically grounded in the concept of class-label energy spectral density (ESD), this dependence constrains its applicability to supervised settings. Therefore, the impact of label quantity on performance needs to be studied. 

3. The explanation of the “class-wise block structure” is somewhat ambiguous. Intuitively, the optimal attention matrix is purely block-diagonal form (where the diagonal blocks are non-zero and the off-diagonal blocks are zero). In Figure 2, the empirical pattern shows substantial inter-class attention, which seems inconsistent with the intuitive clustering.

4. The paper omits comparison and discussion with several state-of-the-art transformer-based graph models, including Polynormer [1], CoBFormer [2], DualFormer [3], and Gradformer [4], which represent the latest advances in structural bias integration. This omission substantially weakens the empirical credibility of the paper’s claimed contributions. 

[1] Polynomial-Expressive Graph Transformer in Linear Time, in ICLR 24. 

[2] Less is More: on the Over-Globalizing Problem in Graph Transformers, In ICML 24.

[3] DUALFormer: Dual Graph Transformer, in ICLR 25.

[4] Gradformer: Graph Transformer with Exponential Decay, in IJCAI 24.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
