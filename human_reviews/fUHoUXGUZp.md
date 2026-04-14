# DLGNet: Hyperedge Classification through Directed Line Graphs for Chemical Reactions

- Decision: Reject
- Scores: 8, 6, 5, 5

## Abstract
Graphs and hypergraphs provide powerful abstractions for modeling interactions among a set of entities of interest and have been attracting a growing interest in the literature thanks to many successful applications in several fields. In particular, they are rapidly expanding in domains such as chemistry and biology, especially in the areas of drug discovery and molecule generation. One of the areas witnessing the fasted growth is the chemical reactions field, where chemical reactions can be naturally encoded as directed hyperedges of a hypergraph. In this paper, we address the chemical reaction classification problem by introducing the notation of a Directed Line Graph (DGL) associated with a given directed hypergraph. On top of it, we build the Directed Line Graph Network (DLGNet), the first spectral-based Graph Neural Network (GNN) expressly designed to operate on a hypergraph via its DLG transformation. The foundation of DLGNet is a novel Hermitian matrix, the Directed Line Graph Laplacian $\mathbb{\vec L}_N$, which compactly encodes the directionality of the interactions taking place within the directed hyperedges of the hypergraph thanks to the DLG representation. $\mathbb{\vec L}_N$ enjoys many desirable properties, including admitting an eigenvalue decomposition and being positive semidefinite, which make it well-suited for its adoption within a spectral-based GNN. Through extensive experiments on chemical reaction datasets, we show that DGLNet significantly outperforms the existing approaches, achieving on a collection of real-world datasets an average relative-percentage-difference improvement of 3.27\%, with a maximum improvement of 5.28\%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduced a novel method that convert the directed hypergraph into Directed Line Graph (DLG), where each hyperedge is transformed into a vertex in the DLG. The DLG Laplacian is proved to be positive semidefinite and spectral-based GNN is introduced for the DLG. This novel representation and associated GNN operation are applied for chemical reaction classification problem.

### Strengths
1. The study is comprehensive. Starting from the mathematical definition of the directed hypergraph, the laplacian is derivated and related properties are proved. Then the convolution operation is introduced and complexity is also analyzed. 

2. The paper includes many details including the public-available code, proofs of theorems, complexity analysis, dataset and model setting in the Appendix, which is very helpful to understand the paper. 

3. The importance of directionality are well assessed in ablation study, which supports the motivation of the proposed method.

### Weaknesses
1. The baseline performance for reaction classification is quetionable: 
   The F1 scores reported in Table 1 for baseline methods are all at random guess level (e.g. F1<0.1 for 10-class classification). However in other reaction classification datasets and baseline methods, the F1 could be larger than 0.8 even for 1000 classes. [1]
   Please verify the baseline metrics, and compare with the baseline methods which could get better prediction performance. 

2. The definition of node and hyperedge for chemical reaction is not clear:
   In Section 1 Line 72, author mentions tackle the reaction classification problem as hyperedge classification task. The Figire 1 also indicates each molecule is a node and each reaction is a hyperedge. 
   However in Figure 2 Dataset-1 and Dataset-2, the head and tail are for single molecule. In Section 2 Line 130, the head set and tail set are for hyperedge e, which means the hyperedge is a single molecule. 
   The author should make the definition consistent. 

3. Proofs could be more clear:
   In line 740 and 743, the second term is transformed from sum(i,j \in E) to sum(i,j \in E: i<=j) by mapping the lower triangle elements to upper triangles. However, the diagonal terms are introduced twice and should be subtracted from the new format. 
   In line 743 and line 745, how the first term was transformed should be further clarified. 

References:
[1] Li, Anchen, Elena Casiraghi, and Juho Rousu. "Chemical reaction enhanced graph learning for molecule representation." Bioinformatics 40.10 (2024): btae558.

### Questions
1. High Level: 
   After converting the directed hypergraph to DLG, the relationship of vertices belongs to the same hyperedge are not reflected in DLG anymore and the representation of the original graph is not updated during learning.
   In Line 294: the feature matrix for the vertices of DLG is a simple aggregation of original feature matrix. 
   Will this exacerbate the learning ability? 

2. The Dataset description could be enriched:
	In Figure 1 Dataset-1 example, only 2 reactnats are involved and it seems only one hyperedge for reaction remains. Then the graph convolution layer becomes linear layer since no neighbors are included. 
	Could you provide the statistics of the reactions in the datasets (e.g. the average number of vertices and average node degree in DLG)?

3. Experiments:
   Line 408-409, with 5-fold cross-validation, why do we still need 50%/25%/25% splition for train/val/test? 

Typo: 
1. Line 308, the output shape should be m x 2c', instead of n x 2c'. 
2. Line 742, the * for first B(u,j) should be removed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the field of organic synthesis, accurately predicting reaction types can help chemists design and optimize synthetic routes. This article proposed DLGNet. This is a novel approach for classifying chemical reactions by leveraging a new spectral-based graph neural network for hypergraphs. DLGNet utilizes a directed line graph Laplacian operator Hermitian matrix to encode the directionality and connectivity between hyperedges. The Hermitian property and positive semidefiniteness of this matrix make it suitable for spectral convolution, which the DLGNet leverages to enhance classification performance.

### Strengths
1. The chemical reactions often involve interactions between multiple substances, rather than simple binary relationships (e.g., reactants-products). Using hypergraphs to represent and analyze chemical reactions can more naturally capture this multivariate relationship.

2. The mathematical framework is innovative and critical. The introduction of "Directed Line Graph Laplacian" is a key mathematical contribution. By constructing a Hermitian matrix, complex-values Laplacian, the authors enable the network to capture both the directionality and connectivity of hyperedges, a sophisticated approaches for chemical reaction classifications.

3. DLGNet has been conducted extensive experiments on three different and diverse real-world datasets. Also including a robust ablation study to demonstrate the importance of directionality in the model. These all show the benefits of the proposed methods.

### Weaknesses
1. The use of Hermitian Laplacian matrix is mathematically convenient. However, using non-Hermitian matrix on the directed graph neural networks may sometimes provide more flexibility in encoding the directionality.

2. The model currently only relies on the molecular Morgan Fingerprints for the node features. However, consider more molecular features, such as electronic descriptors and three dimensional conformations, would provide the model a more comprehensive understanding of chemical reactions.

3. It would be beneficial to add a figure of the DLGNet model architecture in the main text. This would  help readers better understand the model.

### Questions
1.  Add a figure of the DLGNet model architecture in the main text.

2. Although the authors compared the model performance of DLGNet to other published graph neural networks. It would be beneficial to illustrate the differences between DLGNet and the other models to provide readers a clearer understanding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the chemical reaction classification problem by introducing the Directed Line Graph (DLG) associated with a given directed hypergraph. It also deduces a Directed Line Graph Network (DLGNet) which achieves impressive performance on the chemical reaction classification task.

### Strengths
- This paper proposed a new direct hypergraph neural network for chemical reaction prediction.  This application is interesting and the proposed method sounds technique.

### Weaknesses
- The paper says it only focuses on modeling reaction structures without considering any form of hypergraph learning methods. However, it is unclear whether some traditional methods have been proposed for this task. If there exists, it should be added to support the efficiency of your method. Further, the proposed method is obviously can be used for link prediction, can you explain why it cannot achieve good performance on the task?
- Lack of details of how to model the reaction structures to hypergraph edge prediction task. Is it the node of the hypergraph is the molecular? This problem definition should be clearly shown in the main paper. Further, why DLG design can specifically work for this reaction task? In our understanding, the specific application paper should consider some domain knowledge (such as chemical reaction prior information) to make the proposed approach convincing.
- The design of the $Bve=-i, if v\in T(e)$ is weird, what is the theoretical or intuitive motivation? What if we use $Bve=-1, if v\in T(e)$ or we still use the B as described in the typical method like [1].
- This paper lacks some necessary baselines and is without discussion with some related works. For example [1,2,3]. Besides, the Magnetic Laplacian also be used in [4].

minor: Some key matrices should describe their dimensions for ease of reading.

[1]Directed Hypergraph Representation Learning for Link Prediction, AISTATS 2024
[2] Hypergraph Convolutional Networks via Equivalency between Hypergraphs and Undirected Graphs, ICML workshop 2022
[3] Directed hypergraph neural network, 2020
[4] Unified Random Walk, Its Induced Laplacians and Spectral Convolutions for Deep Hypergraph Learning, 2022

### Questions
see weakness

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the problem of chemical reaction classification based on hypergraphs. Firstly, this paper introduces a formal definition of directed line graphs to transform directed hypergraphs, thereby converting edge-level tasks into node-level tasks. Building on this definition, DLGNet is proposed, which updates hyperedge features through local aggregation. Experimental results demonstrate the superior performance of the proposed DLGNet compared to the baseline chemical reaction classification.

### Strengths
1) The picture of modeling chemical reactions using hypergraphs is interesting in GNNs for chemistry and biology. 

2)  Good performance compared to the "baseline".

### Weaknesses
1) The novel of this paper is limited. On the one hand, the transformation of directed graphs to line graphs has been extensively studied, and the proposed DLG does not capture the unique characteristics that distinguish hypergraphs from ordinary graphs. Additionally, the use of complex numbers to represent directionality in directed graph neural networks (such as MagNet [1]) has already been explored. Thus, the contribution of this part appears to be incremental. On the other hand, it is unclear what the rationale for translating directed hypergraphs into directed line graphs for hyperedge classification is. 

2) The paper is poorly organized. Is the introduction of Datasets 1, 2, and 3 a contribution to this paper? Why devote so much space to this in the main text?

3) The discussion and comparison of related work are insufficient. In particular, all compared models are not specific to chemical reaction classification. 

[1] MagNet: A Neural Network for Directed Graphs. NeurIPS 2021

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
