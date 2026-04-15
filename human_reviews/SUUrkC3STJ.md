# VCR-Graphormer: A Mini-batch Graph Transformer via Virtual Connections

- Decision: Accept (poster)
- Scores: 5, 6, 6, 5

## Abstract
Graph transformer has been proven as an effective graph learning method for its adoption of attention mechanism that is capable of capturing expressive representations from complex topological and feature information of graphs. Graph transformer conventionally performs dense attention (or global attention) for every pair of nodes to learn node representation vectors, resulting in quadratic computational costs that are unaffordable for large-scale graph data. Therefore, mini-batch training for graph transformers is a promising direction, but limited samples in each mini-batch can not support effective dense attention to encode informative representations. Facing this bottleneck, (1) we start by assigning each node a token list that is sampled by personalized PageRank (PPR) and then apply standard multi-head self-attention only on this list to compute its node representations. This PPR tokenization method decouples model training from complex graph topological information and makes heavy feature engineering offline and independent, such that mini-batch training of graph transformers is possible by loading each node's token list in batches. We further prove this PPR tokenization is viable as a graph convolution network with a fixed polynomial filter and jumping knowledge. However, only using personalized PageRank may limit information carried by a token list, which could not support different graph inductive biases for model training. To this end, (2) we rewire graphs by introducing multiple types of virtual connections through structure- and content-based super nodes that enable PPR tokenization to encode local and global contexts, long-range interaction, and heterophilous information into each node's token list, and then formalize our $\underline{\textbf{V}}$irtual $\underline{\textbf{C}}$onnection $\underline{\textbf{R}}$anking based $\underline{\textbf{Graph}}$ Trans$\underline{\textbf{former}}$ (VCR-Graphormer). Overall, VCR-Graphormer needs $O(m+klogk)$ complexity for graph tokenization as compared to $O(n^{3})$ of previous works. The [code](https://github.com/DongqiFu/VCR-Graphormer) is provided.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work try to combine PPR and transformer together to perform mini-batch training.

### Strengths
1. The paper aims to solve the mini-batch training of graph transformer.

2. the paper is well written and easy to follow.

3. The performance of this work is good, comparing to the baseline.

### Weaknesses
1. The novelty of this work is not high, as it mainly combines random walk with attention. Tokenize the graph to sequence use random walk is common used. 

2. The code is not provided, its reproducibility is unclear.

### Questions
1. The novelty of this work is not high, as it mainly combines random walk with attention. Tokenize the graph to sequence use random walk is common used. 

2. The code is not provided, its reproducibility is unclear.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
To scale the Transformer to large graphs, this paper proposes a new graph Transformer, called VCR-Graphormer. The key idea of the proposed method is to sample a token list constructed by related nodes on graphs for each target node. In this way, the mini-batch training strategy could be adopted to reduce the training cost. Moreover, the authors leverage techniques like PPR and virtual connections to preserve both local, global, long-range, and heterophilous information.

### Strengths
1.	This paper summarizes four metrics for graph tokenization methods.
2.	This paper leverages existing techniques, like PPR and the graph partition method, to generate the token list for each target node.
3.	This paper provides several theoretical analyses for the proposed method.
4.	Empirical results on different scale datasets seem to indicate the promising performance of the proposed method.

### Weaknesses
1.	Several recent studies on designing graph Transformers with node sampling or node clustering are ignored.
2.	Experimental results are inefficient in demonstrating the merits of the proposed method.

### Questions
1.	I think the proposed method belongs to the line of designing scalable graph Transformers via node sampling. Hence, several necessary studies [1,2,3] on this research topic should be cited and discussed in the paper. [1] and [3] leverage various node sampling strategies to obtain the token list for each node, where a super node-based strategy is also adopted in [3] to preserve the global information. [2] leverage the graph partition-based strategy to reduce the training cost of the Transformer model. These researches are highly related to the proposed method, especially [1] and [3]. 
2.	Based on Q1, I think it is necessary to compare the performance of the proposed method with [1] and [3] to demonstrate the superiority of the proposed method for constructing the node sequence.
3.	I think the results on heterophilous graph datasets are inefficient in demonstrating that the proposed method can handle graphs with heterophily. Important baselines [4,5,6] and datasets [7, 8] are not considered in the experiment. In addition, [8] has revealed the drawbacks of the Squirrel dataset. Hence, experiments on heterophilous graph datasets need to be reorganized to support the claim of handling heterophily property. 
4.	Similarly, the authors have highlighted that efficiency is one of the important metrics for graph tokenization methods. So, the necessary experiment is required to support the above claim.
5. Does the sampling node set of the third component contain super nodes? If the answer is yes, how do you initialize the features of super nodes?



[1] Zhao et al. Gophormer: Ego-Graph Transformer for Node Classification. arXiv 2021.

[2] Kuang et al. Coarformer: Transformer for large graph via graph coarsening. arXiv 2021.

[3] Zhang et al. Hierarchical graph transformer with adaptive node sampling. NeurIPS 2022. 

[4] Bo et al. Beyond Low-frequency Information in Graph Convolutional Networks. AAAI 2021.

[5] Chien et al. Adaptive universal generalized pagerank graph neural network. ICLR 2022.

[6] Li et al. Finding Global Homophily in Graph Neural Networks When Meeting Heterophily. ICML 2022.

[7] Lim et al. Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods. NeurIPS 2021.

[8] Platonov et al. A critical look at the evaluation of GNNs under heterophily: are we really making progress? ICLR 2023.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work the authors propose a way for effective mini-batch training of graph transformers. In their approach, namely Virtual Connection Ranking Graph Transformer (VCR-Graphformer), they build a special token list for each input target node as input to a transformer. This list consists of four components including: 
- (1) the node features of the target node, 
- (2) propagated features for a number of random walk steps and features of nodes with top Personalized PageRank (PPR) when PPR is run on extended graphs (rewiring) with extra virtual nodes connected to original nodes either 
- (3) in the same cluster (structure-based global information) or
- (4) carrying the same label (content-based global information). 

They provide theoretical justification for their direction, experiment with their mini-batching approach over a collection of small, large and heterophilous graph datasets, employing various Graph Neural Network (GNN) and Graph Transformer (GT) architectures as baselines, and report on (a) superior competitive node classification accuracies for their VCR-Graphormer and (b) ablation studies clarifying the role of components in the token list and parameter choices.

### Strengths
- The presentation is easy to follow and the intuition of the approach well supported.

- Types, number of datasets and baseline models are adequate for demonstrating the efficacy of the approach for the node classification task in particular. Ablation studies are very informative (Figures 3 and 4).

### Weaknesses
- The inclusion of additional graph learning tasks (edge/graph classification) would further establish the validity/generality of the token list preparation approach proposed.

### Questions
- VCR-Graphormer seems to be an input preprocessing technique (preparation of the token list for input to standard transfromer layers as in Eq (3.4)) that is compatible with the target training mode (mini-batching) rather than a graph transformer architecture (which is what the reader would possibly expect when coming across the term "VCR-Transformer"). Perhaps emphasizing this view early in the presentation would be beneficial in comprehending the overall idea?

- Do you identify any obstacles for this technique being effective for other (large-scale) graph learning tasks (not node classification)? 

- VCR-Graphormer supports effective heterophilous graph node label learning through one of its token list components. Would this also promote homophilous graph learning?

Minor/typos
- Page 3: share the exactly same -> share exactly the same
- Page 7: eigendecompostion -> eigendecomposition
- Page 9: strcuture -> structure
- Page 9: Targeting node-level tasks like graph classification -> ? (graph classification is not a node-level task)

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes VCR-Graphormer, which resolves the computational challenges of graph transformers in large-scale graphs. The authors tokenize nodes using personalized PageRank (PPR), enabling mini-batch training. Additionally, they introduce virtual connections in the graph to encode local and global contexts, long-range interactions, and heterophilous signals. This approach reduces computational complexity and outperforms or is on par with existing methods in node classification across 12 datasets.

### Strengths
The scalable graph transformer is a hot, interesting, and important field in our research community. Training with mini-batches is memory-efficient by nature. The experiments are extensive (but focus on homophilous datasets). The proposed method outperforms baselines by a large margin, especially for heterophilous graphs.

### Weaknesses
- The paper is hard to follow. Abuse of notations in Eq 3.1, 3.2, and 3.3 might be confusing for readers. Please formally define the operations or use well-known operations. Is {.} a set or a list? Please proper set-builder notations. Please put l (l-th step) to the name of the variable, r_u? Is the concatenation operator applied to both scalars and vectors? The cardinality of T_u in Eq 3.3 is 4 when you note it like this. Plus, it would be nice if the authors explained what insights we can see in a series of theorems 3.2 and 3.3.
- The eigendecomposition is not a core component in NAGphormer. NAGphormer without structural encoding has shown no bad results. We can choose other cheap structural encodings rather than eigendecomposition, for example, PPR vectors as the authors say. Thus, the superiority of the proposed method against NAGphormer (e.g., cubic complexity) should be rewritten focusing on the core components (Hop2Token).
- The performance increase in homophilous and large-scale datasets is marginal. VCR-Graformer's additional modules do not seem to be effective for these datasets. Instead, the proposed method is effective for heterophilous datasets. However, these datasets are small-scale thus, only evaluating small parts of this model (that targets large-scale graphs). It would be nice if the authors can use large-scale heterophilous datasets [Lim, Derek, et al.]. In addition, experiments to evaluate modeling long-range dependency (the third component) are not conducted.
- Runtime evaluation on efficiency is required. Although the time complexity is decreased in theory, the actual computations can be increased by METIS and PPR on three types of graphs.

## References 
- [Lim, Derek, et al.] Lim, Derek, et al. "Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods." Advances in Neural Information Processing Systems 34 (2021): 20887-20902.

### Questions
N/A

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good
