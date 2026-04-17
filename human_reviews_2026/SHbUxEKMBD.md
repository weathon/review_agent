# NEUTAG: Graph Transformer for Attributed Graphs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Graph Transformers (\textsc{GT}) have demonstrated their superiority in graph classification tasks, but their performance in node classification settings remains below par. They are designed for either homophilic or heterophilic graphs and show poor scalability to million-sized graphs. In this paper, we address these limitations for node classification tasks by designing a model that utilizes a special feature encoding that transforms the input graph separating nodes and features, which enables the flow of information not only from the local neighborhood of a node but also from distant nodes, via their connections through shared feature nodes. We theoretically demonstrate that this design allows each node to exchange information with all nodes in the graph, effectively mimicking all-node-pair message passing while avoiding $\mathcal{O}(N^2)$ computation. We further analyze the universal approximation ability of the proposed transformer. Finally, we demonstrate the effectiveness of the proposed method on diverse sets of large-scale graphs, including the homophilic \& the heterophilic varieties.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper develops a new graph Transformer called NEUTAG which is based on the constructed node-feature graph. Moreover, the authors also define positive and negative neighborhood according to the relevance between nodes and features. Then, NEUTAG also leverages the Transformer backbone to learn node representations. The experimental results showcase the effectiveness of NEUTAG for node classification.

### Strengths
1.This paper is well-organized and easy to follow.

2.The authors provide the theoretical analysis of the proposed method.

3.The authors select various baselines for performance comparison.

### Weaknesses
1.The research gap is overclaimed.

2.Mainstream datasets are missing.

3.Some important experiments are missing.

### Questions
1.The authors claim three limitations in existing graph Transformers which lack objectivity. The first limitation only appears in the hybrid-based graph Transformer which needs to combine Transformer with GNN-like modules. The rest two limitations have been widely studied in recent GTs. So, I do not think the authors clearly present the research gaps between NEUTAG and previous methods.

2.The selection of the dataset is inappropriate. For instance, the Chameleon dataset has been shown to contain a significant number of duplicate nodes. Furthermore, the authors did not employ established mainstream datasets for evaluation, large-scale graphs in NAGphormer or different types of graphs Polynormer.

3.The authors emphasize the scalability of GTs but they do not conduct efficiency study in this paper. I think it is required to compare the training cost of NEUTAG and other representative baselines.

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
4

### Summary
This paper proposes a graph transformer that leverages visual nodes for message propagation. The authors provide theoretical analysis to demonstrate the efficiency of the proposed approach.

### Strengths
Some theoretical analisys.

The method is conceptually straightforward.

### Weaknesses
The authors claim that existing GTs cannot perform well on both homophilous and heterophilous graphs. However, many works have demonstrated strong performance on both types of graphs. For sxample, [1][2]...

For the non-scalable, this issue has been extensively addressed by many recent techniques, such as linear transformers[3] and their graph-specific variants, which significantly reduce the computational and memory complexity while maintaining competitive performance.


The paper is not well organized, which makes it difficult to follow. For example, Figure 1 is not referenced in the main text, and the concept of “Feature node” is not clearly defined. This raises a question: how should the connections between feature nodes and graph nodes be defined when the node features are not based on a bag-of-words representation?

The experimental results show that the proposed method does not achieve significant improvement. The datasets with the largest reported gains, Arxiv-year and Snap-patents, in fact, do not yield satisfactory results. For comparison, some directed graph methods such as Dir-GNN [4] achieve much higher performance 64.08 and 73.95 on these two datasets, which is far surpassing the proposed method’s 53.96 and 63.0. Moreover, Dir-GNN is simpler and more lightweight.

[1] GOAT: A Global Transformer on Large‑scale Graphs
[2] Rethinking Graph Transformer Architecture Design for Node Classification
[3] Sgformer: Simplifying and empowering transformers for large-graph representations
[4] Edge directionality improves learning on heterophilic graphs

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes NEUTAG, a scalable transformer that operates on a rewired graph composed of original nodes and additional "feature" nodes. NEUTAG uses both sparse and full attention, capturing homophily and heterophily. Theoretical analyses show that NEUTAG increases graph connectivity and can approximate dense attention. NEUTAG is benchmarked against popular GTs and MPNNs on homophilous and heterophilous benchmarks.

### Strengths
1. The research question on designing more powerful and scalable graph transformers is interesting
2. The approximation results are interesting given the more efficient attention can approximate the dense version
3. The transformer baselines in the experiments are extensive.

### Weaknesses
1. For the limitations of existing graph transformers I'm not sure I appreciate or understand these given my following thoughts:  
* redundant dependency on GNNs. the paper claims that since transformers are universal, GNNs and full attention are redundant together. while transformers are universal, I think it could still be useful to have a component that captures local connectivity as this would require lots of data for the transformer to learn. in other words, the local message-passing by GNNs is a good inductive bias that otherwise a transformer would need to learn from a lot of data. moreover, I think NEUTAG is using a form of message-passing in its local attention along only neighbors, so it doesn't completely do-away with the GNN component. 
* given my argument above that local attention is a form of message-passing, NEUTAG could also inherit the homophily biases similar to other transformers that use a GNN component. 
* given that scalable GTs exist, what are the limitations with GTs like GOAT or polynormer?

2. To me, the core contribution seems to be the proposal of the "metamorphosis" graph, which augments the original graph with new "feature" nodes that connect to original nodes. Rather than a new transformer, this is closer to graph rewiring approaches, which are not discussed as related works or compared against empirically. The other novel component in addition to the metamorphosis graph is the feature2feature full attention which is where the main computational savings come from in comparison to other full attention GTs. From the Appendix ablations though, it seems removing this component only slightly drops performance, whereas removal of local neighbors causes the biggest drop. This seems to indicate that the local neighborhood attention, which is similar to message-passing, is the most important to the architecture. So to sum up, the main gains are coming from the metamorphosis graph, similar to rewiring approaches, and local attention, a form of message-passing, while the feature2feature attention improves performance not as significantly. I would thus like to see comparisons to other rewiring techniques perhaps also with your version of local attention.

3. NEUTAG seems only to apply to graphs with nodes of only binary features (given the present/not present figure). I appreciate that this limitation is mentioned in the Appendix, and I agree that this is a weakness for NEUTAG. This seems limited since in general a graph can have categorical or continuous features, and I'm not sure what the prevalence of graphs with only binary features is. In its current form there doesn't seem to be a discussion on the generalization of NEUTAG to graphs with features that are not just binary vectors, so the approach seems quite limited. 

4. Empirically, Polynormer comes quite close across all benchmarks except for snap, while NEUTAG wins 1/3 on larger benchmarks in comparison to other GTs (table 1).

### Questions
My questions largely follow my weaknesses: 

1. how is the local attention different than message-passing? and if not, why does NEUTAG circumvent the homophily-biases?
2. what is the relation of NEUTAG's metamorphosis step to existing graph rewiring techniques? it seems similar to an approach that adds edges between nodes of similar features. would you expect local attention + graph rewiring to perform similarly?
3. how would you generalize NEUTAG to non-binary features?
4. The connectivity argument makes sense, but if you are adding connections, does this increase the rate of oversmoothing? or does it affect oversquashing since now the degree of each node increases significantly?

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
In this paper, the authors propose a novel GT architecture that employs a special feature encoding to separate nodes and features, enabling information flow not only from local neighborhoods but also between distant nodes via shared feature connections. Experiments are conducted on a variety of graph datasets to validate the approach.

### Strengths
The method is straightforward, and the authors provide supporting theoretical analysis.

### Weaknesses
1. The novelty of introducing new virtual nodes to connect nodes in the graph appears to be limited.

2. The experimental results do not clearly demonstrate the effectiveness of the proposed Graph Transformer method compared to existing GT approaches.

3. Some recent Graph Transformer methods [1–5] that also utilize virtual nodes are not discussed.

4. The paper requires careful proofreading and polishing.
 
[1] Wenhao Zhu, et al. Hierarchical transformer for scalable graph learning, 2023. 

[2] Wenhao Zhu, et al. Anchorgt: Efficient and flexible attention architecture for scalable graph transformers, 2024. 

[3] Weirui Kuang, et al. Coarformer: Transformer for large graph via graph coarsening, 2022. 

[4] Chuang Liu, et al. "Gapformer: Graph Transformer with Graph Pooling for Node Classification." IJCAI. 2023.

[5] Xueqi Ma, et al. "HOGT: High-Order Graph Transformers."

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
1
