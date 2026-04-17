# Learning from Historical Activations in Graph Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 8

## Abstract
Graph Neural Networks (GNNs) have demonstrated remarkable success in various domains such as social networks, molecular chemistry, and more. A crucial component of GNNs is the pooling procedure, in which the node features calculated by the model are combined to form an informative final descriptor to be used for the downstream task. However, previous graph pooling schemes rely on the last GNN layer features as an input to the pooling or classifier layers, potentially under-utilizing important activations of previous layers produced during the forward pass of the model, which we regard as historical graph activations. This gap is particularly pronounced in cases where a node’s representation can shift significantly over the course of many graph neural layers, and worsened by graph-specific challenges such as over-smoothing in deep architectures. To bridge this gap, we introduce HistoGraph, a novel two‑stage attention‑based final aggregation layer that first applies a unified layer-wise attention over intermediate activations, followed by node-wise attention. By modeling the evolution of node representations across layers, our HistoGraph leverages both the activation history of nodes and the graph structure to refine features used for final prediction. Empirical results on multiple graph classification benchmarks demonstrate that HistoGraph offers strong performance that consistently improves traditional techniques, with particularly strong robustness in deep GNNs. Our code is at https://github.com/YanivDorGalron/HISTOGRAPH

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
HISTOGRAPH is a graph pooling method that leverages historical activations from all GNN layers rather than just the final layer. The method uses a two-stage attention mechanism: (1) layer-wise attention that aggregates node representations across all layers using the final layer as a query, and (2) node-wise self-attention that models spatial interactions between nodes. This approach aims to mitigate over-smoothing in deep GNNs and capture multi-scale graph features.

### Strengths
* The method can be used both for end-to-end training and as a post-processing step on frozen pretrained GNNs, making it practical for different scenarios.
* Extensive experiments across graph classification, node classification, and link prediction tasks demonstrate the method's generalizability.
* Strong improvement in benchmark results for the proteins dataset (15%).

### Weaknesses
* While the overall idea is novel, the individual components (layer-wise attention, node-wise attention) are standard techniques. The main contribution is their combination for this specific purpose.
* Storing all intermediate activations (N×L×D) could be memory-intensive for very large graphs or deep networks.
* Tested datasets are not of large nature.

### Questions
* The method assumes all GNN layers produce embeddings of the same dimension Din. How would you handle architectures where layer dimensions vary, which is common in some GNN designs?
* Beyond the over-smoothing mitigation proof, can you provide any theoretical guarantees about the expressiveness of HISTOGRAPH compared to standard pooling methods?
* How critical are the sinusoidal positional encodings for layer positions? Have you tried other encoding schemes or learning the positional embeddings?
* How sensitive is the method to hyperparameters, particularly the hidden dimension D and the number of attention heads?
* Have you tested HISTOGRAPH on industrial-scale graphs (millions of nodes)? What are the practical limitations you've encountered?

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
3

### Summary
This paper proposes a novel and effective GNN architecture, HISTOGRAPH, which mitigates the over-smoothing problem by explicitly modeling and leveraging historical node representations.

### Strengths
1. The motivation is clear; the authors propose leveraging historical representations to mitigate over-smoothing, which is reasonable and well-justified.
2. The experiments are comprehensive, thoroughly validating the effectiveness of their method across various tasks.

### Weaknesses
1. Lacks comparison with some more recent baselines [1].
2. No experimental comparisons were conducted on larger graphs, such as those in the OGB [2] suite. How does the time efficiency compare to the baseline when the graph size increases?
3. How are the historical representations specifically utilized? What are the theoretical advantages of the gating mechanism?
4. Lacks a theoretical analysis of the method's effectiveness.

[1] Wang Y, Liu S, Zheng T, et al. Unveiling global interactive patterns across graphs: Towards interpretable graph neural networks[C]//Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024: 3277-3288.

[2] Hu W, Fey M, Zitnik M, et al. Open graph benchmark: Datasets for machine learning on graphs[J]. Advances in neural information processing systems, 2020, 33: 22118-22133.

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces an architectural modification for graph neural networks (GNNs) consisting in (1) an intra-node attention mechanism that aggregates the representations for each node at different layers (i.e., different message passing iterations), and (2) a global inter-node self attention layer (aggregating information across all nodes). 
The motivation from this architecture comes from the fact that current methods ignore the history of graph activations across layers, even though this may include some useful information.
The experimental evaluation compares the proposed method on standard node and graph classification benchmarks, both when training from scratch and when applying the method to a pre-trained model, and compares against a vast number of competitors. The proposed method shows high performance and an ablation studies explores different attention modifications.

### Strengths
- It is clear that the authors have spent a lot of effort in the experimental section as they compare against a large number of baselines and consider a large number of datasets.
- The proposed method can be easily included into existing architectures (at the cost of some training for the new parameters).

### Weaknesses
- Global self-attention is quadratic in the number of nodes, which makes the method impractical for large graphs.
- Caching in memory the activations at all layers for all nodes can become prohibitively expensive. Together with the above, this makes the proposed method very impractical for large graphs.
- Section 4 is not very convincing as the arguments are too general. Regarding oversmoothing, Proposition 1 is obvious, and in practice different nodes might perform better with different alphas (which however is not allowed). Furthermore applying global self-attention is actually promoting smoothing. Regarding the trajectory filter, the argument can be made for any attention mechanism, and also it is hard argue whether models actually learn to use this information this way.

### Questions
- The computational complexity analysis is a bit confusing. First it is mentioned that the proposed method improves over a naive "joint node-layer attention" which has complexity O(LN^2D) by instead having a method which is O(NLD + N^2D) but then it is (correctly) mentioned that the complexity is dominated by N^2D, which means that there is no advantage. So it seems that the paragraph from line 216 to 220 does not really have much sense. Could you please clarify the point of this paragraph?
- The same happens in the "Frozen Backbone Efficiency" paragraph: as the complexity is dominated by N^2D there is no advantage (I do not doubt that in practice it can have a difference in runtime, but in terms of computational complexity there is no difference). Could you clarify what is the advantage that is mentioned in the paper?
- What are the hyperparameters for the baselines and how where they selected? These details should be included in the paper
- It is mentioned that the method can overcome oversmoothing, but global self-attention actually goes against this. Could you elaborate on why global self-attention would not lead to oversmoothing?
- On PROTEINS the proposed method reaches 97% (almost perfect), while all other methods stop at 80%. This difference is quite striking and should be analyzed. Could the authors comment on this? 
- The runtime analysis shows training time, but I think inference is actually much more important as training only happens once. Could you include some number for inference times? Some plots showing how runtime and memory scale as a function of N and L would be great.

Minor comments:
- There are some imprecisions in the text, specially in the math formalism:
	- lines 166 and 170 define X in different ways
	- in line 183 for computing X^tilde the sizes do not match (I understand there is an implicit broadcasting, but it should be formalized better)
	- in Section 4 the notation for node vectors is not introduced
- It would be interesting to see an analysis of the attention maps to understand which layers are receiving the most attention

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes HISTOGRAPH, a two-stage attention-based pooling framework for Graph Neural Networks (GNNs) that leverages intermediate activations (“historical graph activations”) from all layers rather than only the final layer. The approach first applies layer-wise attention to capture the evolution of node embeddings across depths, followed by node-wise attention to model spatial dependencies. The method can be integrated end-to-end with a backbone GNN or applied as a lightweight post-processing head on frozen models. Experimental results on TU and OGB benchmarks, as well as node classification tasks, demonstrate improved performance and robustness to over-smoothing in deep architectures.

### Strengths
1. Novel perspective: The paper introduces a clear and well-motivated idea of learning from the historical trajectory of node activations, addressing the common limitation of relying solely on the last GNN layer.

2. Comprehensive experiments: Evaluations across multiple datasets (TU, OGB, node classification, and link prediction) with both GIN and GCN backbones show consistent improvements.

3. Well-written and well-positioned: The paper situates HISTOGRAPH clearly within prior works on pooling, residual connections, and over-smoothing mitigation

### Weaknesses
1. Limited interpretability of learned attention weights: While attention is used layer-wise and node-wise, the paper could benefit from deeper analysis of what the model learns—e.g., visualization of layer weights across datasets.
2. The attention mechanism itself is widely adopted and not novel. However, the paper should further clarify why the proposed method achieves such notable performance gains. A deeper analytical discussion and illustrative case studies would substantially strengthen the contribution.

### Questions
see above

### Soundness
3

### Presentation
4

### Contribution
3
