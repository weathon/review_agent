# TopGQ: Fast GNN Post-Training Quantization Leveraging Topology Information

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Graph Neural Networks (GNNs) demand substantial memory and computation as datasets scale in size. Thus, quantization is a promising remedy by compressing full-precision values into low-bit representations. However, existing GNN quantization methods depend on tedious gradient-based updates to preserve accuracy. This quantization time may be a major barrier to real-world deployments as the input graph size scales. To this end, we present TopGQ (Topology-aware GNN Quantization), an accurate post-training quantization framework tailored
for GNNs, alleviating the burden of redundant quantization overhead. We propose Dual-axis scale absorption, which applies scale factors along both activation axes, merging one into the static adjacency matrix. Dual-axis scale absorption attains higher accuracy via addressing outlier nodes. This helps maintain the same computational cost as column-wise quantized inference. We further introduce topology-guided quantization, which exploits the relationship between local graph structure and activation variance. TopGQ enables fast inference for unseen nodes, via a novel node index (TopPIN), a lightweight proxy of activation variance from local structure. With these techniques, TopGQ eliminates the need for retraining while preserving accuracy. Experimental results show that TopGQ is comparable to prior works while reducing quantization time by an order of magnitude, establishing it as a practical solution for efficient and scalable GNN inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TopGQ, a post-training quantization framework for GNNs that exploits graph topology to achieve fast quantization. The method introduces two main techniques: dual-axis scale absorption and TopPIN. The authors claim the speedup in quantization time compared to existing methods while maintaining comparable accuracy across various node and graph classification tasks.

### Strengths
1. The paper tackles a genuine bottleneck in GNN deployment - the excessive quantization time of existing methods (hours to days for large graphs) that limits practical applicability.

2. The authors provide theoretical justification (Theorems 1-2) linking local graph topology to activation variance, which motivates the use of topology-aware quantization.

3. Comprehensive experiments: Extensive evaluation across multiple architectures (GCN, GAT, GIN, GraphSAGE), datasets of varying scales (from Cora to MAG240M with 240M nodes), and both node/graph-level tasks.

### Weaknesses
1.  The claim “quantizing GNNs is known to be difficult due to the varying magnitudes of node activations” lacks empirical support. Is this issue universally observed across all GNN architectures? If the variation arises from graph structure, the use of normalized adjacency matrices could alleviate it. If it instead results from activation outliers, quantitative evidence should be provided—e.g., what proportion of nodes are considered outliers?

2. A major concern lies in the applicability of the proposed techniques. Dual-axis scale absorption assumes a static adjacency matrix, whereas many recent GNNs employ dynamic or learnable edge weights. Although the authors briefly acknowledge this limitation in the appendix, the quantitative impact of the additional cost remains unclear. For example, in classical graph transformers—where edge weights are computed for all node pairs—what is the precise computational overhead introduced by the proposed absorption?

3. The motivation behind several design choices is insufficiently explained. For instance, the reasoning behind Formula (10) is unclear and should be elaborated to justify its formulation.

4. Theoretical results rely on assumptions that are unrealistic in practice—for example, Theorem 1 assumes infinitely large hidden dimensions and zero-mean inputs and weights. These assumptions weaken the practical validity of the theoretical claims.

5. Since the proposed method is a PTQ approach, comparing it primarily against QAT baselines is not entirely fair or informative. The inclusion of more recent PTQ baselines would strengthen the experimental section.

6. What are the numbers in Table 7? Besides, the numerical results in Table 7 show substantial improvement, but the underlying reasons for such gains are not explained. A discussion on why these improvements occur is needed. 

7. What does “GS” denote—GraphSAGE? If so, do the authors have results on more recent GNNs or graph transformer architectures to demonstrate broader applicability?

8. DRA achieves only 1.75 % accuracy on Reddit GCN INT4 and 3.12 % on ogbn-products GCN INT4. Were these results reproduced by the authors or directly taken from the original publication?

9. The selection of k from the k nearest neighbors (mentioned in lines 232–237 and Figure 3c) are not specified. These details are crucial for reproducibility.

10. The reported speed-up in Table 5 appears modest compared to the claimed efficiency gains. Additional clarification or discussion is required to reconcile this discrepancy.

11. The paper focuses on the computation time speed up. However, no empirical results of memory cost are provided.  Will the proposed method introduce additional memory cost?

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TopGQ, a topology-aware post-training quantization (PTQ) framework for GNNs. The motivation is that existing GNN quantization methods are time consuming. 

TopGQ introduces two main components:
	1.	Dual-Axis Scale Absorption, which merges node-wise scaling factors into the adjacency matrix, enabling integer arithmetic compatibility while maintaining per-node quantization precision.
	2.	TopPIN (Topology-Aware Pairwise Index), a lightweight node index derived from local graph structure (node degree and neighbor degree) that predicts activation variance and allows quantization parameter reuse for unseen nodes in inductive settings.

Experiments across diverse GNNs and datasets show that TopGQ does reduce quantization time  while matching or exceeding sota accuracy.
Theoretical sections attempt to connect local topology to node activation variance and provide continuity arguments for quantization parameter sharing.

### Strengths
The experiments cover enough datasets and hardware setups( desktop GPUs and edge-devices)

Even if the industrial motivation is overstated, the efficiency gains would still be useful for frequent retraining or memory-constrained deployment.

### Weaknesses
The cited works discuss dynamic GNN training/inference, not quantization time bottlenecks. Thus, the paper’s claim that “quantization time is a deployment barrier” lacks direct evidence.

Dual-axis absorption is somewhat a reparameterization trick (scaling merge and Hadamard product rewrite). 

Theoretical analysis of TopPIN relies on strong assumptions without providing error bounds or practical ablations verifying these assumptions.

### Questions
Find a stronger motivation for effient Post-Training GNN Quantization.

### Soundness
2

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
5

### Summary
This paper introduces TopGQ, a topology-aware post-training quantization framework for graph neural networks. Its core contributions include: a dual-axis scale absorption technique that maintains the efficiency of integer arithmetic while achieving node-wise quantization accuracy, and a lightweight topological index (TopPIN) designed to rapidly predict quantization parameters for unseen nodes. TopGQ aims to provide a quantization solution for large-scale GNN deployment; however, several concerns regarding its novelty, methodology, and experimental evaluation require further analysis.

### Strengths
1. This paper proposes an interesting idea, supported by a discussion.

2. This paper exhibits a clear structure and is well-written.

### Weaknesses
This paper lacks enough innovation.

1. The core of the proposed method involves shifting a scaling diagonal matrix from left-multiplication to right-multiplication and merging it into the adjacency matrix. However it is a relatively common technique in numerical computing and compiler optimization. TopGQ does not introduce a fundamentally new approach to addressing the intrinsic challenges of GNN quantization, such as the highly non-uniform distribution of activations, as it does not alter the basic quantization paradigm but rather makes a more favorable quantization scheme (node-wise quantization) computationally feasible.

2. The two core contributions of the paper, i.e., "node-wise quantization" and "leveraging graph topology to guide quantization, are not novel concepts within the field. The studies such as A$^2$Q have already explored node-wise or mixed-precision quantization, and the use of centrality measures like node degree has long been established. The primary contribution of TopGQ lies in the straightforward combination of these two existing ideas, without substantial refinement of their underlying principles.

This paper lacks some necessary analysis.

1. The theoretical foundation of the paper (Theorem 1) relies on strong assumptions, such as Gaussian initialization, sufficiently large dimensions, and ReLU activation functions. However, the actual proposed TopPIN is a highly simplified first-order approximation. The transition from the intricate $\phi(v)$ to the simplified TopPIN lacks rigorous theoretical justification and ablation studies. This design choice is not sufficiently supported by analytical reasoning.

2. Does the effectiveness of TopGQ rely on the presence of a distinct topological structure that can be characterized by node degrees? In many real-world graphs (e.g., molecular or knowledge graphs), node features may play a far more critical role than topology, or the graph structure may be more complex than what degree alone can capture. Since DRA employs KL divergence for global range analysis and is applicable to various GNN architectures and datasets, does this make TopGQ's generality inferior to that of DRA?

The experimental design of the paper has some issues.

1. The comparison with relevant baselines is insufficient. When compared against numerous QAT methods, the observed speed advantage stems primarily from the inherent nature of post-training quantization (PTQ) being training-free, rather than from the novel contributions of this work. In terms of accuracy, TopGQ does not demonstrate a significant or consistent superiority over other methods.

2. The paper claims to achieve "orders of magnitude" speedup in quantization. To what extent is this acceleration attributable to the core concept of being training-free versus the specific implementations of TopPIN and dual-axis scale absorption? Shouldn't corresponding ablation studies be designed to isolate and validate the effectiveness of the proposed contributions?

There are obvious problems in the experimental analysis.

1. Figure 1 is not cited in the main text. Is the reference to "Section 1" in the INTRODUCTION section (i.e., in the last sentence of the first paragraph on Page 2) incorrect? Should it instead be referencing Figure 1?

2. The font size in some parts of Figure 3 is somewhat small. Increasing it would improve readability.

### Questions
1. Why was a first-order approximation chosen for TopPIN instead of a second or third-order approximation? Is this design choice supported by experimental evidence?

2. Does the effectiveness of TopGQ depend on the graph having a distinct topological structure that can be characterized by node degrees? How generalizable is the method to graphs with complex topological structures?

3. Does the effectiveness of TopGQ depend on the graph having a distinct topological structure that can be characterized by node degrees? How generalizable is the method to graphs with complex topological structures?

4. In the comparisons with DRA, under what conditions does TopGQ demonstrate superior performance, and in which scenarios does DRA hold the advantage?

### Soundness
2

### Presentation
3

### Contribution
2
