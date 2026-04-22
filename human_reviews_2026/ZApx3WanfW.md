# MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Many real-world graphs, such as financial transaction networks, are edge-attributed multigraphs that feature multiple edges between the same pair of nodes, each with distinct edge attributes. State-of-the-art neural network solutions operating on such edge-attributed multigraphs either preprocess the multigraph by collapsing its multi-edges into a single edge or introduce auxiliary edge features that compromise permutation equivariance. We introduce MEGA-GNN, a graph neural network (GNN) for edge-attributed multigraphs, which overcomes these limitations by employing a two-stage aggregation process in its message passing layers: first, features of the multi-edges between the same two nodes are aggregated, and then messages from distinct neighbors are combined. We show that MEGA-GNN computes a richer set of statistical features than the GNNs that implement only single-stage aggregation in their message passing layers. We evaluate MEGA-GNN on seven financial transaction network datasets and three temporal user-item interaction datasets, demonstrating significant improvements in minority-class F1 scores for illicit transaction detection and ROC-AUC scores for user state-change prediction, respectively, compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MEGA-GNN, a graph neural network structure designed for multigraphs with edge attributes, addressing an important problem with potential applications across various domains. The paper is well written and easy to follow, and experiments are conducted on several real datasets for different tasks. However, the technical novelty of MEGA-GNN is limited, as its integration with existing GNN baselines and the two-stage aggregation approach are not fundamentally new. The architecture closely resembles standard GNNs, with only minor extensions for multiedge aggregation. Additionally, the experimental evaluation lacks depth, and the presentation of results is unclear, making it difficult to assess improvements over existing methods.

### Strengths
1. The paper addresses an important problem concerning multigraphs with edge attributes and proposes a graph neural network structure suitable for various applications.
2. The paper is well written and easy to follow.
3. The experiments are performed on several real-world datasets across different applications.

### Weaknesses
1.	At line 92, the authors state that MEGA-GNN can seamlessly integrate with GNN baselines; however, this may not be a significant advantage or contribution, as it suggests that the proposed MEGA-GNN has limited novelty or differences compared to existing GNNs. This actually highlights the limited technical contribution of this work.
2.	The two-stage aggregation in Definition 4 appears natural but lacks novelty, as it simply introduces another layer of aggregation over the single-stage aggregation. This idea is not new.
3.	In Section 3.3, apart from the aggregation over multiedges, the approach is quite similar to existing GNN architectures. Moreover, the aggregation design in Section 3.3 does not present significant challenges.
4.	The experiments are conducted on three quite different tasks, which makes them more general but insufficient in depth.
5.	There are many existing methods for ETH node classification that should be compared. For example:
-	Enhancing graph neural network-based fraud detectors against camouflaged fraudsters
-	Effective Illicit Account Detection on Large Cryptocurrency MultiGraphs, CIKM 
-	BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection. In Proceedings of the ACM Web Conference

I understand that these studies are focused on ETH anomaly detection, but fundamentally, their methods are designed for multi-graph neural network classification.

### Questions
1.	The presentation of the results in Figure 3 is confusing and difficult to understand. Why not provide a comprehensive table of results for the AML datasets? It is hard to determine whether your method improves upon existing methods.

### Soundness
2

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
This paper introduces MEGA-GNN, a novel graph neural network designed to address the challenges of modeling multi-edge interactions in graphs. The method utilizes a two-stage aggregation process, is designed to be permutation equivariant or invariant, and is evaluated on seven real-world datasets, showing practical application in areas like transaction networks.

### Strengths
- The paper tackles a practical and important problem (multi-edge graphs) that is common in real-world scenarios, such as transaction networks, but often overlooked in GNN research.
- The proposed method is proven to be a universal approximator. It is also shown to be capable of distinguishing between edges originating from the same neighbor versus those from different neighbors, a key capability for multi-edge graphs.
- The core improvement—the two-stage aggregation—is relatively simple to understand yet demonstrates effective results in the experiments.

### Weaknesses
- A significant clarification is needed regarding the use of timestamps. Timestamps are inherently ordered, which seems to conflict with the model's claim of permutation invariance.
    - How does the model handle this apparent contradiction?
    - Furthermore, does the model simply treat timestamps as static features, or does it properly account for the **temporal and causal relationships** implied by transaction data?
- The two-stage aggregation (separating edge-type aggregation) appears *prima facie* to be more computationally complex than a standard RGCN, which aggregates all edge types in a single step.
    - The paper lacks a clear computational complexity analysis. How does MEGA-GNN compare to RGCN in terms of time and memory?
    - Are there any efficiency benefits? If not, the authors should provide clear recommendations for practitioners on the trade-offs: when should one use MEGA-GNN (e.g., for high-dimensional edge attributes) versus a standard RGCN?

**Minor Comments**

1. The meaning of the '$+$' operator in '$m_A(x) + m_B(x)$' is ambiguous. Please explicitly define whether this represents element-wise addition, concatenation, or some other operation.
2. Figures 1 and 2 share many elements and convey very similar information. It would be clearer and more concise to merge them into a single, comprehensive figure.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MEGA, a message-passing framework for multigraphs that operates in two stages: first aggregating over multiple edges between the same pair of nodes, and then aggregating over different neighbors. This two-level design allows the model to distinguish intra-neighbor and inter-neighbor relations, which standard GNNs usually cannot. The authors also provide formal proofs showing that this two-stage mechanism is strictly more expressive than single-stage aggregation and maintains permutation equivariance.

### Strengths
1. The two-stage message passing mechanism is intuitive yet effective, and it directly addresses a structural limitation of current GNNs on multigraphs. The formal proof that this design is strictly more expressive than single-stage aggregation is quite convincing.

2. The paper carefully formalizes permutation equivariance and provides a universality result (under an edge ordering assumption). These theoretical discussions are coherent and help justify the architectural choices.

3. The experiments are thorough within their chosen domains, and MEGA achieves steady improvements over multiple strong baselines. The results are reproducible and the code organization appears clean and principled.

4. It’s nice to see that the authors didn’t just stop at the theoretical part, they also validated scalability and provided a reasonable complexity analysis, showing that the model can be trained efficiently in practice.

### Weaknesses
1. The universality argument depends on having a strict total order over edges. That’s fine for datasets with clear timestamps, but in reality many edges can share identical times or even lack ordering info. It’s not entirely clear how the model behaves in those cases, or whether the ordering assumption limits the practical generality.

2. There are some nice comparisons showing the effect of the two-stage design, but the analysis feels a bit narrow. For instance, it’s unclear how different aggregation choices or varying numbers of edge types would change the results. I would have liked to see a more systematic exploration here — even a small controlled study would help.

3. The datasets are mid-sized, and one of the baselines already runs out of memory. That makes me wonder how MEGA would perform on larger, real-world graphs. Some evidence or discussion about scaling behavior would make the contribution stronger.

4. Most results come from financial transaction or interaction graphs. That’s a relevant testbed, but I’m not sure how general these conclusions are beyond that context. It would be nice to see at least one example from another kind of multigraph, maybe something in social or transportation data.

5. The paper shows aggregate runtime comparisons, but not much insight into where the actual cost comes from. Since the model introduces an additional aggregation stage, some operator-level breakdown or profiling would make the trade-offs clearer.

### Questions
1. How does the model handle cases where multiple edges share the same timestamp or lack ordering information? Is there a deterministic tie-breaking rule, and does that affect permutation equivariance or expressiveness?

2. Could the authors provide more insight into how the two aggregation stages are parameterized? For example, is there any adaptive mechanism to select the number or type of aggregators based on edge multiplicity?

3. How expensive is the artificial-node construction in practice when using neighbor sampling? Does it cause additional indexing or memory overhead during batch training?

4. For the temporal datasets, how do you ensure there’s no information leakage across time? Are all multi-hop neighbors restricted to the same snapshot, or could future edges appear in the receptive field?

5. Have you considered pretraining or reweighting strategies to further mitigate the strong class imbalance in AML datasets? Some contrastive or cost-sensitive techniques might improve performance in the low-positive regime.

6. It would be interesting to see a plot showing how throughput and memory usage scale with the average number of edges per node. This would make the practical benefits of MEGA clearer to readers who care about deployment efficiency.

### Soundness
3

### Presentation
3

### Contribution
3
