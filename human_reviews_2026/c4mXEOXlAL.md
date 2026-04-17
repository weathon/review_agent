# gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4, 6

## Abstract
Graph Neural Networks (GNNs) leverage the graph structure to transmit information between nodes, typically through the message-passing mechanism. While these models have found a wide variety of applications, they are known to suffer from over-squashing, where information from a large receptive field of node representations is collapsed into a single fixed sized vector, resulting in an information bottleneck. In this paper, we re-examine the over-squashing phenomenon through the lens of model storage and retrieval capacity, which we define as the amount of information that can be stored in a node’s representation for later use. We study some of the limitations of existing tasks used to measure over-squashing and introduce a new synthetic task to demonstrate that an information bottleneck can saturate this capacity. Furthermore, we adapt ideas from the sequence modeling literature on associative memories, fast weight programmers, and the xLSTM model to develop a novel GNN architecture with improved capacity. We demonstrate strong performance of this architecture both on our capacity synthetic task, as well as a range of real-world graph benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper redefines *over-squashing* in Graph Neural Networks (GNNs) as two distinct failure modes: **insufficient sensitivity** and **storage capacity saturation**, arguing that existing evaluations often conflate the two. The authors introduce a synthetic task called **Neighbor Associative Recall (NAR)** that isolates and measures capacity alone. Building on this task, they propose a message-passing architecture called **gLSTM**, which incorporates **matrix-based associative memory** and **xLSTM-style gating**. Additionally, they employ **K-hop aggregation** to enhance long-range interactions. The proposed method demonstrates clear capacity advantages on the NAR task and achieves competitive or state-of-the-art results on several GPP and LRGB benchmarks. The authors also analyze the relationship between capacity and sensitivity using Jacobian and Hessian-based measures.

### Strengths
- The paper clearly separates “capacity” and “sensitivity” when analyzing over-squashing, and introduces a shallow, controllable synthetic task (NAR) that isolates and measures storage capacity. The experiments and quantitative analyses strongly support this formulation.


- The proposed gLSTM architecture combines matrix-based associative memory with fast-weight (outer-product) updates, which is well aligned with the message-passing paradigm in graphs; the derivation and implementation details are sound and well explained.


- The method shows strong performance on multiple long-range dependency benchmarks, with comprehensive ablation studies (on gating, K-hop aggregation, positional encoding, etc.) and full hyperparameter tables.


- The paper is reproducible and transparent: code and configurations are released, and the training framework is standardized and well documented.

### Weaknesses
- Coupling between K-hop aggregation and capacity claims:

In real-world benchmarks, the performance gains may stem not only from increased storage capacity but also from enhanced sensitivity due to the denser computation graph created by K-hop aggregation. This coupling makes it difficult to attribute improvements solely to capacity. The paper should include controlled comparisons that fix K-hop settings while varying only the memory and gating components, alongside sensitivity metrics.

- Lack of efficiency and scalability analysis:

The matrix memory in gLSTM increases computational and memory costs quadratically with the hidden dimension. Although parameter counts are aligned across models, the paper does not report training/inference time or peak memory usage. Adding throughput and GPU memory comparisons with standard baselines (GCN, GIN, GCNII, GPS, etc.) would strengthen the empirical claims.

### Questions
- Under the same K-hop and parameter budget, how does gLSTM compare to equivalent non-memory MPNNs (wider or deeper) in both performance and sensitivity metrics such as Jacobian or mixed second-order derivatives?

- Please report training/inference time and peak memory consumption, and provide scaling curves showing how efficiency varies with memory dimension and the number of heads.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes gLSTM, a graph neural architecture that integrates an LSTM-like gating and memory mechanism into the message-passing process of GNNs. The authors argue that by maintaining global and local memory states, the model can alleviate over-squashing problem.

### Strengths
1. The idea of combining LSTM-style gating with graph message passing is conceptually intuitive and aligns with the goal of controlling information flow.

2. The experiments on the synthetic Neighbor Associative Recall (NAR) task provide some evidence that the model can store multiple neighbor features effectively.

3. The overall writing is easy to follow.

### Weaknesses
1. The main novelty of the paper is the adaptation of xLSTM to GNNs. While the motivation is reasonable, most design components are directly inherited from prior LSTM or xLSTM works. The paper feels more like an engineering extension rather than a fundamentally new GNN architecture. The authors should clearly articulate what is truly new in the design beyond re-using LSTM elements.

2. Although the idea of using LSTM to control message passing is intuitive, the technical motivation for several design choices is insufficiently explained. For instance, the roles of 𝑛𝑢 and 𝑚𝑢 on graphs are not clearly defined—are they merely borrowed normalization and stabilization terms from xLSTM, or do they have specific graph-theoretic meaning? This ambiguity makes it difficult to understand how these terms contribute to mitigating over-squashing.

3. The paper lacks a clear description of the datasets and settings. Although LRGB and synthetic graphs are mentioned, there is no systematic list of datasets, splits used or how baselines are chose. It remains unclear how results generalize to standard benchmarks such as OGB datasets or larger real-world graphs.

4. The main analysis of over-squashing is performed on the synthetic NAR task. While this toy setting helps isolate capacity effects, it is unclear whether the claimed improvements hold on realistic graphs. Results on real-world datasets are necessary to substantiate the claim.

5. Recent Graph Transformers and other scalable GNNs have been shown to alleviate over-squashing via global attention mechanisms. However, these models are not compared, leaving it unclear how gLSTM performs relative to current state-of-the-art methods.

6. The proposed design may introduce significant computational overhead due to additional gating, normalization, and memory matrices. No runtime or complexity analysis is provided, especially for large graphs such as papers100M.

7. Although the empirical intuition is sound, the paper lacks a rigorous theoretical justification for why the LSTM-style mechanism fundamentally reduces over-squashing. A formal analysis would strengthen the claims.

8. The experiments only include graph-level classification and synthetic tasks. There are no results on node classification or link prediction, which are precisely the domains where over-squashing is more prominent. This omission limits the paper’s empirical impact.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits over-squashing in message-passing neural networks by emphasizing a **capacity** perspective in addition to the usual **sensitivity** view. It introduces a synthetic evaluation, **Neighbor Associative Recall (NAR)**, designed to isolate node storage capacity in a shallow setting. Building on sequence-modeling ideas, the authors propose **gLSTM**, a GNN architecture with associative (matrix) memory and gating, paired with **K-hop aggregation**. Experiments on NAR and standard graph benchmarks suggest that increasing node-level storage capacity improves long-range reasoning while remaining competitive downstream.

### Strengths
1. **Clear conceptual separation.** The work carefully distinguishes capacity-related vs. sensitivity-related over-squashing and makes the capacity lens the central object of study.
2. **Purpose-built synthetic task.** NAR is a targeted probe for node storage limits that avoids confounding depth-driven sensitivity effects common in prior tests; the setup and metrics make the evaluation legible.
3. **Methodological innovation.** gLSTM adapts associative memory and exponential gates (in the spirit of xLSTM) to graphs, coupled with K-hop aggregation to broaden accessible context. The design choices are motivated and technically transparent.
4. **Empirical rigor and diagnostics.**
    * On NAR, gLSTM maintains **high recall** and then **gracefully saturates** near the memory limit, consistent with the capacity hypothesis.
    * On downstream benchmarks (e.g., GPP Diameter/Eccentricity; LRGB Peptides-Func), results are strong/competitive within a parameter budget.
    * Jacobian/Hessian-style sensitivity analyses help separate capacity effects from sensitivity/optimization effects.

### Weaknesses
1. **Limited Theoretical Guarantees**:  
   As noted in the conclusion, the capacity notion is supported primarily through intuition and empirical performance. There is currently no theoretical quantification or formal bounding of node storage capacity for different architectures (gLSTM, GCN).

2. **Dependency on K-hop Aggregation**:  
   Much of gLSTM's performance gain (Appendix B.2, Table 3 and onward) appears to come from K-hop aggregation, raising the question of whether capacity improvements are due to associative memory or simply enhanced information accessibility.
3. **Baseline Limitations**:  
   The experimental section primarily compares gLSTM to GCN and occasionally to a few other standard models (Table 1); Particularly, the lack of evaluation against strong geometric or spectrum-preserving approaches is an empirical gap.

### Questions
1. How does gLSTM’s time and memory footprint scale as node and edge counts grow? In practice, do you observe memory bottlenecks that differ between sparse and dense graphs.

2. What motivated the choice of K-hop neighborhood specifically, and could alternative aggregation schemes provide similar or even better empirical benefits? Could the authors provide more detail/access to K-hop ablation results in the main body?

3. Could the authors give more interpretative diagnosis when gLSTM or GCN "breaks down" (e.g., NAR neighbor count exceeding memory)?

4. Please include strong capacity-oriented and rewiring baselines for both NAR and downstream tasks, or provide a clear and compelling justification for their omission. In the absence of such comparisons or reasoning, I will have to lower the score due to insufficient empirical validation.

### Soundness
3

### Presentation
2

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
The paper identifies two distinct mechanisms behind “over-squashing” in MPNNs (message-passing graph neural networks): (i) sensitivity over-squashing and (ii) capacity over-squashing. It introduces a shallow synthetic task (Neighbor Associative Recall, NAR) that isolates capacity from sensitivity, and proposes gLSTM, an MPNN whose node states are augmented with an associative-memory matrix updated via fast-weight outer products. On NAR, gLSTM retains high recall until the number of neighbors equals the memory dimension, while vanilla GCN fails much earlier. gLSTM also obtains good performance on some real-world benchmarks.

### Strengths
1. The paper is well-written and well-motivated with the synthetic tasks. The problem targeted is of real significance. 
2. It is wise to disentangling capacity from sensitivity to give an in-depth analysis of capacity over-squashing.

### Weaknesses
1. The proposed method is a simple combination of existing techniques, particularly applying xlstm to graph structures. The tech novelty should be further strengthened. 
2. The proposed method may suffer from high computational overhead, there should be relevant analysis. Moreover, it is unclear the size of the tested benchmarks. It is advised to show that the proposed method can be applied to graphs with more than 1M nodes.

### Questions
How would the proposed method perform when applied to deep mpnns?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the over-squashing phenomenon in GNNs by focusing specifically on storage capacity as distinct from sensitivity. The authors introduce a new synthetic benchmark, NAR, designed to isolate and directly measure capacity over-squashing. Inspired by xLSTM and fast weight programmers from sequence modeling, the authors propose a new GNN architecture (gLSTM) to boost storage capacity. Empirical evaluations demonstrate the effectiveness of gLSTM on the NAR task and several long-range graph benchmarks.

### Strengths
- The paper provides a rigorous re-examination of over-squashing, explicitly distinguishing between storage capacity and sensitivity, with careful argumentation. It clarifies and extends understanding within the community.

- The NAR task is a well-designed, controlled test for probing capacity bottlenecks. It enables discrimination between capacity and sensitivity effects, addressing limitations of prior synthetic benchmarks.

- The paper provides detailed mathematical descriptions of both baseline GNNs and the proposed gLSTM mechanism.

- The authors have conducted extensive ablations and sensitivity analyses to support their claims. Jacobian and Hessian analyses also enhance interpretability.

- The experiments show that gLSTM achieves state-of-the-art or competitive performance in both the synthetic NAR task and real-world benchmarks.

### Weaknesses
1. The related work section omits several directly relevant recent studies [1, 2].  Besides, the authors are encouraged to discuss the difference between over-smoothing [3, 4] and over-squashing. 

2. While the authors propose the NAR task and empirically study capacity limits, there is a lack of formal and quantitative capacity analysis.

3. The move from vector to matrix memory increases parameter counts and computational complexity. There is little analysis or discussion about it.

4. Direct empirical comparison to a broader set of anti-over-squashing methods is missing.

Refs:

[1] Schreier-Coset Graph Propagation, Arxiv 2025.

[2] Over-Squashing in GNNs and Causal Inference of Rewiring Strategies, Arxiv 2025.

[3] SkipNode: On alleviating performance degradation for deep graph convolutional networks, TKDE 2024.

[4] Dropedge: Towards deep graph convolutional networks on node classification, ICLR 2019.

### Questions
Please address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
