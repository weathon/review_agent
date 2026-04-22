# From Contextual Distributions to Messages: Entropy-Guided GNNs

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
The Message Passing Neural Networks (MPNNs) have emerged as the dominant framework for learning on graphs. However, their expressive power is fundamentally restricted by the 1-dimensional Weisfeiler-Lehman (1-WL) test. To further improve the expressive power of MPNNs, existing methods mainly rely on higher-order or subgraph WL tests, that usually require significantly increased memory usage and computational overhead. The aim of this paper is to address the above limitations by introducing a novel message passing paradigm, that can effectively encode and propagate the structural distribution of context around each node. Instead of directly performing the message passing on $k$-tuples or subgraphs, our method encodes and propagates structural information through a compact distributional statistic, i.e., the entropy of the node context. Furthermore, we propose a kernel-based aggregation scheme to quantify the structural distribution similarities between the contexts of different nodes. Theoretical analysis and empirical evaluations indicate that the proposed framework not only achieves higher expressive power but also significantly reduces computational and memory costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ConD-MPNN, a framework that enhances GNN expressiveness by propagating entropy-encoded representations of node contexts instead of performing message passing on k-tuples or subgraphs. The method claims to achieve better expressiveness than 1-WL while maintaining linear complexity. A kernel-based variant is introduced to handle distributional discrepancies. Experiments on synthetic expressiveness benchmarks and real-world datasets show competitive performance.

### Strengths
1. The paper clearly identifies the computational bottleneck of subgraph GNNs and higher-order methods, and the proposed entropy-based encoding is conceptually sound as a compact structural representation.

2. The paper includes evaluations on synthetic benchmarks, real-world graph classification, and regression tasks. The ablation studies and scalability experiments across different backbones  demonstrate thoroughness.

3. The visualizations in Figure 3-4 effectively illustrate how the method works.

### Weaknesses
1. The paper positions itself as an efficient alternative to subgraph GNNs but fails to compare against recent structural encoding methods that make identical efficiency claims, such as [1-3]. Without these comparisons, it is hard to evaluate whether the proposed method actually provides advantages over state-of-the-art structural encodings. 

2. Definition 1's "k-hop context construction using shortest path distance" appears to similar to previous studies such as [1,4], which explicitly constructs contexts at distance k and processes k-hop neighborhoods separately. 

3. The claim that "training is $O(|V|·d)$, same as standard MPNN" is misleading because it ignores that features are augmented to dimension $d^k$, making true training complexity $O(|E|d^{k})$. In the experiments, it would be better to provide empirical comparison with the structural encoding based methods, as they also aim to improve the computational efficiency, which share the similar goal with the proposed method. 

4. Claiming "strictly more powerful than 1-WL and 2-WL" is baseline expectation, many 2023-2024 papers exceed this. The field has moved toward quantitative characterizations (which specific cycles/cliques can be counted at which levels) rather than qualitative WL comparisons. This point makes this work incremental. 

5. The work also missed some direct baselines, such as the structural encoding based methods mentioned earlier and 2-FGNN [5], to name a few.

---

**Reference**

1. Yao et al., Improving the Expressiveness of 𝐾-hop Message-Passing GNNs by Injecting Contextualized Substructure Information

2. Bao et al., Homomorphism Counts as Structural Encodings for Graph Learning

3. Kanatsoulis et al., Learning Efficient Positional Encodings with Graph Neural Networks

4. Abbound et al., Shortest Path Networks for Graph Property Prediction

5. Zhang et al., Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness

### Questions
1. Can you provide direct empirical comparisons against structural encoding based methods in terms of efficiency and performance.

2. The $O(|V|·d^{3k})$ preprocessing, for k=3 with typical dimensions, is this prohibitively expensive?

### Soundness
3

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
This paper proposes ConD-MPNN, a message-passing framework that augments node features with a summary of each node’s k-hop “context” encoded by the von Neumann entropy. A kernelized variant (KerA-MPNN) aggregates messages using a reproducing kernel defined on these context entropies. The paper claims (i) strictly higher expressive power than 1-WL and 2-WL, with the ability to distinguish “many” regular graphs that 3-WL fails on; (ii) comparable training-time complexity to standard MPNNs (messages depend on precomputed scalars); and (iii) improved performance on several benchmarks.

### Strengths
S1: The paper proposes a simple, general plug-in that augments messages with a scalar contextual statistic (entropy), which is architecturally lightweight and could be dropped into many MPNN variants. 

S2: The approach achieves many best-in-column results on QM9 targets, and reports very strong BREC performance -- validating its expressiveness claims and demonstrating its in practice usefulness.

### Weaknesses
W1: The presentation, motivation and methodology are difficult to follow.
W1a: Figure 1 is not self-contained and does not aid understanding. The caption fails to explain the illustration, what the different node colors denote, or what the histograms represent.
W1b: Figure 2 has the same shortcomings as Figure 1 and likewise adds little to the paper; both read as fillers rather than explanatory figures.
W1c: The theoretical statements are informal. For example, Theorem 2 begins “For this specific class of d-regular graphs with V nodes” but never defines the class. Theorem 3 mentions “significantly different node context distributions” informally and concludes that the method “can effectively reduce these differences,” which is also vague. Theorem 1 should be a proposition: by adding auxiliary information to a 1-WL GNN one can trivially subsume the 1-WL test and, as the 1-WL test equals the 2-WL, gnn can subsume it  as well.
W1d: It is unclear if the example in Figure 3 is representative or an outlier. Instead one would need to plot a measurement over multiple nodes. 

W2: Unsupported efficiency claims. The assertion of “significantly reduced computational and memory costs” is not backed by end-to-end time and memory comparisons. The asymptotic discussion in the appendix is likewise unconvincing: since the subgraph size s in subgraph GNNs can be fixed as a constant, their asymptotic computational complexity matches that of the proposed method. The claim is therefore unsupported both asymptotically and empirically.

W3: Reproducibility gaps. The reproducibility statement promises code in the supplementary materials, but no link is provided. Important hyperparameters and data-processing choices are missing or scattered across the appendix. This is especially concerning given the very strong BREC results.

W4: Related work placement. The main paper lacks an in-depth comparison with competitive methods; a related-work section can be is found in the appendix, which reinforces the broader presentation issues.

### Questions
Q1: Could you please formalize Theorems 2 and 3 (including precise definitions/metrics)?

Q2: Can you please report total runtime (including preprocessing) and peak memory vs. subgraph GNNs and k-GNN baselines on (a) a large sparse graph and (b) a dense graph, to support the “significant reduction” claim.

Q3: For Figure 3, how was the center node chosen and why 14 neighbors? Is this example representative or an outlier? Could you aggregate this analysis across many nodes/graphs (e.g., distribution of entropy spread before/after kernel mapping)?

Q4: Can you provide an anonymous link for the code so that reviewers can verify the performance (especially BREC)?

### Soundness
3

### Presentation
1

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
This paper proposes a GNN framework that propagates not only node features but also structural information of ego-subgraphs. The proposed framework can be broken down into the following key steps:

1. **Node Context Extraction**: For each node \(u\), the framework extracts its local context, denoted as \(S_u^k\), which represents the ego-subgraph centered at \(u\) and extending \(k\) hops outward. This ego-subgraph encompasses all nodes within \(k\) hops of \(u\) as well as the edges connecting them, effectively capturing the structural neighborhood around \(u\).

2. **Graph Entropy Computation**: The structural properties of each node's context \(S_u^k\) are encoded using graph entropy, specifically the von Neumann entropy. This computation produces a compact scalar representation that encapsulates the complexity and structural patterns of the ego-subgraph, providing a concise summary of its local topology.

3. **Entropy-Augmented MPNNs (ConD-MPNN)**: The computed entropy values are integrated into Message Passing Neural Networks (MPNNs) as supplementary features for neighboring nodes. This augmentation leads to the development of the ConD-MPNN model, which enhances both the expressiveness and computational efficiency of traditional MPNNs.

4. **Kernel-Based Structural Similarity (KerA-MPNN)**: To further enrich the framework, the entropy values of two ego-subgraphs are leveraged to compute a reproducing kernel between corresponding nodes. This kernel is incorporated into MPNNs as an additional edge feature, capturing structural similarities between the central node and its neighbors. This extension results in the KerA-MPNN model, which significantly improves the framework's ability to encode and utilize structural information.

### Strengths
1. Incorporating structural information of subgraphs into existing GNN models is a novel and promising approach to enhance model performance.
2. The paper provides formal results proving that the contextual distributional encoding (ConD-WL and ConD-MPNN) is strictly more expressive than 1-WL and 2-WL, and empirically distinguishes many regular graphs that are missed by 3-WL (Theorem 1, Theorem 2, and proofs in Appendix A.3, A.5).
3. Figures, such as Figure 1, cogently illustrate how the node context distributions (represented visually as colored histograms on graphs) are incorporated into message passing, offering clearer intuition than is common in related work.
4. The authors provide both theoretical and practical reasons for preferring von Neumann entropy over alternatives such as Shannon entropy, confirmed by discussion in Appendix A.4 and supporting experiments.
5. Reproducibility is well supported, with detailed methodology, extensive appendices, and code made available.

### Weaknesses
1. **Limited Novelty Relative to Advanced GNNs:** While the authors claim that the message passing scheme propagating both node features and contextual distributions is novel, it should be noted that message passing frameworks incorporating node features and structural information have been previously explored in the literature. For instance, [4] presents a similar approach where structural information is integrated with node features in classic GNN architectures. Thus, the claim that this framework represents a significant shift from operations on node-tuples or subgraphs to the structural distribution of node contexts appears overstated.
2. **Insufficient Theoretical Validation of the Reproducing Kernel:** While the authors claim that the proposed context entropy reproducing kernel captures similarities between different structural distributions, they do not provide rigorous mathematical proof that this kernel actually encodes meaningful structural similarities. The theoretical analysis is limited to Theorem 3, which focuses on reducing discrepancies rather than formally establishing the kernel's ability to capture structural similarities. A comprehensive mathematical analysis would require a thorough examination of the kernel's properties and its relationship with the underlying structural distributions. Such rigorous theoretical validation is essential for acceptance in top-tier AI conferences.
3. **Insufficient Theoretical and Empirical Validation of Expressivity:** While the paper compares ConD-MPNN against 1-WL, 2-WL, and 3-WL tests, it lacks theoretical and empirical comparison with k-hop MPNNs. Since the evaluation relies heavily on k-hop MPNNs with structural information, a theoretical comparison with k-hop MPNNs is needed to properly assess the expressivity gains. Furthermore, empirical results of the native k-hop MPNNs on BREC, QM9, ZINC, and Peptides datasets are needed. Without these baselines, the improvement achieved by the proposed methods over standard k-hop MPNNs cannot be quantified.
4. **There are several critical issues in the procedure for constructing the $k$-hop neighborhood:**
    1. The phrase `random walk distance' lacks a widely accepted or standardized definition, potentially causing confusion and ambiguity.
    2. Line 7 is redundant since setting $A_{\text{final}}[A_{\text{final}} > 0] \gets 0$ always results in a zero matrix, negating its intended purpose. Consequently, the operation in line 8 simplifies to $A_{\text{final}} = A^i$. As a result, $A_{\text{final}}[u,v] > 0$ fails to indicate whether the shortest-path distance between nodes $u$ and $v$ equals $i$.
    3. Lines 10-12 should be moved outside the for loop (line 2) since the algorithm returns the context extracted from the fully accumulated neighborhood, making their current placement within the loop redundant.
    4. Line 5, $A_{\text{final}} \gets A_{\text{final}} + A^i$, ensures $A_{\text{final}}[u,v]>0$ when the shortest path distance between $u$ and $v$ is $\leq k$. Thus, the 'random walk' method actually extracts node contexts based on shortest path distances.
5. The main text lacks a clear introduction to the key backbone model, the K-hop MPNN framework, which makes it difficult to fully understand the proposed framework and its components.
6. Two backbones are both named K-GIN: one from (Morris et al., 2019) and another from (Nikolentzos et al., 2020). This shared naming confuses readers. Especially, it is unclear which K-GIN variant is used for ConD-KGIN and KerA-KGIN in Table 4, 5 and 10. In addition, Both K-GIN (Nikolentzos et al., 2020) and KP-GIN (Feng et al., 2022) are k-hop MPNNs. The authors do not clearly distinguish between K-GIN and KP-GIN before the comparison, which may confuse readers.
7. In line 355, the authors claim that both ConD-based and KerA-based MPNNs can distinguish graphs that are not distinguishable by the 4-WL test. However, there is insufficient theoretical or comprehensive empirical evidence provided to substantiate this strong claim.
8. The improvements achieved by the proposed methods on TU datasets are limited.
9. The evaluation on QM9 and ZINC lacks comparison with recent SOTA models such as PPGN+RRWP[1], N$^2$-GNN[2], and GRIT[3]. The proposed methods perform worse than these SOTA models, particularly PPGN+RRWP, failing to demonstrate superiority over recent SOTA models.

### Questions
1. How does the expressive power of the proposed graph-entropy encoding compare to the k‑FWL hierarchy?
2. How does ConD‑KGIN’s expressivity compare to existing k‑hop MPNNs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes to incorporate structural information in the message passing mechanism; the structural information is represented by the von Neumann entropy of the k-hop subgraph of the target node. The results show improved performance when this structural information is encoded in message passing.

### Strengths
- The use of von Neumann entropy metric to incorporate structural information is a novel and interesting exploration in message passing on graphs.

### Weaknesses
- The presentation is unclear, and the entire paper is difficult to understand. For example, the functional forms of **U** and **M** in equations 7, 8, and 9 are not given. This makes it difficult to understand the proposed method. Also, what is the dimensionality of the linear transformation $\phi$?

- It is difficult to conclude that von Neumann entropy is better than Shannon entropy from Table 16. The performances are not statistically different considering the standard deviation.

- The result of the proposed methods in Table 3 compared to GIN and K-GIN are not statistically different. This does not validate the effectiveness of the proposed message passing strategy. Also, the extension to GCN and GraphSAGE backbones in Table 9 has no significant improvement over the baseline.

- It is not clear why K-GIN and KP-GIN are excluded from the baselines in Table 2.

- The kernel-based structural similarity does not seem to bring any advantage compared to the ConD variant; the performances are not statistically different between the two methods.

I recommend that the authors substantially revise the paper to improve the mathematical clarity of the proposed methods and to further investigate why incorporating structural information does not yield meaningful performance gains. The current results directly challenge the core motivation behind injecting additional structural signals for representation learning: the results suggest that message passing schemes in standard architectures such as GCN and GraphSAGE may already implicitly capture the relevant structural information, and that explicitly adding more structure offers no tangible benefit in cases os benchmark or real-world graphs.

### Questions
- It is not clear why "kernel-based aggregation strategy effectively mitigating the discrepancies in node context distributions" leads to better performance.
- In Figure 3 (a), does the entropy value depend on the central node?

### Soundness
2

### Presentation
1

### Contribution
2
