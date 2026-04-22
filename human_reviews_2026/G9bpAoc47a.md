# It Takes a Graph to Know a Graph: Rewiring for Homophily with a Reference Graph

- Avg Score: 2.80
- Decision: Reject
- Scores: 0, 4, 2, 4, 4

## Abstract
Graph Neural Networks (GNNs) excel at analyzing graph-structured data but struggle on heterophilic graphs, where connected nodes often belong to different classes. While this challenge is commonly addressed with specialized GNN architectures, graph rewiring remains an underexplored strategy in this context. We provide theoretical foundations linking edge homophily, GNN embedding smoothness, and node classification performance, motivating the need to enhance homophily. Building on this insight, we introduce a rewiring framework that increases graph homophily using a reference graph, with theoretical guarantees on the homophily of the rewired graph. To broaden applicability, we propose a label-driven diffusion approach for constructing a homophilic reference graph from node features and training labels. Through extensive simulations, we analyze how the homophily of both the original and reference graphs influences the rewired graph homophily and downstream GNN performance. We evaluate our method on 13 real-world heterophilic datasets and show that it outperforms existing rewiring techniques and specialized GNNs for heterophilic graphs, achieving improved node classification accuracy while remaining efficient and scalable to large graphs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper addresses the issue of weak performance of GNNs in homophilous graphs. It proposes to rewire the graph to increase its homophily. First, a reference graph is constructed based mostly on node feature similarity. Then, the original graph is rewired by adding or deleting edges to make it more similar to the reference graph, which increases its homophily. Finally, models are trained on the resulting graph.

### Strengths
The paper is mostly well-written and easy to follow.

### Weaknesses
- The main assumption behind the paper's approach is that GNNs provide weak performance on heterophilous graphs. However, this assumption is outdated: there have been multiple paper showing that GNNs perform perfectly fine on heterophilous graphs (in particular, they typically outperform specialized models), see [1-5]. The paper does not discuss these works at all. The paper also states as its motivation: "Standard GNNs are designed primarily for homophilic graphs, as they rely on the homophily assumption". This is also not the case: none of the papers that developed modern GNNs mention homophily at all. Note that this models do not even have explicit access to node labels, but rather work with node features. Nowadays, any claim that GNNs are designed exclusively for homophilous graphs or do not work well on heterophilous graphs should be supported by strong evidence that directly addresses works [1-5] and shows where they went wrong (and I am not aware of any such evidence).

- The motivation and statement of Theorem 1 seem confusing. Linearly separable embeddings are being discussed. However, there can be two types of linear separability for embeddings: all pairs of embeddings can be separable, or pairs of embeddings from different classes can be separable. What is needed for strong model performance is the second (intra-class) type of separability, but it seems that the paper discusses the first (all-pair) type of separability, which is not directly related to model performance. The paper claims that Theorem 1 ties GNN's ability to generate linearly separable embedding to graph homophily, but then Theorem 1 starts by assuming that the embeddings are linearly separable. It is also not clear how Theorem 1 proves some property of GNNs, when there is no GNN in the theorem. Finally, what is $A_{u,  v}$ in Theorem 1? I cannot find its definition, but it seems like $A$ denotes elements of the adjacency matrix. Then the minimum non-zero element is simply 1, because the adjacency matrix is binary (it is never mentioned that the graph is weighted, and later sections describing the method clearly assume unweighted graphs).

- The experimental results are unreliable. First, the paper uses Squirrel and Chameleon datasets (among others), that have been shown in [3] to be buggy. Cornell, Texas, and Wisconsin datasets have also been criticized in [3] (in particular, the Texas dataset has a class consisting of a single node). More importantly, the results reported in the paper for GCN, GAT, H2GCN and GPRGNN are significantly lower than those reported in [3] for those datasets that are used in both works. For example, GPRGNN achieves an accuracy of 64.85 on the Roman-Empire dataset in [3], while the current paper reports only an accuracy of 20.5 for it. Most notably, however, the current paper reports accuracies of 14.8 and 14.0 for GATv2 and APPNP respectively on the Roman-Empire dataset, which is roughly the performance of the naive majority class prediction (13.96). This clearly shows that the considered baselines were not well-tuned and thus the obtained experimental results cannot be trusted.



[1] Is homophily a necessity for graph neural networks? (ICLR 2022)

[2] Revisiting heterophily for graph neural networks (NeurIPS 2022)

[3] A critical look at the evaluation of gnns under heterophily: Are we really making progress? (ICLR 2023)

[4] Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond (NeurIPS 2023)

[5] Oversmoothing, Oversquashing, Heterophily, Long-Range, and more: Demystifying Common Beliefs in Graph Machine Learning (arxiv preprint)

### Questions
- How is the hyperparameter k (the number of rewired edges) in the proposed method selected? If the main point is to increase graph homophily, why not simply replace the original graph with the reference graph (i.e., rewire all the edges)?

- The reference graph construction is mostly based on node feature similarity. However, GNNs have access to node features. Why is it expected that the edges of the reference graph will provide GNNs with additional information?

- It seems like the sources of the datasets are not mentioned in the paper. In particular, what is the Elliptic/EllipicBitcoin dataset? If it is the dataset with the same name from Pytorch Geometric, then it has strong class imbalance, and accuracy is not an appropriate metric for it (and the naive majority class prediction provides better accuracy on it then the values reported for GNNs in the paper).

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a label-driven diffusion approach to construct a homophilic "reference graph." The goal is to rewire an original graph to improve downstream GNN performance. The paper includes extensive simulations analyzing how the homophily of both the original and reference graphs influences the final rewired graph and subsequent task performance.

### Strengths
1. The method is inherently scalable by design, as it uses a clustering-based approach as its first step.
2. The core idea of constructing a reference graph for label-driven rewiring is novel, yet the proposed solution is simple and elegant.

### Weaknesses
1. There is a significant disconnect between the paper's theoretical grounding and its practical implementation. The clustering step, while efficient, creates a problem:
    - Theories (Theorem 1, Proposition 1, Proposition 2) are derived from a *global* perspective, seemingly assuming all edges can be modified.
    - The implementation, however, performs rewiring *locally* (within clusters). The links *between* clusters are fixed and cannot be modified.
    - This implies the theoretical statements may not hold for the final reassembled graph. The theory must be revised to account for this restriction (i.e., that a subset of edges is immutable). Similarly, the *actual* global homophily may not change as expected due to these fixed inter-cluster links.
2. The paper's central motivation is that increasing homophily is beneficial for GNNs. However, this premise has been increasingly questioned in recent literature [1], which suggests homophily is not universally necessary or beneficial. This makes the paper's *exclusive* focus on refining homophily seem limited or dataset-specific. The authors must address this and justify their motivation in light of this conflicting research.
3. The introduction of a "reference graph" inherently doubles the storage and memory cost, as both the original and reference graph structures must be maintained. This is a significant practical drawback that is not adequately discussed or justified.
4. The method's performance seems highly dependent on the initial clustering step. How does the *balance* of these clusters (e.g., the variance in cluster sizes) influence the final rewiring and downstream GNN performance? This critical factor is not analyzed.

## Minor

1. **Table 1:** To improve clarity, it would be beneficial to add a row for "Averaged Gain" (or similar summary metric) in Table 1 to allow for a more direct comparison of the method's overall effectiveness against baselines.
2. **Figure 1 Analysis:** The analysis in Figure 1 is insightful. Do the same patterns regarding homophily's influence hold true for the other datasets used in the paper? Please include this analysis (perhaps in Appendix B) to demonstrate the generalizability of these findings.

[1] Is Homophily a Necessity for Graph Neural Networks? In ICLR, 2022

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents theoretical and empirical connections between smoothness, homophily/heterophily, and GNN performance. Based on these analyses, authors propose REFine, a feature and label-driven diffusion approach that rewires the graph to be more homophilous. Extensive experiments are conducted on homophilous and heterophilous benchmarks to demonstrate the utility of the approach.

### Strengths
1. The approach of graph rewiring for homophily/heterophily is intuitive and interesting
2. The paper is sound, clear, and well-written.

### Weaknesses
1. **Positioning vs. prior rewiring for heterophily (novelty).**  
  The core idea of REFine is to construct a more homophilous reference graph leveraging the graph **features + training labels** and then **adding/deleting edges**. To me, this is quite is close to **DHGR**, which also leverages similarities between node features and training labels to form the rewired graph. The similarities and differences of REFine in comparison to DHGR as well as its advantages/disadvantages are not clear. I believe the paper would benefit from a sharper distinction (beyond “simpler & guaranteed”) and **direct empirical comparisons** to DHGR/GSL baselines to make the contribution more clear. 

2. **Incremental theory relative to prior spectral/smoothness analyses.**  
  The main smoothness and homophily result (Theorem 1) is very close to existing analyses that link Dirichlet energy/smoothing to homophily/heterophily. One example of where this appears in prior works is in Theorem 3 in "Beyond Homophily in Graph Neural Networks" by Zhu et al., 2020. The subsequent propositions in the paper also state intuitive claims (if the reference graph is more homophilous, then adding its edges/deleting complement edges improves homophily) under certain conditions. Please clarify how these results meaningfully extend prior theory and provide new insights for the limitations of prior methods or advantages of REFine.

3. **Missing critical baseline and mixed empirical gains.**  
While the method largely outperforms the benchmarked rewiring techniques, it is **often comparable** to specialized heterophilous GNNs, with a standout improvement on **Roman-empire** but with no major gains elsewhere. Additionally, the absence of a **DHGR** comparison weakens conclusions about superiority among heterophily-oriented rewiring methods. Including DHGR would strengthen the empirical case.

4. **When to prefer rewiring over heterophilous GNNs.**  
I am unsure as to when a practitioner should choose heterophilous GNNs or REFine. Are there any benefits of using the rewiring technique over GNNs other than just performance? If the performances are similar, why should we want to use a rewiring method over a heterophilous GNN? The paper should clarify when to choose rewiring (e.g., datasets where feature/label-driven reference graphs are reliably more homophilous), and provide guidelines on when specialized heterophilous GNNs remain the better choice.

### Questions
My questions largely stem from the limitations: 

1. What are the advantages and disadvantages of REFine in comparison to DHGR? What are the conceptual differences given DHGR also constructs a rewired graph based on node features similarities and train labels?
2. How do the theoretical results meaningfully build on existing spectral analyses relating smoothness/energy to homophily/heterophily? 
3. Given the connection between oversmoothing, oversquashing, and heterophily, what are the connections between existing rewiring approaches that address oversmoothing and oversquashing to heterophily? why don't their constructions mitigate the heterophily problem?
4. Why is DHGR not a baseline? How does DHGR perform in comparison to REFine?
5. It would be good as to get an intuition for why we would consider using a rewiring approach over a heterophilous GNN. The heterophilous GNNs perform closely in many cases, and if we have established heterophilous GNNs, when is rewiring a better alternative?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the challenge of poor GNN performance on heterophilic graphs. It explores graph rewiring as a strategy to improve GNN performance with theoretical and empirical study. It specifically introduces a label-driven diffusion method to construct the reference graph from node features and available labels, to rewire graphs for increasing homophily. Extensive experiments could show the effectiveness of the proposed method.

### Strengths
1. The idea of addressing the homophily–heterophily issue through graph rewiring is interesting and provides a different perspective compared to traditional architectural modifications.

2. The paper combines theoretical analysis regarding the reference graph with a comprehensive empirical study, and the explanations are generally clear and well-motivated.

3. Some experimental results indeed show significant performance improvements over certain baselines.

### Weaknesses
1. The proposed method lacks clear novelty. Label-guided graph rewiring has already been studied, for example, in *Bose, K., Banerjee, S., & Das, S. (2025). “Can Graph Neural Networks Tackle Heterophily? Yes, With a Label-Guided Graph Rewiring Approach!” IEEE TNNLS*, and the core technical component "label-driven diffusion" used here appears directly derived from existing work Mendelman
& Talmon (2025)., which substantially weakens the contribution.

2. The paper does not convincingly justify why increasing graph homophily is the right direction. Ideally, models should handle both homophilic and heterophilic structures without modifying the graph itself. More puzzlingly, Table 2 shows that the proposed method, which increases homophily, performs better even on heterophilic GNNs (e.g., H2GCN) — this seems conceptually inconsistent and requires further discussion.

3. The baseline comparison is outdated and incomplete. The latest strong methods such as *Bose et al., 2025 (label-guided rewiring) and Barbero et al., 2023, “Locality-Aware Graph Rewiring in GNNs,” ICLR 2024*, and above TNNLS 2025 work are not included.

4. The organization and presentation need improvement. Many experimental results that belong in the main text are buried in the appendix, with duplicated tables (E.1 COMPLETE RESULTS seems same as Table.1), and no bottomline in many tables. The paper feels somewhat rushed.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces REFine, a graph rewiring framework designed to enhance graph homophily using a reference graph, thereby improving the performance of standard Graph Neural Networks (GNNs) on heterophilic graphs. The authors provide a theoretical foundation linking edge homophily, embedding smoothness, and GNN performance, which motivates homophily enhancement. They propose a principled rewiring method guided by a reference graph, with theoretical guarantees on homophily improvement under certain conditions. A label-driven diffusion process is introduced to construct a homophilic reference graph from node features and training labels.

### Strengths
S1 The paper establishes a clear theoretical connection between graph homophily and the smoothness of GNN embeddings (Theorem 1), providing a strong motivation for homophily-enhancing rewiring. 
S2 The experimental section is thorough, evaluating the method across a diverse set of 13 datasets with varying sizes and homophily levels.
S3 The use of the METIS algorithm for graph partitioning, coupled with parallelizable per-cluster computations, makes the method highly scalable to large graphs.

### Weaknesses
W1 The effectiveness of the reference graph construction hinges on the assumption that node feature similarity is indicative of label similarity. In domains where this assumption does not hold, the method's performance may be limited. While the label-driven diffusion aims to mitigate this, the fundamental dependency remains a potential limitation.
W2 The framework is specifically designed and evaluated for the node classification task. Its applicability to other fundamental graph learning tasks, such as graph classification or link prediction, is not explored or discussed, which may limit its perceived utility for the broader graph learning community.
W3 The method involves several key hyperparameters, including the kernel scale ε, the choice between edge addition/deletion, and the number of edges k to rewire. While a grid search is employed, the performance is contingent on proper tuning, which could be computationally expensive and less user-friendly in practice.

### Questions
Q1 The current framework performs either edge addition or deletion in a single rewiring step. Have the authors considered a strategy that performs both operations simultaneously, which could offer finer control over the resulting graph topology?
Q2 The reference graph is constructed in a fixed, non-learnable manner. Could the performance be further improved by integrating the reference graph construction into an end-to-end, trainable framework, perhaps drawing inspiration from GSL paradigms in future work?
Q3 How does the method perform on graphs that are extremely sparse (leading to potential disconnection) or excessively dense (where rewiring might have minimal relative impact)? Are there specific regimes where the method is less effective?

### Soundness
2

### Presentation
3

### Contribution
2
