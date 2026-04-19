# HyperPLR: Hypergraph Generation through Projection, Learning, and Reconstruction

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Hypergraphs are essential in modeling higher-order complex networks, excelling in representing group interactions within real-world contexts. This is particularly evident in collaboration networks, where they facilitate the capture of groupwise polyadic patterns, extending beyond traditional pairwise dyadic interactions. The use of hypergraph generators, or generative models, is a crucial method for promoting and validating our understanding of these structures. If such generators accurately replicate observed hypergraph patterns, it reinforces the validity of our interpretations. In this context, we introduce a novel hypergraph generative paradigm, **HyperPLR**, encompassing three phases: Projection, Learning, and Reconstruction. Initially, the hypergraph is projected onto a weighted graph. Subsequently, the model learns this graph's structure within a latent space, while simultaneously computing a distribution between the hyperedge and the projected graph. Finally, leveraging the learned model and distribution, HyperPLR generates new weighted graphs and samples cliques from them. These cliques are then used to reconstruct new hypergraphs by solving a specific clique cover problem.
We have evaluated HyperPLR on existing real-world hypergraph datasets, which consistently demonstrate superior performance and validate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
***Overview.*** The authors study the hypergraph generation task. 

***Method.*** Specifically, they propose a learning-based hypergraph generative method, named HyperPLR.
HyperPLR learns the (potential) weight of each edge with GNN from the unweighted graph.
Therefore, (1) it generates an unweighted graph with CELL, and (2) it assigns edge weights by using the trained GNN.
Lastly, the weighted graph is transformed into a hypergraph.

***Experiment.*** The authors demonstrated the effectiveness of HyperPLR in several benchmark hypergraph datasets, compared to hypergraph generative baseline methods.

### Strengths
***S1.*** This is the first learning-based hypergraph generative method. 

***S2.*** Overall method design is reasonable.

***S3.*** Various baseline methods are considered.

### Weaknesses
***W1 (Performance).*** Despite the large training cost, performance gain compared to the rule-based methods seems a bit marginal. Specifically, HyperLAP and THERA often outperform the proposed method.

***W2 (Scalability).*** The authors acknowledge that the proposed method lacks scalability, which is a significant limitation given that many real-world hypergraphs are typically large. Despite its slower training and generation, is the proposed method ***capable of*** generating larger hypergraphs (e.g., those with more than 10,000 nodes)? 

***W3 (Independent edge generation).*** Many real-world hypergraphs include timestamps, as interactions occur over time. Furthermore, interactions may influence one another, suggesting that hyperedges may not be independent. However, to my understanding, the proposed method generates pairwise edges without accounting for temporal aspects and dependencies.

Please clarify if my understanding or claims are wrong.

### Questions
See the mentioned weaknesses.

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
4

### Summary
This paper introduces HyperPLR, a hypergraph generative model employing a three-stage pipeline: projection, learning, and reconstruction. HyperPLR first projects the input hypergraph to a weighted projected graph and then utilizes this structure to learn graph structures, including both adjacency and edge weights. Then it reconstructs new hypergraphs by addressing a clique cover problem on the generated weighted projected graph. The authors evaluated HyperPLR by comparing it against SOTA hypergraph generative methods.

### Strengths
- HyperPLR introduces a new approach to hypergraph generation, based on three-stage framework. 
- The authors thoroughly discuss the limitations of the method. 
- Code and datasets are available.

### Weaknesses
**Clarity and presentation issues**
- Important definitions, such as "node degree in a weighted graph," are missing. This makes terms like "the degree of each node corresponds to its total frequency of appearance in hyperedges" unclear. 
- Inconsistent notations (e.g., Graph A in Figure 2 and Graph G in the text) are used.
- "G coefficient/modularity" and "B modularity" in Table 1 lack proper definitions.
- It is unclear in line 202, why weighted projection offers a more "compact representation" compared to the bipartite graph representation of hypergraphs.
- "Strong global structures" in line 214 require better contextual discussions. 
- A typo in line 138: $(v_i, v_j)\in e_i$.

**Method descriptions**
- While the authors suggest the "novelty" of the weighted projection of hypergraphs, this approach is well-studied and employed in prior hypergraph studies, even in early work like "Learning with Hypergraphs: Clustering, Classification, and Embedding" (2006).
- More details are needed about the edge weight prediction GCN module. For example, how is the loss function designed? - Does it require any negative samples? How is the bias in the edge weight distribution addressed?
- What are the bottlenecks in HyperPLR? What is its space/time complexity?

**Evaluation**
- The model's performance is not impressive enough; for example, it is outperformed by HyperLap in many cases. While the authors argue HyperPLR does not require structural parameters directly from the input, non-trivial features such as number of nodes, edge weights, and maximum hyperedges are given.
- More advanced evaluation metrics, such as those used in baseline papers, could have been considered.
- Table 2 lacks a comparative baseline.
- The effectiveness of key components in HyperPLR remains unclear. For example, how does HyperPLR perform when excluding CELL (i.e., using the input graph G directly)? Is the proposed GWC module effective?


**In summary**, the paper has room for improving the clarity of definitions and consistency of notation, highlighting the novelty, providing a more thorough description of the method, and strengthening the experiments with more comprehensive evaluation metrics and ablation studies.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
Hypergraphs are essential in modeling high-order relationships. Hypergraph generative models are crucial for promoting and validating the understanding of these structures. In this paper, the authors introduce a novel hypergraph generative approach, HyperPLR, encompassing three phases: Projection, Learning, and Reconstruction. They have evaluated HyperPLR on existing real-world hypergraph datasets, demonstrating superior performance and validating the effectiveness of HyperPLR.

### Strengths
1. Hypergraph generation is important for understanding the nature of high-order relationships.
2. The proposed method is based on a simple, intuitive idea, yet shows good empirical performance on various datasets.
3. Compared to deep learning frameworks, the proposed method is easier to interpret and trust.

### Weaknesses
1. Technical novelty is limited. (a) The idea to create a hypergraph structure from an ordinary (clique-expanded) graph has been studied already in (Bresler et al., 2024). (b) Training a GNN-based weight predictor for weighted graph structures seems new, but the technical contribution is limited, as the authors use typical GCN with node2vec-based features. (c) The proposed algorithm for weight cover is also a simple greedy algorithm which has no theoretical guarantee on its performance.
2. Writing quality should be improved in general. (a) Section 4 includes things that are part of HyperPLR, e.g., lines 189 - 192, 233 - 243, etc. It would be easier to read the paper if the authors separate them from the “proposed” ones. (b) Some knowledge about MCEC should be given so that readers can understand the propositions in a better context.
3. Theoretical claims do not support the proposed method well. Lemma 2 is straightforward, and it does not always support that WCEC is better than MCEC, considering that acquiring good edge weights is difficult. Even with the GNN predictor, we cannot guarantee the quality of this solution.

### Questions
1. Refer to the weaknesses above.
2. Can we replace CELL with any recent graph generator models, like those introduced in [1]?

- [1] Liu et al. “Generative Diffusion Models on Graphs: Methods and Applications.” IJCAI 2023

### Soundness
2

### Presentation
2

### Contribution
3
