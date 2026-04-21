# Semi-HyperGraph Benchmark: Enhancing Flexibility of Hypergraph Learning with Datasets and Benchmarks

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
Graphs are widely used to encapsulate a variety of data formats, but real-world networks often involve complex node relations beyond only being pairwise. While hypergraphs have been developed and employed to account for the complex node relations, they reduce the flexibility of machine learning systems by totally disregarding simple edges, which to some extent leads to a drop in performance. Additionally, Graph Neural Networks (GNNs) research are normally separated into simple graphs and hypergraphs, and these two classes of methods tend not to interchange. Therefore, there is a need for a more flexible benchmark that allows GNNs to employ both simple edge and hyperedge information. In this paper, we present the *Semi-HyperGraph Benchmark (SHGB)*, a collection of comprehensive datasets combining hypergraphs and simple edges, with an accessible evaluation framework to fully understand the performance of GNNs on complex graphs. SHGB contains 23 real-world hypergraph datasets with simple edges included, across various domains such as biology, social media, and e-commerce. Furthermore, we provide an extensible evaluation framework and a supporting codebase to facilitate the training and evaluation of GNNs on SHGB. Our empirical study of existing GNNs on SHGB reveals various research opportunities and gaps, including (1) evaluating the actual performance improvement of hypergraph GNNs over simple graph GNNs; (2) comparing the impact of different sampling strategies on hypergraph learning methods; and (3) exploring ways to integrate simple edge and hyperedge information. We make our source code and full datasets publicly available at https://anonymous-url/.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the Semi-HyperGraph Benchmark (SHGB) to unify graph and hypergraph benchmarks, attempting to bridge an existing gap where these benchmarks are considered separately in the literature.

SHGB integrates real-world datasets featuring both pairwise edges and hyperedges, enabling researchers to thoroughly assess Graph Neural Networks (GNNs) using a combination of edges and hyperedges.

Experiments reveal that 
* GNNs on hypergraphs may not consistently outperform simple graph GNNs on large networks, 
* sampling strategies enhance GNN performance on hypergraphs, and 
* combining edge and hyperedge information improves predictions on complex graphs.

### Strengths
### Clarity
1. The paper provides a clear demonstration of edge and, more crucially, hyperedge construction through clear illustrations in Figures 1 and 2.
2. The SHGB framework is effectively outlined in Figure 3, ensuring a clear and concise understanding of its structure and components.
3. The paper is well-structured into clear sections, guiding readers logically from the introduction to the conclusion, ensuring a thorough understanding of the research content.

### Weaknesses
### Originality
1. The hyperedges in the datasets are *not natural* but derived from social networks' simple edges, a specified base pair distance in gene data, and e-commerce product image embeddings, which can be modelled as node features. 
2. The hyperedges introduced can be obtained solely from *appropriately chosen graph data and node features*, limiting the originality of the curated hyperedges.
3. The HypergraphSAINT sampler is a straightforward application of GraphSAINT [Zeng et al., 2020] to hyperedges.

$~$
### Significance
4. Appropriately chosen GNNs (e.g., subgraph GNNs, MixHop, JKNets) on meticulously chosen graph datasets would be able to model the information given by the curated hyperedges, limiting the benchmark's potential impact.
5. The significance of the work can be improved by the inclusion of recent hypergraph neural networks [e.g., Chien et al., 2022, Wang et al., 2023] that generate hyperedge embeddings to fully exploit hyperedges.

$~$
### Quality
6. Contrary to the stated claim of extending hypergraphs with simple edges, e.g., see contribution 1 on page 2, the actual contribution involves integrating *curated hyperedges* with *naturally occurring edges* such as mutual followers in social networks and regulatory effects between genes in biological networks.
7. Four different GNNs on pairwise edges are tested in the experiments, whereas the selection of GNNs on hyperedges is restricted to those introduced in a single paper [Bai et al., 2021], further emphasising simple edges over hyperedges.

\
References:
* [Wang et al., 2023]: Equivariant Hypergraph Diffusion Neural Operators, ICLR'23
* [Chien et al., 2022]: You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks, ICLR'22
* [Bai et al., 2021]: Hypergraph convolution and hypergraph attention, Pattern Recognition'21
* [Zeng et al., 2020]: GraphSAINT: Graph Sampling Based Inductive Learning Method, ICLR'20

### Questions
1. From a pure dataset perspective, what unique information do the hyperedges contribute that is not already represented by carefully/appropriately/meticulously chosen graph data and node features?
2. From a pure dataset perspective, are there unique insights into the specific reasons for choosing social networks' friend circles from pairwise edges, gene data within a user-specified base pair distance, and e-commerce product image embeddings as sources for deriving hyperedges?
3.  Have there been examples or scenarios where such derived hyperedges have demonstrated real-world applicability or have been utilised successfully in practical applications?
4. Considering the potential benefits of incorporating recent hypergraph neural networks [e.g., Chien et al., 2022, Wang et al., 2023], are there any challenges or limitations associated with their implementation, and if so, how could these challenges be addressed to ensure seamless integration into the proposed SHGB framework?
5. In what ways do the hyperedges complement or enhance the information captured by naturally occurring edges, such as mutual followers in social networks and regulatory effects between genes in biological networks? 
6. In addition to Table 3, are there specific examples or case studies where this integration has led to unique insights or improved predictions compared to utilising simple edges alone?
7. Why was the selection of GNNs for hyperedges limited to those introduced in a single paper [Bai et al., 2021]? 
8. Were there specific criteria or reasons for this restriction, and how does this choice impact the overall diversity and representation of GNN models applied to hyperedges?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes datasets that contains hyperedges and simple edges for GNNs. The paper reads well but motivation for proposing the 
datasets combining hypergraphs and simple edges is unclear.  Datasets are not from the real-world. Insights for future research are not unclear.  Authors should fix the download links for codes and datasets.

### Strengths
S1. Datasets cover various domains.

S2. The paper reads well.

### Weaknesses
W1. Motivation for proposing semi-hypergraphs is unclear.  Real-world applications for semi-hypergraphs are not discussed. Or importance of semi-hypergraphs for future research is not discussed. See Q1.

W2. Datasets are not effective for GNNs. See Q2.

W3. Datasets are not from the real-world but generated by rules and algorithms. See Q3. 

W4. Insights for future research is not clear. See Q4.

### Questions
Q1: Can authors show real-world applications for semi-hypergraphs? Or can authors show evidences for that proposing semi-hypergraphs is important for future research?

Q2: Figure 4 (a) shows that for most of the datasets, Hyper GNNs and simple GNNs have almost the same accuracy. Can authors give comments on this?

Q3: Can authors explain why not search for real-word datasets but generate by rules or algorithms?

Q4: Can authors show the insights for future research?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work aims to provide a more complete evaluation of hypergraph deep learning models. The approach includes a) constructing new hypergraph datasets that consist of both simple edges and clique-based hyperedges, b) comparing hypergraph neural networks and graph neural networks, and c) properly combining the two types of neural networks. The work also studies hypergraph sampling approaches for hypergraph NNs.

### Strengths
+ It is a valid idea to build hypergraph datasets that hold both simple edges and hyperedges. 

+ It is novel to investigate hypergraph sampling approaches for hypergraph neural networks.

+ It is reasonable to combine GNNs and hypergraph NNs to achieve overall best performance.

### Weaknesses
- The work misses some solid foundations. First, the work seems unaware of how practical hypergraphs are typically models. The work only discusses those datasets used to evaluate hypergraph NNs but really misses the discussion on the entire area that studies hypergraph modeling, higher-order graphs for data analysis, e.g. [1][2].

- Because the work is unaware of that area. The claim that "using hyperedges may overlook simple pairwise node relations and thus make hypergraphs substantially lose useful graph information" is overclaimed. Properly modeling hyperedges as sets and using complex set functions essentially cover simple pairwise relations as a special case [3]. The set representation is much more powerful in principle. The not-idea performance of hypergraph NNs as opposed to GNNs is just due to the non-idea way to construct hypergraphs and the suboptimality of hypergraph NNs. 

- Some recent more principled hypergraph NNs are missing to discuss, e.g. [4][5]. This is a weak point for a benchmark paper. 

[1] Higher-order organization of complex networks, Science 2016

[2] Networks beyond pairwise interactions: Structure and dynamics, Physics Report, 2020

[3] Submodular hypergraphs: p-laplacians, cheeger inequalities and spectral clustering, ICML 2018

[4] Unignn: a unified framework for graph and hypergraph neural networks, IJCAI 2021

[5] Equivariant Hypergraph Diffusion Neural Operators, ICLR 2023

### Questions
no specific questions. 

The authors are suggested to extensively discuss relevant works to address the listed weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
