# Interpretable Hypergraph Neural Additive Networks

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
Hypergraph neural networks have emerged as a powerful framework for learning from higher-order structured data, where relationships among entities extend beyond pairwise connections. However, most current hypergraph neural networks are black-boxes that rely on post-hoc explanation methods to provide model insights. Such post-hoc explanations can be unreliable in high-stakes scenarios and knowledge discovery tasks. We introduce an inherently interpretable hypergraph neural additive network (HGNAN), an
extension of generalized additive models that facilitates interpretability in complex, higher-order relational learning settings. HGNAN provides clear visualizations of both global and local behaviors at the node and hyperedge levels while preserving the expressive power of hypergraphs. We evaluate HGNAN on node classification and hyperedge prediction across various datasets, achieving competitive performance compared to state-of-the-art methods. HGNAN also significantly outperforms existing approaches in recovering missing reactions in metabolic networks, while offering interpretablebiological insights into metabolic processes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces HGNAN, a hypergraph neural additive network designed to achieve inherent interpretability in hypergraph learning. Unlike conventional black-box hypergraph neural networks that rely on post-hoc explanations, HGNAN extends generalized additive models to provide transparent and visualizable decision mechanisms while maintaining strong expressive capability. Experimental results across multiple node classification and hyperedge prediction tasks show that HGNAN achieves competitive or superior performance compared to state-of-the-art methods, and offers interpretable insights in scientific domains such as bioinformatics and social network analysis.

### Strengths
S1: Extending additive networks to hypergraph learning is an interesting and meaningful direction, and the proposed model demonstrates a certain degree of interpretability.

S2: The experimental results are comprehensive and show a good balance between predictive performance and interpretability.

### Weaknesses
W1: The overall writing quality needs improvement, as several parts of the method and experiments are not clearly described. For example, the methodology section only explains how node or hyperedge representations are obtained, but the complete training pipeline and model architecture are not well presented. In addition, some key parameters, such as $W$, are not properly introduced—their dimensions and correspondence with samples remain unclear.

W2: The paper primarily extends the GNAN framework to hypergraphs, with the main difference appearing to be the definition of a distance function on hypergraphs. However, the overall idea is largely similar to GNAN, suggesting that the methodological novelty may be limited.

W3: Although the authors mention that “such extreme high-dimensional datasets are relatively uncommon in many real-world applications,” in practice, many graph datasets feature high-dimensional node attributes, especially those derived from large language model embeddings. This could limit the applicability and scalability of the proposed method.

W4: The paper does not provide any theoretical analysis of time complexity or experimental evaluation of computational cost. It is recommended that the authors include such analyses to strengthen the work.

### Questions
Q1: I would like to know the specific values of $s_{max}$ used for each dataset during the experiments, as this information does not seem to be provided in the paper.

Q2: I am curious about how the results in Figure 2 were computed. In particular, is the weight vector $W$ introduced in Section 3.2 shared across the entire dataset, or is it computed separately for each sample?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **HGNAN (Hypergraph Neural Additive Network)** — an *inherently interpretable* hypergraph neural network framework that integrates the additive modeling principle of **Neural Additive Models (NAMs)** with the structural learning capability of **Hypergraph Neural Networks (HGNNs)**. The key innovation lies in introducing **feature-wise additive shape functions** and **distance-based weighting mechanisms** that together preserve both *interpretability* and *higher-order relational reasoning*. Extensive experiments across **five node classification datasets** and **four hyperedge prediction datasets** demonstrate that HGNAN achieves *comparable or superior predictive accuracy* compared to black-box baselines such as HGNN, AllSetTransformer, and CHESHIRE, while providing interpretable insights at both the **feature** and **structure** levels.

### Strengths
* The work extends **Neural Additive Models** and **Graph Neural Additive Networks** to **hypergraph-structured data**, filling a notable gap in interpretable hypergraph learning.
* The concept of combining **distance-aware structural weighting** with **feature-wise additive modeling** is novel, enabling a direct interpretability mechanism rather than relying on post-hoc explainers.

* Experimental comparisons are comprehensive, covering diverse datasets that span homophilic and heterophilic regimes, ensuring that claims are empirically well supported.
* The interpretability visualizations (e.g., distance-function curves and feature contribution heatmaps) demonstrate genuine transparency into model reasoning.

* The paper is well-organized and readable, with clear notations and step-by-step explanations of key equations.

### Weaknesses
* Since HGNAN assigns a separate neural subnetwork to each feature (as in NAMs), scalability may become an issue for datasets with very high-dimensional feature spaces. Although the paper acknowledges this limitation, empirical evidence on runtime or parameter scaling is limited.

* A baseline for hypergraph node classification is missed[1].

[1] From Hypergraph Energy Functions to Hypergraph Neural Networks

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an hypergraph neural network with an interpretable
mechanism (HGNAN), based on the additive architecture.

The efficacy of the new HGNAN is tested on node prediction and
hyper-edge prediction with results in line with the state of the art and
with the advantage of the interpretability mechanism that is clearly
explained in the experiments section.

The paper is well-written and accurate

### Strengths
The paper presents an innovative algorithm on hypergraphs and provides examples on node classification an hyperedge prediction

### Weaknesses
As stated by the authors a weakness can be the computational cost since a neural network is needed for each feature.

### Questions
Do the hypergraph-based techniques require a larger amount of input data compared to equivalent graphs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces HGNAN, an interpretable hypergraph model that marries NAM-style per-feature shape functions with neighborhood-weighted aggregation for node tasks and distance-weighted aggregation over s-intersection graphs for hyperedge tasks. HGNAN can produce feature-level importance by decomposing the neural network to be a linear combination of NNs; each operates on a different feature independently.

### Strengths
- The problem itself is interesting, and the paper clearly motivates the need for inherently interpretable HGNNs.
- Some results on hyperedge prediction seem good.

### Weaknesses
- Novelty is limited. The paper only decomposes the HGNN into a linear combination of NNs. This is not new and can only produce feature-level explanations. Meanwhile, node/subgraph level explanations are still unknown. 
- The expressive power of the HGNAN due to the linear decomposition is not discussed. What if we have many features? The number of parameters would grow linearly? What if those features are redundant or not linearly correlated? What is the expressiveness of HGNAN compared to other hypergraph neural networks?
- Evaluations are not on standard datasets.
- The paper does not use the standard font of ICLR.

### Questions
- The related work section should rather focus on interpretable by design GNN and post-hoc explainability for HGNN to better position itself in the literature.
- it may be better to start with a clear definition of the targeted explanation. This will better motivate architectural choices.
- The datasets used seem not to be standard. Can you reproduce the results used in other papers, such as cora, citeseer, pubmed, yelp, DBLP-CA?
- Feature explanations come from the linear combination of features are straightforward. Can you derive node explanation using HGNAN? For example, finding a set of nodes or hyperedges that are most influential to the prediction?

### Soundness
1

### Presentation
1

### Contribution
1
