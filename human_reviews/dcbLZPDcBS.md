# Efficient Graph Representation Learning by Non-Local Information Exchange

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5

## Abstract
Graph is an effective data structure to characterize ubiquitous connections as well as evolving behaviors that emerge from the inter-wined system. In spite of various deep learning models for graph data, the common denominator of current state-of-the-arts is finding a way to represent, or encode, graph entities (such as nodes and links) on top of the intricate wiring topology. Limited by the stereotype of node-to-node connections, learning global feature representations is often confined in a graph diffusion process where local information has been excessively aggregated as the random walk explores far-reach neighborhoods on the graph. In this regard, tremendous efforts have been made to alleviate feature over-smoothing issue such that current graph learning backbones can lend themselves in a deep network architecture. However, little attention has been paid to improving the expressive power of underlying graph topology, which is not only more relevant for the downstream applications but also more effective to mitigate the over-smoothing risk by reducing unnecessary information exchange on the graph. Inspired by the notion of non-local mean techniques in image processing area, we propose a non-local information exchange mechanism by establishing an express connection to the distant nodes, instead of propagating information along the (possibly very long) topological pathway node-after-node. Since the seek of express connections throughout the graph could be computationally expensive in real-world applications, we further present a hierarchical re-wiring framework (coined $express\ messenger$ wrapper) to progressively incorporate express links into graph learning in a local-to-global manner, which allows us to effectively capture multi-scale graph feature representations without using a very deep model, thus free of the over-smoothing challenge. We have integrated our $express\ messenger$ wrapper (as a model-agnostic plug-in) with existing graph neural networks (either using graph convolution or transformer backbones) and achieved SOTA performance on various graph learning applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces "express messenger" a model-agnostic plug-in that re-wires the learning graph such that new connections between distant nodes are formed. The aim behind this approach is to achieve better non-local information exchange. The authors claim that their module reduces/eliminates the feature over-smoothing issue. They also integrate the plug-in with existing GNNs and demostrate its effectiveness.

### Strengths
Figure 1 clearly illustrates the idea discussed in this work and aids in the visualization of the proposed methodology

The experiments that are designed to show the effectiveness of the express messenger are thorough

### Weaknesses
(Table 2) Many of the numbers reported for baselines and the proposed method are within the margin of error of each other. This detracts from the effectiveness of the proposed methodology.

### Questions
(Table 3) Going by the pattern of improvement in numbers when the ExM plug-in is included; would the authors be comfortable in agreeing to this?: If the delta of performance between GNN and GNN + ExM is plotted against h (x-axis), it would be an elbow curve with the delta approaching zero as h -> 1 and the elbow being somewhere around h = 0.2. Hope this is clear!

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this submission, the authors propose a new method for GNN learning that focuses on addressing the challenge of over-smoothing in feature representation. Incorporating a novel mechanism called Non-Local Information Exchange (NLE), enhances the ability of GNNs to combine local and global information effectively. The main contribution is the ExM wrapper, which can be integrated with various GNN models to maintain state-of-the-art performance across different graph datasets, particularly improving performance on heterophilous graphs.

### Strengths
1. The paper introduces a new idea to tackle the over-smoothing issue.

2. The experimental results seem interesting. Specifically, the ExM wrapper developed aids most baseline graph neural network (GNN) methods in retaining state-of-the-art (SOTA) performance across various graph datasets with diverse homophily ratios. It is highlighted that the C-ExMP variant of the ExM wrapper often outperforms its counterparts, securing top-3 rankings in multiple datasets​.

### Weaknesses
1. The presentation has a large space for improvement. The reviewer does not think the proof makes sense.

2. The theoretical justification of the proposed method is weak. It is unclear why the proposed wrapper can be applied to general models.

3. Some claims in this paper are too strong. For example, the paper mentioned the expressiveness of GNN has not been explored. However, there are many papers focusing this area including the papers cited in this submission.

### Questions
Overall, the reviewer thinks the submission is not ready for publication. There are presentation issues and the methodology needs a theoretical justification.

Q1. The proof of Proposition is so unclear, why the matrix operations can be linked with sets(these neighbors)?

There are grammar issues including:

1. State-of-the-arts -> state-of-the-art
2. far-reach neighborhoods -> far-reaching
3. over-smoothing issue -> issues.

### Soundness
1 poor

### Presentation
1 poor

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
This paper proposes a novel framework to enhance the expressive power of GNNs and mitigate the over-smoothing in deep GNNs. The authors introduce an innovative non-local information exchange mechanism inspired by non-local mean techniques from image processing. This mechanism directly connects distant nodes, bypassing the traditional sequential propagation of information. Two express messenger wrapper are proposed to rewire the connection. The two wrapper allows to capture global representations thus free of over-smoothing. Extensive experiments are conducted to validate the effectiveness of the proposed method.

### Strengths
1. The commitment to addressing the over-smoothing issue and improve expressiveness in Graph Neural Networks (GNNs) is highly commendable and worthy of research. 

2. Extensive experimental validation demonstrates the method's capability in enhancing performance.

### Weaknesses
1.	I disagree with the claim made in the abstract that "However, little attention has been paid to improving the expressive power of underlying graph topology." In fact, there has been a significant amount of research in recent years on the expressive power of graph neural networks, which is a topic that this paper lacks discussion on. 

2.	In particular, the approach presented in this paper bears similarities to k-hop GNN. Therefore, it is important to provide a detailed discussion and conduct experimental comparisons between the two.

a)	Nikolentzos, Giannis, George Dasoulas, and Michalis Vazirgiannis. "k-hop graph neural networks." Neural Networks 130 (2020): 195-205.

b)	Feng, Jiarui, et al. "How powerful are k-hop message passing graph neural networks." Advances in Neural Information Processing Systems 35 (2022): 4776-4790.

3.	The diagrams included in the article are difficult to comprehend.

4.	The paper lacks experimental or theoretical evidence to support the claim that extracting global structural information can effectively resolve the issue of over-smoothing.

### Questions
1.	Why we need to use proxy measurements instead of existing measurement methods？

2.	Could you please explain the design differences and suitable scenarios for the two types of messenger wrapper.?

3.	The ablation study lacks a comparison that involves aggregation of global information exclusively. The overall experiments do not directly demonstrate the impact of different methods on over-smoothing. It is recommended that this be supplemented.

4.	How does capturing global information overcome the problem of over-smoothing? Could an excessive focus on extracting global information potentially lead to even greater over-smoothing?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
