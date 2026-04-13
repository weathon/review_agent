## Human Reviewer 1

### Summary
The paper introduces STAGE, a novel approach designed to enable zero-shot generalization for Graph Neural Networks (GNNs) across graphs with varying node attribute domains. STAGE aims to learn representation of statistical dependencies between attributes rather than their absolute values. This allows the model to transfer knowledge to unseen domains by leveraging analogous dependencies. Through experiments on multiple datasets, the paper demonstrates STAGE's superior performance in link prediction and node classification tasks, especially in terms of zero-shot cross-domain generalization.

### Strengths
- Generalization from a node attributes view and the two stages for processing graph representation are interesting.
- The paper provides a theoretical analysis, linking STAGE with maximal invariants and statistical dependency measures, which provides theoretical support for the model's generalization capabilities.
- The paper shows STAGE's robustness when facing different attribute domains, which is a very important characteristic in the varied real-world data

### Weaknesses
- Although the paper presents some quantitative results and shows good performance in different link prediction and node classification domains, it lacks some qualitative analysis. For example, it could demonstrate how the model learns that "income level is positively correlated to phone price" from the training set and then discovers that "height is positively correlated with clothing size" in a new domain, thus generalizing to the new domain.
- STAGE is capable of capturing and leveraging feature dependencies in graph data, rather than relying on specific attribute values. The article can illustrate which feature dependencies are effective on the test set after pre-training the model.

### Questions
How well does this model handle larger graph datasets.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper presents STAGE, a method that enables zero-shot generalization of graph neural networks (GNNs) across graphs with different attribute domains. STAGE constructs STAGE-edge-graphs to capture statistical dependencies between attributes instead of absolute values, facilitating transferability to unseen domains. The method shows substantial improvement in zero-shot tasks like link prediction and node classification on graphs with entirely new feature spaces.

### Strengths
1. STAGE's strategy to use statistical dependencies rather than raw attribute values to enhance zero-shot generalization in GNNs is novel for graph machine learning.
2. The theoretical basis for STAGE, connecting maximal invariants and statistical dependencies, is well-articulated and provides a sound foundation for the empirical results.
3. STAGE is shown to be adaptable across domains of varied attribute types and dimensions, a crucial quality for real-world applicability.

### Weaknesses
1. STAGE's two-stage process involving STAGE-edge-graphs, conditional probability matrices, and subsequent embeddings may be challenging to implement or optimize in practice. Details on computational overhead compared to baselines would enhance clarity.
2. While STAGE is effective for pairwise dependencies, it is unclear how it handles more complex dependencies in highly interconnected graphs.
3. The success of STAGE appears dependent on the architecture and expressivity of the underlying GNNs (M1 and M2). Sensitivity analysis on different GNN backbones might clarify robustness across architectures.
4. The evaluation focuses on e-commerce and social network datasets; examining STAGE’s generalizability on domains like biomedical or geospatial networks would strengthen claims of universality.

### Questions
1. How does STAGE handle attribute domains with highly heterogeneous data types, such as unstructured or mixed media data?
2. Could the authors elaborate on the computational cost associated with STAGE compared to baselines, especially in large-scale graphs?
3. Does STAGE's reliance on GNN backbones like NBFNet affect its generalizability? Could alternative GNN architectures be equally effective?
4. Has STAGE been evaluated in terms of the interpretability of the learned dependencies? If so, what methods were used?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces STAGE, a method designed for zero-shot generalization across attributed graphs with distinct attribute domains. STAGE constructs what it calls STAGE-edge-graphs for each edge in a graph, embedding statistical dependencies between attributes at each node pair. The model achieves significant performance gains in zero-shot settings for tasks like link prediction and node classification on various datasets.

### Strengths
1. This paper achieves SOTA results by embedding statistical dependencies rather than raw features.
2. The STAGE is a domain-agnostic framework, which can generalize across disparate attribute spaces.

### Weaknesses
1. The STAGE-edge-graph is a fully connected weighted graph, so I am concerned about the complexity.
2. Edge-based embeddings may limit its ability to capture high-order interactions in graphs.
3. The motivation in the introduction is not presented well. The authors didn't analyze why their proposed method can address the limitations they mentioned before, so it's hard to understand the intrinsic research thinking.
4. The experiments are a little weak. For example, I believe the 4.2 and 4.3 belong to the same type of experiment, they didn't analyze the complexity and the ablation study, and they didn't include the limited research papers they mentioned in the introduction into baselines,  which weakens the convincing.
5. I noticed this paper was submitted to ICML workshop so there is authors' information leakage. Both papers present STAGE for zero-shot generalization of GNNs across different attribute domains, and this paper just extends some real-world testing datasets. 
6. The authors didn't release their code although this is not compulsory, which may limit their reproducibility.

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper studies how to use a pre-trained graph model in any new domain with unseen attributes, enhancing the zero-shot generalization. The authors propose a new model STAGE by learning the representations of statistical dependency between attributes, instead of the attribute values themselves. They also conduct experiments to validate the performance of STAGE across several benchmark datasets.

### Strengths
1.	The paper is well written and easy-to-follow.
2.	Extensive results validate the effectiveness of the proposed model.

### Weaknesses
1.	More discussions on the variants on LLM should be made.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4