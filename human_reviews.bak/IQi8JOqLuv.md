# On Designing General and Expressive Quantum Graph Neural Networks with Applications to MILP Instance Representation

- Decision: Accept (Poster)
- Scores: 5, 8, 6

## Abstract
Graph-structured data is ubiquitous, and graph learning models have recently been extended to address complex problems like mixed-integer linear programming (MILP). However, studies have shown that the vanilla message-passing based graph neural networks (GNNs) suffer inherent limitations in learning MILP instance representation, i.e., GNNs may map two different MILP instance graphs to the same representation. In this paper, we introduce an expressive quantum graph learning approach, leveraging quantum circuits to recognize patterns that are difficult for classical methods to learn. Specifically, the proposed General Quantum Graph Learning Architecture (GQGLA) is composed of a node feature layer, a graph message interaction layer, and an optional auxiliary layer. Its generality is reflected in effectively encoding features of nodes and edges while ensuring node permutation equivariance and flexibly creating different circuit structures for various expressive requirements and downstream tasks. GQGLA is well suited for learning complex graph tasks like MILP representation. Experimental results highlight the effectiveness of GQGLA in capturing and learning representations for MILPs. In comparison to traditional GNNs, GQGLA exhibits superior discriminative capabilities and demonstrates enhanced generalization across various problem instances, making it a promising solution for complex graph tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors proposed a quantum machine learning method to solve mixed integer programming via graph learning.

### Strengths
The proposed method is able to encode MILP graphs that are known to be intractable by message passing GNNs. The authors show that this method is able to surpass GNN baselines, especially on GNN-intractable problems.

### Weaknesses
The examples of GNN-intractable MILP problems are specific to message passing GNNs. Various works on higher-order GNNs have been proposed to overcome this issue in classical GNN architectures. Indeed, the authors provided experimental comparison to graph transformers; but basing the main motivation on a weak GNN lacks persuasiveness.

### Questions
Is the proposed method still superior comparing to higher order GNNs? Examples include subgraph GNNs that can distinguish the graphs in figure 1.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a versatile quantum graph learning architecture (VQGLA) method for solving GNN-intractable mixed-integer linear programming (MILP). The VQGLA encodes features of nodes and edges in quantum circuits, which maintain node permutation equivariance. The new method combine the power of quantum computing and GNN, exhibiting superior discriminative power over classical GNNs.

### Strengths
The authors present a well established method for MILPs. By introducing quantum computing, this new method is able to solve GNN-intractable MILPs. Numerical results can fully support this conclusion. I believe that this work exhibits an interesting application of quantum computing for solving real-world optimization problems.

### Weaknesses
The quantum circuits that are used to encode the constraint node are not straightforward. Could the authors explain how to map mathematical operators onto quantum circuits?
This work exhibits excellent performance of VQGLA in solving some MILPs. It is interesting to discuss the scalability and quantum circuit complexity of this method for general graph structures.
Since quantum neural network is an interdiscipline, the authors would better clarify how to turn a real-world problem into a quantum computing problem.

### Questions
The authors show a good parameter transferability. I wonder if these graphs have the same structures? 
The authors compare the VQGLA and the HEA in their numerical results. Did they encode the graph information in the HEA?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new quantum-driven graph learning model called VQGLA, which aims to overcome the limitations of classical Graph Neural Networks (GNNs) in representing mixed-integer linear programming (MILP) problems. VQGLA leverages quantum computing to provide superior discriminative capabilities for complex optimization tasks like MILP representation. The proposed model introduces a node feature layer, a graph message interaction layer, and an auxiliary layer, ensuring permutation equivariance and flexible circuit configurations. Experimental results demonstrate VQGLA's effectiveness compared to classical GNNs on both GNN-tractable and GNN-intractable datasets.

### Strengths
1. The paper clearly articulates the limitations of classical GNNs in representing MILP problems and makes a strong case for the introduction of a quantum-driven solution. The use of quantum machine learning (QML) is well-motivated, especially in scenarios where classical methods are fundamentally limited, such as GNN-intractable graph representations.

2. The design of VQGLA, including the node feature layer, graph message interaction layer, and auxiliary layer, is well-thought-out and comprehensive. The incorporation of quantum entanglement to enhance discriminative power is innovative, and the theoretical proof of permutation equivariance strengthens the rigor of the work.

3. The experimental results are compelling. VQGLA's superior performance over classical GNNs is demonstrated through multiple benchmarks, including both GNN-tractable and GNN-intractable MILP datasets. The comparison to other quantum and classical methods is thorough, and the reported results support the claims made in the paper.

4. The paper demonstrates the versatility of VQGLA by applying it to various graph tasks, including graph classification, graph regression, and node property prediction. The adaptability of the model to different levels of graph tasks, as well as its ability to handle edge features, is a significant advantage.

### Weaknesses
1. While the paper emphasizes the benefits of using quantum methods, it lacks a detailed discussion on the computational complexity and scalability of VQGLA. The use of fully quantum circuits for large graph instances might introduce significant resource requirements, and this limitation is not addressed sufficiently. It would be helpful if the authors provided more insights into the scalability of VQGLA with respect to graph size and the number of quantum gates required.

2. The explanation of the quantum operations, especially in the auxiliary and graph message interaction layers, could be clearer. The use of quantum gates and entanglement is discussed at a high level, but more detailed explanations and diagrams would make it easier for readers without extensive quantum computing backgrounds to understand the exact role of each component in the quantum circuit.

3. Although VQGLA is compared to classical GNNs and a few quantum models, it would be beneficial to include more comparisons with state-of-the-art classical GNN architectures that have shown effectiveness in handling complex graph tasks, such as Graph Transformers or other hybrid models. This would provide a more comprehensive understanding of how VQGLA stands relative to the best classical methods.

4. The implementation of VQGLA on near-term quantum devices is briefly mentioned, but the impact of quantum noise on the model's performance is not explored in depth. Quantum hardware is prone to noise, and the absence of a discussion on error mitigation strategies or the robustness of VQGLA under noisy conditions weakens the paper's practical relevance.

### Questions
1. Can the authors provide more insights into the scalability of VQGLA, especially concerning the number of quantum gates required for larger graph instances?
2. How does the presence of quantum noise impact the performance of VQGLA? Are there any error mitigation strategies that could be employed?
3. Could the authors include a comparison with more advanced classical GNN models, such as Graph Transformers, to highlight the specific advantages of the quantum-driven approach?
4. Is there any practical guidance on implementing VQGLA on near-term quantum hardware? A discussion on the hardware requirements would be very helpful for future work.

### Soundness
3

### Presentation
3

### Contribution
3
