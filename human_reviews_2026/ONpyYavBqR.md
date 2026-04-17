# A Graph Meta-Network for Learning on Kolmogorov–Arnold Networks

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Weight-space models learn directly from the parameters of neural networks, enabling tasks such as predicting their accuracy on new datasets.  Naive methods -- like applying MLPs to flattened parameters -- perform poorly, making the design of better weight-space architectures a central challenge. While prior work leveraged permutation symmetries in standard networks to guide such designs, no analogous analysis or tailored architecture yet exists for Kolmogorov–Arnold Networks (KANs). In this work, we show that KANs share the same permutation symmetries as MLPs, and propose the KAN-graph, a graph representation of their computation. 
Building on this, we develop WS-KAN, the first weight-space architecture that learns on KANs, which naturally accounts for their symmetry. We analyze WS-KAN’s expressive power, showing it can replicate an input KAN’s forward pass - a standard approach for assessing expressiveness in weight-space architectures. We construct a comprehensive ``zoo'' of trained KANs spanning diverse tasks, which we use as benchmarks to empirically evaluate WS-KAN. Across all tasks, WS-KAN consistently outperforms structure-agnostic baselines, often by a substantial margin.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a graph-based framework for designing meta-networks that operate on the pretrained weights of Kolmogorov–Arnold Networks (KANs).

### Strengths
The paper is well-written and self-contained. Although I am not very familiar with the recent developments on KANs, I was able to understand the architecture clearly. The authors provide a good demonstration of the symmetry properties of the KAN parameter space, along with a clear explanation of how a KAN can be interpreted as a graph. Their approach closely follows the standard methodology used in the literature for constructing metanetworks for traditional MLPs and CNNs.

### Weaknesses
I did not identify any clear weaknesses. Only a few minor typos were found, such as “silo function” in Lemma 4.1 or “weight-space model” in line 10.

### Questions
I have a question regarding the permutation symmetry of KANs presented in Section 3.1. While it is clear that the permutation symmetry preserves the network’s functional behavior, I am wondering whether there exist other types of symmetry. For example, in MLPs and CNNs with ReLU activations, there is a well-known scaling symmetry, which has also been leveraged in the design of graph-based metanetworks [1]. A similar analysis for KANs would help clarify whether additional symmetries exist beyond permutation, and would further justify that modeling permutation symmetry alone is sufficient for constructing the proposed graph-based network.


[1] Ioannis Kalogeropoulos, Giorgos Bouritsas, and Yannis Panagakis. Scale equivariant graph metanetworks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies “learning in weight space” for Kolmogorov–Arnold Networks (KANs). The authors (i) prove KANs admit the same hidden-neuron permutation symmetries as MLPs, (ii) encode a KAN as a directed “KAN-graph” whose edge features are the parameters of each learned univariate function, and (iii) process that graph with a GNN they call WS-KAN. They further show WS-KAN can simulate the forward pass of an input KAN (a standard expressivity criterion in weight-space works), then build several “model zoos” of trained KANs and evaluate on INR classification, accuracy prediction, and pruning-mask prediction—reporting consistent gains over baselines and large speedups vs an oracle pruning routine that requires data passes.

### Strengths
1. Concrete construction of edge features for KANs’ 1D functions; positional encodings are explained and motivated.

2. A standard expressivity result: WS-KAN can approximate a KAN’s forward pass, with proof sketch and full appendix.

### Weaknesses
1. **Lack of empirical symmetry validation:** The work does not experimentally verify the equivariance or invariance properties that the theory implies.

2. **Limited range of equivariant tasks:** Prior works such as Editing INRs [1] and Learning to Optimize [2] evaluate equivariant models on richer, more practical tasks. Incorporating similar benchmarks would strengthen the empirical relevance.

[1] Zhou et al., Permutation Equivariant Neural Functionals. NeurIPS 2023.

[2] Kofinas et al., Graph Neural Networks for Learning Equivariant Representations of Neural Networks. ICLR 2024.

3. **Unaddressed scalability issues:** The paper omits discussion of computational complexity and runtime behavior. Since the KAN-graph grows quadratically with network connectivity, an analysis of how WS-KAN scales with KAN depth and width is needed.


4. **Incomplete design analysis:** Key architectural choices—such as the use of bidirectional message passing and specific positional encodings-are insufficiently justified. Ablation studies would clarify their impact on performance

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents WS-KAN, a meta-network tailored for working with Kolmogorov-Arnold Networks (KANs). The authors introduce a graph representation (KAN-graph) for KANs that accounts for their permutation symmetries, and they employ a GNN structure built on this encoding. They conduct a theoretical examination of the model's expressive capabilities and perform empirical evaluations using a freshly assembled collection of pre-trained KANs across multiple tasks, reliably showing better results than competing methods.

### Strengths
- This paper introduce the first metanetwork for KANs.
- The motivation and design process are presented clearly.
- The KAN model zoo (covering INR classification, accuracy prediction, pruning) provides a valuable benchmark for future WS models on KANs.

### Weaknesses
- The discussion in Section 3.1 on permutation symmetries in the KAN weight space primarily pertains to symmetries in the space of neuron functions and their connections rather than the underlying weight parameters that define those functions. This echoes themes from prior work on weight space symmetries in MLPs [1] and only use permutation symmetries. Could other symmetries extend to KANs?
- Clarification is needed on the scope of each KAN-graph: How is node features initialized, is it associated to input data? If yes, this suggests the WS-KAN operates beyond pure weight space, potentially incorporating input-specific adaptations, which somewhat contradict to the claim being a weight-space-only method.
- In Lemma 4.2, please clearly define what is meant by "the nodes in the first layer are enhanced with the input x", does this imply concatenation, embedding, or another operation, and how does it affect the overall model and expressivity?
- The paper focuses on B-spline parameterization for KAN neurons, but it would be beneficial to comment on alternative parameterizations and whether the proposed meta-network generalizes to them.
- The experiments lack reporting on time and memory complexity. Additionally, how well does the approach scale to generating larger models (eg, deeper and wider KANs)
- As the WS-KAN is GNN-based, I wonder if the method can be tested for generalization, eg, by training the meta-network on small KANs and evaluating its ability to generate parameters for larger, unseen KAN architectures?
- Experiments illustrating the importance of positional encoding (if used in the graph representation) would be valuable, perhaps through an ablation study.

Reference:

[1] Zhao et al. Symmetry in Neural Network Parameter Spaces. arxiv:2506.13018.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces applying weight-space symmetric architecture for Kolmogorov–Arnold Networks (KANs), called WS-KAN, which leverages a graph representation of the network’s computation to respect inherent symmetries in the architecture. The authors highlight the equivalence of permutation symmetries in KANs to those in standard neural networks, especially MLPs, and propose the KAN-graph—a directed graph where nodes represent neurons and edges carry univariate functions. WS-KAN uses existing work of graph metanetwork to process these graphs and simulate the forward pass of the original KAN. Empirical validation is conducted using a model zoo of pre-trained KANs across various tasks. The proposed model consistently outperforms structure-agnostic baselines and provides superior performance in INR classification, accuracy prediction, and pruning mask prediction.

### Strengths
The method is the first to specifically address the design of weight-space models for KANs.

The authors construct a comprehensive "model zoo" of KANs and benchmark WS-KAN across various tasks, demonstrating superior performance over baselines. The extensive experiments on multiple datasets such as MNIST, CIFAR10, and F-MNIST validate the proposed method's robustness.

### Weaknesses
The paper basically uses the existing solution of graph metanetwork proposed for MLPs to apply on KAN weight space, which is intuitive, however contribution is limited. Also experiments e.g. INR classification, accuracy prediction, all follow the standard from weight space symmetry works on MLP/CNN. From the technical solution and application tasks I do not see anything new and specific to KANs.

### Questions
- What does proposition 3.2 imply and how it is related to the main contributions / arguments in the paper? Here "the nodes in the first layer are enhanced with the input x", do you really do that in the implementation?

- Have you tried other weight space symmetry works such as DWSNet and NFN, and do they apply to KANs as well?

### Soundness
3

### Presentation
3

### Contribution
2
