# Node2Net: Node-Specific Parameterization for Expressive Graph Representation Learning

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Graph Neural Networks (GNNs) have emerged as powerful tools for graph learning. Classical message-passing GNNs enforce permutation equivariance at the node level and permutation invariance at the graph level, but these symmetries constrain expressiveness, limiting them to the discriminative power of the 1-WL test. Recent advances such as Graph Transformers extend GNNs with global attention and positional encodings, yet still rely on shared graph-level parameters. In this work, we revisit the symmetry–expressiveness trade-off through node-specific parameterization, where each node contains a small trainable neural network-an approach we term Node2Net. Unlike existing methods that represent each node with a static embedding vector, Node2Net represents each node with a parametric function capable of modeling nonlinear feature interactions and adaptive transformations. Node2Net breaks 1-WL indistinguishability and can act as universal approximators capable of representing arbitrarily complex node-level transformations. Its computational and memory costs scale linearly with the number of nodes and remain practical on standard benchmarks. As a fundamental node representation method, Node2Net is model- and task-agnostic and does not change the transductive or inductive generalization properties of GNN backbones. Extensive experiments on multiple benchmarks demonstrate that Node2Net consistently improves over node feature learning methods, traditional message-passing GNNs, and recent Graph Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Node2Net, a GNN framework where each node is represented by its own small neural network instead of a shared embedding. This design enhances expressiveness beyond the 1-WL limit while remaining computationally efficient. Node2Net is model-agnostic and consistently outperforms traditional GNNs and Graph Transformers across benchmarks.

### Strengths
1. The intuition behind Node2Net is simple, intuitive, and easy to follow. It is model-agnostic and can be seamlessly adapted to different graph learning architectures and tasks.
2. The paper provides numerous examples demonstrating the applicability of Node2Net, including its integration with MPNNs and Graph Transformers, highlighting the elegance and versatility of the approach.
3. The experimental section is comprehensive, with detailed settings and clear explanations of results, effectively showing the advantages and practical impact of Node2Net.

### Weaknesses
1. Although the intuition behind Node2Net is simple, it feels somewhat trivial and insufficiently developed. I think the paper should carefully analyze the trade-off between performance gains and the substantial costs in terms of parameters and computational complexity. Node2Net introduces significant overhead, making it difficult to scale to large graphs—this issue must be explicitly discussed.
2. The performance improvement brought by Node2Net appears limited. As shown in Table 2, the gains over backbone models are minor, and the paper does not compare against recent state-of-the-art baselines.
3. The paper does not provide publicly available code for reproduction, which is an essential requirement for modern learning-based research.
4. The writing quality requires further improvement; see the minor comments below. Additionally, Table 3 reports loss values directly, which is unconventional and potentially confusing; it would be clearer to present convergence curves over time or epochs instead.

**Minor Comments:**  
(1) The figures in the paper are not vector graphics, which do not conform to academic publication standards.  
(2) Figure 2 is difficult to understand, and it is unclear which module the term “Node2Net” represents.  
(3) The mathematical notation in the paper is not clear. Vectors and matrices should be typeset in bold for clarity. It is recommended to follow the formatting guidelines in the official ICLR template.

### Questions
1. Please begin by responding to the Weaknesses part.
2. Please discuss the overhead introduced by Node2Net in terms of parameters and computational complexity, especially from an empirical perspective. Is it feasible to scale to very large graphs (e.g., with hundreds of millions of nodes)? It is worth noting that Node2Vec can handle such scales.

### Soundness
2

### Presentation
2

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
The paper proposes including node specific weight parameters in the GNN model. Experimental results evaluate the proposed approach with various GNN backbones.

### Strengths
* The paper is well written.
* Broad Applicability: The method is model-agnostic and can be integrated into various GNN architectures (node representation methods, MPNNs, Graph Transformers), demonstrating versatility.
* Theoretical Contributions: The paper provides theoretical analysis showing that Node2Net can break 1-WL indistinguishability and is strictly more expressive than static node embeddings.

### Weaknesses
* While consistent, the improvements shown in Tables 1-4 are often marginal (e.g., 83.07% → 83.30% for GCN on Cora). This raises questions about the practical significance of the added complexity.
* While the paper claims linear scaling, the actual runtime and memory overhead compared to baselines is not reported. Each node now requires storing and updating an entire MLP.
* The experiments focus on relatively small graphs (largest is PubMed with ~20K nodes). Scalability to million-node graphs is unclear.
* While the paper claims Node2Net doesn't affect inductive properties, the pre-training step for unseen nodes seems problematic. How well does this work for completely new nodes at test time?
* The paper dismisses node-ID methods as having "randomness and instability" but doesn't provide empirical comparisons to support this claim.
* The paper doesn't provide insights into what types of graphs or tasks benefit most from node-specific parameterization.

### Questions
* Can you provide concrete runtime and memory comparisons? For a graph with 1M nodes, what is the actual memory footprint of storing 1M MLPs?
* Given that Node2Net can break 1-WL, why are the empirical improvements so modest? Are the benchmarks not challenging enough to showcase the theoretical advantages?
* How exactly does the pre-training step work for nodes that appear only at test time? What if these nodes have feature distributions very different from training nodes?
* Can you include empirical comparisons with random node features/IDs to substantiate your claims about their instability?
* Are there specific graph properties (e.g., heterophily, specific structural patterns) where Node2Net shows more significant improvements?
*

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In the submitted manuscript, the authors propose to include a node-wise lightweight MLP in different deep learning methods on graphs, including unsupervised node embedding methods such as node2vec, standard Graph Neural Networks, such as GCNs, GAT and GraphSage and graph transformers such as GraphGPS. The authors show that this addition allows MPNNs to distinguish graphs that are not 1-WL distinguishable. They furthermore provide empirical results on 5 datasets for their different model variants.

### Strengths
- The proposed idea has not been explored in the literature on Graph Neural Networks as far as I know. 

- The ability to map automorphic nodes to different representations could potentially be interesting if properly explored and analysed.

### Weaknesses
- It seems to me that the presented idea does not generalise to unseen nodes and that the authors therefore have to rely on the exclusion of their proposed MLPs for the test set in the case of GNNs and rely on pretraining or gradient scheduling tricks to not harm performance in practice. 

- The observed performance improvements are insignificant in the majority of cases: In Table 2 there is only one dataset for which one GNN shows a significant improvement over the baseline. In Table 4 no significant improvement is observed. So, the only improvements you observed were in the context of the node2vec method, where you introduced your lightweight networks only after training a node2vec model in isolation. I am therefore, not convinced of the empirical benefit of your method. 

- Certain presentation styles like the lists in Lines 126-151, 301-310 and (especially) 447-455 are reminiscent of the writing style of an LLM, the use of which is not disclosed by the authors. I think it may be better to opt for a more compact form of presentation, which may avoid such suspicion.

### Questions
1] I have a fundamental question about your approach: Since MLPs are universal approximators, a global shared MLP (as is standard in GNNs) can learn any well-defined function, which should include almost all functions that your formulation fitting one MLP per node can learn. The only conceptual difference that I see between your approach and the global shared MLP is that you are able to learn functions that are "not well-defined", i.e., functions that map identical inputs to different outputs. This minor difference would also explain why in practice your approach does not yield significant performance improvements on GNNs. Is this right? Or are there further function classes that a global shared MLP cannot learn, which your Node2Net formulation is able to fit? 

2] The added time and memory cost of your method in practice is not clear to me. Could you provide these statistics for the results you show in Tables 1, 2 and 4. I suppose since your Node2Net is usually added after a base model is pre-trained, your method should come at an additional training time cost and since it introduces more parameters, it should almost surely increase the memory cost? 

3] In Table 2 I am curious what the parameter count of the compared models is? Does your Node2Net variant have more parameters than the base model? And if yes, how does the performance of the base model change when you allocate equally many parameters to the globally shared update step as you do in your Node2Net model?

4] The performance of your graph transformer variant is only measured on one dataset. This seems like an insufficient empirical evidence basis to make any conclusive statements about the improvements of your method. Would it be possible for you to include further datasets in your study of graph transformers?

5] Minor Comments:

5.1] Typo: "nerual" in Line 197

5.2] In Line 243 you define the attention mechanism of graph transformers to only aggregate over neighbourhoods of nodes. To the best of my knowledge, most graph transformers aggregate over the whole node set instead. I think it may be appropriate to edit your formula accordingly, unless you discuss a particular kind of graph transformer here?

5.3] The proof of Theorem 1 has two \qed symbols. 

5.4] I found Section 3.5 to have a very low information density. It seems to me that the content of this section could be expressed a lot more concisely. In particular, several of the points mentioned in Lines 301-10 seemed somewhat obvious to me after reading the remainder of your paper.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Node2Net, a node-specific parameterization method designed to enhance the expressiveness of GNNs. The authors equip each node with a learnable function to model nonlinear feature interactions while preserving feature-dependent variability. This approach is akin to adding node IDs to each node, where an MLP is learned for each node in a manner similar to Node2Vec. Node2Net extends the representational power of GNNs beyond 1-WL indistinguishability while maintaining linear scaling in both computation and memory. Experiments are conducted to validate the proposed approach.

### Strengths
- The writing is adequate.
- The idea of learning node embeddings with separate MLPs is novel, though it has not yet been fully developed into a functional approach that can be truly useful.

### Weaknesses
- Why are Node MLPs initially trained as identity mappings?
- There is no clear advantage demonstrated in the GNN comparison experiments.
- The authors claim that Node2Net does not alter the transductive or inductive generalization properties of the backbone. However, the inductive part of this claim is not convincing.
- The authors do not claim permutation invariance in expectation. Several works, such as PF-GNN, perform inductive modeling with learnable positional encodings. Given that both approaches aim to break the symmetry of graphs with learnable position embeddings, the authors should compare their method with PF-GNN.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
