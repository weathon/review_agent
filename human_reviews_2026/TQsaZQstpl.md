# Cross-Domain Pre-training of Transformers on Text-Attributed Graphs via Random Walks

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Pre-training large-scale models with diverse data using the Transformer architecture has driven significant advances in natural language understanding. Motivated by this success, we explore pre-training strategies for graph representation learning that leverage the flexibility of Transformers. A key challenge is enabling a sequence-based Transformer to effectively encode graphs of varying sizes and from diverse domains. To address this challenge, we represent nodes as collections of random walks, allowing the Transformer to learn node embeddings from sequential contexts. We provide theoretical analysis on the expressive capacity of this representation for distinguishing graph structures. We also introduce a novel context prediction loss tailored to random walks. Empirically, we show that the proposed pre-training strategy can be adapted to various downstream graph tasks, highlighting its promise for processing and reasoning with graph-structured data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a unified Transformer-based framework for cross-domain graph pre-training, focusing on text-attributed graphs. The key idea is to represent graph structures as sequential inputs via random walks, enabling the use of Transformer architectures similar to those in NLP.The paper outlines four desiderata for pre-training across diverse graphs and demonstrates strong transferability in out-of-domain evaluation.

### Strengths
Originality：Introduces a new paradigm for cross-domain graph pre-training, using random walks and LLM-based embeddings to unify diverse graph structures；3.1, the dataset-level virtual tokens to distinguish graphs is creative.

Quality : Comprehensive experimental validation across multiple domains demonstrates strong and consistent performance;The results show good transferability even when pre-training on only a few representative datasets, confirming the framework’s generalization ability.

Clarify：.
Significance：Demonstrates that large-scale pre-training on a few datasets can generalize across domains, a valuable insight for developing graph foundation models

Significance：Demonstrates that large-scale pre-training on a few datasets can generalize across domains, a valuable insight for developing graph foundation models

### Weaknesses
While the paper discusses edge incorporation, more analysis on label leakage and task-specific fine-tuning would strengthen the claims.
The ablation on walk length ℓ and number of walks k could include a more detailed complexity discussion.

### Questions
How does the model perform when the node text information is noisy or incomplete?

How would the framework adapt to temporal or dynamic graphs?

### Soundness
4

### Presentation
2

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
The paper presents a Transformer-based model for text-attributed graph representation learning, utilizing random walks to represent graph nodes as sequences. Through a custom context prediction loss for self-supervised pre-training, the approach can handle the graph types from small molecules to large citation networks, and can be easily adapted to various downstream tasks via fine-tuning.

### Strengths
1. The paper proposes a single yet effective approach by transforming nodes into random walk sequences that can be processed by a Transformer, making it adaptable to diverse graph types.
2. The model leverages self-supervised pre-training to adapt to diverse datasets for cross-domain learning, offering significant transferability across tasks.
3. The paper provides a theoretical analysis of random walks’ expressiveness.

### Weaknesses
1. Lacking comparisons with recent studies, such as LLM-based methods like LLaGA  (Chen et al., 2024b), GraphGPT  (Tang et al., 2024), and unified GNN models like UniGraph  (He & Hooi, 2024), which are mentioned in Appendix A.
2. The figures are kind of hard to interpret without accompanying figure captions or explanations.

### Questions
1. In the few-shot learning results shown in Table 6, the proposed method achieves strong performance across nearly all settings, but it underperforms on the ARXIV 5-way task under both 1-shot and 5-shot conditions. Could the authors provide insights into the possible reasons for this?
2. Molecular graphs and knowledge graphs differ significantly in scales. Could the authors discuss some theoretical or empirical guidelines for selecting the walk length $l$ and the number of walks $k$ for these different graph types?
3. Why can representing molecular graph structures through random walks achieve performance comparable to graph transformer models such as GPS, which incorporate structural encodings? Could the authors provide more analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces RWPT, a cross-domain graph pre-training approach that encodes each node using multiple random walks and processes them with a Transformer, trained via a contrastive context prediction loss. The model transfers well across datasets and tasks.

### Strengths
1. Effective cross-domain transfer without retraining the backbone.
2. Random-walk representation captures long-range structure.
3. Strong results across node, link, and graph classification.

### Weaknesses
1. The topic has been widely explored before, lack novelty.
2. Computationally heavy due to long sequences and Transformers.
3. Relies on text-attributed nodes; less applicable to non-text graphs.

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
2

### Rating Number
2

### Confidence
5

### Summary
This work explores pre-training graph representations using Transformer architectures. It represents nodes through collections of random walks, enabling Transformers to model diverse graph structures. The study provides theoretical justification for this representation and introduces a random-walk-based context prediction loss, demonstrating strong transferability.

### Strengths
1. The paper is well-written and easy to follow.
2. The idea of sequentializing graphs with random walks is reasonable.

### Weaknesses
1. The proposed method requires positional encoding based on shortest-path distances, which explicitly injects the graph’s inductive bias into the model. As a result, the reconstruction of local neighborhoods is unsurprising. In practice, computing shortest-path distances for large graphs can be time-consuming, limiting the method’s scalability.

2. The proposed method claims to capture long-range dependencies; however, in the main experiments, the walk length was set to 4, which is insufficient to support such a claim. With 8×4 short walks, the sampled contexts likely resemble local k-hop neighborhoods, suggesting that the properties of random walks are not fully exploited.

3. The method involves the use of advanced LLMs (e.g., LLaMA), which produce high-quality node features initially. For a fair comparison, the authors should clearly specify the feature types used in the baselines, as the observed improvements may stem from the strong textual embeddings rather than the proposed framework itself.

4. Many experimental details are missing, such as dataset splits. Table 1 shows that pretrained graph models perform consistently better than individually trained baselines, which contradicts prior findings that graph pretraining often fails to yield significant gains.

5. The transfer learning results in Table 2 are surprisingly strong, e.g., pretraining on a single out-of-distribution dataset leads to substantial improvements over training from scratch. This is counter-intuitive and arguably too good to be true, yet the authors provide no explanation for it.

6. Table 2 and Fig. 4 show that RWPT achieves significantly larger gains on link- and graph-level tasks than on node-level tasks. The authors should clarify the factors contributing to this discrepancy.

7. Following 5, a Transformer pretrained on limited (and potentially OOD) data is unlikely to learn complex random walk patterns. Given the strong results, the model may be relying on simple heuristics, such as mean aggregation. To verify that the model has learned meaningful interaction patterns, the authors are encouraged to analyze intermediate representations (e.g., attention patterns). Additionally, a simple baseline using non-parametric mean aggregation over LLM features, followed by an MLP task head (optionally with shortest-path encodings), should be included to provide a fair comparison and establish a starting point for evaluating more complex methods such as RWPT.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
