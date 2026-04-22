# DuConTE: Dual-Granularity Text Encoder with Topology-Constrained Attention for Text-attributed Graphs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Text-attributed graphs integrate semantic information of node texts with topological structure, offering significant value in various applications such as document classification and information extraction. Existing approaches typically encode textual content using language models (LMs), followed by graph neural networks (GNNs) to process structural information. However, during the LM-based text encoding phase, most methods not only perform semantic interaction solely at the word-token granularity, but also neglect the structural dependencies among texts from different nodes. In this work, we propose DuConTE, a dual-granularity text encoder with topology-constrained attention. The model employs a cascaded architecture of two pretrained LMs, encoding semantics first at the word-token granularity and then at the node granularity. During the self-attention computation in each LM, we dynamically adjust the attention mask matrix based on node connectivity, guiding the model to learn semantic correlations informed by the graph structure. Furthermore, when composing node representations from word-token embeddings, we separately evaluate the importance of tokens under the center-node context and the neighborhood context, enabling the capture of more contextually relevant semantic information. Extensive experiments on multiple benchmark datasets demonstrate that DuConTE achieves state-of-the-art performance on the majority of them.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces DuConTE, a novel model designed for text-attributed graphs, which integrate both semantic information from node texts and topological graph structure. Text-attributed graphs have significant applications in tasks such as document classification and information extraction. Traditional methods often use language models (LMs) to encode text and graph neural networks (GNNs) for processing the graph’s structural information. However, these methods usually focus on word-token granularity during the text encoding phase and fail to capture the structural dependencies between texts from different nodes.

DuConTE addresses these issues with a dual-granularity text encoder and a topology-constrained attention mechanism. The model employs a cascaded architecture of two pretrained LMs. The first LM encodes semantics at the word-token granularity, while the second operates at the node granularity. During self-attention computation in each LM, DuConTE dynamically adjusts the attention mask matrix based on node connectivity. This adjustment guides the model to better capture semantic correlations that are informed by the graph structure. Additionally, DuConTE evaluates the importance of tokens under two different contexts: the center-node context and the neighborhood context, enabling the model to capture more relevant semantic information that is contextually informed by both the individual node and its graph neighbors.

### Strengths
1. By dynamically adjusting the attention mask based on node connectivity, DuConTE incorporates graph structure into the semantic encoding process. This ensures that the model better understands how nodes are connected, which is a key feature for text-attributed graph tasks.

2. The use of dual-granularity encoding, which operates at both the word-token and node levels, allows for a richer semantic understanding of the nodes and their relationships within the graph.

3. This paper is clear and easy to follow.

### Weaknesses
1. Complexity of Cascaded Architecture: The dual-granularity approach, especially the use of a cascaded architecture of two pretrained LMs, introduces additional computational complexity. This may lead to higher training times and resource consumption, potentially making the model less scalable for large datasets.

2. Traditional LM embedding methods typically first obtain word embeddings, then assist with GNNs to perform node propagation. This cascaded approach in DuConTE is neither novel nor practical.

3. Many existing papers have already addressed the interaction problem, such as GLEM and they only used a single LLM to achieve word-level learning. 

4. The paper contains many formulas, but they are relatively trivial. The overall framework is quite simple and does not provide particularly insightful ideas.

5. The performance on ogbn-product is not that impressive but it's a minor issue as most datasets show better improvements.

### Questions
I suggest authors can consider reducing the complexity of the cascaded two-LM structure. A more streamlined approach could improve both computational efficiency and practical applicability, especially for large-scale datasets. 

It would be helpful to better highlight the novel aspects of the model. While the current method builds on existing approaches, offering deeper insights or introducing more innovative techniques could strengthen the paper’s contribution to the field.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposed DuConTeE, a framework that encode text rich graph using topology constrained attention, in the granularity of text and node. A compose layer is also proposed to encode the texts in node into single node representation.  

Comprehensive experiment and ablation study is performed and experiments demonstrated the effectiveness of the proposed approach.

### Strengths
1. The paper is easy to follow and understand 

2. The experiment and ablation study is comprehensive. 

3. The proposed approach reached state-of-the-art performance on various dataset.

### Weaknesses
1. The proposed approach is not novel, especially for "topology-constrained attention mechanism". Previous impactful work like K-BERT [1], has already proposed similar idea and not cited in this paper. And there are a lot of similar works during that time. Previous work before 2020 is not comprehensively surveyed.  

2. The effectiveness of different modules are incremental, according to ablation study in Table 2. The improvement of Mask and dual representation is small, it seems that only K-BERT like architecture have already reached state-of-the-art performance. This further reduce the novelty of this paper. 

[1] K-BERT: Enabling Language Representation with Knowledge Graph

### Questions
Refer to  the weakness part.

### Soundness
3

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
4

### Summary
This paper focuses on the text encoder component in text-attributed graphs and observes that most existing works overlook the incorporation of structural information during text encoding. To address this, the authors propose DuConTE, a dual-granularity text encoder that integrates graph structure into language model-based text encoding through topology-constrained attention, thereby enhancing semantic modeling for text-attributed graphs. Experimental results on five commonly used datasets demonstrate the effectiveness of the proposed method.

### Strengths
S1: The paper is clearly written, and the proposed method is easy to follow and understand.

S2: Experimental results on multiple datasets demonstrate the effectiveness of the proposed method.

### Weaknesses
W1: The experimental evaluation could be improved. The authors only conduct experiments on the node classification task and report Accuracy as the sole metric. Moreover, on the non-citation datasets (WikiCS and OGBN-Products), the proposed method does not show a significant improvement. It is recommended that the authors further validate the generalizability of their approach on e-commerce networks such as those used in the CS-TAG [1] work.

[1] A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking. NeurIPS 2023.

W2: Some implementation details of the proposed method are not clearly described. For instance, the setting of the key parameter k is not explicitly reported for each dataset, and it remains unclear how the authors handle longer contexts when feeding data into the RoBERTa model (whose maximum context length is 512).

W3: The proposed approach involves multiple attention operations, but the overall time complexity analysis is missing. The authors are encouraged to provide both theoretical complexity and empirical runtime comparisons.

W4: It would be helpful if the authors could include a brief description of the overall training process at the end of the methodology section, as this would improve the reader’s understanding of the proposed framework.

W5: Mechanisms similar to the proposed Topology-Constrained Attention are already quite common in the Graph Transformer literature. Therefore, the methodological novelty of this work may be somewhat limited.

W6: There are several typographical errors in the paper. For example, in Section 2.2, lines 131–132, the phrase “Another work, Revisiting, generates...” contains a typo.

### Questions
Q1： I am curious about the composer component of the proposed method. Would replacing it with a simpler operation, such as mean pooling, lead to a substantial drop in performance?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DuConTE (Dual-Granularity Conceptualization for Text-Attributed Graphs), a new framework designed to enhance text-attributed graph (TAG) learning by jointly leveraging coarse- and fine-grained semantics. The authors observe that prior GNN–LLM fusion models often focus on single-level representations, which limits the model’s ability to generalize across heterogeneous node semantics. To address this, DuConTE introduces two complementary components: a coarse-grained concept encoder that abstracts global node semantics via clustering-driven conceptualization, and a fine-grained text encoder that captures token-level contextual nuances using large language models. The framework then fuses the two representations through a dual-level contrastive alignment mechanism, encouraging the model to maintain consistency between local linguistic details and high-level structural abstractions. Extensive experiments on six text-attributed graph benchmarks show that DuConTE outperforms both GNN-based and hybrid GNN–LLM baselines on node classification and transfer learning tasks. The authors also demonstrate superior efficiency compared to deep fusion approaches, as DuConTE reduces redundant token usage while preserving expressiveness.

### Strengths
The paper’s strengths are threefold: 1) it proposes a conceptually elegant and practically useful dual-granularity design that integrates fine linguistic cues and abstract conceptual features, addressing a key limitation in prior single-granularity fusion methods; 2) the method strikes a strong balance between performance and efficiency, offering comparable or superior accuracy to deep GNN–LLM models but with reduced computational overhead thanks to compact conceptual representations and shared alignment objectives; and 3) the empirical evaluation is thorough and convincing, spanning six datasets, diverse baselines, ablation studies, and zero-shot transfer tests, all of which consistently validate the effectiveness of the dual-level alignment scheme in capturing multi-scale semantics and improving generalization.

### Weaknesses
The paper also has several weaknesses: 1) while well-structured, the novelty is somewhat incremental, as the core idea of combining coarse- and fine-grained semantic representations echoes hierarchical or multi-level contrastive learning paradigms already common in multimodal and text–vision literature, adapted here for TAGs rather than newly invented; 2) the fusion mechanism remains under-analyzed, as the paper provides limited interpretability or diagnostic insight into how the two granularity levels interact or when one dominates the other, leaving questions about robustness to noisy clustering or overly abstract concept nodes; and 3) the scalability discussion is lacking, as the model’s clustering step and dual contrastive objectives may become costly for large-scale graphs or streaming scenarios, yet these practical concerns are not empirically evaluated or theoretically bounded.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
