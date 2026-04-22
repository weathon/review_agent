# LP-RAG: A Link Prediction-Based Framework for Retrieval-Augmented Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Retrieval-augmented generation (RAG) strategies have empowered large language models (LLMs) through integration with external knowledge sources, thereby enabling more accurate, up-to-date, and contextually relevant outputs. Among these, graph-based RAG methods stand out as particularly prominent. These approaches aim to structure external knowledge into graphs and leverage relational reasoning to retrieve relevant information to the context at hand. However, existing approaches remain limited in their ability to exploit query-based semantic cues. In this paper, we propose LP-RAG, a link prediction-based framework for document RAG. Specifically, LP-RAG employs an LLM-prompted chunker and text encoders to construct a graph of similarity relationships among chunks, which is then augmented with chunk-conditioned synthetic queries that emulate potential questions for each chunk. This design enables the incorporation of chunk-specific semantic information for model training. In our framework, retrieval is cast as an inductive link prediction problem, where the goal is to predict chunk–query links. Notably, LP-RAG is model-agnostic and can incorporate any link prediction method (e.g., graph neural network–based predictors). To demonstrate its effectiveness, we evaluate LP-RAG across diverse benchmarks. Results show that LP-RAG consistently outperforms existing graph-based RAG methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LP-RAG, a novel framework that reformulates the retrieval step in Retrieval-Augmented Generation as an inductive link prediction task on a "chunk–query" graph. The core idea involves using an LLM to segment documents into fine-grained semantic chunks and generate synthetic queries for each, constructing a graph where edges represent chunk-query associations. A link prediction model is trained to score these connections, and at inference, a user query is treated as a new node; its link probabilities to all chunks are used for retrieval. Experiments demonstrate that LP-RAG outperforms existing graph-based RAG methods on several benchmarks while significantly reducing token consumption.

### Strengths
​​1  Framing retrieval as a link prediction problem is innovative. It effectively leverages the rich methodology of graph learning, opening a new direction for RAG research.

2 The use of LLM-generated synthetic queries as weak labels provides valuable supervisory signals. The ablation studies convincingly demonstrate the importance of this design choice for performance gains.

​​3 The document chunking strategy via a prompted LLM is a principled alternative to sliding windows, inherently reducing chunk size and directly contributing to the framework's token efficiency.

### Weaknesses
​​1  The computational complexity and resource demands of the proposed method for large-scale corpora are a primary concern. While the method is evaluated on small-scale datasets, its practicality for real-world applications with millions of chunks is questionable. The analysis in Section 3.3 is insufficient to ensure scalability.

​​2  The framework's performance heavily depends on the quality and diversity of the LLM-generated queries. The current strategy to increase diversity lacks quantitative validation. Redundant or low-quality synthetic queries could weaken the supervisory signal and harm generalization. Furthermore, the increased complexity and potential instability introduced by relying on an LLM for both chunking and query generation warrant deeper investigation.

3  Although token cost is reported, the critical metric of end-to-end latency for a user query is absent. The inference step involves computing a node embedding and running a full GNN forward pass, which could introduce significant delay. This latency might offset the benefits gained from token savings, and its absence from the evaluation is a notable omission.

4  The use of a uniform random negative sampling strategy, as mentioned in Section 3.4, is simplistic and likely introduces false negatives. In a k-NN graph, high-degree chunks may be semantically related to many queries, and random sampling could incorrectly label potential positives as negatives. The lack of experimentation with more sophisticated strategies like hard-negative mining is a weakness, as it may limit learning signal quality and final task performance.

### Questions
​​1 To address concerns about the synthetic queries inadvertently overlapping with the test set distribution, it would be better to perform a similarity analysis between the synthetic and real test queries. If a high similarity is found, an ablation study removing the high-similarity synthetic queries and retraining the model would help verify if performance is inflated by potential leakage.

2 How about the performance on large-scale datasets and the relationship between graph size (number of nodes/edges), memory usage, and inference latency?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
LP-RAG reframes RAG retrieval as link prediction on a graph. Documents are split into chunks, chunks are linked by semantic similarity, and each chunk gets synthetic queries that connect to it, forming a chunk–query graph. A link-prediction model is trained to decide whether a query should connect to a chunk; at inference it predicts the best chunks to retrieve for generation, yielding higher accuracy and lower context cost than common GraphRAG baselines.

### Strengths
Strengths:

- Learning-based retrieval: Casts retrieval as link prediction, letting the retriever improve beyond static similarity.

- Encodes query intent: Synthetic queries write likely user phrasing directly into the index, boosting recall for varied asks.

- Token efficiency: Tends to surface shorter, high-signal chunks → smaller context windows.

### Weaknesses
Weaknesses

1. Missing key baselines (e.g., HippoRAG2, GFM-RAG).
   The evaluation omits strong contemporary graph-RAG methods, making it hard to gauge progress against the current state of the art.

2. Scalability challenges.
   The graph grows quickly with chunks and synthetic-query nodes/edges, and link-prediction models (e.g., GNN-based) can become memory- and time-intensive on large corpora.

3. Transferability.
   How transferable is the approach?: adapting to a new corpus typically requires regenerating synthetic queries and (re)training the link-prediction model rather than zero-shot transfer.

### Questions
**Transferability.**
   How transferable is the approach?: adapting to a new corpus typically requires regenerating synthetic queries and (re)training the link-prediction model rather than zero-shot transfer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LP-RAG --- a link-prediction-based approach using GNNs for document retrieval-augmented generation (RAG). Empirical studies on nine multi-hop RAG benchmarks demonstrate the effectiveness of the proposed approach.

### Strengths
**S1.** The paper proposes a novel method for document RAG by leveraging GNNs for link prediction.

**S2.** Empirical studies span a large number of datasets.

**S3.** The paper is overall easy to follow.

### Weaknesses
**W1.** Most importantly, the empirical studies consider a very small subset of the corpus (<100 documents) for each dataset. Existing graph-based approaches for document RAG focus on open-domain question answering (QA) when it comes to multi-hop QA, where the corpus should span at least thousands of papers. Properly evaluating the performance and scalability of the proposed approach against the baselines requires sufficient studies under this setting.

**W2.** The paper fails to compare against several highly relevant works.

- Gutiérrez et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. An impactful work that performs knowledge graph extraction for document RAG. NeurIPS 2024.
- Gutiérrez et al. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. ICML 2025. An improved version of HippoRAG that incorporates chunk nodes.
- Luo et al. GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation. NeurIPS 2025. GNN-based retrieval for document RAG.
- Alonso & Millidge. Mixture-of-PageRanks: Replacing Long-Context with Real-Time, Sparse GraphRAG. Also employs inter-chunk similarities for graph construction.
- Mavromatis & Karypis. GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning. This paper also leverages GNNs for graph retrieval.

Notably, many currently considered baselines are not designed to work well for multi-hop QA. E.g., GraphRAG tackles query-focused summarization, LightRAG is known to not perform well for multi-hop QA, etc.

**W3.** While the proposed approach involves training, the paper does not study the generalizability of the learned retrievers to datasets unseen during training.

### Questions
**Q1.** From section 3.2, it seems that the generated synthetic queries are all one-hop queries. Have you explored generating multi-hop synthetic queries?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to build a graph of text and synthetic queries and then formulate retrieval for RAG as predicting link between incoming queries and text nodes. Specifically, the text nodes are connected as a kNN graph, and then one or several text nodes are used to generate synthetic queries that link to them. Several GNN-based link prediction methods are tested as the retrieval model.

### Strengths
- The idea of formulation retrieval as link prediction is interesting, and the usage of synthetic queries as training data is also intuitive and sound
- The experiment results overall is strong, showing the effectiveness of the method

### Weaknesses
- As claimed in the paper, the chunking strategy is also newly provided and is different from baselines. However, chunking is more considered as part of data process and should be kept identical among compared methods
- I didn't see in the paper what the selection criteria (threshold) is used for LP-RAG. The reproducibility can be further improved

### Questions
- Table 2, Tech, misses the boldfaced best scores, and LP-RAG is not strong as claimed in the text
- I wonder what a inference latency comparison looks like between the proposed GNN-based method and a typical dense retriever with FAISS
- The baseline can be enriched further. Also, I feel it is better to include a baseline that uses the same synthetic queries and their corresponding texts to fine-tune a standard dense retriever and make a comparison, which shows the contribution of the graph-based model

### Soundness
3

### Presentation
3

### Contribution
3
