# Multi-Head RAG: Solving Multi-Aspect Problems with LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Retrieval Augmented Generation (RAG) enhances the abilities of Large Language Models (LLMs) by enabling the retrieval of documents into the LLM context to provide more accurate and relevant responses. Existing RAG solutions do not focus on queries that may require fetching multiple documents with substantially different contents. Such queries occur frequently, but are challenging because the embeddings of these documents may be distant in the embedding space, making it hard to retrieve them all. This paper introduces Multi-Head RAG (MRAG), a novel scheme designed to address this gap with a simple yet powerful idea: leveraging activations of Transformer’s multi-head attention layer, instead of the decoder layer, as keys for fetching multi-aspect documents. The driving observation is that different attention heads learn to capture different data aspects. Harnessing the corresponding activations results in embeddings that represent various facets of data items and queries, improving the retrieval accuracy for complex queries. We provide an evaluation methodology and metrics, multi-aspect datasets, and real-world use cases to demonstrate MRAG’s effectiveness. We show MRAG’s
design advantages over 18 RAG baselines, empirical improvements of up to 20% in retrieval success ratios, and benefits for downstream LLM generation. MRAG can be seamlessly integrated with existing RAG frameworks and benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Multi-Head RAG (MRAG), a simple yet intriguing extension to Retrieval-Augmented Generation (RAG). MRAG leverages activations from different attention heads in the Transformer’s multi-head attention (MHA) layers, rather than the final feed-forward output, to represent distinct semantic aspects of queries and documents. The authors claim this multi-aspect embedding improves retrieval performance for complex queries that require multiple, semantically distinct documents. They also provide datasets, evaluation methodology, and report up to 20% retrieval improvement over standard RAG baselines without additional training or storage overhead.

### Strengths
1. The paper clearly identifies a genuine limitation of current RAG systems in retrieving documents that represent semantically distinct aspects of a complex query.

2. The proposed Multi Head RAG is conceptually simple, practical to implement, and can be directly integrated into existing RAG pipelines and vector databases without additional training or storage overhead.

### Weaknesses
1. The core assumption that each attention head captures a distinct semantic aspect is not empirically validated within the experiments of this paper.

2. The work lacks qualitative evidence such as visualization or case analysis to show how different heads retrieve different information.

### Questions
1. Can the authors empirically demonstrate that each attention head corresponds to a different semantic aspect？

2. Would fine-tuning heads for retrieval improve specialization?

3. Can the authors quantify semantic complementarity across heads to substantiate the “multi-aspect” assumption within your setup?

4. Have the authors tried using cross-attention for aspect embeddings?

### Soundness
2

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
3

### Summary
The paper introduces Multi-Head RAG (MRAG), a retrieval-augmented generation framework that overcomes the limitations of standard RAG when handling multi-aspect queries. Instead of using a single embedding vector for retrieval, MRAG extracts the multi-head attention activations from the final Transformer layer, treating each head as a distinct semantic sub-space. Each sub-vector independently retrieves documents in parallel, and results are merged via a weighted voting scheme based on head importance. This design captures diverse semantic aspects without additional training or computational overhead. Experiments on synthetic and real multi-aspect datasets show that MRAG improves retrieval accuracy and downstream generation by 10–20% over vanilla RAG, while maintaining comparable efficiency and no degradation on single-aspect tasks.

### Strengths
1) Training-free use of multi-head attention activations as aspect-specific embeddings; plug-and-play with any Transformer, no model changes, and same embedding dimensionality as standard RAG (so minimal storage/latency overhead).

2) MRAG matches vanilla RAG’s leading terms while outperforming many recent variants in practicality.

3) Comprehensive evaluation design for multi-aspect queries (three datasets + bespoke metrics) and clear gains in retrieval and downstream generation.

### Weaknesses
1) The reranker is heuristic (voting with head-importance); effects vs. strong cross-encoder rerankers or dense-sparse hybrids aren’t deeply quantified.

2) Fusion strategies can add variance and computational/token cost, tempering the “free lunch” narrative when stacking with other RAG upgrades.

### Questions
- How robust are head-importance weights across domains/models, and does per-query dynamic head weighting beat the offline scoring used here? 

- What happens with extremely high aspect counts (≫25) or overlapping aspects—does retrieval saturate or fragment?

- Can MRAG’s gains persist with strong cross-encoders or rerankers in the loop, and what’s the net latency/throughput at production scale? 

- Beyond text, how well does the approach transfer to vision/graphs where MHA exists but semantics differ?

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
3

### Summary
In this work, to deal with multi-aspectual problems in retrieval augmented generation (RAG), that is, queries requiring the integration of multiple, semantically distinct aspects, the authors propose a new approach, Multi-Head RAG (MRAG). MRAG uses the representations from multi-head attention (MHA) modules of decoder blocks, different from the conventional model that just uses the output of the last decoder block. In addition to their MRAG, the authors created a new benchmark dataset for evaluating MRAG. Experimental results on the dataset show the effectiveness of MRAG against conventional RAG baselines.

### Strengths
- The proposed method for using multiple blocks and their multi-head attentions is novel.
- The authors clarify the position of their proposed method by carefully referring to conventional models.
- The authors created a new benchmark dataset that requires retrieving multi-aspect documents.
- The experiments on the created benchmark dataset show the effectiveness of MRAG.
- The authors discuss the validity of the computational complexity of MRAG.

### Weaknesses
- The experiments are conducted only with GPT-4o. To generalize the discussion about the observed results, additional models such as open language models should be used.

### Questions
- How did you decide the hyperparameters like $k$ in top-$k$?
- Did you compare MRAG with sparse-retriever-based approaches like one using a BM25-based retriever?

### Soundness
2

### Presentation
3

### Contribution
2
