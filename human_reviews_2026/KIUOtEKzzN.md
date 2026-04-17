# NoLLMRAG: LLM-Free Makes Graph-Based RAG Highly Efficient, Effective and Generalizable

- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Graph-based Retrieval-Augmented Generation (graph-based RAG) improves retrieval relevance and multi-hop reasoning compared to traditional RAG by constructing a graph that models relationships among text chunks. However, existing methods heavily rely on LLMs during indexing, resulting in inefficiency and unstable performance across LLM scales. Moreover, during retrieval, the lack of effective mechanisms for extracting query keywords and filtering irrelevant chunks further leads to redundant retrieval, introducing noise and degrading answer quality. To address these limitations, we propose NoLLMRAG, a novel graph-based RAG framework which is LLM-free during indexing and retrieval. It builds a three-layer heterogeneous graph index without LLMs, leverages a graph-statistics-driven keyword extraction to select keywords from queries that are aligned with the corpus, and applies a clustering-based retrieval on co-occurrence subgraphs to select more relevant chunks for generation. Experiments on three datasets and three LLMs demonstrate that NoLLMRAG achieves an average improvement of 41.27\% over the strongest baseline, with indexing speedup of up to 300$\times$ and QA speedup of up to 15$\times$, and maintains robust adaptability for real-time corpus expansion, highlighting its superior performance, efficiency, and generalization across LLMs and domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes NoLLMRAG, a GraphRAG framework that eliminates LLM dependence by leveraging NLP tools to build a three-layer heterogeneous graph and extract query keywords. A clustering-based retrieval algorithm is proposed to retrieve relevant text chunks.

### Strengths
1. This paper demonstrate that leverging the NLP tools instead of LLMs can also achieve good results for GraphRAG.
2. The proposed method demonstrates effiency gains compared with LLM-based methods.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The novelty is limited. It is a common practice to extract keywords using NLP tools such as Named Entity Recognition (NER) [1]. The proposed Three-Layer Heterogeneous Graph is also quite common.
2. The motivation is not clear. The authors claim that clustering-based retrieval algorithm can reduce redundancy and improve retrieval quality and introduce several steps. However, what is the motivation behind these steps?
3. Lacks of baselines and evaluation datasets. The authors only use 3 baselines and 3 datasets. More baselines and datasets should be used to make a comprehensive evaluation.


[1] Named Entity Extraction for Knowledge Graphs: A Literature Overview.

### Questions
1. Why does HippoRAG 2 fails when applying the Qwen2.5-3B model?
2. What is the retrieval performance of the proposed method?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces NoLLMRAG, a graph-based RAG framework that removes LLMs from the indexing pipeline. It leverages graph-statistics-driven keyword extraction and clustering via keyword co-occurrence to better align queries with the corpus. Experiments across diverse QA datasets and LLMs of varying sizes show consistent gains over state-of-the-art graph RAG in accuracy, efficiency, and generalization.

### Strengths
S1: Tackles a timely and important problem for both industry and academia.
S2: Achieves high computational efficiency and low cost by using tokenizer-based token extraction instead of LLM-based entity extraction.
S3: Delivers competitive performance, validated with LLM-as-a-judge evaluations.

### Weaknesses
W1: The effectiveness of the proposed importance score is not compared against established graph ranking methods (e.g., PageRank, SALSA, HITS).
W2: The mechanism of cluster-based chunk retrieval are insufficiently explained.
W3: The construction details of the three-layer heterogeneous graph are missing (node/edge types, counts, and connectivity).

### Questions
1.	Tokenizer choice: Is the tokenizer tied to a specific LLM, or can users select any tokenizer independent of the LLM used for embeddings? Please clarify compatibility, token normalization, and handling of subword units.
2.	Importance score validation: The design resembles TF-IDF but lacks evidence of correlation with node “importance.” If ground-truth importance is unavailable, compare against baselines such as PageRank and SALSA (e.g., overlap percentage of top-k nodes, Kendall tau/Spearman correlation, precision@k).
3.	Graph statistics: Given fine-grained tokenization, the graph likely contains many token nodes. Please report node and edge counts by type (token/chunk/document; token–token, token–chunk, chunk–document), average degrees, sparsity, and memory footprint.
4.	Cluster-based retrieval: Elaborate the procedure (objective, stopping criteria, hyperparameters) and include a step-by-step example on a small corpus to illustrate iterations, cluster formation, and chunk selection, along with complexity analysis.

### Soundness
3

### Presentation
3

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
The method builds a three-layer heterogeneous graph using traditional NLP tools instead of expensive LLM calls. It extracts keywords from queries using graph statistics rather than LLM understanding. The system also uses clustering-based retrieval to reduce redundant information and improve answer quality.

### Strengths
NoLLMRAG cuts out LLMs during indexing and retrieval, which makes things way faster. It's up to 300× quicker at indexing and 15× faster for answering questions compared to other graph-based RAG methods. Instead of expensive LLM calls, it just uses basic NLP tools like spaCy to build the graph. The system stays fast even when you keep adding new documents, so it actually works for real applications.

The retrieval part groups related keywords together to avoid grabbing irrelevant stuff. All these pieces work together to create a system that doesn't rely on LLMs at all, which is different from existing methods.

NoLLMRAG handles complex multi-step questions better while still doing fine on simple ones.

### Weaknesses
The three-layer graph idea is clever, but it would be helpful to understand how it captures relationships without LLMs. Right now the Token-Token edges just connect words that are next to each other in documents. This works okay but might miss connections between related ideas that aren't neighbors.

It could be interesting to try other ways of connecting tokens. Maybe connecting similar tokens across different chunks. These approaches might catch more meaningful relationships than just word order.

The paper doesn't really explain how the system weighs different edge types. Document-Chunk edges show hierarchy while Token-Token edges show sequence. How does retrieval balance these different relationship types? Some discussion of this would help readers understand the design better.

The importance scoring approach is neat, but some choices seem worth exploring more. Why multiply the three frequency measures and then take the log? What about adding them together or picking the maximum? The multiplication could get tricky when any frequency gets really small.

The log transformation makes sense intuitively, but testing other combinations might show if this is really the best approach. The threshold τ = 0.5 also seems like it was picked somewhat arbitrarily. Maybe trying different values or making it adapt to different queries could work better.

### Questions
See my analysis above for specific details on these questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes NoLLMRAG, an LLM-free graph-based RAG framework to address efficiency bottlenecks in existing methods. The approach constructs a three-layer heterogeneous graph using traditional NLP tools, employs graph-statistics-driven keyword extraction, and implements clustering-based retrieval. Experiments show 41.27% average accuracy improvement with up to 300× indexing speedup and 15× QA speedup.

### Strengths
1. **Significant efficiency improvements.** The LLM-free design achieves dramatic speedups (16-300× indexing, up to 15× QA) while maintaining competitive accuracy, addressing an important practical problem in RAG systems.

2. **Good robustness and noise reduction.** Shows strong performance across different LLM scales, especially on small models where baselines fail. The graph-statistics approach effectively reduces retrieval redundancy as validated by ablation studies.

3. **Comprehensive experiments.** Evaluation covers three QA task types, multiple LLM scales, and various metrics including real-time corpus expansion scenarios.

4. **Clear presentation.** Well-structured paper with logical flow and comprehensive experimental results.

### Weaknesses
1. **Essentially a pre-LLM approach.** While advantages over LLM-based methods are discussed, comparison with traditional non-LLM RAG methods (BM25, TF-IDF, hybrid retrieval) is missing, making it unclear whether improvements come from novel contributions or just avoiding LLM overhead.

2. **Limited baseline comparison.** Only three graph-based RAG baselines are compared. Missing comparisons with classical IR methods and traditional graph algorithms weakens the evaluation.

3. **Limited semantic understanding.** The statistical approach cannot handle cross-document concept alignment or synonyms (e.g., "ML" vs "machine learning"), which may hurt performance on complex reasoning tasks. The authors should provide more evidence on whether it hurts cross-document concept alignment to not use LLMs.

4. **Small LLM evaluation only.** Testing limited to small models (3B-7B, GPT-4o-mini). Should evaluate on larger models to better assess generalizability claims.

### Questions
1. **How does your method compare to traditional non-LLM RAG approaches?** It would be valuable to see comparisons with classical methods like BM25-based retrieval, TF-IDF, and hybrid retrieval systems to better understand whether the improvements come from your novel graph construction or from avoiding LLM overhead.

2. **Could you provide broader baseline comparisons?** The current evaluation against three graph-based methods is somewhat limited. Comparisons with additional sophisticated graph indexing methods and classical information retrieval algorithms would strengthen the evaluation.

3. **How does your method handle semantic variations and synonyms?** The statistical approach may struggle with cases where the same concept is expressed differently (e.g., "ML" vs "machine learning"). Could you discuss this limitation and its potential impact on performance?

4. **How does performance scale with larger, more capable LLMs?** The evaluation focuses on smaller models. Testing on larger LLMs like GPT-4o or Claude Sonnet would help validate whether the observed advantages persist when baselines have access to stronger semantic understanding capabilities.

### Soundness
3

### Presentation
3

### Contribution
3
