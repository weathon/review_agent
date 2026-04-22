# BrowseNet: Graph-Based Associative Memory for Contextual Information Retrieval

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Associative memory systems face significant challenges in efficiently retrieving semantically related information from large document collections, particularly when queries require traversing complex relationships between concepts. Traditional retrieval-augmented generation (RAG) approaches often struggle to capture intricate associative patterns and relationships embedded within textual data. To address this limitation, we propose BrowseNet, a novel associative memory framework that leverages query-specific subgraph exploration within a named-entity-based graph for enhanced information retrieval. Our method transforms unstructured text into a graph-of-chunks representation, where nodes encode document chunks with semantic embeddings and edges capture lexical relationships between content segments. By dynamically traversing the graph-of-chunks based on query characteristics, BrowseNet emulates content-addressable memory systems that enable efficient pattern matching and associative recall. The framework incorporates both structural similarity derived from lexical relationships and semantic similarity based on embedding representations to optimize retrieval performance. We evaluate BrowseNet against established RAG baselines and state-of-the-art (SOTA) pipelines using publicly available datasets that require associative reasoning across multiple information sources. Experimental results demonstrate that BrowseNet achieves SOTA performance in exact match score over both the graph-based RAG approaches and the dense retrieval methods. The two-pronged approach combining structural graph traversal with semantic embeddings enables more effective associative memory retrieval, particularly for queries requiring the integration of disparate but related information. The code and datasets are open-sourced at: https://github.com/bisect-group/BrowseNet

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an implementation of GraphRAG. The idea is to decompose the query into a graph of sub-queries using LLM, create a graph of passages based on common enities, and use the query graph to guide the exploration of the passage graph to select chunks to be fed into LLM for generation.

### Strengths
S1. The paper is generally easy to read.

S2. The experiments are well designed, providing retrieval and generation performance as well as extensive ablation studies.

### Weaknesses
W1. Novelty is limited. KG construction (Section 3.1), query decomposition (Section 3.2.1), and answer generation (Section 3.3) are standard practice in the literature. The rest of the approach, KG traversal (Section 3.2.2), is heuristic and its generalizability is not well justified given experiments on only three datasets.

W2. Baselines are focused on recent RAG solutions, but the proposed approach is closely related to early multi-hop QA solutions, which are not compared empirically.

W3. The constructed KG is actually not a KG but a graph of passages connected based on common entities. This is a common practice in conventional multi-hop QA research. It is not a KG that should represent relations between entities.

### Questions
Q1. Can you compare your approach with early multi-hop QA methods that also build a graph of passages based on their similarity (e.g., common entities)?

Q2. How do you demonstrate the generalizability of your heuristic graph traversal?

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
4

### Summary
The paper introduces BrowseNet, a framework for retrieval-augmented generation (RAG) that represents document collections as graphs of semantically related text chunks. Nodes correspond to document segments, while edges represent lexical or semantic associations between them. During query processing, BrowseNet decomponses a question into directed acyclic graphs and then performs query-specific subgraph exploration to retrieve relevant information, combining both structural similarity (through graph traversal) and semantic similarity (through embeddings). The system is evaluated on multiple multi-hop QA datasets, showing competitive or superior performance to existing RAG, GraphRAG, and dense retrieval baselines.

Overall, this is an interesting and well-executed paper that tackles a relevant challenge in retrieval-based question answering: how to retrieve information that is indirectly connected through multiple associative steps. The approach is well-motivated, and the experimental setup is thorough. However, some of the claims, particularly regarding LLM independence and the nature of the “knowledge graph”, feel overstated or insufficiently substantiated.

### Strengths
1. Representing document chunks as graph nodes connected by lexical or semantic relations is a clear and intuitive way to capture associative structures in text.
2. The paper provides ablation studies and analyses of multiple components (e.g., graph generation, retrieval mechanisms, and QA results), which help the reader understand what drives performance.
3. The proposed method performs competitively, and in some cases surpasses, both dense retrieval and graph-based RAG baselines.
4. The paper is well-written and easy to follow, with good structure and figures that illustrate the key ideas effectively.

### Weaknesses
1. While the results are competitive, other claims such as BrowseNet minimizing dependence on LLMs are not really backed by how the method is desgined. It relies on LLMs at several stages, for creating the KG, embedding chunks, decomposing queries, etc. It is therefore not clear what the actual cost is in comparison with other methods.
2. An important source of cost is in retrieval for non-initiator nodes. This requires considering a total of $k^p$ chunks that need to be scored, increasing the cost of the method.
3. The query decomposition is based on a directed acyclic graph, which might lead to a limitation for queries that do involve cycles or cannot be expressed as a DAG. Whether this is a source of issues is not discussed.
4. Calling the constructed graph a “knowledge graph” seems somewhat misleading, since the edges primarily encode textual similarity or shared entity mentions, rather than semantic relations between well-defined concepts. It might be more accurate to refer to it as a semantic chunk graph or entity-linked similarity graph. Clarifying this terminology would prevent confusion for readers coming from the KG community.
5. Some improvements reported (e.g., Table 2, HotpotQA in Table 3) are small, and it’s unclear whether they are statistically significant. Including confidence intervals or significance tests (e.g., paired bootstrap) would increase confidence in the reported gains.

### Questions
1. Could you clarify in what sense the dependence on LLMs is reduced, or whether the key benefit lies more in structuring the outputs of LLMs rather than avoiding them altogether? Have you quantified the computational or monetary cost of these LLM-dependent components relative to baseline RAG methods?
2. The retrieval process for non-initiator nodes appears to involve scoring $k^p$ candidates. Could you provide more details about how this complexity behaves in practice?
3. How would BrowseNet handle queries that involve cycles, mutual dependencies, or other forms of recursive reasoning?
4. The isomorphic accuracy is an interesting way to measure the generated subgraphs. I assume that since the graphs are likely small, you used an exact algorithm for isomorphism check. Could you please elaborate on this?

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
4

### Summary
This paper proposes BrowseNet, a knowledge graph–based associative memory framework for multi-hop contextual information retrieval. Unlike traditional RAG systems that rely solely on semantic similarity, BrowseNet constructs a knowledge graph where document chunks are nodes enriched with embeddings and lexical entity links form edges. For each query, BrowseNet decomposes it into sub-queries and performs structured graph traversal to retrieve relevant subgraphs that better capture reasoning chains. Experiments on HotpotQA, 2WikiMQA, and MuSiQue show state-of-the-art performance in recall and exact match compared to dense and graph-based RAG baselines, while reducing LLM interaction cost through more efficient retrieval.

### Strengths
Proposes a knowledge graph–based traversal approach that decouples queries for more accurate and context-aware retrieval, achieving competitive results in both passage retrieval and answer generation.

GraphRAG is an important domain for facts-required questions answer.

### Weaknesses
The core idea focuses on context retrieval, but the novelty is limited since query decomposition and graph-based iterative retrieval have been explored previously; comparison to similar methods (e.g., SiReRAG, ArchRAG, GraphRAG) is missing.


Zhang, N., Choubey, P. K., Fabbri, A., Bernadett-Shapiro, G., Zhang, R., Mitra, P., ... & Wu, C. S. (2024). Sirerag: Indexing similar and related information for multihop reasoning. arXiv preprint arXiv:2412.06206.
Wang, S., Fang, Y., Zhou, Y., Liu, X., & Ma, Y. (2025). ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation. arXiv preprint arXiv:2502.09891.
Han, H., Wang, Y., Shomer, H., Guo, K., Ding, J., Lei, Y., ... & Tang, J. (2024). Retrieval-augmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309.

Scalability to large, real-world corpora and real-time use cases is not clearly addressed, as experiments rely on controlled corpora with gold evidence and distractors.

Fairness of comparison is unclear, particularly whether the same backbone models (generation, NER, embeddings) were used across baselines, which may confound reported improvements.

The current framework is optimized for structured multi-hop reasoning and may not directly generalize to open-domain retrieval or tasks with unstructured context dependencies.

The approach depends on LLM-based query decomposition, which may introduce structural or semantic errors when sub-queries are misgenerated, leading to cascading retrieval failures.

### Questions
See weakness

### Soundness
2

### Presentation
2

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
This paper proposed a method for gathering context for retrieval-augmented generation (RAG): first process the candidate documents into a form of graph (where nodes are chunks and edges exist when there are shared entities), and explore according to a query-specific subgraph generated by LLMs. The method achieves state-of-the-art performance across a wide range of question answering tasks that require multi-hop reasoning.

### Strengths
- The preprocessing of the corpus is intuitive and based on entities. The idea is widely applicable to other NLP tasks
 - Evaluation is done at all stages of the RAG pipeline: at the construction of the graph, subgraph planning, retrieval, and final answer generation. The evaluations presented are convincing.

### Weaknesses
- The graph constructed probably should not be called a "knowledge graph" -- usually in the context of information extraction a KG has entities as nodes and predicates as edges. The proposal here is a graph on chunks of text (which I agree that is a better format for downstream RAG than a traditional 3-tuple KG)
 - The context is gathered with one shot -- maybe one could execute retrieval multiple times as the LLM executes the query subgraph it generates? As in a beam-search like process, the retrieved context could evolve as the LLM tries to solve the question (as in the style of https://aclanthology.org/2023.acl-long.557/)? 
 - A snippet subgraph of the constructed graph would be illuminating-- I suggest the authors present one in the final version.

### Questions
- L195: $cos$ has a single argument: to denote cosine similarity, please write $\cos \angle (x, y)$.
- L202-L220: Please clarify this search process on how this is related to beam search.

### Soundness
3

### Presentation
3

### Contribution
3
