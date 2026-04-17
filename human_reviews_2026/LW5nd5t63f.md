# E$^2$GraphRAG: Advancing the Pareto Frontier in Efficiency and Effectiveness for Graph-based RAG

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Graph-based RAG methods like GraphRAG demonstrate strong global understanding of the knowledge base by constructing hierarchical entity graphs, but often suffer from inefficiency and rigid, manually defined query modes, limiting practical use. To address these limitations, we present E$^2$GraphRAG, a streamlined graph-based RAG framework that advances the Pareto frontier of Efficiency and  Effectiveness. In the indexing stage, E$^2$GraphRAG utilizes large language models to generate a summary tree, and NLP tools to construct an entity graph from document chunks, with bidirectional indexes linking entities and chunks for efficient lookup. In the retrieval stage, the graph structure filters related entities, while the bidirectional indexes map these entities to their corresponding chunks, supporting an adaptive mechanism that dynamically switches between local and global modes. Experiments show that E$^2$GraphRAG achieves up to $10\times$ faster indexing than GraphRAG while maintaining comparable QA performance, advancing the Pareto frontier with respect to effectiveness and efficiency. Our code is available at https://anonymous.4open.science/r/E-2GraphRAG-8897.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
$E^2GraphRAG$ is a streamlined framework for graph-based Retrieval-Augmented Generation that aims to advance the Pareto frontier of efficiency and effectiveness of the GraphRAG paradigm. The approach enables efficient graph construction and retrieval while maintaining superior performance. Constributions inlcude:
1. integration of summary trees and entity graphs for lightweight indexing
2. adaptive retrieval strategy that automatically selects query modes using graph structure

### Strengths
1. Efficiency Improvements. By replacing LLM-based entity extraction with standard NLP tools and simplifying extracting specific relation to sentence-level "co-occurrence", the method reduces indexing time and computational overhead, making it more practical for large-scale knowledge base.
2. Adaptive Retrieval. The graph-filtering approach for mode selection is innovative and flexible.
3. Strong Empirical Validation, as shown by the experiments.

### Weaknesses
While the proposed model appears novel and promising, I identified several contradictions and points of confusion that could undermine the paper's overall quality and the replicability of its conclusions.

1.  Usage of SpaCy Sacrifices Generalizability Without Reducing Overall Complexity

In discussing the motivation for adopting SpaCy-based NER, the authors claim that it reduces computational costs compared to LLM-based alternatives. However, the construction of the summary tree still necessitates processing the entire corpus through LLMs for recursive summarization, rendering the LLM token costs largely unavoidable. Furthermore, the complexity of LLM-based NER would be approximately $O(N \cdot T)$ (where $N$ is the number of chunks and $T$ is the average token size per chunk). This is comparable to the costs incurred by the summary tree construction. It is not the primary bottleneck relative to those costs. Consequently, employing SpaCy does not alter the overall token complexity of the indexing stage. 

It is also worth noting that SpaCy-based NER can introduce noise and often struggles with domain-specific corpora, potentially compromising the accuracy of entity extraction in specialized contexts (e.g. legal, finance, medical).

2. Limited novelty in indexing stage

The summary tree construction is similar to the community summarization in GraphRAG[1] and the recursive summarization in Raptor[2], which has been widely used.

3. When building the summary tree, the chunks are grouped by taking the consecutive $g$ chunks following the order in the original passages. How can this be generalize to the domain where this bias is not available? For instance, in a knowledge base where all chunks are independent passages. If you use clustering to group chunks in this scenario, the efficiency may be an issue.

- [1] From Local to Global: A Graph RAG Approach to Query-Focused Summarization, arxiv-2404.16130
- [2] RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, ICLR, 2024

### Questions
Plz see above.

### Soundness
3

### Presentation
3

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
This paper proposes E2GraphRAG, a graph-based retrieval-augmented generation (RAG) framework that improves both efficiency and effectiveness over existing methods such as GraphRAG and LightRAG. The key idea is to combine a hierarchical summary tree (built by LLM summarization) with an entity co-occurrence graph (constructed using lightweight NLP tools instead of LLMs). The model builds bidirectional indexes between entities and chunks to enable adaptive retrieval that automatically switches between local and global query modes. Experiments on long-document QA datasets show that E2GraphRAG achieves up to 10× faster indexing and 100× faster retrieval than GraphRAG, while maintaining comparable QA accuracy.

### Strengths
Well-motivated improvement over GraphRAG, targeting efficiency–effectiveness trade-offs.

Clever use of non-LLM entity extraction tools to reduce computational cost.

Comprehensive experiments and ablation studies that clearly demonstrate both speed and accuracy gains.

### Weaknesses
The paper’s novelty lies mainly in the integration and efficiency-oriented redesign of existing graph-based RAG components rather than in proposing a fundamentally new retrieval or reasoning paradigm. It offers practical, well-engineered improvements but limited conceptual originality.

### Questions
N/A.

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
This paper introduce $E^2$GraphRAG, a GraphRAG method with NLP tools to construct graphs. Specifically, the authors first build a hierarchical chunk summarization tree, and then enrich it by linking chunk nodes to an entity-level knowledge graph extracted using NLP techniques. For each query, the proposed retrieval strategy adaptively selects either local or global retrieval. Experimental results show that the method outperforms existing baselines while maintaining efficiency.

### Strengths
1. The designed retrieval mechanisim can automatically choose local or global retrieve.
2. The authors compared the number of input and output tokens of different methods.
3. The authors provided detailed ablation study of the proposed method to demonstrate the effectiveness of different designs.

### Weaknesses
1. The novelty of the proposed method is limited. Constructing knowledge graphs using NLP tools such as entity and relation extraction has been widely explored in prior NLP and RAG-related studies, and thus cannot be considered a key contribution of this work. In addition, the hierarchical summarization tree is largely inspired by RAPTOR, which further reduces the originality of the approach.

2. The overall writing quality needs improvement. The paper reads more like a technical report rather than a well-structured academic paper. It lacks clear motivation, and the rationale behind each design choice is not well explained. Many components appear ad hoc, without sufficient theoretical grounding or empirical justification.

3. The experiments are only conducted on three datasets, which is not sufficient to demonstrate the robustness and generalizability of the proposed method. Additional benchmarks, especially multi-hop QA datasets such as MultihopRAG, should be included to more comprehensively validate the effectiveness of the approach.

### Questions
1. In the retrieval, "queries whose entities are densely connected are processed locally, while others fall back to global retrieval." What if there is only one entity in the query?

2. What is the reason that the authors summary the consecutive chunks? What if chunks in different documents share the similar semantic meaning?

3. Where did the authors get the heuristic of Graph Filtering, "truly relevant entities tend to be semantically related and thus connected in the constructed graph".

### Soundness
2

### Presentation
2

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
This paper introduces E2GraphRAG, a novel Retrieval-Augmented Generation framework that advances the Pareto frontier of efficiency and effectiveness in graph based RAG systems. Its core innovation lies in combining a hierarchical summary tree built with large language models and an efficient entity co occurrence graph constructed using traditional NLP tools like SpaCy. These two components are interconnected through bidirectional entity to chunk and chunk to entity indexes. This hybrid structure supports adaptive retrieval by allowing queries to dynamically trigger either local entity centric retrieval or global dense embedding retrieval, based on detected query characteristics and graph structure. This approach eliminates the rigidity and computational inefficiency found in prior work such as GraphRAG.

### Strengths
- The writing and figures of the paper are clear.
- The discussion of indexing efficiency in graph-based RAG systems for corpora with long documents is meaningful.

### Weaknesses
- The main concern is the novelty in the proposed framework. The core module for constructing the graph combines the "merge and summarize text chunks to construct a hierarchical tree" approach (similar to RAPTOR) with building a concise entity graph from dispersed chunks (similar to knowledge graph construction-based methods, such as GraphRAG and HippoRAG2). The use of lightweight tools such as SpaCy and BERT is also not new compared to existing methods. As a result, the framework seems to be an incremental integration of existing baselines, rather than offering strong innovation.
- In the experiments, the base LLM used in this paper is not consistent with the baselines. Key baselines, such as RAPTOR, HippoRAG2, and LightRAG, are evaluated using stronger generation models, including LLaMA-3.3-70B-Instruct and GPT-4o-mini API, whereas all models are tested solely with a 7B-scale LLM in this paper. This mismatch raises concerns about the credibility of the experimental results, particularly since the proposed method claims to offer better accuracy and efficiency.
- The experiments only consider datasets with long documents, ignoring classic multi-hop question answering GraphRAG datasets [1] such as MuSiQue, 2WikiMultihopQA, and HotpotQA. This omission may lead to unfair comparisons. While indexing efficiency and accuracy on long documents are important, evaluations on standard multi-hop datasets should not be neglected.

[1] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models

### Questions
As noted in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
