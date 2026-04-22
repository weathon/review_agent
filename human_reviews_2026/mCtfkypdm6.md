# LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) is widely used to mitigate hallucinations of Large Language Models (LLMs) by leveraging external knowledge. While effective for simple queries, traditional RAG systems struggle with large-scale, unstructured corpora where information is fragmented. Recent advances incorporate knowledge graphs to capture relational structures, enabling more comprehensive retrieval for complex, multi-hop reasoning tasks. However, existing graph-based RAG (GraphRAG) methods rely on unstable and costly relation extraction for graph construction, often producing noisy graphs with incorrect or inconsistent relations that degrade retrieval quality. In this paper, we revisit the pipeline of existing GraphRAG systems and propose Linear Graph-based Retrieval-Augmented Generation (LinearRAG), an efficient framework that enables reliable graph construction and precise passage retrieval. Specifically, LinearRAG constructs a relation-free hierarchical graph, termed Tri-Graph, using only lightweight entity extraction and semantic linking, avoiding unstable relation modeling. This new paradigm of graph construction scales linearly with corpus size and incurs no extra token consumption, providing an economical and reliable indexing of the original passages. For retrieval, LinearRAG adopts a two-stage strategy: (i) relevant entity activation via local semantic bridging, followed by (ii) passage retrieval through global importance aggregation. Extensive experiments on four benchmark datasets demonstrate that LinearRAG significantly outperforms baseline models. Our code and datasets are available at https://github.com/DEEP-PolyU/LinearRAG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces LinearRAG, a GraphRAG framework that streamlines graph construction by employing lightweight entity extraction instead of complex relation extraction. It constructs a hierarchical graph comprising entities, sentences, and passages and utilizes a two-stage retrieval strategy that integrates both local and global structural information to enhance precision and recall.

### Strengths
S1: Addresses a timely and significant challenge relevant to both industry and academia.
S2: The proposed method is straightforward, achieving the fastest indexing speed alongside very low retrieval latency.
S3: Demonstrates competitive performance in token efficiency, content accuracy, and evaluation accuracy using large language models (LLMs) as judges.

### Weaknesses
W1: Some equations lack clarity and contain unexpected errors.
W2: The effectiveness of the dynamic pruning mechanism is not empirically evaluated.
W3: Since LLMs serve as the evaluators, it would be beneficial to incorporate multiple state-of-the-art LLMs from diverse vendors to mitigate potential evaluation bias.

### Questions
1. Entity activation is a crucial initial step in LinearRAG. However, the authors do not clarify how the framework handles queries from which no entities can be extracted. Does the retrieval process halt in the absence of seed entities? Additionally, if extracted entities are ambiguous and cannot be disambiguated without the query’s full context, could this lead to activation of incorrect entities and degraded retrieval performance? An explanation addressing these scenarios would strengthen the paper.

2. In Equation 5, the matrix M is not defined. Moreover, since σ_q is a ∣S∣×1 vector (per Equation 4), its transpose σ_q^T is 1×∣S∣, which mismatches the dimension ∣V_e∣ of α_q^(t-1) (from Equation 3). A more detailed explanation of this equation and its dimensions is necessary.

3. LinearRAG’s dynamic pruning currently applies a fixed threshold across all expansions. Given that dynamic pruning aims to produce high-quality semantic paths, empirical evaluation of this fixed-threshold approach is warranted but absent from the manuscript. Furthermore, since distant entities are generally less important than closer ones, justification for using a uniform threshold would be valuable.

4. For evaluation using LLMs as judges, it is recommended to include multiple state-of-the-art LLMs from different providers to reduce bias stemming from reliance on a single model.

### Soundness
3

### Presentation
3

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
This paper proposes a new framework for RAG, LinearRAG, =which overcomes the limitations of existing GraphRAG systems. The proposed LinearRAG replaces complex relation modeling with a relation-free hierarchical Tri-Graph, built using lightweight entity extraction and semantic linking. This design scales linearly with corpus size and avoids extra token consumption. The retrieval process follows a two-stage mechanism: First, local semantic bridging activates relevant entities beyond literal matches through semantic propagation. Second, global importance aggregation applies personalized PageRank to rank passages holistically.

Experiments on HotpotQA, 2WikiMultiHopQA, MuSiQue, and a Medical benchmark show that LinearRAG consistently outperforms both vanilla RAG and advanced GraphRAG methods in retrieval quality, generation accuracy, and efficiency. It achieves strong multi-hop reasoning performance with minimal computational overhead.

LinearRAG offers a practical and scalable solution for integrating structured retrieval with large-scale text corpora, avoiding the instability of relation extraction and achieving linear scalability in both construction and retrieval.

### Strengths
- This paper proposes a efficient and lightweight graph construction method, Tri-Graph, which avoids error-prone relation extraction in GraphRAG, maintains linear scalability, and achieves 0 token consumption in retrieval.
- This paper proposes a novel GraphRAG framework that combine local semantic bridging and global importance aggregation, whose effectiveness is valided through extensive experiments.
- This paper conducted extensive experiments to validate the effectiveness of LinearRAG, including sufficient main experiments against GraphRAG and vanilla RAG.

### Weaknesses
- The ablation study seems coarse-grained. For example, when constructing Tri-Graph, the current method simultaneously incorporates co-occurrence relationships between entities and both passages and sentences. How would the final performance be affected if only passages, only sentences, or paragraphs instead?
- The main experiment only compared with vanilla RAG. What about comparisons with some advanced RAG methods, such as LightRAG, Lexical Diversity-aware RAG, and others?

### Questions
- The token-free graph construction process appears similar to the context-graph construction in the paper *"Synthesize-on-graph: ..."*, as both utilize the position relationships between entities and texts as graph relationships. Could you please explain the key distinctions of your method?
- Other questions refer to Weaknesses.

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
This paper proposes LinearRAG, a novel GraphRAG framework designed to address key challenges in existing methods: the difficulty and noise associated with knowledge graph construction, and low retrieval efficiency. The core innovation is constructing a three-layer hierarchical graph (Tri-Graph) comprising entities, sentences, and paragraphs. During retrieval, the model first activates query-relevant entities via local semantic propagation and then uses these as seeds in a Personalized PageRank algorithm on a paragraph-entity bipartite graph to aggregate global importance for retrieval in a single pass. Both the time and space complexity scale linearly with the corpus size, and the entire indexing/retrieval process requires zero LLM API calls. Experimental results on four public datasets demonstrate that LinearRAG significantly outperforms existing GraphRAG methods.

### Strengths
1. The paper presents a strong, well-argued case that relation extraction errors are a primary bottleneck for GraphRAG performance. The decision to "de-relationalize" the graph and defer relation understanding to the LLM during generation is a conceptually clear and impactful design choice. 

2. The proposed Tri-Graph structure is a novel and potentially groundbreaking simplification of the GraphRAG paradigm. The retrieval mechanism, leveraging sparse matrix operations for semantic propagation and PPR, is computationally elegant. 

3. The framework's linear scalability, facilitated by update-friendly appends and the complete absence of LLM calls in indexing/retrieval, is a major strength. The demonstrated linear growth trend on a million-scale corpus underscores its high practical value for real-world deployment.

4. The paper provides an exceptionally thorough evaluation, going beyond standard QA accuracy to include critical efficiency metrics like token cost and latency. This holistic comparison makes the results highly persuasive and relevant for computation-budget-sensitive scenarios.

### Weaknesses
1. The Tri-Graph merges entity mentions based on string matching, which ignores polysemy (different entities with the same name). This is a significant limitation, as it can introduce erroneous co-occurrence edges, potentially leading to the retrieval of irrelevant paragraphs. While high NER accuracy is claimed, the paper does not quantify the impact of this inherent ambiguity on retrieval performance, leaving a major potential source of error unanalyzed.

2. Using binary weights (0/1) for sentence-entity edges fails to capture the varying semantic importance of different entities within a sentence. This uniform weighting strategy is a simplification that may amplify the influence of "noisy" or peripheral entities during the propagation step, potentially harming result relevance and interpretability. 

3. The parameters, particularly the activation threshold and the trade-off coefficient, are noted to be highly sensitive. The paper lacks an analysis establishing a functional relationship between these parameters and query characteristics or graph properties.

### Questions
Could the framework benefit from more informative weights for sentence-entity edges?

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
3

### Summary
The paper proposes LinearRAG, an efficient GraphRAG framework that constructs a relation-free hierarchical graph (Tri-Graph) using lightweight entity extraction and semantic linking over entities, sentences, and passages. For retrieval, LinearRAG adopts a two-stage strategy: (1) local semantic bridging to activate contextually relevant entities beyond literal matches, and (2) global importance aggregation via personalized PageRank to retrieve salient passages. This construction scales linearly with corpus size and incurs no extra token consumption, providing economical and reliable indexing for large corpora. Experiments on four benchmarks show consistent improvements over strong baselines in retrieval quality, generation accuracy, and scalability.

### Strengths
S1: The approach fits production constraints and integrates with standard embedding + reranking pipelines.

S2: The linear retrieval design emphasizes bounded expansions and pruning, offering predictable runtime and memory behavior on large graphs.

S3: The method encourages simple deployment: light graph construction, single-pass adjacency, and top-k selection, reducing system complexity compared to full GraphRAG stacks.

### Weaknesses
W1: Linear constraints risk under-retrieving long-range dependencies; multi-hop reasoning and compositional queries may suffer without fallback mechanisms.

W2: Limited theoretical analysis (e.g., recall/latency trade-offs, worst-case behaviors) could make the contribution primarily engineering rather than principled.

### Questions
Q1: What exact linear retrieval procedure is used (single-pass BFS-like scan, bounded-degree expansion, heuristic queue ordering)? How are thresholds chosen and adapted?

Q2: How are node and edge attributes computed (embedding similarity, co-occurrence, LLM scoring)? Are edge weights normalized and thresholded consistently?

Q3: What fallback exists for multi-hop or compositional queries requiring evidence aggregation? Can limited expansions be temporarily enabled without breaking linearity guarantees?

Q4: Are there systematic trade-off curves (recall vs. latency/memory) across different graph sizes and domains? How robust are results to parameter changes?

Q5: Do comparisons include equal-resource settings against beam search or greedy path selection to isolate algorithmic gains from compute differences?

### Soundness
3

### Presentation
3

### Contribution
3
