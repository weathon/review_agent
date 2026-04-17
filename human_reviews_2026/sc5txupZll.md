# PaX-RAG: Path-augmented and Cross relational Graph Retrieval Augmented Generation via Structural and Semantic Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
RAG

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PaX-RAG, a retrieval-augmented generation framework that replaces chunk-level or triple-level evidence with ordered “path” facts and aggregates multiple paths into path-summaries to better capture multi-entity, ordered relations. For retrieval, it introduces a training-free Optimal Transport–based set-to-set alignment over entity keys and relation phrases, combining entity-side and relation-side matchings for a joint path score. The method then performs path-centric fusion and unions the selected graph evidence with standard text chunks for generation. The authors argue that GraphRAG, PathRAG, and HyperGraphRAG can be seen as special cases of their framework, and report consistent gains over strong RAG baselines across five text domains.

### Strengths
S1. The research question is meaningful, which highlights the limitations of chunk retrieval and binary/hypergraph facts for multi-entity, ordered relations.

S2. Broad, consistent empirical gains across multiple domains and both binary/n-ary sources; ablations indicate all modules matter.

S3. OT for entity and relation sets gives a robust, auditable matching. Further, joint scoring is simple and training-free.

### Weaknesses
W1. Propositions that prior methods are special cases are stated but not shown with constructive mappings or counter-examples. Moving a succinct proof sketch (or a concrete mapping diagram) from the appendix into the main paper would strengthen the claim.

W2. Several strong graph-RAG variants are not compared. Also, ablations are only shown in one domain. Authors should add more recent Graph-based RAG baselines for comparison and include more ablation study results.

W3. The method leans on LLM prompts to segment text and link mentions to canonical entities. There’s limited analysis of extraction accuracy, error types (entity linking, relation phrasing), and robustness across noisy domains.

### Questions
Q1. Could you include explicit mappings from GraphRAG triples and HyperGraphRAG hyperedges to your path-summary formalism, and vice-versa?

Q2. What is the end-to-end latency versus StandardRAG, broken down into path extraction, OT (Sinkhorn) cost, and generation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel RAG framework, named PaX-RAG, that leverages path information to overcome the limitations of traditional chunk-based retrieval, which often overlooks complex multi-entity relationships. Unlike existing methods such as GraphRAG or HyperGraphRAG, it aggregates multiple paths into a single, coherent summary. In this way, it acquires a more holistic and ordered knowledge representation, free from the ambiguity and complexity of hypergraphs. PaX-RAG uses an optimal transport-based retrieval strategy to enhance accuracy by preserving the structural integrity of path sets and improving semantic matching for multi-entity queries. Extensive experiments show that this approach has outperformed other RAG methods in terms of retrieval accuracy, generation quality, and computational efficiency.

### Strengths
1. This paper models multi-entity facts in a novel path-summary structure that unifies ordered relational chains for avoiding knowledge fragmentation. Further, this is coupled with a robust OT-based retrieval strategy for effective set-to-set semantic matching, preserving structural integrity and highly improving query accuracy.

2. The authors have provided extensive theoretical evidence in Appendix E by giving elaborate proofs of the generalization properties of the framework, hence further establishing the theoretical soundness and validity of their proposed approach.

3. The empirical evaluation is extensive and proves that PaX-RAG significantly outperforms standard RAG and other graph-based baselines in multiple disparate knowledge-intensive domains.

### Weaknesses
1. The code is not available, and the implementation details are not clear.

2. Widespread typesetting errors and terminology confusion. For example, in Section 5, the method is repeatedly and incorrectly referred to as “HyperGraphRAG” instead of “PaX-RAG”.

3. Efficiency analysis is missing. Efficiency analysis of graph-based RAG is very important, but the cost for the OT-based retrieval approach and maintaining such a complex “path context graph” remains unknown. Comparisons with other baselines are needed.

4. Only GPT-4o-mini was used to perform the main experiments; this limits the demonstration of the generalization capability of the method. The authors are highly encouraged to further validate the robustness and scalability of the proposed method, PaX-RAG, on smaller open-source models, such as LLaMA or Qwen-based backbones, by providing evaluations to ensure consistent performance of the proposed framework across different architectures and parameter scales.

5. The formulation in Optimal Transport defines transport cost as purely cosine similarity between either entity or relation embeddings. Include a theoretical or empirical analysis of alternative definitions of the cost in OT beyond cosine similarity. Test Euclidean, Mahalanobis, or learned metric variants for sensitivity and retrieval robustness. 

6. Unclear advantage of ordered path construction over hypergraph representation. Despite this paper's claim that ordered path graphs capture multi-entity relations better than hypergraphs, it does not provide direct experimental evidence that isolates this effect. Without a controlled comparison, e.g., applying the same OT retrieval to both path and hypergraph graphs, the claimed structural advantage is not convincingly demonstrated.

### Questions
See above.

### Soundness
3

### Presentation
1

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
The paper proposes PaX-RAG, a novel retrieval-augmented generation (RAG) framework that integrates path-structured knowledge representations and Optimal-Transport-based retrieval.
Unlike traditional chunk-based RAG, GraphRAG (binary edges), or HyperGraphRAG (n-ary edges), PaX-RAG represents knowledge as ordered multi-entity paths aggregated into coherent path-summaries. It introduces a training-free OT distance to measure structural similarity between query and path sets, preserving the relational order and mitigating aliasing or redundancy problems.
The authors show theoretically that existing graph-based RAGs are special cases of PaX-RAG and empirically validate the approach on five domains (medicine, agriculture, CS, legal, mixed) and multimodal (Video-MME) benchmarks. PaX-RAG consistently outperforms all baselines in F1, Retrieval-Similarity, and Generation-Evaluation metrics, demonstrating both improved retrieval accuracy and generation quality.

### Strengths
Originality: Combines path-structured relational reasoning with OT-based retrieval.

Quality: Comprehensive experiments across domains and modalities, ablations isolating each component’s impact.

Clarity: The paper is well structured. The motivation for replacing hyperedges with ordered paths is intuitive and well-explained.

Significance: The technical formulation is generally sound. The OT-based set-to-set retrieval is derived correctly from Sinkhorn-regularized optimal transport.

### Weaknesses
1. The paper uses the Optimal Transport theory to support the proposed framework. However, the connection between OT theory and the core challenges of RAG is unclear. Why is converting to different distributions related to LLMs' needs? Why can the OT theory help us find multi-hop reasoning contexts? Proving existing models are "special-case" does not mean PaX-RAG would be effective. 

2. The proof of Proposition 4 is more like an argument and explanation. There are no rigorous mathematical steps. Thus, the paper gives readers the impression that it just wants to have some theory, though the theory is not that related to context retrieval and LLM reasoning.

3. Instead of discussing OT theory, the authors should discuss the advantages of using paths. I feel like the key idea of PaX-RAG can be summarized as extracting entities, building a bipartite graph G_p = (V, P), and then walking on the graph. By using this expression, what is the actual novelty of PaX-RAG (not the OT theory)? Thus, the paper should find a focus. The current paper title is long, which gives readers the impression that the focus is unclear.

4. The results in Table 2 show that other baselines are much worse than HyperGraphRAG and PaX-RAG. Can we have stronger baselines, given that so many graph RAG papers have been published? E.g., RAPTOR and GFM-RAG? Or does it mean the dataset is not that suitable? It may be because the questions for UltraDomain are generated by LLMs. Using LLM-generated questions to evaluate LLMs is problematic. Then, the authors can include at least one or two traditional datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue), or try some new dataset, e.g., GraphRAG-Bench.

5. The OT-based retrieval's complexity and latency are not benchmarked. Scalability for very large corpora remains unclear.

[1] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation.
[2] GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation.

### Questions
Please check the weaknesses. 

One additional question:
Are path summaries automatically generated via prompting or rule-based merging? How do you ensure their factual consistency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes ​​PaX-RAG​​, a GraphRAG framework that aims to address limitations in existing GraphRAG methods by modeling knowledge as ​​ordered paths​​ rather than isolated chunks or binary relations.

### Strengths
- The idea of using ordered paths as atomic knowledge units generalizes both extraction and graph retrieval, though I think all the existing methods can be abstracted in this way.
- Propositions formally show the existing methods are special cases; this is actually the same thing I mentioned before.

### Weaknesses
- The path extraction is over-claimed but implemented roughly. Pure dependency on LLMs to extract complex paths from single documents may not scale to ​​cross-document scenarios​​, where critical relational chains span multiple sources. No clear mechanism for cross-document path fusion is described. More path mining methods should be designed to support the idea.
- While OT is motivated as superior to vector similarity, ​​no direct ablation​​ compares OT against simpler set-embedding methods (e.g., averaging entity vectors or the concatenation of path words and then vectorizing them). Claims of OT's advantages remain unclear.
- The benchmark datasets are not widely adopted ones. Experiments should be conducted on HotpotQA, MuSiQue and 2Wiki.
- Video MME mismatches with PaX-RAG. What is the point here? Results show improvements when plugging into VLMs but ​​lack comparisons with RAG or GraphRAG baselines​​, making it unclear if gains stem from path-based retrieval or generic context augmentation.

### Questions
- How do you handle paths that require combining information from multiple documents? or how could you ensure the performance without path mining designs while purely relying on LLMs?
- Why not include a direct comparison between OT retrieval and query-path vector similarity?
- What is the point of Video MME? also no comparisons with other baselines?

### Soundness
2

### Presentation
3

### Contribution
2
