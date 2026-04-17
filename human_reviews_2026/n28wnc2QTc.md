# Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Graph-based Retrieval-Augmented Generation (RAG) systems leverage interconnected knowledge structures to capture complex relationships that flat retrieval struggles with, enabling multi-hop reasoning. Yet most existing graph-based methods suffer from (i) heuristic designs lacking theoretical guarantees for subgraph quality or relevance and/or (ii) the use of static exploration strategies that ignore the query's holistic meaning, retrieving neighborhoods or communities regardless of intent. We propose \textit{Query-Aware Flow Diffusion RAG} (QAFD-RAG), a training-free framework that dynamically adapts graph traversal to each query's holistic semantics. The central innovation is \emph{query-aware traversal}: during graph exploration, edges are dynamically weighted by how well their endpoints align with the query's embedding, guiding flow along semantically relevant paths while avoiding structurally connected but irrelevant regions. These query-specific reasoning subgraphs enable the first statistical guarantees for query-aware graph retrieval, showing that QAFD-RAG recovers relevant subgraphs with high probability under mild signal-to-noise conditions. The algorithm converges exponentially fast, with complexity scaling with the retrieved subgraph size rather than the full graph. Experiments on question answering and text-to-SQL tasks demonstrate consistent improvements over state-of-the-art graph-based RAG methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Query-Aware Flow Diffusion (QAFD), a new graph-based retrieval method for retrieval-augmented generation. The key idea is to make graph diffusion sensitive to query semantics by dynamically adjusting edge weights based on semantic similarity. The authors also provide theoretical guarantees on convergence and subgraph recovery, and validate the method on multi-hop QA and text-to-SQL tasks, showing clear improvements over prior graph RAG approaches.

### Strengths
1. The proposed method is conceptually novel, integrating semantic and structural signals in a principled way.

2. Theoretical analysis provides valuable guarantees rarely seen in RAG research.

3. Experiments are comprehensive and show strong performance gains with reduced computation cost.

### Weaknesses
1. Despite the significant innovation in the graph construction process, the method still heavily relies on the quality of pretrained embeddings for semantic similarity, which may limit robustness under domain shift.

2. The method may struggle with handling logical negation. For instance, a query like “not red” may still activate nodes related to “red,” since it is unclear whether the embedding-based similarity explicitly encode negation or exclusion relationships. This could limit the model’s ability to capture fine-grained logical distinctions in retrieval.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Query-Aware Flow Diffusion RAG (QAFD-RAG), a novel framework for graph-based Retrieval-Augmented Generation that addresses the limitation of static, query-agnostic exploration in existing graph-RAG methods. The central innovation is dynamic query-aware graph traversal: instead of fixed edge weights, QAFD-RAG dynamically re-weights edges during exploration based on node-query semantic alignment through principled flow diffusion, prioritizing structurally present and semantically relevant paths. The paper provides theoretical guarantees (exponential convergence, subgraph recovery under mild conditions) and demonstrates empirical superiority over state-of-the-art baselines on question-answering and text-to-SQL tasks.

### Strengths
1. The paper presents a novel graph-based RAG approach, which demonstrates its effectiveness from the perspectives of intuitive understanding, theoretical proof, and experimental results.

2. This approach is training-free, which significantly reduces deployment costs, and holds great potential for broad application.

### Weaknesses
1. The entire framework is built upon LLM's embeddings, thus it is necessary to compare its sensitivity to embedding quality with other existing methods; however, such experiments are currently lacking.
2. As a general graph-based RAG framework, the experimental comparisons across domains remain somewhat limited. It would be beneficial to incorporate experiments in more diverse domains or conduct cross-domain validation to demonstrate the framework's robustness.

### Questions
1. Why is the number of seed nodes treated as a hyperparameter? Is it necessary to enable adaptive selection across different scenarios, why or why not?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes Query-Aware Flow Diffusion RAG (QAFD-RAG) framework for graph-based RAG that incorporates query semantics during the graph diffusion process.
During Indexing Stage, QAFD-RAG builds a knowledge graph from documents. And in the Query Stage, QAFD-RAG dynamically reweights edges and diffuses flow according to query alignment, enabling adaptive graph traversal that are semantically consistent with the query and highlighting reanoning paths.
QAFD-RAG is a training-free framework offering the statistical guarantees for query-aware subgraph retrieval.
Comprehensive experiments on multiple tasks demonstrate that QAFD-RAG achieves better performance compared with existing SOTA graph-based RAG methods, while significantly reducing the number of LLM calls.

### Strengths
1. **Novelty and Strong Motivation**: The motivation of this work is clear and well-argued, effectively highlighting the deficiencies of existing heuristic graph-based RAG methods. And the idea of using graph diffusion theory to deal with the graph traversal process is highly novel, which can effectively help find subgraphs that are semantically relevant with the query.
2. **Rigorous Theoretical Proof**: The provision of theoretical statistical guarantees is a significant strength. The convergence analysis (Theorem 3, Corollary 4) and, more importantly, the statistical recovery guarantees under a signal-to-noise condition (Theorem 7) provide a solid foundation for the method's efficacy. The analysis convincingly shows how query-aware weighting acts as a semantic filter.
3. **Comprehensive Experiments and Strong Performance Improvement**: This work benchmarks against a wide array of strong baselines across two distinct and challenging domains. The experiments results under various metrics provide compelling evidence for the method's effectiveness and efficiency.
4. **Clarity and Reproducibility**: The methodology is well described with precision, including the formulation, the algorithmic procedure, and the derivation process. Besides, code in the anonymous repository is complete and accompanied by some explanatory guidelines in the README, enabling easy reproduction of the work.

### Weaknesses
1. **Limited Analysis of Flow Interpretation**: It is easy for us to understand the basic principle of the flow diffusion process for water. However, why the graph flow diffusion can be meaningful and traverse to semantically relevant nodes by dynamic edge reweighting remains unexplained. The experiments also mainly focus on the final output without the deeper analysis of the diffusion process itself.
2. **Only Compared with Graph-based RAG Methods**: If the study can compare QAFD-RAG with some traditional SOTA RAG methods, or propose insights about how to combine QAFD-RAG with current SOTA RAG methods, it would be better to apply QAFD-RAG in practical scenarios.

### Questions
1. Does the way Knowledge Graph is constructed affect the graph-based RAG methods? I find that the performance of GraphRAG and LightRAG is poor on HotpotQA and 2WikiMultihopQA—is that related to this issue?
2. Will the demand of indexing before retrieval make the QAFD-RAG system hard to fit the scenario where RAG documents are modified frequently? As current KG construction is a static process, will it cost a lot to build such a large KG for graph retrieval?

### Minor Comments

1. It seems that the *Relevance* score of QAFD-RAG in *Legal* dataset in Table 1 is not the best one. HippoRAG (93.60) is better than QAFD-RAG (93.30) and should be marked correctly.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Query-Aware Flow Diffusion for Graph-based RAG (QAFD-RAG), a framework that enhances GraphRAG by incorporating query semantics into graph traversal. The authors design a query-aware diffusion process, where edge weights are dynamically reweighted based on alignment between node embeddings and the query.

### Strengths
1. The authors leverage flow-diffusion for subgraph retrieval.
2. The paper provides theoretical guarantees for convergence and subgraph recovery
3. The athors evaluate the proposed method on differnt tasks, such as General Question answering, Multi-hop QA and Text-to-SQL tasks.

### Weaknesses
1. The paper formatting does not appear to follow the official ICLR template. In particular, the line spacing is noticeably smaller than the required standard,

2. The paper claim existing GraphRAG methods rely on heuristic sugraph search strategies that are holistic query-agnostic. However, there are different retrievers, such as GNN-based Retriever such as GNN-RAG[1], and LLM-based Retriever, such as RoG[2,  which already consider the query semantic.

3. The writing could be improved. There is no logic in the current writting. For example, in line 139, the authors claim "We pose diffusion as a constrained optimization(6)", while the equation 6 is in Line 198. Also, the motivation of different components are also unclear. For example, what is the motivation for equation 4,5?

4. For the Multi-hop QA task, the authors only compare 2 simple GraphRAG baseliens.





[1] Mavromatis, Costas, and George Karypis. "Gnn-rag: Graph neural retrieval for large language model reasoning."
[2] Luo, Linhao, et al. "Reasoning on graphs: Faithful and interpretable large language model reasoning."

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
