# Topology of Reasoning: Retrieved Cell Complex-Augmented Generation for Textual Graph Question Answering

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Retrieval-Augmented Generation (RAG) enhances the reasoning ability of Large Language Models (LLMs) by dynamically integrating external knowledge, thereby mitigating hallucinations and strengthening contextual grounding for structured data such as graphs. Nevertheless, most existing RAG variants for textual graphs concentrate on low-dimensional structures—treating nodes as entities (0-dimensional) and edges or paths as pairwise or sequential relations (1-dimensional), but overlook cycles, which are crucial for reasoning over relational loops. Such cycles often arise in questions requiring closed-loop inference about similar objects or relative positions. This limitation often results in incomplete contextual grounding and restricted reasoning capability. In this work, we propose Topology-enhanced Retrieval-Augmented Generation (TopoRAG), a novel framework for textual graph question answering that effectively captures higher dimensional topological and relational dependencies. Specifically, TopoRAG first lifts textual graphs into cellular complexes to model multi-dimensional topological structures. Leveraging these lifted representations, a topology-aware subcomplex retrieval mechanism is proposed to extract cellular complexes relevant to the input query, providing compact and informative topological context. Finally, a multi-dimensional topological reasoning mechanism operates over these complexes to propagate relational information and guide LLMs in performing structured, logic-aware inference. Empirical evaluations demonstrate that our method consistently surpasses existing baselines across diverse textual graph tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a topology-aware retrieval-augmented generation framework that represents textual graphs as cell complexes to capture higher-order relations such as cycles and loops beyond standard nodes and edges. It retrieves relevant multi-dimensional subcomplexes and performs topological reasoning across dimensions to guide an LLM in generating answers.

### Strengths
- A novel approach to include more relevant higher dimensional information in the RAG pipeline
- Mathematically strong paper. 
- The approach does improve upon the existing baselines.
- The approach provides interpretable reasoning traces

### Weaknesses
- The computational and memory cost of building and reasoning over high-dimensional cell complexes is not quantified or compared with simpler baselines.
- The approach still has its limitations - including more higher dimensional knowledge in case of incomplete/noisy graphs could lead to worse results. Moreover the approach will be very difficult to adapt in cases of dynamic graphs where new nodes/edges are added from time to time.

### Questions
- Can the topological retrieval appriach be adapted to handle multimodal graphs?
- Could the proposed topological framework be used for reasoning tasks that do not explicitly involve graph structures  (eg. scientific question answering or commonsense reasoning)?
- What is the intuition behind describing the graph as a cellular representation? 
- Instead of adding high dimensional knowledge, can't we instead find the relevant 0- and 1- dim and let any reasoning model to reason over this? My intuition is that doing this will have many benefits - limited training, more robust to graph changes etc. Although this won't result in better answers when using a smaller model.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Topology-enhanced Retrieval-Augmented Generation (TopoRAG), a novel framework for textual graph question answering. TopoRAG retrieves not just at the node, edge, or triple level, but also incorporates higher-dimensional topological structures. The authors evaluate TopoRAG on three textual graph QA datasets with three different settings and conduct ablation studies on the number of layers in the reasoning module, the K value of the retrieval module, and other factors.

### Strengths
- The proposed retrieval method enables retrieval at different granular levels, which can capture higher-dimensional topological and relational dependencies.

### Weaknesses
- **Comparing Methods are Not SOTA Models.** The authors compare with models like G-Retriever, SubgraphRAG, and GraphToken, but these are not state-of-the-art (SOTA). Some related work with better performance, such as DoG [1] and GCR [2], is not cited or compared. These models achieved better results on WebQSP than the proposed TopoRAG.

- **Selected Datasets and Their Suitability.** The selected datasets (ExplaGraphs and SceneGraph) may not adequately demonstrate TopoRAG's ability to capture high-dimensional topological and relational dependencies. Since these datasets usually require less than 2-hop reasoning, they might not be the most suitable for demonstrating the need for "higher-dimensional topological structures" to answer questions. The authors should consider datasets that require multi-hop (>3-hop) reasoning.
- **Sufficiency of 2-Cell.** The paper does not explain why (or if) a 2-cell is sufficient to capture higher-dimensional topological structures. Furthermore, there is no analysis of how the top-k 0/1/2 cells contribute to the results.


[1] Kun Li, Tianhua Zhang, Xixin Wu, Hongyin Luo, James R. Glass, and Helen M. Meng. 2025. Decoding on Graphs: Faithful and Sound Reasoning on Knowledge Graphs through Generation of Well-Formed Chains.

[2] Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, and Shirui Pan. Graph-constrained
reasoning: Faithful reasoning on knowledge graphs with large language models. ICML 2025.

### Questions
- In equation 15, will the $x^d (d \in \{1,2,3\})$ have an overlap? For example, if an edge was selected and the circle that contains this edge is also selected, how can such overlapping be addressed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TopoRAG, a framework for textual graph question answering that explicitly models higher-dimensional topological structures. The framework introduces: (i) cellular representation lifting that transforms graphs into cell complexes capturing 0-cells (nodes), 1-cells (edges), and 2-cells (cycles) for closed-loop reasoning; (ii) topology-aware subcomplex retrieval extending Prize-Collecting Steiner Tree to multi-dimensional complexes; (iii) multi-dimensional topological reasoning with message passing across cellular dimensions; and (iv) integration with LLMs for generation. Experiments on three benchmarks (WebQSP, ExplaGraphs, SceneGraphs) show improvements of up to 5.12% accuracy over GraphRAG baselines.

### Strengths
1. The paper is the first to introduce cellular complexes into the GraphRAG framework, explicitly modeling 2-cells to capture closed-loop topological dependencies that are ignored by traditional node-and-edge paradigms.

2. The method design is logically clear, with a coherent pipeline progressing from cellular lifting to topology-aware retrieval, multi-dimensional reasoning, and generation.

### Weaknesses
1. All fundamental cycles—regardless of semantic relevance—are lifted into 2-cells, which may introduce noise or spurious dependencies due to meaningless loops.

2. The experiments use datasets that involve multi-hop reasoning, but their graph structures inherently contain many cyclic dependencies. The paper does not evaluate on purely chain-like or acyclic reasoning tasks, making it unclear whether TopoRAG remains advantageous or introduces redundancy in non-cyclic scenarios.

3. The impact of spanning tree choice is not analyzed: 2-cells depend on the tree construction, yet the paper does not assess how different trees (BFS, DFS, random) affect results, raising concerns about robustness.

### Questions
1. Many fundamental cycles may be semantically irrelevant. Has the paper considered the impact of such low-quality cycles on performance? Are there methods to further reduce their influence?

2. Could the authors add experiments on acyclic or purely chain-style reasoning datasets to verify the method’s performance and generalizability in non-cyclic tasks?

3. Different spanning trees lead to different 2-cell sets. Have the authors tested the impact of tree choice on final performance? If there is an impact, how should it be balanced?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TopoRAG, a topology-enhanced retrieval-augmented generation framework for
textual graph question answering.
The key novelty lies in lifting textual graphs into regular cell complexes—representing nodes, edges, and
higher-order cycles as 0-, 1-, and 2-cells—and retrieving relevant subcomplexes conditioned on the query.
A topology-aware retrieval mechanism selects query-relevant cells by generalizing the Prize-Collecting
Steiner Tree (PCST) formulation to multi-dimensional complexes, and a multi-dimensional reasoning
module propagates information across these 0/1/2-cells.
Finally, the subcomplex representation is integrated into a frozen LLM (Llama-2-7B) via prompt tuning or
LoRA.
Experiments on ExplaGraphs, SceneGraphs, and WebQSP show strong improvements (≈ +5 % Acc/Hit
over GraphRAG baselines).

### Strengths
Introducing cell complex topology into RAG for textual graphs is new. Prior GraphRAG methods (e.g., G-
Retriever, GNN-RAG, SubgraphRAG) restrict reasoning to pairwise edges, while TopoRAG models cyclic
and higher-order relations. For a formal introduction. the paper carefully defines cell complexes,
homology, and the lifting procedure, demonstrating a strong understanding of topological deep learning
foundations. Besides, the whole pipline proposed in the paper provides a systematic way to inject
structured topology into frozen LLMs. Clear improvements over all categories of baselines (prompt-tuned,
LoRA, and pure inference) across multiple datasets. The ablations are detailed and show the contribution
of each module.

### Weaknesses
While using cell complexes for retrieval is novel, the reasoning stage (message passing with faces/cofaces)
resembles existing simplicial or cell complex GNNs (e.g., CWN, SAN, CellNN). It would be better to
integrate clarification about how its update scheme differs algorithmically from those prior works.
Building and storing cell complexes (especially identifying all 2-cells from cycles) may be expensive for
large textual graphs.
While topology sounds interpretable, the paper lacks visualizations or examples showing which cycles or 2-
cells are retrieved and how they affect reasoning.
Ablations remove modules but not the dimensional component (e.g., keeping only 0- and 1-cells).
Showing how much 2-cell reasoning alone contributes quantitatively would help isolate the topological
benefit.
typo:
Line 123: intersecting
Line 148: eq. 2

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
