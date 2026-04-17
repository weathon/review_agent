# GraphSearch: Agentic Search-Augmented Reasoning for Zero-Shot Graph Learning

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recent advances in search-augmented large reasoning models (LRMs) enable the retrieval of external knowledge to reduce hallucinations in multistep reasoning. However, their ability to operate on graph-structured data, prevalent in domains such as e-commerce, social networks, and scientific citations, remains underexplored. Unlike plain text corpora, graphs encode rich topological signals that connect related entities and can serve as valuable priors for retrieval, enabling more targeted search and improved reasoning efficiency. Yet, effectively leveraging such structure poses unique challenges, including the difficulty of generating graph-expressive queries and ensuring reliable retrieval that balances structural and semantic relevance.
To address this gap, we introduce GraphSearch, the first framework that extends search-augmented reasoning to graph learning, enabling zero-shot graph learning without task-specific fine-tuning. GraphSearch combines a Graph-aware Query Planner, which disentangles search space (e.g., 1-hop, multi-hop, or global neighbors) from semantic queries, with a Graph-aware Retriever, which constructs candidate sets based on topology and ranks them using a hybrid scoring function. We further instantiate two traversal modes: GraphSearch-R, which recursively expands neighborhoods hop by hop, and GraphSearch-F, which flexibly retrieves across local and global neighborhoods without hop constraints. 
Extensive experiments across diverse benchmarks show that GraphSearch achieves competitive or even superior performance compared to supervised graph learning methods, setting state-of-the-art results in zero-shot node classification and link prediction. These findings position GraphSearch as a flexible and generalizable paradigm for agentic reasoning over graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GraphSearch, an agentic framework that equips an LRM with a graph-aware query planner that disentangles where to search from what to search for, and a Graph-aware Retriever that constructs structurally grounded candidate sets and ranks them with a hybrid score combining anchor–attribute similarity and query–semantic similarity. Experiments on six datasets for node classification (and three for link prediction) show competitive zero-shot performance vs. Search-o1, GraphICL, and supervised GNN/GraphLLM references, with reported efficiency gains in retrieval latency.

### Strengths
- Identification and separation between structure specification and semantic intent in queries is an interesting observation and the handling with a separate module is well-crafted
- Ablations on structure-aware queries vs. non-structural prompts and α-sensitivity are helpful
- Efficiency analysis is a nice touch

### Weaknesses
- The paper over-claims novelty as “the first framework” for agentic search on graphs; prior work, such as Graph-CoT[1] and RoG[2], already integrates planning, retrieval, and reasoning over graph structures. Please discuss and compare with the existing baseline empirically and how they differ from the proposed method conceptually. 

- Limited baselines and base models: comparisons omit other agentic retrieval augmentation frameworks (e.g., GraphRAG, G-Retriever, ReAct) and stronger LLMs such as Llama-3.1, Qwen-2.5, or Gemma-2, making it unclear how robust the proposed approach truly is and how generalizable it is. 

- The retrieval scoring relies largely on cosine similarity of attributes, raising concerns that structural information only constrains the candidate set rather than contributing meaningfully to ranking.


- No public code or detailed configuration is provided, hindering reproducibility.



[1] Jin, Bowen, et al. "Graph chain-of-thought: Augmenting large language models by reasoning on graphs." arXiv preprint arXiv:2404.07103 (2024).
[2] Luo, Linhao, et al. "Reasoning on graphs: Faithful and interpretable large language model reasoning." arXiv preprint arXiv:2310.01061 (2023).

### Questions
- How exactly is structural information integrated into the final ranking beyond attribute cosine similarity?
- Could the planner and retriever be jointly optimized or adapted via reinforcement learning or feedback, similar to RLVR or Search-o1?

### Soundness
3

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
5

### Summary
GraphSearch is the first framework to extend search-augmented reasoning to graphs, enabling zero-shot graph learning without task-specific fine-tuning. It has two key components and two traversal modes:
1. Graph-aware Query Planner: Separates search space (e.g., 1-hop, multi-hop neighbors) from semantic queries.
2. Graph-aware Retriever: Builds candidate sets based on graph topology and ranks them via a hybrid scoring function.
Traversal Modes:
1. GraphSearch-R: Recursively expands neighborhoods step by step.
2. GraphSearch-F: Flexibly retrieves across local/global neighborhoods without hop limits.

### Strengths
1. It is the first framework that extends agentic search-augmented reasoning to graph-structured data, enabling zero-shot graph learning.
2. Its core contributions include a graph-aware query planner (which decouples search space from semantic queries), a graph-aware retriever (which constructs candidate sets based on topology and uses hybrid scoring), and two traversal modes: GraphSearch-R and GraphSearch-F.
3. Extending search-augmented LRM (Large Reasoning Model) to the graph domain is a meaningful research direction, addressing the issue that existing methods cannot utilize graph structural information.
4. The design idea of decoupling search space (topology) from semantic content is clear, and the hybrid ranking function balances structural and semantic relevance.
5. Experiments on node classification and link prediction are conducted on 6 datasets, with comparisons against multiple baseline methods.

### Weaknesses
1. Essentially, it adapts existing search-augmented reasoning (e.g., Search-o1) to graph data, resulting in limited technical innovation.
2. It does not provide theoretical analysis on how graph structure affects reasoning.
3. It lacks a learning mechanism and fully relies on predefined prompt templates.
4. Figure 4 only demonstrates the impact of query structure awareness, with no ablation studies on other components.
5. It does not analyze the impact of different hop counts and candidate set sizes.

### Questions
1. Strengthen theoretical analysis and provide a formal framework for how graph structure affects reasoning.
2. Design a learning-based query planner instead of relying on fixed templates.
3. Conduct fair comparisons with the latest graph learning methods.

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
The paper introduces GraphSearch, a framework that extends search-augmented large reasoning models (LRMs) to graph-structured data for zero-shot node classification and link prediction. The system separates where to search (graph topology) from what to search (semantic intent) via a Graph-aware Query Planner and a Graph-aware Retriever. Two traversal modes are instantiated: GraphSearch-R (recursive, hop-by-hop expansion) and GraphSearch-F (flexible retrieval over local/global neighborhoods without hop constraints). Experiments on six datasets suggest GraphSearch reaches competitive or better results than supervised graph learners and achieves state-of-the-art in zero-shot settings.

### Strengths
1.Bridging search-augmented reasoning with graphs is important for domains (e-commerce, social, citations) where topology carries critical priors that plain-text RAG overlooks.

2.Disentangling topological scope from semantic query is a clean design that can reduce retrieval noise and focus computation on structurally relevant regions.

3.Demonstrating competitive zero-shot graph learning results on node classification and link prediction, is noteworthy and of practical interest to agents that must operate on new graphs without retraining.

### Weaknesses
1.Much of the method reads as policy design (planner prompts, scope flags, hybrid scoring) rather than a fundamentally new retrieval or reasoning mechanism. The technical contribution should be highlighted.

2.The compared methods are limited given the claim of “first framework” and SOTA zero-shot results. Include: (i) planner–executor RAG on graphs/text, (ii) dense–sparse hybrid retrievers with structural priors, (iii) recent GraphRAG variants, and (iv) supervised GNNs tuned under equal budget. Report identical token/latency budgets and prompt templates for fairness.

3.It would be better to analyze failure modes when the planner emits incorrect structural scopes or when graphs are noisy/sparse.

4.The performance of GraphSearch-R and GraphSearch-F varies significantly across different datasets, and there is a lack of in-depth analysis and explanation for this phenomenon.

5.Zero-shot claim needs a stronger evaluation. Current tasks and datasets are relatively standard. How about compositional, long-range, and heterophilic graphs where structural cues can conflict with semantics?

6.The efficiency evidence is limited. It would be better to break down token usage and latency across think, search, information, answer phases, and compare against chunk-based search with compressed contexts. 

7.Minor Typos:
Lines 140–142: repeated sentences—please remove.
Table 4: the Best metric annotation is incorrect on the Products dataset.

### Questions
How about the performance of GraphSearch with smaller open-source LLMs (e.g., Qwen2.5-7B)?

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
4

### Summary
GraphSearch extends the idea of "search-augmented reasoning" to graph-structured data, aiming to solve graph learning tasks such as node classification and link prediction in zero-shot scenarios without task-specific fine-tuning.
GraphSearch use a Graph-aware Planner to disentangle search space and a Graph-aware Retriever to construct candidate sets and rank candidates based on a hybrid scoring function.

### Strengths
1. **The Combination of Large Reasoning Model and Graph-Aware Reasoning is Illuminating**: GraphSearch use LRM to guide graph search process by generating graph-search instructions, which can enable expressive search space control. And equip LRM with enabling zero-shot graph learning
2. **Detailed Analysis of Comprehensive Experiments**: This work provides a thorough empirical evaluation across six diverse benchmarks for both node classification and link prediction tasks across multiple graph domains. Comparing against strong baselines, this work demonstrates good performance of GraphSearch.
3. **Exemplary Clarity and Presentation**: The paper is well-written and structured, significantly enhancing its accessibility. The authors effectively use visual aids and a clear flow to articulate a complex framework, making it easy for the reader to grasp the core innovations and their motivations.

### Weaknesses
1. **Low Computational Efficiency and Limited Performance Improvement**: All the target tasks to be addressed in this paper, which are node classification and link prediction tasks, are not complex enough to require the introduction of LRM for completion. Introducing LRM for reasoning will significantly increase the time cost of reasoning, while there is no significant improvement in performance compared with methods based on Graph Neural Networks such as GCN and GraphPrompter. 
2. **Missing Related Work about Search-Augmented Graph Learning**: Currently, Some search-augmented frameworks for graph data have already been investigated. For example, RAGraph[1] introduces RAG into graph learning, which is similar with the insight of this paper. It's better to incorporate these related works.
3. **Highly Relies on the Ability of LRM**: Using LRM to generate graph-search instructions places high demands on the capabilities of LRM itself. In the paper, merely employing *Qwen2.5-32B-Instruct* and *Qwen2.5-72B-Instruct-AWQ* models that lack long reasoning capabilities during their own pre-training process is not sufficiently convincing. It is recommended to add experiments with models that possess long reasoning capabilities (such as DeepSeek-R1) and models with smaller parameter sizes to enhance the persuasiveness of the research.

[1] RAGraph: A General Retrieval-Augmented Graph Learning Framework

### Questions
1. Will the attribute of graph information provided in the <information>...</information> lose topological context for the graph data? Since the information in  text modal is very limited.

### Soundness
3

### Presentation
4

### Contribution
3
