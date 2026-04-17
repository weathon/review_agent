# G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Large language models (LLMs) excel at complex reasoning but remain limited by static and incomplete parametric knowledge. Retrieval-augmented generation (RAG) mitigates this by incorporating external knowledge, yet existing RAGs struggle with knowledge-intensive tasks due to fragmented information and weak modeling of knowledge structure. Graphs offer a natural way to model relationships within knowledge, but LLMs are inherently unstructured and cannot effectively reason over graph-structured data. Recent graph-enhanced RAG (GraphRAG) attempts to bridge this gap by constructing tailored graphs and enabling LLMs to reason on them. However, these methods often depend on ad-hoc graph designs, heuristic search, or costly agent pipelines, which hinder scalability and generalization. To address these challenges, we present G-reasoner, a unified framework that integrates graph and language foundation models for scalable reasoning over diverse graph-structured knowledge. Central to our approach is QuadGraph, a standardized four-layer abstraction that unifies heterogeneous knowledge sources into a common graph representation. Building on this, we introduce a 34M-parameter graph foundation model (GFM) that jointly captures graph topology and textual semantics, and is integrated with LLMs to enhance reasoning in downstream applications. To ensure scalability and efficiency, mixed-precision training and distributed message-passing are implemented to scale GFM with more GPUs. Extensive experiments on six benchmarks show that G-reasoner consistently outperforms state-of-the-art baselines, significantly enhances LLM reasoning, and achieves strong efficiency and cross-graph generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
LLMs reason well but are constrained by static, incomplete parametric knowledge. RAG helps by fetching external info, yet current methods falter on knowledge-intensive tasks due to fragmented sources and weak structure modeling. They propose G-reasoner, which unifies graph and language foundation models for reasoning over diverse graph-structured knowledge. Its core is QuadGraph, a four-layer abstraction that standardizes heterogeneous sources into a single graph. They also introduce a 34M-parameter Graph Foundation Model (GFM) that captures both topology and text and integrates with LLMs. With mixed-precision training and distributed message passing for scalability, experiments on six benchmarks show state-of-the-art results, stronger LLM reasoning, better efficiency, and cross-graph generalization.

### Strengths
1. The experimental validation of the method was well conducted based on a variety of experiments.

2. A general-purpose framework that can be extended to any data type was proposed.

3. It achieved higher performance compared to other baselines.

4. Despite its small size of 34M, it demonstrated efficient and strong performance.

### Weaknesses
Overall, this is a well-written paper with solid experiments. However, one weakness is that each component of QuadGraph (such as the document-level hierarchy and the knowledge graph hierarchy) integrates existing concepts rather than introducing entirely new ones, which limits its novelty. The same observation applies to the GFM component as well.

### Questions
There is no mention of this paper’s limitations. What are some of the limitations of this paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes G-Reasoner, a GNN foundation model-based system for unified reasoning over graph-structured knowledge.  The G-Reasoner system contains three main modules. It first constructs a hierarchical heterogeneous QuadGraph, which contains Community, Document, Knowledge Graph, and Attribute four layers. Then it devises a query-dependent GNN as the graph foundation model for relevant context retrieval. Given a question q, the model predicts whether each of the node (can be entity, document, passage, community and attribute) contribute positively to answer the question. To mitigate the label scarcity issue, the authors proposed to leverage pre-trained language models for knowledge distillation. In order to support training on large-scale graph, the authors propose mixed precision training.  Finally, the LLM reasoning step receives top-k relevant node, and use them to augment the LLM for final answer generation.

### Strengths
1.	The concept of a Unified QuadGraph is novel and, to the best of the reviewer’s knowledge, is introduced here for the first time.

2.	The proposed method G-Reasoner, outperforms competitive GNN/Graph Retriever-based methods in multi-hop question answering. 

3.	The prerequisite knowledge from sentence transformers shows to be useful in guiding the GNN. 

4.	The proposed mix precision training technique achieves higher throughput with lower GPU memory usage.

### Weaknesses
1.	G-reasoner does not propose new graph construction methods. Instead, it reuses graph construction techniques from three baseline methods, namely HippoRAG2, LightRAG, and Youtu-GraphRAG. 

2.	G-reasoner is not compared with various RAG systems, including but not limited to Search-o1 [1], Search-R1 [2], RAG-Gym[3], R1-Searcher [4], Collab-RAG [5] and RAG-Star [6], ReSearch [7], IRCoT [8]. 

3.	This manuscript and its related baselines adopt an alternative evaluation setup than the commonly-adopted Multi-hop QA evaluation setting. The commonly-adopted setup constructs the retrieval corpus with  the full Wikipedia dump [2,3,4,6,7] or all supporting plus distractor passages from all training, validation, and test questions [8]. 

In contrast, as stated in section 5.1, the manuscript follows the settings used in Gutierrez et al., 2024 (HippoRAG), which “*collect all candidate passages (including supporting and distractor passages) from our selected (the 1000) questions and form a retrieval corpus for each dataset*”.  This retrieval environment is overly idealized and does not reflect real-world conditions. 

The reviewer understands that constructing a graph over the full text corpus is practically challenging. However, based on the existing experimental results, one cannot conclude that: (1) the proposed graph retrieval method outperforms iterative RAG approaches (or can achieve competitive performance), and (2) the proposed G-reasoner is robust to larger graphs and applicable to open-domain question answering.

[1] Li et al., Search-o1: Agentic Search-Enhanced Large Reasoning Models

[2] Jin et al., Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

[3] Xiong et al., RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation

[4] Song et al., R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning

[5] Xu et al., Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration

[6] Jiang et al., RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement

[7] Xie et al., Interleaved Reasoning for Large Language Models via Reinforcement Learning

[8] Trivedi et al., Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions

### Questions
1. The reviewer suggests the authors to add the following experiments:

a)	Since the performance reported in Table 2 is not directly comparable to the multi-hop QA results in [1–8], re-evaluating some baselines on the current evaluation knowledge corpus (same subset of distractive passages) may help clarify the advantages and necessities of the proposed graph search over iterative dense/sparse document retrieval.

b)	Scaling-up the scope for the document corpus used in evaluation. For example, the MuSiQue dataset only contains 25k questions. Using all supporting documents can be less challenging than the 2Wiki dataset. This may help in validating the effectiveness of the proposed method when being adapted to real-world scenarios.

2. [Minor] Apart from providing references for graph construction solutions, the reviewer suggests the author to include a short description to introduce how we can construct the unified graph in this paper. 

3. For the HotpotQA, MuSiQue, and 2Wiki datasets, *performance can vary substantially under different evaluation settings* (such as open-domain, distractor, etc.)  To avoid confusion, apart from providing relevant references, the reviewer suggests the authors to explicitly state the retrieval scope for evaluation. It can be included in the appendices if the main sections do not have enough space.

The reviewer is willing to raise the score if the aforementioned issues are addressed.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes G-reasoner, a novel framework integrating graph and language foundation models for reasoning over graph-structured knowledge. The authors introduce QuadGraph, a unified graph abstraction, and a 34M-parameter Graph Foundation Model (GFM) powered by Graph Neural Networks (GNNs). The paper is theoretically sound and demonstrates strong experimental performance across six benchmarks, including QA datasets as well as domain-specific benchmarks.

### Strengths
**Novel Contribution:**

The QuadGraph abstraction is a promising innovation that unifies different types of graph structures, such as knowledge graphs and document graphs, into a single standardized representation. This is a key step toward addressing the challenge of generalizing reasoning models across diverse graph types.

The integration of GNN-based reasoning into a graph foundation model that leverages both graph topology and textual semantics is a novel approach that aligns well with current trends in deep learning and knowledge representation.

**Strong Empirical Results:**

G-reasoner consistently outperforms state-of-the-art methods, including graph-enhanced techniques like GraphRAG and HippoRAG, in multi-hop reasoning tasks across diverse domains.

The approach demonstrates strong cross-graph generalization, which is critical for real-world applications involving diverse knowledge domains.

**Efficient Training and Reasoning:**

The use of mixed-precision training and a distributed message-passing mechanism allows the model to scale effectively across large datasets and graph structures, ensuring both efficiency and performance.

### Weaknesses
See Questions.

### Questions
**Model Complexity and Interpretability:**

The GFM model, while powerful, could benefit from further discussion on its interpretability. Complex models like GNN-based foundation models often suffer from interpretability issues, making it hard to explain the reasoning behind predictions. This is particularly important for high-stakes applications such as legal | medical | safety reasoning.

**Lack of Comparison with State-of-the-Art GNN Methods:**

While the model is compared with existing graph-enhanced RAG methods, there is no direct comparison with state-of-the-art GNN-only models or pure graph models that don’t involve retrieval-augmented generation. This comparison could provide clearer insight into the unique benefits of GFM.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces G-Reasoner, a framework designed to unify graph reasoning and language reasoning by integrating Graph Foundation Models (GFMs) with Large Language Models (LLMs). The core idea is that while LLMs are powerful reasoners, they lack structural awareness; conversely, graph structures effectively represent relational knowledge but are underutilized in LLM reasoning. To address this, the authors propose three components: QuadGraph, Graph Foundation Model (GFM),  and a LLM-enhanced reasoning.
Experiments on six benchmarks show consistent improvement over state-of-the-art methods such as GraphRAG, HippoRAG, and GFM-RAG.

### Strengths
1. The paper introduce clear motivation and organization. The introduction frames the LLM-graph reasoning gap well. I also like the figures and tables that effectively summarize the pipeline and results.
2. The paper proposes a four-layer unified graph schema, which is elegant and general enough to cover multiple graph types (KGs, document graphs, hierarchical graphs).
3. The paper also conduct comprehensive evaluation. Benchmarks include both general QA datasets and domain-specific GraphRAG benchmarks. The ablations also clearly demonstrate component contributions. Efficiency experiments are also included.

### Weaknesses
1. The paper has an incremental novelty. The core idea (combining GNN-based reasoning with LLM retrieval) closely follows existing works like GFM-RAG and GNN-RAG; the contribution feels like a well-engineered unification rather than a conceptual breakthrough.
2. The overall methodological depth is limited. For example, the GFM architecture largely reuses standard DistMult-style message passing and MLP updates without introducing new graph learning techniques. The “QuadGraph” abstraction appears more of a schema design than a learnable innovation.
3. The paper can benefits from some interpretability analysis or case-study. The paper shows gains but does not deeply analyze why G-Reasoner helps LLM reasoning or how graph structure contributes. A case-study with visualization of reasoning paths showing the predicted top-k scores from the graph will be great.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3
