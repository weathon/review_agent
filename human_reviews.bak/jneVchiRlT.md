# FusionMaestro: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Table-text retrieval aims to retrieve relevant tables and text to support open-domain question answering. Existing studies use either early or late fusion, but face limitations.  Early fusion pre-aligns a table row with its associated passages, forming ``stars," which often include irrelevant contexts and miss query-dependent relationships. Late fusion retrieves individual nodes, dynamically aligning them, but it risks missing relevant contexts. Both approaches also struggle with advanced reasoning tasks, such as column-wise aggregation and multi-hop reasoning. To address these issues, we propose FusionMaestro, which combines the strengths of both approaches. First, the edge-based bipartite subgraph retrieval identifies finer-grained edges between table segments and passages, effectively avoiding the inclusion of irrelevant contexts. Then, the query-relevant node expansion identifies the most promising nodes, dynamically retrieving relevant edges to grow the bipartite subgraph, minimizing the risk of missing important contexts. Lastly, the star-based LLM refinement performs logical inference at the star subgraph rather than the bipartite subgraph, supporting advanced reasoning tasks. Experimental results show that FusionMaestro outperforms state-of-the-art models with a significant improvement up to 42.6% and 39.9% in recall and nDCG, respectively, on the OTT-QA benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper constructs a novel framework, FusionMaestro, for the table-text task. The method involves a well-designed pipeline that combines the advantages of early fusion and late fusion to generate more accurate table-text retrieval results. Experimental results confirm that FusionMaestro significantly outperforms existing baselines in both table-text retrieval accuracy and end-to-end response accuracy. Besides, the experimental results also proves the contribution of each process in the pipeline and validates the effectiveness of the proposed edge-based information granularity in improving results.

### Strengths
* Although the method and pipeline proposed in the paper are very complex and involve many details, the authors have meticulously written the paper, ensuring logical consistency and smooth readability. The writing level of this article is relatively high.
* Experimental results demonstrate that the proposed method has significant advantages compared to existing early fusion and late fusion methods.
* The experimental analysis is comprehensive and solid, examining the effectiveness of each component proposed.

### Weaknesses
The designed framework is very complex, with intricate steps and a vast amount of detailed information. Although the authors have provided open-source code, this poses a certain level of challenge to understanding the method proposed in this paper.

### Questions
I noticed that the proposed pipeline involves the utilizing LLM to refine the graph information. It can be seen that this step essentially leverages the capabilities of LLM to enhance the overall effectiveness of the pipeline. Is this fair compared to the existing baselines?

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
3

### Summary
In table-text open-domain question answering, a retriever system must retrieve relevant evidence from both tables and text to accurately answer questions. The core challenge in table-text retrieval lies in integrating these two information formats, as textual data is unstructured, while tabular data is highly structured. Existing methods are categorized into two main approaches based on the timing of table and text integration: early fusion and late fusion. Early fusion aligns table rows with associated passages in advance using entity linking, which can overlook query-dependent relationships and include irrelevant context. In contrast, late-fusion dynamically aligns relevant table rows and passages after the query is given, which, however, risks missing relevant contexts.
This paper introduces a new table-text retriever framework, FusionMaestro, which combines both early- and late-fusion strategies. FusionMaestro formalizes the retrieval task as a bipartite graph problem, representing table segments and passages as two disjoint node sets and the links between them as edges. The goal is to identify query-relevant edges within this graph. The pipeline consists of four main stages:
1. Early Fusion: Links are established between text and table data via entity linking to generate an initial graph.
2. Edge-based Bipartite Subgraph Retrieval: Each edge in the graph is embedded based on its connected table and text, while the query is also embedded, and relevant edges are then retrieved to form a subgraph.
3. Query-Relevant Node Expansion: Nodes are embedded, and seed nodes are identified based on their relevance to the query embedding. These seed nodes are then expanded along their edges within the graph.
4. Star-Based LLM Refinement: An LLM identifies the most query-relevant table segments or passages in the graph, filtering out irrelevant nodes.
The refined edge-scored graph ultimately provides a ranked sequence for retrieval results. Experiments on the OTT-QA benchmark demonstrate that this framework significantly improves retrieval performance.

### Strengths
1. The paper first formalizes the retrieval task as a bipartite graph problem, which is an interesting and reasonable approach that clarifies the task.
2. Building on bipartite graph modeling, FusionMaestro effectively combines early- and late-fusion methods. It begins with early-fusion to create an initial graph and then uses edges as the basic retrieval unit within this bipartite structure. This is a strong approach that leverages the advantages of both fusion methods: it selects query-relevant tables and text while integrating semantic information from both text and corresponding tables through early fusion, ultimately improving retrieval accuracy.
3. The proposed method employs prompts to instruct the LLM to aggregate table information and identify documents relevant to the query. By incorporating an LLM, this approach effectively enhances retrieval for queries that demand complex reasoning, such as multi-hop retrieval or column-wise aggregation.

### Weaknesses
1. The experiments involve different types of retrievers, which may lead to an unfair comparison. The proposed method uses ColBERTv2, a late-interaction retriever, which is more resource-intensive than single-embedding retrievers. However, none of the baselines use late-interaction retrievers. Given that the primary contribution of this paper is the retrieval framework itself, experiments should evaluate a single-embedding retriever to ensure consistency with the baselines and to more effectively validate the framework's effectiveness.
2. The paper does not provide many details of the proposed method. (a) **Edge weight calculation**: The final refined edge-scored graph $G_q$ generated by LLM lacks an explanation of how edge weights are calculated. Since the edge weights of $G_q$ are critical for the final output, this is a crucial part of the method. (b) **Role of edge weights**: Each stage in this method generates a graph with edge weights, yet the paper does not explain the purpose of these intermediate edge weights. It appears that edge weights before the “LLM Refinement” step are unused. (c) **Output results**: The authors state that “the refined edge-scored graph $G_q$ is decomposed into a ranked sequence of edges $\epsilon_q$ and returned as the result.” However, the paper does not explain how this decomposition is performed, leaving the reader uncertain about how the final output is generated.
3. Although the experimental results indicate that the proposed method performs well, it is significantly more time- and resource-intensive compared to the baselines. This approach requires the use of ColBERT and Llama-3.1-8B-Instruct, along with multiple rounds of LLM inference, which is highly demanding in terms of both computational resources and time.

### Questions
The questions are listed in the 'Weaknesses' section. Another question is whether you recorded the inference times of the methods in the experiments and if you could provide retrieval time results for the proposed method compared to the baselines.

### Soundness
2

### Presentation
2

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
This paper proposes a novel graph-based approach for table retrieval, named FusionMaestro. The authors first examine the limitations of previous work, noting that early fusion methods, which are query-independent, risk including irrelevant passages, while late fusion methods may only retrieve partial relevant information. Both fusion approaches also primarily rely on semantic matching, making it challenging to answer questions that require complex reasoning.

FusionMaestro aims to address these limitations by combining the strengths of early and late fusion methods. In this approach, early fusion is used to construct an initial graph that links tables to passages, with edges as the retrieval unit. A two-stage ranking mechanism then selects the most relevant edges to the query. During late fusion, the most relevant nodes are identified as seed nodes, and neighboring nodes are integrated into the graph. Finally, an LLM is used for graph refinement, performing column-wise aggregatiol and verifying passages to filter out irrelevant edges. FusionMaestro’s output is a set of ranked edges from the final retrieved graph. Experimental results demonstrate that this method significantly outperforms baselines on two benchmark datasets.

### Strengths
- S1: The paper is well-written, and the approach is clearly motivated, effectively combining the advantages of both early and late fusion methods.
- S2: Reformulating table retrieval as a graph retrieval task is a unique perspective. Using edges as the retrieval unit adds an interesting new approach to the task.
- S3: The authors conduct comprehensive experiments on two benchmark datasets, showing that the proposed method effectively retrieves relevant table information, thereby enhancing table QA performance.

### Weaknesses
- W1: The edge retriever and reranker are critical components, yet the authors do not evaluate their method against different ranking models (e.g., sparse ranking or state-of-the-art dense retrievers).
- W2: The authors note that removing the star-based LLM refinement increased AR@50 but attribute this to LLM hallucination without providing qualitative analysis. Including a representative example or conducting quantitative analysis (e.g., with off-the-shelf hallucination detection models) would strengthen the findings.
- W3: Table retrieval has been extensively explored in information retrieval, but some related work on graph-based methods for table retrieval [1,2] is missing from the references. 


[1] Retrieving Complex Tables with Multi-Granular Graph Representation Learning, SIGIR 2021

[2] MGNETS: Multi-Graph Neural Networks for Table Search, CIKM 2021

### Questions
See Weaknesses.

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
4

### Summary
This paper proposes FusionMaestro, a novel method to table-text retrieval in open-domain question answering. FusionMaestro leverages both early fusion and late fusion techniques, combined with large language model(LLM) reasoning, to address column-wise aggregation and multi-hop reasoning. Specifically, FusionMaestro utilizes a multi-granular retrieval pipeline, integrating edge-based bipartite subgraph retrieval, query-relevant node expansion, and star-based LLM refinement, resulting in significant performance improvements over state-of-the-art models. Additionally, the authors validate the effectiveness of their approach on the OTT-QA and MMQA datasets and conduct ablation studies to demonstrate the contribution of each component.

### Strengths
S1. FusionMaestro combines early and late fusion techniques while utilizing the reasoning capabilities of large language models to refine retrieval results,  effectively addressing the limitations of previous approaches, particularly in handling complex reasoning tasks, and improving overall performance.
S2. FusionMaestro constructs bipartite subgraph using edges as the retrieval unit and further expands these with query-relevant nodes, providing richer contextual information and enhancing both the accuracy and comprehensiveness of retrieval.
S3. The experiments in this paper cover multiple datasets and baselines, along with ablation studies, effectively validating the effectiveness of FusionMaestro.

### Weaknesses
W1. Although the proposed method achieves impressive performance, the combination of multiple components, finer-grained retrieval, and the use of large language models undoubtedly introduce significant computational overhead and complexity. However, the paper lacks an analysis of efficiency to address these concerns.
W2. Although the paper includes an ablation study, the evaluation section only provides AR@k and nDCG@50 scores. Reporting end-to-end performance would be more intuitive, as it would help illustrate the performance gains achieved in relation to the increased computational costs.
W3. Among the baselines compared with FusionMaestro, were any methods that utilized LLMs or LLM-like architectures? If so, for fairness, should the same LLM have been used across comparisons?

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2
