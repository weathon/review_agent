# Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2

## Abstract
Retrieval-Augmented Generation (RAG) and Graph-based RAG has become the important paradigm for enhancing Large Language Models (LLMs) with external knowledge. 
However, existing approaches face a fundamental trade-off. While graph-based methods are inherently dependent on high-quality graph structures, they face significant practical constraints: manually constructed knowledge graphs are prohibitively expensive to scale, while automatically extracted graphs from corpora are limited by the performance of the underlying LLM extractors, especially when using smaller, local-deployed models.
This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER) mechanism to overcome these limitations. 
Our core innovation is the dynamic construction and refinement of a Chunk-Triplets-Community heterogeneous graph index, which pioneeringly incorporates a dual-evolution mechanism of Evolving Query and Evolving Sub-Graph for precise evidence retrieval.
This approach addresses a critical limitation of prior Graph-based RAG methods, which typically construct a static graph index in a single pass without adapting to the actual query. 
A multi-agent system, comprising Constructor, Retriever, Reflector, and Responser agents, collaboratively engages in an iterative process of evidence retrieval, answer generation, sufficiency reflection, and, crucially, evolving query and subgraph. This dual-evolving multi-agent system allows ToG-3 to adaptively build a targeted graph index during reasoning, mitigating the inherent drawbacks of static, one-time graph construction and enabling deep, precise reasoning even with lightweight LLMs.
Extensive experiments demonstrate that ToG-3 outperforming compared baselines on both deep and broad reasoning benchmarks，and ablation studies confirm the efficacy of the components of MACER framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Think-on-Graph 3.0, a framework for RAG on heterogeneous graphs. The core is a Multi-Agent Coupled Evolutionary Retrieval–Generation loop (MACER), where during answering, the query is iteratively refined by a Reflector, while the evidence subgraph is simultaneously expanded or pruned by a Constructor until a Reflector determines that “sufficient evidence” has been reached. Offline, ToG-3 builds a heterogeneous graph with Chunk–Triplet–Community node types. Online, four agents collaborate in a closed loop that formalizes the process as a Markov Decision Process, with a binary sufficiency reward controlling the stopping condition. Experiments show that ToG-3 achieves the best average results.

### Strengths
S1. Combining dual evolution (query + subgraph) with a multi-agent loop is an elegant and effective fix to the brittleness of static GraphRAG systems under noisy LLM extraction formalized via MDP.

S2. The Chunk–Triplet–Community graph unifies multi-granular retrieval in a single embedding space, bridging fine-grained evidence and coarse community reasoning.

S3. ToG-3 achieves top or near-top EM/F1 on HotpotQA, 2WikiMultihopQA, and MuSiQue, and demonstrates domain generalization through ELO win-rates across four UltraDomain subsets.

### Weaknesses
W1. The binary reward depends on whether Suff(q, G_k, a_k)=1, but the paper does not explain the exact implementation, thresholding, or its correlation with EM/F1.

W2. The paper should compare with 1-2 more multi-agent baselines released in recent years, e.g. HM-RAG (with the single modality setting) [1] or Graph Counselor [2].

W3. The authors should polish the writing. For example, Section 3.2.1 contains many unnecessary sparse lines and overly detailed yet non-essential descriptions. It could be written more concisely as a single continuous paragraph instead of being broken into multiple bullet points.

[1] HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation

[2] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning

### Questions
Q1. How is max iteration K chosen? What is the accuracy–latency tradeoff under fixed inference budgets?

Q2. What happens if only evolving-query or evolving-subgraph is introduced into GraphRAG/LightRAG? Or conversely, ToG-3 is reduced to a static graph?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Think-on-Graph 3.0, an graph-based RAG framework over heterogeneous graphs.

The core innovation is the multi-agent context evolution and retrieval mechanism, called MACER in the draft. It employs a set of agents to iteratively refine the query and a chunk-triplet-community subgraph during reasoning. The dynamic nature of this framework addresses the limitation of previous graphRAG algorithms that rely on a static graph. And the approach is formalized as a markov decision process with a dual-evolution loop for adaptive evidence retrieval.

### Strengths
1. The paper tackles a practical challenge in graph-based RAG: the trade-off between graph quality and scalability, especially for open-source, lightweight LLMs in offline or private deployments. 

2. ToG-3 mitigates issues like incomplete triplet extraction, insufficient details by dynamic, query-adaptive refinement on a dynamically updated graph.

### Weaknesses
1. The presentation is poor; instead of adopting a style of professional academic writing, it resembles a course project report. To be honest, most of the content is inconsistent, poorly organized, and reads like AI-generated text.
2. The overall design is purely engineering-oriented, achieved by stacking a set of agents. Due to the lack of preliminary studies and theoretical analysis, the insights behind the design are unclear.
3. The agents are applied iteratively; thus, there is no global planning to control the refinement direction. The refinement of the graph has no supervision information and may easily lead to collapsed results rather than good ones, especially when the input corpora become large-scale or heterogeneous.
4. No efficiency analysis is given to demonstrate its complexity (time and memory).

---
Below are weaknesses on experiments.

5. The baselines are limited. Only HippoRAG-2 and four GraphRAG methods with poor performance (worse than NaiveRAG) are included. Stronger baselines are needed, e.g., GFM[1], RAPTOR[2], KGP[3]
6. The empirical accuracy improvements are marginal across the three datasets. The "Average" column in Table 1 is unnecessary and redundant.
7. Table 2 only compares the indexing and retrieval times, and no token consumption is given. Based on my understanding, a multi-agent system should incur significant token costs.
8. Table 2 does not compare with other baselines (HippoRAG-2, ToG-2). Based on my experience, the indexing and inference times of GraphRAG and LightRAG are quite high, almost the highest in the GraphRAG family. Comparing with these low-efficiency methods undermines the convincingness of your work's efficiency claims. More efficient GraphRAG methods should be compared, such as HippoRAG-2, RAPTOR[2], E^2GraphRAG[4].





- [1] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
- [2] RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- [3] Knowledge Graph Prompting for Multi-Document Question Answering
- [4] E^2GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness

### Questions
plz see above

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Think-on-Graph 3.0 (ToG-3), a novel framework that addresses key limitations in graph-based Retrieval-Augmented Generation (RAG) systems by proposing a Multi-Agent Context Evolution and Retrieval (MACER) mechanism. The approach dynamically constructs and refines a Chunk-Triplets-Community heterogeneous graph index through a dual-evolution process involving evolving queries and evolving sub-graphs. A multi-agent system, comprising Constructor, Retriever, Reflector, and Responser agents, collaboratively performs iterative evidence retrieval, answer generation, and context refinement.

### Strengths
1. The proposed MACER mechanism, together with its dual-evolution process of queries and sub-graphs at its core, has a very intuitive design that is well-motivated.

2. Extensive experiments on a comprehensive suite of benchmarks, including both deep multi-hop and broad reasoning tasks, demonstrate the effectiveness and superior performance of ToG-3. 

3. Detailed ablation studies are also included to further validate the contribution of each core component.

### Weaknesses
1. Marginal Performance Gains: Despite the complex architecture that involves multi-agent collaboration, dual-evolving mechanisms, and heterogeneous graph construction, ToG 3.0 only achieves marginal improvements compared to much simpler baselines. According to Table 1, it has only a minor edge in performance over the more lightweight HippoRAG-2. On specific cases, such as Musique under the EM metric, it is even surpassed by the simplest baseline, NaiveRAG. This raises questions about the practical cost-benefit trade-off of such a complex system.

2. The key baselines are missing. The experimental evaluation does not compare with various important and recent state-of-the-art Graph RAG methods [1-6]. As a result, without considering these relevant baselines, it is hard to verify the claimed superiority of ToG-3.

3. The efficiency analysis is incomplete. The authors have mainly compared ToG-3 against GraphRAG and LightRAG, which are known to be computationally heavy. HippoRAG-2 and other more recent, efficient baselines [1-6] are omitted in the efficiency comparison. This omission weakens the practical efficiency and deployment advantages of ToG-3. A wider comparison is required to solidify its positioning in regard to efficiency.

4. The detailed analysis of the RL-based training is missing. The online MACER process is formulated as a Markov Decision Process, with its dependence on reinforcement learning principles. This introduces significant complexity regarding training stability, reward function and policy hyperparameter tuning, and overall reproducibility. Further discussion on the engineering challenges, convergence guarantees in practice, or sensitivity of the performance to the design of the reward signal is lacking in the paper.

5. While the proposed framework is built upon a graph structure, the paper does not engage or benchmark against established graph algorithms for subgraph retrieval or graph refinement. This omission makes it difficult to assess whether the performance gains come from the novel multi-agent, dual-evolving mechanism or could be partially achieved by applying more specialized graph-theoretic methods to the same heterogeneous graph index. Such a comparison would enhance the contribution and professionalism of the paper.

6. The work presents a multi-agent system but fails to make a compelling case that this is a necessary architecture. The different roles of the agents seem to be a modular decomposition of what could well be a monolithic reasoning process, named Constructor, Retriever, Reflector, and Responser. An ablation study, removing the multi-agent system and using a single, well-prompted LLM in its place, executing the same iterative process, would be far more convincing.

7. Each step in the framework requires multiple calls to the LLM for retrieval, reflection, query evolution, and subgraph evolution. It would be beneficial for the paper to provide an in-depth analysis of how this scales with graph size and query complexity and discuss practical trade-offs for how to balance the achieved performance gains with such a significant increase in computational requirements over simpler, single-shot retrieval methods.


[1] E²GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness.

[2] GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning.

[3] Graph-R1: Towards agentic graphrag framework via end-to-end reinforcement learning.

[4] KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG.

[5] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation.

[6] Align-GRAG: Reasoning-Guided Dual Alignment for Graph Retrieval-Augmented Generation.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents ToG 3.0, as a multi-agent RAG framework mainly focusing on a dynamic heterogeneous graph (chunk-triple-community) and a dual-evolving retrieval mechanism for both query and graph. While the idea of adaptive graph refinement is interesting, the work suffers from critical flaws in novelty, experimental validity, and practicality that undermine its contributions.

### Strengths
- The paper clearly identifies limitations in static GraphRAG methods, especially under resource-constrained settings with lightweight LLMs, among them, I deeply agree with the extraction performance when LLMs cannot follow the instructions with proper format.
- The integration of multi-agent collaboration is designed properly, though it's not new, with iterative graph refinement attempts to address query-dependent reasoning.

### Weaknesses
- The biggest problem is found in Table 1, the EM scores surprisingly exceed F1 scores (e.g., HotpotQA: EM=0.520 vs. F1=0.312). This inversion contradicts typical QA evaluations, where F1 most usually surpasses EM, which is also widely reported in the baselines. It suggests obvious flaws in answer extraction, metric computation, or dataset alignment.
- Bad performance. The proposed method could hardly outperform the baselines, especially when there are concerns about the token costs and the efficiency problem.
- Limited Novelty. The "Chunk-Triple-Community" graph schema is not fundamentally novel. Similar multi-layer graph structures (e.g., E2GraphRAG's hierarchy, RAPTOR’s recursive summarization, Youtu-GraphRAG's knowledge tree, and G-Reasoner's Quato Graph) have been explored extensively. The paper fails to delineate clear advancements.
- The dual-evolving mechanism resembles iterative retrieval-generation paradigms like Self-RAG, which makes sense but not that novel. This design also brings concerns that:
    - Costs. Token costs will increase incredibly when including reasoning, reflection and graph refinement.
    - Latency and efficiency. similar reasons as above.
- MDP formulation is not well justified. It is more like a story.

### Questions
- Why does EM significantly outperform F1? Please provide proper error analysis and clarify it.
- Does the MDP converge reliably with lightweight LLMs?

### Soundness
1

### Presentation
3

### Contribution
1
