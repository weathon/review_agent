## Human Reviewer 1

### Summary
This paper introduces Graph-R1, an RL-post-trained agentic GraphRAG framework. It operates on a constructed knowledge hypergraph and answers queries in a multi-turn fashion, i.e., it repeatedly performs a "think-retrieve-rethink-generate" loop within the knowledge hypergraph environment. Graph-R1 is trained using GRPO, with rewards based on format correctness and answer accuracy. Comprehensive empirical studies are conducted over multiple QA datasets, demonstrating the effectiveness of Graph-R1.

### Strengths
1. RL post-training with a multi-turn query and retrieval design makes sense.
2. The empirical studies are comprehensive.
3. The paper is well written and nicely illustrated.

### Weaknesses
1. The technical contributions seem limited: multi-turn action loop + GRPO + downstream task accuracy reward + format correctness reward are not particularly new. The authors could further clarify the core contributions.
2. The propositions appear decorative rather than informative.
3. I am curious about how the knowledge hypergraphs are built. It is not entirely clear to me from Sec. 2.1. Could the authors further clarify how this step is performed?
4. I am also interested in the generalization behavior of the trained policy. What would happen if the model were trained on all datasets: could it then achieve a universally effective (and maybe better) policy across all test sets?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes Graph-R1, a GraphRAG method that leverage LLMs as agent to interact with constructed knowledge hypergraph. The proposed method is trained via end-to-end reinforcement learning. Experiments on multiple dtasets demonstrate the proposed Graph-R1 outperforms bases while also achieve strong generalization ability.

### Strengths
1. The proposed Graph-R1 leverage LLMs as agent to interact with graph, which solve the issues of finxed retrieval process with only one-time interaction.

2. The experiments are conducted on multiple datasets and include recent baselines.

3. The authors also conduct detaield ablation studies and generalization comparision.

### Weaknesses
1. There is another work GraphRAG-R1 [1], which also use RL to solve the GraphRAG issues. The authors may compare the difference between these two methods.

2. What is the motivation to use Knowledge HyperGraph intead of KG?


[1] Yu, Chuanyue, et al. "GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning." arXiv preprint arXiv:2507.23581 (2025).

### Questions
1. Can the propsoed method be applied to KGQA, which contains the explicit KG.
2. Is the <think> steps necessary for RL?
3. Why does GRPO significantly outperform PPO?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This work applied RL with verified outcome reward to enhance the GraphRAG framework. Specifically, it construct a knowledge hypergraph from a given knowledge source and then let an LLM to perform multi-turn graph retrieval on the constructed hypergraph followed up by reasoning / thinking. These process is improved via GRPO with outcome-directed reward function.

### Strengths
- The paper is well writing.

- The paper did detailed empirical experiments across multiple datasets and also provide insightful case studies.

### Weaknesses
- The research novelty is quite limited. This work applies GRPO to GraphRAG scenario. The most techniques (GRPO, GraphRAG etc.) utilized in this work is already extensively studied. Basically this paper only proves that GRPO can work in the GraphRAG scenario. 

- The proposed OOD setting is not enough to evaluate the generalization ability of the finetuned LLMs. Many datasets such as 2wiki and hotpotqa are all wiki-based datasets. The paper need to evaluate on other domain of data to show the generalization ability.

### Questions
- Some claims are too strong. For example, the paper claims that existing GraphRAG all aim to gather sufficient knowledge in a single fixed retrieval. However, there are already some GraphRAG work such as HybGRAG [1] that leverages multi-retrieval strategy. The paper need to do a more comprehensive literature review.


[1] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](https://aclanthology.org/2025.acl-long.43/) (Lee et al., ACL 2025)

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper presents Graph-R1, an optimized agent for retrieval-augmented generation using GraphRAG. Graph-R1 first constructs a HyperGraph that serves as its backbone retrieval system, enabling structured knowledge retrieval. The agent then employs a multi-turn approach to answer queries by iteratively refining search queries and hypergraph retrieval operations (entity/hyperedge retrieval), allowing for progressive improvement in response quality. Graph-R1 is trained end-to-end using GRPO reinforcement learning, with rewards designed to encourage both correct answer generation and proper response formatting. Experimental evaluation on six datasets demonstrates Graph-R1's effectiveness, significantly outperforming competing GraphRAG approaches and search agents across metrics.

### Strengths
- S1) Graph-R1 demonstrates strong performance, outperforming existing GraphRAG approaches and search agents across multiple benchmarks.

- S2) Ablation studies show that combining enhanced knowledge representations (GraphRAG) with reinforcement learning yields greater performance gains than traditional chunk-based retrieval methods.

- S3) The paper provides comprehensive experimental analysis across multiple dimensions, such as performance, efficiency, response length, and turn count, demonstrating Graph-R1's advantages.

### Weaknesses
- W1) Graph-R1 appears to rely on existing approaches, Search-R1 (Jin et al., 2025) and HyperGraphRAG (Luo et al., 2025), i.e., on Search-R1's reinforcement learning framework and HyperGraphRAG's retrieval operations and graph construction. In addition, HyperGraphRAG seems to outperform Graph-R1 on some benchmarks when using GPT-4o-mini (Table 2), while Graph-R1 requires >15 steps to surpass HyperGraphRAG's performance (Figure 5b).

- W2) Graph-R1's reliance on entity-based retrieval (Section 2.2) may limit its generalizability to queries lacking explicit entities, such as "Summarize the trends in the last 5 financial reports."

### Questions
- Q1) Do Graph-R1 and Search-R1 use identical training data, or are you reporting results from the released Search-R1 version? Additionally, what is the number of training examples per method?

### Soundness
4

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4