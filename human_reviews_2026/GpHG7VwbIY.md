# PilotRAG: Teaching LLMs Multi-Turn Hybrid RAG via Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge, typically from unstructured texts or structured graphs. While recent progress has extended text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), existing graph-based and hybrid RAG methods generally rely on fixed or handcrafted multi-turn retrieval procedures rather than an RL-trained policy, and thus do not support adaptive, decision-based multi-turn reasoning. This limitation restricts their ability to incrementally integrate supplementary evidence as reasoning unfolds, thereby reducing their effectiveness on complex multi-hop questions. To address this limitation, we introduce PilotRAG, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG by dynamically interleaving reasoning, hybrid retrieval, and answer formulation. PilotRAG jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either unstructured texts or structured graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework with a reward function that accounts for both task outcome and retrieval efficiency. By rewarding answer accuracy and efficient retrieval while penalizing redundant retrieval operations, the model learns to retrieve selectively and reason effectively. Experiments on both simple and multi-hop question answering benchmarks demonstrate that PilotRAG significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL for enabling adaptive and iterative retrieval in complex reasoning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a RL method that enables LLMs to interleave reasoning and retrieval actions across both text and graph sources. The model uses a two-stage GRPO-based training procedure to balance answer accuracy and retrieval efficiency. Experiments on multiple QA benchmarks claim significant gains over previous RAG and graph-RAG systems.

### Strengths
1. The paper attempts to unify text and graph retrieval with reinforcement learning, producing an end-to-end framework that is well-articulated.

2. The model shows quantitative improvements over existing multi-turn RAG baselines such as Search-R1 and HippoRAG v2 across multiple benchmarks.

### Weaknesses
1. The main technical idea is a straightforward combination of existing retrieval types (text + graph) within a standard GRPO framework. There is no genuinely new retrieval or RL formulation; the method feels incremental and engineering-oriented.

2. The so-called “theoretical analysis” in the appendix is a bit trivial as it merely re-states GRPO variance reduction logic with algebraic reformatting, adding no formal insight or new guarantees.

3. Although efficiency is emphasized as a reward signal, the paper provides no real complexity or runtime comparison with baselines beyond qualitative plots. There is no measurement of wall-clock savings, GPU utilization, or scalability under large graphs.

4. The ablation only removes the efficiency reward; no experiment isolates hybrid retrieval, GRPO itself, or the contribution of multi-turn structure.

5. Some other baselines, such as R1-Searcher and SimpleDeepSearcher, are not compared to.

### Questions
1. What new insight does “hybrid RRF fusion” bring over existing ensemble retrieval methods like reciprocal rank fusion?

2. Can the authors demonstrate any generalization beyond the closed HotpotQA-like domains, or is this yet another narrow QA-specific RL setup?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper mainly extends graph-text hybrid RAG into multi-turn scenarios. It utilizes RL training by constructing performance rewards and efficiency rewards. It conducts experiments on several datasets.

### Strengths
The methods are clearly stated. Experiments validate the methods on multiple datasets.

### Weaknesses
1. I think the innovation of the paper's methods is limited. In Section 3.1, the overall model architecture is very similar to previous iterative RAGs, such as Search-R1, except that the retrieval methods are expanded to hybrid ones. However, these retrieval methods are also implemented in previous work, as shown in Section 3.1.2. Regarding RL optimization, Stage 1 simply measures the reward for task completion. Although Stage 2 introduces total search time as a loss, this does not align with the core motivation in the abstract, and does not provide targeted optimization for Graph-RAG. Therefore, I consider this paper's approach to be incremental.
2. I think Algorithm 1 needs to be optimized because it's unclear. While I can generally understand what it describes after reading the subsequent sections, some of the symbols used in this algorithm are not explained, making it a bit confusing.
3. Experiments were conducted only on the Qwen 2.5-3B model. Validation on 7B and higher models is strongly recommended.
4. In the provided anonymous Github repository, under "Step 1: Start Services," it is mentioned that Llama-3.1-8B-Instruct was used in the experiments, but this is not mentioned in the paper. Please provide more explanations.

### Questions
See weaknesses above. Please respond to these cons.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to improve current RL-based multi-turn RAG method with hybrid search. Specifically, each search action can be selected from passage retrieval, graph-based retrieval, or a hybrid of both. Then, the model is trained with GRPO with a 2-stage method. In the first stage, the model is trained only to achieve higher accuracy, and in the second stage, the model is also encouraged to be more efficient measured by time.

### Strengths
- Combining text-based and graph-based retrieval in the RL-based multi-turn RAG in principle should help the model achieve higher performance with more adaptive retrieval results
- Experiment results show the method achieves strong performance, especially the stage-2 training shows it can improve efficiency while not sacrificing performance
- Paper writing is clear

### Weaknesses
- The proposed method seems to achieve promising performance on multi-hop QA datasets but the performance on simple QA is clearly hurt with hybrid search (e.g., comparing search-r1 with PILOTRAG w/o efficiency training). Therefore, there should be more analysis on why hybrid strategy fails on simple QA and how to mitigate it.
- Some efficiency comparison could be beneficial to show the effects of stage 2 training other than just improving performance.

### Questions
- When testing the model, what is the ratio of the 3 search actions on different datasets?

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
4

### Summary
This paper tackles the limitation of one-shot retrieval in both traditional and graph RAG systems by fine-tuning a multi-step hybrid RAG policy LLM - PilotRAG. It features a two-stage RLVR pipeline by first performing outcome-oriented training then accuracy-efficiency training . By training a policy LLM, the model learns to interleave the three different retrieval actions and reasoning. After optimizing on hotpotQA, PilotRAG outperforms recent graph RAG and multi-RAG on same generator model (Qwen2.5) across multiple different datasets.

### Strengths
1. This paper proposes an effective and practical reward system to help policy model balance between the reasoning efficiency and tool-use in a hybrid RAG setting (both textual and relational information are presented).

### Weaknesses
1. Some of the technical details are not described in the paper, such as generalization of the graph construction algorithm (re-uses hipporag graph).

2. The paper is not aware of some important baselines that are published on the same topic: hyrbid text and graph rag. For example, HybGRAG[1] introduces hyrbid graph retrieval: choice between vector search and one-hop graph search with a pre-trained LLM (Sonnet3.5). These related work are not discussed, which also makes the claim in the abstract "graph-based RAG remains limited to one-shot retrieval" unappropriate. 

3. In case study, all of the Pilot-RAG examples selects the graph retrieval, neither text nor hybrid retrieval are selected. It seems the performance improvements are from the graph retrieval instead of the RL. 

[1] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases, ACL 25'

### Questions
1. Does Pilot-RAG generalize to different graphs such as thoese in CRAG [1] and hyperlink graphs? 

2. The training dynamics showing non-convergent #actions and #response length. Does this mean model is not converged or trained properly since no emerging reasoning is observed as other R1* work.  

3. Please refer to my other questions in weaknesses. 


[1] CRAG - Comprehensive RAG Benchmark

### Soundness
2

### Presentation
3

### Contribution
2
