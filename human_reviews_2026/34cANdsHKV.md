# Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
With an ever-growing zoo of LLMs and benchmarks, the need to orchestrate multiple models for improved task performance has never been more pressing. While frameworks like Mixture-of-Agents (MoA) attempt to coordinate LLMs, they often fall short in terms of (1) selecting relevant agents, (2) facilitating effective intra-agent communication, and (3) integrating responses efficiently. In this work, we propose Graph-of-Agents (GoA), a new graph-based framework for modeling multi-agent LLM communication. Our approach begins with node sampling, selecting only the most relevant agents by leveraging model cards that summarize each model’s domain, task specialization, and other characteristics. Next, we construct edges between the selected agents by evaluating their responses against one another to determine relevance ordering. Directed message passing is then performed from highly relevant agents to less relevant ones to enhance their responses, followed by reverse message passing to refine the original responses of the more relevant agents. Finally, the updated responses are aggregated via graph-based pooling (e.g., max or mean pooling) to produce a single, unified answer. We evaluate GoA on diverse multi-domain benchmarks (MMLU, MMLU-Pro, GPQA) and domain-specific benchmarks (MATH, HumanEval, MedMCQA), with an agent pool of 6 LLMs spanning multiple domains. Surprisingly, GoA achieves superior performance18 using only 3 selected agents, outperforming recent multi-agent LLM baselines that utilize all 6 agents simultaneously. By adopting a graph structure, GoA offers both scalability and effectiveness through structured message passing—positioning it as a strong candidate for navigating the challenges of the ever-growing LLM zoo.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Graph-of-Agents (GoA), a graph-based framework for orchestrating multi-agent collaboration among Large Language Models (LLMs). GoA addresses three key limitations of existing approaches like Mixture-of-Agents (MoA): (1) inefficient agent selection, (2) unstructured communication, and (3) costly response integration. It introduces a pipeline of node sampling, edge sampling, bidirectional message passing, and graph pooling (max/mean) to produce a final answer. Evaluated on diverse benchmarks (MMLU, GPQA, MATH, etc.), GoA with only 3 agents outperforms MoA and other baselines using 6 agents, demonstrating superior efficiency and effectiveness.

### Strengths
1) GoA creatively reframes multi-agent LLM collaboration as a dynamic graph construction problem, combining agent metadata, relevance-based edge formation, and directional message passing. This is a synthesis of graph neural network concepts with test-time LLM orchestration, distinct from static ensembles or debate-based methods.
2) The methodology is well-structured and technically sound. The ablation study (Table 5) convincingly validates each component (e.g., bidirectional message passing, relevance scoring). Experiments span 6 diverse benchmarks and compare against 6 strong multi-agent baselines, with clear metrics (accuracy, token usage, latency).
3) The paper is clearly written and well-organized.

### Weaknesses
1) The proposed multi-agent collaborative framework is evaluated with only a small number of agents (e.g., 3 in Table 1), which fails to demonstrate the scalability or advantages of the method in more complex, large-scale multi-agent settings. Moreover, The ablation shows k=5 slightly underperforms k=3. Is this due to increased noise or insufficient message-passing depth? Would adding more rounds of message passing help with larger agent sets?
2) The paper lacks a clear comparison with workflow-based multi-agent systems or auto-workflow generation [1]. It remains unclear whether the proposed approach—selecting agents based on task–agent relevance—outperforms hand-crafted workflows, and the selection mechanism itself shows limited novelty.
3) Key graph-based multi-agent baselines, such as MacNet (Qian et al., 2024) and GPTSwarm (Zhuge et al., 2024), are not included in the experiments, weakening the empirical evaluation.
4) The method is not tested on established complex multi-agent benchmarks that require coordination, such as WebAgent, GUI navigation, or WebShop tasks, raising concerns about its generalizability and practical utility.
5) The results show that their model with 3 agents outperforms the baselines with 6 agents. However, the agents set for selection is larger than 3. Moreover, the performance of GoA_max and GoA_mean is unstable for various tasks. Particularly, both GoA_max and GoA_mean perform similarly with the baselines. The improvement of the proposed method is very limited.
6) The manuscript contains grammatical errors (e.g., “... and graph pooling,GoA supports efficient and ...”), which affect readability and professionalism.

[1] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge,
Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic workflow generation. 2025.

### Questions
1) In the main experiment (Table 1), the authors use 3 agents—how many candidate agents were available for selection? How do baseline methods perform when restricted to the same 3 agents? What is GoA’s performance with 6 agents? Which base model is used in Table 1? The comparison appears unfair unless all methods use the same number and type of agents.
2) The method constructs a graph based on agents’ initial responses to determine execution order, then re-executes the task using this order. Does this two-stage process significantly increase computational or implementation complexity? Can the constructed graph capture multi-layer or hierarchical dependencies among agents?
3) Intuitively, the benefit of GoA should grow with more agents, yet Table 3 shows limited gains when increasing agent count. How do other models in Table 1 (e.g., Refine) perform under the same GPT-4o setting used in Table 3? Given that Refine already achieves strong results in Table 1, is GoA’s improvement marginal?
4) You cite router-based ensembles (e.g., Wang et al. 2023c, Hari & Thomson 2023) but don’t compare against them. How does GoA’s node sampling + graph communication compare to learned or prompt-based routing in terms of accuracy and cost?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel method for coordinating multiple LLMs using a graph structure. Each LLM is treated as a node, and directed edges represent relevance-based communication, enabling structured message passing between more and less relevant LLMs. Across multi-domain and specialized benchmarks, GoA outperforms prior multi-agent systems (like Mixture-of-Agents) while using fewer models, offering both scalability and efficiency in test-time collaboration among LLMs.

### Strengths
- The writing flow is very clear, and the motivation for developing this framework is strong and convincing. The node sampling and edge sampling are relatively new methods to me and seems interesting to promote communication efficiency.

- I pretty like the motivation behind the source-target information passing design, which is said to be able to avoid noise while enhance the model’s original answer through aggregate other’s opinion.

- The benchmarks tested are comprehensive, even though the ablation studies can be more comprehensive and insightful. The comparison with many other related methods are also very comprehensive.

### Weaknesses
- The agent here I am afraid is kind of abused. The different nodes from my understanding is just different LLMs. For agent, their core trait is task-oriented and environment interaction, which I think current node (it does not interact with environment using tools) does not entail. The paper may clarify what they view as agent if claim this paper as graph-of-agent.

- I think more ablation studies should be done on the method design motivation itself, like what if we reverse the source-target message passing sequence? Also it can be more insightful by discussing why the results are like this, such as top-k=3 is better than 2 and 5, instead of just describing the numbers which are obvious from the table.

### Questions
- I am wondering when each node is generating the response, are they aware that their response will be routed to others? This awareness may cause difference in model’s behavior I believe.

- Will the domain specific questions always drive GoA to choose the expert model of that certain domain? Why do you think GoA can be better than using the expert model alone, empirically and theoretically?

- I am wondering how you make the MoA and GoA comparable, as prompt input should be quite different, and in pure prompting framework, I think the performance is very sensitive to the instruction inputs. Therefore, how you ensure the scale up / efficiency experiment results are comparable?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Graph-of-Agents (GoA), a graph-based framework designed to improve multi-agent communication among LLMs. GoA addresses the challenges particularly in Mixture-of-Agents (MoA), by providing solutions for selection of agents and intra-agent communications. The proposed framework with 3 agents outperforms other multi-agent baselines with 6 agents on various benchmarks.

### Strengths
1. **Clear ideation:** The paper clearly identifies the limitations of Mixture-of-Agents (MoA), and motivates the need for a selection approach for agents and agent communications.
2. **Extensive experiments:** The authors conduct comprehensive experiments across various multi-domain and domain-specific benchmarks, demonstrating the effectiveness of the GoA framework.
3. **Clear ablations:** The ablation study is well designed for evaluating the impact of different components of the proposed framework.

### Weaknesses
1. **Limited discussions about prior work:** The related work section is brief and lacks depth in discussing prior works on dynamic multi-agent collaboration with LLMs. [1,2,3] show different levels of dynamic agent selection and communication strategies that could undermine the novelty of the proposed method.
2. **More agents may not bring better performance:** The authors claim that GoA with 3 agents outperforms baselines with 6 and more agents. But [1] suggests that increasing the number of agents may not always lead to better performance, especially in self-organized agent systems. Fairer comparisons with different numbers of agents are necessary.
3. **Justification and presentation issues:** The content is redundant by repeating node sampling, edge sampling, and message passing in multiple sections, without enough justification of method design choices. Minor presentation issues exist such as tcboxes should not be labeled as tables and etc.

[1] Liu et al. A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration. COLM 2024.  
[2] Zhuge et al. Language Agents as Optimizable Graphs. ICML 2024.  
[3] Yue et al. MasRouter: Learning to Route LLMs for Multi-Agent Systems. ACL 2025.

### Questions
Could you provide failure cases where GoA does not perform well or even fails to MoA or other baselines? Understanding the limitations of your approach would help in assessing its overall effectiveness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the shortcomings of current multi-agent LLM collaboration frameworks (e.g., Mixture-of-Agents, MoA) in three key aspects: agent selection, effective communication, and response integration, proposing a graph-structured framework named Graph-of-Agents (GoA). The core idea is to model LLM agents as nodes and the inter-agent relevance as directed edges. The paper’s main contributions include: Identifying three core challenges in current multi-agent LLM systems, thus providing clear research directions; Introducing a graph-based formulation of multi-agent collaboration that integrates four essential modules (node sampling, edge sampling, message passing, and graph pooling), and theoretically showing that GoA generalizes MoA, providing conceptual unification; Demonstrating that GoA achieves better performance and efficiency with fewer agents across diverse benchmarks and both open-source and proprietary models (e.g., GPT-4o).

### Strengths
1. The paper proposes a graph-structured multi-agent collaboration framework, integrating node sampling, edge construction, bidirectional message passing, and graph pooling. This design is a meaningful extension of existing multi-agent methods (e.g., MoA) and introduces the idea of modeling multi-agent interaction as a dynamic graph, enabling task-adaptive communication paths—an innovative and elegant formulation.
2. The method is prompt-based and requires no additional training, making it compatible with black-box APIs and thus generalizable to different LLM setups.

### Weaknesses
1. Although GoA improves upon MoA, bidirectional message passing still requires multiple LLM calls, which may cause latency issues in real-time or large-scale agent pools. The paper lacks a systematic comparison of token usage and inference time.
2. The paper does not provide quantitative ablation on the robustness of the node sampling strategy. Although Table 4 includes ablations on the number of agents and message-passing variants, it does not explicitly examine robustness when model card information is incomplete or noisy.
3. While “Proposition 1” claims that GoA is a generalization of MoA, this statement is largely informal, without rigorous theoretical proof. There are no formal results on the convergence, generalization, or optimality of the message-passing mechanism.

### Questions
1. Have the authors experimented with larger or more diverse agent pools (e.g., 10, or synthetic agent ensembles)? If so, did they observe bottlenecks in latency or accuracy?
2. Could GoA be extended to multimodal or non-text agents? Is it possible to generalize this framework to multimodal tasks?

### Soundness
2

### Presentation
2

### Contribution
2
