# CARD: Towards Conditional Design of Multi-agent Topological Structures

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Large language model (LLM)-based multi-agent systems have shown strong capabilities in tasks such as code generation and collaborative reasoning. However, the effectiveness and robustness of these systems critically depend on their communication topology, which is often fixed or statically learned, ignoring real-world dynamics such as model upgrades, API (or tool) changes, or knowledge source variability. To address this limitation, we propose CARD (Conditional Agentic Graph Designer), a conditional graph-generation framework that instantiates AMACP, a protocol for adaptive multi-agent communication. CARD explicitly incorporates dynamic environmental signals into graph construction, enabling topology adaptation at both training and runtime. Through a conditional variational graph encoder and environment-aware optimization, CARD produces communication structures that are both effective and resilient to shifts in model capability or resource availability. Empirical results on HumanEval, MATH, and MMLU demonstrate that CARD consistently outperforms static and prompt-based baselines, achieving higher accuracy and robustness across diverse conditions. The source code is available at: https://github.com/Warma10032/CARD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CARD, a conditional multi-agent communication graph designer for LLM-based agent systems, along with AMACP, an explicit protocol for adaptive multi-agent communication. The core claim is that current multi-agent LLM systems typically assume a static or hand-crafted communication topology, which makes them brittle when the environment changes, for example upgraded language models, missing tools, degraded knowledge sources. CARD models topology as a directed communication graph and learns to generate that graph conditioned on two channels, static agent profiles, such as role, base model, available tools, and dynamic runtime conditions, such as API availability, token cost, search quality, model capability. The method uses a conditional variational-style graph encoder–decoder that embeds both channels, predicts an adjacency matrix of directed communication probabilities, and then thresholds it to get an executable communication protocol. Training optimizes a loss combining task utility, for example accuracy on HumanEval, MATH, MMLU, and a condition-aware communication cost term that reflects per edge token and model usage cost. At inference time, the system can update only the condition embeddings and re decode a new topology without retraining, allowing one shot adaptation when, say, the base LLM changes or a tool disappears. Experiments compare CARD to static topologies, to prompt only adaptations, and to recent automatically learned communication structures such as GPT-Swarm, Aflow, and G-Designer. On HumanEval, MATH, and MMLU, across multiple backbone LLMs, CARD delivers the highest average accuracy and is especially strong under out of domain conditions where model capacity or tool quality differs from training. The paper also analyzes how communication density, edge direction, and cost trade off as the model size, tool quality, or number of agents changes, and claims that CARD produces more robust, more cost efficient, and more easily scalable communication structures than existing approaches.

### Strengths
The paper identifies a real and increasingly important gap. Most existing multi-agent LLM orchestration work either wires up agents in a fixed pipeline or learns a static graph tuned for one configuration, but in practice these systems run over evolving models, tool outages, and changing retrieval quality. The authors formalize this as AMACP, which states that an agent communication topology should be effective at solving the task, cost efficient under current resource constraints, and adaptive to runtime condition changes. They then actually implement that requirement. CARD is not just a heuristic, it is a trainable conditional graph generator that takes as input a structured description of each agent’s role, tools, and model identity, plus a structured description of the current environment, and predicts who talks to whom and in what order. This is materially different from simple prompt engineering, where you append “the math agent is weaker today, rely on search more” to a system prompt, because here the topology itself, the adjacency, changes. The empirical section is broad for this area. They test on coding, on math reasoning, and on MMLU style multi task QA, using several families of LLMs, and they evaluate not just accuracy but also robustness, for example node failures and adversarial perturbations, scalability with agent count, and accuracy versus token cost. CARD tends to either match or beat the best baseline and often keeps a smaller drop when conditions shift, in other words, the performance is more stable than G-Designer and Aflow when the backbone model or the available tool is swapped. The work also takes steps toward interpretability. The paper visualizes learned topologies and shows concrete behavioral adjustments, for instance, weaker models produce denser graphs with heavier reliance on other agents, and weaker retrieval sources change which agent edges carry information into the reasoning path. That makes the claim of conditional adaptation more convincing because we can see structure changes, not just accuracy deltas. Finally, runtime adaptation without retraining is compelling. The method can re instantiate a communication graph under new conditions by just re encoding the condition channel and passing it through the trained decoder, so deployment can, in principle, respond in real time to model upgrades or API outages, and this is exactly the sort of operational knob teams building agentic systems are asking for right now.

### Weaknesses
The technical depth is somewhat uneven. The AMACP objective is framed as a joint optimization over task utility and a communication cost term, but the training procedure in practice seems to boil down to supervised or weakly supervised gradient descent on an aggregate loss that mixes accuracy and a differentiable expected edge cost computed from soft edge probabilities. The paper gives the loss but does not fully unpack how gradients are attributed to edges through multi round communication, or how credit is assigned to long chain interactions, especially given that the system output aggregates all agent responses after several rounds. Without that, it is difficult to judge whether CARD is actually learning a policy for dynamic topology design, or if it is just fitting correlations between condition embeddings and good handcrafted structures. The experimental methodology raises some concerns. The benchmarks are HumanEval, MATH, and MMLU, which are popular and recognizable, but the evaluation protocol for multi agent interaction is only described at a high level, and does not clearly state how prompts are constructed, how tool calls are executed, and how many tokens are allowed per agent per round. HumanEval in particular is typically judged with execution based pass rates, but here it is treated more like accuracy in a dialogue pipeline, that needs to be spelled out precisely. The fairness of comparisons needs more clarity. Some baselines, like debate and chain of thought, are single round or pairwise prompting schemes, others like GPT-Swarm and G-Designer are graph learners, but I did not see cost normalized comparisons where all methods are given the same token budget, number of rounds, or number of agent calls. CARD explicitly optimizes a cost regularizer, so it is important to show, at equal or lower total spend, that it matches or beats baselines, and not just at whatever spend it happens to choose. The robustness and adversarial tests are promising, but they are still driven by synthetic manipulations of single nodes or tools, and they do not explore stronger failure modes like correlated tool outages or cascading hallucinations across agents. The work would feel stronger if it included at least one ablation where multiple nodes are simultaneously unreliable, or where a high capacity model silently degrades in a way that resembles realistic model version drift. Finally, while the paper repeatedly emphasizes adaptation to new models or tools without retraining, all evaluations still assume that you can embed those new conditions in a way the encoders can interpret. It would be good to quantify how far out of distribution you can go before the condition encoder fails, for example, can a model family never seen in training still be placed meaningfully in the graph, or does performance collapse there.

### Questions
Can you describe in more detail how gradients flow through K round communication during training, in particular, when you compute task utility after aggregation, how do you propagate credit to specific edges predicted by the decoder, and how do you avoid vanishing credit assignment for early round communication hops. Can you provide token budget and cost accounting for every baseline and confirm that CARD is not just buying accuracy by spending more interactions. In Table 1 and Figure 8, you mention accuracy versus USD per instance, can you include a strict Pareto style comparison where for each baseline you show its best achievable accuracy at or below CARD’s reported cost. How does CARD behave with simultaneous condition shifts, for example, weaker LLM plus degraded search plus high token cost. The single node perturbation study in Section B suggests graceful degradation, but can you show the multi factor case. How sensitive is the method to the choice of anchor topology A, you mention chain and star as priors, do different anchors converge to the same learned communication patterns, or does A bias the final structure. How much of the gain comes from the condition encoder versus just training a stronger static topology learner, a useful ablation would be to freeze condition embeddings to random noise at inference and see how far performance drops. Finally, what is the exact output metric for HumanEval, is this passpercentage style unit tests, or is it some text judged correctness proxy.

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
This paper introduces a framework (called CARD), which dynamically designs communication topologies for LLM-based multi-agent systems. CARD uses environmental signals to adapt agent connections in real-time. Experiments on three benchmarks show that CARD outperforms static baselines.

### Strengths
1. This paper is well-organized. Most of the content is easy to understand.

2. The proposed CARD framework is interesting and reasonable, which can be applied to various multi-agent cooperation scenarios.

### Weaknesses
1. The experimental improvements over the second-best baseline are modest and may not be statistically significant.

2. It would be better to provide a case study to show the effectiveness of multi-agent communication.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **CARD (Conditional Agentic Graph Designer)**, a conditional graph-generation framework that dynamically designs LLM-based multi-agent communication topologies. It formalizes the **Adaptive Multi-Agent Communication Protocol (AMACP)** to jointly optimize for *effectiveness*, *cost-efficiency*, and *adaptiveness*. Through conditional variational encoding and environment-aware optimization, CARD enables real-time topology adjustment without retraining. Experiments on HumanEval, MATH, and MMLU show consistent improvements over static and prompt-based baselines.

### Strengths
* Technically innovative and conceptually unifying approach to dynamic topology learning.
* Strong theoretical grounding with explicit optimization objectives (Eqs.6,11,12).
* Robust empirical validation across multiple LLMs and benchmarks.
* Intuitive and interpretable visualization of adaptive graph behavior.
* Clear presentation and reproducible methodology with full prompts and configurations.

### Weaknesses
* **Hyperparameter Sensitivity:** Add performance–cost curves for different \$\beta\$ values to visualize robustness.
* **Empirical Validation of Assumptions:** Introduce experiments with fluctuating API or tool availability.
* **Baseline Coverage:** Include stronger recent baselines such as reinforcement-based topology learning methods.
* **Fixed Edge Threshold:** Consider adaptive or learnable \$\tau\$ for improved flexibility.
* **Scalability:** Extend evaluation beyond 10-agent systems to confirm convergence trends.

### Questions
* Does CARD support localized topology updates rather than full recomputation?
* Can the cost term \$w(G;C)\$ be generalized to other resource metrics such as latency or energy?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CARD, a framework that dynamically generates communication structures for multi-agent LLM systems by conditioning on real-time environmental signals like model capabilities and tool availability. It formalizes the AMACP protocol and demonstrates that CARD outperforms static topologies, achieving greater accuracy, robustness, and cost-efficiency on benchmarks like HumanEval, MATH, and MMLU, especially when adapting to unseen conditions.

### Strengths
Strengths:
1.Novel Formulation of Adaptive Topology: The paper moves beyond static or naively learned communication graphs, which is a significant limitation in existing multi-agent systems. The formalization of the AMACP protocol and the CARD framework provides a principled approach for dynamic, condition-aware topology generation.
2.Strong Empirical Validation: The paper provides comprehensive experiments across three major benchmarks (HumanEval, MATH, MMLU) and multiple LLMs. The results consistently show CARD outperforming strong baselines, particularly in "out-of-domain" settings and under simulated environmental changes, demonstrating superior generalization and robustness.
3.Practical Runtime Adaptation: CARD can update the communication topology at deployment time in response to changing conditions without retraining, which is a crucial feature for real-world applications.

### Weaknesses
Weaknesses:
1.Limited Agent-Level Adaptation: The paper adapts the communication topology but does not update agent-level configurations (e.g., individual agent prompts, internal reasoning steps, or tool-selection strategies) based on conditions. Jointly optimizing both topology and agent behaviors could lead to further performance gains.
2.Scalability and Complexity: While a scalability analysis is provided, the computational overhead of the encoder-decoder graph generation module for very large agent ensembles (e.g., 50+ agents) remains a potential concern and is not thoroughly explored.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
