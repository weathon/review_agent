# DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Travel planning (TP) agent has recently worked as an emerging building block to interact with external tools/resources for travel itinerary generation, ensuring enjoyable user experience.
Despite its benefits, existing studies rely on hand-craft prompt and fixed agent workflow, hindering more flexible and autonomous TP agent.
This paper proposes DeepTravel, an end-to-end agentic reinforcement learning framework for building autonomous travel planning agent, capable of autonomously planning, executing tools, and reflecting on tool responses to explore, verify, and refine intermediate actions in multi-step reasoning.
To achieve this, we first construct a robust sandbox environment by caching transportation, accommodation and POI data, facilitating TP agent training without being constrained by real‑world APIs limitations (e.g., inconsistent outputs).
Moreover, we develop a hierarchical reward modeling system, where a trajectory‑level verifier first checks spatiotemporal feasibility and filters unsatisfied travel itinerary, and then the turn‑level verifier further validate itinerary's detail consistency with tool responses, enabling efficient and precise reward service.
Finally, we propose the reply-augmented reinforcement learning method that enables TP agent to periodically replay from a failures experience buffer, emerging notable agentic capacity.
We deploy trained TP agent on DiDi Enterprise Solutions App and conduct comprehensive online and offline evaluations, demonstrating that DeepTravel enables small-size LLMs (e.g., Qwen3-32B) to significantly outperform existing frontier LLMs such as OpenAI-o1/o3 and DeepSeek-R1 in travel planning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper DeepTravel presents an end-to-end agentic reinforcement learning framework for autonomous travel planning. It trains agents to plan, execute, and refine itineraries through a sandboxed environment, hierarchical reward modeling, and replay-augmented learning. Deployed on DiDi’s app, DeepTravel enables smaller LLMs to outperform larger models like OpenAI-o1/o3 and DeepSeek-R1 in travel planning tasks.

### Strengths
The paper’s main strength lies in its innovative integration of agentic reinforcement learning for autonomous travel planning, combining a robust sandbox environment, hierarchical reward modeling, and replay-augmented learning to enable effective tool interaction and reasoning. It demonstrates promising empirical performance, showing that smaller LLMs can surpass larger state-of-the-art models, and provides real-world validation through deployment on DiDi’s platform.

### Weaknesses
While the proposed sandbox environment effectively stabilizes training by mitigating API inconsistencies, it also raises concerns about closed-world limitations. By caching transportation, accommodation, and POI data, the framework may train agents on a static or outdated representation of the travel environment, potentially limiting generalization to real-world, dynamically changing conditions. Although the authors mention a daily update mechanism, the paper lacks detailed analysis on how frequently and extensively this data is refreshed, or how well the sandbox replicates real-world variability. Consequently, the practical usefulness and robustness of the trained agent outside this controlled setting remain uncertain, and further empirical validation with live, dynamic data sources would strengthen the paper’s claims.

The design of the trajectory-level and turn-level verifiers appears to be highly domain-specific and handcrafted, relying on travel-oriented rubrics and prompt templates. This raises concerns about generalizability, reliability, and reproducibility. Since these verifiers directly determine the reward signals, the overall agent performance may critically depend on their specific design choices, such as rubric formulation, prompt phrasing, or the underlying reward model’s calibration. However, the paper appears to provide limited ablation or sensitivity analysis to assess how changes in verifier design impact learning outcomes. Without clearer justification, it is difficult to evaluate whether the reported performance gains stem from genuine improvements in agentic reasoning or from carefully tuned, domain-dependent reward mechanisms, which could limit reproducibility and broader applicability.

The necessity and novelty of the proposed Replay-Augmented Reinforcement Learning (RARL) algorithm are not sufficiently justified. While the idea of replaying failed experiences is presented as a key contribution, it appears conceptually similar to existing experience replay or replay buffer mechanisms widely used in RL, including prior work like GRPO. The paper does not clearly explain what makes this replay-augmented approach fundamentally different or superior beyond applying it to the travel-planning domain.

Moreover, it remains unclear why existing RL algorithms could not achieve similar effects with appropriate tuning or data sampling strategies. Without a deeper theoretical motivation or comparative analysis isolating the benefits of the proposed replay mechanism, the contribution risks appearing incremental rather than novel, and its necessity in achieving the reported performance improvements remains questionable.

The comparison with state-of-the-art reasoning LLMs (e.g., OpenAI-o1/o3, DeepSeek-R1) may be unfair, as these models are not specifically trained or optimized for travel planning. Their weaker performance could reflect domain mismatch rather than genuine inferiority. Moreover, since evaluation relies on DeepTravel’s own reward verifiers and limited human checks, the setup may favor the proposed system, calling into question the fairness and validity of the comparisons.

All experiments and evaluations appear to rely heavily on DiDi’s proprietary platform, APIs, and datasets, which may not be publicly accessible. This dependence significantly limits reproducibility and independent verification of the reported results. While the authors provide training details and claim to release prompts, the potential absence of open-source data, sandbox implementation, and evaluation environment means that other researchers may find it difficult to replicate or validate the findings. As a result, the scientific transparency and reproducibility of the work ay be at the weak side.

### Questions
How does the cached sandbox data reflect real-world dynamics, and can the trained agent generalize beyond this controlled environment?

How sensitive is the agent’s performance to the design of the handcrafted verifiers, and how is their reliability and reproducibility validated?

What distinguishes the proposed replay-augmented RL from existing methods, and how are fairness and reproducibility ensured given reliance on DiDi’s proprietary setup?

Could the authors provide stronger evidence or ablation results showing that this new algorithm yields distinct and essential improvements compared to standard approaches?

### Soundness
2

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
This paper focuses on the practical business scenario of "travel planning" and designs an automated optimization algorithm based on Large Language Models (LLMs). The method is tested on different types of datasets and in a production environment, showing some improvements compared to open-source models. However, I do not believe this work is a good fit for this conference. It would be more appropriate for a conference with an applied track, such as KDD.

### Strengths
- This paper is easy to follow
- The algorithm design is simple and straightforward

### Weaknesses
- This work is too engineering-focused and lacks significant academic contribution.
- The learning curves presented in Figure 4 are weird; they show no clear evidence of policy learning because there is no significant reward improvement.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose an end-to-end agentic reinforcement learning framework for autonomous travel planning. It enables a LLM to plan itineraries by calling travel-related tools inside a sandboxed environment and refining its reasoning through RL rather than static prompts. 

* Self-expanding sandbox: built from cached real API data (DiDi flights, hotels, maps). It starts almost empty and grows as the agent explores new queries. Cached responses are replayed deterministically, while a daily refresh partially updates data to mimic real-world dynamics.

* Hierarchical reward modeling: A trajectory-level reward checks global consistency (route feasibility, timing), and a turn-level reward, implemented via DeepSeek-R1, ensures each step’s tool use and reasoning match tool outputs. If any turn fails, the entire trajectory is penalized.

* Training pipeline:
  1. Cold-start SFT: 1K verified trajectories distilled from DeepSeek-R1; fine-tune Qwen to learn the `<think>`/`<tool_call>` structure, masking `<tool_response>` tokens.
  2. RL phase (GRPO-style): roll-out groups of 8 trajectories per query in the sandbox; compute verifier rewards; update with GRPO loss and replay previously failed queries for continual improvement.

* Evaluated on 6 K + real user queries and synthetic benchmarks.
  DeepTravel-32B outperforms OpenAI-o1/o3 and DeepSeek-R1 while cutting hallucinations by > 50 %.

### Strengths
* Introduction of an Agentic RL framework for travel planning with real deployment.
* Sandbox design: stable, replayable environment that solves API inconsistency and rate-limit problems.
* Hierarchical reward: reward system combining high-level feasibility with low-level factual consistency.
* Practical impact: deployed inside a commercial platform (DiDi)
* Scalable idea: structure could extend to other multi-tool reasoning domains

### Weaknesses
* Reward quality relies on DeepSeek-R1 judgments; could introduce bias.
* No comparison to human or optimization-based planners
* Agentic behavior is constrained by predefined setting; not full open-world
* Evaluation metric is binary and may miss aspects like personalization or cost-tradeoffs

### Questions
* How consistent are DeepSeek-R1 verifier judgments vs human annotators?
* How are failed queries sampled when training?
* Can the agent handle unseen cities or updated APIs without re-training?

### Soundness
4

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
The paper introduces DeepTravel, an RL framework for autonomous travel planning. DeepTravel enables the agent to plan, execute external tools, and iteratively refine itineraries. The framework includes a sandbox environment built from cached transportation, accommodation, and POI data, a reward system with trajectory and turn-level verifiers, and a reply-augmentation method for learning from failed experiences. The proposed approach outperforms larger state-of-the-art reasoning models (used in zero-shot mode) on the target benchmark.

### Strengths
- The paper is easy to read and understand. The graphical illustrations are clear and informative.
- The presented approach significantly outperforms current state-of-the-art models in solution quality on travel planning tasks.

### Weaknesses
W1: The proposed approach is primarily engineering work rather than a conceptual innovation. The simulated API cache closely resembles existing agentic environments used for tool-based or web-interaction training, such as ReTool and WebSailor. The hierarchical verifier component functions as a rule-based reward layer, merely providing additional supervision signals instead of introducing a novel learning principle. Similarly, the replay augmentation strategy represents a simplified variant of well-known curriculum or prioritized replay mechanisms in reinforcement learning [3–6]. Consequently, the overall novelty of the paper is limited.

W2: Although the paper claims an end-to-end agentic RL framework, many components are manually supervised and tied to proprietary systems. The reward model uses human-engineered scoring rules, and the sandbox relies on internal APIs and unavailable data, making the framework neither fully end-to-end nor externally reproducible. Even though appendices are extensive, external researchers cannot replicate the environment or results.

W3: Evaluation metrics are internal and mostly rule-based. Conducting only 50 human evaluations online is too few for robust validation. Moreover, the online evaluation uses queries collected from the production environment rather than deploying the system to real users.

W4: The paper doesn’t transparently discuss how the system was adapted to the nine reasoning LLMs it was compared to. Did the authors attempt to improve them, for example by providing in-context data from the dataset used to train DeepTravel?

[1] Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021). Prioritized Level Replay. In Proceedings of the International Conference on Machine Learning (ICML), pp. 4940–4950. PMLR.

[2] Andrychowicz M, Wolski F, Ray A, Schneider J, Fong R, Welinder P, McGrew B, Tobin J, Pieter Abbeel O, Zaremba W. Hindsight experience replay. Advances in neural information processing systems. 2017;30.

[3] Shao Z, Wang P, Zhu Q, Xu R, Song J, Bi X, Zhang H, Zhang M, Li YK, Wu Y, Guo D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. 2024 Feb 5.

**Minor suggestions**:
- Make the descriptions of tables and plots more self-contained to improve clarity and readability.

### Questions
Q1: Does the proposed simulator/environment support multi-turn dialogue? For example, can the user see a response and add further requests, or does the system need to ask clarifying questions?

Q2: What is the token budget and inference cost for the baseline LLMs?

### Soundness
2

### Presentation
3

### Contribution
2
