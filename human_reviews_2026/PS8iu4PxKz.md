# MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MobileIPL, a novel framework to enhance the reasoning abilities of VLM-based mobile agents for tasks involving mobile GUIs. Addressing limitations in existing approaches, such as overfitting from supervised fine-tuning and reliance on costly manual step-level reward annotation, the authors propose Iterative Preference Learning, which constructs diverse reasoning trajectories through Monte Carlo Tree Search and uses rule-based scoring for intermediate thoughts instead of process reward models. Feedback from scored leaf nodes is backpropagated to earlier reasoning steps, forming thinking-level preference pairs for Direct Preference Optimization to improve both final action selection and the reasoning process itself. To boost generalization and prevent overfitting, a three-stage instruction evolution strategy generates varied Q&A pairs grounded in real UI screenshots, enhancing agents’ contextual understanding.

### Strengths
- The paper introduces a creative method (MobileIPL) for improving reasoning in mobile GUI agents, leveraging an iterative sampling and learning process that avoids expensive and hard-to-scale process-level manual annotation. By combining Monte Carlo Tree Search (MCTS) with rule-based rewards and backward value propagation, the paper sidesteps unstable process reward models and manual step annotation, streamlining the learning process.
- Unlike many prior works that optimize only for final task reward/accuracy, this method explicitly scores and optimizes at the level of intermediate "thinking" steps, fostering better structured and generalizable agent reasoning.
- The approach is thoroughly evaluated on three widely recognized benchmarks, using both in-domain and out-of-domain tasks, and consistently outperforms several strong recent baselines and continual pretraining models. The analysis demonstrates the benefits of each introduced component (e.g., iterative rounds, instruction evolution), strengthening the validity of the methodological choices.

### Weaknesses
- The method entirely relies on rules (format, type, spatial distance, and F1 for input) to quantify step-level reward. These may not capture nuanced or partially correct reasoning paths, edge-case actions, or subtle mistakes that a learned process reward model or human judgements might identify. Is the rule-based approach robust to complex/ambiguous UI layouts or fuzzy matching of instructions? Even with more granular step-wise signals, rule-based reward systems are susceptible to reward hacking (the model may exploit rules to maximize reward without genuinely improving reasoning). 
- The iterative sampling and tree construction (MCTS-like CoaT-tree) can be computationally expensive, scaling poorly with both the number of sampled trajectories (K) and the depth of reasoning. Can the authors report the runtime and resource requirements, especially compared to simple SFT or even RL-based pipelines?
- The process involves choices of sampling number K, discount factor c, DPO data filtering thresholds, etc. There is insufficient discussion of robustness to these parameters, or general guidance about their selection for new domains.
- Although the instruction evolution is partly filtered by humans, it remains heavily reliant on LLM-generated QA pairs. Are these synthetic instructions always realistic and helpful? Does the method generalize if downstream tasks differ greatly from the synthetic questions generated?

### Questions
See Weaknesses.

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
3

### Summary
#### Main Content Summary

* **Problem Statement**: VLM-based mobile agents rely on the **Chain of Action-Planning Thoughts (CoaT)** paradigm, but the **scarcity of diverse CoaT trajectories** severely restricts their generalization ability. Existing self-training solutions either overlook the correctness of **intermediate reasoning steps** or require **expensive process-level annotations** to construct Process Reward Models (PRMs).
* **Proposed Solution (MobileIPL)**: The paper introduces **MobileIPL (Iterative Preference Learning)**. This method is designed to enhance the agent's thinking process by **iteratively learning preferences** to distinguish between high-quality and low-quality reasoning paths. This mechanism effectively optimizes the correctness of intermediate thoughts **without the high cost of manual process-level labeling**.
* **Observed Failure Modes**: Analysis of negative samples reveals consistent reasoning errors, including **Rough page description**, **Hallucinated Thought** (e.g., believing the agent is on the Play Store when it is on the Home page), and **Fabricated Position and Elements** (e.g., generating an action for a non-existent element).

### Strengths
#### Strengths (1-2 points)

* **1. Cost-Effective and Scalable Process Improvement**
    * **Detail**: MobileIPL successfully addresses the bottleneck of requiring **expensive process-level annotations** for building Process Reward Models (PRMs). By utilizing an Iterative Preference Learning approach, the method provides a scalable and cost-efficient mechanism to automatically guide and improve the quality of the agent's intermediate reasoning steps.

* **2. Direct Focus on Intermediate Reasoning Quality**
    * **Detail**: Unlike many self-training strategies that only optimize for the final task success rate, MobileIPL's core mechanism explicitly targets the **correctness, detail, and diversity of the "thoughts" (intermediate reasoning steps)**. This helps the agent build a more robust and trustworthy foundation for planning and execution.

### Weaknesses
#### Weaknesses (3-4 points)

* **1. Failure to Address Root Causes of Hallucination**
    * **Detail**: The paper highlights severe errors like **Hallucinated Thought** and **Fabricated Elements**. While MobileIPL attempts to suppress these errors via preference learning, it **does not fundamentally solve** the VLM's underlying limitations in **visual grounding** and accurate internal state tracking, leaving the model susceptible to imagining non-existent states or elements.

* **2. High Domain Specificity and Limited Generality**
    * **Detail**: The proposed method is highly specialized for **VLM-based mobile agents** operating within the structured environment of **GUI tasks**. The direct applicability of MobileIPL's preference learning strategy may be restricted outside of this specific domain, such as in continuous control, physical robotics, or less-structured web navigation.

* **3. Risk of Error Accumulation in Iterative Training**
    * **Detail**: As an iterative self-training technique, the approach carries an inherent risk of **compounding errors** or training instability. If the preference model in earlier iterations inaccurately favors flawed reasoning patterns, these errors could be amplified throughout the process, leading the final agent to converge on a suboptimal or unreliable policy.

* **4. Constraint by the CoaT Architectural Paradigm**
    * **Detail**: The entire framework is built upon enhancing the existing **Chain of Action-Planning Thoughts (CoaT)** structure. This reliance prevents the model from exploring or learning potentially more efficient **non-sequential, parallel, or highly flexible** reasoning processes that might be required for solving complex, multi-faceted tasks more optimally than a strict linear chain allows.

### Questions
### Open Research Questions for MobileIPL

* **1. Generalization Beyond GUI and Abstract Action Spaces**
    * **Question**: How can the Iterative Preference Learning (IPL) framework be successfully extended and validated in **more complex, less-structured embodied environments**, such such as physical robotics or 3D navigation, where actions are continuous and visual states are highly dynamic and less deterministic than in a GUI?

* **2. Mitigating the Root Cause of Hallucination in the Thought Process**
    * **Question**: Can MobileIPL's approach be integrated with **explicit visual grounding mechanisms** or **internal state tracking models** to not just penalize (via preference learning) but fundamentally **prevent** the generation of Hallucinated Thoughts and Fabricated Elements?

* **3. Optimizing the Preference Learning Signal for True Causality**
    * **Question**: How can the preference signal be refined to better distinguish between a **causally correct** but less detailed thought sequence and a **highly detailed** but ultimately incorrect or inefficient one, especially in cases where a flawed process yields a correct final result (the rare case mentioned in the paper)?

### Soundness
2

### Presentation
2

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
The paper proposes MobileIPL, a self-training framework for vision-language mobile GUI agents that enhances reasoning via Iterative Preference Learning. It introduces a CoaT-tree built through Monte Carlo sampling, uses rule-based rewards with backward credit assignment to form thinking-level DPO pairs, and mitigates overfitting via a three-stage instruction evolution strategy. Experiments on AITZ, AMEX, and AndroidControl show strong performance and out-of-domain generalization.

### Strengths
- The CoaT-tree construction via iterative sampling enables fine-grained reasoning optimization without manual step annotations and seems novel.

- Experimental results on AITZ, AMEX, and AndroidControl indicates that MobileIPL outperforms previous strong baselines with less data usage.

### Weaknesses
- A few recent related works are not well-discussed. For example, TCPO [1] proposes thought-centric preference optimization, which is similar to the Thinking-level DPO proposed in this work. TreePO [2], TreeRL [3], SPO [4] all introduces tree-structure rollout and value backpropagation, which is similar to the iterative sampling process of MobileIPL.

-  The clarity is not clear. For example, (1) Section 3.3 is not a complete part. It introduces Iterative Preference Learning. However, this section ends with the construction of $\\beta _ \\mathrm{pairs}$ and $\\gamma _ \\mathrm{pairs}$, without the preference optimization process. Although the DPO process is mentioned in Algorithm 1, it is not clear enough for clarity. Also, it is not clearly demonstrated what $\\alpha$, $\\beta$, $\\gamma$ mean, are they sampling trees (Line 260) or numbers (Line 248) or qualities (Line 247)? (2) what is $d(x,y)$ in Line 223? It is not explained before or after. **Make sure all the notations or concepts are clearly explained before using them.**

- The motivation behind Iterative Preference Learning is unclear. Specifically, if an outcome-based reward is accessible and the values of intermediate reasoning steps can be obtained via MCTS, why not directly apply GRPO or PPO with step-level or segment-level advantage estimation—as in SPO [4]? Such an approach would provide finer-grained reward signals than DPO and is also simpler. Could the authors please provide further justification or experimental results to clarify this design choice?

[1] TCPO: Thought-Centric Preference Optimization for Effective Embodied Decision-making. EMNLP 2025

[2] TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling

[3] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. ACL 2025

[4] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models. Neurips 2025

### Questions
Please see the weaknesses for the major concerns.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces MobileIPL, a self-training framework for mobile GUI agents that enhances the intermediate “thinking” or reasoning steps of vision-language models interacting with mobile interfaces. The key innovations are (1) instruction evolution — using a model like GPT‑4o to generate diverse Q&A pairs from real mobile UI screenshots to enrich training data and avoid reasoning collapse; (2) construction of a CoaT-tree (Chain of Action-Planning Thoughts) via iterative sampling of reasoning trajectories at the action level, scoring leaf nodes through rule-based rewards, and back-propagating values to generate thinking-level preference pairs (T-DPO) for training; and (3) iterative rounds of preference learning to improve the agent’s reasoning diversity and correctness without requiring heavy process-level annotations. Empirical results on GUI benchmarks (AITZ, AMEX, AndroidControl) show that MobileIPL outperforms strong baselines (like continual-pretraining agents) and generalizes better to out-of-domain mobile UI tasks.

### Strengths
The GUI agents are becoming more and more popular recently, which is a very interesting and highly practical direction.

The overall presentation and paper writing is very clear. The pipeline components, i.e., SFT data collection and preference training annotation and filtering, are all illustrated well, which are helpful in advancing open-source GUI agents development.

The results are comprehensive and convincing, especially demonstrating OOD performance.

### Weaknesses
From a novelty standpoint, the primary contributions of this work lie in building a comprehensive end-to-end pipeline, whereas most of the technical components appear to be adaptations of existing methods or relatively straightforward extensions, especially in light of the recent surge of research on agentic system design and RL-based training. I sincerely appreciate the considerable engineering effort invested in large-scale SFT data collection and RL system implementation—this is clearly valuable for the open-source community and not trivial to execute. Nevertheless, the limited methodological innovation remains the main reason I am unable to assign a higher score.

### Questions
My primary concern, as noted in the weaknesses section, relates to the level of technical innovation in the proposed approach. I would welcome further clarification from the authors regarding the novel methodological contributions and how they advance beyond existing agent-based and RL-training frameworks.

### Soundness
3

### Presentation
3

### Contribution
2
