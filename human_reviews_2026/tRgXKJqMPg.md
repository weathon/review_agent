# OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
A critical challenge in developing capable AI agents is defining their "action space''—the set of possible actions they can take. These spaces can range widely, from generating code and using language skills to operating on latent representations or raw joystick controls.
Through a large-scale study in Minecraft, we discovered a major dilemma: no single action space is universally best. The most effective action space is highly task-dependent, which complicates the goal of building one generalist agent that can handle everything. To solve this, we introduce Chain-of-Action (CoA), a novel framework that unifies high-level abstracted actions and low-level control actions within a single model. With CoA, an abstract goal is not just a final command; instead, it serves as an intermediate reasoning step that guides the model to generate the precise, executable actions needed to complete the task. Furthermore, we show that an "All-in-One" agent, trained on a diverse mix of action spaces using CoA, learns a more generalizable policy. This unified agent achieves a new state-of-the-art, outperforming strong, specialized baselines. To support the research community, we are releasing the OpenHA (Open Hierarchical Agents) suite, which includes our benchmark of over 800 tasks, curated datasets, source code, and all model checkpoints at: \url{https://anonymous.4open.science/anonymize/OpenHA-ACFE}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Chain-of-Action is a unified framework that integrates high-level abstract reasoning with low-level control through an intermediate “thinking” step, allowing agents to translate abstract goals into precise executable actions. Large-scale experiments in Minecraft across 1,000 tasks demonstrate that CoA enhances generalization and decision-making, outperforming specialized baselines. The authors further develop an agent trained on mixed action spaces and release the OpenHA suite to facilitate future research on hierarchical and generalizable action learning.

### Strengths
1. The paper introduces the Chain-of-Action framework, that unifies high-level reasoning and low-level control through intermediate abstract actions. 
2. The paper presents experiments across over 1,000 Minecraft tasks and ablation analyses that convincingly demonstrate the effectiveness of CoA and the proposed All-in-One agent.
3. The work makes a substantial contribution to the field of generalist AI agents by revealing the task-dependent nature of action spaces. The public release of the OpenHA benchmark suite further enhances the paper’s long-term impact and reproducibility.

### Weaknesses
1. The model is still fundamentally based on Qwen2-VL, without introducing substantial architectural innovation or new pretraining strategies.
2. From Figure 1, the proposed method shows little difference from standard VLA architectures and does not demonstrate a clear performance advantage in practice, despite its claimed end-to-end design.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents OpenHA, a family of hierarchical embodied agents in Minecraft, unified under a proposed Chain-of-Action (CoA) framework. CoA models abstract actions as intermediate “thought” tokens before producing low-level primitives, supporting both fast and slow inference modes. The authors also construct a large-scale benchmark (~800 tasks) covering embodied, GUI, and combat settings. Experiments across multiple action spaces (Skill, Grounding, Motion, Latent, Text) and an All-in-One training setup show promising transfer and efficiency benefits.

### Strengths
1. Addresses a timely and meaningful problem: unifying multiple action spaces for embodied agents.
2. CoA offers a clear and flexible framework connecting high-level planning with low-level execution.
3. The openness of the system (800 tasks, public code, and checkpoints promised) can benefit the community.

### Weaknesses
1. CoA is mainly a token-level decomposition inspired by Chain-of-Thought, no new loss, optimization, or theoretical insight.
2. The number of baselines is relatively small, causing concern about the real performance of the work.
3. If there may exist programmatic labeling bias: the rule-based pipeline may leak structural information from expert trajectories (App. B.2).
4. Lack of analysis of why CoA improves reasoning, and fast/slow mode trade-offs lack quantitative modeling.
5. It is not fully clarified whether the same level of retraining or data preprocessing was applied to them. If baselines merely reuse results from the original paper (under different training conditions), comparisons may be biased.
6. Experiments are limited to Qwen2-VL-7B, raising questions about generalizability to other LLM architectures.

### Questions
Please refer to the Weakness for more details.

### Soundness
2

### Presentation
3

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
This paper, OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft, presents a thorough investigation into the problem of action space design for agentic models. The authors propose a Chain-of-Action (CoA) framework that unifies high-level abstracted actions and low-level environmental actions under a single end-to-end formulation. Built atop this, they introduce OpenHA, a family of hierarchical agents trained and evaluated across over 800 Minecraft tasks.

### Strengths
High significance and scope – The paper tackles a fundamental but underexplored question: how should an agent’s action space be represented and unified across abstraction levels? The results have strong implications for both foundation models for agents and future generalist VLA research.

Chain-of-Action (CoA) formulation – The idea of modeling abstracted actions as intermediate reasoning tokens in a single autoregressive process is elegant and novel. It unifies hierarchical decomposition with the reasoning paradigm of LLMs and avoids multi-stage training limitations of classical hierarchical architectures.

Large-scale, fair evaluation – The benchmark (≈1000 tasks) and ablations (Tables 2–4) are impressively comprehensive. The authors control for token counts, use consistent pretrained backbones (Qwen2-VL-7B), and evaluate across multiple domains (embodied, GUI, combat), providing strong empirical support for their claims.

Open-source contribution – The planned release of code, data, and checkpoints is a major community contribution, filling an urgent need for reproducible baselines in embodied agent research.

### Weaknesses
Computational cost – The unified CoA (slow mode) inference trades speed for accuracy (Table 3), but there is no discussion of scalability for real-world or multi-agent settings. Some estimate of inference-time compute (e.g., tokens/sec, cost per step) would strengthen the practicality discussion.

Minecraft-centric evaluation – Although Minecraft is a flexible testbed, validation on at least one different embodiment domain (e.g., GUI interaction or robotic control) would make the generality claim more robust.

Limited analysis of latent actions – The latent-action baselines are mentioned but not deeply analyzed; a closer look at learned latent embeddings and their interpretability could have further enriched the findings.

### Questions
How sensitive is CoA’s performance to the format and tokenization of abstracted actions? Could it generalize to non-text-based or vector-quantized action codes?

Could the All-in-One training be seen as a form of multi-action-space alignment analogous to multi-modal pretraining? If so, have you examined shared representations?

How are hierarchical inference modes switched dynamically at runtime? Can the agent adaptively select between fast/slow reasoning based on task complexity?

How does the CoA formulation interact with memory or trajectory length limits (e.g., 15-step working memory in Sec. C)? Does truncation degrade reasoning?

### Soundness
3

### Presentation
3

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
This paper addresses the critical challenge of defining an effective "action space" for generalist AI agents in complex environments, particularly in the game of Miinecraft. The authors' finding is that no single abstracted action space is universally optimal; the most effective action representation is highly task-dependent, and they propose CoA (unifying high-level abstract actions and low-level control actions within a single, end-to-end autoregressive model) and OpenHA (grounding all diverse, high-level abstractions in the same primitive, low-level action space). The author validate them on a benchmark of over 800 tasks in Minecraft, which they also plan to release as part of their contributions.

### Strengths
1. The paper tackles a fundamental problem in agent development. The finding that "the optimal action space is task-dependent" is demonstrated through the large-scale experiments, which motivates the two methods proposed.
2. In Table 4, the demonstrated fact that a single agent (OpenHA) trained on a mixture of action spaces can outperform all specialist agents is a significant result.
3. The benchmark the author plans to release, given that it is broad and manually-verified, which includes all the codes, datasets, and model checkpoints, is a solid contribution.

### Weaknesses
1. Only the Minecraft environment is considered. How does this method generalize to other environments? Like GUI automations or other complex games?
2. No human evaluation of the interpretability / reasoning transparency of the “thought" step. This remains unverified.

### Questions
1. Why was TA chosen as the primitive representation, among others like RA?
2. The experiments on CoA and OpenHA focus only on Motion (MotionCoA) and Grounding (GroundingCoA) actions. Skills actions like gathering wood, crafting a pickaxe, and mining stone seem like a very natural fit for the high-level "thought" step in the CoA framework. What are their performances on these tasks?
3. Does the CoA framework, which requires data triplets $\{(o_t​,A_t​,a_t​)\}$, require more data and training to converge compared to a standard VLA model trained only on data pairs $\{(o_t​,a_t​)\}$?

### Soundness
3

### Presentation
3

### Contribution
4
