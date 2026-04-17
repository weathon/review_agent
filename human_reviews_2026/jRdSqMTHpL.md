# The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Enabling LLMs to effectively operate long-horizon task which requires long-term planning and multiple interactions is essential for open-world autonomy. Conventional methods adopt planning with actions where a executable action list would be provided as reference. However, this action representation choice would be impractical when the environment action space is combinatorial exploded (e.g., open-ended real world). This naturally leads to a question: As environmental action space scales, what is the optimal action representation for long-horizon agents? In this paper, we systematically study the effectiveness of two different action representations. The first one is conventional planning with actions (PwA) which is predominantly adopted for its effectiveness on existing benchmarks. The other one is planning with schemas (PwS) which instantiate an action schema into action lists (e.g., “move [OBJ] to [OBJ]” → “move apple to desk”) to ensure concise action space and reliable scalability. This alternative is motivated by its align- ment with human cognition and its compliance with environment-imposed action format restriction. We propose cognitive bandwidth perspective as a conceptual framework to qualitatively understand the differences between these two action representations and empirically observe a representation-choice inflection point between ALFWorld (∼35 actions) and SciWorld (∼500 actions), which serve as evidence of the need for scalable representations. We further conduct controlled experiments to study how the location of this inflection point interacts with different model capacities: stronger planning proficiency shifts the inflection rightward, whereas better schema instantiation shifts it leftward. Finally, noting the suboptimal performance of PwS agents, we provide an actionable guide for building more capable PwS agents for better scalable autonomy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a cognitive perspective on how LLM agents handle planning in environments of increasing complexity. The authors argue that action-based agents suffer from cognitive overload as the action space expands, whereas schema-based agents alleviate this burden by reasoning over abstract action schemas. The paper introduces the cognitive bandwidth perspective which models the agent’s reasoning as a limited resource distributed over stages such as environment understanding, planning, and schema instantiation. Experiments across four text-based environments, TextCraft, WebShop, ALFWorld, and SciWorld, show that planning with action performs better in simple settings while planning with schema scales better to larger complicated environments. The study positions schema-based reasoning as a cognitively efficient alternative for long-horizon agent scalability.

### Strengths
* Clear conceptual motivation: The paper tackles a fundamental but often overlooked question of how agents should represent actions. This framing provides a fresh angle for studying reasoning efficiency in LLM-based agents for long-horizon tasks.
* Cognitive theory integration: The idea of modeling LLM reasoning through cognitive bandwidth is novel and offers an intuitive explanation for performance drops in complex environments.
* Systematic environment scaling: Using four progressively complex benchmarks from TextCraft to SciWorld gives a logical structure to the experimental comparison.
* Controlled cognitive stress test: The authors cleverly simulate increasing environment understanding load by injecting distractor actions, demonstrating how planning with action performance collapses under noise while planning with schema remains stable.

### Weaknesses
* Weak rationale for the cognitive-load claims: The claim that planning with actions' load concentrates in environment understanding and planning with schemas shifts it to schema instantiation is never empirically validated. No token-level or attention-based measurement of reasoning load is provided. The argument relies only on indirect correlations e.g., error rates rather than direct evidence. For example, in an environment where abundant number of different action schemas exist, this assumption cannot always hold.
* Inaccurate characterization of environment complexity: The paper states that ALFWorld has ~35 actions, but ALFWorld which is based on ALFRED actually contains hundreds of grounded action combinations from 11 action schemas and 84 object classes. The authors should provide a detailed explanation of how this figure of 35 actions was derived.
* Unclear explanation of ALFWorld results: Planning with schemas performance in ALFWorld is extremely low (e.g., GPT-4.1 = 16 %, Gemini-2.0-flash = 1%), despite the schema space being small (11 schemas with 35 actions). I think such failure cannot be explained just by SI overhead. An in-depth analysis and a plausible explanation are needed.
* Lack of methodological transparency and reproducibility: The paper’s Reproducibility Statement promises all prompt templates and implementations in the appendix, but none are provided. Key details like prompt format, environment wrappers, model parameters, and handling of invalid actions are missing. This limits proper comprehension and appreciation of the provided results as well as the reproducibility of the results.
* Presentation clarity issues (tables and figures): In Table 2, the “Delta” column lacks an explicit definition that makes readers infer that it means (PwS – PwA) difference. Also in Figure 2, the legend for the gray shading incurs confusion as the grey colored bar does not exist in the chart. As such, all the figures and tables need to be reexamined and revised for clearer presentation.

### Questions
Please refer to the Weaknesses section for the raised issues.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the "Cognitive Bandwidth Bottleneck" faced by large language models (LLMs) when handling long-horizon tasks and compares two action representation paradigms: **Planning with Actions (PwA)** and **Planning with Schemas (PwS)**. The cognitive bandwidth bottleneck occurs when LLMs are unable to effectively process long-term task planning due to limited information processing capacity. PwA, which uses predefined action sequences, works well for short tasks but is inefficient for long-term tasks. In contrast, PwS, which employs schemas to represent task components, handles long-term dependencies more effectively, offering greater flexibility and improved task planning. The paper concludes that PwS provides a better solution for long-horizon tasks by overcoming the cognitive bandwidth limitation of PwA.

### Strengths
## Strengths

The paper introduces a novel framework for comparing two action representation paradigms—**Planning with Actions (PwA)** and **Planning with Schemas (PwS)**—through the lens of **Cognitive Bandwidth**. This conceptual framework is particularly effective in explaining how the cognitive capacity of large language models (LLMs) is allocated during task execution. By formalizing the trade-offs between PwA and PwS, the framework highlights that PwS offers a more scalable and flexible approach, inspired by human cognition, by reducing the complexity of action representations through schemas. In contrast, PwA relies on a detailed action list, which, while effective for short tasks, is less efficient for long-horizon tasks due to the high cognitive load at the environment understanding (EU) stage. PwS, by shifting the cognitive load to the schema instantiation (SI) stage, allows for better handling of long-term dependencies, making it a more reliable and effective choice for complex tasks. This insight into cognitive bandwidth and its impact on action representation provides a strong theoretical foundation for advancing LLM-based task planning systems.

### Weaknesses
## Weaknesses

While the paper provides a solid theoretical framework for comparing **Planning with Actions (PwA)** and **Planning with Schemas (PwS)**, several weaknesses in the experimental evaluation and analysis can be identified:

1. **Limited Experimental Scope for PwA**: No results are provided  in TextCraft on PwA scenarios. So the results are not clear in simple environments. I recommend the **babyAI** benchmark could serve as a needed test for PwA, which might give more evidence of your contribution.

2. **Insufficient Comparison in Complex Scenarios**: Although **AlfWorld** is expected to be a more complex environment than **WebShop**, the experimental results show that **PwS** does not significantly outperform PwA in this case. This discrepancy raises questions about whether PwS is truly more effective in handling more intricate tasks, as the framework suggests. Further clarification are needed to explain this result and whether additional factors such as task characteristics or environmental constraints influence performance.

3. **Confuse about Relationship between Action Paradigm Choice and Environment**: The paper also needs to provide a discussion on how the choice between PwA and PwS might relate to the environment itself. Specifically, it is unclear whether the selection of the action representation paradigm is solely dependent on the intrinsic properties of large language models (LLMs) or if it also interacts with environmental factors and low-level skills. A more detailed explanation of how the environment, task complexity, and model capabilities influence the choice between PwA and PwS would provide more practical insight into their applicability in real-world scenarios.

### Questions
See weakness above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the scaling challenge for Large Language Model (LLM) agents in environments with combinatorially exploding action spaces by comparing two core action representations: Planning with Actions (PwA) and Planning with Schemas (PwS). PwS aims to ensure scalable and compliant action generation by instantiating an abstract schema (e.g., "move [OBJ] to [OBJ]") into a concrete action (e.g., "move apple to desk").

Empirical results across environments of varying complexity (ALFWorld $\approx$ 35 actions to SciWorld $\approx$ 500 actions) reveal a representation-choice inflection point. PwA is superior in the low-to-medium action space due to lower schema instantiation (SI) overhead, but PwS gains an advantage in high-complexity environments (SciWorld) where the environment understanding load of PwA becomes prohibitive. This suggests the optimal representation is task-dependent. The paper concludes that post-training focused on multi-turn tool use enhances SI capability, shifting the inflection point leftward.

### Strengths
1. The PwS paradigm is promising for real-world autonomy as it aligns with human cognitive mechanisms and inherently produces interface-compliant outputs that scale more reliably than free-form generation.
2. The conclusions offer actionable steps for future agent development, specifically identifying that post-training which emphasizes multi-turn tool use significantly improves Schema Instantiation (SI) capability, which is necessary to shift the inflection point to the left and make PwS broadly viable.
3. Empirical evidence is compelling, demonstrating a Representation-Choice Inflection Point located between ALFWorld ($\approx$ 35 actions) and SciWorld ($\approx$ 500 actions), providing quantified evidence of the scaling limitations inherent in the conventional PwA paradigm.
4. I like Figure 1 which shows a high-level overview of the relationship between environment complexity and agent effectiveness. Helped me understand the paper better.

### Weaknesses
1. The Cognitive Bandwidth Perspective is explicitly defined as a conceptual and qualitative framework, lacking any quantifiable metrics or probes for the postulated bandwidth ($\mathcal{B}(\mathcal{M})$) or load ($\mathcal{L}_{stage}$) variables.

2. The paper notes that PwS agents demonstrate suboptimal performance compared to PwA in the low-to-medium complexity ALFWorld, suggesting the fundamental bottleneck of Schema Instantiation (SI) is underdeveloped in current models, despite being the proposed solution for scalability.

3. The behavioral comparison's analysis of "Repetitive Actions" and "Invalid Actions" relies on either simple rule-based filtering (for Invalid Actions) or an external LLM (Kimi-K2) to analyze and generate ground truth for failure modes, which could introduce potential noise and dependence on a specific model's subjective judgment.

### Questions
1. To overcome the qualitative nature of the Cognitive Bandwidth Perspective, could the authors propose a quantifiable proxy for Schema Instantiation (SI) load ($\mathcal{L}_{SI}$), perhaps based on the number of objects visible in the observation (combinatorial complexity of grounding) or the token count of the generated action string? This could help replace the subjective binary categorization in Section 5.
2. The paper argues PwA suffers from high Environment Understanding (EU) load and long context pressure. Yet, for PwS, the final grounded action still needs to be inserted into the history for subsequent interaction steps (Eq. 1, 2). Could the authors measure the total historical token growth for successful trajectories in SciWorld under both PwA and PwS? This would provide quantitative evidence on how much PwS actually mitigates the "long context pressure" throughout the entire multi-turn trajectory.
3. The current schemas for ALFWorld and SciWorld are derived by replacing all object arguments with a generic `[OBJ]` token. How robust is this simple abstraction across different instruction styles? Does providing the LLM with slightly different schema formats, such as `move [FRUIT] to [DESK]` (using explicit types) or `move [OBJ1] to [OBJ2]`, significantly change the inflection point's location for models categorized with poor SI capability?
4. Since Long Reasoning Models (LRMs) have marginally improved PwS performance but failed to solve the SI bottleneck, did the authors experiment with prompting strategies that specifically augment the SI stage? For example, using a Chain-of-Thought (CoT) prompt structured as: "Thought: To instantiate 'move [OBJ] to [OBJ]', I must first identify available objects, then check if they are movable, and finally generate the action 'move apple to desk'"? Measuring performance change here would directly probe the bottleneck.

Would appreciate answers to these questions. I am willing to increase my scores if the rebuttal is convincible for my questions.

### Soundness
3

### Presentation
3

### Contribution
2
