# Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills

- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing approaches are limited by language ambiguity, noisy outputs, and limited context windows, which makes long-horizon tasks hard to solve.
While closed-loop feedback has been explored, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often simple for humans to identify. 
Moreover, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable and extendable skills. 
To address these issues, we propose a human-in-the-loop lifelong skill learning and code generation framework that encodes feedback into reusable skills and extends their functionality over time.
An external memory with Retrieval-Augmented Generation and a hint mechanism supports dynamic reuse, enabling robust performance on long-horizon tasks.
Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in feedback rounds. 
It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces LYRA, a human-in-the-loop lifelong learning framework for LLM-based code generation in robotic manipulation, focusing on acquiring and extending reusable skills to handle long-horizon tasks like building structures. The framework encodes human feedback into modular skill functions stored in an external memory, using retrieval-augmented generation (RAG) and user hints for dynamic reuse, addressing issues like noisy LLM outputs, catastrophic forgetting, and limited context windows in prior code-as-policies approaches. They demomnstrate reasonable performance on benchmarks (Ravens, Franka Kitchen, MetaWorld) and real-world tasks (e.g., "build a house" with >20 primitives).

### Strengths
1. The combination of human-in-the-loop skill learning with lifelong capability extension through user-designed curriculum is a reasonable contribution. The hint mechanism for guiding RAG retrieval is simple but effective. The paper clearly articulates limitations of existing LLM-based code generation approaches (language ambiguity, catastrophic forgetting, limited context) and proposes a well-motivated solution.

2. Evaluation spans three simulation benchmarks plus real-world deployment, demonstrating broader applicability than many code generation works

3. Impressive long-horizon performance: They successfully solve "build a house" requiring 20+ primitives represents a genuine advance in manipulation complexity.

### Weaknesses
1. Limited Novelty: Individual components (RAG for code generation, human-in-the-loop feedback, skill libraries) exist in prior work. The main contribution is their integration for robotic manipulation.

2. Insufficient comparison to state-of-the-art: The paper mentions VLA foundation models (OpenVLA, π0, GR00T) in related work but does not compare against them experimentally. Given these models address similar long-horizon manipulation problems, the lack of comparison significantly weakens claims about superiority.

3. The heavy reliance on human-in-the-loop for feedback and curriculum design limits scalability to real deployments without experts.

### Questions
1. Can the authors provide comparisons with stronger baselines on the benchmarks like OpenVLA, π0?

2. How much human time (in hours) was required to build the skill library for each benchmark? This is critical for assessing practical feasibility. Please provide this analysis in the paper.

3. Since the proposed is based heavily on the human-in-the-loop procedure for feedback, can the authors comment on possible alternatives to avoid this limitation, and how can they be effectively deployed?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LYRA, a human-in-the-loop lifelong code generation framework for robotic manipulation that integrates LLM-based program synthesis, human feedback, and skill memory retrieval. Unlike prior LLM-only closed-loop approaches that struggle on long-horizon tasks, LYRA encodes user feedback into reusable, modular “skills”, stores them in external memory with retrieval-augmented generation (RAG), and enables users to guide the agent through hint-based dynamic skill selection. The framework continually expands its capabilities via a user-designed curriculum, achieving lifelong skill acquisition without catastrophic forgetting. Experiments across Ravens, Franka Kitchen, MetaWorld, and real-world Franka FR3 tasks demonstrate great success rate, outperforming LLM-only baselines.

### Strengths
1. The integration of human-in-the-loop feedback, external memory, and modular skill representation provides a robust alternative to end-to-end LLM-driven methods.
2. The framework emphasizes skill inheritance and continual expansion, a key step toward reusable and interpretable robotic skills.
3. Covers simulation (Ravens, Franka Kitchen, MetaWorld) and real-world deployment, with quantitative and qualitative evidence.

### Weaknesses
1. While comparisons to LLM code-generation baselines are thorough, the paper omits strong lifelong or hierarchical RL baselines (e.g., HiLLa, BOSS, Text2Reward-type methods).
2. The framework heavily relies on manual feedback and a user-designed curriculum. Specifically: (1) user-guided skill code generation, (2) user-guided skill capability extension with few-shot examples, and (3) user-provided hints to direct the agent toward the correct subset of skills as the number of skills |Z| and behaviors |E| increases. This reliance on human intervention can be costly, particularly when scaling to hundreds of skills or when the users are less experienced.

### Questions
1. "a meta-prompt that explicitly asks the agent to preserve prior functionality while adapting to new tasks" To keep the balance between stability (preserving old skills) and plasticity (acquiring new functionality), the paper uses 1) Meta-prompt Regularization 2) Modular Code Extension 3) Re-evaluation Loop. How is the evaluation result on this part? What is the Average number of corrections (NoC)?
2. Few-shot examples that show mappings from instructions to task-specific code plan would be really helpful for LLM to generate skill program. These examples are retrieved based on the semantic similarity between the new instruction and previously seen instructions. My question is: are these examples environment-specific to those used in the experiments, or are they designed to be environment-agnostic? Moreover, how robust and generalizable is the proposed approach when faced with a new task that lacks a corresponding example in the existing code library?
3. "The agent can freely modify implementation details under the reserved skill name" How detailed the user's instruction would be(Does it require user be very clear about the attributes of the objects in the envrionments?)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper aims to address limitations in using LLMs for robotic code generation, specifically their struggles with long-horizon tasks, language ambiguity, and catastrophic forgetting. The authors propose LYRA, a human-in-the-loop (HITL) lifelong learning framework. LYRA enables an agent to learn reusable skills as Python functions, guided by human feedback. These skills and successful task examples are stored in an external memory and retrieved using Retrieval-Augmented Generation to solve new tasks. The framework includes a user-designed curriculum for extending skill capabilities and a "hint" mechanism for guiding retrieval. Experiments in simulation (Ravens, Franka Kitchen, MetaWorld) and on a real Franka FR3 robot show LYRA achieves relatively strong performance and improvement in feedback efficiency, successfully solving complex tasks like "build a house".

### Strengths
- Involving HITL feedback, a user-driven curriculum to LLM-robotic control is novel and interesting.
- The evaluation is thorough, spanning three distinct simulation benchmarks (Ravens, Franka Kitchen, MetaWorld) and a real-world Franka FR3 robot.
- The ablations are insightful: LYRA w/o memory and LYRA w/ LLM feedback directly prove the value of the two core components.
- The "build a house" task serves as a good and compelling demonstration. Claiming to be the first to solve this task, the authors clearly show how a complex behavior can be decomposed into 12 hierarchically-learned skills.

### Weaknesses
- The paper combines several well-established ideas: HITL, RAG, and code-generation. The authors should clarify whether the framework introduces specific novel contributions versus the integration of existing ideas.

- The paper strongly motivates the need for HITL by showing LLM-only feedback is unreliable, but it does not adequately quantify the cost of the human. The "42% efficiency improvement" is measured in "Average number of corrections". This metric is useful but incomplete. It hides the human's cognitive load and time-per-round. A human correction might be 1 round but take 10 minutes, while an LLM correction takes 10 rounds at 30 seconds each.

### Questions
- The "hint" mechanism is described as "simple yet effective", but its implementation and UI are omitted. How does a user provide this hint? Is it free-form text that is then parsed?

- What is the mechanism for the "roll back from failure"? If a 20-step plan (like "build a house")  fails at step 15, does the user correct from that state, or must the entire task be restarted? How is task state managed?

### Soundness
3

### Presentation
4

### Contribution
3
