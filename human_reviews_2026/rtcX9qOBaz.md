# VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
As LLMs with agentic abilities are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VitaBench, a benchmark designed to evaluate LLM-based agents in real-life applications. VitaBench formalizes task complexity along three dimensions—reasoning, tool, and interaction—and proposes a rubric-based sliding window evaluator to assess multi-turn trajectories. With the extensive experiments conducted, the author show that even the best current models achieve only around 30% success in cross-scenario tasks, underscoring the remaining challenges in developing capable real-world LLM agents.

### Strengths
- This paper is well-organized and clearly written.

- This paper addresses the gap in the absence of evaluations of LLM Agents in real-world settings.

- The proposed task complexity formalization provides a reasonable way to quantify the task difficulties.

### Weaknesses
- Although the paper claims to benchmark “LLM-based agents” in real-world scenarios, the evaluation is actually conducted on single LLM models acting as direct function-calling agents, rather than on established agent frameworks or architectures.

- The benchmark evaluates current models as static agents without adaptation or learning. It remains unclear how VitaBench can guide or support the training of future agent models beyond static evaluation.

### Questions
Have the authors considered evaluating existing agent frameworks？

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
This paper presents VitaBench, a benchmark designed to evaluate agents across three interactive daily-life scenarios: food delivery, in-store consumption, and online travel services. The benchmark comprises 100 cross-scenario tasks, 300 single-scenario tasks, and 66 tools. Agent performance is assessed using a rubric-based sliding window evaluator to handle long-horizon trajectories. Experiments on multiple LLMs demonstrate the difficulty of the benchmark, as the best-performing model achieves only a 30% success rate in the cross-scenario setting.

### Strengths
1.	The paper is well-written and clearly describes the benchmark and evaluation results.
2.	The benchmark focuses on realistic, real-world scenarios, making it valuable for guiding real agent development. Its components, including the user simulator, evaluator, and hyperparameter settings, are shown to be robust in the reliability analysis.
3.	The paper provides thorough analysis of the evaluation results, from aspects of reasoning complexity, task complexity, and interactive complexity.

### Weaknesses
1.	Although the benchmark is well designed, there already exist many benchmarks addressing similar tasks. It is unclear how much new insight VitaBench contributes beyond existing efforts.
2.	The sliding window evaluator is intended to manage long trajectories, but it raises the question of whether evaluation should occur over the entire trajectory. An alternative approach could be to extract key outcomes (e.g., booking results) and apply the rubrics to those results directly. Additionally, the permanent satisfaction criterion across windows may fail when an agent meets a rubric early but changes later.

### Questions
1.	Could you provide more details about the tools? Specifically, what are the 66 tools, and are their returned values (e.g., store_info) real or simulated?
2.	Do variations in simulated user attributes or communication styles influence the results differently?
3.	What does the “score” in Table 4 represent?

### Soundness
3

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
4

### Summary
This paper introduces VitaBench, a new benchmark designed to evaluate LLM-based agents on complex, interactive tasks grounded in real-world, "life-serving" applications. The benchmark spans three domains (food delivery, in-store consumption, online travel) and comprises 66 tools, 100 cross-scenario tasks, and 300 single-scenario tasks. The authors motivate their work by arguing that existing benchmarks lack the necessary complexity in terms of information volume, tool interdependencies, and dynamic user interaction.

### Strengths
1. The paper correctly identifies a critical need for more challenging and realistic agent benchmarks. VitaBench represents a significant step up in scale and complexity compared to prior work, pushing the evaluation frontier toward long-horizon. The ambition to create a "life-serving simulation environment" is good and moves the field in the right direction.

2. The proposed rubric-based sliding window evaluator is a  practical solution to the problem of evaluating long-form agent trajectories with LLMs.

### Weaknesses
1. The benchmark's entire pipeline is built on LLMs. The agent is an LLM, the user simulator is an LLM (gpt-4.1), and the final evaluator is another LLM (claude-3.7-sonnet). While the authors conduct reliability studies (Section 5.1), this "LLM-evaluating-LLM-interacting-with-LLM" setup raises concerns about potential biases and circularity. The benchmark may inadvertently measure how well one model's behavior and style align with another's, rather than objective task success. For example, the user simulator's "cooperative" nature might favor agents with similar inherent tendencies, and the evaluator might be more lenient towards reasoning patterns similar to its own. This methodological concern is significant and undermines the objective grounding of the benchmark's results.

2. The simulator is prompted to faithfully convey all the points from a pre-written, detailed instruction set (page 13, "Must ensure every detail from instructions is mentioned"). Real users are not like this. They forget their own constraints, provide contradictory information ("I want the cheapest option... no, not that one, it looks bad"), and change their minds midway through a task. A key skill for a real agent is navigating this human inconsistency, which VitaBench's user doesn't seem to model. The paper notes that user behavior like "impatience" is explicitly prompted (e.g., "If the agent repeats the same question... show impatience," line 719). This is a scripted reaction. A real impatient user might interrupt, abandon the conversation, or start making demands outside the original scope. Moreover, The agent can make as many tool calls as it wants without penalty. In reality, API calls can have monetary costs, and excessive interaction turns frustrate the user and increase the probability of abandonment. **I think the benchmark tests an agent's ability to follow a complex conversational script and using tools rather than handle truly emergent, unpredictable human behavior thus not a realisttic benchmark.**

### Questions
The benchmark is explicitly focused on "Chinese contexts" (Appendix B, line 646), while many of the top-performing models evaluated (like the GPT and Claude series) are developed by Western companies and primarily trained on English-language and Western-centric data.Many reasoning tasks are steeped in cultural common sense. For example, the user request in Figure 3 mentions booking a dinner for a "10th anniversary celebration" near the "Huangpu River." While anniversary is a universal concept, the specific expectations around such an event, typical restaurant choices, or etiquette might differ. A model trained on Western data might make different assumptions than one with deeper knowledge of modern Chinese urban culture. The example trajectory's mention of a "Tangshan time-honored restaurant" is a perfect example of a culturally specific concept (老字号, lǎo zì hào) that an English-centric model might only understand at a surface level. I was wondering if author considering this.

### Soundness
2

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
This paper introduces VitaBench, a real-world–oriented agent benchmark spanning three domains (delivery, in-store consumption, OTA).
The tasks require multi-turn dialogue, temporal/spatial reasoning, proactive clarification, and tool chaining. Results show strong difficulty: ~30% success on cross-scenario and <50% on single-scenario tasks with various state-of-the-art models.

### Strengths
1. Clear three-axis task-complexity framework (reasoning/tool/interaction) and cross-scenario composition without domain-specific policies.
2. The data are collected from authentic platform data.
3. Rubric + sliding window evaluator with reported human agreement (κ≈0.828) and ablations justifying design choices.
4. Comprehensive experiments and error taxonomy (reasoning dominates) study,

### Weaknesses
1. Compute/latency opacity: Many-turn trajectories and 4-run protocol imply high cost; paper lacks concrete cost accounting per task/model.
2. The main results are based on only one basic function calling agent. It is unknown how different agent design would affect the results.

### Questions
1. What is the cost analysis of different models? 
2. Is any other agents/techniques being tested? Can agent design affect results?

### Soundness
3

### Presentation
3

### Contribution
3
