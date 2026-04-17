# Meta-Researcher: Empowering Planning and Reflection Mechanisms in Large Reasoning Models for Advanced Deep Research Abilities

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Deep research significantly reduces the time and cost of information gathering for researchers by collecting and integrating vast amounts of data. However, its uncontrollable planning and reflection phases during reasoning lead to errors or gaps in information collection, and make it challenging to ensure timely reflection for correcting and supplementing information—thereby performing suboptimally in complex tasks requiring extensive data gathering. To address this limitation, we propose Meta-Researcher, an End-to-End Reinforcement Learning-based Deep Research Method designed to equip Large reasoning models (LRMs) and non-reasoning models with metacognitive capabilities for autonomously executing the research process of "Task Planning - Information Gathering - Process Reflection - Problem Solving'', thereby effectively tackling complex problems that require multiple rounds of information collection and reasoning. Firstly, our approach standardizes LRMs to explicitly output controllable planning and reflection processes rather than implicitly including them within reasoning, thus ensuring that LRMs demonstrate metacognitive abilities in practice. Secondly, we perform end-to-end optimization through the Group Relative Policy Optimization (GRPO) strategy to enhance the active decision-making capabilities of LRMs while strengthening the metacognitive process. Extensive experiments on two tasks — closed-ended question answering and open-ended topic research — demonstrate that Meta-Researcher significantly outperforms existing deep search methods, deep research methods, and proprietary systems. Our approach enhances the reliability and applicability of LRMs in complex task scenarios, offering a new paradigm for developing intelligent agents with autonomous research capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Meta-Researcher, an end-to-end reinforcement learning framework that empowers Large Reasoning Models (LRMs) with explicit metacognitive capabilities for deep research tasks. The key innovation is making the planning and reflection processes controllable by explicitly outputting them rather than keeping them implicit within reasoning. The framework consists of four components: task planning, tool calling, process reflection, and question answering, optimized using Group Relative Policy Optimization (GRPO) with carefully designed rewards. Experiments on closed-ended QA (GPQA, GAIA, Bamboogle, HLE) and open-ended research (Glaive) demonstrate improvements over existing methods.

### Strengths
1.	Clear framework design: The four-component architecture is intuitive and the use of "virtual tools" (Task Planning Tool and Process Reflection Tool) to ensure explicit outputs is a good design choice.
2.	Comprehensive experimental evaluation: The paper evaluates on multiple diverse benchmarks covering both closed-ended and open-ended scenarios, with detailed ablation studies demonstrating the value of each component.
3.	Practical applicability: Experiments on different model scales (7B, 14B, 32B) and non-reasoning models show the framework's broad applicability.

### Weaknesses
1.	Limited algorithmic novelty: The paper essentially combines existing techniques (task decomposition, web search, reflection, GRPO) without introducing fundamentally new methods. The main contribution is engineering-focused rather than conceptually novel. It's primarily about formatting outputs explicitly and reward engineering.
2.	Weak theoretical foundation: (1) The proposed "metacognition" definition (Section 3) lacks rigorous justification and theoretical grounding; (2) No formal analysis of why explicit output of planning/reflection is superior to implicit reasoning; (3) The connection between the framework design and actual metacognitive capabilities is assumed rather than demonstrated.
3.	Missing critical information: (1) No computational cost analysis for training time, inference latency, API costs for web searches not reported; (2) How does the multi-round search and reflection compare to simpler approaches in terms of cost-effectiveness? (3) Limited scalability analysis: What happens with longer contexts or more complex problems?
4.	All experiments use the same web search API and similar domains (academic/factual questions).
5.	It’s unclear how the method performs with different information sources, noisy/contradictory information, or specialized domains.
6.	Some design may have problems. For example, the thinking length penalty (Equation 10) may hurt performance on genuinely complex problems.

### Questions
1.	Hyperparameter sensitivity: How sensitive is the method to the specific reward weights and thresholds? Was there extensive hyperparameter search, and how would practitioners set these for new tasks?
2.	Ablation on RL: What happens if you use the same explicit format with supervised fine-tuning only, without RL? Is RL necessary or is it primarily the format that helps?
3.	Information loss: How much information is lost during context truncation, and how does this affect final performance? Can you quantify this?
4.	Reflection mechanism: Why is it necessary to explicitly call a "reflection tool" rather than just prompting the model to reflect? What does the tool-calling format add?
5.	Failure modes: Can you provide systematic analysis of when the method fails? What types of questions or scenarios are still challenging?

### Soundness
2

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
This paper introduces Meta-Researcher, a reinforcement learning (RL) framework designed to endow large reasoning models (LRMs) with explicit metacognitive abilities for autonomous research. Instead of relying on implicit reasoning within the chain-of-thought, the framework enforces a structured “Task Planning → Information Gathering → Process Reflection → Problem Solving” loop. Meta-Researcher implements two virtual tool: a task-planning tool and a process-reflection tool;  to externalize these cognitive phases, allowing the model’s internal reasoning process to become observable, rewardable, and trainable. Training is performed using Group Relative Policy Optimization (GRPO) with layered rewards (format, accuracy, and thinking-length).

### Strengths
1. Making planning and reflection processes explicit and controllable through virtual tools is a sensible design choice.
2. The paper evaluates on diverse tasks spanning closed-ended QA and open-ended research, with thorough comparisons against both RAG and autonomous search methods.
3. The progression from format rewards to combined rewards addresses the challenge of lacking intermediate supervision in a principled way.

### Weaknesses
1. The open-ended experiments rely on a small (30-sample) Glaive subset and LLM-as-judge scoring, limiting robustness.
2. We don’t see qualitative cases of when reflection fails  e.g., reflection triggers but still misses key evidence, or over-reflects and gets penalized.

### Questions
1. How were the specific reward weights (δ=1e-4, η=2048, ξ=1.0) chosen? How sensitive is performance to these choices?
2. Do plan/reflect stages measurably improve evidence attribution (e.g., source coverage, citation precision/recall, redundancy reduction)?
When does reflection hurt (over-reflection, topic drift)?
3. if the planning tool is not called, the overall format reward becomes 0. This is a hard gate. Did the authors observe training instability or low sample efficiency in early stages due to this?
4. The method encourages multiple tool calls and multiple reflections. What is the average tool budget per example compared to WebThinker, RAgent?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Deep research is a very useful tool. However, it is often difficult to control, and reasoning errors lead to gaps in the collected information. Therefore, they propose Meta-Researcher, an end-to-end RL-based Deep Research method that supports task planning, information gathering, process reflection, and, finally, problem-solving. Their structured approach makes deep research more controllable and allows RL to enhance decision-making abilities. Their experiments show that their method outperforms existing deep research recipes.

### Strengths
1. The authors address one of the currently most interesting applications of LLMs, deep research. While most deep research recipes are proprietary, they make their recipe publicly available. 
2. The authors aim to decompose the complex deep research pipeline into  4 key components (task planning, tools calling, process reflection, question answering) that operate in two modes (closed-ended question answering and open-ended topic research, which are sensible. 
3. Designed a comprehensive list of reward components, which generally seem well motivated to enable effective RL fine-tuning.
4. The benchmarks across closed-ended question answering and open-ended research are well selected.
5. The authors provided the prompts used in every part of the pipeline in the Appendix, which can be useful for future work.

### Weaknesses
1. Some important definitions (e.g., BGE model) or citations (GRPO) are missing 
2. Generally, there is a lack of ablation studies. In particular, no sensitivity analysis of the individual reward components. Therefore, it may be that the reward function is overly complex. Can you verify that all components are necessary? Also, the two-stage training is not ablated. Is it necessary?
3. It would be valuable to see an analysis of how many steps/turns the deep research performs. Especially, how does the RL fine-tuning change this behavior? For example, does RL considerably increase the number of tool calls or result in generally longer interactions than not using RL? 
4. The outcome rewards for closed-ended question answering are clear. However, there is no clarity on how the outcome of open-ended research is graded. Is LLM as a judge used on the listed dimensions? How does performance vary if you switch the judge? How reliable are the judged scores even? An analysis of this would help the paper
5. The paper focuses on Qwen 2.5-72B-Instruct. It is unclear if the findings transfer to other models. It would be essential to verify their findings with more recent reasoning models, such as Qwen3 (even if at a smaller scale), to ensure that all components and reward mechanisms are necessary. Otherwise, it is possible that the found recipe is specific to Qwen 2.5-72B.
6. It would be helpful to the reader to provide more insights into the end-to-end RL component, such as learning curves and other important metrics, which are currently missing. This also includes the computational efforts, technical challenges, etc, that come with end-to-end RL training in a difficult multi-turn problem. At the moment, it seems like RL is more used as a black-box mechanism. 
7. Additionally, there is no comparison of wall-clock times of the compared methods in Table 1 (e.g., between direct reasoning, enhanced reasoning, and autonomous search). This would be important to see for users, to understand the tradeoffs of different approaches better.

### Questions
Some questions are asked above. Here are some more: 

1. The “BGE model” is listed, but not cited or explained. Can you clarify why, what the abbreviation stands for, and how it works?
2. What is $R_{temp}^{tool}$? This is not specified. 
4. Can you provide ablations on the importance of the reward components and two-stage training? 
5. Can you provide a quantitative analysis of how RL changes the agent's behavior (e.g., tool call frequency, etc.)?   
5. In Appendix B.1.2, the authors state that OpenResearchBench is constructed by themselves. Is this dataset going to be released?
6. How does the proposed system compare to proprietary deep research recipes from frontier labs? It is not necessary to beat these recipes, but for the reader, it would be important to see how these methods compare. If it is not feasible to conduct a full evaluation on all downstream tasks, an analysis on a restricted task set would also help.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper describes a system, a prompting strategy and tool-calling harness to build a deep research system. The authors emphasize that the system uses planning, tool-calling, using search to gather information and reflection as crucial components. Secondly, the authors use end-to-end RL with GRPO to optimize the overall system.

### Strengths
Building a useful system to solve challenging tasks such as “deep research” on top of LLM remains a challenge and deserves careful attention from the research community. Tuning these systems end-to-end with RL is a promising direction and public research can have high impact.

### Weaknesses
My biggest concern is around the training/evaluation methodology for the results involving RL: It seems there is no distinction between training and test tasks and I am concerned that RL optimization might have been performed directly on the test problems? 

Furthermore, there are very few ablations describing the impact of individual design choices of the overall system. For example: What is the impact of the two training stages described in 4.4.4? I.e. which gains come from training the model to adhere to the correct formatting vs actually improving the models capabilities? 



Minor comment: 

The paper frequently employs excessive adjectives and overly complex terminology. While individual instances might be justifiable, the overall impression is one of pretentiousness. For example, terms like "virtual tools," "autonomous reflection," "autonomous search," and "multi-tool collaborative calling mechanism" effectively describe one LLM initiating sub-LLM calls with predefined prompts.

I also don’t understand why running RL with reward only for adhering to the toll-calling format is “genuinely instill(ing) metacognitive capabilities for autonomous reflection” (line 310).

### Questions
Do you use distinct training/test tasks for RL? Do you train one model across all benchmark sets and how to you mix the differently sized datasets?

What exactly is the difference between “closed ended” and “open ended” mode? I infer that open-ended tasks use a LLM-as-judge to score the result. Figure 1 suggests open-ended is using additional tools and more reflection iterations? Lines 190 to 205 are not clearly expressing the difference. “integrating external information with the reasoning capabilities” seems to be necessary for both.   

When the meta-researcher triggers a call to the task-planning or process-reflection tools, do the sub-agents receive more than the explicitly passed arguments? (the JSON tool-call syntax “{name: .. arguments: }” suggest arguments and intermediate results are passed explicitly only). The example on page 22 however suggests that the reflection tool is “inline” and has full access to the meta-researchers token stream?

How are the planning and reflection tools sub-trajectories handled for the RL updates? Are they doing their own GRPO update steps? What exactly comprises a group then? 

B.2 Implementation details mentions $s_p = 6$ for the reward calculation in the “Process Reflection Phase” What is this reward and how is it used? Just returned as a number to the meta-agent?

### Soundness
2

### Presentation
1

### Contribution
2
