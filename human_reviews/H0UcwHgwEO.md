# LLF-Bench: A Benchmark for Interactive Learning from Language Feedback

- Decision: Reject
- Scores: 5, 6, 5, 3

## Abstract
We introduce a new benchmark, LLF-Bench (Learning from Language Feedback Benchmark; pronounced as ``elf-bench''), to evaluate the ability of AI agents to interactively learn from natural language feedback and instructions. Learning from language feedback (LLF) is essential for people, largely because the rich information this feedback provides can help a learner avoid much of trial and error and thereby speed up the learning process. Large Language Models (LLMs) have recently enabled AI agents to comprehend natural language --- and hence AI agents can potentially benefit from language feedback during learning like humans do. But existing interactive benchmarks do not assess this crucial capability: they either use numeric reward feedback or require no learning at all (only planning or information retrieval). LLF-Bench is designed to fill this omission. LLF-Bench is a diverse collection of sequential decision-making tasks that includes user recommendation, poem writing, navigation, and robot control. The objective of an agent is to interactively solve these tasks based on their natural-language instructions and the feedback received after taking actions. Crucially, to ensure that the agent actually learns from the feedback, LLF-Bench implements several randomization techniques to ensure that the task isn't familiar to the agent and that the agent is robust to various verbalizations. In addition, LLF-Bench allows configuring different types of feedback to study how agents respond to them. Together, these features make LLF-Bench a unique research platform for developing and testing LLF agents.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors introduce LLF-Bench, a benchmark for evaluating the ability of AI agents in Learning from Language Feedback (LLF). LLF-Bench provides a suite of 8 diverse tasks including movie recommendation, poem writing, numerical optimization, and robot control. Unlike traditional RL gym environments, LLF-Bench environments provide no numerical reward, instead providing a natural language instruction at the beginning, and natural language feedback to the agent at each step. Language feedback is generated via manually written feedback templates for consistency and efficiency, while ensuring a diverse set of paraphrased templates to prevent overfitting.

### Strengths
- LLF-Bench provides a useful benchmark and research platform for LLF, a very interesting contemporary topic which has lacked systematic environments for evaluation.
- Good conceptual discussion of how this compares to standard RL and why LFF has advantages over RL.
- Compatibility with existing OpenAI Gym environments is a clever design choice that makes this easily extensible.
- Useful distinction of 3 different types of feedback: Reward, Future Feedback, Hindsight Feedback
- Use of templates (as opposed to LLM-generated feedback) is a sensible choice for reproducibility and cost/compute efficiency.

### Weaknesses
My biggest complaint is insufficient experiments and analysis on the benchmark.
- In my mind, LLF-Bench needs to provide evidence justifying itself as a meaningful improvement over simply passing in standard RL gym observations and rewards into a language model. The paper provides good motivation for why we should expect language feedback to be more efficient that traditional rewards, but fails to provide evidence for this hypothesis.
- Relatedly, LLF-Bench introduces 3 interestingly different kinds of feedback types (Reward VS Future VS Hindsight), but do little to show why they are useful. In theory, comparing the setups between Table 2 and Table 3 gives us an ablation of how removing Future Feedback affects performance, but the separate presentation of the tables makes it very difficult to compare, and the text does not discuss the findings as far as I can see.
- The authors also don't clearly motivate why they run the specific experiments of Table 2 and Table 3. Table 2 makes sense as a default / main setup, but ablations should be clearly motivated and state what questions they seek to answer. For example, I would be interested to see comparisons of: [Reward] vs [Reward + Hindsight] vs [Reward + Hindsight + Future] or similar experiments trying to isolate the impact of each feedback type.
- It would be useful to also compare LLM results against RL-trained agents using rewards as baselines, since that would allow us to compare how quickly RL agents learn from reward VS LLMs with language feedback. This would provide some useful investigation into claims that feedback is more efficient that RL rewards.

### Questions
- Something else I was looking for was further elaboration on which kinds of environments stand to benefit most from natural language feedback. The paper alludes to the idea that some tasks benefit more from language feedback:
> "real-life use cases, where a user needs LLM-based agents to handle tasks whose solution cannot be directly inferred from the task description and has to be learned from interactions and feedback instead (such as “make the title text larger” or “wrap the code with an error-catching block.”)."

  But it is not clear how the selected environments and the constructed feedback provide these interesting properties. Would be curious if there was more careful thought in the selection of these particular environments.

- Is there an error in Appendix F.2? "Example Feedback" includes a "Thought: From the feedback, it seems that..."

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new benchmark designed to provide language model feedback across 8 diverse decision-making tasks. The benchmark is implemented on the OpenAI Gym framework, by modifying observation space with adding task-specific instructions and feedback text. Language feedback is generated based on reward, anticipated future feedback, and hindsight feedback, structured through task-specific templates.

### Strengths
Recognizing the potential of human feedback to guide agents in leveraging real-world data more effectively, this benchmark explores a valuable direction in agent learning. The paper presents extensive experimental results, utilizing both open-source and proprietary language models, highlighting the challenges for many current LLMs. The authors provide detailed descriptions of each task, including observation and action spaces, which enhances the clarity of the work. The selected tasks cover a wide range of observation and action modes.

### Weaknesses
While learning from language feedback could be valuable for tasks requiring human interaction, the current paper relies on a limited set of 4-20 language templates per task, which may lack the diversity needed to effectively mimic authentic human feedback. Additionally, other benchmarks, such as MINT bench, already provide language feedback during interactions, which weakens the novelty of LLF-Bench. The selection of tasks may also be less aligned with current missions of agent-human interaction, further impacting the applicability of the language feedback approach.

### Questions
1. **Task Selection for Rich Human Feedback**: To fully utilize the potential of language feedback in human-to-human learning contexts, would it be more effective to include tasks that are inherently cooperative? For example, humans have powerful intuitions on space and physics, which encourages more visual and physics related tasks, such as the current Metaworld task.

2. **Varied Levels of Language Feedback**: Could the paper explore different levels of language feedback detail? For example, as shown in Figure 1, level-1 feedback might provide an abstract comment such as "The arm is moving too fast/wrong direction"; level-2 feedback could include directional guidance, such as "The arm should move slightly left"; level-3 feedback might provide detailed statistics, as in the example. This approach would better mimic the often abstract nature of real-world human feedback, potentially adding practical value to the benchmark.

3. **Diversity in Language Feedback**: Generating feedback from a limited set of templates lacks the diversity seen in feedback generated by large language models, which can be easily prompted to create varied responses in different styles and with minimal repetition. Could using an LLM to generate feedback improve feedback diversity and realism? For example, using LLMs to first generate more templates, directly using LLMs with different seed prompts to generate diverse feedbacks, or using different LLMs to generate feedbacks.

4. **Effect of Vision Input on Performance**: In Table 4, adding vision input at the input stage seems to harm performance. Even though the authors claim vision input would help after removing future reward, there are still three tasks with significant performance drop. Could the authors provide an explanation for why vision input seems to negatively impact outcomes in these tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose a new framework for evaluating LLM agents, called LLF-Bench. The framework consists of a variety of tasks that require an LLM agent to solve, and provide both scalar and textual feedback for actions taken by the agent.

### Strengths
The benchmark consists of diverse domains, from recommendation, to navigation, to even robotic manipulation (which to my knowledge,  is not typically used as a evaluation domain for LLM agents). The inclusion of textual feedback in addition to scalar rewards also sets it apart from existing benchmarks.

### Weaknesses
I think the set of tasks chosen is diverse and reasonable. However, I do think the benchmark would be much more valuable if it also included a realistic dialogue task, such as negotiation or persuasion. I think such tasks might be the primary examples where some form of RL fine-tuning is needed. 

Another potentially bigger concern I have is in the empirical evaluation. While the authors compare against different LLMs, they only consider one method, namely a Reflexion-based agent. To truly understand the nuances of each considered task, I think the authors need to also evaluate different non-RL and RL based methods, including ReAct [1] and LATS [2], respectively. 

[1] https://arxiv.org/pdf/2210.03629

[2] https://arxiv.org/pdf/2310.04406

### Questions
(1) While I see that including textual feedback provides value to training LLM agents, it would be nice to better understand its computational cost. Namely, it would be interesting to see how much slower generating trajectories when the environment must provide textual feedback beyond the scalar reward. 

(2) As alluded to earlier, I think it is important to consider more baselines beyond just a Reflexion-based agent. That way, we can better understand which tasks are difficult and may require explicit planning or multi-step reasoning to solve.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies learning from language (LLF) setup where AI agent learns from interactive language feedback. To facilitate research in this direction, the authors propose LLF-Bench, providing a reproducible setup for learning from language feedback. LLF-Bench includes diverse tasks ranging from poem writing to low-level robot control. Also, it features diverse feedback types and randomization of the environment to prevent overoptimization towards a specific setup.

### Strengths
1. The proposed benchmark, LLF-Bench, includes a diverse set of tasks ranging from text writing to robotics. 
2. The benchmark is simple and easy to use using the commonly adopted OpenAI Gym-like API.

### Weaknesses
1. While “proposing a formal setup of Learning from Language Feedback (LLF)” is claimed to be one of the main contributions, LLF has been widely studied in prior work[1, 2] and the formulation in the paper does not seem to be novel compared to those.
2. The experiments mainly focus on quantitatively assessing existing language models within the benchmark, with limited analysis specific to LLF. For instance, how does LLF compare to using scalar feedback alone in exact performance? How does randomizing the textual description of task instructions and feedback impact the agent's performance?
3. The authors emphasize the challenges of designing rewards and the risk of unexpected outcomes from suboptimal rewards in reinforcement learning. However, similar issues could arise for LLF if the feedback provided is suboptimal or misaligned with the task instruction.
4. Despite being presented in main Table 4, it is not clear why experiments with image-based observations are meaningful for understanding LLF, and no justification is currently provided regarding this.

[1] Reflexion: Language Agents with Verbal Reinforcement Learning. Shinn et al. NeurIPS 2023.

[2] Yell At Your Robot: Improving On-the-Fly from Language Corrections. Shi et al. RSS 2024.

### Questions
1. How do agents using only scalar rewards compare to LLF agents? Does it show a slower learning speed?

2. In Figure 1, future feedback includes a very detailed coordinate for the next action. Does the environment assume the availability of optimal action in feedback? 

3. In Sec 3.2, while it is mentioned that LLF-Bench implements the task instruction as part of the environment rather than part of the agent, it is not clear how this paradigm makes a technical difference in the environment and contributes to “the development of more general learning agents” as mentioned in Sec 3.2. 

4. In Sec 2.2, while it is claimed that the reinforcement learning agent has to not only learn to solve the task but also learn to understand the task’s objective, this might not be the case for training agent that has language understanding capability (eg., LLM) using reinforcement learning with instruction, which is the prevalent type of agent in this paper.

### Soundness
2

### Presentation
2

### Contribution
2
