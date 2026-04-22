# CogniPair: GNWT-inspired cognitive architecture for generative agents for Social Pairing -  Dating & Hiring Applications

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 4, 4, 8, 4

## Abstract
Current large language model agents lack authentic human psychological processes necessary for genuine digital twins. We present the first computational implementation of Global Workspace Theory (GNWT), creating agents with multiple specialized sub-agents (emotion, memory, social norms, planning, goal-tracking) coordinated through a global workspace broadcast mechanism. This architecture allows agents to maintain consistent personalities while evolving through social interaction. Our CogniPair simulation platform deploys 551 GNWT-Agents for speed dating interactions, grounded in real data from the Columbia University Speed Dating dataset. Evaluations show strong psychological realism, with agents achieving 72\% correlation with human attraction patterns and outperforming baselines in partner preference evolution (72.5\% vs. 61.3\%). Human validation studies confirm our approach's fidelity, with participants rating their digital twins' behavioral accuracy at 5.6/7.0 and agreeing with their choices 74\% of the time. This work establishes new benchmarks for psychological authenticity in AI systems and provides a foundation for developing truly human-like digital agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper implements AI agents based on the Global Workspace Theory to model human psychological processes that are essential for developing genuine digital twins. They test their agent in the context of dating. Their testbed involves 551 GNWT-Agents for speed dating interactions and shows that the agents achieve a high alignment with humans. They also conduct a human validation study by asking human participants to evaluate AI versions of themselves, which supports that their agents demonstrate a high correlation with the original humans.

### Strengths
- They develop AI agents that implement human psychological processes necessary for creating genuine digital twins.
- They involve many AI agents (551 GNWT-Agents) for evaluation
- Their result reveals that their agents achieve a higher alignment with humans compared to other techniques.
- They also conduct a human validation study to support their argument.

### Weaknesses
- Their results might be just due to using more computational resources than other baselines. For example, they may have achieved better outcomes in the Multi-Agent Debate setup simply by using a larger number of debating agents. Similarly, combining the multi-agent debate framework with memory-enhanced techniques could have demonstrated the performance gains. The paper should compare the resource usage of their method and competing approaches. It is not clear that the higher alignment with humans is from their proposed design (i.e., the Global Workspace Theory) rather than from simply increased resource allocation.

- Although it is true that their agents show a significantly higher alignment with human behavior, a 72% correlation with human attraction patterns is not sufficient I think. Moreover, the evaluation is limited to a single domain (dating). These points limit the overall contribution of the paper.

- I appreciate their efforts to include human validation studies. However, these studies could have been conducted more rigorously. They only evaluate their own agents and not the baseline agents. Without comparing against baselines, it is difficult to determine whether the reported values genuinely reflect superior human alignment.

- Minor: There is no content for section A.34.

### Questions
Please see the weaknesses

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
4

### Summary
This paper introduces a computational implementation of Global Workspace Theory (GNWT) for building psychologically realistic AI agents. The proposed architecture integrates multiple specialized sub-agents (emotion, memory, planning, social norms) coordinated via a global workspace, enabling consistent yet adaptive personalities. Using a speed-dating simulation with 551 agents grounded in real human data, the model shows strong psychological realism—achieving a 72% correlation with human attraction patterns and outperforming baselines. Human validation further confirms behavioral fidelity. Overall, the work sets a new benchmark for psychological authenticity in digital twin and human-like AI research.

### Strengths
The reviewer as a researcher in computational cognitive architecture likes this work very much, which tries to construct agents based on some theories from the cognition field, e.g., global workspace theory, instead of searching modules over a vast and discrete space for better performances on math or code tasks.

The strengths are listed as follows:

(1) Start to seriously consider psychological behavior and social behavior gaps, and propose to establish agents based on global workspace theory, by modelling emotion, social norms, planning, etc.

(2) The agent simulation results seem to be very correlated with human data, which may verify the effectiveness of this architecture.

### Weaknesses
As a research paper to be publised on top conference, there are several concens to be clarified:

(1) The method or technology is very vaguly introduced in this paper. I have no deep impression on how to implement the architecture in details, e.g., how the core modules (emotion, social norms, etc.) interleave each other? Please use mathmatical formula to illustrate these interactions.

(2) Why the authors choose emotion, social norms, memory, goal tracking, memory as the main modules, instead of other modules, like attention, reasoning?

(3) Figure 1 and Figure 2 are too "abstract" so that it shows a very conceptual framework for the audience. Such framework can be obtained from many previous cognitive architecures. It is best to repaint these figures and give much more info. to others.

(4) How to learn the parameters by using Eqn. 3? Give more details on the implementations.

(5) Although the reviewer encourages to propose agents based on previous theories, the proposed method has to compare with the state-of-the-art self-evolving/improving agents, e.g., ADAS, AFLOW (maybe some other methods), to further demonstrate its better performances.

### Questions
As the reviewer mentioned before, this work is facinated by its motivation and experimental results. However, there are so many details missing that the authors should answer these questions:

(1) What is the motivation to choose those five modules, i.e., memory, planning, goal tracking, emotion, social norms, instead of others?

(2) Explain more technical details to convince the reviewer that there is a lot of things in this work.

(3) Summarize the unique advantages of the proposed agent compared with other self-evolving agents instead of doing more experiments during rebuttal.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes CogniPair. It introduces the Global Workspace Theory into the agent workflow design by designing submodules with a workspace for an agent. Then, they conduct experiments by deploying the agents in dating interactions and show that these agents outperform baseline agents without global workspace in terms of human correlation.

### Strengths
- The motivation of using the Global Workspace Theory to design agents and study their social interactions is novel and interesting.
- The setting of dating and hiring are interesting problems and a topic worth studying using agent workflows. I think the evaluation with digital twins is interesting.

### Weaknesses
- the design space lacks enough scientific support since simulating human brain is still an open-ended problem. For example, in the paper "Parallel Processing Modules（Similar to Unconscious Modules) " contains modules such as memory / emotion etc, but are not actually verified. What these modules should be, how each of them should work, and also how they should interact to simulate human brain are all unclear. While I appreciate the author's attempt to apply these concepts to actual agent design, I think the actual technical part is not well studied enough to make the approach valid.

### Questions
- I'm curious what the author think about how to introduce concepts in psychologic / cognitive science to the design of agentic workflow?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents CogniPair, a cognitive architecture for generative agents inspired by the Global Neuronal Workspace Theory (GNWT) from cognitive neuroscience. CogniPair models social cognition as a process of dynamic broadcasting between specialized modules (Emotion, Intention, and Perspective buffers) coordinated through a Global Workspace controlled by a probabilistic broadcast mechanism.

### Strengths
This is a well-written, comprehensive paper that operationalizes the GNWT within a generative-agent architecture. The system design, evaluation, and cross-disciplinary framing are strong, and the work makes a meaningful contribution to socially aware cognitive modeling in LLM agents.

### Weaknesses
While the paper states that demographic data are ingested from the Columbia Speed Dating dataset or synthetic balanced profiles, it does not specify which demographic attributes are used (e.g., gender, age, cultural background) or how they are normalized, balanced, or anonymized.

### Questions
Could you specify the exact demographic attributes used and describe how they were pre-processed (e.g., binning, balancing, anonymization)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces CogniPair, a novel platform featuring GNWT-Agents, which represent the first computational implementation of the Global Workspace Theory (GNWT) for generative agents. Designed to address the lack of authentic human psychological and social dynamics in current LLM agents, the GNWT-Agent architecture coordinates specialized sub-agents (e.g., emotion, memory) via a global workspace broadcast. This mechanism enables agents to maintain a consistent personality while dynamically evolving their preferences through social interactions. The platform was successfully applied to social pairing tasks like speed dating and hiring, achieving a 72% correlation with human attraction patterns and significantly outperforming baselines in predicting partner preference evolution (up to 72.5% accuracy).

### Strengths
1. Pioneering Cognitive Architecture (First GNWT Implementation)
The paper presents the first computational implementation of the Global Workspace Theory (GNWT) for generative agents. This is a significant theoretical breakthrough, providing LLM agents with a human-like "consciousness" mechanism (the global workspace broadcast) that integrates information from specialized sub-agents (e.g., emotion, memory), resulting in agents with high Psychological Fidelity.

2. Achieves Authentic and Dynamic Personality Evolution
CogniPair successfully overcomes the "static" nature of previous LLM agents. By utilizing the GNWT broadcast mechanism, the agents can dynamically update their internal states and preferences based on social experience. This leads to superior performance in modeling partner preference evolution, achieving 72.5% accuracy and showing that the agents can genuinely learn and adapt over time.

3. High Social Realism and Practical Generalization
The agents demonstrate strong social simulation capabilities, achieving a 72% correlation with human attraction patterns in the speed dating scenario. Furthermore, the architecture proves its generalization capability by successfully transferring the model to different social decision contexts, such as job interviews (with 81% accuracy), highlighting its broad practical value for various social pairing applications.

4. Strong Validation Through Human Studies
The model's psychological authenticity is validated through human trials. Participants rated the behavioral accuracy of their digital twins at 5.6/7.0 and agreed with their twins' choices 74% of the time. This high degree of human endorsement significantly boosts the credibility and real-world applicability of the CogniPair framework.

### Weaknesses
1. The title is not smooth.
2. The architecture design chooses emotion, memory and social norms. But why do you choose to save value and personality into memory, not in a separate module just like "emotion"? In dating scenarios, value and personality also play very important roles. 
3. The cognitive architecture is simplified, and there is no Theory of Mind. I understand that building such a big system is difficult, but Theory of Mind is inevitable when talking about dating tasks. 
4. Missing Analysis: The paper lacks a detailed quantitative analysis of the latency and cost increase (e.g., in terms of API calls or tokens per decision) when moving from a baseline agent to the full GNWT-Agent.
5. Missing SOTA Social Agents: The comparison largely focuses on Reflexion-Style Agents. The paper omits direct comparisons with other state-of-the-art architectures specifically designed for complex social interaction, such as those that explicitly model Theory of Mind (ToM).
6. Missing GNWT Mechanism Baselines: To truly validate the design, the paper should include ablations comparing the implemented GNWT features against alternative designs, such as using local broadcasting (instead of global) or different attention/priority mechanisms in the global workspace.
7. Insufficient Behavioral Granularity: The human validation focuses on final choice agreement and overall accuracy (5.6/7.0). It lacks fine-grained human evaluation of micro-behaviors, such as the naturalness of emotional expression or conversational flow.

### Questions
See above. I might change my rating, but I need the authors' rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
