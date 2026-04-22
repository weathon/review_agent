# ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
As large language models (LLMs) become increasingly integrated into daily life, there is growing demand for AI assistants that are not only reactive but also proactive and personalized. While recent advances have pushed forward proactivity and personalization individually, their combination remains underexplored. To bridge this gap, we introduce ProPerSim, a new task and simulation framework for developing assistants capable of making timely, personalized recommendations in realistic home scenarios. In our simulation environment, a user agent with a rich persona interacts with the assistant, providing ratings on how well each suggestion aligns with its preferences and context. The assistant’s goal is to use these ratings to learn and adapt to achieve higher scores over time. Built on ProPerSim, we propose ProPerAssistant, a retrieval-augmented, preference-aligned assistant that continually learns and adapts through user feedback. Experiments across 32 diverse personas show that ProPerAssistant adapts its strategy and steadily improves user satisfaction, highlighting the promise of uniting proactivity and personalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ProPerSim, a simulation framework for developing AI assistants that integrate proactivity and personalization in home environments. It also presents ProPerAssistant, a retrieval-augmented AI assistant trained using ProPerSim to adapt its strategies based on user feedback. ProPerAssistant demonstrates improved user satisfaction across 32 diverse personas, highlighting its ability to adapt to individual preferences and contexts. This work bridges the gap between personalized and proactive AI systems, advancing the development of context-aware, user-centric recommendations.

### Strengths
1. The paper addresses an important gap by combining personalization and proactivity, offering a new perspective on AI assistant design.

2. The simulation framework is well-structured and supports realistic evaluation through diverse user personas and rich feedback loops.

3. ProPerAssistant demonstrates measurable improvements in user satisfaction, showcasing its practical applicability and potential for real-world integration.

### Weaknesses
1. The dataset mentioned in the paper is neither provided in the supplementary materials nor shared with an anonymous link. This raises concerns about whether the dataset will be made publicly available in the future.

2. While the paper claims to integrate proactivity and personalization as its motivation, the focus of both the dataset and the proposed method appears to be predominantly on personalization. The treatment of proactivity seems rigid, as it only involves providing suggestions at fixed time intervals rather than adapting to users' behaviors. Such an approach lacks flexibility and does not align well with real-world user scenarios.

3. The paper includes comparisons with only a limited set of baselines, specifically a few variations of ProPerAssistant. However, it does not compare the proposed method against other existing personalization baselines. This omission raises concerns about the effectiveness of the proposed approach.

4. The authors report performance results based on only one type of large language model (LLM) and do not conduct ablation studies on other LLMs. This raises doubts about the generalizability of their method and dataset across different models.

### Questions
See weakness.

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
This paper investigates the proactive personalization problem in the conversational ai scenario. Due to the rare study in this track, this work proposes ProPerSim to simulate this scenario and further propose ProPerAssistant which coupled with a memory system and a preference-aligned training strategy to proactively generate personalized interventions for user.

### Strengths
* The research scenario is very interesting and to the community’s focus.

* The proposed ProPerSim simulation is concrete and consider personality, environment, user’ memory.

* The proposed ProPerAssistant learns from user feedback via intermediate state and user preference is novel.

### Weaknesses
Some minors from practical:

* The baseline selections are not convincing, the ProPerSim would of course outperforms than no memory, AR memory and ARS memory since ProPerSim are trained based on these configurations and contain them all. The comparisons here are more like ablation. Authors may consider different preference alignment comparisons etc. 

* The experiments lack the evaluation of intervention. Since it’s proactive personalization, it would be straightforward to see what the rate of successfully proper interventions is.

* Latency/efficiency analysis on the daily data generation should be included.

### Questions
* How does adopting a purely time-based framing of proactivity potentially limit the assistant's ability to respond to specific user-triggered events (which prior work used) that might require instantaneous intervention, rather than waiting for the next predetermined timestep?

* Is the challenge of improving performance for these complex personas fundamentally attributable to the memory system's inability to retain the necessary temporal and content granularity required to meet these highly specific demands, given that the memory relies heavily on time block summarization?

### Soundness
4

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
3

### Summary
In this work, the authors introduce a simulation framework and benchmark for developing AI assistants that are both proactive and personalized. The framework models a simulated home environment where a user agent performs daily routines, with multiple distinct user personas and realistic activity patterns generated using GPT-4o. The assistant’s objective is to determine when and what actions to recommend to the user. To address this, the authors propose a retrieval-augmented, preference-learning assistant that maintains a memory of past interactions, generates multiple recommendation candidates, and learns from user feedback through preference optimization. Extensive experiments demonstrate that this approach substantially improves user satisfaction across a wide range of simulated personas.

### Strengths
The main strength of this paper lies in its timely motivation to create assistants that are both proactive and personalized. The authors create extremely controlled agent setup to study the design of such systems. Under this setup, they consider multiple constraints that the assistant need to adhere to, creating an extremely rich setting to ablate different scenarios. The authors also validate their proposed setup with a human evaluation study, which increases validity of their claims about naturalness and persona alignment.

In addition, the authors propose intuitive system that builds on 2 key components: (a) structured memory storing past experiences, and (b) DPO training based on user feedback. Through extensive evaluation, their proposed system shows consistent improvement across all their persons. Overall, this is an extremely strong paper that will be of great interest to the scientific community.

### Weaknesses
I don't see many weaknesses with the paper. In fact, the authors have gone to extreme details in designing their system.

I have a few questions with the experimental setup.

(a) The definition of proactivity is narrow. Under the current definition, proactivity is reduced to timing control, not anticipation of latent goals or needs of the user. This is because the authors define it as whether an action should be taken at a particular time. However, by proactivity, I would also define it in terms whether an action can return a reward faster or change the trajectory of decisions on the user. How can that be modeled under the current framework?

(b) Noisy or delayed feedback: One key missing component in the current setup is that it assumes almost perfect feedback from the user always. User Preferences can be noisy or even delayed. How would such noise be incorporated into such a setup? And how would the proposed system change under presence of such noise?

(c) LLM-as-a-judge is performed with Gemini 2.0 Flash. The rubric used in table 6 is extremely broad. How would the results change when the evaluation is performed with a different LLM?

### Questions
Please check my questions above.

### Soundness
3

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
This paper created a benchmark for simultaneously evaluating the proactivity and personalization of an agent. The benchmark is a simulated environment where a user which is powered by an advanced LLM conducts daily activities, simulating human behaviors. At the same time, an agent observes the activities of the user and decides whether or not to interact with the user by providing recommendations. The agent will receive a score of its recommendation based on the user's personalized rubric evaluated by an LLM. The users are created to cover a broad set of persona. Besides the benchmark, the authors also provide an agent that performs well under this simulated environment by utilizing preference alignment and retrieval-augmented generation.

### Strengths
1. This paper considers the problem of benchmarking both proactivity and personalization of an agent. It can augment the existing evaluations which mainly focus on either proactivity or personalization. 

2. The presentation of this paper is clear and easy to follow.

### Weaknesses
1. The definition of proactivity is too narrow. In this proposed benchmark, proactivity is mainly described as the ability to initiate the interaction with the user by providing recommendations. On the contrary, previous works on benchmarking proactivity covers a broad range of tasks including coding, writing, and daily life scenarios. Also, the quantification/evaluation of proactivity seems to be mixed into personalization. For example, how can one infer from the evaluation results that the agent is proactive but bad at personalization?

2. The proposed ProPerAssistant can be regarded as a baseline of this benchmark instead of a contribution, since there is no new innovation in this agent which mainly based on memory retrieval and preference optimization. For example, the statement at line 406 "Notably, ProPerAssistant achieves this without relying on computationally expensive reinforcement
learning methods, instead leveraging an efficient DPO-based approach to learn user preferences". This is the contribution of DPO. And I think the statement is a little bit misleading since the agent is trained with online DPO, which necessitates the collection of online trajectories, it is still expensive.

3. The benchmark is not practical to use. the simulation and evaluation are both powered by advanced LLMs which induces lots of API costs. And since the evaluation is based on LLM-as-a-judge. There can be bias or hallucination. Although in section 4.3, the authors conducted manual checking of the actions and evaluations, it is a limited checking and I still concern the reliability of the benchmark.

4. As an agent capable of actively engaging the interaction and infer the personality of a user based on interaction history, it should be able to generalize to users with other persona without re-training. But there is limited discussion (not sure the first paragraph in section 6.4 is under this setting?) on the generalization ability of ProPerAssistant.

### Questions
I also raised these questions in the weaknesses part. Please refer to that part for more contents.

1. How can one infer the separate performance of proactivity or personalization from the evaluation results? It seems low proactivity or bad personalization will decrease the score. I know the goal of this benchmark is to evaluate the joint performance. But it is also crucial to know what the agent is bad at currently for future improvement.

2. What is the setting of Adaption Across Diverse Personas in section 6.4? What's the training set of users and what's the evaluation set?

### Soundness
2

### Presentation
3

### Contribution
2
