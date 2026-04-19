# Simulating Human-like Daily Activities with Desire-driven Autonomy

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Desires motivate humans to interact autonomously with the complex world. In contrast, current AI agents require explicit task specifications, such as instructions or reward functions, which constrain their autonomy and behavioral diversity. In this paper, we introduce a Desire-driven Autonomous Agent (D2A) that can enable a large language model (LLM) to autonomously propose and select tasks, motivated by satisfying its multi-dimensional desires. Specifically, the motivational framework of D2A is mainly constructed by a dynamic $Value\ System$, inspired by the Theory of Needs. It incorporates an understanding of human-like desires, such as the need for social interaction, personal fulfillment, and self-care. At each step, the agent evaluates the value of its current state, proposes a set of candidate activities, and selects the one that best aligns with its intrinsic motivations. We conduct experiments on Concordia, a text-based simulator, to demonstrate that our agent generates coherent, contextually relevant daily activities while exhibiting variability and adaptability similar to human behavior. A comparative analysis with other LLM-based agents demonstrates that our approach significantly enhances the rationality of the simulated activities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Desire-driven Autonomous Agent (D2A), a framework for simulating human-like daily activities based on intrinsic motivations rather than explicit tasks or external rewards. Inspired by Maslow's Theory of Needs, D2A prioritizes actions that fulfill a hierarchy of desires (e.g., physiological, social, self-actualization), allowing it to autonomously select actions that align with its motivational framework. This desire-driven approach contrasts with traditional task-oriented agents by focusing on fulfilling internal motivations, which provides the agent with the capacity for more adaptable and human-like behavior.

### Strengths
**1. Proactive Action Based on Intrinsic Motivation**: The proposed D2A framework demonstrates an ability to proactively initiate actions driven by intrinsic motivations. Through the integration of a value system and a desire-driven planner, the framework establishes a dynamic interaction between desires and actions, wherein lowered desire values trigger corresponding activities to restore balance. This mechanism, though relatively simple, allows the agent to engage autonomously in daily activities in a manner that mirrors proactive human behavior, setting it apart from purely reactive or task-driven models.

**2. Human-Inspired Intrinsic Motivations Across Life Dimensions**: Unlike recent approaches that focus on intrinsic motivations for exploration or collaboration, the D2A framework offers a multi-dimensional model inspired by human needs. By integrating eleven desire dimensions (e.g., physiological, social, and self-fulfillment needs), D2A provides a broader, more human-like motivational structure. This approach goes beyond typical reward-driven or exploratory motivations by simulating daily life activities that dynamically balance internal desires, reflecting human motivation patterns more authentically. This novelty enhances the agent's potential for replicating realistic, human-inspired behaviors within single-agent environments.

### Weaknesses
**1. Inconsistent Application of Maslow’s Theory**: While D2A is presented as being inspired by Maslow’s Theory of Needs, it does not implement the theory’s hierarchical structure. According to Maslow, higher-level desires are pursued only once lower-level needs are met, but D2A treats each desire independently, allowing the agent to pursue higher-level needs without satisfying foundational ones. This weakens the theoretical foundation and could make the agent's behavior feel less authentically human-like.

**2. Overly Simplistic Desire-Action Dynamics**: Despite its innovative multi-dimensional motivational structure, the D2A framework uses a straightforward linear deduction of desire values to simulate the fluctuation of needs. This approach falls short of capturing the organic variation in human desires, which often intensify or wane in response to context, time, or recent actions. By exploring more complex dynamics—such as non-linear decay, situational adjustments, or time-of-day cycles—the framework could better reflect realistic interactions between desires and actions. The current simplicity reduces the authenticity of the agent's behavior, limiting the depth of its desire-driven model.

**3. Narrow Focus on Physiological Desires in Single-Agent Setting**: The experimental results in the paper primarily emphasize physiological desires, with minimal exploration of higher-level motivations such as social connectivity or self-actualization. Additionally, the study tests only one agent, limiting insights into potential interactions or complex social behaviors that might arise in multi-agent settings. These restrictions make the results narrowly focused and reduce the paper's ability to demonstrate the full range of behaviors that D2A could potentially simulate, ultimately constraining the framework’s demonstrated impact.

**4. Opaque GPT-4o Evaluation Methodology**: The evaluation of human-likeness using GPT-4o lacks transparency. Key details—such as the prompts used, scoring consistency, and validation of the assessment criteria—are not provided, making it difficult to gauge the robustness of the results. This lack of methodological clarity limits confidence in whether the evaluation effectively captures nuanced human-like behavior or merely reflects surface-level patterns.

**5. Limited Qualitative Evaluation of Generated Action Sequences**: The paper lacks an in-depth qualitative analysis of the action sequences generated by D2A, making it difficult to assess the framework’s true effectiveness. The brief, simplistic sequence shown in Appendix O does not illustrate the model’s potential for creating complex, dynamic routines, leaving the impact of D2A’s design unclear. More detailed, varied sample sequences, along with thoughtful discussion, would provide stronger evidence of D2A's capabilities in simulating realistic human-like behaviors.

### Questions
Please answer the issues mentioned in the weaknesses.

### Soundness
3

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
2

### Summary
The paper introduces a Desire-driven Autonomy framework for LLM-based agents to simulate human-like daily activities. The framework is inspired by Maslow's theory of needs and includes 11 dimensions of human-like desires. The Desire-driven Autonomous Agent (D2A) operates based on intrinsic motivations, autonomously proposing and selecting tasks that fulfill its motivational framework. The authors developed a flexible text-based activity simulator using Concordia components, supporting various agent types and textual environments for reliable interaction and evaluation. They conducted simulations in a detailed textual home environment with a Game Master providing relevant observations. The experiments demonstrated that D2A generates appropriate activities effectively and efficiently, achieving described desires. A comparative analysis with three baseline approaches (ReAct, BabyAGI, and LLMob) showed that D2A generates more natural, coherent, and plausible daily activities.

### Strengths
- The proposed framework allows agents to operate based on intrinsic motivations, which is a significant departure from existing task-oriented AI agents that rely on explicit instructions or external rewards. 
- The paper is well written and has nice figures.
- The authors conducted a comprehensive comparative analysis with three baseline approaches (ReAct, BabyAGI, and LLMob) to evaluate the effectiveness of their framework. The development of a flexible text-based activity simulator using Concordia components is another strength of the paper.

### Weaknesses
I think there are some weaknesses in this paper:
- Limited Technological Innovation. The paper primarily focuses on the conceptual framework and theoretical underpinnings of the desire-driven autonomy approach. While the idea of using intrinsic motivations inspired by Maslow's theory of needs is innovative, the technological implementation details might not be as groundbreaking or novel within the recent advancements in the field of AI and LLMs. The work seems in the flow of LLM agents, while I think it is more of a LLM project rather than a technologically-solid paper.
- Although the paper demonstrates the effectiveness of the D2A agent in a specific textual home environment, there might be questions about its generalization and scalability to other environments or domains. 
- The authors have established a set of concepts, such as human needs, desires, characteristics, and values, to guide the model's behavior. Although I understand the authors' intent to direct the generation of human-like daily activities, I am uncertain about the definition and composition of these intermediate variables. They are determined by the authors without sufficient psychological support or experimental validation to substantiate the overall design. How is the overall pipeline designed? I.e., for the five-level Maslow model, why you further define 11 desire dimensions?
- The experiment is limited. The testing environment is confined to a single room containing a kitchen, living area, bedroom, and bathroom. It is unclear whether there is any randomness or variation between each epoch's setup, such as the rooms in the house or the items within the room. The experiments should be conducted in a wider variety of settings to minimize the impact of environmental bias and to get general ideas. How many different scenes or settings are included in the experiment? Also, for LLMs, have the authors studied how the prompt design will influence the overall results?

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is concerned with simulating realistic trajectories of human activities in household environments. The authors introduce a text-based home environment as well as an LLM-based agent whose outputs aim to be activity traces as diverse as possible.

Inspired by Maslow's theory of needs, their agent incorporates 11 desire scalars (with each a target value, variable across agents). These desires are split according to the levels of Maslow's hierarchy. The desired levels, current levels, previous activities, and other environment information is provided to the LLM-agent in a tree-of-thought framework to generate the next activity.

The authors find the generated trajectories are deemed more likely by judge LLMs, and decrease dissatisfaction compared to other LLM-agent baselines.

### Strengths
## Originality

The paper is quite original, as I have not seen Maslow's hierarchy of needs used in the context of a text agent.

## Quality

The paper is well-written, experiments are quite well-designed, several seeds are provided to account for variability. The figures are nice, and the main one does a good job of summarizing how the agent works. The results of the paper support the claims made in the introduction and the abstract.

## Clarity

The paper was easy to follow and the points are clear and easy to grasp.

## Significance

The paper demonstrates a recipe for creating more realistic human daily trajectory activities. I can see the type of agent developed here being useful for other applications, such as creating LLM-based non-playable characters in video games.

### Weaknesses
* I am not completely convinced of the end-goal of this paper, specifically, building sequences of human activities. I see the authors justifying this goal in the potential for generating data for psychological, economic or sociological academic study. However, the validity of the generated behavior with respect to at least one downstream application is not investigated in the paper. How to make sure the data generated is useful in these contexts?
* The introduction also briefly argues that building agents with behaviors aligning with human ones will guarantee their intelligence (the Turing test argument). But unfortunately this does not seem to be the case; I cannot see how giving agents simulated desires will make them score higher on GSM8K for instance.
* A human-judge evaluation of the validity of the AI judge would be nice (although I am still pretty convinced of the comparison results)
* I think there is a methodological flaw in the design of the human reference agent: the human is (as far as I understood) given a list of the desires, and the criteria is whether the agents are able to minimize dissatisfaction for those same desires. A better reference would be a dataset of real human activities on a stay-at-home day;

### Questions
* My foremost question would be about the practical use of generating human daily activity data. I do not know this subject, and it might be the focus of an entire subcommunity which I am not familiar with. It would be important for the authors to elaborate on this point, and I am ready to reconsider my score if the demonstration is convincing;
* What do the statistics of the generated activities look like?
* Is there a list of predetermined activities one may do? (predefined action space?)

## Notes

* I think that the naturalness of activities might also come from the fact that desires are satisfied for quite some time giving the agent the opportunity to concentrate on more diverse activities than in the baselines.

## Suggestions

* I believe this paper should cite Park et. al 2023 (https://arxiv.org/abs/2304.03442), a seminal work in human behavior simulation, and Colas et. al. 2023, which presents an intrinsically-motivated agent operating on other principles than Maslow's hierarchy (https://arxiv.org/abs/2305.12487).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a novel framework, the Desire-driven Autonomous Agent (D2A), designed to simulate human-like behaviour in daily activities. Traditional task-oriented AI agents primarily operate based on specific instructions or external rewards, but this approach often limits their ability to display intrinsic motivations similar to humans. This paper proposes an alternative: a motivation system inspired by Maslow’s hierarchy of needs, enabling the agent to autonomously generate and select activities based on desires like social interaction, self-care, and personal fulfilment. The D2A framework uses a system of 11 dimensions representing different human desires. The agent evaluates its current state and intrinsic motivations to decide on activities that align with its desires, generating more coherent and varied behaviours compared to existing models. The study uses Concordia, a text-based simulator, where a Game Master provides the environmental context for D2A to interact in a simulated household environment. The results suggest that D2A successfully simulates human-like activities by aligning with intrinsic motivations, which opens new avenues for developing desire-driven agents in various applications.

### Strengths
Originality: The paper presents a novel approach to simulating human-like daily activities using a desire-driven framework inspired by Maslow’s hierarchy of needs. Unlike traditional AI agents that rely on specific instructions or task-based rewards, the Desire-driven Autonomous Agent (D2A) framework introduces intrinsic motivation as the driving factor. This approach is unique in that it models a human-like motivational system, enabling the agent to select actions autonomously based on intrinsic desires rather than predefined goals.

Quality: The paper is good quality with descriptive figures and clear results.

Clarity: The paper is well structured with a clear distinction provided between their proposed method and past methods and how theirs performs better.

Significance: This work holds significance for fields focused on human-like AI, agent-based simulations, and real-world applications requiring adaptive behaviour. By adopting an intrinsic motivation model, this framework could be used for applications such as social robotics, interactive gaming, and assistive technology, where human-like adaptability and engagement are essential.

### Weaknesses
There are a few weaknesses.
In Section 3, Problem Formulation, the math is not very clear, and it is also not explained in more detail how the activity distribution could be generated.

In Section 6.3.1, Naturalness, Coherence and Plausibility are used to evaluate the activity sequences, but these three dimensions seem to have been picked arbitrarily and I am not sure if they are enough to rigorously test the outputs.

Evaluation is done using GPT-4o but how are we to ensure that these evaluations can be taken at face value. I think the paper would do well to do a human verification of these evaluations and how reliable they are.

### Questions
Listed in the weaknesses section itself.

- Make section 3 more clear by adding more details.
- Give better reasoning for using the 3 dimensions of naturalness, coherence and plausibility.
- measure the reliability of GPT-4o evaluations.

### Soundness
2

### Presentation
2

### Contribution
2
