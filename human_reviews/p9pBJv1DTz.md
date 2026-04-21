# MindAgent: Emergent Gaming Interaction

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 3, 3, 8

## Abstract
Large Language Models (LLMs) can perform complex scheduling in a multi-agent system and can coordinate agents to complete sophisticated tasks that require extensive collaboration. However, despite the introduction of numerous gaming frameworks, the community lacks adequate benchmarks that support the implementation of a general multi-agent infrastructure encompassing collaboration between LLMs and human-NPCs. We propose a novel infrastructure--- MindAgent---for evaluating planning and coordination-emergent capabilities in the context of gaming interaction. In particular, our infrastructure leverages an existing gaming framework to (i) require understanding of the coordinator for a multi-agent system, (ii) collaborate with human players via instructions, and (iii) enable in-context learning based on few-shot prompting with feedback. Furthermore, we introduce CuisineWorld, a new gaming scenario and its related benchmark that features a multi-agent collaboration efficiency and supervises multiple agents playing the game simultaneously. We have conducted comprehensive evaluations with a new auto-metric collaboration score CoS for assessing the collaboration efficiency. Finally, MindAgent can be deployed in real-world gaming scenarios in a customized VR version of CuisineWorld and adapted in the broader "Minecraft" gaming domain. Our work involving LLMs within our new infrastructure for general-purpose scheduling and coordination can elucidate how such skills may be obtained by learning from large language corpora.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a new text-based gaming multi-agent benchmark CuisineWorld, inspired by the video game Overcooked, proposes a method MindAgent for central control of multiple agents using LLM by designing prompts, and conducts experiments with the benchmark and the method including human experiments.

### Strengths
- The paper introduces a new text-based multi-agent cooperation benchmark, with a vivid visual appearance.

- The paper conducts experiments involving more than 2 agents and human.

- The paper shows some efforts in transferring the method in real-world gaming scenarios Mindcraft.

### Weaknesses
1. The Benchmark's Contribution Falls Short

- The CuisineWorld, based on the video game Overcooked which is popular for measuring multi-agent cooperation and has many existing environments already[1][2]. The benchmark introduces more cuisines but falls short in diversifying kitchen maps when compared to previous Overcooked environments.

- Table 12 made a good summary of related benchmarks though some related embodied multi-agent cooperation benchmarks with most of the features in the table are missing, such as [3][4].
 
- While CuisineWorld boasts an appealing visual design, it does not directly align with the paper's experiments. The paper primarily relies on text-based states and interactions (Figure 26, even during human experiments, where humans are restricted to a textual interface with lots of text description of the game state)

- In section D.2, the statement "In this game, the actions of human players can be obtained from an inverse dynamic model by checking pre-conditions and post-effects" is confusing to me. How was Figure 22 obtained? It seems to be in real-time with human players using the keyboard to move, how's the time step defined here and the "goto" action for the LLM agent implemented?

- The benchmark predominantly employs a "new auto-metric collaboration score CoS for assessing the collaboration efficiency". However, this metric is defined as the average task success rate with different time intervals for each dish that appears highly tailored to the specific environment and lacks a clear connection to "cooperation efficiency".

- The absence of a train/test split is concerning, as prompt engineering can substantially impact performance. A thorough understanding of how the prompt is tuned for different tasks is crucial.

- From C.3.4, level 3 has only two similar recipes which differ only with the words "salmon" and "tuna", only one demonstration may decrease the task difficulty significantly for the LMs, raising doubts about the benchmark formation.

2. Claims Need Stronger Support from Results

- In section 5.1, The paper claims "more agents will lead to higher collaboration efficiencies. Thus, indicating that the LLM dispatcher can coordinate additional agents to execute tasks more efficiently". However, from Table 1, 4 agents perform worse than 3 agents in most scenarios, and even 2 agents achieve the highest score in levels 2 and 4, which contradicts the claim. It may provide more insights if the paper can provide more analysis on these contradictory results instead of only ablating on level_3, which seems to be the only level that "looks normal" (a.k.a 2 agents < 3 agents < 4 agents with somewhat clear gaps)

- Table 4, "For a fair comparison, all tests employed identical prompt inputs" using "identical" prompt for different LM families may not be "fair", especially if the prompt is "tuned" specifically for one model. More details on the prompt engineering process may help clarify these concerns.

- Table 3 presents a perplexing scenario where four agents using a two-agent demo outperform four agents using a four-agent demo, albeit marginally. This result could benefit from a clearer explanation.

- Novel game adaptation of Minecraft seems very promising, but the details are extremely limited. It would be more convincing if there were more details and formal experiments on it. For example, how is the "adaptation" conducted? What's the additional human effort required (such as prompt engineering and rounds of re-playing for prompt tuning)?

3. Method's Limited Contribution

- The method employed in the paper primarily relies on intensive prompt engineering, without introducing novel designs for multi-agent cooperation.

- It's well known that LLMs can perform "in-context learning", providing demonstrations and reasoning steps can help improve performance, there's nothing new to take away from the method and experiments.

- As mentioned above, the "emergent ability" of MindAgent is not well supported by the results.

- Providing only screenshots of part of the prompts as in Figures 6 and 7 is not enough. More details on the full prompt and game episode may help clarify these concerns.

[1] On the utility of learning about humans for human-ai coordination
[2] Too many cooks: Bayesian inference for coordinating multi-agent collaboration
[3] Building cooperative embodied agents modularly with large language models
[4] A Cordial Sync: Going Beyond Marginal Policies for Multi-Agent Embodied Tasks

### Questions
Please address the concerns raised in the Weakness.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an infrastructure, MindAgent, for evaluating planning and coordination capabilities in the context of gaming interaction. To facilitate multi-agent planning capabilities of LLMs, the paper designs an effective set of prompt templates, memory modules, and state and action processing modules. Additionally, the paper reformulates the optimization objectives and constraints of multi-agent planning into natural language descriptions and uses them as prompts to guide LLM planning. The paper also introduces a virtual kitchen game called CuisineWorld as a benchmark for LLM-based multi-agent planning. Furthermore, the paper evaluates the planning capabilities of various LLMs in multi-agent collaboration tasks and human-agent collaboration tasks in CuisineWorld using MindAgent.

### Strengths
***Originality & Significance***

Although this work is mainly application-oriented for LLMs, its novelty lies in proposing an infrastructure to explore the potential of LLMs in multi-agent planning and conducting experiments on multiple LLMs. I believe this work will inspire the LLM community and provide a valuable test bed for evaluating LLM capabilities.

***Quality & Clarity***

This work provides detailed descriptions and examples of the components of MindAgent. The paper also offers a good description of the environment setup and level settings in CuisineWorld. Furthermore, the paper validates the abilities of multiple LLMs, such as GPT-4, through multi-agent collaboration and human-agent collaboration tasks, and provides detailed experimental settings. I think the paper is clear and of high quality.

### Weaknesses
Please refer to the questions section.

### Questions
1. Have the authors considered incorporating a human-agent communication module in MindAgent? As suggested by Gao et al. [1], introducing interpretable human-agent communication into collaborative games can effectively improve human-agent collaboration performance and human subjective preferences. Natural language is the best medium for human-agent communication, which is also a natural advantage of LLM-based agents in human-agent collaboration.

2. Can the design of MindAgent support collaborative games that require more domain-specific knowledge? For example, Multiplayer Online Battle Arena (MOBA) games [1], First-person Shooter (FPS) games [2], and Diplomacy [3] have very complex gameplay and their outcomes heavily depend on the planning and collaboration capabilities of the agents. The authors could discuss how the infrastructure needs to be modified when extending it to other games, which would enhance the generalisability of the infrastructure.

---

[1] Gao, Yiming, et al. Towards Effective and Interpretable Human-Agent Collaboration in MOBA Games: A Communication Perspective. ICLR. 2023.

[2] Jaderberg, Max, et al. Human-level performance in 3D multiplayer games with population-based reinforcement learning. Science. 2019.

[3] FAIR, et al. Human-level play in the game of Diplomacy by combining language models with strategic reasoning. Science. 2022.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes an infrastructure for LLM-based agents to perform task distribution across a number of agents. It focus on planning and coordination ability of LLM with in-context learning from a few examples. The work tests on a multi-agent gaming environment, CuisineWorld, as well as a few other multi-agent collaboration environments, and obtain promising results.

### Strengths
Contributions:

1. Evaluate LLM agent in a multi-agent gaming environment; test LLM's ability to serve as a task allocator.

2. Proposes a collaboration score to evaluate and benchmark coordination agents.

### Weaknesses
1. Technical novelty: The technical novelty of this work is lacking because the multi-agent setting is really just adding an LLM call to allocate what different agents should do. Concretely, it is a simple prompting and what is supposed to be an optimization problem is all packed into one LLM call. This doesn't seem to be a very principled way of studying the agent allocation. 

2. Problem setting: the multi-agent collaboration problem is reduced to a top-down allocation: a distributor distributes tasks for agents to do. But for a collaboration problem to work, there should also be communication between the agents, which this work does not study.
- the title is also quite misleading: it is titled "emergent interaction" however, there is no real interaction between the agents, i.e., communication of agent's individual's ability, limitation, progress etc. It is more of a allocation / coordination problem of a central agent.

3. Experimental results: The work evaluate on Minecraft environment, but there has been quite a few LLM-based Minecraft agents, and the work does not offer comparison between this method and existing works as baselines.

### Questions
1. What are the common failure cases of the LLM agent allocation? Under what environment / optimization constraints would the agent fail to allocate works correctly?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new benchmark CuisineWorld for evaluating the planning and coordination capabilities of LLMs in multi-agent settings. Different from previous research, this benchmark is characterized by (1) its incorporation of multi-task objectives, (2) the involvement of more than two agents, and (3) the utilization of a centralized system for coordination. In this context, an LLM functions as the centralized system. At every timestep, the LLM processes each agent's state along with a prompt including recipes, demonstrations, the fundamental rules of the game, and memory history, and outputs the optimal actions for the agents. The paper demonstrates that GPT-4 has a strong planning capabilities on the proposed benchmark.

### Strengths
- The creation and development of a new benchmark, entailing a substantial amount of effort, is a significant strength of the paper.
- The introduction of a novel metric, CoS (Coordination Score), to assess coordination capabilities is a noteworthy contribution.

### Weaknesses
- The proposed LLM coordination system, MineAgent, lacks novelty as its approach of leveraging scratchpad or memory has been extensively explored in prior research.
- The proposed environment features limited state and action spaces and furnishes all the requisite recipes to solve tasks, possibly oversimplifying the challenge. In this situation, heuristic or RL planners could be readily employed. However, the paper does not provide comparisons with these approaches.
- The proposed environment bypasses low-level control, further oversimplifying the problem. From my understanding, an agent can move to any location with a single action, without the need to consider spatial information. This eliminates the need for spatial reasoning in the LLM planner.
- The paper claims that the LLM can seamlessly adapt to new planning problems across different domains, but this assertion is questionable considering the dependence on context length. While the simplicity of the current setting allows all environment information to be described within 1K tokens, this approach may prove inadequate for more challenging environments. Additionally, engaging in manual prompt engineering to enhance performance may entail significant monetary costs.
- The description of the environment setting requires further clarification. Does $\tau_\mathrm{int, (1)}$ mean that a new task will be added at every timestep? What is the maximum horizon of an episode?
- The paper only highlights the emergent behavior of GPT-4 while neglecting to discuss its potential drawbacks. A more balanced perspective could be achieved by including examples of GPT-4's failure cases.

### Questions
See the weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new gaming scenario and related benchmark based on a multi-agent virtual kitchen environment, CuisineWorld.  It introduces MindAgent, which demonstrates the in-context learning multiagent planning capacity of LLMs and brings several prompting techniques that help facilitate their planning ability. Extensive evaluations are conducted with multiple LLMs and prompting settings on the benchmark, including deploying the system into real-world gaming scenarios.

### Strengths
The work is solid and important to the community which provides a benchmark that supports the implementation of a general multi-agent infrastructure that encompasses collaboration between large language models (LLMs) and human-NPCs.

The paper is well-written and organized.

### Weaknesses
1.	The font size of Figure 4 is too small.
2.	The paper seems not provide a clear definition of the terms $q_{pim}$ and $c_{pim}$ in Equation 2.

### Questions
As weaknesses.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
