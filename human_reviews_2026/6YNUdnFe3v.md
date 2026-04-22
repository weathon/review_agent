# Ella: Embodied Lifelong Learning Agents with Non-Parametric Memory

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Situated within human society, embodied agents are continuously exposed to diverse streams of information, ranging from visual observations to natural language interactions. A central challenge is enabling them to learn from and effectively leverage this information over extended periods. To address this, we introduce Ella, an embodied lifelong learning agent designed to accumulate experiences and acquire knowledge across hours of social interaction in a 3D open world. At the core of Ella’s capabilities is a structured, non-parametric,  long-term multi-modal memory system that stores, updates, and retrieves information effectively. It consists of a name-centric semantic memory for organizing acquired knowledge and a spatiotemporal episodic memory for capturing multimodal experiences. By integrating foundation models with this non-parametric memory system, Ella retrieves relevant information for decision-making, plans daily activities, builds social relationships, and evolves autonomously while coexisting with other intelligent beings in the open world. We conduct capability-oriented evaluations in a dynamic 3D open world where 15 agents engage in social activities for days and are assessed with a suite of unseen controlled evaluations. Experimental results show that Ella can influence, lead, and cooperate with other agents well to achieve goals, showcasing its ability to learn effectively through observation and social interaction. Our findings highlight the transformative potential of combining non-parametric memory systems with foundation models for advancing embodied intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
ELLA is a system for an autonomous agent designed to operate on the order of days. It combines a long-term spatial memory based on multi-layer 3D scene graphs, along with LLM and multimodal models.

The authors demonstrate a case study in a multi-agent by running several copies agents in a model scenario where agents in a town are asked to plan a party. The authors give somewhat elaborate personalities to each of the agents, and designate 4 predetermined social groups such as "fitness enthusiast" and "ai enthusiast".

### Strengths
The paper tackles an ambitious and important problem - enabling embodied agents to learn and operate over extended temporal scales (days) in social environments. I found it a fun read, and there were plenty of good cogsci + psychology references.
* The evaluation setup is creative and goes beyond typical navigation or manipulation tasks.
* The paper includes ablation studies with oracle perception, showing the impact of perception errors on the overall system performance.
* The inclusion of experiments with open-source foundation models (DeepSeek-R1, Qwen2.5) shows the system depends heavily on the LLM used

### Weaknesses
Overall this is an interesting setup, but the paper introduces several things, and currently the experiments do not feel complete or well-matched to the specific things introduced.

**Benchmark**
On the one hand, the paper introduces a multi-agent setting and setup. This is the only evaluation used; a multi-agent setup on two hand-defined tasks in a single hand-defined setting. If this is the main focus of the submission, then considering progress in the field, I think there must be more recent baselines that have come out in the past two years and could be compared in the benchmark? Why were these tasks and metrrics chosen -- what behaviors do they measure and why are existing benchmarks not sufficient. Multi-agent is not my main familiarity, so I leave this to the other reviewers.

**Memory**
On the other hand, this paper introduces a RAG-like memory that queries a 3D scene graph and combines it with an LLM. It's not clear to me if this is supposed to be a core contribution of the paper; it is listed in the contributions -- but seems to play a minor role if the focus is on the multi-agent setting where oracle perception could be used. 

If the scene graph is important, then why is the multi-agent setting needed to evaluate the efficacy of the scene graph, and why do existing long-term memories not suffice and where does this plug the gap? I see that this plugs a gap with the Coela baeline, but other existing work can, too. 

**4.1.1: Hierarchical scene graph as spatial memory**
There are plenty of 3D scene graph approaches that could serve as a starting point and are intended to support long-term memory that interfaces nicely with LLMs, with established benchmarks for single-agent performance (e.g. SayPlan https://sayplan.github.io/, or something like this: https://arxiv.org/pdf/2510.16643 -- the second paper came our recently; I just found it by googling. The point is this _type_ of eval, not necessarily a comparison to this specific paper.)

The authors do cite a couple scene graph works (e.g. conceptgraphs, hydra), but the actual method is not explicitly related to any of them. My opinion is that submission would be stronger if the method itself built more off of existing literature -- it would be easier to pinpoint the contribution of this paper, and fewer ablations would be needed to understand the design choices.

For the specific long-term memory module; there are no ablations, except maybe figure 1b where the analysis is simply how many nodes stored (not whether it is effective, or how effective compared to baselines. 


**4.2 Spatiotemporal Episodic Memory**
The authors propose a handcrafted retrieval rule, similar to conceptgraphs, that builds on feature similarity and spatiotemporal recency. There are no ablations or analysis of the design choices here.

### Questions
There is a classical lifelong learning paper "ELLA: Efficient Lifelong Learning Algorithm" https://proceedings.mlr.press/v28/ruvolo13.html that may be confusing for some readers given the name. I just wanted to bring this to the authors' attention -- perhaps they could consider some ways to avoid confusion. Modifying the acronym, letting the reader know, etc.

Benchmark:
* How were the personalities for the characters in the scenario chosen?
* Do all agents in the scenario in all groups use the same agent (e.g. ELLA or Coela or Generative Agents)? 

Memory:
* I have some questions about the generality of the approach. Since the model builds in a layer for buildings, would it work outdoors, in a forest, or underwater?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Ella, an embodied agent designed for lifelong learning within a social, 3D open-world environment. The central challenge the paper addresses is the inability of current foundation model-based agents to continuously learn from new information over long periods, often suffering from catastrophic forgetting when parameters are fine-tuned. The core contribution is Ella's structured, non-parametric, long-term multi-modal memory system. Ella integrates this memory system with foundation models using a planning-reaction framework.

### Strengths
1. This paper introduce a novel and grounded memory architecture which utilizes dual semantic/episodic memory.
2. Ella demonstrates a clear and significant performance advantage over baselines in the new social tasks (Table 1). The analysis effectively pinpoints the baselines' failures. 
3. This paper also includes thoughtful analysis which is a good base for future work.

### Weaknesses
1. As the authors note, the retrieval system is based on feature similarity and does not fully exploit the rich graph structure of the name-centric semantic memory. This limits the agent's reasoning to relatively simple, direct lookups rather than complex, multi-hop queries (e.g., "Who did I meet at the place where I bought coffee yesterday?").
2.  The large performance gap between Ella and "Ella + Oracle Perception" suggests that the current perception pipeline is a major bottleneck. While using open-set models is a strength, the paper does not deeply analyze how perception errors (e.g., misidentifying agents, failing to track objects) propagate into the long-term memory and cause cascading failures in reasoning.

### Questions
1. How does memory scales? The 9-hour simulation generated ~161MB of memory per agent. If you were to run this simulation for a simulated month, what new challenges would you anticipate for the retrieval system?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on life long learning agents which act in situated environments. The agents have a long term spatial memory where they store precise spatial representations for locations and events that happen over time. These memories are used to plan, react and communicate with other agents in a continual learning setting. Summarized conversations are also stored in episodic memory which builds up over time. Experiments are performed in the “Virtual Community” environment with 15 unique agents.

### Strengths
The paper aims to tackle a very ambitious problem of learning in situated agents with non-parametric memory, where the agent environment simulation happens over a timescale of days. This is also a very important and challenging problem given that the current set of foundation models (and thus agents) have no explicit sense of memory and most often memory is added on top of these agents.

### Weaknesses
The paper is not really well written, it is very confusing in parts. It is not really well motivated at all. While I agree that agents need memory and some form of non-parameteric memory would be important, the paper does not present convincing arguments/experiments on why the agent would need precise spatial memory for high level semantic goals. The details on how the agent acts to generate low level actions is missing (Are they even generated, e.g. when the agent generates a goal “go to office”, how is this implemented?) All of these details are critical and completely missing from the paper. The experimental section is also quite week, there aren’t any ablations. Some of the experiment settings e.g. (need of robust perception for embodied social agents) are not motivated at all, instead results are presented directly. All of this makes it very hard to understand how the paper is implemented.

### Questions
*Hierarchical Scene Graph as Spatial Memory*: The paper goes into details on how the spatial memory is being created but it is not clear how it is being used by the agent. The paper mentions that the agent uses/needs spatial memory to act, but what is the action here? Are these navigation actions, manipulation actions or other high level semantic actions. If the latter then why do we need spatial memory. Also, it is quite unclear why precise spatial memory (e.g. which object is at what location) is needed in the future. If an agent goes from place A to place B, then the spatial locations of object in place A (spatial memory of A) is not really needed at place B. 

*Daily Schedule*: The daily schedule of the agent shown in Figure 3 (c) consists of mostly high level goals (e.g. commute to office). How are these high level goals grounded in the environment? It seems relying on memory alone would be insufficient for many problems (e.g. navigation) which would require planning based on current context instead of past context? 

The paper has a very very ambitious goal, to achieve memory for really really long horizon actions (e.g. at the level of days) when the environment clock is ticking at 1 FPS. Does this mean that the number of actions the agent takes in the environment is 1.5 days * 86400? Or does the paper assume some hierarchical actions. I think many such critical details are completely missing from the paper.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an approach for lifelong multimodal embodied reasoning over long contexts (1.5 days) in the open-world by ingesting visual observations and social interactions. The proposed method implements two different forms of non-parametric memory: a graph-based representation for semantic memory and spatiotemporal episodic memory, in addition to retrieval and reactionary modules (environment interaction and social conversations). The method achieves outperforms adapted baselines on two hand designed social interaction settings testing perception, memory retrieval, reasoning and conversational capabilities.

### Strengths
- The paper considers a novel setting of open-world reasoning and social coordination over long contexts (1.5 days).
- The tasks of agent persuasion and team-based task completion considered in the paper are challenging and interesting.
- The accompanying task illustrations greatly improve the understanding of tasks and method performance.

### Weaknesses
1. The method appears to be a multimodal extension of prior works: two distinct forms of memory from CoELA and distinct retrieval and reactionary pipelines from Generative Agents, limiting the technical contributions of the proposed method. 
2. While the method adds multiple changes over prior works and compares to them, it doesn't carefully study the contribution of each of these changes to the overall improvements. Some of these changes are: name-based graph representation for semantic memory, multimodal data in episodic memory, spatial information in episodic memory, replacing importance with location in relevance computation. These studies would show the importance of these additions to the method.
3. The authors do not report results on standard benchmarks, e.g., TDW-MAT or C-WAH from CoeLA (possibly with oracle perception).
4. Some of the choices seem to be non-intuitive or possibly hand-designed for solving the target tasks.
    
    a. Same weight to recency as semantics or spatial closeness when calculating may not be generally appropriate: a query could specifically be interested in memories from distant past.

    b. Using closeness of visuals for identifying dynamic objects may not be sufficient, e.g., for dealing with multiple instances of objects. It may be necessary to incorporate further reasoning based on object semantics and temporal distance between the two spottings may be necessary.

     c. Semantic matching to memory matches only images in query to images and texts in query to texts in memory. It may be necessary to incorporate cross-modal matching (e.g., if a memory entry only includes descriptions of objects).

5. Storing entities seen over multiple days to the graph semantic memory could drastically increase the memory size with further increase in number of days. The agent may need to selectively forget memories to avoid large memory blowups.

### Questions
1. Are the tasks of Leadership Quest and Influence Battle inspired from prior works? Can authors share more details on the choice of these tasks?
2. Can authors clarify the novel contributions to the method over prior work beyond storing multimodal data and using active perception in place of oracle perception components?
3. Results: Why is Leadership Quest performance for Detroit (or Influence Battle for London) worse with access to oracle perception?
4. Results: Is there evidence to show the inefficiency with oracle perception is actually because of more interactions between agents? Conversely, does the non-oracle approach frequently miss identifying known agents?
5. What are the remaining failures on these settings? Are there limits to the maximal achievable performance on the tasks (e.g., it might be infeasible to persuade all agents for the party in the given time limit if they are initialized very far)?  
6. How are the prompts designed for each of the two tasks and how do they differ between the tasks? How much prompt engineering is required to specify goals for solving these tasks?

### Soundness
3

### Presentation
4

### Contribution
2
