# Real-Time Reasoning Agents in Evolving Environments

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
Agents in the real world must make not only logical but also *timely* judgments. This requires continuous awareness of the dynamic environment: hazards emerge, opportunities arise, and other agents act, while the agent's reasoning is still unfolding. Despite advances in language model reasoning, existing approaches fail to account for this dynamic nature. We introduce *real-time reasoning* as a new problem formulation for agents in evolving environments and build **Real-time Reasoning Gym** to demonstrate it. We study two paradigms for deploying language models in agents: (1) reactive agents, which employ language models with *bounded reasoning computation for rapid responses*, and (2) planning agents, which allow *extended reasoning computation for complex problems*. Our experiments show that even state-of-the-art models struggle with making logical and timely judgments in either paradigm. To address this limitation, we propose **AgileThinker**, which simultaneously engages *both reasoning paradigms*. AgileThinker consistently outperforms agents engaging only one reasoning paradigm as the task difficulty and time pressure rise, effectively balancing reasoning depth and response latency. Our work establishes real-time reasoning as a critical testbed for developing practical agents and provides a foundation for  research in temporally constrained AI systems, highlighting a path toward real-time capable agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces AgileThinker, a dual-thread architecture for LLM agents that must act while reasoning in dynamic environments. It defines the real-time reasoning problem, where environments are dynamic and the agent’s computation time directly affects outcomes. Using the new Real-Time Reasoning Gym benchmark, the authors show that combining a fast reactive thread (for immediate responses) with a slower planning thread (for long-term reasoning) enables a better balance between speed and deliberation. Experiments demonstrate that AgileThinker outperforms purely reactive or planning agents under varying cognitive load and time pressure, with results validated both in simulated and real wall-clock settings.

### Strengths
1) The paper introduces real-time reasoning as a new challenge for LLM agents and proposes AgileThinker, a dual-thread framework that combines fast, reactive behaviour with slow planning. This is a clear conceptual advance beyond existing static or turn-based agent paradigms.
2) The authors validate their approach through a carefully designed benchmark (Real-Time Reasoning Gym), multi-game testing, controlled ablations, and statistical significance tests (appendix). The experimental evidence is strong and aligns with the theoretical motivation. 
3) The paper is well-written, well-illustrated, and accessible, effectively communicating complex timing and coordination concepts. I found the work really interesting.

### Weaknesses
1) All experiments use simulated game-like settings (Freeway, Snake, Overcooked); they may not fully represent complex real-world tasks (e.g., robotics, human-AI interaction). The results demonstrate concept validity but not necessarily scalability to multimodal or continuous-control domains.
2) The paper needs to explore comparable baselines like: 
   - Li, Yaoru, Shunyu Liu, Tongya Zheng, and Mingli Song. 2025. “Parallelized Planning-Acting for Efficient LLM-Based Multi-Agent Systems.” arXiv [Cs.AI]. arXiv. http://arxiv.org/abs/2503.03505.

### Questions
1) What are some alternatives to TR, i.e., alternatives to fine-tuning this hyperparameter for each domain/task?
2) What are the foreseeable computational and latency overheads of running two LLM threads in real-world systems, especially under networked or resource-constrained deployment?
3) How does AgileThinker compare with Yaoru et al.? Unless there are strong reasons not to, this is a good baseline to test AgileThinker.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose Real-Time Reasoning Gym, an environment consisting of three games (Freeway, Snake, and Overcooked) that are designed to test LLM agents in dynamic environments. Dynamic environments are those where the environment progresses at a fixed rate regardless of whether the agent has been able to come up with an action or not. To evaluate how current methods perform in this new setting, the authors study both reactive agents (non-thinking models, budget forcing) as well as planning agents (thinking models, code as policies). They find that each type of agent has its limitations, motivating the novel AgileThinker agent which runs two parallel threads (one for reactive, one for thinking) and mostly matches or outperforms all baselines in all settings.

### Strengths
1. The work provides a very novel and interesting problem setting for language agents, lifting the unrealistic assumption that the environment waits for agents to execute their actions.
2. The motivation is clear and overall the paper is well-written
3. The quantitative results are very strong, and the authors additionally provide a qualitative case study further illustrating the benefits of AgileThinker.

### Weaknesses
I could not discern any significant weaknesses, but I do have some questions in a few places (see questions section).

### Questions
1. Figure 3: what does “reference” mean? And why is there two frames marked as “Reactive Agent Output” under the Static setting? I thought the agent only outputs actions, but here somehow the observations seem to be marked as output?
2. Page 4: “For each game, we normalize the scores by the highest score the agent could get in that game, so we always have a score between 0 and 1.” → could the authors clarify exactly how this normalization works maybe by giving an example?
3. Page 5: “(2) code plans where a thinking model is used to generate a code snippet that automatically produces actions based on observation input (Liang et al., 2022; Zhang et al., 2025). Although (1) is often easier to generate, (2) is more adaptive to potential changes.” → could the authors explain this a bit more? As in, why is generating code plans more adaptive to potential changes? Also, it’s not immediately obvious to me that generating code should fall under planning?
4. Figure 6: for the “Planning Reasoning” yellow box in the middle, why is there only two apples instead of 3? In all other cases there seems to be three.
5. Figure 6: is this a case study of real trajectories?
6. Figure 5: it seems that in the bottom right plot, AgileThinker does better than the planning methods even though the time pressure is very lenient. I would have expected these methods to be at best on par here, but somehow AgileThinker outperforms the planning methods - could the authors comment on why this might be the case?
7. Page 8: “we also plot the cumulative distribution function (CDF) of R’s token usage across all game trajectories without constraints” → does this mean for the purpose of creating the CDF, R gets to see all of P? Or how is the partial reasoning reference point determined if there is no constraints?
8. Table 2: what’s the cognitive load setting for these experiments?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the real-time decision-making into LLM based agents. Instead of having an agent-environment loop, the proposed real-time reasoning gym imposes a time pressure on agent reasoning manifested in the form of token limits, and tests how well agents can complete tasks under the pressure of bounded reasoning. The proposed gym features three diverse synthetic agent environments with different challenges that arise from dynamic environments, and experiments show that previous LLM agents perform poorly on these challenges, especially when time-pressure is high, sheding light on a new direction of research.

### Strengths
1. Real-time reasoning of AI agents under time pressure is an under-evaluated direction in the current wave of AI agents research, and this paper presents a good setting and three diverse synthetic environments to prototype research in this direction. Experiments show that with increased time pressure and task complexity, existing agent performance decrease significantly from the case where the environment waits indefinitely for agent reasoning.

2. The paper presents AgileThinker, an agent design that combines a planner and a reactive agent in a manner that controls planner output with a token budget for the agent to stay under the overall allotted time pressure limit. This agent design, albeit simple, shows gains over purely planning or reactive agents especially when high time pressure is applied on the agent.

### Weaknesses
1. The assumption that token limit is a good proxy for actual wallclock time is valid only when all agents are implemented with the same LLM served with the same hardware. Even if hardware independence is desirable, it is unlikely that different models will share the same slope (TPOT) and intercept (whatever factor that goes into time-to-first-token), because of model size, model architecture, etc. The paper's formulation and experiments neglected this important consideration. 

2. As also seen in the result in Section 6, one crucial factor the token budget limitation neglects is the time-to-first-token cost of LLM calls in these language agents. Contrary to what the authors lead the reader to believe, the step time in the results presented in the paper for these simulated games is not 100s of milliseconds (as those familiar with these games would believe), but minutes (5.5 minutes to be precise). This is an important caveat, and undermines the real-time nature of the gym itself, since the environment is still unduely waiting for the agent to perform indefinite internal operations.

3. The paper could do a better job contextualizing itself in the literature.

    1. There is a limited set of agent designs that this paper experimented with, especially open-source implementations of agents from prior work to help the reader understand whether the baselines used in this paper are valid or competitive, and more importantly, whether the proposed AgileThinker framework can be a drop-in improvement for existing agent implementations.
    2. Real-time agents have long been a subject of AI research, from best-first search (A* and variants) and robotics [1], to more recently real-time customer support AI agents [2]. The paper could use a better review of related work in this direction, and better distinguish the key contributions it's making. The concept might not be entirely new, but it's useful to call it out in the new paradigm of AI agent design and evaluation.

[1] OpenVLA: An Open-Source Vision-Language-Action Model. (https://arxiv.org/pdf/2406.09246)

[2] Asynchronous Tool Usage for Real-Time Agents (https://arxiv.org/pdf/2410.21620)

### Questions
1. Table 2 and Section 6: what's the metric shown here, and what's the wallclock limit for these agents?
2. How does the proposed framework generalize to cases where the environment updates at a cadence that is not predetermined, e.g., one update takes 1s and another takes 10s?
3. How does AgileThinker generalize to other agent designs?

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
This paper addresses the limitation of existing LLM-based agents in dynamic environments (where environments evolve during agent reasoning) by formulating "real-time reasoning" as a new problem and building Real-Time Reasoning Gym (with Freeway, Snake, Overcooked) to evaluate it. It studies two agent paradigms—reactive agents (bounded computation for speed) and planning agents (extended reasoning for complexity)—finds their flaws, and proposes AgileThinker, a dual-thread model that combines both by letting the reactive thread use partial planning traces. Experiments show AgileThinker outperforms single-paradigm agents as cognitive load and time pressure increase, and wall-clock tests confirm the linear correlation between tokens and real inference time, verifying its practicality.

### Strengths
1. It innovatively defines the "real-time reasoning" problem for LLM agents (addressing the flaw that environments evolve parallel to agent reasoning) and builds Real-Time Reasoning Gym with three games (Freeway, Snake, Overcooked) to systematically control cognitive load and time pressure (using tokens as a hardware-agnostic time proxy).
2. The proposed AgileThinker has an innovative dual-thread design: its reactive thread references partial planning traces for real-time decisions, solving the limitations of existing dual-system methods (independent operation or waiting for completion), and experiments with the same model family verify its superiority over single-paradigm agents.
3. It emphasizes practicality: wall-clock experiments confirm a strong linear correlation between token count and real inference time (R²=0.9986).

### Weaknesses
1. AgileThinker lacks an adaptive mechanism for thread resource allocation: the optimal token budget for the reactive thread ( $N_{TR}$ ) varies across environments (e.g., ~5k tokens for Freeway vs. ~2k tokens for Snake/Overcooked) and requires manual empirical tuning, with no solution proposed to dynamically adjust it based on real-time environmental changes.
2. Experimental scenarios are disconnected from real-world complexity: all experiments are conducted on three simulated games (Freeway, Snake, Overcooked), and there is no verification of AgileThinker’s performance in real-world complex scenarios (e.g., SWE-Bench, BrowseComp), weakening the practical reference value of the results.
3. The contribution is unclear. Existing works [1] control whether models conduct deep thinking through mode switching; what is the essential difference between AgileThinker and these works?

[1] Wu, S., Xie, J., Zhang, Y., Chen, A., Zhang, K., Su, Y., & Xiao, Y. (2025). ARM: Adaptive Reasoning Model. ArXiv, abs/2505.20258.

### Questions
In terms of evaluation tools, what unique contributions does the Real-Time Reasoning Gym (with three games: Freeway, Snake, Overcooked) bring to the field of real-time agent evaluation?

### Soundness
2

### Presentation
2

### Contribution
2
