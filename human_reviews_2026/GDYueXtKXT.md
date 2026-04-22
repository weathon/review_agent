# Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
While the advancement of large language models has spurred the development of AI agents to automate tasks, numerous use cases inherently require agents to collaborate with humans due to humans' latent preferences, domain expertise, or the need for control. To facilitate the study of human-agent collaboration, we introduce Collaborative Gym (Co-Gym), an open framework for developing and evaluating collaborative agents that engage in bidirectional communication with humans while interacting with task environments. We describe how the framework enables the implementation of new task environments and coordination between humans and agents through a flexible, non-turn-taking interaction paradigm, along with an evaluation suite that assesses both collaboration outcomes and processes. Our framework provides both a simulated condition with a reliable user simulator and a real-world condition with an interactive web application. Initial benchmark experiments across three representative tasks-creating travel plans, writing related work sections, and analyzing tabular data-demonstrate the benefits of human-agent collaboration: The best-performing collaborative agents consistently outperform their fully autonomous counterparts in task performance, achieving win rates of 86% in Travel Planning, 74% in Tabular Analysis, and 66% in Related Work when evaluated by real users. Despite these improvements, our evaluation reveals persistent limitations in current language models and agents, with communication and situational awareness failures observed in 65% and 40% of cases in the real condition, respectively. Released under the permissive MIT license, Co-Gym supports the addition of new task environments and can be used to develop collaborative agent applications, while its evaluation suite enables assessment and improvement of collaborative agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel asynchronous, non-turn-taking framework for evaluating collaborative agents. The system design features a message-passing mechanism, a notification protocol (supporting wait-delay actions and private-public divisions), and an interface accommodating both real-human and LLM-simulated users.

Through evaluation on three instantiated tasks, the authors find that: 1) While collaborative agents struggle with task completion, they yield better quality on completed tasks. 2) The simulated evaluation setting achieves high fidelity to evaluations with real humans. 3) Communication and situational awareness remain prevalent issues in human-AI collaboration. 4) The non-turn-taking design incentivizes more performant and satisfying collaboration.

### Strengths
Overall, this paper is well-written and easy to follow. The framework design rigorously considers key aspects of human-AI collaboration from prior studies (e.g., proactive human control, non-turn-taking), while providing flexibility for adding new tasks. The experimental findings are particularly insightful, illustrating important lessons and failure modes in current human-AI collaboration.

### Weaknesses
From a high-level perspective, this is a strong paper on *human-AI collaboration*. (I give my scores based on this calibration) However, since it is framed as a contribution to *infrastructure/software*, further improvements are needed to strengthen the paper:

1. **Lack of Infrastructure-Level Detail**: As an infrastructure paper, the programming model and system design are not sufficiently detailed. Key content is relegated to the appendix, leaving it unclear how a new task is practically implemented. The code example (Figure 2) is too high-level to understand the core mechanics (e.g., how the `event_loop` and `event_handler` operate). Furthermore, the notification protocol is not analyzed from a systems perspective: How are potential deadlocks or inefficiencies handled in highly uncertain, massively parallel environments? How are message backlogs managed? The choice of Redis, a lightweight in-memory database, also raises questions about system robustness and fault tolerance.

2. **Unclear Implementation and Justification for Dual Control**: While dual control is an important feature for human-AI interaction, its implementation in an asynchronous, non-turn-taking system could bring extra complexity. The paper does not discuss the engineering challenges (e.g., reliably sending a 'pause' signal to a multi-threaded agent). Moreover, the necessity of this feature for the selected tasks is not clearly justified. The paper notes it improves delivery rates and user satisfaction for human collaborators, but it lacks a clear ablation study in the human evaluation setting to quantify the benefits of dual control.

### Questions
1. Could the authors provide a qualitative example or a concrete walk-through of how the "Initiative-Taking" metric is computed? This would help build intuition for this metric (i.e., why this metric is useful and necessary from an intuitive sense).

2. The claim on L395 ("Consistent with the simulated condition...") appears to contradict the data. In the simulated condition, collaborative agents "struggle to complete the tasks," but in the human-AI condition, the task delivery rate is 100%. Please clarify this inconsistency.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents Collaborative Gym (Co-Gym), a framework for developing and evaluating human–agent collaboration. Unlike most existing benchmarks that assume turn-taking or single-control settings, Co-Gym enables parallel and bidirectional communication between humans and agents in shared task environments. The framework supports both simulated (with LM-based user simulators) and real (interactive web application) conditions. Experiments on three representative tasks—travel planning, related work writing, and tabular analysis—show that collaborative agents consistently outperform fully autonomous agents and turn-taking workflows in overall task quality and user satisfaction.

### Strengths
1. The paper proposes a well-designed and extensible infrastructure that supports both simulated and real human–agent collaboration, complete with an evaluation suite for outcomes and collaboration processes.

2. The comparison between turn-taking and current paradigms, along with process-level metrics such as initiative entropy, provides valuable insight into how interaction structure impacts collaborative performance.

### Weaknesses
1. The ablation study on turn-taking could be more comprehensive. It appears that the turn-taking version involves alternating actions without explicit communication. For a fairer comparison, it would be useful to include a variant that allows turn-taking with communication, to isolate the effect of interaction timing from that of message passing.

2. The error analysis is rich but mixes issues caused by the framework design (e.g., communication protocol, observation structure) with those caused by inner agent reasoning (e.g., LM limitations). A clearer discussion of which errors are systemic versus agent-specific, and potential framework-level mitigations (e.g., proactive confirmation policies, adaptive notification thresholds), would strengthen the interpretation.

3. Task diversity: All three benchmark tasks are text- or knowledge-work oriented. It would be valuable to include more interaction-heavy or dynamic scenarios (e.g., collaborative gaming or creative design) to test whether the framework generalizes to conversationally rich or time-sensitive collaboration domains.

### Questions
The paper reports that collaborative agents have lower delivery rates than fully autonomous agents, yet achieve higher task performance when tasks are completed. This raises questions about whether the framework’s coordination or timeout mechanisms limit the collaborative agents, and whether such outcomes are artifacts of the environment design rather than true behavioral inefficiency.

### Soundness
2

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
3

### Summary
The authors introduce a new framework for developing and evaluating human-agent collaboration ability of LLM agents - Co-Gym. Co-Gym provides an agent-agnostic interface for collaboration, allows for non-turn-taking interaction, and an evaluation suite for assessing outcomes and processors. To enable coordination in a non-turn-taking fashion, they use a notification protocol to notify the agents in either shared or private manner of various events.

The authors evaluation on three tasks - creating travel plans, writing related works sections, and analyzing tabular data. The metrics used are delivery rates and task performance (quality). The authors also use a measure for initiation entropy and overall satisfaction of the human collaborator on Likert scale.

The results show that in simulation (mocked LM-based human interaction using static databases with privileged information), the collaborative agents have low-delivery rate compared to fully autonomous agents, but the quality of outcome is higher for the human-agent teams. In the real-world, the delivery rates for collaborative agents is better. The authors also analyze failure modes and show that communication and situational awareness are the most common issues.

### Strengths
1. Co-Gym allows for dynamic non-turn taking and dual-control interactions, which provides high level of flexibility for the human-agent collaboration.
2. There is a concept of private and shared information in the collaboration, which is an interesting and realistic aspect of the framework.
3. It is a unique and interesting finding that collaboration works better in the real-world than in simulation for their tasks.

### Weaknesses
1. The tasks in Co-Gym can only be textual, including the description of states, instructions, action space, etc. It is harder to evaluate human-agent collaboration with Co-Gym in real-world or vision-based environments, where the states can not be defined using text.
2. The tasks that the authors choose to evaluate on are pretty limited. While more tasks can be (and have been) added to Co-Gym, a more thorough evaluation is warranted to demonstrate the effectiveness of the approach and the ability of this framework to evaluate the human-agent collaboration ability.
3. The collaborative agent in the manner that Co-Gym implements them fails to achieve higher success rates than a fully autonomous agent in simulation, casting a doubt on the approach and the task quality that have been added to the framework. It also shows that human-involvement is necessary to evaluate the agents' ability to collaborate, which casts doubt on the overall approach.

### Questions
1. How can we extend this framework to environments which are not text-only?
2. How do we work with continuous action spaces within this framework?
3. How can the readers gain some confidence on the sim-only evaluation (done using GPT-4o)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoGym, a framework for implementing dual-control, non-turntaking environments. The authors implement three tasks (travel planning, writing, data analysis) and show that multi-agent or human-agent collaboration in these tasks outperforms non-collaborative agents.

### Strengths
- tackles an important problem: understanding human-AI collaboration and building benchmarks that better align with how it is natural to use the models
- interesting and well-motivated setting/idea: dual-control and non-turntaking interaction
- well-designed environment abstractions (notifications)
- broad range of tasks (travel planning, writing, data analysis/coding)
- nice new evaluation metric for initiative in collaboration (initiative entropy)

### Weaknesses
**Core contribution of the paper is unclear**

I'm not sure what point to take away from this paper. I agree with the introductory points that collaboration in general is important, but CoGym specifically focuses on "dual-control, non-turntaking collaboration." This seems to be the main contribution of the framework relative to other benchmarks with user simulators, but the experiments mainly focus on evaluating collaboration vs. no collaboration, rather than the dual-control/non-turntaking aspects. I think the new ideas in the paper are actually quite interesting, but I'm not sure what to make of it and what we should expect the community to do with CoGym.

Do the authors think that:
- all collaborative tasks should be dual-control, non-turntaking interaction?
- a specific subset of collaborative tasks? what subset? When should one build environments into Co-Gym vs. prefer turntaking interaction?

Notably, the task performance difference between turn-taking and non-turntaking is negligible. I'm not surprised that non-turntaking interaction is preferred since it's less constraining, but I think the point would be stronger if the authors could identify situations where non-turntaking interaction is important for performance.

This is made more confusing by the fact that the authors repurpose tasks designed for the single-agent setting. These tasks fundamentally are solvable without collaboration, but the paper makes them into collaborative tasks by hiding some relevant information. The paper does not focus on this design decision, but I think the choice of what information to hide importantly determines what kind of collaboration is necessary. For example: does the user just need to tell the agent a few relevant facts, or is there more complex coordination necessary (motivating dual-control/non turn-taking)? One idea: maybe the user needs to make fine-grained edits in order to communicate intent, which would make dual control very useful.

**Limited understanding of how well the user simulator matches real humans in this setting**

The analysis of the alignment between failure modes of simulated/real condition is helpful, but very limited. I imagine that there are a lot of differences between user simulator and human behavior, especially in this new async setting. I think it's important to better understand -- if we do evaluation in CoGym, where and when we can generalize these insights to real humans -- not just in error modes, but in *how* the interaction happens, for instance. Even more qualitative analysis would be helpful!

**I think the paper has a lot of potential but I would have loved to see deeper thinking about these points to warrant a strong accept.**

### Questions
- I think the most interesting section is around "discovery of effective patterns" -- these insights get at some of my questions above around when dual-control, non-turntaking interaction is necessary. I would love to see more involved discussion or understanding of this!
- Conclusions drawn from the simulated results are slightly confusing/potentially misleading as described right now (Sec 5.1.1).**
Because this is collaboration with user simulators, these results are better described as "failure modes/strengths of multi-agent vs. single-agent systems." I think these results are pretty interesting and important, but better framed as disentangling how much of task failures are due to the model having different capabilities when it has to converse in dialogue (vs. operate in a single-agent setting), and how much of task failure is due to _collaboration_ specifically (e.g. failing to ask the right questions, share information with the user, etc.). As written, it sounds like "collaboration in general leads to better task performance," but since this is "collaboration with another model," I don't think it should be read as such.
This confusing conclusion also appears in the introduction: "In the simulated condition...collaborative agnets have a lower delivery rate as human-agent teams sometimes fail to reach their goals" (but in the simulated condition these are *agent-agent* teams?)

### Soundness
2

### Presentation
4

### Contribution
2
