# Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Humans learn abstractions and use them to plan efficiently to quickly generalize across tasks---an ability that remains challenging for state-of-the-art large language model (LLM) agents and deep reinforcement learning (RL) systems.
Inspired by the cognitive science of how people form abstractions and intuitive theories of their world knowledge, Theory-Based RL (TBRL) systems, such as TheoryCoder, exhibit strong generalization through effective use of abstractions. However, they heavily rely on human-provided abstractions and sidestep the abstraction-learning problem.
We introduce TheoryCoder-2, a new TBRL agent that leverages LLMs' in-context learning ability to actively learn reusable abstractions rather than relying on hand-specified ones, by synthesizing abstractions from experience and integrating them into a hierarchical planning process. We conduct experiments on diverse environments, including BabyAI and VGDL games like Sokoban.
We find that TheoryCoder-2 is significantly more sample-efficient than baseline LLM agents augmented with classical planning domain construction, reasoning-based planning, and prior program-synthesis agents such as WorldCoder. TheoryCoder-2 is able to solve complex tasks that the baselines fail, while only requiring minimal human prompts, unlike prior TBRL systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TheoryCoder-2 is a theory-based reinforcement learning (TBRL) agent that leverages LLMs’ in-context learning to automatically synthesize reusable high-level abstractions from experience and integrate them into hierarchical planning for program-synthesis agents. It is evaluated on environments including BabyAI and VGDL games (e.g., Sokoban), compared to LLM-augmented classical planning, reasoning-based planners, and prior program-synthesis agents. TheoryCoder-2 demonstrates substantially improved sample efficiency and generalization—solving complex tasks that baselines fail—while requiring minimal human guidance.

### Strengths
- Innovative integration of LLMs and TBRL. TheoryCoder-2 advances theory-based RL by using LLMs to learn high-level abstractions (PDDL operators) rather than relying on hand-coded ones.

- Hierarchical coupling of learned abstractions + executable world model. The combination of PDDL-level planning with a learned Python transition model (plus predicate classifiers) is a principled way to ground abstractions for execution.

- Comprehensive evaluation with multiple baselines, ablation studies, and diverse metrics (token cost, compute time, success rate)

### Weaknesses
- Limited analysis of abstraction quality. While the paper highlights the ability to actively learn reusable abstractions rather than relying on human-provided ones, it remains unclear what kinds of abstractions are newly discovered beyond those already captured by hand-specified ones. The paper would benefit from a deeper analysis showing whether the learned abstractions differ qualitatively or functionally from human-designed ones, and to what extent they are more general, compositional, or transferable.

- Limited domain diversity / ecological validity. Experiments are restricted to VGDL grid games and BabyAI — discrete, object-oriented toy domains.

### Questions
1. Is it possible to incorporate objective measures of abstraction quality—such as correctness, granularity, or redundancy—to more rigorously assess the learned abstractions?

2. Could the paper include concrete examples or comparisons showing cases where the learned abstractions outperform or complement human-specified ones? 

3. The solution rate of TheoryCoder-2 appears to remain below 90%, could the paper analyze or discuss the primary failure reasons?

4. Is it feasible to adapt or extend TheoryCoder-2 to more diverse or ecologically valid domains?

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
4

### Summary
This paper presents TheoryCoder-2, a new TBRL method that learns reusable abstractions for hierarchical planning. Unlike prior systems such as TheoryCoder (Ahmed et al., 2025), which rely on manually provided PDDL abstractions, TheoryCoder-2 uses LLMs to synthesize abstractions from experience via in-context learning. The basic idea is to incrementally grow a library of abstractions. Empirically, TheoryCoder-2 has better sample efficiency and generalization compared to baseline LLM-based planners (LLM + P, WorldCoder, reasoning LLMs), solving tasks that others fail on, while requiring minimal human input.

### Strengths
-  The paper addresses a limitation in TBRL and many program-synthesis agents (dependency on human-input). The proposed approach is an interesting solution to this issue.

- To the best of my knowledge, the combination of (i) PDDL abstraction synthesis via in-context LLM prompting, (ii) creation of a curriculum-based abstraction collection, and (iii) grounding through low-level Python models is novel.

- The idea fits ICLR.

### Weaknesses
To me, the main weakness of the paper is the scope of the experimental analysis. The experiments are not sufficient for a strong, general conclusion that TheoryCoder-2 achieves scalable, human-like abstraction learning. There is only one curriculum with eight problems evaluated in the experimental results, so I am not confident taking conclusions from it. 

Usually, I think that good ideas can overcome bad experimental results, but in the case of this paper, the main motivation is to show that the method "is a significant improvement in the applicability of TBRL, and an important step towards building AI systems that learn like humans". For that, I would need to see much more challenging and meaningful benchmarks.

On top of that, I am quite confused by section 4.1 First, comparing Sokoban to Maze and Labyrinth is like comparing the Earth with a soccer ball. I find it quite hard to believe that the `moveontop` abstraction would not *hurt* performance in Sokoban.The fact that Sokoban seems much easier (from the raw data) than the other two, makes me think that the task used is not appropriately challenging. Additionally, Labyrinth and Maze are just isomorphic problems, so I find the presence of both in the curriculum a bit underwhelming.

Last, a minor complain about the presentation: I think the paper deserves a detailed revision and editing. In the first half of the paper, in particular, I found that many passages had words that were used with some very ad-hoc meaning. While this is often harmless and just a matter of style, it gave me an impression that the paper had been rushed and the text was too vague for a scientific paper.

Example: (italics are mine)

> The domain file specifies abstract actions (called “operators”, e.g., “open door”) and their preconditions/effects, as well as abstract
states (e.g., “door unlocked”) which are *summarized* through Boolean predicates that capture task-relevant features. The problem file specifies the initial state and goal conditions for a particular task. Together, these files are *consumed* by a *classic* PDDL planner.

None of the marked words is intended to mean what it says in the text: "summarized" means "fully described"; "consumed" simply means "read'; "classic" is probably just a typo and I assume you mean a "classical planner". (And, of course, the description of a PDDL domain and instances is incomplete.)

### Questions
Could you please explain me the Sokoban task you had in the curriculum and how the moveontop abstraction actually help there? I am clearly missing something.

Do you have results for other curricula that do not involve moving on a grid?

### Soundness
3

### Presentation
2

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
The paper introduces TheoryCoder-2, a system that solves agent tasks through learning reusable abstractions to solve new tasks. It focuses on the domain of the TBRL paradigm, which attempts to model the environment and interactions through program synthesis. This work builds on theoryCoder, which represents theory through Python programs. It mainly improves in the way that it does not require hand-crafted abstractions. Instead, it only needs a task-agnostic, high-level prompt and enables the system to automatically learn abstractions through in-context learning ability of language models. The system is also able to grow its learned library through further interaction with the environment. The paper evaluates on environments such as BabyAI and VGDL games, and shows that their method has good accuracy with  great sampling efficiency, compared to other frameworks that also utilize LLMs.

### Strengths
- The paper is well-written and very easy to follow. It introduces a domain, the task, and method in a way that is friendly to the broader RL/ML community.
- The improvement from TheoryCoder to TheoryCoder2 is significant. It gets rid of domain specific human annotation, which shows potential to scale to more complex, unseen environments
- Using programs as a representation of complex environments and their transitions is interesting and shows a direction of world model.

### Weaknesses
- Though the modeling through programs works well on these environments, I am unsure about how the TBRL paradigm and TheoryCoder2 will perform on more complex, real-world tasks. Recent advancements on Agent Task have proposed more complex tasks e.g., ALFWorld, Webshop, WebArena. I doubt that this paradigm and method can still discover and model the transition function. This is because purely modeling the very complex environment with code is difficult (if not impossible). The authors may want to show that such method may also work in more complex environments through experiments.

- The experiment is not sufficient. More specifically, it chooses weaker baseline methods for comparison. For example, (according to citation), ReAct is used as a baseline to be compared. While ReAct is a famous paradigm, several methods have been proposed based on it, and have achieved better results. The authors may need to add experiments that compare more stronger baseline. Moreover, it seems that the LLM used for different methods are different: some baseline methods use o4-mini and the proposed method use GPT-4o, the author may consider keep them consistent.

### Questions
- See weaknesses. 

- Could the authors elaborate on the potential to extend this framework to embodied tasks?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TheoryCoder-2, a theory-based reinforcement learning (TBRL) agent that learns reusable abstractions from experience via large language models’ in-context learning ability. Unlike previous systems that rely on hand-crafted abstractions, TheoryCoder-2 autonomously synthesizes and grounds abstractions for hierarchical planning. Experiments on BabyAI and VGDL games demonstrate performance improvement in sample efficiency and generalization over LLM-based and program-synthesis baselines.

### Strengths
- The paper focuses on an interesting and timely direction: automatically learning reusable abstractions with LLMs. 
- The presentation is clear and the paper is generally easy to follow.

### Weaknesses
- The method resembles prior work on learning and reusing skills through curriculum learning and code-based representations. The main difference here is expressing these skills in the PDDL format, but similar high-level ideas (e.g., learning, storing, and recalling code snippets or functions for reuse), have been explored in many recent works such as [1]. However, these related baselines are not discussed in sufficient depth, which makes it hard to assess what is genuinely new beyond the specific implementation.
- The paper does not include a direct comparison between TheoryCoder-2 and human-coded abstractions (e.g., hand-crafted PDDL operators). Without this comparison, it is difficult to understand the gap between the generated abstractions vs human-written ones. 

---
[1] Liu, Anthony Z., et al. "Interactive and Expressive Code-Augmented Planning with Large Language Models." arXiv preprint arXiv:2411.13826 (2024).

### Questions
Have the author tried running with different model scales (e.g., GPT-4o vs smaller models). How sensitive is the abstraction quality to model size or reasoning capacity?

### Soundness
2

### Presentation
3

### Contribution
2
