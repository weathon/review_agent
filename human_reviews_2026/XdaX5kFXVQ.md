# Tree-of-Options: Temporally Extended World Modeling, Planning, and Execution with Large Language Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
With commonsense knowledge embedded, Large Language Models (LLMs) have been repurposed as world models that can be exploited by principled planning algorithms such as Monte Carlo Tree Search (MCTS).
Prior works have been limited to exploiting LLMs for low-level world modeling, i.e., predicting immediate next world states and rewards upon primitive actions, which makes them unfit for long-horizon tasks where prediction errors compound quickly over time. 
This work develops an alternative framework where LLMs perform world modeling on temporally extended actions (options), to overcome their limitations in precise world modeling at small temporal scales.
At this temporal abstraction level, LLMs will also be competent in suggesting reasonable options, enabling effective planning using MCTS.
To execute the planned options with the primitive actions, we again turn to LLMs by prompting them to synthesize code implementing option-conditioned policies, which LLMs are known to excel at.
Empirical results in Minecraft show that this approach substantially improves performance over prior LLM-based planners on long-horizon, compositional tasks for embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Tree of Options (ToO), using temporally extended actions instead of primitive actions to improve large language models' performance in long-horizon, complex tasks. The authors choose Minecraft as the target environment and experiment ToO on four long-horizontal tasks against CoT and ToT+MCTS, showing promising results.

### Strengths
1. For long-horizon tasks, using options as instead of actions is straightforward and convincing. Options can be viewed as a higher level action comparing to primitive actions. Building the tree of options shortens the length of trajectory and facilitates LLMs to make decisions on a higher level.
2. Presentation of Figure 2 is straightforward, where ToO obtains a "flatter" distribution of lower-level attempts. Flatter distribution of lower-level attempts could potentially help LLMs jump out of local maxima and go to the ultimate goal faster.

### Weaknesses
1. Contribution is limited. I would like to view ToO as a variant of Language Agent Tree Search (LATS) [1], where the actions are replaced by options, where LLM initially critics on options and then try implements with actual feedbacks with environments.

2. The experiment results partially supports the argument, but not fully. The long-horizon tasks should be harder and longer, like some tasks where the Diamond tools are involved. Also, for all the tasks, the baselines (CoT and ToT + MCTS) are able to achieve successfully. Some results that the baselines cannot achieve within some reasonable extended budgets comparing to ToO would support the authors' argument much better, where options do help LLMs perform reliably in long-term, complex tasks.

3. The implementation of options is also limited. The authors use a prebuilt Voyager [2] skill library to implement Minecraft, and a trial-error fashion style based on the semantic similarity between options and skill library function names. While those implementations are simple and straightforward, a more general or principled way to implement execution of options would be appreciated.

4. The related work section needs to be better. For instance, this paper is not cited: "Mastering Board Games by External and
Internal Planning with Language Models" [3].

[1] Zhou, Andy, et al. "Language agent tree search unifies reasoning acting and planning in language models." arXiv preprint arXiv:2310.04406 (2023).
[2] Wang, Guanzhi, et al. "Voyager: An open-ended embodied agent with large language models." arXiv preprint arXiv:2305.16291 (2023).
[3] Schultz, John, et al. "Mastering board games by external and internal planning with language models." arXiv preprint arXiv:2412.12119 (2024).

### Questions
1. Would you mind providing additional experiments results on longer horizontal tasks? (Connection to weakness 2)

2. For the option, how many actor code generation attempts needed to fully implement the option? 

3. What's the difference between option and action? For me, it seems like the options would be slightly general queries to retrieve from Voyager [1] skill library for code implementations comparing to detailed verbal actions. This is a nice contribution with simple solutions but there's no fundamental differences between options and actions. 

4. What's the cost comparison among CoT, ToT + MCTS, ToO (iteration =5) and ToO (iteration = 10)? Dollar computation would be sufficient (including OpenAI embedding model costs for ToO).

[1] Wang, Guanzhi, et al. "Voyager: An open-ended embodied agent with large language models." arXiv preprint arXiv:2305.16291 (2023).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Tree of Options (ToO) framework, a novel planning method designed to enhance the performance of Large Language Models (LLMs) on long-horizon embodied decision-making tasks. The framework integrates Monte Carlo Tree Search (MCTS) with a high-level, LLM-based world model to plan over an abstract space of temporally extended actions, or "options." Experiments conducted in the Minecraft environment demonstrate that ToO achieves superior efficiency and reliability on complex tasks when compared to baseline methods.

### Strengths
- By planning at the "option" level, the framework astutely leverages the high-level reasoning strengths of LLMs while mitigating their known weaknesses in handling low-level control details. The integration of MCTS with an option-level world model is a well-motivated and insightful contribution.
- The choice of Minecraft as a testbed for long-horizon tasks is highly appropriate. Furthermore, the comparison against the ToT-MCTS baseline effectively ablates and highlights the core contribution of the option-level world model.

### Weaknesses
- The framework involves multiple nested LLM calls within a single MCTS iteration, which appears computationally expensive. The paper does not discuss the associated costs or the scalability of this approach.
- The success of each component (e.g., dynamics model, feasibility check) is highly dependent on carefully engineered prompts. This approach can be brittle and may require significant re-engineering for new domains or even for slightly different tasks, raising questions about its generalizability.
- While "actor attempts in code generation" is a useful proxy for efficiency, the evaluation would be more persuasive if it also prominently featured standard metrics such as overall task success rate, total wall-clock time, and the total number of LLM API calls, some of which are currently in the appendix.

### Questions
- Given that ToO involves multiple nested LLM calls per MCTS iteration, could the authors provide a quantitative analysis of the planning costs (e.g., number of API calls, token consumption, latency) and discuss the framework's scalability as the number of MCTS iterations and the branching factor increase?
- The current implementation relies on one-shot planning. How might the system adapt to significant discrepancies between the world model's prediction and the actual outcome of an action? A discussion of strategies for online replanning and the associated trade-offs with the high planning cost would be valuable for assessing the framework's practicality.
- Could the evaluation be made more comprehensive by supplementing the primary metric ("number of code generation attempts") with standard metrics like task success rate and total execution time in the main text?
- It would be interesting to hear the authors' perspective on the trade-offs of the iterative code generation approach—a form of trial-and-error learning—compared to traditional reinforcement learning methods for executing complex options.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Tree of Options (ToO) — a framework that integrates Large Language Models (LLMs) into temporally abstract world modeling and planning via Monte Carlo Tree Search (MCTS). Unlike previous LLM-based planners that operate on primitive actions (leading to compounding prediction errors), ToO models temporally extended actions (“options”) in natural language form.

### Strengths
1. Introduces a language-level option framework—bridging classical hierarchical RL (options) and LLM reasoning.
2. Integrates LLMs into MCTS for structured exploration and evaluation over temporally extended actions. The combination of option-driven dynamics and reward predictors gives a coherent planning architecture.
3. Uses LLMs’ strength in program synthesis to translate abstract options into executable skills. Iterative refinement with feedback increases robustness against generation errors.

### Weaknesses
1. The approach depends on carefully hand-engineered prompts (for option generation, feasibility checks, reward prediction, etc.).
2. While qualitative trajectories are shown, there is limited quantitative analysis on computational cost, token usage, or LLM query efficiency compared to baselines.
3. The paper lacks a formal discussion of convergence, optimality guarantees, or the relationship between option-level abstraction depth and search efficiency.
4. Ablations isolate world modeling and feasibility checks but do not examine how MCTS hyperparameters, rollout length, or option vocabulary size affect performance

### Questions
More discussions about Weaknesses.

### Soundness
3

### Presentation
2

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
This paper presents ToO, a framework that lets LLMs plan and act through temporally extended actions instead of step-by-step moves. It combines an LLM-based world model, which predicts how the environment changes when a high-level option is executed, with a MCTS  planner that selects the best option sequence. The system then prompts the LLM again to generate executable code for each option. Experiments in Minecraft show that it handles long, multi-step tasks more reliably than Chain-of-Thought or Tree-of-Thought baselines, producing steadier and more feasible plans.

### Strengths
- The pape introduces the options concept from RL into the context of LLM-based planning.

- The experiments provid qualitative analyses of the slow is fast, option dependency stability, and the role of feasibility validation, which make the behavioral insights richer.

### Weaknesses
- The overall presentation of the paper could be improved. Figure 1 lacks clear annotations and does not clearly illustrate how the world modeling component is integrated into the framework. 

- The main results section relies heavily on visualizations (Figures 2–4). As these figures are not sufficiently explained, it takes some effort for me to understand their logic. Since they mainly show a few specific tasks, the results feel more like case studies. It would greatly strengthen the paper to include a summary table with clear metrics such as task success rate or average step length.

- All experiments are conducted in the Minecraft environment with very similar task settings (mining, crafting, milking, etc.). The paper would benefit from validation in additional environments.

### Questions
- When an option is judged infeasible, how exactly does the feasibility module generate the “alternative actions”? Is there a specific prompting strategy or rule to ensure these replacements remain consistent and task-relevant?

- Could the authors clarify how the weights in Eq. (6) are determined?

### Soundness
2

### Presentation
1

### Contribution
2
