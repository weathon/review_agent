# EvoCurr: Self-evolving Curriculum with Behavior Code Generation for Complex Decision-making

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
While large language models (LLMs) demonstrate remarkable capabilities across diverse domains, they fail catastrophically on high-complexity tasks requiring long-horizon reasoning and multi-step coordination. To address this problem, we present EvoCurr, a self-evolving curriculum learning framework that enables LLMs to solve complex decision-making problems through cooperative multi-agent learning. The core of EvoCurr is a multi-agent cooperative system where a Designer agent generates adaptive task sequences and a Solver agent produces executable solutions through coordinated interaction. Both agents share identical rewards based on task performance and proximity to the target task, creating a fully cooperative framework that naturally aligns their objectives for progressive skill acquisition. A critical innovation is the accepted-floor constraint that prevents difficulty regression below previously solved levels, ensuring monotonic skill advancement while preventing catastrophic forgetting. The framework enforces feasibility through a validation gate and supports both open-loop code generation and closed-loop policy learning paradigms. We evaluate EvoCurr on two complementary domains: StarCraft II micro-management and Overcooked coordination tasks. On StarCraft II micro-management, where the Solver generates Python behavior-tree scripts for complex tactical scenarios, EvoCurr achieves average combat winning rates above 90\% while state-of-the-art models achieve less than 50\% when directly attempting these scenarios. On Overcooked coordination tasks, where the Solver uses multi-agent reinforcement learning to train cooperative policies, EvoCurr achieves 20\% higher task completion rates (measured by dish orders delivered) compared to direct training. Our results demonstrate that EvoCurr provides a principled, domain-agnostic approach for extending LLM capabilities to complex decision-making tasks previously beyond their reach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces **EvoCurr**, a self-evolving curriculum learning framework designed to enable Large Language Models (LLMs) to tackle high-complexity decision-making tasks that require long-horizon reasoning and multi-step coordination, where LLMs typically fail. The framework operates as a cooperative two-agent system where an LLM **Designer** generates adaptive task sequences, and a **Solver** produces executable solutions, supporting both open-loop code generation (behavior trees) and closed-loop policy learning (Multi-Agent Reinforcement Learning). A key innovation is the accepted-floor constraint and feasibility gate, which prevent difficulty regression and ensure monotonic skill advancement by only accepting valid tasks above a previously mastered level. EvoCurr was evaluated on StarCraft II micro-management and Overcooked coordination tasks; in StarCraft II, it achieved combat winning rates over 90% in scenarios where direct code generation failed, and in Overcooked, it achieved 20% higher task completion rates compared to direct training, successfully extending LLM capabilities to previously unreachable complex tasks.

### Strengths
1. The proposed framework can be applied to a wide range of other tasks.

2. Compared to the baseline proposed in the paper, significant improvements were achieved in both StarCraft and Overcooked tasks.

### Weaknesses
1. Overall, the framework appears relatively simple, consisting solely of a designer and a solver. This makes the framework not entirely suitable for all tasks.  The author needs to provide further evidence to demonstrate the applicability of this framework. 

2. I have reservations about the novelty of the paper.  This paper designs a curriculumlearning framework and validates its effectiveness in the LLM domain. I personally argue that this framework represents a simplification of current coding agents such as Claude Code, Gemini CLI, and GitHub Copilot Agent. I wonder whether existing coding agents can accomplish this task after integrating with simulation testbeds like SC2. **Please note that Coding Agent is not equivalent to an LLM.**

3. The baseline for comparison needs to be more comprehensive. Currently, it only includes various base models without comparing other curriculum learning algorithms or agentic-type algorithms.

### Questions
1. In Section 3.1, how are the difficulty function and distance function defined?

2. Where are the results for GPT-5, Claude 4, and Gemini 2.5 mentioned in Line 323?

typo:
1. Line 324, there may be an extra "." here.
2. Line 355, "scores ¡ 0.5" should be sorces < 0.5？

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a curriculum learning method that helps LLMs solve complex, long-horizon tasks by progressively increasing task difficulty in a semi-automated manner. The proposed method, EvoCurr, consists of a designer that adaptively proposes new tasks while a solver produces solutions via behavior-tree code generation or multi-agent RL. Curriculum generation enforces two rules: the accepted-floor constraint, to avoid regressing below mastered difficulty levels, and a feasibility gate, to filter invalid tasks based on success rate. Experiments on StarCraft II micro-management and Overcooked coordination demonstrate that EvoCurr can significantly increase success rates compared to the baselines.

### Strengths
- The paper is easy to follow and well-written. The figures explain EvoCurr's components clearly, and the experimental setup and results are well demonstrated too.
- The designer-solver setup is a modular concept that connects curriculum learning to LLM-based solution generation.
- Using two difficult domains as examples for open-loop and closed-loop settings shows that the framework is general in certain aspects.
- Empirical evidence shows that EvoCurr has benefits over direct baselines.

### Weaknesses
- Although described as 'self-evolving', the method is not so automated as the difficult levels are pre-defined and do not depend on agent capabilities, but rather heuristically picked properties. So I'd call EvoCurr semi-automated.
- The proposed framework does not seem as novel and generalizable as it is described, as it is a heuristic adoption of existing ideas into two particular domains. For example, the enforced rules sound similar to return and task similarity constraints in self-paced learning, but they are less general.
- No ablation study is carried out.

### Questions
- What do you mean by self-evolving, if what and how it can evolve is already constrained based on pre-determined difficulty levels?
- Which rules affect the performance the most? An ablation study looking into this may be nice.
- How would EvoCurr generalize to domains where task similarity cannot be measured based on such properties?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EvoCurr, an adversarial-like curriculum learning framework. It has two cooperating agents: a Designer, which creates tasks with gradually increasing difficulty, and a Solver, which solves them. The goal is to help large language models (LLMs) handle more complex decision-making tasks. The authors tested their approach in two environments: StarCraft II micro-management and Overcooked, and claim their method achieves improved performance compared to direct baselines.

### Strengths
- The paper structure and presentation are clear and easy to follow.
- Figures and tables are well-made, helping readers easily grasp the main idea.

### Weaknesses
- The proposed approach isn't really new; similar frameworks have already been explored before. Even though the authors applied it to LLMs, it doesn't really introduce anything new.
- The method relies too heavily on heuristic choices and manually set hyper-parameters (like the difficulty measure $d(C)$ and acceptance threshold $\tau$). This makes the method seem overly simplistic.
- The experimental environments feel somewhat toy-like, raising questions about the generalizability of the results.
- Using acceptance rate as the curriculum metric seems arbitrary. It might be more reasonable to use loss signals or something more meaningful.
- The contributions listed in the intro feel exaggerated since the core ideas have appeared previously.

### Questions
- Can the authors provide more discussion about potential generalization beyond these simplified scenarios?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces EvoCurr, a self-evolving curriculum framework for inference-time problem solving with LLMs. A Designer LLM proposes the next task configuration; a Solver either (i) generates executable behavior-tree code (open-loop, interpretable) or (ii) trains a MARL policy (closed-loop) under a fixed budget. Two rules named accepted-floor (no regression below the last mastered difficulty) and a feasibility gate (syntax/logic/runtime checks) yield monotonic skill acquisition without manual difficulty metrics. The method reaches 90% win-rate on 12 SC2 micro-management tasks after 4–6 curriculum steps and delivers ~20% higher order completion on Overcooked than direct training under matched budgets.

### Strengths
- Inference-time curriculum with simple rules (accepted-floor + feasibility gate) removes the need for hand-crafted difficulty metrics and schedules. 

- The behavior trees and the syntax & code critic make debugging and analysis practical. 

- Compelling results on 12 SC2 tasks and Overcooked with clear acceptance criteria and curriculum traces.

### Weaknesses
- The major concern is the limited baseline. To more clearly clarify the advantage of the proposed method, the authors are recommended to compare the proposed method to stronger curriculum RL baselines  or strong search-based planners beyond the “one-shot direct code” baseline.

- Directly using code as policy may limit the capacity. Behavior-tree size and LLM context length may cap complexity; ablations on tree depth/lines of code vs performance would be useful.

### Questions
- Could the authors report the env-steps, compiles, wall-clock per accepted curriculum vs direct baselines to demonstrate its converging process?

### Soundness
3

### Presentation
3

### Contribution
3
