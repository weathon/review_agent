# Expanding the Action Space of LLMs to Reason Beyond Language

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Language Models (LLMs) are powerful reasoners in natural language, but their actions are typically confined to outputting vocabulary tokens. As a result, interactions with external environments---such as symbolic operators or simulators---must be expressed through text in predefined formats, parsed, and routed to external interfaces. This overloads the model's language with both reasoning and control duties, and requires a hand-crafted parser, external to the LLM. To address this, we decouple environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary. The model starts reasoning in the default language environment, but may trigger routing actions and switch to an external environment at any time. From there, the model can only invoke environment-specific actions, receive feedback from the environment, and potentially route back to language as a result. To promote effective exploration of the expanded action space and new environments, we introduce ExpA Reinforcement Learning (EARL) with counterfactual policy optimization.
On tasks requiring multi-turn interactions and contingent planning, EARL outperforms strong baselines with vocabulary-constrained actions. It performs robustly across calculator-based multi-task learning and, in the partially observed sorting problem, achieves perfect Sort-4 accuracy while self-discovering an efficient algorithm competitive with classical designs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ExpA, a method to add additional non-language tokens to LLMs for environment interaction. It also introduces EARL, a reinforcement learning algorithm training LLM for better using ExpA. It conducts experiments comparing with standard prompting and GRPO fine-tuning methods, showing strong performance on some mathematical reasoning and sorting tasks.

### Strengths
+ ExpA is a good intuitive method to explore how LLM should interact with the environment. Instead of directly using natural language, the additional action tokens introduced by ExpA can ensure more accurate interactions.
+ EARL is a novel RL method for fine-tuning LLMs for ExpA. The counterfactual rollouts ensure efficient exploration of the newly-added action space compared to the standard GRPO.
+ Experiments show strong performance on Qwen-2.5 3/7B models compared to non-ExpA natural language methods and/or standard GRPO.

### Weaknesses
+ The benchmark in the evaluation is not very fascinating. The usage of additional calculator has far been explored by many works. The usage of LLM to sort sequences are also less interesting, as it can be easily solved by many traditional non-machine learning algorithms. It could be much more interesting if the author can show the application of ExpA in robotics tasks.
+ Countdown and GSM8k\* use extremely large numbers, which is naturally a good fit of ExpA since it can learn how to use external calculator. However, such kind of tasks are not generally the use case of LLMs (people can use simply DFS for Countdown, and the GSM8k\* is a modified version just to enhance the importance of a calculator). Thus, the chosen application does not show the strong potential in applications of LLMs of introducing additional action tokens. The same also holds for Arithmetic.
+ For Arithmetic and Count, other two Calc-Bench tasks, ExpA does not show very strong performance against standard SFT/prompt + GRPO methods.
+ Table 2 lacks a baseline of Prompt+GRPO (presented later in ablation only for Countdown). The current numbers look that Prompt+CPO is worse than Prompt+GRPO. It does not show clear advantage of CPO in general domain.
+ The paper lacks some baselines of introducing external action tokens (such as ToolkenGPT). It only compares with baselines without such action tokens.

### Questions
+ Is the model fine-tuned on each of the task, or there is only one model fine-tuned with mixture of the tasks?
+ Please see weakness.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ExpA—an expanded action space that adds routing and environment-specific actions to an LLM’s policy—and EARL, an RL recipe that encourages the LLM to use these actions. The agent begins in a language environment; it can route into a tool (e.g., a calculator), where decoding is constrained to that tool’s valid actions, receives observations back as text, and then routes back to language. This design explicitly decouples reasoning tokens from control actions, eliminating the need for hand-engineered string parsers that detect special tool blocks. Weight initialization ties new action logits to the embeddings of their textual descriptions, and CPO injects counterfactual “forced routing” at likely positions inferred from the pretrained next-token distribution. Empirically, on an Arithmetic task, Countdown, GSM8K*, and a count-based task, as well as a partially observed Sorting setup, the approach often outperforms vocabulary-only baselines (Prompt+GRPO, Prompt+CPO, SFT+GRPO).

### Strengths
* **Conceptual clarity & clean mechanism.** The MDP framing and Algorithm 1 make the control flow crisp: select a routing action, then act within the environment from an action set, append observations to the history, and return to language.  
* **Well-motivated learning recipe.** The policy head covers both vocabulary and expanded actions but is environment-masked; new action logits are initialized from their description tokens, giving a strong prior, and counterfactual rollouts encourage invoking tools at plausible points.

### Weaknesses
I sort them in decreasing order of importance to my score. I am primarily concerned about the first two, which, if fully addressed, would encourage me to increase my score.
* **The source of gains is under-isolated.** Two plausible drivers are (i) cleaner, in-distribution inputs to the LLM while inside tools (as seen in the case study appendix), and (ii) action-space restriction (fewer invalid generations along the learning process). Adding appropriate baselines (e.g., language-only tool calls with subset-token decoding when inside <calculator>, or cleaning the format of tool calls after returning)—is missing and would provide significantly more substantial insights.  
* **CPO vs GRPO effects are mixed across tasks/baselines.** While EARL (ExpA+CPO) is often best and converges quickly on Countdown, some tables and plots show baselines where GRPO rivals or beats CPO on subtasks.  
* **“Removes reliance on external parsers” is slightly overstated.** Routing/dispatch is no longer string-parsed, which is a real advance; however, the system still needs parsers for the routing tokens and appends textual descriptions of actions to history for learning/context, so scaffolding remains.   
* **Positioning vs prior “expanded-action” agents.** The paper contrasts ExpA with a token-only tool calling, but the related work discussion underplays earlier systems that already output non-linguistic control (e.g., VLA/robotic policies like RT-2 \[1\]). Tightening this comparison would prevent over-claiming novelty and clarify what is new.  
* **Reproducibility details.** Appendix configs reference many flags without a compact table mapping them to semantics, or without clearly explaining which common configs are used across different methods and what the specific differences are. 

\[1\] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., Florence, P., Fu, C., Arenas, M. G., Gopalakrishnan, K., Han, K., Hausman, K., Herzog, A., Hsu, J., Ichter, B., . . . Zitkovich, B. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *ArXiv*. [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Expanded Action Space (ExpA), a novel framework for training language models to interact with external tools and environments by directly selecting actions, rather than issuing text-based commands. The authors contend that existing tool-using LLMs depend on special tokens and external parsers, which hinders end-to-end learning and optimization. ExpA instead offers a unified action space where the model can either produce language tokens or choose environment actions, allowing reinforcement learning to jointly optimize reasoning and tool-use behaviors. To support this, the authors propose EARL, a reinforcement learning algorithm that incorporates counterfactual policy optimization to help the model identify when tool usage is advantageous. Experimental results on arithmetic and sorting tasks demonstrate improved performance and the emergence of procedural strategies, which the authors argue showcases ExpA’s potential to develop more capable and principled LLM-based agents.

### Strengths
1. The paper introduces the Expanded Action Space (ExpA), allowing language models to interface with external environments by selecting environment-specific actions directly, rather than encoding tool use through special tokens and external parsers. This clean separation between language reasoning and environment control supports true end-to-end training.

2. The authors present EARL (ExpA Reinforcement Learning), which incorporates counterfactual policy optimization (CPO) to encourage exploration of infrequent but critical routing actions.

3. Across multi-step, contingent-planning tasks, EARL consistently outperforms strong baselines that restrict actions to the model’s vocabulary. Ablation studies highlight the effectiveness of CPO over GRPO and demonstrate that the approach is effective even with base models lacking supervised fine-tuning.

4. On the sorting task (Sort-4), EARL achieves 100% accuracy, with the resulting policy converging to a compact, near-optimal decision tree in terms of swaps and comparisons.

### Weaknesses
-The work is a natural extension of RL-formulated code execution agents (e.g., LEVER, RA-ABL, RCI, CodeAct), tool-use planners with policy heads, and counterfactual credit assignment in modular RL. The abstraction change (tokens → actions) is in line with past works ideas. 

-comparisons against strong modern baselines (SFT+function calling, ReAct-style agents, grammar-based parsers) under identical budgets are lacking. Moreover, ablations study that isolate the effect of counterfactual routing is missing.

-The paper argues for an “RL-first” tool-use paradigm, but the experiments do not convincingly isolate RL’s necessity or superiority. The results show that performance improves under specific setups, yet there is no systematic comparison against modern SFT+RLHF tool frameworks or hybrid architectures. The counterfactual rollout idea is interesting, but the description and ablation depth are insufficient to confirm it is the primary driver versus simpler credit assignment heuristics.

-Moreover, environments used in experiments are toy-scale. Evaluation is limited to small, closed environments (arithmetic helpers, sorting tasks). These settings, while pedagogically clean, do not demonstrate scalability to heterogeneous real-world tools (APIs, OS environments, web search, memory systems, code interpreters). The method’s practicality in complex, partially observable, or noisy environments remains unclear. As a result, the claim of enabling “general tool-using agents” feels premature.

### Questions
-The paper claims to “remove the need for parsers” and introduce a fundamentally new paradigm, but this claim is weakened by the lack of engagement with prior work in RL-based tool agents, program-induction systems, and LLMs with environment-action heads (e.g., LEVER-like systems, code-exec agents, policy-head planning architectures, and work in embodied LLM agents). The contribution reads primarily as re-applying standard RL policy-head design to LLM tool use. The shift from textual tool invocation to direct action logits is a conceptual refinement, but the paper does not clearly differentiate itself from existing literature on action-space augmentation or structured decision heads in language-conditioned agents.  Can authors provide stronger comparative analysis to justify novelty.

-To strengthen the general-agent claim, the authors should evaluate the approach in richer, noisy, and heterogeneous tool settings rather than only toy arithmetic and sorting domains. In particular, it would be compelling to show that ExpA scales to real-world tools such as web search/browsing, OS/file manipulation, code execution, and persistent memory—and can operate across multiple tools within a single task. 

-Experiments that incorporate partial observability, API failures, rate limits, and evolving tool schemas would demonstrate robustness and practical viability. Further, testing zero-shot generalization to unseen tools given only natural-language/tool-spec descriptions would clarify whether the architecture truly enables flexible tool use rather than overfitting predefined action templates. 

Finally, comparisons against strong modern baselines (SFT+function calling, ReAct-style agents, grammar-based parsers) under identical budgets, and ablations isolating the effect of counterfactual routing, would help validate that the proposed action-space formulation—and not just engineering scaffolding—drives the observed performance gains.

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
The paper proposes Expanded Action space (ExpA), a policy class where an LLM’s choices expand from vocabulary tokens to actions. The model can either stay in the language environment or route into an external environment and then select environment-specific actions until it routes back. To make models actually use these new actions, the authors introduce ExpA RL (EARL) with counterfactual policy optimization (CPO) trick. For each rollout, they build a counterfactual trajectory to encourage exploration of rarely invoked but crucial tools. On two benchmarks Calc-Bench and Sort-4, the approach outperforms prompt+RL and SFT+RL baselines.

### Strengths
1. A clean reframing of tool use as an explicit expanded action space, removing parser scaffolding.
2. The approach shows great improvement on the benchmarks (+10.46 EM overall), and it could have positive impacts on future LLM action capability and Agent development.
3. The paper is well written and easy to follow. It gives explanations of most of the technical details.

### Weaknesses
1. ExpA replaces an external parser with a learned action head, but observations are still appended as language tokens. This looks like a relocation of the parser boundary rather than a fundamentally different interface.
2. For large |E|, the performance of the approach is unclear. The author said the approach is a “scalable framework”, but this is a forward-looking claim. The paper doesn’t quantify behavior as the number of tools/actions grows.

### Questions
Most experiments are performed on Qwen 2.5 0.5, 3, 7b. Is the method still work on other models with different architectures or larger parameters?

### Soundness
2

### Presentation
3

### Contribution
2
