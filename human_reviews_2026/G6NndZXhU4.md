# DRIP: Decompositional reasoning for Robust and Iterative Planning with LLM Agent

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Research on LLM agents has shown remarkable progress, particularly in planning methods that leverage the reasoning capabilities of LLMs. However, challenges such as robustness and efficiency remain in LLM-based planning, with robustness, in particular, posing a significant barrier to real-world applications. In this study, we propose a framework that incorporates human reasoning abilities into planning. Specifically, this framework mimics the human ability to break down complex problems into simpler problems, enabling the decomposition of complex tasks into preconditions and subsequently deriving subtasks. The results of our evaluation experiments demonstrated that this human-like capability can be effectively applied to planning. Furthermore, the proposed framework exhibited superior robustness, offering new perspectives for LLM-based planning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DRIP, a planning framework for LLM agents based on backward reasoning and task decomposition, aimed at enhancing robustness in long-horizon planning tasks. Its core contribution lies in formalizing a human-like problem decomposition mechanism for LLM planning, realized through the construction of a dynamic, goal-driven plan via an executability reasoning tree. Experimental results in both BlockWorld and Minecraft environments demonstrate its superior robustness compared to forward-reasoning baselines.

### Strengths
1. **Originality:** The paper presents a systematic implementation of backward reasoning for LLM planning, offering a clear and contrasting alternative to the predominant paradigm of forward reasoning.

2. **Quality:** The proposed method is well-designed and rigorously described. The experimental design is comprehensive, effectively validating the framework across both structured (BlockWorld) and open-world (Minecraft) tasks.

3. **Clarity:** The paper is clearly structured. The inclusion of overview diagrams, detailed algorithm pseudocode, and a comprehensive symbol table greatly aids in understanding the proposed framework.

### Weaknesses
1. **Insufficient Experimental Comparison:** The empirical evaluation lacks direct comparisons with other recent backward reasoning methods. This omission makes it difficult to precisely assess the unique advantages and distinctive contributions of DRIP within the landscape of backward reasoning approaches.

2. **Limited Generalizability Validation:** The framework's performance is validated only in the BlockWorld and Minecraft domains. Broader assessment on more diverse and realistic task benchmarks—such as robotic manipulation or everyday planning tasks—is needed to fully establish its general applicability.

3. **Incremental Nature of Contribution:** The core idea of backward reasoning is well-established in classical AI planning. While the work is solid, the primary novelty lies in its effective adaptation and demonstration using LLMs, rather than in introducing a fundamentally new reasoning paradigm.

### Questions
* Q1: Have the authors considered a hybrid approach that strategically combines DRIP's backward reasoning with elements of forward reasoning? This could potentially strike a more optimal balance between robustness and planning efficiency, mitigating the observed increase in subtask steps in Minecraft.

* Q2: How does DRIP handle scenarios with ambiguous goal states or multiple concurrent goals? Could the authors discuss the framework's stability in such settings and any potential strategies to address these challenges?

* Q3:Are there plans to evaluate DRIP in more complex simulated environments or on real-world physical tasks? This would significantly strengthen the claims regarding its practicality and robustness for real-world applications.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors consider planning an important problem, but current premier LLMs fall short of generating robust plans. The paper devises a planning process grounded in cognitive psychology, stating that the proposed DRIP framework leverages human-inspired decomposition to enhance LLMs’ planning capabilities. They argue that the novelty and effectiveness of DRIP lie in performing both forward, top-down reasoning and backward reasoning. The introduction is light on technical specifics and on clear statements of novelty; the authors should at least explain why backward reasoning is helpful. The tooth-brushing example is weak and offers little insight.

### Strengths
This paper points out that planning is an important problems, and LLMs could be helpful.

It causally reason that backward reasoning can be helpful to reduce computational cost.

Limited studies on "simple" planning problem seems to yield some improvement.  But the paper should step up to deal with serious planning problems such as supply-chain management, etc.

### Weaknesses
There are several shortcomings in this paper.

1. Related work coverage. It is not yet comprehensive for a planning paper. The section emphasizes decomposition and regression planning but misses four pillars that serious planners consider essential: uncertainty and belief tracking, plan repair and rollback, tool-grounded interaction, and memory or context management. It also needs a brief evaluation critique.

2. Native LLM limitations unaddressed. Context loss on long horizons is a known problem. Self-validation is inherently limited in light of Gödel’s incompleteness results. The paper does not discuss these issues.

3. Cost of backward search. Backward search can explode when many goal configurations are admissible. What constraints are used (landmarks, goal ordering, HTN templates, causal graphs) to prevent exponential cost? The empirical study should examine efficiency and effectiveness trade-offs.

4. Persistent memory. LeCun has noted that LLMs lack persistent memory and therefore struggle with long-horizon planning. This fundamental issue should be addressed.

5. Evaluation realism. The empirical study uses rudimentary problems and does not stress-test the proposed schemes. The authors are encouraged to consider planning work from the database and systems community since the 1980s, including the recent SagaLLM work (VLDB 2025).

### Questions
1. Positioning and related work. Can you provide a comparative assessment that covers:

* native LLM limits such as context loss and attention narrowing,
* structured speculative methods such as Tree of Thought and successors,
* persistent memory and transactional stability such as SagaLLM (VLDB 2025).

Explain why each is relevant or not to your planning setup and how your method addresses the gaps.

2. Does search complexity and pruning follow rigorous theories?

Both forward and backward reasoning can exhibit exponential branching. How do you constrain backward alternatives in practice? Specify the constraints and heuristics you use, for example landmarks, goal ordering, HTN templates, causal graphs, bidirectional search, or admissible heuristics, and report their effect on complexity and token cost.

3. Grounding, commonsense, and uncertainty. You did mention in "limitations" that commonsense could be an issue.  
How does DRIP handle commonsense and locale dependent logistics that break pure context reasoning, for example landing time versus airport exit time, baggage claim, customs, or rental queues? Describe your belief tracking under partial observability, your information gathering actions, and any tool based validation or buffer policies, and evaluate their impact on plan validity.

### Soundness
1

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
The paper proposes DRIP, a planner that alternates backward goal decomposition with forward execution. On BlocksWorld, DRIP outperforms CoT and ReAct in terms of success. In Minecraft, DRIP achieves the highest success rate, trading off trading-step efficiency for robustness via finer-grained subtasks. Overall, DRIP is a lightweight, LLM-agnostic approach that scales from classical to open-world tasks.

### Strengths
1) The paper solves an interesting problem. The method cleanly separates planning from execution and is explained with good figures.

2) The paper compares against LLM baselines (CoT, ReAct) across two domains.

3) Results show consistent gains on tasks in both a classical benchmark, BlocksWorld, and Minecraft.

### Weaknesses
1) BlocksWorld is altered (3 ops, multi-hold), which weakens comparability to prior work. I suggest that the authors 
   - Add a parallel track with the traditional 4-operator, single-gripper domain and 
   - Include classical planning baselines, such as Fast Downward. Report success, plan-length gap to optimal/classical planning baselines, and expansions/time.

2) The study did not use GPT-4 (only Claude 3.5) in Minecraft, so cross-model conclusions are thin. Please include Minecraft with GPT-4o. 

3) I also did not understand the "Manual" condition: specify the number of participants, the decision rules, the inter-rater checks, and whether participants could correct invalid steps. Please specify what the human condition actually was and the protocol.

4) The paper should also include an English variant as well; while not central, this may help eliminate any language effects on the accuracy (which look dismal)

- There are many minor grammar/punctuation issues; the paper requires a careful read.

### Questions
1) Why were classical planners not tested with conventional/original problem specifications? I strongly encourage adding that baseline (plus see my comments in the Weaknesses section).
2) What was the human protocol for Manual conditions?
3) Are the conditions met to use Fisher's test conditions met for your study design?
4) Will you add GPT-4o for Minecraft study?
5) Was there a specific reason for not comparing Japanese and English specifications?
6) Could you please provide citations to support this statement: "In contrast, LLMs offer a unique advantage in their ability to dynamically generate and adapt rules based on their extensive pre-trained knowledge."?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DRIP, a backward-reasoning, decomposition-first planning framework for LLM agents. Given a goal, an LLM recursively decomposes it into prerequisite subtasks; an executability module filters which subtasks can run under the current state; successful child nodes propagate executability upward (checkParentExec), yielding a plan. Experiments on BlockWorld (hard split, 6–15 blocks, modified rules to allow lifting stacks) and Minecraft (“mine diamond” from scratch) show higher robustness than forward approaches (CoT, ReAct). With Claude 3.7 Sonnet, DRIP hits 40.9% vs CoT 23.6% and ReAct 9.1% on BlockWorld; a manual-execution variant reaches 82.7%, indicating the gains are from planning rather than actuation. In Minecraft, DRIP succeeds on diamond 4/5 trials (ReAct 1/5, CoT 0/5). The paper is clear about limitations, LLM decomposition errors, reliance on natural-language state, more LLM calls than CoT, and occasional inefficiency in open-world tasks.

### Strengths
1. Clear decomposition loop with explicit executability check and upward propagation; easy to implement. 
2. Robustness gains on BlockWorld (large effect vs ReAct; solid vs CoT on Claude) and open-world Minecraft where many forward plans stall. Manual actuator study isolates planning quality from execution bugs, showing good methodology. 
3. DRIP uses ~4–5 fewer steps than baselines in successful cases. 
4. Honest limitations and discussion (need for formal state, LLM call budget, trade-off between step count and success).

### Weaknesses
1. Non-standard BlockWorld setup (multi-block lifting & holding) inflates branching and may favor the proposed decomposition; please also report standard constraints for comparability. 
2. Small-N in Minecraft (n=5 per resource) and single seed/model for many parts; results could be noisy. 
3. No comparison to planner-assisted LLMs (e.g., LLM+P/Task-graphs) or hybrid symbolic planners with LLM heuristics. 
4. Executability via natural language is brittle; the paper shows this, but there’s no quantitative analysis of that component (accuracy/confusion). 
5. Efficiency trade-off in Minecraft (more subtasks than ReAct in its single success) is under-analyzed. What’s the token/call budget? 
6. Novelty relative to recent backward-planning with LLMs (e.g., explicit backward search/goal regression) needs sharper positioning.

### Questions
1. Report DRIP/CoT/ReAct under the standard “one block in hand, must clear top” constraints. How do the conclusions change?
2. Provide a labeled set of state–action pairs to measure precision/recall of “Executable/Unexecutable/Unnecessary,” and error breakdowns that lead to plan failure.
3. Report tokens and LLM calls per solved instance; DRIP vs ReAct vs CoT, and for Minecraft, include code-generation retries.
4. (a) depth cap / tree-policy; (b) re-decomposition strategy; (c) swapping backward step with least-to-most prompting. 
5. Add a hybrid symbolic baseline (e.g., PDDL planner with LLM goal translation) or LLM+P. 
6. How sensitive are results to language (the BlockWorld prompts were in Japanese)? Any cross-language trials? 
7.  Can DRIP reconcile goal maintenance vs temporary goal violations (e.g., allowing undo/redo with bookkeeping)? 
8. Will you release code, prompts, and Minecraft environment scaffolding to ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
2
