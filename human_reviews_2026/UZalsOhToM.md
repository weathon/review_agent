# LogicGuard: Improving embodied LLM agents through temporal logic based critics

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large language models (LLMs) have shown promise in zero-shot and single step reasoning and decision-making problems, but in long-horizon sequential planning tasks, their errors compound, often leading to unreliable or inefficient behavior. We introduce LogicGuard, a modular actor–critic architecture in which an LLM actor is guided by a trajectory-level LLM critic that communicates through Linear Temporal Logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. LogicGuard supports both fixed safety rules and adaptive, learned constraints, and is model-agnostic: any LLM-based planner can serve as the actor, with LogicGuard acting as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LogicGuard to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. To demonstrate generality, we evaluate LogicGuard across two distinct settings: short-horizon general tasks and long-horizon specialist tasks. On the Behavior benchmark of 100 household tasks, LogicGuard increases task completion rates by 25\% over a baseline InnerMonologue planner. On the Minecraft diamond-mining task, which is long-horizon and requires multiple interdependent subgoals, LogicGuard improves both efficiency and safety compared to SayCan and InnerMonologue. These results show that enabling LLMs to supervise each other through temporal logic yields more reliable, efficient and safe decision-making for both embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an actor-critic method consisting of two parts. An LLM-actor, which can be any LLM-based planner, and an LLM-critic, called LogicGuard. LogicGuard uses trajectory history to infer rules in a reactive fragment of LTL, which are then used in conjunction with a verifier to allow the actor’s actions to proceed or to provide feedback to the actor. The proposed method is evaluated in two domains with two different planners, showing an increase in task completion and efficiency over implementations that do not use LogicGuard.

### Strengths
This work tackles an important question and is mostly well-written. This is a very crowded area and the authors have taken great pains to make clear the distinction between this paper and related work.

The proposed architecture makes a lot of sense, and the fact that it contains a verifier after *both* LLMs helps avoid the problem of “LLMs all the way down.”

It is nice to see efficiency gains as well as safety improvements in the results.

### Weaknesses
The authors overstate the application of LTL in this work. The work uses a *fragment* of LTL, that is purely reactive. That is, all specifications are of the form “Always preconditions implies next postconditions”. This may be a useful fragment, and indeed the results show some positive effect. But the authors should be very clear up front about the fact that their work is for a very small fragment of LTL.

The graph planning portion is not very clear to me. It is quite common in LTL to use graph traversal for planning, even with reactive fragments. See for example [1] for incremental graph construction and pruning, and [2,3] for reactive synthesis as a graph game. How does the graph approach compare to these? Is there any novelty to this part compared to standard work in LTL?

I’m not sure the communication via LTL is entirely novel. There is plenty of work in automatically generating LTL specifications from examples using LLMs. For example, [4] and [5]. The authors should comment on the novelty of this aspect.

[1] Vasile CI, Li X, Belta C. Reactive sampling-based path planning with temporal logic specifications. The International Journal of Robotics Research. 2020;39(8):1002-1028. doi:10.1177/0278364920918919

[2] Ehlers, R., Khalimov, A. (2024). “Fully Generalized Reactivity(1) Synthesis”. Tools and Algorithms for the Construction and Analysis of Systems. TACAS 2024.

[3] Bloem, Roderick, Krishnendu Chatterjee, and Barbara Jobstmann. "Graph games and reactive synthesis." Handbook of model checking. Cham: Springer International Publishing, 2018. 921-962.

[4] Vazquez-Chanlatte, Marcell, et al. "L LM: Learning Automata from Demonstrations, Examples, and Natural Language." (2025).

[5] Gupta et al. “Integrating Explanations in Learning LTL Specifications from Demonstrations”

### Questions
How is S_unsafe determined? Is it entirely inferred by the critic?

See above for questions about graph planning.

How necessary is the LLM as critic? It seems like something like [6] could be used to mine these specifications. Is there a reason it must be an LLM?

Can this method extend to more general forms of LTL specifications?

[6] Hasanbeig, Mohammadhosein, et al. "Deepsynth: Automata synthesis for automatic task segmentation in deep reinforcement learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 9. 2021.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LogicGuard, a modular actor–critic framework where the actor is guided by a trajectory-level LLM critic, communicating through Linear Temporal Logic (LTL). The proposed method improves task completion rates on both short-horizon and long-horizon tasks.

### Strengths
1. The system combines symbolic reasoning with the generalization ability of LLMs.
2. Every constraint has a verbalized explanation, helping users inspect the agent’s decision process.

### Weaknesses
1. Offline critic analysis and frequent rule updates require repeated LLM calls, which can be expensive and potentially unstable.
2. This work hand crafts a large number of rules and complex prompts. This constraints the environment and lowers the task difficulty.

### Questions
1. Can learned LTL rules transfer across tasks or environments, or must they be relearned each time?

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
5

### Summary
LogicGuard proposes a modular actor-critic architecture where an LLM actor generates high-level actions while a trajectory-level LLM critic communicates constraints via Linear Temporal Logic (LTL). The system addresses compounding errors in long-horizon sequential planning by automatically generating formal constraints that guide LLM-based agents. Evaluated on Behavior-100 (household tasks) and Minecraft diamond-mining, LogicGuard achieves 25% improvement in task completion rates and 23% efficiency gains over baseline InnerMonologue, while enabling SayCan to complete previously impossible tasks. This work presents itself as actor-critic framework but this work appears as LLM as Jude. Paper is written very nicely and enjoyed reading the paper.

### Strengths
1. Novel Integration of Formal Methods with LLMs:
The use of LTL as a communication protocol between actor and critic is innovative and addresses a critical gap in LLM-based planning. Prior work using natural language feedback, LTL provides verifiable, machine-checkable constraints with formal guarantees but this work went ahead to have a online actor critic.

2. Strong Empirical Results Across Diverse Settings:
The paper demonstrates substantial improvements in both generalist (Behavior: 47%->72% completion) and specialist settings (Minecraft: 23% efficiency gain, 23%->4.5% failure rate). The fact that SayCan completely fails without LogicGuard (0/5 success) but succeeds consistently with it (5/5) is particularly compelling.

3. Modular and Model-Agnostic Design:
The architecture allows direct integration with existing LLM planners (InnerMonologue, SayCan) as a "logic-generating wrapper," enhancing practical applicability. The separation of online actor and offline critic loops is well-motivated.

4. Theoretical Grounding:
Theorem 1 bounds the number of possible LTL rules generated from trajectories, providing formal guarantees about constraint space complexity. The graph-theoretic formulation of planning as bipartite graph traversal is elegant

5. Automatic Atomic Proposition Generation:
The automated AP generation for Behavior dataset demonstrates scalability potential, though hand-engineered APs remain necessary for specialist tasks

### Weaknesses
1. Missing ablations and baselines:There is no direct comparison between online vs. offline critic modes, which is crucial to understand trade-offs in adaptability and computational overhead. Baseline coverage (e.g., other LLM-based critics or symbolic guardrail methods) is limited. What is the contribution of each critic source (environment feedback, graph-based efficiency, over-constrained states)?

2. Evaluation scale and statistical rigor: Reported experiments appear based on small sample sizes (e.g., 5 trials for Minecraft) and lack confidence intervals or significance testing. Reported gains, while large, are not statistically supported. 

3. Sensitivity to AP design: The system heavily relies on the quality and coverage of atomic propositions (APs). The paper itself acknowledges failures in auto-generated APs but provides no quantitative analysis or robustness study.

4. Limited theoretical contribution: Theorem 1 offers only a trivial bound on rule count rather than a meaningful performance or safety guarantee. Either the analysis should be expanded or reframed more modestly.

(Minor):

1. Theorem 1 bounds constraints at O(N) where N is trajectory length, but exponential state space (2^n) remains problematic.

### Questions
**(General Note:)** 1.  The proposed framework appears to function more as an LLM-as-Judge, where the model evaluates and corrects its own reasoning without a learning loop or external reward signal. Could the authors clarify why this is framed as an actor-critic setup rather than a judgment-based architecture? The distinction affects how we interpret its generalizability and learning dynamics.

2. Can you report comparisons between the offline critic (used in this work) and an online critic that generates or enforces constraints during rollout? This would clarify whether the offline setup is a performance vs. efficiency trade-off.

3. How sensitive is LogicGuard’s performance to AP quality? For example, if APs are noisy, missing, or partially wrong, how often do induced rules misfire or block safe actions?

4. Could you clarify how over-constrained or conflicting laws are detected and pruned? The threshold-based deletion policy is mentioned but not analyzed quantitatively.

5. Please provide statistical metrics (mean ± CI) for Behavior and Minecraft results . This would make the empirical section much stronger.

6. Please comapre and clarify how LogicGaurd differs from work like Veriplan [1], CotTL[2]. As for example veriplan and other recent work on formal verification for LLM planning would help situate LogicGuard among existing logic-verification frameworks, although not on actor critic side. Similarly Cot-TL also employs automata-based temporal checking to verify plans, similar to your LTL critic. 

7. The critic’s induced laws are interpretable, but how scalable is this approach when the number of APs grows large (e.g., combinatorial explosion)?

 8. Constraint Accumulation: Do LTL constraints accumulate over time? Is there constraint pruning? The adaptive removal mechanism (triggered beyond fixed threshold) is mentioned but not evaluated. IN case I missed it, please point out relevent section.

9. SayCan baseline uses hand-engineered affordance function while LogicGuard versions use critic-generated laws - inconsistent comparison. Please clarify or did I missed something.

[1] VeriPlan: Integrating Formal Verification and LLMs into End-User Planning

[2] CoT-TL: Low-Resource Temporal Knowledge Representation of Planning Instructions Using Chain-of-Thought Reasoning

### Soundness
3

### Presentation
4

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
The paper proposes LogicGuard, a modular framework designed to augment any embodied LLM agent with a symbolic verification layer. The method consists of an LLM agent which outputs a feasible action and a _critic_ LLM model that analyzes past trajectories to add new laws as Linear Temporal Logic (LTL) from  constraints in natural language, which are then formalized and enforced online by a verifier. They verify empirically their framework on Behavior and Minecraft environments.

### Strengths
- The problem addressed by the paper is interesting, they want to automatically add constraints and verify them for embodied LLM agents.
- The integration of LTL-based symbolic constraints with LLM-driven planning is practically interesting.
- The experimental evaluation on two domains (Behavior and Minecraft) demonstrates empirical improvements in completion and efficiency rates.

### Weaknesses
- The contribution is primarily empirical rather than methodological. The proposed framework lacks a formal or algorithmic novelty beyond combining existing elements (LLM planners, constraint checking, and symbolic reasoning).
- The entire pipeline is heavily based on LLM, which introduces noise and risk of hallucinating at multiple stages such as: (i) _State grounding_ (ii) _Constraint generation_ which relies on natural language rules induced by an LLM and (iii) _Constraint translation_ where the LLM maps language to atomic propositions, which may not generalize beyond simulator settings.
- While this might work in practice is hard to ensure safety on the constraint generation process. I think there is no automated way to ensure non-redundancy or correctness of generated laws.

### Questions
- In a practical scenario how do you enforce safety with respect to generated constraints?
- The approach heavily relies on LLM outputs, how the problem of hallucinations affects this approach?

General comments:
While the idea of symbolic self-supervision for LLM agents is interesting, the proposed method relies to much on LLM accuracy and consequently the contribution feels closer to an empirical demonstration rather than a theoretically or methodologically grounded approach. The experimental results are encouraging but in this current form, the contribution feels limited for a top-tier venue.

### Soundness
3

### Presentation
3

### Contribution
1
