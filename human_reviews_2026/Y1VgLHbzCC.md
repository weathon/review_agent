# One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Pre-trained large language models (LLMs) show promise for robotic task planning but often struggle to guarantee correctness in long-horizon problems. Task and motion planning (TAMP) addresses this by grounding symbolic plans in low-level execution, yet it relies heavily on manually engineered planning domains. To improve long-horizon planning reliability and reduce human intervention, we present Planning Domain Derivation with LLMs (PDDLLM), a framework that automatically induces symbolic predicates and actions directly from demonstration trajectories by combining LLM reasoning with physical simulation roll-outs. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains with minimal manual domain initialization and automatically integrates them with motion planners to produce executable plans, enhancing long-horizon planning automation. Across 1,200 tasks in nine environments, PDDLLM outperforms six LLM-based planning baselines, achieving at least 20% higher success rates, reduced token costs, and successful deployment on multiple physical robot platforms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PDDLLM, a novel framework that enables LLMs to autonomously generate symbolic planning domains from a single human demonstration. PDDLLM derives both automatically by integrating LLM reasoning with physical simulation. Its Logical Constraint Adapter further connects high-level symbolic plans with low-level motion planning, allowing end-to-end execution. Evaluations across over  nine environments demonstrate that PDDLLM outperforms LLM-based baselines in success rate, planning efficiency, and token cost.

### Strengths
- PDDLLM automatically derives predicates and actions from a single demonstration without manual engineering or predefined templates.
- Strong empirical results, outperforms six LLM-based baselines across 1,200 tasks.
- Demonstrates effective transfer of learned domains and actions to new, unseen tasks and environments, on multiple robot platforms.

### Weaknesses
- The difference from InterPreT is unclear. It seems PDDLLM replaces human-crafted planning examples with a single demonstration and a predefined constraint pool for predicate generation. More clarification on how this differs conceptually or technically would strengthen the paper.
- Is GPT-4o necessary to achieve the reported performance? Have the authors tested other LLMs?
- Since the method lacks a feedback loop, it seems to require perfect constraint design, perfect predicate specification, and perfect action proposals. A single failure at any step could lead to a catastrophic outcome. It is unclear why the method is described as robust and generalizable, could the authors provide examples demonstrating successful performance with imperfect intermediate steps?

### Questions
Questions:
- Once a predicate is proposed with constraints, it appears that it is never updated. What happens if the constraints are incorrect? How do you ensure that the hyperparameters within these constraints are generalizable?
- In Figure 2’s illustration, the x, y, and z axes all share the same hyperparameter u. Why is this the case, and how was u selected?
- How do you handle low-level failures? Although the motion planning is formulated as a constrained optimization problem, the inequality constraints can accumulate errors. For example, in the stacking task, the third block may fall if the second block is not perfectly centered on the first one.

Comments: 
- L201-203, PDDL clauses have no comma in it

Missing literature: 

Zhu et al., PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models. 2025

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The submission proposes a framework to derive a symbolic planning domain from a single demonstration. It achieves this through predicate “imagination” from simulator roll-outs summarized by an LLM and action “invention” from logical state transitions. Then the produced domain can be used with a PDDL motion planner. The authors evaluate across  several tasks and environments and report large gains over LLM-enabled planners (with a fixed planning budget), includingl real-robot executions on three platforms.

### Strengths
The submission tackles the costly manual domain-spec bottleneck in TAMP and positions the work among LLM planners and domain-inference lines of work. The end-to-end automation pipeline (predicate imagination, action invention  and LoCA) is, in my understanding, the main contribution/novelty. The tasks are varied in difficulty and nature (Tower of Hanoi, bridge building, burger cooking), and multiple SOTA baselines are included (LLMTAMP, LLMTAMP-FF/FR, o1-TAMP, R1-TAMP, RuleAsMem). Analysis of time-limit is provided, which can be relevant when planning under a fixed time budget in real applications. Real robot demonstrations strengthen the empirical support of the frameworks effectiveness.

### Weaknesses
Since the pipeline still relies on some hand-chosen design choices (e.g. $u$ for subspace granularity), statements about constructing domains "without manual predesign" might be exaggerated.

How often does the limited operator set miss required invariants? First-order predicates come from discretized subspaces while higher-order ones use a limited set of logical operators/quantifiers. The limitations section admits missing complex predicates (e.g., ordering constraints), which can materially affect plans in more challenging domains.

### Questions
Can you add an ablation for u and for the number of parallel prompts? Including domain quality metrics (missing/redundant predicates) 

Can you include more quantitative per platform metrics for real robot experiments?

Can you disambiguate token costs across domain derivation and per-task planning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PDDLLM, a framework that automatically generates the PDDL planning domain from one demonstration by combining LLMs with physical simulation. It introduces a Logical Constraint Adapter (LoCA) to automatically ground the generated symbolic actions into motion constraints, enabling seamless integration with motion planners and real-robot execution. Experiments on 9 tasks show that PDDLLM outperforms baselines and achieves performance comparable to expert-designed domains.

### Strengths
1. This paper aims to automatically generate planning domains from one demonstration to reduce manual engineering efforts, which is a valuable goal for the field. 
2. Experiments on 9 tasks show that the proposed method achieves a high success rate, outperforming other baselines.

### Weaknesses
1. The paper relies on a physics simulator to evaluate the physical feasibility of predicates. However, such simulation-based evaluation may fail to capture complex dynamics, limiting the method’s generalization to real-world settings. The current experiments only involve simple rigid-body interactions, so it remains unclear how the proposed approach would perform with more complex objects such as deformable materials or fluids.
2. There are several unclear aspects in the paper:
(1) It is not clearly stated whether the predicates must be predefined or can be freely generated by the LLM based on the task and scene. If predicates are required in advance, this could constrain the generality of the proposed framework.
(2) The paper mentions that “the range of each feature is divided into intervals, with the length of each interval being a hyperparameter,” but does not specify how this hyperparameter is determined. Is it fixed manually or generated adaptively by the LLM?

### Questions
The paper claims that the framework can integrate knowledge across demonstrations. In this context, during the evaluation, were demonstrations from other tasks used to assist the construction of the domain model for the current task?

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
This paper presents a new method that is capable to generate a complete planning domain from scratch, without relying on any predefined predicates or actions. The method employs the reasoning ability of LLMs to do generation based on demonstration.

### Strengths
This work targets an interesting and meaningful problem in planning.

The proposed methodology does not rely on pre-defined predicate space and action model, which reduce the effort of human annotation.

Experimental in real robot environments demonstrates the effectiveness of the proposed method.

### Weaknesses
**Major**

Successful deployment of the proposed method requires a perception function that can accurately extract the continuous states from objects. It is unclear what types of the perception function the proposed method can work along well.

It was not extensively discussed in the paper how robust the proposed method is with respect to any noises in the perception process.

It seems that applying the proposed method in real applications requires setting up a same digital copy in a simulation. This limits the potential applicable areas as setting up simulation in some domains requires laborious manual modelling even for feature parameters. It would be better to discuss what kind of domain the proposed method can easily handle, and what domains the proposed method may encounter big challenges.

**Minor**

Some paragraph is not written with clear motivations, which makes the reading not easy to follow. For example, L180-L196, it is unclear why the method is dividing the feature space. It would be better to clearly explain the target problem at the moment, e.g., use parallel simulations to imagine predicates by summarizing the simulation roll-outs with LLMs, and explain what the challenges are.

### Questions
Please see the weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
2
