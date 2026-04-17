# Information Seeking for Robust Decision Making under Partial Observability

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Explicit information seeking is essential to human problem-solving in practical environments characterized by incomplete information and noisy dynamics. When the true environmental state is not directly observable, humans seek information to update their internal dynamics and inform future decision-making. Although existing Large Language Model (LLM) planning agents have addressed observational uncertainty, they often overlook discrepancies between their internal dynamics and the actual environment. We introduce Information Seeking Decision Planner (InfoSeeker), an LLM decision-making framework that integrates task-oriented planning with information seeking to align internal dynamics and make optimal decisions under uncertainty in both agent observations and environmental dynamics. InfoSeeker prompts an LLM to actively gather information by planning actions to validate its understanding, detect environmental changes, or test hypotheses before generating or revising task-oriented plans. To evaluate InfoSeeker, we introduce a novel benchmark suite featuring partially observable environments with incomplete observations and uncertain dynamics. Experiments demonstrate that InfoSeeker achieves a 74% absolute performance gain over prior methods without sacrificing sample efficiency. Moreover, InfoSeeker generalizes across LLMs and outperforms baselines on established benchmarks such as robotic manipulation and web navigation. These findings underscore the importance of tightly integrating planning and information seeking for robust behavior in partially observable environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Existing large language model (LLM)-based planners for agents in partially observable environments have limitations. These agents often ignore the mismatch between their internal dynamics and the actual environment, leading to decision biases. To address this issue, the paper proposes an Information Seeking Decision Planner (InfoSeeker), an LLM-based framework that combines task-oriented planning with information seeking. InfoSeeker prompts the LLM to actively gather information through planned actions before generating or modifying task plans, ensuring alignment between internal dynamics and the real-world situation. The authors also introduce a new benchmark dataset to test agent performance under conditions of observation uncertainty and environmental noise. Experimental results show that InfoSeeker outperforms baseline methods on this new benchmark, validating the effectiveness of the proposed approach.

### Strengths
1、Combination of Task Planning and Information Seeking:
Although the innovation is not strong, the framework presents the idea of incorporating information seeking into task planning. Theoretically, it offers an approach for solving some partially observable problems.

2、Cross-Model Applicability:
InfoSeeker can be tested across different LLM models, showing a certain level of generality.

### Weaknesses
1、Simplicity of the Framework:
The framework is quite simple both theoretically and in implementation. It lacks deeper innovation or the ability to solve more complex problems.

2、Lack of Depth in Experimental Design:
The experiments are limited to simple benchmark tasks, and all tasks and datasets were designed by the authors themselves, which makes the validation of its general applicability insufficient.

3、Lack of Real-World Validation:
The experimental results are primarily based on simulated environments (and in pure text), which cannot verify the actual effectiveness of the framework in real-world, complex environments.

4、Issues with the Information Extraction Module:
The deficiencies in the information extraction module led to some failures in the experiments, and the paper does not propose effective solutions to address this issue.

5、Low Final Pass Rate on the TravelPlanner Benchmark:
On the TravelPlanner benchmark, all methods show very low final pass rates. Although InfoSeeker performs the best, the absolute improvement is too small to convincingly demonstrate its significant advantage in real, complex tasks.

### Questions
1、How can the framework in the paper be applied in more complex and dynamic real-world environments? Will it perform well in highly practical tasks?

2、Does the design of the custom dataset affect the generalizability of the results? Can this method demonstrate its advantages on other public datasets? Please add further experiments.

3、Regarding the failures in the information extraction module, are there clear optimization solutions, or have other technological approaches been considered to improve its performance?

### Soundness
2

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
4

### Summary
To address the problem of LLM-based planners in interactive tasks with uncertain outcomes, the authors propose the Information Seeking Decision Planner (InfoSeeker), an LLM decision-making framework that integrates task-oriented planning with information seeking. It proactively obtains environmental information alongside with execution history, and then revises its execution plan based on updated beliefs (world knowledge). Through experiments on multiple self-made benchmarks and two recent public benchmarks, the authors demonstrate that the proposed method is promising.

### Strengths
1. Dealing with uncertainty in LLM-based planning is an important and less-explored topic.
2. The proposed method is effective. 
3. The paper is well-written.

### Weaknesses
1. Limited novelty, it seems that such a method is a combination of ReAct + Reflexion.
2. The baselines are incomplete. For example, ReAct+Reflexion, Tree of Thought, Self-consistency, or MCTS-based planning method can be used as the baselines.

### Questions
1. What are the token consumptions?
2. What are the differences between InfoSeeker and LLM3?
3. Some related works are missing. Please perform a more comprehensive survey regarding LLM and robust/uncertainty decision making/reasoning.

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
The paper proposes InfoSeeker, an LLM-based decision-making approach in partially observable settings that integrates information seeking actions to align the LLM-agent’s belief of the dynamics with that of the environment for effective task-oriented planning. Information seeking is done by running diagnostic trials and observing the outcomes to detect shifts in environment dynamics. To evaluate their approach, the paper also presents text-based benchmarks that include uncertain dynamics in tasks such as robot arm control and navigation, color mixing, and block stacking based on existing benchmarks.

### Strengths
- The paper is generally well-written and clear.
- The contributions of the paper in the context of existing works are communicated clearly. Specifically, the paper adds information seeking as an explicit step before generating/refining the plans as opposed to prior works that rely solely on reactive feedback.
- The paper additionally focuses on dealing with uncertainty in environment dynamics, in addition to observation uncertainty considered in prior works.
- The formalization of the problem in terms of the POMDP framework is meaningful.
- Experiments and related ablation studies are promising. Failure analysis is helpful to readers.

### Weaknesses
- The paper tackles only a very specific kind of uncertainty in dynamics: those that are deterministic but unknown/unmodeled beforehand (e.g. deterministic shift in gripper pose due to misaligned robot arm). Many problems of interest in the broader planning under uncertainty domain are more focused on uncertainty due to stochasticity (e.g. stochastic action outcomes due to imperfect/noisy actuators).
- There are few places in the paper where the authors claim their approach to make “optimal” decisions under uncertainty. Perhaps, “effective” might be a better word for it, since “optimal” is a little strong given that in some experiments, the paper’s approach underperforms existing baselines. Claiming optimality might also call for establishing formal guarantees that no other approach outperforms the paper’s approach. This might come off as misleading especially since the presented approach is theoretically formalized in the context of the standard POMDP framework, which often has clear definitions of optimality.
- The proposed benchmark is limited and feels hand-crafted as acknowledged by the authors.

### Questions
- In line with the first weakness mentioned above, what are the limitations and potentials of the proposed approach for dealing with the uncertainty due to stochasticity in addition to those due to unknown/unmodeled dynamics?
- Given that the proposed benchmarks are limited in scope (acknowledged in the limitations), what are the challenges and limitations of the proposed approach for applications in more realistic planning scenarios in the real world (e.g. with an embodied AI agent)?

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
3

### Summary
The paper introduces an LLM-based decision-making framework that interleaves task‑oriented planning with explicit information‑seeking actions. The goal is to align the agent's internal dynamics with the actual environment, particularly under partial observability and noisy or shifted dynamics. The method prompts the LLM to diagnose inconsistencies, design small diagnostic trials, extract insights from those trials, and then revise task plans. The authors also propose a text-based benchmark that injects uncertainty not only in observations but also in environmental dynamics. Empirically, it reportedly achieves large performance gains on the new benchmark and shows improvements on LLM3-style robotic manipulation and TravelPlanner web navigation.

### Strengths
1. The work moves beyond observation uncertainty to uncertain/shifted dynamics, a critical but under‑evaluated challenge for LLM agents.

2. The work shows a practical, training‑free loop and the cycle is simple to adopt and demonstrably helpful across tasks.

### Weaknesses
1. The claimed formal connection to POMDPs remains interpretive. There is no explicit specification of the belief representation, the backup operator the prompts approximate, or a principled value‑of‑information objective that the information‑seeking step maximizes.

2. The writing is not very clear. For example, the text refers to internal dynamics, it is unclear what gets updated (offsets? action‑effect map? parameters), where it is stored (text memory? structured map?), and how it is consumed during planning. 

3. There is no clear trigger for when to seek information, no seek budget, and no rule for choosing diagnostic actions. It is very underspecified. 

4. 4.1 Talks about 5 tasks, and yet only 4 are explained (I suppose the last one is split into 2, but it does not make it a new task). And in 5.1, what are 11 tasks from the benchmark? The writing is very hard to follow. 

5. There is an inconsistent mixing of steps, attempts, and wall‑clock time across sections without normalization, weakening sample‑efficiency claims.

6. Adding to (5): Efficiency is multi-dimensional (env steps, LLM calls, tokens, time). Claims are based partly on a wall‑clock comparison. But, token/LLM‑call budgets and env‑step budgets are not jointly reported.

7. Similarly, isn't a "one‑minute" cap API‑latency‑ and hardware‑dependent? It is not a stable cross‑paper comparator.

8. It is unclear whether baselines had identical total interaction budgets (seek+plan vs plan‑only), identical temperature/top‑p, timeouts, and truncation policies.

9. Only two LLMs are tested and per‑model variability and versioning are not reported. I would strongly suggest the authors to ablate a few more LLMs.

### Questions
Please see the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
