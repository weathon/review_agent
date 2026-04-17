# Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Spatial embodied intelligence under partial observability requires agents to actively acquire missing information rather than passively consume complete observations.
While multimodal foundation models excel at passive perception and reasoning, their ability to support active, self-directed exploration to build and maintain a coherent spatial belief remains unstudied.
We therefore propose Theory of Space, defined as an agent's ability to construct, revise, and exploit a spatial belief through self-directed active exploration under partial observability.
We implement Theory of Space using a benchmark with textual and visual environments. Rather than solving specific tasks, the goal is curiosity-driven exploration to build a complete, accurate spatial belief. 
A core innovation is spatial belief probing: we prompt it to reveal its internal spatial belief as a cognitive map at each step, letting us measure the quality of its underlying spatial belief.
Our evaluation of state-of-the-art models on a suite of downstream tasks reveals critical bottlenecks: (1) \textbf{The Active-Passive Gap}: Performance degrades when agents must autonomously gather information (e.g., \textsc{GPT-5.2}: $57.1{\to}46.0$); (2) \textbf{Inefficiency}: Models explore in an unsystematic way and with high redundancy, failing to match the efficiency of program-based proxies while producing no better results.
Through belief probing, we diagnose that perception acts as an initial bottleneck, yet global beliefs suffer further from \textbf{instability} that causes spatial knowledge to degrade over time. 
Finally, a false belief paradigm reveals \textbf{Belief Inertia}: agents fail to overwrite obsolete priors, an effect especially severe in vision-based models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper argues that today’s VLM agents excel at passive or task-driven spatial reasoning but largely fail to build a coherent understanding of space through exploration. It proposes “Theory of Space” (ToS): an agent’s ability to construct, update, and use an internal spatial belief from partial, egocentric observations, and introduces a benchmark that makes agents actively explore text and vision environments (procedural multi-room layouts in a grid world and a ThreeDWorld+Objaverse simulator).

 At every step the agent must externalize its belief as a cognitive map (JSON), which affords assessing not just task accuracy but belief correctness, consistency, and stability. The suite spans route and survey tasks (e.g., pairwise relations, action-to-view, allocentric mapping, mental rotation, localization) and adds belief-update probes via perspective taking and false-belief variants. To disentangle exploration from reasoning, they compare active on-policy agents with passive runs over standardized proxy trajectories (SCOUT for vision, STRATEGIST with AC-3 for text) and score exploration efficiency via information gain or coverage. Results across several state-of-the-art models show strong text-world performance but large drops in vision, common failures in ego-motion updates and global consistency, and a tendency to terminate exploration too early, highlighting the need for agents that manage spatial belief as a first-class capability.

### Strengths
S1.  While the setup is admittedly a bit toyish, the core capability it targets—actively constructing, updating, and using a spatial belief from partial egocentric views—is exactly what real robots need to operate safely and robustly in cluttered, changing environments; from home assistants that must map and remap rooms, to warehouse and hospital robots that navigate multi-room spaces with moving obstacles, the paper’s emphasis on belief management, uncertainty reduction, and allocentric–egocentric conversion translates directly into high-value, real-world behaviors.

S2. The paper cleanly isolates “active belief construction under uncertainty” as the missing capability between passive reasoning and task-driven foraging, giving the field a crisp target. Also, they require stepwise cognitive-map JSON forces models to externalize internal spatial state, letting evaluators inspect correctness, consistency, and stability—not just final-task accuracy.

S3. Dual-modality design: mirrored text and vision worlds over identical layouts quantify the real modality gap and let researchers attribute errors to perception vs. spatial inference.

### Weaknesses
W1. Discrete bins (5 egocentric angle bins, 6 distance bins, 8 allocentric directions) compress geometry in ways that can hide small-but-critical errors (or inflate accuracy by near-miss snapping); no sensitivity analysis shows how bin granularity changes rankings or conclusions.

W2. Layouts are grid-based with tree-structured room graphs (no loops), weakening tests of global consistency (e.g., cycle closure) and long-range metric drift—the very failure modes cited (egomotion, global map maintenance) would be more exposed in loopy topologies and irregular geometries.

W3. The authors grant an action that returns ground truth, acknowledge cost-minimization under a step budget, and evaluate information gain via AC-3 in text mode—yet they don’t ablate Query usage, price, or availability. That omission makes it hard to know how much of the reported “spatial belief construction” truly reflects inference from partial observations versus selective peeks at the answer.

### Questions
The statistical robustness is unclear: the paper mentions 100 seeds but not per-task item counts, variance across seeds, or multiple-comparison controls. Given many tasks and modalities, some reported gaps could reflect sampling and prompt stochasticity rather than stable capability differences.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new Theory of Space framework that tests the agent's ability to learn the unobserved structure of its spatial environment from partial observations.
They implement a benchmark including text-based and visual environments for curiosity-driven exploration, and prompt (SOTA foundation) models to present their cognitive map at each step to track the underlying spatial model.
They document common failure modes in how models' form spatial beliefs, including path integration errroe and incorrect map inference.
The primary focus of the benchmark is exploration: the agent must make its own decisions on what to observe next.
They find that the models perform much better in pure text-based worlds compared to visual ones.

### Strengths
Theoretical soundness: The paper is well written and well grounded in cognitive theory.

Technical soundness: A stack of tools and technologies is integrated effectively to design the ToS benchmark. A large number of models is evaluated

### Weaknesses
This work aims to produce a benchmark for human-like spatial understanding and exploration. While this motivation, and the ToS approach as a means of achieving it are very well theoretically justified, the paper offers no evidence of how a human would behave in these tasks.

The models are compared to two proxy agents, which they claim to execute a theoretically optimal path, to establish an upper bound on exploration ability.  However, no proof of proxy agent optimal is given.

Both weaknesses arise from overstated claims. While would take a lot of work to support these claims, the paper would be still valuable if the claims were more accurately reframed.

### Questions
How would foundation models compare to human performance? 

Is it possible to design a human study where people navigate the same worlds? 

It is worth considering that while human map drawings are notoriously poor, their navigation abilities are often excellent. Have the authors considered to what extent this pattern of knowledge accessibility may be parallelled in machines?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the Theory of Space (ToS), a framework for evaluating the ability of foundation models to actively construct, update, and utilize an internal spatial belief of an environment from partial observations. The paper designed experiments to (1) ask an LLM/VLM agent to actively explore a partially observable environment for establishing the internal spatial model, (2) use a proxy agent script to derive a ``passive" spatial model to disentangle active exploration from passive reasoning, and (3) ask the agent to represent its spatial belief for direct accessment. Evaluations across modern LLM/VLMs show a modality gap (stronger on text than vision) and diagnose failure modes like egomotion update errors and poor global map maintenance; humans outperform most models with modest exploration budgets.

### Strengths
1. The paper designs spatial belief evaluation around task-agnostic, uncertainty-reducing exploration, rather than passive reasoning and goal-directed task completion, adding an active aspect to spatial belief construction. This makes the paper original. 
2. Spatial understanding under partial observability is central for embodied agents and planning. The ToS framework fills in the gap for LLM evaluation in enactive cognition by evaluating the goal-agnostic exploration of active LLM agents. 
3. Each experiment in the benchmark is well-designed for evaluating the representation, updating, and utilization of the internal space model. The metrics used for measuring the performance are reasonable, including the exploration efficiency against the AC-3 algorithm, belief quality assessment by correctness and consistency, and task success rate.

### Weaknesses
1. Relatively simple spatial environment and limited statistical results: most experiments use two connected $6$ by $6$ rooms with $9$ objects, and some with varying room size. While the small size of the spatial environment avoids memory capacity as a confounding factor for ToS performance, its simplicity could potentially trivialize the active exploration aspect of the agent. The generalization beyond the grid room is therefore unclear. As a consequence, there are also no statistical results on the performance. 
2. Exploration efficiency analysis: the exploration efficiency is mainly analyzed by the information-gain curve. It's unclear whether limited exploration thoroughness originates from policy or belief integration failures.

### Questions
1. How do you ensure the cognitive map probing reflects the true internal state rather than post-hoc rationalization?
2. Is the termination action fully decided by the LLM agent? What could be the cause that leads to agents stopping prematurely?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Theory of Space (ToS), an evaluation paradigm for whether LLM/VLM agents can actively construct, update, and use an internal spatial belief (a cognitive map) through exploration. It builds a procedural multi-room world (text and vision), defines route/survey/update tasks, decouples exploration from reasoning using proxy agents, and probes the belief itself (serialized maps) rather than only task success. Empirically, top models perform well in text but struggle in vision. Active and passive performance are close, suggesting belief construction and maintenance (not coverage alone) is the bottleneck.

### Strengths
- Concepts and definition contribution: ToS captures the ability to (1) construct a globally consistent belief from partial views, (2) update it as new evidence conflicts, (3) utilize it for downstream spatial tasks. The two-phase evaluation separates Exploration to gather observations and build a belief from Reasoning on route/survey/update tasks to test utilization. The task formulation itself is interesting, although this paradigm has been explored in early embodied AI benchmarks like EXCALIBUR (I knew EXCALIBUR evaluates where the agent goes; ToS evaluates what the agent infers and remembers, there is a difference).
- Belief probing & metrics: Models output a cognitive map (JSON) that can be directly examined. The paper provides metrics for positional, directional, and facing accuracy, plus a composite score. It also introduces a formal Exploration Efficiency metric based on information gain with AC-3 pruning.
- Interesting findings: Active performance roughly equals passive performance (but with premature stops). Active scores are close to passive, with gaps often stemming from premature termination and inefficient exploration, not pure lack of path coverage (see information-gain curves). Belief management matters: stepwise probing shows errors compound from perception to integration (egomotion) to stability. Text models rank higher on correctness/consistency, while in vision, perception is the bottleneck.

### Weaknesses
- The current environment relies on symbolic discretization: angles and distances are bucketed into categorical bins (e.g., {near, mid, far}) and rendered with calibration cues (reference grids, constant lighting). This yields clean supervision but strips away realistic sensory ambiguity such as partial occlusions, depth uncertainty, and texture variation. Models may appear spatially consistent only because the environment removes real-world ambiguities. This inflates performance and may not transfer to continuous or noisy perception (e.g., real egocentric videos or robot camera feeds).
    - Possible action items: Add a continuous-valued variant where direction and distance are real-valued regressions rather than categorical bins; introduce sensor noise and texture perturbations (random lighting, viewpoint jitter, partial occlusion).

- The results show a significant gap between text and vision performance, but the source of this gap (perceptual encoding vs. multimodal integration vs. belief updating) remains ambiguous. Without identifying the bottleneck, it's unclear whether improvements should target visual representation learning, grounding alignment, or belief maintenance logic.
    - Possible action items: Implement modular ablations: Perfect-perception oracle (feed ground-truth object positions); Perfect-integration oracle (feed clean percepts, test memory/belief fusion); Perfect-stability oracle (test only inference drift).

- Despite the paper's emphasis on ToS, the core scientific question (can an embodied agent build a consistent internal map from partial observations) closely parallels prior work such as EXCALIBUR (CVPR 2023) and Emergence of Maps in the Memories of Blind Navigation Agents (NeurIPS 2023). Both earlier studies explored spatial map emergence and information-gain-based exploration. The conceptual novelty is less about introducing a new phenomenon and more about repurposing it into an evaluation benchmark. Without explicit differentiation, readers may view ToS as incremental rather than foundational.

### Questions
- Space and ToM has been extensively studied in past psychology research. See https://arxiv.org/abs/2310.19619 for example. Your benchmark includes "false-belief / perspective-taking" tasks, but these seem to involve the agent's own belief about the environment rather than what another agent believes, i.e., ToS did address spatial belief of a single agent about objects and its own perspective, but did not appear to include multiple agents with independent goals or belief states. How do you justify this choice?
- Can ToS-style belief actually improve downstream navigation/manipulation? JSON is a strong choice, and vision seems to play a minimal role here.
- How sensitive are results to Query and termination policy? Could Query or early-stop heuristics be creating artificial gaps between active and passive? Maybe provide cost-sweeps over Query availability/penalties and termination criteria; plot accuracy vs. action cost curve?

 - - -
I am happy to change my ratings if the above Weaknesses and Questions are addressed.

### Soundness
2

### Presentation
3

### Contribution
2
