# CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Vision-and-Language Navigation (VLN) primarily focuses on a single-agent-centric approach that executes human instructions step-by-step. In real environments with high demand or parallel workflows, collaboration VLN offers distinct benefits including shorter makespan and greater robustness through parallelism and role specialization. Collaboration VLN also brings new challenges including congestion, handoff errors, and rendezvous timing, which single-agent formulations overlook. Current datasets and protocols remain single-agent centered, which hides opportunities for assistance and ignores inter-robot interference. We fill this gap with Collaborative Long-Horizon VLN benchmark (\textbf{CoNavBench}), consisting of 4048 single and collaborative episodes with graph-level annotations and a collaboration type taxonomy that controls handoff styles and rendezvous patterns. To generate and evaluate at scale, we build \textbf{NavCraft}, an automated graph-grounded data generation platform. A two-stage hierarchical agent first produces a long-horizon base mission for the primary robot and then instantiates helper robots, allocates subgoals, and specifies validated handoffs and rendezvous. The agents operate with a scene graph in the loop derived from Habitat-Sim, which enables reachability checks, travel time, and interference assessment, and iterative schedule repair via an efficiency tool library. As a reference, we provide a collaborative baseline based on a finetuned Qwen2.5-VL-3B. Trained with CoNavBench, collaborative policies reduce makespan and improve reliability over strong single robot counterparts, yielding \textbf{18.11\%} step level success. Anonymous Website: https://navcraft.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CoNavBench, a large-scale benchmark designed to evaluate collaborative vision-and-language navigation (VLN) under multi-agent, long-horizon settings. The authors further propose NavCraft, a graph-grounded data generation and validation platform that constructs semantically annotated scene graphs, generates single-agent base tasks (NavCraft-S), and lifts them into collaborative multi-robot schedules (NavCraft-C). The experiments use Qwen2.5-VL-3B/7B models fine-tuned on the dataset, demonstrating improved performance over single-robot baselines and validating the benefit of collaboration in task success and path efficiency. Overall, this work establishes an interesting benchmark for multi-agent embodied navigation by unifying data generation, simulation, and evaluation.

### Strengths
The paper convincingly identifies a major gap in the VLN landscape: the lack of standardized evaluation for multi-agent, cooperative scenarios, and fills it with a well-structured benchmark and taxonomy.
NavCraft’s design (scene-graph generation, task validation, and efficiency tools) is technically detailed, modular, and reproducible within Habitat-Sim, enabling scalable and context-aware task generation.
The framework successfully merges semantic graph annotation, spatial reasoning, and LLM-driven language prompts to produce feasible, diverse, and verifiable tasks.
Finetuned Qwen2.5-VL models show consistent improvements in SR, SPL, and CSR across both single- and multi-agent settings, confirming the utility of CoNavBench for training collaborative policies.

### Weaknesses
*Limited collaboration scope*

Only two fixed handoff types (A1, A2) are explored, which may not fully capture the diversity of real collaborative behaviors (e.g., concurrent exploration, dynamic task reassignment).

*Data generation*

The dependence on GPT-4o-mini and closed APIs limits long-term reproducibility; no quantitative analysis is provided on the variability or bias of generated instructions.

*Lack of detailed time-efficiency evaluation*

Although collaboration is claimed to reduce makespan, this paper needs more clarification on collaboration acceleration and time-based metrics.

*Incomplete ablations for important modules*

Components such as the memory-aware mechanism, efficiency tool library, and profile-conditioned sampling are presented but not individually quantified in their contributions.

### Questions
Besides the weakness mentioned above, the reviewer has some extra questions below:

How does the benchmark handle multi-agent interference and collision during simulation? Are these events explicitly annotated or only indirectly measured by task failure?

Could the authors provide a quantitative comparison between NavCraft-generated and manually designed tasks to justify realism and linguistic fidelity?

Why does the Qwen2.5-VL-3B outperform 7B in several collaborative tasks? Does this stem from optimization saturation or data scarcity?

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
3

### Summary
1. CoNavBench Benchmark: It includes 4048 single-robot and multi-robot collaborative tasks, designed to test how robot collaboration can optimize task completion time and improve efficiency in long-horizon tasks.

2. NavCraft Platform: An automated data generation platform for collaborative tasks that uses the scene graph from Habitat-Sim to create semantically rich environmental layouts and generate collaborative tasks. It includes two stages of task generation: NavCraft-S (for generating single-robot tasks) and NavCraft-C (for generating collaborative tasks).

3. Collaborative Models and Performance Evaluation: Experiments using the Qwen2.5-VL-3B model show that the collaborative model improves step-level success by 18.11% and reduces task completion time compared to the single-robot model.

### Strengths
By introducing multi-robot collaboration, the CoNavBench benchmark optimizes the completion time of long-horizon tasks. Compared to single-robot tasks, collaborative robots can perform multiple subtasks simultaneously, effectively reducing task delays and idle time, significantly improving overall task efficiency. The CoNavBench benchmark includes 4048 single-robot and multi-robot collaborative tasks, providing a comprehensive evaluation of collaborative navigation systems in complex tasks. It tests how robots coordinate and distribute tasks in scenarios where multiple robots work together.

### Weaknesses
1. The inference speed of large models is very slow; will there be a significant delay in task completion?
2. Will the two agents collide in the environment?
3. There are many VLN methods based on large models [1]; why were they not compared with these methods?
4. Are the initial positions of the two agents the same, or is one agent directly summoned?




[1] Cheng, An-Chieh, et al. "Navila: Legged robot vision-language-action model for navigation." arXiv preprint arXiv:2412.04453 (2024).

### Questions
same as weakness.

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
2

### Summary
The paper introduces CoNavBench, a collaborative long-horizon Vision-and-Language Navigation (VLN) benchmark designed to study multi-robot cooperation under language-guided tasks. It extends prior single-agent datasets by supporting multi-agent task decomposition, handoff scheduling, and rendezvous-based cooperation. To generate the data, the authors build NavCraft, a graph-grounded generation platform with semantic scene graphs, hierarchical planning agents (NavCraft-S / NavCraft-C), and an on-graph efficiency tool library for validation and optimization. Experiments demonstrate reduced makespan and improved step-level success over single-agent baselines using Qwen2.5-VL policies.

### Strengths
1. CoNavBench provides a large-scale benchmark (4048 episodes) covering both single and multi-robot navigation with rich annotations for collaboration types and performance metrics.
2. The proposed NavCraft system establishes a structured data-generation pipeline grounded in semantic graphs, incorporating task synthesis, collaboration lifting, and efficiency validation, which enhances data consistency and scalability.
3. The framework introduces a useful utility for checking reachability, congestion, and timing constraints directly on scene graphs, which is a practical contribution for large-scale synthetic task design.
4. The manuscript provides dataset statistics, training configurations, and baselines clearly, allowing others to replicate and evaluate under consistent protocols.

### Weaknesses
1. Visual clarity is not acceptable at all, for example, Figure 9 is difficult to interpret and provides almost no new information beyond stating that the training loss decreases. It lacks analytical insight and does not meaningfully support the main claims. The visual examples fail to illustrate collaborative interaction or path optimization clearly. Comparisons are blurry or incomplete. An example is the visualization of collaborative cases, the authors didn't provide any useful information in these cases to support the claim.
2. While the data-generation pipeline heavily relies on prompt engineering for NavCraft, there is no systematic or theoretical analysis of its design choices, prompt sensitivity, or ablation, leaving uncertainty about robustness.
3. The entire framework and experiments are conducted in simulation (e.g., Habitat-3). Although acceptable at this stage, no real-world experiments or transfer evaluations are included to verify deployability or generalization.

### Questions
1. Could authors provide more clear visualization results to clarify the claims, for example, how Figure 9 contributes analytically, whether it reflects convergence stability, mode collapse, or loss behavior differences between different models?
2. Are there plans to improve qualitative visualizations to more clearly demonstrate cooperative efficiency or multi-agent coordination?
3. Have you analyzed how different prompt templates or role-conditioning strategies affect the generated tasks’ diversity or validity?
4. Is there any plan to evaluate NavCraft or CoNavBench tasks on real robots or real-world scans to assess practical applicability?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce CoNavBench, a new benchmark for "Collaborative Long-Horizon Vision-and-Language Navigation" where multiple robot agents collaborate to follow instructions inside 3D simulated homes (using Habitat-Sim). They introduce Navcraft, an algorithm to generate these multi-robot cooperation scenes where agents share and hand off subtasks. It contains 4048 episodes annotated with scene graphs, but more can be generated by Navcraft. Navcraft uses a two-stage hierarchy, Navcraft-S and NavCraft-C, to first generate a single-robot task, then break it down into a two-robot collaborative task, with graph-based scene understanding and validation. The authors fine-tune Qwen2.5-VL models on this benchmark, reporting improvements over single-agent baselines.

### Strengths
This is a novel benchmark for robotic cooperation in 3D home environments. The proposed mehtod can generate an arbitrary number of scenarios. The paper presents a clear and interesting pipeline (NavCraft-S and NavCraft-C) for task creation, graph annotation and validation. It is overall clear with illustrative figures. The authors use standard metrics for measuring performance on the benchmark, and provide a thorough baseline and ablation analysis with various LLM APIs.

### Weaknesses
- The benchmark, while innovative, is still limited to two agents and specific relay-style tasks (one robot carries something from A to B, hands it off to a second robot to carry it from B to C). Dynamic collaborations are not explored. 
- NavCraft relies on proprietary LLM APis for data generation, which can lead to issues for reproducibility of the benchmarking. This limitation is addressed by the authors. 
- Some parts are dense in formulas and could benefit from a high-level intuitive explanation beforehand.

### Questions
- Can the author expand on how do the results vary based on the chosen collaboration type (A1 vs A2) ? 
- How does this approach (generate single-agent task then split) compare against directly generating a two-agent task? 
- Authors report using two robots for the task instead of a single one improves performance. This can be explained by each robot being able to learn its task more efficiently. Do the authors believe this difference would disappear with SOTA large language models?

### Soundness
3

### Presentation
3

### Contribution
3
