# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

- Decision: Reject
- Scores: 8, 2, 6, 4, 8

## Abstract
Synthetic data generation via simulation represents a promising approach for enhancing robotic manipulation. However, current synthetic datasets remain insufficient for robust bimanual control due to limited scalability in novel task generation and oversimplified simulations that inadequately capture real-world complexity. We present RoboTwin 2.0, a scalable framework for automated diverse synthetic data generation and unified evaluation for bimanual manipulation. We construct RoboTwin-OD, an object library of 731 instances across 147 categories with semantic and manipulation labels. Building on this, we design a expert data generation pipeline by utilizing multimodal large language models to systhesize task-execution code with simulation-in-the-loop refinement. To improve sim-to-real transfer, RoboTwin 2.0 applies structured domain randomization over five factors (clutter, lighting, background, tabletop height, language instructions). Using this approach, we instantiate 50 bimanual tasks across five robot embodiments. Experimental results demonstrate a 10.9% improvement in code-generation success rates. For downstream learning, vision-language-action models trained with our synthetic data achieve 367% performance improvements in the few-shot setting and 228% improvements in the zero-shot setting, relative to a 10-demo real-only baseline. We further evaluate multiple policies across 50 tasks with two difficulty settings, establishing a comprehensive benchmark to study policy performance. We release the generator, datasets, and code to support scalable research in robust bimanual manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes RoboTwin 2.0, a scalable simulation framework for bimanual manipulation that integrates Multimodal LLM-based code generation with simulation-in-the-loop feedback for automated expert data creation, systematic domain randomization (clutter, lighting, background, height, and language), embodiment-aware grasp adaptation, and an open-source benchmark spanning 50 dual-arm tasks across 5 robot embodiments and 100K trajectories. The system shows substantial gains in automated code success, improves few-/zero-shot sim-to-real (and real-to-sim-to-real) transfer, and releases a unified dataset and evaluation benchmark.

### Strengths
- Strong engineering contribution: impressive scale and integration of object assets, code-generation feedback, and domain randomization.
- Clear sim-to-real and real-to-sim-to-real improvements with controlled ablations where domain-randomized data notably enhances robustness.
- Extensive benchmark coverage and solid comparisons to prior datasets.
- Methodological novelty: closed-loop LLM+VLM code generation for robotic simulation, reducing human effort.
- The paper is well-written.

### Weaknesses
- Limited conceptual depth: while the system design is technically strong, the paper reads primarily as a dataset + pipeline release; the underlying algorithmic innovations (e.g., feedback controller, grasp adaptation) remain engineering-focused rather than introducing new learning principles.
- Empirical analysis could be deeper: results rely on average success rates; missing are fine-grained analyses of failure cases, generalization to new unseen tasks beyond the 50 predefined ones, or comparisons to other synthetic-data pipelines.

### Questions
- How does the MLLM pipeline generalize to novel APIs or unseen task primitives beyond those encoded in the current skill library?
- What are the compute and time costs and how scalable is the process across clusters?
- Can domain randomization parameters be tuned automatically based on sim-to-real validation feedback?
- Are there plans to include tactile or force feedback channels?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This submission introduces RoboTwin 2.0, a framework for generating bimanual robot manipulation data in simulation. It consists of MLLM-guided code generation, domain randomization, object grasping pose annotation, and an object dataset. Experiments show that it achieves a better generation quality, and the proposed system can be used for robotic model training. However, several concerns exist regarding the proposed system and demonstrated results.

### Strengths
- The submission is well structured with nice figures for illustration.
- The proposed system is comprehensive with code generation, dataset, and evaluation.
- The submission successfully demonstrates an end-to-end pipeline of robotic model training in simulation and real-world deployment.

### Weaknesses
- It is unclear how does the proposed system compare to existing robotic simulation benchmarks at a system level.
- All components proposed in the system is not new. It's unclear what's the uniqueness behind the proposed method.
- For the MLLM-aided code generation, there have been many works in that front, such as Ha et al., 2023; Wang et al., 2024; Genesis, 2024; Nasiriany et al., 2024; etc. It's unclear how does the proposed code generation pipeline differ from others.
- The code generation pipeline still relies on pre-defined APIs, which means the complexity of robotics tasks the pipeline can generate as well as the scalability of such pipeline are limited. Also, what are those API functions? How much human effort has been spent in them?
- The importance of domain randomization has become a common sense. Moreover, the types of domain randomization proposed in this submission only cover visual aspects. However, it's more critical to have physical domain randomization as suggested by Tan et al., 2018; OpenAI 2019; Makoviychuk et al., 2021, etc, which unfortunately the proposed system is incapable of. 
- The choice of the five supported robot arms seems arbitrary and not well justified. In fact, those five arms share similar 6-DoF or 7-DoF kinematic structures. It's not diverse enough.
- On the similar vein, the proposed method to annotate grasping poses for different arms is arbitrary too. More principled ways should consider the kinematics and reachability of different robot arms.
- Missing details about the proposed 50 tasks. Do they require dexterous manipulation? Are they short-horizon or long-horizon? These tasks look trivial and not impressive.
- Regarding experiments in sec. 4.1, they are more like ablation studies of the proposed system. More comprehensive comparison to other similar LLM/MLLM-aided robot simulation benchmarks should be presented.
- Regarding the claim about policy robustness derived from experiments in Sec. 4.3, because these tasks are mostly static and there are no external disturbance, it's overclaiming to call robust.
- Sec. 4.4, there are many evidences showing that sim-real co-training can help, e.g., Maddukuri et al., 2025. What's new derived from these experiments?
- Many important details are missing in the paper. See questions above.

## References
- Ha et al., Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition, CoRL 2023.
- Wang et al., RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation, ICML 2024.
- Genesis: A Generative and Universal Physics Engine for Robotics and Beyond, 2024.
- Nasiriany et al., RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots, RSS 2024.
- Tan et al., Sim-to-Real: Learning Agile Locomotion For Quadruped Robots, RSS 2018.
- OpenAI, Solving Rubik's Cube with a Robot Hand, arXiv 2019.
- Makoviychuk et al., Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning, arXiv 2021.
- Maddukuri et al., Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation, RSS 2025.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces RoboTwin 2.0, a scalable framework for automated diverse synthetic data generation and unified evaluation for bimanual manipulation. It includes an object asset library RoboTwin-OD, an expert data generation pipeline that utilizes multimodal large language models, and a structured domain randomization scheme. The 50 benchmark contains 50 bimanual tasks across five robot embodiments.

Empirically, the authors evaluate both code-generation success rates for task generation and robot policy learning via vision-language-action models trained on their synthetic data. Experimental results show improve code-generation quality and significant task performance improvements in both few-shot and zero-shot settings.

### Strengths
1. Dedicated effort into curating tasks with diverse object assets, domain randomization schemes, and support multiple robot embodiments. These are all important aspects in studying bimanual robot manipulation. 

2. Good presentation quality. The authors provided extensive details and visualizations for the task design and simulation implementation details. It's difficult to organize the content when it comes to writing a benchmark paper, and the authors have presented information in a clear way and easy to follow for the readers.

3. Real robot setup and evaluation on sim-to-real transfer. Real world experiment results show clear improvement from using the synthetic simulation data over using only real world demonstrations

### Weaknesses
1. Tasks are limited to table-top manipulation of mainly rigid objects. Compared to some prior works (e.g. RoboCasa), the lack of scene-level assets such as kitchen countertops or cabinets limits the diversity of possible tasks. This is perhaps beyond the scope of the current work, but a larger scene setting would make the benchmark much more useful in studying more tasks and diverse robot behaviors. 

2. Lack of qualitative results on the policy learning performance. It would have been much clearer to show videos of the successful policy rollouts and failure modes. Especially for the real world evaluation experiments, providing videos would provide much more information on how smooth and how fast can the arms move when achieving the task. But the submission did not include any supplementary materials. 

3. Nitpicking: 1) Figure 3 -- the code snippets are way too small and hard to read; 2) Table 3 -- the best-performing methods/tasks should be bolded. 3) Figures 5, 6, 8 all contain relatively small images, would be better to enlarge them for readibility.

### Questions
1. In the object asset library, many objects contain complex geometries and inner holes -- what do their collision shapes look like and how does the physics simulation parse the shapes?

2. What's the control frequency for the robot arms? How are the controllers implemented in both sim and real?

3. Where is the camera for real world evaluation settings? What image processing and/or augmentation is needed?

### Soundness
2

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
4

### Summary
RoboTwin 2.0 is a scalable synthetic data generator and benchmarks for bimanual manipulation. It leverage an MLLM-in-the-loop-code-generation pipeline, enhance domain randomization, and embodiment-aware grasping for a large scale of tasks, different embodiment, and trained on this data, and show large gain in code-generation success and meaningful real-world robustness.

### Strengths
1.RoboTwin 2.0 introduces a closed-loop MLLM-based code-generation system that can automatically create, execute, and refine robot-control scripts until they succeed. This minimizes human supervision while maintaining data quality, allowing the collection of 100 k+ expert trajectories across diverse dual-arm tasks. It demonstrates how large-scale synthetic data for manipulation can be generated programmatically and verified via multimodal feedback.

2.The framework emphasizes five-axis domain randomization (scene, texture, lighting, tabletop, and language) and supports five robot embodiments. This produces visually, spatially, and linguistically varied demonstrations that teach policies to generalize across setups and robot morphologies—something missing in prior single-arm, static-scene datasets.

3.RoboTwin 2.0 doesn’t stop at simulation—it establishes a standardized dual-arm benchmark and shows meaningful sim-to-real gains (≈ +24 % few-shot, +21 % zero-shot) on real hardware. These results validate that richly randomized synthetic data can substantially reduce real-world data needs while improving robustness.

### Weaknesses
1.The paper claims existing benchmarks have weak domain randomization. Is there empirical evidence comparing OOD manipulation benchmarks—for example, The Colosseum, GEMBench, or others?

2.The sim-to-real evaluation covers four bimanual tasks on a single platform (COBOT-Magic) with one policy backbone (RDT), which limits how broadly the gains generalize to other tasks, robots, and policies. Are the real-world results statistically significant, and if so, what tests and effect sizes are reported?

3.Much of the diversity comes from an LLM-templated instruction pool and an 11k Stable-Diffusion–derived texture library. Despite curation, both are synthetic and templated, which may not fully reflect free-form language or real-world background statistics.

### Questions
Refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present RoboTwin 2.0, a simulation platform that provides tasks and datasets featuring dual robot arms. The authors use LLMs and MLLMs to generate task templates and expert demonstrations for these tasks. They incorporate domain randomization to expand the scope of training data. They perform experiments comparing policy learning baselines, and show that co-training with a small amount of real world data enables significantly improved performance on these real world tasks.
 
Generally, I think this is a good paper and I would advocate for its acceptance.

### Strengths
- The paper is generally well-written, with a clearly written motivation section, logical flow, and legible figures
- A comprehensive simulation framework with large-scale assets and the ability to generate tasks and datasets automatically with LLMs
- Comprehensive experiments, first in simulation to benchmark numerous policy learning algorithms, and in the real world to show the utility of the simulation data for learning real world tasks

### Weaknesses
- Details on the expert code generation pipeline are very sparse, and while some details are provided in the appendix, it would be helpful to provide additional context on all the pieces outlined in Figure 3, in the main text (assumptions, inputs, outputs, etc).
- It would be helpful to break down the domain randomization factors further. Which factors contribute most to learning robust behaviors, among scene clutter, lighting, tabletop heights, textures, etc? 
- The real robot experiments use a very small amount of real-world demonstrations (10 demos). How would the insights hold with larger amounts of real-world data (dozens of demonstrations)? It would be interesting to see if the simulation data provides a benefit in more data-rich real-world settings.

### Questions
See points raised in the "weaknesses" section

### Soundness
3

### Presentation
3

### Contribution
3
