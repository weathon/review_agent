# Compositional Multimodal Reasoning for Long-Horizon Robotic Manipulation in Scientific Experiments

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Long-horizon robotic manipulation in scientific experiments requires strict procedural dependencies, multi-stage reasoning, and domain-aware manipulation skills that remain challenging for existing multimodal planning systems. Existing Vision-Language-Action (VLA) models excel at multimodal understanding but often lack explicit symbolic knowledge, limiting their compositional and interpretable planning ability. We present Compositional Multimodal Planner (CoMP), a hierarchical reasoning framework that decouples task understanding, perceptual reasoning, and skill execution for complex experimental procedures. CoMP consists of: (1) a task-level interpreter using chain-of-thought prompting to infer task logic, (2) a mid-level multimodal planner that integrates future scene prediction
to enable visually grounded reasoning, and (3) a low-level skill controller that executes actions via reinforcement learning. This decoupled design enables each component to be optimized independently, improving controllability, extensibility, and generalization without fine-tuning large models. To facilitate evaluation, we introduce a benchmark dataset for scientific experiment tasks. Experiments on both our benchmark and RLBench show that CoMP achieves strong performance and superior compositional generalization compared to competitive baselines, highlighting the advantages of structured and decoupled multimodal planning for long-horizon scientific workflows.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CoMP, a modular framework for long-horizon robotic manipulation in scientific experiments. CoMP separates (i) a task-level LLM planner using chain-of-thought, (ii) a mid-level multimodal planner that predicts future scene frames and maps subtasks to action primitives, and (iii) a low-level RL controller (e.g., DDPG). The authors also introduce a simulation suite and benchmark, report gains over several baselines (including on RLBench), and present a limited sim-to-real evaluation using detection + depth for coordinate transfer.

### Strengths
- Clear modularity & interpretability: Decoupling planning, grounding, and control is well-motivated for long-horizon laboratory workflows and aids debugging/verification.
- New benchmark & analyses: A domain-specific benchmark emphasizing procedural dependencies is valuable.
- Empirical coverage: Results on both the proposed suite and RLBench show consistent improvements on several tasks

### Weaknesses
- Real-robot validation is limited: The physical experiments are narrow in scope and rely on a sim-to-real coordinate mapping; robustness (sensor noise, calibration drift, safety) is under-characterized. Strengthen with more tasks, seeds, and error/failure analyses; report success variance and safety incidents.
- Training–to–deployment gap: Clarify whether the RL policy is trained only in simulation and how it transfers (domain randomization, dynamics mismatch, contact modeling). Provide fine-tuning/none, and quantify performance drop from sim to real.
- Efficiency & latency: The pipeline appears complex (CoT planning → visual prediction → MLM primitives → RL). There is no analysis of runtime (per-step latency, FPS), computational cost, or throughput—critical for long-horizon tasks.
- Baseline completeness and fairness: Include comparisons to simpler Dual-System (System-1 + System-2) paradigms (e.g., “Hi Robot”-style simple planner + skills) and rule-based/task-graph planners to demonstrate that gains require the full CoMP stack. Explicitly discuss fairness for re-implemented baselines (training data, hyperparameters).

### Questions
- RL training & transfer: Was the RL policy trained exclusively in simulation? What domain randomizations were used? Any real-robot fine-tuning? How is stability ensured under perception errors?
- Calibration pipeline: How is the YOLO + depth → simulation mapping calibrated and maintained over time? What is the failure rate due to calibration drift or occlusions?
- Runtime profile: What are the end-to-end latencies for (a) CoT planning, (b) future-frame prediction, (c) MLM primitive generation, and (d) RL control? Where is the bottleneck?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CoMP, a modular framework for long-horizon robotic manipulation in scientific lab settings. It decouples: (i) a task-level LLM planner using chain-of-thought (CoT) with a verification–correction loop, (ii) a mid-level multimodal planner combining a conditional diffusion–based future frame predictor and a multimodal LLM to turn predicted goals into action primitives, and (iii) a low-level RL controller (DDPG). The authors also introduce a small simulation benchmark in CoppeliaSim and report higher success rates than several baselines on their benchmark and mixed results on RLBench, plus limited sim-to-real tests using YOLOv8 + depth mapping.

### Strengths
1. The modular decomposition is easy to understand: a language-based task planner decomposes goals into symbolic subtasks, a mid-level component grounds each subtask using the current and predicted goal images, and a low-level controller executes the resulting action primitives. The paper’s overview figure and method section make this pipeline intuitive.
2. Clear ablations illustrate the role of key modules: the paper varies the LLM and the MLM and reports CoT step-wise ablations, which together help attribute where gains come from.
3. The descriptions of each module are clear in both the main text and the appendix, with concrete prompt templates, training details, and a breakdown of the verification–correction operations; this level of disclosure makes the system easier to understand and (partially) reproduce.

### Weaknesses
1. **Overly complicated pipeline.** The system performs subtask decomposition twice—first via a task-level LLM planner and then again via GPT-4o in the mid-level. It’s unclear why a single VLM could not directly produce the decomposition and grounding, and what unique value the initial LLM adds.

2. **Limited novelty.** The core ideas—LLM/VLM-based subtask decomposition and future image prediction—are well-trodden areas with substantial prior art. The paper does not sufficiently articulate what is new beyond combining known components.

3. **Missing imitation-learning setting.** Most competitive VLA systems and many baselines in the literature rely on imitation learning. The paper uses only RL, making comparisons incomplete and potentially unfavorable to the method’s practicality.

4. **Incomplete empirical validation.** Experiments are almost entirely in simulation with only minimal real-robot results. Claims about long-horizon manipulation and “scientific workflow” utility are not substantiated on physical hardware.

5. **Scalability concerns.** The many moving parts (planner LLM, GPT-4o grounding, predictor, RL controller) raise questions about data efficiency, engineering overhead, and whether the framework scales to large, diverse task suites without prohibitive complexity.

### Questions
1. **On the dual decomposition:** Why is a separate task-level LLM needed if a modern VLM can directly output grounded subgoals/action primitives?

2. **On novelty and positioning:** What specific technical contributions differentiate this work from prior LLM/VLM planning and future-prediction pipelines? Beyond integration, are there new algorithms, training objectives, or guarantees? A related-work table mapping differences would help.

3. **On imitation learning:** Can you include IL baselines and/or an IL variant of your method? If not, please justify why RL is necessary here and report how many demonstrations or episodes would be needed to match RL performance.

4. **On real-world validation:** Can you expand physical-robot evaluation (more tasks, repetitions, failure breakdowns) and report success metrics with confidence intervals? What are the dominant real-world failure modes (perception vs. planning vs. control)?

5. **On scalability and cost:** What is the training/inference cost and data requirement of each module, and how do these scale with task count and horizon length? Any evidence that the architecture remains tractable and reliable when moving to a large multi-task setting?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a non end-to-end pipeline for long-horizon robotic experimentation.

Specifically, the framework first uses a large language model (LLM) to decompose a high-level experimental goal into a sequence of symbolic subtasks {s}, then employs a multimodal language model (MLM) to further break down each subtask into fine-grained action primitives. Finally, a reinforcement learning (RL) controller is used to train and execute the corresponding physical actions.

The paper also introduces a simple benchmark and training dataset containing basic chemical operations (e.g., pouring, stirring, mixing).
Experimental results show that the proposed pipeline, CoMP, achieves state-of-the-art performance on both the proposed benchmark and public datasets.

### Strengths
* Clear motivation: the authors convincingly argue that end-to-end approaches fail to generalize in long-horizon scenarios (due to catastrophic forgetting), and therefore adopt a hierarchical decomposition strategy (LLM -> MLM -> RL). The logic of this design is coherent and well-motivated.

* Dataset contribution: the paper provides a compact but useful benchmark simulating chemical laboratory operations.

* Strong results: CoMP achieves SOTA performance across evaluated benchmarks.

### Weaknesses
* In the public benchmark (Table 4), CoMP performs poorly on the "pick cup" task. This is surprising since a long-horizon planner should, in theory, handle short-horizon tasks more easily, i.e., it shouldn't hurt the original capability of short-horizon tasks.

* The dataset is overly simple, covering only a limited set of basic chemical actions.

* It is unclear how fine-grained CoMP’s control actually is, i.e., in the “pour” task, does the system control liquid volume (e.g., in milliliters), or is the simulation limited to a symbolic pouring gesture without real fluid modeling?

* The RL training procedure is insufficiently described: is the controller trained from scratch, or initialized via imitation learning?

### Questions
The questions are included in the weaknesses.

### Soundness
2

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
- This paper presents CoMP, a compositional and decoupled framework for long-horizon robotic planning in scientific experiments.

- CoMP combines task-level CoT decomposition, multimodal visual prediction, and RL-based control for robotic planning and control.

- This paper also introduces a benchmark dataset for scientific experiment tasks.

### Strengths
- It is sound to employ a modular approach rather than an end-to-end method for long-horizon robotic manipulation in laboratory environments.

- Focusing on autonomous experimental systems for science is highly insightful.

### Weaknesses
- A primary concern is the potential for the entire system to be overly complex, redundant, and time-consuming. Why did the authors choose not to leverage a single powerful Vision-Language Model (VLM), such as GPT-4o or Gemini, to handle both task-level planning and mid-level planning concurrently?

- Is the visual prediction module essential for planning? For a fairer comparison in Table 3, comparing MLM (LLaMA3.2)+RL and MLM+RL is insufficient due to the different base MLLMs used. The authors should have conducted an ablation using MLM (GPT-4o, without vision input) + RL as a baseline. Furthermore, I argue that visual prediction may not be strictly necessary for sub-task planning, as similar grounding could potentially be achieved by employing highly detailed text prompts.

- The visual prediction module necessitates the use of expert demonstration samples. Therefore, the authors' claim of operating "without trajectory-level supervision" is arguably misleading. Although action information is not explicitly used, the cost of acquiring the required demonstration data is comparable to that of standard Imitation Learning (IL) methods.

- The comparison against several IL based works may not be entirely fair, as those baselines do not require a separate visual prediction module. What would be the performance outcome if the authors trained the IL policy models using the exact same demonstration data?

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
