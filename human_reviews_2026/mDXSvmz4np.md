# TRACE: Coarse-to-Fine Automated Evaluation of Mobile Agents with Safety Considerations in Realistic Environments

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
The online evaluation of mobile agents is becoming increasingly important for both accurately assessing agent capabilities and providing reward signals for online reinforcement learning. Evaluating mobile agents on complex multi-step tasks remains challenging, as existing work suffers from limitations in reliability and generality, while overlooking issues of environmental realism and operational safety. This paper introduces TRACE (TRajectory-based Automated Coarse-to-fine Evaluation), a fully automated vision language model (VLM)-based method designed to evaluate arbitrary mobile agents across diverse environments. TRACE evaluates agent trajectories in a two-stage manner, first through step-wise assessment and then through overall judgment, which significantly reduces evaluation difficulty and enhances reliability. Potentially risky or harmful operations are also detected simultaneously during the step-wise assessment. Furthermore, we construct TRACEBench, a scalable benchmark consisting of 187 tasks from 35 commonly used mobile applications, to better reflect the actual performance of agents in realistic online environments. Task design explicitly considers operational safety, and evaluation metrics cover three key dimensions: task completion, safety, and resource consumption. Experiments show that TRACE achieves an F1 score of 0.836 with the open-sourced Qwen2.5-VL-72B-Instruct, indicating high precision as well as better usability and cost-effectiveness. Extensive evaluation of 8 representative mobile agents on TRACEBench reveals that current mobile agents still have substantial room for improvement, particularly in terms of task completion and operational safety.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes TRACE, a coarse-to-fine automated evaluation framework for mobile GUI agents that integrates step-level reasoning with task-level verification. It introduces TRACEBench, a benchmark of 187 real-world tasks across 35 apps with safety-aware and multi-modal evaluation metrics. Experiments show that TRACE achieves evaluations closely matching human judgments and outperforms prior automatic evaluators such as SPA-Bench.

### Strengths
1. The paper tackles an important and underexplored problem: evaluating GUI-based agents, which involves unique challenges beyond standard LLM-as-judge settings.
2. The proposed coarse-to-fine evaluation framework is novel, improving upon prior rule-based or single-VLM evaluators through structured step-level and task-level reasoning.

### Weaknesses
1. The overall focus of the paper is somewhat unclear. The main contribution appears to be developing a new evaluation framework for assessing the trajectories of GUI agents, which could help benchmark existing agents or assist in training stronger ones. However, the description of the proposed method is too brief and lacks sufficient clarity. It is difficult to understand how the method actually works and why it performs better than previous approaches. This lack of detail also affects the design of experiments and the explanatory figures.

2. The method section is overly simple and provides few technical details. Key implementation components are missing, making it hard to grasp how the framework operates in practice. Moreover, the illustration figures fail to convey useful information. Although the paper emphasizes a “coarse-to-fine” evaluation strategy, it does not clearly explain how step-wise information is incorporated into the final evaluation process.

3. While the paper presents a large number of experiments, the experimental focus seems misplaced. If the main goal is to validate the effectiveness of the evaluator, a large portion of the experiments (e.g., Table 1) is spent on evaluating different GUI agents, which is not the main point of the paper. At least this experiment should not be presented as the most important result.

4. The paper missed discussion with previous works on evaluting GUI agents in the related work section.

### Questions
1. If all step-level evaluations are run in parallel, how does each step obtain information from the previous ones? For instance, an agent may collect intermediate results or context in earlier steps, and later actions could depend on those results in ways that are not explicitly related to the task description. Similarly, an agent might perform recovery behaviors to fix earlier mistakes, which are also not directly tied to the original goal. How does TRACE handle such dependencies when evaluating each step independently?

2. How does TRACE automatically generate task milestones? In cases where a task can be completed in multiple valid ways (e.g., checking the weather through different apps or directly searching online), how does TRACE ensure that alternative valid solutions are not penalized?

3. TRACEBench contains a relatively small number of tasks compared to prior benchmarks. Given that this paper focuses on evaluation rather than agent training, why did the authors not use existing benchmarks such as AiTW to demonstrate TRACE’s effectiveness and generality?

4. Why are timeout and safety issues explicitly analyzed in the experimental section? These aspects seem to relate more to agent design or behavior, rather than to the evaluation framework itself. Clarifying the motivation behind including these metrics would help.

5. The experimental description for Table 1 is somewhat unclear. Are recall and precision defined by comparing the agent’s self-reported evaluation results with TRACE’s evaluation, or between TRACE’s evaluation and the human-annotated ground truth? If it is the former, how does this result demonstrate the effectiveness of TRACE as an evaluator?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces TRACE (TRajectory-based Automated Coarse-to-fine Evaluation), a fully automated VLM-based framework for evaluating mobile GUI agents. TRACE adopts a two-stage coarse-to-fine design that first analyzes individual actions and then performs an overall judgment based on aggregated evidence. The authors also release TRACEBench, a benchmark of 187 mobile tasks across 35 apps with explicit safety-focused cases. Experiments show that TRACE achieves an F1 score of 0.836 against human annotations, outperforming SPA-Bench and demonstrating improved reliability and automation. The work aims to advance realistic, safety-aware, and scalable evaluation of mobile agents.

### Strengths
1. **Approach and design**:
TRACE proposes a coarse-to-fine decomposition that isolates local reasoning from global evaluation, substantially reducing VLM context complexity. This staged design is intuitive and effective, producing a strong correlation (F1 = 0.836) with human judgment while remaining fully automated.
2. **Practical and accessible implementation**:
TRACE attains competitive evaluation accuracy using an open-source VLM (Qwen2.5-VL-72B-Instruct). The reported improvements in F1 score, precision, and recall over SPA-Bench provide solid empirical evidence of its effectiveness.
3. **Clarity and presentation quality**:The paper is clearly written and well structured, with sufficient methodological descriptions and illustrative examples that make the proposed framework easy to follow. The inclusion of multiple VLM evaluators (Qwen, Doubao, GPT-4o, Gemini) also helps demonstrate the robustness and general applicability of the proposed approach.

### Weaknesses
1. **Unfocused contribution scope**: The paper claims three contributions—pipeline, benchmark, and safety detection—but only the pipeline represents a substantive methodological novelty. The benchmark and safety modules, though useful, blur the central narrative. A tighter focus on evaluating the pipeline across existing benchmarks (e.g., Mobile-Bench, SPA-Bench) would better clarify its impact.
2. **Limited benchmark novelty and scale**:
TRACEBench’s 187 tasks are modest compared with prior work (e.g., Mobile-Bench 832, Mobile-Bench-v2 > 12 k). The Chinese-only design restricts accessibility and limits cross-lingual generality. The rationale for this language choice and the proportion of newly designed versus adapted tasks should be better justified.
3. **Insufficient safety evaluation rigor**:
The safety detector is prompt-based and lacks quantitative validation (precision/recall) or baseline comparison against simple heuristics. Without such analysis, it appears closer to prompt engineering than a verified safety metric. The trade-off between Success Ratio (SR) and Safety Ratio (SFR) could also be explored more systematically.
4. **Lack of ablations and pipeline analysis**:
Although the step-wise design is conceptually strong, the paper does not test simpler baselines such as (i) final-screenshot-only or (ii) single-pass full-trajectory evaluation. An ablation quantifying how much each stage contributes to accuracy would strengthen the argument that coarse-to-fine evaluation is necessary.
5. **Uncontrolled execution environment and variance**:
TRACEBench relies on real mobile devices, introducing nondeterminism (ads, residual states). The paper acknowledges this but does not quantify benchmark variance or report results from repeated runs. Without this, the statistical significance of agent performance differences remains unclear.
6. **Limited comparison to other VLM/LLM evaluators**: SPA-Bench is the only baseline. Other recent works (e.g., Android Agent Arena [1] and OSUniverse [2]) also propose automated evaluators using VLMs or LLMs. Evaluating against these would clarify TRACE’s relative advantage.
7. **Milestone reliability not evaluated**:
The overall judgment phase critically depends on accurate milestone decomposition. Errors in milestone extraction could cascade into incorrect final judgments, yet no quantitative assessment of this stage’s reliability is provided.
8. **Metric definition and motivation**:
Metrics such as Complete Recall/Precision (CR/CP) are introduced but not well-motivated relative to standard success metrics. Concrete examples showing when CR/CP provide distinct insights would help justify their inclusion.

### **References**
[1] Android Agent Arena: A3 Benchmark for Mobile GUI Agents. https://arxiv.org/abs/2501.01149

[2] OSUniverse: Benchmark for Multimodal GUI-Navigation AI Agents. https://arxiv.org/abs/2505.03570

### Questions
1. **Benchmark motivation**:
What unique features of TRACEBench justify introducing a new dataset instead of expanding existing ones such as Mobile-Bench or SPA-Bench? How many of the 187 tasks are genuinely new versus adapted?
2. **Language choice**:
Why were all task instructions written in Chinese, including for global apps? 
3. **Environment control and reproducibility**:
When using real devices, how do you ensure fairness across runs? Was any emulator, snapshotting, or re-initialization used?
4. **Milestone reliability**:
Have you measured the correctness of automatically generated milestones? How sensitive is overall performance to errors at this stage?
5. **Metric usefulness**:
Please illustrate cases where CR/CP provide insights beyond success ratio, or consider simplifying the metrics if redundant.

### Soundness
2

### Presentation
2

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
This paper addresses the challenges of evaluating mobile agents in real-world environments by proposing TRACE, a fully automated, VLM-based "coarse-to-fine" evaluation framework. TRACE performs step-wise assessment to extract screen clues, action effects, and safety risks, followed by an overall judgment stage that integrates task milestones and the final state. The authors also construct TRACEBench, a benchmark comprising 187 tasks across 35 commonly used applications, with a particular emphasis on safety and real-world scenarios. Extensive experiments on eight representative mobile agents demonstrate the advantages of TRACE in terms of accuracy, generalizability, and automation.

### Strengths
1.TRACE decomposes evaluation into step-level analysis and overall judgment, significantly reducing the complexity of understanding long trajectories and improving assessment accuracy and reliability.
2.RACEBench covers diverse daily applications and tasks and deliberately includes 17 risky tasks, systematically introducing quantitative safety metrics for mobile agent evaluation for the first time.
3.TRACE operates without manual annotations or predefined rules, relying solely on task descriptions, screenshots, and action sequences. It supports evaluation across environments, tasks, and agent architectures, demonstrating strong scalability.

### Weaknesses
1. Running TRACEBench directly on real devices means environmental noise (e.g., pop-up ads, residual states) can lead to inconsistent results. While parameterization and repeated runs are proposed as mitigations, this may still affect evaluation stability.
2.Experiments show TRACE's performance varies with the chosen VLM (e.g., Doubao performs best, while GPT-4o and Gemini struggle on some tasks). This could limit its applicability in resource-constrained or specific linguistic environments.
3.Although TEX and TEV metrics are mentioned, there is insufficient analysis of practical deployment concerns like computational overhead and response time for large-scale evaluations.

### Questions
1.	Beyond parameterization and repeated runs, does TRACE incorporate other mechanisms (e.g., context-aware reasoning or dynamic adaptation) to enhance robustness against environmental stochasticity like ads or dynamic content?
2.	Have you considered adapting TRACE for non-Chinese environments or other mobile platforms (e.g., iOS)? Would your recommendations for VLM selection differ in multi-lingual or cross-platform scenarios?

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
The paper proposes TRACE, a coarse-to-fine method for judging mobile agent trajectories, along with a new benchmark for mobile agents, TRACEBench. The paper evaluates and analyzes several existing models on TRACEBench.

### Strengths
* The paper proposes a new benchmark for mobile agents, with greater focus on apps popular in China and on safety considerations.
* The paper proposes a new methodology for building LLM evaluators for mobile agents, with improvements in F1 over a method from prior work. 
* The paper analyzes the performance of several agents on the proposed benchmark.

### Weaknesses
* The paper has several contributions: (1) TRACE, a new method for building a LLM evaluator by decomposing the task into a pipeline system, (2) TRACEBench, a new benchmark for mobile agents, and (3) empirical analysis of several models on TRACEBench. However, each contribution is a bit underdeveloped. For example, the paper lacks justification for TRACE's complexity (see point below), despite this contribution being most emphasized in the title and narrative. The new benchmark could potentially be a more compelling contribution, but it is not well justified with respect to existing benchmarks like AndroidWorld. A new benchmark focusing on apps more popular in China could be useful to test i18n capabilities of agents beyond English, but I didn't quite understand the justification otherwise. A greater focus on safety judgements could also be useful, but this contribution was a bit lost in the current narrative.
* Prior work has also proposed various pipelined systems for improving automated UI agent evaluation, e.g. WebJudge from https://arxiv.org/abs/2504.01382. Since this paper proposes a relatively complex pipeline for evaluation, it would be useful to perform a more comprehensive analysis of which specific choices are most important and how they compare with prior work. For multimodal models with sufficient context windows, how does the approach compare with naively encoding the complete trajectory (the paper claims performance would be better, but this claim is not empirically supported)? Does this difference depend specifically on the context window and screen understanding capabilities of the underlying model? Without more analysis, it's difficult to understand the key take-aways from the various design decisions of TRACE. If a new LLM evaluator is indeed the key contribution, it would be useful to evaluate it across multiple benchmarks, e.g. you could analyze agreement with heuristic rewards on AndroidWorld compared with LLM evaluators from prior work.

Summary: I think the paper could be improved for a future submission by either (1) focusing on the benefits of the proposed benchmark, relaxing some of the claims related to the specific LLM evaluator or (2) expanding the empirical analysis of different design choices related to LLM evaluators, extending comparisons of different evaluators to existing benchmarks.

Nits:

* The method is described as "coarse-to-fine", but TRACE is actually the reverse. It starts with fine step-level judgements and then makes a coarse trajectory judgement. This was perhaps not the best terminology.
* Figure 1 - Why no arrow from (4) to (5)?

### Questions
See weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2
