# GeoBench: Rethinking Multimodal Geometric Problem-Solving via Hierarchical Evaluation

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Geometric problem solving constitutes a critical branch of mathematical reasoning, requiring precise analysis of shapes and spatial relationships. Current evaluations of geometric reasoning in vision-language models (VLMs) face limitations, including the risk of test data contamination from textbook-based benchmarks, overemphasis on final answers over reasoning processes, and insufficient diagnostic granularity. To address these issues, we present GeoBench, a hierarchical benchmark featuring four reasoning levels in geometric problem-solving: Visual Perception, Goal-Oriented Planning, Rigorous Theorem Application, and Self-Reflective Backtracking. Through six formally verified tasks generated via TrustGeoGen, we systematically assess capabilities ranging from attribute extraction to logical error correction. Experiments reveal that while reasoning models like OpenAI-o3 outperform general MLLMs, performance declines significantly with increasing task complexity. Key findings demonstrate that sub-goal decomposition and irrelevant premise filtering critically influence final problem-solving accuracy, whereas Chain-of-Thought prompting unexpectedly degrades performance in some tasks. These findings establish GeoBench as a comprehensive benchmark while offering actionable guidelines for developing geometric problem-solving systems. Our benchmark and
code are released at https://github.com/FrontierX-Lab/GeoBench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces GeoBench, a hierarchical benchmark for multimodal geometric problem solving. It defines four reasoning levels: Visual Perception, Goal-Oriented Planning, Rigorous Theorem Application, and Self-Reflective Backtracking, instantiated via six multiple-choice tasks built from problems generated and formally verified by TrustGeoGen. The dataset has 1021 items and is used to evaluate a range of general and reasoning MLLMs. Key empirical findings: (i) performance drops as task difficulty rises; (ii) sub-goal decomposition and irrelevant-premise filtering correlate most with final-answer accuracy; (iii) Chain-of-Thought can hurt faulty-branch localization; (iv) some OOD generalization is observed to GeoQA/Geometry3K.

### Strengths
- Hierarchical evaluation design. The four-level tasks with difficulty level increase is principled, and the task design reflects real-life use cases. 

- Using TrustGeoGen with verified reasoning graphs reduces label noise.

- Comprehensive evaluations cover many popular general and reasoning MLLMs and CoT ablation highlighting task-specific limits.

### Weaknesses
* While TrustGeoGen with formal verification is a strength, the heavy reliance on synthetic problems may narrow the distribution. It remains unclear how faithfully the tasks reflect real contest or classroom diagrams.

* Results are reported mostly as point accuracies; significance tests, confidence intervals, or bootstrap estimates are missing.

* A random-choice baseline is trivial; a properly calibrated human baseline (e.g., novice vs. expert) is needed to contextualize difficulty and headroom.

* Inaccurate or overstated claims in results discussion.

  * The statement around lines 346–348 (e.g., *“... T.S. task where other models generally perform poorly.”*) is not accurate, as **o1** attains similar performance.
  * The claim *“Reasoning MLLMs consistently outperform General MLLMs … confirming their superior reasoning capabilities …”* overgeneralizes; several non-reasoning models outperform some reasoning models on certain tasks. These statements should be revised to reflect the actual per-task and per-model variability.

* Some benchmark details require further clarification (see the Questions section).

### Questions
* What is difference between GeoBench and GeoBench-Solving, is the later a subset of the former one? 

* The results in Table 4 and Table 5 have a lot overlap, which may not necessary.

* A proper reference for TrustGeoGen at line 155 is missing. 

* Important evaluation details such as temperature, max tokens, retry logic, and related decoding parameters are not fully reported.

* In Section 3.2, how the distractors are designed? 

* In Table 4, the DeepSeek-R1 results is obtained via image textual descriptions. This may cause unfair comparison as models may benefit from accurate image textual descriptions, especially for geometric problem solving. How the other models perform when the image is replaced by the textual descriptions?

* Some results in Table 4 seems unintuitive. For example, Qwen2.5-VL-7b performs pretty well on N.P. tasks but suddenly very poor on S.P. tasks, which all belong to Level 1 tasks. 

*  In Section 4.3, how the feature vectors are generated? I am a bit confused by this section. 

*  In Section 4.4, why the other two benchmarks are OOD? What is the point of studying other two benchmarks in your paper? 

*  In Section 4.5, it would help to include side-by-side qualitative examples with vs. without CoT, highlighting where CoT helps or hurts. Annotated failure cases would strengthen the argument.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes GeoBench, a hierarchical benchmark to evaluate geometric reasoning in multimodal large language models. It introduces four levels: visual perception, goal-oriented planning, rigorous theorem application, and self-reflective backtracking which is implemented through six tasks derived from formally verified problems generated using TrustGeoGen. Across 1,021 examples, the authors show that reasoning-oriented models like OpenAI-o3 outperform general MLLMs, though performance decreases sharply with task complexity. The analysis highlights that sub-goal decomposition and irrelevant premise filtering are key determinants of geometric problem-solving accuracy and that Chain-of-Thought prompting is not universally beneficial.

### Strengths
I liked the clarity in the paper's writing and the results are comprehensive, and have two strengths to highlight:

- Hierarchical evaluation grounded in cognitive theory: The benchmark’s structure, inspired by the van Hiele model, allows precise diagnosis of reasoning abilities rather than measuring final-answer accuracy alone.

- Comprehensive and formally verified dataset: Using TrustGeoGen ensures rigorous, contamination-free problem generation, making GeoBench a strong diagnostic tool for evaluating how MLLMs reason through geometric logic.

### Weaknesses
The benchmark relies on synthetic, clean diagrams and controlled premises. This limits assessment of robustness to real-world variability such as hand-drawn figures, scanned textbook noise, ambiguous markings, and imperfect annotations. Adding a real-diagram slice or perturbation suite would strengthen ecological validity.

### Questions
I do not have any particular concerns with the paper, except maybe including some more MLLMs such as Molmo, Math-LLaVA and MathPuma. The authors can show some preliminary results on these models

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Addressing three major limitations in current geometric reasoning evaluation—the risk of test data contamination, overemphasis on final answers, and insufficient diagnostic granularity—this paper proposes a more scientific and refined hierarchical benchmark called GeoBench. Through systematic evaluation, this benchmark not only reveals the capability limitations of existing AI models in handling complex geometric problems but, more importantly, precisely diagnoses the core factors affecting model performance.

### Strengths
1. It identifies key limitations in geometric reasoning evaluation (e.g., data contamination, overemphasis on answers) and addresses them through GeoBench—a hierarchical benchmark that decomposes geometric reasoning into distinct stages

2. The benchmark leverages the TrustGeoGen methodology to generate tasks verified for logical rigor, ensuring data novelty and mitigating contamination risks. This establishes a reliable foundation for equitable model evaluation.

### Weaknesses
--The evaluation framework lacks a necessary human verification step. Given the complexity of the dataset problems (as shown in Figure 4), establishing a performance baseline from human experts is crucial. Furthermore, the scope of evaluation should be expanded to include advanced mathematical reasoning agents—particularly those capable of using tools for exploration or constructing auxiliary lines—in order to assess the true capabilities of current models under problem-solving paradigms that closely resemble human approaches.

--Regarding the reasoning graphs relied upon in the synthetic data generation process, key details—such as the quality of these graphs and the fidelity with which models adhere to them—are not sufficiently elaborated. This raises concerns about potential risks to synthetic data quality: if models do not strictly follow the intended reasoning logic, or if problems are constructed based on flawed reasoning graphs, the validity and reliability of the resulting problems may be called into question.

### Questions
N/A

### Soundness
2

### Presentation
3

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
This work presents a benchmark to evaluate multimodal geometric reasoning by VLMs/MLLMs. The hierarchical evaluation proposed offers a novel assessment framework that goes beyond answer accuracy metrics. Additionally, this helps identify the exact steps at which models fail in the solving process, providing actionable insights into improving model performance. Finally, this work identifies the failure of the common CoT approach when it comes to geometric reasoning.

### Strengths
1. The hierarchical framework is a useful diagnostic tool to provide actionable insights into pitfalls in the reasoning process.
2. The benchmark presented in this work goes beyond data collation and adds a unique set of information for analysing model performance.

### Weaknesses
1, This work does not detail how the automatically generated tasks are verified for accuracy and legitimacy
2. Likewise, there is no insight into how the automatically-generated problems are distributed in terms of logical and reasoning complexity. In addition to the empirical comparison against established benchmarks and their levels, the work could benefit from a deeper, and qualitative, analysis of the complexity and difficulty of the problems in this benchmark,

### Questions
Questions:
1. How does this benchmark ensure diversity in the questions generated?
2. Can this generation and evaluation framework be extended to geometric problems in the 3D space?

Suggestions:
1. This work would benefit from further proofreading. Some errors that could be fixed:
- 1.1 L161: TrustGenGen -> TrustGeoGen
- 1.2 L250: Capitalizing the first letter of the section.

### Soundness
2

### Presentation
3

### Contribution
3
