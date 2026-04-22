# TURTLEAI: Benchmarking Multimodal Models in Turtle Graphics for Visual Programming and Reasoning

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
Multimodal vision-language models (VLMs) have achieved remarkable success in fundamental visual tasks like image captioning and visual question answering. However, their performance on complex visual tasks requiring integrated visual reasoning and problem-solving capabilities remains underexplored. To bridge this gap, we introduce TurtleAI, a multimodal benchmark to evaluate VLMs on visual programming and reasoning tasks in the Turtle Graphics domain. Our benchmark contains 823 visual programming tasks that challenge VLMs to generate Python code to replicate patterns in images. Evaluation of 20 VLMs reveals that state-of-the-art models like GPT-4o and Qwen2-VL-72B struggle with these tasks, achieving success rates of only 26.5% and 11.8% respectively. Our analysis reveals that models often fail to align their code implementation with visual reasoning. To address this misalignment, we propose TurtleAI-Datagen, a data generation framework that creates large-scale synthetic datasets consisting of task-code pairs. Using just 10 initial samples, TurtleAI-Datagen generates over 700k samples. Fine-tuning on this dataset significantly reduces errors arising from the misalignment between visual reasoning and program synthesis, improving Qwen2-VL-72B's performance by over 20%. We will release the benchmark publicly to facilitate future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a multimodal benchmark, TURTLEAI, which challenges vision-language models (VLMs) to generate Python code for drawing patterns. TURTLEAI consists of three components: TURTLEAI-DS (a collection of datasets), TURTLEAI-Eval (an evaluation framework), and TURTLEAI-Datagen (a data generation framework).
TURTLEAI-DS contains pairs of images and their corresponding Python code. A VLM is required to generate Python code based on a given image. The generated image is then compared with the original image using TURTLEAI-Eval. TURTLEAI-Datagen is designed to generate (image, code) pairs and chain-of-thought (CoT) reasoning examples for fine-tuning VLMs.
Experimental results demonstrate that existing VLMs struggle to perform well on these tasks, but fine-tuning the models significantly improves their performance.

### Strengths
-	The paper introduces a new benchmark to facilitate the development of VLMs.
-	It reveals that existing VLMs struggle with visual programming tasks.
-	It proposes a data generation framework for fine-tuning.

### Weaknesses
- In TURTLEAI-Eval, which compares drawings in a transformed, normalized space, the line width is standardized to a fixed value of 1. However, certain shapes may require width information for accurate representation. For instance, a solid rectangle can be considered as a very thick line, and its width plays a crucial role in distinguishing it from other shapes.
- In Symbolic comparison and Embedding-based comparison, there is a predefined threshold. What impact does this threshold have on the evaluation results? How is this threshold determined, and is there any experimental evidence to support it?
- In TURTLEAI-DATAGEN, two codes are randomly selected from the dataset to extract their high-level mutation pattern. However, the selected codes may not have a clear mutation pattern (e.g., adding a loop to the code).
- In stage 3 of TURTLEAI-DATAGEN, is there any quality check for the generated CoT reasoning?

### Questions
If the writing could clarify the code mutation process in more detail, especially the extraction of high-level mutation patterns from reference codes, and include some concrete examples, it would enhance the quality of the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces TURTLEAI, a dataset and benchmark for visual to code generation in Turtle Graphics, along with a synthetic data pipeline that reportedly improves model performance.      However, the benchmark is evaluated only on its own synthetic distribution, and the observed improvements are likely due to domain exposure rather than genuine reasoning gains.      OOD results show that finetuning actually makes models worse on hand-drawn sketches, suggesting harmful overfitting.      No external datasets, no comparisons to existing sketch to code benchmarks, outdated baselines, and poor presentation quality further weaken the contribution.      While releasing the dataset could be valuable to the community, the scientific impact and novelty of the paper are limited.

### Strengths
1. Clear problem formulation and a well defined task scope.
The paper focuses on visual-to-code generation via Turtle Graphics, which is a structured setting where correctness can be objectively evaluated via execution.

2.  Infrastructure contribution.
The authors provide a dataset, an evaluator, and code for data synthesis and benchmarking.      If fully released, the benchmark could serve as a reproducible testbed for small scale visual to program induction.

3.  Diagnostic experiments.
The paper analyzes failure types and includes limited OOD testing, which helps reveal the limitations of current VLMs on synthetic geometric tasks.

### Weaknesses
1.   Major validity issue: the benchmark is only evaluated on its own synthetic data, with no external datasets or established baselines.
The paper trains on data generated by its own pipeline and then evaluates on a benchmark created from the same distribution.     This closed loop validation prevents demonstrating scientific novelty, generality, or impact.     There is no evidence that performance gains reflect reasoning improvements rather than domain overfitting or memorization.

2.  OOD experiments contradict the core claims.
In Figure 7, models fine-tuned on TURTLEAI data perform significantly worse on hand-drawn OOD sketches.     Since both tasks are visually to code mapping, this drop indicates the model becomes *less* general and more brittle after training suggesting harmful domain overfitting rather than capability improvement.     This undermines the core contribution of the dataset and training pipeline.

3.  The baseline success rates are extremely low (~10%), making “20% improvement” uninformative.
Because current VLMs have never been trained on synthetic Turtle-style graphics, any improvement after domain exposure is expected and does not imply methodological novelty.     The paper does not demonstrate that its data synthesis strategy is better than simpler alternatives such as random sampling, Self-Instruct, Evol-Instruct, or manual template expansion.

4.  No comparison with existing visual to program or sketch-to-code benchmarks.
The work ignores relevant prior datasets (e.g., SVG/TikZ program induction, sketch-to-code datasets, visual UI-to-code tasks).     Without external validation, it is unclear whether TURTLEAI measures general reasoning or just fits a narrow toy domain.

5.  The data and task space are extremely toy-like.
All visuals are synthetic geometric primitives with perfect rendering (no noise, occlusion, perspective, thickness variation).     Claims about “general visual reasoning” or “broad program synthesis” are overstated relative to the simplicity of the domain.

6.  Paper presentation quality is below conference standard.
All tables are mislabeled as figures, and captions are consistently placed incorrectly.     This violates standard formatting guidelines and indicates insufficient care in preparation.

### Questions
1.  Why are there no experiments on existing sketch-to-code or visual program induction benchmarks?     Without external evaluation, how can the benchmark claim scientific relevance beyond its own synthetic sandbox?
2.  In Fig.7, why does fine-tuning severely degrade performance on hand-drawn sketches?     Doesn’t this imply the dataset harms general visual programming rather than improves it?
3. What proportion of generated samples are incorrect, redundant, or semantically invalid?     Is there any human auditing, or are models only self-evaluating their own output?
4.  How do you demonstrate that the proposed data synthesis approach is superior to simpler strategies (e.g., template mutation or random augmentation)?
5. Many baselines used (e.g., GPT-4V, Qwen2-VL) are outdated relative to 2024–2025 VLMs.     Why are recent models (GPT-4o, GPT-5，Qwen2.5-VL，Qwen3-VL，etc.) missing?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces TURTLEAI, a multimodal benchmark designed to assess visual programming and reasoning in Turtle Graphics. It provides 823 tasks requiring models to generate Python code that reproduces target images. Experiments on 20 VLMs show that state-of-the-art models struggle, with GPT-4o and Qwen2-VL-72B achieving low success rates. The authors also propose TURTLEAI-Datagen, a synthetic data generation pipeline producing 700k+ samples from 10 seeds, improving model performance by over 20% after fine-tuning.

### Strengths
1. The paper offers a well-structured task setup linking visual reasoning to program synthesis and conducts systematic experiments across many VLMs, yielding a solid empirical assessment of model limitations in structured visual-to-code settings.

2. The synthetic data framework is executed at scale and empirically improves model performance, demonstrating practical value for enhancing VLM capability in controlled visual programming tasks.

### Weaknesses
1. Limited novelty relative to prior benchmarks: Similar multimodal visual-to-code and graphics reasoning benchmarks already exist (e.g., NAACL 2025 TurtleBench: A Visual Programming Benchmark in Turtle Geometry https://aclanthology.org/2025.naacl-long.607/
). The paper primarily repackages an existing paradigm (image-to-code in a constrained graphics domain) rather than introducing a fundamentally new task or evaluation angle.

2. Narrow and arguably low-impact domain: Turtle Graphics is a highly simplified, pedagogical environment with limited real-world relevance. Performance in this synthetic sandbox does not clearly translate to practical multimodal programming, robotics, CAD/graphics reasoning, or general visual planning tasks. The paper lacks evidence that gains on Turtle tasks meaningfully correlate with improvements on broader multimodal program synthesis or vision-reasoning benchmarks, raising questions about external validity and actual scientific payoff.

3. Evaluation and insights remain shallow: While the paper reports success rates and shows synthetic data improves scores, the analysis stops short of deeper failure categorization, ablation across visual complexity factors, or diagnostics that could reveal why models fail (e.g., perceptual ambiguity vs. planning vs. code correctness). The work also does not benchmark against alternative data augmentation or curriculum approaches, nor does it explore whether improvements generalize beyond the Turtle setting, limiting interpretability and impact of the proposed method.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
