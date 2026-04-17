# OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across various visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the absence of a dedicated and systematic benchmark. To address this gap, we propose OCR-Reasoning, a novel benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. Specifically, OCR-Reasoning comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Unlike existing text-rich image understanding benchmarks that only provide a final answer, this benchmark additionally provides a detailed step-by-step reasoning process. This dual annotation enables the evaluation of both the models' final answers and their reasoning processes, thereby offering a holistic assessment of text-rich reasoning capabilities. By leveraging this benchmark, we conducted a comprehensive evaluation of the latest MLLMs. Our results demonstrate that even the most advanced MLLMs exhibit substantial difficulties in text-rich image reasoning tasks, with none achieving an accuracy above 50\% on our benchmark, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at https://github.com/SCUT-DLVCLab/OCR-Reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reveals severe capability gaps in current advanced multimodal slow-thinking systems when performing text-rich image reasoning tasks, attributing this limitation fundamentally to the absence of a systematic and specialized evaluation benchmark. To address this challenge, this paper introduces the OCR-Reasoning benchmark, characterized by its large-scale manual annotations, coverage of multi-dimensional reasoning tasks, and core feature of "dual annotation" providing both answers and detailed step-by-step reasoning processes.

### Strengths
This paper accurately identifies a crucial gap in multimodal reasoning research by proposing the first benchmark specifically designed for systematically evaluating "text-rich image reasoning" capabilities
﻿
The constructed OCR-Reasoning benchmark surpasses traditional models that focus solely on answer correctness. By incorporating annotations of step-by-step reasoning processes, it achieves dual evaluation of both model reasoning paths and final answers, providing a more comprehensive and profound perspective for diagnosing model capability shortcomings.

### Weaknesses
1. The study primarily focuses on diagnostic evaluation but does not validate the effectiveness of its proposed "thinking with images" reasoning approach. A critical question remains unanswered: can explicitly training models to generate such visual reasoning chains on the proposed benchmark lead to significant gains in reasoning performance? The paper would be significantly strengthened by including fine-tuning experiments that demonstrate whether and how leveraging this benchmark for training, as opposed to merely for evaluation, improves model capabilities on text-rich reasoning tasks.

2. The paper would benefit from a more thorough comparative analysis between the proposed OCR-Reasoning dataset and existing text-rich benchmarks (e.g., TextVQA, DocVQA, ChartQA). While the novel "dual annotation" is highlighted, a quantitative and qualitative comparison is needed to clearly delineate its advantages. This should explicitly detail the dataset's superiorities in terms of data quality (e.g., the depth and consistency of reasoning chain annotations), data characteristics (e.g., the diversity and complexity of reasoning types beyond simple QA), and coverage (e.g., the inclusion of scenarios that require multi-hop reasoning).

3. Although the benchmark's quality is commendable, its scale (approximately 1,069 examples) may limit its comprehensiveness and statistical power. The covered visual domains might not fully represent the vast spectrum of real-world text-rich images. To enhance the robustness and generalizability of the findings, the authors could consider expanding the dataset to include more diverse sources, such as web screenshots (for UI/UX reasoning), financial documents (for complex table and report understanding), academic papers, or product manuals. This would ensure that the benchmark tests reasoning capabilities across a broader range of practical contexts.

4. The benchmark is constructed manually, which ensures high quality but is not scalable for generating large-scale training data. A discussion on potential methodologies for scalable training data construction is a crucial missing piece. The authors could propose and discuss semi-automatic or synthetic data generation techniques that could produce large volumes of text-rich image reasoning data. For instance, exploring how advanced models (like GPT-4V) could assist in drafting reasoning chains for human review, or how to create synthetic tasks that inherently require visual-textual reasoning, would greatly enhance the practical utility and impact of this work beyond evaluation.

### Questions
see the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the OCR-Reasoning benchmark, designed to evaluate the reasoning capabilities of Multimodal Large Language Models (MLLMs) in complex text-rich image scenarios. Unlike existing benchmarks that focus on text extraction, OCR-Reasoning includes both final answers and step-by-step reasoning processes for 1,069 annotated examples across six core reasoning categories. The benchmark reveals significant limitations in current MLLMs, with top models achieving only around 50% accuracy in reasoning tasks, highlighting struggles in integrating visual, textual, and logical information. The study emphasizes the importance of Chain-of-Thought prompting and provides valuable insights into common failure modes such as calculation, spatial comprehension, and logical errors. This benchmark aims to inspire future advancements in multimodal reasoning systems.

### Strengths
The paper demonstrates strong originality by introducing a new benchmark, OCR-Reasoning, that evaluates multimodal large language models on text-rich image reasoning—a domain previously underserved by existing datasets focused mainly on text extraction. Its dual annotation of final answers and reasoning steps represents a creative and meaningful extension of prior benchmarks.

In terms of quality, the study employs a rigorous and transparent methodology, including systematic dataset curation, expert annotation, and comprehensive evaluation across a broad spectrum of state-of-the-art models. The experimental design is sound, with clear baselines, detailed performance breakdowns, and thoughtful error analysis.

The paper exhibits strong clarity, with a well-structured presentation, clear motivation, and informative figures that effectively illustrate dataset design and results. The writing is precise and technically competent, enabling easy understanding of both the problem and its importance.

Regarding significance, the work makes a timely and impactful contribution to the multimodal reasoning community. By exposing the current limitations of MLLMs in integrating textual and visual reasoning, it establishes an essential benchmark that will likely guide future research on improving model reasoning and evaluation frameworks.

### Weaknesses
The dataset is relatively small (1,069 samples), limiting generalization and coverage of diverse real-world scenarios. The reliance on LLM-as-Judge introduces potential bias; incorporating human or cross-model validation would improve reliability. The paper lacks deeper diagnostic analysis explaining why models fail, and provides limited quantitative comparison with prior benchmarks. Finally, details on dataset release and reproducibility are insufficient, which may hinder adoption.

### Questions
1. Dataset Scale and Coverage:
Could the authors clarify whether there are plans to expand OCR-Reasoning beyond 1,069 samples? A larger and more diverse dataset (e.g., multilingual, domain-specific, or handwritten documents) would improve the benchmark’s representativeness.
2. Evaluation Bias in LLM-as-Judge:
How do the authors mitigate potential bias when using LLMs to evaluate reasoning quality, especially if the judging model shares architecture or training data with the tested models? Would cross-model or partial human evaluation be feasible for validation?
3. Reasoning-Type Analysis:
The paper shows category-wise results but lacks deeper diagnostics. Could the authors provide finer-grained error analyses or ablations—for example, separating failures due to OCR errors, visual reasoning, or logical inference?
4. Comparison with Existing Benchmarks:
It would be helpful to include a more direct experimental comparison or transfer evaluation with existing datasets (e.g., DocVQA, ChartQA, OCRBench). How does OCR-Reasoning specifically challenge models beyond these benchmarks?
5. Reinforcement Learning Methods:
The paper notes that RL-based methods perform poorly. Could the authors elaborate on how a better reward design for text-rich reasoning might look, or what specific factors caused these RL models to fail?
6. Dataset Accessibility and Reproducibility:
Please clarify the intended release details—license, format, annotation schema, and evaluation scripts. Ensuring full reproducibility will significantly strengthen the paper’s long-term impact.
7. Future Directions:
The authors mention possible improvements in reward design and dataset expansion. Could they outline a concrete roadmap for how OCR-Reasoning might evolve into a standardized benchmark suite for multimodal reasoning?
These clarifications and extensions could meaningfully strengthen the paper’s rigor, reproducibility, and long-term value to the community.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes OCR-Reasoning, a benchmark for evaluating MLLMs’ text-rich image reasoning, with 1,069 human-annotated examples (6 core reasoning abilities, 18 tasks) and annotations of both reasoning processes and answers (unlike existing benchmarks only annotating answers). Evaluations show top closed-source MLLM  doesn’t exceed 50% on it, while open-source ones perform worse.

### Strengths
1. Filling text-rich image reasoning evaluation gaps: Existing text-rich image benchmarks focus on text extraction but lack systematic reasoning assessment. OCR-Reasoning addresses this, measuring MLLMs’ reasoning in practical scenarios.
2. Sample design forcing reasoning: Few answers in its samples are directly extractable from OCR results; models must actively reason, avoiding reliance on text extraction to truly reflect their reasoning levels.
3. Comprehensive annotations for in-depth evaluation: Unlike benchmarks that only annotate final answers, OCR-Reasoning  labels both reasoning processes and answers, enabling holistic analysis of models’ problem-solving abilities

### Weaknesses
1. Limited dataset scale: Most of the data collection and annotation processes rely on manual work, and the high associated costs result in the dataset scale being only comparable to previous methods, failing to achieve larger-scale expansion

### Questions
1. OCR-Reasoning annotates both reasoning processes and final answers, while existing text-rich image benchmarks (e.g., DocVQA, OCRBench) mostly only annotate final answers and their samples’ answers are often directly extractable from OCR results. What are the specific core differences between OCR-Reasoning and these benchmarks in terms of sample design and annotation logic? What key role does this difference play in evaluating the true reasoning capabilities of MLLMs?
2. The paper states that existing reinforcement learning (RL) methods perform poorly on OCR-Reasoning, due to mismatched reward functions and a disconnect between training data and the benchmark’s scenarios. Does the study propose preliminary improvement directions (e.g., specific reward function design ideas, training data selection criteria)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work addresses the lack of systematic benchmarks for evaluating Multimodal Large Language Models (MLLMs) in text-rich image reasoning. It proposes OCR-Reasoning, a benchmark with 1,069 human-annotated examples covering 6 core reasoning abilities and 18 practical tasks, featuring both final answers and step-by-step reasoning processes. Extensive evaluations of LLMs, MLLMs, and document-oriented MLLMs show that no model achieves over 50% accuracy, with image input outperforming OCR text alone and CoT prompting benefiting most models. The core contributions include the first reasoning process-annotated benchmark for text-rich images, systematic model evaluation, and identification of key improvement directions.

### Strengths
1. OCR-Reasoning is the first benchmark to systematically assess reasoning processes in text-rich image scenarios, addressing a long-overlooked need.
2. The comprehensive evaluation includes multiple model categories and zero-shot settings, ensuring generalizable results.
3. Detailed error analysis and qualitative case studies deepen understanding of model limitations beyond accuracy metrics.

### Weaknesses
1. While the handwritten data in OCR-Reasoning provides valuable transcribed college-level STEM problems, it would be beneficial to consider incorporating more everyday real-world handwritten scenarios to further enhance the benchmark's coverage of diverse text-rich reasoning tasks commonly encountered in practice.

2. The paper presents an interesting observation that CoT prompting may have backfired on VL-Rethinker-7B, potentially due to conflicting built-in reflection mechanisms. It would strengthen this finding if the authors could provide additional ablation studies or experiments to further validate this hypothesis and better understand the underlying mechanisms.

3. The human validation for the LLM-as-Judge method demonstrates careful evaluation on DouBao-1.5-Vision-Pro. To further establish the robustness of this evaluation approach, it would be valuable to extend the validation across additional models and reasoning categories, which could help address potential concerns about judge bias and generalizability of the assessment methodology.

### Questions
The zero-shot evaluation effectively demonstrates out-of-the-box model capabilities. Have the authors explored or considered exploring few-shot prompting or fine-tuning scenarios on OCR-Reasoning? Could such experiments provide insights into whether models achieve substantial improvements with modest amounts of task-specific guidance?

### Soundness
4

### Presentation
4

### Contribution
4
