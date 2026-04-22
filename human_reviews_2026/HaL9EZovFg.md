# XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Omni-modal large language models (OLLMs) aim to unify audio, vision, and text understanding within a single framework. While existing benchmarks have advanced multimodal evaluation, it remains unclear whether OLLMs achieve modality-invariant reasoning or inherit modality-specific biases. We introduce \textbf{XModBench}, a large-scale tri-modal benchmark explicitly designed to measure cross-modal consistency. XModBench contains 60K multiple-choice questions across five task families and systematically covers all six cross-modality directions, enabling diagnosis of task competence, modality disparity, and directional imbalance. Experiments show that even the strongest model, Gemini 2.5 Pro, (i) struggles with spatial and temporal reasoning, achieving less than 60% accuracy, (ii) suffers from modality disparities, with performance dropping by over {20 points} on average when audio inputs replace text, and (iii) exhibits directional imbalance, with a {9-point gap} when using vision as context versus using text as context.
The findings suggest that OLLMs fall short of modality-invariant reasoning, and XModBench provides a fundamental diagnostic tool for evaluating and improving their overall cross-modal competence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents XModBench, a large-scale tri-modal benchmark for evaluating cross-modal consistency in omni-language models (OLLMs) that handle text, vision, and audio.
Unlike previous multimodal benchmarks, XModBench focuses on whether models maintain consistent reasoning across modalities.
It includes 60K multiple-choice questions spanning five task families and six modality directions, and introduces diagnostic metrics for task competence, modality disparity, and directional imbalance.
Experiments on leading OLLMs (e.g., Gemini 2.5 Pro, Qwen2.5-Omni) show that current models still lack modality-invariant reasoning, especially in spatial, temporal, and audio-related tasks.

### Strengths
Strengths

Novel Focus: Targets an important but underexplored problem — evaluating cross-modal consistency rather than just multimodal performance.

Comprehensive Benchmark Design: Covers six modality directions and five task families, providing a balanced and systematic tri-modal evaluation.

Insightful Diagnostics: Introduces clear metrics (modality disparity and directional imbalance) that reveal hidden biases and asymmetries in current OLLMs.

### Weaknesses
This paper compares model performance differences across modalities for the same question. However, it does not discuss whether such differences are caused by information loss during modality conversion — for example, when a video or audio question is converted into text, the textual description cannot fully capture the visual or auditory content.

The paper does not explore whether specific training strategies or data construction methods could help mitigate these shortcomings in cross-modal consistency.

### Questions
None

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
4

### Summary
The paper introduces XModBench, a large-scale, tri-modal (audio, vision, text) benchmark designed to evaluate omni-modal large language models (OLLMs). The core objective is to move beyond task-specific accuracy and measure cross-modal consistency—the ability of a model to produce consistent answers when the same semantic content is presented in different modalities. XModBench comprises 60,828 multiple-choice questions systematically covering five task families (perception, spatial reasoning, temporal reasoning, linguistic understanding, and external knowledge) and all six possible cross-modal directions between the three modalities. The authors use this benchmark to evaluate a range of OLLMs, including the Gemini series and several open-source models. The key findings demonstrate that even state-of-the-art models like Gemini 2.5 Pro lack true modality-invariant reasoning, exhibiting significant performance drops on spatial/temporal tasks, major disparities when inputs are switched (e.g., text vs. audio), and directional imbalances (e.g., V->T vs. T->V).

### Strengths
1. The paper tackles a crucial question: are OLLMs truly modality-invariant? It moves evaluation beyond simple accuracy on multimodal tasks and proposes a novel, principled method for measuring cross-modal consistency. The design, which permutes modalities for the same semantic question, is the paper's core strength.



2. The benchmark is comprehensive. It contains over 60K questions , covers 5 diverse task families (from perception to external knowledge) , and 17 subtasks. This breadth ensures that the findings are not artifacts of a single domain.



3. High-Quality Curation. The authors detail a rigorous data curation and verification process. The explicit use of "human in-the-loop verification" and multiple rounds of testing by annotators addresses major concerns about the quality and ambiguity of web-sourced or generated data .

4. Actionable Diagnostics. The paper doesn't just rank models. It provides specific, interpretable diagnostic metrics—modality disparity and directional imbalance —that allow researchers to pinpoint where and how their models are failing. The failure case analysis in Section 4.5 and Figure 6 reinforces this with qualitative examples

### Weaknesses
1. Accessibility and Cost: The benchmark's primary strength—its scale—is also a potential weakness for adoption. Evaluating a model on 60,828 question-answer pairs, many of which involve multiple modalities, appears to be a computationally expensive process. The paper does not mention the availability of a smaller, standardized "lite" subset for researchers with limited compute. Furthermore, no information is provided on the practical costs of evaluation, such as total token usage for API-based models (like the Gemini series) or GPU hours for open-source models. This omission could be a significant barrier to widespread adoption and reproducibility.

2. Limited Analysis of SOTA Performance: The paper's results clearly establish Gemini 2.5 Pro as the top-performing model, yet one that still has significant flaws. However, the analysis is largely limited to reporting these scores and failures. The paper would be strengthened by a deeper discussion hypothesizing why this model performs so much better on average than its open-source counterparts. Is it its training data, a specific architectural choice, or better-aligned encoders? A more in-depth analysis of the causes of SOTA performance (and its limitations) would be more impactful than just documenting the performance itself. Could be helpful if we could analyze how it was trained even with a guess.

### Questions
Given the impressive scale of the benchmark, have the authors considered releasing a standardized "lite" subset? A smaller, balanced subset would significantly lower the barrier to entry, allowing for more rapid experimentation and broader adoption by the research community.

Could the authors provide an estimation of the computational cost to run the full XModBench evaluation? Specifically, what is the approximate token usage (input and output) for evaluating an API-based model like Gemini, and what are the estimated GPU-hours for an open-source model?

The performance of Gemini 2.5 Pro is a key data point. While its failures on spatial/temporal tasks are clear , its overall superiority is also evident. Do the authors have any insights or hypotheses as to why this model demonstrates relatively better cross-modal consistency and overall competence compared to the other models tested?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces XModBench, a large-scale tri-modal benchmark (audio, vision, text) designed to evaluate cross-modal consistency in omni-modal large language models (OLLMs). The dataset contains 60K multiple-choice QA pairs rendered in six modality directions (e.g., audio→text, image→audio), covering five task categories: perception, spatial reasoning, temporal reasoning, linguistic understanding, and external knowledge. The authors benchmark several frontier models (e.g., Gemini 2.5 Pro) and report three key findings: (i) OLLMs still struggle with spatial/temporal reasoning, (ii) performance drops significantly when audio replaces text, and (iii) models show strong directional imbalance (e.g., text→vision vs. vision→text). The benchmark aims to serve as a diagnostic tool for measuring modality-invariant reasoning.

### Strengths
The paper explicitly targets cross-modal consistency, a dimension often ignored in existing multimodal benchmarks.

Each question instance is rendered across all six modality mappings, enabling controlled comparison and directional analysis.

Large scale and broad coverage.
The dataset includes >60K QA samples spanning 17 subtasks, with balanced modality construction.

Relevance to current model trends.
As many new models claim “omni-modality,” this benchmark fills a timely evaluation gap.

### Weaknesses
Table 2 is overloaded and hard to interpret.
The key conclusions (e.g., text→image > audio→text; Gemini has lowest variance) are meaningful, but the table is dense, lacks focused analysis, and could be split into smaller tables aligned with each main claim.

Interesting modality-swap results but no deeper investigation.
The paper observes asymmetric performance (e.g., vision→text vs. text→vision) but does not analyze why. For example:
– Do any models use interleaved multimodal training data?
– Do models with such data show smaller swap gaps?

Dataset quality control is unclear.
The benchmark claims 60K samples but does not report human validation, error rate, or annotation quality checks.

No discussion of answer-option bias.
Some modalities may allow shortcut guessing (e.g., lexical cues in text choices). There is no “noise-input” baseline to rule this out (e.g., Gemini with shuffled / blank modality input).

Lack of analysis for <25% performance cases.
Several settings score worse than random guessing (25% for 4-choice MCQ), but the paper does not explain whether this is due to instruction following, noisy inputs, or poor distractor design.

### Questions
Dataset quality
Have you conducted human verification on a subset of the 60K samples? If so, what is the estimated annotation error rate?

Distractor bias
Can models guess correct answers without context? Please provide “no-input” or “noise-input” baselines to quantify answer-option bias.

Modality swap analysis
Do any evaluated models train on interleaved multimodal corpora (e.g., narrated video, audiocaps)? If yes, do they exhibit smaller directional gaps?

Table 2 clarity
Would you consider splitting Table 2 into multiple focused tables (e.g., task competence, disparity, imbalance) to improve readability?

Below-random performance
For conditions where models score <25%, what is the failure mode? Instruction refusal? Systematic misalignment? Poor distractor construction?

Benchmark extensibility
Do you plan to release tools for adding new modality pairs (e.g., text↔3D, audio↔video)?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces XModBench, a large-scale tri-modal benchmark specifically designed to measure cross-modal consistency in Omni-modal Large Language Models (OLLMs) by systematically covering six cross-modality directions for audio, vision, and text. XModBench comprises over 60,000 multiple-choice questions across five task families and 17 subtasks, enabling diagnostic assessment of task competence, modality disparity, and directional imbalance.

### Strengths
- Comprehensive Diagnostic Scope: XModBench provides a large-scale, systematically balanced tri-modal QA benchmark, covering all six modality permutations (audio, vision, text) for both the context and candidate answers. The benchmark covers five diverse task families (perception, spatial, temporal, linguistic, external knowledge), each with multiple subtasks.

- Detailed Empirical Analysis: The authors conduct a detailed empirical analysis of cutting-edge OLLMs, including a performance breakdown by task and modality configuration (Table 2). This analysis effectively identifies the significant lack of capability or competence in current OLLMs within the audio domain.

### Weaknesses
- XModBench primarily focuses on isolated cross-modal alignment (e.g., T→V, V→T) and fails to cover true mixed tri-modal capabilities (e.g., Image+Vision+Audio→Text/Image). This combined modality reasoning is arguably the critical differentiator separating OLLMs from Multimodal Large Language Models (MLLMs) and specialized speech models.

-  The data curation shows an over-reliance on GPT-5 as the primary question generation tool. However, the quality assurance (QA) or filtering process for this synthetic data is not clearly elaborated. This risks labeling XModBench as a 'silver' dataset rather than a 'gold' standard, where prioritizing data quality over sheer quantity is paramount.

-  Contextualization against MLLMs: The analysis lacks comparison against the performance of MLLMs focused on traditional ASR or image-to-text (I2T) tasks, such as Qwen-VL or Intern-VL. It remains unclear whether OLLMs maintain a performance advantage in these specific subdomains when compared to these more focused MLLMs.

### Questions
same as weakness

### Soundness
2

### Presentation
3

### Contribution
3
