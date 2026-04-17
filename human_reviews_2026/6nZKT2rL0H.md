# OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Spatial reasoning is a key aspect of cognitive psychology and remains a bottleneck for current vision-language models (VLMs). While extensive research has aimed to evaluate or improve VLMs' understanding of basic spatial relations, such as distinguishing left from right, near from far, and object counting, these tasks cover only the most elementary layer of spatial reasoning and are largely approaching saturation in the latest reasoning models. In this work, we introduce OmniSpatial, a comprehensive and challenging benchmark for spatial reasoning, grounded in cognitive psychology. OmniSpatial covers four major categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, with 50 fine-grained subcategories. Through careful manual annotation, we construct over 8.4K question-answer pairs. Extensive experiments show that both open- and closed-source VLMs exhibit significant limitations in comprehensive spatial reasoning. We also explore two strategies—PointGraph (explicit scene graph cues) and SpatialCoT (novel-view chain-of-thought)—to bolster spatial reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OmniSpatial, a comprehensive and challenging benchmark designed to evaluate the spatial reasoning of VLMs. Grounded in cognitive psychology, OmniSpatial features over 8.4K manually curated question-answer pairs across four key categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking. Experiments show that even SOTA VLMs struggle significantly, performing far below human accuracy. To bridge this gap, authors propose two novel strategies PointGraph and SpatialCoT, which leverage structured scene graphs and multi-view synthesis to improve the model’s reasoning capabilities.

### Strengths
This paper’s most significant contribution is exposing that top AI models fail at complex spatial reasoning where humans excel, clearly defining a crucial and challenging direction for future VLM research.
It presents the highly original concept of SpatialCoT, which enhances reasoning by simulating mental imagery. This creative fusion of 3D novel-view synthesis with chain-of-thought prompting represents a significant conceptual advance for tackling view-dependent and perspective-taking tasks.
The work is distinguished by its quality, evident in the meticulous manual creation of its 8.4K question-answer pairs, which achieved a high inter-annotator agreement and transparent evaluation across a wide spectrum of leading AI models.

### Weaknesses
The paper demonstrates that models fail on complex tasks but does not offer a deep analysis of the reasons. Without a breakdown of specific error types, the work provides limited actionable guidance for researchers to develop targeted architectural improvements.

### Questions
Given the performance gap between frontier models and humans, it's important to consider whether current methods can help VLMs catch up. If not, what future research or scaling approaches could bridge this gap?

### Soundness
3

### Presentation
3

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
The paper proposes a new Visual Question Answering (VQA) benchmark, OmniSpatial, designed to comprehensively evaluate spatial reasoning capabilities. The benchmark highlights a broad and in-depth coverage of spatial relation tasks. The authors carefully design four key dimensions of spatial reasoning to be evaluated: perspective-taking, spatial interaction, dynamic reasoning, and compositional understanding. These four categories present significant challenges and substantially advance the complexity of spatial evaluation for current vision-language models (VLMs).
The proposed benchmark provides a comprehensive evaluation across diverse conditions, supported by a large collection of curated web images. The paper conducts experiments across multiple model series and sizes, including reasoning, closed-source, and open-source VLMs, while also providing a human baseline for comparison.
In addition, the authors investigate two approaches aimed at improving VLM spatial understanding on this benchmark, PointGraph and SpatialCoT. Both  yield consistent improvements across different VLMs.

### Strengths
- The proposed benchmark introduces a new and challenging evaluation setting that explores aspects of spatial reasoning rarely addressed in previous datasets. It is notably more complex and comprehensive than prior benchmarks.
- The question annotations involve a human-in-the-loop process to ensure clarity, answer uniqueness, and the resolution of ambiguous spatial references.
- The evaluation includes a wide range of VLMs—covering reasoning-focused, open-source, closed-source, and human baselines—demonstrating the benchmark’s broad coverage, thorough experimental setup, and comprehensive comparisons across models.
- Results demonstrate the significant shortcomings of VLMs across different type of spatial relation
- The paper also introduces two promising approaches, PointGraph (which incorporates an explicit scene graph as input) and SpatialCoT (which generates multi-view points from a given image to provide diverse spatial perspectives). These methods consistently improve model performance across different VLMs.
- Paper shows that fine-tuning models with this dataset shows potential for transferability to other VLM benchmarks.
- The paper is well-written and includes clear illustrations that help the audience understand the proposed spatial relation benchmark and its evaluation scope.

### Weaknesses
- There is no qualitative analysis of failure cases. Investigating these failures would strengthen the paper further. Providing a few examples and categorizing the errors could help reveal which aspects of reasoning need improvement—such as perception, logical reasoning, or consistency.
- The paper only demonstrates the effectiveness of SpatialCoT on the perspective-taking task. How does this approach affect performance on other task types? This might raise some concern that it make model perform worth in other tasks that does not require perspetive taking.
- Minor issue, Table 4 is never mentioned in discussion.

### Questions
- In the question generation process, is a fixed template used, or are LLM involved? If the process relies on template-based questions, would it be possible to incorporate LLMs to increase the diversity of question types? If it already use LLM, the paper should expicitly said so.

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
This paper presents OmniSpatial, a new and comprehensive benchmark aimed at evaluating higher-level spatial reasoning in VLMs beyond basic left–right or counting tasks. It provides 8.4K human-curated QA pairs across four categories—dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking—covering 50 task types. Evaluating 36 models shows that state-of-the-art VLMs achieve only 56% accuracy, far below human performance, with notable weaknesses in geometric reasoning and non-egocentric perspective shifts. The paper also introduces PointGraph and SpatialCoT as two strategies to improve spatial reasoning, both yielding modest gains.

### Strengths
- The paper is well written and easy to understand.
- The dataset construction is solid and carefully annotated by humans.
- The evaluation is comprehensive.

### Weaknesses
**Training Data Leakage Concern**
- While the dataset is manually curated, some sources (e.g., web images, exam-style questions) may overlap with model pretraining corpora. A clearer discussion on leakage mitigation, measurement, and dataset decontamination would strengthen the benchmark’s credibility.

**Compute Cost of SpatialCoT**
- The proposed SpatialCoT relies on multi-view synthesis, which appears computationally expensive. A discussion of its runtime, resource requirements, and potential lightweight or more practical alternatives would improve the clarity of its applicability.


**Lack of Discussion on Related Works**
-  I have seen prior works that also incorporate structured spatial information through text-based scene representations (e.g.,[1]). The PointGraph idea seems to be closely related to this one. It would be appropriate to acknowledge and discuss such related methods when introducing PointGraph in Sec. 3.3.1 to better position the contribution.

**Missing Error Bars in Reporting Results**
The main table does not present confidence intervals, variance, or statistical testing. As this is a benchmark paper, stronger evidence of robustness and significance is needed. Reporting standard deviations or significance tests can better support the claims and ensure results are reliable.

Overall, I believe the paper could be a good contribution to the community, and I would be happy to reconsider the score if the above concerns are satisfactorily addressed.

### Questions
- To what extent can models answer correctly without looking at the images? Since most questions are binary or 4-way multiple choice, some may be solvable from textual priors alone (e.g., Fig. 3: “I am entering a highway, I will encounter a ‘Give Way’ sign”). Have the authors evaluated a text-only baseline to isolate true visual reasoning?

- How is PointGraph different from existing methods like [1]?

- Can the authors provide an estimated compute overhead of SpatialCoT and discuss practicality for deployment?

- How do the authors assess or mitigate potential data leakage, especially for web- or exam-derived content that may exist in model training corpora? Is there any decontamination or measurement of overlap?

[1] Wang et al., Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models, NeurIPS 2024

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
2

### Summary
This paper introduces OmniSpatial, a large-scale benchmark designed to evaluate comprehensive spatial reasoning in vision-language models (VLMs). It organizes tasks into four key categories, including dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, covering 50 subtypes and 8.4K manually curated QA pairs. The benchmark integrates multiple data sources (web, cognitive tests, driving exams, and prior embodied datasets) with high annotation consistency (Krippendorff’s α = 0.84).

The authors further propose two methods to enhance VLM spatial reasoning:
1. PointGraph – providing explicit scene graphs for spatial structure.
2. SpatialCoT – enabling multi-view reasoning using novel-view synthesis (InstantMesh).

They benchmark 36 VLMs (GPT-4.1, Gemini-2.5, Qwen-VL, InternVL, etc.) and show that while leading reasoning models (e.g., o3, Gemini-2.5-pro) achieve ≈56% accuracy, human performance reaches 92%. Fine-tuning on OmniSpatial improves performance (+7.8 points) and transfers modestly to other spatial benchmarks (e.g., VSI-Bench +2 points)

### Strengths
1. Focused and Systematic Scope
The paper maintains a clear focus on spatial reasoning, defining it precisely, covering its cognitive dimensions, and avoiding unnecessary general multimodal extensions. This conceptual focus makes OmniSpatial a coherent and practically usable benchmark.

2. Rigorous Manual Curation:
- The dataset is human-annotated, multi-sourced, and cross-validated with strong inter-annotator agreement, addressing common weaknesses of synthetic or template-based datasets.

### Weaknesses
1. Lack of Deep Analysis or Failure Studies
The paper could benefit from qualitative examples showing why models fail (e.g., depth reasoning errors, frame-of-reference confusion, or temporal misalignment)

2. Marginal Quantitative Gains
The improvements from PointGraph and SpatialCoT are modest (≈1–2 points per dimension), raising questions about their practical impact.

### Questions
1. How do PointGraph and SpatialCoT interact? Are their improvements additive or overlapping?
2. Can the authors provide qualitative examples illustrating typical model errors (e.g., misinterpreting object orientation, inconsistent frame of reference)?

### Soundness
3

### Presentation
3

### Contribution
3
