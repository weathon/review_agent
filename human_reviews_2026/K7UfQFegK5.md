# FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Front-end engineering involves a complex workflow where engineers conceptualize designs, translate them into code, and iteratively refine the implementation. While recent benchmarks primarily focus on converting visual designs to code, we present FullFront, a benchmark designed to evaluate Multimodal Large Language Models (MLLMs) \textbf{across the full front-end development pipeline}. 
FullFront assesses three fundamental tasks that map directly to the front-end engineering pipeline: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase).
Unlike existing benchmarks that use either scraped websites with bloated code or oversimplified LLM-generated HTML, FullFront employs a novel, two-stage process to transform real-world webpages into clean, standardized HTML while maintaining diverse visual designs and avoiding copyright issues.
Extensive testing of state-of-the-art MLLMs reveals significant limitations in page perception, code generation (particularly for image handling and layout), and interaction implementation. Our results quantitatively demonstrate performance disparities across models and tasks, and highlight a substantial gap between current MLLM capabilities and human expert performance in front-end engineering.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Proposes a full-pipeline benchmark for front-end work with three tasks: Webpage Design, Perception QA, and Code Generation. Datasets span hundreds to thousands of items across these stages.

### Strengths
1. Evaluates the full front-end pipeline from concept, perception, and implementation (3 tasks, 8 sub-tasks).
2. Combines visual similarity with a Code Score to evaluate fine-grained elements.
3. Benchmarks a wide set of open-source and proprietary models and analyzes task-level differences.

### Weaknesses
Minor weaknesses:
1. Typos and labeling inconsistency.
2. Numerical inconsistencies in reported results and category counts.

Major issues:
1. Incomplete specification of the Code Score.
2. Same-family bias from using a model as the judge while evaluating related models.
3. Data-pipeline bias due to using certain models to construct the dataset.

### Questions
1. Typo: In Fig.6, "InterenVL3-78B" should be "InternVL3-78B" (Line 435); In the caption of Table 6, "Blacnk" and "Isonlation" (Line447-448). In Table 1, "DiINOv2" (Line272). "outperforme" in Line 107.
2. Unreasonable numbers: In Table 3, the GPT-5 just got 0.58 on Img of Gemini Visual Score (Line 307); The authors mentioned "15 categories", but claimed "twelve categories" in Figure 31.
3. Missing weight values: the actual weights of Code Score are missing. I am concerned about the reproducibility.
4. Model-as-judge bias: using Gemini 2.5 Flash as the visual judge while also evaluating Gemini models risks same-family bias.
5. Data source with bias: HTML-v1 built by GPT-4o and refined to HTML-v2 by Claude 3.7 Sonnet, then those models are evaluated. The evaluation metrics may favor Claude and GPT-4o.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes FullFront, a unified benchmark evaluating the full front-end engineering pipeline: Webpage Design, Webpage Perception QA, and Webpage Code Generation.

### Strengths
1. Compared to prior works, FullFront transforms real-world websites into clean, standardized HTML to avoid copyright issues.
2. Many models are comprehensively benchmarked on three subtasks.

### Weaknesses
1. While the authors claim they unify multiple components into one cohesive evaluation pipeline, the implementation and results look like three separate benchmarks to me; the analysis of the connection between different parts seems weak.
2. Webpage Design seems to benchmark text-to-image generation capability. This part is kinda less motivated, since why do we want such MLLM to generate a website in image form? What makes it necessary to ask them to generate an image instead of generating code + rendering?
3. Misleading sample size: It seems that a non-trivial amount of experiments is conducted on FullFront-mini, which only contains 10 webpage design data points and 50 webpage code generation data points. The selection process is not justified.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

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
This paper introduces FullFront, a benchmark that evaluates MLLMs across the full front-end workflow—design, perception, and code—using reconstructed real-page data and complementary visual/code metrics validated against human judgments. Experiments reveal substantial gaps from human performance, with persistent weaknesses in fine-grained perception, layout fidelity, image handling, and interaction implementation despite strong results from leading proprietary models. The work is timely and likely useful for the community, though its reliance on closed models and the potential metric bias should be weighed carefully.

### Strengths
+ The benchmark mirrors real front-end workflows rather than a single slice: it covers conceptualization (Webpage Design), perception (Webpage Perception QA), and implementation (Webpage Code Generation) with concrete task counts across eight subtasks. This end-to-end framing is rare and useful for diagnosing capability gaps.
+ The dataset is grounded in real webpages and reconstructed into standardized, copyright-safe HTML through a two-stage, MLLM-assisted pipeline, addressing common issues of bloated scraped code and oversimplified LLM HTML in prior corpora.
+ The evaluation combines visual and code-level metrics and validates them against human judgments with high Spearman correlations, supporting automated evaluation as a reasonable proxy for human preference.

### Weaknesses
- Several construction and scoring steps rely on proprietary models (e.g., GPT-4o/Claude in the pipeline; Gemini-based visual scoring), which can introduce system bias and limit strict reproducibility; data/code release is contingent on acceptance.
- The engineered “Code Score” aggregates DOM and style attributes with fixed design choices; even with strong human correlation, such choices may privilege particular implementation patterns and under-reward acceptable alternatives.
- The results clearly show large model–human gaps (e.g., <60% model accuracy vs. >95% human on perception), but the causal link between perception skill and code quality remains only lightly probed; deeper ablations could clarify what actually transfers across stages.

### Questions
1. How reproducible is the dataset and evaluation given reliance on closed models (GPT-4o/Claude for curation; Gemini for visual scoring)? Please quantify any bias toward these systems and provide an open, drop-in alternative or calibration protocol.


2. “Code Score” encodes specific structural/style weights. What sensitivity and failure analyses show that valid alternative implementations aren’t systematically penalized? Include counterexamples where humans deem outputs equivalent but the metric disagrees.

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
4

### Summary
FullFront is a new benchmark consisting of three types of tasks related to various aspects of the front-end development cycle, including: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase).

Webpage Design is an image generation task where the model needs to generate the webpage design based on textual descriptions of synthetic webpages. Webpage perception QA is a QA task based on both real and synthetic webpages, including 75 samples of multi-window QA tasks. The questions are generated by GPT-4o with manual inspection. The webpage code generation task takes visual design as input and expects code generation as output. This task includes image-to-code, text-to-code, interaction authoring, and code refinement. 

The authors did benchmarking of various closed and open models on these tasks.

### Strengths
- New resource for benchmarking MLLM capability on front-end engineering. 

- Relatively thorough coverage of different capabilities and model families. 

- Nice to include to human evaluation too.

### Weaknesses
- I'm not fully convinced by why we need an aggregate benchmark for front-end development. Some of the capabilities are quite distinct. For example, design (image) generation vs QA vs code generation. You have to use different models for the benchmarking because most models can't do image generation at all. Then what's the point of putting all of these tasks into one benchmark?

- I understand the authors put in effort to curate new data for many of the tasks in the benchmark. But I believe for most of the sub-tasks, there exist prior benchmarks that test the same capability. What's the unique contribution here apart from putting all the result tables into one paper?

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
