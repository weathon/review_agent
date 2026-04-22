# Beyond Classification Accuracy:  Neural-MedBench and the Need for Deeper Reasoning Benchmarks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Recent advances in vision-language models (VLMs) have achieved remarkable performance on standard medical benchmarks, yet their true clinical reasoning ability remains unclear. Existing datasets predominantly emphasize classification accuracy, creating an evaluation illusion in which models appear proficient while still failing at high-stakes diagnostic reasoning. We introduce Neural-MedBench, a compact yet reasoning-intensive benchmark specifically designed to probe the limits of multimodal clinical reasoning in neurology. Neural-MedBench integrates multi-sequence MRI scans, structured electronic health records, and clinical notes, and encompasses three core task families: differential diagnosis, lesion recognition, and rationale generation. To ensure reliable evaluation, we develop a hybrid scoring pipeline that combines LLM-based graders, clinician validation, and semantic similarity metrics. Through systematic evaluation of state-of-the-art VLMs, including GPT-4o, Claude-4, and MedGemma, we observe a sharp performance drop compared to conventional datasets. Error analysis shows that reasoning failures, rather than perceptual errors, dominate model shortcomings. Our findings highlight the necessity of a Two-Axis Evaluation Framework: breadth-oriented large datasets for statistical generalization, and depth-oriented, compact benchmarks such as Neural-MedBench for reasoning fidelity. We release Neural-MedBench at https://neuromedbench.github.io/ as an open and extensible diagnostic testbed, which guides the expansion of future benchmarks and enables rigorous yet cost-effective assessment of clinically trustworthy AI.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work provided Neural-MedBench, which is an expert-curated, neurology-focused benchmark that shifts evaluation from breadth to depth by testing multi-modal clinical reasoning, showing current VLMs chiefly fail for reasoning rather than perception despite strong shallow scores. However, there remain some unresolved issues, with single-judge reliance, opaque challenge validation, and unclear length normalization tempering the claims.

### Strengths
1. High reasoning density: a depth-oriented benchmark with greater clinical relevance than shallow QA-style responses.
2. The benchmark and items are curated by medical experts, which is more credible than LLM-generated or auto-converted content.
3. Annotations are not single-point labels but structured, narrative explanations that mimic clinical reasoning, yielding high clinical value.
4. Items are challenging and come with explicit difficulty tiers.

### Weaknesses
1. The paper should specify which baseline model(s) were used for the pre-release “challenge validation,” including the exact prompts and their performance.

2. Grading relies primarily on GPT-4o as the automatic judge; despite high correlation with clinicians, model-specific bias may persist. Please explain why a multi-model, anonymous voting scheme was not adopted.

3. “Normalized in length to control for spurious effects of input size on model performance”: length-related performance is itself part of model capability. Why normalize for this task, and how exactly was it implemented (e.g., truncation, abstractive summarization, padding to a fixed length, specific upper/lower thresholds, etc.)?

### Questions
See the weaknesses

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
This paper introduces Neural-MedBench, a compact yet reasoning-intensive benchmark designed to evaluate clinical reasoning capabilities of vision-language models (VLMs) in neurology. The authors propose a "Two-Axis Evaluation Framework" distinguishing between Breadth (statistical generalization on large datasets) and Depth (reasoning fidelity on complex cases). The benchmark comprises 120 expert-curated neurology cases with 200 tasks spanning differential diagnosis, lesion recognition, and rationale generation. Evaluation of state-of-the-art VLMs (GPT-4o, Claude-4, MedGemma) reveals significant performance drops compared to breadth-oriented benchmarks, with error analysis showing reasoning failures (51%) dominate over perceptual errors (27%).

### Strengths
1. The paper is well-written. 
2. The paper presents the Two-Axis (Breadth/Depth) evaluation framework provides a compelling perspective on AI evaluation that addresses a genuine gap in current medical AI benchmarking.
3. The rigorous multi-stage expert curation process involving senior neurologists and neuroradiologists, with consensus review, ensures benchmark quality and clinical validity.

### Weaknesses
1. The bechmark is small-scale benchmark compared with exists medical benchmark, such as OmniMedVQA, GMAIMMbench.
2. Some medical VLM method need to added to compared，such as, Huatuo-vision, Lingshu. While, GPT-4o maybe a outdate model, gpt5 is welcome.

### Questions
1. Prompt Sensitivity: Did you test the sensitivity of the evaluation prompts to variations in wording?
2. Medical Student Performance: What explains the substantially lower performance of medical students compared to existing large models in Table 2?
3. BERTScore Discrimination: The BERTScore differences in Table 2 appear minimal across models. Does this metric provide sufficient discriminative power?
4. Senior Physician Baseline: Given the relatively low performance of senior physicians on the benchmark, were all questions validated by multiple experts to ensure correctness and clinical appropriateness?

### Soundness
4

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
3

### Summary
This paper introduces Neural-MedBench, a focused and reasoning-intensive benchmark designed to evaluate the clinical reasoning capabilities of vision-language models (VLMs) in the neurology domain. This benchmark integrates multi-sequence MRI scans, structured electronic health records, and clinical notes, and encompasses three core task families: differential diagnosis, lesion recognition,
and rationale generation. A hybrid scoring pipeline is proposed for better evaluation.

### Strengths
A two-axis evaluation framework is proposed, which can be complementary assessments of both breadth (statistical generalization) and depth (reasoning fidelity). Neural-MedBench is the first neurology-focused benchmark explicitly designed to operationalize the Depth axis, comprising 120 multimodal, expert-curated diagnostic cases with 200 reasoning-intensive tasks. Experimental results show that current VLMs fail primarily at reasoning, despite strong performance on existing large-scale datasets. A systematic error analysis and a human performance baseline are also provided.

### Weaknesses
1. The definition of three difficulty levels is unclear. 
2. A limitation section should be added in main body.

### Questions
1. The details of grader validation and validation process are missing. 
2. Some important VLMs are missing, such as Gemini 2.5 pro and GPT-5.
3. What the different of Base VLMs and General VLMs? 
4. It is intersting to provide more discussion for the following case: Why the pass@1 acc of medical student is significant lower than VLMs? Why the pass@1 acc of medical student in Multi-round dialogue is significantly better than Direct diagnosis?

### Soundness
3

### Presentation
3

### Contribution
2
