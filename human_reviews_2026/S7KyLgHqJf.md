# M3CoTBench: Benchmark Chain-of-Thought of MLLMs in Medical Image Understanding

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models by encouraging step-by-step intermediate reasoning, and recent advances have extended this paradigm to Multimodal Large Language Models (MLLMs). In the medical domain, where diagnostic decisions depend on nuanced visual cues and sequential reasoning, CoT aligns naturally with clinical thinking processes. However, current benchmarks for medical image understanding generally focus on the final answer while ignoring the reasoning path. An opaque process lacks reliable bases for judgment, making it difficult to assist doctors in diagnosis. 
To address this gap, we introduce a new M3CoTBench benchmark specifically designed to evaluate the correctness, efficiency, impact, and consistency of CoT reasoning in medical image understanding. M3CoTBench features  (1) a diverse, multi-level difficulty dataset covering 24 examination types, (2) 13 varying-difficulty tasks,  (3) a suite of CoT-specific evaluation metrics (correctness, efficiency, impact, and consistency) tailored to clinical reasoning,  and (4) a performance analysis of multiple MLLMs. M3CoTBench systematically evaluates CoT reasoning across diverse medical imaging tasks, revealing current limitations of MLLMs in generating reliable and clinically interpretable reasoning, and aims to foster the development of transparent, trustworthy, and diagnostically accurate AI systems for healthcare.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces M3CoTBench, a benchmark for evaluating Chain-of-Thought reasoning in medical image understanding MLLMs. The dataset contains 1,079 image-QA pairs across 24 medical imaging modalities with expert-annotated reasoning steps and four evaluation dimensions.

### Strengths
- Addresses Critical Gap: First comprehensive benchmark for CoT reasoning in medical imaging - important for clinical AI transparency and trust.
- High-Quality Curation:
  - Diverse coverage: 24 modalities from 55 public datasets
  - Rigorous annotation: Multi-stage validation with medical experts
  - Clinical alignment: 4-step reasoning framework mirrors diagnostic workflows
- Novel Evaluation Framework: Four dimensions (correctness, efficiency, impact, consistency) provide comprehensive CoT assessment beyond accuracy.
- Extensive Evaluation: Tests 13 MLLMs including general-purpose, reasoning-focused, and medical-specific models with interesting findings about CoT effectiveness.

### Weaknesses
Methodological Concerns: 
- The dataset comprises only 1,079 images, relatively small compared to other medical reasoning benchmarks (e.g., OmniMedVQA with 118K+ images).
- Potential Bias: Although reasoning steps undergo expert validation and revision, their initial generation by GPT-4o may introduce biases inherent to its reasoning style, which might persist despite subsequent human refinement.
- Evaluation Circularity: The study uses GPT-4o both to generate reasoning chains and to evaluate them against GPT-4o-based gold standards, creating a circular evaluation loop.
- The paper does not specify which MLLMs were used for flagging potentially incorrect reasoning steps.

Evaluation Concerns:
- Despite the multi-expert validation process, no inter-annotator agreement scores are reported.
- Confidence intervals and significance tests for performance differences are not provided.
- The number of experts involved and procedures for resolving disagreements are not described.


Conceptual Issue:
- Counterintuitive Findings: The universally negative impact of reasoning across models raises questions about the benchmark’s design, the quality of the Chain-of-Thought implementation, and the validity of using GPT-4o as the evaluation reference.

### Questions
- How do you address evaluation circularity when using GPT-4o to assess GPT-4o reasoning?
- What are the inter-annotator agreement scores during expert validation?
- Why do most models show negative reasoning impact, is this a CoT implementation issue or benchmark design problem?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces M3CoTBench, a benchmark designed to evaluate chain-of-thought (CoT) reasoning in multimodal large language models (MLLMs) for medical image understanding. M3CoTBench comprises a diverse dataset spanning 24 imaging modalities (X-rays, MRIs, endoscopy, etc.) and 13 task types, ranging from low-level tasks like image quality assessment to high-level clinical reasoning such as diagnosis and treatment planning.

### Strengths
1.  This paper introduces M3CoTBench, encompassing 24 imaging modalities to evaluate MLLMs' understanding capabilities across diverse medical imaging contexts.
2. The benchmark introduces tailored metrics to assess reasoning quality across four dimensions: correctness of each reasoning step, efficiency cost, impact on final answer accuracy, and logical consistency—providing a more nuanced evaluation beyond traditional accuracy measures.

### Weaknesses
1. M3CoTBench spans 24 modalities and 13 task types, but contains only 1,079 image-based QA pairs. Given this broad coverage, does each category have sufficient samples? The paper does not appear to provide per-category statistics.
2. The benchmark’s dataset, while diverse, is relatively small (only 1079 Q&A pairs) compared to other medical VQA datasets, which may limit the statistical breadth of evaluation.

### Questions
1. LLaVA-CoT exhibits relatively strong performance compared to Gemini 2.5 Pro. The authors attribute this to its architecture and training process, which emphasize structured reasoning chains while minimizing irrelevant or misleading steps. However, given that Gemini 2.5 Pro also incorporates thinking capabilities, what accounts for this performance difference?

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
5

### Summary
This paper proposes M3CoTBench, a benchmark designed to evaluate Chain-of-Thought (CoT) reasoning in multimodal large language models (MLLMs) for medical image understanding. It constructs a diverse dataset of medical images with clinically grounded reasoning annotations and proposes multidimensional evaluation metrics covering correctness, efficiency, impact, and consistency. Experiments on both open- and closed-source models show that while CoT improves interpretability, it does not always enhance diagnostic accuracy, revealing limitations in current MLLMs’ clinical reasoning and highlighting the need for more trustworthy and efficient medical CoT systems.

### Strengths
1. The paper tackles an emerging yet underexplored topic: evaluating Chain-of-Thought reasoning in medical multimodal LLMs, which is both timely and relevant to advancing trustworthy medical AI.
2. The benchmark is validated on a broad range of both open- and closed-source MLLMs, providing a well-rounded comparison that highlights current model limitations and practical challenges in clinical reasoning.

### Weaknesses
1. The definition of CoT in the medical area is unclear. Although the paper claims that its Chain-of-Thought (CoT) formulation “mirrors clinicians’ cognitive workflow”, the reasoning template shown in the Appendix appears overly simplified. It typically only has four steps: examination type -> key features -> key conclusion -> additional analysis. It is unclear why this sequence represents a gold standard reasoning path in clinical diagnosis. Is it based on any references, such as guidelines in medicine?
2. The justification for diverse reference reasoning paths is insufficient. The paper mentions that “multiple valid reference reasoning paths may exist” and evaluates by matching the generated path to the most similar reference. While this makes sense conceptually, it is unclear how the annotation process ensures both diversity and correctness of reference reasoning paths. In the medical domain, it remains questionable whether clinicians indeed exhibit substantially diverse CoTs. If so, where does this diversity arise? Is it in identifying different key features (Step 2) or in drawing different key conclusions (Step 3)?
3. The Reasoning Impact evaluation simply measures the performance difference between models with and without CoT, which seems redundant. This metric does not provide new insight into reasoning quality.
4. The reasoning progression across CoT steps is weak. From the provided CoT examples, the reasoning flow among steps is not clearly causal or hierarchical. Steps 3 and 4 appear to be direct deductions from Step 2, while Step 1 (examination type) is largely independent of the reasoning process itself. As a result, it is difficult to claim that the sequence truly reflects a step-by-step reasoning chain rather than a loosely connected checklist.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
