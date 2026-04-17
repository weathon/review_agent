# PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Recent advances in Large Language Models (LLMs) have enabled impressive capabilities across diverse reasoning and generation tasks. However, their ability to understand and operate on real-world engineering problems, such as Printed Circuit Board (PCB) placement and routing, remains underexplored due to the lack of standardized benchmarks and high-fidelity datasets. To address this gap, we introduce PCB-Bench, the first comprehensive benchmark designed to systematically evaluate LLMs in the context of PCB design. PCB-Bench spans three complementary task settings: (1) text-based reasoning with approximately 3,700 expert-annotated instances, consisting of over 1,800 question-answer pairs and their corresponding choice question versions, covering component placement, routing strategies, and design rule compliance; (2) multimodal image-text reasoning with approximately 500 problems requiring joint interpretation of PCB visuals and technical specifications, including component identification, function recognition, and visual trace reasoning; (3) real-world design comprehension using over 170 complete PCB projects with schematics, placement files, and design documentation. We design structured evaluation protocols to assess both generative and discriminative capabilities, and conduct extensive comparisons across state-of-the-art LLMs. Our results reveal substantial gaps in current models’ ability to reason over spatial placements, follow domain-specific constraints, and interpret professional engineering artifacts. PCB-Bench establishes a foundational resource for advancing research toward more capable engineering AI, with implications extending beyond PCB design to broader structured reasoning domains. Data and code are available at https://github.com/digailab/PCB-Bench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PCB-Bench, a novel and comprehensive benchmark designed to evaluate the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) on real-world Printed Circuit Board (PCB) design tasks. Authors evaluate several strong models on PCB-Bench.

### Strengths
1. The multimodal PCB task is an important and meaningful direction. PCB-Bench is the first benchmark designed to evaluate MLLMs’ capabilities in PCB design.

2. The benchmark’s three-task structure is comprehensive and well thought out.

3. The paper is exceptionally well-written and clearly presented.

### Weaknesses
1. The evaluation is somewhat limited. Only a few models are tested. It would be beneficial to include more models, such as the Intern-VL series and models of different sizes, to provide a more thorough understanding of the benchmark’s effectiveness.

2. In the benchmark construction pipeline (Figure 4), "Commercial MLLMs" are listed as a "Knowledge Source." Could the authors elaborate on how these models were used in this process?

3. Some related works are missing. For example, EEE-Bench [1] and Circuit [2] are relevant benchmarks related to circuit design and understanding, and they should be discussed in related works.

4. The performance of most models on the benchmark appears relatively high, which raises questions about its difficulty and suitability for driving future research.

References:
[1] EEE-Bench: A Comprehensive Multimodal Electrical and Electronics Engineering Benchmark. CVPR 2025.

[2] Circuit: A Benchmark for Circuit Interpretation and Reasoning Capabilities of LLMs.

### Questions
How the answers are parsed from model responses?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a benchmark containing three types of datasets: text-to-text QA and CQ, Image-and-Text QA and CQ, and a real-world PCB design repository. Based on this, the authors execute three corresponding tasks and benchmark various LLM backbones.

### Strengths
- The collection of the PCB datasets is important and meaningful, which is a basic reason I am leaning towards giving a positive rating.
- The authors give a clear comparison between current benchmarks, which I believe PCB-BENCH is competitive and advanced.
- The dataset is released. Though it is incomplete, I am convinced of the comprehensive quality.

### Weaknesses
- The introduction of how to construct the datasets requires a more detailed clarification.
- The benchmark could benefit from providing statistical results of the datasets, such as how many instances are in each class/topic/difficulty.
- The visualizations in Figure 1 really help with understanding, but some figures are vague, and I cannot see any information on them. It would be beneficial for the authors to use clearer or even vector graphs.
- The instances in task 3 are directly collected from OSHWHub, which makes this paper weak in workload.

### Questions
- The authors claim that domain experts write the questions. I just wonder what kind of domain experts they are, like academics or industry professionals.
- It seems that lots of LLMs achieve 100% accuracy on task 1, routing. Maybe it is too simple for these SOTA LLMs to do this job, right?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents PCB-Bench, a comprehensive benchmark for evaluating the reasoning, multimodal understanding, and real-world engineering capabilities of large language models (LLMs) and multimodal LLMs (MLLMs) for printed circuit board (PCB) placement and routing.

PCB-Bench has three complementary task settings and integrates three modalities: textual reasoning, image-text reasoning, and real PCB design comprehension. It includes over 3,700 text QA instances, 500 multimodal tasks, and 170 complete design projects with schematic diagrams and placement files.

The benchmark aims to bridge the current gap between foundation models' open-domain reasoning and the highly structured, constraint-driven nature of electronic design automation (EDA).
 Results from multiple state-of-the-art models (GPT-4o, GPT-5, Gemini-2.5, Claude-Opus-4.1, Qwen-VL-Max, etc.) show that while these models exhibit some knowledge of PCB principles, they struggle significantly in multimodal reasoning, domain rule adherence, and real-world design understanding.
 
PCB-Bench fills an important gap by providing standardized tasks, expert-curated questions, and reproducible metrics for EDA-oriented foundation-model assessment.

### Strengths
- Novel Domain and Multimodal Integration
PCB-Bench is the first dataset to integrate textual, visual, and structured PCB design data, covering the entire reasoning chain, from physical layout interpretation to rule-based design explanations.  This bridges the gap between EDA and current MLLM benchmarks.

- Expertly curated and Content.
The QA and CQ datasets are based on validated engineering principles (e.g., impedance control, EMI, high-speed routing).  The inclusion of over 25 subtopics across macro and micro-level placement ensures depth and realism.

-Comprehensive Model Evaluation
The authors benchmarked eight major LLM/MLLM systems under a unified zero-shot protocol, revealing clear differences in factual reasoning and semantic generation quality.

- High Practical Impact.
The paper connects foundation models to PCB design automation, a real-world domain where LLMs could help with debugging, constraint checking, and educational tools.

### Weaknesses
- Limited Novelty in Dataset Construction Approach
Although the domain is new, the methodology (expert-curated QA pairs + image-text matching) follows established benchmark pipelines. The contribution lies in domain adaptation rather than methodological innovation.

- Missing fine-grained statistical validation.
There is no explicit agreement among annotators or confidence interval analysis. To improve the dataset's credibility, a quantitative quality measure could be used beyond relying on expert curation.

- Evaluating Diversity
Only foundation models trained on general data were tested, and no domain-specific baselines (such as fine-tuned models or SPICE-guided reasoning agents) were included. This limits one's ability to distinguish between reasoning limitations and domain unfamiliarity.

- Lack of Complexity or Difficulty in stratification.
While "easy" and "hard" labels exist, difficulty calibration is not quantified (for example, through expert scoring or performance correlation).  Future releases may benefit from difficulty metadata.

- Fairness in Model Comparison.
The paper does not specify whether all models were evaluated with identical input resolutions, context lengths, or prompt templates. This may complicate the interpretation of differences between GPT-5, Gemini, and Qwen-VL.

### Questions
- Annotation Process and Quality Assurance:
How many domain experts contributed to the curation process, and how were disagreements resolved? Was inter-annotator agreement assessed, or were examples reviewed by several experts?

-Difficulty Calibration: Are "easy/hard" tags assigned objectively (e.g., the number of design rules involved) or subjectively by annotators?Could the authors provide a question breakdown by subtopic (e.g., EMI, differential pair routing, power delivery)?

-Representation Bias
The dataset depends mainly on open-source OSHWHub designs. Do these projects overrepresent specific device categories (such as USB hubs)? Could this bias affect generalization to industrial boards with more dense routing?

- Have the authors considered testing domain-specific systems(such as RL-based tuned models or retrieval-augmented models)? Will combining PCB-Bench with simulation-based feedback loops (such as constraint violation detection) give more realistic metrics?

### Soundness
3

### Presentation
3

### Contribution
4
