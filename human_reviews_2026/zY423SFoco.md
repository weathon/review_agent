# TREAT: A Code LLMs Trustworthiness / Reliability Evaluation and Testing Framework

- Decision: Reject
- Scores: 4, 6, 2, 2, 6

## Abstract
Large foundation models are fundamentally transforming the software engineering landscape, demonstrating exceptional capabilities across diverse tasks such as code generation, debugging, and testing. Despite this rapid progress, a significant gap remains in how to comprehensively evaluate these models' trustworthiness in real-world software engineering scenarios. Existing benchmarks suffer from limited task scope and fail to incorporate critical evaluation aspects such as the robustness and reliability of models. To bridge this gap, we present an evaluation framework called TREAT (Code LLMs Trustworthiness / Reliability Evaluation And Testing) that provides a holistic assessment of model performance in code intelligence tasks. Our evaluation framework addresses key limitations in existing approaches with four main improvements: (1) Multi-Task Holistic Evaluation that spans diverse software engineering activities rather than limited coding tasks; (2) Multi-Language and Multi-Modality Assessment that extends beyond traditional single-language, text-only benchmarks to include multi-modality coding tasks; (3) Robustness Assessment that evaluates model reliability under semantically-preserving code transformations; and (4) Rigorous Evaluation Methodology that enhances the trustworthiness of evaluation results through diverse evaluation prompts and adaptive solution extraction. Based on this evaluation framework, we assess 26 state-of-the-art models and uncover both their strengths and limitations, yielding several key insights:(1) Current models show substantial performance variation across programming tasks; (2) Multi-modal language models demonstrate specific performance limitations in UI code generation and edit; (3) Existing models exhibit severe robustness issues on coding tasks; (4) Our multi-prompt evaluation method can mitigate potential evaluation bias from single prompts and obtain more reliable results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
## Paper Summary

This paper proposes TREAT, a benchmark and evaluation framework designed to assess trustworthiness and reliability of Code LLMs in realistic software-engineering scenarios. It covers:
1. 10+ tasks spanning code generation, summarization, reasoning, review, test generation, vulnerability detection, etc.
2. Multi-language support (up to 12 languages)
3. Multi-modality evaluation including UI-to-Code tasks
4. Robustness evaluation using code transformations and misleading cues
5. Multi-prompt evaluation to mitigate prompt bias
The authors evaluate 26+ state-of-the-art models, uncovering notable weaknesses in robustness, multi-modal UI tasks, and cross-task consistency.

### Strengths
## Strengths

1. Comprehensive coverage of SE tasks. Holistic lifecycle-oriented evaluation instead of narrow codegen-only benchmarks.

2. Incorporates robustness testing. Perturbations reveal vulnerability to misleading comments and code structure changes.

3. Large-scale model coverage. Commercial + open-source, reasoning vs non-reasoning LLMs.

### Weaknesses
## Weaknesses

1. Lack of novelty in dataset construction. Many task datasets are combinations or resampling from existing public benchmarks such as PRIMEVUL, EvalPlus, DESIGNBENCH, GEEKSFORGEEKS/HackerRank, GitHub mined code, etc. There is no clearly new task definition, only integration. The framework is large and ambitious but conceptual novelty is thin: It aggregates, expands, and re-labels existing evaluation components instead of introducing fundamentally new methodology.


2. Imcomplete and misleading claim in Table 1. First Table 1 miss some releated work, including some dataset used in this paper. Second, Multi Prompt techniques is used in DyCodeEval [1], Multi-Modality is used in SWE-Bench-Multimodal [2], and Robustness Evaluation seems to be a plugin appraoch can be applied to any seed dataset [3, 4], which does not belong to the contribution of this work.

3. Many details are pushed into Appendix. Insufficient explanation for: (1) sample selection (2) deduplication, (3) data quantity per task/language
and (4) contamination filtering and verification. 


4. Data contamination risk is acknowledged but not enforced or measured. Given that many evaluated models have been trained on GitHub/HackerRank, results may overestimate model capability. Specifically, some evaluated models in Table 2 were released after certain dataset releases; considering the lack of transparency of the pretraining data of the evaluated models and the potential inclusion of benchmark data in model training, reported scores may be inflated.-level. Even authors admit lacking repository-level realism (dependencies, build systems). Software engineering isn’t only function puzzles. I would suggest the auhtor also consider some repository task, such as SWEBench.


5. Missing related work. The paper would benefit from acknowledging and comparing against recent benchmarks that also focus on code LLMs.




[1]. Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination (ICML 2025)

[2]. SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains? (ICLR 2025)

[3]. DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation (ACL 2025 Finding)

[4] Is Your Benchmark (Still) Useful? Dynamic Benchmarking for Code Language Models

[6]. Codereval: A benchmark of pragmatic code generation with generative pre-trained models (ICSE 2024)

[7]. EvoCodeBench: An Evolving Code Generation Benchmark Aligned with Real-World Code Repositories (Neurips 2025)

[8]. Evaluating and Improving LLM-based Competitive Program Generation

[9]. PPM: Automated Generation of Diverse Programming Problems for Benchmarking Code Generation Models (FSE 2024)

### Questions
1. How do you justify TREAT as a “novel benchmark” given that most tasks are resampled or aggregated from existing datasets?

2. Can you provide more details on sample selection, deduplication, and contamination filtering? How consistent is data quantity across tasks and languages, and how might this affect cross-task comparisons?

3. Have you measured or verified contamination in the datasets with respect to the pretraining data of the evaluated models?

4. Given that evaluation is function-level, how do you ensure that TREAT reflects real-world software engineering challenges like repository-level dependencies and build systems?

### Soundness
3

### Presentation
3

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
This paper introduce a benchmark for evaluating the trustworthiness of large foundation models in real-world software engineering scenarios. It provides a very comprehensive setting to span the software development lifecycle with more than 10 takss and languages. Also, the holistic evaluation framework try to use different prompts to reduce the bias in evaluation.

### Strengths
1. this submission provides lots of detailed experiment results and analysis which comprehensively support the insights behind current large foundation models. 
2. the evaluation across multiple programming language and even including UI code generation tasks is very interesting and bridges the visuals and code. 
3. multi-prompt strategy and adaptive answer extraction helps to improve the fairness of this submission and real-world alighment.
4. the UI code editing or reparing tasks aligns with real deceloper workflows.
5. robustness testing with strucural, misleading comments, and misleading hints.

### Weaknesses
1. Adding more types of level can further enhance the benchmark (e.g., repo-level)

2. Model-as-judge may introduce some bias in evaluation. it's better to add more comparison between gpt-4o and human in some subset of testing and analyze the difference. 

3. Pass@1 is the primary metric for some tasks. It is better to consider more like partial correctness or runtime issues. 

4. In robustness evaluation, instead of rule-based perturbations, using some real-world git-diffs may aligh more with the real scenerios.  

5. Authors can try to analyze the influence of image resolution across different models in benchmarking.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents TREAT, an evaluation framework for assessing LLMs on code intelligence tasks. The framework evaluates 26 models across 10+ tasks spanning code generation, summarization, translation, reasoning, review, test generation, vulnerability detection, and UI-based tasks. The work incorporates multi-language support (Python and Java primarily but extended 12 languages), robustness evaluation using semantically-preserving perturbations, and multi-prompt evaluation. Key findings include performance variation across tasks, robustness issues, and bottlenecks in multimodal tasks.

### Strengths
- The paper evaluates 26 state-of-the-art models across diverse tasks, languages, and modalities, offering insights and providing extensive empirical data that could be useful for the community.

- The use of three prompts per task demonstrates prompt sensitivity, though the magnitude and practical implications remain unclear (see notes below). 

- The paper provides detailed appendices with experimental setup, promises code/data release, and supporting reproducibility.

### Weaknesses
The paper lacks both novelty and practicality. 

First, the paper is fundamentally an ensemble of existing benchmarks and methods with minimal innovation. The benchmark directly samples from or reuses PolyHumanEval, HumanEval+, MBPP, PrimeVul, SymPrompt/CodaMosa, DesignBench, and CodeCrash without significant modification. For instance, code generation uses problems from GeeksforGeeks and HackerRank with EvalPlus-style test augmentation, code translation uses PolyHumanEval and GeeksforGeeks, vulnerability detection directly adopts PrimeVul, multimodal tasks use DesignBench, and robustness evaluation uses CodeCrash. The multi-prompt approach, while useful, is a straightforward extension that simply runs three paraphrased prompts and averages results. No novel evaluation metrics, methodologies, or theoretical frameworks are introduced. The paper reads as a benchmark aggregation exercise rather than advancing evaluation methodology.

Seccond, there is critical data contamination problem rendering the benchmark impractical. The paper's heavy reliance on public platforms and existing benchmarks creates severe contamination issues that fundamentally undermine its utility: GeeksforGeeks and HackerRank are explicitly known to be in training corpora of major LLMs. The HumanEval series of benchmark (e.g., PolyHumanEval and HumanEval+) are also known to be extensively contaminated. Similarly, GitHub repositories are likely in training data given pre-training data scraping practices. These all making performance metrics potentially measure memorization rather than capability. While the paper mentions contamination as a "limitation", it failed to provide viable mitigation strategy, making the benchmark impractical for evaluating current or future models.

Finally, some of the tasks heavily rely on LLM as judge, and there was no proposal on how to mitigate the internal biases of these models, making results unreliable and untrusted especially the scale of performance difference is rather small. For example, the authors acknowledge GPT-4o's bias when GPT-5 scores lowest (26.9%) on code review despite excelling at other tasks, attributing this to preferring "simpler outputs" in Appendix. This admission directly contradicts using it as an objective judge.

### Questions
Please see "Weaknesses" for the major ones. In addition, 

1. Could the authors add statistical significance when presenting results?

2. What is the actual performance changes across prompts? While ranking is interesting, it'd be good to see the actual performance and variation to understand whether this is a problem that we should worry about at all.

3. There lacks sufficient literature review on robustness in machine learning for code and LLM in general.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a comprehensive evaluation framework for assessing the holistic performance of code LLMs. The framework integrates multiple code-related tasks, including code generation, code review, test generation, and code summarization, and extends to multi-language and multi-modality settings. It also incorporates robustness evaluation by applying semantically-preserving perturbations to the inputs. Experimental results reveal that existing large language models have notable robustness issues in coding tasks.

### Strengths
- The framework is comprehensive, covering diverse coding-related tasks, multiple programming languages, and multimodal settings. This provides a broad and unified view of code LLM capabilities.
- The inclusion of robustness evaluation through semantically-preserving perturbations is practical and relevant. The finding that existing models degrade significantly under prompt perturbations is insightful and highlights an important real-world limitation.

### Weaknesses
- The size and scale of each sub-benchmark are not clearly reported, making it difficult to assess coverage and statistical significance.

- When enhancing the prompt diversity, the prompt diversification process relies on manual validation, which limits scalability and reproducibility.

- On certain tasks such as Code Review, the evaluation results fail to effectively distinguish between models of different sizes or architectures, suggesting limited sensitivity of the metric or dataset.

- The contribution appears incremental. The framework largely aggregates existing datasets from prior work without introducing substantial new task design, annotation, or evaluation methodology.

### Questions
Many datasets are sourced from public websites, which are also used in the pretraining data of large code models. How does the paper ensure that evaluation data are not contaminated by training overlap?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TREAT, a framework for holistic evaluation of coding capabilities of LLMs, covering different coding tasks (like generation, summarization, test generation, etc), languages, multi-modality and robustness. They benchmarked this framework across different LLMs and noticed that no one model excels at all the tasks, while also showing that these LLMs are not robust to semantic code perturbations.

### Strengths
- Holistic benchmark with coverage across different tasks, languages, containing multimodal and robustness assessments. 
- Section 5.3 and 5.4 present interesting findings, where models with thinking exhibit better robustness to code perturbations and that model's evaluation results are sensitive to changes in prompt.

### Weaknesses
- Using GPT-4o as the only LLM judge may bias the scores towards GPT* models, for tasks beyond code correctness. Why did the authors not think of an ensemble based ranking? 
- In page 2, the authors mentioned "Current state-of-the-art models exhibit substantial performance variation and specialization across
different programming tasks" to be one of the novel findings. However, LiveCodeBench paper also discusses similar findings in Figure 4 (https://arxiv.org/pdf/2403.07974). This seems to be a misrepresentation.

### Questions
- Why is only GPT4o used as an LLM Judge?

### Soundness
3

### Presentation
3

### Contribution
3
