# CoolPrompt: An Automatic Prompt Optimization Framework for Large Language Models

- Avg Score: 2.40
- Decision: Reject
- Scores: 2, 4, 2, 2, 2

## Abstract
The effectiveness of Large Language Models (LLMs) is highly dependent on the design of input prompts. Manual prompt engineering requires a domain expertise and prompting techniques knowledge that leads to a complex, time-consuming, subjective, and often suboptimal process. We introduce CoolPrompt as a novel framework for automatic prompt optimization. It provides a complete zero-configuration workflow, which includes automatic task and metric selection, also splits the input dataset or generates synthetic data when annotations are missing, and final feedback collection of prompt optimization results. Our framework provides three new prompt optimization algorithms ReflectivePrompt and DistillPrompt that have demonstrated effectiveness compared to similar optimization algorithms, and a flexible meta-prompting approach called HyPE for rapid optimization. Competitive and experimental results demonstrate the effectiveness of CoolPrompt over other solutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper presents CoolPrompt, an automatic prompt optimization framework designed to be "zero-configuration". It automates the full optimization pipeline, including automatic task and metric selection, and synthetic data generation for when annotations are missing. The framework introduces three new optimization algorithms: HyPE (a rapid meta-prompting method),

### Strengths
1. The primary strength is the "zero-configuration" framework that aims to automate the entire autoprompting pipeline. This includes valuable components like a Task Detector, Synthetic Data Generator, and a PromptAssistant for optimization feedback, addressing significant practical barriers for users.

### Weaknesses
1. The evaluation is limited which doesn't benchmark against other prompt optimization methods such as OPRO. 
2. I reserves conservation on the novelty of framework. 
3. The proposed three prompt optimization approaches is not thoroughly benchmarked and analyzed if they are the main contribution.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CoolPrompt, a zero-configuration framework for automatic prompt optimization that combines (1) a rapid meta-prompting method HyPE, and (2) two longer-running autoprompting algorithms ReflectivePrompt and DistillPrompt. The system also includes automatic task/metric detection, an LLM-driven synthetic data generator, and a PromptAssistant that provides feedback; experimental comparisons across several benchmarks show competitive results versus other autoprompting libraries.

### Strengths
1. The paper demonstrates an end-to-end system perspective by presenting a complete pipeline that includes task detection, synthetic data generation, optimization strategies, and feedback mechanisms, making it valuable as an applied system.
2. It introduces diverse optimization strategies—HyPE for meta-prompting, ReflectivePrompt for evolutionary reflection, and DistillPrompt for iterative distillation—that complement each other and are well motivated.
3. The work incorporates several practical features, such as model-agnostic integration through LangChain, automatic metric selection, and feedback generation, which strengthen its engineering contributions.
4. The authors conduct thorough ablations on synthetic data and meta-prompt design, evaluating synthetic data quality across models and refining the HyPE meta-prompt, making these analyses appropriate and informative.

### Weaknesses
1. The experimental setup is weakly grounded — each task uses only 30 samples and three runs, which is insufficient for statistically reliable conclusions.
2. The paper does not report standard deviations, confidence intervals, or significance tests, making it difficult to assess robustness. The optimization experiments rely on only 30 samples per task and three runs, which is too small to draw statistically significant conclusions. No confidence intervals, variance estimates, or error bars are provided.
3. Several baseline results appear inconsistent or potentially misconfigured (e.g., extremely low BertScore values), raising questions about fairness and reproducibility.
4. The paper lacks human evaluation for generative tasks and does not discuss known failure cases or limitations. All evaluations rely on automated metrics like BertScore or EM, which are insufficient for tasks involving coherence, relevance, or factual consistency. A small-scale human study would have strengthened the evidence.
5. Conceptual novelty is modest; the work focuses more on system integration than on algorithmic advancement.
6. Key configuration details are missing - including random seeds, prompt templates for baselines, number of LLM calls, runtime, and cost estimates. Without these, reproducibility is difficult, and results cannot be independently verified.Key configuration details are missing — including random seeds, prompt templates for baselines, number of LLM calls, runtime, and cost estimates. Without these, reproducibility is difficult, and results cannot be independently verified.
7. The paper does not analyze cases where automatic prompt optimization fails (e.g., tasks with ambiguous metrics or open-ended generation), nor does it provide insights into when human-in-the-loop guidance remains necessary.

### Questions
1. How many optimization trials and LLM calls were performed per method, and what were the compute costs?
2. How consistent are results when the same optimization is repeated with different random seeds?
3. How were baselines (Promptify, DSPy, etc.) configured, and were identical datasets, metrics, and generation parameters used?
4. What explains anomalies such as the very low BertScore for some baselines in Table 1?
5. How is circularity avoided when using GPT-4o or GPT-3.5 both as optimizer and as evaluation model?
6. What mechanisms ensure that automatically generated synthetic data are accurate and not hallucinated or mislabeled?
7. Can the authors demonstrate transferability?  Do optimized prompts generalize to unseen LLMs or unseen domains?
8. How does the proposed system handle tasks where evaluation metrics cannot be automatically inferred (e.g., subjective or multi-objective tasks)?

### Soundness
3

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
This paper introduces a new prompt optimization framework, CoolPrompt, that receives a generic description of the task, and then uses a series of optimization processes to iteratively refine the prompt using real data, synthetic data, and LLMs. Three central techniques are used: (1) using an LLM to generate new prompts, (2) a genetic algorithm to extend and synthesize prompts, and (3) and an iterative generate and distill approach. The approach is tested on five type of tasks, with prompts from the second technique attaining higher performance on four of five

### Strengths
- Presents multiple options for how to optimize prompts automatically. Methods uses a series of steps that can each potentially contribute to the optimization

- Minor, but I agree with the authors that this paper is filling a key gap in prompt optimization for smaller open-weight models

- Compares performance on five types of tasks

### Weaknesses
- The paper describes a single framework but three approaches are analyzed (Table 2) and it is not clear to me whether the framework picks one of these methods or if their prompts are somehow aggregated. Given the superior performance of the ReflectivePrompt setup, it's not clear why the other approaches are needed (perhaps they do better for some settings?)

- The whole pipeline is relatively complex, which isn't necessarily bad. However, it's not clear what in this pipeline is contributing to a better prompt. Some type of ablation analysis would be very helpful here beyond what's in A.2. It would help to see what impact each stage has on the resulting performance.

- While the models are getting better performance with the prompt, I would be very interested in the timing of the pipeline, given its complexity. Compared to other approaches, how much longer (or shorter) is CoolPrompt?

- The tasks, while diverse, are relatively old with AG News being over a decade. I'm not opposed to older tasks if they still pose a challenge but given the age, most pretrained/large language models have seen this data which adds a potential confound. Also, given the relatively high performance on all tasks, even for the manual zero-shot prompt, it would be useful to see whether these approaches work for more recent and more challenging tasks.

- Minor, but many experiment details are moved to the appendix. however, the paper has nearly a page of extra space. It would make the paper much more readable to have this content in the main part.

### Questions
- I was confused by the comment on line 251 that CoT requires task-specific exemplars. I don't think this is true since you could just include the "Think step by step" command in the prompt to elicit CoT output. Could you clarify what is meant here?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CoolPrompt, an automatic prompt optimization framework for large language models. The framework provides a zero-configuration workflow including automatic task detection, metric selection, synthetic data generation, and optimization feedback. The authors propose three optimization methods: HyPE for rapid optimization, and ReflectivePrompt and DistillPrompt for long-term optimization. Experiments on multiple benchmark datasets demonstrate the framework's effectiveness.

### Strengths
1. This work proposes a relatively complete prompt optimization framework with commendable attention to data aspects (including automatic data generation and task detection), which has significant practical value for real-world applications.
2. The work introduces several optimization algorithms synthesized from existing approaches (HyPE, ReflectivePrompt, DistillPrompt) and demonstrates improved performance over baselines across multiple tasks.

### Weaknesses
1. The paper's presentation requires significant improvement. Figures 1, 2, and 3 use crude flowcharts that rely heavily on text stacking, lacking clear visual hierarchy, making it difficult for readers to quickly grasp key information and inter-module relationships.
2. While claiming to present a complete framework, the paper omits critical implementation details. Particularly, the synthetic data generation module, listed as a core contribution, is only described with a four-step outline (Appendix A.5.3) without providing specific generation algorithms, prompt templates, or quality control mechanisms, making this contribution difficult to verify and reproduce.
3. The experimental design has significant flaws. First, the benchmark datasets are dated and the experimental scale is too small (only 30 samples total, with just 6 samples in the training set). Second, the comparison targets are inappropriate, only comparing against engineering frameworks like Promptify and AdalFlow, rather than mainstream optimization algorithms such as OPRO, DSPy, and APE. Finally, the experiments lack statistical significance testing, ablation studies, and cost analysis.
4. The paper lacks substantial academic novelty. ReflectivePrompt and DistillPrompt are essentially combinations of existing evolutionary algorithms and Tree-of-Thoughts methods. While HyPE shows some originality, it lacks in-depth analysis. The synthetic data generation is not a novel method but merely an engineering integration. Overall, this reads more like a technical report for an engineering project rather than an academic paper with significant methodological contributions.

### Questions
Please provide detailed implementations of key modules, particularly the complete algorithmic workflow and data quality control mechanisms for synthetic data generation.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an automatic prompt optimization system for large language models (LLMs) that includes automatic task and metric selection, and synthetic data generation. The proposed method provides three prompt optimization strategies: ReflectivePrompt, DistillPrompt, and HyPE. The effectiveness of the proposed method is verified through experiments on several tasks, including question answering, mathematical reasoning, and text classification.

### Strengths
- This paper provides a zero-configuration framework for prompt optimization, which is useful for users who are not familiar with prompt engineering.
- In the proposed method, three different prompt optimization strategies are provided for short- and long-term optimization.
- The experimental results demonstrate that the proposed method exhibits competitive performance on different tasks with existing prompt optimization methods.

### Weaknesses
- Although this paper proposes a complete pipeline for prompt optimization, each component of the proposed method is based on existing methods. For instance, the prompt optimizers, HyPE and ReflectivePrompt, are based on prior works of HyDE (Gao et al., 2023) and Reflective Evolution (Ye et al., 2024). The technical novelty of the proposed method is limited, and the core innovation of this paper is not clear.
- The experimental evaluation is weak:
    - Only the gpt-3.5-turbo model is considered. It is unclear whether the proposed method works well for other LLMs.
    - Only the performance of each prompt optimization method is evaluated. However, the cost of each method in terms of LLM API calls or token consumption is not discussed.

### Questions
- How is the computational cost of each prompt optimization method in terms of LLM API calls or token consumption?
- The authors claim that other automatic prompt optimization frameworks have limitations of usage only for proprietary LLMs. However, the reviewer cannot understand the reason for this point. What are the technical limitations and difficulties of existing methods when using custom LLMs?

### Soundness
2

### Presentation
2

### Contribution
2
