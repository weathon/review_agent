# DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
The fast-growing demands in using Large Language Models (LLMs) to tackle complex multi-step data science tasks create a emergent need for accurate benchmarking. There are two major gaps in existing benchmarks: (i) the lack of standardized, process-aware evaluation that captures instruction adherence and process fidelity, and (ii) the scarcity of accurately labeled training data. To bridge these gaps, we introduce DARE-bench, a benchmark designed for machine learning modeling and data science instruction following. Unlike many existing benchmarks that rely on human- or model-based judges, all tasks in DARE-bench have verifiable ground truth, ensuring objective and reproducible evaluation. To cover a broad range of tasks and support agentic tools, DARE-bench consists of 6,300 Kaggle-derived tasks and provides both large-scale training data and evaluation sets. Extensive evaluations show that even highly capable models such as gpt-o4-mini struggle to achieve good performance, especially in machine learning modeling tasks. Using DARE-bench training tasks for fine-tuning can substantially improve model performance. For example, supervised fine-tuning boosts Qwen3-32B’s accuracy by 1.83× and reinforcement learning boosts Qwen3-4B’s accuracy by more than 8×. These significant improvements verify the importance of DARE-bench both as an accurate evaluation benchmark and critical training data.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The author suggests DARE-bench, designed to evaluate and train LLMs for data science workflows. It addresses two major gaps in existing benchmarks: (1) lack of process-aware evaluation capturing instruction adherence and modeling fidelity, and (2) lack of accurately labeled training data. DARE-bench includes 6,300 verifiable, executable Kaggle-derived tasks divided into two families. The benchmark enables reproducible automatic evaluation via sandboxed code execution. Experiments show that general-purpose LLMs perform poorly on these tasks without domain-specific training, but fine-tuning on DARE-bench data dramatically improves both process fidelity and prediction accuracy. The paper demonstrates up to 8× improvement in smaller models after reinforcement learning fine-tuning, and also validates generalization improvements on external datasets like DSBench.

### Strengths
1. Introduces a large-scale, executable, and verifiable benchmark that measures both process fidelity and predictive accuracy, filling a key gap in LLM evaluation for data science.

2. Provides a rich dataset derived from Kaggle that supports both evaluation and supervised/RL-based training, improving reproducibility and scalability.

3. Demonstrates concrete, measurable performance improvements and reduced execution failures after fine-tuning, validating benchmark utility for both evaluation and model training.

4. Clearly integrates determinism and sandbox execution to ensure fair, reproducible comparison across models, a major methodological advantage.

### Weaknesses
1. Heavy reliance on Kaggle-derived data may bias task diversity toward structured tabular and forecasting problems, limiting broader domain generalization.

2. The benchmark’s focus on reproducibility via deterministic setups may underrepresent realistic data science variability or stochastic modeling behavior.

3. The paper lacks ablation or error analysis to quantify which aspects of fine-tuning drive improvements most.

### Questions
1. How does DARE-bench handle tasks involving stochastic algorithms (e.g., random forest, neural nets) while maintaining deterministic evaluation?

2. Are the reference “ground truth” codes verified manually, or automatically synthesized and how is correctness guaranteed?

3. Does reinforcement learning with DARE-bench tasks risk overfitting to procedural templates rather than improving general data reasoning?

4. Can this benchmark be extended beyond Kaggle-style tabular tasks, e.g., to unstructured text or multimodal datasets?

### Soundness
3

### Presentation
3

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
This paper introduces DARE-bench, a benchmark designed to evaluate and train LLMs for data science tasks, addressing two critical gaps in existing benchmarks: (i) the absence of process-aware, verifiable evaluation (e.g., measuring instruction adherence) and (ii) scarcity of high-quality labeled training data. Derived from 6,300 Kaggle datasets, DARE-bench includes two task families—process-aware instruction-following (with reference-code ground truth) and ML modeling (with dataset ground truth)—covering classification, regression, and time-series forecasting. Key features include verifiable ground truth (enabling objective, human-judge-free evaluation) and a dual role as both an evaluation tool and training resource. Evaluations show strong LLMs (e.g., Qwen3-32B, gpt-4o-mini) perform poorly on baseline tests, but supervised fine-tuning (SFT) and reinforcement learning (RL) using DARE-bench data yield dramatic gains: SFT improves Qwen3-32B’s accuracy by 1.83×, and RL boosts Qwen3-4B’s accuracy by over 8×.

### Strengths
1. It addresses two key gaps in existing benchmarks: it enables verifiable, process-aware evaluation (relying on reference-code or dataset ground truth, no human/model judges) and provides 6,300 Kaggle-derived tasks as large-scale training data, ensuring objective, reproducible assessments . 
2. Its task coverage is comprehensive—covering classification, regression, time-series forecasting, with two variants (instruction-following/ML modeling) probing core DS capabilities, outperforming peers (e.g., DS-1000, DSBench) in time-series support and training task provision .

### Weaknesses
1. Tasks are almost exclusively tabular, excluding multimodal inputs (e.g., text-image combinations, code-diagram interactions) common in modern DS.
2. Generating large-scale executable trajectories (for training data) is costly, and rejection sampling strategies may introduce biases toward shorter trajectories.

### Questions
As listed in weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces DARE-bench, a benchmark for evaluating and training LLMs on data science (DS) tasks. Considering that existing benchmarks lack of process-aware evaluation and scarce labeled data, the DARE-bench offers 6,300 Kaggle-derived tasks (covering classification, regression, time-series forecasting) with verifiable ground truth. Tasks include two variants: Instruction Following (IF, testing workflow adherence) and ML Modeling (MM, testing outcome accuracy). Experiments show baseline LLMs perform poorly, but SFT and RL using DARE-bench data drastically improve performance.

### Strengths
1. The paper is good-writing and easy to follow.  The benchmark provides comprehensive evaluation scope, specifically, it covers diverse DS tasks (including underrepresented time-series forecasting) and enforces real-world constraints (execution time, interaction turns), enhancing practical relevance.
2. DARE-BENCH serves both as an evaluation tool and a large-scale training resource, with proven effectiveness in improving LLM performance via SFT/RL.

### Weaknesses
1. Lack of Comparison with Specialized DS Agents. The paper evaluates general-purpose and code-centric LLMs  but omits comparisons with specialized data science agents, which are explicitly designed for multi-step DS workflows. This gap makes it hard to contextualize DARE-bench’s utility. It is unclear whether the benchmark’s gains (via fine-tuning) can match or surpass the performance of purpose-built DS agents,
2. Provide more explanations about the Instruction Following (IF) and ML Modeling (MM) metrics. The paper frames IF (workflow adherence) and MM (outcome accuracy) as two core DS capabilities to evaluate together, but fails to justify their joint necessity. Especially given Table 4’s results showing no clear correlation between the two metrics. For example, GPT-5 scores highest in classification-IF (69.81) but ranks mid-tier in classification-MM (43.40); Claude-Sonnet-3.7 excels in MM tasks (e.g., regression-MM: 63.20) but lags GPT-5 in IF.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Large Language Models (LLMs) are increasingly adopted for complex multi-step data science (DS) tasks, yet existing benchmarks suffer from two critical gaps: a lack of process-aware evaluation (e.g., instruction adherence and process fidelity) and scarce high-quality labeled training data. To address these, this paper introduces DARE-BENCH, a training-focused benchmark for evaluating LLMs’ DS capabilities, encompassing both machine learning (ML) modeling and instruction following.

Derived from Kaggle datasets, DARE-BENCH includes classification (Instruction Following/IF, ML Modeling/MM), regression (IF, MM), and time-series (eXogenous Features/XF, Canonical Forecasting/CF) tasks, split into 95% training and 5% test sets. Unlike benchmarks relying on human/model judges, all tasks have verifiable ground truth (reference outputs for IF tasks, original dataset labels for MM tasks), ensuring objective, reproducible evaluation via a sandboxed code execution environment.

Extensive evaluations show that even advanced LLMs (e.g., gpt-4o-mini, Qwen3-32B) perform poorly on baseline tests, especially in time-series tasks. However, fine-tuning with DARE-BENCH yields significant improvements: supervised fine-tuning (SFT) increases Qwen3-32B’s accuracy by 1.83×, while reinforcement learning (RL) boosts Qwen3-4B’s accuracy by over 8×. External validation on DSBench further confirms generalization.

### Strengths
DARE-BENCH has several strengths against previous work.  
Unlike counterparts that only assess final-answer accuracy, DARE-BENCH uniquely evaluates both ML modeling performance and instruction fidelity, filling the void of process-aware assessment. It also provides 6,300 Kaggle-derived tasks with verifiable ground truth (reference outputs for IF tasks, original labels for MM tasks). The training data seems to be valuable. 
In addition, its four-stage pipeline minimizes human effort, enabling large-scale task generation (6,300 tasks) while ensuring realism—e.g., 20% noise injection in IF tasks to simulate real-world data issues.

### Weaknesses
The task diversity is limited. It exclusively covers tabular data, lacking support for multimodal DS tasks (e.g., text-image fusion, speech-data analysis), restricting applicability to broader DS scenarios.

### Questions
1. How do you validate the quality of your generated dataset? 
2. Do you have qualitative and quantitative analysis?
3. What's the detailed dataset statistics for your dataset, e.g., information like how many tool calls, how many tokens are your prompt or your completion.

### Soundness
2

### Presentation
2

### Contribution
3
