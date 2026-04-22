# MLE-Smith: Scaling MLE Tasks with Automated Multi-agent Pipeline

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
While Language Models (LMs) have made significant progress in automating machine learning engineering (MLE), the acquisition of high-quality MLE training data is significantly constrained. Current MLE benchmarks suffer from low scalability and limited applicability because they rely on static, manually curated tasks that demand extensive time and manual effort to produce.
We introduce MLE-Smith, a fully automated multi-agent pipeline, to transform raw datasets into competition-style MLE challenges through an efficient generate--verify--execute paradigm for scaling MLE tasks with verifiable quality, real-world usability and rich diversity. 
The proposed multi-agent pipeline in MLE-Smith drives structured task design and standardized refactoring, coupled with a hybrid verification mechanism that enforces strict structural rules and high-level semantic soundness. It further validates empirical solvability and real-world fidelity through interactive execution.
We apply MLE-Smith to 224 of real-world datasets and generates 606 tasks spanning multiple categories, objectives, and modalities, demonstrating that MLE-Smith can work 
effectively 
across a wide range of real-world datasets.
Evaluation on generated tasks shows that the performance of eight mainstream and cutting-edge LLMs on MLE-Smith tasks is strongly correlated with their performance on carefully human-designed tasks, highlighting the effectiveness of the MLE-Smith in scaling up MLE tasks while maintaining task quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the scarcity of high-quality, scalable training data for Machine Learning Engineering (MLE) tasks. The authors propose MLE-Smith, a fully automated multi-agent pipeline designed to convert raw datasets into competition-style MLE challenges. The pipeline operates on a "generate-verify-execute" paradigm, utilizing specialized agents to create tasks. A hybrid verification mechanism is employed to ensure structural integrity, semantic soundness, and empirical solvability. The authors apply this pipeline to 224 real-world datasets, generating 606 new MLE tasks.

### Strengths
1. The paper tackles an important problem. The bottleneck in creating high-quality, large-scale benchmarks for MLE agents is a real obstacle to progress in the field.

### Weaknesses
1. The contribution of the paper is limited. The proposed "generate-verify-execute" pipeline is a relatively common and established strategy for automated data synthesis and benchmark generation. Many prior works have employed similar paradigms, and the paper does not sufficiently articulate how this multi-agent application fundamentally differs from or improves upon those, beyond its application to the MLE domain. 
2. While the verification mechanism ensures that tasks are executable and solvable (i.e., an agent can run code and get a non-trivial score), it does not appear to guarantee the correctness of their reference solutions. It is possible for a generated task to pass all verification checks while containing subtle logical flaws, or for the intended solution path to be suboptimal. The validation relies on "non-trivial predictive performance" rather than a strong guarantee of ground-truth correctness, which could impact the quality of the benchmark for training.
3. The experimental evaluation is lacking in-depth analysis. The results are largely aggregated, focusing on high-level Elo rankings and correlations. While this shows that the benchmark can rank models, it fails to provide insight into why models perform as they do or what specific challenges the new benchmark presents. The paper lacks a qualitative or fine-grained analysis of model failures. For example, what kinds of tasks are most difficult? What specific errors do top-performing agents make? Without this analysis, it is difficult to confirm the "challenging" nature of the benchmark, as it's unclear if it introduces new, harder problems or simply more of the same problems found in existing benchmarks.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Automating machine learning engineering (MLE) is an important task in LLM. However, the acquisition of high-quality MLE training data is difficult. In this work, the authors introduce MLE-smith, a fully automated multi-agent pipeline, to transform raw datasets into competition-style MLE challenges through an efficient generate–verify–execute paradigm for scaling MLE tasks with verifiable quality, real-world usability and rich diversity and a hybrid verification mechanism that enforces strict structural rules and high-level semantic soundness. The proposed benchmark utilizes 224 datasets to derive 606 tasks spanning multiple categories, objectives, and modalities.

### Strengths
1. The writing is clear and easy to understand.
2. The design of the pipeline is reasonable, detailed and economical. The pipeline incorporates a hybrid verification stack: deterministic assertions (format/structure-checks), semantic reviews (via agent), and execution-based validation (empirical solvability).  This provides multiple levels of guarantee that the generated tasks are structurally correct, semantically meaningful, and actually executable by agents while the cost is mere $2.11 per dataset. Given the cost, the scalability of this method is ensured.
3. They conduct comprehensive experiments on 8 different language models to testify to the quality of their dataset and their strong correlation with human-designed tasks.

### Weaknesses
1. I think the authors should include more details about the hybrid verification check in the Appendix, to ensure the reproducibility of this work.

### Questions
1. The difference of Refactor and Assertions seems ambiguous to me. It looks like that they are both executing formal checks. I would appreciate it if the authors could further clarify their differences, e.g. Refractor is for execution while Assertions is for check only?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the critical bottleneck in scaling the creation of high-quality Machine Learning Engineering (MLE) benchmarks, which are essential for developing and evaluating sophisticated AI agents. The key of this work is MLE-Smith, a fully automated, multi-agent pipeline designed to transform raw datasets into competition-style MLE challenges at scale. The authors demonstrate the efficacy of their system by applying MLE-Smith to 224 real-world datasets, successfully generating 606 fully verified tasks spanning a wide range of modalities, objectives, and domains.

### Strengths
1.	The manuscript is well-motivated and well-written. The paper tackles a problem of significant importance. The ability to automatically generate diverse, high-quality MLE benchmarks at scale would be a major catalyst for research in autonomous MLE agents. Moreover, the proposed generate-verify-execute pipeline is well-presented and can be easily understood.

2.	The authors successfully demonstrate the system's capability at scale. They generate 606 tasks from 224 diverse datasets. The subsequent evaluation is extensive, involving eight different LLMs on a 100-task benchmark.

### Weaknesses
1.	The paper's main justification for task quality is that its performance rankings correlate with the MLE-Dojo benchmark. While this is a useful check, it feels like a narrow definition of "quality." Relying only on this metric makes it hard to judge other important aspects, like whether the tasks are truly novel, realistic, or test a wide range of skills. The paper's claims would be much more convincing if backed by more evidence, such as qualitative feedback from human MLE experts or an analysis showing that the tasks require diverse problem-solving strategies.

2.	The paper proposes a sophisticated system with multiple components, but there are no ablation studies to show what each part is actually contributing. For example, how much does performance change if you alter the agent's prompt or remove a specific reasoning step? Without this analysis, it's hard for the reader to know which components are essential to the system's success and which might be less important.

### Questions
1.	Could you consider supplementing the benchmark correlation with qualitative feedback from human MLE experts to provide a more holistic validation of task quality?

2.	Could you provide further analysis on the diversity of skills or problem-solving strategies required by your tasks, beyond the performance ranking correlation?

3.	How sensitive is the system's performance to changes in key components, such as the agent's prompt or the removal of a specific reasoning step?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MLE-Smith, a fully automated multi-agent system that converts raw datasets into competition-style machine learning engineering (MLE) tasks using a generate–verify–execute paradigm. Three specialized agents—Brainstormer, Designer, and Refactor—collaborate to create, standardize, and validate MLE tasks. Performance rankings on MLE-Smith tasks strongly correlate with those on human-curated benchmarks (MLE-Dojo), Pearson r ≈ 0.98, Spearman ρ ≈ 0.95, demonstrating high realism. Overall, MLE-Smith achieves scalable, diverse, and verifiable generation of MLE challenges for agent training and benchmarking.

### Strengths
1. Thorough evaluation – Quantitative correlation between synthetic and real tasks across multiple LLMs; diverse modalities and metrics.
2. Strong empirical realism – Demonstrated high rank and score correlation with human benchmarks, suggesting the tasks are faithful surrogates for real-world ones.
3. Reproducibility – Detailed description of environment, budgets, and execution setup; appendix lists datasets and code schema.

### Weaknesses
1. Ablation missing: The pipeline is explicitly broken into three agent roles (Brainstormer, Designer, Refactor) and a three-layer verification mechanism (Assertions, Reviews, Execution-based validation). However, it does not provide an empirical ablation study to justify the necessity of each individual component.
2. The pipeline was tested on 300 datasets from Kaggle. These datasets are typically well-structured and pre-cleaned for competition. It is unclear how MLE-Smith would handle "rawer" datasets (e.g., unstructured server logs, complex scientific data) that lack clear, pre-identified features or labels, and which may require significant domain expertise to formulate a task. In practice, those rawer data are even more useful since it requires good feature engineering strategies. 
3. Lack of in-depth discussion: The paper states 807 tasks were generated and 606 were "fully verified". This implies a failure rate of ~25% (201 tasks). The paper does not provide a breakdown of why these tasks failed. clearer framing of scientific insight (why this works) would strengthen the whole paper as well.

### Questions
1. Could MLE-Smith handle domains outside Kaggle-like structured datasets?
2. Are there known failure patterns or bottlenecks in verification throughput? How many failed at the Assertions, Reviews, and Execution-based Validation stages, respectively?
3. What empirical benefit does the multi-agent separation provide versus a monolithic prompting approach?

### Soundness
2

### Presentation
3

### Contribution
2
