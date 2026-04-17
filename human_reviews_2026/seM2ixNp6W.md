# Who Judges the Judge? LLM Jury-on-Demand: Building Trustworthy LLM Evaluation Systems

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
As Large Language Models (LLMs) become increasingly integrated into high-stakes domains, there is a growing need for evaluation methods that are both scalable for real-time deployment and reliable for critical decision-making. While human evaluation is reliable, it is slow and costly. Single LLM judges are biased, and static juries lack adaptability. To overcome these limitations, we propose LLM Jury-on-Demand – a dynamic, learning-based framework for scalable and context-aware evaluation. Our method trains a set of reliability predictors to assess when LLM judges will agree with human experts, leveraging token distributions, embeddings, and structural input features. This enables a two-tiered adaptation - first selecting an optimal jury per dataset, then assigning dynamic, instance-specific weights to its members. Experiments on summarization and RAG benchmarks show that our dynamic jury system achieves significantly higher correlation with human judgment than both single-judge and static-jury baselines. These results highlight the promise of adaptive, learning-based juries for building scalable, more reliable and trustworthy evaluation systems for modern LLMs in high-stakes domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes "LLM Jury-on-Demand," a dynamic, learning-based framework for evaluating LLM outputs. The core idea is to move beyond static evaluation methods by training a set of "reliability predictors" that assess, on an model-by-model basis, how likely each LLM judge is to agree with human experts. This allows the system to dynamically assemble an optimal jury for each data point and assign instance-specific weights to its members, aiming for a more scalable, reliable, and context-aware evaluation system.

### Strengths
1. The paper is well-written and clearly presents the core concepts of the proposed framework, making it easy for readers to understand the main ideas and contributions.

2. The authors validate their approach across a wide range of datasets for both summarization and RAG tasks. This extensive evaluation, despite potential issues in the experimental setup noted below, provides evidence for the method's generalizability.

### Weaknesses
1. **Systematic Evaluation of Judge Pool Composition**: The experiments are conducted using a fixed pool of 9 LLMs, which includes a mix of open and closed-source models of varying sizes. *This experimental choice could significantly influence the results**. It is unclear how the framework would perform with a different composition of judges, particularly a pool of models with **more comparable capabilities**. As shown in Figure 3 (Summarization Groundedness), the selection process appears to converge on a few specific models. A more comparable judge pool might lead to different outcomes and would also provide a more challenging test for the "**Static Jury**" baseline, which is naturally **disadvantaged by the inclusion of weaker LLMs**. An analysis of the framework's performance under different judge pool configurations is needed.

2. Potentially Flawed Human-LLM Alignment Metric: The paper's method for measuring alignment between an LLM judge and human experts is based on the *absolute difference* between their normalized scores. As seen in Figure 3, this leads to counter-intuitive results where a model like Claude 3.7 Sonnet is the top choice for "Summarization Groundedness" but is almost never selected for "Summarization Completeness". Rather than concluding that the model has poor alignment on completeness, it is more plausible that this is due to a **systematic scoring bias** (e.g., the model consistently gives higher scores than humans). Even with such a bias, the model's preference ranking of outputs might still align well with human judgment, a nuance that the current metric fails to capture (as suggested by the **rank correlations in Figure 2**, Claude 3.7 Sonnet still can achieve very high alignment with humans on "Summarization Completeness"). **Since this alignment score is the training label for the reliability predictors, any bias introduced by this metric is critical and warrants further investigation.**

3. **Insufficient Baseline Comparisons**: The conclusion that an ensemble of LLM judges outperforms a **single judge** is largely expected. To demonstrate the effectiveness, comparsion should be done with other ensemble methods. However, the authors only compare their dynamic jury to a simple averaging baseline (also with some weak LLMs in the pool). This is **insufficient to demonstrate the novelty and effectiveness** of the proposed selection mechanism. More advanced and relevant baselines should be included, such as:
    - A static "Top-K" jury, where the best-performing K models on the entire validation set are selected.
    - Methods from the LLM routing or mixture-of-experts literature, which are conceptually similar to the task of selecting the best model for a given instance.

### Questions
- Could you please clarify how the *Kendall's Tau correlation* was computed in your experiments? Specifically, does it measure the correlation of rankings over the entire test set for each task, or is it calculated differently?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LLM Jury-on-Demand, a dynamic, learning-based framework for automated LLM evaluation that predicts instance-level reliability for each candidate judge and then assembles a context-aware jury whose members’ raw scores are weighted by predicted reliabilities. The system (1) extracts rich textual features from inputs/outputs (token/size/embedding/text-complexity features; Fig.1, p.3), (2) trains per-judge XGBoost classifiers to predict whether a judge’s score will be “good” vs “bad” relative to human scores (Sec.3.3), and (3) for each instance selects the top-K judges by predicted reliability and aggregates their scores via reliability-weighted averaging (Sec.3.4). Experiments on summarization and RAG metrics (groundedness, relevance, completeness) across multiple datasets show the Jury-on-Demand consistently improves Kendall’s Tau correlation with human judgments versus single-judge and static-ensemble baselines (Fig.2, Table 1). The paper analyzes judge selection patterns and feature importance (e.g., embedding PCA vs compression ratio) and provides ablations and implementation details in the appendix.

### Strengths
* Clear problem with practical solution. The paper addresses a concrete weakness of current LLM-as-judge approaches (judge reliability varies by instance) and proposes an implementable remedy (per-judge reliability predictors + dynamic jury selection).

* Practical gains. Jury-on-Demand consistently achieves higher Kendall’s Tau than baselines; improvements are shown at both aggregate and dataset levels

### Weaknesses
Several areas limit the paper’s completeness and broader impact:

* First, the method’s sensitivity to hyperparameters (τ for reliability labeling, K for jury size) is not systematically analyzed; these choices likely influence both reliability prediction and final performance.
* Second, baselines could be stronger, comparing against static weighted ensembles or calibration-based weighting would clarify the added value of instance-level adaptivity.
* Third, statistical significance and effect sizes are not explicitly reported, even though claims of superiority rely on small Tau differences.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the limitations of existing Large Language Model (LLM) evaluation methods—slow and costly human evaluation, biased single LLM judges, and inflexible static juries—by proposing LLM Jury-on-Demand, a dynamic, learning-based evaluation framework. The framework first extracts multi-dimensional text features (text size, special words, complexity, and embedding-related features) from input data (e.g., source text, generated summaries for summarization tasks). It then trains XGBoost-based reliability predictors to assess how well each LLM judge aligns with human experts, dynamically selects the most reliable judges to form an optimal jury for each data point, and aggregates scores using reliability as weights.

### Strengths
* Introduces a creative combination of existing ideas (LLM-as-a-Judge, multi-model juries, LLM performance prediction) into the LLM Jury-on-Demand framework.

* The methodology is rigorously designed: (1) Multi-dimensional feature engineering covers key signals for reliability prediction; (2) XGBoost-based reliability predictors are tailored to each "judge-task-metric" combination, ensuring targeted prediction; (3) Experiments use 9 diverse LLMs as judges, span 2 core tasks (summarization/RAG) and multiple datasets.

* The work targets a critical pain point in high-stakes LLM deployment—lack of scalable, reliable evaluation.

### Weaknesses
* Over-Reliance on Human-Annotated Data Limits Scalability: Reliability predictors depend entirely on human-annotated datasets for training，this creates a critical bottleneck, high-stakes domains (e.g., legal/medical) often lack such annotations, making the framework hard to deploy there. 

* Insufficient Analysis of Jury Size (K) and Tolerance (τ) Hyperparameters: K and τ are optimized via validation sets but provides no details on: (1) the search range of K (e.g., 3–9 judges) and how different K values impact performance; (2) why τ varies across judges/tasks or its sensitivity to results.

### Questions
NO ETHICS STATEMENT and REPRODUCIBILITY STATEMENT

Please see the weaknesses.

### Soundness
3

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
4

### Summary
This work proposes a multi-LLM evaluation framework with dynamically selected judges for specific model contexts. The jury allocation mechanism is learned via a feature engineering technique from the instances of LLM inputs and outputs, where the task is to learn the most predictive features for judge reliabilities (alignment with human experts). With the learned scores for each LLM judge, the system is able to select the top performing judges and assemble their final evaluations with learned weights.  Experiments are conducted on a summarization benchmark and a RAG benchmark, where the results show the proposed dynamic system provide better alignment with manual evaluations than static and single jury baselines.

### Strengths
1 The learning-based design of feature selection is an effective way to discover judge reliabilities. Most of the candidate features are intuitive can be related to specific tasks or criteria.

2 The dynamic evaluation system is logically sound and technically proficient. One can easily identify the purpose of each component.

3 The experiment results show good alignment with human experts. The authors also give interesting and insightful findings on the analysis section.

4 The structure of this paper is well-organized and the writing is generally easy to understand.

### Weaknesses
1 This assembled jury method certainly brings more computation burden or api calls of multiple LLMs as juries, yet its performance does not seem to win by a large margin to justify the additional cost (e.g. on DialSumEval and SummEval, the best single Judge is only 0.01 lower than the proposed the multi-judge method)

2 The proposed method is also heavy on hyperparameter-tuning (K, multiple \tau(s), and multiple training parameters of XGboost). The paper should conduct ablation experiments on the robustness against more parameter settings, since it is unlikely to have a good validation set in real scenarios.

### Questions
please refer to weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
