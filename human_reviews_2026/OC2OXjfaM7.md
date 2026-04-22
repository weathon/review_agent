# Efficiently Estimating Data Efficiency for Language Model Fine-tuning

- Avg Score: 5.20
- Decision: Reject
- Scores: 4, 4, 8, 4, 6

## Abstract
While large language models (LLMs) demonstrate reasonable zero-shot capability across many downstream tasks, fine-tuning is a common practice to improve their performance. However, a task's \textit{data efficiency} --- i.e., the number of fine-tuning examples needed to achieve a desired level of performance --- is often unknown, resulting in costly cycles of incremental annotation and retraining. Indeed, we demonstrate across a curated set of 30 specialized tasks that performant LLMs may struggle zero-shot but can attain stronger performance after fine-tuning. This motivates the need for methods to predict a task's data efficiency \textit{without} requiring incremental annotation. After introducing a concrete metric that quantifies a task's data efficiency, we propose using the \textit{gradient cosine similarity of low-confidence examples} as a way to predict data efficiency based on a small number of labeled samples. We validate our approach on the collected set of tasks with varying data efficiencies, attaining 8.6 % error in overall data efficiency prediction and eliminating hundreds of unnecessary annotations. Our experiment results and implementation code are available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies how to predict a task’s data efficiency for LLM fine-tuning without exhaustive annotation. It (1) defines data efficiency as the area under the performance–data curve (AUC) normalized between zero-shot and “human-level” accuracy and then (2) proposes CoS-Low, the median pairwise gradient cosine similarity computed only on the lowest-confidence 10% of examples (sampled 32 points). (3) It maps CoS-Low to AUC via linear regression and reads off a predicted data budget for a target accuracy.
The authors validate the effectiveness of their method on 30 realistic specialized tasks.
This paper focuses on an interesting and practical topic, `data efficiency prediction`, but there are still unclear parts that need to be clarified by the authors.

### Strengths
1. **Lightweight method**: using a small set of data samples to effectively predict the AUC of the given task
2. **Diverse datasets in experiments**: using 30 tasks in experiments to support the effectiveness of the proposed method.
3. **Additional data and results**: providing more details and experimental results in the appendix for better presentation.

### Weaknesses
1. **Unclear research goal**: This paper focuses on the data efficiency of a task (i.e., the number of fine-tuning examples needed to achieve a desired level of performance) in the abstract. But the main content discusses how to estimate the AUC. One of my concerns is the relationship between AUC and data efficiency. How to use the AUC (a number) to estimate the number of fine-tuning examples when I want to fine-tune a given model to a certain accuracy? How can I use AUC to guide an efficient fine-tuning?
2. **Lack of discussion on generalization**: The experiments in this paper mainly use classification datasets with accuracy as the metric. However, the generation task covers many application scenarios of LLMs (e.g., summarization, translation, and questions-answering). It is unclear whether CoS-Low holds for generation tasks using BLEU / ROUGE as the metric. 
3. **Lack of theoretical supports**: This paper lacks an analysis of the effectiveness of its design and sufficient theoretical support. The authors claim that they select the top 10% of examples with the highest difficulty and lowest confidence because these examples are effective in improving model performance. However, existing research [1][2] on data selection/pruning shows that using only a small number of high-difficulty examples does not improve model performance as much as randomly selected data due to a lack of diversity. This clearly contradicts the authors' claims.

### Questions
1. What is the relationship between the data efficiency and the AUC in this paper? How to use the AUC to estimate the number of fine-tuning examples when users want to fine-tune a given model to a certain accuracy `Acc`?
2. What is the generalizability of Cos-Low on generation tasks and datasets? 
3. Please clarify the disparity between the observation data selection paper and the claims in this paper. How do the AUC predictions change when using randomly selected data samples compared to using the top 10% of data samples?
4. Typos:
   (1) Extra ] in line 128; (2) What does the `jje` mean in Line 1235? Is it an abbreviation from prior work?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of estimating the data efficiency—the number of examples required for a desired performance for fine-tuningLLMs. The authors first propose a formal metric for data efficiency based on the performance-data AUC. They then introduce CoS-Low, a novel predictor based on the gradient cosine similarity of low-confidence examples , which can be computed using only a small number of labeled samples. The authors empirically validate their approach across 30 downstream tasks , demonstrating that CoS-Low strongly correlates with the true data efficiency (Spearman correlation of 0.675) and achieves a low prediction error (8.6%).

### Strengths
1. Significant and Practical Problem: The paper addresses a significant and practical problem in the era of LLMs: efficiently estimating the data requirements for fine-tuning on downstream tasks. 
2. Novel Methodology: The approach of constructing a predictor for data efficiency, rather than relying on costly incremental annotation and retraining, is a novel and valuable perspective. The proposed CoS-Low metric, which leverages gradient cosine similarity among low-confidence examples, is an interesting and non-obvious choice.
3. Clarity and Writing: The paper is generally well-written, clear, and easy to follow.

### Weaknesses
1. **Inconsistent Definition of "Data Efficiency"**: The paper presents inconsistent definitions of its core concept. The abstract defines data efficiency as "the number of fine-tuning examples needed to achieve a desired level of performance". However, Section 3.1 formally defines it as the AUC of the performance-data plot. These two concepts are not consistent. The AUC measures the rate of performance gain, while the abstract's definition refers to a specific data budget required to hit a performance target. Arguably, the definition from the abstract is what practitioners actually care about, and the AUC is only a proxy for this. This inconsistency confuses the paper's primary claim.

2. **Unreliable Upper Bound for Performance**: The methodology relies on normalizing the performance curve using a "maximum attainable (human-level) performance" as the upper bound. This assumption is questionable for two reasons:
It is well-established that LLMs can and do exceed human-level performance on many benchmarks, making this an unreliable ceiling for model capability.
"Human-level performance" is itself a vague and often noisy metric, highly dependent on the expertise and consistency of the specific annotators for that dataset, rather than a true theoretical maximum. This arbitrary upper bound could introduce bias into the AUC calculation.

3. **Critical Omission of Data Sampling Methodology**: The reliability of the paper's ground-truth data efficiency curves is undermined by a critical lack of detail. The authors state they fine-tuned on various data budgets (e.g., 50, 100, 200, 500...), but fail to describe their sampling methodology. Crucial questions remain unanswered: How were these subsets selected? Were they random samples? Was the 100-sample set a superset of the 50-sample set, or an independent draw?
This is a significant omission. As evidenced by a large body of work on data selection, which studies how to achieve optimal results under a fixed data budget, the choice of examples (not just the quantity) dramatically impacts fine-tuning performance. By failing to discuss or define a systematic sampling strategy (e.g., reporting mean and variance over multiple random draws), the paper's ground-truth curves and their corresponding AUCs lack reliability.

4. **Lack of Actionable Insights**: The analysis stops short of providing truly actionable insights for practitioners. While the paper maps CoS-Low to an AUC and then to a predicted performance curve , it doesn't adequately "close the loop" by providing a clear answer to the user's question: "How many samples and which samples do I need to achieve a certain target performance?".

### Questions
1. Could you please clarify the exact methodology used to sample the data subsets for fine-tuning at different budget sizes (e.g., 50, 100, 500 samples)? How did you account for the significant performance variance known to be caused by which specific data points are selected, not just the quantity?

2. How should a practitioner practically translate the predicted AUC value into a concrete, actionable estimate for the number of samples required?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ReLiNet (Relational Linear Network), a novel architecture designed to improve relational reasoning and out-of-distribution (OOD) generalization in graph-structured data. Unlike standard Graph Neural Networks (GNNs) that rely on nonlinear message-passing schemes, ReLiNet employs a linear relational operator that is augmented with a structured residual correction term, yielding a hybrid model that retains interpretability while maintaining strong expressive power.

### Strengths
1. The paper takes a bold and interesting stance by revisiting linearity as a desirable inductive bias in relational models. This goes against the prevailing trend of ever more nonlinear message-passing architectures. The argument is well-justified both intuitively and empirically: simpler linear propagation can yield better extrapolation under graph shifts.
2. The authors provide clear derivations showing that ReLiNet’s linear relational operator can be viewed as a constrained instance of a first-order spectral filter. They further show connections to graph kernels and linear graph Laplacian smoothing, offering interpretability advantages rarely seen in modern GNNs.

### Weaknesses
1. Although ReLiNet excels on graph-structured data, its utility for non-graph relational reasoning (e.g., text, vision-language relational datasets) is not demonstrated. The claim that “ReLiNet generalizes to any structured relation learning task” feels overstated.
2. On the larger OGB datasets, the performance improvement over GAT and Graph Transformer baselines is modest (~0.5–1% absolute). While statistically significant, it may not be practically impactful without additional benefits like efficiency or explainability metrics.

### Questions
1. Could the authors clarify whether ReLiNet can incorporate edge features or dynamic edges without retraining the linear operator?
2. The residual nonlinear correction term resembles a low-rank adaptation layer—did the authors explore alternative parameterizations (e.g., adapters, attention-weighted skip connections)?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a method to predict a task's "data efficiency”, which is the amount of fine-tuning data needed for a large language model to reach a specific performance level. The goal is to make this prediction efficiently, using only a small number of labeled examples, thereby avoiding costly, large-scale data annotation. By analyzing 30 different tasks, the authors show that performance improves significantly after fine-tuning. They propose that the "gradient cosine similarity of low-confidence examples" (CoS-Low) is the most effective predictor. This metric is used in a simple linear regression model to estimate the performance curve and the required data budget for a new task.

### Strengths
+ The authors empirically demonstrate a strong correlation between their proposed CoS-Low metric and the actual data efficiency of a task, making it a reliable signal for prediction.
+ The study validates its core motivation by showing that across a diverse set of 30 tasks, fine-tuning consistently leads to significant performance improvements over the model's initial zero-shot capability, highlighting the practical need for such an estimation method.

### Weaknesses
- The analysis is capped at a 5000-example budget, so it is not clear how the method works beyond that point.
- The authors do not evaluate models on cross-benchmarks, which is hard to measure the impact of training on cross-domain. 
- The paper assumes that performance is a monotonically non-decreasing function of the data size. The authors acknowledge this is a simplification and state that in the rare cases where it wasn't true, they adjusted the data to enforce the assumption, which may not reflect all real-world scenarios.

### Questions
Why was a power function chosen to model the performance curve f(n), especially when the performance graphs in Figure 2 show varied shapes that are not all clearly exponential? Could the authors plot the predicted performance curve f(n) against the ground truth curve for a few examples?

The method aggregates per-sample gradients into a single median value (CoS-Low) to predict task-level efficiency. How much information is lost in this aggregation, and does it obscure how individual examples or subsets of data impact model learning?

Are the per-sample gradients computed on the model's final answer token(s) only, or across the entire generated response?


The CoS-Low metric exclusively uses the 10% of examples with the lowest model confidence. What is the justification for disregarding the learning signal from the remaining 90% of the data?


The regressor is trained using a "hold-one-out" setting, where all tasks except the target task are used for training. Given that each task has a unique learning curve, what is the rationale for assuming that a regressor trained on 29 diverse tasks can accurately predict the curve for a held-out one?

What is the source for the "known estimate of human-level performance" used as the maximum attainable performance for each task?

Selecting low-confidence examples for analysis seems related to curriculum learning or active learning. Was there any exploration of using this method to select data for the training set itself, rather than using randomly chosen data points for fine-tuning?

In Table 6, why does the CoS-Low metric computed on rank-64 LoRA gradients show a stronger correlation (0.675) to data efficiency than when computed on full model gradients (0.628)? This seems counterintuitive, as the full gradients should contain more information.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a lightweight method to compute task-specific data efficiency. The major contributions of paper include curating a multi-domain set of 30 tasks to study how the llama 8B model performs when finetuned over 50-5000 samples. The paper proposes a formal definition of data efficiency based on AUC and covers prediction strategies for a task's data efficiency- including Cos-Low - a median gradient cosine similarity based on low confidence examples. Finally, the paper shares experimental results of predicting data efficiency over these 30 tasks, evaluated through hold-one-out prediction.

### Strengths
1. The paper attempts to quantify task-specific data efficiency that currently relies on expensive trial and error and costly empirical tuning. Understanding the data efficiency can help wisely expend the data annotation budget per task. Moreover, this method requires no labeled validation data, unlike most domain adaptation or reweighting-based methods. 
2. The experimental setup is well motivated, with thorough ablations and multi-model validation (Llama-3.1, Mistral, Qwen). The proposed CoS-Low metric consistently outperforms baseline predictors in estimating data efficiency across 30 downstream tasks.
3. The work bridges several literatures (task difficulty, active learning, multitask gradient conflict) and contributes a simple yet interpretable signal (gradient cosine similarity on low-confidence examples).

### Weaknesses
1. The mapping from predicted AUC to performance curve relies on simplifying assumptions ( for instance, human-level saturation in 5k examples per task) that may not hold for long-tailed or complex tasks (for instance, the discussion on MMLU in Section 6).
2. Cos-Low may conflate data noise / out-of-distribution samples with genuine difficulty due to the reliance on low-confidence samples.
3. Task interaction effects remain unaddressed - modern LLMs are used in a multi-task setting and this approach can be improved by studying how gradient conflicts behave when multiple tasks are trained jointly.

### Questions
1. Can the authors share the relationship (if any) between Cos-Low to predict data efficiency and data valuation methods like influence functions or data valuation [1][2][3]
2. How would cross-task interactions affect CoS-Low’s predictive power when fine-tuning models on multiple tasks jointly, as is typical in instruction or tool-use tuning?
3. How stable is the metric under domain shift—for example, when low-confidence examples come from out-of-distribution data?


[1] GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration
[2[ Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models
[3] What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Function

### Soundness
3

### Presentation
2

### Contribution
3
