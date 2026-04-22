# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Identifying how training samples influence/impact Large Language Model (LLM) decision-making is essential for effectively interpreting model decisions and auditing large-scale datasets. Current training sample influence estimation methods (also known as influence functions) undertake this goal by utilizing information flow through the model via its first-order and higher-order gradient terms. However, owing to the large model sizes of today consisting of billions of parameters, these influence computations are often restricted to some subset of model layers to ensure computational feasibility. Prior seminal work by Yeh et al. (2022) in assessing which layers are best suited for computing language data influence concluded that the first (embedding) layers are the most informative for this purpose, using a hypothesis based on influence scores canceling out (i.e., the cancellation effect). In this work, we propose theoretical and empirical evidence demonstrating how the cancellation effect is unreliable, and that middle attention layers are better estimators for influence. Furthermore, we address the broader challenge of aggregating influence scores across layers, and showcase how alternatives to standard averaging (such as ranking and vote-based methods) can lead to significantly improved performance. Finally, we propose better methods for evaluating influence score efficacy in LLMs without undertaking model retraining, and propose a new metric known as the Noise Detection Rate (NDR) that exhibits strong predictive capability compared to the cancellation effect. Through extensive experiments across LLMs of varying types and scales, we concretely determine that the first (layers) are not necessarily better than the last (layers) for LLM influence estimation, contrasting with prior knowledge in the field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors study one particular problem in influence function approximation, i.e., which layer to focus on. They invalidated the existing claim that the first layer is the best and proposed some method to select the layer (based on two newly proposed metrics NDR and AUC). Multiple empirical studies are conducted to answer 4 research questions.

### Strengths
On the positive side, I enjoy reading the draft for the following reasons. 

First, the problem of identifying influential samples is an interesting and important one and in fact existing approaches are less than satisfactory. 

Second, some of the results are interesting and relevant, such as those for answering RQ1 and RQ2 (the empirical part).

### Weaknesses
On the less positive side, the draft can be improved from the following aspects.

First, the authors seem to be unaware of some recent work questioning the usefulness of the influence function for LLMs (“Do Influence Functions Work on Large Language Models?”). If we were to trust the empirical results (which is not unlike some of your observations) and reasoning in that paper, there is in fact little point in patching this existing influence function estimation method. Kindly discuss. 

Second, some of the empirical results and conjectures are rushed and may not be reliable from what I read, given the many variables for such studies, such as the data distribution and the optimization algorithm and so on. 

The following are some detailed comments.

Section 2: RELATED WORKS

Comment: Some important recent work is missing such as “Do Influence Functions Work on Large Language Models?” (EMNLP 2025).

Page 4: “Additionally, our proposed Rank method ignores incorrectly predicted validation samples …”

Comment: Kindly explain why.

Page 5: “[RQ4]. How reliably can … ”

Comment: Without first introducing what NDR is, stating this RQ here seems not useful for the readers.

Page 7: Figure 2

Comment: How reliable are these results given the many details involved in this kind of experiment (such as the data distribution, the performance of the optimization algorithm)?

Page 7: “For Llama, influence functions perform worst among all settings: none of the
configurations surpass the uniform removal baseline …”

Comment: This perhaps questions the usefulness of influence functions for LLMs fundamentally. 

Page 8: “DataInf and Cosine frequently outperform TracIn, particularly on early–middle attention layers of deeper models, suggesting that these layers encode representations that are most informative for noise filtering.”

Comment: Given the limited scale of the experiments and the complexity of the problem, it is not clear how strong this empirical evidence is. 

Page 9: “We validate the hypothesis that NDR and AUC can serve as reliable proxy metrics
for influence estimation by …”

Comment: Kindly explain why intuitively these are good estimations.

### Questions
How do you justify the relevance of your work in the presence of the recent observation made in "Do influence functions work for LLMs"?

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
3

### Summary
This paper challenges the assumption that embedding layers are best for computing influence functions in LLMs, showing that middle attention layers often perform better and that the "cancellation effect" metric is unreliable. The authors propose novel aggregation methods (Rank and Vote) that outperform standard averaging, and introduce proxy metrics (Noise Detection Rate and AUC) to evaluate influence functions without costly retraining, demonstrating through experiments on GLUE benchmarks across multiple LLMs that their approaches significantly improve detection of detrimental training samples.

### Strengths
1. The paper provides both theoretical (Theorem 5.1) and empirical evidence challenging prior assumptions about optimal layers for influence estimation.

2. The paper introduces well-motivated aggregation strategies (Rank and Vote) that outperform standard averaging, revealing layer-specific behaviors and improving influence estimation performance.

### Weaknesses
**1. Limited Evaluation Setting**: The experiments rely solely on synthetically injected label noise (20% uniform flipping) on GLUE benchmarks, which may not reflect real-world data quality issues.

**2. Inconsistent Results Across Models**: The findings show notable inconsistencies, particularly for LLaMA-3.2 1B where influence functions fail to outperform random filtering, and the best-performing layers vary across models, suggesting the conclusions may not generalize.

### Questions
1. How do the proposed methods perform on real-world noise beyond synthetic uniform label flipping?

2. Why do influence functions fail on LLaMA-3.2 1B, and how to select layers given different LLM architectures?

### Soundness
2

### Presentation
2

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
This paper shows that middle attention layers, not the first embedding layers, are most effective for estimating data influence in LLMs. It proposes new aggregation methods and a Noise Detection Rate metric, achieving better and more reliable influence estimation across multiple models and datasets.

### Strengths
1. The paper is clearly written and well organized, making it pleasant and easy to follow.

2. The work challenges the established conclusion [1] that the embedding layer is the most informative for LLM data influence estimation. The authors provide both theoretical and empirical analyses showing that the "gradient cancellation effect" can be unreliable in practice, and offer a counterexample (Theorem 5.1). This contributes a fresh perspective and theoretical insight to the field of LLM interpretability.

3. The motivation is clearly presented and supported by experiments. The proposed method is conceptually simple, easy to implement, and practically useful.

4. The experimental setup is comprehensive, covering multiple datasets and LLM architectures. It also evaluates several influence estimation methods, which strengthens the empirical analysis.

5. The authors have provided open-source code to facilitate reproducibility, which is highly appreciated.

```
[1] Yeh, Chih-Kuan, et al. "First is better than last for language data influence." Advances in Neural Information Processing Systems 35 (2022): 32285-32298.
```

### Weaknesses
1. The current baselines are mostly classical. Comparing with more recent influence estimation approaches would make the study more comprehensive and convincing.

2. Some recent studies [2] restrict gradient computation to specific layers for efficiency reasons. Computing influence across all layers in large LLMs could be computationally expensive. A discussion or quantitative analysis of the computational cost of the proposed approach would strengthen the paper.

3. The current analysis primarily reports overall performance metrics. Including visualizations that illustrate how influence rankings of individual samples differ across layers would provide valuable interpretability.

4. The intuition or mechanism explaining why middle layers perform better remains somewhat underexplored. A deeper discussion would be helpful.

5. The paper could also benefit from a more detailed analysis of how architectural differences between LLMs affect the observed influence patterns and performance differences.

```
[2] Chhabra, Anshuman, et al. "Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models." International Conference on Machine Learning (2025).
```

### Questions
How sensitive is the positional voting approach to k?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies which LLM layers are informative for sample-level influence and proposes simple cross-layer aggregation (Rank/Vote) to combine per-layer scores. It further uses no-retrain proxies (e.g., NDR/AUC) to pre-screen configurations across models and tasks.

### Strengths
- Addresses a practical question (which layers to use and how to aggregate) with simple, general aggregation operators (Rank/Vote).
- Provides broad empirical evaluation across models/tasks and includes no-retrain proxies (NDR/AUC) that are useful in practice.

### Weaknesses
- **Related-work positioning should be strengthened.** Add a concise paragraph clarifying scope vs. **knowledge editing** (e.g., ROME, MEND, MEMIT):  Explicitly discuss the differences and connections in terms of **“where” (layers/locations)** and **“how” (locality vs. cross-layer aggregation)**. **No new experiments are required**—a brief positioning and citations will suffice.

- **Novelty is relatively simple.** (Aggregation is straightforward; theory is light.)

### Questions
Same with weakness

### Soundness
2

### Presentation
3

### Contribution
3
