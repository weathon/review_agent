# BEDTime: A Unified Benchmark for Automatically Describing Time Series

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent works propose complex multi-modal models that handle both time series and language, ultimately claiming high performance on complex tasks like time series reasoning and cross-modal question-answering. However, they skip evaluations of simple and important foundational tasks, which complex models should reliably master. They also lack direct, head-to-head comparisons with other popular approaches. So we ask a simple question: *Can recent models even produce generic visual descriptions of time series data?*   In response, we propose three new tasks, posing that successful multi-modal models should be able to *recognize, differentiate,* and *generate* language descriptions of time series. We then create **BEDTime**, the first benchmark dataset to assess models on each task, comprising four datasets reformatted for these tasks across multiple modalities.  Using **BEDTime**, we evaluate 13 state-of-the-art models, and find that (1) surprisingly, dedicated time series foundation models severely underperform, despite being designed for similar tasks, (2) vision–language models are quite capable, (3) language-only methods perform worst, despite many lauding their potential, (4) all approaches are clearly fragile to a range of realistic robustness tests, indicating avenues for future work.*

*All of our code and data needed to reproduce our results will be made public.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BEDTime, a benchmark for evaluating language models' ability to recognize, differentiate, and generate natural language descriptions of univariate time series. The benchmark unifies four existing datasets (three synthetic, one real-world) and evaluates 13 models across three modalities: text-only LLMs, vision-language models (VLMs), and time-series language models (TSLMs). Key findings include: (1) VLMs substantially outperform other modalities, (2) dedicated TSLMs underperform despite being designed for similar tasks, and (3) all models show fragility under realistic perturbations like missing data and noise.

### Strengths
**S1. Addresses important evaluation gap**: Existing work evaluates LLMs, VLMs, and TSLMs in isolation using different datasets and metrics. This paper provides the first unified benchmark enabling direct comparison across modalities. The finding that VLMs outperform specialized TSLMs is surprising and valuable for the community.

**S2. Comprehensive experimental coverage**: The evaluation is exceptionally thorough. Tables 2-12 systematically present results across all conditions. This experimental rigor strengthens the empirical claims.

**S3. Valuable practical insights:** The finding that VLMs consistently outperform specialized TSLMs is surprising and actionable.

**S4. Clear presentation of results**: Tables 2-12 and Figures 2-5 systematically present results across conditions, making it easy to trace findings back to specific experimental settings.

### Weaknesses
**W1. Inadequate TSLM evaluation:** Only two TSLMs are evaluated: ChatTime and ChatTS. Appendix F reveals ChatTime has severe instruction-following failures with 61-83% refusal rates. This leaves only one functional TSLM. Drawing conclusions like "TSLMs severely underperform" from a single working model is unjustified. Recent surveys cite 10+ relevant TSLMs including Time-LLM, LLaTA, UniTime, and TEST. The omission of these models is a fundamental flaw. The paper's central claim about TSLM underperformance cannot be trusted without broader model coverage.

**W2. Unfair input representation across modalities:** VLMs receive matplotlib plots with axes, labels, and grids. LLMs receive raw comma-separated numbers. TSLMs get Python lists. These are not equivalent inputs. The matplotlib formatting provides structural cues (trend direction from slope, magnitude from visual height) that raw numbers lack. VLM superiority may reflect representation advantage rather than visual reasoning ability. The paper lacks ablations: What if VLMs received raw plots without annotations? What if LLMs received formatted tables? Without these controls, the fairness of comparison is questionable.

**W3. Over-reliance on synthetic data:** Three of four datasets are synthetic. The only real dataset (TRUCE-Stock) shows 20-30% lower accuracy than synthetic ones. The paper does not analyze why real data is harder or what properties cause difficulty. Claims about real-world applicability are weakened when 75% of evaluation uses synthetic data. The benchmark may not predict deployment performance.

**W4. NLI metric has unvalidated domain mismatch:** The generation task uses NLI models trained on general text to evaluate time series descriptions. Table 3 shows extremely low bi-directional entailment (2.65-14.41%). This indicates either poor generation quality or metric failure. The paper acknowledges domain mismatch but provides no validation that NLI scores correlate with human judgments. Without this validation, generation results may be unreliable.

**W5. Limited human evaluation:** Only 340 samples (24% of SUSHI) are human-evaluated by "research group members." This introduces potential bias. Standard practice uses external annotators on larger samples (1000+). The small sample size limits statistical power for comparing 13 models across 6 criteria.

**W6. Oversimplified negative sampling:** Distractors are selected as "most dissimilar" time series via DTW/Euclidean distance. This creates easy discrimination. Real errors involve subtle confusion between similar patterns. Table 7 shows 96-98% accuracy on SUSHI, suggesting ceiling effects. The benchmark may overestimate model robustness.

**W7. Critical results buried in Appendix:** Chain-of-Thought prompting shows 10-15% accuracy improvements (Figure 3) but appears only in Appendix E.2. ChatTime's 83% refusal rate appears only in Appendix F. These are major findings that should be in the main text.

**W8. Missing diagnostic analyses:** No class-stratified results (trends vs. seasonality vs. anomalies). No length-based breakdown despite variable sequence lengths. No error analysis of what mistakes models make. These missing analyses limit diagnostic value.

### Questions
Q1. Why is real data performance so much lower? TRUCE-Stock shows accuracy much lower than synthetic data. What properties of real series cause difficulty? Does synthetic data dominance limit real-world validity?

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
5

### Summary
In this paper, the authors proposed a new time series understanding benchmark datasets to evaluate the ability of LLM/VLM/TSLM to generate language descriptions of time series. By curating existing datasets, the authors create 3 tasks for evaluation: description recognition, description differentiation and open generation. Extensive experiments show the limitations of current LLM/VLM/TSLM.

### Strengths
S1: The paper proposed a new time series understanding datasets for the task of describing time series.
S2: Extensive evaluations of SOTA LLM/VLM/TSLM are performed.

### Weaknesses
W1: The paper did not talk much about how they ensure the quality of the generated benchmarks. For instance, the authors make sampling and select the most dissimilar option as the incorrect description. Human evaluation should be included to ensure the quality.
W2: Most of the included datasets are from existing work and they are mostly synthetic and small. For instance, the TRUCE dataset only contains 12 points. The largest one only contains 2048 points.

### Questions
Q1: Why not include more datasets into the benchmark. For instance, the one used in timeseriesexam and ChatTS.
Q2: The performance of VLM will be strongly affected by the resolution of the plotting of the time series. However, this part is not evaluated.
Q3: It seems that the questions are mostly from the caption of the time series data. However, the timeseries description task are frequently asked for local details. Can the proposed benchmark cover these aspects?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BEDTime, a unified benchmark designed to evaluate models' capabilities in describing time series data using natural language. The benchmark formalizes three tasks: (1) **Recognition** - true/false question-answering to assess whether a given statement about a time series is correct; (2) **Differentiation** - multiple-choice question-answering to select the most appropriate description; and (3) **Generation** - open-ended natural language description generation. The authors consolidate four recent datasets to enable direct model comparisons and evaluate 13 state-of-the-art models, including language-only models, vision-language models, and time-series language models.

The main findings indicate that language-only models perform poorly, vision-language models show promise, and pre-trained multimodal time-series language models outperform large language models but still have significant room for improvement. All methods demonstrate fragility in robustness tests. While the paper addresses an important need for standardized evaluation in time-series text multimodal understanding, several critical limitations regarding literature coverage, experimental design, and benchmark comprehensiveness prevent it from being a complete contribution to the field.

### Strengths
* **Well-defined task framework**: The formalization of three distinct tasks (Recognition, Differentiation, Generation) provides a clear structure for evaluating different aspects of time-series description capabilities. This taxonomy is useful for understanding model strengths and weaknesses across different reasoning types.
* **Comprehensive model evaluation**: The evaluation of 13 diverse models across different categories (language-only, vision-language, time-series language models) offers valuable insights into the current state of the field and reveals interesting patterns about which architectures are most effective.
* **Robustness testing**: The inclusion of robustness tests (though results showing fragility across all methods) is valuable for identifying limitations and areas for improvement in current approaches.
* **Unified benchmark creation**: The consolidation of four datasets into a single benchmark addresses the practical need for standardized evaluation, which could facilitate future research and fair comparisons.

### Weaknesses
## Weaknesses

### 1. Insufficient Literature Review and Comparative Analysis

The paper's literature review is significantly incomplete and lacks proper contextualization of the proposed work within the existing landscape of time-series text multimodal research. Specifically:

**Missing Comparative Table**: The paper does not provide a comprehensive table comparing existing datasets and benchmarks. Such a table should detail:

- What tasks each existing benchmark covers
- What data modalities and characteristics they use
- What evaluation metrics are employed
- What gaps exist that BEDTime aims to fill
- How BEDTime's contribution differs from prior work

Without this comparative analysis, it is difficult to assess the novelty and significance of the proposed benchmark. The authors should provide a detailed table (suggested format below) that clearly positions their work:

| Benchmark/Dataset      | Tasks                                    | Data Type     | Metrics                      | Key Contributions | Limitations | BEDTime's Addition |
| ---------------------- | ---------------------------------------- | ------------- | ---------------------------- | ----------------- | ----------- | ------------------ |
| [Existing Benchmark 1] | ...                                      | ...           | ...                          | ...               | ...         | ...                |
| [Existing Benchmark 2] | ...                                      | ...           | ...                          | ...               | ...         | ...                |
| BEDTime                | Recognition, Differentiation, Generation | Univariate TS | Accuracy, Generation metrics | Unified framework | ...         | ...                |

**Omitted Related Work**: The paper fails to discuss several highly relevant works that are directly related to time-series text multimodal understanding and benchmarking:

- **MTBench**: A comprehensive benchmark for multimodal time-series understanding that should be discussed and compared.
- **ITFormer**: Work on bridging time series and natural language for multi-task temporal-textual QA, which is highly relevant to the tasks proposed in this paper.
- **Time-MQA**: Multi-modal question answering for time series, which addresses similar problems and should be included in the related work discussion.

These omissions make it difficult to assess whether BEDTime offers meaningful advances over existing benchmarks or merely re-formalizes already-studied problems. The authors should provide a thorough discussion of how BEDTime relates to and improves upon these existing works.

### 2. Lack of Evidence for Downstream Task Correlation

The paper argues that descriptive capabilities are important, but provides **no empirical evidence** that performance on the proposed benchmark tasks correlates with improved performance on downstream applications. This is a critical gap for a benchmark paper, as the value of a benchmark is fundamentally tied to its ability to predict or correlate with real-world performance.

The paper should:

- Conduct experiments demonstrating correlation between BEDTime task performance and downstream task performance (e.g., forecasting, anomaly detection, classification)
- Provide theoretical or empirical justification for why these description tasks are predictive of broader time-series reasoning capabilities
- Discuss what specific downstream tasks would benefit from strong performance on recognition/differentiation/generation tasks

Without this validation, it remains unclear whether improving on BEDTime actually translates to improved real-world performance, which undermines the benchmark's utility.

### 3. Limitation to Univariate Time Series

The benchmark is exclusively evaluated on **univariate time series data**, despite the fact that multivariate time series are ubiquitous in practice and represent a fundamentally different and often more complex problem. This limitation is particularly concerning because:

- **Inter-dimensional correlations are fundamental**: The relationships between different dimensions in multivariate time series are often the most informative aspect of the data (e.g., cross-correlations in sensor networks, lead-lag relationships in financial data, spatial-temporal dependencies). A benchmark that ignores this critical characteristic may not capture the full complexity of real-world time-series understanding.
- **Architecture implications**: Models that work well on univariate data may not generalize to multivariate scenarios, where understanding relationships between dimensions is crucial. The current evaluation does not test whether the conclusions (e.g., "vision-language models perform well") hold in multivariate settings.
- **Limited practical applicability**: Many real-world applications involve multivariate time series. A benchmark that only considers univariate data has reduced practical value.

The authors should:

- Extend the benchmark to include multivariate time-series datasets
- Analyze whether the current findings (e.g., vision-language models outperforming language-only models) hold in multivariate scenarios
- Investigate whether understanding inter-dimensional relationships requires different capabilities that are not captured by the current tasks
- Discuss how the three proposed tasks (Recognition, Differentiation, Generation) should be adapted or extended for multivariate scenarios

This is not just a "future work" consideration—it is a fundamental limitation that calls into question the comprehensiveness of the benchmark as a foundational evaluation framework for time-series text multimodal understanding.

### 4. Limited Scope of Evaluation

The evaluation, while covering 13 models, could be more comprehensive:

- **Ablation studies**: More detailed analysis of why certain architectures perform better would strengthen the paper
- **Error analysis**: Systematic analysis of failure cases would help identify common patterns and areas for improvement
- **Cross-domain evaluation**: More analysis of how models generalize across the four different datasets would be valuable

### Questions
## Questions

1. **Literature comparison**: Can the authors provide a comprehensive table comparing existing time-series text multimodal benchmarks (including MTBench, ITFormer, Time-MQA, and others), clearly articulating what unique contribution BEDTime makes?
2. **Downstream task correlation**: Can the authors conduct experiments showing that performance on BEDTime tasks correlates with performance on downstream applications (e.g., forecasting, anomaly detection)? What is the theoretical or empirical justification for why description tasks are predictive of broader capabilities?
3. **Multivariate extension**: The paper focuses exclusively on univariate time series. Given that multivariate time series with inter-dimensional correlations are common in practice and represent a fundamentally different challenge, how do the authors plan to extend BEDTime to multivariate scenarios? Would the current conclusions about model performance change if evaluated on multivariate data?
4. **Task design rationale**: Why were these three specific tasks (Recognition, Differentiation, Generation) chosen? What is the theoretical or empirical justification that these tasks comprehensively evaluate time-series description capabilities?
5. **Dataset selection**: How were the four datasets selected? What criteria were used to ensure they are representative and sufficiently diverse?
6. **Evaluation metrics**: For the Generation task, what metrics are used to evaluate the quality of generated descriptions? How are semantic correctness, fluency, and relevance assessed?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a benchmark that evaluates a model's capability to recognize, differentiate, and generate language descriptions of time series. The benchmark aggregates four datasets and evaluates 13 models spanning three paradigms, including language models, vision language models, and time series language models. Results show that overall vision language models outperform the other two types.

### Strengths
1. The paper presents the first benchmark explicitly focused on language-based understanding of time-series characteristics, enabling apples-to-apples comparisons across modalities.

2. The paper compares a large set of models spanning three types of popular paradigms. It also conducts a series of robustness studies given e.g., missing values, varying scales and noise levels to provide insights on current methods.

### Weaknesses
1. The benchmark contains mostly synthetic datasets and statistical properties. It would be helpful to generalize to more real-world datasets, and incorporates semantic grounding, for example describing the potential domains that the time series come from (etiological reasoning).

2. The study primarily shows that existing models fall short, but does not propose methods to address these gaps. In addition, the evaluation focuses on zero-shot performance and does not consider fine-tuned models.

3. The ability of describing a time series does not necessarily translate to other tasks. For instance, visual encodings can capture global shape but lose numerical precision, so VLMs that excel at descriptions might still underperform on forecasting or numerically sensitive tasks.

### Questions
Minor: Better to show the overall rankings in Table 2 as well.

### Soundness
3

### Presentation
3

### Contribution
3
