# TelecomTS: Observability Dataset for Multi-Modal Time-Series and Language Analysis

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 4, 8

## Abstract
Modern enterprises generate vast streams of time series metrics when monitoring complex systems, known as observability data. Unlike conventional time series from domains such as weather, observability data are zero-inflated, highly stochastic, and exhibit minimal temporal structure.  Despite their importance, observability datasets are underrepresented in public benchmarks due to proprietary restrictions. Existing datasets are often anonymized and normalized, removing scale information and limiting their use for tasks beyond forecasting, such as anomaly detection, root-cause analysis, and multi-modal reasoning. To address this gap, we introduce TelecomTS, a large-scale observability dataset derived from a 5G telecommunications network. TelecomTS features heterogeneous, de-anonymized covariates with explicit scale information and supports a suite of downstream tasks, including anomaly detection, root-cause analysis, and a question-answering benchmark requiring multi-modal reasoning.  Benchmarking state-of-the-art time series, language, and reasoning models reveals that existing approaches struggle with the abrupt, noisy, and high-variance dynamics of observability data. Our experiments also underscore the importance of preserving covariates’ absolute scale, emphasizing the need for foundation time series models that natively leverage scale information for practical observability applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
A real-world observability dataset in time series domain, de-anonymized and without normalization.

### Strengths
1. Built on a real-world physical system.
2. De-anonymized.
3. No normalization, preserving the original data scale.
4. High resolution.
5. Synthetic anomalies appear reasonable, supported by (i) a literature review and (ii) human evaluation;however, please note **this is not a human-in-the-loop verification process, it is an open procedure without feedback**.
6. Good insight: in certain threshold-triggered systems, the erratic, zero-inflated nature of these series makes forecasting less critical; anomaly detection and multi-step reasoning for downstream decision-making are more important.

### Weaknesses
1. The paper feels incomplete. It claims to include a Q&A dataset, but there appear to be only 64 time-series Q&A and 5 network Q&A samples (as shown in Table 1). I may be misunderstanding, but **if Network QA indeed has only 5 samples, that is far too few;** results based on such a small sample are not statistically meaningful.
2. There is an apparent inconsistency: the paper argues that the erratic, zero-inflated nature of these series makes forecasting less critical, yet the experiments still focus on forecasting (Section 4.4). This conflicts with the Introduction; **this work should focus on downstream control for network congestion and anomaly fixing rather than prediction.**
3. In Section 4.1, all models perform poorly, which raises **concerns about data quality**. If the dataset contains substantial noise, you should place greater emphasis on preprocessing and provide insights into data quality and preprocessing. ***A dataset alone, without meaningful insights, is engineering rather than research.***
4. A similar issue appears in Section 4.2. The anomaly duration analysis essentially reduces task difficulty; I do not think this setting is necessary. Instead, focus more on preprocessing to **ensure data quality**.
5. Root cause analysis should include more details.

Overall, the paper is ***incomplete and lacks useful insights***; I mainly see engineering effort rather than research contributions.

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a telecom observability benchmark built from high-cadence (multi-KPI) device/base-station traces, including induced conditions (mobility, congestion) and a real jammer setup. The authors procedurally synthesize anomaly segments from literature-derived types and KPI-level symptom mappings, and use LLMs to generate ticket text/templates. The benchmark evaluates anomaly detection, root-cause classification, forecasting, and QA, with LLM and time-series model baselines.

### Strengths
- Thorough real-world data collection and setup: The authors instrument 18 KPIs at high cadence, align trace, spatial zones, vary mobility and congestion, with a real jammer. I think this setup is thorough.

- Broad task suite: The work evaluates anomaly detection, anomaly-type classification, forecasting, and QA with clear context settings and failure analyses.

- Insight on scale handling: Results highlight the importance of preserving absolute scale for this specific type of domain. Models that encode local mean/std (Mantis-style features) perform robustly.

- Principled synthetic pipeline: Anomaly types are grounded in telecom literature with KPI-symptom mappings, parameterized timings, and human verification, providing controllable coverage beyond rare real incidents.

### Weaknesses
1. QA tasks emphasize statistic extraction rather than reasoning.  

   Labels for several QA tasks are deterministic outputs of simple statistics (means/variances, FFT periodicity, linear trend), so tasks can be solved by lightweight scripts or classifiers. Even for the root cause analysis, it can be framed as a 1 step classification tasks. I don't think these tasks strongly probe multi-hop or cross-channel reasoning.

2. Ambiguity between LLM performance and LLM usage.  

   The paper reports that GPT-style models underperform on anomaly detection, yet LLMs are used in the pipeline (for ticket/text generation). This invites confusion unless the boundary is stated explicitly and the contribution of tickets to downstream performance is measured through ablations (see more in question 2).

3. Missing ultra-simple baselines and efficiency context.  
   Strong but simple baselines such as DLinear ([arXiv:2205.13504](https://arxiv.org/abs/2205.13504)), AutoAR ([arXiv:2411.02796](https://arxiv.org/abs/2411.02796)), per-KPI linear regression ((https://openreview.net/pdf?id=wfyc8vLcq0)), and naïve last/seasonal predictors are absent as comparison models. Given recent evidence that simple models can match or beat heavier foundation models in time series, this limits interpretability; compute/latency reporting is also desired.

4. Prompting & evaluation fragility for LLMs

    For anomaly/RCA experiments, models are given optional context about KPI behaviors and asked to output a strictly formatted "conclusion block" for regex parsing. Results may be highly prompt-sensitive; mis-formatting can incur evaluation penalties unrelated to capability. (Please report prompt ablations and parsing robustness.)

### Questions
- How do the authors quantitatively validate real-vs-synthetic similarity (e.g., KS/MMD on KPI features, or train-on-synthetic→test-on-real and the reverse)?
- The duration/inter-arrival processes are modeled as exponential; how sensitive are results to that choice? Were alternative priors considered?
- Please clarify the boundary: LLMs generate only tickets/templates without attending to the time series right? If so, does adding tickets measurably improve multimodal performance versus a no-ticket ablation?
- An external validation of some of your key insights would help (for example, does scale handling helps on an external observability dataset as well?)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces TelecomTS, a large-scale observability dataset from a 5G telecommunications testbed, aiming to fill a critical gap in public time series datasets used for evaluating representation learning in the context of observability. TelecomTS features de-anonymized, heterogeneous covariates with explicit scale information and supports various downstream tasks. Through extensive benchmarks, the authors show that even powerful time-series foundation models and LLMs perform poorly in this regime, rationalizing the unique, under-explored challenges.

### Strengths
**Fills a substantial gap in public benchmarks**
TelecomTS addresses the scarcity of openly available, realistic observability datasets by releasing a resource with explicit scale, real and synthetic anomalies, and a range of tasks.

**Comprehensive downstream tasks**
The dataset supports not just classic time-series tasks, but also anomaly detection, anomaly duration localization, root cause analysis, and question-answering tasks.

**Extensive description and explanation of the data curation process**
The paper carefully documents the construction of the dataset, from 5G network setup and controlled jamming for real anomalies to principles for synthetic anomaly generation and diverse Q&A template construction.

### Weaknesses
- No direct comparison between normalized data and raw data with scale information.
- Lack of comparison (conceptual or quantitative) with datasets in other domains, especially regarding temporal dependencies.
- The scale of the QA portion is far from yielding significant comparison results.

### Questions
1. Could the authors provide a small-scale direct ablation or controlled experiment quantifying the performance improvement within a single model architecture when the absolute scale is preserved or not?
2. Do authors have a plan to expand the number of QA pairs?

### Soundness
3

### Presentation
3

### Contribution
3
