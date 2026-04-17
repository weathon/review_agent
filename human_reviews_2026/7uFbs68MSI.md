# Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring.

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We propose a post-hoc adaptive conformal anomaly detection method for monitoring time series that leverages predictions from pre-trained foundation models without requiring additional fine-tuning. Our method yields an interpretable anomaly score directly interpretable as a false alarm rate (p-value), facilitating transparent and actionable decision-making. It employs weighted quantile conformal prediction bounds and adaptively learns optimal weighting parameters from past predictions, enabling calibration under distribution shifts and stable false alarm control, while preserving out-of-sample guarantees. As a model-agnostic solution, it integrates seamlessly with foundation models and supports rapid deployment in resource-constrained environments. This approach addresses key industrial challenges such as limited data availability, lack of training expertise, and the need for immediate inference, while taking advantage of the growing accessibility of time series foundation models. Experiments on both synthetic and real-world datasets show that the proposed approach delivers strong performance, combining simplicity, interpretability, robustness, and adaptivity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a post-hoc, model-agnostic adaptive conformal anomaly detection framework that leverages pre-trained Time Series Foundation Models without additional fine-tuning.  Experiments on a total of seven benchmark datasets exhibit consistent superior performace over existing methods.

### Strengths
The paper focuses on key pain points in industrial time series anomaly detection such as limited high-quality data, lack of training expertise, and non-stationary data distributions. These are critical challenges in fields like predictive maintenance.  Experiments are comprehensive and well-designed that cover a total of seven benchmark datasets.

### Weaknesses
One caveat is that this paper focuses exclusively on univariate time series anomaly detection, without extending it to multivariate time series scenarios, despite multivariate time series are common in industrial monitoring and predictive maintenance.  In addition, it provides no details on computational overhead or scalability of the proposed method. For practical scenarios requiring real-time monitoring, computational costs are critical and could limit practical deployments.

### Questions
Can the proposed method be extended to deal with multi-dimensional time series data?

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
3

### Summary
This paper presents **W1-ACAS** (1-Wasserstein Adaptive Conformal Anomaly Score), a post-hoc adaptive conformal anomaly detection method for monitoring time series using pre-trained foundation models without requiring fine-tuning. The key contributions include: (1) producing interpretable anomaly scores directly linked to false alarm rates (p-values), enabling transparent threshold selection; (2) using weighted quantile conformal prediction that adapts online to distribution shifts by learning optimal weights via Wasserstein distance minimization; (3) providing a model-agnostic solution that works seamlessly with any time series foundation model while preserving theoretical guarantees; and (4) addressing practical industrial challenges such as limited data availability, lack of training expertise, and need for immediate deployment. Experiments on synthetic and real-world benchmark datasets demonstrate that W1-ACAS achieves strong detection performance with better calibration and lower false positive rates compared to baseline methods, while maintaining robustness to temporal distribution shifts.

### Strengths
1. Interesting idea of improving TSFMs' performances on anomaly detection (even if the TSFMs are trained for forecasting)
2. Solid mathematical derivations

### Weaknesses
1. Missing LLM usage section
2. The baselines mainly consist of TSFM-based baselines. Some of these TSFMs are not specifically trained for anomaly detection, so it's easy to improve their performances.
3. Should also include more recent (after 2023) deep-learning based baseline.
4. Improvement not that impressive give the high standard deviation

### Questions
1. What is the fundamental difference between this paper and [1]
2. Does this also work on fine-tuned non-FM anomaly detection model?
3. The TSFMs are forecasting models. They are not idea for contextual anomalies. Does this method help to alleviate that disadvantage?
4. Is this applicable to reconstruction-based models.




[1]  Margaux Zaffran, Olivier Féron, Yannig Goude, Julie Josse, and Aymeric Dieuleveut. Adaptive conformal predictions for time series. In Proceedings of the ICML. PMLR, 2022.

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
This work introduces a post-hoc adaptive conformal anomaly detection method for time series data, while combining predictions from pre-trained foundation models without fine-tuning.

### Strengths
- Sound approach that builds on the conformal prediction for non-exchangeable scenarios formulation (Barber, 2023) for a concrete application in time series anomaly detection
- A good number of datasets considered in the evaluation
- Promising results

### Weaknesses
Within the baselines, the paper could consider a broader number of competitors, encompassing classical and deep learning based. All the considered baselines report very poor results, although there are well-established and well-performing AD methods in the literature. One gets the impression that the evaluation is not fully fair. The authors may consider [1] for a starting point. 

[1] https://doi.org/10.1016/j.patcog.2022.108945

### Questions
- Table 1 provides a summary of the performance of the proposed method over all the datasets. While the full table appears in the appendix, it would be helpful to include a per-dataset analysis in the main text to provide readers with insights into which scenarios the proposed method performs best/worst.
- It is surprising to see that the baselines have a drop in performance when using point-adjusted F1. Usually, this is not the case. Methods tend to have a higher PA-F1 than a standard one. Could you explain?
- From the results, it seems there is no clear best foundation model to be coupled with the proposed method. It would be nice to discuss this. Where do the differences in performance metrics stem from? A per-dataset analysis, as previously suggested, could provide insights.
- How sensitive is the proposed method to the different hyperparameters that need to be adjusted?

### Soundness
3

### Presentation
2

### Contribution
3
