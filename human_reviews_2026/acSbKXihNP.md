# Hierarchical Periodic Stationarization for Non-stationary Time Series Forecasting

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Time series forecasting (TSF) has advanced rapidly through benchmark-driven competition. However, we find that state-of-the-art models struggle to predict even a simple long-period sine wave, despite ample training data. One reason is that existing benchmarks underrepresent the non-stationary characteristics prevalent in real-world time series, leading to misleading evaluations. Moreover, standard stationarization methods inherently introduce substantial information loss during the stationarization process. To investigate this, we introduce \textit{controlled} datasets that expose information loss incurred by standard z-normalization-based stationarization methods, widely used in TSF models. To address this limitation, we propose Hipeen, a hierarchical periodic stationarization method that achieves stationarization through representing the value into multiple periodic components, minimizing information loss. Hipeen, with a linear backbone, successfully forecasts highly non-stationary signals—controlled datasets and large-scale stock datasets—substantially outperforming current SOTA models (8 stationarization methods and 8 baselines), while maintaining strong performance on conventional benchmarks. Our results highlight the importance of preserving critical information during stationarization and provide a new approach for robust TSF in non-stationary environments. All code and models will be released in the final version.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper indicates that many state-of-the-art forecasting models still fail to predict even a simple long-period sine wave, largely because existing datasets underrepresent the non-stationary characteristics commonly found in real-world time series—leading to misleading forecasting. To address this, the authors introduce controlled datasets to reveal the potential information loss caused by the widely used z-score normalization method. Furthermore, they propose Hipeen, a hierarchical periodic stationarization technique that decomposes time series into multiple periodic components, thereby reducing information loss during stationarization. The proposed approach is thoroughly validated on synthetic, stock, and long-horizon forecasting datasets, demonstrating superior performance in MSE and MAE metrics.

### Strengths
1. The paper conducts extensive experiments across diverse datasets and scenarios, and further constructs tailored datasets to validate the motivation and observed phenomena.
2. The proposed approach offers insightful perspectives on handling non-stationarity, differing from traditional methods by emphasizing the importance of preserving critical information during the stationarization process.
3. The empirical validation is clear and comprehensive, with rich experimental evidence supporting the claims

### Weaknesses
1. Figure 1 appears overly crowded, which affects readability, and both axis labels are too small.
2. The current baselines are not well-suited for the stock forecasting task; I suggest including additional baselines specifically designed for financial or stock data.
3. Using only MSE and MAE for stock prediction lacks sufficient persuasiveness—metrics such as drawdown or risk-adjusted returns would better reflect real-world forecasting quality.
4. On the long-horizon forecasting benchmarks (Table 4), the improvements are not significant; notably, Leddam* performs even better, especially considering the low predictability of the Exchange dataset.
5. The paper could include more discussion of post-SAN works such as FAN and DDN, which also incorporate normalization and frequency-domain operations.

### Questions
see weeknesses.

### Soundness
2

### Presentation
2

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
This paper addresses the challenge of stationarization for non-stationary time series forecasting. Instead of relying on conventional z-normalization-based approaches, the authors propose a novel framework named Hippen, which represents each time series value as a combination of multiple periodic components. The authors conduct extensive experiments on both controlled synthetic datasets and real-world benchmarks across diverse data scenarios. The results demonstrate that Hippen achieves superior performance compared to existing stationarization methods.

### Strengths
- The paper introduces a novel and conceptually interesting approach to stationarization for non-stationary time series forecasting. By decomposing each value into multiple periodic components rather than applying standard z-normalization, the proposed Hippen framework aims to retain essential information typically lost in conventional preprocessing.
- Extensive experiments are conducted on both synthetic and real-world datasets, and the proposed method shows consistent and significant improvements across different data conditions and forecasting models.

### Weaknesses
- Some claims need further clarification:
  - One of the central claims of this paper is that gradients and absolute values are essential for non-stationary time series forecasting but are discarded by conventional stationarization methods. This argument requires further justification. For example, methods like RevIN restore both the mean and standard deviation after normalization, which effectively recover much of the original information. The paper should clarify in what specific sense such information is “discarded” and why Hippen preserves it better.

  - The mechanism by which the proposed transformation retains gradients and absolute values, and how this process alleviates the impact of non-stationarity, needs a clearer intuitive explanation. Currently, the experimental evidence is convincing, but the theoretical or conceptual reasoning behind the improvement is underdeveloped.

  - What does the hierarchical periodicity metioned in line 255 refer to? And why the cosine distance loss function help to account for it?

- It is recommended to include more investigation experiments:
  - In Appendix C.2.2, the ensemble mechanism with $E=16$ is mentioned. It would be informative to analyze the effect of different ensemble sizes $E$ on performance, to assess the robustness and scalability of the approach.

  - It would also be valuable to investigate whether Hippen can serve as a model-agnostic plugin, similar to RevIN, that can enhance the performance of various backbone forecasting models.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
2

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
This paper points out that current time series forecasting models struggle to predict even simple periodic functions such as sine waves. The authors argue that this arises from a gap between real-world non-stationarity and the stationary assumptions often used in training, and that standard stationarization techniques inevitably lead to information loss. To address this, the paper introduces new controlled datasets and proposes a hierarchical periodic stationarization method, Hipeen. Experiments show that applying this method to existing backbone and MLP models yields notable performance gains.

### Strengths
1. Clear and reasonable motivation, easy to understand.
2. Extensive experiments, including new datasets.

### Weaknesses
1. The writing and structure could better highlight the differences from prior work and the paper’s unique contributions. Related work discussion should also be more comprehensive.
2. Applying long-horizon forecasting techniques to short-term financial stock prediction is questionable, as most long-horizon TSF methods are not well-suited for such tasks.
3. The use of MSE/MAE metrics shows only moderate advantages on long-term tasks, making it difficult to fully demonstrate the superiority of the proposed method.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
