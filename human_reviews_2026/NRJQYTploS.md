# RockTS: Robust Time Series Forecasting based on Information Bottleneck and Optimal Transport

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 4, 6, 4, 4

## Abstract
Time series forecasting plays a crucial role in numerous real-world applications. Existing works mostly assume clean and regular historical sequences for predicting future ones. However, real-world time series data often contain anomalous subsequences that deviate from the normal patterns of the entire series, posing challenges to accurate forecasting. In this paper, we propose RockTS, a novel end-to-end framework for robust time series forecasting based on Information Bottleneck and Optimal Transport, which integrates the detection and imputation of anomalous subsequences into the forecasting task through a unified optimization objective. RockTS first introduces a detection process for anomalous patterns based on Information Bottleneck, which compresses representations of time series while retaining the information more relevant for effective forecasting. It then imputes the detected anomalous regions with normal patterns through a novel reconstruction strategy based on Optimal Transport for forecasting. Experiments on multiple real-world and synthetic datasets demonstrate that RockTS achieves superior robustness and forecasting performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents RockTS, a novel, end-to-end framework designed to tackle the critical real-world problem of robust time series forecasting by jointly addressing data anomalies. The core contribution is an integrated pipeline that utilizes the Information Bottleneck principle to effectively detect anomalous subsequences and subsequently employs Optimal Transport to perform principled and robust imputation, demonstrating significant performance gains over traditional forecasting methods when dealing with noisy and contaminated datasets.

### Strengths
1. The paper's motivation to jointly handle anomaly detection and robust time series forecasting within a single, unified framework is highly novel and valuable, addressing a critical limitation of existing works that often assume clean data.

2. The proposed methodology, which strategically employs the Information Bottleneck for detection and Optimal Transport for imputation, is theoretically well-grounded and logically sound, providing a strong, principled foundation for the model's robustness.

3. The experimental section is relatively comprehensive, covering various real-world datasets and successfully demonstrating the superiority of the proposed RockTS model over several strong baseline models, particularly under conditions of data contamination.

### Weaknesses
1. The ablation study is thorough but could be strengthened by replacing the Information Bottleneck and Optimal Transport components with more recent and advanced alternatives from the literature on time series anomaly detection and imputation, respectively, to better validate the specific design choices of RockTS.

2. The necessity of the imputation step should be more rigorously justified; specifically, the authors should explore whether irregular time series forecasting models could be directly applied to the sequences after masking anomalies, given that Optimal Transport does not introduce new information. Furthermore, the authors should discuss the potential impact of the OT-based imputation method on the general field of irregular time series prediction.

3. Given that the paper is submitted to ICLR 2026, the comparison against baselines like iTransformer (a 2024 ICLR submission) appears slightly dated; the authors must include comparisons with more recent and state-of-the-art models published in major conferences in 2024 and 2025 to ensure a fair and contemporary evaluation.

### Questions
Please refer to Weaknesses.

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
This paper proposes RockTS, an end-to-end framework for robust time series forecasting that addresses the challenge of anomalous subsequences in real-world data. RockTS integrates an Information Bottleneck-based detector to identify and mask anomalous subsequences, and an Optimal Transport-based reconstruction module to impute these regions with normal patterns. The entire process is jointly optimized with the forecasting task, resulting in improved robustness and prediction accuracy. Experiments on real and synthetic datasets demonstrate the superior performance of RockTS over existing methods.

### Strengths
S1. RockTS is the first end-to-end framework that directly tackles anomalous subsequences in time series forecasting.
S2. It introduces an adaptive Information Bottleneck-based detector to identify and retain only forecasting-relevant regions.
S3. The Optimal Transport-based reconstruction strategy effectively imputes masked regions while preventing the re-emergence of anomalies.

### Weaknesses
W1. The claimed novelty is limited: RockTS mainly composes an IB-based mask and an OT-based imputer on top of existing predictors, so theoretical or algorithmic breakthroughs over prior denoising/imputation work are unclear.

W2. The mask and reconstruction are tightly coupled, so mask errors directly propagate to imputation and forecasting, yet the paper lacks analysis of robustness to wrong masks.

W3. The paper favors hard binary masks via Gumbel-Softmax but provides no direct empirical comparison to soft/probabilistic weighting, leaving the necessity of discretization unproven.

W4. The necessity of OT-based reconstruction is not fully justified. The authors do not compare to simply deleting masked subsequences or using much simpler imputation strategies across tasks.

W5. The training objective leverages future targets to shape masks (IB predictiveness term), raising concerns about train–inference mismatch and potential information leakage that may inflate reported gains.

### Questions
Q1. Can the authors clarify and quantify RockTS’s theoretical or empirical advantages over prior denoising+imputation combinations, and highlight any unique insights beyond composing IB and OT?

Q2. How sensitive is end-to-end performance to mask false positives/negatives; can you provide a robustness study or mitigation (e.g., uncertainty-aware masks, mask calibration)?

Q3. Can you report experiments replacing hard binary masks with soft probability weights (and hybrids) to show whether discretization indeed improves forecasting and imputation?

Q4. Have you compared OT reconstruction to the simple alternative of dropping masked segments or to lightweight imputers, and what is the marginal benefit versus computational cost?

Q5. How do you prevent train-time use of future y from creating leakage or unrealistic masks; what are results if masks are learned without access to y or with strictly causal training?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RockTS, a robust method for time series forecasting. Specifically, the authors adopt a detection-imputation pipeline. First, anomalies in the time series are detected based on the Information Bottleneck principle; then, the anomalous segments are imputed using normal patterns derived from optimal transport. Experiments on both synthetic and real-world datasets demonstrate that RockTS achieves better performance than baseline methods.

### Strengths
1. The proposed methods are both sound and novel, as they employ the Information Bottleneck principle to detect anomalies in time series.

2. Visualization results demonstrate that RockTS has strong reconstruction capabilities for anomalous time series.

3. Experiments indicate consistent and significant improvements over all baseline methods.

### Weaknesses
1. In the experiments, the authors primarily compare methods designed for general time series forecasting, rather than those specifically developed for robust time series forecasting, such as the robust forecasting approaches discussed in the related work section.

2. There is a lack of ablation studies on hyper-parameters, such as the value of $\alpha$ in Equation (15).

### Questions
1. For the imputation part, what would be the outcome if we impute the time series using simple approaches such as linear interpolation? I raise this question because it can help investigate whether detection or imputation is more important for robust time series forecasting.

### Soundness
3

### Presentation
3

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
This work proposes an end-to-end robust forecasting framework RockTS that integrates anomalous-subsequence detection and imputation directly into the forecasting objective. It employs an Information-Bottleneck (IB) mask to eliminate prediction-irrelevant segments and uses an Optimal-Transport (OT) refill to reconstruct the masked areas into normal patterns while preventing the reappearance of anomalies.

### Strengths
1. It is reasonable to explicitly incorporate anomalous subsequences into the main objective of time series prediction, rather than removing them during preprocessing or relying on robust loss functions.

2. The combination of Information Bottleneck (IB) to generate masks based on predictive correlations and Optimal Transport (OT) to fill in values and prevent abnormal reproduction is novel.

3. The paper is well organized and presents the method with clarity.

### Weaknesses
1. Despite being proposed as a robust method, it has not been evaluated under a wide range of abnormal scenarios, such as varying noise levels and different lengths of anomalous segments.

2. The baseline comparison primarily includes standard forecasting models, while robustness-oriented methods are represented only by TAFAS.

3. The paper reports inference time per sample but does not provide a theoretical analysis of the algorithm’s time complexity.

### Questions
1. Additional experiments with robust baselines under diverse noise conditions are needed to validate the claimed robustness.

2. Which specific components or stages are included in the reported inference time per sample?

3. The paper should include a theoretical analysis of the model’s time complexity.

4. Given that the appendix claims the model can be deployed in a real-time environment, what is its memory overhead on GPU and CPU?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Real-world time series data often contain anomalous subsequences that deviate from the regular patterns of the entire sequence, posing challenges for accurate forecasting. This paper proposes a novel end-to-end framework for robust time series forecasting called RockTS, which first introduces an anomaly pattern detection process based on the information bottleneck, which compresses the representation of the time series while retaining information more relevant to effective forecasting. Then, it reconstructs the detected anomalous regions into normal patterns  based on optimal transport for prediction.

### Strengths
(1) It introduces an adaptive detector based on the information bottleneck to detect anomalous subsequences in the prediction process and retain prediction-relevant regions. 

(2) It designs a reconstruction strategy based on optimal transport (OT) to fill in the detected anomalous regions with normal data patterns. 

(3) Experiments show that ROCKTS outperforms classic time series forecasting algorithms, and ablation experiments demonstrate the effectiveness of the proposed modules.

### Weaknesses
(1) The code implementation does not seem to align well with the paper's description. For example, the paper mentions using MSE loss, but it appears that MAE loss is used in the code.

(2) There are concerns about unfair comparisons. (i) It seems that some methods' results are referenced from published papers, but most of them use MSE loss, while the authors' method uses MAE loss. (ii) Some results differ significantly from the published methods at length 512. For example, the average of 512-PatchTST-ETTh1 over four prediction lengths is 0.331 form its paper, while the authors report 0.351. 

(3) The baselines used are relatively old, and there is a lack of more recent baselines, such as CATS and DeformableTST. 

(4) The advantages on large-scale datasets such as Electricity, traffic, and weather are not obvious. For example, on traffic, it is even worse than PatchTST. Since the authors' code is mainly modified from PatchTST, this suggests that the newly added modules may be harmful for some datasets. After artificially injecting noise into the dataset, the accuracy of ROCKTS deteriorates severely compared to the original ROCKTS. If the ROCKTS modules were truly effective, such a significant deterioration should not occur. Based on the above analysis, considering that the Gumbel softmax is unstable and difficult to optimize, and there are no explicit supervision information for anomaly detection in the paper, this raises concerns about the effectiveness and generalizability of the IB-based Detector.

(5) Anomaly detection is entirely unsupervised. We do not know whether the current sample contains anomalies. It is highly likely that normal patterns are mistakenly corrected. This may also be why it performs worse than the baseline model on large-scale datasets. Moreover, the current method forces the model to identify anomalies, which may not be reasonable and could lead to the risk of performance degradation.

(6) In terms of efficiency, compared with patchTST, the inference overhead nearly doubled on the weather, electricity, and traffic datasets, while the accuracy remained the same or even poorer than PatchTST. The efficiency trade-off does not seem to have yielded the expected performance improvement.

### Questions
The output mask of Gumbel softmax lacks explicit supervision information for anomalies and non-anomalies. Even under the implicit constraints of mutual information theory, it seems hard to imagine that it can effectively detect truly anomalous subsequences? For Gumbel softmax, it seems highly likely that normal patterns are misjudged as anomalies, leading to a deterioration in model performance.

### Soundness
2

### Presentation
3

### Contribution
2
