# Spectral Retrieval-Augmented Time-Series Forecasting

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Time series forecasting leverages historical patterns to predict future values, but traditional methods face challenges when dealing with complex, non-stationary patterns that are difficult to memorize during training. Retrieval-augmented approaches have emerged as promising solutions by retrieving similar historical patterns to enhance predictions. However, existing retrieval methods suffer from two fundamental limitations: spectral blindness, which overlooks critical frequency-domain characteristics that capture underlying periodic structures, and temporal recency, which treats all historical data equally without emphasizing recent, more relevant patterns. In this paper, we propose SpecReTF, a novel retrieval method that addresses these issues by converting time series into windowed frequency representations, measuring similarity with a combined metric that captures both amplitude and phase information. To balance recency and historical context, we apply an exponential moving average weighting scheme that emphasizes recent windows. Extensive experiments on benchmark datasets demonstrate that SpecReTF outperforms time-domain retrieval methods, achieving superior forecasting accuracy across diverse, non-stationary time series.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel frequency-domain-based similarity method, SpecReTF, for Retrieval-Augmented time series forecasting. The work aims to address two identified limitations in existing approaches: "spectral blindness" and "temporal recency". The motivation is intuitive, and the experimental setup is relatively comprehensive.

### Strengths
- The paper is clearly structured, making the proposed methodology easy to follow.
- The identification of "spectral blindness" and "temporal recency" provides an easily understandable motivation for the work.
- The results demonstrate the effectiveness of the proposed method against the chosen baselines.

### Weaknesses
1. The two primary solutions, while effective, appear somewhat heuristic and lack elegance or a learned approach. Specifically:
- In Equation (7), the amplitude and phase similarities are combined through a simple, unweighted summation. This approach may be suboptimal, especially given that these two similarity measures are on different scales and capture distinct aspects of the signal. The work would significantly benefit from exploring automated methods for adaptively tuning or weighting the contribution of amplitude versus phase similarity.
- Similarly, the Recency-Weighted Aggregation employs a relatively rigid weighting scheme. Investigating more data-driven weighting mechanisms could be a valuable direction for future work.
2. The paper would be strengthened by a more profound discussion contrasting frequency-domain matching with established time-domain methods. Key questions remain unexplored:
3. What is the unique advantage and necessity of frequency-domain features compared to time-domain shape features (e.g., DTW) or deep learning-based embeddings? Under what scenarios would frequency-domain matching be uniquely superior?
4. What would be the effect of a hybrid approach that combines both time-domain and frequency-domain information?

### Questions
See Weakness.

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
This paper proposes the SpecReTF method to address the shortcomings of existing RAG methods in time series prediction, which neglect key frequency domain features of potential periodic structures and fail to emphasize capturing recent, more relevant patterns in time. The pipeline (i) converts times series into windowed frequency representations using Short-time Transform(STFT), (ii) uses a combined metric that incorporates amplitude and phase information to measure similarity, (iii) apply an exponential moving average weighting scheme that emphasizes recent windows. Extensive experiments on benchmark datasets demonstrate that SpecReTF outperforms time-domain retrieval methods, achieving superior forecasting accuracy across diverse, non-stationary time series.

### Strengths
（i）This work focuses on the practical and important issue of the ignoring the distribution of energy across frequency bands leads to the misidentification of periodic patterns and temporal relevance.

（ii）This paper proposes a novel retrieval-augmented time series forecasrting method that performs similarity matching in frequency domain, which integrates similarity score for each frame is calculated by combining Jensen-Shannon divergence (to measure amplitude distribution) and cosine similarity (to measure phase alignment). An exponential moving average is then used to weight the frame-level similarity scores, thereby enhancing the influence of the most recent window while gradually reducing the weight of older windows.

(iii) SpecReTF is a novel retrieval-augmented forecasting architecture that combines frequency-domain analysis with recency-weighted pattern retrieval to address non-stationarity.

### Weaknesses
(i) It lacks hyperparameters (step size, embedding dimension, etc.) for automatic selection or robustness analysis of the Short Time Fourier Transform (STFT).

(ii) Calculated a composite similarity score for each frame using only Jensen-Shannon divergence (to measure amplitude distribution) and cosine similarity (to measure phase alignment), without comparing it with other methods to explain its advantages.

(iii) The computational complexity and actual inference overhead are not detailed, but frequency domain retrieval/STFT will significantly increase the cost.

(iv) Although frequency domain similarity is emphasized, no real-world examples are shown to illustrate that the retrieved segments are indeed more reasonable.

(v) The lack of comparative experiments with other time series prediction methods that also employ retrieval-augmented demonstrates the effectiveness of the proposed method.

### Questions
See Weaknesses

### Soundness
2

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
4

### Summary
The paper proposes a retrieval-based time-series forecasting framework that incorporates both a spectral perspective and temporal recency into the retrieval process.

### Strengths
Although the work may appear somewhat incremental - mainly replacing the similarity measurement in conventional retrieval-based time-series forecasting methods with one based on the frequency domain - the authors effectively establish the motivation through Figure 1 in the introduction. The intuition behind adopting the spectral view is clearly conveyed.

### Weaknesses
1. In Figure 3, the model’s performance appears insensitive to the decay factor $\alpha$. This raises doubts about whether temporal recency, one of the paper’s main claimed contributions, is truly impactful. Moreover, it remains unclear how practitioners should determine an appropriate value of $\alpha$ in real-world settings.

1. In the experiments, the “No Retrieval” configuration (a simple linear predictor) in Table 2 outperforms most baselines in Table 1. This makes me question whether the experiments were conducted under a fair and consistent setting.

### Questions
1. Justify the necessity of temporal recency.

2. Verify the experimental setting and guarantee the fair comparison.

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
4

### Summary
This paper introduces SpecReTF, a spectral retrieval-augmented time series forecasting method that overcomes key limitations in existing approaches by decomposing time series into frequency-domain components, measuring similarity using a combined amplitude and phase metric, and incorporating temporal recency weighting. The proposed method demonstrates superior forecasting accuracy across multiple benchmark datasets, establishing new state-of-the-art performance.

### Strengths
1.	The paper effectively identifies and analyzes "spectral blindness" as a key limitation in existing time-domain retrieval methods, providing compelling evidence through both theoretical discussion supported by visual examples.
2.	The combination of amplitude spectrum analysis (Jensen-Shannon divergence) and phase coherence measurement (cosine similarity) effectively addresses the identified spectral blindness problem.
3.	Extensive experiments across eight diverse benchmark datasets demonstrate consistent improvements over state-of-the-art baselines, with the method achieving superior performance on most evaluation metrics.

### Weaknesses
1.	The "no retrieval" results in Table 2 (which uses only two linear layers) are significantly better than the DLinear results in Table 1 on several datasets. This is contradictory because DLinear, a purpose-built linear model, should perform at least as well as this simple baseline. This inconsistency raises concerns about whether all baselines in Table 1 were evaluated under the same experimental settings as SpecReTF. Please clarify this discrepancy to ensure the fairness and validity of the comparisons.

2.	The paper claims three contributions, but they are largely overlapping and describe a single core innovation: the frequency-aware similarity metric. The architectural framework and the recency-weighting scheme, while useful, appear to be supporting components rather than distinct conceptual advances. This conflation could be seen as overstate the breadth of methodological contribution. The work's primary novelty resides in the new similarity measure, and the contributions should be reframed to more accurately reflect this.

3.	The hyperparameter study in Appendix D reveals that the model's performance is sensitive to the number of retrieved segments (K) and the STFT window size, and that the optimal values for these parameters vary across different datasets.  This observed sensitivity appears to be at odds with the paper's claim of the method's robustness.

### Questions
1.	Why does the simple "no retrieval" linear model in Table 2 outperform the purpose-built DLinear in Table 1? Does this indicate inconsistent experimental settings for the baseline models? 

2.	The three contributions essentially revolve around the frequency-domain similarity measure. Could the main contribution be reframed to focus on this core innovation?

3.	Given the sensitivity to K and STFT window size shown in Appendix D, how does this align with the paper's claim of robustness? Furthermore, how should these hyperparameters be determined in practice for new datasets?

### Soundness
2

### Presentation
3

### Contribution
2
