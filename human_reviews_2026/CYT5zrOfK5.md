# LOBBen-TM: A Benchmark Study of Limit Order Book Prediction with Temporal Modeling

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
We introduce LOBBen-TM, a limit order book (LOB) benchmark with temporal modeling for deep learning on open-sourced LOB data that unifies evaluations across tasks, features, and assets. Our work makes four major contributions: (i) On the Mid-Price Trend Prediction (MPTP) task, we assess state-of-the-art LOB models with a standardized full LOB feature set with time-sensitive features on two assets, equities (FI-2010) and cryptocurrency (Bitcoin), to probe cross-asset generalization under a common protocol. We further benchmark a common LOB feature taxonomy (basic, time-insensitive, time-sensitive) and conduct an ablation on FI-2010. (ii) We extend the study to Mid-Price Return Forecasting (MPRF), jointly evaluating LOB-specific architectures and top-tier general time-series predictors on FI-2010 with MSE, R2, and Pearson correlation. (iii) To enhance multivariate time series prediction models on LOB returns, we propose a lightweight Cross-Variate Mixing Layer (CVML) that plugs into existing models. Empirically, results show that the standardized full feature set yields robust MPTP performance across FI-2010 and Bitcoin while revealing asset-dependent ranking shifts. Besides, time-sensitive features provide sizable improvements on FI-2010, underscoring the importance of temporal signal modeling. Last but not least, our proposed CVML architecture substantially boosts general time series prediction models on MPRF, narrowing the gap to LOB models and advancing return forecasting on LOB data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces LOBBen-TM, a comprehensive benchmark for deep learning on Limit Order Book data. It unifies evaluation across two tasks, multiple feature sets, and two asset types including FI-2010 equities and Bitcoin. The benchmark standardizes a full LOB feature taxonomy with basic, time-insensitive and time-sensitive categories. Time-sensitive features significantly improve performance on one task for FI-2010 equities. Cross-asset experiments show model rankings shift between equities and crypto, reflecting asset-specific dynamics. For the other task, LOB-specific models outperform general time-series architectures, emphasizing domain-specific inductive biases. To address this gap, the authors propose CVML, a lightweight plug-in that enhances multivariate signal mixing in general time-series models. CVML consistently boosts performance across metrics, reduces input noise and improves attention-based modeling of cross-variate and temporal dependencies.

### Strengths
1. Establishes a unified evaluation benchmark for Limit Order Book data named LOBBen-TM, addressing the gaps of existing benchmarks in task coverage and asset diversity.
2. Proposes a three-level LOB feature taxonomy including basic, time-insensitive, and time-sensitive features, and clarifies the significant improvement effect of time-sensitive features on mid-price trend prediction through ablation experiments, providing clear guidance for feature selection.
3. Conducts the joint evaluation of LOB-specific models and top-tier general time-series models on the mid-price return forecasting task, clearly revealing the importance of domain-specific inductive biases.
4. Designs a lightweight Cross-Variate Mixing Layer that can integrated into existing models, enhancing the signal mixing capability of general time-series models for LOB data and narrowing the performance gap with LOB-specific models.

### Weaknesses
1. Limited asset diversity, only two types of assets are evaluated, broader market coverage would strengthen generalization claims.
2. Shallow analysis of failure modes: The study doesn’t deeply investigate why general time-series models underperform on raw LOB data beyond “low signal-to-noise.”
3. The selected time-series baseline models are not diverse enough. Incorporating additional time-series foundation models such as the recent Chornos-2 [1] and Moirai-2 [2], as well as frequency-domain-based models like FITS [3], would help enhance the credibility of the conclusions.
4. Most results focus on short horizons, and longer-term forecasting performance is not explored. Providing longer horizons would help observe changes in prediction accuracy brought by models in long time-series forecasting.

### Questions
See Weakness.

### Soundness
3

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
The paper talks about LOBBen-TM, a unified benchmark for Limit Order Book prediction with a focus on temporal modeling. It evaluates state-of-the-art models across mid-price classification and forecasting, making comparisons in performance. The paper builds a standardized feature taxonomy, tests it on equity and crypto data, and introduces a new Cross-Variate Mixing Layer to try to capture multivariate relationships. Results show time-sensitive features greatly boost prediction accuracy, though cross-asset transferability is low, and CVML improves model performance as measured by R^2.

### Strengths
* Establishes a unified benchmark covering both trend and return forecasting under consistent protocols.
* Demonstrates that time-sensitive signals are critical for predictive accuracy.
* Provides a simple, effective improvement to general time-series architectures via CVML which seems to improve MSE and R^2
* Consistent metrics are used, and ablations are run

### Weaknesses
* Limited dataset diversity - only FI-2010 and Bitcoin datasets are evaluated.
* No discussion of economic or trading significance beyond statistical metrics – what about P&L?
* Short forecast horizons (K=1–10) may miss longer-term dependencies.
* Limited theoretical explanation for CVML’s effectiveness.
* Heavy reliance on FI-2010 results for conclusions.
* Use of vanilla hyperparams from other methods may not get the best from these (as most were tuned for use with different financial assets).

### Questions
The key focus on MSE ignores the importance of returns and associated P&L in finance. Many of the other works cited consider things like Sharpe ratios etc – and some go on to directly maximize these rather than min MSE. Can you comment on this?

Can you comment on the focus on mid-price, rather than top of book price on buy and sell sides? Realistic trading needs to cross the spread, so forecasting both – and then paying the spread – is important. Adding on top of this, there is normally a commission or trading fee. 

Why is the target not log returns? Any reason?

In the experiments are just 3 seeds enough to get meaningful stats in results? (I don’t think so)

Can you comment on the short horizons used, and if longer horizons might behave differently.

In Fig 1, DeepLOBAtt seems anomalous – any reasons for this?

References – you need to protect capitals, like {MLP} etc

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
This paper introduces LOBBen-TM, a unified benchmark for deep learning on open-source Limit Order Book (LOB) data that standardizes evaluation across assets, features, and tasks. It spans Mid-Price Trend Prediction (MPTP) and Mid-Price Return Forecasting (MPRF), covering equities (FI-2010) and crypto (Bitcoin). The authors benchmark a taxonomy of features (basic, time-insensitive, time-sensitive) and propose a Cross-Variate Mixing Layer (CVML) to enhance multivariate modeling in time-series backbones. Results show that time-sensitive features consistently improve accuracy, that model rankings vary across assets, and that CVML boosts general TS models, narrowing the gap with LOB-specific architectures.

### Strengths
This paper illustrates a unified framework for standardized LOB evaluation across tasks, features, and assets.

As a benchmark study work, it is up to date and demonstrates the clear value of time-sensitive features for LOB modeling.

The assets are made open, which facilitates the community and provides a clear benchmarking framework that encourages transparent comparisons.

### Weaknesses
Cross-asset coverage and related results are discussed in the proposal and experiments, but they are not very convincing. 

The study did not provide insights into the pros and cons of LOB prediction tasks, nor did it address the challenges, potential dataset limitations, and computing power requirements in LOB trading. Therefore, it is information-rich but not industrial.

### Questions
It would be good to highlight the extent of the dollar term's impact on the LOB prediction. Could you illustrate the differences in dollar terms between benchmarks for a trading system that could use them for systematic trading?

### Soundness
2

### Presentation
2

### Contribution
2
