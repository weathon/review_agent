# Variability Aware Recursive Neural Network (VARNN): A Residual-Memory Model for Capturing Temporal Deviation in Sequence Regression Modeling

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Real-world time series data exhibit nonstationary behavior, regime shifts, and temporally varying noise that degrade the robustness of standard regression models. We introduce the Variability Aware Recursive (VARNN) Neural Network, a novel residual-aware architecture for supervised time series regression that learns an explicit error memory from recent prediction residuals and uses it to recalibrate subsequent predictions. VARNN augments a feed-forward predictor with a learned error memory state that is updated from residuals over a short context steps as a signal of variability and drift, and then conditions the final prediction at the current time. We study four configurations along two orthogonal axes: (i) residual memory as instantaneous (RM) an embedding of the current innovation only, or accumulative (ARM) that augment current innovation with past residual memory states to capture drift and volatility bursts; and (ii) the presence or absence of an activation memory (AM), which carries the previous latent activation to enrich short-term temporal dynamics and stabilize predictions under noise. Across nine datasets from three important domains, Energy, Healthcare, and Environmental monitoring, experimental results demonstrate that VARNN achieves superior performance and attains the lowest test MSE with minimal computational overhead over static, dynamic, and sequence baselines. Our findings show that the VARNN model offers robust predictions under a drift and volatility environment, highlighting its potential as a promising framework for time series learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Variability-Aware Recursive Neural Network (VARNN), which improves robustness in time-series regression by modeling prediction residuals as a learned “error memory.” Lightweight and efficient, VARNN updates a residual-memory pathway to recalibrate future predictions. On three experimental datasets, it showed better performance compared to the considered baselines. Variants and ablation studies further validate the design choices.

### Strengths
1. The paper is clearly written and easy to follow.
2. VARNN proposes a novel approach by integrating prediction residuals as a stateful memory, effectively addressing non-stationarity and distinguishing it from standard RNNs or lag-based models.
3. Its effectiveness is demonstrated through one-step-ahead forecasting with covariates on three datasets (Appliances Energy Prediction, BIDMC, Beijing), showing consistent improvements in test MSE.

### Weaknesses
1. The baselines considered in the paper are quite limited compared to the range of methods available in the time series forecasting/regression literature.


a) For static regression baselines, it would have been interesting to include a dynamic version using features like $X = [y_{t-w+1}, \dots, y_{t-1}, x_t] \rightarrow Y = y_t$. This could be applied to Ridge (linear), MLP, and potentially CatBoost (which tends to overfit less than Random Forest).


b) Another way to include temporal dynamics in classical regressors is to construct features as $X = [x_t, t] \rightarrow Y = y_t$ where t is rescaled between 0 and 1 over the window w (start of window \(t=0\), end \(t=1\)). Classical regressors like Ridge, CatBoost, or the recent TabPFN could perform well with this setup.


c) Deep learning baselines for time series, such as PatchTST, are not considered. These could potentially be adapted for one-step-ahead forecasting.

2. Only three datasets are considered, and they are relatively specific. Testing the method on a broader range of datasets, including cases with and without covariates, would strengthen the evaluation.

3. The paper focuses on one-step-ahead prediction, which is somewhat limited; extending VARNN to multi-horizon forecasting would be valuable for real-world applications.

### Questions
The paper is well written and interesting. However, the experimental section does not consider many elements from the literature, which makes it hard to fully trust the model’s effectiveness. Addressing (even partially) the points discussed above (in weaknesses) regarding the experiments could increase confidence in the results.

### Soundness
2

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
3

### Summary
In this work, the author introduces VARNN, a novel recurrent architecture that explicitly models prediction residuals as a first-class memory state to improve performance in non-stationary time-series regression. The idea is motivated and empirically validated across multiple domains.

### Strengths
1. The idea of elevating prediction residuals to an explicit memory state is innovative and well-motivated. The paper clearly identifies a gap in how existing models (ARX, RNNs, etc.) handle temporal variability and proposes a light-weight yet effective mechanism to address it.
2. The experimental design is thorough, covering three diverse domains and multiple baseline families (static, dynamic, recurrent). The consistent superiority of VARNN across datasets strengthens the claim of robustness under non-stationarity.

### Weaknesses
1. There is a sentence in the Abstract whose initial letter is not capitalized. It seems not to be a complete sentence and the logical connection is also somewhat lacking. i.e., a lightweight sequence-to-one architecture...
2. While the proposed model is empirically validated, there is little theoretical analysis or intuition for why the residual memory mechanism improves generalization under non-stationarity. A deeper discussion on the connection to error-correction models or adaptive filtering would strengthen the contribution.
3. The paper does not compare with several recent strong baselines designed for non-stationary time series (e.g., N-BEATS, or transformer-based forecasters).

### Questions
Please see Weaknesses!

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes the Variability-Aware Recursive Neural Network (VARNN) — a residual-memory architecture for time-series regression. The key idea is to explicitly model prediction residuals as a recurrent state to handle non-stationarity, variability, and drift in temporal data. The authors evaluate several variants (instantaneous vs. accumulative residual memory, with/without activation carry-over) on three datasets (energy, healthcare, environmental) and report consistent MSE improvements over ARX/NARX and RNN baselines.

### Strengths
1. The idea of using residuals as a first-class state is conceptually simple and potentially useful.
2. Empirical evaluation spans multiple domains and baselines, with consistent performance gains reported.

### Weaknesses
1. The paper is very difficult to follow: grammar and phrasing issues are pervasive from the first paragraph (“Regression is is one of the fundamental tasks/ a fundamental task…”). The exposition is verbose, repetitive, and at times inconsistent in notation. There are many typos and formatting errors that hinder readability.
2. The paper claims to propose a new neural architecture, but offers no theoretical analysis or formal justification.
3. Comparisons are relatively weak. Only simple baselines (ARX, MLP, RNN) are included. Missing comparisons with modern strong regressors for time series, including Transformers (Informer, TFT, PatchTST) or State-space models (SSM/Mamba), which are natural choices for regression under non-stationarity.
4. The proposed mechanism, feeding residuals back as input, is not conceptually new.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new neural architecture, VARNN, a neural for time-series "regression" that explicitly incorporates prediction errors (called innovations) as a recurrent state. Unlike standard RNNs that implicitly absorb variability into hidden states, VARNN maintains a dedicated "residual memory" that tracks recent prediction errors and uses this signal to calibrate subsequent predictions. Experiments are conducted to  evaluate the proposed approach across three datasets (appliance energy, healthcare heart rate, and air quality), compared to a few baselines.

### Strengths
* The core idea of the paper is well motivated and well explained.
*  The architecture is clearly described, flexible and lightweight.
* The paper presents ablation experiments and several variants to  apprehend  the proposed architecture.

### Weaknesses
* While the paper frames the problem as "time-series regression," the task is fundamentally one-step-ahead forecasting. This framing issue leads to a critical gap in the literature review: none of the recent state-of-the-art forecasting methods are cited or discussed (including PatchTST (Nie et al., 2023), Autoformer (Wu et al., 2021), DLinear (Zeng et al., 2023), etc). This omission is problematic because these methods represent the current competitive landscape. Without acknowledging or comparing against them, it is impossible to assess VARNN's true contribution to the forecasting literature.

* The core claim—that "elevating innovations  to first-class signals" is novel—overlooks substantial prior work as  Kalman Filters  and litterature in deep learning related to state space models, or ARMAX models.

* The experimental evaluation is too narrow to support the paper's claims: Only vanilla RNN is evaluated among recurrent models; LSTM and GRU are absent, despite being standard baselines. No comparison with methods designed for non-stationarity: The related work cites RevIN (Kim et al., 2022), GARCH+DL (Han et al., 2024), and de-stationing methods, but none are evaluated. No classical forecasting baselines: ARIMA, Exponential Smoothing. No modern deep forecasting methods: DLinear,  or any of the Transformer-based methods mentioned above. The current baselines function as a proof-of-concept that VARNN's architecture is functional, but they do not establish its value relative to the state-of-the-art.

* Only three datasets are evaluated, all relatively small and domain-specific. No evaluation on standard forecasting benchmarks (M4, Electricity, ETT, Traffic, Weather) that would enable comparison with published results

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
1
