# Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Time-series forecasting finds broad applications in real-world scenarios. Due to the dynamic nature of time series data, it is important for time-series forecasting models to handle potential distribution shifts over time. In this paper, we initially identify two types of distribution shifts in time series: concept drift and temporal shift. We acknowledge that while existing studies primarily focus on addressing temporal shift issues in time series forecasting, designing proper concept drift methods for time series forecasting has received comparatively less attention.

Motivated by the need to address potential concept drift, while conventional concept drift methods via invariant learning face certain challenges in time-series forecasting, we propose a soft attention mechanism that finds invariant patterns from both lookback and horizon time series. Additionally, we emphasize the critical importance of mitigating temporal shifts as a preliminary to addressing concept drift. In this context, we introduce ShifTS, a method-agnostic framework designed to tackle temporal shift first and then concept drift within a unified approach. Extensive experiments demonstrate the efficacy of ShifTS in consistently enhancing the forecasting accuracy of agnostic models across multiple datasets, and outperforming existing concept drift, temporal shift, and combined baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies two types of distribution shifts in time series forecasting: temporal shift (marginal distribution) and concept drift (conditional distribution). The authors propose a new method, SAM (soft attention masking), to mitigate concept drift by learning invariant patterns from both lookback and *horizon* exogenous features ($X^L$ and $X^H$). This mechanism learns a surrogate feature, $X^{SUR}$, which is predicted as an auxiliary task. The paper also presents ShifTS, a model agnostic framework that first uses normalization (like RevIN) to handle temporal shift, and then uses a backbone model to jointly predict the target $Y^H$ and the surrogate $\hat{X}^{SUR}$. Experiments on six datasets show that ShifTS consistently improves the performance of various backbone models, including current state of the art ones.

### Strengths
- The paper clearly distinguishes between temporal shift and concept drift, tackling a significant and practical problem in time series forecasting.
- The core idea of using *horizon* exogenous features ($X^H$) to define a stable surrogate target ($X^{SUR}$) is novel. This surrogate acts as an effective regularization target during training, forcing the model to learn future relevant patterns.
- The ShifTS framework is practical and model agnostic. It cleanly integrates a known temporal shift solution (normalization) with the proposed concept drift solution (SAM).
- The experiments are comprehensive. They show consistent performance improvements across six datasets and multiple strong backbone models, and also outperform other distribution shift baselines.

### Weaknesses
- The claim that SAM finds "invariant patterns" is not well supported by the mechanism. The method (Equation 1) is a learnable attention mask, not an explicit invariance optimization like in IRM. It seems to learn a *useful compression* of future features, but calling it "invariant" is a strong claim that needs better justification.
- The paper introduces significant complexity with the SAM module and the aggregation MLP. However, the performance gains on SOTA models like iTransformer are sometimes small (e.g., <5% MAE gain on ETTh2/ETTm2). The marginal benefit versus the added complexity is questionable in these cases.
- A critical and much simpler baseline is missing. The paper argues predicting raw $X^H$ is too hard, but $X^{SUR}$ is easier. This assertion must be tested by comparing ShifTS to a simpler multi task model that just predicts the raw $X^H$ as the auxiliary task.

### Questions
1.  Can you please clarify how the SAM mechanism (Equation 1) specifically enforces invariance? The high weight patterns are defined as invariant, but it is unclear how the optimization process encourages this property over just learning a stable predictive signal.
2.  To justify the complexity of SAM, could you add a baseline that replaces the $X^{SUR}$ target with the raw $X^H$? This would involve a multi task loss $\mathcal{L} = \mathcal{L}_{TS} + \lambda \cdot MSE(X^H, \hat{X}^H)$. This comparison is essential to prove that the SAM slicing and attention is superior to just predicting the raw future features.
3.  What is the impact of applying ShifTS to "near-stationary" datasets like Traffic or Weather, which were excluded? Does the method degrade performance in the absence of significant shifts, or does it robustly "do no harm"?
4.  The aggregation step ($Agg(\cdot)$ in Algorithm 1 and Figure 2) appears to be a key component. Could you provide more detail on its architecture and how the final $\hat{Y}^H$ is computed from the initial forecast and the $\hat{X}^{SUR}$ prediction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper categorizes the general concept drift in time series into types: concept drift and temporal shift (as defined more precisely in Definition 2.1 and Definition 2.2). To solve the concept drift problem, it proposes the soft attention mechanism (SAM) to find the invariant patterns in lookback and horizon windows. The core idea of this paper is illustrated well in Fig. 1. A method-agnostic framework called ShiftTS is proposed to deal with both temporal and concept drifts in a unified framework. Experiments demonstrate the good performance of the proposed method.

### Strengths
1.	The paper is well written and easy to follow. The idea proposed is simple. 
2.	The invariant patterns are learned through the surrogate feature $X_{SUR}$ and this is the core contribution of this paper. The basic idea is to concatenate the lookback and horizon windows, and model the conditional distributions for local patterns using the soft attention matrix M. 
3.	Experiments demonstrate the good performance of the proposed method.

### Weaknesses
1.	The categorization of temporal shift and concept drift is very similar to different sources of concept outlined in Section 2.1 in [R1], but in the context of time series with some differences (the temporal shift is for Y, and the concept drift is the same as Source II in [R1] ). 
2.	The method mitigating temporal shift (or the marginal distribution shift of Y) looks quite standard in the literature. 
3.	The proposal of mitigating concept drift may be incremental for some datasets and base algorithms.

### Questions
1.	The analysis in Section 4.2 looks reasonable to me. The improvement of the proposed method depends on the data and the base algorithm simultaneously. Fig.3(a) plots the performance gain vs. the mutual information between X and Y. In fact, this plot studies the effectiveness of ShiftTS w.r.t. the so-called concept drift defined in Def 2.2 in this paper. A good example is the Exchange dataset, which shows low mutual information but high performance improvement. Thus, it suggests that the performance is mainly due to the mitigation of the so-called temporal concept drift. This is also consistent of Fig. 3(b), which shows that the most significant performance improvement is between ShiftTS\TS and Base. Therefore, at least on the Exchange dataset, the proposal of learning form surrogate feature $X_{SUR}$ looks incremental.

### Soundness
2

### Presentation
3

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
This paper addresses the problem of **distribution shifts in time-series forecasting**, focusing specifically on **concept drift**—a relatively under-explored issue compared to temporal shift. The authors propose:

1. **SAM**: A mechanism to identify invariant patterns in exogenous features across lookback and horizon windows, enabling more stable conditional distribution modeling.
2. **ShiftS**: A unified, model-agnostic framework that first mitigates temporal shift (via normalization) and then concept drift (via SAM), improving generalization across diverse forecasting models.

Extensive experiments on six real-world datasets show that ShiftS consistently boosts forecasting accuracy across multiple base models and outperforms existing distribution-shift baselines.

### Strengths
- **Originality**: SAM is a novel approach to handling concept drift without relying on environment labels or online retraining.
- **Quality**: The method is well-designed, with careful attention to both theoretical motivation and practical implementation.
- **Clarity**: The problem formulation and methodology are clearly explained, and the experiments are thorough and convincing.
- **Significance**: The paper fills a clear gap in the literature and provides a practical tool for improving time-series forecasting under distribution shifts.

### Weaknesses
- **Theoretical Guarantees**: The method lacks theoretical analysis (e.g., error bounds or convergence guarantees), which is noted in the limitations but could strengthen the contribution.
- **Dependence on Horizon Exogenous Data**: SAM relies on $\mathbf{X}^H$ , which may not always be available or accurately predictable in practice. The paper addresses this via surrogate forecasting, but the impact of prediction error on final performance is not deeply analyzed.
- **Limited Scope**: The method is evaluated only on univariate forecasting with exogenous features. Its applicability to multivariate or purely endogenous settings remains unclear.

### Questions
1. How does SAM perform when $\mathbf{X}^H$ is not available during training or is highly noisy? Have you tested scenarios with missing or imperfect exogenous data?
2. Could SAM be adapted for online settings where concept drift occurs continuously? The paper criticizes online methods but does not explore whether SAM can be extended in that direction.
3. The mutual information analysis is insightful—have you considered using $ I(\mathbf{X}^H; \mathbf{Y}^H) $ as a criterion for applying SAM in practice?
4. The framework currently uses RevIN for temporal shift mitigation. Have you experimented with more advanced methods (e.g., SAN) in the full ShiftS pipeline, and if so, how do they compare?

### Soundness
3

### Presentation
3

### Contribution
3
