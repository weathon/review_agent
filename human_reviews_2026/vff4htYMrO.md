# Reviving Error Correction in Modern Deep Time-Series Forecasting

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Modern deep-learning models have achieved remarkable success in time-series forecasting. Yet, their performance degrades in long-term prediction due to error accumulation in autoregressive inference, where predictions are recursively used as inputs. While classical error correction mechanisms (ECMs) have long been used in statistical methods, their applicability to deep learning models remains limited or ineffective. In this work, we revisit the error accumulation problem in deep time-series forecasting and investigate the role and necessity of ECMs in this new context. We propose a simple, architecture-agnostic error correction model that can be integrated with any existing forecaster without requiring retraining. By explicitly decomposing predictions into trend and seasonal components and training the corrector to adjust each separately, we introduce the Universal Error Corrector with Seasonal–Trend Decomposition (UEC-STD), which significantly improves correction accuracy and robustness across diverse backbones and datasets. Our findings provide a practical tool for enhancing forecasts while offering new insights into mitigating autoregressive errors in deep time-series models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper revisits the issue of error accumulation in deep time-series forecasting and proposes a universal, architecture-agnostic Universal Error Corrector (UEC) and its variant UEC-STD with seasonal–trend decomposition. The method can be plugged into existing forecasting backbones without retraining and significantly reduces long-horizon prediction errors. The authors provide theoretical motivation, clear formulation, and extensive experiments across multiple datasets and backbones, showing consistent improvements in accuracy and robustness.

### Strengths
1. Architecture-agnostic Universal Error Corrector (UEC) can work with various deep backbones without retraining or architecture modification, and needs low training overhead.
2. The experimental results demonstrate that the proposed method achieves strong performance.
3. The paper is well-structured, with a logical flow from motivation to methodology and experiments.

### Weaknesses
1. The entire UEC module is trained on the validation set. This data was already used to select the best checkpoint for the backbone model. Training a new model component on this same data risks overfitting to the specific error patterns of the validation set and does not guarantee generalization to the test set. 
2. The UEC-STD's success hinges on a moving-average decomposition with a fixed, hard-coded kernel size of 25. This choice is never justified, ablated, or discussed. What if the data has a different seasonality, or no seasonality at all? The method's performance is fundamentally tied to this single, arbitrary hyperparameter.
3.  The correction strength $\beta$ seems to be a critical crutch for the model. Table 8 shows that the automatically-selected $\beta$ is often very small (e.g., 0.1 or 0.3). This implies that the raw output of the UEC is often highly noisy, and its utility is only realized by severely damping its predictions.

### Questions
1. Can the authors provide a stronger justification for training on the validation set? How can we be confident that the UEC is not simply overfitting to the validation set's error distribution?
2. What is the performance of UEC-STD if $\beta$ is fixed to 1.0 for all experiments?
3. Please provide an ablation study on the moving average kernel size (ks=25). How sensitive is the model's performance to this parameter? What happens if ks is set to 5, or 50, or chosen based on a dataset's known seasonal period?

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
4

### Summary
The proposed UEC-STD framework is novel, practical, and demonstrates consistent improvements across diverse settings. While some theoretical aspects could be strengthened and a broader evaluation would be beneficial, the work provides both immediate practical utility and valuable insights for future research.
The architecture-agnostic nature of the approach is particularly valuable, as it enables practitioners to improve existing models without the computational cost of retraining. The seasonal-trend decomposition strategy is well-motivated and effectively implemented.

### Strengths
1. Architecture-Agnostic Design: The proposed Universal Error Corrector (UEC) framework can be integrated with any existing forecaster without retraining, addressing a major limitation of previous approaches
2. Seasonal-Trend Decomposition (UEC-STD): The explicit separation of trend and seasonal components for targeted correction is theoretically sound and aligns with established time series analysis principles
3. Practical Implementation: The method requires minimal computational overhead (∼10% of backbone training time) while delivering consistent improvements

### Weaknesses
1. Methodological Limitations:
- Decomposition Simplicity: The moving average approach for seasonal-trend decomposition, while practical, may be too simplistic for complex, non-stationary time series
- Hyperparameter Sensitivity: The method requires careful tuning of λs-λt coefficients and correction strength β, which could limit out-of-the-box applicability
- Assumption of Decomposability: The approach assumes that time series can be cleanly separated into trend and seasonal components, which may not hold for all real-world data
2. Experimental Considerations:
- Limited Real-World Testing: While benchmark datasets are comprehensive, additional testing on more diverse, noisy real-world datasets would strengthen the claims
- Computational Overhead: Although minimal compared to backbone training, the additional inference time (UEC prediction at each autoregressive step) could be problematic for real-time applications
- Scalability Concerns: The method's performance on very high-dimensional multivariate time series needs further validation

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses error accumulation in deep learning-based time-series forecasting during autoregressive inference, where predictions are recursively used as inputs. The authors propose the Universal Error Corrector with Seasonal-Trend Decomposition (UEC-STD), a simple and architecture-agnostic post-processing module that can enhance any pre-trained forecasting model without requiring retraining. UEC-STD explicitly decomposes predictions into trend and seasonal components and learns separate corrections for each, optimizing a weighted loss function. The method is trained on validation data while keeping the backbone forecaster frozen, making it practical and efficient. Experiments across 7 datasets and 3 modern forecasting models (TimeMixer, TimesNet, TimeXer) demonstrate that UEC-STD consistently reduces both MSE and MAE by approximately 2.1% and 0.8% on average, with particularly strong improvements on datasets like ETTm1 (4.78% MSE reduction), while adding minimal computational overhead (roughly 10% of backbone training time).

### Strengths
1. Error correction models are indeed an important aspect of forecasting that is currently missing from deep-learning-based approaches2. 2. The ablation study is well-designed and comprehensive

### Weaknesses
1. The presentation of Figure 2 is unclear, making it difficult for the reader to interpret the key takeaways.
2. Averaging results across all backbones in Tables 1 and 3 may obscure performance nuances. (No one will use all backbones at the same time in practice) A more informative presentation would be to disaggregate these results, showing performance for each backbone individually. Additionally, tracking the number of "first place" finishes per backbone would offer a clearer comparison of methods.
3. The reported improvements are modest, making it difficult to assess their significance. To validate these gains, please provide statistical significance tests. Furthermore, the 2% gain is not a lot when compared to 14% error increase depicted in Figure 1(b).
4. An inconsistency arises as only the UEC-STD variant shows consistent improvement, while other related UEC architectures do not. This selective improvement is counter-intuitive and requires further analysis or explanation.

### Questions
1. What are the look-back length, per-step forecast horizon, and number of steps in the main experiments?
2. The analysis could be strengthened by examining performance trends over time. It would be valuable to see if the method's improvement (relative to baselines) changes as the forecast horizon extends deeper into the future.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a method named Universal Error Corrector with Seasonal-Trend Decomposition (UEC-STD), which leverages error correction mechanisms for time series forecasting and can be used to enhance model performance. The authors apply their proposed method to baseline models and show that it improves the performance in prediction accuracy.

### Strengths
1. Overall this paper is clearly written and easy to understand. 

2. The proposed method using error correction mechanism is moderately novel and may be potentially applicable to similar problems in this domain.

### Weaknesses
1. The baseline models in Table 1 do not represent the state-of-the-art (SOTA) methods for time series forecasting. In order to justify the significance of this research contribution, the authors should clearly demonstrate that the proposed UEC framework can further improve a wide range of SOTA methods.  

2. In addition to MAE and MSE, the authors should evaluate their proposed method with MAPE (mean absolute percentage error) which is robust under different scales of the time series values.

3. The authors should also evaluate their proposed method on standard benchmark datasets for time series forecasting, such as the M4 competition dataset.  

4. Recent works in time series forecasting should be reviewed in Related Works.

### Questions
1. Why are some evaluation results not available in Table 1? 

2. How does UEC perform when it is applied to the SOTA methods for deep time series forecasting?

### Soundness
3

### Presentation
3

### Contribution
2
