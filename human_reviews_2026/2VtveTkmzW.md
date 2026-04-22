# Contextual and Seasonal LSTMs for Time Series Anomaly Detection

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Univariate time series (UTS), where each timestamp records a single variable, serve as crucial indicators in web systems and cloud servers. Anomaly detection in UTS plays an essential role in both data mining and system reliability management. However, existing reconstruction-based and prediction-based methods struggle to capture certain subtle anomalies, particularly small point anomalies and slowly rising anomalies. To address these challenges, we propose a novel prediction-based framework named Contextual and Seasonal LSTMs (CS-LSTMs). CS-LSTMs are built upon a noise decomposition strategy and jointly leverage contextual dependencies and seasonal patterns, thereby strengthening the detection of subtle anomalies. By integrating both time-domain and frequency-domain representations, CS-LSTMs achieve more accurate modeling of periodic trends and anomaly localization. Extensive evaluations on public benchmark datasets demonstrate that CS-LSTMs consistently outperform state-of-the-art methods, highlighting their effectiveness and practical value in robust time series anomaly detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper identifies key shortcomings of prior approaches to univariate time-series (UTS) anomaly detection and proposes CS-LSTM. This dual-branch architecture explicitly separates long-term seasonal dynamics from short-term contextual fluctuations. By assigning distinct roles to the two LSTM branches, the method aims to detect both *small point anomalies* and *slowly rising interval anomalies.* Across several benchmarks, the approach is reported to outperform prior methods in both accuracy and runtime.

### Strengths
- **Clear problem framing**. This paper is, to my knowledge, the first to explicitly explain the two challenges of *small point anomalies* and *slowly rising anomalies* in UTS, and to analyze why popular reconstruction- and prediction-based baselines struggle with them.
- **Probabilistic predictions**. Instead of deterministic outputs, the model predicts $\mu$ and $\sigma$, allowing $\sigma$ to act as a data-driven tolerance band and enabling a principled anomaly score.
- **Architectural novelty with complementary roles**. The design assigns seasonal/long-term dynamics to the S-LSTM and contextual/short-term dynamics to the C-LSTM, yielding strong empirical performance. The dual-branch decomposition is simple, interpretable, and appears novel in this setting.

### Weaknesses
- **Generalization beyond the targeted anomaly types.** The evaluation focuses on cases where the two targeted anomaly types are present. It remains unclear how the model behaves when anomalies fall outside these categories (e.g., regime shifts without seasonality, changes in variance, or adversarial bursts).
- **Scope limited to univariate data.** Many real-world series are multivariate. It is not apparent how this UTS-specific design would extend to multivariate settings.
- **Hyperparameter sensitivity.** The method introduces several windowing hyperparameters (e.g., $w_{s}$, $w_{c}$, $k$). The relative importance of these choices, guidance for tuning, and robustness across datasets and tasks are not fully clarified; optimal values are likely to vary substantially across domains, which may complicate deployment.

### Questions
- **Training efficiency.** Table 4 reports inference latency. How does training time compare to strong baselines under the same hardware and data conditions (including windowing/FFT overhead)?
- **Mask estimation.** Please describe the procedure to construct the mask used in the masked Gaussian NLL in full detail (initialization, update rule, thresholds/criteria).

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
The paper introduces CS-LSTMs, a dual-branch prediction framework for detecting subtle anomalies in univariate time series (UTS). It consists of two complementary LSTM branches, S-LSTM (Seasonal branch) captures evolving periodicity via wavelet-based time–frequency representations, and C-LSTM (Contextual branch) models short-term dependencies and local dynamics in the time domain.

A wavelet transform first decomposes the signal into trend, seasonal, and noise components, allowing each branch to process denoised inputs. Their outputs are fused for final anomaly prediction, optimized by a masked probabilistic loss that focuses learning on normal patterns while suppressing anomalous effects.

Across multiple benchmarks, CS-LSTMs consistently outperform reconstruction- and prediction-based baselines. Ablation and transfer experiments verify that the dual-branch design and noise-decomposition mechanism jointly enhance robustness and generalization. The approach effectively overcomes prior models’ difficulty in detecting small point and gradually rising anomalies by integrating time- and frequency-domain representations and modeling the evolution of periodicity.

### Strengths
- The separation of seasonal and contextual learning provides complementary modeling of periodic and local behaviors, enabling precise detection of small or slowly evolving anomalies.
- The use of wavelet-based decomposition allows simultaneous handling of non-stationarity and noise suppression, improving interpretability and robustness compared with pure time-domain models.
- Demonstrates consistent superiority over SOTA baselines across several datasets, supported by transferability and ablation experiments showing each module’s distinct contribution.
- Addresses a realistic gap in anomaly detection—capturing weak, evolving irregularities that traditional reconstruction or forecasting models often miss.

### Weaknesses
- Key mechanisms—especially the noise-decomposition pipeline, fusion process, and masked loss formulation—lack detailed mathematical description and ablation justification.
- The paper does not analyze how the method scales to multivariate time series or handles cross-variable dependencies.
- Efficiency metrics are presented in relative terms without absolute computational cost (e.g., GPU time, memory).

### Questions
- How exactly does the wavelet-based noise decomposition feed into the dual-branch architecture? Need clear data flow and mathematical formulation.
- How are mask and ~mask generated and applied in the loss function to handle anomalies during training?
- How is ‘evolution of periodicity’ defined and quantified within the S-LSTM branch?
- Why does the paper choose LSTM over Transformer-based alternatives (Informer, Autoformer) for subtle anomaly modeling?
- Can CS-LSTMs handle multivariate or cross-correlated series, and what modifications would be required?
- Recent works such as Dual-TF (Nam et al., 2024), TFAD (Zhang et al., 2022), FAD (Li et al., 2023), and WaveletAE (Zhao et al., 2020) also integrate time–frequency representations for anomaly detection. How do CS-LSTMs differ from these dual-TF or frequency-aware models in terms of architectural design, ability to capture evolving periodicity, and computational efficiency?

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
3

### Summary
This paper addresses anomaly detection in univariate time series (UTS) by identifying three key challenges that limit existing methods' ability to detect subtle anomalies. The authors propose CS-LSTMs, a dual-branch framework where S-LSTM captures long-term periodic patterns through frequency-domain analysis and C-LSTM models short-term local dynamics. A noise decomposition strategy is introduced to improve robustness against unlabeled anomalies during training. Experimental results on four benchmark datasets demonstrate that CS-LSTMs achieves superior F1 scores compared to ten baseline methods while reducing inference time.

### Strengths
- **Systematic analysis of detection failures in existing methods.** The paper identifies why current approaches fail on small point anomalies and slowly rising segment anomalies, attributing this to inadequate integration of local trends with periodic variations.
- **Principled noise decomposition strategy.** The wavelet-based approach effectively filters noise while preserving trend and seasonal components, enabling robust learning of normal patterns despite unlabeled anomalies in training data.
- **Well-motivated dual-branch architecture.** S-LSTM captures long-term periodic evolution via frequency-domain analysis while C-LSTM models short-term local dynamics, providing complementary temporal perspectives that enhance detection capability.
- **Strong empirical results with efficiency gains.** CS-LSTMs consistently outperform ten baselines across four datasets while reducing inference time and using fewer parameters, demonstrating practical deployment value.

### Weaknesses
- Equation 6 appears to rely on the function or definition presented later in Equation 7. For clarity and logical coherence, the authors should introduce Equation 7 before Equation 6.
- The paper lacks critical implementation details including specific hyperparameters such as  batch size, number of training epochs,window sizes, decomposition Level L.
- The paper provides limited explanation for why LSTMs are better over other recent architectures (Transformers, state-space models) that have also demonstrated superior performance on time series tasks.
- While Section 5.4.1 shows noise decomposition helps, the paper doesn't compare the MAD against other alternatives (pooling-based decomposition ,Robust STL or recent decomposition approaches).
- All datasets come from web/IT monitoring domains, limiting generalizability claims to other time series applications (medical, financial, environmental).

### Questions
- How is the decomposition level L in Equation 2 determined Is it already set as hyperparameters? or is it just automatically derived based on discrete wavelet transform (DWT)?
- In equation 7, there’s Kronecker product in the equation while Appendix E shows element-wise multiplication for the same operation. Which is correct? Couldd the author please clarify and ensure consistent notation.
- In Section 5.4.2, the presentation is confusing. If the top graphs vary seasonal window with fixed context window (and bottom panels do the reverse), why do the fixed window values differ across datasets (e.g., context=4 for Yahoo vs. context=48 for NAB)? Could the author please clarify the experimental setup and using different baseline window sizes across datasets.
- Table 4 reports overall inference time but doesn't break down the wavelet decomposition overhead. How much time does the noise decomposition step consume relative to the dual-branch prediction?

### Soundness
3

### Presentation
2

### Contribution
2
