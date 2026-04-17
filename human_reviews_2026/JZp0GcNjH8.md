# Towards Reliable Spatiotemporal Epidemic Forecasting via Steering Diffusion Inference

- Decision: Reject
- Scores: 4, 6, 2, 2, 8

## Abstract
Reliable epidemic prediction is vital for public health response and resource allocation, especially in rapidly evolving outbreaks. 
Despite the recent attempts to integrate the epidemic mechanistic model into data-driven forecasting models, existing approaches still lack explainability and robustness.
To bridge this gap, we propose **EpiDiff**, an epidemiology-aware diffusion framework that incorporates mechanistic estimations and their posterior uncertainties into the forecasting process. **EpiDiff** features a flexible and high-capacity diffusion backbone specifically designed for spatiotemporal epidemic data, enabling accurate and robust sequence prediction. By quantifying the uncertainty of mechanistic forecasts and using it to steer the diffusion model at inference, **EpiDiff** dynamically adjust the data-driven prediction with the guidance from epidemic model.
Extensive experiments on real-world epidemic datasets demonstrate that **EpiDiff** consistently outperforms state-of-the-art baselines in both accuracy and robustness, while offering improved explainability for epidemic forecasting.
Our code and datasets are available at https://anonymous.4open.science/r/epidiff-4782.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EpiDiff, a hybrid framework for epidemic forecasting that integrates mechanistic SIR models with a diffusion-based deep learning model. The key innovation is its ability to quantify the uncertainty of the mechanistic model's predictions. This uncertainty score is then used to dynamically "steer" the diffusion model during inference, allowing the model to rely more on data-driven patterns when the mechanistic model is less confident. Experiments on real-world epidemic datasets show this approach consistently outperforms state-of-the-art baselines in both accuracy and robustness.

### Strengths
1. The core contribution is a novel mechanism that uses the posterior uncertainty of a mechanistic model to dynamically steer a diffusion model . This attempts to balance trust between the mechanistic prior and the data-driven backbone, moving beyond simpler hybrid methods that just use model outputs as static features.


2. The empirical evaluation tests the model on three real-world datasets specifically chosen to represent different non-stationary scenarios (stable, positive-shift, and negative-shift). The ablation study (Table 2) effectively isolates the contribution of each model component, demonstrating the framework's performance in these specific cases.


3. The framework offers a specific form of explainability by visualizing the mechanistic uncertainty (Fig. 3). This feature allows a user to identify which of the model's components (the data-driven backbone or the mechanistic prior) is driving the final forecast, offering a view of the model's internal dynamics.

### Weaknesses
1. Limited and Incomplete Definition of "Robustness"

The paper's claim of superior "robustness" is based on a limited and incomplete definition. The experiments define robustness almost exclusively as the ability to handle non-stationary distribution shifts, such as the covid-JP outbreak. However, the experiments fail to test for robustness against other critical, real-world data challenges, namely (1) missing values and (2) noisy data.

2. Limited Experimental Scope

The experimental scope is limited. The range of horizons and lookback window sizes studied is narrow, and there is a lack of experiments on larger datasets that feature more nodes and complex temporal patterns.

3. Narrow Uncertainty Quantification and Misleading Explainability

The method's uncertainty quantification is relatively narrow. It only quantifies parameter uncertainty (e.g., $\beta, \gamma$), while ignoring the far greater model uncertainty (that SIR is a flawed model) and data uncertainty (reporting noise). Basing the system's "trust" on this incomplete uncertainty model undermines the central claim of reliability. Consequently, the "explainability" is also misleading; visualizing the model's internal uncertainty is merely a self-referential explanation of its components, not a true epidemiological insight into the causes of an outbreak.

### Questions
1. How does the model perform across different forecasting horizons (e.g., 1–8) and lookback window sizes (e.g., 8–36)?
2. Does the model generalize effectively to larger datasets while remaining robust under realistic conditions with missing values and noisy data?
3. How can the model be adapted to account for and explain data uncertainty?
4. Although the experiments claim to address non-stationary settings, there is no quantification of the data’s distribution shift. How does the model perform under varying degrees of distribution shift?

### Soundness
2

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
The paper proposes EpiDiff, a hybrid epidemic forecasting framework that combines diffusion-based deep learning with mechanistic SIR modeling. By using uncertainty-aware steering to guide the diffusion process, EpiDiff achieves accurate and robust forecasts while maintaining epidemiological consistency, outperforming prior neural and hybrid methods on COVID-19 and influenza datasets.

### Strengths
1. A motivation that closely aligns with real-world epidemic applications.
2. Clear and easy-to-follow writing.
3. Comprehensive theoretical analysis provides strong support for the proposed method.

### Weaknesses
1. Why does introducing the SIRS prior help address the OOD problem? The data fitted by SIRS are still IID, and SIRS itself does not inherently possess generalization capability.
2. Regarding the claim in line 241, how does the diffusion backbone integrate diverse contextual signals such as mobility patterns, vaccination records, or other exogenous covariates?
3. In Table 2, why does removing Steering improve performance on the COVID-US dataset? COVID-US is a stationary dataset, but that does not necessarily mean it is non-trending; parameter estimation should still be feasible.
4. In Figure 3, the right-hand plot suggests that uncertainty appears to be node-dependent rather than time-dependent. Does this imply that certain nodes are inherently harder to predict?

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a diffusion model that incorporates an SIR model as a classifier-type guidance for improving time series forecasting in epidemics. This approach is tested in a few datasets.

### Strengths
* The motivation and underlying technical ideas are reasonable and do not appear to contain major errors.

* Good presentation and visualizations.

### Weaknesses
* This paper addresses a spatiotemporal setting in which time series are predicted for multiple locations. However, the SIR model used in the study is defined for a single location. It is well established that when modeling spatiotemporal dynamics, the SIR framework must incorporate connections between locations to capture the exchange of cases across them (see, for example, [1]). In addition, it is well known that the SIR model is not suitable for modeling COVID-19 dynamics, see [4] for a review on more appropriate models. Therefore, the mechanistic model employed in this paper is not appropriate for the spatiotemporal setting considered.

* The methodology section includes an extensive discussion on uncertainty quantification, yet the experiments do not report a single metric evaluating uncertainty calibration. The paper’s motivation highlights the “lack of sufficient flexibility to handle uncertainty,” but it remains unclear why the authors chose not to present any uncertainty analysis. This omission is particularly notable given that several recent works on diffusion models for forecasting explicitly emphasize and evaluate uncertainty calibration (see, for example, [2]), whereas this paper appears to disregard it entirely.

* The mechanistic guidance is not a novel technical contribution, despite the authors’ emphasis, as similar ideas have already been explored in several prior works (e.g., [3]).

* While the authors claim that the mechanistic model enhances explainability, no experiments are presented to demonstrate or validate this claim.

[1] Lloyd, A.L. and Jansen, V.A., 2004. Spatiotemporal dynamics of epidemics: synchrony in metapopulation models. Mathematical biosciences, 188(1-2), pp.1-16.

[2] Rühling Cachay, S., Zhao, B., Joren, H. and Yu, R., 2023. Dyffusion: A dynamics-informed diffusion model for spatiotemporal forecasting. Advances in neural information processing systems, 36, pp.45259-45287.

[3] Huang, J., Yang, G., Wang, Z. and Park, J.J., 2024. DiffusionPDE: Generative PDE-solving under partial observation. Advances in Neural Information Processing Systems, 37, pp.130291-130323.

[4] Adiga, A., Dubhashi, D., Lewis, B., Marathe, M., Venkatramanan, S. and Vullikanti, A., 2020. Mathematical models for covid-19 pandemic: a comparative analysis. Journal of the Indian Institute of Science, 100(4), pp.793-807.

### Questions
How do you account for the fact that different regions (e.g., U.S. states) vary greatly in scale? For instance, California, being a populous state, will naturally have much higher disease incidence than a smaller state like Arkansas. Do you apply any normalization or weighting to ensure that your evaluation is not biased toward performance in large regions while overlooking smaller ones?

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes EpiDiff, a hybrid spatiotemporal epidemic forecasting framework that combines mechanistic models with diffusion-based neural networks. The authors proposed a novel uncertainty-aware mechanistic guidance (via Laplace approximation and nonlinear transforms) for steering the inference process (at evaluation phase) of the predictive diffusion model. The method demonstrates strong performance on COVID-19 and influenza datasets, especially under distribution shifts.

### Strengths
The paper is well-written, ideas are well-presented and easy to follow.
The idea of using Laplace approximation for the posterior distribution and propagates the parameter uncertainty to predictive uncertainty via support points is novel and efficient.
Comprehensive experiments and ablation studies showcasing the efficacy of the proposed components.

### Weaknesses
The parameters $\hat{\theta}_i$ are estimated separately for each history window $y^{t−\kappa:t−1}$ which raises concerns about the consistency of the mechanistic model over one epidemic season. Also the prior distribution $p(θ_i)$ is not specified.
    The estimated parameters for the mechanistic model are not shown, which limits the interpretability of the model.
    As shown in the ablation study, model performance is sensitive to guidance scale $\tau$, which limits the application of the model in real-world setting.
    The model is a direct combination of previous works ([1] and [2]), the technical novelty is somewhat limited.


[1] Wen, Haomin, et al. "Diffstg: Probabilistic spatio-temporal graph forecasting with denoising diffusion models." Proceedings of the 31st ACM international conference on advances in geographic information systems. 2023.
[2] Singhal, Raghav, et al. "A general framework for inference-time scaling and steering of diffusion models." arXiv preprint arXiv:2501.06848 (2025).

### Questions
How the estimated parameters of mechanistic model change over time? And how did you choose the prior?
    How does the framework perform when the mechanistic model is mis-specified, which is a common scenario for real-world data with small sample size?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces EpiDiff, a hybrid forecasting framework designed for spatiotemporal epidemic prediction. It combines mechanistic epidemic models (like SIR) with a diffusion-model-based deep learning backbone. EpiDiff classifies and quantifies uncertainty in mechanistic estimations and uses this uncertainty to guide a spatiotemporal diffusion model during inference, adapting the influence of mechanistic models based on their confidence. Extensive evaluations on COVID-19 and influenza datasets demonstrate EpiDiff’s superiority over state-of-the-art baselines in both accuracy and robustness, especially under distribution shifts. The model also improves interpretability by quantifying and visualizing uncertainty, providing explainable forecasts that help gauge reliability. The approach is validated with comprehensive experiments, ablation studies, and sensitivity analysis, and the authors commit to releasing their code and data for reproducibility.

### Strengths
1. Unified framework for leveraging mechanistic epidemic models with spatiotemporal diffusion model for performance plus interpretability is novel
2. Uncertainty-aware guidance which can modulate mechanistic parameters on forecasts are very useful
3. Extensive experiments show SOTA performance over popular baselines: both traditional, deep learning and hybrid methods.
4. Case studies shoe importance of interpretability in visualizing and quantifying uncertainty
5. Ablation studies are extensive to show importance of important methodological choices

### Weaknesses
1. SIR dynamics are too simple for many epidemics. How scalable is it for more complex mechanistic models?
2. How sensitive are model performance w.r.t mechanistic estimates? Do these estimate correlate with other models or reports for specific epidemics?
3. How scalable is it across number of geographies? Analysis on complexity and running time would be useful

### Questions
See questions

### Soundness
3

### Presentation
3

### Contribution
4
