# Improving Extreme Wind Prediction with Frequency-Informed Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Accurate prediction of extreme wind velocities has substantial significance in industry, particularly for the operation management of wind power plants. Although the state-of-the-art data-driven models perform well for general meteorological forecasting, they may exhibit large errors for extreme weather—for example, systematically underestimating the magnitudes and short-term variation of extreme winds. To address this issue, we conduct a theoretical analysis of how the data frequency spectrum influences errors in extreme wind prediction. Based on these insights, we propose a novel loss function that incorporates a gradient penalty to mitigate the magnitude shrinkage of extreme weather, and we theoretically justify its effectiveness via a PDE-based energy–enstrophy analysis. To capture more precise short-term wind velocity variations, we design a novel structure of physics-embedded machine learning models with frequency reweighting. Experiments demonstrate that, compared to the baseline models, our approach achieves significant improvements in predicting extreme wind velocities while maintaining robust overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the persistent "amplitude underestimation" problem in extreme wind speed prediction. Through a detailed frequency-domain theoretical analysis, it demonstrates that standard data-driven models (trained with MSE loss) systematically underestimate the magnitude and short-term variability of high-frequency wind components. To address this, the authors propose a novel loss function with gradient penalization to mitigate amplitude shrinkage, and design a physics-embedded architecture (leveraging the Navier-Stokes equation) along with frequency separation and reweighting modules. Experiments on ERA5 meteorological datasets show that the proposed method significantly outperforms classic models (such as CNN, ConvLSTM, PINN), especially for extreme wind scenarios, and captures both overall and extreme case prediction more accurately.

### Strengths
The work skillfully combines rigorous frequency-domain analysis (revealing frequency-dependent amplitude shrinkage) with tailored engineering solutions in both loss function and model architecture—providing an effective, innovation-driven answer to a real-world pain point in extreme wind forecasting.

### Weaknesses
The core innovation (gradient-penalized loss) lacks thorough theoretical exploration regarding its effect during optimization. The observed instability with large λ is only empirically described (U-shaped curve), with no detailed theoretical analysis as to why or how excessive λ causes non-convergence, possible oscillation, or even gradient explosion, nor are additional regularization strategies discussed.

Although the model's backbone is described as a combination of physics-embedding and neural networks, experiments do not present clear ablation studies to isolate the contribution of the Navier-Stokes module. Only coarse comparisons ("Ours" vs "NS-Op") are given, so the true benefit of embedding physics remains ambiguous.

Extreme wind samples are inherently rare, but the paper does not clarify how many such cases exist in the data, nor does it analyze the trade-off between extreme case data volume and model robustness. The generalization capability (e.g., performance when transferring to new regions or under few-shot settings) is not systematically evaluated.

### Questions
For gradient penalization's optimization instability at large λ, can the authors provide a detailed theoretical convergence analysis? Is there a risk of uncontrolled high-frequency oscillations? Are there any additional regularization measures?

Can the authors supply full ablation experiments for the Navier-Stokes physics module to quantitatively show its standalone benefit compared to standard neural network baselines?

How robust is the model when extreme wind samples are extremely scarce or when transferring to new regions? Can the authors add few-shot or cross-region generalization experiments?

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
3

### Summary
This paper introduces an improved physics-informed approach for extreme wind prediction.

### Strengths
- An interesting theoretical view of frequency-domain error behaviors to enhance the model design for extreme wind prediction
- Significantly improved performance demonstrated in experiments

### Weaknesses
- Limited domain: the proposed method is only applicable to extreme wind prediction
- Limited evaluation: the experiments are based on sampled data, and the forecasting horizon is fixed as one-hour and the lookback includes 23 hours.

### Questions
- Please give a comprehensive introduction of your sampled data, such as how many hours in total, across which regions, etc.
- The frequency masking level seems to be a critical hyperparameter. Did you find an optimal masking threshold to be consistent across different geographic regions and weather regimes, or does it require case-specific tuning? Is there a potential to make this threshold learnable?
- Besides, the forecasting horizon and lookback length could be importance factors to see the robustness of the proposed approach.

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
2

### Summary
This paper investigates the underestimation of extreme wind speeds in data-driven weather forecasting models, a persistent issue in both academic research and industrial applications such as wind power management. The authors propose a frequency-aware learning framework that integrates theoretical analysis in the Fourier domain with innovative model and loss function design.

The study first demonstrates through analytical proof that traditional mean squared error training induces frequency-dependent amplitude decay, where high-frequency components are systematically weakened due to spatial translation errors. To address this, the authors propose:
(1) A gradient penalty loss function to enhance sensitivity to amplitude errors and mitigate high-frequency signal attenuation.
(2) A physically embedded neural network architecture: Employing a simplified Navier-Stokes core, it integrates physically grounded convection, diffusion, and pressure modules with a learnable “volume force” network.
(3) Band separation and weight rebalancing mechanism: Input signals are decomposed into high- and low-frequency components, processed separately via Fourier filters, and equipped with time-domain attention mechanisms for each band.

Experiments on the ERA5 meteorological reanalysis dataset demonstrate that this model consistently outperforms CNN, ConvLSTM, and PINN benchmark models in RMSE metrics, achieving significantly improved accuracy in extreme wind regions. Analysis further confirms the stabilizing effect of gradient penalties and the optimal trade-off relationship regulated by their coefficient λ.

### Strengths
(1) Strong theoretical insight and motivation: The paper provides a clear Fourier-domain error decomposition explaining why high-frequency components are underestimated by standard MSE. The analytical backing is mathematically sound and bridges physical intuition and machine-learning loss design, a commendable improvement over purely empirical approaches in weather forecasting.
(2) Innovative frequency-informed hybrid framework: The integration of gradient-penalized loss, Navier–Stokes-based backbone, and frequency-domain reweighting demonstrates an elegant blend of physics knowledge and deep learning. The model architecture and spectral treatment are well-motivated, providing an interpretable mechanism for addressing extreme-event underestimation.
(3) Comprehensive experimental design and convincing results: Clear comparisons with strong baselines such as CNN, ConvLSTM and PINN. Analysis of the λ hyperparameter and frequency masking ablations demonstrates methodological robustness. Results on both overall and extreme-attentive errors substantiate the theoretical claims.

### Weaknesses
(1) Limited experimental diversity and scale: The study relies on ERA5 data from selected regions, with no cross-region or cross-time validations to assess generalization ability. Suggestion: Extend evaluations to multiple climate zones and temporal ranges (e.g., 6h, 48h forecasts) to assess robustness and transferability of the frequency-informed model.
(2) Computational overhead and implementation detail gaps: The physics-embedded backbone and frequency separation introduce heavy computation during both training and inference. Quantitative data on training time, convergence speed, or complexity trade-offs are missing. Suggestion: Provide complexity analysis versus baseline models, and discuss deployment feasibility for operational forecasting.
(3) Limited connection between frequency-domain theory and physical interpretability: The gradient-penalized term and frequency reweighting are theoretically justified, but it remains unclear how these modifications alter learned spectra or physical consistency over time.
Suggestion: Include frequency-spectrum visualizations pre- and post-training, or energy distribution comparisons to ground truth, to confirm the mitigation of amplitude shrinkage empirically.

### Questions
（1）How does the proposed framework perform under longer-term forecasts or coarse-resolution settings where statistical noise dominates over high-frequency content?
（2）Can the gradient-penalized loss lead to overfitting sharp gradients or instability in turbulent regions, and how is λ chosen or adapted dynamically during training?
（3）How generalizable is the physics-embedded structure? Could it extend effectively to 3D atmospheric models or other variables (temperature, humidity) without major redesigns?

### Soundness
3

### Presentation
3

### Contribution
3
