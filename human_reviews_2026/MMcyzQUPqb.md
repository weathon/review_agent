# ClimateAR: Multi-Scale Autoregressive Generative Modeling for Climate Forecasting

- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Accurate Seasonal‑to‑interannual climate forecasting provides critical support for decision-making in agriculture, energy, and disaster preparedness. Current deterministic models often fail to capture climate uncertainty, while existing generative approaches oversimplify the system by neglecting key spatiotemporal dependencies and cross-scale interactions. To address these limitations, we introduce **ClimateAR**, an AutoRegressive generative model for probabilistic Climate forecasting. The framework incorporates two novel components: (1) an aligned tokenizer that bridges and aligns heterogeneous simulation and real-world data to improve transferability across domains, and (2) a mixed-scale conditioning mechanism that captures multi-scale climate interactions for robust probabilistic forecasting. Extensive evaluations on the ERA5 reanalysis dataset show that ClimateAR achieves state-of-the-art performance, improving anomaly correlation skill by 29.27\% on average compared to leading baselines. Code is available at https://anonymous.4open.science/r/ClimateAR-956D.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ClimateAR, an autoregressive generative model for probabilistic climate forecasting. It addresses limitations in existing deterministic and generative models by capturing multi-scale spatiotemporal dependencies and climate uncertainty. ClimateAR features two key innovations: an aligned tokenizer for bridging simulation and real-world data, and a mixed-scale conditioning mechanism to model cross-scale climate interactions. Extensive evaluations and training is done on the range of datasets including ERA5, CMIP6 and ORAS5 datasets. The results show that ClimateAR outperforms state-of-the-art weather forecasting models. The model demonstrates strong performance in forecasting critical climate phenomena like El Niño–Southern Oscillation (ENSO). ClimateAR is computationally efficient and offers robust transferability across simulated and real-world datasets.

### Strengths
The paper addresses critical gaps in existing models by incorporating multi-scale spatiotemporal dependencies and explicitly modeling climate uncertainty. The use of an aligned tokenizer to bridge simulated and real-world data and a mixed-scale conditioning mechanism to capture cross-scale interactions represents a creative combination of existing ideas tailored to the unique challenges of climate forecasting. The model's robustness is validated through zero-shot forecasting, ENSO prediction, and ablation studies, showcasing its adaptability and effectiveness.

### Weaknesses
One of paper's main claim is "the capture of inherent climate uncertainty". However, only RMSE and ACC are used as evaluation metrics and it is nowhere quantified in the paper of how uncertain or certain the model is. If authors can run experiments on few ensembles and quantify uncertainty using metrics like CRPS or log likelihood. This would strengthen the paper.
When referred to Climate Forecasting, it goes beyond monthly forecasts upto several years or decadal forecasts. The results are shown upto 10 months only which in my opinion doesn't justify the climate forecasting part.
The evaluation done against the data-driven models such as graphcast, pangu weather were inherently built for weather forecasting (from short-term to medium-range forecasting upto 10-15 days, not climate forecasting.
Also, the RMSE results show the performance tend to de grade after 8 months lead time on several variables.

### Questions
Please address some of the concerns mentioned in weakness sections.
I think the problem needs to be reframed as long-term weather forecasting instead of climate forecasting.
OR
The forecasting and evaluation framework needs to be adjusted according to the climate forecast problem.
There are many works on climate forecasting that goes beyond yearly forecasts. Please have a look at them. Few of those works include ACE2 and NeuralGCM.
If the framework and evaluation is reformulated fairly, I'll consider changing my score.

### Soundness
3

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
This paper adapts a visual autoregressive (AR) model for probabilistic climate forecasting with two key innovations. First, an aligned tokenizer with a "shallow-separation, deep-sharing" architecture bridges the domain gap between simulated and real-world data. Second, a mixed-scale conditioning mechanism effectively handles high-dimensional climate inputs by combining global and local guidance.

### Strengths
This paper introduce an innovative approach to tackle to problem of long term predictions. In addition investigate the domain shift between simulated and real-world data. The model achieves impressive results, showing a significant average improvement in ACC over tested baselines.

### Weaknesses
- A central claim of the paper is the model's superiority in probabilistic forecasting and capturing climate uncertainty. However, the primary evaluation relies on the Anomaly Correlation Coefficient (ACC), a deterministic metric that assesses the capability of the ensemble mean. This metric does not evaluate the quality of the predicted probability distribution itself (e.g., its spread or calibration). To fully substantiate the claims of probabilistic skill, the inclusion of a proper probabilistic metric, such as the Continuous Ranked Probability Score (CRPS) and Spread/Skill-Ratio would be essential.
-  While the paper provides an efficiency study for model training, it omits a detailed analysis of the inference cost. Key details regarding the inference time and computational resources required to generate a full ensemble forecast are missing. This information is critical for assessing the model's practical viability for operational applications, where computational efficiency is often a major constraint. In addition it would be helpful to have a better understanding if the gain obtained from the new architecture comes from a higher complexity compared to the baseline.

### Questions
- The VQ tokenizer is a critical component of ClimateAR, yet its design choices lack a detailed ablation study. Specifically:
a) Codebook Utilization: The paper does not provide an analysis of the codebook utilization or perplexity. This is important, as low utilization (i.e., "codebook collapse") can indicate an inefficient latent space where only a fraction of the learned codes are used, suggesting the codebook may be unnecessarily large.
b) Codebook Size: While the authors provide a hyperparameter study for the number of codebooks (partitions N), there is no corresponding study on the size of the codebooks (V, set to 4096). An ablation on this key hyperparameter would be needed to understand the trade-offs between representational capacity and model efficiency.
- The paper claims probabilistic superiority but primarily uses a deterministic metric (ACC of the ensemble mean). Could the authors provide results using a proper probabilistic score, such as the Continuous Ranked Probability Score (CRPS), to validate this core claim?
- How does the model ensure that the generated token sequences decode into physically plausible climate states, especially concerning conservation laws? Was any analysis performed to verify this?
- What is the practical inference cost to generate a full 200-member ensemble forecast, and how does this compare to the operational baselines?
- Could the authors provide a targeted ablation study to isolate the impact of the hybrid-scale prompt (C_mix) on the model's overall performance?
- Given the widespread success of diffusion models in scientific generative modeling, the paper lacks a compelling justification for choosing an autoregressive framework. Could the authors either include a state-of-the-art conditional diffusion model as a baseline or provide a more rigorous argument for why the AR paradigm is fundamentally better suited for long-range climate forecasting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a visual autoregressive model for climate forecasting. Given one timestep of climate model data, it predicts a distribution over the next timestep. It computes tokens corresponding to multiple resolutions of the data, and uses a transformer to predict the tokens at the next timestep,  before decoding the tokens in observations space, effectively capturing dynamics happening at multiple spatial scales. 

Authors evaluate the model on ERA5 and ORA5 data, and show that the model outperform all baselines.

### Strengths
The paper is sound and well written. Using a visual autoregressive model to capture multi-scale dependencies is a great idea, and the model also allows to represent uncertainty, although this is not explored nor validated by experiments.

### Weaknesses
The experiments and results could be strengthened. 

First, the RMSE is not a great metric for climate emulators. When doing climate forecasting, the variables are not predictable and we're instead interested in capturing climate statistics and dynamics. It is thus important to report additional metrics, such as mean + std. dev., or the power spectral density of the climate indices (as done in [1]) or the time-dependent area-weighted global mean and the area-weighted global mean bias and RMSE of time-mean fields (as done in [2]).

Second, it would be interesting to highlight learned relationships between multi-scale processes. For example by computing regression coefficient between climate indices (such as ENSO or IOD) with variables over the entire grid. 

Third, authors claim that the model does probabilistic climate forecasting, but this is not shown or evaluated in any of the experiments. Does the model learn a sensible uncertainty representation? I believe that the authors need to evaluate it since they argue that it is a strength of the model. To do so, it might be useful to evaluate the model on climate model data, where there are many more available years  than ERA5. Authors could show the uncertainty in the climate forecast (does it increase as you forecast longer time horizons?), continuous ranked probability score (by looking at different initial conditions for the different models) and maps of the uncertainty to check if predictable processes are associated with less uncertainty (are oceanic temperatures associated with less uncertainty than atmospheric temperatures over land?). 

The authors should also highlight the limitations of the model in the discussion section.  
 
[1] Hickman et al., Causal Climate Emulation with Bayesian Filtering, 2025

[2] Watt-Meyer et al., ACE: A fast, skillful learned global atmospheric model for climate prediction, 2023

### Questions
The Intra-scale Mixed Token paragraph is a bit unclear to me. 

In Equation 8, why are you doing down(f')? Isn't f'k already at the correct resolution? 

Why does it help to do the intra scale mized token? Since the "f" is essentially a concatenation of the r (from Fig. 1 a), and this is replacing the r with the f. 

You're then adding all r' into Cmix, but r' for low k are already contained in the Intra-scale Mixed Token. Isn't this information contained twice then? Or is Cmix only looking at higher resolution tokens?

### Soundness
3

### Presentation
4

### Contribution
3
