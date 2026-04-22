# ResCast: Enhancing Global Medium-range Precipitation Forecasting with Residual Diffusion Model

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Machine learning techniques have been successfully applied to global weather forecasting, achieving significant results across various applications. However, existing data-driven machine learning methods struggle to provide accurate medium-range meteorological predictions. As a result, precipitation forecasts regressed from these predictions are less accurate, making reliable medium-range precipitation forecasting difficult. The root causes of these issues are error accumulation in meteorological variable forecasts and a lack of effective variable interaction in the precipitation regression module. In this paper, we propose ResCast, a novel approach to global medium-range precipitation forecasting by combining meteorological residual diffusion modeling and precipitation regression.
The diffusion component consists of (i) the Details Network (DetNet), which captures global features, and (ii) the Multi-Attention U-Net (AttUnet), which generates residuals for meteorological variables to reduce prediction bias. Then, a precipitation regression module quantifies the influence of residual-enhanced meteorological variables on precipitation, improving forecast accuracy. We evaluate our approach on the ERA5, an established dataset from the ECMWF, using comprehensive metrics and compare global medium-range precipitation forecasts against four state-of-the-art baselines (such as ENS, GraphCast, etc.). The results demonstrate the effectiveness and superiority of the proposed framework. The code implementation can be found in https://anonymous.4open.science/r/ResCast-78BD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors outline a method using residual diffusion with an additional L2 loss for medium-range precipitation forecasting. Furthermore, they propose a precipitation regressor that uses multi-frequency 2D-DCT components to weight channels and output rain fields. Results demonstrate a moderate improvement over previous SOTA methods.

### Strengths
- The methodology is described in detail, and the appendix and codebase aid reproduction of results. There are clear metric definitions including cosine-latitude weighting and explicit ACC/F1 formulas.

- The diffusion residuals are described with an explicit training objective.

- The ablation study shows the core contribution, the residuals, helping core-variable skill as lead time grows.

### Weaknesses
- In my view, the novelty of the paper makes it more suited for a climate/earth science conference or journal. What’s implemented is essentially a conditional residual DDPM (AttUNet) guided by DetNet features, then a DCT-based regressor. This is something interesting and practical for weather modelling, but not a new diffusion paradigm.

- The comparison with GenCast is important and should be in results instead of appendix. This also raises the question of the advantage of this model over GenCast. 

- The "Conservation Law Loss" needs either better justification or renaming. The loss function reduces to L2 on residuals. While this encourages small residual energy, it doesn't encode flux balances. On line 944, it is unclear how the flux equation relates to the residual equation.

- A main limitation mentioned, the error accumulation and over-smoothing, perhaps needs more evidence.

Minor things:
- Algorithm 2 cites Eq ??
- Consider moving all the math in the main paper to one section for clarity. Lots of variables are introduced so it will be easier to keep track.
- line 98: Unet -> U-Net
- Table 1: HERS -> HRES (I assume)

### Questions
See weaknesses:

- How do you justify the technical novelty of the paper?

- What specific advantages are there of using this model over GenCast?

- Can you define (1) Error Accumulation in predicting fundamental meteorological variables; (2) insufficient modeling of Variable Interaction?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ResCast, a novel framework designed to enhance global medium-range precipitation forecasting. It addresses the critical challenges of error accumulation and insufficient variable interaction prevalent in current data-driven methods. The ResCast framework integrates a residual diffusion model with a precipitation regression module. It first employs DetNet and AttUnet to capture global trends and generate residuals, thereby improving the prediction accuracy of fundamental meteorological variables. These enhanced variables are then utilized for the precipitation regression task. Experimental results on the ERA5 dataset demonstrate that ResCast outperforms several state-of-the-art machine learning models and traditional numerical weather prediction systems, exhibiting accuracy and stability across multiple evaluation metrics.

### Strengths
* The paper introduces a residual diffusion approach. By focusing on predicting residuals, this method effectively mitigates the over-smoothing problem commonly observed in traditional meteorological forecasts.
* The introduction of a Conservation Law Loss function is a novel idea.
* The paper conducts a comprehensive and systematic evaluation against a strong and diverse set of baselines. This includes comparisons with both traditional numerical weather prediction models (e.g., ENS, HRES) and state-of-the-art machine learning models (e.g., GraphCast, FourCastNet).
* The proposed model demonstrates significant performance gains across multiple evaluation metrics. Its superiority is particularly pronounced in the medium-range forecasting window.

### Weaknesses
1. **Lack of Theoretical Justification for the Diffusion Model.** The paper lacks a rigorous theoretical analysis explaining **why diffusion models are inherently well-suited** for predicting meteorological residuals. A core motivation is tackling error accumulation, yet the mechanism by which the residual diffusion component mitigates this remains underdeveloped. The authors should provide a deeper analysis, perhaps by visualizing residual trends or quantifying the suppression of error growth in key variables (e.g., temperature, humidity), as the current explanation is relatively superficial.

2. **Unclear Physical Significance of the Conservation Law Loss.** The physical significance of the proposed Conservation Law Loss ($L_{conservation}$) is not sufficiently clear. Although Appendix C.3 provides a theoretical derivation, its **connection to the practical impact** observed during training (e.g., the Energy Ratio in Figure 7) needs to be more explicitly established.

3. **Weak theoretical grounding for the integration of residual and base predictions.**  
  The paper does not sufficiently address a fundamental question: if the base predictor (e.g., ViT) were highly accurate, would residual correction still be necessary or beneficial? As illustrated in Figure 3, the proposed pipeline involves a multi-stage refinement process. However, it remains unclear whether **errors or biases introduced at intermediate stages** could propagate and potentially degrade the final prediction—especially if the residual model itself learns to compensate for systematic flaws in the base model rather than genuine physical residuals.

4. **Vague Definition of "Global Trend Knowledge".** The concept of "global trend knowledge," which DetNet is purported to capture, is **vaguely defined** and lacks a precise formulation.

5. **Insufficient Ablation Studies.** The ablation studies are not comprehensive enough to isolate the contributions of key components. The analysis is limited to DetNet and the Conservation Law Loss, while the **contribution of AttUnet is not analyzed at all**. Critically, the paper lacks a direct comparison against a baseline *without* the residual mechanism (e.g., only the ViT backbone) for both the base variables and the final precipitation forecast. This omission makes it difficult to clearly ascertain the **independent contribution** of the residual diffusion component.

### Questions
1. In Figure 18, ResCast exhibits slightly worse performance than GenCast in the 8–15 day forecast range. Could the authors analyze the potential reasons for this degradation?

2. In Appendix F (Discussion), the authors mention computational resource constraints and note that training was stopped at 10,000 steps. Given that ResCast is a multi-stage model with potentially high training overhead, could the authors provide further clarification on the computational cost (e.g., GPU-hours, memory usage) and how it compares to baseline models like GenCast?

3. The paper claims that ResCast addresses two key limitations: error accumulation in meteorological variable forecasts and insufficient variable interaction in the precipitation regression module. If these issues are effectively mitigated, why was the model not evaluated over even longer forecast horizons (e.g., beyond 15 days) to demonstrate its potential advantages in extended-range prediction?

4. Several figures in the paper would benefit from improved visual design.

### Soundness
3

### Presentation
2

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
The paper proposes ResCast, a novel approach to global medium-range precipitation forecasting. First, the authors design a diffusion-based residual component, including the Details Netword (DetNet) and Multi-Attention Unet (AttUnet), to effectively predict residuals, reducing error for fundamental meteorological variables. Second, the authors introduce a precipitation regression module to quantify the interactions between meteorological variables. This mechanism improves the accuracy of precipitation forecasts. ResCast surpass machine learning baselines (GraphCast, FourCastNet) and state-of-the-art traditional NWP models (ENS, HRES) on the ERA5 dataset.

### Strengths
S1. The authors design the Fundamental Meteorological Predictor based on the Vision Transformer (ViT) to effectively capture spatial information. To obtain more accurate predictions for the next step, the architecture uses the Lat-weighted MSE to optimize the parameters of the deterministic predictor. 

S2. The authors propose a Precipitation Regression Component to calculate the influence scalar for each channel. The idea is interesting and promising. 

S3. The motivation shown in Figure 2 is interesting and easily understandable.

### Weaknesses
W1. The paper shows limited originality, as the proposed approach builds upon existing diffusion-based and regression frameworks without introducing methodological innovations.

W2. According to Table 1, the experimental comparison with state-of-the-art models is insufficiently comprehensive. Comparisons with more SOTA models are needed.

W3. In the red box of Stage 1 in Figure 3, the description of the diffusion process is unclear and lacks sufficient detail. 

W4. The paper mentions that the dataset used is ERA5. However, no additional datasets were employed in the experiments to verify the model’s performance.

### Questions
1. In the caption of Figure 3, the authors mention that “DetNet (yellow) is used to extract global motion information”. However, DetNet adopts a ResNet-like structure, which typically extracts spatial features rather than temporal ones. Why can temporal features be extracted here? Or in other words, why does the model need to extract temporal features at this stage?
2. In Stage 1 of Figure 3, the diffusion process within the red box is unclear, and the green box part has not been explained. 
3. Regarding Table 1, what metrics do the numbers such as 5 and 6 in the first row represent?
4. Regrading Figure 4, in the 10-day performance, I find that GraphCast appears to be closer to the ground-truth than ResCast (both in the lower-left boxed area and in the extreme precipitation regions).
5. In the ablation study, the authors mention that “our precipitation regression module achieves improvements of 5%-10% across multiple metrics”. Where are these results presented? It would be clearer if the comparative results were shown in a table. 
6. Regarding the Impact of Conservation Law Loss, it seems that the paper does not provide detailed experimental results to demonstrate how significant the effect of the Conservation Law Loss is.

### Soundness
3

### Presentation
3

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
The paper proposes a machine learning framework for medium-range precipitation forecasting, structured as a two-stage pipeline combining meteorological residual diffusion and precipitation regression. In the first stage, residual diffusion enhances the predictions of a Fundamental Meteorological Predictor (based on Vision Transformer) for key atmospheric variables such as temperature, wind, and specific humidity. This refinement is achieved using DetNet (with residual blocks) and AttUnet (with spatial attention) to model residuals, improving prediction accuracy. In the second stage, the corrected meteorological predictions are used by the precipitation regressor that forecasts precipitation through a combination of 2D Discrete Cosine Transforms (DCTs), fully connected layers, and convolutional operations. The approach is evaluated on the ERA5 dataset, focusing primarily on 5–15-day forecast horizons.
While this is a highly relevant and less studied problem, the manuscript has severe shortcomings in description of the approach, in relationship with existing work, and in the empirical evaluation.

### Strengths
- Tackles a problem where not much work has been done (medium-range precipitation forecasting).
- Using predictions for fundamental weather variables to forecast precipitation is an interesting approach that most data-driven models have not considered. However, is it the way ERA5 integrates precipitation into the reanalysis data, so it should be a valid way when observational data is lacking?
- Ablation on the precipitation regressor (Table 2 right) seems strong.
- There are many details in the paper and code.

### Weaknesses
Models
-	Incorrect claims on ML medium-range forecasting:
	15: “ML models struggle to provide accurate medium-range meteorological predictions” is not correct. Checking the WeatherBench2 scorecard, and papers like GraphCast, Aadvark, Aurora, GenCast, PanguWeather, AIFS, etc. shows how good ML models are compared to traditional approaches.
- Incorrect claims on precipitation prediction:
The paper claims medium-range forecasting models like GraphCast, GenCast, FuXI, FourCastNet, and ClimaX predict precipitation by regressing precipitation from fundamental meteorological variables at the desired lead time (16, 79, 252, 446)
•	To my knowledge, only FourCastNet has a AFNO model that takes the predicted state for outputting precipitation.
•	ClimaX only works with annual precipitation in the ClimateBench dataset.
•	GraphCast and GenCast do not explicitly model precipitation and even claim to not evaluate it due to ERA5 biases (more details below).
•	FuXI does not provide details.
- the paper claims that precipitation forecasting remains a significant challenge (56). It is not only because of performance (18) and specific nature (41-46), but also because of the data and evaluation (details later). Most models use ERA5, which shouldn’t be used for precipitation (GraphCast, WeatherBench2).
o	NeuralGCM is not considered at all in the introduction, related work, or baselines. However, it outperforms the mid-range precipitation forecast of the ECMWF ensemble on precipitation.

Experiments / Data
o	Baselines: only ML models compared are GraphCast and FourCastNet. Lacking are GenCast (diffusion based) and NeuralGCM (best model evaluating precipitation).
- GraphCast does not consider precipitation: “Note, we exclude total precipitation from the evaluation because ERA5 precipitation data has known biases [15]”.
- FuXI claims to output precipitation, but it is not evaluated either.
-	GenCast also highlights limitations of evaluating with ERA5, yet shows that it outperforms ENS model: “Owing to our lack of confidence in the quality of ERA5 precipitation data, we exclude precipitation results from our main results and refer readers to Supplementary Information section B.2”.
- The paper considers GenCast, but only for a sample case without reporting aggregated metrics and including results in Table 1. The case analyzed in Appendix G3 even shows greater performance for GenCast than ResCast.
- NeuralGCM is not considered in the inputs, even though it is one of the leading medium-range precipitation forecasting models.
- The paper claims that models trained on ERA5 data have reached the capability of medium-range precipitation forecasting, but they have to surpass traditional NWP models (70-73). Which models other than FourCastNet? 
-	The paper even cites WeatherBench2 (310). However, WeatherBench2 explicitly states: The quality of ERA5 depends on the variable in question. For surface variables, the sparsity of observations and difficulty of representing smallscale physics in the underlying model can cause larger discrepancies with observations. This is especially true for precipitation, which is not directly assimilated into ERA5 (e.g., through radar observations) and often show large differences to rain gauges or radar precipitation estimates (e.g., (Lavers et al., 2022), (Andrychowicz et al., 2023)). The precipitation evaluation using ERA5 shown here should really be seen as a placeholder for more accurate precipitation data. Operational weather services like ECMWF verify their forecasts with direct observations, e.g., from weather stations, in addition to using assimilated ground truths. This is something we are looking to add to WeatherBench in the future.
- Fig 15 compares against stations, but it is only one sample and no metrics. Also, it is unclear the range of the color scale and how extreme the different colors can be. 
- Future work hopes to integrate satellites and radar and it is essential for avoiding biases in ERA5. Shouldn’t this be required for evaluation (i.e. using observational data known to be more accurate)?
- NeuralGCM: We consider both IMERG and the Global Precipitation Climatology Project [37] (GPCP; a dataset not used in training) as ground truth for precipitation.
- ERA5 does not use observations and estimates precipitation from other parameters, so it would be a reason why ResCast is able to capture precipitation like this. Evaluation should be considered with additional data.

Experiments:
-	The evaluation is only on days 5-15. What about days 1-5? 
-	Most models are available 1-10 days, wouldn’t it make sense to compare with that as well?
-	Why is GenCast not reported in Table 1 as a baseline?
-	The metrics only consider CSI and F1 at 1mm for daily accumulation (very light rainfall). This is extremely low for daily accumulations. 
-	354: claim better extreme value performance, but no metrics?
-	485: future work includes to study at least one case study of extreme precipitation. So, it has not been considered yet?

Contributions:
-	Contribution 1: residual predictions reduce error for fundamental meteorological variables.
-	Why is this not evaluated? The predictions should be better than other ML models and NWPs. Overall metric evaluation of temperature, wind, specific humidity, and other predicted variables should be reported.
-	The ablation shows that the residuals help in ResCast. But is ResCast better than the other models? Compare with ML models and NWPs.
-	Contribution 2:
-	443: regressing precipitation is essential. Is this proved? Does it perform better than GenCast that predicts precipitation directly? Tried to predict precipitation directly? 
-	468-473: Claim is that direct precipitation prediction fails to meet medium range forecasting needs (GenCast). Is this proved anywhere? Could include GenCast in Table 1 at least…
-	At least the ablation on the precipitation regressor (against other regressor) looks strong.
-	Conservation Law Loss (208). Even though it claims the conservation law in fluid dynamics and physical constraints, it just makes sense as a regularization term to encourage small residual values (meaning that the Fundamental Meteorological Predictor needs to make more accurate predictions and rely less on the residual correction).
-	411: The result makes sense, but it may be more about having residuals close to 0 than trying to apply conservation laws.
-	457: “Generating precipitation prediction can violate physical constraints (energy conservation)”. Can this not also happen with the precipitation regressor? Figure 7 doesn’t show energy ratio of 1 constantly…

Minor comments
-	431: Diffcast is not deterministic, it is a combination of a deterministic module and a stochastic module.
-	Highlight the spatio-temporal resolution of ResCast (daily and coarser than ERA5).
-	Table 2 makes no sense to share rows between the ResCast ablation and Precipitation Regression ablation.
-	51: a lot more successful models that (Pathak et al., 2022; Hu et al., 2023; Nguyen et al., 2023) – GraphCast, GenCast, Aurora, Aadvark, Pangu-Weather…
-	Graphcast -> GraphCast (53)
-	Which -> these (185)
-	Correct citation style in text: 154, 291, 300

### Questions
- Contradiction between 161 and 917 in App C.3. Are the parameters of the Fundamental Meteorological Predictor updated or frozen?
- Eq. 9: Is it really summation? It has dimension H x W (266), right? 
- Please clarify how the fully connected layer works in the precipitation regressor? Is it unclear from the text (269) and the figure (3). Is it across channels or do you flatten the spatial dimensions?

### Soundness
2

### Presentation
2

### Contribution
2
