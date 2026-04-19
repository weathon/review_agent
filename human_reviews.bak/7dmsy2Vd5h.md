# Comparing and Contrasting Deep Learning Weather Prediction Backbones on Navier-Stokes and Atmospheric Dynamics

- Decision: Reject
- Scores: 8, 3, 5, 3

## Abstract
Remarkable progress in the development of Deep Learning Weather Prediction (DLWP) models positions  them  to become competitive with traditional numerical weather prediction (NWP) models. Indeed, a wide number of DLWP architectures---based on various backbones, including U-Net, Transformer, Graph Neural Network (GNN), and Fourier Neural Operator (FNO)---have demonstrated their potential at forecasting atmospheric states. However, due to differences in training protocols, forecast horizons, and data choices, it remains unclear which (if any) of these methods and architectures are most suitable for weather forecasting and for future model development. Here, we step back and provide a detailed empirical analysis, under controlled conditions, comparing and contrasting the most prominent DLWP models, along with their backbones. We accomplish this by predicting  synthetic two-dimensional incompressible Navier-Stokes and real-world global weather dynamics. In terms of accuracy, memory consumption, and runtime, our results illustrate various tradeoffs. For example, on synthetic data, we observe favorable performance of FNO; and on the real-world WeatherBench dataset, our results demonstrate the suitability of ConvLSTM and SwinTransformer for short-to-mid-ranged forecasts. For long-ranged weather rollouts of up to 365 days, we observe superior stability and physical soundness in architectures that formulate a spherical data representation, i.e., GraphCast and Spherical FNO. In addition, we observe that all of these model backbones ``saturate,'' i.e., none of them exhibit so-called neural scaling, which highlights an important direction for future work on these and related models. The code is available at \url{https://anonymous.4open.science/r/dlwp-benchmark-F88C}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a fair comparison of the performance of widely used deep learning models for weather prediction. The authors standardize parametric settings, inputs, outputs, and training methods across models, and evaluate their performance using Navier-Stokes dynamics simulations, as well as medium- and long-term weather prediction. The study highlights each model's strengths and weaknesses.

### Strengths
This paper addresses a significant gap in the field, as there is currently no comprehensive and fair comparison of DLWP models. While many studies claim superior performance for their models, it remains unclear whether this is due to the backbone architecture, diagnostic variables, training, or inference strategies. By focusing specifically on the backbone models, the authors conduct rigorous experiments to empirically assess their forecasting potential. The findings are valuable for the DLWP field, offering novel insights, such as the superior performance of ConvLSTM, differences between Pangu-Weather and SwinTransformer, and the influence of FourCastNet's patch size.

### Weaknesses
1. The experiment of synthetic Navier-Stokes simulations seem to play a limited role. As noted in line 160, there is a significant gap between the univariate Navier-Stokes simulation and real atmospheric dynamics. Given that the authors aim to evaluate backbone models' performance in the more complex weather forecasting task in section 3.2, dedicating one-third of the main text to the simpler univariate Navier-Stokes simulation seems unnecessary. In light of the results from section 3.2, the findings from section 3.1 appear less relevant and, in fact, somewhat confusing:
    
    (a)  Section 3.1 highlights the superiority of TFNO, yet this model is absent from section 3.2. Since TFNO appears in Figures 13, 14, and 15, its performance in RMSE metric should have been assessed by the authors.
    
    (b) In Figures 13, 14, and 15, TFNO, ConvLSTM, and UNet underperform compared to other models.
    
    (c) In Figure 16, models that perform well in section 3.1 (TFNO, ConvLSTM, UNet) exhibit poor stability.

2. In section 3.2, the authors evaluate the performance of different backbone models using the 5.625 deg ERA5 dataset. The experiments provide limited guidance for selecting backbone models for operational weather prediction, which typically relies on the 0.25 deg ERA5 dataset. However, large-scale experiments at this resolution are obviously costly, so this is an existential but understandable drawback :) .

### Questions
Overall, as the first paper to provide a fair comparison of various DLWP models, this work has the potential to make a significant contribution to the field. However, I recommend the authors reconsider the emphasis placed on section 3.1 and expand the experimental results in section 3.2, particularly by including RMSE metrics for variables such as t850, u10, and v10.

Here are other questions:
1. Lines 291-293: The ACC metric is not provided in section 3.2.1. Additionally, section 3.2.1 presents RMSE metrics for geopotential only up to 7 days, not 14. Given the presence of 8 prognostic variables, it would be beneficial to include the RMSE and ACC metrics for all variables, potentially in the appendix in a format similar to Figure 2. 
2. In Figures 2 and 18, the authors observe that SwinTransformer outperforms Pangu-Weather in terms of RMSE. To my knowledge, the primary difference between Pangu-Weather and SwinTransformer is the use of Earth-specific positional bias in Pangu-Weather. Intuitively, this difference alone should not lead to such a performance gap. I suggest that the authors standardize other hyperparameters (e.g., layers, embedding dimensions) between Pangu-Weather and SwinTransformer, and present additional results, such as RMSE for geopotential with 1M parameters, to clarify this discrepancy.
3. Lines 339-340: The authors limit the optimization cycle to 24 hours (4 steps). While there is no established standard for optimization lead time, I question whether ConvLSTM, being the only RNN-based model, is particularly sensitive to this hyperparameter. State-of-the-art models like FourCastNet (2 steps), Pangu-Weather (1 step), and GraphCast (1 step in pretraining) use shorter optimization cycles. Training with 4 steps may become resource-intensive at higher spatial resolutions, which could be a limitation of ConvLSTM.
4. Lines 340-341: The authors evaluate the backbone models using initial conditions at 00z. I wonder if fixing the initial time at 00z simplifies the overall weather prediction task. Could the authors test whether models trained on 00z initial conditions also perform well with 12z initial conditions in the test set?
5. Lines 489-490: In Figures 5 and 15, the authors note that SFNO performs well in predicting wind fields, accurately capturing real-world wind patterns. They attribute this to SFNO's adherence to physical principles. However, given SFNO's performance in Figure 2, I question whether this claim holds true for all prognostic variables.
6. Line 863: Since $x,y \in \mathbb{N}$, it follows that $x+y \in \mathbb{N}$. Therefore, in the authors’ setting, $f \equiv 0.1$. I think there must be some mistake. Otherwise, the Navier-Stokes simulation is too simple.
7. Line 1228-1229: The ‘]’ of heads in layers in Pangu-Weather is missing.
8. In Figure 14, why are the only two graph neural networks smoothed in Zonally averaged forecasts?
9. In Figures 2, 17, 18, and 19, I observe that the confidence intervals for some models, particularly FourCastNet, are notably wide across the three random seeds. Upon reviewing the code, I suspect this may be due to the gradient clipping, which is set equal to the learning rate ($\leq 10^{-3}$). When multiplied by the learning rate, the step size of the gradient descent ($||\eta *\text{Clip}(\nabla f)||_{2}$) is less than $1\times 10^{-6}$, which is likely too small for effective exploration of the parameter landscape. As a result, model performance may be highly dependent on initial parameters or random seeds. My question is, why was the gradient clipping value set equal to the learning rate? Is there a specific reference for this choice?
10. Line 684-686: unify the reference.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper analyzes the performance of different network structures for weather prediction. Experiments were conducted on both synthetic and real data, and a benchmark was established. They also suggested network structures suitable for mid-term and long-term forecasting.

### Strengths
The design of the backbone network greatly affects the performance of machine learning models. This article provides an analysis of the backbone network's performance in weather forecasting.

### Weaknesses
This article seems like an experimental report. It includes introductions to several classic backbone networks, settings for two experimental datasets, and descriptions of the results. However, this paper lacks insights that previous work did not reveal.

### Questions
1. The authors introduced a new benchmark for weather forecasting, but they didn't clearly explain how it differs from previous research, such as in data construction and task definition.
2. The authors analyzed several backbone networks, but only showed some quantitative results without providing more insights, such as proposing new designs for backbone networks.
3. The authors used synthetic and real data to train these models, but they did not discuss the differences between these data and the data used by existing state-of-the-art models.
4. The number of model parameters used by the authors seems small, but current weather prediction models use a large number of parameters. With such a big difference in parameter count, is the conclusion reliable?
5. With only 1K and 10K samples in experiments 1 and 2, are these numbers too small? Can the conclusions be trusted?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper provides a comparative study of various architectures such as U-Net, Transformers, Graph neural networks, ConvLSTM, and Neural Operators that have shown their potential to serve as backbones in Deep Learning Weather Prediction (DLWP) models. This work includes a systematic and detailed empirical analysis under controlled conditions, controlling for parameter count, training protocol, and prognostic variables. All the models are evaluated by benchmarking on two systems: synthetic Navier-Stokes and real-world weather datasets. The paper focuses on short-to-mid-ranged forecasts, long-ranged (climate length) forecasts, and physics-backed forecasts, intending to provide better architectural design choices supporting the DLWP research for various forecasting tasks. Based on their observation, ConvLSTM is better at short and mid-range forecasts on weather data. For stable long-ranged forecasts, aligned with physics principles, spherical representations such as in GraphCast and Spherical FNO, show superior performance.

### Strengths
- The experiments in this study are extensive, and the analysis is presented in a clear, organized manner. The details of the experiments are thoroughly explained and the set of models chosen for the comparison is justified well.
- The paper includes long-range forecasts for lead times as long as 365 days (and more), which is important and not included in most DLWP studies. These results can be insightful to this line of research.

### Weaknesses
- The spatial resolution used in the paper for global weather prediction is too coarse (5.625 degrees) as compared to the 0.25-degree resolution used in recent weather forecasting models such as FourCastNet, PanguWeather, and GraphCast. 
- Using the backbones of the DLWP models for performing prediction on the Navier-stokes system does not seem very relevant to the contributions of this work. The paper also says “A direct transfer of the results from Navier-Stokes to weather dynamics is limited”. Moreover, FNO working so well on Navier-Stokes has already been shown before. 
- The paper claims to be studying physically meaningful forecasts. This is a crucial aspect of weather forecasting and should be a critical factor in comparing models. However, the paper doesn’t go into much detail on this aspect. For instance, physics-based metrics and power spectrum plots [1] are needed to investigate if the models can capture small-scale (high-freqeuncy) features in their forecasts. 
[1] Nathaniel, Juan, et al. "Chaosbench: A multi-channel, physics-based benchmark for subseasonal-to-seasonal climate prediction." 2024.

### Questions
- What is the justification behind using a coarse spatial resolution for weather prediction?
- The authors should add more on why they chose to evaluate and compare the models on the Navier-Stokes system. 
- There needs to be more analysis to understand the physical soundness of various models. This should include physics-based plots/metrics as suggested before, and a discussion comparing models on this aspect of their forecasting skill.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper aims to conduct a comprehensive evaluation and comparisons of deep learning backbones for weather forecasting. Authors selected seven widely used networks and conducted a large number of experiments on both synthetic dataset and a low resolution Weatherbench dataset.

### Strengths
Pros:

1. Fair and comprehensive evaluations of the influence of network backbones to DLWP is important. 
2. Apart from different backbones, authors also evaluate the influence of parameter numbers, which would be informative for exploring the parameter scaling law for DLWP.

### Weaknesses
Cons:

1. According to my experiments, tuning the parameters for weather prediction (e.g., on weatherbeach) is hard and can cause significantly different results, which hampers the reliability of the results. 
2. According to my experiments, different models have different rates of convergence, which is not considered and analysized in the paper and further hampers the reliability of the results. 
3. In table 1, the models saturated easily, which is not consistent with existing weather models that have more than 1B parameters. I would suggest the authors to explore model techniques to save the memory.
4. Some important works in the field are not considered and discussed, such as FengWu, FengWu-GHR, and Stormer.

### Questions
please refer to the weakness

### Soundness
2

### Presentation
3

### Contribution
2
