# Series-to-Series Diffusion Bridge Model

- Decision: Reject
- Scores: 8, 6, 6, 3

## Abstract
Diffusion models have risen to prominence in time series forecasting, showcasing their robust capability to model complex data distributions. However, their effectiveness in deterministic predictions is often constrained by instability arising from their inherent stochasticity. In this paper, we revisit time series diffusion models and present a comprehensive framework that encompasses most existing diffusion-based methods. Building on this theoretical foundation, we propose a novel diffusion-based time series forecasting model, the Series-to-Series Diffusion Bridge Model ($\mathrm{S^2DBM}$), which leverages the Brownian Bridge process to reduce randomness in reverse estimations and improves accuracy by incorporating informative priors and conditions derived from historical time series data. Experimental results demonstrate that $\mathrm{S^2DBM}$ delivers superior performance in point-to-point forecasting and competes effectively with other diffusion-based models in probabilistic forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses point-to-point time-series forecasting through the use of diffusion models. The authors propose a unified non-autoregressive framework that encompasses most existing diffusion-based time-series models. Building on this framework, they introduce S2DBM, which incorporates the Brownian Bridge process. By integrating historical information via informative priors and conditions, S2DBM effectively reduces randomness and enhances forecasting accuracy.

### Strengths
1. The presentation of this paper is clear and easy to follow, with algorithms well articulated.
2. The paper provides a unified framework for several existing diffusion-based time series models, which is well-supported by theoretical foundations. The insights offered are impressive.
3. Both point-to-point and probabilistic forecasting are addressed in the reverse process, broadening the algorithm's application scope.

### Weaknesses
1. Some expressions reference prior work without sufficient explanations, which can hinder comprehension. For instance, in the last equation of Section 3.1, the introduction of \(y_\theta\) relies solely on a citation, making it time-consuming for readers to fully grasp its meaning.
2. Table 4 presents results only for a horizon of 96, omitting long horizon settings. This limits the analysis and may leave important insights unaddressed.

### Questions
1. In Table 2, your models do not achieve the best performance across all datasets. Could you explain why linear models perform best in several cases, while your models do not?
2. Since $c = E(x)$ and $h = F(x)$ are two important components of your algorithms, I am curious about the role of F in enhancing performance. What impact does it have? An ablation study of F may help.

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
This paper presents the Series-to-Series Diffusion Bridge Model ($S^2DBM$), a promising approach in time series forecasting using diffusion models. 
Traditional diffusion models often struggle with deterministic point-to-point predictions. 
$S^2DBM$ addresses this by leveraging the Brownian Bridge process to reduce noise and improve accuracy in reverse estimations, effectively capturing temporal dependencies in time series data. 
The model incorporates historical data as informative priors, stabilizing diffusion and enhancing point-to-point prediction capabilities. 
Experimental results on various datasets show that $S^2DBM$ outperforms existing diffusion-based and other state-of-the-art time series models, demonstrating superior accuracy in both deterministic and probabilistic forecasting tasks.

### Strengths
S1.
$S^2DBM$ integrates the Brownian Bridge process into time series forecasting using diffusion models.
By redefining the diffusion framework, the authors introduce a new model and consolidate various non-autoregressive diffusion techniques into a comprehensive framework, elucidating their interrelationships and underlying principles. 

S2.
The authors conduct thorough theoretical groundwork. 
The empirical evaluations are robust, utilizing diverse real-world datasets to benchmark the model's performance against SOTA methods.

S3.
The draft is well-structured and organized. 
The introduction succinctly outlines the problem context and motivates the need for the proposed model.

### Weaknesses
W1. 
Some assumptions and derivations in this work could be better justified. 
For example, the choice of using a Brownian Bridge process warrants a more detailed discussion on why it is preferred over other stochastic processes. 
Providing empirical evidence or theoretical reasoning for the choice of this process could strengthen the argument for its effectiveness in reducing randomness in forecasting.

W2.
While the paper presents strong performance metrics, it lacks robustness testing under adverse conditions, such as noisy inputs or missing data. 
Future experiments should include scenarios with varying levels of data quality to demonstrate how $S^2DBM$ handles real-world challenges.

W3.
Currently, the experiments focus primarily on a single configuration. 
Investigating how different settings of hyperparameters, such as the choice of the prior predictor or conditioning modules, impact performance could provide valuable insights. 
For instance, evaluating the effects of varying the number of diffusion steps or using alternative conditioning mechanisms could highlight the robustness of $S^2DBM$ across diverse scenarios.

### Questions
Q1.
Can you elaborate on the computational efficiency of $S^2DBM$, particularly in relation to larger datasets or real-time forecasting applications?

Q2.
What specific advantages does the Brownian Bridge process offer over other stochastic processes for time series forecasting in the context of your model?

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
2

### Summary
This paper revisits the application of diffusion models for time series forecasting, presenting a unified framework that consolidates these methods. Building on this framework, the authors incorporate the Brownian bridge process to enhance prediction accuracy. The results demonstrate that the proposed approach outperforms existing diffusion-based forecasting models.

### Strengths
The authors provide a comprehensive summary of existing models, highlighting that their primary differences lie in the formulation of $\hat{\gamma_t}$. By introducing the Brownian bridge process into diffusion-based time series forecasting models, they establish a range of relevant properties.

### Weaknesses
1. In Line 259, the title of Proposition 1 is "Brownian Bridge between Historical and Predicted Time Series." However, the Brownian bridge’s endpoint is set to $h$, the Prior Predictor’s forecasted value of $x$. The paper provides no explanation as to why $h$ is chosen as the endpoint for the Brownian bridge. 
2. It would be beneficial for the authors to clarify, from a theoretical perspective, why the Brownian bridge is integrated into the diffusion model for time series forecasting. Specifically, how does its ability to "pin down" the diffusion process at both ends help reduce instability from noisy inputs and enable the accurate generation of future features based on historical time series?
3. In line 195, Theorem 1 concerns non-autoregressive diffusion processes. However, the summarized models, CSDI and SSSD, do not appear to be non-autoregressive models. Could there be an issue with the theorem here?

### Questions
See weakness plz.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a comprehensive framework that encompasses most existing diffusion-based methods. Building on this foundation, the authors introduce the Series-to-Series Diffusion Bridge Model (S2DBM). Experimental results demonstrate that S2DBM delivers superior performance.

### Strengths
This paper utilizes the Diffusion Bridge Model to help the reverse process start from a more deterministic state, reducing the instability caused by noise and thereby facilitating better predictions.

### Weaknesses
weakness:
1. There is a notation issue with \(\hat{\gamma}_t\) in line 203; the writing needs to be standardized. Additionally, it needs to be clarified whether the values of \(\hat{\alpha}_t\), \(\hat{\beta}_t\), and \(\gamma_t\) should have a specific relationship to conform to the diffusion model.
2. In line 411, it is mentioned that a comparison with timediff was made, but Table 2 does not include TimeDiff data while other baselines are present.
3. The content of the paper appears to primarily build on existing work by combining the model guidance from TMDM and the non-autoregressive approach of timediff with the existing Brownian Bridge process. This integration seems to lack novelty.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
