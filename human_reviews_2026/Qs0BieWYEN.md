# ARROW: An Adaptive Rollout and Routing Method for Global Weather Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Weather forecasting is a fundamental task in spatiotemporal data analysis, with broad applications across a wide range of domains. Existing data-driven forecasting methods typically model atmospheric dynamics over a fixed short time interval, e.g., 6 hours, and rely on naive autoregression-based rollout for long-term forecasting, e.g., 5 days. However, this paradigm suffers from two key limitations: (1) it often inadequately models the spatial and multi-scale temporal dependencies inherent in global weather systems, and (2) the rollout strategy struggles to balance error accumulation with the capture of fine-grained atmospheric variations. In this study, we propose ARROW, an Adaptive-Rollout Multi-scale temporal Routing method for Global Weather Forecasting. To contend with the first limitation, we construct a multi-interval forecasting model that forecasts weather across different time intervals. Within the model, the Shared-Private Mixture-of-Experts captures both shared patterns and specific characteristics of atmospheric dynamics across different time scales, while Ring Positional Encoding  accurately encodes the circular latitude structure of the Earth when representing spatial information. For the second limitation, we develop an adaptive rollout scheduler based on reinforcement learning, which selects the most suitable time interval to forecast according to the current weather state. Experimental results demonstrate that ARROW achieves state-of-the-art performance in global weather forecasting, establishing a promising paradigm in this field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ARROW, an Adaptive Rollout and Routing framework for global weather forecasting (GWF).
The authors address two long-standing issues in data-driven GWF:Insufficient modeling of multi-scale spatiotemporal dependencies, as most methods independently train single-interval forecasting models (SIFMs).Rigid autoregressive rollout schemes, which use fixed or greedy intervals regardless of atmospheric dynamics.
To overcome these, ARROW integrates two key innovations:Multi-Interval Forecasting Model (MIFM) with a Shared–Private Mixture-of-Experts (S&P MoE) architecture and Ring Positional Encoding (RPE) for Earth’s spherical geometry.Adaptive Rollout Scheduler (AR Scheduler) trained via Deep Q-learning to dynamically select forecasting intervals conditioned on the current weather state.

### Strengths
* The paper is well-written, with a clear problem statement and a well-organized structure.

* The authors propose a Ring Positional Encoding (RPE) that is more suitable for global prediction tasks. Theoretically, in the 1D case, the encoding ensures that points closer on the circular domain remain close in the RPE space. The visualization further demonstrates that along the latitude, points near the boundaries are represented with shorter distances, capturing the circular nature of global coordinates.

* The paper introduces a Shared-Private Mixture of Experts architecture, which enables a single model to produce predictions at multiple temporal resolutions within one iterative step.

* The authors further propose an AR Scheduler Fine-tuning strategy that adaptively selects the number of rollout steps during inference. It employs a Deep Q-Network (DQN) to estimate the state–action value function, and an Adaptive Rollout Fine-tuning Algorithm is designed to jointly optimize the DQN and the environment.

### Weaknesses
* The experiments are conducted under a limited data setting — using a resolution of 128×256 and only six variables. It is recommended to follow Pangu-Weather by maintaining a 25 km resolution and using 13-level variables, which would make the results in Table 1 more convincing.

* Pangu-Weather performs predictions via multi-step rollouts of several models, but the paper does not clarify whether such multi-model rollouts were considered or compared in the experiments.

* There is insufficient experimental evidence to illustrate how the Adaptive Rollout Scheduler (Fine-tuning) actually operates. Its mechanism is only reflected in the RMSE values in Table 3, without showing the distribution of selected rollout lengths or how the scheduler adapts to different cases.

* Similarly, the paper lacks sufficient analysis to demonstrate how the Shared & Private Mixture of Experts (S&P-MoE) functions. Although the ablation setting is described in Section 2, there is no analysis of the routing behavior or how experts are dynamically selected in different situations.

* In the case study, the paper does not compare its predictions against other methods, making it difficult to assess whether the proposed approach provides improved capture of fine-grained atmospheric variations compared to existing models.

### Questions
* In the AR Scheduler Fine-tuning strategy, what exactly does the term “environment” refer to? Could you please provide a more detailed explanation of its definition and role within the fine-tuning framework?

* Regarding fine-tuning strategies, how does the proposed Adaptive Rollout compare with other approaches such as Pangu-Weather’s rollout scheme, Fengwu’s replay-buffer-based fine-tuning, Fuxi’s multi-timescale cascaded fine-tuning, and GraphCast’s multi-step fine-tuning?

* How would the AR Scheduler Fine-tuning strategy perform when combined with the Pangu-Weather model? Given that the Pangu-Weather GitHub repository provides multi-step model checkpoints, would it be possible to test AR Scheduler Fine-tuning in conjunction with these checkpoints? I am particularly interested in seeing how the results would compare.

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
A novel data-driven global weather forecasting model that introduces a Reinforcement Learning-based Adaptive Rollout Scheduler for flexible time steps and a Multi-Interval Forecasting Model (MIFM) with a Shared-Private Mixture-of-Experts, achieving state-of-the-art accuracy.

### Strengths
1.Novel use of Reinforcement Learning (Q-learning) to formulate and solve the autoregressive prediction process as a sequential decision-making problem, directly addressing the inflexibility of fixed-step forecasting.

2.Introduces a Shared-Private Mixture-of-Experts (S&P MoE) for efficient, unified modeling of multi-scale temporal dependencies, and uses Ring Positional Encoding (RPE) for physically-informed spatial modeling of the Earth's spherical geometry.

### Weaknesses
1.Lack of AR Scheduler Insight: Insufficient analysis on the learned policy of the Adaptive Rollout Scheduler. It is unclear how the policy chooses time steps (e.g., $6\text{h}$ vs. $24\text{h}$) in response to specific weather conditions, which limits interpretability.

2.Limited Architectural Novelty: The concept of using multi-interval/lead-time based forecasting (the basis of MIFM) is not entirely new, with prior work (e.g., MetNet) having explored similar ideas.

3.Inference Time Trade-off Unclear: The paper lacks discussion and comparison of the total inference time or computational cost of ARROW against fast baselines (like Pangu-weather). The adaptive nature might lead to more model calls and slower overall prediction.

4.The core physical motivation for Ring Positional Encoding (RPE) needs clearer explanation and justification beyond just latitude circularity.

### Questions
1.Policy Interpretation: Provide an in-depth analysis of the learned AR Scheduler policy. How do its chosen time steps ($\delta$) correlate with quantifiable meteorological indicators (e.g., instability, magnitude of change in Z500)?

2.Optimization Details: Provide more detail on the RL hyperparameters and comment on the stability and convergence of the challenging alternating optimization paradigm used to train the model and the policy jointly.

3.RPE vs. Advanced PE: Given the rise of more advanced positional encoding techniques like Rotary Positional Encoding (RoPE) which are effective in capturing relative relationships, did the authors consider or test a comparison between RPE and methods like RoPE, especially since RoPE could potentially be adapted to model relative positions within the spatial grid? What specific limitations of RoPE (or similar methods) make RPE a superior or more suitable choice for the geophysical context of global weather forecasting?

### Soundness
2

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
This work proposes ARROW, an Adaptive-Rollout Multi-scale temporal Routing method for Global Weather Forecasting. It includes a multi-interval forecasting model that forecasts weather across different time intervals, Ring Positional Encoding that encodes the circular spatial information, and an adaptive rollout scheduler, which selects the most suitable time interval to forecast.

### Strengths
- Strong motivation to address circular spatial representation of Earth and innovative rollout design.
- Clear paper writing and easy to follow.
- Clear code and present the data downloading and model running.

### Weaknesses
- Limited baselines. Is it to compare with GraphCast and GenCast from Google? These two show powerful performance in global weather forecasting.
- Lack uncertainty. Is it possible to study uncertainty in predictions, since weather evolution is an inherently uncertain process? The probabilistic methods may be a possibility.
-  Any further explanations on loss functions in Eq. (7)?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
4
