# Flow-based Conformal Prediction for Multi-dimensional Time Series

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 0

## Abstract
Time series prediction underpins a broad range of downstream tasks across many scientific domains. Recent advances and increasing adoption of black-box machine learning models for time series prediction highlight the critical need for uncertainty quantification. While conformal prediction has gained attention as a reliable uncertainty quantification method, conformal prediction for time series faces two key challenges: (1) \textbf{leveraging correlations in observations and non-conformity scores to overcome the exchangeability assumption}, and (2) \textbf{constructing prediction sets for multi-dimensional outcomes}. To address these challenges, we propose a novel conformal prediction method for time series using flow with classifier-free guidance. We provide coverage guarantees by establishing exact non-asymptotic marginal coverage and a finite-sample bound on conditional coverage for the proposed method. Evaluations on real-world time series datasets demonstrate that our method constructs significantly smaller prediction sets than existing conformal prediction methods, maintaining target coverage.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a conformal prediction method for multi-dimensional time series that addresses two main challenges: temporal dependencies and multi-output prediction. The approach combines a Flow model with Transformer-based context encoding to model residual distributions, using Euclidean distance for non-conformity scoring. Trained via Flow Matching, the method provides theoretical coverage guarantees and achieves tighter prediction sets than baselines on real-world datasets.

### Strengths
1. The integration of flow matching with conformal prediction is original and conceptually appealing.
2. The method provides solid coverage guarantees, both marginal and conditional.
3. Experimental results show significant improvements over baselines.
4. The paper is well written and easy to follow, with clear motivation and presentation.

### Weaknesses
1. While flow matching may help model the distribution of prediction residuals, it can be computationally expensive.
2. Results may be sensitive to hyperparameters, and extensive hyperparameter search can further increase computational cost.

### Questions
1. Please compare computational cost against baselines and report sensitivity to hyperparameters.
2. Please empirically validate Assumptions 4.1, 4.3, and 4.7–4.11 in the experiments (or provide diagnostics/proxies).
3. I'm not sure about the key reason that the flow-based model yields substantially smaller prediction sets than baselines. What properties of flows drive the gains, and why alternatives like quantile regression or transformer encoders cannot achieve similar improvements?

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
This paper introduces a method for uncertainty quantification in multidimensional time series forecasting. The approach begins by using any base forecaster to generate point predictions and residuals. It then encodes recent historical data as contextual information to condition a continuous normalizing flow, which maps an isotropic Gaussian source distribution to the situational residual distribution.

### Strengths
(1) The paper features a clear structure and rigorous logic, facilitating readers' comprehensive understanding of the methodology.  

(2) The paper introduces a multidimensional uncertainty quantification approach centered on conditional continuous flow. By mapping source distributions to residual distributions, it establishes a unified pathway for constructing prediction sets, demonstrating novelty and general applicability.  

(3) The paper presents a comprehensive argumentative framework, rigorously defining the assumptions and applicability of marginal coverage and conditional coverage respectively, accompanied by rigorous proofs.   

(4)The experimental design encompasses multiple datasets, diverse output dimensions, and various base predictors. Evaluation metrics are clearly defined, and results effectively validate the proposed methodology

### Weaknesses
(1) The paper lacks a corresponding overview of methods and a concise algorithmic workflow.  

(2) The rationale for selecting the source distribution and scoring metric is not sufficiently discussed. The paper adopts an isotropic Gaussian distribution as its core design without presenting alternative source distributions or concluding on differences in coverage and set size.  

(3) The claim of conditional coverage lacks empirical support. While presented as a theoretical contribution, the experiments only report overall coverage and set size without showing coverage performance grouped by scenario or context.

### Questions
(1) Is the precise marginal coverage established solely on theoretical grounds? In practice, do flow estimation errors and base predictor errors influence coverage bias? If so, can an upper bound or consistency conclusion be provided?  

(2) At a given nominal level, does the constructed forecast set size exhibit near-optimality (minimal volume)? As the sample size approaches infinity, does it converge consistently to the optimal solution? What assumptions or conditions are required?  

(3) Are the assumptions for achieving conditional coverage, such as the strong mixing condition, empirically supported in real data? If not satisfied, can a robust adaptation scheme be proposed that accommodates both the data and the proposed method?  

(4) If a distribution shift occurs during the testing phase, will the context encoder and conditional stream cause coverage degradation? Does the method adaptively handle distribution shifts while still meeting the specified coverage rate?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the construction of calibrated prediction intervals for multivariate time-series 
data.  Building on Guided Flow, the proposed method transforms residuals 
into a Gaussian variable and then derives a confidence region by inverting the  
Gaussian distribution. Some theoretical guarantees for the method are provided in the paper, 
and the proposed method is evaluated on numerical experiments.

### Strengths
1. The paper considers an important yet challenging problem: constructing 
calibrated prediction intervals for multivariate time-series data.
2. The proposed method has provided some fresh perspectives on the problem.

### Weaknesses
1. Although the goal is to construct prediction intervals, the proposed method does not seem to 
be a "conformal prediction" method in the usual sense: it does not leverage any (approximate) exchangeability 
of the conformity scores to determine the prediction region.
Instead, it seems closer to a nonparametric method that directly models the residual distribution and then constructs prediction sets.

2. The theoretical guarantees are unclear. For marginal coverage, Proposition 4.6 states that the prediction set produced by the algorithm satisfies coverage guarantee "if the ball $B_{\alpha}$ defining the prediction set in equation (5) has probability mass $1-\alpha$". This is a strong "if", since $B_\alpha$ is derived under Gaussian distribution, while $\hat e(y_i)$ need not be exactly or even close to Gaussian in the presence of approximation error or model misspecification. As written, it is unclear what unconditional coverage guarantee the method provides and under what assumptions (or calibration procedures) the condition is satisfied. 

3. The theoretical results rely on numerous assumptions that are not fully justified. A more thorough discussion—motivating each assumption, assessing plausibility in typical applications, and exploring sensitivity to violations—would greatly strengthen the work.

### Questions
Please refer to the "Weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The proposed work presents a novel conformal production method for multidimensional time series by combining a transformer and a flow model. The obtained prediction region incorporates the historical context of the time series and is based on the transformation of residuals, creating an Euclidean ball in the latent space of flow residuals.  The efficacy of the proposed method is evaluated across several settings and various datasets.

### Strengths
The experiments are conducted with multiple base predictors, and various baselines have been compared against, with FCP coming out as the best in the experiments.

The experiments are repeated across several runs, providing uncertainty statements that give a better picture of the method's efficacy.

### Weaknesses
To best judge a conformal method, it is imperative to compare its performance across different significance levels, and plotting a calibration curve is particularly helpful. 

Most of the theoretical results are well-established or fundamental. 

See the "Questions" below for more.

### Questions
1. Line 016, "leveraging correlations in features and non-conformity scores" Did the authors want to allude to the exchangeability assumption in the traditional CP setup?

2. The proposal is to use a flow with Classifier-free guidance; however, the idea only requires a kind of flow-based model that can transform a given distribution to another and is bijective. This makes the idea somewhat restrictive, given that a practitioner might want to use the Diffusion Model, classifier-based guidance, or a discrete flow. 

3. The idea to model the residual distribution is already present in the literature. For example, Res-CONTRA [1] transforms the residuals as a means of learning the calibration, while also utilising an Euclidean ball to define conformity scores. 

4. Line 045: If I am not wrong, the cited Barber et al 2023 does not require the exchangeability assumption; I believe this is a mis-citation.

5. Line 077: "Despite these efforts, existing methods remain limited to univariate outcomes or assume access to multiple i.i.d. time series." This statement seems misleading, given that many methods nowadays discuss multivariate outcomes or single time series setups.

6. Line 095: There are more works apart from Xu et. al 2024 that also discuss single time series settings with multiple step-ahead forecasting. CAFHT, JANET [2, 3] is one such example. Note that their work is also applicable to multiple trajectories. More importantly, any idea, such as ACI, can incorporate multivariate responses by simply using a non-conformity score that caters to multivariate responses. 

7. Line 113: with y_i = f(x_i), it looks like there is no dependence on history y_{i-1}. Is that intentional?

8. Line 119: Similarly,  z_i seems only to include x. I mean to say there is a lack of clarity around the notations.

9. The idea of using a ball is already present in CONTRA, such as in Eq. 7. Furthermore, they have a computationally efficient version, as one only needs to work around the boundary of the ball. 

10. One issue with using Flow-matching is solving ODE or computing the divergence for the area computation. From this perspective, it seems more natural to use discrete flows. 

11. Theorem B.4, Lemma 4.5, and Proposition 4.6 present a similar result to one in CONTRA. 

12. Conditional coverage results, while important, depend on strong assumptions such as strongly mixing.

13. Line 373-377: I am confused about the data splits here. Why is it that there is no calibration set needed for FCP? Furthermore, using the training and the validation sets for calibration skews the coverage guarantees, as far as I can understand. 

14. Line 385: The target confidence is set as high as 0.95, which is okay. But it is necessary to show the performance of the proposed method with different significance levels. A calibration curve might be helpful to see if the proposed method works well in all cases.

15. The theoretical statements are there for conditional coverage, but there is no empirical evidence for the same.


[1] CONTRA: Conformal Prediction Region via Normalizing Flow Transformation Zhenhan FANG · Aixin Tan · Jian Huang
[2] Conformalized Adaptive Forecasting of Heterogeneous Trajectories Yanfei Zhou, Lars Lindemann, Matteo Sesia
[3] JANET: Joint Adaptive predictioN-region Estimation for Time-series Eshant English, Eliot Wong-Toi, Matteo Fontana, Stephan Mandt, Padhraic Smyth, Christoph Lippert

### Soundness
2

### Presentation
3

### Contribution
1
