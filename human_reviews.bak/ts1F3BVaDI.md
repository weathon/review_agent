# Marginalization Consistent Mixture of Separable Flows for Probabilistic Irregular Time Series Forecasting

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Probabilistic forecasting models for joint distributions of targets in irregular time
series are a heavily under-researched area in machine learning with, to the best of
our knowledge, only three models researched so far: GPR, the Gaussian Process
Regression model (Dürichen et al., 2015), TACTiS, the Transformer-Attentional
Copulas for Time Series Drouin et al. (2022); Ashok et al. (2024) and ProFITi
(Yalavarthi et al., 2024b), a multivariate normalizing flow model based on invertible
attention layers. While ProFITi, thanks to using multivariate normalizing flows,
is the more expressive model with a better predictive performance, we will show
that it suffers from marginalization inconsistency: it does not guarantee that the
marginal distributions of a subset of variables in its predictive distributions coincide
with the directly predicted distributions of these variables. Also, TACTiS does not
provide any guarantees for marginalization consistency.
We develop a novel probabilistic irregular time series forecasting model, Marginal-
ization Consistent Mixtures of Separable Flows (moses), that mixes several nor-
malizing flows with (i) Gaussian Processes with full covariance matrix as source
distributions and (ii) a separable invertible transformation, aiming to combine
the expressivity of normalizing flows with the marginalization consistency of
Gaussians. In experiments on four different datasets we show that moses outper-
form other state-of-the-art marginalization consistent models, perform on par with
ProFITi, but different from ProFITi, guarantees marginalization consistency.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a new model, called moses (marginalization consistent mixture of separable flows) for probabilistic forecasting of irregular time series. In particular, the proposed model mixes several normalizing flows with Gaussian Processes with full covariance matrix as source distributions and a separable invertible transformation. Some empirical results are provided to demonstrate the effectiveness of moses in achieving marginalization consistency and lower Marginal Negative Log-likelihood when compared to existing methods.

### Strengths
- Overall the paper is well written and the presentation is easy to follow
- The design of the proposed marginalization consistent model that combines simple, separable transforms with a richer source distribution (a Gaussian Process with full covariance matrix) seems innovative

### Weaknesses
- Empirical results are limited to simple tasks and it is not clear how effective the proposed method would work on longer temporal sequences with higher spatial resolution 
- The importance of marginalization consistency is not adequately demonstrated across different task settings. In other words, the paper does not address and perform detailed empirical studies to investigate how effective is maintaining marginalization consistency in improving the considered metrics for different lengths of training sequences, different lengths of forecast horizons and different dimensions (spatial resolutions) of training sequences
- Missing details on the training of the model: choices of optimization algorithm, batch size, learning rate, choices of architectural hyperparameters, etc. for each of the considered task

### Questions
- To empirically demonstrate marginalization consistency, instead of using the proposed two simple bivariate distributions, can we use simulated data from nonlinear (possibly chaotic) dynamical systems (with possibly complex stationary distributions)? I believe testing the proposed method on various such datasets could help to strengthen the paper  
- How effectively would the proposed method perform for longer-horizon forecasts? Will the maintenance of marginalization consistency help improving longer term forecasting?

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
3

### Summary
The work blends in the ideas of Gaussian Process Regression with Normalising Flows to develop a probabilistic forecasting model for joint distributions of targets in irregular time series. The proposed method shows the property of marginal consistency. The model can be queried in two ways; 1. by asking for joint predictive distribution of all variables and then marginalising out the variables not to be queried; 2. by directly querying the variable.

### Strengths
1. The paper focuses on a rather under-researched area providing a way for marginalisation consistency.
2. The standard deviation in the experimental results provides a good idea of the method's performance.
3. The idea of using mixtures of conditional normalising flows to counter expressivity issues is clever.

### Weaknesses
1. Could the author explain why we might want marginalisation consistency for models when we can directly query the desired variables and get the desired distribution?
2. Can authors comment on the performance of Moses if instead of mixtures of univariate normalising flows, a multivariate NF is used? This will surely violate the marginalisation consistency, but do we get better njNLL?
3. Is it possible to create prediction intervals on desired queries and attain the desired coverage due to marginalisation consistency?
4. Table 1 only compares njNLL, the metrics such as Energy Score were criticised, but it would be helpful to see a comparison of other such metrics.
5. The proposed methods seem to be lagging significantly wrt CRPS and MSE
6. A trivial question: I don't see a motivation for section C.3 in the appendix as one can expect Moses to not work that well for unconditional density estimation.

Edit: Thanks for the response, I have adjusted my score accordingly.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper considers the problem of forecasting joint distributions of targets in irregular time series, which is a relatively seldom research topic in deep learning community. In particular, the authors show that one of the competitor methods, ProFITi, suffers from marginalization inconsistency. To tackle this issue, the paper proposes a novel method, called moses, that mixes several normalizing flows with Gaussian Processes as source distributions and a separable invertible transformation, aiming to combine the expressivity of normalizing flows with the marginalization consistency of Gaussians. The method was checked in a few experimental settings.

### Strengths
The paper has a few strengths overall, which I will outline below. 


**Strengths:**
1. Considering the issue of marginalization consistency from the theoretical perspective.
2. Finding the obvious drawbacks (e.g., lack of marginalization consistency) of the baselines.
3. Proposing a simple, but effective, novel method for the considered problem. The model itself is simple and is combining Gaussian processes with a mixture of normalizing flows.

### Weaknesses
Despite its strengths, the paper has a few weaknesses.

**Weaknesses:**

1. However, I like the theoretical findings of the proposed method (marginalization consistency), I’m really concerned about the experimental setting and obtained results. In particular, the experimental settings seems to be relatively simple. Moreover, the results are usually worse than ProFITi.
2. Moreover, I don’t know why the authors used the specific normalizing flows (discrete-time) where we need to choose the specific invertible transformations. Why not to use the continuous-time normalizing flows (e.g., FFJORD)? I found the lack of the comparison between the used normalizing flow models a weakness of this paper. 
3. Moreover, we don’t know what is the computational (and memory) cost of using the normalizing flows.
4. The comparison in Fig. 1 seems to be unfair to me, since the number of GMM’s parameters is much smaller than the number of moses’s parameters.
5. Regarding the measure of marginalization inconsistency, the authors use Wasserstein distance. However, the number of samples used for obtaining the results is small.
6. Severe computational cost with increasing the number of variables and using of computational heavy normalizing flows are a key limitations of this method. Because of that, I would like to see any higher-dimensional problem to see if this model is feasible to such problems.

### Questions
I would like to see especially the experiments and responses to the issues mentioned as weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
