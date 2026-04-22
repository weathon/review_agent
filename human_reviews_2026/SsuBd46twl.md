# Eliciting Numerical Predictive Distributions of LLMs Without Auto-Regression

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Large Language Models (LLMs) have recently been successfully applied to regression tasks---such as time series forecasting and tabular prediction---by leveraging their in-context learning abilities. However, their autoregressive decoding process may be ill-suited to continuous-valued outputs, where obtaining predictive distributions over numerical targets requires repeated sampling, leading to high computational cost and inference time. In this work, we investigate whether distributional properties of LLM predictions can be recovered _without_ explicit autoregressive generation. To this end, we study a set of regression probes trained to predict statistical functionals (e.g., mean, median, quantiles) of the LLM’s numerical output distribution directly from its internal representations. Our results suggest that LLM embeddings carry informative signals about summary statistics of their predictive distributions, including the numerical uncertainty. This investigation opens up new questions about how LLMs internally encode uncertainty in numerical tasks, and about the feasibility of lightweight alternatives to sampling-based approaches for uncertainty-aware numerical predictions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In an empirical paper, authors target numerical prediction in LLMs by decomposing prediction into magnitude classification and scaled value regression. The key contribution is learning the magnitude and a normalized value separately, and using the predictions before LLM token generation. 

While it is hinted that the idea enables uncertainty-aware numerical 
prediction without repeated sampling, evaluation is limited to maximum Pinball loss and IQR prediction in generated data.

While the methods empirically work well, we gain no insight regarding the LLM representation behind.

The paper is well written and source code is provided anonymously.

While the paper presents convincing empirical results for the applicability of the decomposed numerical prediction learning in LLMs, I miss theoretical or architectural explanations. Also, the uncertainty considerations are shomewhat limited.

### Strengths
+ New contribution towards improved numeric prediction capabilities of LLMs
+ Very well written paper
+ Source code is available

### Weaknesses
- No theoretical and/or architectural explanation of how the LLMs represent the numeric range and scaled value
- The uncertainty part could be strengthened
- Comparison with methods that learn the distribution of regression problems (e.g. Bayesian NN, Mixture Density Networks), while not absolutely necessary, could make the contribution more valuable

### Questions
The discussion of Table 1 could be more elaborate. Do I understand well that the LLM raw output is much worse than the mean or median of several of its outputs? I see no explanation, not even in the Appendix.

3.1 pinball loss: why the maximum over quantiles? Why not the integral, as in CRPS [Alexander Jordan, Fabian Kruger, and Sebastian Lerch. Evaluating probabilistic forecasts with scoringrules. Journal of Statistical Software, 90(12):1–37, 2019, Diane Bouchacourt, Pawan K Mudigonda, and Sebastian Nowozin. Disco nets: Dissimilarity coefficients networks, Neurips'16]?

In the training procedure of eqs (6-7), how can you handle if a multimodal distribution has modes of different order of magnitude? Isn't eq (6) too restrictive?

While the main goal is certainly not to provide the best model for distribution learnting, could you compare other methods that learn the distribution, e.g. [Shengyang Sun, Changyou Chen, and Lawrence Carin. Learning Structured Weight Uncertainty in Bayesian Neural Networks. AISTATS'17]. 

Fig. 6 only shows MAE, it would be interesting to see e.g. Pinball Loss, compared to non-LLM models for learning distributions.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors explore the ability to extract the predictions of an LLM dependent on its hidden states of its final layers using a variety of methods. For their main experiment, they adopt a mantissa+exponent floating point style method of calculating the regression model, with the mantissa trained as a regression problem and the exponent trained as a classification problem. Only the exponent is trained first, then the mantissa is used to fine-tune the results. They then compare this against the point estimates of the greedy, median, and greedy methods of the LLM. They additionally train a quantile-regression probe using pinball loss to quantify uncertainty.

### Strengths
Strengths
This is an interesting dive into whether both 1. point estimates and 2. uncertainty can be recovered from the LLM's hidden state.

They show advantages against standard LLM sample-based prediction in Figure 4, along with improved results over GP in the tables for the time-series regression task.

They also demonstrate some generalization properties, which can be instrumental when dealing with distribution shift.

### Weaknesses
Though the author's main goal seems to be moving towards sidestepping autoregressive generation, they only explore one-step prediction in the current results. Additionally, for some of the main other results, they calculate statistics and last step predictions, it may be worth it to show average MSE across time rather than MSE against the average in time.

Currently, the experiments seem to be all done on synthetic datasets. While this is useful for the uncertainty metrics, having some error-based analysis on some standard real-world time series datasets could strengthen the paper.

### Questions
If LLMs generate results autoregressively and the model does not work as well for the greedy approach, does this mean that it would still be hard to capture the results of the LLMs purely from the hidden states?

How much contribution does the floating-point formulation actually bring for the probe, compared to just predicting the value directly? If it is significant, then how would this method compare to using it against a standard regressor (without LLM) with the same floating-point formulation?

Why is the LLM sample error so high in the beginning for Figure 4?

### Soundness
3

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
3

### Summary
The paper introduces a set of regression probes to predict the distributional properties of LLM-based time-series forecasters. By training the probes on synthetic data, the probes can predict quantities such as the greedy prediction, mean, and median with good accuracy without autoregressive decoding in both ID time-series and, to a less extent, OOD time-series.

### Strengths
- I appreciate the careful design going into designing the parameterization and loss functions of the probes, which is important for handling values with large ranges.
- The empirical result shows good agreement between the probe predictions and actual values obtained from decoding.
- The computational efficiency of the probe over decoding / sampling is appealing.
- Overall good presentation.

### Weaknesses
- A main finding of the paper is that LLM’s predictive distribution is encoded in its internal activations. This statement seems trivially true. The predictive distribution (jointly over all future tokens) is fully determined given the hidden states of the LLM (KV cache) as a direct consequence of the model architecture.
- The probes only make predictions about a single next value, rather than a future sequence. In practice, time-series forecasters are used to make extended, variable-length predictions over a non-trivial horizon, where this approach does not trivially generalize over to. In other words, the probe only shortcuts autoregression over the digits, but not time steps.
- I'm not convinced that the probes provide significant computational speedup over directly decoding from the LLM. In both approaches, most of the computation is in running the LLM forward pass on the input sequence. Decoding from the LLM is much cheaper, especially for predicting only the very next value. Thus, the savings from the probes can be negligible relative to the overall cost.

### Questions
- Can the proposed approach be generalized to predict distributional properties of a future sequence, rather than a single value?
- How much total runtime or compute saving do the probes provide over decoding from the LLM across the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the question of whether LLM's hidden representations encode distributional information about their numerical predictions rather than relying on autoregressive decoding when it comes to utilizing LLMs for time-series forecasting regression tasks. They Introduce a probing model that separately predict the order of magnitude ( treated as classification ) and the multiplicative residual to predict the mean, median and the greedy predictions from the LLM internal representations. They further extend their results to predicting multiple quantiles of the underlying distribution utilizing quantile regressors. They lastly validate generalization capabilities of the underlying predictors.

### Strengths
1. the paper is very well written. All experiments all well-motivated, clearly explained, and results are well presented. 

2. The idea of eliciting numerical predictions and uncertainties from hidden states is timely, and can possible be extended to tasks beyond time series forecasting. 

3. The experiments are sound, and different aspects such as different kind of generalizations are well considered, thus are comprehensive. I found the calibration analysis interesting extension of the experiments and that it strengthens the credibility of the results. 

Lastly, and to summarize, the paper is well written, and the findings are interesting, informative, and of interest to a broader community. The evidence that distributional information about LLM predictions are recoverable from hidden states is enlightening followed by sound experimental evidence.

### Weaknesses
Most of the weaknesses are well acknowledged in the paper. These include (i) accessing hidden states may decrease the practicality of the method, (ii) relying on extensive autoregressive sampling for training the probes. 

1. While the paper positions itself only within the context of regression and time-series forecasting, it does not sufficiently discuss prior work that also show hidden states encode future outputs and uncertainty in other modalities and tasks. There is a growing literature that shows similar behaviors that probe internal states in the context of analyzing sentiment, factuality, intent, jailbreaking, chain-of-thought reasoning and etc. The related works would benefit from addressing these works as well for better positioning the contribution within a larger context.

2. The experiments in section 2.2 focus on [-1,1] range but its not very clear how the method behaves for larger value ranges. although the authors mention testing other ranges, the results don't show how performance and calibration changes with scale, and it would be interesting to see how performance metric across different ranges change; whether larger numeric ranges introduce instability, or if normalization essentially removes that issue.

### Questions
1. How sensitive are the probe's predictions to the choice of layers used ? It would be interesting to analyze which layers contribute the most to recovering the numerical predictions, and whether the same choice of "best" layers is consistent across tasks and different settings studied in the paper under "generalization". 

2. One form of generalization that is not studied in the experiments is whether a probe trained on llama2 for example generalizes ( or not ) to different models from the same family ( for example llama3 ). Results discussing this would be interesting.

### Soundness
4

### Presentation
4

### Contribution
3
