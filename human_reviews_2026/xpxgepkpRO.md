# Inference-time Scaling for Time-series Processing

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Scaling laws have fundamentally driven AI progress, particularly in large-scale models. However, as Web-scale pretraining data for such models nears saturation, focus increasingly shifts to new paradigms like inference-time scaling. While validated across various AI domains, its application to time-series tasks remains largely unexplored. This study addresses this gap by investigating whether inference-time scaling can be successfully adapted for time-series processing. First, multiple candidate outputs for a given input are generated based on a trained model. Second, motivated by the principle that better candidates reconstruct the observed data more accurately, we compute the reconstruction error for each candidate output. Third, these errors are used to determine weights of each candidate, and the final prediction is then formed as a weighted combination of the candidates. We present specific algorithmic instantiations of this new framework for two fundamental time-series tasks, namely forecasting and missing value imputation. Furthermore, we provide a theoretical analysis for the forecasting case to support the method's validity from a Bayesian uncertainty perspective. Extensive experimental evaluation across 7 benchmark datasets for both tasks convincingly verifies the effectiveness of our methodology: Incorporation of our methodology during the inference phase led to performance improvements in all 9 recent time-series methods. Source codes have been uploaded in the supplementary files.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an ensembling method of time series forecasting/imputation models. Basically, the idea is, given a piece of time series to-be-forecasted/imputated and a forecasting model, the authors use MC drop-out to generate multiple forecasting/imputation candidates. After generating, a backward model backcasts the original time series (in forecasting cases); or some original pieces are masked and a backward model backcasts the original time series (in imputation cases). Authors carry out theory analysis based on Bayesian uncertainty theory. Authors also provide experiment results for forecasting and imputation, showing that the proposed method performs better than WTA or average. Higher mse reduction is shown for larger number of candidates.

### Strengths
1. The paper studies the interesting idea to use multiple predictions from same model.
2. Authors provide reasonable theory analysis and support for the proposed method.
3. Authors provide comprehensive experiments to validate that using weighted average is better than simple average or best of N.

### Weaknesses
1. Could you please provide more ablation study on the choice of $\sigma_e$? Especially, please specify is $\sigma_e$ globally chosen or is it specially tuned for each model each dataset on each prediction horizon/imputation mask percentage. As shown in Figure 4, Figure 16, 17, etc., WTA in some cases would lead to worse performance. Please specify how you choose $\sigma_e$ in more detail, and perhaps carry out sensitivity study of $\sigma_e$.

### Questions
Please see weakness. Also questions that are not considered as Cons:

1. The reverse model here is somewhat similar to reward model for LLM. The authors train a backward model with same architecutre/design for each foreward model for consistency, which I think is indeed reasonable. Given that said, according to my understanding and common sense, a stronger reward model would lead to better inference time scaling result for LLM, and perhaps a stronger backward model would lead to better inference time scaling result for time series forecasting. I notice in FIgure 18 and Figure 26 that in the PatchTST as foreward \& backward model case, the WTA and weighted-avg-of-8 lead to worse performance in some cases. Would this be caused by, that the PatchTST backward model is not a nice backward model? Could you use some other backward model with PatchTST foreward model, to see if this would improve PatchTST inference time scaling performance?
2. Following 1., what's your considered best backward model? Would the backward model performance related to the forward forecasting performance? (I know to answer this question in great detail would require a lot of experiments. I'm not requiring the authors to conduct this amount of thorough experiments just to answer this question. Perhaps some intuitive answer would be enough for this question. My main concern is still written in Weakness section.)

One further question for discussion:
Recent observations (e.g. https://www.arxiv.org/abs/2510.02729) show that these time series forecasting benchmarks have almost been saturated. Other concerns come from the fact that it doesn't seem to make sense that one simple neural network can reach sota for all time series tasks without any context (e.g. https://neurips.cc/virtual/2024/108471), so perhaps we should combine more things together (e.g. features, NNs, strategies, etc.). What's your opinion on these thoughts? Compared to proposing methods that risk overfitting these datasets, what future direction do you think would be good for further research of our time series community?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a general inference-time scaling (ITS) framework for enhancing time-series processing tasks, including forecasting and imputation. The key idea is to enhance model performance at inference time without retraining by generating multiple candidate outputs via Monte Carlo Dropout, evaluating each candidate through reconstruction errors using a backward model, and aggregating them through error-based weighting. The authors justify the approach with a Bayesian uncertainty interpretation, showing that the weighting scheme corresponds to a probabilistic ensemble that reduces epistemic uncertainty. Experiments are performed across seven benchmark datasets and nine models.

### Strengths
1. The paper tackles an underexplored direction — inference-time scaling for time-series models — and presents a clear, well-motivated framework. Most current work focuses on squeezing more data into pre-training to obtain better foundation models; shifting attention to smarter inference is an important and complementary perspective. I would, however, have appreciated a deeper discussion comparing ITS to other inference-focused approaches (e.g., meta-learning based ([1], [2]) test-time adaptation or context-based methods [3] / in-context learning).
2. The Bayesian interpretation of the weighting scheme gives theoretical grounding to what would otherwise be a heuristic ensemble technique.
3. The authors validate their procedure across multiple tasks (forecasting and imputation) and over a large set of methods, which supports the claimed generality of ITS.

[1] Learning deep time-index models for time series forecasting, Woo et al, ICML 2023

[2] Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations, Le Naour et al, TMLR 2024

[3] From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models, Hoo et al, 2025

### Weaknesses
1. The relative improvements are small. For forecasting, the average relative gain for methods like PatchTST and iTransformer (Table 1 and Appendix F.2) is under 2%, which is modest. Moreover, some experimental choices for forecasting are questionable: using a look-back window of 96 to predict horizons of 96, 192, 336, and 720 is not convincing in my view (longer history windows are typically required for long horizons).
2. For imputation, relative gains are larger, but the experimental design weakens the claim. Most baselines are forecasting models rather than imputation-specific methods; including dedicated imputation baselines (e.g., BRITS, SAITS, CSDI) would be more convincing. Also, the sequences used for imputation are very short (L = 96), which reduces realism and makes it harder to judge real-world efficacy.
3. The computational-cost discussion is underdeveloped and not fully convincing. Section 4.3 and Appendix F.5 only report runtime analyses on two small datasets and for the smallest settings (forecasting input=96 then predict=96, imputation input=96 with 12.5% masking). Because the performance gains are relatively mild, a thorough runtime comparison is essential to justify the extra inference cost. I would expect a comparative table showing end-to-end inference time (in seconds) on a single GPU across multiple datasets and settings.

### Questions
Here are my suggestions :

1. Please substantially expand the inference-time computational analysis (Weakness 3). Provide a comparative inference-time table (seconds per example) on the same GPU across representative datasets and multiple settings (varying L and S, and different mask rates). This will clarify the trade-off between accuracy and cost.
2. Strengthen the discussion comparing ITS with other inference-focused approaches, such as meta-learning-based test-time adaptation and context/in-context learning methods (see Strength 1). A short empirical or conceptual comparison would help place ITS in the broader landscape.
3. Improve experimental quality: use longer look-back windows for long-horizon forecasting, include imputation-specific baselines (BRITS, SAITS, CSDI), and evaluate on longer sequences to better reflect practical imputation scenarios.

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
This paper applies inference-time scaling to time-series models by generating multiple predictions via MC Dropout, scoring them using reconstruction error from a backward model, and combining them with learned weights. Experiments on forecasting and imputation tasks show consistent improvements across 9 methods and 7 datasets, at the cost of 1-15x slower inference.

### Strengths
- The paper is the first to address inference time scaling in time series forecasting.
- Using a reverse-temporal model for reconstruction verification is a good domain-specific adaptation.
- First systematic study of inference-time scaling for mainstream time-series models (prior works seem to have only worked on LLMs for TSF).
- Comprehensive evaluation across architectures and tasks.

### Weaknesses
- The theoretical insight is naive and does not provide much insight. The method is not novel in other domains, so it is more like an engineering work with good empirical results.
- The baselines are simple (see Figure 4). Although there are no prior works but there should be more sophisticated weighting methods they can compare with. This can demonstrate if the backward model actually provides effective guidance or if the effect simply comes from averaging.
- The computation cost is somewhat high (1-15x slower with 2-12% improvement), and they did not mention or count the computation cost for training the backward model.

### Questions
Can you try to justify the effectiveness of the backward model as a guidance more clearly? I believe comparing with more comprehensive baselines (for example entropy-guided or average) would also solve this.
I will raise my score if you solve this well (since the score I would like to give now is somewhat between 4 and 6).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an inference-time scaling method for time series data. The proposed method can be applied to both time series forecasting and time series imputation tasks. The core idea is to create multiple outputs from a pretrained model and then weight the outputs based on how well they reconstruct the input. The proposed method yields improved performance in the experiments, demonstrating the efficacy of the proposed approach.

### Strengths
The paper presents a robust empirical evaluation, utilising multiple datasets and methods. 

The reweighting strategy is an interesting and effective method based on the reconstruction that improved performance across datasets.

### Weaknesses
While the experiment section is strong, it only reports single-time statistics. To understand the full efficacy of methods, it would be better to have the experiments repeated over multiple runs and presented with the standard deviations. 

See "Questions*" below for more

### Questions
1. Line 67-68: These line seems unnecessary given the same information appears in the contributions just below:

2. The multi-candidate generation is done via Monte Carlo Dropout, which can provide very narrow sets of candidates. Furthermore, the probabilistic statements made from it are largely miscalibrated. It would be interesting to see how the results differ when using generative models such as flow-based models with or without calibrated probabilistic statements via Conformal Prediction.

3. Line 160-161: A backward model needs to be trained for reconstruction. However, a strong backward model might be able to reconstruct from any of the given candidates. An interesting example could be how diffusion models work; even if there is complete noise, one might be able to reconstruct the true distribution, even if the score function is known.

4. Line 204: X belong to R (F X C), does it need to only reconstruct S steps in the past? Also, why was it needed to divide the prediction horizon into segments? All of the prediction horizon could be predicted at once, no?

5. The theoretical analysis seems restricted to MC dropout; are any other forms of probabilistic candidate generation supported?

6. Line 341: "the Exchange dataset, we follow the hyperparameter settings of the Weather dataset": Why were the same hyperparameters used as in the Weather dataset? The datasets seem quite different. Isn't it better to perform a hyperparameter search for the Exchange dataset separately? 

7. How critical is the training of the backward model? How well-trained should it be?

8. Line 419: "Exchagne" is written instead of "Exchange".

9. Is there any rule of thumb to find the right number of candidates, K?

10. Only reports single-time statistics. To understand the full efficacy of methods, it would be better to have the experiments repeated over multiple runs and presented with the standard deviations.

### Soundness
3

### Presentation
3

### Contribution
3
