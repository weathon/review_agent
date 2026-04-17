# TimeSeed: Effective Time Series Forecasting with Sparse  Endogenous Variables

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Time series forecasting is widely applied across various domains. In real-world applications, there are many scenarios where endogenous variables are missing. Recent studies show that incorporating exogenous variables can significantly enhance the predictive accuracy of endogenous variables. However, the lack of a complete historical context introduces significant uncertainty in temporal dependence capture, particularly in systems characterized by non-stationary behavior. To address these challenges, we propose TimeSeed, specifically designed for scenarios with sparsely observed endogenous variables. Technically, TimeSeed reconstructs l sufficient endogenous series from both complete exogenous series and sparsely observed endogenous series, utilizing two types of data to extract stable information. Building on this foundation, we effectively transforming the challenging original prediction task into a sequence-based prediction task. Moreover, TimeSeed is built entirely upon linear layers, which significantly reduces computational costs. Experiments conduct on seven real-world datasets demonstrate that TimeSeed consistently outperforms state-of-the-art models in forecasting accuracy, achieving an average reduction of 13.01\% in MSE and 7.54\% in MAE, with a model size of only 0.19M parameters. Code is available at this repository: \url{https://anonymous.4open.science/r/Alistair-7}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduce TimeSeed which is specifically designed for time series forecasting scenarios with sparsely observed endogenous variables. Technically, TimeSeed leverages dense exogenous and sparse endogenous sequences within a two-stage paradigm of context reconstruction and hierarchical prediction. Experiments demonstrate that TimeSeed consistently outperforms state-of-the-art models in forecasting accuracy.

### Strengths
1. The paper is well-organized and easy to follow.
2. The proposed TimeSeed is a lightweight model built entirely upon linear layers, which significantly reduces computational costs. 
3. The experiment is extensive and quite detailed.

### Weaknesses
1. The paper focuses on the sparse forecasting scenario, which is a conceptually complex and challenging scenario. However, practical evidence on this setting is limited, as the experiments primarily rely on existing multivariate benchmarks. 
2. The presentation of TimeSeed lacks clarity. For instance, line 162 of the main text states, "we design the TDA and FDA blocks to predict trend and periodic features for sparse endogenous sequences, respectively." In contrast, line 171 states, "the TDA is designed to learn the periodic features of endogenous variables by leveraging exogenous variables."
3. Furthermore, the paper does not provide adequate explanations or experimental validation regarding the significance and effectiveness of decomposing and modeling the time domain and frequency domain separately.
4. The term "physical similarity" in line 53 is unclear in the context of time series forecasting. A more precise explanation is required.

### Questions
1. Why did the authors choose to introduce a two-stage decomposition and forecasting paradigm based on reconstruction instead of directly predicting the future of endogenous variables? The paper lacks detailed explanations and experimental evidence to justify this choice.
2. In Figure 3 (right), the authors used the Pearson coefficient to estimate the correlation between the reconstructed sequences and the ground truth. Why did the authors choose the Pearson coefficient instead of Dynamic Time Warping (DTW), which is generally more suitable for time series data?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TimeSeed, a model for time series forecasting under the specific and challenging setting where the target (endogenous) variable is sparsely observed, while correlated (exogenous) variables are fully observed. The proposed method operates in a "reconstruct-then-predict" fashion. It first generates a complete historical representation of the target series by aggregating information from three modules: a Time Domain Aggregator (TDA) for periodic patterns, a Frequency Domain Aggregator (FDA) for trend information (both derived from exogenous series), and an Adaptive Scale Reconstructor (ASR) which incorporates the sparse endogenous signal itself. This reconstructed series is then fed into a simple linear forecaster. The authors show that this efficient, linear-based model achieves state-of-the-art results on seven benchmark datasets under this sparse setting, outperforming numerous recent and complex models.

### Strengths
1) The paper formalizes and tackles the f(X, S) -> Y problem, a relevant and common scenario in real-world applications where a target signal is costly or difficult to measure continuously.

2) Strong Empirical Results: Within its defined experimental setup, TimeSeed demonstrates a significant and consistent performance advantage over a wide range of strong baselines.

3) The model is extremely lightweight and computationally efficient, making it a practical and attractive solution for deployment.

4) The authors' commitment to re-running all baselines under a fair and unified framework is a major strength.

### Weaknesses
1) The main weakness is the limited conceptual novelty. The model is largely a thoughtful recombination of existing ideas (patching, FFT) from recent time series literature, applied to a new problem variant. The contribution feels more like system-building than the introduction of a new fundamental modeling principle.

2) The central "reconstruction" narrative is not backed by an explicit reconstruction loss, which makes the model's inner workings less interpretable and the claims less grounded than they appear. The model is not learning to reconstruct the past per se, but rather learning a useful latent representation for forecasting.

3) The evaluation is almost entirely based on a uniform sparsity pattern, which is not representative of many real-world missing data scenarios (e.g., random or block-wise missingness). The model's robustness to more realistic data imperfection is not sufficiently demonstrated.

4) The paper does not compare against a straightforward two-stage pipeline of "imputation model + forecasting model." This baseline is essential for determining if the proposed integrated architecture provides a tangible benefit over a simple, modular approach.

### Questions
Could you clarify the design choice of not using a direct reconstruction loss on the historical window? Given the paper's central narrative is "reconstruction," it seems this would not only align the training objective with the story but also provide a more grounded way to supervise and evaluate the TDA, FDA, and ASR modules.

The ASR module uses Gumbel-Softmax for a hard selection of a single resolution. What is the justification for this restrictive choice over a soft, weighted combination of features from all reconstructed scales (O_q), which would be a more general approach and potentially capture information across multiple resolutions simultaneously?

A critical baseline seems to be missing: a two-stage approach where a state-of-the-art imputation model (e.g., SAITS, or even a masked version of a model like PatchTST) is first used to fill the sparse endogenous series, followed by a separate state-of-the-art forecasting model. How do you expect TimeSeed to perform against such a pipeline? This comparison is crucial to validate the benefits of your integrated "reconstruct-then-predict" architecture.

How does the model's performance degrade as the sparsity pattern becomes more challenging, for example, with large, contiguous blocks of missing data instead of uniform samples? The current reliance on TDA/FDA might be robust, but this needs to be empirically verified to claim general applicability.

### Soundness
3

### Presentation
3

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
The paper presents TimeSeed, a simple yet effective framework for forecasting when the target variable is only sparsely observed. It rebuilds a dense context of the target using exogenous variables and the few available target points, then forecasts normally. The model combines three parts: 1) Time-Domain Aggregator captures periodic patterns; 2) Frequency-Domain Aggregator extracts trend information using FFTs; 3) Adaptive Scale Reconstructor upsamples sparse data at multiple resolutions. TimeSeed is fully linear, very lightweight, and achieves improvements over baselines across seven datasets.

### Strengths
1. TimeSeed presents a effective approach to forecasting with sparse target data. It reframes the problem as a reconstruction-forecasting task, using simple linear modules that separate periodic and trend information through time- and frequency-domain modeling. 

2. The paper is clearly written and supported by various experiments, showing consistent gains on ablations that validate each component.

3. The model’s clarity, lightweight design, and efficiency make it practical for real-world deployment, while its empirical results show contribution for scenarios with limited target observations.

### Weaknesses
1. The authors did not use real datasets with sparsely observed endogenous variables and fully available exogenous variables. Instead, they created such conditions artificially by applying predefined sparsity ratios to general datasets that originally contain complete endogenous and exogenous information. This design choice somewhat weakens the paper’s practical significance. To better demonstrate the real-world relevance of this problem, it would be valuable for the authors to identify or include datasets that naturally reflect this sparse-endogenous scenario.

2. How exactly is the sparsity ratio implemented? Providing a detailed description of how the sparse endogenous variables are constructed from the full datasets would help alleviate concerns about selection bias and fairness in comparison. When the sparsity ratio is high, different random seeds could substantially alter the time series behavior (e.g. especially for data with sparse-peak patterns). The impact of such randomness and its implications for statistical significance should be discussed more thoroughly.

3. Additionally, were the baselines trained on the imputed, non-observed timesteps? Training models on data points known to be inaccurate could cause them to learn false inter-series relationships, which would unfairly hurt the performance of the models that focus on learning precise inter-series dependencies.

### Questions
1. In Tables 1 and 2, it seems unusual that DLinear outperforms almost all other baselines, given that it does not take exogenous variables as input and has very few parameters. This observation reinforces the concern raised in Weakness 3 about how the baselines are adapted or trained. Moreover, the setup of the experiment in Table 2 needs clearer explanation: in Section 4.1, the paper states that “endogenous variables are missing.” In that case, what exactly does a DLinear-type model use as input?

2. Is the Adaptive Scale Reconstructor reconstructing the fully dense time series, or only the sparsely observed parts? How is the reconstruction quality of the historical series evaluated. Does the model apply a reconstruction loss on the full series?

3. For input lengths longer than 96, how does extending the context window affect performance? It would be helpful if the authors could include additional experiments or discussion on this aspect.

If the authors' response adequately addresses my questions and concerns mentioned above, I am willing to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2
