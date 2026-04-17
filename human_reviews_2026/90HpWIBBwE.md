# Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
In many forecasting settings, the target series comes with exogenous covariates: promotions and prices for retail demand, temperature for energy load, calendar/holiday flags for traffic or sales, and grid load or fuel costs for electricity prices. Ignoring such exogenous covariates can seriously degrade forecasting accuracy, especially when they signal phase changes or spikes in the target series. Most current time-series foundation models (e.g., \texttt{Chronos}, \texttt{Sundial}, \texttt{TimesFM}, \texttt{TimeMoE}, \texttt{TimeLLM}, and \texttt{LagLlama}) ignore exogenous covariates and make forecasts solely from the time-series history, limiting their performance. In this paper we focus on bridging this gap by developing \texttt{ApolloPFN}, a prior-data fitted network (PFN) that is time-aware (unlike prior PFNs) and that natively incorporates exogenous covariates (unlike prior univariate forecasters). Our design introduces two major advances: (i) a synthetic data generation procedure tailored to resolve the failure modes that arise when tabular (non-temporal) PFNs are applied to time-series, and (ii) time-aware architectural modifications that embed the inductive biases needed to fully exploit the time-series context. We demonstrate that \texttt{ApolloPFN} achieves state-of-the-art results across benchmarks containing \emph{exogenous} information such as M5 and electric price forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ApolloPFN, a Prior-Fitted Network (PFN) designed for zero-shot time series forecasting that incorporates exogenous covariates.

### Strengths
1. The focus on zero-shot forecasting with exogenous variables addresses real industrial needs where fine-tuning is undesirable.
2. The evaluation covers multiple realistic benchmarks, and shows consistent improvements especially on tasks with exogenous information.

### Weaknesses
1. While the combination is effective, individual components are relatively incremental. RoPE is a standard positional encoding, and the SNGN algorithm is essentially a reversal of existing graph generation methods. The main contribution seems to be recognizing that tabular PFNs need temporal adaptations, which is somewhat obvious in hindsight.
2. How are the frequencies (φ₁, φ₂) and amplitudes (α₁, α₂) sampled for time-dependent root nodes? This is critical for understanding what temporal patterns the model learns.
3. The paper doesn't discuss how synthetic SCMs relate to real time series structures. Why should random DAGs with MLPs/decision trees as node functions capture real forecasting scenarios?
4. What's the distribution over graph sizes, number of features, etc.. These details matter for reproducibility.
5. The paper identifies that TabPFN-TS "fails to understand temporal autocorrelations" but doesn't explain why adding a running index feature isn't sufficient to break order invariance.
6. How does ApolloPFN perform when the forecast horizon H is much larger than what was seen during training?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ApolloPFN, a time-aware prior-fitted network for zero-shot time-series forecasting with exogenous variables. Existing foundation models (e.g., Chronos, Moirai) often ignore exogenous covariates or require fine-tuning, while TabPFN-TS lacks temporal inductive bias due to its i.i.d. training assumption. ApolloPFN addresses these issues through two innovations: (1) a temporal synthetic data generation procedure and (2) architectural modifications adding RoPE positional encodings and expanded attention to capture temporal dependencies. Experiments on M5 and electricity price forecasting benchmarks demonstrate that ApolloPFN achieves state-of-the-art zero-shot performance with exogenous inputs and remains competitive on classical univariate benchmarks despite being smaller than large TSFMs.

### Strengths
1. The proposed temporal SCM generation procedure and the SNGN algorithm are well-motivated and supported by empirical evidence. The design effectively introduces temporal dependencies into the synthetic data, which allows the model to learn meaningful temporal structures during training. This approach represents a thoughtful and principled way to bridge the gap between tabular PFNs and time-series forecasting tasks.

### Weaknesses
1. Many of the “failure mode” demonstrations (e.g., Fig. 2) rely on single illustrative examples rather than aggregated or statistically supported analyses. Without quantitative evidence across a larger number of series or benchmarks, it is difficult to assess whether these issues with TabPFN-TS are systematic or merely anecdotal, which somewhat weakens the empirical foundation of the argument.

2. Most datasets containing exogenous covariates (such as M5 and electricity price forecasting) are relatively small in scale and limited in diversity. It remains unclear how ApolloPFN would generalize to larger, more complex multivariate datasets.

3. The experimental comparison includes a limited set of baselines and datasets, which makes the empirical results less solid and comprehensive.

4. The proposed architectural modifications—such as the incorporation of positional embeddings and expanded attention—are relatively standard and have been widely adopted in time-series Transformer variants. As a result, the architectural novelty of ApolloPFN is somewhat limited, with the main contribution lying more in the adaptation of PFNs to temporal contexts rather than in introducing fundamentally new modeling mechanisms.

5. The overall writing and exposition of the paper are somewhat average. Several sections could benefit from clearer explanations, better structural flow, and more precise terminology. Improving the presentation would help readers more easily grasp the key motivations, methodological details, and experimental setups.

### Questions
1. Since the model employs an encoder-only Transformer architecture, is there a defined maximum sequence length for T (time steps) and F (features) during training? If such limits exist, what are the specific maximum values for T and F, and what are the implications of these limits on the model’s ability to process long or high-dimensional time series? Additionally, are there strategies, such as memory-efficient attention mechanisms that could potentially allow the model to handle sequences beyond these limits?

2. Is the model capable of handling variable-length sequences for both T (time steps) and F (features) during inference? If so, was this flexibility explicitly supported during training, for example by including samples of varying lengths, to improve the model’s generalization across sequences of different sizes? 

3. In the design of the “Expanding Attention” mechanism, what is the rationale behind allowing all points to attend to each other? Could a more restricted attention mechanism, such as causal or masked attention that only considers past or neighboring points, be sufficient to capture temporal dependencies? 

4. What is the total amount and detailed configuration of the synthetic data used for training? For example, how many samples, series lengths, or graph instances were generated?  And how significant is the impact of the synthetic data generation parameters on the final performance of the model?

### Soundness
2

### Presentation
1

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
This paper describes a novel prior-data fitted network, ApolloPFN, that improves the drawbacks of TabPFN by 	adapting to the current date, synthetic data generation and time-aware architectural modifications. The model is then validated across datasets with exogenous  variables and shows improved inferences.

### Strengths
1. Paper is clear and well presented. 
2. Framework reforms well across the two datasets considered by the authors (M5 dataset and electricity price forecasting).
3. Paper presents an algorithm that improves PFM architectures for exogenous features.

### Weaknesses
1. While the framework is established for multivariate datasets, the comparison with just two datasets seems limited. Other multivariate datasets like wind power forecasting etc can be used to understand the performance of this algorithm further. 
2. Benchmark models can be improved by considering TTM and/or Flowstate which also works with exogenous features and is in the top 10 of Gift-Eval dataset.

### Questions
1. How well does the algorithm work with univariate datasets? Has it been tested with all the standard benchmark datasets like ETTh1 etc from time series foundation models literature. 
2. How does the computational cost change of ApolloPFN as compared to TabPFN?

### Soundness
2

### Presentation
3

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
The paper targets zero-shot forecasting with exogenous variables—a gap in many TSFMs (Chronos, Sundial, TimesFM, TimeLLM, LagLlama) that often ignore or require finetuning for exogenous covariates. It critiques TabPFN-TS (order-invariant tabular PFN with TS features added) and proposes ApolloPFN, contributing: (i) a time-aware synthetic data generator (graph-based with time-dependent roots) to better match TS priors; and (ii) architectural biases that respect temporal order. On benchmarks with exogenous info (e.g., M5 aggregations and electricity price), ApolloPFN achieves SOTA zero-shot performance and outperforms TabPFN-TS on average; it is competitive against Moirai and large univariate TSFMs.

### Strengths
Adapting the PFN paradigm to time series with native exogenous handling and explicit time-aware inductive bias is a meaningful advance. The critique of using order-invariant tabular FMs for TS is convincing (forecasting requires order sensitivity), and the synthetic-prior design tailored to TS is a good fit for PFN training. Given the practical importance of zero-shot + exogenous, the contribution is significant.

* The paper clearly articulates failure modes of TabPFN-TS (order-invariance, weak trend extrapolation, poor calibration under regime changes) and motivates architectural/time-aware changes. Examples illustrate the gap.


* The experimental comparisons show gains on electricity and M5 aggregations with exogenous, where ApolloPFN is SOTA or competitive. The authors note the Moirai confound: training exposure to public benchmarks complicates strict zero-shot comparisons—where the transparency is appreciated.


* A comprehensive table on classical univariate shows ApolloPFN is competitive too, though the conceptual focus is on the setting with an increased number of exogenous variables

### Weaknesses
I believe the paper has a limited analysis of probabilistic calibration and robustness to exogenous shift, which can be needed in real-life time-series settings. I also believe that the heavy reliance on synthetic priors deserves more discussion on alignment to real exogenous processes. I agree with the scaling constraints from quadratic attention as noted by the authors, which is a considerable weakness.

### Questions
- Can the authors provide calibration (PIT, coverage) and counterfactual sensitivity analyses for exogenous covariates (e.g., price elasticity sanity checks)?


- Is it possible to add ablations isolating each time-aware architectural change and the synthetic prior components that matter most?


- Is it feasible to design and evaluate robustness under exogenous distribution shift (e.g., unseen promo patterns)?


- Can you clarify zero-shot rigor, e.g., ensuring no leakage from synthetic priors into test covariate regimes; specifying any exogenous data preprocessing rules?

### Soundness
3

### Presentation
2

### Contribution
2
