# U-Cast: Learning Hierarchical Structures for High-Dimensional Time Series Forecasting

- Decision: Reject
- Scores: 6, 6, 6, 6, 2

## Abstract
Time series forecasting (TSF) is a central problem in time series analysis. However, as the number of channels in time series datasets scales to the thousands or more, a scenario we define as **High-Dimensional Time Series Forecasting (HDTSF)**, it introduces significant new modeling challenges that are often not the primary focus of traditional TSF research. HDTSF is challenging because the channel correlation often forms complex and hierarchical patterns. Existing TSF models either ignore these interactions or fail to scale as dimensionality grows.  To address this issue, we propose **U-Cast**, a channel-dependent forecasting architecture that learns latent hierarchical channel structures with an innovative query-based attention. To disentangle highly correlated channel representations, U-Cast adds a full-rank regularization during training.  We also release **Time-HD**, the first benchmark of large, diverse, high-dimensional datasets. Our theory shows that exploiting cross-channel information lowers forecasting risk, and experiments on Time-HD demonstrate that U-Cast surpasses strong baselines in both accuracy and efficiency. Together, U-Cast and Time-HD provide a solid basis for future HDTSF research.  Our [code](https://anonymous.4open.science/r/Time-HD-Lib-1A71) and [benchmark](https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series) are available to ensure reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of high-dimensional time series forecasting (HDTSF), where the number of channels can reach into the thousands or tens of thousands, common in practical settings such as finance, climate, or sensor networks. The authors introduce U-CAST, an attention-based forecasting architecture designed to explicitly model latent hierarchical channel correlations through a query-based attention mechanism, and further regularizes representation learning with a novel full-rank regularization term. To benchmark scalable methods, the authors also release Time-HD, a dataset suite spanning 16 high-dimensional, highly-correlated real and synthetic datasets. The paper presents theoretical analysis, empirical ablation, and comprehensive benchmarking on Time-HD, where U-CAST demonstrates improved forecasting accuracy and efficiency compared to a suite of recent baselines.

### Strengths
**1. Clear Focus on a High-Impact Problem:** The paper tackles HDTSF, a setting that is under-explored yet highly relevant for real-world applications at scale. The formalization and motivation of the problem are well articulated.
Theoretical Analysis: The authors present a mathematically clear and rigorous argument (Theorems 1 and 2, with detailed proofs on pp. 3–4, 19–21) demonstrating how incorporating cross-channel dependencies can reduce forecasting risk, with the benefit growing in higher dimensions.


**2. Extensive Experiments:** Table 4 (pages 8–9) presents results on 16 datasets, showing U-CAST achieving top MSE/MAE in most cases, with consistent performance gains relative to strong baselines.


**3. Efficiency Claims Supported:** The computational complexity analysis (Appendix L) is detailed, and Figure 2 (page 9) directly illustrates that U-CAST is both faster and more memory-efficient than iTransformer at high C, justifying claims of scalability.

### Weaknesses
**1. Limited Theoretical Depth in Hierarchy Discovery:**  While the theory (Theorems 1–2) rigorously analyzes the benefit of cross-channel correlation in VAR processes, it does not directly address the identifiability, learnability, or optimality of the hierarchical latent structure produced by U-CAST's attention mechanism. No guarantees (even informal) are offered for whether the architecture can recover ground-truth groupings under any conditions, nor how deviations from "true" hierarchy affect forecasting risk.

**2. Unclear Generalizability Beyond Correlated Data:**  Time-HD is valuable but by construction consists almost exclusively of highly-correlated channels (Table 3, Correlation column). This leaves unclear how U-CAST would perform if faced with more heterogeneous or less-structured high-dimensional data.

### Questions
1. Can the authors comment on, or simulate, performance when channels are weakly correlated or only loosely grouped ?
2. Can the authors provide empirical evidence on whether the attention-based hierarchical latent representations discovered by U-CAST match known hierarchical structures in real-world datasets?
3. In the NIPS 2024 workshop[1], some researchers pointed out that current methods sometimes use the "drop-last" trick [2] to improve performance. Therefore, It is recommended that you clarify whether the "drop - last" operation was used in your paper in the implementation details section of your paper for transparency.

[1] Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation

[2] TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of forecasting time series data with a large number of potentially interdependent channels. The authors first highlight the lack of a comprehensive evaluation benchmark for this domain, and they tackle this by introducing 16 high-dimensional time series datasets called Time-HD. Next, they propose a new architecture that learns a latent hierarchical structure among the channels using an attention-based approach. Additionally, they introduce a method to enforce disentanglement between channels through a full-rank loss function.

### Strengths
1- The motivation of both investigating the evaluation of channel dependency forecasting methods and proposing ones is timely.

2- The benchmarks proposed can be useful for properly evaluating methods.

3- The performance reported sounds good.

### Weaknesses
1. While a new benchmark is proposed, it would be useful to test against existing benchmarks for sanity checks, even if they have fewer channels. This ensures the new benchmark isn’t designed to favor the method, and provides a safety check. Additionally, a synthetic benchmark with known channel correlations (ranging from 0 to a specific value) could show how MSE behaves as the number of channels increases, with expected trends like monotonic decrease or stabilization when correlation is zero and comparison with competitors that would not exhibit such trend.

2. The theory presented is somewhat trivial, as the conclusions could be made without detailed calculations. It would be more valuable to quantify the MSE gap explicitly rather than just claiming qualitative improvement.

3. The architecture doesn’t seem very novel, and the importance of each module should be more clearly explained.

### Questions
1. Could you explain the rationale behind the architectural choices? Why is the first module called "channel embedding" if it just maps time to another dimension? Does it account for recurrent structures and time dynamics? Also, why choose a transformer architecture over other methods for learning the latent network?

2. It would be helpful to explore the trade-off of the parameter $\alpha$ in the loss function, especially in an oracle setting. How sensitive is it (worst vs best case) across different benchmarks? This would help show the robustness of the approach.

3. The theoretical claims seem trivial, it's obvious that more correlated channels would lead to better performance. Could the authors provide a more in-depth analysis? How much of a gain can we expect, and can specific algorithms be used to narrow this gap?

4. It would be useful to test on well-known forecasting benchmarks for sanity checks. Additionally, creating a synthetic benchmark with known channel interactions (e.g., for 1000 channels with specific correlations) and tracking MSE as the number of channels increases could reveal useful insights.

5. Some related works on multi-task learning for handling channel dependencies in forecasting are missing. These might be relevant, such as:

a) A Multi-Task Learning Approach to Linear Multivariate Forecasting

b) Analyzing Multi-Task Regression via Random Matrix Theory for Time Series Forecasting

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes U-CAST, a channel-dependent forecasting architecture designed for High-Dimensional Time Series Forecasting (HDTSF). It addresses scalability and complex inter-channel correlations that emerge when the number of channels grows to thousands or more.
U-CAST introduces two main components:
- A Hierarchical Latent Query Network (HLQN) that learns latent multi-scale channel structures via query-based attention;
- A Full-Rank Regularization term to encourage disentangled and non-redundant channel representations.

The paper also introduce TIME-HD, a new benchmark suite of 16 large-scale datasets spanning various domains.

### Strengths
- **Novel formulation of HDTSF** as a distinct problem setting, highlighting the gap in scaling existing TSF models.
- **Comprehensive benchmark (TIME-HD)** that fills an important void in the community, covering large, diverse, and highly correlated datasets.
- **U-CAST design is conceptually clear and efficient**, combining hierarchical latent queries with disentanglement regularization.
- **Strong empirical performance**: consistently achieves the best or second-best results across 16 datasets, outperforming iTransformer in both accuracy and memory usage (Fig. 2, Table 4).
- **Theoretical analysis** provides formal grounding for the benefits of channel-dependent modeling and rank regularization.
- Extensive **ablation studies and sensitivity analyses** confirm robustness to hyperparameters and the contribution of each component (Table 11, Fig. 5–7).

### Weaknesses
1. **Limited conceptual novelty despite strong engineering**
   - The core components (hierarchical queries and full-rank regularization) are **incremental extensions** of standard attention architectures and covariance penalties.
   - The paper’s theoretical analysis (VAR-based) is elegant but **not specific to U-CAST** — it supports general CD modeling rather than justifying this architecture’s unique design.
2. **Lack of deep analysis on why it works well**
   - While the paper visualizes covariance evolution (Fig. 3) and attention maps, these are **qualitative illustrations**, not rigorous analyses. There is no quantitative or causal explanation for how full-rank regularization specifically improves hierarchical learning or reduces overfitting.
3. **Theoretical connection between rank regularization and hierarchical attention is weak**
   - Theorems 3 and 4 establish full-rank and entropy monotonicity but do **not mathematically link these to better forecasting performance**.
   - The claimed “disentanglement” effect is intuitive but not quantified (e.g., via mutual information or redundancy metrics).

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of high dimensional time series forecasting on data sets with 1000s of inter-dependent time series. The main idea is that using the correlations between the time series can lead to more accurate time series forecasting.

It is theoretically shown that if there is an actual flow of information between the time series, it can lead to a more accurate forecast. The proposed time series forecasting method U-Cast models the correlations between various time series using attention layers in a scalable manner, to produce the final forecasts.

The final contribution is a collection of datasets for high dimensional forecasting for evaluation. This is a collection of datasets with correlations between the time series which can leveraged for accurate prediction.

Experimental results show that the proposed U-Cast method is more accurate in terms of MSE and MAE compared to other state-of-the-art baselines.

### Strengths
This paper presents several good contributions
- The proposed method U-Cast is more accurate compared to other baselines, thus able to make use of the correlations between the time series to produce accurate forecasts.
- The dataset is useful to the time-series community to move away from traditional datasets, enabling evaluation on more diverse sets of datasets.
- Formal theoretical proofs establish the intuition that using the correlation information can lead to more accurate forecasts.

### Weaknesses
- The main weakness of this paper is that U-Cast is not evaluated on traditional multi-variate time series datasets such as Exchange, Electricity, ETTh, Weather. While the method is not designed for such datasets, it would be good see those results for a more complete evaluation.
- Foundational time series models are also missing from the evaluation, it would be interesting to see how the model compares against these baselines.

### Questions
- How does U-Cast compare to foundation time series forecasters? While U-Cast should have an advantage of foundational models, the advantage needs to be experimentally verified.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces U-CAST, a novel architecture designed to address the scalability and structural modeling challenges in High-Dimensional Time Series Forecasting (HDTSF). Unlike traditional models that ignore or poorly scale with inter-channel dependencies, U-CAST learns latent hierarchical channel structures through a query-based attention mechanism and employs full-rank regularization to disentangle correlated representations. The authors also release TIME-HD, the first large-scale benchmark suite containing 16 datasets (1k–20k channels) across diverse domains. Theoretical analysis demonstrates that exploiting cross-channel information lowers forecasting risk, and extensive experiments show that U-CAST outperforms strong baselines such as iTransformer and TSMixer in both accuracy and efficiency. Overall, the paper contributes a scalable model, a comprehensive benchmark, and an open-source library that collectively establish a new foundation for research in high-dimensional time series forecasting

### Strengths
The authors provide formal proofs showing that channel-dependent (CD) models have lower Bayes risk than channel-independent ones when cross-channel dependencies exist, lending solid theoretical support to the model design 

The paper conducts broad empirical comparisons with both channel-independent and channel-dependent baselines, consistently showing U-CAST’s superior accuracy and efficiency, as well as meaningful ablation studies confirming the contribution of each component

### Weaknesses
My main concerns lie in the definition of “high-dimensional” used in the proposed benchmark and the design rationale of the model architecture.

1. Questionable definition of high dimensionality.

The paper defines “high-dimensional” time series by treating measurements from different objects or spatial locations (e.g., stations or sensors) as separate channels. However, in conventional time-series analysis, “dimensions” typically refer to distinct attributes or variables of a single object—such as temperature, humidity, and wind speed in a weather dataset—rather than independent time series collected from different spatial points. While the signals from different locations may be temporally aligned and correlated, this setup is more appropriately considered a spatio-temporal prediction problem rather than a high-dimensional multivariate one, which has also been well studied. This represents a potential conceptual misuse of “dimensionality” that weakens the claimed novelty of the benchmark.

2.	Unintuitive and unconventional architectural design.

The proposed U-CAST architecture raises several design concerns.

(1) First, using attention for up/down scaling along the token (channel) dimension is unconventional and lacks clear justification. The “Latent Query Attention” mechanism, where the latent queries Q_l are entirely learnable parameters, from my understanding is similar to linear or low-rank attention methods such as Linformer or Performer, which sacrify the attention performance, especially your Q_l is not even derived from H and total parameterization make it become part of W_k (I dont think this should be called attention, if this understanding is correct). Also, why not use some conventional methods for these up/down sampling like convolution or mlp 

(2) Second, the model appears to process the temporal dimension through a single predictor module at the bottleneck, without interleaving temporal and channel modeling layers, could you justify? In related domains such as video or spatio-temporal forecasting, it is common practice to alternate or integrate spatial and temporal modeling to better capture cross-dimensional dependencies.

(3) Third, the paper would benefit from additional ablation studies to isolate the contribution of each architectural component. In particular, starting from a well-established channel-independent baseline and progressively adding the proposed modules would provide more convincing evidence for the effectiveness of each design element. Proposing an entire new (and arguble) architecture from scratch is hard to convince me (unless largely pretrained with open weights avilable).

### Questions
Are you applying any patching to derive the token sequence, if yes, is the patching CD or CI?

What are the model size to be considered (I saw you are using 8A/H100 to power the training, why is it so costly)?

### Soundness
2

### Presentation
3

### Contribution
2
