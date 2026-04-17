# Boundary-Aware Tokenization for Event-Driven Time-Series Forecasting

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Transformer-based large sequence models have recently been extended from language to time-series to capture long-range dependencies and heterogeneous dynamics. However, unlike language, time-series lack a natural dictionary for principled tokenization: existing large sequence models often resort to fixed-length tokens or patches for computational efficiency. This design can obscure regime changes, expend attention on low-information tokens, and restrict the effective context length. We address this limitation with Boundary-aware tokenization, which initiates new tokens only at predicted regime changes in the time-series, analogous to how spaces delimit words in language. At its core, the model integrates an unsupervised boundary detector to form variable-length chunks, an intra-chunk fusion module to derive chunk-level token embeddings, and a smoothing module to stabilize training, before passing the resulting tokens to Transformer-based modules. We further add a gating refinement that fuses fixed- and variable-length representations before the forecasting decoder, enabling adaptive selection during pre-training based on data patterns. This design directly addresses event-driven regime changes, while remaining robust in stationary regimes. Across diverse benchmarks, our method reduces forecasting error by 10.5\% on average, with learned chunks aligned with true regime boundaries. We also show that the model adaptively reverts to fixed-length tokenization in stationary time-series.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Boundary-Aware Tokenization, a Transformer-based framework that dynamically segments time-series into variable-length chunks aligned with event boundaries. Unlike fixed-length tokenization used in PatchTST, BT-LSM detects regime changes using an unsupervised boundary detector based on first- and second-order embedding dynamics. Within each chunk, a mixture-of-experts fusion module combines multiple pooling statistics to produce chunk-level tokens, followed by a chunk-level Transformer and a causal smoothing module for stability. A gating refinement further combines fixed- and variable-length tokenization to ensure robustness on both stationary and event-driven data. Experiments across datasets show ~10.5% average forecasting error reduction compared to strong baselines. Ablation and visualization support the claim that learned boundaries align with real regime changes.

### Strengths
1. Dynamic tokenization is a crucial problem for time-series models. The paper insightfully draws an analogy between language tokenization and event segmentation in time-series, and proposes a boundary-based dynamic tokenization approach that effectively addresses the inefficiencies of fixed patching.

2. The proposed pipeline (boundary detection → MoE fusion → chunk Transformer → smoothing → gating → decoder) is well-motivated and internally coherent, with each component designed to tackle a specific modeling challenge.

3. Theorem 1 formally establishes invariance to intra-chunk resampling—an elegant theoretical property that enhances the model’s robustness to irregular sampling.

### Weaknesses
1. If the original multivariate time series exhibit strong temporal lags or misalignments across variables, the proposed method would still produce a single set of shared boundaries for all variables. This may limit its ability to capture variable-specific regime changes.

2. In the test datasets, what proportion of cases have g₍var₎ > g₍fix₎? This statistic would help clarify whether the performance improvement mainly stems from the proposed dynamic tokenization mechanism.

3. The paper lacks visualization of the gating weights among the different embedding experts (attention pooling, mean, max, etc.). Such visualization could provide insights into which features contribute most to the model’s representation.

4. The experiments involve a relatively small set of datasets and baselines, which somewhat weakens the empirical evidence supporting the paper’s claims.

5. Section 1 title: “Introductionn” → should be corrected to “Introduction.”

### Questions
1. Does the decoder-only Transformer incorporate positional embeddings to preserve temporal ordering? If so, please clarify what type of positional encoding is adopted.

2. How are the variable-length and fixed-length representations (z₍var₎ and z₍fix₎) aligned before being fused by the gating module? It appears that they are aligned sequentially, which may imply that the two tokens being fused could correspond to quite different original temporal positions.

3. How is the target boundary rate chosen in practice? Is it tuned as a hyperparameter or derived from data statistics? It would be helpful to clarify whether the model’s performance is sensitive to this setting.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Boundary-aware Tokenization Large Signal Model (BT-LSM) for time-series, introducing a lightweight unsupervised boundary detector and mixture-of-experts chunk embeddings. The proposed model allocates tokens adaptively to event-driven transitions, avoiding uniform waste. Experimental results demonstrate that BT-LSM achieves over 10.5% lower forecasting error at matched compute budgets across diverse benchmarks, including energy, power, and traffic data.

### Strengths
- The paper is well-motivated and addresses a significant limitation in the literature.
- The proposed method is evaluated on multiple benchmark datasets, demonstrating superior performance compared to baselines.

### Weaknesses
- The paper omits evaluations on widely used benchmarks in the time series forecasting community, such as the ECL and Weather datasets. Similarly, the baseline does not include several key state-of-the-art models, such as DLinear and TimesNet, which are commonly used for benchmarking and would provide a more robust comparative analysis. 
- The description of the experimental setup is vague and lacks sufficient details. For instance, the length of the input sequence appears to be 144 based on visualization, but there is no explicit mention of this in the paper. This lack of clarity makes it difficult for readers to fully understand the conditions under which the experiments were conducted.
- The experimental evaluation is primarily focused on the ETTh1 and ETTh2 datasets for multi-horizon forecasting. This narrow scope raises concerns about the generalizability of the proposed method to other datasets and forecasting tasks.
- The paper lacks the ablation analysis on the input length. This is a critical omission, as input length is a significant parameter in time series forecasting models.
- The paper does not analyze whether the proposed model can maintain its effectiveness under different conditions, such as shorter input sequences or varying levels of data sparsity. These scenarios are common in practical forecasting tasks, and further analysis is needed to ensure the method's applicability in such settings.

### Questions
See Weakness

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
The paper introduces a boundary-aware tokenization scheme for time series forecasting, which starts new tokens at event changes. The model contains an unsupervised boundary detection module, an intra-chunk fusion module, a smoothing module and a gating refinement module that dynamically selects fixed-length and boundary-aware tokenization. Experiments across multiple datasets indicate consistent improvements from the proposed tokenization.

### Strengths
1. The proposed tokenization scheme has clear motivation and the design is new. The unsupervised boundary detector based on velocity and acceleration does not involve additional training overhead.

2. Experiments on multiple datasets show the effectiveness of the proposed tokenization scheme. The paper also presents many case studies to show the benefits.

### Weaknesses
1. The paper does not compare with some recent time series tokenization schemes [1,2,3]. For example, [1] also moves beyond fixed encodings via pattern-based tokenization.

2. The proposed tokenization scheme is not lossless. One cannot deterministically reconstruct the original series from tokens, which may limit certain applications.

3. How robust is the hard boundary of 0.5?

4. It would help to show a controlled comparison where only the tokenization differs (fixed vs proposed boundary-aware vs other existing tokenization schemes) under different forecasting backbones, to demonstrate model-agnostic gains and isolate the contribution of tokenization.

[1] Byte Pair Encoding for Efficient Time Series Forecasting

[2] Enhancing foundation models for time series forecasting via Wavelet-based tokenization

[3] TOTEM: Tokenized Time Series Embeddings for General Time Series Analysis

### Questions
1. Are boundaries/tokens shared across variables or detected per channel in multivariate time series?

2. Minor typo: accross -> across at Line 399

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes **BT-LSM (Boundary-aware Tokenization Large Signal Model)** for event-driven time-series forecasting, addressing the limitation of fixed-length tokenization by adaptively forming variable-length tokens aligned with regime changes. It integrates an unsupervised boundary detector, mixture-of-experts (MoE) chunk embedding, chunk smoothing, and a gating refinement that fuses fixed- and variable-length representations. BT-LSM concentrates model capacity on event transitions while remaining robust in stationary regimes, reducing forecasting error by 10.5% on average across diverse benchmarks and aligning learned chunks with true event boundaries.

### Strengths
1. The paper addresses a critical flaw of fixed-length tokenization in time-series forecasting—its inability to align with event-driven regime changes—by introducing an unsupervised boundary detector. This detector leverages embedding dynamics (velocity, acceleration, energy change) to identify natural chunk boundaries, ensuring tokens concentrate on critical transitions (e.g., spikes, inflections) rather than wasting capacity on redundant stationary spans. Unlike supervised segmentation methods (e.g., SIMTSeg, U-Time) that require labels, this unsupervised design is broadly applicable across time-series domains.
2. The proposed gating refinement module fuses variable-length (event-aligned) and fixed-length (stationary-optimized) representations, enabling BT-LSM to adapt dynamically. In stationary time-series (e.g., smooth cycles), the gate prioritizes fixed-length tokens to preserve short-range statistics; in bursty/irregular data, it shifts to variable-length tokens to capture events. This design eliminates the trade-off between event sensitivity and stationary robustness, a limitation of purely fixed or variable tokenization methods.
3. The paper proves a resampling invariance theorem (Theorem 1), ensuring BT-LSM’s chunk embeddings and forecasts remain unchanged under intra-chunk resampling (e.g., varying sensor sampling rates).

### Weaknesses
1. The related work focuses on temporal-domain tokenization methods but omits direct comparisons to frequency-domain forecasting models (e.g., FEDformer, TimesNet) that excel at capturing periodic patterns. This leaves uncertainty about BT-LSM’s performance relative to frequency-aware approaches, especially for time series with strong periodicity but weak event signals.
2. BT-LSM uses padding-and-masking to handle variable-length chunks in batch processing, which becomes inefficient for extremely long sequences (e.g., multi-year high-frequency data). The paper does not explore alternative batching strategies (e.g., chunk-level bucketing) to mitigate this, restricting its application to moderate-length time series.
3. While BT-LSM performs well on datasets with clear event patterns (e.g., solar spikes, traffic peaks), it lacks evaluation on low signal-to-noise ratio (SNR) or highly irregular time-series (e.g., sparse medical sensors, non-periodic industrial anomalies). No experiments demonstrate its robustness to such edge cases, limiting generalizability to real-world "messy" data.

### Questions
1. The paper mentions that the boundary detector relies on parameters like boundary probability threshold and minimum chunk length, but lacks a systematic optimization method. What specific hyperparameters of the boundary detector have the most significant impact on BT-LSM’s forecasting performance? Is there a potential adaptive adjustment strategy (e.g., learning hyperparameters via data-driven methods) that can reduce manual tuning efforts across different datasets?
2. Since BT-LSM has not been tested on low SNR or highly irregular time-series (such as sparse medical sensors), what modifications to the boundary detector or chunk embedding module might help the model better filter noise and capture valid event boundaries in such challenging data scenarios?
3. The padding-and-masking strategy used by BT-LSM becomes inefficient for extremely long time series. Are there alternative batch processing strategies (e.g., chunk-level bucketing, hierarchical chunking) that the authors have considered to improve the model’s scalability, and what preliminary results or feasibility analyses exist for these strategies?

### Soundness
2

### Presentation
2

### Contribution
2
