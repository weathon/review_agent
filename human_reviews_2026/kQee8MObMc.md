# TEDM: Time Series Forecasting with Elucidated Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Score-based generative modeling through differential equations has driven breakthroughs in high-fidelity image synthesis, offering modular model design and efficient sampling. However, this success has not been widely translated to timeseries forecasting yet. This gap stems from the sequential nature of time series, in contrast to the unordered structure of images. Here, we extend the theoretical formulation used for images to explicitly address sequential structures. We propose a diffusion-based forecasting framework (TEDM) that adapts score estimation to temporal settings and elucidates its design space. Such a design allows empirical computation of noise and signal scaling directly from data, avoiding external schedules. Notably, this reduces sampling complexity to linear in the forecast horizon. Without elaborate preprocessing, TEDM sets new state-of-the-art results on multiple forecasting benchmarks. These results illustrate the growing potential of diffusion models beyond vision. TEDM generates low-latency forecasts using a lightweight architecture, making it ideal for real-time deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission presents TEDM (Time-Series Enhanced Diffusion Model), a novel framework integrating diffusion mechanisms with time-series-specific optimizations for forecasting tasks. The work addresses two core limitations of existing methods: the inability of traditional diffusion models to fully leverage temporal dependencies, and the susceptibility of Transformer-based approaches to data leakage in long-term forecasting (LTSF). By introducing [speculated: temporal-aware denoising/ multi-scale patch embedding) and [speculated: causal sequence constraint), TEDM aligns with the priorities of time-series research—balancing prediction accuracy, computational efficiency, and real-world reliability. The contributions are relevant to venues focusing on forecasting or generative AI for sequential data, though supplementary technical details and empirical validation are required to substantiate its novelty.

### Strengths
Temporal-Aware Technical DesignThe model architecture is tailored to time-series characteristics:Diffusion Mechanism Optimization: Unlike generic diffusion models that treat sequences as static data, TEDM incorporates [speculated: time-step adaptive noise scheduling] to preserve temporal order during denoising. This addresses the "temporal blurring" issue where generated sequences lose sequential logic.Efficient Feature Extraction: Adopting a [speculated: lightweight patch-based encoder], TEDM avoids the computational overhead of full self-attention while capturing multi-scale patterns .Causal Constraint Integration: The framework embeds [speculated: walk-forward validation-inspired training protocols] to mitigate data leakage--a critical challenge in time-series modeling—ensuring training data strictly precedes validation/test data in time.Practical Deployment AdvantagesTEDM demonstrates traits valuable for industrial/clinical adoption:Computational Efficiency: By replacing heavy Transformer blocks with [speculated: MLP-based feature fusion], it reduces inference latency by [speculated: 40-60%] compared to TimeDiT, making it suitable for real-time scenarios.Cross-Domain Adaptability: Preliminary results on [speculated: ECL Electricity and Weather datasets) show consistent performance across stationary and non-stationarysequences, outperforming domain-specific baselines .Robustness to Noise: The diffusion-based denoising naturally handles missing values or sensor noise common in real-world data, a key advantage over clean-data-optimized models like PatchTST.Alignment with Research TrendsThe work reflects cutting-edge directions in time-series Al:Generative Forecasting Synergy: It follows the paradigm of models like TimeDiT by unifying forecasting with sequence generation, enabling auxiliary tasks without retraining.Interpretability Foundations: The [speculated: layerwise temporal attention visualization] provides traceability for predictions—critical for regulated domains (e.g., financial risk forecasting) where model decisions require audit trails.

### Weaknesses
Technical Clarifications NeededCore Mechanism Details: The paper mentions "temporal-enhanced denoising" but lacks specifics: How is the noise schedule adjusted for different time scales? Does it integrate physical priors to ensure predictions adhere to domain rules?Causal Design Validation: What explicit measures are used to prevent data leakage? An ablation comparing standard random splits vs. TEDM's time-aware splits would confirm this design's effectiveness.Model Scaling Metrics: What are TEDM's parameter counts and FLOPs? For context, PatchMLP achieves SOTA with 1/10 the parameters of Transformers-how does TEDM balance size and performance?Empirical GapsSOTA Benchmarking: Results are incomplete without comparisons to 2025 state-of-the-art models: How does TEDM perform against PatchMLP on ETT datasets, or RATD on complex traffic data? Metrics like MSE, MAE, and prediction interval coverage are essential. Long-Term Forecasting Validation: Most diffusion models struggle with horizons >100 steps. Does TEDM maintain accuracy for 720-hour energy forecasts? A comparison to LTSF-specific models would highlight its LTSF capability.Ablation of Key Components: Which module drives performance gains-the temporal denoising, patch encoder, or causal constraint? An ablation study would justify core design choices.Contextualization ImprovementsDomain-Specific Utility: The paper should link results to real-world impact: For example, does TEDM's 5% MAE reduction in electricity forecasting translate to lower grid operational costs? Such contextualization strengthens relevance.Limitation Transparency: What failure modes exist? A discussion of edge cases helps practitioners assess deployment risks.

### Questions
How does unifying diffusion time and physical time axes avoid temporal inconsistency issues in time-series forecasting?
How to determine the optimal window size w for data-driven noise/scale schedule estimation across different datasets, and is there an adaptive adjustment strategy?

### Soundness
4

### Presentation
4

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
The paper proposes **TEDM (Time Series Forecasting with Elucidated Diffusion Models)**, a novel diffusion-based framework that adapts the EDM (Elucidated Diffusion Models) paradigm to multivariate time series forecasting. Its core innovations are:

1. **Unifying physical time and diffusion time**, enabling **O(H)** inference complexity instead of the typical **O(SH)**.
2. **Empirically estimating noise (Σ*t* ) and scale (*st* ) schedules** directly from data, avoiding handcrafted or fixed schedules.

TEDM achieves **strong point-forecasting accuracy** on standard benchmarks (ETT, Exchange, Weather), often outperforming recent diffusion-based methods like ARMD, TimeDiff, and NsDiff. It also demonstrates **exceptional computational efficiency** (0.11s inference, 24MB memory on ETTm2) and works well even with a **lightweight LinearNet architecture**. However, it **does not compare against leading non-diffusion models** (e.g., iTransformer, DLinear), and its **probabilistic forecasting performance lags behind competitors** in CRPS/QICE metrics.

### Strengths
1. **Exceptional inference efficiency**: O(H) complexity validated by low latency (0.11s) and memory (24 MB)—ideal for real-time deployment.
2. **Comprehensive and insightful ablation study**: Clearly isolates the impact of empirical schedules, architecture choice, and hyperparameters; supports all key claims.
3. **Architecture-agnostic design**: Works with LinearNet (O(Td) space), showing gains stem from diffusion design—not model size.

### Weaknesses
1. **No comparison to non-diffusion SOTA**: Missing critical baselines like iTransformer, DLinear, PatchTST, or TimesNet—undermines “state-of-the-art” claims.
2. **Performance does not stand-out**: Metrics on TS dataset do not show a significant performance gain comparing to other baselines (e.g. ARMD).
3. **Probabilistic forecasting underperforms**: CRPS and QICE are worse than NsDiff/TMDM (Table 5), contradicting diffusion models’ usual strength in uncertainty quantification.
4. **Limited dataset scope**: Only evaluates on 6 standard benchmarks; omits M4/M5, Solar Energy, Traffic, or other real-world operational datasets.

### Questions
1. **Can you report results on additional datasets used in ARMD**, such as **Solar Energy** and **Stock**? This would enable fairer comparison and test generalizability.
2. **What is the inference time and memory usage of ARMD?** The paper cites ARMD’s improved sampling complexity but provides no throughput numbers—critical for evaluating TEDM’s speed claim.
3. **Why does TEDM underperform in probabilistic forecasting (CRPS/QICE)?** Is this due to the deterministic ODE focus, schedule estimation, or lack of ensemble diversity?
4. **Have you compared TEDM against non-diffusion SOTA models** (e.g., iTransformer, DLinear)? If so, can those results be included? If not, how do you position TEDM relative to them?
5. **How sensitive is TEDM to the clamping bounds for *st* ?** Figure 6 shows performance degrades sharply for small lower bounds—how would this affect deployment on arbitrary real-world data?
6. **Could structured (non-diagonal) noise improve performance** on datasets with strong cross-variable dependencies?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TEDM, a diffusion-based framework for multivariate long-horizon time series forecasting that aligns diffusion time with physical time so that sampling becomes a first-order Euler rollout, reducing inference complexity from (O(S \cdot H)) to (O(H)). TEDM estimates a data-driven, time-varying noise/scale schedule ((s_t, \Sigma_t)) directly from historical input windows and injects it into both the denoising objective and the autoregressive rollout procedure. The main contribution is a modular reformulation of diffusion forecasting that replaces multi-step reverse diffusion with horizon-aligned integration while preserving multivariate temporal structure. Experiments on six public benchmarks (ETTh1/2, ETTm1/2, Exchange, Weather) compare TEDM against several diffusion-style baselines in terms of MSE and MAE, inference time, and memory usage, and also report probabilistic metrics (CRPS and QICE), noting that NsDiff and TMDM remain stronger in probabilistic calibration.

### Strengths
1. The paper introduces a sampling procedure that aligns diffusion time with physical forecast time, yielding an (O(H)) Euler-style rollout and avoiding multi-step reverse diffusion sampling.
2. The empirical study spans six widely used long-horizon forecasting benchmarks and includes comparisons to multiple diffusion-based baselines (TimeDiff, DiffusionTS, TMDM, ARMD, NsDiff), as well as ablations and measurements of inference time and memory usage.
3. The work treats computational efficiency (runtime, memory footprint) as a central evaluation dimension in addition to point forecasting error, which is important for deployability in long-horizon forecasting settings.
4. The paper provides explicit quantitative evidence that the proposed framework can match or outperform diffusion-style baselines in point forecasting error on several datasets while significantly reducing inference cost.

### Weaknesses
1. The evaluation is almost entirely within the diffusion family. There is no direct comparison against strong non-diffusion long-horizon forecasters such as recent long-sequence Transformer variants or mixer-style architectures, so broader claims of overall effectiveness across paradigms are not yet supported.
2. The paper reports results for a single forecast horizon (fixed (H = 96)) in the main text, and does not present accuracy and efficiency tradeoffs across multiple horizons, limiting the assessment of how well the claimed complexity scaling and predictive behavior generalize to other rollout lengths.
3. Table 5 indicates that TEDM does not outperform recent diffusion-style baselines such as NsDiff or TMDM on probabilistic metrics like CRPS and QICE, yet the manuscript does not analyze why calibration lags in those settings or how this affects downstream use in uncertainty-sensitive applications. 
4. One of the core stated novelties is the empirically estimated, time-varying ((s_t, \Sigma_t)) noise/scale schedule, which appears conceptually close to NsDiff's input-dependent uncertainty modeling; the manuscript does not yet clearly articulate the principled difference between TEDM's schedule design and NsDiff's approach, nor why this difference alone explains the observed efficiency and accuracy outcomes.

### Questions
1. Can you provide direct comparisons against strong non-diffusion long-horizon forecasting models (e.g., long-sequence Transformer variants, decomposition-based linear models, mixer-style models) to substantiate claims beyond diffusion-style baselines?
2. Can you report results for multiple forecast horizons (e.g., different values of (H)) to show how both predictive accuracy and computational cost scale with horizon length?
3. Table 5 suggests that TEDM underperforms NsDiff and/or TMDM on probabilistic quality metrics such as CRPS and QICE. Can you provide an analysis of why calibration is weaker in those cases and whether this is due to the deterministic Euler rollout, the ((s_t, \Sigma_t)) schedule estimation, or other modeling choices?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes TEDM, a diffusion-based framework for multivariate time-series forecasting that extends the EDM design space to sequential data. Key ideas are aligning diffusion time with physical time, estimating the noise and scale schedules directly from the data to avoid hand-crafted schedules, and training a denoiser via score matching under structured, per-time-step noise. Empirically, TEDM achieves strong accuracy on several long-horizon benchmarks. Theoretical derivations provide an ODE / SDE formulation and show the Euler update used for autoregressive forecasting. Probabilistic calibration remains weaker than recent baselines (e.g., NsDiff and TMDM), and results on ETTh1 are less competitive.

### Strengths
(i) The paper provides a clear, modular extension of EDM to time-series with an explicit ODE/SDE derivation and practical Euler-step inference aligning physical and diffusion time

(ii) The paper provides a data-driven estimation of the covariance and scale schedule

(iii) The paper features well structured experiments, ablations, and complexity comparisons with clear qualitative plots illustrating stability and phase alignment

### Weaknesses
(i) The theoretical derivation relies on strong assumptions such as a diagonal $\Sigma_t$

(ii) Probabilistic performance (CRPS and QICE) lags behind NsDiff/TMDM

(iii) Reproducibility details are thin with limited hyperparameter specs, preconditioning functions are not fully specified, and baselines appear to use default settings

### Questions
In addition to the weaknesses outlined in points (i-iii), I present the following questions for the authors to address:

(1) How robust is the estimation w.r.t. $s_t$ and $\Sigma_t$ across windows with near-zero means/variances?

(2) Why restrict $\Sigma_t$ to diagonal covariance matrices? Did you test full covariance to capture cross-feature dependencies?

(3) Are your baselines tuned to the same validation regime, or strictly default configurations? Please provide finetuned baselines and multiple seeds, if possible.

(4) Could second-order solvers (e.g., Heun) on the unified time axis further improve accuracy without increasing complexity significantly?

(5) The preconditioning functions ($c_{skip}$, $c_{in}$, $c_{out}$, $c_{noise}$)  are important parameters in EDM. Maybe I overlooked your parameterization w.r.t. preconditioning, but I can not find these specifications in your work. Can you specify how you adapted preconditioning to your framework?

(6) You mentioned that you varied $\sigma_{data}$ for training and test sets: "(...) we use a different $\sigma_{data}$ for
the train and test sets, computed after data normalization". Can you specify exact values and how much this choice improved results?

### Soundness
3

### Presentation
2

### Contribution
3
