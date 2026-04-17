# When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Recent efforts have extended the foundation model paradigm from natural language to time series, raising expectations that pre-trained time-series foundation models generalize well across downstream tasks. In this work, we focus on time-series anomaly detection, in which time-series foundation models detect anomalies based on the reconstruction or forecasting error. Specifically, we critically examine the performance of five popular families of time-series foundation models: MOMENT, Chronos, TimesFM, Time-MoE, and TSPulse. We find that for each model family using varying model sizes and context window lengths, anomaly detection performance does not significantly differ to simple one-liner baselines: moving-window variance and squared-difference. These findings suggest that the key assumptions underlying reconstruction-based and forecasting-based methodologies for time-series anomaly detection are not satisfied for time-series foundation models: anomalies are not consistently harder to reconstruct or forecast. The results suggest that current approaches for leveraging foundation models in anomaly detection are insufficient. Building upon our insights, we propose alternative directions to effectively detect anomalies using foundation models, thereby unlocking their full potential for time-series anomaly detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates whether TSFMs deliver reliable gains for TSAD compared with simple statistical baselines. The study evaluates reconstruction- and prediction-based TSFMs across multiple window sizes and reporting metrics, and finds limited or inconsistent advantages over one-line baselines.

### Strengths
The question is timely and practically important, given the rapid adoption of TSFMs and the community’s push for standard, reproducible TSAD evaluations. The experimental coverage across representative TSFMs, scoring paradigms, and window settings provides useful negative results and actionable insights.

### Weaknesses
- Regarding the core claim that TSFMs are not significantly better than statistical baselines, the paper relies on averaged F1/AUC-ROC without formal statistical testing.
- Regarding methodological transparency and practical guidance, it lacks of parameter counts, FLOPs, peak memory, and energy (GPU-hours), and plot accuracy versus cost to justify when TSFMs are warranted over one-line baselines.
- Regarding the explanation that prediction-based TSFMs underperform due to single-step horizons leveraging local anomaly leakage, the paper may need an ablation over the forecasting horizon H. It will be beneficial to plot anomaly-detection metrics as H increases from 1 to L/4 and L/2 (with L the window length) to verify if longer horizons better separate anomalous regimes.
- Minor comments:
    - typo “detecte anomalies” → “detect anomalies”
    - typo “requires” → “require” (line 470)

### Questions
Since density-based TSAD approaches [1–3] represent another important direction in anomaly detection, could density scoring offer a complementary mechanism for TSFMs?

[1] Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. ICLR 2022.

[2] Detecting Multivariate Time Series Anomalies with Zero Known Label. AAAI 2023.

[3] CaPulse: Detecting Anomalies by Tuning in to the Causal Rhythms of Time Series. 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper critically evaluates five families of time-series foundation models (TSFMs)—MOMENT, Chronos, TimesFM, Time-MoE, and TSPulse—for time-series anomaly detection (TSAD). The central finding is that, across model sizes and window lengths, TSFM-based anomaly scores (reconstruction/forecast errors) often perform on par with simple “one-liner” baselines (moving-window variance and squared difference), challenging the assumption that anomalies are generally harder to reconstruct or forecast. The paper then proposes practical fixes: ensembling multiple forecasting horizons, detecting anomalies in hidden representations, and modest end-to-end fine-tuning with labeled anomalies.

### Strengths
- Originality: Moves beyond leaderboard reporting to interrogate the mechanism behind TSFM-for-TSAD, revealing why reconstruction/forecast error can collapse to simple variance/neighbor baselines.
- Quality: Evaluates five prominent TSFM families under consistent setups (normalization, tokenization, architecture choices summarized), with direct, like-for-like baseline comparisons.
- Clarity: Effective visualizations diagnose single- vs long-horizon failures and show how representation-space detection helps.
- Significance: Practical guidance—multi-horizon ensembles, embedding-space detectors, light supervised alignment—can immediately influence TSAD deployments.

### Weaknesses
1. Data/setting coverage is limited. Experiments focus on univariate benchmarks; richer multivariate and domain-diverse evaluations (industrial, finance, sensors) would strengthen external validity.
2. Scope is narrow. The study centers on TSFMs vs. simple baselines; a fuller comparison against strong classical/modern TSAD methods (e.g., change-point, matrix decomposition, graph-based multivariate models) under matched conditions would be informative.
3. Statistical rigor. Many insights are qualitative; add paired tests, effect sizes, and multiple-comparison controls for key claims.

### Questions
1. Normalization sensitivity: If replacing z-scaling/sliding-window normalization with RevIN or sequence-level scaling, do the “TSFMs ≈ one-liners” findings materially change?
2. Horizon selection: Have you tried uncertainty-aware or autocorrelation-aware adaptive horizon selection instead of max-over-h ensembling? How does it trade off early detection vs false alarms?
3. Label granularity: If evaluated with event-level metrics (e.g., tolerant windows) instead of point-level, do core conclusions persist?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper critically evaluates the effectiveness of time-series foundation models (TSFMs) for anomaly detection, examining five popular model families: MOMENT, Chronos, TimesFM, Time-MoE, and TSPulse. The authors' main finding is that across different model sizes and context window lengths, these TSFMs perform no better than simple one-liner baselines—moving-window variance for reconstruction-based methods and squared-difference for forecasting-based methods. The results reveal that the fundamental assumption underlying these approaches—that anomalies are harder to reconstruct or forecast—does not hold for current TSFMs. The paper contributes a comprehensive empirical analysis demonstrating this limitation, emphasizes the importance of visualizing data and algorithm outputs rather than relying solely on aggregate metrics, and proposes three alternative directions for future research: combining multiple forecasting horizons, leveraging hidden representations for anomaly detection, and performing end-to-end fine-tuning with labeled anomalies.

### Strengths
1. very important question to answer
2. a lot of experiments on many TSFMs, covering both reconstructing and forecasting models
3. agree that visualization in anomaly detection is important
4. good, detailed analyses

### Weaknesses
1. Focus only on univariate time series. TBH, I don't agree with the argument in footnote 1 in page 2 because univariate is easier (so the baseline methods might work on univariate but not on multivariate) and TSFMs are trained on multivariate time series.
2. Need to see tabularized results to compare baselines against TSFMs to be more convincing. (seems like TSFMs are still slightly better than baselines)
3. Some of the TSFMs are not trained for anomaly detection, so comparison is not fair

### Questions
1. Could you clarify the conventions used in the box plots? Specifically, what do the solid and dashed lines represent—for instance, are they the mean and median?
2. On small-scale univariate time series, a specialized baseline method may perform well, but identifying the correct baseline often requires data-specific pre-analysis. Are you able to identify without pre-analysis? The value proposition of a FM lies not in outperforming a perfectly tuned baseline, but in its ability to be applied universally to any dataset and achieve robust, reasonable results without such data-specific selection.
3. It is worth testing if combining TSFMs with baseline methods improves performance. A lack of improvement would indicate that the TSFM may not be offering additive predictive power beyond the baselines

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates why time-series foundation models (TSFMs) underperform on time-series anomaly detection (TSAD) when used with standard error-based pipelines. The core empirical finding is that, for reconstruction-based use (e.g., MOMENT), sliding-window z-normalization makes the final anomaly score a moving-variance detector, since the normalized-space MSE is nearly constant across windows. For forecasting-based use, the paper shows single-step horizons can miss sequence-type anomalies. Across several TSFMs and baselines, simple one-line methods (moving variance, last-value/centered mean predictors) are competitive. The paper argues for (i) representation-space anomaly detection instead of error-based scoring, and (ii) more appropriate horizon settings.

### Strengths
- **Clear mechanistic insight**: neat derivation tying de-normalization to $(\sigma_i^2)$ and explaining the empirical collapse to variance detection.
- **Diagnosis** of a widely used pipeline (sliding z-norm + error scoring).
- **Useful negative results** that recalibrate expectations vs. simple baselines.
- **Horizon analysis** highlighting why single-step forecasting misses sequence anomalies.
- **Constructive direction** toward **representation-space** anomaly scoring.

### Weaknesses
- **Over-reach in generality**: conclusions for MOMENT/z-norm are at times presented as TSFM-wide without parallel derivations or tests for TimesFM/Time-MoE/Chronos/others.
- **Fairness of scope**: several forecasting models did **not** originally claim TSAD capability; the paper risks attributing pipeline failures to **model families**.
- **Framing**: occasionally reads as a critique of **TSFMs per se**, while the evidence mainly indicts the **error-based TSAD pipeline**.

### Questions
1. **Normalization ablations**: What happens under RevIN, MinMax, and global (non-sliding) normalization for MOMENT and at least one forecasting TSFM? Does the variance-collapse remain?
2. **Model-wise derivations**: Can you provide an analog of the de-normalization argument for TimesFM/Time-MoE (z-scaling) or explain decisively why it does/doesn’t apply?    
3. **Chronos/mean-scaling**: If Chronos underperforms, is it for the **same** reason? Please dissect with controlled experiments.
4. **Representation-space TSAD**: Present systematic results (kNN/density/one-class SVM) across multiple TSFMs/datasets, not just point cases—does this consistently beat one-liners?

### Soundness
2

### Presentation
3

### Contribution
2
