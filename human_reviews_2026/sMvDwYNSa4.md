# NetBurst: Event-Centric Forecasting of Bursty, Intermittent Time Series

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Forecasting on widely used benchmark time series data (e.g., ETT, Electricity, Taxi, and Exchange Rate, etc.) has favored smooth, seasonal series, but network telemetry time series---traffic measurements at service, IP, or subnet granularity---are instead highly bursty and intermittent, with heavy-tailed bursts and highly variable inactive periods. These properties place the latter in the statistical regimes made famous and popularized more than 20 years ago by B.~Mandelbrot. Yet forecasting such time series with modern-day AI architectures remains underexplored. We introduce NetBurst, an event-centric framework that reformulates forecasting as predicting when bursts occur and how large they are, using quantile-based codebooks and dual autoregressors. Across large-scale sets of production network telemetry time series and compared to strong baselines, such as Chronos, NetBurst reduces Mean Average Scaled Error (MASE) by 13-605x on service-level time series while preserving burstiness and producing embeddings that cluster 5x more cleanly than Chronos. 
In effect, our work highlights the benefits that modern AI can reap from leveraging Mandelbrot's pioneering studies for forecasting in bursty, intermittent, and heavy-tailed regimes, where its operational value for high-stakes decision making is of paramount interest.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
### *Summary*

- This work focuses on network telemetry time series, which are highly bursty and intermittent. Existing models are not well adapted to this regime, such as Chronos which both ignores timestamps and uses uniform binning. The authors propose NetBurst, a model architecture which forecasts bursty time series forecasting as prediction of both the time gap between bursts, and burst intensity. The authors convert two existing telemetry datasets, PINOT and MAWI, as time series datasets, which they then convert to event datasets based on fixed activity thresholds. The authors ablate the importance of the quantile codebooks over uniform binning, and the relative importance of timing the burst vs predicting its intensity. The authors analyze clusters of the model embeddings.

### *Contributions*

- NetBurst, a model architecture that forecasts as event prediction  
- New time series datasets derived from existing telemetry data  
- Additional mentioned contributions: evaluating netburst on PINOT and MAWI, analyzing netburst embeddings through cluster analysis

### Strengths
### *Originality*

- Novel architecture, NetBurst  
- Novel time series datasets (through transformation of raw logs)

### *Quality*

- evaluates against multiple ML baselines  
- Includes ablation studies

### *Clarity*

- Clear paper structure  
- Clear figures

### *Significance*

- Properly motivates the importance of the statistical regime for applied problems, as well as the unique challenges of this statistical regime for the mentioned baselines

### Weaknesses
### *Originality*

- **Missing related work on bursty forecasting**: There is no mention of other burstiness-specific baselines in the paper. If this is indeed the case, then I’d mention the absence of any burstiness-adapted models in both the classical and recent ML literature, I would mention this explicitly, as it is an important point for both the novelty of the problem setting/method and for evaluating the relevance of the baselines that NetBurst is compared against.

### *Quality*

- **Event-specific MASE appropriateness:** Why is MASE on events the appropriate metric for bursty series with irregular seasonalities? This ignores false positives during long gaps and ignores time warps. The ablation helps understand this behaviour within your own model, but not for the other baselines. It seems like you’d need a pair of metrics that disentangles burst *timing* vs *size*, as your architecture does. If a sharp burst prediction is incorrectly timed, MASE will effectively punish this *even more* than predicting no burst at all, but evaluating MASE only on the events would ignore these false positives. Since you’re dealing with non-seasonal series, a metric that allows for some form of time distortion would provide more insight into *why* NetBurst does well.  
- **Missing bursty baselines:** The related work mentions burstiness-specific models (e.g. Hawkes Process models) and the baselines section mentions “classical” baselines, but there are no traditional/classical baselines compared against. Importantly, there are no models made specifically for bursty series or Hawkes processes, as these are dismissed around line 200 with little additional justification. This might be appropriate for very specialized venues that are familiar with this literature niche, but for a general ML conference, additional justification is warranted for dismissing seemingly relevant baselines. Some examples of simple baselines \+ bursty baselines:  
  - Repeating the mean value for the entire forecast window  
  - Repeating the last value for the entire forecast window  
  - Croston methods   
  - Some potentially relevant Hawkes-specific models that turned up from asking ChatGPT:  
    - SAHP  
    - Transformer Hawkes  
    - Graph Neural Hawkes

### *Clarity*

- It's unclear why you cannot just call this a model architecture, instead of a framework. The word “framework” has lost all meaning in this context. It’s about as imprecise as saying a “system”. What makes NetBurst a “framework”?
- Jargon in abstract: “service-level”  
- Figure 1 caption: I would spend more time walking the reader through the interpretation of the figure, and conveying the “why”, for example, what is the relevance of the Fano factor? Intuitively I know that variance/mean being bigger means more variability, but this is a (wasted) opportunity to reinforce the motivation. Furthermore, the introduction should point to this plot/section to back up its claims, as one of my first thoughts when reading the intro was “do they provide any evidence that telemetry data is different from ETT or Electricity?", to which the answer is "yes, see figure 1", so the text should point to this. Also, for (c), it’s not clear what the takeaway for the reader is supposed to be other than “this is hard to predict”: you might want to expand on which design choices of existing TS-FoMos make forecasting in this regime difficult (no timestamps, uniform binning), as this sets up your contributions.
- If you want to introduce your results table early, then the caption should point toward descriptions of the datasets being evaluated (4.1), otherwise I have no idea what the numbers are supposed to mean.
- No Gap between segments 120 and 121 is weird, please include some gap for legibility.

### *Significance*

- It’s unclear why Mandelbrot is brought up multiple times beyond one initial citation to foundational work.
- “We begin with the most challenging regime: service level time series at 100 ms resolution.” It’s unclear why this regime is the most challenging. Is this known in the literature? If so, please point to this. Otherwise, is this determined by the results? Then this should be made clear, otherwise the reasoning about the significance of results in this regime is circular.

### Questions
# What will change my score

My current score is a 5 (borderline reject, rounded to 4 by rating system). Satisfactory answers to the below questions can significantly change my opinion on the paper. Q1 and Q2 are crucial in my opinion as they pertain to originality, quality and significance: answering both of these questions would raise my score to 6, while additional answers to the other questions could raise the score to 8.

Q1: **How is evaluating MASE only on events** (as determined by a tuned threshold) **appropriate?** This ignores any false positives, and does not account for possible time warping. What does MASE look like across the whole series when comparing methods? Are there any false positives?

Q2: **Why are there no Hawkes process models, or traditional baselines?** Instead of ruling these out *before* evaluation, they should first be evaluated, *then* ruled out based on performance, with a potential explanation for why they don’t perform well. 

### *other*

- **What is performance sensitivity to the fixed activity threshold?** It’s fine to state it as a limitation, but to actually assess how important this limitation is would require a sensitivity analysis of performance to different thresholds.  
- **Why does Table 1 not include NetBurst?** Without this, it’s hard to make an apples-to-apples comparison on typical datasets. Does NetBurst still perform well on typical time series datasets?


- The Wasserstein distance (WD) is a distance between probability distributions. **What exactly are you measuring the WD between?**
- Why do you model inter-burst as a gap (i.e. a difference wrt the previous burst), but burst intensity as an absolute (independent of the previous burst)?

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
3

### Summary
The paper proposes viewing bursty data as events and modeling this instead of the full time series. They achieve this by first processing the data by converting it into events, tokenizing using quantiles and fitting a separate intensity and timing models. The ablation studies show how each part of the network influences the results. The authors compare to some common baselines and show good results.

### Strengths
Converting rare outlier or burst values into events is a very natural approach. The resulting model is elegant and sound. Most of the paper is clear and easy to understand. The choices of architecture and data processing are reasonable. Ablation studies answer most of the the questions I would have. The results are good.

### Weaknesses
It is disappointing that this model does not handle "dense" time series with bursts but is converting a time series to a sparse representation of events and modeling them as a temporal point process. Ideally, I would like to see a conventional time series model which is enriched with outliers or bursts viewed as events. The model would predict normally most of the time but also predict the bursts on top of that. Unless I'm missing something, this model is not doing that.

Formatting is bad. Figures and tables are too close to the text.

### Questions
- In my understanding you don't learn the quantile codebook? If not, codebook name is a bit confusing given VQ-VAE.
- Can you comment on learning all of this in real number space instead of using tokens? Taking into account that it's possible to do some kind of data processing which will flatten large values and it's also possible to learn heavy tailed distribution.

### Soundness
3

### Presentation
2

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
This paper introduces NETBURST, a novel event-centric framework designed specifically for forecasting bursty, intermittent, and heavy-tailed time series, a regime characteristic of network telemetry data but poorly handled by existing state-of-the-art models. Instead of treating the data as a continuous sequence, NETBURST reframes the problem as predicting a series of discrete events. It first "eventizes" the raw signal into two streams: Inter-Burst Gaps (IBG) for timing and Burst Intensities (BI) for magnitude. Each stream is then tokenized using distribution-aware quantile codebooks to preserve tail fidelity, and forecasted independently by dual autoregressive Transformer models. The final forecast is reconstructed by combining the predicted event streams. On large-scale production network datasets, NETBURST achieves orders-of-magnitude improvements in accuracy (MASE) over strong baselines like Chronos while preserving distributional integrity and producing more meaningful embeddings.

### Strengths
*   **Novelty:** The core strength of this paper is its fundamental rethinking of the forecasting problem for this specific data regime. The shift from a sequence-value regression paradigm to an event-prediction paradigm is a powerful and well-justified conceptual leap. By explicitly separating "when" a burst occurs (IBG) from "how large" it is (BI), the model directly addresses the entanglement of timing and magnitude that destabilizes conventional forecasters. This event-centric view is a significant and novel contribution.

*   **Empirical Results:** The paper presents some of the most dramatic performance improvements seen in recent forecasting literature. The orders-of-magnitude reduction in MASE (13-605x) compared to state-of-the-art models like Chronos and Lag-Llama on real-world network data is a massive and undeniable result (Figure 3, Table 3). This is not an incremental improvement but a complete change in performance tier, demonstrating that NETBURST solves a problem that prior models fundamentally cannot handle. The gains are consistent across multiple datasets and granularities.

*   **Domain Knowledge:** Every component of the NETBURST pipeline is well-motivated by the statistical properties of the target data, drawing inspiration from Mandelbrot's work on heavy-tailed distributions. The use of quantile tokenization to combat the tail-erasing effect of uniform binning is a critical and intelligent design choice. The dual autoregressive model directly mitigates error cascades between the distinct statistical processes of timing and magnitude. These choices are validated through excellent ablation studies (Figure 5) that clearly demonstrate the contribution of each component.

*   **Experimental Evaluation:** The evaluation goes far beyond simple point-forecast accuracy. The authors assess distributional fidelity using Wasserstein distance, showing that NETBURST preserves the crucial "burstiness" of the original data. Furthermore, the analysis of embedding quality (silhouette scores and t-SNE plots in Figure 6) is a key contribution, demonstrating that the event-centric representations are not only better for forecasting but are also more semantically meaningful and useful for downstream operational tasks like clustering. This focus on the practical utility of the model's outputs is a major strength.

### Weaknesses
The paper has significant issues, and the authors did not sufficiently discuss its potential weaknesses.

1.  **Narrow Applicability and Specialized Domain:** The NETBURST framework is explicitly and brilliantly designed for one specific, albeit important, statistical regime: bursty, intermittent, heavy-tailed time series. The paper does not provide any experiments or discussion on how the model would perform on the smooth, seasonal benchmarks (like ETT, Electricity) where conventional models excel. While this is not the paper's focus, it raises the question of whether NETBURST is a specialized tool or a general-purpose forecaster. Its core "eventization" step would likely be ineffective or even detrimental for continuous, non-sparse data.

2.  **Reliance on a Heuristic and Potentially Brittle "Eventization" Threshold:** The entire pipeline is predicated on the initial "eventization" step, which relies on a fixed activity threshold, `Tact`, to define what constitutes a "burst." This threshold is a critical hyperparameter that is set manually. The model's performance could be highly sensitive to this choice, and an improperly set threshold could lead to a complete failure to identify relevant events or, conversely, to treat noise as events. The paper does not provide a sensitivity analysis for `Tact` or discuss methods for setting it automatically, which could be a significant practical hurdle.

3.  **Modeling Bursts as Atomic Spikes:** The current model simplifies each burst into a single value (BI) representing its total intensity, which is then placed at the start of the burst during reconstruction. This ignores the internal structure, shape, and duration of the bursts themselves. For operational tasks that might depend on the profile of an event (e.g., distinguishing a short, intense spike from a longer, sustained period of high activity), this loss of information could be a significant limitation.

### Questions
I will appreciate if the authors could answer my following questions: 
*   **Question 1:** The eventization step is crucial and depends on the activity threshold `Tact`. How sensitive is the model's performance to the choice of this threshold, and how should a practitioner set this value for a new, unseen dataset? Could this threshold be learned or adapted?

*   **Question 2:** The paper's narrative focuses exclusively on bursty, intermittent data. How would you expect NETBURST to perform on standard smooth and seasonal benchmarks like the ETT or Electricity datasets? Would the eventization process fail, and does this imply that NETBURST is a specialized, rather than general-purpose, forecasting architecture?

*   **Question 3:** The Oracle analysis in Figure 5b is very insightful. It shows that timing is the dominant error source for sparse Service traces. Why do you think modern Transformer architectures struggle so much with predicting the timing (IBG) of these rare events, and does your work suggest a need for fundamentally different temporal modeling approaches beyond standard self-attention?

*   **Question 4:** By collapsing each burst into a single BI value, the model discards information about the burst's duration and internal shape. In what operational scenarios would this be a critical limitation, and how could the NETBURST framework be extended to model or generate richer, intra-burst dynamics?

*   **Question 5:** The dual autoregressive models for IBG and BI are trained independently. Is there any information lost by ignoring the potential statistical dependence between the timing of a burst and its magnitude (e.g., do longer idle periods tend to be followed by larger bursts)? Have you explored coupling the two models, for instance, by conditioning the BI prediction on the predicted IBG?

*   **Question 6:** Your work draws a powerful connection to Mandelbrot's studies. Beyond network telemetry, what other specific domains (e.g., in finance, climate science, or social networks) do you see as the most promising and immediate applications for the NETBURST framework, and what new challenges might arise in those contexts?

### Soundness
2

### Presentation
2

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
The authors highlight the challenges of forecasting network telemetry time series - which are highly bursty and intermittent. They propose an architecture NetBurst, an architecture that uses quantile-based codebooks and dual autoregressors that predict when bursts occur and how large they are, which are then combined into a forecast. Compared to other models, NetBurst reduces MASE significantly on service-level timeseries. The authors present ablation analyses on the model.

### Strengths
- I like the fact that the authors delved deep into a specific type of time series and collated datasets specific to this domain of time series. We need more such deep-dive analyses instead of generic time series benchmarks (which are also very useful)
- The authors motivate their work very well. The preliminary analyses that they present in Fig 1 with Fano factors, autocorrelation and local example is interesting. 
- The presented model is simple and builds on prior work (Chronos) with targeted modifications. Although it deserves better explanation, to my understanding, the model makes sense and doesn’t have any “fluff”.
-  I like the analyses such as “Do quantile codebooks improve fidelity over uniform binning?”. It shows that the authors attempted to deeply understand the model’s results.

### Weaknesses
- Many parts of the paper are unclear, all of which cloud my evaluation of the paper. I mention all these in the questions section. The only reason for my score of 6 is the presentation which is very underwhelming.

### Questions
- Are the time series foundation models (TSFMs) (Chronos, Lag-Llama) being finetuned on the given datasets? Does Table 1 present results of these models being finetuned (if it’s a TSFM)? I have the same question for all results presented in the paper. This can be made clear throughout the paper and also in the abstract.

- Suggestion: The authors state “Continuous-time point-process models capture sparsity but fail to handle heavy-tailed magnitudes and long-range dependence.”. References should be given for those readers who don’t know what continuous time point process models

- “our work is an instance of Mandelbrot meets AI.” I”ve not heard of that. Can you give references?

- The authors claim “Models either dilute bursts among zeros or learn shortcuts, such as predicting zero everywhere. This explains the misleadingly low errors of DeepAR on some sparse series—it minimizes loss by ignoring rare events entirely.” - can the authors give empirical evidence for this? I don’t see any reference to their experiments which gave them this conclusion.

-  “Table 1 illustrates this directly: despite pretraining, Chronos collapses on network telemetry data.” Do you pretrain Chronos from scratch, or finetune it?

- “we fix an activity threshold $T_{act}$ and declare a burst whenever consecutive windows exceed this threshold.” Is this threshold determined per dataset? Is this a hyperparameter? Or is it the same for all dataset?
Also this is a limitation of the framework that should be highlighted.

- If I'm right, the authors pre-process the time series into two time-series that have the inter-burst gap and the burst intensity.
Is this adaptable to other datasets? The framework depends on the data being pre-processed so it would be ideal to have the data pre-processing pipeline adaptable to new datasets.

- If the authors are indeed training all the model, I think it is worth adding zero-shot baselines without training to show how much performance you can get without training. This will help the reader understand why training is necessary to show results on this benchmark.

- As a side note, - I’m curious if event prediction baselines be more appropriate for this paper. I'm curious why the authors did or did not consider them.

### Soundness
3

### Presentation
2

### Contribution
3
