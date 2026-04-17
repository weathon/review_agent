# PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. 
PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance on the evaluated benchmarks with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting.
Code is available at this repository: https://github.com/neumyor/PhaseFormer_TSL

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PhaseFormer, a lightweight Transformer architecture for time series forecasting that replaces conventional patch-based tokenization with a novel phase-based tokenization paradigm. Instead of segmenting sequences into contiguous time windows, the model aligns values at identical offsets across cycles (phases) to form compact and stable phase tokens. A lightweight phase-based routing Transformer is introduced to enable efficient cross-phase interaction via a two-stage "phase-to-router" and "router-to-phase" attention mechanism. Experiments on seven benchmark datasets (ETTh1/2, ETTm1/2, Weather, Electricity, Traffic) show that PhaseFormer achieves comparable or superior accuracy to existing patch-based and frequency-based models.

### Strengths
1. Conceptual novelty: introduces a new phase-based view for modeling periodic time series, complementing the patch and frequency paradigms.
  2. Efficiency and scalability: achieves over 99.9% parameter and FLOP reduction without losing accuracy.
  3. Clarity of presentation: writing, diagrams, and methodology are clear and logically connected.

### Weaknesses
1. Representational limitation: phase tokenization compresses intra-cycle information; it remains unclear whether this sacrifices fine-grained dynamics.
  2. Incremental architecture: routing attention is inspired by Perceiver-style cross-attention; architectural novelty mainly lies in its adaptation to phase-level modeling.
  3. Strong periodicity assumption: the model assumes a fixed, known period; performance on irregular or multi-periodic data is not analyzed.

### Questions
1. The paper does not include a comparison against CycleNet[1], which also explicitly models periodic patterns to enhance forecasting performance.
  2. How sensitive is PhaseFormer to inaccurate or variable period lengths? Could adaptive or learned phase estimation improve robustness on datasets with weak periodicity?
  3. Have the authors examined whether phase tokenization retains sufficient intra-cycle information? For instance, reconstruction experiments or mutual information analysis could verify that the compressed phase embeddings still capture critical fine-grained temporal cues.

[1] Lin S, Lin W, Hu X, et al. Cyclenet: Enhancing time series forecasting through modeling periodic patterns[J]. Advances in Neural Information Processing Systems, 2024, 37: 106315-106345

### Soundness
3

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
4

### Summary
This paper proposes PhaseFormer, a phase-centric framework for long-term time-series forecasting. Instead of patch tokens, the method aligns values at identical offsets across cycles to form phase tokens, argues (empirically + theoretically) that phase tokens are more stationary and lower-dimensional than patch tokens, and introduces a lightweight cross-phase routing mechanism (phase→router aggregation, router→phase distribution) plus a shared predictor. Experiments on 7 benchmarks report strong accuracy with ~1k parameters and very large FLOPs reductions versus patch-based Transformers, with an explicit complexity analysis and ablations.

### Strengths
The paper motivates phase tokens via t-SNE/MMD/PCA analyses showing lower drift and dimensionality than patch tokens, and formalizes stability under cycle perturbations (Theorem 1). This gives a principled basis for efficiency and generalization.  The cross-phase routing layer (few routers, shared predictor) is easy to implement and yields large efficiency gains; on Traffic, FLOPs are reduced by ≈99.99% vs PatchTST/Crossformer while improving error.

### Weaknesses
1. While TimeMixer is included, PatchMLP  is not in the main tables. Given the paper’s efficiency narrative, a direct comparison would strengthen claims. Please report PatchMLP with the same look-back (720), horizons, and official settings. 
2. The main table shows FITS slightly outperforming PhaseFormer on ETTh2 (e.g., 0.334/0.382 vs 0.346/0.388 for MSE/MAE), and PatchTST essentially ties on ETTm2. Discuss failure modes and whether non-strictly periodic signals undermine the phase assumption. 
3. L_phase is chosen via frequency-domain/autocorrelation analysis and inputs are circularly padded to multiples of L_phase. Please provide a sensitivity study to period mis-specification and show robustness when cycles drift or are multi-periodic (e.g., Traffic/Electricity), as well as clarify any risk that circular padding induces boundary artifacts for long horizons.
4. The method follows a channel-independent paradigm (treating each variable separately). For multivariate datasets with strong cross-variable dependencies, can PhaseFormer leverage cross-variable structure beyond phase-wise routing? A small study contrasting channel-independent vs channel-dependent variants would help.

### Questions
As in weaknesses.

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
4

### Summary
This paper proposes PhaseFormer, a highly efficient Transformer architecture for long-term time series forecasting.
Instead of modeling local patch tokens, the authors introduce a novel phase tokenization mechanism that reorganizes the input sequence into a phase across period structure, capturing cross-cycle consistency.
A lightweight Cross-Phase Routing Layer with two submodules—Phase to Router (aggregation) and Router to Phase (distribution)—enables global information exchange with only a few learnable routers.
The authors also provide a theoretical justification via the Phase Tokenization Stability Theorem, arguing that phase representations are more stable and low-dimensional under periodic perturbations.
Experiments on seven benchmarks show that PhaseFormer achieves comparable or better accuracy than SOTA models while reducing parameters and FLOPs by orders of magnitude.

### Strengths
1. The idea of aligning data across cycles (phase tokenization) directly targets the instability of patch-based representations. Both theoretical and empirical analyses (PCA, MMD, t-SNE) support the claim that phase tokens form a more stable, low-dimensional subspace.
2. PhaseFormer drastically reduces parameters and FLOPs (up to 99.9% less than PatchTST) while maintaining or even improving forecasting accuracy across multiple datasets.
3. The Cross-Phase Routing Layer elegantly combines efficiency and expressiveness. Router attention visualizations reveal interpretable phase dependencies.
4. Extensive comparisons and ablations confirm the effectiveness of both phase tokenization and the router mechanism.

### Weaknesses
1. The theory and architecture depend on locally stable periodicity. The method’s behavior on weakly periodic or nonperiodic series is not fully tested, though acknowledged as a limitation.
2. The paper mentions autocorrelation-based detection but does not fully describe how multiple or unstable periods are handled. Automatic phase-length estimation and failure cases should be detailed.

### Questions
1. The paper mentions that phase tokenization depends on the cycle length $L_{phase}$, yet the main text only briefly states that it is detected via autocorrelation. Could the authors clarify how $L_{phase}$ is determined for each dataset and whether it is fixed or adaptive during training?
2. Table 5 shows the number of routers $M$ affects performance. Could the authors provide a short discussion or guideline on selecting $M$? For example, is there a trade-off pattern across datasets?
3. In the Phase Tokenization Stability Theorem, the perturbation analysis is central. Could the authors clarify whether this theorem assumes a fixed period or allows mild variations? This would help readers understand the scope of applicability.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper continues a line of research on highly parameter-efficient univariate long-term forecasting for time series with a dominant periodicity. It establishes a new best performance among these parameter-efficient models on standard baselines. Additional analysis is provided on the role of different approaches for reshaping time series into tokens.

### Strengths
- The model appears to improve forecasting performance relative to models of comparable size in the given domain.
- The architecture used is substantially novel and demonstrates useful approaches for minimizing parameter usage in these kinds of models.

### Weaknesses
- The comparison to parameter-efficient models is up to date but the general comparison to time series forecasting models is not, with the most recent evaluated large models being published in May 2024. (See [1] for examples of more recent models.) While I don't think this comparison is crucial for the paper, claims relating to the state of the art should be amended to limit the scope to parameter-efficient models.
- It's not clear how the statistical model in Theorem 1 relates to periodic signal forecasting. For instance, the data is modelled as a low rank matrix with additive noise, and it's not explained how that relates to a univariate time series with a dominant periodicity.
- The practical relevance of the approach seems limited, since it relates to univariate forecasting on signals with a known dominant periodicity where extremely low parameter counts are needed. Prior works establishing that this was possible with reasonable performance were an interesting result, but that's well-established now, and it's not clear to me that incremental improvements to performance in this regime are impactful. The reliance on a small number of very well-studied benchmarks also raises the possibility of overfitting.
- Important details on hyperparameter tuning and model variants used for experiments are not given, making it difficult to evaluate the soundness of experiments - see Questions.

[1] GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation https://huggingface.co/spaces/Salesforce/GIFT-Eval

### Questions
- The code shows significantly different hyperparameters being used for different datasets and horizons, but hyperparameter tuning is not discussed in the paper. How was hyperparameter tuning done?
- Were hyperparameters tuned for the baseline models? If not, were dataset-specific hyperparameters used or were they kept the same for all datasets and horizons?
- Are different model sizes used for different horizons? This seems to be the case in the code but the horizon is not given for Figure 1.b) and Figure 4.
- Can you explain how the terms used in Theorem 1 specifically relate to a time series signal and time series operations on it?

### Soundness
2

### Presentation
2

### Contribution
2
