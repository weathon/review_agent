# Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting

- Decision: Reject
- Scores: 6, 8, 4, 4, 2, 4

## Abstract
In the era of increasingly complex AI models for time series forecasting, progress is often measured by marginal improvements on benchmark leaderboards. However, this approach suffers from a fundamental flaw: standard evaluation metrics conflate a model's performance with the data's intrinsic unpredictability. To address this pressing challenge, we introduce a novel, predictability-aligned diagnostic framework grounded in spectral coherence.
Our framework makes two primary contributions:  the **Spectral Coherence Predictability (SCP)**, a computationally efficient ($O(N\log N)$)  and task-aligned score that quantifies the inherent difficulty of a given forecasting instance, and the **Linear Utilization Ratio (LUR)**, a frequency-resolved diagnostic tool that precisely measures how effectively a model exploits the linearly predictable information within the data. We validate our framework's effectiveness and leverage it to reveal two core insights. First, we provide the first systematic evidence of "predictability drift'', demonstrating that a task's forecasting difficulty varies sharply over time.  Second, our evaluation reveals a key architectural trade-off: complex models are superior for low-predictability data, whereas linear models are highly effective on more predictable tasks. We advocate for a paradigm shift, moving beyond simplistic aggregate scores toward a more insightful, predictability-aware evaluation that fosters fairer model comparisons and a deeper understanding of model behavior. Codes and data are available at https://anonymous.4open.science/r/TS_Predictability-C8B7.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a forecasting evaluation framework built on two spectral diagnostics: SCP, which estimates how much of the future is linearly predictable from the past, and LUR, which measures how much of that predictable portion a given model actually captures. Experiments illustrate that predictable energy correlates with errors and that certain models better exploit specific bands, offering practical guidance for improving forecasters. The approach is computationally light and broadly applicable across models, making it a useful diagnostic layer atop standard forecasting benchmarks.

### Strengths
- Clear, theory-backed diagnostics.
- Model-agnostic and frequency-aware.
- Instance-level insight (predictable energy vs. error)
- Efficient: no extra training; few hyperparameters and straightforward implementation.

### Weaknesses
- Linear-only notion of predictability: The methodology defines “predictability” via linear coherence (SCP) and evaluates utilization via LUR. This assumes linear, LTI-style dependencies capture what matters. The paper does not establish when this proxy is sufficient. A good way to do this is probably demonstrate tightness of the linear lower bound against a strong linear forecaster, there has been many paper showing this in the past ([1](https://arxiv.org/abs/2205.13504)), ([2](https://arxiv.org/abs/2411.02796)), ([3](https://openreview.net/pdf?id=wfyc8vLcq0)).

- Equal-length constraint (history = horizon): The framework and experiments fix \(|x|=|y|=N\) to share a DFT grid. This is atypical in practice (commonly \(N_h \neq H\)) and may bias frequency resolution/weighting. Sensitivity or an extension to unequal lengths is missing.

- Boundary correction handles only mean jump: SCP subtracts a mean mismatch at the split but not slope/curvature changes. If a trend shift occurs at the boundary, the bound can over-attribute error to “unpredictable” power.

- Univariate treatment for inherently multivariate problems: Analyses are largely per-channel.

- Limited model coverage in key figures: Central plots (e.g., per-sample “predictable energy vs. error”) highlight a single baseline (DLinear). Generality claims are weaker without parallel plots for stronger nonlinear models.

- No direct evidence that spectral diagnostics beat time-domain diagnostics: Frequency-domain analysis is motivated but not compared against time-domain counterparts.

### Questions
- Does the framework and its conclusions go beyond forecasting? Can it extend to anomaly detection or tasks that do not assume a strict past→future order (e.g., retrospective labeling, two-sided contexts)?

- Similar approach also work on time-domain right? is there any intuition or direct evidence that working on spectral domain is better than that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce two new metrics, Spectral Coherence Predictability (SCP), and. Linear Utilization Ratio (LUR). This aims to provide more thorough understanding of the model and data relationship and to provided directions to optimize/tackle in the model for time series forecasting.

### Strengths
- The authors propose metrics which aim to reduce guesswork when diagnosing model performances, which is a welcome notion in the community.
- Overall, the paper is well written.

### Weaknesses
--

### Questions
- How well do the authors believe their metrics hold for time-series extraction tasks, not forecasting? For example, in the rPPG domain. wherein the PPG time series is extracted from facial/skin video?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the issue of predictability in time-series forecasting, which aims to revolutionize the evaluation of time-series models. It proposes SCP for measuring difficulty of a given forecasting instance and LUR for measuring how effectively a model exploits the linearly predictable information within the data. Experiments are performed on public time-series datasets to verify the proposed claims

### Strengths
1. This paper investigates the evaluation of time-series forecasting, which is an interesting perspective in this field.

2. This paper provides a detailed algorithm table, which is helpful for readers to understand the flow of the algorithm.

3. The paper is written well with clear logic flows.

### Weaknesses
1. Authors note that `standard evaluation metrics conflate a model’s performance with the data’s intrinsic unpredictability`, which seems to be an important motivation of this work. However, to my view, this conflation does not impede the validity of existing metrics for evaluation. Notably, the data’s unpredictability seems to be fixed, so the rank of the standard metrics could reflect the rank of model’s performance despite the interference of data’s unpredictability.

2. Eq. 3 in preliminary seems to be $R^2$ ，which is a widespread metric; it would be not very necessary to introduce it with large volume of texts, a concise definition is sufficient.

3. The squared coherence in Eq.5 is an established metric. For example, it is included in the term `coherence (sometimes called magnitude-squared coherence)` in Wikipedia. However, no reference is provided but it is necessary to clarify the contribution of existing studies.

4. Rigorous theoretical evidence is necessary but lacking to demonstrate the benefits of the proposed metrics over the state-of-the-art standard evaluation metrics. The theoretical discussion in lines 211-215 is good, but it only concerns with the property of the proposed metric, without evidence on the advantage of the proposed metrics over the state-of-the-art standard evaluation metrics even on certain scenarios.

5. The proposed metrics could be linear, i.e., there would be no guarantee that they could be effective for non-linear datasets or models, which is a critical limitation in the deep learning age. 

6. The dataset coverage is not sufficient. Currently there are only three types of datasets: ETT (the 4 subsets are often merged), ECL, Weather. As a evaluation-centered paper, diversity of datasets in terms of sample number, complexity, context, variable number, etc, should be critically considered in experiment design.

### Questions
Please see the weaknesses above

### Soundness
3

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
4

### Summary
This paper argues that current forecasting metrics didn't distinguish the influence of model inadequacy and intrinsic data unpredictability. Based on this, they proposed a unified framework which contains Spectral Coherence Predictability (SCP) and Linear Utilization Ratio (LUR). This framework helps to show the predictability drift and an architectural trade-off.

### Strengths
1: This paper shows a wealth of visualizations including different types of figures and tables, which are easy to understand.

2: This paper clearly make irreducible error explicit from the model performance and address evaluation issues inside this.

3: Their new designed SCP and LUR offers insights into different diagnostics, showing the evidence of predicability shift.

### Weaknesses
1: From this paper's implementations, it seems that model's implementation depends on several factors like window length, type, ridge and second-order stationarity within windows. However, we didn't see enough sensitivity analysis or ablation study on these choices.

2: In this paper's methods and experiments, the current formulation seems all focuses on linear coherence. How about any multivariate scenarios? How about any conditional coherence scenarios? How's your framework extend to these scenarios? More explanations or discussions about this should be mentioned.

3: For SCP, it seems that it is derived under the assumptions that the related time series is locally stationary and it is linear. However, in real-world forecasting, the scenario is more like a nonlinear or nonstationary process. So these assumptions may not hold. When the signal contains abrupt shifts or external factors, SCP may overestimate or underestimate the real case. How your SCP adapt to the real-world forecasting?

4: For LUR, it seems that it has a strong dependence on frequency-band partitioning. LUR requires to divided into several frequency bands and then compute based on each band. Different band boundaries may lead to different LUR values. So, your results may reflect how the frequency bands were chosen rather than intrinsic model behavior.

### Questions
Most questions are listed in the Weakness.

Beside them, more questions: \
1: Can your SCP/LUR be adapted to irregular timestamps or missing segments?

2: How sensitive is SCP to the spectral estimation setup?

3: Is there a statistical metric or method to summarize LUR performance in more details?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new diagnostic framework for time series forecasting that separates model performance from the data’s inherent predictability. The authors introduce two key metrics: Spectral Coherence Predictability (SCP) and Linear Utilization Ratio (LUR). Experiments show that forecasting difficulty can drift over time (“predictability drift”) and that complex models perform better on less predictable data, while linear models excel on more predictable tasks.

### Strengths
1. The technical methodology is clearly presented, with detailed pseudocode provided. The released code demonstrates strong reproducibility.

2. The conclusion that complex models perform better on less predictable data, while linear models excel on more predictable tasks is insightful and provides valuable understanding of model behavior.

### Weaknesses
1. The scientific and engineering contributions of the current work are limited. The authors mainly apply two existing statistical methods to analyze six small-scale time series datasets and five baseline models. Such a limited experimental setup is insufficient to support the claimed conclusions. More comprehensive and convincing experiments are needed.

2. The paper lacks strong case study visualizations to illustrate how SCP and LUR analyze time series data in practice. Their characteristics are only reflected through numerical metrics, which contradicts the authors’ stated motivation of moving beyond simplistic aggregate scores.

3. The authors fix the historical look-back window to ensure experimental fairness. However, different models exhibit varying sensitivity to the look-back length, which is typically treated as a tunable hyperparameter in time series forecasting. It is recommended that the authors evaluate multiple look-back window lengths for the same prediction horizon to fully and fairly assess each model’s potential performance.

### Questions
1. The paper lacks case study visualizations and analyses of real forecasting scenarios. It remains unclear under what circumstances the annalysis of SCP and LUR actually helps models achieve better predictive performance.

2. The proposed SCP and LUR metrics should be more thoroughly discussed in relation to the traditional and widely used autocorrelation function (ACF) analysis in time series forecasting. The authors are encouraged to clarify both the differences and connections between these approaches.

3. The concluding discussion on model design is insufficiently developed. The authors should elaborate on what constitutes a “complex” model versus a “linear” model, and further analyze the Pareto frontier between these two categories for the given datasets.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a spectral-coherence–based evaluation framework for time-series forecasting that (i) estimates a linear MSE lower bound (SCP) aligned with the task loss and (ii) computes a Linear Utilization Ratio (LUR) to diagnose how much linearly explainable signal a model actually captures, with band-wise (frequency) breakdowns. Experiments on synthetic and real datasets show good calibration of the bound, reveal predictability drift over time, and uncover architecture–difficulty complementarity when metrics are stratified by predictability.

### Strengths
1. The evaluation framework is task-aligned and scalable: the lower bound matches the MSE objective and its time complexity is O(NlogN), enabling large-scale use.
2. The diagnostic perspective is sufficient. LUR with band-wise views provides actionable insights beyond aggregate MSE/MAE.
3. The empirical study is clear. Convincing synthetic and real-world evidence is provided , including per-bucket analyses that expose capability differences obscured by averages
4. In practical relevance, the framework can guide model selection, curriculum design, and evaluation protocols.

### Weaknesses
1. Assumption scope: Reliance on (local) stationarity and second-order structure; tightness under strong nonstationarity/nonlinearity/exogenous drivers is under-characterized.
2. Hyperparameter sensitivity: Limited analysis of Welch/windowing choices, band partitioning, and associated uncertainty.
3. Multivariate extension: Joint, multichannel formulations (partial coherence, coherence matrices) are left for future work.

### Questions
1. Can the method support adaptive windowing/change-point handling with uncertainty estimates?
2. Please report systematic sweeps over Welch/window parameters and provide bootstrap CIs for predictability and LUR.
3. On controlled processes, how do SCP/LUR correlate with MI/entropy-rate surrogates versus compute cost?

### Soundness
3

### Presentation
3

### Contribution
3
