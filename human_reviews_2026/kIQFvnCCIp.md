# Balanced Scaling Using Nonlinear Dynamic Metrics in Multivariate Time Series Modeling

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Time-series foundation models have shown strong capability in tasks such as forecasting across diverse domains by leveraging informative waveform representations. The main challenge in building a generic multivariate time series model lies in adaptability and consistent pattern extraction across systems that differ in autocorrelation, sensitivity to initial conditions, and the complexity of their underlying dynamical structure, whether reflected in univariate or multivariate signals. Prior approaches often fall into two extremes: specialized models trained separately for individual systems or large-scale foundation models trained on heterogeneous collections of time series with limited dynamical grounding. Motivated by the Platonic Representation Hypothesis, we achieve a heuristic observation that models across domains tend to converge toward a shared representation space that encompasses systems expressible in time-series form, including systems governed by differential equations, canonical analytical functions, and stochastic processes. In this work, we introduce Pangu-TS, a Pre-trained modality Agnostic Network for Generic multivariate Time Series modeling. Pangu-TS is pre-trained on a benchmark dataset designed with a more balanced distribution of types of time series systems, quantified by several nonlinear dynamical metrics from chaotic theory. Through this analysis, we uncover an empirical balancing law, showing that maintaining representative distributions of dynamical systems is essential for controlling the patterns learned by the model. Alongside this, Pangu-TS demonstrates both strong zero-shot forecasting ability in real-world data and promising latent representation quality on various downstream tasks, as validated across benchmarks in fields of digital healthcare, battery life health, and civil monitoring.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a method to estimate the “chaos” of a set of datasets containing time series data, using three metrics (DFA, LE and PE). Then, the authors present a foundation model trained on a dataset collection curated according to their proposed metrics. From their results, the authors show that their time series foundation model outperforms two existing foundation models, Sundial and Panda.

### Strengths
- The authors use three existing metrics to estimate 'chaos balance' inside pre-training datasets for time series foundation models. They show (Fig 5 and 6) how a higher “chaos” (according to their metrics) leads to improved performance in generative tasks, when using their proposed foundation model. This research direction is interesting.
    
- The proposed foundation model outperforms two existing baselines, Sundial and Pandas, on the presented tasks.

### Weaknesses
Major:

- **W1** - Experimental evaluation could be improved:
     - From the experiment, it was not clear to me whether their foundation model architecture is indeed outperforming existing approaches due to novel methodological implementations or due to the data.
     - All reported results lack an estimate of experimental uncertainty, i.e., variability across random initializations with different seeds. This is critical to ensure significance of the results and all associated claims.
     - Comparison is limited to two baselines.

- **W2** - To me, the interesting aspect of the paper was the impact of the data balance score on a foundation model's performance. However, this is limited to a single ablation study, appearing more as an addendum than the focus of the paper. Overall, the paper scope, as presented, looks limited and a bit “over the place”. It’s not clear if the paper is more about the analysis of the chaos metrics or about proposing a foundation model (of which the novelty could be argued).
    
- **W3** - The methodological description in Section 2.1 is often unclear. In particular, from the text, it wasn't clear to me where Equation 4 comes from. Also, the discrete Fourier series is not appropriate for general time series data, and the authors should be better off starting from the continuous Fourier transform.
    
- **W4** - The authors base their foundation model on an existing architecture, which is, however, not peer-reviewed and, indeed, not accepted for publication upon review by peers ([https://openreview.net/forum?id=4x83oH6Oy6](https://openreview.net/forum?id=4x83oH6Oy6 "https://openreview.net/forum?id=4x83oh6oy6")).
 

Minor:

- Fig. 3 is very small, with many unnecessary elements and icons. Fig.4 is also very small.

### Questions
- **Q1** (related to W1) - Is the superiority of PANGU-TS w.r.t. PANDA statistically significant?

- **Q2** (related to W1) - Would the authors consider re-training other existing foundation models, using the same strategy and datasets, in order to provide more generalizability to their results?

- **Q3** - Could the authors elaborate on the advantages of "chaos balance" apart from improved accuracy?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces PANGU-TS, a multivariate time-series foundation model pretrained on a chaos-theoretic balanced dataset, inspired by the Platonic Representation Hypothesis. The authors argue that dataset balance—quantified via nonlinear dynamic metrics such as DFA, Lyapunov exponent, and Permutation entropy—is more critical than raw dataset scale for achieving generalizable temporal representations.
Extensive experiments across chaotic, wearable, battery, and infrastructure domains demonstrate strong zero-shot forecasting and representation transferability. The study’s novelty lies in applying chaos theory to dataset characterization and proposing that balanced dynamic diversity enhances pretraining quality.

### Strengths
- The paper introduces a fresh perspective by integrating chaos theory metrics into dataset evaluation, linking nonlinear dynamics with representation learning.
- PANGU-TS exhibits competitive or superior zero-shot forecasting compared to existing foundation models (e.g., Chronos, SUNDIAL, PANDA) across multiple real-world domains.
- The work provides early empirical evidence for the Platonic Representation Hypothesis, highlighting that qualitative balance may outweigh dataset size for robust pretraining.
- The authors detail their pretraining hyperparameters, masking strategies, and release plans for code and scripts, enhancing research transparency.
- The proposed framework bridges chaotic system modeling and AI foundation modeling, offering potential for wide application in scientific, industrial, and healthcare time-series settings.

### Weaknesses
- The *Platonic Representation Hypothesis* is explained vaguely. The paper does not clearly articulate how this philosophical concept operationally connects to model design or chaos metrics.
- The procedures for computing and combining chaos-based balance metrics (DFA, LE, PE) are insufficiently specified, making reproduction difficult.
- The claim that “balanced data matters more than sheer scale” lacks strong empirical justification. Figure 6 and related analyses require clearer numerical interpretation and stronger comparisons with large-scale baselines.
- Implementation specifics, such as the definition of the aggregation operator (∘), [MASK] token training dynamics, and Conv1D multivariate treatment, are underexplained.
- The paper overlooks potential biases and privacy risks when applying PANGU-TS to sensitive domains (such as healthcare) and provides minimal reflection on limitations or misuse.

### Questions
- How does the hypothesis translate into concrete architectural or training choices?
- What empirical or mathematical form does a “Platonic” representation take in time-series space?
- How was the weighting factor ($\alpha$) selected, and how sensitive are results to its value?
- Can you provide statistical evidence (e.g., correlation or p-values) supporting that balance, not scale, drives performance?
- What happens when model size or dataset volume is controlled while varying bthe alance?
- Could an ablation explicitly show performance variation with different balance vs. scale settings?
- How does Conv1D process multivariate inputs—independently per channel or through cross-channel fusion?

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
This paper introduces the Platonic representation hypothesis for the domain of multi-variance time series. Building on this hypothesis, the authors propose clustering data using metrics from chaotic theory to construct a more balanced multivariate time-series pre-training dataset. Based on this dataset, this paper introduces the PANGU-TS model, which achieves strong performance in zero-shot forecasting across diverse multivariate time-series scenarios, as well as on other downstream tasks.

### Strengths
To the best of my knowledge, this is the first work that attempts to establish a unified mathematical representation for multivariate time-series data in the context of time-series foundation models, and to propose a systematic data-mixing strategy based on it. 
The experimental results demonstrate that, in time-series settings, a principled data-mixing scheme can be more critical for improving model performance than simply scaling up the data volume.

### Weaknesses
1. The transition from the Platonic representation hypothesis to using chaotic metrics for data balance assessment is entirely heuristic. The paper does not clarify the relationship between these chaotic metrics and the coefficients in Equation (4). It remains unclear how the Platonic representation hypothesis concretely supports the subsequent methodological design.
2. The core methodology and assumptions of the paper, such as the generality of Equation (4), the selection of chaotic metrics, and the specific clustering and mixing strategies, lack theoretical justification, and the overall approach relies heavily on empirical validation.
3. However, the experimental evidence is not sufficiently solid, with several issues:
  a. Baseline selection: The paper does not compare against some newer and stronger univariate foundation models. The justification provided is unconvincing, especially since the referenced works only compare against earlier or smaller univariate models. In addition, the authors do not evaluate against NormWear, which shares a similar architecture with PANGU-TS.
  b. Comparison with PANDA: On the chaotic evaluation dataset, PANGU-TS does not outperform PANDA, suggesting that the proposed data-mixing strategy may sacrifice performance in domain-specific settings.
  c. Ablation studies: The ablations are simplistic and only demonstrate that mixing data from different sources yields improvements; they do not establish the effectiveness of the specific mixing scheme proposed in the paper.

### Questions
1. In Figure 3.A, the connection arrows among the Encoder, Lightweight Decoder, and Transformer appear somewhat unclear. What is the correct input–output flow among these components?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper argues that many time-series systems can be viewed as a composition of base signals plus timestamp-wise perturbations, and that balancing the distribution of dynamical types before pretraining improves generalization. Building on this, the authors propose PANGU-TS, a pretraining pipeline that quantifies dataset balance with nonlinear dynamical metrics (DFA, Lyapunov exponent, persistent entropy) and then trains a channel-aware MAE on a more balanced corpus.

### Strengths
1. Chaos-theory–based balance assessment: The use of DFA, Lyapunov exponent, and persistent entropy gives a clear diagnostic of “what kinds of dynamics” are present and how evenly they are covered, K-means with an elbow rule is used to visualize and quantify balance. 

2. Domain diversity: Pretraining/evaluation spans heterogeneous sources (e.g., wearable/health, chaotic systems, battery, civil monitoring), aligning with the paper’s goal of modality-agnostic time-series foundations.

### Weaknesses
1. Limited theoretical grounding for PRH. The Platonic Representation Hypothesis is presented as a motivating assumption rather than a theorem; formal identifiability/equivalence guarantees are not established. 

2. Training relies on MSE reconstruction only, there is no explicit spectral/topological or physics-consistency regularization, which could better preserve frequency-domain or structural properties. 

3. Forecasting/simulation is framed generatively via masking, but the paper provides limited analysis of error accumulation, mode collapse, or noise amplification under long rollouts.

4. While many hyperparameters are listed, broader reporting (multiple seeds/variances, error bars) could strengthen claims of robustness and replicability.

### Questions
1. If we mix architecturally different backbones (CNN-style, non-autoregressive transformers, state-space models), do the learned representations still converge to a common latent space? What are potential counterexamples or boundary conditions for PRH in time series? 

2. Have you tried scheduled sampling/teacher forcing or a denoising step during generation to mitigate error drift on long horizons? Any quantitative study on horizon length vs. degradation?

3. With total tokens/length/variates held constant, how much of the gain comes purely from balancing the dynamical mix vs. from adding more data? A strict controlled ablation would clarify causal impact.

### Soundness
2

### Presentation
2

### Contribution
2
