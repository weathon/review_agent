# Mamba Unchained: A Permutation-Invariant Approach to Multivariate Time Series

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Time series data in domains such as climate science, finance, and biomedicine present a significant challenge for scalable modeling due to their multi-scale temporal patterns, complex inter-variable dependencies, and frequency-specific structures. While recent advances in architectures like Transformers and state space models (SSMs) have shown promise, they are often limited by either high computational costs or an inability to capture time-varying cross-variable interactions. To overcome these limitations, we propose a variable-invariant two-dimensional state space model that eliminates variable ordering dependence by leveraging a global, permutation-invariant descriptor to condition temporal dynamics. This design allows for the efficient processing of variable-axis updates through an effective pooling operation, which maintains global correlations and enables full parallelism. We further enhance this architecture with a multi-branch design that incorporates distinct pathways for long- and short-horizon temporal features and a dedicated frequency-domain pathway, all integrated via a lightweight gating mechanism. Through extensive experiments, our model consistently outperforms state-of-the-art baselines on forecasting, classification, and anomaly detection tasks. Comprehensive analysis confirms our model’s efficiency, robustness, and ability to capture diverse temporal–spectral patterns.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Mamba Unchained, a variable-invariant two-dimensional state space model (2D SSM) designed for multivariate time series analysis. Unlike conventional 1D or 2D SSMs that depend on variable ordering or sequential variable-axis scans, the proposed method introduces a global permutation-invariant pooling mechanism to model cross-variable dependencies. This design allows simultaneous and order-free variable-axis updates, improving both scalability and robustness. The model further adopts a multi-branch architecture incorporating long-term, short-term, and frequency-domain pathways, fused via an adaptive gating mechanism. Experiments across forecasting, classification, and anomaly detection tasks show model’s good performance.

### Strengths
1.The proposed variable-invariant 2D SSM fills a key gap in existing literature by resolving the artificial variable ordering dependence of conventional 2D SSMs (e.g., Chimera)；

2.The multi-branch framework (long/short-horizon temporal + frequency-domain pathways) is well-motivated;

3.The experiments cover diverse tasks (forecasting, classification, anomaly detection) and benchmarks.

### Weaknesses
1.The paper lacks an overall architectural diagram that describes each component of the model, as well as its inputs and outputs. Such a diagram is essential for readers to intuitively understand how the variable-invariant 2D SSM, multi-branch pathways (long-horizon temporal, short-horizon temporal, frequency-domain), and lightweight gating mechanism interact with one another, and how raw multivariate time series data flows through these modules to generate final predictions or representations;

2.It is unclear where the results of these baseline models are derived from—whether they are quoted from other existing literatures or obtained through the authors’ own experimental testing;

3.Temporal Convolutional Network (TCN) is a relatively old model architecture, yet the paper shows it achieves surprisingly strong performance—even outperforming more recent advanced models such as iTransformer and PatchTST. This unexpected result requires further explanation: are there specific modifications to the TCN (e.g., improved normalization, adjusted kernel size) in the experiments?

4.The proposed model does not outperform baseline models on tasks such as short-term forecasting and time series classification. For short-term forecasting on the M4 dataset, it only achieves the second-best performance (behind Chimera); for time series classification on UEA datasets, its average accuracy (74.4%) is slightly lower than that of Chimera (75.3%);

5.The description of the experimental setup is unclear, such as the length of the historical window and the number of epochs?

6.The paper does not provide code to verify reproducibility. While a GitHub link is mentioned in Section 5.1 (Footnote 1), the code and running script inside are incomplete;

7.Many formulas lack punctuation marks.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a variable-invariant two-dimensional state space model that eliminates dependency on variable ordering by regulating temporal dynamics through the use of globally permutation-invariant descriptors. Extensive experiments demonstrate that the proposed model consistently outperforms state-of-the-art benchmark models in prediction, classification, and anomaly detection tasks.

### Strengths
1. The study is rich in experimental content. The paper conducted a substantial number of qualitative and quantitative experiments to validate the effectiveness of the methodology.
2. The paper possesses a robust theoretical foundation, thereby providing theoretical validation of the method's interpretability.

### Weaknesses
1. Lack of comparison with relevant prior work. The abstract asserts that existing SSMs or Transformers suffer from high computational costs or fail to capture cross-variable interactions over time. However, the recent work TimePro [1] has addressed this issue. Notably, TimePro is also based on SSM, rendering it highly relevant to this paper. The authors should clarify distinctions from TimePro and include comparative experiments to validate the novelty and efficacy of their approach.

[1] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable-and Time-Aware Hyper-state [ICML'25]

2. Limited readability. The author should provide a schematic diagram of the model to enhance module details and improve the paper's readability.

3. Lack of efficiency metrics. The authors have not provided efficiency comparisons with relevant works such as Simba, Chimera, and TimePro within the paper. According to the Method section, the proposed approach incorporates multi-branching and Fourier transforms, which to some extent reduce its efficiency. The authors should furnish efficiency metrics including comparisons of parameters, FLOPs, and latency.

4. It is recommended that the optimal values in Tables 7 and 8 be bolded to facilitate readers' observation of the specific performance of the proposed method.

### Questions
Please refer to the weakness. I will raise my score if the author resolves my issues.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a permutation-invariant 2D state space model (2D SSM) for multivariate time series. Instead of sequentially scanning variables, it introduces a global permutation-invariant descriptor ψ(t) to jointly model temporal and variable dynamics, enabling full parallelism across variables. Built on this, a multi-branch Mamba architecture with long-/short-term and frequency-domain branches is designed, fused via lightweight gating. Experiments show superior performance, efficiency, and robustness to variable permutations across forecasting, classification, and anomaly detection tasks.

### Strengths
1. The problem is clearly defined, and the design is simple yet effective. By replacing variable-axis recursion with a permutation-invariant global aggregation, the method directly addresses the core challenge of “no natural variable order” in applying 2D SSMs to multivariate time series, while reducing 2D scanning to 1D temporal scanning plus parallel aggregation.

2. The multi-domain modeling design is well-motivated. Long-/short-term temporal branches capture multi-scale dynamics via different discretization steps, while the frequency-domain branch models spectral structures. These are adaptively fused through gating, achieving strong expressiveness with low implementation cost.

### Weaknesses
1. Replacing variable-axis scanning with a global aggregation is the core of this work. In the paper, what specific forms of ϕ (e.g., mean pooling, gating, or set-attention) are actually used as the default setting? How do different choices of ϕ affect GPU throughput and peak memory usage? Has a systematic ablation study been conducted?
2. The advantage is less evident in univariate or low-dimensional settings. On the M4 (single-channel) dataset, the method is suboptimal, and the authors acknowledge that the benefit of permutation invariance is limited in this case. It is suggested to more systematically characterize the performance transition as the number of variables C varies.
3. The rationale for choosing the frequency-domain branch hyperparameter Δf is unclear. Table 4 shows that a smaller Δf yields better results, but no adaptive mechanism or theoretical justification is provided, despite the large variation in spectral distributions across datasets.

### Questions
1. Does the permutation-invariant conditional independence assumption have any side effects? Are there local interactions among variables, such as pairwise or higher-order relationships, that might be disrupted by the parameter-sharing constraint imposed in the model? What would happen if the parameters were not shared? Have any related experiments been conducted?
2. The paper only compares the efficiency between the proposed global aggregation and the 2D-SSM baseline, without presenting a direct comparison of FLOPs and memory usage against existing methods such as Chimera. Could the authors provide a detailed mathematical derivation of the computational complexity?

### Soundness
2

### Presentation
3

### Contribution
3
