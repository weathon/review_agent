# ChaoticFuzz: Fuzzy-Based Graph Representation for Spatiotemporal Learning

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Chaos theory addresses a unique class of dynamical systems that, despite operating under deterministic rules, exhibit extreme sensitivity to initial conditions, where even minor perturbations can lead to vastly different trajectories. Rather than modeling such systems directly in the time domain, chaotic time series are often reconstructed in higher-dimensional phase spaces, where latent structures such as attractors and fixed points emerge. However, forecasting within this context remains challenging, as conventional time series models are ill-suited to capture the complex interplay between spatial and temporal dynamics. In this work, we propose ChaoticFuzz, a novel framework that transforms univariate chaotic time series into graph-structured data by leveraging fuzzy clustering in phase space. Our framework encodes temporal trajectories and fuzzy membership degrees to construct weighted graphs, enabling the application of Graph Neural Networks for accurate long-term prediction. This graph-based representation preserves essential spatiotemporal patterns that are often lost in traditional approaches. Experiments on several benchmark chaotic systems demonstrate that ChaoticFuzz significantly outperforms state-of-the-art methods, which reflect a model’s ability to follow complex dynamical behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper investigates forecasting in chaotic dynamical systems, which are known for their extreme sensitivity to initial conditions. 
It motivates the need for models capable of capturing complex temporal dependencies that traditional approaches such as ARIMA and RNNs often fail to represent. 
To address this, the authors propose a method that embeds time series data into a high-dimensional phase space and applies fuzzy clustering to uncover latent structures in the data.

From the resulting membership matrix, which encodes the degree of association between data points and clusters, a graph representation is constructed. 
This graph is then used as input to a Graph Neural Network to perform time series forecasting.

The paper reports that, somewhat surprisingly, recent models such as Chronos and TimesFM (2024) perform worse than an LSTM baseline. 
The reported RMSE values are consistently higher than the baseline and are not clearly emphasized, 
which raises questions about the robustness of the experimental setup and the tuning of comparison models.

### Strengths
the idea of using fuzzy clustering for time series is interesting

results for DTW seem in favour of the propose method.

### Weaknesses
the RMSE results are negative

### Questions
what are the limitations / tradeoffs of your method?

why chronos and timesFM (from 2024) perform worst than LSTM?

is possible to compare with other methods for chaotic systems?

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
3

### Summary
The authors propose the ChaoticFuzz method for univariate chaotic time series, which transforms the input into weighted graphs via phase-space reconstruction, and based on the similarity of fuzzy membership degrees, which encodes proximity and uncertainty. The extracted graphs are treated by attention-based GNN encoders for spatiotemporal forecasting and evaluated on chaotic univariate time series datasets against common time series forecasting baselines, showing competitive performance.

### Strengths
The paper is easy to follow, an important aspect for the overall clarity of the presented approach. The proposed method is heavily dependent on principles of fuzzy clustering, while combining phase-space reconstruction with graph structure learning, a relatively novel and unconventional approach in the fields of time series forecasting and spatiotemporal forecasting, which supports its originality. Results demonstrate that some recent time series foundation models are significantly outperformed by the proposed methods for particular cases.

### Weaknesses
Several weak points of the paper span poor positioning against related works in spatiotemporal forecasting and the limited scope of the presented experimental setup. 
More specifically:
1. **[Positioning against graph-based approaches]** There are several related works in extracting underlying graph structures from time series data, focusing on multivariate time series and their correlation (see baselines used in (Yi et al, 2023)). Methods include sparse graph structure learning and fully-connected graphs, with some approaches also applying to univariate time series. It is hard to position the paper in the time series spatiotemporal community since related methods and their limitations are not discussed.
2. **[Positioning against works on dynamical systems]** The chaotic univariate time series and non-linear dynamical systems are interconnected. While there are several related works in machine learning for dynamical systems and spatiotemporal forecasting [Li et al, 2020; Wang et al., 2022; Kissas et al., 2022], those are not discussed or experimentally evaluated in the presented results.
3. **[Scope of selected datasets]** The datasets selected for experiments do not represent a common benchmark in the time series forecasting field, while the baselines chosen have been proposed for standard time series datasets, multivariate in most cases, with some univariate examples as well (see datasets in Time-Series-Library: https://github.com/thuml/Time-Series-Library, M4 is univariate). Similarly, for dynamical systems and spatiotemporal forecasting, there are additional datasets used [Herzen et al., 2022] (see also datasets used in papers in point 2). Therefore, the significance of the contribution is limited experimentally to very few, assumption-constrained datasets. 
4. **[Application scope]** It seems that the application scope of the proposed method is limited to univariate and chaotic time series, and it remains unclear how it can generalize to real-world time series that are noisy, have correlated variables, multiple covariates, distribution shifts, and lack stationarity.
5. **[Significance of results and architectural design choice]** Performance improvements are, in several cases, incremental/missing compared to baselines. Additionally, no ablation studies are provided to justify the design choice/effect of basic modules of the proposed baselines, e.g., the graph structure module, the attention-based GNN, clustering loss, etc. 

- Yi, Kun, et al. "FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective." Advances in neural information processing systems 36 (2023): 69638-69660.
- Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
Wang, Rui, et al. "Koopman neural forecaster for time series with temporal distribution shifts." arXiv preprint arXiv:2210.03675 (2022).
- Kissas, Georgios, et al. "Learning operators with coupled attention." Journal of Machine Learning Research 23.215 (2022): 1-63.
- Herzen, Julien, et al. "Darts: User-friendly modern machine learning for time series." Journal of Machine Learning Research 23.124 (2022): 1-6.

### Questions
1. Can the authors better position themselves against related works in graph learning-based methods for time series forecasting and machine learning methods for dynamical systems?
2. Is the proposed method competitive for real-world time series datasets, beyond the selected simple chaotic systems?
3. Can the proposed method be extended to time series common challenges, such as non-stationarity, multi-dimensionality, additional correlations?
4. Can the authors experimentally justify the effect of different components of their proposed method for forecasting? 
5. Can alternative design choices for graph learning/clustering give similar performance?
6. Can the authors explain with examples the motivation behind the introduction of their architectural design?

### Soundness
2

### Presentation
2

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
This paper introduces ChaoticFuzz, a novel framework for forecasting chaotic time series by transforming univariate sequences into fuzzy graph representations. The approach reconstructs time series in phase space using Takens’ embedding theorem, applies Fuzzy C-Means clustering to assign soft memberships, and builds a weighted graph encoding both temporal proximity and uncertainty. The resulting graph is processed by Graph Neural Networks (GNNs) enhanced with a global attention mechanism to model long-range dependencies. Experiments on several benchmark chaotic systems demonstrate that ChaoticFuzz achieves lower Dynamic Time Warping (DTW) distances and competitive RMSEs compared to baselines like LSTM, GRU, PatchTST, Attraos, and TimesFM. The results suggest that fuzzy graph encoding preserves the complex spatiotemporal structure of chaotic dynamics better than conventional time-series architectures.

### Strengths
The paper makes a creative and well-motivated contribution by integrating principles from chaos theory, fuzzy systems, and GNNs into a unified framework. The use of fuzzy clustering in phase space to construct graph representations is elegant and interpretable, offering a principled way to capture uncertainty and nonlinear dependencies. The model’s design—particularly the phase-space encoder-encoder —is well thought out, combining spatial (graph-based) and temporal (LSTM + attention) components. Experimental results across multiple chaotic benchmarks are strong, with consistently lower DTW scores indicating superior alignment with true dynamics. Visualizations of phase-space trajectories and DTW warping paths convincingly demonstrate ChaoticFuzz’s ability to reproduce underlying attractor behavior, supporting the paper’s interpretability and robustness claims.

### Weaknesses
The paper lacks an in-depth complexity or scalability analysis—important since phase-space reconstruction, fuzzy clustering, and GNN inference can be computationally heavy for high-dimensional or multivariate systems. The comparisons to foundation models (TimesFM, CHRONOS) are somewhat superficial, as those models were used zero-shot without fine-tuning, limiting fairness. Additionally, while the use of DTW is insightful, the authors’ claim that it better captures “true dynamics” than RMSE could be further validated by quantitative measures of attractor fidelity (e.g., Lyapunov exponents, correlation dimension). Finally, the work focuses on synthetic benchmarks, and real-world chaotic datasets (e.g., weather, EEG, or finance) would strengthen the empirical impact.

### Questions
How does ChaoticFuzz perform on multivariate chaotic systems or real-world nonlinear datasets?

What is the computational complexity of the fuzzy clustering and GNN stages as a function of embedding dimension and cluster count?

How sensitive are results to the fuzzification coefficient m or the chosen number of clusters?

Can the learned graph adjacency matrix be interpreted physically (e.g., to identify attractor regions or transition probabilities)?

Would replacing fuzzy clustering with probabilistic or manifold-based clustering (e.g., Gaussian mixture models) yield similar benefits?

### Soundness
3

### Presentation
3

### Contribution
3
