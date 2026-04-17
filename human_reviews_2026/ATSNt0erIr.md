# Learn Bullish Moves via EigenCluster Tokens

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Conventional tokenization schemes, such as point-wise and patch-wise methods, are poorly suited for financial time series data due to excessive token counts, sparse distributions, and heightened out-of-vocabulary risks---an issue not explicitly addressed in prior work. This paper introduces a novel tokenization approach for financial time series. By clustering scalar projections of eigenvectors from multi-window Open-High-Low-Close (OHLC) price matrices, our method generates compact and semantically meaningful tokens, enabling Transformer-based models to effectively identify next-day close price increase patterns. Extensive experiments on S\&P 500 and CSI 300 datasets show our approach outperforms market baselines by 6--9\% in precision, while reducing token vocabulary size to 51--101 tokens and sequence length by 75\% versus point-wise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EigenCluster Tokens, a clustering-based tokenization method for financial time series that uses eigendecomposition and multi-scale analysis to generate compact, meaningful tokens. It enables Transformers to predict bullish movements with 6–9% higher precision than market baselines, reduces token sequence length by 75%, and achieves superior efficiency and real-world trading performance.

### Strengths
+ Originality: Proposes a novel, domain-specific tokenization paradigm—EigenCluster Tokens—by combining spectral decomposition and clustering for financial time series, reframing prediction as bullish cluster identification.
+ Quality: Methodologically rigorous with sound mathematical formulation and thorough experiments across multiple datasets, baselines, and ablation settings.
+ Clarity: Well-structured and clearly written, with intuitive visualizations (e.g., multi-scale workflow, cluster separation) and logical progression from problem to solution.
+ Significance: Highlights tokenization—not architecture—as the key bottleneck in financial forecasting with Transformers. Offers practical benefits: compact representations, faster inference, and actionable trading signals.

### Weaknesses
+ Lack of Justification for Key Methodological Choices:
  + Arbitrary Scalar Projection Function: The function to project high-dimensional features into a scalar (Eq. 6), using a specific combination of $sin()$ and $L2$-norm, lacks theoretical or intuitive explanation. It is unclear why this complex form is superior to simpler, more interpretable alternatives (e.g., using the principal components directly).
  + Counter-intuitive "Bullish Cluster" Criterion: The rule for selecting the "bullish cluster" by favoring the "smallest" size (Eq. 9) is counter-intuitive. This could lead the model to overfit to a few unrepresentative outliers. A more robust criterion would balance the cluster's size with its predictive precision.
+ Insufficient and Potentially Misleading Comparisons:
  + Weak Baseline Models: The comparison is primarily against many weak baseline (e.g. XGBoost, LSTM, Lasso), which are not state-of-the-art baselines. A more convincing evaluation would involve comparing against established time series models like PatchTST or Autoformer.
  + Overly Simplistic Financial Backtest: The trading backtest is insufficient as it only compares against market indices. It should include stronger quantitative baselines (e.g., Time-Series Momentum) and report standard risk-adjusted performance metrics like the Sharpe Ratio and Maximum Drawdown.
+ Serious Concerns about Generalizability:
  + Hand-Tuned Hyperparameters: The formula for selecting the optimal number of clusters $K$ (Eq. 10) and its weights were empirically hand-tuned on a single dataset. This is a classic case of "hyperparameter fitting" that severely undermines the method's credibility and claims of generalization, making it feel more like a custom solution than a general framework.
  + Lack of Online Applicability: The entire tokenization process is performed offline. The paper fails to discuss how the method would adapt to continuously arriving new data in a real-world, online trading scenario, which is a critical practical omission.

### Questions
1. My primary concern is the justification for using a deep 8-layer Transformer with an extremely short input sequence of 9 tokens, trained on a limited set of index data. This setup raises significant questions about overfitting and the actual utility of a complex self-attention architecture. To address this, please **provide plots of the training and validation loss curves** from your experiments to demonstrate that the model is learning generalizable patterns rather than memorizing the training data. Additionally, please provide a clear justification—ideally supported by a comparative experiment with a simpler model like an LSTM—for why a deep Transformer is necessary and effective in this short-sequence, low-data regime.
2. The question to addresses the computational scalability of the multi-scale tokenization. The proposed method requires $n$ separate eigendecompositions on increasingly large matrices, which seems computationally prohibitive for the longer input sequences (e.g., n > 96) common in time series forecasting. To assess the practical viability of your approach, could you please provide both a formal complexity analysis and an empirical study showing **how the tokenization time, training time, and model precision scale as the sequence length $n$ increases**? This is crucial to understand whether the performance benefits justify the potentially exponential growth in computational cost.
3. The concern about the validity of the baseline comparisons. The paper compares its method against non-standard, discretized versions of *point-wise* and *patch-wise* tokenization. The standard approach in modern time series Transformers is to directly project continuous patches into the embedding space via a linear layer, which is a stronger baseline. To properly validate the contribution, the authors must add an experiment comparing their method against this standard **linear projection baseline** under the same model architecture. Without this, the paper's claims of superiority are not sufficiently substantiated, as it has not competed against the most common and robust alternative.

### Soundness
3

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
The paper proposes eigencluster tokenization, a multi-scale spectral method that discretizes OHLC time series into semantic tokens for Transformer-based next-day up-move prediction. It achieves higher precision and faster inference than point- or patch-wise baselines, with complementary gains from multi-scale design, eigendecomposition, and clustering.

### Strengths
1. The paper identifies a clear and significant research problem, focusing on tokenization and out-of-vocabulary (OOV) challenges in financial time-series modeling, which are crucial for improving model generalization and interpretability.

2. The proposed approach shows good engineering feasibility, as the multi-scale prefix, eigendecomposition, and clustering pipeline forms a concise and implementable framework that reduces token and sequence length while preserving key temporal information.

3. The authors exhibit commendable efforts toward reproducibility by providing code access and detailed implementation descriptions in the appendix, which facilitates transparent verification and future research replication.

### Weaknesses
1. The current scalar projection design lacks theoretical grounding and comparative evaluation. It is recommended to test alternative mappings (e.g., first principal component, eigenvector projection, multi-component clustering, or nonlinear functions like tanh) and report robustness across settings.

2. In terms of clustering methodology, automatic selection approaches such as the Bayesian Information Criterion, Akaike Information Criterion, Bayesian Gaussian Mixture Models, or time-series cross-validation should be compared. The paper should also justify the rationale and advantages of the specific clustering method it adopts.

3. There already exist other clustering-based tokenization methods, but the paper does not cite or compare them to demonstrate the advantages of its proposed tokenization approach.

4. The paper does not provide sufficient justification for the effectiveness of the chosen prefix windowing approach, and the selection of the prefix window size lacks rationale and comparative experiments. 

5. The baselines include only point- and patch-wise discretization, but not learned tokenizers such as VQ-VAE, discrete autoencoders, or transformer-based quantizers, which would provide a stronger benchmark for the proposed approach.

6. The paper lacks a principled discussion of why eigen-decomposition and 1D clustering should preserve predictive structure or how they relate to temporal-spectral representations.

### Questions
1) What is the theoretical motivation for the specific scalar projection design? Have alternative mappings or nonlinear transformations been tested to confirm robustness?

2) What is the principled link between eigendecomposition + 1D clustering and the preservation of predictive temporal–spectral structure in price series?

3) Why was 1D K-means chosen over automatic or probabilistic clustering methods (e.g., BIC/AIC-based selection or Bayesian GMM)? How sensitive are the results to this choice?

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
This paper proposes a new financial time series discretization method, Eigencluster Tokenization, which performs multi-scale eigendecomposition on OHLC matrices and clusters eigenvector projections to generate compact and semantically meaningful tokens. The approach aims to overcome three major issues in conventional point-wise and patch-wise tokenization for financial data: excessive token counts, sparse distributions, and out-of-vocabulary (OOV) risks. The authors conduct cross-market experiments on both S&P 500 and CSI 300 datasets.

### Strengths
1. The paper clearly discusses the challenges of financial tokenization that differ from language modeling (e.g., high variability, OOV risk), and proposes a multi-scale eigendecomposition and clustering approach on OHLC matrices. Made adaptations specific to financial data characteristics, such as using C_t−n+k−1 for normalization.


2. The definition of the most bullish cluster enhances model interpretability.

3. The evaluation on both S&P 500 and CSI 300 datasets demonstrates strong performance, along with significant reductions in token count and computational cost.

### Weaknesses
1. The explanation of Table 1 is unclear.

2. The clustering method is limited to K-means only; the impact of alternative clustering approaches (e.g., GMM, Spectral Clustering) is not explored.

3. Constituent stocks are part of the indices, and depending on the index composition, certain stocks may have a dominant weight and be strongly correlated with index movements. The paper does not discuss this dependence.

4. The font size in Figure 6 are too small, making it hard to read; figure annotations are insufficient (no explanation of labels for each point).

5. The paper lacks comparison with other recent tokenization methods.

### Questions
1. Explain the tables and figures more clearly and make them easier to read.

2. Provide a discussion on the dependence or correlation among the test stocks.

3. Compare the proposed method with other state-of-the-art tokenization approaches.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an eigen-cluster tokenization method for applying Transformer architectures to financial time series. It argues that conventional point-wise and patch-wise tokenization approaches are ill-suited for financial data because of three structural issues: excessive token cardinality, sparse token representation, and out-of-vocabulary tokens.

To address these problems, the authors propose a multi-scale spectral clustering pipeline. OHLC matrices are decomposed via eigen decomposition to derive dominant temporal components, and their scalar projections are clustered to produce discrete tokens. The resulting cluster tokens are used in a Transformer to predict next-day price increases. The method reportedly reduces the token vocabulary to around 50–100 while improving precision by 6–9 pp over point- and patch-wise baselines across S&P 500 and CSI 300 datasets.

### Strengths
- Clear motivation from tokenization theory: The paper correctly identifies structural mismatches between standard tokenization and financial data — particularly OOV tokens arising from non-stationarity and sparse embedding updates. This diagnosis is compelling and empirically supported with vocabulary size and OOV analyses.
- Methodological novelty: Using eigen decomposition and clustering as a discrete representation mechanism is conceptually interesting and extends recent ideas from computer-vision token reduction (e.g., Clusterformer, PACAViT). The multi-scale prefix-window representation captures temporal hierarchy, which is intuitively aligned with financial market regimes.
- Experimental breadth.: The authors conduct cross-market experiments (US/China), ablations, and backtests. The ablation results substantiate that clustering and eigen decomposition both materially affect performance.

### Weaknesses
- SMOTE and data-generation bias: The use of SMOTE to balance “bullish clusters” is problematic for financial time-series forecasting. SMOTE interpolates samples in Euclidean space, implicitly assuming smooth similarity structures; yet in asset-price data, temporal autocorrelation and regime-dependence make such interpolation unrealistic. Oversampling can distort minority-class distributions and generate spurious trajectories. The paper acknowledges in appendix that SMOTE materially changes sample counts, but does not demonstrate that synthetic samples preserve realistic dynamics. This undermines the credibility of the reported precision gains.
- Heuristic cluster selection (Eq. 10): The rule combining three weighted terms (bullish probability, cluster size, and K penalty) is purely empirical with fixed weights (25.0, 0.8, 0.2). No sensitivity or robustness analysis is presented. Because market regimes differ across periods, applying the same weights to all datasets introduces arbitrariness and potential overfitting to the chosen calibration period.

### Questions
As noted in the weaknesses, using SMOTE for financial time series is highly unusual. The paper should clarify why this oversampling method was chosen and why it is appropriate, given that SMOTE interpolates data points and may distort temporal and distributional structures essential to financial data.

Similarly, the weighting scheme in Eq. (10) (25.0, 0.8, 0.2) seems empirically chosen without explanation or sensitivity analysis. The authors should specify how these weights were determined and whether the results remain robust under different settings.

### Soundness
3

### Presentation
3

### Contribution
3
