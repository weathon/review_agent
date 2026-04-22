# Integrating Selective State-Space Models and Bayesian Graph Attention for Uncertainty-aware Time-Series Analysis

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
This paper presents \textbf{BIMAMBA \& Bayesian-MAGAC}, a unified framework that integrates bidirectional Selective State-Space Models with Bayesian Multi-head Adaptive Graph Attention Convolution for uncertainty-aware financial forecasting. The framework addresses two fundamental challenges: capturing long-range temporal dependencies across volatile market regimes while maintaining linear complexity, and learning adaptive cross-sectional structure with calibrated predictive uncertainty. BIMAMBA processes sequences bidirectionally via reversible state-space filters, extracting complementary temporal features while preserving strict causality. MAGAC constructs dynamic adjacencies through Gaussian kernel and attention blending, followed by Chebyshev spectral filtering for multi-scale aggregation. The Bayesian extension treats adjacencies and spectral filters as stochastic variables via Monte Carlo Dropout and DropEdge, yielding posterior predictive distributions with closed-form variance propagation at $\mathcal{O}(N)$ complexity. Comprehensive evaluations on U.S. equity indices demonstrate that the architecture achieves substantial improvements in both point prediction accuracy and uncertainty calibration compared to established baselines, with statistically significant correlation between predicted uncertainty and prediction difficulty, suggesting practical utility for risk-aware decision making in financial applications.
\textit{The code is available on GitHub but has been hidden to preserve anonymity during the review process.}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a model that uses a bidirectional Mamba backbone combined with Bayesian graph attention for financial forecasting. The authors claim their model achieves state-of-the-art performance on three financial datasets.

### Strengths
The paper proposes a lightweight architecture that demonstrates strong performance on the evaluated financial datasets.
Some design choices, such as the use of bidirectional encoding, are well-justified and intuitive.

### Weaknesses
_Insufficient Citations and Details_: Several core components of the proposed model appear to be derived from prior work without proper attribution. For instance, the bidirectional Mamba architecture has been explored in numerous previous studies (e.g., [1], [2], [3]). A similar issue is present in Sections 3.2.2 and 4. Since the paper does not claim to have invented Bayesian Graph Neural Networks, it should cite the original proposers of this idea (likely [4]). Furthermore, the authors do not provide citations for the datasets used, which hinders reproducibility and disregards the contribution from dataset makers.

_Lack of Ablation Studies_: The authors provide explanations for their design choices in Section 3.2.3, which is appreciated. However, these explanations remain as claims without empirical support. The paper shows that the full model show performance improvement, but it does not demonstrate the individual contribution of each component (e.g., the use of $\beta=\text{softmax}(\gamma)$ versus a directly learned $\beta$). Further, if the authors are claiming the MAGAC backbone as a novel contribution (see 3), it is critical to compare its performance against existing backbone architectures.

_Confusing Terminology_: The terminology in Section 3.2 is confusing. "Graph Attention" is a well-established technique, first introduced in [5]. However, the description in Section 3.2.1 appears to describe standard attention mechanisms, not specifically graph attention. Similarly, while "graph convolution" is mentioned, no corresponding graph convolutional structure is detailed. This ambiguity makes it difficult to determine whether the authors are proposing a novel network architecture or have misconceptions about existing ones (GAT, GCN, etc.).

_Other Issues_: Figure 1 need additional polish--the texts are not clear. I also believe for $A_{mn}^{g}$ and $\tilde{A}_{mn}^{(g,s)}$, the use of $\text{softmax}(\exp(\cdot))$ might be a mistake since otherwise it tends to cause instabilities (so you have $\exp(\exp(x))$).
Overall, the exact contributions of the paper are unclear. For the parts that are relatively clear, there is a lack of theoretical or empirical evidence to support the authors' claims. Therefore, I recommend that this manuscript undergo significant revision.

[1] Zhu, Lianghui, et al. ‘Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model’. arXiv [Cs.CV], 2024, arxiv.org/abs/2401.09417. arXiv.

[2] Liang, Aobo, et al. ‘Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting’. arXiv [Cs.LG], 2024, arxiv.org/abs/2404.15772. arXiv.

[3] Erol, Mehmet Hamza, et al. ‘Audio Mamba: Bidirectional State Space Model for Audio Representation Learning’. IEEE Signal Processing Letters, vol. 31, 2024, pp. 2975–2979,

[4] Hasanzadeh, Arman, et al. ‘Bayesian Graph Neural Networks with Adaptive Connection Sampling’. CoRR, vol. abs/2006.04064, 2020, arxiv.org/abs/2006.04064.

[5] Veličković, Petar, et al. ‘Graph Attention Networks’. arXiv [Stat.ML], 2018, arxiv.org/abs/1710.10903. arXiv.

### Questions
The authors should directly address the specific concerns raised in the "Weaknesses" section above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors have used bidirectional MAMBA for time and Bayesian multi-head graph attention for space to perform forecasting. The work claims to preserve the causal and anti-causal dependencies across the temporal dimension.

### Strengths
- Novel architecture design 
- Good experimental results

### Weaknesses
- No ablation studies
- Equations are not numbered
- Limited number of datasets
- MAE metric should also be reported, as RMSE can dramatically shrink components less than 1
- The training is slow compared to other baselines
- No paragraph to clarify notations
- sparse citations
- Experimental results cannot be verified

### Questions
1. Is there any reference on the benefits of processing the temporal sequence in both forward and reverse orders?
2. Were any ablation studies performed that show that bidirectional MAMBA is better than regular MAMBA? (both in terms of performance and training time)
3. How does the performance of the proposed model change with the size of the training data available? 
4. What are the confidence intervals of the reported metrics?
5. Is the matrix P an anti-diagonal matrix of ones?
6. How are the sequences being merged? Does '+' represent element-wise addition?
7. Could the authors please cite some prior work on BGNN, esp. for time series forecasting?
8. Is there a way through which the results can be verified? I do not see any link to the code.
9. In Table 1, for which dataset is the training time reported?
10. Could the authors please comment on the causality in the data, and if any tests were performed? See: _Pearl, J. (2009). Causality. Cambridge University Press_

### Soundness
2

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
3

### Summary
The paper proposes a hybrid sequence–graph framework for equity forecasting that couples a bidirectional Mamba (Bi-Mamba) state-space encoder with a Bayesian Multi-head Adaptive Graph Attention Convolution (MAGAC) layer. The Bi-Mamba encoder processes price windows in both forward and reverse directions and feeds sequence features to MAGAC; MAGAC builds an adaptive adjacency by blending a Gaussian kernel on learnable node embeddings with attention-based scores, then applies Chebyshev spectral filtering and multi-head aggregation. A lightweight Bayesian treatment is introduced by applying MC-Dropout to node embeddings and DropEdge at inference to obtain posterior predictive means/variances optimized via heteroscedastic Gaussian NLL. Experiments on NASDAQ/NYSE/DJIA report very large gains in RMSE/IC/RIC over LSTM/Transformer/GNN baselines, with modest MACs and parameter counts claimed for the proposed model.

### Strengths
1.   A linear-time temporal encoder (Mamba) with an adaptive spectral/attention GNN is a coherent and potentially scalable combination for long horizons. The bidirectional Mamba block and residual/normalization design are clearly described.
2.   The MAGAC construction (Gaussian + attention blend, Chebyshev supports, head mixing) is laid out step-by-step with equations that are easy to implement.
3.    The paper spells out how MC-Dropout on node embeddings and inference-time DropEdge are used to derive mean/variance and the closed-form variance propagation through a linear head, optimizing a heteroscedastic Gaussian NLL. This is clearly written and easy to reproduce technically.
4.   The paper reports MACs/params/epoch time and positions the model against Transformers with respect to efficiency (albeit for S=1). Consistent, large improvements on all benchmarks, with excellent calibration.
5.   Consistent, large improvements was reported on all benchmarks, with excellent calibration.

### Weaknesses
W1. The reported IC values are extraordinarily high for daily single-day returns (e.g., IC=0.9413 on NASDAQ). The paper defines IC/RIC as correlations with realized single-day returns, but offers no safeguards against common leakage vectors (e.g., improper chronological splits, cross-sectional standardization using future info, or target leakage through feature engineering). Please justify these numbers or audit the pipeline.
Additionally, the evaluation uses a fixed L=5 with bidirectional processing; while the text claims “causality is preserved,” there is no precise description of how labels are formed and how windows are cut to guarantee no look-ahead within the window (e.g., predicting $r_{t+1}$ from $[t−4,t]$). Provide a rigorous data construction diagram and masking rules.
W2. The dataset description states N=82 features per day; however, the graph notion (nodes = assets) is not specified (how many tickers? which universe? how are nodes aligned over time? dynamic graph cadence?). The text also equates $E = d_{model}$ with “number of graph nodes (model width),” which is dimensionally inconsistent—E is typically feature width, not the number of assets. Later equations sum over E as if it were the node count. This confusion must be resolved with explicit tensor shapes (batch, nodes, time, channels) and a clear asset universe size.
W3. The paper claims “well-calibrated uncertainty” and “without inflating computational cost,” yet no calibration metrics (NLL comparisons, CRPS, PIT/QQ plots, ECE, PICP/ACE) are reported. Moreover, Table-reported MACs are for S=1 while the method elsewhere enables S=10 at inference; the “no overhead” claim is therefore misleading. Also, DropEdge is applied only at inference, effectively evaluating a different model than trained. Provide proper calibration evaluation and fair compute accounting for the posterior estimates.
W4. There is no ablation disentangling the gains from bidirectionality, Gaussian+attention blend, Chebyshev order, number of heads, Bayesian sampling, or DropEdge. Without such analysis, the source of improvements is unclear.
W5. The authors acknowledge restrictions to daily closing prices and the need for theory on Bayesian MAGAC stability, but neither robustness checks (e.g., regime shifts, crisis periods) nor sensitivity analyses are presented.

### Questions
Please see above W1-W5.

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
3

### Summary
This paper proposes Bi-Mamba Bayesian-MAGAC, combining bidirectional selective state-space models (Bi-Mamba) with Bayesian multi-head adaptive graph attention convolution (MAGAC) for financial time-series forecasting. Bi-Mamba processes sequences bidirectionally with linear complexity, while MAGAC constructs adaptive graph topologies and performs spectral filtering. Experiments on NASDAQ, NYSE, and DJIA show improvements over baselines.

### Strengths
1. The integration of linear-time Mamba for temporal dynamics with graph convolution for cross-asset relationships effectively addresses both long-range dependencies and inter-asset correlations in financial forecasting.

2. The Bayesian treatment captures both epistemic and graph structural uncertainty through MC-Dropout and DropEdge, with closed-form variance propagation maintaining computational efficiency.

### Weaknesses
1. Evaluation uses only three datasets with implausibly high IC values (~0.94) compared to baselines (~0.2-0.3), raising concerns about potential data leakage or experimental setup issues.

2. The code is not available for reproducibility.

### Questions
See the Weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
