# FATE: Feature-Wise Graph Attention with Multi-Period Temporal Encoding for Stock Return Forecasting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Forecasting stock returns remains challenging due to the dual complexity of \textit{temporal heterogeneity} and \textit{structural instability}. On the one hand, stock return series contain signals at multiple horizons: short-term fluctuations, medium-term trends, and long-term cycles. On the other hand, relations across stocks are non-stationary and noisy: correlations strengthen or vanish under sector rotations, liquidity shocks, or macro events. However, existing temporal models often process each stock independently, neglecting cross-stock dependencies and blur signals across horizons. Moreover, structure-aware models typically rely on static or single-view relation-graphs that are brittle under market shifts. To overcome these limitations, we propose \textit{FATE}, which couples \textit{ multi-view dynamic graphs} with a \textit{multi-period temporal encoder} to jointly capture cross-stock relations and horizon-specific signals, and further employs \textit{feature-wise graph attention} with a learned graph to integrate these signals coherently while suppressing noise and enhancing interpretability. Experimental results demonstrate that FATE consistently outperforms strong temporal and graph baselines in correlation metrics and investment simulations, which provides more informative and robust signals for stock return forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FATE, a framework for stock return forecasting that jointly addresses temporal heterogeneity (signals at different time scales) and structural instability (changing inter-stock relations). The model's key innovations are a multi-period temporal encoder to disentangle horizon-specific patterns and a dynamic graph fusion module that creates a robust, denoised relational structure. Extensive experiments on the CSI 500/1000 datasets demonstrate that FATE achieves state-of-the-art results, significantly outperforming strong baselines.

### Strengths
+ Originality: The primary originality lies in the creative and synergistic combination of several well-motivated ideas to tackle the specific dual challenges of stock forecasting:
  + The multi-period temporal encoder provides an explicit and effective mechanism for disentangling signals across different time horizons, which is a more direct approach than what is seen in typical single-stream sequence models.
  + The dynamic graph fusion module is sophisticated, combining pre-defined dynamic graphs, learned correlations, and a market-state vector. This represents a significant step up from models that use static or single-view graphs.
+ Quality: The methodology is sound, and claims are convincingly validated through a rigorous experimental protocol that includes strong baselines, comprehensive ablations, and realistic portfolio backtesting.
+ Clarity: The paper is exceptionally well-written, with a clear motivation and logical structure that makes the complex architecture easy to understand.

### Weaknesses
+ Lack of Sensitivity Analysis for Key Design Choices: The paper relies on seemingly arbitrary hyperparameters without sufficient justification or analysis. For instance, the choice of temporal partitions $(T/4, T/2, T)$ and the graph binarization threshold (τ) are critical to the model's design, yet their impact on performance and robustness is not explored.
+ Unsubstantiated Interpretability Claims: The paper claims to enhance interpretability but provides no supporting analysis. It misses a clear opportunity to inspect its own interpretable components—such as the adaptive horizon weights, graph fusion weights, or feature gates—to offer concrete insights into what the model learns about market dynamics during different conditions.
+ Computational Cost is Ignored: The model is complex, yet the paper entirely omits any discussion of its computational complexity. Key practical metrics like model size, training time, or inference latency are not reported, making it difficult to assess the method's real-world viability against simpler baselines.

### Questions
1. Regarding the graph fusion module (Section 4.4), the calculation $T = sigmoid(Q K^T)$ for the stock correlation matrix raises questions. This formulation lacks the scaling factor common in attention mechanisms, potentially leading to $sigmoid$ saturation and vanishing gradients. Furthermore, its necessity is not justified against simpler baselines like **cosine similarity**. Could the authors please discuss its impact on training stability, and provide an ablation study comparing it to simpler correlation metrics to validate its effectiveness?
2. FATE employs a purely learnable vector $M$ for its market state rather than real index data as in MASTER[1], and critically, fails to include MASTER as a baseline in the experiments. Could the authors justify their learnable market state design and explain the omission of this highly relevant and strong baseline? 
3. The performance drop from removing the *feature-wise gating* in SuperAttention (the "NoGate" variant in Tables 3 & 4) is marginal. This appears to contradict the paper's claim that this mechanism is a key contribution for "suppressing noise." Can the authors explain this small difference and provide stronger evidence to justify the necessity and practical value of this complex component?
4. The graph-regularized loss ($L_E$ in Sec 4.6) requires a set of ground-truth positive ($E^+$) and negative ($E^-$) edges for supervision, but the paper omits how these sets are constructed. Could the authors please clarify their definition?
5. The **ICW** metric is described in Section 5.1 as a method that "upweights top-decile predictions," but its mathematical formula is omitted. Could the authors please provide the precise definition used to calculate ICW to ensure reproducibility?
6. The ablation study shows the multi-period temporal module is the primary performance driver (evidenced by the large drop in the "SinglePeriod" case). This raises a critical question: *is the complex graph module necessary?* To justify its inclusion and complexity, the authors should provide a "**Temporal-Only**" ablation—bypassing the entire graph module and using only the temporal output for prediction—to quantify the actual contribution of the graph architecture against the full FATE model.
7. In the portfolio backtests (Figure 2), simple temporal models like GRU and MLP show strong cumulative returns, challenging the necessity of FATE's complex graph architecture. To demonstrate that FATE's advantage is not just higher returns but also superior robustness, could the authors please report key risk-adjusted metrics like the Sharpe Ratio and Maximum Drawdown? These metrics would clearly quantify FATE's risk-management benefits over simpler models.

[1] Li, Tong, et al. "Master: Market-guided stock transformer for stock price forecasting." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 1. 2024.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FATE, a novel framework for stock return prediction. FATE combines multi-view, dynamically constructed relation graphs, a multi-period temporal encoder, a graph fusion mechanism integrating temporal, relational, and market-wide information, and a feature-wise graph attention module. Experiments on the CSI 500/1000 datasets demonstrate that FATE outperforms all baselines. Ablation studies and visualizations further validate the contributions of each architectural component.

### Strengths
1. The architecture of FATE is well-motivated. FATE constructs multiple graph views grounded in economic rationale, adoptes multi-period temporal representation learning, and adaptively fuses multi-view graphs and horizon-specific signals.
2. Ablation results provide evidence that each module—multi-period encoding, graph fusion, gating, and regularization—adds value to the overall performance.

### Weaknesses
1. While the SuperAttention module and sparse graphs are lauded for "interpretability", the paper lacks concrete examples, case studies, or visualizations of attention weights or learned graphs on actual stock subgraphs. This omission weakens the interpretability claims and leaves readers with limited insight into how the model captures market dynamics.
2. The experiments are limited to the China stock market, and the scalability of FATE to larger markets (e.g., S&P 500 or global indices) is not discussed.

### Questions
The multi-period temporal encoding and feature-wise graph attention mechanisms may introduce significant computational overhead. Could the authors provide more details on the training time and scalability to larger datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the core challenges in the field of stock return prediction: temporal heterogeneity (stock returns contain multi-scale signals such as short-term fluctuations, medium-term trends, and long-term cycles) and structural instability (the correlations between stocks exhibit non-stationarity due to factors like industry rotation and liquidity shocks). It proposes a unified framework named FATE to address the limitations of existing models. The core designs of FATE include:

1.Multi-view dynamic graph construction

2.Multi-cycle temporal encoding

3.Graph fusion module

4.SuperAttention mechanism

In terms of experiments, the authors validated on the CSI 500 and CSI 1000 datasets from 2017 to 2024: FATE significantly outperforms baselines such as MLP, LSTM, and THGNN in metrics including IC and RankIC. Ablation experiments prove that modules like multi-cycle encoding, graph fusion, and SuperAttention all contribute positively to performance.

### Strengths
FATE's design is deeply aligned with the characteristics of financial markets. Its multi-cycle encoding can match the decision-making logic of short/medium/long-term investors, and its multi-view dynamic graph can cover dimensions of price (opening/closing prices) and liquidity (trading volume/turnover rate). This effectively addresses the "insufficient adaptability" issue of general temporal-graph models in financial scenarios. Additionally, SuperAttention filters effective neighbor information through feature gating, which specifically suppresses noise in financial data and outperforms the single scalar weight mechanism of traditional GAT.

### Weaknesses
1.The paper consistently uses 6 types of features—opening price, closing price, highest price, lowest price, trading volume, and turnover rate—to construct dynamic graphs. However, it does not explain the basis for selecting these 6 features. For instance, both trading volume and turnover rate reflect liquidity, yet the information redundancy between them is not analyzed.

2.The main text only mentions that the global market vector M is "learnable" but fails to specify its input features (e.g., whether it includes macro-micro indicators such as market indices, interest rates, and trading volume), the logic for dimension setting, and the training method.

3.There is a lack of ablation experiments to verify the impact of reducing the number of views (e.g., to 4 types) on model performance. This makes it difficult to prove the necessity of 6 views and may lead to redundant model complexity.

4.The data preprocessing section only mentions "cross-sectional standardization of all features" but does not explain how to handle outliers commonly seen in financial data (such as extreme daily stock price fluctuations and sudden surges in trading volume).

5.Experiments are only validated on the component stock datasets of two A-share indices: CSI 500 and CSI 1000. They are not extended to other markets (e.g., U.S. stocks, Hong Kong stocks) or different asset types (e.g., bonds, commodities), making it hard to fully demonstrate the model’s generalization ability in non-A-share scenarios.

6.The paper does not report the model’s training/inference time consumption or memory usage. Although the model needs to recalculate 6 dynamic graphs, retrain the gating network, and perform multi-head attention computation every day, it does not explain the time difference in its computational scale compared with conventional methods.

7.The feature gating function of SuperAttention uses tanh activation, but the paper does not discuss whether other functions could be adopted instead.

### Questions
1.Which component—multi-cycle temporal encoding, multi-view graph fusion, or SuperAttention—contributes most to FATE’s IC improvement over baselines (e.g., FactorGCL, Transformer)? Could targeted experiments be designed (e.g., replacing FATE’s multi-cycle encoding with single-cycle and comparing performance degradation) to clarify the contribution ratio of each module?

2.Given that FATE includes modules such as multi-view graph construction and multi-cycle encoding, what are its training and inference speeds? Will the model’s computational complexity rise significantly when the number of stocks increases (e.g., exceeding 1,000)? Could efficiency metrics under different data scales (e.g., single-epoch training time, inference latency) be supplemented to evaluate its practical deployment feasibility?

3.Financial data often has missing values due to trading halts, market closures, etc. The paper does not explain how such missing values are handled (e.g., interpolation, masking mechanisms) nor verify the model’s performance stability under different missing rates (e.g., 5%, 10%). Could robustness test results and details of handling schemes for data missing scenarios be added?

4.Regarding the number of Transformer heads in multi-cycle encoding: The paper mentions that multi-cycle branches process signals via Transformer multi-head self-attention but does not specify the selection of head count (e.g., 8 heads, 12 heads) or the basis for it. It also does not verify how different head counts affect the ability to capture multi-scale signals (e.g., whether too few heads cause missed dependencies or too many induce overfitting). Could the optimization process of this hyperparameter and experimental comparisons be supplemented?

### Soundness
2

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
4

### Summary
This paper proposes the FATE framework to address the temporal heterogeneity and structural instability in stock return forecasting. Method A multi-view dynamic graph, short -, medium -, and long-term multi-period time encoding, and feature-level graph attention mechanism are combined to capture the dynamic relationship across stocks and separate signals at different time scales.

### Strengths
This work achieves stable and leading results on two large data sets, CSI 500 and CSI 1000. The authors detail the data processing, training pipeline, and baseline setup, and provide full implementation details to ensure good reliability and reproducibility of the study results.

### Weaknesses
1. In the construction stage, the model uses a large amount of historical data to form a multi-view dynamic graph, and the calculation process is complex, but it does not provide any analysis on computational efficiency or Scalability.
2. Defining "graph noise" as a "weak link" is imprecise, since spurious strong links can also be noise. Papers lack empirical validation of the effectiveness of noise culling or interpretability case analysis.
3. Authors claim that the dynamic graph structure is better than the predefined static graph, but do not provide direct comparative experiments to support this conclusion.
4. All experiments are based on the CSI, which lacks cross-regional and cross-market generalization verification and does not conduct parameter sensitivity analysis.

### Questions
See weaknesses.

Further, how to ensure that the assumption of equivalence between "weak link" and "noisy" holds? Can spurious strong links also act as noise? Are there case studies?

### Soundness
2

### Presentation
2

### Contribution
2
