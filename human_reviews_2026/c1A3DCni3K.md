# A Set-Sequence Model for Time Series

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Many prediction problems across science and engineering, especially in finance and economics, involve large cross-sections of individual time series, where each unit (e.g., a loan, stock, or customer) is driven by unit-level features and latent cross-sectional dynamics. While sequence models have advanced per-unit temporal prediction, capturing cross-sectional effects often still relies on hand-crafted summary features. We propose Set-Sequence, a model that learns cross-sectional structure directly, enhancing expressivity and eliminating manual feature engineering. At each time step, a permutation-invariant Set module summarizes the unit set; a Sequence module then models each unit’s dynamics conditioned on both its features and the learned summary. The architecture accommodates unaligned series, supports varying numbers of units at inference, integrates with standard sequence backbones (e.g., Transformers), and scales linearly in cross-sectional size. Across a synthetic contagion task and two large-scale real-world applications—equity portfolio optimization and loan risk prediction—Set-Sequence significantly outperforms strong baselines, delivering higher Sharpe ratios, improved AUCs, and interpretable cross-sectional summaries.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Set-Sequence, a model for large-scale time series prediction. It first uses a Set module to extract a permutation-invariant population summary, then combines it with individual features in a Sequence module (e.g., Transformer or RNN) for prediction. Experiments show improved accuracy, efficiency, and interpretability across multiple tasks.

### Strengths
The paper presents a clear and modular architecture that integrates a permutation-invariant Set module with standard sequence models, offering linear scalability and theoretical grounding through connections to Deep Sets. Its empirical evaluation—spanning large-scale financial datasets such as CRSP equities and CoreLogic mortgages—demonstrates strong predictive and economic performance. However, the validation remains limited to financial domains, and the interpretability analysis, while suggestive of meaningful latent structure, is primarily correlational and not deeply explored.

### Weaknesses
1) The motivation is partially unclear: while the paper aims to handle large cross-sections of time series efficiently, it is not well distinguished from the classical setting of multivariate time series modeling. If each entity were viewed as a “variable,” the proposed Set-Sequence formulation would resemble feature aggregation rather than feature selection. However, the paper does not clarify this distinction nor position its contribution relative to multivariate temporal modeling, leaving its conceptual novelty somewhat ambiguous.

2) While the paper motivates itself by the need to capture cross-entity interactions, it does not explicitly model pairwise or structured dependencies as in graph or attention-based methods. Instead, it addresses a different, more specific challenge: efficiently summarizing population-level context under exchangeability when the number of entities is extremely large. This makes the contribution one of scalability and global aggregation, rather than relational modeling in the strict sense.

3) The dataset includes only the top 500 market-cap equities per period and excludes missing returns, which may introduce survivorship or selection bias. It is recommended to include a less-filtered baseline (e.g., fixed universe or delisted stocks) and year-by-year sensitivity analyses to ensure robustness.

4)  The model assumes exchangeability and uses pooled moments, which may fail under strong non-exchangeable or localized dependencies. A small synthetic experiment comparing Set-Seq with attention or graph models under such conditions would clarify its expressive limits.

5) The paper demonstrates the expressive capability of the model; however, in practical testing, its performance shows a clear gap compared to attention-based mechanisms. The paper does not clearly explain the cause of this discrepancy and only highlights the model’s superiority in terms of time complexity.

6)  The paper mainly uses mean and polynomial pooling. Including or emphasizing ablations on alternative pooling methods (e.g., quantile, robust, or group-wise pooling) would better illustrate expressivity–cost trade-offs.

### Questions
1) How is the proposed setting fundamentally different from classical multivariate time series modeling (multivariate time series feature selection, reweighting), beyond computational efficiency?

2) If the model does not explicitly capture pairwise or structured dependencies, can it truly model cross-entity interactions?

3) Does restricting the dataset to the top 500 equities introduce survivorship or selection bias, and how robust are the results under a less-filtered universe?

4) How does the exchangeability assumption affect performance when dependencies are localized or non-exchangeable?

5) Why does the model underperform attention-based baselines despite its claimed expressivity advantages?

Would exploring alternative pooling functions (e.g., quantile or group-wise) change the expressivity–efficiency trade-off?

### Soundness
2

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
5

### Summary
This paper addresses high-dimensional time series prediction with correlated units (e.g., stocks, loans, sensors). It needs to capture cross-sectional dependencies among M units and temporal dynamics over T steps, but direct joint modeling is computationally infeasible. Existing methods rely on hand-crafted cross-sectional features, fail to adapt to dynamic unit counts, and miss latent correlations—though the problem itself is important.

The paper proposes the Set-Sequence model (Set module for linear-complexity permutation-invariant cross-sectional summaries; Sequence module for unit temporal dynamics). Evaluated on a synthetic loan default task, 36,600 U.S. equities, and 5M mortgage records, it outperforms baselines (Transformer, S4, domain models): 4.4–10.2× lower KL divergence, 22%–42% higher equity Sharpe ratio, and 0.683 mortgage AUC.

### Strengths
Strength 1: Set-Sequence models cross-sectional dependencies as a permutation-invariant set summary. Linear pooling plus a shallow readout is proven to uniformly approximate any continuous permutation-invariant target. This result supplies an expressive lower bound under exchangeability and directly informs the choice of pooling order and hyper-parameters.

Strength 2: The module plugs into any temporal backbone with Θ(M) complexity. A single scan and per-unit temporal update reduce FLOPs by an order of magnitude relative to self-attention when M≥2500. Units can be added or dropped at inference without retraining, and memory grows linearly, so one A6000 GPU handles 5 M samples.

Strength 3: Consistent, measurable gains are observed on synthetic contagion, equity-portfolio, and mortgage-default tasks. Compared to strong baselines, it achieves 10× KL reduction, +42 % Sharpe, and +4 average AUC, with improvements that grow monotonically with pool size. The learned summary correlates 0.67 with the realized foreclosure rate, delivering an auditable macro signal for risk management.

### Weaknesses
Limited methodological novelty: The proposed Set-Sequence model lacks substantial innovation. It simply combines an existing permutation-invariant set encoder with a standard sequential backbone, following the design principles of Deep Sets and Set Transformer with only minor modifications. Theoretical results are mostly restatements of known approximation properties rather than new insights. Overall, the contribution feels more like an engineering integration than a novel modeling paradigm.

Outdated and insufficient experimental comparisons: The experiments rely on relatively old baselines and omit recent state-of-the-art time-series architectures such as iTransformer, TimeMixer, PatchTST, and Mamba. The reported improvements over older models therefore do not convincingly demonstrate the superiority of the proposed method. Moreover, the empirical evaluation lacks comprehensive analyses such as cross-task generalization, robustness, or hyperparameter sensitivity.

Shallow theoretical and architectural analysis: The claimed linear scalability is only valid under a simple mean-pooling assumption, which is not evaluated in heterogeneous or strongly correlated scenarios. The paper does not analyze how this simplification affects representational capacity or generalization performance, nor does it study how the look-back parameter interacts with temporal dependencies. Consequently, the theoretical analysis remains shallow and fails to substantiate the claimed advantages of the architecture.

### Questions
Sensitivity to look-back parameter 𝐿: The paper fixes 𝐿=3 without justification. How sensitive is the model to this choice? Does increasing 
𝐿 enhance long-term dependency modeling or simply increase computation? A short sensitivity analysis would clarify its role.

Robustness of linear aggregation: The Set module employs mean pooling for cross-sectional summarization. Given potential heterogeneity among units, is this pooling stable or biased toward large or noisy samples? Have the authors explored normalization or adaptive weighting schemes?

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
The paper tackles prediction over large cross-sections of units (loans, stocks, customers) where each unit is a time series. The authors propose a Set-Sequence architecture that factors the problem into two natural dimensions: Set modeling across entities at each time step (the “cross-section”). Sequence modeling through time for each entity (the “temporal” dimension). Experiments on (i) a synthetic contagion/default task, (ii) U.S. equity portfolio construction, and (iii) mortgage risk prediction (~5M loan-months; 52 features) show consistent gains, plus interpretable set summaries that correlate with latent factors.

### Strengths
-  The idea is clear: set over units → sequence over time.
- The method is scalable,  with theoretical complexity analysis, and can be extended when M is large.
- Empirical results are broad and strong, both in Synthetic contagion and real-world datasets.
- The model gives the Interpretability.

### Weaknesses
- The Set-Sequence architecture extends the idea of multiple instance learning mean pooling to the temporal domain. There are a few related works that are encouraged to be discussed or compared.

[1] Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems 30 (2017).

[2] Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." International conference on machine learning. PMLR, 2018.

[3] Chen, Xiwen, et al. "TimeMIL: Advancing multivariate time series classification via a time-aware multiple instance learning." ICML'24


- The empirical section mainly compares against early Transformer-style and S4/LongConv models. Given the rapid development of efficient long-sequence and cross-sectional architectures, incorporating or at least discussing stronger, contemporaneous methods would be encouraged.

### Questions
- Did you compare early-vs-late insertion or stacking multiple Set layers? Any diminishing returns?
- Will synthetic generator, equity feature pipelines, and mortgage preprocessing be released to enable reproduction?
- This is just an open question: can the Set-Sequence concept be verified beyond financial or factor-style datasets?
- To be honest, I am not sure about the assumption of exchangeability in real-world applications. Are they only approximately exchangeable or any exceptions?

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
The paper introduces a Set-Sequence model that specifically learns the cross-sectional structure in time series. The model breaks down the task into two prongs: summarizing the unit set and modeling each unit’s dynamics conditioned on their features and learned summaries. It is scalable to variable numbers of units. Empirical results show that the model outperforms the alternatives and provides interpretable summaries.

### Strengths
- The proposed model addresses the challenge of cross-sectional modeling.

- Linear scaling in the number of units is a practical advantage.

- The expressivity result under exchangeability maps naturally to factor-model intuition: cross-sectional moments summarize latent common variation.

### Weaknesses
- The assumption of exchangeability is strong, which may not generally hold in real markets across all periods.

- It’s unclear how stable the model is across market regimes, such as bull vs. bear and calm vs. crisis. Performance may be different at distinct periods and events.

### Questions
- Does the permutation-invariant property hold across common financial time series and data in other domains?

- How does the set summary learned by the model relate to known economic structures?

- What is the model’s performance across market regimes?

- Do conclusions hold using different window sizes and set-summary sizes?

### Soundness
2

### Presentation
2

### Contribution
2
