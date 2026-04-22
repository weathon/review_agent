# Binning of Linear Models for Time Series Forecasting

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Linear models have been gaining attention in the time series forecasting (TSF) field, as several variations of modified linear layers have shown excellent performance across benchmarked datasets. This work extends these efforts by exploring the application of the binning technique to linear models. The conceptual rationale is based on the idea that binning can serve as a simple means of efficient learning through isolating temporal patterns. Instead of relying on temporally adjacent observations, binning adopts another perspective by exposing binned linear layers to periodically grouped data, treating temporal neighbors independently. Both conceptual analysis and empirical experimentation are conducted, with the results demonstrating that this modification can lead to improvements in prediction error. This work positions binning as a simple and effective technique, contributing to the exploration of representations structured by periodicity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a binning technique for time series forecasting that partitions the lookback window based on periodicity rather than temporal adjacency. Each bin contains data points that are w time steps apart (e.g., all same-hour observations for hourly data with daily period), and is processed independently by a separate linear layer. The authors provide theoretical analysis showing that binning eliminates seasonal variation within bins and reduces parameters from LH to LH/w. The technique is validated on synthetic data and five benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, Electricity) using four linear model variants.

### Strengths
The periodicity-based binning technique they provided sounds interesting and novel. Unlike some previous value-based discretization (e.g., https://arxiv.org/pdf/2005.10111), this work groups data by temporal phase within periods, which is conceptually intuitive for seasonal time series.
They provided strong theoretical foundations for their method as well as empirical validation in both synthetic datasets and real benchmarks. 
They made a comprehensive analysis for each experiment, especially the weight visualization, which provides useful insights.

### Weaknesses
1. Although the binning technique in this work is novel (as far as I know), there were some other related works (like https://arxiv.org/pdf/2005.10111) that also use the 'binning' concept in a different way. The authors should make more comparisons with such related works.
2. They made some theoretical claims that were not thoroughly validated in experiments. For instance, they claimed a significant parameter efficiency ($LH \to LH/w$), but why is it not reflected in the experiments? Did the binning use less parameters compared to the baselines? Also, they claimed the method can be applied to any model, but they only did experiments on linear ones.
3. They made strong assumptions in their theoretical part, for example they assumed there is only one single period and it is perfectly periodic throughout the entire time series, which should be more carefully dealt with (for example in AutoFormer they only required approximate periods and learns it). Also they assumes the period $w$ is known, which is acceptable in real cases but should be more clearly stated.
4. Their method consistently fails in Etth2 (and for some models in some other datasets). They made a superficial discussion in the Appendix but it would be better if they can analyze deeper, for example why binning fails here and in what cases should be use binning.
5. More ablation studies on experiment hyperparameters should be done, and they should do more comparisons with other methods instead of only the linear models themselves.

### Questions
I think there are many interesting ideas in this paper which are underdeveloped, and there are some other concerns. If the author manage to address some of these concerns I think this would significantly improve the quality of this paper (and of course I'd considering giving higher score).
1. As mentioned in 'weaknesses', was the parameter efficiency reflected in your experiments? If not, why not?
2. Can you test the method on non-linear models? Is it also effective?
3. Can you explain what may happen if the assumptions are slightly imperfect? For example what will hapen if you got a wrong $w$? How can you handle if their are multiple seasonalities?
4. You only tested the $b=w$ case but left other cases as future works. Can you briefly justify this (for example why not $b=w/2$ or $b=2w$) either theoretically or empirically?
5. As mentioned in 'weaknesses', the binned models consistently fail on ETTh2 across all model types, but only a superficial discussion was made, which is unsatisfactory. Have you investigated why? Does ETTh2 violate your periodicity assumptions? Would using a different period w or number of bins b help? Understanding this failure would strengthen the paper by clarifying when binning should/shouldn't be used. 
6. Section 2 discusses SparseTSF in detail and notes it also uses periodic sampling, but you never provide a direct empirical comparison. Could you include SparseTSF results in Table 1 to show the marginal contribution of your binning approach compared to this closely related method?
7. Tables 1-2 show that global relative binning (grb) generally outperforms local absolute binning (lab), but the reason is not clearly explained. Is this because grb provides better scale normalization, or because it allows better cross-series learning? Could you provide more analysis on when each strategy is preferable?
8. Figure 2 (synthetic experiments) shows binned models perform best at high TO ratios (oscillation-dominated). Can you characterize which real datasets have high vs. low TO ratios? Does this explain why binning works well on some datasets (ETTm2) but fails on others (ETTh2)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In the paper under review, a binning method for linear models for time series forecasting is proposed, where the binning is used to capture the periodic / seasonal part of the time series.

### Strengths
1) The paper is well written and easy to follow.
2) The approach is simple and can be applied to any linear forecasting model.

### Weaknesses
1) The paper makes strong assumptions about the structure of the time series. It is assumed that the (original) time series model decomposes in a seasonal and a trend components. This only reflects a small subset of time series models.

2) The paper leaves many aspects unclear. What is the purpose of the bin function f(.)? Is it implicitly assumed that w < L holds? How are f, b, and w selected in the experiments?

3) The experiments are limited. The used datasets are rather simple with a low sample frequency. There is data with high frequency from for instance the manufacturing domain, e.g., RUL forecasting. Further, the benefits of the binning are very mixed.

4) Overall the contributions of this paper are too limited. It feels more like a best practice addon to linear models, which should be better presented as workshop contributions rather than contributions to a large AI/ML conference.

### Questions
Can the binning also applied to nonlinear models like transformers?

What is the purpose of the bin function f(.)? Is it implicitly assumed that w < L holds? How are f, b, and w selected in the experiments?

### Soundness
1

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
The paper explores the idea of applying binning for time-series forecasting.

### Strengths
The perspective is somewhat novel and provides an interesting way to think about different way of representing temporal periodicity.

### Weaknesses
**Major Comments**

1. The assumption in Line 113 is too strong and impractical. How can one know the period $w$ in advance and set the number of bins accordingly? Moreover, how would the method handle time series that exhibit multiple or varying periods?

2. In the methodology section, it is unclear what it means to “stack” the predictions of different bins. Does this simply mean summing them, or is there another aggregation mechanism?

3. The experimental evaluation is very limited. Many recent and representative time-series forecasting models after DLinear and RLinear are missing from the baselines. In addition, the experiments use only ETT and Electricity datasets, which makes the evaluation insufficiently comprehensive.

4. The paper does not explain how the number of bins is determined. Furthermore, it would be important to show how performance changes as the number of bins varies.

**Minor Comments**
5. The paper should at least cite the baseline models NLinear, DLinear, and RLinear.

### Questions
1. Clarify the assumption regarding the period ($w$).
Does the method assume that the period is known in advance, or does it use an arbitrary or learned value? Please make this explicit and discuss how the approach handles varying or unknown periods. (W1, W4)

2. Clarify the meaning of “stacking” in the method.
It is unclear what “stack” precisely means in this context - does it refer to summing, concatenation, or another form of aggregation? Please specify this in the methodology section. (W2)

3. Expand and strengthen the experimental evaluation.
Consider including more recent and representative baselines as well as additional datasets to demonstrate the robustness and generalizability of the proposed approach. (W3)

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
3

### Summary
- The paper proposes a binning scheme that partitions the input window by phase with respect to an assumed period and trains per-bin linear mappings; horizon forecasts are formed by concatenating bin outputs.

### Strengths
- Clear seasonal/trend decomposition argument; bin-wise constant seasonal offset is intuitive
- The method is simple and portable, easy to reproduce and reason about.

### Weaknesses
- Even if bins match the true phase, the model forbids dependencies whose effective lags aren’t exact multiples of the period (e.g., holiday spillovers), so cross-phase signals alias or vanish. This induces approximation bias unless you add cross-bin coupling/regularization or a residual path that models inter-bin interactions.
- The narrative leans on rapid training-loss convergence on synthetic TO cases; this is not equivalent to systematic test improvement on real data (esp. in non-stationary time-series forecasting task)

### Questions
- Can you provide a diagnostic for when binning helps? E.g., estimate a trend-to-oscillation indicator on real data and correlate it with gains/losses (why ETTh2 drops)?
- What happens when the period is wrong, drifting, or multi-periodic? Can you show w!=b, learned/non-uniform bins, or online re-alignment?

### Soundness
2

### Presentation
3

### Contribution
2
