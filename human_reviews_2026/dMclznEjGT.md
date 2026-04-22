# ReNF: Rethinking the Design Space of Neural Long-Term Time Series Forecasters

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
Neural Forecasters (NFs) are a cornerstone of Long-term Time Series Forecasting (LTSF). However, progress has been hampered by an overemphasis on architectural complexity at the expense of fundamental forecasting principles. In this work, we return to first principles to redesign the LTSF paradigm. We begin by introducing a Multiple Neural Forecasting \textcolor{red}{Proposition that provides a theoretical motivation} for our approach. We propose Boosted Direct Output (BDO), a novel forecasting paradigm that synergistically hybridizes the causal nature of Auto-Regressive (AR) models with the stability of Direct Output (DO). In addition, we stabilize the learning process by smoothly tracking the model's parameters. Extensive experiments show that these principled improvements enable a simple MLP to achieve state-of-the-art performance, outperforming recent, complex models in nearly all cases, without any specific considerations in the area. Finally, we empirically verify our proposition, establishing a dynamic performance bound and identifying promising directions for future research. The code for review is available at: \url{https://anonymous.4open.science/r/ReNF-A151}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ReNF, a simple MLP-based model for long-term time series forecasting (LTSF). It introduces the Multiple Neural Forecasting Theorem (MNFT) and a Boosted Direct Output (BDO) strategy that combines auto-regressive and direct forecasting benefits. ReNF achieves state-of-the-art performance on major benchmarks with fewer parameters, showing that strong forecasting principles can outperform complex architectures.

### Strengths
1. The motivation is clear, it aims to simplify the problem while providing solid theoretical justification.
2. The experiments are comprehensive, with representative baselines and strong overall performance.

### Weaknesses
1. Theorem 1 claims that an accurate forecast is theoretically achievable even with a rather weak generator. Could the authors clarify how this insight is formally derived from the theorem’s assumptions?

2. Why would combining multiple methods yield more accurate results than any single candidate? Please provide either theoretical or empirical justification.

3. The paper introduces a new training strategy, BDO, but this raises concerns about computational cost. Does the “Efficiency” metric in Table 1 refer to training or inference (testing) time?

4. The paper lacks results averaged over multiple random seeds. Including the standard deviation would make the evaluation more statistically reliable.

### Questions
see weaknesses

### Soundness
3

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
4

### Summary
In this paper, authors provide a new framework based on neural forecasters. Based on the new introduced multiple neural forecasting theorem, it provides both theoretical and experimental support for the effectiveness of this model.

### Strengths
1. The comprehensiveness of Multiple Neural Forecasting Theorem and experimental discussion. 
2. This model has a better performance than many state-of-art baselines.
3. There is a relatively comprehensive discussion about the EMA and BDO in the ablation study section.

### Weaknesses
1. Can authors provide an explanation about the existence of empirical bound? Can we use more advanced model to have a much better performance, or this bound is unsolvable?
2. Can this model combine with other TSF structures, such as transformer?
3. Why adopted both deterministic and frequency based loss for this model? Are there any theoretical support?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper targets long-term time-series forecasting (LTSF), arguing that progress has over-focused on architecture tricks while under-using supervision and stable training. It introduces a Multiple Neural Forecasting Theorem (MNFT): with multiple forecast candidates from a neural forecaster, an (oracle) combination can bound error. A crude upper bound scales with horizon, data range, and number of candidates. Assumptions include bounded data, multiple candidates, independent output elements, and bounded bias. Paper proposes Boosted Direct Output (BDO): recursively forecast longer segments by concatenating previous (shorter-horizon) predictions to the inpu, mixing AR-like causality with DO’s low error propagation. Trains an MLP with a stage-wise, time+frequency loss, heavier weight on earlier (shorter) stages. Uses EMA "shadow" weights for evaluation to smooth validation/test curves and avoid unstable early stopping.

### Strengths
- Reproducibility checklist and code provided

- Theoretical results are presented

- BDO recipe that adds causal structure to DO without heavy architecture

### Weaknesses
- The focus on long horizon problems undermines the claims of approach's generality

- Theoretical assumptions A1-A4 are not motivated and the significance of Theorem 1 is unclear

- BDO strategy, which seems to be one of core contributions seems trivial, if it is not accompanied with additional block-level supervision. The latter is not described in the paper in any level of detail at all.

- EMA is not novel

### Questions
> Furthermore, the field’s focus has shifted towards designing specialized modules for properties that
are believed to be beneficial for LTSF, such as multi-scale (Wang et al., 2024b) and non-stationary
(Liu et al., 2022). However, the progress has become erratic because these architectural additions
often yield subtle gains while overlooking fundamental principles.

- I actually like this reflection a lot. However, I feel that the authors fall in the same trap as many previous works: by focusing on a subset of problems represented only by long-horizon benchmarks, authors fail to answer the question about generality of their approach and its applicability to wide array of problems. Could you please provide additional results on M4, M3 and TOURISM benchmarks? I have observed in the past that the works focusing on long-horizon benchmarks fail to deliver any tangible gains on other benchmarks, resonating with the authors' original sentiment.

- The practical applicability of theoretical assumptions A1-A4 is not clear. Could you please provide a discussion for each, linking it to the practice of forecasting? I am especially interested in A3, how does this work in practice? For example, if we generate each sequence from the same architecture weights, how can each element in an output series be independent?


- Can you explain the significance of Theorem 1? The only takeaway I get from it is that no matter how many candidates $c$ are generated, there is an irreducible error bound related to $b$ and $\sigma_t$, which is arbitrarily loose. In my view, this is trivial and does not provide insight on how the proposed MNFT can improve on top existing approaches.

> Furthermore, higher-quality historical data and a more powerful NFM (which reduces the estimation
error b) both serve to constrain the solution space, leading to better predictions. 

- I do not understand how a more powerful NFM should necessarily lead to the reduction of estimation
error $b$. What is the mathematical principle here and how is this related to overfitting? This seems like a rather hand-wavy argument to me.

- I do not understand the significance of Figure 2 and I do not believe it is referenced/discussed in text.

- BDO strategy, which seems to consist in combining the multi-horizon forecast with AR forecast by feeding generated chunks back into the input is a standard practice in today forecasting. For example, Chronos does it. Could you please explain the novelty in more detail? For example, the key contribution behind BDO seems to be the supervision of each output $k$ (e.g. Figure 4), but this supervision scheme is not described mathematically in Section 2.5. Could you please provide the exact formula for BDO loss? Additionally, the BDO supervision is not ablated. What happens if block level supervision is replaced with the supervision of the final output only?

- EMA of model weights is not novel. It is a well-established training/evaluation trick (a.k.a. Polyak/EMA averaging) used for decades and popularized across modern deep learning (e.g. attention is all you need, diffusion models, and many vision/NLP training recipes). Using EMA can absolutely help stability and accuracy, but it should not be positioned as a core contribution. Also, when comparing to other models as baselines, those models should be trained with EMA as well.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles long-term forecasting by revisiting simple architectures rather than increasing model complexity. The authors argue that well-selected MLPs can match or exceed the performance of recent long-horizon forecasting models. They introduce the Multiple Neural Forecasting Theorem (MNFT) to guide model complexity selection.

However, MNFT provides little novel insight. MNFT essentially restates the law of large numbers under unrealistic independence assumptions for forecast outputs. The theoretical contribution feels forced and disconnected from practical forecasting settings.

### Strengths
1. The paper provides valuable empirical evidence that many recent long-horizon models underperform compared to carefully tuned MLPs, reinforcing skepticism toward architectural inflation in this area.

2. The authors attempt to provide a theoretical framework for ensemble-style reasoning about neural forecasting models, which is a valuable direction.

### Weaknesses
1. The evaluation relies exclusively on overused long-horizon benchmarks (ETT, Weather, Traffic, Electricity). These datasets are heavily saturated, often exhibit weather-driven dynamics, and are not suitable for assessing general forecasting capability. Accurate weather-related forecasts are known to degrade sharply beyond 2–3 weeks due to chaos in atmospheric systems. To strengthen their claims, the authors should test on broader benchmarks, such as M1, M3, or M4, which are more diverse and less prone to overfitting.

2. The exposition of MNFT is opaque. Figure 1 and its caption are confusing and should be rewritten for clarity.

3. The notation and assumptions in MNFT are inconsistent and often unjustified:

    3.1 Defining “forecasting machines” via tuples is unnecessarily convoluted.

    3.2 Assuming finite means and variances excludes heavy-tailed distributions.

    3.3 Assumption A3 (independent forecast generation) is unrealistic in both direct and multi-step setups.

    3.4 Assumption A4 (bounded expected error leading to bounded forecast error) is tautological and makes the theorem effectively vacuous. Also makes MNFT an LLN-like result. I want to remind the authors that the independence assumptions in the time series domain do not hold.

4. The paper omits key related work, notably NHITS (Neural Hierarchical Interpolation for Time Series), which provides a stronger theoretical connection between network capacity, basis expansions, and approximation guarantees.

### Questions
1. Given the large number of available baselines in long-horizon forecasting, why were comparisons limited to DUET, TimeDistill, Timer-XL, iTransformer, TimeMixer, PatchTST, Crossformer, and DLinear? Are these truly the most competitive methods?

2. Typo on line 098 (page 2): the goal should be to approximate Y_f, not X_f.

### Soundness
2

### Presentation
1

### Contribution
1
