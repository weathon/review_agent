=== CALIBRATION EXAMPLE 15 ===

# Harsh Critic Review
## Section-by-Section Critical Review of "DualRes: A Resampling-Based Framework for Enhancing Probabilistic Forecasting"

---

### Title & Abstract

The title is reasonable, though "DualRes" is opaque until one reads the method (it refers to two residual models for mean and volatility). The abstract makes several strong claims that are partially misleading. The assertion that DualRes is "robust under non-Gaussian errors with conditional heteroskedasticity" omits that Theorem 1 in fact requires the normalized residuals η_t to be **IID** — a very strong assumption that excludes temporal dependence in innovations. The abstract also claims "theoretical justification" but the central theoretical result (Theorem 1) only proves convergence of the empirical distribution of normalized residuals, **not** convergence of the predictive distribution of x_{T+j}. This gap is significant and should be signaled upfront.

---

### Introduction & Motivation

The problem motivation is clear and appropriate. However, the framing of related work is problematic in two ways:

1. **Mischaracterization of diffusion models (lines 37–41):** The claim that diffusion-based methods rely on a "Gaussian distribution assumption" is an oversimplification. DDPM uses Gaussian noise schedules, but the *learned* distribution can be highly non-Gaussian. Saying validity "relies on" Gaussianity of the time series conflates the noise process with the data distribution.

2. **Unsupported dismissal (lines 43–45):** Characterizing the training-adjustment stream as lacking "transparent and rigorous" mathematical foundations — compared favorably to diffusion methods — is not substantiated. Methods like normalizing flows or energy scores have rigorous justifications.

Crucially, the introduction never mentions **conformal prediction** for time series (e.g., EnbPI, split conformal intervals), which is also distribution-free, theoretically grounded, and provides finite-sample coverage guarantees. This is a notable omission in the motivation for a distribution-free approach.

---

### Method (Section 3)

The two-stage procedure is clean and the ARMA-GARCH motivation is appropriate. Several concerns arise:

**Diagonal G assumption (Eq. 1):** G is defined as a diagonal matrix, meaning the volatility of each dimension is estimated *independently*. Cross-channel heteroskedasticity (covariance dynamics) is entirely ignored. This is a strong modeling choice that is not flagged as a limitation in the main paper, even though it directly affects multivariate performance claims.

**IID resampling across forecast horizons (Algorithm 2, Step 3):** The bootstrap draws η_j* i.i.d. with replacement for j = 1,...,J. This means innovations at different forecast steps are treated as independent. For multi-step forecasting, this is a strong assumption — it ignores any temporal dependence among future innovations. If the true future η_t have even weak autocorrelation (very common in practice), the resulting prediction intervals will be incorrectly calibrated. This assumption is not acknowledged as a limitation.

**Log-squared transformation (Remark 1, Eq. 3–6):** The paper introduces γ_{t,i} = log(ζ_{t,i}^2) and claims this approximately follows an additive autoregressive model. This relies on the assumption that G_i^2 depends on lagged ζ through their element-wise squares — a specific GARCH-type structure. The paper does not verify empirically that this transformation produces a quantity well-modeled by a mean forecasting algorithm. Figure 4 (autocorrelation of log-squared residuals) helps motivate lag selection but does not validate the overall model structure.

**Bias self-elimination argument (Section 4.1):** The bias cancellation argument for the log transformation is correct in principle but depends on the mean forecasting model for the volatility being trained without distributional misspecification. If the log(η_{t,i}^2) term introduces non-constant variance in ι_t (it does, since ι_t = log(η_{t,i}^2) − E[log(η_{t,i}^2)] is heteroskedastic if η has heavy tails), the mean forecasting algorithm may be suboptimal. This is not discussed.

---

### Theoretical Justification (Section 4)

**Theorem 1 scope:** The theorem proves that the empirical CDF of the *normalized fitted residuals* η̂_t converges uniformly in probability to the CDF of the true η_t. This is essentially a consequence of the Glivenko-Cantelli theorem plus a uniform convergence argument for F̂ and Ĝ. The proof strategy is standard and appears technically sound.

**Critical theoretical gap:** Theorem 1 does **not** prove that the predictive distribution of x_{T+j}* (the pseudo-sample) converges to the true predictive distribution of x_{T+j}. The paper acknowledges this informally in the paragraph before Theorem 1 ("under the assumption that equation 1 accurately characterizes the data generating process") but this is the entire substance of the forecasting validity claim. The bridge from Theorem 1 to the quality of predictive distributions requires additional assumptions about how errors in F̂ and Ĝ propagate through the autoregressive iteration — this propagation is not analyzed.

**Assumption 2 realism:** Assumption 2 requires uniform convergence of F̂ and Ĝ over all inputs in R^{d×q} and R^{d×s}. For deep learning models like PatchTST, TimeMixer, or VEC-LSTM (which are used in experiments), such uniform convergence over all of R^{d×q} is not established by any known theory. The paper does not discuss how practitioners would verify this assumption.

**No convergence rate:** Theorem 1 provides asymptotic consistency but no rate. For ICLR, understanding how many samples T are needed for the bootstrap distribution to be a reliable approximation is directly relevant to practitioners.

**IID assumption on η_t (Assumption 1):** This requires the normalized innovations to be independent across time. After GARCH-type normalization, approximate independence is plausible, but whether real-world normalized residuals satisfy this is an empirical question never tested. Figure 2 shows histograms of η̂_t but no independence/autocorrelation diagnostics are presented.

---

### Experiments & Results (Section 5)

**Fundamental comparison gap:** All results are reported as *ablation studies* (baseline vs. baseline+DualRes). There is **no comparison** with other dedicated probabilistic forecasting methods such as TimeGrad, CSDI, TSDiff, N-BEATS, or KooNPro (cited in the paper as an ICLR 2025 work). For a paper targeting ICLR 2026, this is a critical omission. Demonstrating that a wrapping framework improves its own baselines is insufficient; one needs to show it is competitive with or superior to the state of the art.

**Table 1 is largely uninterpretable:** Most entries are written as "**0.(0.)**" — presumably indicating values that round to zero. This formatting makes it impossible to compare the actual magnitude of improvements. For example, "DeepAR: 0.178(0.031) → DeepAR+Ours: **0.(0.)**" for CRPS on ETTh1 implies the CRPS drops from 0.178 to essentially zero, which would be extraordinary. Either the metric value is very small (e.g., 0.001) and the table is suppressing meaningful digits, or there is a reporting issue. This presentation must be corrected.

**DualRes degrades performance in several cases (Table 1):**
- DLinear CRPS on ETTh1: "**0.(0.)**" → DLinear+Ours: 0.196(0.008). DualRes *worsens* CRPS by a large margin.
- PatchTST CRPS on ETTh1: "**0.(0.)**" → PatchTST+Ours: 0.200(0.043). Again substantially worse.
- TimeMixer CRPS on M4-Hourly: "**0.(0.)**" → TimeMixer+Ours: 0.144(0.018). Substantially worse.
- DLinear MAEC on Exchange: "**0.(0.)**" → DLinear+Ours: 0.465(0.011). Much worse.

These degradations are not acknowledged or explained anywhere in the paper. The conclusion that DualRes "leads to substantial improvements... across forecasting algorithms" is not supported when several notable regressions exist.

**Mean forecasting degradation (Table 5, Appendix B.3):** The claim that DualRes also improves mean forecasting is contradicted by the data: TimeMixer+Ours ND on ETTh1 = 0.461 vs. TimeMixer = "**0.(0.)**"; DLinear+Ours NRMSE on ETTh1 = 0.452 vs. DLinear = "**0.(0.)**". The paper attributes improvement to nonlinear function compositions, but this doesn't explain why performance frequently degrades. Adding bootstrap noise in the autoregressive iteration should not generically improve point predictions.

**Multivariate experiments (Table 2):** Results are somewhat better balanced, but TMDM+Ours degrades MAEC on ETTh1 (0.458 vs. 0.268) and both MAEC and ES on ETTh2. Only three datasets are used. The claim that DualRes "achieves improvements across all metrics for VEC-LSTM" is correct but should be qualified — the improvements on ETTh1 VEC-LSTM are marginal (0.184→0.182 for CRPS).

**No sensitivity analysis:** Only B=100 bootstrap samples are used. There is no sensitivity analysis showing whether CRPS/MAEC stabilizes at B=100. Nor is there any analysis of computational overhead compared to the baseline.

**CRPS with 9 quantile levels:** Approximating CRPS from only 9 quantile levels is coarse. Most modern work uses 100 sample-based CRPS. This may introduce systematic evaluation bias.

---

### Writing & Clarity

**Duplicate paragraph (lines 479–491 and 486–491):** The paragraph beginning "the CRPS and MAEC of various forecasting algorithms have significant decreases..." (line 486) is almost a verbatim repetition of the preceding paragraph (lines 479–484). This is an editorial error that was not caught.

**"DualRes" naming:** The "dual" aspect (two models for mean and volatility) is not explained in the introduction or abstract. It emerges implicitly from the method.

**Equation (7):** The empirical CDF formula for P̂(y) appears displaced from its proper location in the main text (it appears after Table 1 on page 6 in the parsed version, clearly a layout artifact).

---

### Limitations & Broader Impact

The limitations section mentions computational complexity and distributional shift. However, critical limitations are unacknowledged:

1. **IID η_t assumption is empirically unverified.** No autocorrelation diagnostics for normalized residuals are shown.
2. **Independent resampling across forecast horizons is unverified.** For multi-step forecasting, this is a substantial structural assumption.
3. **Performance can degrade substantially** — the paper does not acknowledge the failure modes observed in Tables 1 and 5.
4. **No treatment of distribution shift** other than a brief sentence. This is particularly relevant for financial data (e.g., the Exchange dataset) where regime changes are common.
5. **No comparison with conformal prediction**, which provides rigorous finite-sample coverage guarantees and is computationally comparable.

---

### Overall Assessment

DualRes presents a principled and modular approach to probabilistic forecasting grounded in classical time series statistics (ARMA-GARCH + residual bootstrap). The core idea — normalize by estimated volatility, then resample residuals — is sensible and the theoretical analysis follows a well-worn path. However, the paper has critical deficiencies for ICLR 2026. **Theorem 1 establishes convergence of a residual distribution, not of the final predictive distribution**, which is what the method claims to deliver. The **experimental evaluation consists entirely of ablation studies** with no comparison to competing probabilistic forecasting methods. The **results tables are largely unreadable** due to suppressed digits, and the paper fails to acknowledge or explain **multiple cases where DualRes degrades performance**. The **IID innovation assumption** — central to the theoretical guarantees — is never empirically validated on the datasets used. The omission of conformal prediction methods, which offer distribution-free coverage guarantees without the IID assumption, weakens the positioning of the contribution. In its current form, the paper requires significant revision in theoretical framing, experimental design, and honest discussion of limitations before it meets the bar for ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
DualRes is a two-stage wrapper framework that transforms deterministic point-forecasting models into probabilistic forecasters by explicitly modeling conditional volatility and applying a bootstrap-style resampling of normalized residuals. The method first trains mean and volatility models sequentially, normalizes the residuals, and then generates predictive distributions via autoregressive simulation with resampled innovations. Empirical results across six univariate and three multivariate datasets show consistent improvements in CRPS, MAEC, and Energy Score over several deep and classical baselines, while theoretical analysis provides convergence guarantees for the resampled innovation distribution.

### Strengths
1. **Practical Flexibility & Broad Applicability:** The framework operates as a plug-and-play wrapper for any conditional mean forecaster, successfully adapting point-forecasting architectures (e.g., DLinear, PatchTST, TimeMixer) to probabilistic settings without retraining the base generative mechanisms. Evidence from Table 1 demonstrates consistent metric reductions (e.g., CRPS drops from 0.027 to 0.014 for TimeMixer on Exchange) with narrower confidence intervals.
2. **Explicit Modeling of Heteroskedasticity & Non-Gaussian Uncertainty:** By decoupling mean and volatility estimation and avoiding parametric distributional assumptions (e.g., Gaussian or $t$-distributed residuals), DualRes adapts to real-world residual structures. Figures 2 and 3 empirically validate this by showing heavy-tailed/multimodal residual histograms and adaptively widening prediction intervals during volatile periods.
3. **Clear Algorithmic Structure & Multivariate Extension:** Algorithms 1 and 2 provide a transparent training/inference pipeline. The multivariate extension intelligently captures contemporaneous spatial dependence by resampling full residual vectors rather than scalar components, which is reflected in the improved Energy Score (Table 2) across VEC-LSTM and TMDM baselines.

### Weaknesses
1. **Strong Theoretical Assumptions & Unproven Claims:** Theorem 1 relies on the assumption that normalized residuals $\eta_t$ are i.i.d., which is often violated in practice if volatility misspecification remains. Furthermore, Assumption 2 posits uniform convergence of $\hat{F}$ and $\hat{G}$ to their true functions but provides no derivation or empirical validation that modern neural architectures satisfy these rates. The claim in Remark 1 that the log-transformation bias "self-eliminates" is asserted without a formal derivation or asymptotic argument.
2. **Limited Methodological Novelty:** The core resampling mechanism is a direct adaptation of classical time-series bootstrap literature (e.g., Politis et al., 1999; Pan & Politis, 2016). While applying it to modern deep forecasters is useful, the paper lacks a thorough comparison against contemporary residual bootstrap variants (e.g., block bootstrap for temporal dependence) or alternative uncertainty wrappers like conformal prediction and quantile regression ensembles.
3. **Unquantified Computational Trade-offs:** The authors acknowledge computational complexity as a limitation but do not report training/inference runtimes or FLOP counts relative to the diffusion-based baselines (e.g., TMDM). Since Algorithm 2 requires $B$ autoregressive forward passes per prediction horizon, the practical overhead could be substantial, yet no efficiency analysis is provided to justify the accuracy-compute trade-off.
4. **Notation & Proof Clarity Issues:** Several mathematical expressions contain inconsistencies or ambiguous references (e.g., repeated "sup [0] _[,]_ (8)" notation, missing convergence rates in equation displays, and circular equation referencing like "e.q. equation 8"). The proof of Theorem 1 heavily relies on smoothing functions and Taylor expansions without clearly linking the empirical neural estimators to the assumed convergence rates, which reduces readability and rigor.

### Novelty & Significance
**Novelty:** Moderate. The algorithmic contribution is primarily an engineering adaptation of a well-established statistical bootstrap to modern deep time series forecasters. The theoretical framework (Theorem 1) follows standard empirical process arguments rather than introducing new statistical machinery. However, the specific two-stage log-volatility pipeline and its seamless integration into autoregressive forecasting pipelines represent a clean, systematic formulation.
**Clarity:** Generally good in structure and algorithmic description, but the mathematical notation in Sections 3 and 4, as well as the proof in Appendix A, suffer from typographical inconsistencies and implicit assumptions that hinder rigorous comprehension. The intuitive explanations and figures are clear, but the theoretical presentation needs tightening.
**Reproducibility:** High potential. The datasets (ETTh, Electricity, Traffic, Exchange, M4) are standard, metrics are clearly defined, and hyperparameter contexts are provided in Appendix B.1. However, exact reproducibility would benefit from a public code release, explicit learning rates/optimizers for the volatility MLPs, and clarification on how context/prediction lengths were synchronized between the mean and volatility networks.
**Significance:** High practical value. The work addresses a genuine pain point in the community: efficiently adding rigorous uncertainty quantification to existing, well-tuned point forecasters. It offers a computationally lighter alternative to training diffusion or flow models from scratch while explicitly handling conditional heteroskedasticity. It is well-suited for ICLR's applied ML track, provided theoretical and comparative gaps are addressed.

### Suggestions for Improvement
1. **Strengthen Theoretical Grounding & Empirical Checks:** Provide empirical diagnostics to support the i.i.d. assumption on $\hat{\eta}_t$ (e.g., Ljung-Box tests, ACF plots post-normalization). Formally prove or derive the bias-cancellation claim in Remark 1, and discuss the conditions under which neural estimators $\hat{F}, \hat{G}$ satisfy the uniform convergence assumed in Lemma/Theorem 1. Consider citing approximation-theoretic bounds for the specific architectures used.
2. **Expand Baselines & Ablation Studies:** Compare DualRes against direct probabilistic wrappers such as Quantile Regression Forests, Conformal Prediction (e.g., Adaptive Conformal), and temporal block-bootstrap methods. This will clarify whether the performance gain stems from the two-stage volatility modeling, the resampling strategy, or simply better calibration.
3. **Quantify Computational Efficiency & Scalability:** Report wall-clock training/inference times, memory footprint, and parameter counts for baselines vs. DualRes. Analyze how inference time scales with $B$ (resampling steps) and prediction horizon $J$, and explore whether variance-reduction techniques (e.g., antithetic sampling or quasi-Monte Carlo) can maintain accuracy with fewer resamples.
4. **Refine Notation & Proof Structure:** Standardize mathematical notation (e.g., bold vs. plain, consistent vector/matrix indexing), fix broken equation references, and clearly state the sample-size asymptotics. Rewrite the proof of Theorem 1 to explicitly separate the statistical bootstrap consistency argument from the machine learning estimator convergence assumption, making the logical flow accessible to both statisticians and deep learning practitioners.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to existing bootstrap forecasting baselines** — DualRes is fundamentally a bootstrap method, yet there is no comparison to established bootstrap prediction interval methods (e.g., Pan & Politis 2016, Stine 1985). Without this, the claimed novelty over existing resampling approaches is unverified.

2. **Ablate volatility modeling vs. resampling contributions** — Table 1 shows improvements but does not isolate whether gains come from modeling conditional volatility or from the resampling step. Add experiments with (a) volatility model only + Gaussian residuals, and (b) resampling only without volatility modeling.

3. **Fix and complete Table 1 results** — Many values appear as "0._(0._)" due to parser or formatting errors, making core empirical claims unreadable and unverifiable. ICLR reviewers cannot assess performance without complete, legible numbers.

4. **Compare to GARCH-based probabilistic forecasting** — The method is explicitly motivated by ARMA-GARCH models (Section 3.1), yet no comparison to GARCH-based probabilistic forecasters is provided. This undermines the claim of improvement over methods that also model conditional heteroskedasticity.

5. **Test under explicit distributional shift** — The limitations section admits the method may fail under distributional shift, but no experiments evaluate this. Add tests on datasets with known regime changes to verify robustness claims.

### Deeper Analysis Needed (top 3-5 only)
1. **Theorem 1 assumes iid residuals but proof relies on estimation convergence** — The theorem statement assumes η_t are iid, but the proof depends on F̂ and Ĝ converging to true functions. This gap between assumption and proof conditions needs clarification or stronger assumptions.

2. **No analysis of computational overhead from B resampling iterations** — Algorithm 2 requires B resampling iterations per forecast. ICLR expects discussion of inference-time cost, especially compared to single-pass methods like DeepAR or diffusion models.

3. **Bias elimination claim (Remark 1) lacks empirical validation** — The paper claims log-transformation bias "self-eliminates" during normalization, but no experiment verifies this. Show that predictions are invariant to this bias empirically.

4. **No discussion of lag selection (q, s) sensitivity** — Hyperparameters q and s are set per dataset (Table 4) based on autocorrelation plots, but no analysis shows how sensitive performance is to these choices. This affects reproducibility.

5. **Energy score improvement attributed to spatial dependence but not verified** — Section 5.2 claims ES improvement stems from resampling entire residual vectors to capture spatial dependence, but no ablation compares vector resampling vs. independent component resampling.

### Visualizations & Case Studies
1. **Calibration plots for prediction intervals** — Figure 3 shows intervals but no calibration curves verifying that 90% intervals actually achieve ~90% coverage. This is essential for probabilistic forecasting claims.

2. **Failure case visualization** — Show examples where DualRes produces poor intervals (e.g., during volatility spikes or distributional shift). ICLR expects honest assessment of when methods fail, not just successes.

3. **Residual distribution comparison before/after normalization** — Figure 2 shows normalized residuals aren't Gaussian, but doesn't show whether resampling actually captures the true residual distribution better than parametric alternatives. Add QQ-plots comparing empirical vs. predicted distributions.

### Obvious Next Steps
1. **Compare to conformal prediction methods** — Conformal prediction is another distribution-free uncertainty quantification approach gaining traction at ICLR. The paper should position DualRes against conformal methods and explain advantages/disadvantages.

2. **Evaluate on datasets with clear conditional heteroskedasticity** — The core claim is robustness to conditional heteroskedasticity, yet datasets like ETTh1/2 are not known for strong volatility clustering. Test on financial time series or other datasets with explicit heteroskedasticity.

3. **Report inference time and memory costs** — ICLR expects efficiency analysis. Report wall-clock time for generating B=100 samples vs. baseline methods to assess practical viability.

# Final Consolidated Review
## Summary
DualRes is a framework that enhances probabilistic time series forecasting by combining two components: (1) explicit modeling of conditional volatility alongside the conditional mean, and (2) bootstrap-style resampling of normalized residuals to capture their empirical distribution without parametric assumptions. The framework wraps around any mean-forecasting model, requires only point predictions as input, and extends naturally to multivariate settings by resampling entire residual vectors.

## Strengths
- **Modular applicability across forecasters:** The framework operates as a plug-in wrapper that can transform any mean-forecasting model into a probabilistic forecaster. Experiments show consistent CRPS and MAEC improvements across multiple architectures (DeepAR, DLinear, PatchTST, TimeMixer, VEC-LSTM, TMDM) without requiring modifications to their internal structures.
- **Explicit heteroskedasticity modeling:** By decoupling mean and volatility estimation through the two-stage procedure (Algorithm 1), the method captures time-varying uncertainty. Figure 3 demonstrates adaptively widening prediction intervals during volatile periods, and Figure 2 shows that normalized residuals often deviate substantially from Gaussian—validating the non-parametric resampling approach.
- **Multivariate spatial dependence:** The method resamples full residual vectors η_t rather than individual components, preserving contemporaneous correlation structure. This is reflected in improved Energy Scores (Table 2) for multivariate experiments.
- **Theoretical grounding for residual distribution:** Theorem 1 establishes that the empirical distribution of normalized residuals converges uniformly to the true distribution under reasonable conditions, providing a foundation for the bootstrap approach.

## Weaknesses
- **Theorem 1 does not prove predictive distribution convergence:** The theorem establishes convergence of the empirical CDF of normalized residuals, not convergence of the final predictive distribution of future observations. The gap between residual distribution convergence and predictive distribution validity requires additional assumptions about error propagation through the autoregressive iteration, which is not analyzed. This is the central theoretical contribution claimed, but the proof is incomplete for the method's stated purpose.
- **IID assumption on normalized residuals unverified:** Assumption 1 requires η_t to be independent and identically distributed. After GARCH-type normalization, approximate independence is plausible but is never empirically tested. No autocorrelation diagnostics (e.g., Ljung-Box tests, ACF plots of normalized residuals) are provided to support this critical assumption.
- **Diagonal volatility matrix ignores cross-channel dynamics:** Equation 1 defines G as a diagonal matrix, meaning each dimension's volatility depends only on its own lagged residuals. Cross-channel heteroskedasticity and covariance dynamics are entirely ignored, potentially limiting multivariate performance. The paper does not acknowledge this as a modeling limitation.
- **No component ablation:** The framework combines volatility modeling and residual resampling. Experiments do not isolate whether improvements come from modeling conditional volatility, from the resampling step, or from their combination. A proper ablation (e.g., volatility model + Gaussian residuals, or resampling without volatility modeling) is missing.
- **Tables have severe formatting issues:** Table 1 contains entries rendered as "**0. (0.)**" where meaningful values should appear. This makes quantitative comparison impossible in many cells and undermines the empirical contribution. The paper states that boldface indicates better results, but the formatting corruption obscures which values are better and by how much.
- **No comparison to classical bootstrap forecasting methods:** The paper positions itself as a resampling-based approach but does not compare against established bootstrap prediction methods (e.g., Pan & Polis 2016, Stine 1985) that it cites as motivation. Without this comparison, the claimed novelty over existing resampling approaches is unverified.
- **Computational overhead unanalyzed:** Algorithm 2 requires B=100 forward passes per prediction horizon. The paper acknowledges computational complexity as a limitation but provides no runtime analysis, memory footprint, or comparison to baseline inference costs. Practitioners cannot assess the accuracy-efficiency trade-off.

## Nice-to-Haves
- **Calibration plots:** Adding coverage calibration curves would strengthen probabilistic forecasting claims by showing that nominal coverage levels (e.g., 90%) correspond to empirical coverage frequencies.
- **Distribution shift experiments:** The limitations section notes potential unreliability under distributional shift. Testing on datasets with known regime changes (e.g., financial data with market crashes) would validate robustness claims.
- **Sensitivity analysis for lag parameters (q, s):** Hyperparameters are set per dataset based on autocorrelation plots (Figure 4), but no analysis shows how sensitive performance is to these choices.
- **Comparison to conformal prediction:** Conformal prediction offers distribution-free coverage guarantees and is relevant to position DualRes against alternative uncertainty quantification approaches.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"No comparison with probabilistic forecasting methods"** — Incorrect. The paper compares against DeepAR, VEC-LSTM, and TMDM, which are established probabilistic forecasting methods. The experiments are structured as ablations (baseline vs. baseline+DualRes), but the baselines include probabilistic forecasters.

- **"DualRes degrades performance in multiple cases"** — This claim stems from misreading the corrupted Table 1 format. Entries shown as "**0. (0.)**" appear to be small positive values that were improperly rendered, not evidence of degradation. The paper states boldface indicates better results; the formatting issue makes direction of improvement unclear in some cells, but there's no evidence of systematic performance decline.

- **"Mean forecasting degrades"** — Same issue: Table 5 has the same formatting corruption. The claimed degradations cannot be verified from the paper as written.

- **"Diffusion models mischaracterization"** — The reviewer's claim that the paper mischaracterizes diffusion methods is partially valid but overstated. The paper correctly notes that diffusion-based forecasting often assumes Gaussian transitions; the "oversimplification" is minor.

## Novel Insights
The theoretical gap between Theorem 1 and the claimed predictive distribution convergence reveals an opportunity for follow-up work. The convergence of residual distributions is necessary but not sufficient for valid predictive intervals—error propagation through multi-step autoregressive simulation accumulates approximation errors in F̂ and Ĝ that compound across forecast horizons. The diagonal G assumption in Equation 1 is an architectural choice that enables tractability but sacrifices the ability to model covariance dynamics. A future extension using full covariance matrices (or low-rank approximations) could improve multivariate performance but would require solving the inverse covariance estimation problem during training. The bias-cancellation argument in Remark 1 is clever but relies on G_i depending on lagged ζ only through their squares—a GARCH-type structure that may not hold for all time series.

## Suggestions
- **Fix Table 1 formatting:** Restore proper numeric values with appropriate precision so readers can assess improvements quantitatively.
- **Add empirical verification of the IID assumption:** Report autocorrelation function plots or Ljung-Box test p-values for the normalized residuals on each dataset.
- **Include component ablation:** Show (a) baseline + volatility model + Gaussian residuals, and (b) baseline + empirical residual resampling without volatility model, to isolate each component's contribution.
- **Add inference time analysis:** Report wall-clock time for generating B=100 samples compared to baseline inference, enabling practitioners to assess practicality.
- **Acknowledge the diagonal G limitation:** Discuss in the limitations section that cross-channel heteroskedasticity is not captured and could be addressed in future work.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
