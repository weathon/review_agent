=== CALIBRATION EXAMPLE 33 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "DualRes: A Resampling-Based Framework for Enhancing Probabilistic Forecasting" is accurate and descriptive. The abstract makes four key claims: (1) no Gaussianity assumption, (2) flexibility in model choice, (3) theoretical justification, and (4) empirical gains on six real-world datasets. All four are addressed in the paper to varying degrees.

However, the abstract's framing of DualRes as improving over methods that "rely on Gaussianity" is somewhat overstated. Several recent methods (e.g., Chronos, normalizing flows) do not rely on Gaussian assumptions. The abstract should more carefully position the contribution relative to the full landscape of probabilistic forecasting.

---

### Introduction & Motivation

The motivation—that diffusion-based methods rely on Gaussian transitions while training-adjustment methods lack transparency—is directionally valid but imprecise. The characterization of training-adjustment methods (Le Guen & Thome, Rasul et al., Hasson et al.) as lacking "transparent and rigorous" mechanisms is debatable; several of those methods (normalizing flows, level-set approaches) have well-understood theoretical foundations. This framing overstates the gap that DualRes fills.

The contributions are clearly listed and, importantly, match what the paper actually delivers. However, the claim that DualRes has "theoretical justification" that outclasses black-box approaches sets up an expectation that Theorem 1 does not fully satisfy (see below).

---

### Method / Approach

**Model specification (Eq. 1):** The generative model is a vector heteroskedastic autoregressive model, a natural generalization of ARMA-GARCH. The key structural restriction is that G is diagonal, meaning the per-component volatilities evolve independently. In multivariate settings this is a significant simplification—real financial or climate data often has cross-variable volatility spillovers (as in DCC-GARCH). While the paper partially compensates by resampling entire residual vectors η_t (thereby preserving the cross-variable dependence in the innovations), the volatility structure is still univariate per channel. This architectural choice is not explicitly flagged as a limitation in the method section.

**Log-transformation for G (Remark 1, Section 4.1):** The transformation R(x) = (log(x₁²), ..., log(x_d²))⊤ reduces the problem of learning G to a mean-forecasting problem. This is elegant and practically important. The bias term E[log(η_{t,i}²)] is correctly identified and shown to cancel during normalization and resampling. The argument in Section 4.1 is convincing.

**A key implicit assumption** (stated mid-proof, not prominently in the algorithm): G_i² must depend on ζ_{t-1},...,ζ_{t-s} *only through their element-wise squares*. This is the GARCH-style assumption enabling the log-transform. It is not prominently stated in the main algorithm (Algorithm 1) or the model equation, only in a remark during Section 4.1. Given that this determines whether the volatility model is identifiable from the log-squared residuals, it warrants explicit statement in the algorithm or model definition.

**Inference procedure (Algorithm 2, Eq. 4):** The iterative generation of pseudo-samples x*_{T+j} accumulates approximation errors at each step. For j > 1, F̂ and Ĝ are evaluated at pseudo-residuals ζ̂*_{T+j-s,...}, which themselves are estimated. No discussion of error propagation across the J forecasting steps is provided.

---

### Theoretical Justification (Section 4 and Appendix A)

**Theorem 1** is the paper's main formal contribution. It guarantees that the empirical CDF P̂(y) of the estimated normalized residuals η̂_t converges in probability (in sup-norm) to the true CDF P(y) of η_t.

Several issues arise:

1. **I.i.d. assumption on η_t:** Assumption 1 requires η_t to be i.i.d. This is the regime where a classical Glivenko-Cantelli theorem trivially applies. The main work of the proof is bounding the estimation error from F̂ and Ĝ. If the model (1) is *correctly specified*, then by construction η_t should indeed be i.i.d. But this means the theorem's whole validity hinges on model (1) being correct. A brief discussion of what happens under misspecification—e.g., remaining serial dependence in η̂_t—is entirely absent. In practice, real residuals may retain mild autocorrelation even after GARCH-type normalization (e.g., leverage effects), which would invalidate the theorem's premise.

2. **Uniform convergence assumption (Assumption 2):** The proof requires sup_Y ||F̂(Y) − F(Y)|| →_p 0 and sup_Y |Ĝ_i(Y) − G_i(Y)| →_p 0 *uniformly over all of R^{d×q} and R^{d×s}*, respectively. For neural networks (the primary models used in experiments), such uniform consistency does not generally hold without explicit regularity conditions. The paper simply assumes this, which is circular with respect to the practical setting. At minimum, this assumption should be acknowledged as non-trivial for deep learning architectures.

3. **Gap between Theorem 1 and the actual claim:** Theorem 1 guarantees convergence of the *marginal* distribution of η_t. However, the ultimate goal—forecasting the *joint* distribution of (x_{T+1}, ..., x_{T+J})—requires the iterative structure of Algorithm 2, which propagates η*_j through F̂ and Ĝ for J steps. No formal guarantee is provided for the quality of the resulting joint predictive distribution. The connection from "η̂ converges to the right distribution" to "the J-step-ahead predictive distribution is accurate" is only argued informally in Section 3.2.

4. **No convergence rates:** Theorem 1 provides only consistency (→_p 0 as T → ∞). For practical guidance on how large T needs to be relative to q, s, d, and J, finite-sample bounds would be considerably more useful.

---

### Experiments & Results

**Design and baselines (Section 5.1):**

The paper frames its evaluation as "ablation studies," comparing Base Model vs. Base Model + DualRes. This is the appropriate design for a plug-in framework. However, there is a glaring omission: **there is no comparison to any dedicated probabilistic forecasting method (TimeGrad, CSDI, TSDIFF, Chronos, etc.) as a standalone baseline.** An ICLR reader cannot determine whether, e.g., DLinear+DualRes is competitive with a purpose-built probabilistic model. The paper presents DualRes as a boost over mean-forecasting models, but without this comparison, it is unclear whether the resulting system is actually state-of-the-art or merely "better than a weak baseline."

**Table 1 — inconsistent wins:**

The table shows cases where DualRes *hurts* performance:
- DLinear+Ours CRPS on ETTh1 is 0.196, while DLinear alone appears better on that metric (winning that column per the boldface).
- PatchTST+Ours shows 0.200 CRPS on ETTh1, while PatchTST alone wins that entry.

These regressions are never discussed. The narrative in Section 5.1 focuses only on cases where DualRes wins ("average CRPS of TimeMixer on Exchange decreases from 0.027 to 0.014"), creating a misleading impression of consistent improvement. The authors should explain why DualRes sometimes degrades performance and whether the cases where it hurts are predictable.

**Duplicate paragraph in Section 5.1:** The paragraph beginning "As demonstrated in Table 1, incorporating information on conditional volatility..." is nearly identically repeated as the following paragraph beginning "the CRPS and MAEC of various forecasting algorithms have significant decreases..." This is an uncorrected writing error that undermines the paper's polish.

**Missing ablation — the paper's most significant experimental gap:** DualRes has two main components: (1) modeling conditional volatility (heteroskedasticity), and (2) resampling from the empirical distribution of normalized residuals (non-parametric innovation distribution). Without an ablation that isolates each—e.g., comparing against a version that uses volatility modeling but assumes Gaussian η_t, and against a version that resamples without volatility normalization—it is impossible to attribute where the gains actually come from. Figure 2 (histograms showing non-Gaussianity) provides qualitative motivation, but cannot substitute for a controlled ablation.

**Multivariate experiments (Table 2):**

Only three datasets and two baselines are used. TMDM+Ours fails to consistently improve: MAEC worsens on ETTh1 (0.458 vs. 0.268) and ES worsens on ETTh2 (7.326 vs. 6.933). The Energy Score is the most direct measure of multivariate distributional quality, so the regression on ETTh2 is particularly concerning. No explanation is offered.

**Computational cost:** The limitations section flags computational complexity, but no wall-clock times or memory profiles are reported. This makes it impossible to evaluate the practical overhead of B=100 bootstrap resamples in production forecasting contexts.

**Hyperparameter selection:** The context length for the volatility model is chosen by inspecting autocorrelation plots (Figure 4). This requires dataset-specific tuning and introduces a potential source of data leakage if done improperly. The paper does not describe the train/validation/test split used for this selection.

---

### Writing & Clarity

The main body is generally readable. The notation is occasionally dense but manageable. Beyond the duplicate paragraph noted above, the placement of equation (7) is confusing—it appears in the middle of the experiments section (page 6) but belongs logically with Section 4.2. The ordering seems to be a PDF layout artifact, but it breaks the narrative flow significantly.

The description of Algorithm 2, line 4, uses γ̃* and ζ̂* somewhat inconsistently with Algorithm 1's notation. Remark 2 clarifies this but it requires cross-referencing between algorithms to follow.

---

### Limitations & Broader Impact

The limitations section acknowledges (1) computational complexity and (2) sensitivity to distributional shift. Both are real. However, several failure modes are not discussed:

- **Model misspecification:** If the diagonal-G assumption is violated (common in financial time series), or if η_t retains serial correlation, the entire framework's validity is undermined. This is the most practically important failure mode.
- **Short time series:** The Glivenko-Cantelli convergence requires T to be large relative to q and s. For short time series (e.g., N2 Hourly with median 960 steps and window 312 + prediction 48), the bootstrap approximation may be poor. This is not discussed.
- **Boundary behavior of the log-transform:** R(x) = log(x²) is undefined when any component of x is exactly 0. In practice, if the model fit is near-perfect for some observations, this can cause numerical instability. No mention is made of how this is handled.

---

### Overall Assessment

DualRes presents a practically motivated and conceptually clean framework: use a two-stage GARCH-inspired approach to separate conditional mean, volatility, and innovation distribution, then apply bootstrap resampling to avoid parametric distributional assumptions. The log-transformation trick for reducing volatility learning to mean learning is clever, and the plug-in design (wrapping any mean forecaster) is genuinely flexible. However, for ICLR, the paper has several significant weaknesses that prevent a straightforward acceptance. Most critically, there are no comparisons against any dedicated probabilistic forecasting method, making it impossible to assess where DualRes-enhanced models stand in the field. Theorem 1's theoretical guarantees rely on i.i.d. innovations and uniform consistency of deep neural networks—assumptions that are implicitly invoked but not verified or adequately discussed. The multi-step-ahead forecasting accuracy is not theoretically bounded. Experimental results are inconsistent (DualRes sometimes hurts), with no explanation, and the key ablation separating the two main components is absent. The contribution, while practically useful, may be perceived as primarily an engineering application of well-known statistical ideas (GARCH + bootstrap) to the DL time-series setting, which is a marginal fit for ICLR's novelty bar without stronger empirical and theoretical support.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DualRes, a resampling-based framework that enhances probabilistic time series forecasting by explicitly modeling conditional heteroskedasticity and resampling normalized residuals. The method decouples conditional mean and volatility estimation, allowing existing mean-forecasting models to be adapted for probabilistic prediction without Gaussian assumptions. Extensive numerical experiments on real-world datasets demonstrate improved coverage and distributional accuracy compared to standard baselines.

### Strengths
1.  **Flexibility of Integration:** DualRes acts as a robust wrapper that allows purely mean-oriented models (e.g., DLinear, TimeMixer) to produce high-quality probabilistic forecasts. As shown in Table 1, integrating DualRes with these baselines consistently reduces CRPS and MAEC across six univariate datasets.
2.  **Empirical Robustness:** The experimental validation is comprehensive, covering both univariate (6 datasets) and multivariate (3 datasets) settings. Table 2 demonstrates performance gains in capturing spatial dependence (via Energy Score) and reducing uncertainty errors (MAEC) in multivariate scenarios.
3.  **Theoretical Justification:** The paper provides a specific theorem (Theorem 1) justifying the convergence of the empirical residual distribution to the true underlying distribution, grounding the resampling step in rigorous statistical theory (citing Glivenko-Cantelli).
4.  **Practical Innovation:** Remark 1 and the associated derivation effectively address the difficulty of learning volatility functions by transforming them into a mean-forecasting problem via logarithmic squaring, a clever workaround that aligns with existing deep learning tooling.

### Weaknesses
1.  **Computational Overhead:** The inference stage requires generating $B$ bootstrap samples per forecast (set to 100 in experiments). While feasible for offline analysis, this significantly increases inference latency compared to parametric methods or single-sample generators, a drawback not fully quantified in terms of wall-clock time.
2.  **Limited Baseline Comparison:** The paper compares against DeepAR and transformers but lacks comparison against Conformal Prediction (CP) or Ensemble methods. CP is a dominant alternative for distribution-free uncertainty quantification that currently offers strong coverage guarantees for ICLR-relevant deep learning settings.
3.  **Strong Modeling Assumptions:** Theorem 1 relies on the assumption that the learned mean $F$ and volatility $G$ converge uniformly to the true functions. For complex deep learning estimators, uniform convergence is a non-trivial claim, and the paper does not analyze sensitivity to volatility model misspecification (e.g., if $G$ fails to capture structural breaks).
4.  **Distributional Assumption Nuance:** While the authors claim "No Gaussianity assumption," the validity depends on normalized residuals $\eta_t$ being i.i.d. and following a fixed distribution $P$. If the underlying data-generating process shifts or $F/G$ are highly misspecified, the i.i.d. assumption of residuals may break down, which needs more discussion.

### Novelty & Significance
*   **Novelty:** Moderate. Combining GARCH-like volatility modeling with bootstrap resampling is established in econometrics, but its systematic adaptation to modern deep time-series baselines as a unified framework is a valuable contribution. The specific log-transform trick to enable mean-forecasters for volatility adds algorithmic novelty.
*   **Significance:** High. Probabilistic forecasting is critical for risk management (finance, energy). A method that improves coverage and stability without requiring complex architectural changes offers practical utility.
*   **Clarity:** High. The structure is logical, separating training, inference, and theory clearly, despite minor OCR artifacts in equations.
*   **Reproducibility:** High. The Appendix (Section B) provides detailed dataset descriptions, hyperparameters, and metric definitions (CRPS, MAEC, ES).

### Suggestions for Improvement
1.  **Comparison with Conformal Prediction:** Include a comparison with Split-CP or Conformal Prediction baselines on the same datasets to contextualize the coverage guarantees and computational trade-offs of DualRes against the current state-of-the-art for valid inference.
2.  **Complexity Analysis:** Add a table or figure measuring inference latency (time per forecast) and memory usage between DualRes and standard parametric approaches to quantify the performance cost of the resampling step.
3.  **Sensitivity to Volatility Model:** Provide an ablation study where the volatility model $G$ is intentionally underspecified or over-specified to demonstrate how robust DualRes is to errors in the volatility estimation stage.
4.  **Discussion on Stationarity:** Expand the Limitations section to discuss how the method handles non-stationary data where the residual distribution itself changes over time, as this contradicts the implicit i.i.d. assumption of the bootstrap samples.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Complete the results tables** — Table 1 and Table 2 contain numerous missing values displayed as "0._(0._)" which completely undermines the empirical claims. ICLR reviewers cannot evaluate performance improvements without actual numbers.

2. **Add comparison with SOTA probabilistic forecasting methods** — The paper claims advantages over diffusion-based methods but only compares against DeepAR, DLinear, PatchTST, and TimeMixer. Include Chronos, TimesNet, or other recent probabilistic forecasting SOTA to justify the contribution claim.

3. **Ablation on resampling vs. volatility modeling** — The paper claims both components matter, but no experiment isolates whether improvements come from the volatility estimation, the resampling step, or their combination. Add a "DualRes without resampling" baseline.

4. **Coverage calibration experiments across quantile levels** — MAEC aggregates across 9 quantile levels but doesn't show whether coverage is accurate at extreme quantiles (e.g., 95%, 99%) where probabilistic forecasting matters most for risk assessment.

5. **Long-horizon forecasting evaluation** — Residual distribution approximation should matter more for longer prediction horizons, but all experiments use short horizons (24-48 steps). Test at 96+ steps to validate the core claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Theorem 1 assumes i.i.d. residuals but real time series violate this** — The theoretical guarantee requires independent normalized residuals, yet the paper acknowledges temporal dependence exists. Analyze what happens when residuals exhibit autocorrelation and how this affects coverage.

2. **Log-transformation bias cancellation claims need verification** — Remark 1 states the constant bias "self-eliminates" but provides no empirical evidence. Show that the bias actually cancels in practice across datasets.

3. **Sensitivity analysis on hyperparameters (lags q, s, resampling times B)** — The method introduces multiple new hyperparameters but Table 4 shows fixed values without justification. Show how performance varies with these choices.

4. **Failure mode analysis** — The limitations section briefly mentions distributional shift but doesn't test it. Show empirical results on datasets with known structural breaks or regime changes where the method should fail.

5. **Computational complexity quantification** — The paper acknowledges computational cost as a limitation but provides no runtime comparisons. ICLR requires understanding trade-offs; add training/inference time vs. baseline methods.

### Visualizations & Case Studies
1. **Calibration plots comparing predicted vs. empirical coverage** — Show reliability diagrams for prediction intervals at multiple confidence levels (80%, 90%, 95%) to demonstrate whether intervals are well-calibrated, not just low MAEC.

2. **Residual distribution comparison (predicted vs. empirical)** — Figure 2 shows fitted residuals but doesn't compare the resampled predictive distribution against actual test residuals. Add QQ-plots or distribution overlays.

3. **Case study showing where DualRes succeeds/fails** — Pick specific time series examples where the method substantially improves or worsens predictions. This reveals whether gains are consistent or dataset-dependent.

4. **Prediction interval width comparison with baselines** — Figure 3 shows DualRes intervals but doesn't compare interval widths against baseline methods. Narrower intervals with same coverage would validate the volatility modeling claim.

### Obvious Next Steps
1. **Runtime and memory profiling** — ICLR expects efficiency analysis for new frameworks. Report GPU hours, memory usage, and scalability with sequence length compared to diffusion-based alternatives.

2. **Test on datasets with known conditional heteroskedasticity** — The core claim is handling heteroskedasticity, but standard benchmarks (ETTh, Electricity) aren't known for this. Include financial or volatility-rich datasets where GARCH-like behavior is documented.

3. **Multivariate spatial dependence analysis** — The paper claims improved energy scores capture spatial dependence but doesn't visualize or analyze which variable correlations are better captured. Add correlation matrix comparisons.

4. **Robustness to model misspecification** — Test DualRes when the conditional mean model is deliberately weakened. The method claims to boost any mean forecaster, but this needs validation under poor mean estimation.

# Final Consolidated Review
## Summary
DualRes is a resampling-based framework for probabilistic time series forecasting that decomposes the problem into three stages: (1) learning a conditional mean model F, (2) learning a conditional volatility model G via a log-transformation trick that converts volatility estimation to mean forecasting, and (3) bootstrap resampling of normalized residuals to avoid parametric distributional assumptions. The method operates as a wrapper that can enhance any mean-forecasting model to produce probabilistic forecasts, demonstrated on six real-world datasets.

## Strengths
1. **Flexible integration with existing models**: DualRes can wrap any mean-forecasting architecture without modification. The log-transform trick (Remark 1, Eq. 3) that converts volatility estimation to mean forecasting via R(x) = log(x²) is elegant—it allows practitioners to reuse existing forecasting infrastructure for the volatility component. This design choice meaningfully lowers the barrier to probabilistic forecasting adoption.

2. **Theoretical grounding with explicit assumptions**: Theorem 1 provides formal justification for the resampling procedure, establishing convergence of the empirical residual distribution to the true distribution under Assumption 1 (i.i.d. η_t) and Assumption 2 (uniform convergence of estimators). The proof in Appendix A is complete and follows classical Glivenko-Cantelli arguments combined with estimation error bounds.

3. **Empirical gains across baselines and datasets**: Table 1 shows CRPS improvements on 14 of 24 univariate metric-dataset combinations and Table 2 shows improvements on 12 of 18 multivariate combinations. The MAEC (coverage accuracy) metric improves in most cases, indicating that prediction intervals are well-calibrated.

## Weaknesses
1. **No comparison to dedicated probabilistic forecasting methods**: The paper evaluates DualRes-enhanced versions of mean-forecasting models (DLinear, PatchTST, TimeMixer) and one probabilistic model (DeepAR), but does not compare against modern dedicated probabilistic forecasting baselines like TimeGrad, CSDI, Chronos, or diffusion-based approaches. The paper explicitly positions itself as an alternative to diffusion-based methods that "rely on Gaussian assumptions" (Introduction, paragraph 1), making this omission particularly problematic. Without these comparisons, readers cannot assess whether DualRes-enhanced models are competitive with the state-of-the-art or merely improve upon weak baselines.

2. **Missing ablation isolating core components**: DualRes combines (a) conditional volatility modeling and (b) non-parametric resampling of the innovation distribution. The paper attributes gains to both, but provides no controlled experiment isolating their individual contributions. A proper ablation would include:
   - DualRes full (volatility + resampling)
   - Volatility only (assume Gaussian η_t)
   - Resampling only (no volatility normalization)
   Figure 2 shows non-Gaussian residuals, motivating resampling, but cannot substitute for quantitative ablation. Without this, the attribution of gains remains speculative.

3. **Inconsistent performance improvements left unexplained**: Table 1 shows regressions: DLinear+Ours CRPS on ETTh1 is 0.196, worse than DLinear alone (which wins that column with boldface). Table 2 shows TMDM+Ours degrades MAEC on ETTh1 (0.458 vs. 0.268) and Energy Score on ETTh2 (7.326 vs. 6.933). The Energy Score regression is particularly concerning as it measures multivariate dependence—the paper's claimed strength. These failures are never discussed, nor are conditions under which DualRes should or should not be applied.

4. **Strong theoretical assumptions not verified**: Assumption 1 requires η_t to be i.i.d. If the volatility model G is misspecified (e.g., diagonal G when cross-variable volatility spillovers exist, or wrong lag order s), normalized residuals may retain serial correlation, invalidating the theorem. Assumption 2 requires uniform convergence sup_Y ||F̂(Y) − F(Y)|| →_p 0 over all of R^{d×q}—this is non-trivial for neural networks without explicit regularity conditions. The paper does not discuss failure modes, provide sensitivity analysis, or test whether η̂_t actually satisfies independence (e.g., via Ljung-Box tests).

5. **Table formatting issues obscure results**: Tables 1-2 contain values displayed as "0._(0._)" for best entries, where actual numerical values should appear. This formatting artifact makes it impossible to evaluate the magnitude of improvements in multiple cells.

6. **No computational cost quantification**: The inference stage requires B=100 bootstrap resamples, plus forward passes through F̂ and Ĝ for each. The limitations section acknowledges complexity but provides no wall-clock time, memory usage, or comparison with baselines. For production forecasting systems, this overhead must be understood.

7. **Theorem guarantees marginal distribution, not joint predictive distribution**: Theorem 1 establishes that P̂(y) → P(y) for the normalized residuals. However, the forecasting goal is the joint distribution of (x_{T+1}, ..., x_{T+J}). The iterative structure in Algorithm 2 (Eq. 4) propagates estimation errors across J steps—no formal guarantee connects marginal residual convergence to multi-step predictive quality.

## Nice-to-Haves
1. **Comparison with Conformal Prediction**: CP methods offer alternative distribution-free coverage guarantees; comparing DualRes against CP would contextualize trade-offs between the generative bootstrap approach and post-hoc calibration.

2. **Long-horizon evaluation**: All experiments use prediction horizons of 24-48 steps. Testing at longer horizons (96+ steps) would validate whether the bootstrap approximation quality degrades with J.

3. **Calibration reliability diagrams**: MAEC aggregates across quantile levels but does not show whether extreme quantiles (90%, 95%, 99%) are well-calibrated—critical for risk management applications.

4. **Dataset-specific heteroskedasticity validation**: The core claim is handling conditional heteroskedasticity, but standard benchmarks (ETTh, Electricity) are not known for strong ARCH/GARCH effects. Testing on financial or volatility-rich datasets would strengthen the empirical case.

## Removed Points
These points are flagged to be removed, treat them with caution:

- *"Duplicate paragraph in Section 5.1"* — This is a minor writing/polishing error that does not affect the scientific contribution.

- *"Characterization of training-adjustment methods as lacking transparency"* — This framing disagreement is subjective; the paper's position that black-box neural approaches lack interpretability compared to simulation-based methods is reasonable within scope.

- *"Log-transformation undefined at zero (log(x²) when x=0)"* — This is a standard numerical implementation issue handled with epsilon adjustments in practice; not a fundamental theoretical flaw worth emphasizing.

- *"Comparison against Conformal Prediction as required baseline"* — CP operates under a different paradigm (post-hoc calibration vs. generative modeling). Including it would be informative but is not required for the paper's stated contribution.

## Novel Insights
The key insight—separating mean and volatility estimation via log-transform while resampling innovations—is a principled statistical decomposition applied to deep learning forecasting. Unlike GARCH which assumes parametric innovation distributions, DualRes's bootstrap approach handles arbitrary residual distributions. However, the diagonal volatility structure (G is d×d diagonal) is a significant restriction: cross-variable volatility spillovers common in finance (modeled by DCC-GARCH, BEKK) cannot be captured. The resampling of entire vectors η_t preserves cross-sectional dependence in innovations, but the volatility model remains univariate per channel. This is partially compensated by the vector resampling but represents an architectural tradeoff not explicitly discussed.

## Suggestions
1. **Add dedicated probabilistic forecasting baselines**: Report results for TimeGrad, CSDI, or Chronos on the same datasets to establish where DualRes-enhanced models stand relative to state-of-the-art.

2. **Provide ablation study**: Compare three variants—volatility-only (Gaussian η), resampling-only (no G), and full DualRes—to isolate the contribution of each component.

3. **Explain performance regressions**: Discuss conditions under which DualRes degrades performance (observed in Tables 1-2) and provide guidance on when practitioners should expect improvements.

4. **Verify residual independence empirically**: After fitting F and G, report Ljung-Box or similar tests on normalized residuals η̂_t to assess whether the i.i.d. assumption holds.

5. **Report computational overhead**: Include training time, inference time, and memory usage comparisons between DualRes and baselines.

6. **Fix table formatting**: Replace "0._(0._)" placeholders with actual numerical values.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
