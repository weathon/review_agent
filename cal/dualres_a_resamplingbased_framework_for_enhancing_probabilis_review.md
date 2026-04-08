=== CALIBRATION EXAMPLE 41 ===

# Final Consolidated Review
## Summary

DualRes proposes a resampling-based framework for probabilistic time series forecasting that decomposes the problem into conditional mean estimation, conditional volatility estimation, and bootstrap resampling of normalized residuals. By training two separate models (for mean and volatility) and using non-parametric resampling rather than parametric distributional assumptions, DualRes can convert any mean-forecasting algorithm into a probabilistic forecaster that accounts for conditional heteroskedasticity and non-Gaussian residual distributions.

## Strengths

- **Flexible plug-and-play design for converting mean forecasters to probabilistic forecasters.** The key insight—leveraging the log-transformation in Remark 1 so that both $F$ and $G$ require only mean-forecasting algorithms—is practically valuable. This allows practitioners to apply DualRes on top of existing architectures (DLinear, PatchTST, TimeMixer) without architectural changes, lowering the adoption barrier.
- **Non-parametric residual modeling avoids restrictive distributional assumptions.** Figure 2 directly demonstrates that normalized residuals across six datasets deviate substantially from Gaussian densities (exhibiting heavy tails, skewness, and multimodality), validating the core motivation for resampling over parametric density specification.
- **Theoretical grounding for the bootstrap procedure.** Theorem 1 establishes that the empirical CDF of the resampled normalized residuals converges in probability to the true distribution, providing formal justification absent from many empirical deep learning forecasting papers.
- **Consistent empirical improvements across diverse benchmarks.** Tables 1 and 2 show CRPS and MAEC improvements when DualRes is applied to multiple base models on six univariate and three multivariate datasets, including large-scale gains (e.g., TimeMixer CRPS on Exchange: 0.027→0.014; TMDM CRPS on Electricity: 0.655→0.292).

## Weaknesses

### Major:

- **No ablation isolating the contribution of volatility modeling from resampling.** The paper attributes improvements to both conditional heteroskedasticity modeling and residual distribution capture, but never tests what happens with only volatility modeling (Gaussian residuals with learned volatility) or only resampling (homoskedastic residuals with bootstrap). Without this ablation, it is unclear which component drives the gains, undermining the paper's core claim that both components are essential.

- **The i.i.d. residual assumption central to Theorem 1 is not empirically verified.** Algorithm 2 resamples $\hat{\boldsymbol{\eta}}_t$ under the assumption that they are i.i.d. (Assumption 1). However, no diagnostics—such as autocorrelation function (ACF) plots of the normalized residuals, Ljung-Box tests, or ARCH-LM tests on squared residuals—are presented. If the volatility model $G$ is misspecified (which is plausible given its simple architecture, discussed below), remaining serial dependence in $\hat{\boldsymbol{\eta}}_t$ would violate the i.i.d. requirement and potentially invalidate the prediction intervals. Figure 2 shows marginal distributions but says nothing about temporal dependence.

- **Primary baselines are weak probabilistic forecasters.** Three of four univariate baselines (DLinear, PatchTST, TimeMixer) are mean-forecasting models retrofitted with t-distribution outputs—a minimal and non-competitive probabilistic approach. While DeepAR and TMDM are genuinely probabilistic, the paper does not compare against other strong distribution-free or heteroskedasticity-aware methods (e.g., conformal prediction wrappers, quantile regression networks, or other recent probabilistic SOTA). This makes it difficult to assess DualRes's standing in the broader probabilistic forecasting landscape.

### Minor:

- **Diagonal $G$ limits multivariate dependence modeling.** Equation 1 defines $G$ as a diagonal matrix, which assumes conditional independence across dimensions given the past. While resampling full residual vectors preserves empirical cross-correlation in the noise, this does not capture dynamic conditional correlations (as in DCC-GARCH-type models). The paper claims multivariate capability, but the volatility scaling is dimension-wise independent—a limitation worth acknowledging explicitly.

- **Integration with native probabilistic models is unclear.** When DualRes is applied to DeepAR or TMDM (Table 1–2, "+Ours" rows), the paper does not explain whether DualRes replaces or supplements these models' native uncertainty estimates. DeepAR already models conditional volatility autoregressively; does DualRes discard DeepAR's variance parameters and re-learn $G$ from residuals? Clarifying this interaction is important for understanding what DualRes actually adds to already-probabilistic models.

- **Computational overhead is acknowledged but not quantified.** Section 6 mentions complexity as a limitation, and $B=100$ resampling iterations (Appendix B.1) imply approximately 100× inference cost per forecast step. No runtime comparison or wall-clock analysis is provided, making it difficult to assess practical deployability.

- **Log-transformation bias cancellation lacks empirical verification.** Remark 1 claims the constant bias from estimating $\log(G_i)$ self-eliminates during normalization and sampling (Eqs. 2 and 4). The theoretical argument is sound but sketched rather than formally derived, and no experiment directly measures the residual bias magnitude with and without the claimed correction.

- **Asymptotic guarantees without finite-sample characterization.** Theorem 1 assumes consistent estimators $\hat{F} \to F$ and $\hat{G} \to G$ uniformly (Assumption 2), which is a strong condition for neural networks. No finite-sample coverage bounds or convergence rates are provided, limiting the practical interpretability of the theoretical contribution.

### Trivial:

- The abstract's statement "DualRes requires only mean forecasts" could be more precise—it requires only mean-forecasting *algorithms*, but needs two such models (for mean and volatility) as part of a two-stage procedure.

## Nice-to-Haves

- Lag sensitivity analysis for $q$ and $s$: Table 4 shows dataset-specific choices with no ablation. Since the volatility model's performance directly depends on these lags, a sensitivity study would strengthen the empirical contribution.
- Prediction horizon degradation analysis: Algorithm 2 uses iterative sampling where errors can compound. Showing how CRPS/MAEC evolves across prediction steps $j = 1, \ldots, J$ would reveal practical limits.
- Testing on datasets with known heavy-tailed or skewed distributions where Gaussian assumptions provably fail, to more directly validate the non-Gaussian robustness claim.
- Calibration/reliability diagrams plotting nominal vs. empirical coverage across quantile levels, which would be more informative than aggregate MAEC.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Missing Conformal Prediction literature"**: Per rules, I do not flag missing related works as I cannot verify their existence or relevance comprehensively.

- **"Diffusion models don't all rely on Gaussian assumptions"**: While some modern diffusion variants relax Gaussian transitions, the standard framework (Ho et al., 2020; Rasul et al., 2021) does use Gaussian noise schedules. The paper's characterization is reasonable as a motivation, and the core contribution is independent of this claim.

- **"Garbled table values in Table 1 and 2"**: This is a PDF extraction artifact, not a paper problem. The actual paper renders these correctly.

- **"No statistical significance tests"**: The paper reports 95% confidence intervals over 5 runs. Additional tests would be a nice-to-have, but single-run evaluation with confidence intervals is common practice in this field and not a flaw.

- **"Limited novelty—just combines GARCH + bootstrap"**: While the individual components are classical, the systematic application to modern deep learning backbones with the log-transformation trick (Remark 1) and the theoretical convergence result constitutes a meaningful methodological contribution. Novelty is moderate but not absent.

- **"Reproducibility—no code release"**: Per rules, I do not flag missing code or reproducibility concerns about undisclosed implementation details.

## Novel Insights

The paper's most underappreciated contribution is the conceptual decomposition of probabilistic forecasting into three independently modifiable components—conditional mean, conditional volatility, and residual distribution—rather than treating uncertainty estimation as an end-to-end learned output. This decomposition mirrors classical econometric reasoning (ARMA-GARCH) but importantly reveals that most modern deep forecasting backbones implicitly assume a fixed, often Gaussian, residual distribution. The empirical evidence (Figure 2) that real-world residuals systematically violate parametric assumptions suggests that the performance bottleneck in many probabilistic forecasters may not be in the mean or volatility estimation, but in the distributional assumption on innovations—a point the community should take seriously when designing next-generation probabilistic models.

## Suggestions

- Add an ablation with three conditions: (a) mean-only with Gaussian/t-distributed residuals (current baselines), (b) mean + learned volatility with Gaussian residuals, and (c) full DualRes. This directly isolates the marginal contribution of each component.
- Include ACF plots and formal independence tests on the normalized residuals $\hat{\boldsymbol{\eta}}_t$ to verify the i.i.d. assumption that Theorem 1 requires.
- Explicitly describe how DualRes integrates with DeepAR and TMDM—specifically whether it replaces or augments their native uncertainty outputs—to clarify the "+Ours" experimental setup.
- Provide a brief runtime comparison (e.g., seconds per forecast batch) between base models and DualRes-augmented versions to quantify the practical cost of resampling.
- Consider discussing the diagonal-$G$ limitation more explicitly in Section 5.2, noting that dynamic conditional correlations are not modeled and identifying this as a concrete direction for extension.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
