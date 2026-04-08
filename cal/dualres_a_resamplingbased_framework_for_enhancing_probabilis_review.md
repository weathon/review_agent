=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary

DualRes proposes a resampling-based framework that enhances probabilistic time series forecasting by decoupling conditional mean and volatility estimation, normalizing fitted residuals by estimated volatility, and bootstrap-resampling the normalized residuals to construct predictive distributions without parametric assumptions. The framework is flexible: since a logarithmic transformation converts volatility estimation into a mean-forecasting problem, any mean-forecasting algorithm can serve as both the mean and volatility model, enabling even purely deterministic forecasters to produce probabilistic outputs.

## Strengths

- **Practical model-agnostic flexibility**: The log-transformation trick (Remark 1, Eq. 3) allows any mean-forecasting model to estimate conditional volatility, meaning purely deterministic architectures (DLinear, PatchTST, TimeMixer) can be upgraded to produce full predictive distributions without architectural changes. This is a concrete, specific engineering contribution not shared by most probabilistic forecasting frameworks.

- **Non-parametric residual handling motivated by real data**: Figure 2 directly demonstrates that normalized residuals across all six datasets deviate from Gaussian density—exhibiting heavy tails, skewness, or multimodality—providing empirical justification for the resampling approach over parametric alternatives.

- **Explicit treatment of conditional heteroskedasticity**: Unlike many deep probabilistic forecasting methods that treat volatility implicitly through learned likelihood parameters, DualRes models volatility as a first-class component (Algorithm 1, Step 2), producing prediction intervals whose widths vary adaptively across forecast horizons as shown in Figure 3.

- **Theoretical grounding**: Theorem 1 provides an asymptotic consistency guarantee for the bootstrap procedure, establishing that the empirical CDF of resampled pseudo-normalized residuals converges to the true CDF as the sample size grows, grounding the empirical heuristic in formal statistical theory.

## Weaknesses

1. **Ablation-only evaluation without comparison to dedicated probabilistic forecasting baselines**. The entire experimental design compares each base model against itself augmented with DualRes. There is no comparison against standalone, purpose-built probabilistic forecasting methods (e.g., TimeGrad, normalizing flow-based forecasters, quantile regression forests). It is therefore impossible to determine whether DualRes+base achieves competitive absolute performance against the best available probabilistic models, or whether an end-to-end trained probabilistic model would simply outperform it. This is a critical gap for a paper positioned as a general enhancement framework.

2. **Theorem 1's i.i.d. assumption on normalized residuals is not empirically validated**. The theoretical guarantee requires that η_t are independent and identically distributed (Assumption 1). Even after two-stage volatility normalization, normalized residuals in real time series data frequently retain autocorrelation or ARCH effects. The paper shows histograms of η̂_t (Figure 2) to motivate non-Gaussianity, but presents no diagnostic checks for serial independence (e.g., Ljung-Box tests on η̂_t or η̂²_t, autocorrelation plots). Without this verification, the core theoretical guarantee does not straightforwardly transfer to the empirical setting, and it is unclear whether a block bootstrap would be more appropriate.

3. **Inconsistent improvements across model/dataset combinations are unacknowledged and unexplained**. In Table 2, applying DualRes to TMDM degrades MAEC on ETTh1 from 0.268 to 0.458 and ES on ETTh2 from 6.933 to 7.326. In Table 1, several readable entries show CRPS degradation when DualRes is applied (e.g., DLinear on ETTh1 goes from a lower baseline value to 0.196). The paper attributes improvements to heteroskedasticity and non-Gaussian residuals but provides no analysis of when or why DualRes can harm performance, nor diagnostic tools to predict whether applying it will be beneficial for a given model-dataset pair. This undermines the generality claims.

4. **The diagonal structure of G excludes cross-series volatility dependence in multivariate settings**. Equation 1 defines G as a diagonal matrix, meaning each series' volatility depends only on its own past squared residuals, ruling out volatility spillovers between variables. This is a common feature in financial and energy data. The multivariate advantage is therefore limited to capturing cross-sectional dependence through joint vector resampling, while cross-series heteroskedasticity remains unmodeled. This restriction is not acknowledged as a limitation.

5. **Computational overhead from inference-time resampling is not quantified**. DualRes requires B=100 full autoregressive forward passes through both F̂ and Ĝ during inference to generate pseudo-samples (Algorithm 2, Eq. 4). The paper acknowledges computational complexity as a limitation in Section 6, but provides no empirical measurement of inference time or memory relative to base models. For a framework whose stated advantage is model-agnostic flexibility, the absence of any runtime analysis makes it difficult to assess practical deployability, especially for large-scale or real-time settings.

6. **The bootstrap cannot extrapolate beyond the range of observed residuals**. Since DualRes resamples from historical η̂_t with replacement, it cannot generate residuals larger than the maximum observed during training. For risk assessment applications (finance, energy), this means the framework fundamentally cannot capture tail events or "black swan" scenarios that exceed the training support—a limitation that parametric methods with heavy-tailed distributions can address. The paper briefly mentions distributional shift but does not discuss this more fundamental inability to extrapolate, which is particularly consequential for the target applications.

7. **Imprecise characterization of prior diffusion-based methods**. The Introduction states that "the validity of such methods in general relied on the assumption of time series having Gaussian distribution." This conflates the Gaussian transition kernel assumption (standard in diffusion forward processes) with the data distribution assumption. Diffusion models can learn to generate non-Gaussian data distributions through the reverse process, even with Gaussian transitions. The paper's phrasing overstates the limitation of prior work, weakening its own motivation.

## Nice-to-Haves

- Ablation study separating the contribution of volatility modeling from resampling (e.g., Gaussian residuals + volatility model vs. i.i.d. resampling without volatility model vs. full DualRes), to isolate which component drives the observed gains.
- Comparison with Conformal Prediction methods to situate DualRes within the broader landscape of distribution-free uncertainty quantification.
- Analysis of long-horizon performance degradation, since autoregressive bootstrap iteration (Eq. 4) may compound errors over longer prediction horizons.
- Block or stationary bootstrap variants for residuals that exhibit remaining temporal dependence after normalization.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness**: "Log-transformation bias cancellation could break with regularization/dropout differences between training and inference G." — This misunderstands the mechanism: the same estimated Ĝ (with fixed weights) is used in both normalization (Eq. 2) and sampling (Eq. 4), both in inference/eval mode. The constant bias cancels algebraically regardless of how the model was regularized during training. This is not a valid concern.
- **Weakness**: "Missing related works (Chronos, TimeCCT, GARCH bootstrap)." — Per rules, do not flag missing related works without external confirmation of their existence and relevance.
- **Weakness**: "Table 1 data corruption makes empirical claims unverifiable." — This is a PDF parser artifact, not a paper problem. The bold formatting interferes with extraction.
- **Weakness**: "No finite-sample convergence bounds." — Demanding finite-sample bounds for an asymptotic bootstrap consistency result is beyond what is standard for this type of contribution.
- **Weakness**: "Reproducibility concerns about undisclosed code, seeds, preprocessing." — Per rules, reproducibility nitpicks about implementation details are removed.
- **Weakness**: "No Broader Impact section." — ICLR does not mandate a Broader Impact section; this is a formatting/style nitpick.
- **Strength**: "The paper is well-written / the topic is important." — Generic strengths removed per rules.

## Novel Insights

The paper reveals an underappreciated structural insight: in standard two-stage mean-plus-volatility models, the constant bias incurred when learning log-volatility via mean-forecasting methods is algebraically self-eliminating because the *same* biased volatility estimate appears in both the normalization divisor and the rescaling multiplier during inference. This means practitioners need not correct for the well-known Jensen's inequality bias in log-variance estimation—a non-obvious practical consequence that removes a common objection to this type of pipeline. However, this elegant cancellation also creates a hidden fragility: if the volatility model is ever re-estimated or modified between the normalization and rescaling steps (e.g., due to online updating), the bias no longer cancels, and the predictive distribution can become miscalibrated in ways that are difficult to diagnose.

## Suggestions

- Add empirical diagnostics for the i.i.d. assumption: report Ljung-Box test p-values and autocorrelation plots for η̂_t and η̂²_t across datasets, and discuss whether block bootstrap would be more appropriate when dependence persists.
- Provide wall-clock inference time comparisons between base models and DualRes-augmented models to enable practitioners to assess the practical cost-benefit tradeoff.
- Explicitly acknowledge the diagonal G limitation for multivariate settings and discuss it as a scope restriction; consider noting that extending to non-diagonal G (e.g., via Cholesky-based approaches) would be a natural future direction.
- Add an ablation isolating the contribution of volatility modeling versus resampling to substantiate the claim that *both* components matter.
- Discuss failure modes explicitly: identify conditions under which DualRes degrades performance (e.g., when the volatility model is poorly specified or when residuals are near-i.i.d. Gaussian) and provide guidance on when practitioners should avoid applying the framework.

---

**Axis assessments:**

- **Novelty**: Moderate. The synthesis of classical bootstrap inference with modern deep learning forecasters is useful, but the individual components (two-stage mean/volatility estimation, i.i.d. bootstrap) are well-established in statistics. The log-transformation trick for volatility estimation is a practical engineering insight, not a deep conceptual advance.

- **Technical soundness**: Moderate. The theoretical result is correct under its stated assumptions, but the assumptions (i.i.d. residuals, smooth G) are not verified against the implementation and may not hold in practice. The bias cancellation argument is sound but the framework lacks robustness analysis for when assumptions are violated.

- **Empirical support**: Moderate. Improvements are demonstrated across multiple datasets and base models, but the evaluation is ablation-only (no standalone probabilistic baselines), key theoretical assumptions are unvalidated, and several model-dataset combinations show degradation that is not analyzed.

- **Significance**: Moderate-to-good. The framework addresses a genuine practical need—upgrading deterministic forecasters to probabilistic ones—and the flexibility it provides is valuable. However, the significance is tempered by computational overhead, limited multivariate modeling, and the inability to extrapolate beyond observed residuals.

- **Clarity**: Good. The paper is well-structured, algorithms are clearly presented, and the notation is consistent. The theoretical section is dense but accessible, and the empirical section provides useful visualizations of prediction intervals and residual distributions.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
