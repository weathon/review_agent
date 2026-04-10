=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
## Summary
This paper proposes DualRes, a resampling-based framework for probabilistic time series forecasting. It separates the estimation of conditional mean and conditional volatility, then generates forecasts by bootstrapping normalized residuals. This design aims to handle conditional heteroskedasticity and non-Gaussian errors without parametric assumptions, while allowing any mean-forecasting model to be adapted for probabilistic forecasting. Experiments on six real-world datasets demonstrate improvements in probabilistic metrics when DualRes is applied to various base forecasters.

## Strengths
- **Flexible, model-agnostic design**: DualRes requires only a mean-forecasting model for both its mean and (via transformation) volatility components. This enables a wide range of existing algorithms (e.g., DeepAR, DLinear, PatchTST) to be used for probabilistic forecasting, as validated in ablation studies across multiple datasets (Table 1).
- **Explicit handling of non-Gaussianity and heteroskedasticity**: The framework explicitly models conditional volatility and uses a non-parametric bootstrap of residuals, avoiding the common Gaussian residual assumption. Figure 2 effectively shows real-world normalized residuals deviate from Gaussianity, and Figure 3 illustrates how prediction interval widths vary due to captured volatility.
- **Theoretical grounding**: Theorem 1 provides a convergence guarantee for the empirical distribution of normalized residuals under specified assumptions, linking the method to bootstrap theory and offering a principled foundation.
- **Effective multivariate extension**: By resampling the full residual vector, the method naturally captures empirical spatial dependencies in multivariate settings, evidenced by improvements in the energy score metric (Table 2).

## Weaknesses
### Major:
- **Incomplete comparison to the state-of-the-art for heteroskedasticity**: The empirical validation primarily compares DualRes-enhanced versions of base models against those models' default probabilistic adaptations (e.g., fitting a t-distribution). While this shows the value of the wrapper, it does not establish that DualRes surpasses dedicated, modern probabilistic forecasting methods that also explicitly model heteroskedasticity and non-Gaussian errors (e.g., advanced GARCH variants, specific non-Gaussian diffusion models). This gap limits the assessment of the method's relative contribution.
- **Lack of ablation studies on core components**: The paper does not systematically isolate the contribution of its key design choices. For instance, an ablation replacing the bootstrap with a parametric distribution (using the same estimated volatility) would quantify the value of non-parametric residual modeling. Similarly, the impact of the two-stage training versus a joint approach, or the choice of the logarithmic transformation for volatility, is not analyzed. This makes it difficult to attribute gains to specific innovations.
- **Strong theoretical assumptions with unclear practical satisfaction**: Theorem 1 relies on assumptions like the uniform consistency of the mean and volatility estimators (Assumption 2, Appendix A). While these are standard for theoretical analysis, their validity for complex, modern forecasting models (like transformers or RNNs) is not discussed. The paper does not provide guidance on how to verify or ensure these conditions in practice, leaving a gap between theory and application.

### Minor
- **Increased computational cost not quantified**: The method requires training two models and performing iterative resampling during inference. While mentioned as a limitation, the paper does not quantify the additional training/inference time or memory overhead relative to the base models, which is important for practical adoption.
- **Limited analysis of sensitivity and failure modes**: The performance of DualRes depends on the accuracy of the underlying mean and volatility models. There is no analysis showing how performance degrades when these models are misspecified (e.g., wrong lag order) or when the data exhibits distributional shift. While distributional shift is noted as a limitation, its impact is not empirically explored.
- **Evaluation metrics could be more comprehensive**: The primary metrics are CRPS and MAEC. While appropriate, supplementing them with metrics like weighted quantile loss for specific tails, or calibration plots, would provide a more nuanced view of distributional accuracy, especially for extreme events.

### Trivial
- **Minor typographical inconsistencies**: For example, the phrase "resampling assisted probabilistic forecasting" appears in the section title but the framework is called "DualRes" elsewhere. This does not affect understanding.

## Nice-to-Haves
- A sensitivity analysis on the bootstrap sample count `B` and the lag parameters `q` and `s` would provide practical tuning guidance.
- Visual case studies comparing the shape of the generated predictive distribution to a historical estimate for specific series would vividly demonstrate the capture of non-Gaussian features.
- An exploration on classic financial datasets with known stylized facts (like volatility clustering and heavy tails) would be a natural and compelling testbed given the method's inspiration.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strengths removed:**
- "The paper is well-written." (Generic strength, removed per hard rule)
- "The topic is important." (Generic strength, removed per hard rule)
- "The experiments are extensive." (While experiments are broad, this phrasing is generic. The specific evidence of experiments across six datasets is kept as a strength.)

**Weaknesses removed or weakened:**
- **"Theorem 1 assumes i.i.d. innovations, contradicting the handling of heteroskedasticity."** (Removed - This misreads the paper. Theorem 1 concerns the normalized residuals **η**_t, which the method aims to make approximately i.i.d. through correct volatility normalization. The theorem's assumptions are about the target of the estimation, not a limitation of the method's applicability to heteroskedastic data.)
- **"Comparisons are unfair because baselines use a naive t-distribution fit."** (Weakened to a minor point - The paper explicitly states this is the standard way these mean-forecasting models are adapted for probabilistic forecasting in frameworks like GluonTS. The comparison demonstrates DualRes's value as a *wrapper* for these models. The valid, major weakness is the lack of comparison to other advanced probabilistic methods, not the fairness of this particular baseline.)
- **"The bias elimination in the log-volatility estimation is circular and not addressed."** (Removed - The paper directly addresses this in Remark 1 and Section 4.1, arguing the bias self-eliminates in the normalization and resampling steps. While the reasoning could be expanded, the concern is addressed within the paper's scope.)
- **"Confidence intervals of '0.000(0.000)' are implausible."** (Removed as a likely formatting/rounding artifact in the provided text, not a substantive methodological flaw.)
- **"Missing related work."** (Removed per hard rule - The reviewer cannot externally confirm missing citations.)
- Requests for "theoretical proofs under misspecification" or "user studies" are moved to Nice-to-Haves, as they demand practices not standard for this type of algorithmic empirical paper (soft rule).

## Suggestions
- Conduct a direct empirical comparison with state-of-the-art probabilistic forecasting methods that explicitly model heteroskedasticity and non-Gaussian errors (e.g., advanced GARCH models, Non-stationary Diffusion) to clearly delineate the contribution.
- Perform a comprehensive ablation study to quantify the individual contribution of the volatility modeling and the non-parametric resampling components.
- Include a subsection quantifying the computational overhead (training/inference time, memory) of DualRes compared to the base models.
- Add a practical discussion or simple diagnostic on how users might check if their mean and volatility models are sufficiently accurate for DualRes to be effective.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
