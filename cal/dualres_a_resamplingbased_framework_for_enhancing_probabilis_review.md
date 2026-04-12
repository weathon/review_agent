=== CALIBRATION EXAMPLE 44 ===

# Final Consolidated Review
## Summary
This paper proposes DualRes, a modular framework for probabilistic time-series forecasting that augments an arbitrary mean forecaster with a second-stage volatility model and residual resampling. The key idea is to estimate conditional mean and conditional volatility separately, normalize residuals, and then resample these normalized residual vectors to generate predictive distributions, aiming to handle heteroskedasticity and non-Gaussian residuals without committing to a parametric output distribution.

The paper’s main value is practical: it offers a plug-in way to turn deterministic or mean-focused forecasters into probabilistic ones, and the experiments show sizable gains over the authors’ chosen wrappers/baselines on several datasets. However, the theoretical guarantees are limited to an i.i.d.-innovation setting, and the empirical evaluation does not fully disentangle which part of the method is responsible for the gains or establish how broadly they extend beyond the specific baseline constructions used here.

## Strengths
- **A genuinely modular probabilistic forecasting recipe that can retrofit mean forecasters.** The paper’s most distinctive strength is that DualRes only requires mean forecasts for both the conditional mean and the transformed volatility process, as stated in Remark 1 and operationalized in Algorithms 1–2. This makes it straightforward to attach uncertainty quantification to models such as DLinear, PatchTST, and TimeMixer, which are not natively probabilistic.
- **Explicit treatment of both conditional heteroskedasticity and nonparametric residual distribution modeling.** Rather than only replacing a Gaussian head with another parametric family, the method separately models volatility and then resamples normalized residuals. Figure 2 directly supports the motivation that normalized residuals need not look Gaussian, and the multivariate inference procedure resamples whole residual vectors, which is a sensible mechanism for preserving cross-sectional dependence.
- **The log-squared transformation argument is a useful technical device.** Section 4.1 explains how the volatility-learning problem can be turned into a mean-forecasting problem via \(R(x)=\log(x^2)\), and the paper gives a concrete argument for why the constant bias introduced by this transform cancels in the normalization/rescaling pipeline. This is a practically relevant construction, not just a generic “we use a second model” story.
- **Empirical improvements are often large on the reported setup.** On the authors’ evaluation protocol, DualRes improves CRPS/MAEC for many univariate settings and often improves CRPS/MAEC/ES in multivariate settings as well, with some especially large gains (e.g., on Exchange and Electricity in the tables). These results make the method practically interesting even if the evaluation is not yet fully convincing as a broad claim.
- **The paper does provide a scoped theoretical statement rather than purely heuristic motivation.** Theorem 1 is not a full end-to-end forecasting validity theorem, but it does formally justify consistency of the empirical distribution of normalized residuals under stated assumptions. The theory is limited, but it is relevant to the resampling component actually used.

## Weaknesses

###: Fatal
- None.

### Major:
- **The empirical comparisons do not fully substantiate the broad headline claim of “enhancing probabilistic forecasting” across existing algorithms.**  
  The strongest gains in Table 1 come from converting mean-forecasting models into probabilistic ones via a fairly weak baseline construction: “their distributional indices are obtained through fitting a t-distribution to the predictive values.” Beating such parametric wrappers mainly shows that nonparametric residual modeling can outperform a simplistic parametric uncertainty layer; that is useful, but narrower than the paper’s framing. For natively probabilistic models like DeepAR, the comparison is more meaningful, but overall the evaluation still leaves open whether the gains come from a generally superior probabilistic forecasting framework or from outperforming relatively limited uncertainty baselines.
- **The core theoretical justification depends on i.i.d. normalized residuals, but the paper does not empirically verify that this assumption is even approximately satisfied after the proposed normalization.**  
  Theorem 1 explicitly assumes “\(\eta_t\) are independent and identical distributed,” and Algorithm 2 then samples these residuals i.i.d. with replacement. This is a real limitation, not a reviewer misunderstanding. The paper argues that the volatility model removes conditional heteroskedasticity, but it does not show residual autocorrelation / dependence diagnostics after normalization. Since the practical validity of simple residual resampling hinges on this whitening step, the absence of such analysis materially weakens the technical soundness of the end-to-end story.
- **The paper does not isolate the contribution of its main components.**  
  The experiments compare “base model” vs “base model + DualRes,” but DualRes combines at least two conceptually distinct interventions: (i) a conditional volatility model and (ii) nonparametric residual resampling. There is no ablation comparing, for example, parametric residuals + volatility model, resampling without volatility normalization, or simpler residual bootstrap variants. As a result, it is hard to tell whether the reported gains are driven mainly by heteroskedastic scaling, by nonparametric residual modeling, or by both. This matters because the paper’s conceptual claim is precisely about the importance of both ingredients.
- **The novelty is moderate rather than strong at ICLR standards.**  
  The paper is best understood as a modernized residual-bootstrap / heteroskedastic forecasting pipeline wrapped around neural forecasting models, rather than a fundamentally new probabilistic forecasting paradigm. The contribution is practical and reasonably well-motivated, but the method mostly recombines familiar ingredients: mean forecasting, volatility estimation through transformed residuals, and residual bootstrap sampling. That does not invalidate the paper, but it lowers the bar for empirical and technical validation, which here still feels incomplete.

### Minor
- **The theoretical scope is narrower than the presentation sometimes suggests.**  
  Theorem 1 establishes convergence of the empirical CDF of normalized residuals under strong assumptions, including uniform convergence of \(\hat F\) and \(\hat G\) over the full input domain. It does not provide a substantive multi-step predictive validity guarantee for the final forecast distribution after recursive simulation, nor does it analyze error accumulation in the multi-step recursion of Algorithm 2. The paper is not mathematically wrong here, but the theory supports a more limited claim than the prose occasionally implies.
- **The assumption behind the log-squared volatility transformation is restrictive and deserves more discussion.**  
  Section 4.1 requires that \(G_i^2(\cdot)\) depends on past residuals “only through their element-wise squares.” The authors do state this assumption explicitly and tie it to GARCH-style modeling, so this is not a hidden flaw. Still, it restricts the class of volatility mechanisms that the proposed learning trick can represent, and the practical implications of that restriction are not explored.
- **Compute overhead is acknowledged but not quantified.**  
  The method requires repeated resampling and recursive generation at inference; Appendix B states \(B=100\) forecast samples are used for CRPS/MAEC. Section 6 notes computational complexity as a limitation, but the paper gives no runtime or inference-cost comparison, making it difficult to judge the practical trade-off relative to the reported gains.
- **Calibration analysis is somewhat limited.**  
  The paper reports CRPS, MAEC, and ES, which are reasonable choices, but MAEC averages over a set of central coverage levels and may miss failures in tails or specific quantile regions. Since the method is motivated partly by better uncertainty quantification and risk assessment, more granular calibration evidence would strengthen the empirical case.
- **Some claims around “stability” are not well supported.**  
  The paper interprets narrower confidence intervals across runs as increased stability. That is suggestive, but narrower intervals over repeated runs are not, by themselves, evidence of better uncertainty quantification or better calibrated forecasts. The stronger evidence is in the scoring metrics, not the interval-width narrative.

### Trivial
- **Robustness to distribution shift is only discussed as a limitation, not tested.**  
  This is not a core flaw because the paper explicitly scopes it as a limitation (“if future observations have a distributional shift, the proposed method may no longer be reliable”), but a small empirical stress test would have made the boundaries of the method clearer.
- **The handling of zero or near-zero residuals under the \( \log(x^2) \) transform is not discussed.**  
  Since the volatility model is trained on \(\log(\hat\zeta_t^2)\), a short practical note on numerical stabilization would improve clarity, even if this is likely straightforward in implementation.

## Nice-to-Haves
- Add component ablations: volatility model only, resampling only, and simpler residual-bootstrap variants.
- Diagnose whether normalized residuals are actually close to i.i.d. after the proposed second stage, e.g., with ACF/PACF or dependence tests on \(\hat\eta_t\).
- Include more detailed calibration plots such as PIT histograms, reliability diagrams, or tail-coverage tables.
- Quantify inference/runtime overhead and sensitivity to the number of resamples \(B\).
- Compare against dependent-resampling variants (e.g., block bootstrap) as an extension, especially since the current method’s theory assumes away residual dependence.
- Clarify the practical numerical treatment of the log-squared transform for tiny residuals.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper mischaracterizes diffusion models because Gaussian forward/noise assumptions do not imply Gaussian observation distributions.”**  
  This may be directionally fair, but it leans into broad external positioning rather than the core technical assessment of this submission, and I cannot verify the full external claim set here. I therefore do not keep it as a central weakness.
- **Complaints about missing specific external baselines such as Chronos or other named works.**  
  Per instruction, I do not criticize the paper for omitting particular related works or named baselines that require external confirmation. The retained version of this point is the more general and paper-verifiable concern that the current baseline construction is limited.
- **Reproducibility complaints about heuristic hyperparameter selection from ACF plots and omitted implementation details.**  
  The appendix already provides substantial setup details, and hyperparameter-selection nitpicks of this sort are not central enough to retain.
- **Concerns that some cited models/datasets/tools may be unavailable or unverifiable.**  
  Removed by rule.
- **Pure formatting/table parser issues.**  
  The garbled values in the extracted text are parser artifacts, not paper problems.

## Novel Insights
The most important synthesis across the reviews is that this paper is strongest when framed as a **practical uncertainty retrofit for mean forecasters**, not as a broad new foundation for probabilistic forecasting. Its central technical gamble is not merely “bootstrap helps,” but that a learned volatility model can sufficiently whiten residuals so that simple i.i.d. resampling becomes valid enough in practice. That makes residual-whitening diagnostics the missing bridge between the theory and experiments: if the authors can show that the second stage meaningfully removes dependence and heteroskedasticity, the paper’s current modularity and empirical gains become much more convincing. Without that bridge, the work remains promising but only partially validated.

## Suggestions
- Add a focused ablation table separating the effect of volatility normalization from the effect of nonparametric residual resampling.
- Empirically test the paper’s key assumption by showing ACF/dependence diagnostics for residuals before and after normalization.
- Reframe the contribution more precisely as a modular enhancement / retrofit for existing forecasters, rather than implying a broadly superior probabilistic forecasting paradigm.
- Strengthen the empirical case with at least one stronger nontrivial probabilistic comparison beyond simple parametric wrappers around point forecasts.
- Report runtime or inference-cost overhead relative to baseline methods.
- Expand the calibration analysis beyond MAEC, especially for tails and extreme quantiles.
- Clarify in the main text exactly what the current theorem guarantees and what it does not guarantee for recursive multi-step forecast distributions.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
