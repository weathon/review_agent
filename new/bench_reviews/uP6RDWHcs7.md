Now I have sufficient context for calibration. Let me write the final review.

Key calibration anchors:
- Injective flows for star-like manifolds (accepted poster, avg 6): novel theoretical insight, exact Jacobian for a restricted class of manifolds, clear but limited scope
- Lifting Architectural Constraints of Injective Flows (accepted poster, avg 6.5): novel estimator for injective flows, good experiments, but similar concerns about exact density
- Riemannian Flow Matching (accepted oral, avg 8): strong framework paper with clear advantages, SOTA results
- Generative Marginalization Models (rejected, avg 5.6): marginalization-based approach, concerns about self-consistency
- Marginalization Consistent Mixture (rejected, avg 5.3): limited experiments, unclear novelty
- MLE for Flow Matching (rejected, avg 4): overclaimed, limited experiments

Now let me write my review.</think>## Summary

Marginal Flow proposes a density estimation framework where a model q_θ(x) is defined by marginalizing latent parameters w sampled from a learnable distribution q_θ(w) via an unconstrained neural network f_θ. The density is evaluated as a finite-sample average (1/N_c)∑q(x|w_{θ,i}), enabling both sampling and density evaluation without computing Jacobian determinants or solving ODEs. The framework supports lower-dimensional base distributions (manifold learning), flexible parametric families, and both forward and reverse KL training objectives.

## Strengths

- **Simple and flexible construction.** The idea of defining a density model as a marginal over neural-network-generated mixture components is conceptually straightforward, easy to implement, and naturally supports unconstrained architectures, variable-dimension base distributions, and different parametric families (Gaussian, Wishart, Dirichlet). The framework is genuinely modular.

- **Efficiency advantages are real and meaningful.** Avoiding Jacobian determinant computation and ODE solving while still enabling density evaluation is a genuine practical advantage over NF and FM. The runtime comparisons in Figure 3 (even if not perfectly controlled) demonstrate a significant computational edge, and the single-step sampling mechanism is inherently faster than diffusion or FM sampling.

- **Broad experimental coverage across diverse settings.** The paper evaluates on synthetic 2D densities (forward and reverse KL), SBI benchmarks, Wishart distributions on positive-definite matrices, and image latent-space manifold learning. This breadth demonstrates the framework's versatility.

- **The ability to swap q(x|w) families is elegant.** The Wishart experiment (Section 4.3) is a clean demonstration that the framework extends naturally to structured domains by choosing domain-appropriate parametric families, which competing approaches cannot easily do.

## Weaknesses

### Major:

- **The "exact density evaluation" claim is misleading.** The paper repeatedly claims "exact density evaluation" (Abstract, Table 1, Sec. 2.2) on par with normalizing flows. However, the evaluated quantity q_θ(x) = (1/N_c)∑q(x|w_{θ,i}) is a stochastic Monte Carlo approximation of the true marginal integral in Eq. 1. Because the w_{θ,i} are resampled at each evaluation, the computed density is a random variable - different evaluations at the same x yield different values. Normalizing flows provide a deterministic, closed-form log q_θ(x) for a fixed model. Marginal Flow provides an unbiased *estimate* of a marginal density. While the individual terms q(x|w_{θ,i}) are exactly evaluable (a genuine advantage over VAEs), calling the overall procedure "exact density evaluation" without qualification is an overclaim. The paper never analyzes the Monte Carlo variance or how N_c should scale with dimension or target complexity, making it impossible to assess when the approximation is adequate. This directly affects the comparison in Table 1 and the reverse KL experiments that rely on this "exact" likelihood.

- **Limited scalability evidence.** All experiments operate in very low dimensions: 2D synthetic benchmarks, low-dimensional SBI posteriors, Wishart matrices (10×10 and 100×100 with specialized parametrization), and 10–20 dimensional VAE latent spaces. There are no experiments on standard tabular density estimation benchmarks (e.g., UCI datasets like POWER, GAS, MINIBOONE) where NF, FM, and other baselines are commonly evaluated. Without such evidence, the claim that Marginal Flow is a general-purpose density estimation framework that "overcomes limitations altogether" is not substantiated for realistic problem scales.

- **N_c sensitivity and computational tradeoff is unanalyzed.** The central hyperparameter N_c controls a fundamental accuracy–cost tradeoff: larger N_c reduces Monte Carlo variance but increases evaluation cost linearly. The paper provides no systematic ablation of how density estimation quality, training stability, gradient variance, or test log-likelihood vary with N_c across different dimensionalities. This is a critical gap because the method's practical utility depends on N_c being manageable, and the "orders of magnitude faster" claim depends on N_c being small.

- **The conceptual distinction from mixture models is insufficiently established.** The paper argues that marginalization via resampling "prevents collapse to a GMM" and decouples modeling capacity from N_c (Sec. 2.1). But in practice, for any fixed N_c, the model at evaluation time *is* a finite mixture with N_c components. The only difference from a standard mixture is that components are resampled from a learned distribution. While this provides regularization and could aid optimization, the paper provides only a single qualitative figure (Fig. 1) to support the claim, with no quantitative comparison against well-tuned MDNs or large mixtures. This makes it hard to assess whether the improvement comes from the marginalization mechanism itself or just from having better optimization dynamics.

### Minor:

- **Manifold learning claims require qualification.** When q(x|w) is Gaussian with nonzero variance, the model support is all of R^d, not a lower-dimensional manifold. The model concentrates probability mass near a lower-dimensional structure (via small learned variances), which is useful but not a singular density on a manifold. The paper should clarify this distinction rather than claiming "learning distributions on lower-dimensional manifolds" without qualification.

- **SBI results are underreported.** The claim of "state-of-the-art" on the SBI benchmark is made in a single paragraph with results deferred to Appendix Figure 14, with no per-task numbers, error bars, or comparison details in the main text. For a competitive benchmark, this is insufficient to substantiate the claim.

- **Missing connection to Mixture Density Networks and kernel density estimation.** The closest related work (Bishop's MDNs, kernel density estimation with learned bandwidths) is not discussed. Since Marginal Flow is effectively a learned mixture model with resampled components, clarifying what the resampling mechanism adds beyond an MDN or neural KDE would help position the contribution.

### Trivial:

- **Gradient variance is not discussed.** Maximizing log[(1/N_c)∑q(x|w_{θ,i})] involves gradients through a sum of exponentials, which can have high variance for small N_c or high dimensions. While this doesn't invalidate the method, some analysis would strengthen the paper.

## Nice-to-Haves

- Systematic N_c ablation showing convergence properties and cost-accuracy trade-offs across dimensionalities.
- Evaluation on standard tabular density estimation benchmarks (UCI datasets) for direct comparison with published NF and FM results.
- Formal analysis of universal approximation properties of the model class.
- Discussion of failure modes and limitations (e.g., behavior in high dimensions, sensitivity to variance parameters σ).

## Removed Points

- **"Runtime comparisons are not credibly supported and may be unfair" (Harsh Critic #4):** The harsh critic claims the runtime comparisons lack sufficient detail on model sizes, architectures, and accuracy-controlled comparisons. However, the paper does provide comparison details in the appendix (Section A.3.1), and showing wall-clock time for density evaluation/sampling is a standard comparison. The concern about "accuracy vs. time" is valid but goes beyond what's standard in this field — most papers compare raw runtime. **Downgraded to nice-to-have rather than a standalone weakness.**

- **"The manifold + exact density claim is internally inconsistent" (Harsh Critic #3, severity level: "structural/fatal"):** While the manifold claim does require qualification (noted above as a minor weakness), the harsh critic overstates this as a "fatal" inconsistency. Many generative models (GMMs, VAEs, diffusion models) model densities that concentrate near manifolds without requiring singular measures. Calling the model "inconsistent" is too strong — the model does provide a legitimate full-dimensional density that concentrates near a learned structure, which is a common and useful approach. The issue is one of imprecise language, not logical contradiction. **Downgraded from structural to minor.**

- **"Comparison to finite mixture models is conceptually confused" (Harsh Critic #2, severity level: "structural"):** The harsh critic claims the marginalization story "does not stand as a principled differentiator." While the conceptual distinction could be sharper, the resampling mechanism does provide a genuine difference from standard GMMs: the mixture components are shared parameters of f_θ, enabling generalization beyond the N_c components drawn at any evaluation. The paper should have quantitatively demonstrated this advantage, but dismissing it as "conceptually confused" is too strong. **Kept as a major weakness but with more moderate framing.**

- **"SBI state-of-the-art claims lack sufficient detail" (Harsh Critic #5):** While the SBI section could be more detailed, the results are in the appendix and the C2ST metric is standard for SBI. This is more of a presentation concern. **Downgraded to minor.**

- **"Demand for GMM/KDE baselines" (Spark):** This overlaps with the mixture model comparison concern (Major weakness #4) but comparing against KDE specifically is less essential since KDE is not a learned generative model in the same category. **Merged into Major #4.**

## Novel Insights

The resampling-based marginalization mechanism provides an interesting optimization benefit that the paper doesn't fully articulate: by resampling the mixture components at each training iteration, the model effectively performs stochastic averaging over component locations, which acts as a form of implicit regularization. This is reminiscent of dropout-like noise in training, and it would clarify the contribution to explicitly connect it to such mechanisms rather than claiming it achieves a "true" marginalization that transcends finite mixtures.

## Suggestions

- Retract or heavily qualify the "exact density evaluation" claim. At minimum, acknowledge that the density is a Monte Carlo estimate with stochasticity that decreases with N_c, and provide empirical analysis of how much the estimate varies across resamplings for typical N_c values.
- Add a systematic N_c ablation study across different dimensionalities, reporting test log-likelihood, density quality metrics, and computational cost as a function of N_c.
- Evaluate on at least one standard tabular density estimation benchmark (e.g., POWER, GAS, or MINIBOONE from the UCI suite) to demonstrate scalability beyond 2D problems.
- Compare against mixture density networks (Bishop, 1994) with similar parameter counts to isolate the contribution of the resampling mechanism versus the mixture architecture.

## Score and Decision

**Calibration reasoning:**
- Injective flows for star-like manifolds (accepted poster, avg 6): novel but restricted scope, clear theoretical contribution, limited experiments. Marginal Flow has broader scope but weaker theoretical grounding and similar experimental limitations.
- Lifting Architectural Constraints of Injective Flows (accepted poster, avg 6.5): addresses similar problems (manifold + density), has stronger experimental validation including image data. Marginal Flow is less mature experimentally.
- Generative Marginalization Models (rejected, avg 5.6): similar "marginalization" framing, concerns about self-consistency and limited validation.
- Maximum Likelihood for Flow Matching (rejected, avg 4): overclaimed theoretical contribution with limited experimental evidence — shares the "overclaiming" weakness with Marginal Flow.

Marginal Flow introduces a clean and potentially useful framework, but the core positioning relies on an overclaimed "exact density" property, the experimental validation is limited to low dimensions, and the conceptual novelty relative to mixture models is insufficiently established. It falls between the stronger accepted papers (6-6.5) and the weaker rejected ones (4-5). The overclaiming pushes it below borderline accept.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>