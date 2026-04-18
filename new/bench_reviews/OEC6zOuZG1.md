## Summary

This paper studies the random feature model (RFM) under anisotropic (spiked covariance) data in the proportional asymptotic limit, asking when and how the RFM outperforms linear models. The authors extend universality results from isotropic to spiked covariance data (Theorem 1), establish that the RFM is equivalent to a noisy polynomial model whose degree depends on the input–label correlation (Theorem 2), and identify conditions under which the RFM reduces to a noisy linear model (Corollary 3). Numerical simulations support the theoretical predictions.

## Strengths

- **Well-motivated research question.** The gap between theory (isotropic data: RFM ≈ noisy linear model) and practice (RFM often beats linear models) is real and important. The paper tackles it with a clear theoretical framework.

- **Meaningful extension of universality to spiked data.** Theorem 1 extends the universality framework of Hu & Lu (2023) from isotropic to spiked covariance data. While the approach follows the same Lindeberg method, handling anisotropic inputs introduces non-trivial complications with the Hermite expansion structure, making this a legitimate technical contribution.

- **Insightful connection between correlation and polynomial degree.** Theorem 2 and the η condition (Eq. 15) provide a graded, principled characterization linking input–label alignment to the effective polynomial degree of an equivalent model. The key insight—that higher alignment requires higher-order polynomial models rather than linear ones—is valuable and cleanly articulated.

- **Clear boundary for the linear regime.** Corollary 3 precisely identifies when η = O(n^{−1/2}) suffices for linear equivalence, and Figure 1a effectively illustrates this boundary in (α, θ)-space.

- **Hermite-coefficient interpretation (Remark 4).** The interplay between Hermite coefficients of σ and σ_∗ determining whether the model reduces to lower-order equivalents is a useful and interpretable finding, well illustrated in Figure 2.

## Weaknesses

### Major

- **The headline claim that "RFM outperforms linear models" is not directly supported by the theoretical results.** The theorems establish *equivalences* between models under matching conditions (Theorems 1–2) and conditions for when equivalence *breaks* (Corollary 3). They do not prove that any concrete nonlinear activation (e.g., ReLU, tanh) achieves lower population risk than any linear model. The empirical comparison in Section 5 uses an "optimal linear activation" σ_linear(x) = a₀ + a₁x (Eq. 21) whose coefficients are "determined numerically to minimize the generalization error"—i.e., tuned with oracle access to the test distribution. This is not a standard linear baseline (e.g., ridge regression on raw features), and it actually *strengthens* the linear competitor, making the comparison even harder for nonlinear activations. Despite this, the nonlinear advantage appears only in specific regimes (high α, large θ), and in some regimes the oracle-tuned linear model actually outperforms ReLU/Softplus (Figure 3a). The paper's title and abstract make a strong claim ("Random Features Outperform Linear Models") that the theory and experiments do not rigorously deliver.

- **The η condition is central but opaquely connected to interpretable model parameters.** The quantity η in Eq. (15) governs all equivalence results, but its precise dependence on (α, θ, n, k) is not made explicit. The paper asserts that η = O(n^{−1/4}) "with high probability when β < 1/2" (start of §5), but this is stated without proof or reference to the appendix. For i.i.d. Gaussian f_i, the maximum over k ≈ cn rows of |v^T f_i| scales as √(log n)/√n, not 1/√n, and it is unclear whether the claimed probabilistic bound is correct or whether a union bound over k rows weakens the exponent. Since all nontrivial equivalence claims depend on this condition, the absence of a rigorous connection between η and the main model parameters leaves the reader unable to determine whether the proven regimes actually cover the simulation settings shown.

### Minor

- **Theorem 1, while not tautological, has limited conceptual novelty.** For ridge regression with squared loss, it is intuitive that matching the second-order statistics of (features, labels) yields matching performance. The real contribution lies in verifying conditions (10)–(11) for the spiked setting (done in Theorem 2), not in the universality statement itself. The presentation inflates Theorem 1's role as a "core theoretical contribution."

- **Assumption (A.6) (σ is odd) excludes ReLU, the most commonly used activation.** The paper acknowledges this only empirically ("empirical evidence suggests our findings remain valid"). Since all experiments with ReLU are therefore outside the theorem's scope, results like Figure 3 lack formal backing for the primary activation of interest.

- **The oracle-tuned polynomial baseline (Eq. 22) is hard to interpret as a genuine model.** The coefficients b₀,...,b₄ are chosen to minimize generalization error, which requires access to the population distribution. How this interacts with the ridge regression over ω is unclear (is ω still jointly optimized? Are b's chosen in an outer loop?), and the resulting comparison does not map to any implementable algorithm.

- **β < 1/2 restriction limits the theory's applicability.** Experiments in Figure 3c show results up to β ≈ 1, beyond the proven regime. The paper acknowledges this only in passing. It would be helpful to discuss whether β < 1/2 is fundamental or a proof artifact, and to include convergence plots validating that the asymptotic results hold at n = 400–500.

### Trivial

- The symbol t_i in Eq. (15) is undefined; context suggests f_i. This is a typo, not a conceptual issue.

## Nice-to-Haves

- A direct comparison with standard ridge regression on raw features (not within the RFM framework) would clarify whether the claimed advantage holds against genuine linear baselines.
- Quantification of the generalization error gap (not just equivalence conditions) between the RFM and the optimal linear model, even as an asymptotic expression, would substantially strengthen the "outperforms" claim.
- Experiments varying the polynomial degree l in the equivalent model to directly validate Theorem 2's prediction about how l must grow with η.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Theorem 1 is "tautological" or "almost definitional"** (Harsh Critic, point 2): Theorem 1 requires matching second moments up to o(1/polylog k) in operator norm, and then uses Lindeberg's method to prove convergence in probability of training and generalization errors to the same limit. This is a legitimate technical statement requiring proof; it is not a one-liner. However, the conceptual content—that moment matching implies performance matching for ridge regression—is somewhat expected, so the theorem's novelty is correctly identified as incremental rather than tautological.

- **No comparison with standard ridge regression on raw features** (Spark): This would be a nice addition but is not the paper's comparison framework. The paper studies different activations within the RFM, which is a self-consistent comparison. Adding ridge regression on raw x would change the architecture entirely and is outside the paper's stated scope.

- **Label-flipping confounds SNR with correlation** (Spark): While label-flipping does affect SNR, the paper uses the norm of input–label correlation as the x-axis, which controls for this. The experiment is designed to show the qualitative trend, not to provide a rigorous test of the theory.

- **CIFAR-10 is out-of-distribution w.r.t. theoretical assumptions** (Harsh Critic, point 5): This is standard for theoretical ML papers—real-data experiments illustrate qualitative relevance. The paper does not claim the CIFAR-10 experiment validates the theorems; it says it illustrates "how our results translate to real-world datasets" and confirms "our insights." The framing is acceptable.

- **Missing experiments with multiple spike directions** (Spark): The rank-1 spiked covariance is a standard and tractable model in this area. Multiple spikes would be a natural extension but is outside the paper's stated scope.

- **(A.3) appears garbled** (Harsh Critic): The notation "y ~ σ_*(εx/...)" vs. "y = σ_*(ξ^T x/...)" appears to be a presentation inconsistency in the assumptions section, not a substantive error, since the model is correctly stated in Eq. (6) and used consistently thereafter.

## Novel Insights

The key insight—that under spiked covariance data, the degree of the equivalent polynomial model for the RFM is governed by the input–label correlation parameter η, which itself depends on the alignment α between the spike direction γ and the label direction ξ—is genuinely novel and goes beyond prior universality results that collapse everything to a linear model under isotropy. When α is large, higher-order Hermite terms become relevant, and nonlinear activations can exploit this structure in a way that linear models cannot. This provides a principled answer to "when does nonlinearity help?" that is more nuanced than a simple binary.

## Suggestions

- Reframe the title and abstract claims to accurately reflect what is proven: the paper establishes *conditions for equivalence* between the RFM and noisy polynomial models, and shows *empirically* that nonlinear activations outperform oracle-tuned linear activations in high-correlation regimes. Remove "outperforms linear models" as a definitive claim unless a rigorous risk comparison is added.
- Provide a proof (at least in the appendix) for the claim that η = O(n^{−1/4}) under the assumed settings, including proper handling of the maximum over k rows and the union bound.
- Discuss (even briefly) whether the β < 1/2 restriction is a proof artifact or fundamental, and provide some empirical evidence that the theoretical predictions hold at the finite n used in experiments.

## Score and Decision

Calibration anchors:
- **zxqdVo9FjY** (Generalization for Spiked Covariances): Reject, scores 3–6 (avg ~4.8). Restricted assumptions, incremental contribution, unclear applicability.
- **MY8SBpUece** (Non-Linear Feature Learning with One Gradient Step): Reject, scores 5–6 (avg ~5.5). Gaussian assumption, strong assumptions, but genuine conceptual progress.
- **OdpIjS0vkO** (More is Better): Accept (poster), scores 6–8 (avg ~6.7). Clean theoretical results, well-motivated, good experiments.
- **UrKbn51HjA** (Breakdown of Gaussian Universality): Accept (poster), scores 3–6 (avg ~5.25). Extends universality to breakdown conditions; comparable theoretical contribution but with cleaner framing.

This paper has genuine theoretical contributions extending universality to spiked data, but the headline claim overreaches what is proven, the central η condition lacks a rigorous probabilistic analysis, and the experimental baselines use oracle tuning. Relative to the spiked covariance paper (zxqdVo9FjY, rejected at ~4.8), this paper has more conceptual novelty (the polynomial-degree connection). Relative to the feature-learning paper (MY8SBpUece, rejected at ~5.5), it has a similar profile of interesting ideas under restrictive assumptions with an overclaim in the framing. Relative to the Gaussian universality breakdown paper (UrKbn51HjA, accepted at ~5.25), this paper has comparable technical depth but weaker framing of claims.

I assign a score of **5** — the theoretical contribution of extending universality to spiked data and connecting polynomial degree to input–label correlation is real and interesting, but the overclaiming in the title/abstract and the gap between the proven results and the experimental setup (oracle baselines, ReLU outside theory scope, unproven η bound) prevent a stronger assessment.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>