Now I have enough context. Let me synthesize my final review, carefully cross-checking the harsh critic's claims against the actual paper content.

Let me verify some specific claims about the paper:

1. The harsh critic claims the "4K" denominator seems arbitrary and could cause problems. Let me check: The paper specifies κ(x,x') = Ψ(x-x') ≤ K with Ψ(0) = K. Since κ(x,x) ≤ K and ‖μ_P‖²_ℋ = E[κ(x,x')] ≤ K, each norm squared is at most K. So ‖μ_P‖² + ‖μ_Q‖² ≤ 2K, making 4K - ‖μ_P‖² - ‖μ_Q‖² ≥ 2K > 0. The denominator is indeed bounded away from zero. However, the harsh critic's claim that ‖μ_P‖² can approach K is correct - it can. But 4K - 2K = 2K > 0, so the denominator doesn't vanish. The "4K" choice vs "2K" question still stands though as a design choice.

2. The harsh critic claims the advantage in Theorem 10 is only with probability ≥ 1/65. Let me verify: The paper states "Furthermore, following holds with probability ς ≥ 1/65 over samples X and Y." This is indeed a very weak bound.

3. On the "MMD is less informative" motivation: This is actually a reasonable point. The paper's Figure 1 argument is that for the same MMD value, different norm configurations yield different p-values, suggesting different "closeness levels." The harsh critic argues this simply reflects variance differences affecting power - not a deficiency in MMD as a closeness measure. This is a substantive philosophical point but the paper's perspective has merit: if you want a measure that correlates with statistical distinguishability (which is what closeness testing cares about), then MMD alone doesn't capture it.

Let me now compile the final review.</think>Now I have sufficient context from similar papers and the detailed critic reviews. Let me synthesize the final review.

## Summary

This paper proposes the norm-adaptive maximum mean discrepancy (NAMMD) for kernel-based distribution closeness testing (DCT). NAMMD scales MMD by the norm of distribution embeddings in an RKHS, arguing that this makes the measure more informative for comparing closeness levels across distribution pairs with different norms under a common kernel. The paper provides asymptotic distribution results, sample complexity bounds, and theoretical results claiming NAMMD achieves higher test power than MMD, validated on synthetic and real-world datasets.

## Strengths

- **Addresses an underexplored problem**: Distribution closeness testing (DCT) beyond discrete/TV-based settings is genuinely important, and extending DCT to kernel methods for complex data is a meaningful direction.
- **Comprehensive theoretical analysis**: The paper provides asymptotic distribution results (Theorem 2), concentration bounds (Lemmas 6–7), sample complexity (Theorem 8), Type-I error control (Theorem 5), and power comparisons (Theorems 10, 12). This is a thorough theoretical treatment for a new statistic.
- **Practical case studies**: The ImageNet variant comparison, confidence margin evaluation, and adversarial perturbation detection experiments (Figures 3–5) demonstrate a concrete application pipeline for DCT without ground-truth labels, which is practically relevant.

## Weaknesses

### Fatal
None.

### Major

- **The core motivation that "MMD is less informative" conflates effect size normalization with statistical power**: The paper's central claim is that MMD is deficient because distribution pairs with the same MMD value but different RKHS norms yield different p-values (Figure 1). However, this is simply the standard statistical phenomenon that smaller variance (associated with larger norms) produces smaller p-values for the same effect size. It does not demonstrate that MMD is a defective closeness measure—rather, it shows MMD is a proper distance (invariant to scale transformations of the data) while NAMMD deliberately conflates distance with concentration. Whether this conflation is desirable depends on the application, and the paper does not provide a decision-theoretic or task-level justification for why closeness should be normalized by variance/concentration. The p-value argument for defining "closeness levels" is circular: it defines closeness post hoc by what the test resolves rather than by a pre-established metric.

- **The NAMMD normalization lacks principled justification and raises design questions**: The denominator $4K - \|\mu_P\|^2 - \|\mu_Q\|^2$ is introduced to make NAMMD ∈ [0,1] and increase with norms, but no derivation from an optimization objective, invariance property, or approximation of a canonical metric is provided. Why 4K rather than 2K or another constant? While the denominator is bounded away from zero (since $\|\mu_P\|^2 \leq K$ and $\|\mu_Q\|^2 \leq K$, so the denominator ≥ 2K > 0), the specific choice of 4K is not justified beyond producing [0,1] range. The normalization is also not invariant to kernel rescaling κ → cκ, and the implications of this sensitivity are not analyzed.

- **Theoretical "higher test power" claims are weaker than stated**: Theorems 10 and 12 prove one-sided dominance (if MMD rejects, NAMMD also rejects) plus existence of cases where NAMMD succeeds while MMD fails—with probability only ≥ 1/65. This is not a uniform power improvement; it establishes that NAMMD's power is at least as high under specific distributional conditions (including the condition $\|\mu_{P_1}\| + \|\mu_{Q_1}\| < \|\mu_{P_2}\| + \|\mu_{Q_2}\|$ in Theorem 12), but does not show that the improvement is practically relevant or typical. The "higher test power" narrative in the abstract and throughout the paper overstates what these theorems deliver.

- **Empirical improvements over MMD in the core comparison (Table 1) are marginal and within noise**: The differences in two-sample test power between NAMMD and MMD under the same kernel are consistently very small—e.g., 0.600→0.616 on blob/Gaussian, 0.563→0.566 on higgs/Gaussian, 0.707→0.713 on hdgm/Gaussian. Many improvements fall within one standard deviation. No statistical significance tests are provided. This aligns with the concern raised for similar divergence proposals (e.g., Representation JSD, where reviewers noted "there is no clear advantage of RJSD-D over MMD-D, as the gaps are very small"). The practical value of replacing MMD with NAMMD for standard two-sample testing is not convincingly demonstrated.

### Minor

- **The norm-ordering condition $\|\mu_{P_1}\| + \|\mu_{Q_1}\| < \|\mu_{P_2}\| + \|\mu_{Q_2}\|$ in Theorem 12 is unverified**: This condition is essential for NAMMD's provable advantage in DCT. The paper states it is "often met in practice" because norms are "typically positively correlated with MMD value," but provides no empirical analysis of how often it holds or what happens when it fails. A failure case analysis would strengthen the paper.

- **Table 2's comparison of NAMMD (kernel-based) vs. Canonne's TV test (histogram-based on 50-element discrete support) is asymmetric**: The advantage may come primarily from using kernel embeddings on real-valued data rather than from the NAMMD statistic itself. Notably, at ε'=0.7 on blob, the TV-based test actually outperforms NAMMD, which the authors attribute to "preserved structure in 2D"—but this shows that NAMMD is not universally superior even in the one comparison provided.

- **Type-I error experiments for ε > 0 are relegated to the appendix**: For a hypothesis testing procedure, verifying that Type-I error is properly controlled in finite samples (not just asymptotically) is essential. The main text should present at least basic Type-I error calibration.

- **Global kernel selection for multi-pair DCT is acknowledged as open**: The paper's main application scenario (comparing closeness across multiple distribution pairs) requires a fixed global kernel, but the authors admit that deriving a test power estimator for multiple distribution pairs "remains an open question." This limits the practical applicability of the DCT framework they propose.

### Trivial

- The column label "Canonne's NAMMD" in Table 2 is confusing, as Canonne's test uses total variation rather than NAMMD. It should be labeled "Canonne's TV test."

## Nice-to-Haves

- Comparison against alternative normalization schemes (e.g., variance-normalized MMD, MMD/(‖μ_P‖+‖μ_Q‖)) in the closeness testing experiments to demonstrate that the specific $4K - \|\mu_P\|^2 - \|\mu_Q\|^2$ denominator is principled rather than arbitrary.
- Empirical validation of the norm-ordering condition across all experimental settings, reporting how frequently it holds.
- Statistical significance tests (e.g., paired t-tests or bootstrap CIs) for the differences in Table 1.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"NAMMD is not invariant under kernel rescaling"** (harsh critic): While true, this is not necessarily a flaw—invariance under kernel rescaling would prevent NAMMD from capturing the norm information that is its entire purpose. The non-invariance is by design, though it should have been noted.

- **"The denominator vanishes"** (harsh critic): The critic claimed "the denominator vanishes or becomes arbitrarily small as $\|\mu_P\|^2 + \|\mu_Q\|^2 \to 4K$." Since $\|\mu_P\|^2 \leq K$ and $\|\mu_Q\|^2 \leq K$, the sum is at most 2K, so the denominator is at least 2K > 0. This claim is factually incorrect.

- **"NAMMDFuse comparison uses different sample sizes"** (harsh critic): The paper explicitly states that NAMMDFuse, MMDFuse, MMDAgg, and ACTT use twice the test sample size because they don't require training, and acknowledges this is intentional. This is standard practice in the testing literature (training-free methods can use all data for testing). This is not an unfair comparison—it's a legitimate design difference, not an asymmetry that favors the baselines.

- **"No comparison with any normalized/scaled MMD variant"** (spark reviewer): While desirable, the paper's primary comparison is with the original MMD under the same kernel, which is the most direct comparison. Alternative normalizations would require separate derivation of asymptotic distributions and testing thresholds, which is beyond the scope of this paper.

- **"The connection between TV-based ε' and NAMMD-based ε is unclear"** (spark reviewer): While the mapping between scales is relevant, the paper sets ε = NAMMD(P₁,Q₁,κ) directly in Definition 3, making this a threshold set from reference distributions rather than an arbitrary cross-scale comparison. The concern is noted but not a fundamental flaw.

## Novel Insights

The paper's central observation—that the same MMD value can correspond to very different levels of statistical distinguishability depending on distribution concentration—is both valid and useful for practical testing. However, this is better understood as a fact about *test power* under MMD rather than a deficiency of MMD as a *metric*. NAMMD effectively incorporates a variance-based correction into the distance measure itself, making it implicitly a "power-weighted" metric. This is a reasonable engineering choice for closeness testing, but the framing as correcting a flaw in MMD is misleading—it's more accurately described as designing a statistic optimized for higher test power through norm-adaptive scaling, at the cost of losing the pure-distance interpretation that MMD provides.

## Suggestions

- Reframe the motivation more honestly: rather than claiming MMD is "less informative," present NAMMD as a variance-adjusted statistic that achieves higher test power than MMD under certain conditions, with a clear acknowledgment of when those conditions fail.
- Add failure case analysis for the norm-ordering condition in Theorem 12—show empirically what happens when it's violated and how frequently it holds.
- Provide paired significance tests for the improvements in Table 1 to establish whether the observed gains are statistically meaningful.
- Include at least one additional DCT baseline (e.g., a kernel-based method adapted for closeness testing) to make the DCT experimental comparison more informative than the current apples-to-oranges comparison with Canonne's TV test.

## Score and Decision

Calibration papers:
- **Representation JSD (WnqD3EiylC)**: New RKHS divergence with small empirical improvements over MMD, rejected with scores 3–6 (avg ~5). Reviewers criticized marginal gains and lack of clear advantage over MMD.
- **Regularized MMD (yqaN7MfkFU)**: MMD variant for variable selection, rejected with scores 3–6. Criticized for marginal improvements and lack of theoretical grounding.
- **Practical Kernel Learning for CI Test (GPcSYm89wK)**: Kernel parameter optimization for CI tests, rejected with scores 3–5. Criticized for marginal power improvements over median heuristic.
- **Model Equality Testing (QCDdI7X3f9)**: MMD-based two-sample test for a practical application, accepted (poster) with scores 6–8. Praised for practical relevance and clear methodology.
- **PQMass (n7qGCmluZr)**: New distribution comparison method, accepted (poster) with scores 6. Praised for simplicity and effectiveness.

This paper shares the core weakness of Representation JSD—a new divergence/statistic with marginal empirical improvements over MMD and theoretical advantages that are weaker than claimed. The NAMMD improvements in Table 1 are consistently within standard deviations, mirroring the "no clear advantage" criticism that sank RJSD (scores 3–6, rejected). The theoretical power advantage results (probability ≥ 1/65) are far weaker than their presentation suggests. However, the paper does address a genuinely important problem (kernel-based DCT) and provides a more complete theoretical framework than RJSD, including sample complexity and Type-I error control. The DCT application scenario is novel and practically relevant, going beyond standard two-sample testing.

The paper has a meaningful problem formulation and solid technical framework, but the core contribution—a norm-adapted MMD—is not convincingly shown to provide material practical advantages over standard MMD. The theoretical "higher test power" results are technically valid but practically uninformative. This places it below accepted papers like Model Equality Testing and PQMass that demonstrated clear, substantial empirical gains, and roughly at the level of the rejected RJSD paper.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>