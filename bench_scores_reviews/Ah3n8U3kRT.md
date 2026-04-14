## Summary
This paper introduces median clipping to zeroth-order (ZO) optimization and multi-armed bandits (MAB) with symmetric heavy-tailed noise, enabling rates that hold for any κ > 0 — including distributions with unbounded expectation — where prior methods based on gradient clipping degenerate as κ → 1. The three proposed algorithms (ZO-clipped-med-SSTM, ZO-clipped-med-SMD, Clipped-INF-med-SMD) achieve non-degenerating Õ(d²ε⁻²) iteration complexity for ZO optimization and Õ(√(dT)) regret for MAB by constructing an unbiased, bounded-variance gradient estimator via a batch median of two-point finite-difference samples along shared directions under a symmetric noise density assumption (Assumption 3).

---

## Strengths

- **Non-degenerating rates for κ ≤ 1 via a structurally novel estimator.** The core technical insight is that each ZO gradient sample g(x, **e**, ξ) is a scalar multiple of a fixed direction **e**, so the coordinate-wise median over samples with the same **e** reduces to a scalar median. Symmetry of the scalar noise then makes this median unbiased with bounded variance for any κ > 0, while the competing clipping-based approach (ZO-clipped-SSTM [20]) gives rates that blow up as κ → 1. This is a genuine and non-trivial departure from previous proof strategies.

- **Tight matching of the bounded-variance rate for the Lipschitz oracle.** For the Lipschitz oracle, Theorem 1's stochastic term Õ(d(M₂² + d∆²/κ^(2/κ))R²/(bε²)) replaces the prior Õ(1/b · (√d M₂'/ε)^(κ/(κ-1))) term from [20], eliminating the ε-exponent blowup. The paper is careful to state this matches the finite-variance lower bound only in terms of ε-dependence, and explicitly notes "for the symmetric noise only, we are not aware of any proved bounds," which is honest and appropriate.

- **Unified treatment across ZO and MAB.** The paper adapts median estimation coherently to constrained optimization (via mirror descent, Theorem 2) and to MAB via importance-weighted median estimators (Theorem 3), yielding an Õ(√(dT)) regret bound that matches the optimal lower bound under bounded variance. This unification strengthens the contribution's scope.

- **High-probability guarantees throughout.** All three main theorems provide high-probability (not just in-expectation) bounds, which is the appropriate and harder guarantee in the heavy-tailed regime where variance is infinite.

- **Empirical demonstration of median's advantage for κ ≤ 1.** Figure 3 (ZO experiments) directly shows that for α = κ ≤ 1, median-equipped methods converge where non-median counterparts diverge or stagnate, validating the headline theoretical claim in a controlled synthetic setting that mirrors the framework of prior work [20].

---

## Weaknesses

### Fatal
None identified.

### Major

- **Total oracle complexity obscured by "iteration" framing.** Each iteration of all three algorithms costs (2m+1)·b oracle calls, where m = ⌈2/κ⌉ + 1. As κ → 0, m → ∞, so the total oracle complexity scales as Õ(κ⁻¹ · d²ε⁻²) for the ZO methods — not Õ(d²ε⁻²) as the abstract implies. Table 1 does note "b/κ calls per iteration" for the proposed method vs. "b calls" for [20], but this factor is never consistently folded into a total oracle complexity comparison with baselines. A paper making strong claims about "any κ > 0" must prominently report total oracle cost; the current framing is technically correct but systematically misleading when comparing against prior work.

- **The σ² constant (4/κ)^(2/κ) grows superexponentially as κ → 0.** Lemma 1 gives σ² ∝ (2m+1)(4/κ)^(2/κ) for the independent oracle. As κ → 0 this diverges superexponentially (e.g., κ = 0.1 gives (40)^20 ≈ 10^32). The non-degenerating ε-exponent is preserved, but the constants render bounds vacuous for very small κ. The paper does not discuss any threshold below which the method becomes impractical, leaving the "any κ > 0" claim potentially misleading in practice.

- **Figure 1 (MAB) contradicts the paper's empirical superiority claim.** The figure description shows HTINF achieves *lower* average expected regret (~0.1 vs ~0.2 for Clipped-INF-med-SMD) and *higher* probability of best-arm choice (~0.9 vs ~0.6 for the proposed method). The paper's claim that "HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does" is not supported by these metrics — both HTINF's regret and best-arm probability appear to converge in Figure 1. If the paper means something specific by "convergence in probability of regret," it needs a clearly defined and explicitly plotted metric (e.g., tail probability of regret exceeding a threshold). As presented, Figure 1 shows the proposed method performing worse, not better, on all reported quantities.

- **MAB experiments use only d = 2, making the headline Õ(√(dT)) d-scaling claim empirically unvalidated.** Theorem 3's contribution over prior work is precisely the optimal Õ(√(dT)) regret in place of the degraded Õ(d^((κ-1)/κ) T^(1/κ)) scaling. Yet the only bandit experiment fixes d = 2, where the distinction between Õ(√(2T)) and Õ(2^((κ-1)/κ) T^(1/κ)) is trivial. Experiments with d ∈ {10, 50, 100} are essential to substantiate the claimed advantage.

### Minor

- **Algorithm 2 Step 4 likely contains a notation error.** It reads g^{k+1}_{med} = Med^m(x^{k+1}, **e**, {ξ}), but x^{k+1} has not been computed at that point in the loop; this should almost certainly be x^k. This weakens reproducibility confidence for the constrained algorithm.

- **The main text does not explain why the coordinate-wise median is unbiased.** The crucial insight — that each sample g(x, **e**, ξ^i) = (scalar) · **e**, so the "coordinate-wise" median is simply the scalar median times a fixed vector — is not stated in the main text. Without this, readers cannot readily verify Lemma 1's unbiasedness claim. The paper defers to the appendix ("We refer to Appendix A for more details"), but for the linchpin lemma of the whole paper, the main text should at least state the scalar-structure argument in a sentence.

- **INF-clip [7] is discussed as a direct competitor in related work but excluded from experiments.** The paper explicitly describes INF-clip (Section 4) and explains its architectural shortcoming (clipping before importance weighting). As the most architecturally similar competing method, its omission from Figure 1 is unexplained.

- **Assumption 3's translation to the MAB setting is unexplained.** Theorem 3 says "the conditional probability density function for each loss satisfies Assumption 3 with Δ, κ > 0," but Assumption 3 was defined for a two-point oracle with noise φ(ξ|x, y) and scale function B(x, y). In the MAB setting there is no pair (x, y) — what does B(x, y) correspond to? This mapping needs at least a brief clarification.

- **ZO experiments use only 3 launches in the main text.** With symmetric α-stable noise and heavy tails, 3 runs cannot characterize statistical variability meaningfully. The paper defers to the appendix for "enlarged number of launches," but the main figure should not be presented with this caveat unexplained.

### Tiny

- **The cryptocurrency portfolio experiment validates a full-feedback modification of Algorithm 3**, explicitly noted as departing from the MAB setting. While the authors acknowledge this, it means Section 5.2 provides no empirical support for the bandit theory and should not be presented as a validation of "Clipped-INF-med-SMD."

- **Remarks 1 and 2** (smooth and PL objectives) are stated without proofs in the main body. These are fine as extensions but contribute to a breadth impression that is not fully substantiated within the visible paper.

---

## Nice-to-Haves

- **Asymmetric noise analysis moved from appendix to main text**, with a quantitative degradation plot (e.g., optimality gap vs. skewness parameter), so practitioners can assess whether their noise is "close enough" to symmetric for the method to remain competitive.
- **Experiments varying d for both MAB and ZO** to test the theoretical d-scaling claims directly.
- **Total oracle call count on the x-axis** in Figure 3 (replacing iterations/samples), enabling a fair comparison that accounts for the (2m+1) per-iteration overhead.
- **A quantitative characterization of the practical κ threshold** below which the (4/κ)^(2/κ) constant makes the bounds vacuous, helping practitioners understand the method's practical regime.
- **Discussion of an adaptive or data-driven scheme for selecting m** when κ is unknown, complementing the current suggestion of a fixed m=3 grid search.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **[Harsh Critic] "Matches optimal is unfair without oracle-cost normalization" (abstract-level claim).** Partially valid (see Major weakness on total oracle cost), but the paper explicitly says in Theorem 1 that "optimal" means matching the finite-variance bound in terms of ε-dependence only, and Table 1 already shows "b/κ calls." The abstract claim is loose but the body is appropriately careful.

- **[Harsh Critic] "Theorem 1 is hard to parse."** The dual formula (independent / Lipschitz oracle) in one display is dense but standard for this sub-field. Not a substantive issue.

- **[Harsh Critic] "R = ‖x₀ − x*‖ requires knowledge of the optimum."** This is a standard requirement for clipped accelerated methods in the non-smooth setting (e.g., [15, 20]); criticizing it as a novel limitation is scope creep.

- **[Harsh Critic] "Broader impact statement is too thin."** This is a theoretical optimization paper; no specific social harms are foreseeable. The brevity is acceptable.

- **[Harsh Critic] "Acceleration appears only of theoretical interest because SGD outperforms SSTM in Figure 3."** The SGD-vs-SSTM comparison in the figure compares total samples, and it is well-known that accelerated non-smooth methods may need careful tuning of constants to outperform SGD empirically. The key empirical claim is median vs. non-median, not SSTM vs. SGD. This is not a weakness of the paper's contribution.

- **[Harsh Critic] Claim that Assumption 3 is "narrower than advertised" because it requires absolute continuity.** The paper says it "covers a majority of symmetric absolutely continuous distributions," explicitly signaling this scope. Criticizing the assumption for not covering discrete distributions is a scope expansion, not a flaw.

- **[Harsh Critic / Spark Finder] Missing related works.** Removed per instructions.

---

## Novel Insights

The most intellectually interesting insight not made explicit enough in the paper itself is the following: in the zeroth-order two-point estimator g(x, **e**, ξ) = (d/2τ)(f(x+τ**e**) − f(x−τ**e**) + φ(ξ|...)) · **e**, the noise enters only as a scalar multiplied by a fixed direction. This scalar structure means "coordinate-wise median" and "scalar median" are identical here — the difficulty of median estimation for general vector-valued quantities simply does not arise. The entire theoretical advantage of the paper (non-degenerating rates, unbiasedness of the median estimator) flows from this structure together with symmetry: the scalar noise being symmetric ensures the median is unbiased, and Assumption 3's tail envelope ensures the variance of the median estimator is bounded. Making this structural observation explicit in the main text would substantially clarify why the method works and why generalizing to asymmetric or non-scalar noise is genuinely hard.

---

## Suggestions

1. **Restate all main results in terms of total oracle calls**, not just iteration count. Show a table comparing total oracle calls for this work vs. [20] across a range of κ (e.g., κ = 0.5, 1.0, 1.5), so readers can see where the per-iteration (2m+1) overhead is and is not worth paying.

2. **Fix or clarify Figure 1**: Either (a) add an explicit plot of the probability that regret exceeds a fixed threshold (showing the proposed method's tail is lighter), or (b) replace the current metric with one that makes "convergence in probability" concrete. As it stands, the figure contradicts the paper's textual claims about superiority.

3. **Add MAB experiments with d ∈ {10, 50, 100} and include INF-clip as a baseline.** Without d-scaling experiments, the Õ(√(dT)) headline result has no empirical support distinguishing it from the degraded Õ(d^((κ-1)/κ) T^(1/κ)) baseline.

4. **Add one paragraph in Section 3.1.2 explaining the scalar structure of g(x, e, ξ) and why it makes the coordinate-wise median equal to the scalar median.** This is the intuitive key to Lemma 1 and should not be buried in an appendix.

5. **Fix the likely typo in Algorithm 2 Step 4** (x^{k+1} → x^k) and verify notation consistency across Algorithms 2 and 3.

6. **Add a brief quantitative discussion of the regime κ < 0.5**, acknowledging that (4/κ)^(2/κ) and m = ⌈2/κ⌉ + 1 together make the method impractical for very small κ, so the "any κ > 0" claim is an asymptotic statement rather than a practical guarantee.

---

## Evaluation

**Originality**: High. Applying median-based robust estimation to ZO optimization and MAB, exploiting the scalar structure of two-point gradient estimators, and establishing non-degenerating rates for κ ≤ 1 is a genuine and non-trivial contribution. The prior literature had a clear gap at κ = 1, and this paper fills it under a well-motivated symmetry assumption.

**Importance of the research question**: Meaningful. Heavy-tailed noise with κ ≤ 1 arises in finance and other domains, and the κ → 1 degeneration of prior ZO methods was a real theoretical limitation.

**Whether claims are well-supported**: Mixed. The theoretical claims are generally carefully stated and (based on visible proofs) plausible, but the empirical support for several headline claims is weak or internally contradictory (Figure 1, d=2 MAB).

**Soundness of experiments**: Below ICLR standard given the practical claims made. The ZO synthetic experiments are the most credible (direct comparison to prior work), but the MAB experiments have interpretability problems and the crypto experiment doesn't validate the bandit theory.

**Clarity of writing**: Generally adequate for a theory paper, with the notable exception that the central intuition behind Lemma 1 is missing from the main text, and Section 5.1's interpretation of Figure 1 is at odds with the figure itself.

**Value to the research community**: Moderate-to-high. The theoretical framework (Assumption 3, median estimator design, connection to MAB) is reusable, and the result fills a concrete gap. Value would be substantially higher with credible empirical validation.

**Contextualization relative to prior work**: Good for ZO (comparison with [19, 20] is concrete and quantified). Adequate for MAB (related works are described), though the absence of INF-clip from experiments is a gap.