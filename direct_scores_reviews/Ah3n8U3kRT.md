## Summary
This paper introduces median-clipping-based zeroth-order algorithms (ZO-clipped-med-SSTM and ZO-clipped-med-SMD) for non-smooth convex optimization, and extends the technique to the stochastic multi-armed bandit (MAB) problem under symmetric heavy-tailed noise with any κ > 0. The key innovation is a novel oracle model (Assumption 3) that encodes both symmetry and power-law tail behavior, enabling the construction of unbiased gradient estimators with bounded second moment even when the noise distribution has unbounded expectation (κ ≤ 1). For ZO optimization, the methods achieve $\tilde{O}(d^2\varepsilon^{-2})$ iterations — matching optimal rates for bounded-variance problems — for any κ > 0, whereas prior work (ZO-clipped-SSTM/SMD) degenerates as κ → 1 and is undefined at κ = 1. For MAB, the proposed Clipped-INF-med-SMD achieves $\tilde{O}(\sqrt{dT})$ regret, matching the optimal lower bound for bounded-variance settings.

## Strengths

- **Genuine extension of the heavy-tail frontier to κ ≤ 1.** Prior ZO methods ([19, 20]) achieve high-probability convergence only for κ ∈ (1, 2] and degenerate as κ → 1. This paper is the first to handle κ ∈ (0, 1] in the zeroth-order setting, including Cauchy noise with undefined expectation. This is a non-trivial barrier that is cleared by the combination of Assumption 3 (power-law envelope on the noise density) and the component-wise median estimator.

- **Technically non-trivial Lemma 1.** The unbiasedness of the median estimator under Assumption 3's symmetry, and the derivation of bounded second moment $\sigma^2 = O(dM_2^2 + d^2\Delta^2(4/\kappa)^{2/\kappa})$ requiring only $m > 2/\kappa$ samples, is a substantive technical result that goes beyond straightforward application of prior median analysis. The proof approach is explicitly noted as distinct from earlier works.

- **Rates matching bounded-variance optimal for ZO optimization.** Theorem 1 (Lipschitz oracle) achieves $\tilde{O}(\max\{d^{3/2}M_2R/\varepsilon,\; d(M_2^2 + d\Delta^2/\kappa^{2/\kappa})R^2/(b\varepsilon^2)\})$, which for fixed κ has the same ε and d scaling as the optimal ZO bound under bounded variance. Table 1 provides a clear, honest contrast against the baseline's $(\sqrt{d}\varepsilon^{-1})^{\kappa/(\kappa-1)}$ factor that blows up at κ = 1.

- **Empirical validation of the median clipping effect under extreme tails.** Figure 3 cleanly shows that for κ ≤ 1, median-clipping methods significantly outperform non-median counterparts, while matching them for κ > 1. This directly validates the core theoretical claim where it matters most.

## Weaknesses

- **Potential theoretical gap in the MAB section: symmetry under importance weighting.** The MAB algorithm applies the median operator to importance-weighted estimators $\hat{g}_{t,i} = g_{t,i}/x_{k,i}$ (for chosen arm) and $0$ otherwise. Even if the raw noise $\xi_t$ satisfies Assumption 3's symmetry, the importance-weighted estimator $\hat{g}_t$ has a manifestly asymmetric distribution (zero with probability $1 - x_{k,i}$, heavy-tailed with probability $x_{k,i}$). The paper does not show that $\hat{g}_t - \mathbb{E}[\hat{g}_t]$ satisfies Assumption 3, which is the premise on which Lemma 1's bounded second moment and unbiasedness are derived. If this gap is not addressed in the appendix proof, Theorem 3's regret bound rests on an unjustified application of Lemma 1. This is the most serious concern in the paper.

- **Experimental conclusion in §5.1 appears inconsistent with Figure 1.** The paper claims "HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does." Yet Figure 1 (per the figure caption) shows HTINF achieving the *highest* probability of best-arm selection (~0.9) and the *lowest* average expected regret (~0.1), while the proposed method stabilizes at probability ~0.6 and regret ~0.2. Claiming that HTINF "does not have convergence in probability" while it empirically dominates both metrics in Figure 1 is logically inconsistent with the displayed results, unless "convergence in probability" is being used in a narrow theoretical sense that must be made explicit. This casts doubt on the reliability of the paper's empirical narrative.

- **Growing constant $(4/\kappa)^{2/\kappa}$ is obscured in abstract and Table 1.** Lemma 1 and Theorem 1 explicitly include the factor $(4/\kappa)^{2/\kappa}$ in $\sigma^2$, which grows without bound as κ → 0, and $m = 2/\kappa + 1$ means per-iteration oracle cost also grows as $O(1/\kappa)$. The abstract's claim that methods "require $\tilde{O}(d^2\varepsilon^{-2})$ iterations for any κ > 0" and Table 1's non-degenerating rates absorb this exploding factor into the $\tilde{O}$. While the theorems themselves are honest, the top-level framing obscures the practical degradation for very small κ. At minimum the abstract or Section 6.2 should note that the hidden constant grows as $(4/\kappa)^{2/\kappa}$.

- **Off-by-one in Algorithm 3.** Line 6 computes $\sigma_{med}^{k+1}$ but line 7 clips $\sigma_{med}^k$ (without +1 subscript). At iteration k=0, the update would reference $\sigma_{med}^0$ which has not been computed. If intentional (e.g., the update is lagged), this must be explained; if a typo, it should be corrected since index consistency is critical for algorithm correctness and proof validity.

- **Contribution §1.1 references "Assumption 4" that does not appear in the main text.** The paper lists "Theory I: ...Assumption 4 (our novel theoretical zeroth-order oracle)" but the main text only defines Assumptions 1–3. Readers cannot evaluate the stated contribution without tracking down the appendix. This appears to be a numbering inconsistency from a revision; it should be corrected.

- **ZO experiments use only 3 runs, and the unexplained acceleration failure is not discussed.** Figure 3 shows that SGD-based variants (ZO-clipped-med-SGD) consistently outperform the accelerated SSTM variants (ZO-clipped-med-SSTM) across all κ values tested. This is theoretically surprising — acceleration should help for convex problems — and is practically important since the paper's primary theoretical algorithm is the SSTM variant. No explanation is provided.

## Nice-to-Haves

- **MAB experiments with d > 2 arms.** The sole MAB benchmark uses d = 2 arms, far too small to validate the claimed $\tilde{O}(\sqrt{dT})$ dimension scaling. Adding experiments with d ∈ {10, 50, 100} would substantiate Theorem 3's claims about dimension dependence.

- **Comparison on total oracle queries, not iterations.** Since each iteration of ZO-clipped-med-SSTM requires $(2m+1)\cdot b$ calls, and $m = 2/\kappa + 1$ grows with decreasing κ, comparing on total oracle calls (not just iterations) would give a more complete picture. Table 1 already notes the per-call overhead ($b/\kappa$ calls), but a sample-complexity plot would be informative.

- **Adaptive selection of m.** The theoretically optimal m = 2/κ + 1 requires knowledge of κ. The paper discusses using m = 3 as a fallback for κ ≥ 1, but an adaptive or data-driven scheme for unknown κ would improve practical usability and is noted as future work.

- **Theoretical or empirical characterization of symmetry tolerance.** The method relies on Assumption 3's symmetry, but Section 6.1 argues robustness to mild asymmetry. Quantifying how much skewness degrades performance (at least empirically, supplementing the appendix results §D.2.1) would strengthen the applicability claims.

- **ML-relevant ZO benchmarks.** The ZO experiments use only a synthetic least-squares problem. Including a standard zeroth-order benchmark (e.g., black-box hyperparameter tuning or an adversarial attack scenario) would broaden appeal and relevance for the ICLR community.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "The abstract's 'any κ > 0' is technically wrong."** The bound does hold for any κ > 0 — the $(4/\kappa)^{2/\kappa}$ factor is inside the constant, not in the asymptotic class. The theorems are honest. The issue is framing, not correctness. Moved to the "obscured constant" weakness above.

- **Harsh critic: Assumption 3 does not cover "majority of distributions."** The claim that the assumption is "strictly stronger than bounded κ-th moment with symmetric density" and the demand for a counterexample are overly picky for this type of paper. The characterization is indeed imprecise, but the paper explicitly provides worked examples (Cauchy) and notes reductions to standard assumptions for κ ∈ (1, 2].

- **Harsh critic: Cryptocurrency experiment should compare against HTINF/APE.** The cryptocurrency experiment is explicitly described as a real-world illustration in the full-feedback setting (disclosed by the authors), not a head-to-head MAB algorithm comparison. The disclosure and framing are clear; demanding HTINF/APE comparison here is scope creep for a demonstration task.

- **Harsh critic: Broader impact statement is two sentences.** Pure formatting/style criticism with no bearing on scientific content.

- **Positive reviewer: Lipschitz oracle is hard to verify in black-box settings.** This is a common limitation of any Lipschitz-type assumption in ZO theory and is not specific to this paper. Not a substantive weakness here.

- **Spark finder: Provide a lower bound matching the proposed rates.** Proving matching lower bounds for zeroth-order optimization with symmetric heavy-tailed noise is a significant open theoretical problem outside this paper's scope. Appropriate as a future direction, not a weakness.

## Novel Insights

The most insightful observation emerging from this review is the subtle relationship between the median's variance-reduction mechanism under Assumption 3 and the structural cost hidden in $(4/\kappa)^{2/\kappa}$: the paper achieves "optimal-rate matching" at the price of a super-exponentially growing constant as κ → 0, and the improvement over prior work is most dramatic not as κ → 0 (where the constant explodes) but around κ ≈ 1, where prior rates are actually *undefined* (due to the $\kappa/(\kappa-1)$ exponent) while the proposed method remains well-behaved. The regime κ ∈ (0.8, 1.2] is therefore the paper's true sweet spot, not the extreme heavy-tail regime. This observation, which is not clearly stated in the paper, would help set realistic expectations for practitioners.

## Suggestions

1. **Clarify the symmetry-under-importance-weighting question** (most critical): either prove in the main text that the centered importance-weighted estimator satisfies Assumption 3, or provide a separate lemma for Theorem 3's proof that does not rely directly on Lemma 1. This is required to establish the MAB result.

2. **Fix or explain the off-by-one in Algorithm 3 line 7** (σ_med^k vs. σ_med^{k+1}).

3. **Reconcile §5.1's conclusion with Figure 1**: either re-examine whether "convergence in probability" in the theoretical sense applies here, or add explanation distinguishing theoretical guarantee from empirical figure. If HTINF empirically dominates, this should be acknowledged honestly along with the point that it lacks theoretical guarantees for the experimental κ.

4. **Add a brief discussion on why SSTM (accelerated) underperforms SGD (non-accelerated)** in Figure 3, as this phenomenon is practically important and potentially expected under heavy-tail settings.

5. **Add a quantitative note on the $(4/\kappa)^{2/\kappa}$ constant** to the abstract or Section 6.2, specifying numerically for e.g. κ ∈ {0.25, 0.5, 1} what the effective constant is, so readers can calibrate when the proposed method is truly superior in total oracle calls.

---

**Axis evaluations:**
- **Novelty:** High — extending median estimation to ZO optimization and MAB under κ > 0 is a genuine first, and Assumption 3 is an original oracle formulation.
- **Technical soundness:** Moderate-to-good for the ZO optimization part; uncertain for the MAB part due to the unresolved importance-weighting symmetry question.
- **Empirical support:** Weak-to-moderate — the ZO Figure 3 is convincing; the MAB experiment (d=2 only, apparently contradictory conclusions) and crypto experiment (full-feedback, non-algorithmic baselines) are insufficient.
- **Significance:** Moderate-to-high for the heavy-tailed optimization community; the κ ≤ 1 regime is genuinely important.
- **Clarity:** Moderate — theorems are clearly stated, but the Assumption 4/3 numbering inconsistency, Algorithm 3 index issue, and §5.1 narrative weaken overall clarity.

MY FINAL SCORE: <pineapple>5.5</pineapple>