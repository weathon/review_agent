Now let me search for calibration papers to score this properly.Now I have enough information to write the complete final review.

---

## Summary

The paper proposes median clipping techniques for zeroth-order (ZO) non-smooth convex optimization and multi-armed bandit (MAB) problems under symmetric heavy-tailed noise. The core theoretical contribution is removing the κ > 1 restriction of prior work (ZO-clipped-SSTM): by exploiting distributional symmetry through a novel oracle assumption (Assumption 3) and a component-wise median estimator (Lemma 1), the authors achieve Õ(d²ε⁻²) iteration complexity for any κ > 0 in ZO optimization and Õ(√(dT)) expected regret in MAB—both matching optimal rates under bounded variance. The algorithms are applied to synthetic ZO optimization and real-world cryptocurrency portfolio selection experiments.

---

## Strengths

- **Genuine theoretical advance: removal of the κ > 1 barrier.** Table 1 makes this concrete: ZO-clipped-SSTM [20] has iteration complexity Õ((√dM₂/ε)^{κ/(κ−1)}) that diverges as κ → 1 and is undefined for κ ≤ 1, while ZO-clipped-med-SSTM achieves Õ(max{d^{3/2}M₂R/ε, d(M₂² + dΔ²/κ^{2/κ})R²/(bε²)}) for any κ > 0. Filling this gap is a real contribution.

- **Well-constructed technical core (Assumption 3 + Lemma 1).** The novel oracle assumption (Eq. 4) conditions on the noise density p(u|x,y) rather than a moment bound on ξ, enabling exploitation of symmetry. Lemma 1 then shows the component-wise median of 2m+1 samples is unbiased (Eqs. 10–11) with bounded second moment for all κ > 0 with m > 2/κ. The pipeline (randomized smoothing → two-point oracle → symmetry via Eq. 7 → coordinate-wise median → clipping) is modular and well-motivated.

- **Optimal MAB regret theorem (Theorem 3).** The bound Õ(√(dT)) matches the Ω(√(dT)) lower bound for stochastic MAB with bounded variance. Prior heavy-tailed MAB results achieve only Õ(d^{(κ−1)/κ}T^{1/κ}), which is strictly worse for κ < 2.

- **ZO optimization experiments (Figure 3) clearly validate the median advantage for κ ≤ 1.** For α = κ ∈ {0.75, 1.0}, median-based methods converge while non-median counterparts stagnate or diverge. For κ > 1, median methods match baseline performance. This directly supports the paper's core claim in the ZO setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Misleading MAB empirical claim contradicts Figure 1 results.** The abstract claims methods "do not lose to SOTA approaches and dramatically outperform them for κ ≤ 1." Yet Figure 1 (the sole MAB experiment, using d=2 arms and Cauchy noise, which corresponds to κ = 1) shows HTINF achieving average regret ≈ 0.1 and probability of best-arm selection ≈ 0.9, while Clipped-INF-med-SMD achieves ≈ 0.2 regret and ≈ 0.6 probability—strictly worse on both metrics. The paper's framing in Section 5.1 ("HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does") is false on its face: HTINF converges to 0.9 while the proposed method stagnates at 0.6. Both methods fail to reach 1.0, but HTINF is unambiguously better. The abstract's empirical headline claim is not supported and is misrepresented by this framing.

- **Theoretical justification of median in Algorithm 3 (MAB) is missing.** Algorithm 3 applies the coordinate-wise median to 2m+1 importance-weighted vectors {ĝ_t}, where each ĝ_t has exactly one non-zero coordinate (the chosen arm A_t). For coordinate i, the number of non-zero observations is Binomial(2m+1, x_{k,i}). When x_{k,i} < 0.5 (which is typical as the algorithm concentrates on the best arm), the majority of observations are exactly 0 and the coordinate-wise median is 0 regardless of the noise. This means: (a) the distribution of the i-th component of each ĝ_t is a mixture of 0 (with probability 1−x_{k,i}) and a continuous distribution (with probability x_{k,i}), which is NOT symmetric around any non-zero value; (b) the unbiasedness proof in Lemma 1 rests on symmetry of all 2m+1 samples around ∇f̂_τ(x), which requires each sample to be symmetric—a condition that does not obviously hold here. The paper states "we assume noise ξ_t satisfies Assumption 3" without verifying that this assumption is inherited by the importance-weighted estimator ĝ_t. The main text provides no argument that Lemma 1 applies in the MAB setting. The proof of Theorem 3 resides entirely in the appendix and cannot be checked here; but the theoretical gap in the main text is real and concerning.

### Minor

- **Figure 3 shows the primary proposed ZO algorithm (ZO-clipped-med-SSTM) consistently underperforms the simpler ZO-clipped-SGD baseline.** The figure caption states that both SGD variants converge faster than both SSTM variants across all κ values. The paper does not explain when the theoretical acceleration of SSTM over SGD materializes in practice, or why the asymptotic advantage does not appear at the scales tested. The main algorithm being beaten by a simpler baseline without discussion is a gap.

- **Regret bound constant in Theorem 3 diverges as κ → 0.** The bound in Eq. (14) contains c² = (32 ln d − 8)·(8M₂² + 2Δ²(2m+1)(4/κ)^{2/κ}), where (2m+1)(4/κ)^{2/κ} = (4/κ + 3)(4/κ)^{2/κ} diverges as κ → 0. The headline "Õ(√(dT)) for any κ > 0" is technically correct but the pre-constant grows without bound, making the bound practically vacuous for small κ. This is inadequately discussed; the limitation section only addresses κ → 0 for the adaptive setting.

- **Experiment in Section 5.1 uses only d = 2 arms.** The theoretical claim in Theorem 3 is Õ(√(dT)) for d-arm bandits, and the linear scaling with d is one of the headline advantages over prior work (which achieves Õ(d^{(κ−1)/κ}T^{1/κ})). Validating this claim requires experiments at larger d (e.g., d = 10, 50). A single two-arm experiment does not test the key d-dependent claim.

### Trivial

- The notation "b/κ calls" in Table 1 header is an asymptotic simplification of the actual (4/κ+3)·b calls per iteration stated in Theorem 1. This is asymptotically correct but imprecise; a clarifying remark would avoid confusion.

---

## Nice-to-Haves

- **Multi-arm MAB experiments (d > 2):** Experiments at d = 10, 50 with varying κ would validate the headline d-dependence claim and are needed to close the gap between theory and practice.
- **Total oracle complexity comparison:** A plot of function gap / regret vs. total oracle calls (not just iterations) would allow a fair comparison accounting for the (4/κ + 3) oracle overhead per iteration.
- **Adaptive κ estimation:** The method requires knowing κ to set m = 2/κ + 1. The paper acknowledges this but leaves it for future work. Even a heuristic discussion of robustness to misspecified κ would strengthen the practical case.
- **Portfolio experiment baselines:** Section 5.2 compares only to static strategies. A comparison against EXP3 or online gradient descent in the full-feedback setting would be more informative.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Issue 3 (oracle-count inconsistency in Table 1):** "b/κ calls" in Table 1 is O-notation for the dominant κ-dependent term in (4/κ+3)·b. This is an asymptotic simplification, not a false claim. The actual per-iteration cost is stated correctly in Theorem 1. Not a substantive error.

- **Harsh Critic's claim that complexity is not "independent of κ":** The paper claims rates "match optimal bounds for bounded variance for any κ > 0"—this is about the ε-dependence (Õ(d²ε⁻²)), not the κ-dependent constants. The critics's point that the constant grows with κ is captured in the Minor tier above; the core claim about ε-rates is not falsified.

- **Strength Finder strength #6 ("experimental confirmation of dramatic improvement for κ ≤ 1"):** Partially valid for ZO (Figure 3, within median vs. non-median comparison), but conflicts with the Major weakness that Figure 1 shows MAB underperformance. Moved to a weaker formulation in Strengths above.

- **Strength Finder generic strength about problem importance:** Not specific to paper contributions; removed.

---

## Novel Insights

The key technical insight—that the two-point oracle (Eq. 7) preserves the symmetry of the noise (because φ(ξ|x+τe, x−τe) is symmetric in u when p(u|x,y) = p(−u|x,y)), enabling the component-wise median over 2m+1 independent noise realizations to have bounded second moment for any κ > 0—is a clean and genuinely novel connection between distributional symmetry and ZO gradient estimation. This observation, formalized in Assumption 3 and Lemma 1, is the intellectual core of the paper and represents a real advance over the standard bounded-moment approach of prior ZO heavy-tailed work. The limitation is that the same argument's applicability to the MAB importance-weighted setting is not established.

---

## Suggestions

1. **Fix the Figure 1 framing:** Either run experiments at κ < 1 (Lévy-alpha-stable with α < 1, or Cauchy which has κ < 1) where HTINF's guarantee fails and compare, or honestly report that HTINF numerically outperforms the proposed method at κ ≈ 1 while explaining the theoretical difference (convergence guarantees vs. empirical averages). The current framing is misleading.
2. **Clarify the MAB theory:** Either prove a lemma analogous to Lemma 1 for the importance-weighted case, or explicitly state the additional assumptions under which Theorem 3 holds.
3. **Explain Figure 3 SSTM vs. SGD gap:** Discuss at what problem scales the theoretical acceleration of SSTM over SGD is expected to manifest.
4. **Sharpen κ → 0 discussion:** Give explicit numerical values of the constant c² for representative κ (e.g., κ = 0.5, 0.25) to help the reader assess when Theorem 3 is practically informative.

---

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/2pNLknCTvG.md` (uniINF, avg 7.5, Spotlight): Heavy-tailed MAB, parameter-free best-of-both-worlds — substantially stronger than this paper; empirical and theoretical claims both well-supported.
- `/home/wg25r/review_agent/human_reviews/AfhNyr73Ma.md` (ZO stability, avg 7.0, Poster): Technically sound ZO theory, well-presented, solid experimental confirmation.
- `/home/wg25r/review_agent/human_reviews/7t8aKBeATc.md` (ZO normalized, avg 3.5, Reject): ZO paper with limited novelty, no fundamental advance; this paper is clearly more original.
- `/home/wg25r/review_agent/human_reviews/DIAaRdL2Ra.md` (Adafactor convergence, avg 5.0, Reject): Theory paper with correct results but insufficient novelty and experimental support.
- `/home/wg25r/review_agent/human_reviews/4jzjexvjI7.md` (MAB continuous-time, avg 2.33, Reject): Very weak MAB paper; this paper is well above this level.

**Positioning:** The ZO optimization theory (Sections 3–3.3) is at the AfhNyr73Ma level (≈7): novel, clean, fills a real gap. However, the MAB application (Section 4–5.1) is substantially weaker: the theoretical justification is incomplete and the empirical claim in Figure 1 actively contradicts the abstract's headline claim. The combined paper sits between the 5.0 anchors (theory correct but experiments weak) and the 7.0 anchor (solid all-around). Given the Major weakness on the empirical misrepresentation and the theoretical gap in the MAB section, I place this paper at **4.5**: the ZO theory merits publication but the paper as submitted contains a misleading empirical claim about its main MAB algorithm and an unjustified theoretical extension to the MAB setting that together are too significant to overlook.

**Axes:**
- *Originality:* Moderate–good. The extension from κ > 1 to κ > 0 is genuine; the connection to median estimation is novel in the ZO context.
- *Importance of research question:* Good. Heavy-tailed ZO optimization is practically motivated and the κ ≤ 1 gap is real.
- *Claims well-supported:* Weak. The abstract's empirical headline claim is contradicted by Figure 1; the MAB theoretical argument has a gap.
- *Soundness of experiments:* Weak. d = 2 arms in MAB, accelerated algorithm underperforms simpler baseline, misleading framing.
- *Clarity of writing:* Moderate. Theory is clearly presented; experiment interpretation is misleading.
- *Value to research community:* The ZO optimization part has clear value; the MAB part as currently written has questionable value.

**Decision: Reject.** The ZO optimization contribution is publishable in isolation, but the MAB section, which is presented as a co-equal contribution, has both a theoretical gap and an empirical misrepresentation that must be resolved before the combined paper is acceptable.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>