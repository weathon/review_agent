Now I have enough context to synthesize a final review. Let me organize my thoughts:

Key issues verified against the paper:

1. **MAB experiment overclaiming**: The abstract says "Our methods do not lose to SOTA approaches and dramatically outperform them for κ ≤ 1." But in the MAB experiment (Fig 1), HTINF achieves lower regret (~0.1 vs ~0.2) and higher best-arm probability (~0.9 vs ~0.6). The paper argues HTINF lacks "convergence in probability" but this is not well-supported by the figure – the percentile bands would need to show HTINF has much wider variance. Even if the high-probability claim is technically correct, the word "superior" and "dramatically outperform" in the context of overall SOTA comparison is misleading.

2. **Framing of rate improvements**: The comparison in Table 1 and abstract compares rates under symmetric noise (new method) vs rates under general heavy-tailed noise (baselines). The improvement partly comes from the stronger assumption, not solely from the algorithm. The paper acknowledges this in Section 6.1 but the abstract is misleading.

3. **Only 3 runs for ZO experiments**: The paper mentions 3 runs but says more are in the appendix. MINOR concern since heavier runs exist.

4. **Table 1 oracle calls inconsistency**: The table says "b/κ calls" but theorem says (2m+1)b = (4/κ+3)b per iteration. This is a real inconsistency. MINOR.

5. **Variance bound scaling**: (4/κ)^{2/κ} grows super-exponentially as κ → 0. Worth noting. MINOR.

6. **d-dependence**: For Lipschitz oracle, the paper gets d² factor vs prior (√d)^{κ/(κ-1)}. For κ > 1, this can be worse. Paper discusses this at line 246. PARTIALLY ADDRESSED.

7. **MAB experiment only tests with κ=1**: Should test with κ < 1 where prior methods have no guarantees. VALID suggestion but not fatal.

Let me also verify: does the paper's Figure 1 actually show the "convergence in probability" claim? The description says "0.95 and 0.05 percentiles for regret." If HTINF's percentile bands are wide (which they might be for heavy-tailed distributions), then the paper's claim about "convergence in probability" could be valid even though mean regret is lower.

Looking at the figure caption: "Shaded regions represent 0.95 and 0.05 percentiles for regret." For HTINF (green), the shaded region appears to show wider percentiles vs. the proposed method (blue). This could support the convergence-in-probability claim, but the paper doesn't clearly state or quantify this.

This is an important nuance: the paper might be RIGHT about convergence in probability even though HTINF wins on mean regret. However, the text "confirms the efficiency of the proposed method" and the abstract's "do not lose to SOTA" are overclaims.

Now let me finalize.</think>Now I have verified the key claims against the paper. Let me compose the final review.

## Summary

The paper proposes median clipping for zeroth-order (ZO) non-smooth convex optimization and multi-armed bandits (MAB) under symmetric heavy-tailed noise. The core technical contribution shows that component-wise median estimation of 2m+1 gradient-difference samples yields an unbiased, bounded-variance gradient estimator for any κ > 0 (including distributions with unbounded or infinite variance), enabling convergence rates that match the bounded-variance optimal rates (Õ(d²ε⁻²) for ZO optimization and Õ(√(dT)) for MAB). This extends prior work that degenerates as κ → 1⁺ and cannot handle κ ≤ 1.

## Strengths

- **Sound theoretical contribution extending κ range to (0, ∞):** Lemma 1 establishes that the component-wise median of 2⌊2/κ⌋+1 samples yields an unbiased gradient estimate with bounded second moment for any κ > 0 under symmetric noise. This is the paper's key result and resolves a genuine gap — prior ZO methods [19, 20] require κ ∈ (1, 2] and their rates degenerate as κ → 1. The non-degenerating rates (Table 1) are a meaningful improvement.

- **Optimal regret for symmetric heavy-tailed MAB:** Theorem 3 achieves Õ(√(dT)) regret for any κ > 0, matching the Ω(√(dT)) lower bound for bounded-variance MAB, which is a strong result since prior heavy-tailed MAB algorithms all require κ > 1.

- **ZO experiments demonstrate clear improvement for κ ≤ 1:** Figure 3 shows that median-based methods (green curves) converge substantially faster than non-median baselines under Lévy-stable noise with α = κ ∈ {0.75, 1.0}, where the baselines stagnate. The improvement is visually unambiguous for the regime the paper targets.

- **Assumption 3 is well-motivated and correctly generalizes prior oracle models:** Section 3.1 explicitly shows (via Remark 3) how it reduces to standard assumptions when κ ∈ (1, 2], and justifies why symmetry cannot be trivially incorporated into prior frameworks.

## Weaknesses

### Fatal
None.

### Major

- **MAB experimental results contradict the breadth of claimed superiority.** The abstract states "Our methods do not lose to SOTA approaches and dramatically outperform them for κ ≤ 1." Section 5.1 claims the results "confirm the efficiency of the proposed method." However, Figure 1 (the only MAB experiment) shows HTINF achieving lower average regret (~0.1 vs. ~0.2) and higher probability of selecting the best arm (~0.9 vs. ~0.6). The paper's stated defense — that HTINF and APE "do not have convergence in probability" — is a valid theoretical distinction (Theorem 3 provides high-probability bounds), but the figure does not clearly substantiate this claim: the 0.95/0.05 percentile bands would need to show HTINF has dramatically wider concentration, and this is not quantified or discussed. Even if the convergence-in-probability claim is correct, the sweeping language of "do not lose to SOTA" and "superior performance" (Section 5 opening) is misleading given the MAB results. The ZO experiments and MAB *theory* are strong, but the MAB *experiments* overclaim.

- **Rate comparisons conflate the contribution of the symmetry assumption with that of the median technique.** Table 1 and the abstract compare the proposed methods (operating under symmetric noise, κ > 0) against ZO-clipped-SSTM (operating under general noise, κ > 1). The dramatically better rates partly arise from exploiting the stronger symmetry assumption, not solely from the algorithm. While Section 6.1 acknowledges this limitation and Table 1 labels the proposed column "symmetric noise," the abstract framing — "match the best-known ones for the case of the bounded variance" and "dramatically outperform them for κ ≤ 1" — obscures this distinction. A reader scanning the abstract would not realize the comparison is across different assumption classes.

### Minor

- **Table 1 lists "b/κ oracle calls per iteration" for the proposed method, but Theorem 1 states each iteration requires (2m+1)·b = (4/κ + 3)·b oracle calls (with m = 2/κ + 1).** These are inconsistent (e.g., at κ = 2, the table implies b/2 while the theorem gives 5b). This should be corrected.

- **The variance bound σ² in Lemma 1 includes a (4/κ)^{2/κ} factor that grows super-exponentially as κ → 0.** For κ = 0.75, this is ~3.16; for κ = 0.1, it exceeds 10^{17}. The paper should explicitly discuss how this constrains the practical regime where the method offers meaningful improvement, rather than leaving it implicit.

- **The ZO experiments use only 3 runs with no error bars (Figure 3).** For heavy-tailed noise where outliers can dominate, this is a concern, though the paper mentions additional experiments in Appendix D.2 with more runs.

### Trivial
None.

## Nice-to-Haves

- Test MAB performance under κ < 1 noise, where prior algorithms have *no theoretical guarantees*, to more compellingly demonstrate the practical advantage of the proposed method in the regime where it is uniquely applicable.

- Report high-probability regret metrics (e.g., 95th percentile) for the MAB experiment rather than just mean regret, to match the theoretical contribution's emphasis on convergence in probability vs. in expectation.

- Discuss practical guidance for choosing m and κ, including the trade-off implied by the (4/κ)^{2/κ} factor.

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Harsh critic's claim that the MAB experiment "directly contradicts" claimed superiority (Fatal):** The harsh critic frames this as a structural contradiction. However, the paper's theoretical contribution is specifically about *high-probability* bounds (Theorem 3 provides controlled large deviations), and there is a legitimate distinction between convergence in probability vs. in expectation. The issue is *overclaiming*, not a fundamental contradiction of the core contribution. Downgraded to Major.

- **Harsh critic's claim that the rate comparison is "misleading" because different assumptions yield different rates (treated as Fatal by harsh critic):** The paper does label "symmetric noise" in Table 1 and discusses the limitation in Section 6.1. The improvement is real under the new assumption class. The issue is framing in the abstract, not deception. Downgraded to Major.

- **Harsh critic's claim that Assumption 3 is not novel ("standard in robust statistics literature"):** Assumption 3 combines symmetry with a specific sub-Cauchy density bound on the noise in the two-point oracle setting, which is a non-trivial formulation that enables the median technique to work in the ZO framework. The novelty claim is for the *oracle formulation*, not the individual statistical concepts. Removed.

- **Harsh critic's claim about worse d-dependence for κ > 1 being misleadingly omitted:** The paper explicitly discusses this at line 246: "In terms of d, we obtain the factor dM₂² + d²Δ²/κ^{2/κ} instead of (√d·M₂ + √d·Δ)^{κ/(κ-1)} from [20]." The paper is transparent about this trade-off. Removed as a weakness.

- **Harsh critic's claim that the cryptocurrency experiment "is not a bandit setting":** The paper explicitly acknowledges this ("Portfolio selection has the full feedback for all assets, while, in standard bandits, we observe only one asset per step") and adapts the algorithm accordingly. This is a legitimate real-world application test, not a misrepresentation. Removed.

- **Harsh critic's claim about asymmetric noise appendix results being unverifiable:** The appendix is stripped by the parser. The paper states the claim in Section 6.1 and references D.2.1; this is a parser artifact, not a paper flaw. Removed.

- **"Only 3 runs" as a Fatal issue:** Downgraded to Minor since more runs exist in the appendix and the ZO convergence curves are visually clear.

## Novel Insights

The core insight — that exploiting symmetry via median estimation provably yields bounded-variance ZO gradient estimators for *any* κ > 0, including distributions with infinite or undefined expectation — is genuinely novel and resolves a degeneracy in the existing ZO optimization literature. However, the practical utility of this result is circumscribed: (1) the symmetry assumption limits applicability; (2) the super-exponential growth of (4/κ)^{2/κ} means the method's practical advantage is concentrated in the moderate κ regime (say κ ≥ 0.5); (3) the MAB experiments, while consistent with the *theory*, do not clearly demonstrate practical advantage over existing methods even under symmetric noise, creating a gap between the theoretical and empirical contributions.

## Suggestions

- Rewrite the abstract and Section 5 introduction to carefully scope claims: replace "do not lose to SOTA" with a statement specifying under symmetric noise conditions, and acknowledge where experiments show baselines outperforming on mean metrics.

- For the MAB experiment, either add a clearly labeled metric that shows the convergence-in-probability advantage (e.g., 95th-percentile regret) or tone down the experimental claims accordingly.

## Score and Decision

**Calibration anchors used:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| 2pNLknCTvG (uniINF) | 7.5 | Heavy-tailed MAB with strong theory and BoBW property; cleaner experiments. The paper under review is weaker because its MAB experiments overclaim. |
| fjf3YenThE (ZO variance reduction) | 5.33 | Solid ZO theory with moderate experimental validation. Similar niche area; the paper under review has a stronger theoretical extension (κ > 0) but weaker experimental validation. |
| tsNLIBlG4p (soft-clipping) | 4.0 | Clipping analysis with convergence proofs but limited practical advantage over existing methods and overclaimed novelty. |
| c2OtbtZXFC (Stiefel manifold) | 4.67 | Good theory but experiments contradicted claims; withdrawn. |
| FE7PY7e4tr (manifold topology) | 5.25 | Strong theory with misleading experimental framing. |
| TTrzgEZt9s (DRO) | 8.0 | Strong theory with clear practical gains; much cleaner than the paper under review. |

The paper makes a genuine theoretical contribution (extending ZO heavy-tailed optimization and MAB to κ > 0 via median estimation under symmetry), but the experimental section overclaims and the abstract obscures the role of the stronger assumption. This pattern — strong theory with misleading experimental framing — matches papers in the 4.5–5.5 range. The theoretical contribution is stronger than typical papers in this range, but the overclaiming and weak MAB experiments prevent a higher score. Relative to the uniINF paper (7.5), this paper has comparable theoretical novelty but noticeably weaker experiments and more overclaiming. Relative to the soft-clipping paper (4.0), this paper has a clearer and more impactful theoretical contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>