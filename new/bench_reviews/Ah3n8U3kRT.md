Now I have enough information to write the final review. Let me compile all findings.---

## Summary

This paper proposes median-clipping techniques for zeroth-order (ZO) convex optimization and multi-armed bandits (MAB) under symmetric heavy-tailed noise. The key contribution is a novel gradient estimator (Lemma 1) that achieves bounded second moment for any κ > 0 by taking component-wise medians of (2m+1) oracle calls, where m > 2/κ. This extends prior ZO work (which degenerates at κ = 1 and is undefined for κ ≤ 1) to the entire regime κ > 0, achieving iteration complexities matching optimal bounded-variance rates. The MAB extension yields Õ(√dT) regret for symmetric heavy-tailed rewards.

---

## Strengths

1. **Genuine gap filled for κ ≤ 1 (Table 1, Theorem 1)**: Prior ZO methods (ZO-clipped-SSTM [20]) have rates with exponent κ/(κ−1) that blow up as κ → 1 and are undefined for κ ≤ 1. The proposed methods achieve κ-independent iteration complexity Õ(d²ε⁻²) for any κ > 0, including noise with unbounded mean. This is the first finite high-probability bound in this regime.

2. **Lemma 1 is technically non-trivial**: Proving that the component-wise median of (2m+1) symmetric heavy-tailed samples is unbiased with bounded second moment σ²/b for any κ > 0 (m > 2/κ) requires a fundamentally new proof structure distinct from standard clipping-based proofs. The paper correctly notes that prior oracles cannot be directly adapted to exploit symmetry.

3. **ZO optimization experiments support the main claim (Figure 3)**: The experiments on L1 regression with α-stable noise across κ ∈ {0.75, 1.0, 1.25, 1.5} show median-based methods significantly outperforming non-median counterparts for κ ≤ 1, directly validating the paper's central practical contribution.

4. **Unified theoretical framework**: The paper consistently covers unconstrained (Algorithm 1, SSTM-based) and constrained (Algorithm 2, SMD-based) settings, as well as extensions to strongly convex and PL-condition objectives in the appendix.

---

## Weaknesses

### Fatal
None — the theoretical framework is internally consistent.

### Major

- **Figure 1 is misrepresented in the main text.** Section 5.1 states: *"HTINF and APE do not have convergence in probability, while our Clipped-INF-med-SMD does, which confirms the efficiency of the proposed method."* The paper's own embedded figure description directly contradicts this: HTINF achieves average expected regret of ~0.1 vs ~0.2 for the proposed method (lower is better), and HTINF reaches probability of best-arm selection ~0.9 vs ~0.6 for the proposed method (higher is better). HTINF outperforms the proposed algorithm on **both** primary metrics displayed. The abstract's claim to "dramatically outperform [SOTA]" is directly grounded in this experiment, but the figure does not support it. The interpretation offered in the paper — that HTINF does not exhibit "convergence in probability" — is the opposite of what the graphs show: HTINF's best-arm probability converges to 0.9 while the proposed method stabilizes at a lower 0.6. This is a material misrepresentation of the empirical results.

- **MAB experiment is poorly designed to demonstrate the paper's unique advantage.** The sole MAB experiment (Figure 1) uses Cauchy noise (κ = 1), which is at the boundary of HTINF's valid regime (κ ∈ (1, 2]). The paper's distinctive theoretical claim is that it handles κ ≤ 1 where HTINF and APE are undefined. No experiment shows the proposed method operating at κ < 1 against a baseline that fails there. As a result, the experiment that could have unambiguously demonstrated the method's advantage was never run. The abstract's "dramatically outperform" claim applies to κ ≤ 1 but is supported by zero empirical evidence in the main paper.

### Minor

- **Total oracle complexity omitted from Table 1 comparison.** The proposed method requires (2m+1)·b ≈ (4/κ + 3)·b oracle calls per iteration, while ZO-clipped-SSTM uses b calls per iteration. For κ = 1, this is a 7× overhead; for κ = 0.5, an 11× overhead. The table compares iterations only; total oracle complexity — the actual hardware cost — is not computed or compared for the overlapping regime κ ∈ (1, 2]. The table caption acknowledges "b/κ calls per iter" for the proposed method, but the practical implication is not analyzed. This overstates the advantage in the κ > 1 regime.

- **Optimality claim for Theorem 3 needs qualification.** The paper states the Õ(√dT) regret is "optimal compared to the lower bound Ω(√dT) for stochastic MAB with the bounded variance of losses." This comparison is against the bounded-variance lower bound, not a lower bound specific to the symmetric heavy-tailed regime. For κ ∈ (0, 2) under the standard moment assumption, the lower bound is Ω(T^{1/κ}). Whether this tighter bound applies or fails under Assumption 3 (symmetry) is left unaddressed. The paper's claim that symmetry enables bypassing the T^{1/κ} barrier is implicitly remarkable but no matching lower bound or impossibility argument under Assumption 3 is provided.

- **Section 5.2 cryptocurrency experiment uses weak baselines.** The experiment compares against static strategies (hold ETH, Efficient Frontier), which are not algorithmic competitors in the online learning sense. No comparison against standard online mirror descent or any adaptive portfolio algorithm is given. Combined with the single-year (2023) evaluation window with specific cryptocurrency behavior, the generalizability of this experiment is limited.

- **Figure 3 anomaly unexplained.** Across all four noise settings (κ ∈ {0.75, 1.0, 1.25, 1.5}), the SGD variants outperform the SSTM variants (whether or not median is used). This is unexpected given that SSTM has acceleration guarantees. The paper does not comment on why the theoretically accelerated SSTM underperforms the non-accelerated SGD in practice.

### Trivial

- The self-reference issue for setting parameter A = ln(4K/β) ≥ 1 in Theorem 1 (A requires K, which is the output) is a known circularity in such analyses. The paper leaves it unaddressed, though this is standard practice in the field.

---

## Nice-to-Haves

- An experiment with κ ∈ {0.5, 0.75} in the MAB setting demonstrating that HTINF/APE fail while the proposed method succeeds would be the most direct validation of the paper's unique contribution and would resolve the major weakness.
- Total oracle complexity curves (not just iteration counts) as a function of κ ∈ (1, 2] for both the proposed method and ZO-clipped-SSTM would clarify the trade-off in the overlapping regime.
- A formal characterization of which distributions satisfy Assumption 3 (Eq. 4) would strengthen the theoretical framing beyond the two Cauchy examples in Remark 5.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Not yet released" or existence concerns**: None raised.
- **Misread Section 5.1 claim about "convergence in probability" being a proof issue**: The harsh critic is correct that the paper misrepresents Figure 1, but the exact phrase "convergence in probability" in a probabilistic sense (converging to 1 as T→∞) rather than finite-time convergence might be what the authors intended. Even under that interpretation, HTINF clearly has higher probability (0.9) of best-arm selection than the proposed method (0.6), making the claim still problematic. Kept as major weakness.
- **Strength Finder claim that "Figure 1 shows Clipped-INF-med-SMD achieving convergent regret and best-arm probability where HTINF and APE do not converge in probability"**: This directly conflicts with the verified figure description and is removed from strengths.
- **Harsh critic's claim about Section 3.1.1 lack of formal proof for Assumption 3 coverage**: The paper acknowledges this is an example-based justification and refers to Appendix A. The coverage claim is standard practice in such theoretical papers. Weakened to a nice-to-have.
- **Circularity in parameter A**: This is a standard known practice in SSTM-type analyses. Kept only as trivial.
- **Adaptive κ criticism as "underplayed"**: The paper explicitly acknowledges this as a limitation in Section 6.1, which is an appropriate discussion. The computational cost for κ → 0 is real but the paper's core practical claims are for κ ≥ 0.5. Moved to nice-to-have.
- **Request for d-scaling MAB experiment**: Nice-to-have, not a core flaw.

---

## Novel Insights

The core theoretical insight — that symmetry of the noise distribution, not just bounded κ-th moments, is sufficient to enable a median-based estimator with bounded second moment for any κ > 0 — is genuinely novel. This reframing (exploiting distributional shape rather than moment bounds) could be of independent interest to the broader heavy-tailed stochastic optimization community. The construction in Lemma 1, which converts a problem with arbitrarily heavy tails into one with effectively bounded variance via the median, represents a structural trick that may extend to other first-order and online learning algorithms under symmetry.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| uniINF (Heavy-tailed MABs, parameter-free BoBW) | `/human_reviews/2pNLknCTvG.md` | 7.50 | Stronger: parameter-free, extends to adversarial, no experimental misrepresentation |
| ZO optimization (min-variance two-point estimators) | `/human_reviews/ywFOSIT9ik.md` | 6.80 | Similar domain; somewhat weaker contribution but cleaner experimental narrative |
| General ZO stability framework | `/human_reviews/AfhNyr73Ma.md` | 7.00 | Similar domain; clean framework with no experimental misrepresentation |
| Convergence analysis Adam vs SGDM | `/human_reviews/mEBSeSk49H.md` | 4.25 | Theoretical contributions with overclaimed scope; some comparison gaps |
| Subspace optimization GaLore | `/human_reviews/udtrtwkvk5.md` | 5.25 | Theoretical contribution with counterexample + new algorithm, missing baselines |
| Genitourinary cancer ML (low anchor) | `/human_reviews/zeobgjmUCc.md` | 1.00 | Clearly weak; entirely different domain |

**Reasoning**: The paper's theoretical contributions (Lemma 1, Theorems 1–3, extension to κ ≤ 1) are comparable in quality to papers scoring 6–7. However, the major weakness — a clear misrepresentation of Figure 1 in the main text, where the proposed method is claimed to outperform HTINF when the figure shows the opposite, and the absence of any MAB experiment for κ < 1 that would actually demonstrate the paper's unique advantage — pulls the score below the 7-range anchors. The ZO optimization experiments are supportive but the MAB experimental section is unreliable as presented. This places the paper below uniINF (7.5) and the general ZO papers (6.8–7.0), but above clearly weak papers. The theoretical merit prevents a score below 4.5, but the experimental misrepresentation of central results warrants a lean-reject outcome.

**Score: 5.0**

The paper makes a real and non-trivial theoretical contribution for κ ≤ 1 but the empirical claims in the MAB section actively misrepresent what Figure 1 shows. A revision correcting the interpretation of Figure 1, adding a κ < 1 MAB experiment, and computing total oracle complexity comparisons would substantially strengthen the submission.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>