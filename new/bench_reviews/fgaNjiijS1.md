Now let me search for calibration papers to score this work properly.Now I have enough to write the full review. Let me synthesize everything.

---

## Summary

This paper proposes the *Norm-Adaptive MMD* (NAMMD), a normalization of MMD by the denominator `4K − ‖μ_P‖² − ‖μ_Q‖²`, yielding a bounded [0,1] distance that adjusts for the RKHS norms of distributions. The core insight is that for fixed MMD, pairs with higher norms (more concentrated distributions) are actually farther apart in a statistically meaningful sense. The paper proves asymptotic normality (Theorem 2), Type-I error control (Theorem 5), sample complexity bounds (Theorem 8), and superiority over MMD in test power for both two-sample testing (Theorem 10) and distribution closeness testing (Theorem 12). Experiments validate consistent improvements over MMD across 20 dataset-kernel combinations and practical DCT case studies on ImageNet variants.

---

## Strengths

- **Clean, geometrically motivated normalization (Definition 1, Remark in §3):** NAMMD's denominator `4K − ‖μ_P‖² − ‖μ_Q‖²` reflects that `Var(P,κ) = 1 − ‖μ_P‖²`, so larger norms imply tighter concentration, making the same MMD gap more discriminating. Figure 1(c-d) directly demonstrates this through the p-value correlation argument.

- **Comprehensive theoretical treatment:** The paper establishes a full testing framework — asymptotic distribution (Theorem 2), consistent variance estimation (Lemma 4), Type-I error control (Theorem 5), large-deviation bounds (Lemma 6), sample complexity upper bounds (Theorem 8), and NAMMD=0 iff P=Q (Lemma 9). Together these provide principled guarantees absent from the prior DCT literature on complex data.

- **Theorem 10 proves NAMMD strictly dominates MMD for two-sample testing:** Under the same kernel, whenever MMD rejects the null, NAMMD does so with high probability, and there exist cases (with probability ≥ 1/65) where NAMMD rejects and MMD does not. This is a provable, non-trivial advantage.

- **Consistent empirical improvement across all 20 settings (Table 1):** NAMMD outperforms MMD in every one of the 5 datasets × 4 kernels combinations. While individual magnitudes are modest, the consistent direction is statistically significant by a sign test (p ≈ 10⁻⁶ under H₀ of equal performance), confirming the direction of the theoretical prediction.

- **Practical case studies with real-world utility (Figures 3–5):** The ImageNet-variant experiments show NAMMD more sharply separating closeness levels aligned with ground-truth accuracy margins, offering label-free model evaluation — a practical contribution beyond the testing literature.

- **Extension of DCT to complex, high-dimensional data:** Prior DCT methods (Canonne et al.) are limited to discrete 1D distributions. This paper is the first kernel-based DCT framework that directly handles continuous, high-dimensional data.

---

## Weaknesses

### Fatal
None.

### Major

- **Theorem 12's key condition is unverifiable in practice (§4.3):** The DCT superiority theorem requires `‖μ_{P₁}‖ + ‖μ_{Q₁}‖ < ‖μ_{P₂}‖ + ‖μ_{Q₂}‖`. In Definition 11, P₂ and Q₂ are explicitly *unknown* distributions. The paper's justification — "norms of mean embeddings are typically positively correlated with MMD value" — is stated without proof and is not empirically verified in any experiment. In general DCT settings there is no principled reason why the unknown pair must have larger norms than the reference pair. Without empirical verification that this condition holds in the DCT experiments (§5.2), Theorem 12 does not establish the claimed DCT advantage over MMD, which is the paper's central DCT theoretical contribution.

### Minor

- **Individual improvements in Table 1 are small and lack significance testing:** Most gains in Table 1 are well within one standard deviation (e.g., blob/Gaussian: 0.600→0.616, Δ=0.016 vs. σ=0.090). No pairwise significance tests (e.g., McNemar over repeated trials) are reported. While the *consistency* across all 20 settings is strong evidence, reporting a significance test would substantially strengthen the empirical case, especially for reviewers skeptical of small effect sizes.

- **DCT comparison with Canonne's test conflates two advantages (Table 2):** NAMMD substantially outperforms Canonne's total variation test. However, this comparison captures both the kernel-vs.-TV advantage (already established in the two-sample testing literature) and any DCT-specific advantage of NAMMD over MMD. Without a comparison of NAMMD-DCT versus MMD-DCT in Table 2, it is impossible to isolate how much of the gain is due to the NAMMD normalization specifically versus the use of kernels at all.

- **Potential formula issue in Theorem 2:** The asymptotic variance formula reads `σ²_{P,Q} = √(4E[H_{1,2}H_{1,3}] − 4(E[H_{1,2}])²) / (4K − ‖μ_P‖² − ‖μ_Q‖²)`. Standard U-statistic CLT theory would put the raw variance (not its square root) in the numerator. It is unclear whether this is a typographic artifact of PDF parsing or a mathematical inconsistency; the authors should clarify.

- **Section 5.1 comparison with training-based methods is confounded:** NAMMDFuse uses 2× more test samples than methods requiring training (MMD-D, MEMabid, AutoML). The paper is explicit about this trade-off, but conclusions comparing against those methods are limited, since the sample-size advantage conflates with the algorithmic advantage of NAMMD normalization.

### Trivial

None beyond possible formula parsing issues.

---

## Nice-to-Haves

- A power curve (test power vs. sample size, as in Figure 2) for NAMMD vs. MMD under the same kernel in Table 1's settings would clarify whether NAMMD's advantage is consistent across sample sizes or concentrated at particular regimes.
- Empirical verification that the condition `‖μ_{P₁}‖ + ‖μ_{Q₁}‖ < ‖μ_{P₂}‖ + ‖μ_{Q₂}‖` holds in Section 5.2's DCT experiments would directly validate Theorem 12's applicability.
- Comparing NAMMD-DCT against MMD-DCT in Table 2 (rather than only against Canonne's TV test) would isolate the normalization's DCT-specific contribution.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Circular motivation" (Harsh Critic):** While the use of p-values to illustrate NAMMD's advantage has a superficial circularity, the core geometric argument — that distributions with higher RKHS norms are more concentrated (`Var(P,κ) = 1 − ‖μ_P‖²`), so the same MMD gap is more discriminating — is *not* circular. It is a valid geometric property of RKHS mean embeddings. The harsh reviewer overstated this as a structural problem; the motivation, while imperfect, is substantively valid.

- **"Table 2 comparison with Canonne is fundamentally unfair":** The comparison is appropriate as a demonstration that kernel DCT outperforms TV-based DCT on complex, structured data — which is the paper's stated contribution. The setup treats both methods as DCT competitors on the same distributions, which is fair. Reduced to a Minor weakness above about isolating the kernel-vs.-NAMMD contribution.

- **"Section 5.1 NAMMDFuse superior because more test samples":** The paper explicitly discloses the 2× sample-size convention for training-free methods and the comparison against MMDFuse (same sample-size budget) is the cleanest fair comparison. The concern is a valid caveat but not a flaw.

- **Strength Finder's "NAMMDFuse is a simple drop-in replacement":** This is true but generic — not an independent strength, merely a consequence of the definition.

---

## Novel Insights

The key conceptual insight — that the RKHS norm `‖μ_P‖²` equals `1 − Var(P,κ)` for bounded kernels, so that distributions with higher norms are more concentrated and thus have a more meaningful MMD gap — provides a clean geometric rationale for normalizing MMD. This is not merely a scaling trick: it connects the kernel DCT problem to the geometry of the RKHS unit ball, giving NAMMD a principled interpretation as a ratio of inter-distribution distance to the "available spread" of the distributions in the RKHS. This framing could generalize to other distance measures defined in inner product spaces.

---

## Calibration and Score

**Anchor papers retrieved:**

| Path | Avg Score | Comparison |
|---|---|---|
| `z9j7wctoGV.md` (Deep Kernel Relative Test) | **6.00** | Closest topically: kernel test for a specific applied problem, novel use of relative testing. Accepted (poster). Comparably scoped contribution; the paper under review has stronger formal theory but smaller empirical gains. |
| `QCDdI7X3f9.md` (Model Equality Testing) | **6.50** | Two-sample testing applied to API model comparison. Accepted (poster). That paper had clearer and larger empirical gains; this paper has stronger theoretical structure. |
| `GPcSYm89wK.md` (Practical Kernel CI Learning) | **4.50** | Kernel test with marginally better power, theoretical gap in asymptotic analysis. Rejected. The paper under review has more complete theory and more systematic experiments. |
| `yqaN7MfkFU.md` (Regularized MMD) | **4.40** | MMD modification with marginal improvements and theoretical questions. Withdrawn. The paper under review has a cleaner normalization with full asymptotic theory. |
| `Ip6UwB35uT.md` (Conditional Testing via Conformal p-values) | **7.00** | A strong hypothesis-testing paper with rigorous finite-sample guarantees and multiple problem settings. Accepted (poster). Better experimental rigor and broader scope than the paper under review. |

**Positioning:** The paper falls between the 4.40–4.50 reject cluster and the 6.0–6.5 accept cluster. It is noticeably stronger than GPcSYm89wK and yqaN7MfkFU in theoretical completeness and consistent empirical demonstration. It is weaker than the 6.0–7.0 papers due to: (1) the Theorem 12 condition being unverifiable in practice, (2) small individual effect sizes in Table 1 without significance testing, and (3) the DCT-specific contribution not being cleanly isolated experimentally. The most comparable accepted paper is z9j7wctoGV (avg 6.0), which also had a kernel testing contribution with application focus. The paper under review has comparable depth but a more notable theoretical gap (Theorem 12 condition). This positions it slightly below 6.0.

**Final Score: 5.5**

This is a borderline paper with a real, clean contribution (NAMMD normalization + full theoretical framework), consistent but small empirical improvements, and one notable Major weakness (Theorem 12's unverifiable assumption). The contribution is genuine but the empirical and theoretical gaps prevent a comfortable accept. A revised submission with empirical verification of the Theorem 12 condition and DCT-isolated comparisons would be significantly stronger.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>