Now I have all the information I need. Let me write the final review.

## Summary

This paper studies sparse recovery under heterogeneous noise from two quality sources (high-quality with variance σ₁² and low-quality with variance σ₂²), establishing sufficient conditions for information-theoretic recovery in agnostic and informed settings, and a necessary-and-sufficient phase transition for LASSO recovery. The paper introduces the "Price of Quality" γ — the number of low-quality samples equivalent to one high-quality sample under the sufficient condition — showing γ ≤ 2 in the agnostic IT setting, γ → ∞ in the informed IT setting, and γ = 1 for the LASSO (sample count threshold depends only on n₁+n₂).

## Strengths

- **The LASSO phase transition (Theorem 3) is a genuine, non-trivial contribution.** It cleanly extends Wainwright (2009) to heterogeneous noise by providing necessary (Eq. 26) and sufficient (Eq. 27) conditions, showing the threshold depends only on total sample size n₁+n₂. The proof technique — using QR decomposition and Haar measure properties (Lemma D.6) to handle the non-Wishart structure introduced by the diagonal Σ — is a real technical advance over the homogeneous-noise proof.

- **The "Price of Quality" is a clean, interpretable metric.** The linear trade-off form α₁n₁ + α₂n₂ ≥ n* (Eqs. 9, 16) makes γ = α₁/α₂ well-defined and interpretable, and the closed-form expressions across SNR regimes (Eqs. 12–14, 18–21) provide immediate practical guidance.

- **The distinction between agnostic and informed settings is well-motivated and practically relevant**, correctly identifying that the key difference is whether Σ⁻¹ rescaling is available. The practical implication — "quantify uncertainty in the annotations and rescale the loss accordingly" (§5) — is directly supported by the provable gap between the agnostic and informed Price of Quality.

- **Systematic analysis across all SNR regimes** for both agnostic and informed settings gives a complete picture of when quality matters most, and the generalization to arbitrary invertible Σ (Remark 3.4, Eqs. 22–23) shows the results are not artifacts of the two-block assumption.

## Weaknesses

### Fatal
None.

### Major

- **The Conclusion claims the informed information-theoretic threshold is "sharp," but Remark 3.3 explicitly states necessity is unproven.** The Conclusion (§5) states: "the informed information-theoretic threshold and the LASSO threshold are sharp." Yet Remark 3.3 says: "Establishing full necessity in the heterogeneous setting remains an interesting direction for future work." This is an internal contradiction. Since the informed condition (16) is sufficient but not proven necessary, calling it "sharp" is inaccurate. This matters because the paper's central narrative — the "fundamental difference" between bounded agnostic γ ≤ 2 and unbounded informed γ → ∞ — depends on both conditions being tight. If the agnostic condition is loose (which the paper acknowledges) and the informed condition's necessity is unproven, the contrast could partly reflect different levels of looseness rather than a genuine structural difference in the underlying problems.

- **The agnostic γ ≤ 2 bound is a property of the relaxed sufficient condition, not a proven property of the recovery problem.** The paper acknowledges in Remark 3.2 that the sufficient condition (9) arises from a relaxation of the cubic equation (37) in the Chernoff bound, and is "not expected to be information-theoretically sharp." The true agnostic Price of Quality could be larger than 2; the relaxed bound simply cannot detect it. While the abstract and §1.2.1 include the qualifier "for this sufficient condition to hold," the Conclusion (§5) drops it: "one high-quality sample is never worth more than two low-quality samples." This inconsistent framing presents a property of a loose bound as if it were a fundamental finding about the problem. The paper would be significantly strengthened by either tightening the bound or consistently and prominently qualifying the claim throughout.

### Minor

- **Theorem 3 requires n₁, n₂ = ω(s), excluding the practically relevant fixed-n₁ regime.** In mixed-quality data applications, one typically has a fixed small number n₁ = O(1) of expert labels combined with many low-quality labels. The current result does not cover this case, which is the most practically important setting.

- **The LASSO "striking robustness" framing in the abstract slightly overstates the finding.** While the sample-count threshold n_ALG is genuinely independent of the noise split, Proposition 4.1 shows recovery also requires σ²_avg = o(n/((1+s/ρ²)log(p−s))), which constrains how much high-variance data can be added. The paper acknowledges this, but the abstract's "striking robustness" language could mislead readers into thinking low-quality data is freely substitutable without any noise-level constraint.

### Trivial
None.

## Nice-to-Haves

- **Numerical verification of the sufficiency gap.** A figure comparing the sufficient condition (9) against the numerically optimized Chernoff bound (solving equation 37) for representative parameter regimes would directly quantify the looseness that undermines the agnostic Price of Quality claims.

- **Simulation of the LASSO phase transition under heterogeneity.** Empirical recovery probability as a function of n for different (n₁, n₂) splits at fixed total n would make the "robustness" claim concrete and visually compelling.

- **Establishing necessity for the informed condition (Theorem 2).** The informed MLE setting is the natural place to establish tight IT results since the Chernoff exponent can be optimized in closed form (§3.2 proof sketch). Proving necessity would convert a suggestive sufficient condition into a genuine threshold.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Omidiran & Wainwright sufficiency-only criticism:** The harsh critic claimed the paper does not mention that Omidiran & Wainwright (2008) only establishes sufficiency. In fact, the Conclusion explicitly states: "although this was shown only for the sufficient condition, with no corresponding result on necessity." This criticism is factually wrong.

- **Parser artifacts (σ₄², etc.):** These are PDF extraction artifacts, not author errors.

- **Edge case when σ₁ ≈ σ₂:** The harsh critic notes the Price of Quality is "least meaningful" when noise levels are close. This is trivially obvious and not a weakness — the concept is designed for the regime where σ₁ ≪ σ₂.

- **SNR regime specification for n_INF and n_ALG in §1.1.1:** These are standard results from Reeves et al. (2019) and Wainwright (2009). The paper discusses SNR dependence immediately after in the same section. Minor presentation issue at best.

- **Demand for experiments as a core weakness:** This is a theory paper; experiments would strengthen it but are not required by community standards for this type of contribution.

- **Missing related works claims:** Cannot verify without external sources; removed per rules.

- **Strength finder's "striking contrast" strength conflicts with verified Major weakness about sufficient-only conditions:** The claimed "fundamental dichotomy" between bounded agnostic γ and unbounded informed γ is weakened by the fact that the agnostic bound is sufficient-only and potentially loose. Downgrading this strength — the contrast is suggestive but not proven to be fundamental.

## Novel Insights

The paper's most insightful observation is that the LASSO's insensitivity to noise heterogeneity (Price of Quality = 1 at the sample-count level) coexists with a genuine noise-level constraint (Proposition 4.1), creating a subtler picture than "quality doesn't matter algorithmically": quality doesn't affect *how many* samples you need, but it does affect *how noisy* those samples can be on average before recovery becomes impossible. This dual structure — sample-count independence coupled with noise-averaging dependence — is more nuanced than the paper's "striking robustness" framing suggests, and is itself a contribution worth emphasizing more precisely.

## Suggestions

- Fix the internal contradiction: either establish necessity for the informed condition (making the "sharp" claim accurate) or revise the Conclusion to clearly distinguish which thresholds are sharp (LASSO) from which are sufficient-only (both IT conditions).
- Consistently qualify the agnostic γ ≤ 2 claim throughout, including in the Conclusion, or provide a numerical comparison showing the gap between the relaxed and optimized Chernoff bounds is small enough that the qualitative conclusion (γ bounded vs. γ unbounded) is robust.
- Discuss the fixed-n₁ regime limitation explicitly and consider whether partial results (e.g., for n₁ = s·polylog(p)) are achievable with current techniques.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Scaling Laws and Spectra of Shallow NNs | Q3yLIIkt7z | 7.00 | High anchor: precise phase diagram, complete characterization — our paper's LASSO result is comparable in quality but the IT results are sufficient-only and the framing overclaims |
| Computational Bottlenecks for Diffusions | rAjHUNXybH | 7.33 | High anchor: sharp algorithmic barriers instantiated on concrete problems — stronger than our paper because both IT and algorithmic results are tight |
| Identifiability of Sparse Linear ODEs | BYBKqpZteT | 6.00 | Medium anchor: sharp threshold but limited assumptions — similar profile to our paper (good theory result, practical limitations) but our paper has the additional framing issue |
| Noisy Zeroth-Order Hard-Thresholding | E5NwyihhEc | 5.00 | Medium anchor: valid but unsurprising theoretical contribution — our paper has a stronger contribution (Theorem 3) but also overclaims more |
| Graph-guided Sparse Learning | IMQyjIxVew | 4.67 | Medium-low anchor: valid technical contribution but unclear formulation value — our paper is clearly stronger |
| Gradient Flow Convergence | b36drMoKir | 0.50 | Low anchor: hidden assumption trivializes the result — our paper is far above this; Theorem 3 is a genuine contribution |
| Large Weights / Layer Collapse | TW5DEgtacg | 2.00 | Low anchor: sufficient conditions too strong, overclaimed — our paper's overclaim is less severe; Theorem 3 is solid |

The paper sits between the medium-scoring anchors. It is clearly above papers like E5NwyihhEc (5.00) and IMQyjIxVew (4.67) because Theorem 3 is a genuine, non-trivial contribution with complete (necessary and sufficient) characterization. However, it falls below BYBKqpZteT (6.00) because the overclaimed framing in the Conclusion and the internal contradiction on sharpness are significant issues for a theory paper whose narrative depends on comparing tightness across settings. The LASSO result alone would merit ~6, but the misleading framing of the IT results as "fundamental" when they are sufficient-only, combined with the Conclusion's incorrect sharpness claim, brings the score down.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>