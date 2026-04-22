Now I have all the information needed to write the review. Let me also check the inconsistency between equations (9) and (12) one more time by re-reading exactly.

The key facts:
- Equation (9) coefficient of n₁: $\log\left(1 + \frac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_2^2}\right)$
- Equation (12) numerator: $\log\left(1 + \frac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_1^4}\right)$
- These denominators differ: $2\sigma_2^2$ vs $2\sigma_1^4$
- The paper explicitly says γ is defined from the coefficients in (9) (lines 189-192)
- Equation (14) uses the formula from (12) with $2\sigma_1^4$ and arrives at $\gamma \simeq 2 - \sigma_1^2/\sigma_2^2$

This inconsistency is real and significant. If the correct coefficient from (9) were used, the low SNR₂ approximation would give a different result.

Now let me write the final review.</think>Now I have all the information needed. Let me compose the final review.

## Summary

This paper studies sparse recovery under mixed-quality (heterogeneous Gaussian noise) data, establishing sample-size conditions for both information-theoretic and algorithmic recovery. It introduces the "Price of Quality" γ—the number of low-quality samples replaceable by one high-quality sample—as a scalar summary of the quality-quantity trade-off. For the agnostic setting (decoder unaware of noise levels), γ is bounded by 2 under the sufficient condition; for the informed setting (decoder knows noise levels), γ can grow arbitrarily. For the LASSO, a necessary-and-sufficient phase transition shows the threshold depends only on total sample size and average noise, making high- and low-quality data contribute equally—a striking contrast to the IT thresholds.

## Strengths

- **Complete LASSO phase transition (Theorem 3):** Both necessity (part i, eq. 26) and sufficiency (part ii, eq. 27) are proven, extending Wainwright (2009) to the heterogeneous-noise setting. The finding that the threshold $n_{\mathrm{ALG}}$ is unchanged and the regularization condition involves only $\sigma_{\mathrm{avg}}^2$ is clean and potentially surprising.

- **Novel proof technique for LASSO under heterogeneous noise (§4):** The classical Wishart-structure argument breaks down because $\Sigma$ is no longer scalar. The proof overcomes this via QR decomposition of $X_S$ (eq. 49) and Haar-measure analysis (Lemma D.6), providing a new technical tool applicable beyond this setting.

- **Conceptual framework of Price of Quality:** Defining γ = α₁/α₂ as a single interpretable scalar quantifying the quality–quantity trade-off is a useful conceptual contribution. The stark contrast—γ bounded by 2 (agnostic IT), γ → ∞ (informed IT), γ = 1 (LASSO)—captures a genuine and important structural insight.

- **First conditions for sparse recovery under heterogeneous noise:** To the best of our knowledge, this is the first work providing sample-size conditions for this setting, filling a gap in the sparse recovery literature.

## Weaknesses

### Fatal

None.

### Major

- **Formula inconsistency between eq. (9) and eq. (12):** The sufficient condition (9) defines the n₁ coefficient as $\log\!\bigl(1 + \tfrac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_2^2}\bigr)$, with denominator $2\sigma_2^2$. However, the Price of Quality definition (12) defines γ's numerator as $\log\!\bigl(1 + \tfrac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_1^4}\bigr)$, with denominator $2\sigma_1^4$. Since γ is explicitly described as the ratio of the coefficients from condition (9) (lines 189–192: "The coefficients of the sample sizes reveal that one unit of high-quality data is worth: γ := …"), these should match. This inconsistency in the paper's central definition—different denominators for what should be the same coefficient—casts doubt on the precise meaning of γ and the validity of the derivation in (14), which depends on the (12) form. The entire framework relies on this being correct.

- **Overclaimed sharpness of the informed IT threshold:** The conclusion (line 354) states "the informed information-theoretic threshold and the LASSO threshold are sharp." For the LASSO, this is justified by Theorem 3 (both necessity and sufficiency). For the informed IT threshold, Theorem 2 only provides a *sufficient* condition, and Remark 3.3 explicitly notes: "Establishing full necessity in the heterogeneous setting remains an interesting direction for future work." Claiming sharpness without a necessity proof is an overstatement, and the caveat is buried in a remark rather than reflected in the conclusion.

- **Price of Quality γ ≤ 2 is a property of a loose sufficient condition, not of the IT limit:** The central headline finding—"one high-quality sample is never worth more than two low-quality samples"—holds only under Theorem 1's sufficient condition, which the authors explicitly acknowledge is not expected to be information-theoretically sharp (Remark 3.2: "not expected to be information-theoretically sharp"). This bound γ ≤ 2 could be an artifact of the Chernoff relaxation rather than a fundamental property. The abstract and introduction present this finding prominently without adequate qualification that it applies only to a possibly-loose sufficient condition. This does not invalidate the result but significantly limits the strength of the "never worth more than two" claim.

### Minor

- **Incomparable signal models across sections:** The IT results (§3) assume binary signals β* ∈ {0,1}ᵖ, while the LASSO result (§4) assumes β* ∈ C_{p,s}(ρ). The paper's narrative contrasts these thresholds, but they are not directly comparable under different signal assumptions. Remark 3.1 argues the binary case is "hardest" (supporting the sufficiency direction), but the LASSO necessity direction (Theorem 3 part i) might not hold for binary signals.

- **No numerical or experimental validation:** The paper is entirely theoretical. Simulations confirming (e.g.) that LASSO recovery indeed depends only on total n and σ²_avg, or illustrating the gap between the sufficient condition and empirical recovery, would strengthen the presentation and help readers assess the tightness of the agnostic IT bounds.

### Trivial

- The alternative estimator suggested in Remark 3.2 (weighting by 1/Yᵢ²) is mentioned without any analysis, which raises more questions than it answers. A brief comment on feasibility would help, but this is a minor scope issue.

## Nice-to-Haves

- Numerical experiments comparing the sufficient condition boundaries against empirical recovery rates, which would reveal the gap between Theorem 1's condition and the true threshold and help assess whether γ ≤ 2 is tight.
- Explicit characterization of the "more data can hurt" phenomenon implied by Proposition 4.1 (increasing n₂ can push σ²_avg beyond the feasibility bound (30)).

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **The informed IT Chernoff optimization being "exact" (Harsh Critic point 2 partial):** The harsh critic notes the Chernoff optimization in the informed case is "exact" but this still only gives sufficiency. The paper does acknowledge this (Remark 3.3). The partial overclaim in the conclusion is kept as a Major weakness above, but the broader point that the Chernoff exponent is tight in the homogeneous case is a reasonable argument for likely tightness that the paper does not fully rely on.

- **Remark 3.4 generalization being conservative with σ_max(Σ):** This is a minor observation about a remark; the paper explicitly flags it as a generalization and does not overclaim tightness. Removed as too minor.

- **"More data can hurt" for LASSO (Harsh Critic §4 note):** This is implicit in Proposition 4.1's condition on σ²_avg, but is not explicitly discussed. Moved to Nice-to-Haves since it's an interesting observation the authors could highlight but not a flaw in their results.

- **Missing experiments (Harsh Critic):** Moved to Minor; pure theory papers are common in this community, but empirical validation would strengthen the paper in this case because the tightness of the agnostic sufficient condition is a key open question.

- **Demand for tighter agnostic IT bounds or analysis of eq. (37):** This is beyond the paper's stated scope. The authors acknowledge this as future work. Moved to Nice-to-Haves.

- **Reproducibility/implementation concerns:** Removed per rules.

- **The conclusion's speculation about algorithmic thresholds being "more robust" (Harsh Critic §5):** The conclusion offers this as a broader observation supported by one example plus cited work; this is reasonable speculative discussion and not an overclaim. Removed.

- **Formatting artifacts, typos, notation issues:** Removed per rules.

## Novel Insights

The most distinctive insight is the fundamental decoupling between IT and algorithmic robustness to data heterogeneity: in the agnostic IT setting, one high-quality sample replaces at most two low-quality ones (γ ≤ 2), yet algorithmically the LASSO treats both equally (γ = 1). This means computational recovery is *more robust* to data quality variation than any agnostic information-theoretic decoder. However, this insight's precision is undermined by the formula inconsistency in the central definition of γ, which must be resolved.

## Suggestions

- **Resolve the formula inconsistency between equations (9) and (12)** before publication. Verify which denominator is correct (likely the one from (9), since it derives from the Chernoff bound), update the Price of Quality definition and all subsequent approximations consistently, and re-derive equation (14).
- **Qualify the "sharpness" claim for the informed IT threshold** in the conclusion, or at minimum add "expected to be sharp" rather than stating it as established, to match the honest caveat in Remark 3.3.
- **Add a prominent qualification** in the abstract/introduction that γ ≤ 2 in the agnostic setting holds under a sufficient condition that is not proven tight, to avoid readers interpreting it as a fundamental IT bound.
- Add even a simple simulation figure to validate the LASSO phase transition prediction and illustrate the (n₁, n₂) recovery boundary.

## Score and Decision

**Calibration anchors:**

1. **High-scoring anchors (avg >7):** 8KcjEygedc (7.5, Accept Poster) — theoretical framework for data curation with exact phase transitions; Q3yLIIkt7z (7.0, Accept Oral) — phase diagram with LASSO connections; AUXvYQlQLZ (7.33, Accept Oral) — necessary and sufficient conditions, clean theory. These papers have clean, complete results with no major internal inconsistencies.

2. **Medium-scoring anchors (4-6):** hisAy19yMP (5.5, Reject) — sufficient-only conditions for identifiability, flagged as key weakness; YM6KIpl6aR (5.0, Reject) — fundamental efficiency-confidence bound with tightness concerns; TP4gsjpi7z (3.5, Reject) — phase transition in shuffled regression with approximation concerns.

3. **Low-scoring anchors (<3):** TW5DEgtacg (2.0, Reject) — overclaimed necessity from sufficient conditions, logical errors.

**Comparison:** This paper has a genuine and important contribution (the LASSO phase transition with the novel QR/Haar proof technique), plus a useful conceptual framework (Price of Quality). However, the formula inconsistency between (9) and (12) is a significant mathematical error in the paper's central definition — it's not a cosmetic typo but an error affecting what γ means. The overclaimed sharpness of the informed IT threshold and the insufficient qualification of the γ ≤ 2 bound are serious but addressable presentation issues. Compared to the high-scoring anchors (7+), those papers have clean, complete results without internal inconsistencies. Compared to the medium anchors (4-5.5), those have sufficient-only conditions flagged as weaknesses — this paper shares that issue. The formula error puts this paper below the high-scoring tier. The LASSO result and the conceptual framework are real contributions above the purely weak/overclaimed anchors (3-4 range). I place this paper in the upper range of the medium tier, penalized by the formula error but buoyed by the solid LASSO contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>