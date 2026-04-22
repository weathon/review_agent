Now I have all the information needed to write the final consolidated review. Let me synthesize everything.

## Summary

This paper studies sparse signal recovery when observations come from two sources with different noise levels (high-quality and low-quality). It establishes the first sufficient conditions on sample sizes $(n_1, n_2)$ for both information-theoretic and LASSO-based algorithmic recovery. The central conceptual contribution is the "Price of Quality" γ—the number of low-quality samples that can replace one high-quality sample while preserving the sufficient condition. In the agnostic setting (decoder unaware of noise levels), γ is uniformly bounded below 2; in the informed setting (decoder knows per-sample variances), γ can diverge to infinity. A LASSO phase transition theorem shows the algorithmic threshold depends only on total sample size and average noise, revealing a contrast between information-theoretic and algorithmic sensitivity to data quality.

## Strengths

- **First results on a natural and timely problem.** Mixed-quality data is ubiquitous (weak supervision, LLM-annotated data, multi-site clinical trials), and this paper provides the first formal analysis of sparse recovery under heterogeneous noise. The two-source model is the right starting point (§1.1.2, §1.2).

- **Theorem 3 (LASSO phase transition) is the paper's strongest contribution.** It provides a tight two-sided characterization (both necessity and sufficiency) showing the LASSO threshold $n_{\text{ALG}} = 2s\log(p-s) + s + 1$ is independent of individual noise levels $\sigma_1^2, \sigma_2^2$ (eq. 26–27), with the regularization condition depending only on $\sigma_{\text{avg}}^2$ (eq. 28). This is a non-trivial extension of Wainwright (2009) requiring a QR decomposition of $X_S$ and Haar measure arguments (§4, Lemma D.6) to handle the broken Wishart structure.

- **Conceptually useful Price of Quality framework.** Even if the specific numerical bound γ ≤ 2 is tied to the proof technique, the idea of quantifying the trade-off between high- and low-quality data via the slope of the sample-complexity boundary is a good conceptual contribution that future work can refine (eq. 5, §1.2.1).

- **Transparent about limitations.** The paper clearly states the sufficiency-only nature of the IT results (Remark 3.2), the non-optimality of the agnostic estimator (Remark 3.2), and the difficulty of the informed LASSO extension (Remark 4.2). The conclusion explicitly notes that "the agnostic information-theoretic condition is sufficient but not proven tight" (§5).

## Weaknesses

### Fatal
None.

### Major

- **The central agnostic-vs-informed contrast is partially driven by asymmetric bound tightness, which the paper insufficiently addresses.** The agnostic sufficient condition (Theorem 1, eq. 9) results from a *relaxed* Chernoff bound—the paper notes in Remark 3.2 that optimizing the exponent leads to a cubic equation whose solution would be tighter. The informed condition (Theorem 2, eq. 16) optimizes the Chernoff exponent exactly. The headline contrast—bounded γ in the agnostic setting vs. unbounded γ in the informed setting—could therefore be partly an artifact of the agnostic bound being looser rather than a fundamental structural difference. The paper acknowledges this asymmetry in Remarks 3.2 and 3.3 but does not discuss its implications for interpreting the contrast. A tighter agnostic analysis might yield a much larger γ, potentially narrowing the gap between the two settings. Without at least a numerical investigation of how much tightness is lost (e.g., solving the cubic from Remark 3.2 for specific parameters), the reader cannot assess whether the qualitative contrast is robust or primarily proof-driven. The claim of a "fundamental difference" in the abstract is partially undermined by this gap.

- **The paper's story is incomplete: no informed LASSO result.** The stated contribution is to expose "a fundamental difference between how the information-theoretic and algorithmic thresholds adapt to changes in data quality." For information-theoretic thresholds, both agnostic and informed settings are analyzed; for algorithmic thresholds, only the agnostic setting is covered. The informed extension is acknowledged as nontrivial (Remark 4.2), but its absence means the central comparison is between (agnostic IT, informed IT) on one side and only (agnostic ALG) on the other. Whether the LASSO's noise-averaging robustness persists when the loss is reweighted is an open question that directly affects the paper's narrative arc.

### Minor

- **No numerical verification of the sufficient conditions.** Simulations plotting recovery probability as a function of $(n_1, n_2)$ for various $(\sigma_1^2, \sigma_2^2)$ would allow readers to assess how loose the agnostic bound is and whether the γ ≤ 2 finding is empirically approximately correct. This is standard practice even in theoretical papers in this area (e.g., the sparse phase retrieval literature includes numerical phase diagrams).

- **The agnostic coefficient for $n_1$ depends on $\sigma_2^2$** (eq. 9): the value assigned to a high-quality sample in the sufficient condition is affected by the noise level of low-quality samples, which is presumably an artifact of the unweighted loss function and the Chernoff relaxation rather than a meaningful property. This is not discussed in the paper but deserves acknowledgment.

- **Remark 3.4 (generalization to arbitrary Σ) is stated without proof.** The extension to general invertible Σ giving conditions (22) and (23) is presented as a natural extension of the proof strategy but without even a sketch of why the arguments carry through.

### Trivial
None.

## Nice-to-Haves

- Solving the cubic equation from Remark 3.2 numerically for specific parameter values and comparing the resulting γ against the relaxed γ ≤ 2 bound would directly address the tightness concern and substantially strengthen the paper's conceptual claims.
- Phase diagrams in the $(n_1, n_2)$ plane for concrete $(\sigma_1^2, \sigma_2^2, s, p)$ would make the Price of Quality immediately visible.
- Establishing even partial necessity (e.g., showing recovery fails when both $n_1$ and $n_2$ are too small) would elevate the IT results from "sufficient conditions with interpretable coefficients" to a characterization of the trade-off.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the abstract omits the "under our sufficient condition" qualifier**: This is factually incorrect. The abstract explicitly states "one high-quality sample is never worth more than two low-quality samples *for this sufficient condition to hold*" (line 23). The body text is also consistent in including this qualifier (§1.2.1, §3.1, §5). The overclaiming concern is more about the broader framing ("fundamental difference") than about the specific γ ≤ 2 statement.

- **Harsh critic's claim about Remark 3.1 overstating WLOG**: The remark does not use the phrase "without loss of generality." It carefully states that results "can be viewed as applying to signals whose non-zero components are at least 1 in magnitude" and provides the correct intuition that detecting a component of magnitude 1 is at least as hard as detecting a stronger component. The presentation is adequate.

- **Demand for the paper to address problems outside its scope** (e.g., variance-aware agnostic estimators, sub-Gaussian extensions): The paper explicitly scopes its analysis to the unweighted least-squares estimator and Gaussian noise. Requesting analysis of more sophisticated agnostic estimators or non-Gaussian settings is a nice-to-have, not a weakness.

- **Formatting/presentation nitpicks**: Removed per instructions.

## Novel Insights

The paper identifies a genuinely interesting structural phenomenon: the LASSO's algorithmic threshold is completely insensitive to noise heterogeneity (depending only on total sample size and average noise), while information-theoretic recovery is sensitive to quality differences. This mirrors a broader pattern in the sparse recovery literature where algorithmic thresholds appear more "robust" to model perturbations than information-theoretic ones (cf. sparse design results of Omidiran & Wainwright 2008). The Price of Quality framing—while not fully established quantitatively—captures a real and practically important insight: knowing uncertainty levels and rescaling the loss accordingly is disproportionately valuable when quality disparities are large.

## Suggestions

- Add a short numerical section (even 1–2 figures) solving the cubic from Remark 3.2 for specific parameters and comparing the resulting γ to the relaxed γ ≤ 2. This would cost little space and directly address the most important open question about the paper's central claim.
- Moderate the "fundamental difference" language in the abstract to acknowledge that the agnostic γ ≤ 2 result is derived from a non-tight sufficient condition. For example: "expose a qualitative difference...though the agnostic bound may tighten with further analysis."
- Discuss why the agnostic γ should be bounded independent of the bound tightness: is there an information-theoretic argument that, without quality knowledge, high-quality data has fundamentally limited marginal value? Even a paragraph of informal reasoning would strengthen the interpretability of the result.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** Sparse phase retrieval with provable algorithm (BlkxbI6vzl, avg 7.0, poster) — has both theory and experiments with tighter results; Streaming ℓ_p regression (Kpjvm2mB0K, avg 8.0, spotlight) — deep theory with tight upper and lower bounds; Data selection under weak supervision (HhfcNgQn6p, avg 7.75, oral) — mixed-quality data with theory + experiments + practical insights.
- **Medium band (4–6):** Multi-index model with sufficient conditions (QY52D9BeJo, avg 6.0, reject) — sufficient conditions that miss structural details, similar overclaiming pattern; Sparse RL sample complexity (nIEjY4a2Lf, avg 6.0, poster) — sufficient conditions for a new problem; Denoising under distribution shift (WmB803HJkD, avg 4.33, reject) — pure theory with questionable assumptions and no tight results.
- **Low band (<3):** Increasing batch size/LR (l2odw7OiNw, avg 2.5, reject) — acceleration claims from loose upper bounds, likely incorrect; Neural bounds on Bayes error (Hh0Cg4epYY, avg 2.33, reject) — oversold claims with loose bounds and poor soundness.

The paper under review is clearly above the low-band papers: its LASSO result (Theorem 3) is tight and technically sound, the problem is well-motivated, and the paper is honest about limitations. However, it falls below the high-band papers: those have tight characterizations, experimental validation, or more complete stories. Compared to medium-band papers, it is comparable to the multi-index model paper (avg 6.0, rejected) which also had sufficient-but-not-necessary conditions and was dinged for not fully establishing its conceptual claims. However, our paper has a stronger algorithmic result (Theorem 3 is tight two-sided) and identifies a more directly practical problem. I place it slightly above the QY52D9BeJo anchor but below the nIEjY4a2Lf anchor (which was accepted as poster). The paper's incomplete story (no informed LASSO) and the non-tightness of the central conceptual claim pull it down from a clear accept, but the genuine contributions—particularly Theorem 3—keep it from being a reject.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>