Now I have enough information to write the final review.

## Summary

This paper introduces the "Price of Quality" (γ) framework to quantify sample-size trade-offs when recovering sparse signals from mixed-quality observations (heterogeneous noise with known variances σ₁² < σ₂²). The authors establish sufficient conditions for support recovery under two settings: agnostic (decoder unaware of per-sample noise provenance) and informed (decoder knows per-sample variances). A key finding is that the LASSO phase transition in the agnostic setting depends only on average noise σ²_avg, matching the homogeneous-noise threshold — revealing a fundamental asymmetry between information-theoretic and algorithmic behavior under data heterogeneity.

---

## Strengths

- **LASSO phase transition (Theorem 3) is technically sound and two-sided.** The result that the LASSO threshold depends only on σ²_avg — regardless of the (n₁, n₂) split — is both surprising and non-trivial to prove. The proof overcomes the broken Wishart structure via a QR decomposition and Haar measure argument (Lemma D.6), constituting a genuine extension of Wainwright (2009). Theorem 3 is accompanied by the exact matching lower bound, making it a sharp characterization.

- **Sharp agnostic vs. informed contrast.** Theorems 1 and 2 together establish a qualitative distinction: the Price of Quality is uniformly bounded below 2 in the agnostic setting (Eq. 12–14) but can grow without bound in the informed setting (Eq. 19–21, γ → +∞ when SNR₁ is high and SNR₂ is low). This asymmetry is the paper's core conceptual contribution and is clearly demonstrated.

- **Clean generalization to arbitrary noise structure (Remark 3.4).** Extending both Theorems 1 and 2 from two-block noise to arbitrary non-singular Σ via singular-value conditions (Eqs. 22–23) adds genuine scope without requiring separate proof machinery.

- **Proposition 4.1 provides a necessary and sufficient condition** on the noise scaling regime for the LASSO regularization sequence to exist, closing the loop on when Theorem 3 applies.

- **Clear, actionable practical implication.** The paper concisely translates the theoretical contrast into a concrete recommendation: "whenever possible, quantify uncertainty in the annotations and rescale the loss accordingly" — directly grounded in the divergent Price of Quality behavior between the two settings.

---

## Weaknesses

### Fatal
None.

### Major

- **IT results (Theorems 1 and 2) are one-sided sufficient conditions with no matching lower bounds.** The "Price of Quality" framework is inherently incomplete without converse results. For the agnostic setting, the γ ≤ 2 bound is explicitly a property of a relaxed Chernoff inequality (Remark 3.2 acknowledges: "The condition in Theorem 1 is sufficient and is not expected to be information-theoretically sharp"). For the informed setting, the claim of approximate sharpness in Remark 3.3 rests only on an analogy with homogeneous-noise results, not on a proved converse. Without lower bounds, the paper cannot characterize what recovery actually requires — only what the chosen estimator provably achieves. The γ ≤ 2 result in particular cannot be interpreted as an information-theoretic statement about the fundamental substitution rate between high- and low-quality samples. This is the central limitation of the paper's framing.

### Minor

- **The abstract slightly overstates the γ ≤ 2 result.** The abstract states "one high-quality sample is never worth more than two low-quality samples" without qualification. While the body text is careful (line 245–246: "under our sufficient condition…"), and the conclusion explicitly flags "the agnostic information-theoretic condition is sufficient but not proven tight," the abstract reads as a fundamental limit rather than a property of the sufficient condition. This should be corrected for clarity.

- **The agnostic MLE estimator (8) may not be optimal for heterogeneous noise.** Remark 3.2 notes that alternative reweighted estimators (e.g., reweighting by |Yᵢ|²) could outperform the homogeneous ℓ₀-minimizer without knowing individual σᵢ². Theorem 1 therefore characterizes a specific (potentially suboptimal) estimator. Stating this explicitly in the main theorem would help readers understand the scope of the result.

- **No numerical validation.** The paper is purely theoretical, but even a phase transition simulation comparing LASSO recovery rates at σ²_avg versus σ₁² and σ₂² would help validate Theorem 3 at finite sample sizes and demonstrate the practical relevance of the asymptotic threshold.

- **The n₁, n₂ = ω(s) condition in Theorem 3** excludes cases where one group is small (e.g., n₁ = O(s)), which may be practically relevant when high-quality samples are rare. The paper does not discuss whether this assumption can be relaxed.

### Trivial
- The informed LASSO case is left open (acknowledged in Remark 4.2). No algorithmic result accompanies Theorem 2.

---

## Nice-to-Haves

- A phase diagram in (n₁, n₂) space illustrating the sufficient condition curves (Eqs. 9, 16) alongside the n* threshold and the "equal quality" diagonal would make the Price of Quality geometry immediately interpretable.
- Solving the exact Chernoff exponent (the cubic equation referenced in Remark 3.2) would either tighten the γ ≤ 2 bound or potentially recover it as sharp, which would substantially strengthen the IT contribution.
- A GLS-LASSO analysis for the informed setting would complement Theorem 2 with an algorithmic result.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "γ ≤ 2 is a fundamental misrepresentation in the abstract and conclusion."** The paper body is careful (lines 245–246 and the conclusion both qualify the result as being under the sufficient condition), and the issue is reduced to a minor imprecision in the abstract, not a structural misrepresentation.

- **Harsh Critic: "Estimator (8) is suboptimal — the result only holds for one specific estimator."** This is a genuine observation but the paper acknowledges it in Remark 3.2. It is a limitation of scope, not a flaw.

- **Harsh Critic demanding empirical validation as a major gap.** This is a theory paper in the tradition of Wainwright (2009) and related sparse recovery literature. Single-run empirical validation of phase transitions is not standard in this subfield; demanding it would be scope creep.

- **Strength Finder — "practical implication is concrete."** Retained as a minor strength; moved to supporting role since it follows directly from Theorem 2 rather than being independently validated.

---

## Novel Insights

The most genuinely novel observation is the **fundamental asymmetry in how data heterogeneity interacts with information-theoretic versus algorithmic recovery**. In the agnostic setting, heterogeneity has bounded impact on IT thresholds (γ ≤ 2) and *no* impact on the algorithmic LASSO threshold (which collapses to the average noise level). In the informed setting, heterogeneity can make IT thresholds arbitrarily sensitive to quality composition. This reveals a new axis — noise provenance awareness — along which the IT-vs.-algorithmic gap manifests, complementing the Overlap Gap Property literature (Gamarnik & Zadik) which characterizes this gap via algorithmic hardness rather than sample efficiency.

---

## Suggestions

1. Prove or formally rule out matching lower bounds for Theorem 1. Even a partial lower bound via Fano's inequality in the heterogeneous noise setting would significantly sharpen the paper's contributions.
2. Add a phase transition figure in the (n₁, n₂) plane showing the sufficient condition (9) as a curve, together with the homogeneous threshold n* and the "equal quality" line.
3. Revise the abstract to state γ ≤ 2 as a property of the sufficient condition: e.g., "under our sufficient condition, one high-quality sample is never worth more than two low-quality samples."

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison |
|---|---|---|
| `/human_reviews/BlkxbI6vzl.md` | 7.0 | Sparse phase retrieval — two-sided theory + experiments; paper under review has stronger LASSO result but no experiments and weaker IT results |
| `/human_reviews/7liN6uHAQZ.md` | 6.75 | Sketching for regularized LS with minimax rates — comparable theoretical depth; similarly lacks one-sided converseness |
| `/human_reviews/e8qXTxMgPg.md` | 6.5 | Sparse vector dimensionality reduction with lower bounds — similar scope but has matching bounds |
| `/human_reviews/QY52D9BeJo.md` | 6.0 | Multi-index model sample complexity via phase transitions — similar theoretical style, rejected despite solid content |
| `/human_reviews/UrKbn51HjA.md` | 5.25 | High-dimensional factor mixture with incomplete characterization — analogously one-sided |
| `/human_reviews/xGvPKAiOhq.md` | 8.0 | Over-parameterization + gradient descent — tight two-sided bounds throughout; paper under review lacks this completeness |
| `/human_reviews/lK0WxHeups.md` | 2.5 | SGD complexity — incremental, uninspiring contribution; paper under review is substantially stronger |
| `/human_reviews/Hh0Cg4epYY.md` | 2.33 | f-divergence Bayes error bounds — weak and one-sided; paper under review is far stronger |

**Assessment:** The paper falls comfortably above the low anchors and is in the upper half of the medium tier. Theorem 3 is a strong two-sided result. The IT theorems are acknowledged sufficient-conditions only, which prevents the paper from matching the high-scoring anchors (which typically offer tight, two-sided theory throughout). The novel conceptual framework and the clean LASSO result make this comparable to the 6.5–7.0 papers in the sparse recovery literature. The one-sided IT results hold the paper back from the 7+ tier. Score: **6.5**.

**Originality:** Moderate-to-high. The "Price of Quality" framing and the specific mixed-quality noise model are novel.  
**Importance:** Moderate. The problem is motivated and theoretically clean; practical impact depends on whether the Gaussian-design, known-sparsity assumptions are representative.  
**Support for claims:** Theorem 3 is fully supported (two-sided); Theorems 1–2 are partially supported (one-sided, acknowledged).  
**Experimental soundness:** N/A (pure theory).  
**Clarity:** Good; notation is consistent and results are well-organized.  
**Value to community:** Genuine contribution to high-dimensional statistics and sparse recovery literature.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>