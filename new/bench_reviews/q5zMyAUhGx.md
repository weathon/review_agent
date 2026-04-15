Now let me search for calibration papers to anchor the score.Now I have enough information to write the final consolidated review. Let me verify the Corollary 2 assumption issue I noticed.Now I have all the information I need to write the consolidated review.

---

## Summary

This paper establishes the first rigorous generalization bounds for Kolmogorov–Arnold Networks (KANs), covering two settings: (1) activation functions represented as linear combinations of basis functions, yielding bounds that scale with the ℓ₁ norm of coefficient matrices and layer Lipschitz constants with only logarithmic dependence on combinatorial parameters; and (2) activation functions lying in a low-rank RKHS/Sobolev space, yielding bounds that scale polynomially with the underlying ranks. Both settings accommodate unbounded regression-type losses through a truncation argument. Empirical experiments track the derived complexity measure against excess loss over training epochs on simulated and real datasets.

---

## Strengths

- **First generalization bounds for KANs.** The paper addresses a clear and timely gap: despite rapid empirical adoption of KANs, no formal generalization theory existed. This is the primary contribution and it is genuine.
- **Clean compositional covering-number decomposition.** Proposition 1 provides a technically careful iterative upper bound on the covering number of a compositionally structured class, adapted to per-layer operator norms. This generalizes prior MLP results non-trivially.
- **Unbounded loss extension (Theorem 3).** Using a truncation argument to remove the boundedness assumption is a meaningful technical contribution; the result applies to squared loss, pinball loss, and Huber loss. This is more practically relevant than margin-only analyses.
- **Novel low-rank RKHS analysis.** The results in Section 2.3 for activations in Matérn RKHS spaces appear genuinely new, and the connection to fine-tuning via low-rank updates (Remark 6) is a well-motivated practical analogy.
- **Empirical correlation observed.** Across all six experimental settings, the derived complexity measure visually tracks the excess loss over training, which is at least consistent with the theory's predictions.

---

## Weaknesses

### Fatal
*None triggered. The core theoretical results are internally consistent and the mathematical contribution is genuine.*

### Major

- **Assumption 2 silently broadens the function class beyond standard KANs, weakening the central positioning.** The paper's headline is about KANs, defined in Eq. (1) via edge-wise *univariate* activation functions. However, Assumption 2 allows each output coordinate to be an arbitrary linear combination of *multivariate* basis functions g_{ij}^{(l)}(·). Remark 2 explicitly acknowledges this: "The form of Ψ in Assumption 2 is indeed more general than the additive structure in (1)." While Remark 2 also shows how the standard KAN parameterization (Eq. 5) is a special case, the key consequence is that Theorems 1–3 are *not* exploiting the distinctive additive/edge-wise univariate structure of KANs — they provide bounds for a broader class of Lipschitz compositional networks that merely contains KANs. The KAN-specific complexity advantages attributed to the architecture are therefore not established by the theorems as stated. This mismatch between the headline claim and the actual scope of the analysis is a meaningful structural issue.

- **The empirical section does not substantiate the claim of "practical relevance" due to normalization artifacts and absent cross-validation.** The paper explicitly states (Section 3): "we normalize the values of the complexity measures so that the maximum value of the complexity measure is equal to the last value of the excess loss." This forced rescaling means the visual alignment in Figure 2 is partly definitional — any roughly monotone quantity would be made to end at the same point as the excess loss curve. There is no reporting of unnormalized values, no formal correlation statistic, no multiple seeds, no cross-architecture validation, and no demonstration that controlling the proposed complexity measure improves generalization. The abstract's claim that "numerical results demonstrate the practical relevance of these bounds" is overstated relative to this evidence.

- **Potential error in Corollary 2: wrong assumptions referenced.** Corollary 2 (Section 2.3, low-rank RKHS setting) states: "Suppose Assumptions 1, 2 and 4 hold." However, Corollary 2 is the analogue of Corollary 1 for the low-rank RKHS setting and follows Theorem 5, which operates under Assumptions 4 and 5 (the RKHS assumption). Assumptions 1 and 2 pertain to the basis-function setting from Section 2.2, not the RKHS setting. This appears to be a copy-paste error from Corollary 1, but if the statement is faithful to the intended theorem, it affects the validity of the claim. This requires explicit correction.

- **The RKHS contribution (Theorems 4–5, Corollary 2) has zero empirical support.** All six experiments test only the basis-function complexity measure (Section 2.2). The low-rank RKHS analysis — which the abstract and contributions section present as a major result — is entirely unvalidated empirically. Given that this constitutes roughly half the theoretical contribution of the paper, this is a significant omission.

### Minor

- **Width dependence is absorbed, not eliminated, by the norm constraints.** Theorem 1 is correctly stated: no dependence on d̃ or p̃ outside log factors. However, the dominant complexity term α̃ depends on B_i, C_i, and products of Lipschitz constants ρ_j, all of which may scale with width, basis size, and parameterization. The discussion following Theorem 1 foregrounds the logarithmic dependence while underemphasizing the potentially strong implicit dependence through α̃. This can mislead readers into thinking KAN complexity is essentially width-insensitive.

- **No lower bounds or rate discussion.** The paper acknowledges the absence of lower bounds as future work, but without a matching lower bound, it is unclear how tight the derived rates are. The dominant terms scale as n^{-1} and n^{-(s-1)/(2s)}, and the paper does not compare these with minimax optimal rates for nonparametric regression under compositional structure (as e.g. Schmidt-Hieber (2020) does for MLPs).

- **Assumption 4's practical scope is underdiscussed.** The envelope condition sup_{f∈M} |L(f(·),·)| ≤ G(·) is required uniformly over the entire function class, which can be quite strong for rich deep models under unbounded losses. The paper states the condition but provides no discussion of when it is realistic for KAN classes as trained in practice.

- **Practical connection of the low-rank RKHS setting to standard KAN implementations is unclear.** The paper does not explain when practical KAN parameterizations (B-splines, Fourier, etc.) correspond to the low-rank Matérn RKHS structure, how ranks would be estimated during training, or why this model is more natural for KANs than for generic compositional networks.

### Trivial

- **Lipschitz constant upper bound in Remark 5 may be loose.** The paper uses ρ* ≤ ‖A‖_σ c_l √(b_l) as the empirical estimate of Lipschitz constants. The looseness of this bound is not discussed and directly inflates the empirical complexity measure.

---

## Nice-to-Haves

- Show raw (unnormalized) complexity curves alongside excess loss, or report Pearson/Spearman correlation coefficients to rigorously quantify agreement.
- Vary architecture parameters (depth, width, number of basis functions) in experiments to directly test the claimed scaling behavior and the log-only dependence on combinatorial parameters.
- Add a direct main-text comparison between the KAN bound's structure (B_i, ρ_l) and the MLP bound's structure (spectral norms) to concretely articulate what is structurally new; currently this comparison is deferred to Appendix A.1.
- Provide proof sketches in the main text explaining how Maurey's sparsification lemma is adapted to the KAN setting, since this is the key technical tool.
- Add even a preliminary regularization experiment adding λ·α̃ to the training loss to assess whether the complexity measure is practically useful as a regularizer.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "No dependence on combinatorial parameters is potentially misleading"** — This is technically a valid communication concern, but the bound as stated in Theorem 1 is correct. The claim of "no dependence outside logarithmic factors" on d̃ and p̃ is literally true; the implicit dependence through α̃ is a separate matter that is standard in this class of results. Moved to the Minor weaknesses section in weaker form.

- **Human Finder: "Lack of proof sketches in main text"** — Removed; proof sketches are methodological convention not universal to learning theory papers, and the proofs are provided in the appendix. This is a Nice-to-Have at most.

- **Human Finder: "Missing comparison with MLP generalization bounds in main text"** — Partially removed; Appendix A.1 already provides this comparison. The absence from the main text is at most a Minor presentation preference, noted as a Nice-to-Have.

- **Spark: "No comparison across different basis function choices"** — Removed as scope creep. The theory is explicitly designed for general basis functions; validating every basis type empirically is not necessary to establish the general bound. The current single-basis experiments are adequate for demonstrating the phenomenon.

---

## Novel Insights

The most substantive novel insight across all three reviews is the following synthesis: **the paper's empirical evaluation actually undermines rather than supports its theoretical contribution**. The normalization procedure in Figure 2 is not a mere presentation choice — it collapses the distinction between "the complexity measure predicts the level of the excess loss" and "the complexity measure has the same monotone trajectory shape." The former would be a genuine practical contribution; the latter can be achieved by almost any quantity that monotonically increases with overfitting. If the authors were to show raw bound magnitudes and compare them against the actual generalization gap, readers could assess whether the bound is vacuous or meaningful. This validation gap is the most actionable path to strengthening the paper's impact.

---

## Suggestions

1. **Fix Corollary 2:** Update its assumption statement to reference Assumptions 4 and 5 (not 1, 2, and 4), consistent with Theorem 5 and the RKHS setting.
2. **Report raw/unnormalized complexity values** in Figure 2 alongside excess loss; include a scatter plot with quantified correlation coefficients.
3. **Add empirical evaluation of the RKHS setting** (Theorems 4–5): implement at least one experiment with low-rank activation updates and track ξ₀ against excess loss.
4. **Reframe the theory contribution** more precisely: either (a) specialize the results to the univariate KAN structure of Eq. (1) as the primary claim, or (b) be explicit in abstract/contributions that the results apply to a broader compositional class containing KANs.
5. **Add rate comparison with minimax optimal nonparametric rates** under composition assumptions, analogous to Schmidt-Hieber (2020) for MLPs.

---

## Score and Decision

**Calibration papers considered:**

| Paper | Topic | Decision | Scores |
|-------|-------|----------|--------|
| `ydlDRUuGm9.md` | KAN expressiveness & spectral bias | Accept (Poster) | 6, 6, 8, 5 (avg ~6.25) |
| `Y7lc4aZ4iP.md` | Rademacher complexity bounds for CNNs | Reject | 6, 3, 3 (avg ~4) |
| `oV72wHuRNy.md` | VC dim for low-rank networks | Reject | 6, 3, 3, 3 (avg ~3.75) |
| `kOtFuzoA93.md` | Kernel models & Rademacher bounds for NNs | Withdrawn | 5, 3, 5, 3 (avg ~4) |

**Positioning:** The paper under review has stronger novelty than the rejected generalization-bounds papers (it is genuinely the first KAN-specific analysis), which pushes it above the ~4 average of those rejections. However, it falls short of the accepted KAN expressiveness paper on two key dimensions: (1) empirical support is substantially weaker and relies on methodologically questionable normalization; (2) a potential theorem-statement error in Corollary 2 needs correction. The absent empirical validation of the RKHS contribution (half the theoretical content) is a particularly significant gap.

**Axis assessment:**
- *Originality*: Good — first generalization bounds for KANs, novel RKHS analysis.
- *Importance*: Moderate-to-good — KANs are a timely architecture; theoretical foundations matter.
- *Claims well supported*: Weak — theoretical claims are generally sound; empirical claims are overclaimed.
- *Soundness of experiments*: Poor — normalization renders the central empirical claim inconclusive; no RKHS experiments; single run.
- *Clarity*: Adequate — theory is clearly presented; empirical section could be more transparent.
- *Value to community*: Moderate — the covering-number machinery and unbounded-loss treatment are useful contributions, but the positioning and empirical validation need substantial work.

**Final score: 5.0** — The paper has a genuine and timely theoretical contribution, but the combination of a potential statement error (Corollary 2), substantial empirical overclaiming, and zero empirical support for the RKHS contribution places it below the acceptance bar without revision.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>