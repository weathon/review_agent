Now I have sufficient calibration data. Let me compile my final review.

## Summary

This paper provides the first generalization bounds for Kolmogorov–Arnold Networks (KANs) under two settings: (1) activation functions represented as linear combinations of basis functions (Theorems 1–3, using covering number arguments), and (2) activation functions lying in a low-rank RKHS (Theorems 4–5). The basis-function bounds scale with ℓ₁ norms of coefficient matrices and Lipschitz constants, with no explicit dependence on combinatorial parameters (width, basis count) outside logarithmic factors. The unbounded-loss extension via truncation broadens applicability beyond classification. Empirical evaluation shows the complexity measure tracking excess loss during SGD training across six settings.

## Strengths

- **First generalization bounds for KANs**: This is a timely and genuinely useful contribution. KANs are a rapidly growing architecture class, and providing the first rigorous complexity and generalization analysis fills an important gap in the literature. Theorems 3 and 4 are the main deliverables.

- **Low-rank RKHS analysis is novel**: Theorems 4–5, which bound generalization when activation functions belong to a low-rank RKHS, appear genuinely new. As the authors note in Section 1.2, comparable results for MLPs are not available. Remark 6's connection to LoRA-style fine-tuning is an interesting practical implication.

- **Extension to unbounded losses**: Unlike Bartlett et al. (2017), which requires bounded loss (ramp loss for margin classification), Theorem 3 handles unbounded regression-type losses (squared, pinball, Huber) via a truncation argument, broadening the framework's applicability.

- **Accommodates diverse basis function choices**: The framework in Assumption 2 covers B-splines, wavelets, RBFs, Fourier series, and polynomial bases, with Remark 3 specifically quantifying Lipschitz constants for B-splines.

- **Norm-adaptive covering number construction**: Proposition 1's iterative construction works with general norms at each layer, extending the norm-adaptive approach of Bartlett et al. (2017) to the KAN setting.

## Weaknesses

### Fatal
None.

### Major

- **Empirical validation has significant methodological issues**: Figure 2 normalizes the complexity measure so that its maximum equals the last value of the excess loss (Section 3: "we normalize the values of the complexity measures so that the maximum value of the complexity measure is equal to the last value of the excess loss"). This scale-fitting procedure artificially creates visual alignment. The paper then claims "tightly correlates" and "closely follows the shape of the excess loss," but reports no correlation coefficient, statistical test, or comparison to baseline complexity measures (e.g., parameter count, Frobenius norm). Since the complexity measure is an upper bound-derived quantity that tends to decrease during training (as SGD typically shrinks norms), any roughly monotonically-decreasing quantity would look visually similar after this normalization. This is the paper's only empirical argument for "practical relevance," and it is not convincingly established. The claim requires (a) reporting actual Pearson/Spearman correlations, (b) comparing against trivial baselines, and (c) at minimum reporting raw values or independent scaling.

- **Basis-function theoretical results are primarily adaptation, not new insight**: The covering number analysis for the basis-function case (Propositions 1–2 → Theorems 1–3) directly adapts the Bartlett et al. (2017) / Anthony et al. (1999) framework. When KAN activation functions are linear combinations of basis functions (Assumption 2, eq. 5), the layer mapping Ψ(x) = A·g(x) is structurally equivalent to a fixed-nonlinear-layer followed by a linear layer—precisely the setting where existing MLP tools apply. The paper acknowledges this lineage (Section 1.2) but does not demonstrate that the KAN structure yields qualitatively different generalization behavior or tighter bounds than an equivalent MLP analysis. The five listed contributions (Section 1.1, items i–v) are features of the proof technique rather than insights about KANs specifically. For a learning theory paper, the absence of any result showing a qualitative difference between KANs and MLPs in terms of generalization is a significant gap. The low-rank RKHS analysis partially compensates for this, as it is genuinely novel.

### Minor

- **"No dependence on combinatorial parameters" claim can be misleading in practice**: The abstract and Section 1.1 state the bound "has no dependence on combinatorial parameters (e.g., number of nodes) outside of logarithmic factors." While technically correct about the bound's *form*, B_l (the ℓ₁ norm of the coefficient matrix at layer l) inherently scales with d_l × p_l when individual coefficients are bounded below, and ρ_l as estimated via Remark 5 scales as ‖A‖_σ · c_l · √(b_l), introducing b_l explicitly. For any trained network with bounded per-parameter magnitude, the bound grows polynomially with width—the same situation as Bartlett et al. (2017) for MLPs. The paper should acknowledge this caveat explicitly rather than presenting it as a distinguishing feature.

- **Corollary 2 references wrong assumptions**: Corollary 2 (line 275) states "Suppose Assumptions 1, 2 and 4 hold" but is derived from Theorem 5 in the low-rank RKHS section, which requires Assumptions 4 and 5. This appears to be a copy-paste error from Corollary 1, which correctly references Assumptions 1, 2, and 4 for the basis-function setting.

- **Lipschitzness of Ψ_l can be restrictive for certain basis functions**: Assumption 2 requires globally Lipschitz activation functions. For B-splines (the most common KAN basis), this is reasonable and the paper discusses it in Remark 3. However, for polynomial or spline basis functions on unbounded domains, the Lipschitz constant can be very large or infinite, making the bound vacuous. The paper does not discuss when this assumption is reasonable beyond B-splines.

- **Conditions for ζ₀ to yield a decreasing bound are not discussed**: In Theorem 3, ζ₀ = α̃³ log(2d̃p̃)(nC″/τ)^{2/s′} where s′ > 0. The (nC″/τ)^{2/s′} term grows with n. For the overall bound to decrease with n, we need s′ > 1 (so that the complexity term √ζ₀/n ∼ n^{1/s′−1} decreases). For 0 < s′ ≤ 1, the bound would not decrease, which is counterintuitive. The paper should discuss the conditions under which the bound is meaningful.

- **Loose upper bounds for ρ_j in experiments**: The Lipschitz constants ρ_j are estimated using the upper bounds from Remark 5 (ρ* ≤ ‖A‖_σ · c_l · √(b_l)), which can be quite loose. The paper does not report actual values or assess tightness, making it difficult to evaluate whether the bounds are informative in practice.

### Trivial
None significant.

## Nice-to-Haves

- Compare KAN bounds to analogous MLP bounds on the same tasks to establish whether KAN generalization is qualitatively different
- Derive a lower bound or prove tightness of the upper bounds (acknowledged as future work in Section 4, item ii)
- Show raw (unnormalized) complexity and excess loss curves side by side to reveal whether the complexity measure tracks magnitude, not just trend direction
- Discuss more explicitly when Assumption 2 (global Lipschitzness) is reasonable for basis functions beyond B-splines

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that paper "cannot be independently verified" or that models/tools are unreleased**: Removed per hard rules; if the paper cites it, it exists.
- **Demand for missing appendix or proofs in appendix**: Removed per hard rules; the parser strips appendix sections, which exist in the original submission.
- **Formatting/style nitpicks and typos**: Removed per hard rules; these are parser artifacts.
- **Demand for reproducibility details like undisclosed hyperparameters or implementation details**: Removed per hard rules; these are impractical to include in a submission.
- **Critic's claim that "any monotonically-decreasing-during-training quantity would look similar after normalization"**: Partially removed; this is an overstatement. The normalization makes the *scale* match, but the *shape* still needs to align—any monotonically decreasing quantity would match in trend but not necessarily in shape. The valid core of this point (lack of statistical metrics, no baselines, normalization creates visual alignment) is retained in the Major weakness above.
- **Strength Finder's claim about "strong empirical correlation" from Figure 2**: Removed as a strength because it conflicts with the verified Major weakness about empirical methodology. The correlation claim is undermined by the normalization and lack of statistical analysis.

## Novel Insights

The paper reveals an interesting structural analogy: when KAN activations are linear combinations of basis functions (the dominant practical parameterization), the KAN layer Ψ(x) = A·g(x) is mathematically equivalent to a fixed-feature layer followed by a linear map. This means that the generalization behavior of KANs under this parameterization may not be qualitatively different from MLPs—an insight the paper does not explicitly draw but which emerges from the analysis. The genuinely distinguishing contribution is the low-rank RKHS analysis (Theorems 4–5), which exploits structure unique to KANs (activations as functions in a low-rank function space) and has no MLP counterpart, suggesting that KANs' generalization advantage, if any, may lie in the functional-space structure of their activations rather than the basis-function parameterization.

## Suggestions

- Replace the normalized complexity plots in Figure 2 with (a) raw complexity and excess loss curves on separate axes or dual y-axes, and (b) report Pearson/Spearman correlation coefficients along with comparisons to trivial baselines (parameter count, spectral norm). This would turn the current suggestive-but-unconvincing figure into strong evidence.
- Add a brief discussion in Section 2.2 explicitly comparing the KAN bound form with the Bartlett et al. (2017) MLP bound under analogous assumptions, identifying whether any structural difference exists in the rate or norm dependencies.
- Correct the assumption references in Corollary 2 from "Assumptions 1, 2 and 4" to "Assumptions 4 and 5."

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| "How many samples to train a DNN?" | q6zrZbth1F.md | 7.0 | First lower bounds for deep nonlinear networks—more novel than adapting covering numbers. This paper is below it. |
| "Path-norm toolkit for modern networks" | hiHZVUIYik.md | 7.33 | New toolkit with sharp bounds evaluated numerically—more technical novelty. This paper is below it. |
| "Fantastic Generalization Measures Nowhere to Be Found" | NkmJotfL42.md | 7.0 | Conceptually impactful impossibility result—far more transformative. This paper is below it. |
| "Compute-Optimal LLMs Provably Generalize Better" | MF7ljU8xcf.md | 6.0 | Novel empirical Freedman inequality + timely application; has novel technical tool. This paper is comparable or slightly below it (less novel technique, weaker empirics). |
| "Generalizability based on Expressive Power" | 8wAL9ywQNB.md | 6.6 | New angle (expressive power) on generalization, but restricted to two-layer nets. This paper is slightly below it (similar incremental issues, also restricted settings). |
| "KAAN" | 3VOKrLao5g.md | 4.25 | Architecture paper seen as incremental. This paper is clearly above it—has substantial theoretical substance. |
| "Weight Decay induces low-rank bias" | 3zw9NhLhBM.md | 2.2 | Trivial theoretical contribution, unrealistic assumptions. This paper is well above it. |
| "Generalization Bounds for Neural ODEs" | B8qoU7kgSF.md | 3.0 | Largely unoriginal bounds, loose, no comparison to prior work. This paper is above it—more careful and complete. |
| "Low-Dimensional Representation and Generalization" | A9yKCUQNnc.md | 3.0 | Naive concentration bounds, no comparison. This paper is above it. |
| "Slicing MI Generalization Bounds" | Piod76RSrx.md | 5.5 | Incremental theoretical improvement, borderline. This paper is comparable. |

The paper sits in the 5.0–5.5 range. It is above the clearly weak theoretical papers (2–3 range) and above purely incremental architecture papers (~4), but below the genuinely novel learning theory contributions (7+). It is comparable to borderline papers in the 5.5 range: real contributions (first KAN bounds, novel low-rank RKHS analysis) but with significant limitations (incremental basis-function analysis, flawed empirical methodology).

## Score and Decision

**Originality**: The basis-function analysis adapts existing MLP techniques; the low-rank RKHS analysis is genuinely novel. The overall originality is moderate—valuable as a first analysis of a new architecture, but not methodologically transformative.

**Importance of research question**: KANs are a timely and increasingly important architecture; understanding their generalization properties is a significant research question.

**Claims well supported**: The theoretical claims are rigorous and correctly derived. The "practical relevance" claim is not well supported due to methodological issues in the empirical validation.

**Soundness of experiments**: The experiments have significant methodological flaws (normalization forcing visual alignment, no correlation metrics, no baselines).

**Clarity**: The paper is well-written with transparent proof strategies and clear acknowledgment of its debt to prior work.

**Value to community**: Provides a foundational reference for KAN generalization theory; the low-rank RKHS connection to LoRA-style fine-tuning is practically relevant.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>