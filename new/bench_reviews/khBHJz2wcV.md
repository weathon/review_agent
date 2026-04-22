## Summary

This paper proposes a post-training fine-tuning framework for flow-matching generative models that enforces parameter-dependent PDE constraints via weak-form residuals and adjoint matching. The key innovation is a joint evolution scheme that simultaneously generates physically consistent solutions and infers latent physical parameters (e.g., material coefficients, source terms) without requiring paired training data—addressing a real gap in the physics-constrained generative modeling literature. The method is validated across four PDE families (Darcy, elasticity, Helmholtz, Stokes) and a natural-image demonstration.

## Strengths

- **Joint evolution of solutions and latent parameters without paired training data is genuinely novel.** Existing physics-constrained generative methods assume known parameters; the surrogate base flow construction in Section 3.2 (v^base_{t,α} = (α̂₁ − α_t)/(1−t)) and the joint adjoint-matching formulation provide a principled way to evolve α alongside x when no ground-truth parameter flow exists. The Stokes experiment (Section 4.5, Figure 5) provides the strongest evidence: the joint model achieves MMD_α ≈ 0.07–0.13 while both ablations (Base AM, Base AM+φ) remain at 0.22–0.28, despite similar residual levels—directly demonstrating the joint flow's contribution to parameter inference.

- **Weak-form PDE residuals are a well-motivated technical choice.** Transferring derivatives to test functions via integration by parts avoids the numerical instability of strong-form residuals (Section 3.1), which is a known pitfall of PINNs-style losses. The Helmholtz experiment (Table 2) shows the full joint AM model achieving the lowest weak residuals (4.3×10⁰) and lowest MMD_x (0.06) simultaneously.

- **Favorable constraint–distribution trade-off compared to baselines.** In linear elasticity (Table 1), the method achieves BC error 1.71×10⁻⁶ (vs. FM's 6.98×10⁻⁵) while maintaining MMD_x of 0.15 (vs. PBFM's 0.92), demonstrating a superior Pareto front over prior methods.

- **Honest depiction of trade-offs in the Darcy experiment.** Figure 3 provides the most informative evaluation in the paper: Panel (a) shows increasing λ reduces both residuals and diversity (measured via 1 − SSIM), and Panel (b) shows varying λ_f trades residual reduction for distributional fidelity. This transparency is commendable and practically useful.

- **Computational efficiency.** Fine-tuning requires only 20 gradient steps in under 15 minutes on a single GPU (Section 4.1), after which sampling proceeds at base-model cost with no inference-time adjustments.

- **Comprehensive ablation design.** The paper systematically ablates the joint evolution component (Base AM vs. Base AM+φ vs. full joint AM) across all PDE experiments, isolating the contribution of each design choice.

## Weaknesses

### Fatal
None.

### Major

- **MMD metrics against D_ref partially conflate physics consistency with distribution preservation, undermining the central "without distorting" claim.** The reference set D_ref is "a synthetic, clean dataset generated under the target PDE specification assumed during fine-tuning" (Section 4). Since the base model was trained on data that violates this specification (noisy observations, different BCs, damped physics), any method that shifts samples toward PDE consistency will tend to improve MMD against D_ref regardless of whether it preserves the original training distribution. This makes it difficult to evaluate the paper's most distinctive claim—that it achieves physics consistency *while* preserving distributional fidelity—from MMD alone. The paper partially addresses this through Fig. 3a (SSIM diversity, orthogonal to physics) and Fig. 3b (MMD_x against the *base* dataset), but these analyses appear only in the Darcy section. The abstract's claim of "without distorting the underlying learned distribution" is overstated: the Darcy experiment itself shows diversity decreases with λ (Fig. 3a), and the conclusion's softer "without significantly affecting the sample diversity" is more defensible but still not rigorously established across all experiments. The paper would be strengthened by reporting distributional metrics against the *original training distribution* or consistently using diversity measures (as in Fig. 3a) across all PDE experiments.

- **The inverse predictor φ is never validated against ground-truth parameters, leaving the "accurate recovery of latent coefficients" claim unsupported by direct evidence.** φ is trained by minimizing PDE residuals on base-model samples (Section 4: "we first sample from the base generator and pre-train the inverse predictor φ to recover α by minimizing the (PDE) residual"). Since base-model samples violate the PDE, φ is learning a compromise rather than recovering true α. φ's predictions then define the surrogate base flow (Section 3.2), the regularization direction v^reg, and the MMD_α metric. Yet φ's predictions are never compared against ground-truth α values. While MMD_α against D_ref provides distributional evidence, it does not measure point-wise prediction accuracy. If ground-truth α is available for D_ref (which it should be, since D_ref is synthetically generated), reporting prediction MSE or correlation between φ(x) and true α would substantially strengthen the inverse-problem claims.

### Minor

- **FM+ECI anomaly in Table 1 is acknowledged but unexplained.** FM+ECI achieves BC error = 0 (exact boundary satisfaction, by construction) but relative strong residual of 1.01×10³—the highest of any method by two orders of magnitude. The paper mentions that PBFM and FM+ECI "present high residuals" but does not explain why exact BC enforcement leads to catastrophically bad interior solutions. Understanding this failure mode would help readers assess when hard-constraint methods are inappropriate.

- **PBFM failure in Stokes experiment is reported without explanation.** The paper states "PBFM fails to converge to meaningful velocity–pressure fields" (Section 4.5) but provides no analysis of why. Understanding why a training-time physics-constrained method fails while post-training fine-tuning succeeds would strengthen the paper's narrative.

- **The surrogate base flow for α is a heuristic without theoretical justification.** The direction v^base_{t,α} = (α̂₁ − α_t)/(1−t) is a linear interpolation toward φ's prediction, with no analysis of how errors in φ propagate through the joint evolution or how this compares to alternative base-flow constructions. This is a reasonable design choice that works empirically, but a discussion of its limitations would be valuable.

- **The natural-image experiment (Section 4.6) does not validate PDE-constrained generation.** The "α" is a color transformation and the "constraint" is PickScore preference optimization. While the paper frames this as "cross-domain utility," the experiment demonstrates adjoint matching with a preference reward—a use case already covered by Domingo-Enrich et al. (2025)—rather than validating any PDE-specific claim. The framing could be more precise.

### Trivial
None.

## Nice-to-Haves

- Report ground-truth parameter accuracy (MSE or correlation between φ(x) and true α) for at least one PDE where ground-truth α is available. This single addition would substantially strengthen the inverse-problem claims.
- Report distributional metrics against the original training distribution (not just D_ref) or include SSIM diversity measures consistently across all experiments, to properly isolate distribution preservation from physics consistency.
- Ablation on the number and characteristics of test functions (N_test, polynomial order, length-scale) for the weak-form residual, since the residual drives the entire fine-tuning.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Equation 3 block-Jacobian formatting (all entries showing J_xx)**: This is a parser artifact; the original submission would display the correct block structure (J_xx, J_xα, J_αx, J_αα). Removed per formatting-artifact rule.
- **Scaled noise schedule memoryless property (Lemma 1 in Appendix)**: The harsh critic questions whether scaling preserves the memoryless property and says they cannot verify the appendix proof. The paper cites a specific lemma in the appendix; the proof exists but the parser strips appendices. Removed per missing-appendix rule.
- **Guidance mechanism deferred to Appendix E.4**: The harsh critic says this makes it "hard to assess." This is a standard practice of deferring implementation details to the appendix. Removed per missing-appendix rule.
- **Reproducibility concerns about test function details**: The harsh critic flags that N_test and test function characteristics are in Appendix D.3. These details are documented and the paper references their location. Removed per reproducibility-nitpick rule.
- **Missing references**: The PBFM reviewer mentions the paper itself as a missing reference for the PBFM paper, but per rules we do not flag missing related works.

## Novel Insights

The most interesting tension in this paper is that its strongest experimental evidence (Darcy, Fig. 3) actually undermines its most distinctive abstract claim ("without distorting the underlying learned distribution"). The Darcy ablation honestly reveals a controllable trade-off rather than a free lunch, which is more scientifically valuable than the overclaimed abstract. This suggests the paper's true contribution is not "physics without distortion" but rather "a principled, tunable knob for navigating the physics–diversity frontier"—a contribution that is genuine and practically important, but framed too strongly in the abstract. The Stokes experiment's demonstration that the joint flow is essential for parameter-distribution fidelity (even when residuals are similar across ablations) is the paper's most compelling and underappreciated result.

## Suggestions

- Re-frame the abstract and conclusion to reflect the genuine trade-off: replace "without distorting" with "with controllable distributional shift" or "while limiting distributional distortion," matching what the experiments actually show.
- Add a single table or figure reporting φ prediction error against ground-truth α for at least one PDE where α is known (e.g., Darcy, where α is drawn from a known Gaussian process). This would directly validate the inverse-problem claims.
- Include SSIM diversity metrics (as in Fig. 3a) across all PDE experiments, not just Darcy, to consistently evaluate distribution preservation independent of physics consistency.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Relation to paper under review |
|-------|-----------|-------------------------------|
| PBFM (tAf1KI3d4X) | 5.5 | Most direct comparator: physics-constrained flow matching, accepted as poster. This paper addresses a harder problem (parameter-dependent constraints, inverse problems) but has similar evaluation gaps (overclaimed trade-off, missing analysis). |
| FALCON (FbssShlI4N) | 7.0 | Flow-matching with strong theoretical grounding and clean evaluation. Clearly above this paper, which lacks comparable theoretical rigor and has evaluation confounds. |
| Flow Marching (nnRB90w2kv) | 2.5 | Generative PDE foundation model, rejected for incomplete benchmarking and unclear methodology. This paper is clearly above—more focused, better experiments, cleaner framing. |
| FourierFlow (a3sRspQ62b) | 2.5 | Flow matching for turbulence, rejected for poor methodology. This paper is substantially stronger. |
| Adjoint Matching/FOCUS (qwS1bEqdrS) | 3.5 | Original adjoint matching paper, withdrawn. This paper extends that framework with a novel joint evolution scheme, which is a genuine contribution. |
| RealPDEBench (y3oHMcoItR) | 7.5 | PDE benchmark with strong evaluation design. Far above this paper in evaluation rigor. |
| Physics-Informed Distillation (hW7P3x9W8A) | 4.0 | Similar domain (PDE-constrained diffusion), withdrawn. This paper has stronger empirical results. |

This paper sits above the low-scoring anchors (2.5) by a wide margin—it has a genuine novel contribution and solid experimental design. It sits below the high-scoring anchors (7.0+) due to evaluation confounds and overclaimed scope. Relative to PBFM (5.5), the most direct comparator, this paper addresses a harder and more novel problem (joint parameter inference without paired data) but has comparable evaluation weaknesses (overclaimed trade-off, missing analysis of baseline failures). The core idea is strong enough for acceptance, but the evaluation gaps prevent a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>