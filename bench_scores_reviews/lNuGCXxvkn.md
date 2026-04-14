## Summary

This paper develops an asymptotic Sobolev-norm learning curve for kernel ridge and ridgeless regression applied to elliptic linear inverse problems governed by PDEs. The central theoretical finding is that the PDE forward operator — by amplifying high-frequency components — effectively stabilizes the variance of min-norm interpolators, enabling benign overfitting in fixed spatial dimensions where standard regression would produce tempered or catastrophic overfitting. A secondary contribution is characterizing how the choice of Sobolev-norm inductive bias (parameter β) affects convergence, establishing a smoothness threshold above which the rate becomes independent of the specific inductive bias, and showing this threshold matches one previously identified in the Bayesian inverse-problem literature.

---

## Strengths

- **Fixed-dimensional benign overfitting via PDE structure.** Theorem 4.2 and Remark 7 constitute a genuine and specific finding: the negative exponent p of the differential operator shifts the variance bound exponent from `max{λβ', −1}` (pure regression) to `max{2p + λβ', −1}` (inverse problem), and since p < 0, this can push variance below the regression baseline even without dimensional growth or kernel engineering. This mechanism — the inverse problem operator acting as a spectral smoother — is clearly articulated through the spectral transformation Σ̃ = A²Σ^β and gives a principled reason for variance stabilization that is new relative to prior kernel-interpolation analyses.

- **Unified regularized + ridgeless framework recovering known rates.** The paper analyzes both ridge-regularized (Theorem 4.1) and min-norm interpolating (Theorem 4.2) estimators in a single spectral framework. Critically, Remark 5 shows the regularized bound reproduces the minimax-optimal rate from Lu et al. (2022) at the optimal γ, providing a meaningful sanity check that the framework is correctly calibrated. Simultaneously extending to interpolators in the same setting, and establishing where the dominant terms depend vs. do not depend on β, is a non-trivial analytical step.

- **Smoothness threshold matching the Bayesian literature.** The finding that the threshold λβ ≥ λr/2 − p — above which the convergence rate becomes independent of the inductive bias — coincides with the analogous condition identified in Bayesian inverse problems (Knapik et al., 2011; Szabó et al., 2013) and with empirical understanding in semi-supervised learning is a surprising and useful connection. It elevates the result from an isolated technical bound to a structurally motivated condition. Extending this threshold to the ridgeless/interpolating regime is new.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Bounded-observation assumption inconsistent with Gaussian noise model.** Assumption 2.2(a) stipulates that observations y are almost surely bounded by M, yet Section 3 explicitly sets ε ∼ N(0, σ²I_{n×n}), making y unbounded almost surely. If proofs use boundedness to invoke standard concentration inequalities, the Gaussian model is not formally covered by the stated assumptions. The paper needs to either relax Assumption 2.2(a) to sub-Gaussian or finite-variance noise, or clarify that the Gaussian model satisfies the technical conditions actually used in the proofs (rather than the stated ones).

- **Critical dependence of benign-overfitting claims on ρ_{k,n} is underemphasized.** The headline claim — benign overfitting in fixed dimension — is established under Theorem 4.2, but both the variance bound (scaled by ρ²_{k,n}) and bias bound (scaled by ρ³_{k,n}) critically depend on the concentration coefficient. Remark 6 acknowledges that ρ_{k,n} = Θ(1) requires sub-Gaussian features, and in the worst case can grow as Õ(n^{2p+βλ−1}), which can substantially weaken or even eliminate the benign-overfitting conclusion. The abstract and main body consistently present benign overfitting as a consequence of the PDE structure, without foregrounding that this requires a separate, non-trivial assumption on the feature behavior. The paper should state clearly, in the main theorems, under exactly which feature conditions the benign-overfitting exponents hold, and what happens in the worst case for ρ_{k,n}.

- **Experiments are too limited and indirect to substantiate the theoretical claims.** All experiments are conducted on a single 2D Poisson equation with one ground-truth function, using finite-width neural networks — not the kernel estimators the theory covers. There are no kernel experiments, no systematic variation of PDE order p (which is the central determinant of variance stabilization), no variation of the inductive bias parameter β directly (activation smoothness is an indirect proxy), and no comparison of regularized vs. interpolating estimators under controlled conditions. For a theory paper at ICLR, this leaves the theory entirely unvalidated in its own setting and makes Figure 1(Left)/(Middle) illustrative rather than confirmatory. At minimum, a controlled synthetic kernel experiment — e.g., Matérn kernel + Laplacian operator with known spectral decay — varying n at different values of p and β to check the predicted exponents, would substantially strengthen the paper.

- **No lower bounds; benign vs. tempered regime is one-sided.** All results are upper bounds. "Benign overfitting" in the strict sense requires that risk vanishes, but without matching lower bounds the upper bounds may be loose. It is not possible from the present results to determine whether the benign/tempered/catastrophic trichotomy is tight or an artifact of proof looseness, especially given the max{·,−1} and max{·,−2p+λ(β'−2β)} exponents in Theorem 4.2. The paper should acknowledge this limitation, or present even partial lower bounds for the variance in the inverse-problem interpolation setting.

### Minor

- **Theory-to-experiment gap is not adequately bridged.** The experiments use overparameterized finite-width neural networks, and the connection to kernel regression is justified only informally via NTK heuristics. The paper frames Figure 1 as validating theory "beyond kernel methods" but this is an overstatement; the NTK approximation is not verified to hold in this inverse-problem setting, and the activation function smoothness is at best a proxy for the spectral decay parameter β. Section 5 should explicitly label these results as heuristic evidence rather than validation of the stated theorems.

- **Diagonalizability assumption (Assumption 2.2(d)) limits scope more than framing suggests.** The requirement that A and Σ be simultaneously diagonalizable is strong and excludes most practical geometries beyond the torus. While this is standard in theoretical kernel-based inverse-problem analysis (acknowledged in Remark 2), the paper's broad framing around "physics-informed machine learning" and PINNs is at odds with this restriction. A short discussion of what happens qualitatively when this assumption fails, or under which conditions it approximately holds, would calibrate the reader's expectations.

- **Practitioner takeaway about activation smoothness is heuristic, not theorem-derived.** Section 4.3 states that higher-order PDEs "require smoother activation functions," presenting this as a consequence of the theory. However, the formal results are for kernel estimators with prescribed spectral decay; the link between activation smoothness and the β parameter for finite neural networks is not established analytically. This guidance should be explicitly labeled as conjectural.

### Tiny

- The abstract says "the convergence rate is actually independent to the choice of (smooth enough) inductive bias" without the qualifier that this independence holds for β above a threshold and is subject to the ρ_{k,n} caveat. The abstract should reflect these conditions.
- No dedicated limitations section; the key caveats (diagonalizability, kernel-only theory, ρ_{k,n} dependence) are scattered across remarks rather than consolidated.

---

## Nice-to-Haves

- **Spectral visualization of the transformed kernel eigenspectrum.** A plot of Ã² Σ^β eigenvalues vs. standard kernel Σ eigenvalues for a concrete example (e.g., Matérn + Laplacian) would make the variance-stabilization mechanism intuitive and help readers assess whether the concentration assumptions hold in practice.
- **Bias-variance decomposition plots.** Plotting bias and variance separately vs. n for both PINN and standard NN interpolators would directly validate the claim that the PDE operator specifically suppresses variance, rather than improving the combined risk through other mechanisms.
- **Explicit benign vs. tempered parameter-regime table.** A table stating: for these ranges of (p, λ, r, β, β'), the variance exponent is negative (benign); for these ranges it is only bounded (tempered); for these it diverges (catastrophic) — would make Theorem 4.2 much more accessible and clarify the practical scope of the main result.
- **Characterization of ρ_{k,n} for physics-informed kernels.** Showing that for, e.g., Matérn kernel + Laplacian on the torus the sub-Gaussian feature condition holds, and thus ρ_{k,n} = Θ(1), would close the gap between the theoretical claim and the setting for which it is actually established.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they either misread the paper or are overly pedantic.*

- **Threshold inconsistency between Section 1.1 and Remark 5.** The harsh critic flags that Section 1.1 writes `λβ ≥ λ^r/λ^p − p` while Remark 5 writes `λβ ≥ λr/2 − p`. The former is almost certainly a PDF-to-text parsing artifact of `\frac{\lambda r}{2}` rendering as `λ^r/λ^p` (which would equal λ^{r−p}, a nonsensical expression in context). The Remark 5 expression is the coherent one and appears consistently in the applications. This is not a scientific inconsistency.

- **Novelty is "just a reparameterization."** The critic suggests the results might follow by a black-box application of Barzilai & Shamir (2023) to the transformed kernel K̃. However, the extension to the inverse-problem operator setting, the derivation of the closed-form representer theorem under the Sobolev norm with operator A, the introduction of the concentration coefficient for the transformed kernel Σ̃ = A²Σ^β, and the derivation of threshold conditions involving p are genuine technical contributions, not trivial substitutions.

- **Concern about φ and ψ basis confusion.** The critic raises notation inconsistency between L₂ and RKHS inner products. The paper defines φ_i = √λ_i ψ_i as the RKHS basis and ψ_i as the L₂ eigenbasis, and introduces ψ-maps and φ-maps explicitly with their relationships stated. The notation is dense but internally consistent given careful reading; this is not a mathematical error.

- **Claim that experiments should include confidence intervals/error bars.** For neural-network experiments demonstrating qualitative phenomena (noise profiles, convergence trend with activation smoothness), single-run plots are acceptable; demanding multiple-run statistics here would be an atypical rigor requirement for this type of demonstration figure.

- **Criticism of the "first rigorous upper bound" claim.** The paper scopes this claim specifically to min-norm kernel interpolators for fixed-dimensional physics-informed settings. Within that narrow scope, the claim is plausible; the critic's objection is largely about lack of a comprehensive literature survey, which does not mean the claim is false.

---

## Novel Insights

The most significant insight synthesized from the reviews goes beyond the paper's own framing: the paper implicitly establishes a *spectral duality* between the forward PDE operator and the inductive bias in the interpolation regime. The forward operator A with p < 0 amplifies high-frequency components in the forward direction, but this amplification means the inverse-problem objective *penalizes* high-frequency errors more heavily, effectively acting as spectral regularization without any explicit regularizer. This is structurally dual to adding Sobolev norm regularization: increasing |p| has the same qualitative effect on variance as increasing β. The smoothness threshold condition λβ ≥ λr/2 − p makes this duality precise — p and β enter symmetrically in the admissibility condition. This perspective suggests that for practitioners, the choice of activation smoothness (proxy for β) and the PDE order (p) should be co-designed, and that for sufficiently high-order PDEs, relatively weak inductive bias may suffice for benign behavior. The connection to the Bayesian threshold suggests this duality may be fundamental rather than an artifact of the proof technique.

---

## Suggestions

1. **Add at least one synthetic kernel experiment** on a 1D or 2D Poisson / Schrödinger problem using the actual kernel estimator (Matérn or RBF kernel + discretized Laplacian), varying n at multiple values of p and β, and plotting empirical excess risk against the theoretically predicted exponents. This is the single highest-value addition to strengthen the paper.

2. **Resolve the bounded-y / Gaussian-noise inconsistency** in Assumption 2.2(a). Either extend the assumption to sub-Gaussian/finite-variance noise and verify the downstream inequalities hold, or add a footnote in Section 3 explaining that the proof uses only finite-variance properties and Gaussianity is for concreteness.

3. **Make Theorem 4.2 self-contained with a regime table.** Add after Theorem 4.2 a corollary or remark that explicitly lists the parameter conditions under which (i) V → 0 and B → 0 (benign), (ii) V bounded but nonzero (tempered), and (iii) V → ∞ (catastrophic), so readers can immediately identify which operating regime applies to their problem.

4. **Characterize ρ_{k,n} for at least one concrete kernel-operator pair.** Show (in the appendix) that the shift-invariant Matérn kernel on the torus with the Laplacian satisfies the sub-Gaussian feature condition, so ρ_{k,n} = Θ(1) in that case, providing a complete end-to-end statement of benign overfitting.

5. **Separate heuristic from rigorous in Section 4.3.** Label the practitioner guidance on activation smoothness and higher-order PDEs explicitly as "conjectural extension" and reference the NTK connection, to avoid overstating what is formally established.

6. **Add a limitations paragraph** consolidating: (i) joint diagonalizability of A and Σ; (ii) kernel theory vs. NN experiments gap; (iii) ρ_{k,n} assumption; (iv) linear self-adjoint elliptic operators only.