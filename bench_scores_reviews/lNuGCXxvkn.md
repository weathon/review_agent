## Summary

This paper develops a theoretical framework for kernel ridge and ridgeless regression applied to linear inverse problems governed by elliptic PDEs. The central contribution is showing that the PDE operator (with spectral decay exponent $p < 0$) inherently stabilizes variance, enabling benign overfitting even in **fixed input dimension**, a setting where standard regression yields only tempered or catastrophic overfitting. The authors additionally characterize how inductive bias from Kernel Sobolev Space (KSS) norms affects convergence rates, showing that sufficiently smooth inductive bias renders the rate independent of the specific smoothness parameter $\beta$, with a threshold that surprisingly matches conditions derived in the Bayesian inverse problem literature.

---

## Strengths

- **Benign overfitting in fixed dimension via PDE structure**: The central insight—that the PDE operator acts as a spectral smoother ($p < 0$ attenuates high-frequency variance), enabling benign overfitting where standard regression cannot—is conceptually clean and mechanistically specific. This is not a recycled result; it is a qualitative departure from existing benign overfitting theory, which typically requires high-dimensional or data-dimension-growing settings.

- **Unified non-asymptotic framework spanning both estimator types**: The same bias-variance machinery (Theorems 3.6–3.7) handles both regularized and interpolating estimators. For the regularized case, the paper recovers the minimax optimal rate from Lu et al. (2022), which provides a non-trivial sanity check on tightness; for the interpolating case, it yields the first rigorous upper bound in this inverse-problem setting, covering benign, tempered, and catastrophic regimes.

- **Surprising connection to Bayesian smoothness threshold**: The smoothness threshold on $\beta$ derived from the minimax rate analysis ($\lambda\beta \geq \frac{\lambda r}{2} - p$) matches the condition derived in the Bayesian inverse problem literature (Knapik et al., 2011; Szabó et al., 2013). This cross-paradigm correspondence is a genuinely non-obvious finding that anchors the frequentist result in established statistical theory.

- **Clear operator-theoretic mechanism**: The noise stabilization effect through $\Delta^{-1}$ (the Green's function acts as a smoothing kernel suppressing high-frequency error) is explained clearly in §5, connecting the abstract spectral analysis to an interpretable physical mechanism.

---

## Weaknesses

### Fatal
None.

### Major

- **Benign overfitting is conditional on an assumption not in the main theorem body.** Theorem 4.2's headline claim—that physics-informed interpolation achieves benign overfitting—depends critically on $\rho_{k,n}$ being bounded. This requires sub-Gaussian features, a condition stated only in Remark 6, not in the theorem hypotheses. In the worst case (noted in the same remark), $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$, which can entirely cancel the variance stabilization effect. The paper does not characterize *when* sub-Gaussian behavior vs. the worst case applies for the spectrally transformed kernel $\tilde{K}$, nor does it provide a corollary with explicit sufficient conditions guaranteeing risk convergence to zero. For a paper whose headline is "benign overfitting in fixed dimension," leaving the central conclusion dependent on an uncontrolled quantity is a significant gap. The paper should provide a standalone corollary making the conditions for benign overfitting fully explicit, or characterize precisely when $\rho_{k,n} = \Theta(1)$ holds for the PDE-transformed kernel.

- **Co-diagonalization assumption is strong and insufficiently analyzed.** Assumption 2.2(d)—that $\mathcal{A}$ and the kernel covariance operator $\Sigma$ share the same eigenbasis—is the linchpin of all spectral analysis. Remark 2 acknowledges this and cites the torus + shift-invariant kernel case (justified by Bochner's theorem). However, the paper does not discuss what happens when this alignment breaks down even approximately: since the variance stabilization mechanism ($p < 0$ in the transformed spectrum $\tilde{\Sigma} = \mathcal{A}^2\Sigma^\beta$) relies entirely on the product of the two spectra being well-ordered, any significant misalignment could destroy the effect. Most practical PDE domains (e.g., irregular geometries, non-uniform sampling, Dirichlet boundary conditions on non-toroidal domains) do not satisfy this assumption. The contribution's practical reach is much narrower than the paper's framing implies without this discussion.

- **Experimental scope is insufficient for the paper's claims.** All experiments use a single PDE (2D Poisson equation, $p = -1$), one domain, and one ground truth function. The paper's central claim involves the parameter $p$, yet $p$ is never varied empirically. Similarly, no experiment directly tests the spectral parameter $\beta$ in the kernel setting, and no log-log convergence rate plot is provided to verify whether the predicted rates ($n^{-\lambda(\beta'-r)/(2p+\lambda\beta+1)}$) are reflected in practice. The activation-smoothness experiment is only a coarse proxy for $\beta$ and validates qualitative monotonicity, not the quantitative rate. At minimum, the paper needs: (1) at least one direct kernel-level experiment validating the bounds in their native setting (the entire theory is for kernel estimators, not neural networks), and (2) experiments varying the PDE order $p$ to support the mechanism claim.

- **Theory-to-experiment gap: kernel theory validated only by neural network experiments.** The experiments use PINNs (neural networks), while the theorems cover kernel estimators. The paper frames this as validating findings "beyond kernel methods," but without establishing an NTK-type equivalence or providing any kernel experiment, the experiments do not validate the theoretical guarantees themselves. This is a structural problem: a reviewer cannot assess whether the theory is tight from these experiments.

### Minor

- **The role of $\rho_{k,n}$ in worst-case behavior is underdeveloped.** Even granting Remark 6, the paper does not show that the worst case $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$ is avoidable in practice, nor does it identify structural properties of $\tilde{K}$ that preclude it. Without this, Theorem 4.2 cannot be used as a reliable guide to when benign overfitting holds.

- **No matching lower bounds for the interpolating regime.** Without lower bounds for the min-norm interpolator, it is unknown whether the variance bound in Theorem 4.2 is sharp or a loose upper bound. For the regularized case, the match with Lu et al. (2022) provides external validation of tightness; for the interpolating case, no such validation exists.

- **Notation is very dense and some operator relationships are underexplained.** The paper introduces many simultaneous operators ($S, S^*, \hat{S}_n, \Sigma, \mathcal{L}, \phi, \psi, \Lambda_{\mathcal{XY}}, \tilde{\Sigma}, \tilde{K}$) without a compact summary of their relationships. The relationship between $\phi$ (mapping to $\ell_2^\infty$ using $\sqrt{\lambda_i}$) and $\psi$ (without the eigenvalue scaling) is confusing because both map from $\mathcal{H}$ to sequences but serve different roles in the operator expressions. A diagram or summary table of operators in an appendix would substantially aid comprehension of Theorems 3.6–3.7.

- **The practical guidance on activation smoothness is heuristic.** §4.3 draws a prescription that "higher-order PDEs require smoother activation functions." While intuitive, the bridge from KSS theory (infinite-dimensional kernel) to finite-width PINN activations is not established theoretically—it requires an NTK-type argument that is not provided. This should be explicitly labeled as a heuristic/empirical conjecture rather than a theorem-derived prescription.

### Tiny

- **Assumption 2.2(a) (bounded outputs) and the Gaussian noise model in §3 are technically incompatible.** Almost surely bounded $y$ and $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ cannot simultaneously hold. This is a minor inconsistency common in the literature and does not affect the main results, but it should be acknowledged (e.g., sub-Gaussian noise suffices).

---

## Nice-to-Haves

- A phase diagram plotting regions of benign/tempered/catastrophic overfitting in $(p, \beta)$ space based on the exponents in Theorem 4.2 would make the three-regime coverage concrete and practically useful.
- An experiment with varying PDE order (e.g., Poisson $p=-1$, Biharmonic $p=-2$) showing increasing noise stability as $p$ becomes more negative would directly demonstrate the core mechanism.
- A log-log plot of empirical risk vs. sample size $n$ for a kernel estimator, overlaid with the predicted rate, would provide quantitative validation of the bounds.
- A brief analysis of perturbation robustness for the co-diagonalization assumption (e.g., what is the rate degradation if $\mathcal{A}$ and $\Sigma$ are only approximately jointly diagonalizable?) would significantly broaden the paper's claimed applicability.
- Adding the Bayesian-frequentist threshold correspondence as a formal proposition (rather than a remark) would strengthen one of the paper's most striking findings.

---

## Removed Points

*These points are flagged for removal; treat them with caution — they may reflect reviewer misreading rather than genuine paper flaws.*

- **[REMOVED — parser artifact] Smoothness threshold inconsistency between §1.1 and Remark 5.** The harsh critic claims the threshold in §1.1 ($\lambda\beta \geq \lambda^r/\lambda^p - p$) conflicts with Remark 5 ($\lambda\beta \geq \lambda r/2 - p$). Reading the paper, the §1.1 formula is almost certainly a PDF-to-text parsing corruption of the same mathematical expression. Remark 5, §4.3, and Table 1 all consistently present $\lambda\beta \geq \frac{\lambda r}{2} - p$ as the threshold. This is not a genuine inconsistency.

- **[REMOVED — formatting/parser] Equation (1) vs. Equation (3) norm vs. squared norm.** The critic claims (1) uses $\gamma_n\|f\|_{\mathcal{H}^\beta}$ (non-squared) while (3) uses $\gamma_n\|f\|_{\mathcal{H}^\beta}^2$ (squared), calling this a substantive discrepancy. Reading the paper, equation (1) in §1 (the informal problem statement) appears to be a presentation shorthand/parser issue. The formal problem statement in Lemma 3.1 (equation 3) uses the squared norm, which is standard for ridge regression. No legitimate inconsistency.

- **[REMOVED — misread] Criticism that "variance is not truly independent of $\beta$."** The critic notes $\tilde{\Sigma}$ involves $\beta$. However, the paper explicitly makes the more nuanced claim in §4.3: the variance *bound's exponent* is independent of $\beta$ (it depends on $2p + \lambda\beta'$, where $\beta'$ is the evaluation norm, not the regularization norm). The statement is precise; the critic misidentifies $\beta$ (regularization) and $\beta'$ (evaluation) as the same parameter.

- **[REMOVED — scope creep / non-standard requirement] Demanding theoretical proofs for neural network experiments.** The demand for an NTK-equivalence theorem to justify using neural networks as validation of kernel theory goes beyond standard expectations for this type of paper. The neural network experiments are explicitly framed as going "beyond kernel methods." The absence of a formal equivalence is a limitation worth noting (kept as a minor weakness), but the absence of a proof is not a fatal flaw.

- **[REMOVED — misread] Criticism that the paper claims to show all three overfitting regimes but only proves benign overfitting.** Theorem 4.2 provides upper bounds whose exponents can be positive, zero, or negative depending on the parameter regime, explicitly covering all three cases. The paper's claim of covering all three regimes is accurate; the critic expected a separate theorem for each regime, which is not the standard form in this literature.

---

## Novel Insights

The most genuinely novel observation synthesized from the reviews—partially surfaced by the Spark Finder but not fully articulated in the paper itself—is the following: the variance stabilization mechanism of the inverse problem ($p < 0$) and the inductive bias smoothness threshold ($\lambda\beta \geq \lambda r/2 - p$) are **coupled** in a specific way. As $|p|$ increases (higher-order PDEs), variance stabilization strengthens *and simultaneously* the smoothness requirement on $\beta$ becomes stricter. This creates a design principle for physics-informed learning: the benefit of using a PDE-constrained model comes with an obligation to match the inductive bias smoothness to the PDE order. The matching condition—which the paper shows recovers the Bayesian threshold from Knapik et al. (2011)—is therefore not merely a curiosity but a design law linking PDE order, kernel smoothness, and generalizability. This connection between the frequentist rate and the Bayesian posterior contraction condition across both the regularized and interpolating regimes is the paper's most surprising and underemphasized contribution.

---

## Suggestions

1. **Add a standalone corollary to Theorem 4.2** that states explicit sufficient conditions (including on $\rho_{k,n}$) under which benign overfitting holds (risk $\to 0$). This is the paper's headline claim and deserves a self-contained formal statement, not a theorem-plus-remark reconstruction.

2. **Include at least one kernel-level experiment** (not a neural network) directly comparing different $\beta$ values on a benchmark inverse problem with known spectral structure. Even a synthetic 1D example with a Matérn kernel and a Laplacian operator would validate the bound at the correct level of abstraction.

3. **Add an experiment varying PDE order $p$** (e.g., $\Delta$ vs. $\Delta^2$) to empirically demonstrate that variance stabilization strengthens as $|p|$ grows, directly testing the mechanism the theory predicts.

4. **Characterize when $\rho_{k,n} = \Theta(1)$ vs. grows** for the PDE-transformed kernel $\tilde{K}$. This is the single most important open question raised by the paper's own analysis; at minimum, provide a proposition for the sub-Gaussian feature case bounding $\rho_{k,n}$ under the paper's main assumptions.

5. **Add an operator/notation summary** (a table or diagram) in the main body showing the relationship among $S, \Sigma, \tilde{\Sigma}, \tilde{K}, \phi, \psi$. This would make Theorems 3.6–3.7 parseable without reading the appendix.

6. **Discuss the co-diagonalization assumption more carefully**: identify which practical PDE-kernel pairs beyond the torus + shift-invariant kernel satisfy it, and provide at least an informal discussion of what the results would look like under approximate diagonalization.

---

**Overall assessment:** The paper makes a genuine and non-trivial contribution to the theoretical understanding of physics-informed learning by identifying the spectral mechanism behind variance stabilization in fixed-dimensional inverse problems. The novelty is high and the core technical approach is sound. However, the paper's central benign-overfitting claim currently rests on an assumption ($\rho_{k,n} = \Theta(1)$) that is not in the theorem statement and is not characterized under the paper's main assumptions; this must be resolved. The experiments are too narrow to validate the theory at its own level of abstraction—no kernel experiment is provided despite the theory being exclusively about kernel estimators. Technical soundness is good at the level of proof structure, but the presentation of Theorems 3.6–3.7 is too abstract for the key mechanism to be clearly readable. Empirical support is weak for the paper's ambitions. Significance is potentially high if the main gap (benign overfitting conditionality) is properly resolved and the experimental grounding is strengthened.