## Summary

This paper derives non-asymptotic Sobolev-norm learning curves for kernel ridge and ridgeless regression applied to elliptic linear inverse problems. The central finding is that the PDE operator—because its eigenvalues *grow* with index (negative $p$)—dampen high-frequency variance sufficiently to enable benign overfitting in fixed spatial dimension, contrasting with standard regression where this requires high-dimensional asymptotics. A secondary contribution quantifies how the smoothness of the inductive bias ($\beta$) affects convergence: rates are independent of $\beta$ once a threshold $\lambda\beta \ge \frac{\lambda r}{2} - p$ is met, a condition that surprisingly matches Bayesian inverse-problem literature.

---

## Strengths

- **Benign overfitting in fixed spatial dimension via inverse-problem structure.** The specific mechanism—eigenvalues of $\mathcal{A}$ growing with index ($p < 0$) causing the spectrally transformed covariance $\tilde{\Sigma} = \mathcal{A}^2 \Sigma^\beta$ to have a steeper effective decay than a pure kernel covariance—is a concrete and non-obvious explanation distinguishing inverse problems from regression. This is not a generic claim but is precisely tracked through Theorem 4.2 and Remark 7.

- **Unified bias-variance framework covering both ridge and ridgeless estimators.** The same spectral decomposition apparatus (Theorems 3.6, 3.7) and the same concentration coefficient $\rho_{k,n}$ yield bounds for both settings by changing how $k$ is chosen, and the regularized case provably recovers the known minimax rates of Lu et al. (2022), providing internal consistency validation.

- **Cross-paradigm agreement of the smoothness threshold.** The condition $\lambda\beta \ge \frac{\lambda r}{2} - p$ derived from a frequentist upper-bound analysis independently reproduces the smoothness condition from Bayesian inverse-problem theory (Knapik et al., 2011; Szabó et al., 2013). This unexpected alignment strengthens confidence in the correctness of the bound.

- **Empirical confirmation that sufficiently smooth activations give rate-independent convergence.** Figure 1 (Left) shows convergence curves for ReLU through ReLU$^4$; the ReLU$^3$ and ReLU$^4$ curves are near-indistinguishable, directly illustrating the theoretical threshold prediction. This is a clean qualitative test even if not a quantitative rate validation.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Bounded-output assumption (2.2a) is inconsistent with the Gaussian noise model used throughout the analysis.** Assumption 2.2(a) states $y$ is bounded almost surely by $M$, yet Section 3.2 and Theorem 4.2 assume $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, which is unbounded. Because the concentration inequalities in Theorem 3.5 rely on boundedness, this is not a cosmetic inconsistency—it is a gap in the proof foundations. The paper must either replace boundedness with a sub-Gaussian assumption throughout, or verify that the specific Gaussian noise case is handled separately. This needs to be corrected before the theory can be trusted as stated.

- **No kernel experiments despite all theory being about kernel estimators.** The theoretical contributions concern kernel ridge/ridgeless regression (Lemma 3.1, Theorems 3.6, 3.7, 4.1, 4.2), yet every experiment uses neural networks. There is no direct empirical test of the kernel estimator, no comparison of ridged vs. ridgeless kernel behavior at matched settings, no verification of the predicted polynomial rate exponents (e.g., $n^{\lambda(\beta'-r)/(2p+\lambda\beta+1)}$), and no ablation over $p$ (PDE order) or $\beta$ (Sobolev regularization exponent) in the kernel regime. This creates a fundamental mismatch between the paper's theoretical core and its empirical support.

- **The bias bound carries an explicit $1/\delta$ factor, rendering the high-probability statement unusually weak.** Theorem 3.7 bounds bias as $\lesssim \rho_{k,n}^3 \cdot \frac{1}{\delta} \cdot [\ldots]$, and this factor propagates to Theorem 4.2. A bias bound that blows up as $\delta \to 0$ means the stated rates are only achievable at constant confidence levels, not in the usual high-probability sense. Neither the paper nor the appendix discusses whether this $1/\delta$ is removable (e.g., by a union bound or Markov inequality argument) or is a fundamental artifact of the proof technique. This undermines the precision of the main learning-curve claims.

- **The benign overfitting conclusion depends critically on $\rho_{k,n} = \Theta(1)$, which is only established under sub-Gaussian features.** Remark 6 acknowledges that in the worst case $\rho_{k,n} = \tilde{O}(n^{2p+\beta\lambda-1})$; since the variance and bias bounds scale as $\rho_{k,n}^2$ and $\rho_{k,n}^3$ respectively, benign overfitting in the general case is not established—it relies on the sub-Gaussian assumption that the paper simultaneously claims to avoid. The main text does not adequately foreground this conditionality; Section 4.2's prose implies a cleaner conclusion than the theorem actually delivers.

### Minor

- **Only a single PDE (2D Poisson) with one synthetic ground truth is tested.** While the Poisson equation is illustrative, the paper claims general applicability to elliptic inverse problems. The Schrödinger equation in Example 2.3 is introduced as a motivating case but never experimentally tested. A second PDE—especially a higher-order one to test the $p$-dependence of variance stabilization—would considerably strengthen the empirical story.

- **Neural network experiments do not verify proximity to the kernel/NTK regime.** The theoretical–experimental bridge depends on networks operating in or near the lazy training regime. No diagnostic (e.g., relative parameter change, NTK alignment) is provided to justify using kernel theory to explain the neural network results. Without this, the connection is speculative.

- **No comparison between regularized and interpolating estimators in experiments.** One of the paper's primary contributions is unifying ridge and ridgeless estimation, yet Figure 1 only shows the PINN interpolator vs. a plain NN interpolator. A direct ridged vs. ridgeless comparison under matched conditions would validate the regime transition claim.

- **The simultaneous diagonalization assumption (Assumption 2.2d) is strong and its limitations are underexplored.** The paper correctly cites its prevalence in prior work and notes that it holds for shift-invariant kernels with the Laplacian on the torus. However, it does not discuss what happens to the rates under perturbative misalignment, nor whether the spectral framework degrades gracefully or catastrophically when the assumption is violated. A brief analysis of sensitivity would significantly improve practical credibility.

- **The closed-form solution (Lemma 3.1) uses the population covariance $\Sigma^{\beta-1}$, not an estimable empirical quantity.** The paper mentions the semi-supervised analogue in passing but does not analyze the practically realizable estimator. The gap between the oracle estimator studied and a computable implementation should be stated explicitly as a limitation, especially given the practical framing of the introduction.

### Tiny

- **Smoothness threshold notation is inconsistent between Section 1.1 and Remark 5/Section 4.3.** Section 1.1 writes the threshold as $\lambda\beta \ge \frac{\lambda^r}{\lambda^p} - p$ whereas Remark 5 and Section 4.3 write it as $\lambda\beta \ge \frac{\lambda r}{2} - p$. The latter version is consistent with the surrounding analysis; the former appears to be a formatting artifact. Since this threshold is a core advertised contribution, the inconsistency should be corrected and a single canonical statement should appear prominently.

- **Abstract overstates the generality of the rate-independence claim.** The statement "the convergence rate is actually independent to the choice of (smooth enough) inductive bias" omits that this holds above a smoothness threshold on $\beta$ that itself depends on problem parameters; below the threshold the rate is suboptimal. A brief qualification in the abstract would be more accurate.

---

## Nice-to-Haves

- **Log-log convergence rate plots with predicted theoretical slopes overlaid.** Figure 1(Left) uses only 5 sample sizes on a linear scale. Log-log plots with slope annotations would allow readers to assess whether the predicted exponents (e.g., $n^{\lambda(\beta'-r)/(2p+\lambda\beta+1)}$) are empirically supported.

- **Spectral visualization of variance stabilization.** Plotting the effective eigenvalues $\lambda_i^\beta p_i^2$ vs. $i$ for several values of $p$ (including $p=0$ for comparison) would directly illustrate the paper's core mechanism and make the theoretical insight more accessible to practitioners.

- **A perturbation analysis or discussion of near-commutativity.** Even an informal argument about how rates degrade when $\mathcal{A}$ and the kernel covariance operator nearly (but not exactly) share eigenvectors would address the most frequently raised practical concern.

- **Direct verification that the kernel method's ridge vs. ridgeless transition matches Theorems 4.1 and 4.2.** A controlled kernel experiment varying $n$, $\beta$, and PDE order would close the gap between theory and experiment in the cleanest possible way.

---

## Removed Points

*These points were flagged for removal. Treat them with caution; they may reflect misreadings.*

- **"Fixed dimension" is not formalized.** The critic argued that "fixed dimension" is never defined. However, the paper uses polynomial eigendecay $\lambda_i \propto i^{-\lambda}$ with $\lambda > 1$, which is the standard capacity condition holding in any fixed spatial dimension $d$; Remark 2 explicitly ties $\lambda$ to Matérn/Sobolev kernels on $\mathbb{T}^d$. The contrast with high-dimensional benign overfitting (which typically requires intrinsic dimension diverging) is made via citation. The framing is adequate, not missing.

- **Neural-network experiments compare "PINN vs. NN" unfairly.** The critic implied this comparison favors the PINN and is therefore meaningless. However, the asymmetry is intentional: the purpose is to show that incorporating inverse-problem structure (PDE operator) changes the noise sensitivity of an otherwise identical interpolating architecture. This is exactly the comparison needed to validate the benign overfitting claim, and the asymmetry favors the baseline (plain NN), not the author's method.

- **Claims that the discussed kernel estimator is too abstract without worked examples.** The paper provides Example 2.3 (Schrödinger equation on the torus), Table 1 with parameter semantics, and Remark 2 connecting co-diagonalization to Fourier modes. While more worked-out rate instantiations would help (addressed in Nice-to-Haves), the abstraction level is acceptable for a theory paper targeting kernel inverse-problem experts.

- **Claims about unfairness of comparisons with Barzilai & Shamir (2023).** The paper explicitly recovers Barzilai & Shamir's result as a special case ($p=0, \beta=1, \beta'=0$), which is a genuine consistency check, not an unfair comparison.

- **Requests for theoretical proofs bridging to neural networks.** The paper explicitly frames the neural network section as validation "beyond kernel estimators" and does not claim the theory directly covers neural networks. Demanding formal NTK-regime proofs is outside the paper's stated scope.

---

## Novel Insights

The most genuinely novel insight in this work—one not merely restating the paper's claims—is the realization that the spectral structure of the *inverse* operator, rather than high intrinsic dimensionality, can serve as the mechanism for variance self-regularization. In standard regression, benign overfitting requires the effective rank of the feature covariance to grow with $n$; here, the PDE operator shifts the effective eigenvalue spectrum so that the transformed covariance $\tilde{\Sigma} = \mathcal{A}^2 \Sigma^\beta$ satisfies the necessary spectral-decay conditions for benign overfitting even when the ambient (spatial) dimension is fixed. A direct corollary is that the two axes traditionally conflated—data dimension and statistical complexity—can be decoupled in the inverse-problem setting, with physical operator order substituting for dimensionality. The surprising agreement between the frequentist smoothness threshold and the Bayesian posterior contraction condition (Knapik et al., 2011) reinforces this and hints at a deeper structural reason why both paradigms converge on the same regularity requirement for inverse problems.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | High. The link between PDE operator order and benign overfitting in fixed dimension is novel; the unified kernel Sobolev framework for ridge and ridgeless inverse-problem estimators has not appeared in this form before. |
| **Importance of research question** | Substantial. Understanding generalization of overparameterized PDE solvers is an open and practically relevant question. |
| **Claims well-supported** | Partially. The regularized case is well-supported and recovers prior rates. The interpolation case has meaningful gaps: the $1/\delta$ bias factor and the $\rho_{k,n}$ conditionality mean the benign overfitting claim is established only under additional assumptions that are not always foregrounded clearly. |
| **Soundness of experiments** | Weak. The experiments are exclusively neural-network-based despite an entirely kernel-theoretic paper; there are no quantitative rate validations; and the setting is limited to a single 2D PDE. |
| **Clarity of writing** | Moderate. The high-level narrative is clear and Sections 1–2 are readable, but Theorems 3.5–3.7 are not digestible in the main text without more interpretive corollaries, and there are notational inconsistencies. |
| **Value to the research community** | Meaningful, particularly for the physics-informed ML theory community. The variance-stabilization insight and the smoothness threshold are actionable results. |
| **Contextualization relative to prior work** | Mostly adequate. Comparisons to Lu et al. (2022), Barzilai & Shamir (2023), and Haas et al. (2024) are present, though the Lu et al. comparison could be made more explicit (a direct proposition mapping theorem parameters would help). |

Overall, the paper contains a real and interesting theoretical contribution. However, the inconsistency between the boundedness assumption and Gaussian noise model, the absence of any kernel experiments, the $1/\delta$ bias issue, and the under-qualified dependence on $\rho_{k,n}$ in the main benign-overfitting claim represent substantive issues that need to be addressed before the work fully delivers on its promises.