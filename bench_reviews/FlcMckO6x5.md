## Summary

This paper develops the theoretical foundations for Separable Neural Networks (SepNNs)—architectures that factorize multivariate functions into linear combinations of univariate functions parameterized by lightweight factor MLPs. The contributions are threefold: (1) a universal approximation theorem for CP, TT, and Tucker SepNNs using Stone-Weierstrass arguments; (2) NTK regime characterization showing deterministic kernel convergence under infinite width+rank and random kernel under infinite width+fixed rank; and (3) an efficient Separable Preconditioned Gradient Descent (SepPGD) method that alleviates spectral bias with O(nD) complexity on D-dimensional grids with n points per dimension.

## Strengths

- **Unified approximation theory for multiple SepNN types:** Theorem 1 provides a clean, unified proof of universal approximation for CP, TT, and Tucker SepNNs, extending prior work (Cho et al., 2023) from bivariate to general multivariate settings. The Stone-Weierstrass approach is elegant and the verification that the separable algebra separates points, contains constants, and is closed under algebraic operations is done carefully for all three decomposition types.

- **NTK decomposition revealing structural insight:** Lemma 1 shows the SepNN NTK decomposes as a weighted sum of factor NTKs, which is a non-trivial structural result. The distinction between deterministic NTK (Theorem 2: infinite width + infinite rank) and random NTK (Corollary 1: infinite width + fixed rank) provides genuine insight into why practical SepNNs with small rank exhibit different training behavior than wide standard networks.

- **SepPGD exploits separable structure for efficient preconditioning:** The key algorithmic insight—decomposing the large n^D × n^D preconditioner into D smaller n × n factor preconditioners via the Kronecker structure (Lemma 2)—is both theoretically justified and practically significant. Table 1 clearly shows the complexity advantage over prior NTK-PGD methods for the gradient formulation step.

- **Consistent empirical improvements across applications:** SepPGD shows meaningful PSNR gains (e.g., 26.48→33.30 on Plane image, Table 5) and faster convergence across KRR, INRs, and PINNs, with useful ablation studies on rank R, modulation parameter k, and update frequency (Tables 2–7).

## Weaknesses

### Major:

- **Theory-practice gap in NTK regime:** The deterministic NTK (Theorem 2) and the spectral bias analysis (Eq. 5) both require W,R→∞, but practical SepNNs use small fixed rank (e.g., R=64–500 in experiments). Under fixed rank, Corollary 1 shows the NTK converges to a *stochastic* kernel, and Remark 3 admits that "the training dynamic can not be characterized uniformly using a fixed NTK matrix." Since SepPGD is designed based on infinite-rank spectral properties yet evaluated in the fixed-rank regime, there is no theoretical guarantee that the preconditioner correctly identifies and adjusts the relevant eigenmodes when the NTK is random. The paper acknowledges this gap (Section A.1.2, A.4) but only offers heuristic probability bounds (Chebyshev inequality) and speculative connections to random feature models. For a paper whose core contribution is the interplay between NTK theory and preconditioning, this gap between the regime where the theory holds and the regime where the method is applied is a significant weakness.

- **SepPGD cannot precondition PDE residual loss in PINNs:** Appendix A.12 states: "For the PDE residual loss, which involves derivatives, we do not employ the SepPGD algorithm, as extending PGD to derivative-based losses requires substantially different algorithmic treatment." In physics-informed learning, the PDE residual loss is often the dominant and most challenging component—especially in data-scarce regimes where PINNs are most needed. That SepPGD can only accelerate the data-fitting and boundary/initial condition components significantly limits its utility in the very scientific ML domain the paper targets. This limitation should be discussed prominently in the main text, not relegated to the appendix.

- **NTK analysis limited to CP SepNN; TT and Tucker extensions are unproven:** The paper's NTK theory (Lemma 1, Theorem 2, Corollary 1) is derived exclusively for the CP formulation. Footnote 1 states: "the NTK analysis is primarily conducted for the CP SepNN, while we believe it can be readily extended." Similarly, Section A.1.2 says extensions are "a valuable direction for future research." Since Theorem 1 covers all three decomposition types and the introduction presents TT and Tucker as first-class SepNN variants, the absence of any NTK derivation (even a sketch) for these cases leaves the theoretical contribution incomplete.

### Minor:

- **No explicit approximation error rates:** Theorem 1 proves existence of approximations but does not quantify how rank R or width W must scale with dimension D or target function smoothness to achieve ε-approximation. The paper acknowledges this (Section A.1.2): "our current theoretical analysis... does not yet provide explicit approximation error rates in terms of network rank or width." This limits the practical guidance the theory can offer for hyperparameter selection.

- **No empirical validation of the spectral bias alleviation mechanism:** The paper's core claim is that SepPGD alleviates spectral bias by adjusting the NTK eigenvalue distribution. Figure 1(d) shows the *initial* eigenvalue spectrum, but no figure tracks how eigenvalues evolve during training with vs. without SepPGD. Plotting the NTK condition number or eigenvalue decay over training steps would directly validate that the mechanism works as theorized, rather than only observing faster convergence as indirect evidence.

- **Experiments limited to D≤3 despite theoretical advantage growing with D:** The O(nD) vs. O(n^D) efficiency gap is the central practical motivation, yet all experiments use D=2 or D=3 where the advantage is modest. Demonstrating the method on D≥5 problems would substantiate the scaling claims that distinguish SepPGD from prior work.

### Trivial:

- The notation in Definition 1 and Eq. (8) is dense with tensor operations (unfold, fold, mode-d products). While standard in the tensor literature, a brief intuitive explanation or pseudo-code algorithm in the main text would improve accessibility for the broader ICLR audience.

## Nice-to-Haves

- Derivative-aware preconditioning that extends SepPGD to PDE residual losses, making the method fully applicable to PINNs.
- High-dimensional benchmarks (D≥5) to empirically demonstrate the scaling advantage.
- Explicit non-asymptotic convergence or generalization bounds for the fixed-rank regime, even if loose, to partially bridge the theory-practice gap.
- Comparison with domain-specific SOTA INR methods (e.g., SIREN + Fourier features, TensoRF) to verify that convergence gains do not come at the cost of final representation quality.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Complexity claim is misleading because handling the residual tensor requires O(n^D):** The harsh critic argued that the O(nD) claim ignores the cost of handling the n^D residual. However, SepNNs on grid inputs exploit the separable structure to compute forward passes and gradients without materializing the full n^D output tensor (this is the well-established efficiency advantage of SepNNs per Liang et al., 2022; Cho et al., 2023). The O(nD) claim in Table 1 specifically refers to the gradient formulation complexity, and Remark 4 separately discusses preconditioner construction complexity. The claim is accurate for what it states, though the paper could be clearer about what is and isn't included in the O(nD) figure.

- **Unfair comparison with MSK because MSK runs out of memory:** The paper provides comparisons with MSK where feasible and explains the memory limitation. The asymmetry (full-batch SepPGD vs. MSK that cannot run full-batch) actually demonstrates SepPGD's advantage, not an unfair comparison. Per the rules, this concern is removed.

- **Formatting and style issues:** Removed per rules.

- **Missing related works:** Removed per rules as we cannot verify existence of specific references.

- **Reproducibility concerns about undisclosed hyperparameters:** The paper provides extensive ablation studies (Tables 2–7) and detailed experimental settings in Appendix A.12. Removed per rules.

- **Demand for confidence intervals or multiple random seeds in large-scale benchmarks:** Removed as nice-to-have; single-run evaluation with convergence curves is standard in this area.

## Novel Insights

The decomposition of the SepNN NTK into a weighted sum of factor NTKs (Lemma 1) reveals that the spectral bias of SepNNs has a fundamentally different origin than standard MLPs: it arises from the *product* structure across dimensions (the a_d vectors) compounding with the individual factor NTK spectra. This means the effective condition number of the SepNN NTK can be much worse than any individual factor NTK, as the eigenvalue decay is multiplicative across dimensions. This structural insight explains why SepNNs are particularly prone to spectral bias and why factor-level preconditioning (SepPGD) is both necessary and sufficient—the Kronecker structure means that adjusting each factor's spectrum independently propagates to the full NTK through the product structure.

## Suggestions

- Add a figure tracking NTK eigenvalue distribution (or condition number) over training steps with and without SepPGD to directly validate the spectral bias alleviation mechanism.
- Discuss the PINN PDE residual loss limitation in the main text (Section 5 or Section 6), not only in the appendix, and characterize the scenarios where SepPGD provides the most benefit within PINNs (data-rich vs. data-scarce).
- Provide even a rough sketch of how the NTK analysis extends to TT and Tucker SepNNs, or explicitly scope the NTK contribution to CP only in the title/abstract to avoid overclaiming.
- Add at least one experiment with D≥4 to demonstrate the scaling advantage that motivates the method.

---

**Axis evaluations:**

- **Novelty:** Moderate-to-high. The NTK derivation for SepNNs and the SepPGD algorithm are novel. The approximation theorem is a natural but non-trivial extension of prior bivariate results.

- **Technical soundness:** Moderate. The proofs for approximation theory and NTK convergence under infinite rank are rigorous. However, the gap between infinite-rank theory and fixed-rank practice is a real soundness concern for the core claim that SepPGD alleviates spectral bias via the theorized mechanism.

- **Empirical support:** Moderate-to-strong. Consistent improvements across tasks, but limited to low dimensions and lacking direct validation of the spectral mechanism. The PINN experiments are incomplete due to the PDE residual exclusion.

- **Significance:** Moderate-to-high. SepNNs are increasingly important in scientific ML, and a principled optimization method with theoretical backing addresses a real need. The scope is narrower than claimed (grid inputs, CP only, no PDE residual preconditioning).

- **Clarity:** Moderate. The logical flow is clear, but the dense tensor notation in Definition 1 and the scattered discussion of limitations across main text and appendices reduce accessibility.