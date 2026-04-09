## Summary

SCaSML introduces a physics-informed inference-time scaling framework that improves pre-trained PDE surrogates (PINNs, GPs) by deriving and solving a "Structural-preserving Law of Defect"—a new semi-linear PDE governing the surrogate's error—via Multilevel Picard (MLP) Monte Carlo iterations. The method provably achieves a convergence rate bounded by the product of the surrogate and simulation errors, and empirically reduces errors by 20–80% on semi-linear parabolic PDEs up to 160 dimensions.

## Strengths

- **Structural preservation enabling high-dimensional stochastic solvers:** The key insight—that subtracting the surrogate's approximate PDE from the original yields a defect PDE that *preserves the semi-linear structure and Lipschitz constants of the original* (Lemma D.11)—is non-trivial and precisely what makes MLP solvers applicable. Without this preservation, the dimension-independent convergence of MLP would not carry over. This is the theoretical linchpin of the paper and is rigorously proved.

- **Multiplicative error bound with practical implications:** Theorem 2.5 shows the final error scales as the *product* of the MLP simulation error and the surrogate error, not their sum. This means the correction step's cost *decreases* as the surrogate improves, yielding the improved scaling law of Corollary 2.6 (from $O(m^{-\gamma})$ to $O(m^{-\gamma-1/2})$). The empirical scaling law verification (Figure 4, Appendix G.3) with measured slope changes consistent with the theory adds credibility.

- **Stabilization of MLP in very high dimensions:** The most compelling empirical result is the LQG experiment (100–160D), where the naive MLP solver fails catastrophically (relative $L^2$ error > 5.0) while SCaSML successfully corrects the PINN to ~0.05–0.10 error. This demonstrates that the surrogate-as-control-variate effect is not merely incremental—it makes simulation feasible in regimes where it would otherwise diverge.

- **Comprehensive experimental validation with statistical rigor:** The inclusion of 10-run repeated experiments with paired t-tests (Appendix G.4, Tables 2–6, all $p \ll 0.001$), fixed-budget Pareto analyses (Appendix G.7–G.8), and inference-time scaling curves across multiple PDE families goes well beyond the typical single-run evaluation in SciML papers.

## Weaknesses

### Major:

- **Gap between Assumption 2.4 and practical PINN training:** The theoretical guarantees require both (1) the PDE residual $\epsilon$ to be uniformly bounded by $e(\hat{u})$, and (2) the $W^{1,\infty}$ defect norm to be bounded by $e(\hat{u})$. In the PINN literature, it is well-established that small PDE residuals do not guarantee small solution errors—PINNs can stagnate in local minima with low loss but poor accuracy, or exhibit spectral bias where low-frequency modes are well-approximated but high-frequency errors persist. While the $W^{1,\infty}$ condition partially addresses this (it bounds the defect itself, not just the residual), the paper does not verify that standard PINN training with Adam actually produces surrogates satisfying these assumptions in practice, nor does it characterize the failure regime where the surrogate is too poor for SCaSML to offer improvement. A surrogate quality ablation—deliberately degrading the surrogate and showing where correction stops being effective—would significantly strengthen the practical applicability of the theory.

- **Inference-time latency trade-off misrepresented in framing:** The abstract claims SCaSML "fuses the speed of machine learning with the rigor of numerical simulation," but Table 1 shows SCaSML is 5–60× slower than the pure surrogate at inference (e.g., LCD 10d: 0.45s vs. 6.77s; VB-GP 60d: 1.68s vs. 57.79s). The actual trade-off is *accuracy vs. latency*, not speed vs. accuracy. This is a fundamental characteristic of the method, not a flaw, but the framing should be corrected. The "elastic compute" language (Remark 2.2) is more accurate and should be the primary framing.

- **Impact of clipping on theoretical bounds is unanalyzed:** The clipping/thresholding in Algorithm 2 is essential for numerical stability, and the thresholds differ dramatically between methods (e.g., 0.01 for SCaSML vs. 10.0 for naive MLP in LQG). The theoretical error bounds assume Lipschitz continuity and do not account for the clipping. Since clipping is a hard nonlinearity that violates the Lipschitz assumption, there is a gap between theory and practice for the most challenging problems. The caption of Figure 12 acknowledges the clipping "trade-off" but no analysis of how different thresholds affect the bound is provided.

### Minor:

- **Heuristic budget allocation:** The 1/(d+1) split between training and inference compute (used in Appendix G.7) is presented without justification or optimality analysis. Whether this is near-optimal or far from it significantly affects the practical efficiency claims.

- **Global error evaluation cost:** The method corrects the surrogate pointwise, yet Table 1 reports global $L^2$, $L^\infty$, and $L^1$ errors over test sets of ~1200 points. The total inference cost scales linearly with the number of query points, which is not explicitly discussed and could be prohibitive for applications requiring dense spatial coverage.

## Nice-to-Haves

- **Direct variance reduction measurement:** The core mechanism is variance reduction via the surrogate acting as a control variate. A direct comparison of MLP estimator variance with and without the surrogate correction (not just the final error) would empirically validate the theoretical variance scaling argument from Section 2.1.

- **Verification that trained PINNs satisfy smoothness assumptions:** Assumption F.1 (Gevrey-class regularity) and the $W^{1,\infty}$ bounds are used in the proofs but never empirically verified. A simple check—e.g., measuring the defect's smoothness properties on a held-out set—would bridge theory and practice.

- **Failure mode characterization:** Explicitly demonstrating a case where the surrogate error is too large, causing the defect PDE solver to diverge or fail to improve, would establish clear applicability boundaries and help practitioners decide when to deploy SCaSML.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Sign convention inconsistency between Equations (1) and (6):** The harsh critic flagged a potential sign issue, but this appears to be a parser artifact. The original PDE (1) has $-\partial u/\partial r + Lu + F = 0$, equivalently $\partial u/\partial r + Lu + F = 0$ after rearrangement, and the residual in (6) consistently uses the same sign convention. No actual inconsistency exists.

- **Lipschitz constant degradation from surrogate gradients:** The harsh critic speculated that large surrogate gradients could degrade the effective Lipschitz constant of $\tilde{F}$, negating dimension-independent convergence. This is factually incorrect: Lemma D.11 explicitly proves that the modified nonlinearity $\tilde{F}$ inherits the *exact same* Lipschitz constant $L$ as $F$, because the surrogate-dependent terms cancel in the difference (proof: $F(\hat{U} + w_1) - F(\hat{U} + w_2)$ eliminates the surrogate background state $\hat{U}$).

- **Missing comparisons with FNO, DeepONet, or other neural operators:** The paper targets pointwise PDE solutions in 100–160 dimensions. Neural operators like FNO are designed for fixed low-dimensional grids (2–3D) and do not naturally scale to these dimensions. The comparison category is inappropriate for the problem setting.

- **Demanding confidence intervals for benchmark evaluations:** Single-run evaluation is the norm in the high-dimensional PDE literature; the paper already goes beyond this by including 10-run statistical tests in the appendix.

- **Missing related works on neural control variates:** Per rules, missing related work citations are not a valid weakness without external source confirmation.

- **Reproducibility concerns about hyperparameters:** The paper provides detailed hyperparameters (learning rates, optimizer settings, network architectures, sample sizes) in Sections 3.1–3.4.

## Novel Insights

The paper's most profound observation is not the defect correction itself—which is classical—but the specific way it enables a *type mismatch resolution* between machine learning and stochastic simulation: neural surrogates excel at learning smooth, low-frequency solution components (spectral bias) while Monte Carlo methods have convergence rates *independent of the integrand's smoothness*. By defining the defect PDE so that its source term $\epsilon$ is precisely the high-frequency residual the neural network struggled with, the framework channels each method's strength toward the component it handles best. This is more than a control variate; it is a structural decomposition of the problem by frequency, where the "variance reduction" is not just numerical but conceptual—the surrogate and simulator are solving different problems that happen to sum to the answer.

## Suggestions

- Re-frame the abstract and introduction to emphasize the accuracy-vs-latency trade-off rather than "speed + rigor," and position "elastic compute" as the primary practical contribution.
- Add a surrogate quality ablation (e.g., intentionally under-trained PINN with high error) to empirically map the boundary where SCaSML ceases to improve over the baseline, directly addressing the gap in Assumption 2.4.
- Include a brief analysis or discussion of how the clipping threshold interacts with the theoretical Lipschitz-based error bounds, even if only to bound the bias introduced by thresholding.

## Axis Evaluation

- **Novelty:** High. The specific combination of structural-preserving defect PDE derivation + MLP correction + inference-time scaling framing is novel. The structural preservation result (Lipschitz constants carry over exactly) is a clean theoretical contribution.

- **Technical soundness:** Moderate-to-high. The core theory is rigorous and the proofs are detailed. The main gap is the unverified assumptions about surrogate regularity and the unanalyzed impact of clipping on the bounds.

- **Empirical support:** High. Extensive experiments across multiple PDE families, dimensions, and surrogate types, with statistical tests and fixed-budget analyses. The LQG stabilization result is particularly convincing.

- **Significance:** High for the SciML community. The framework provides a principled and theoretically grounded way to add compute-time accuracy to neural PDE solvers, addressing a real need for trustworthiness in scientific applications.

- **Clarity:** Moderate. The paper is well-organized but suffers from occasional overclaiming (speed vs. accuracy framing) and heavy notation. The separation of linear warmup (Section 2.1) from the general case (2.2) is pedagogically effective.