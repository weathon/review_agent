## Summary

This paper introduces Sensitivity-Constrained Fourier Neural Operators (SC-FNO), which augment standard FNO training with a sensitivity loss $L_s$ that supervises parameter Jacobians $\partial \mathbf{u}/\partial \mathbf{p}$ against pre-computed ground truth. The core claim is that this regularization simultaneously preserves forward solution accuracy and dramatically improves inverse problem performance, sensitivity estimation, and robustness under distribution shift. The idea is well-motivated, and the low-dimensional experiments provide compelling evidence that standard FNO learns poor parametric sensitivities even when forward predictions appear accurate.

## Strengths

- **Well-motivated problem with clear methodology.** The paper correctly identifies poor parametric sensitivity in neural operators as a bottleneck for inversion and optimization, and proposes a sensible, easy-to-understand regularization strategy (Section 2.1, Equation 6).
- **Strong low-dimensional empirical evidence for sensitivity accuracy.** Table 1 and Figure 3 clearly demonstrate that FNO learns highly inaccurate sensitivities (e.g., FNO $\partial u/\partial \alpha$ $R^2 = 0.206$ vs. SC-FNO $0.987$ on PDE2; FNO relative $L^2$ error $0.723$ vs. SC-FNO $0.0112$ for Navier-Stokes in Table 2), while SC-FNO eliminates unphysical oscillations. This directly motivates the need for the proposed loss.
- **Clear inversion improvement in low-dimensional settings.** Figure 1 shows SC-FNO achieving tight clustering around the diagonal for multi-parameter inversion of PDE1 ($R^2 = 0.986$, relative $L^2 = 0.036$) where FNO scatters badly ($R^2 = 0.642$, relative $L^2 = 0.222$).
- **Robustness to input perturbations and concept drift.** Table 1 shows that at $\lambda = 0.4$ perturbation, FNO solution $R^2$ drops to $0.529$ on PDE1 while SC-FNO maintains $0.912$; Figure 5 further shows graceful degradation for SC-FNO versus immediate collapse for FNO.
- **Practical flexibility in gradient generation.** Section 3.5 and Table 5 demonstrate that SC-FNO works with both automatic-differentiation and finite-difference-generated sensitivities, making it applicable to non-differentiable legacy solvers.

## Weaknesses

### Fatal
None.

### Major
- **Impossible $R^2$ values for Jacobian metrics in Tables 3 and 4.** Table 3 reports FNO "Mean Jacobian" $R^2 = 3.11$ (N=500) and $-5.8373$ (N=100) for PDE4; Table 4 reports FNO Jacobian $R^2 = 4.332$ (N=500) and $-14.012$ (N=100) for the 82-parameter zoned PDE2. Under the standard definition $R^2 = 1 - \text{RSS}/\text{TSS}$, values greater than 1 are mathematically impossible. This indicates a fundamental error in the computation, definition, or reporting of the primary metric used to validate sensitivity accuracy in the high-dimensional and limited-data regimes. While the relative $L^2$ values in these tables remain plausible and tell the same qualitative story, the impossible $R^2$ values collapse the credibility of the quantitative evidence for these critical experiments.
- **Implausibly identical per-parameter inversion metrics in Figure 2 and its table.** The table under Figure 2 reports per-parameter $R^2$ values for simultaneous multi-parameter inversion that are identical to three decimal places across all five parameters within each model–PDE combination (e.g., FNO $R^2 = 0.635$ for every parameter of PDE1; SC-FNO $R^2 = 0.945$ for every parameter; PDE2 shows the same pattern with $0.85$ and $0.96$). Because these parameters have different physical roles, scales, and identifiability, exact equality is statistically impossible. This strongly suggests the values were aggregated, erroneously copied, or otherwise mishandled. Because multi-parameter inversion is a headline result, this undermines confidence in a core experimental claim, even though Figure 1 still visually demonstrates the improvement.
- **Abstract overclaims inversion scalability to 82 parameters.** The abstract states SC-FNO "exhibits superior performance in parameter inversion tasks, accommodates more complex parameter spaces (tested with up to 82 parameters)." However, Section 3.4 only reports forward solution accuracy and Jacobian error for the 82-parameter zoned PDE2 (Table 4); no inversion experiment is performed or reported. Inversion in 82 dimensions is qualitatively harder than evaluating forward Jacobians, and its omission means the paper has not substantiated its most ambitious scalability claim in the context where it is most needed.

### Minor
- **Inversion protocol is under-specified.** Section 3.1 states that backpropagation was used to optimize parameters by minimizing discrepancy, but it does not report initialization strategy, optimizer, learning rate, number of steps, or whether parameters were constrained to physical bounds. These details are necessary to assess whether FNO's poor inversion performance is an inherent limitation or an artifact of the optimization setup.
- **Efficiency claims do not fully address data-generation cost for high-dimensional cases.** Section 3.4 and Section 3.6 claim SC-FNO "even reduces training time" and uses fewer training samples. However, generating labels for the sensitivity loss $L_s$ requires either a differentiable solver or $O(d)$ finite-difference runs per training sample. For the 82-parameter case, if finite differences were used, the upfront data-generation burden would be substantial. The paper does not specify which gradient-generation method was used for the 82-parameter experiment, nor does it report total wall-clock time including dataset preparation. Without this accounting, readers cannot assess the practical computational trade-off in the high-dimensional regime.

### Trivial
None.

## Nice-to-Haves
- Perform and report an inversion experiment for the 82-parameter zoned PDE2 to align with the abstract's scalability claim.
- Report end-to-end wall-clock cost (data generation + model training) for the high-dimensional case to substantiate practical efficiency claims.
- Show optimization trajectories or loss landscapes during inversion to clarify whether FNO fails due to vanishing/erroneous gradients versus non-convexity.
- Add variance estimates or repeated runs, as all reported metrics are point estimates without standard deviations.

## Removed Points
These points are flagged to be removed; treat them with caution.

- **"The introduction's claim that estimating $\partial \mathbf{u}/\partial \mathbf{p}$ 'has not been studied for neural operators' is strong."** The paper explicitly hedges this with "to the best of our knowledge" (Section 1), so this is not an overclaim.
- **Criticisms about missing appendix proofs or absent references.** The parser strips appendix sections; they exist in the original submission.
- **Stronger version of the efficiency criticism.** While the paper could more transparently account for data-generation cost, the abstract and main text are careful to frame efficiency claims in terms of "training time" and "training data requirements," which are technically accurate descriptions of the neural network training phase. The criticism that this is intentionally misleading overstates the case.

## Novel Insights
Beyond the paper's own contributions, the reviews highlight an interesting observation that deserves deeper analysis: SC-FNO-PINN consistently provides only marginal improvements over pure SC-FNO across all tests (Table 1), suggesting that PINN-type equation loss—which supervises spatial-temporal derivatives but not parametric sensitivities—is largely ineffective for improving inversion and sensitivity accuracy. This implies that direct supervision of $\partial \mathbf{u}/\partial \mathbf{p}$ may be more impactful for coupled optimization tasks than enforcing the PDE residual alone, a point that runs counter to much of the physics-informed learning literature and could motivate future work on which physical constraints matter most for operator learning.

## Suggestions
- Define and correct the Jacobian evaluation metric in Tables 3 and 4. If the standard $R^2$ definition was not used, state the alternative explicitly; otherwise recompute. At minimum, rely on the relative $L^2$ values in those tables until the metric is fixed.
- Provide actual per-parameter inversion metrics in Figure 2 or explicitly label the value as an aggregate statistic if aggregation was intended. Do not copy the same number across rows.
- Include an 82-parameter inversion experiment, or revise the abstract to clarify that 82 parameters were tested only for forward accuracy.

## Score and Decision

**Calibration comparison:**
- **High anchor:** PhyMPGN (avg 8.0) and MgNO (avg 6.5) feature solid, credible experiments throughout with no metric computation errors and well-controlled baselines. The paper under review has a comparably clear methodology and motivation, but falls well below these anchors due to the impossible $R^2$ values and suspicious identical per-parameter metrics.
- **Medium anchor:** KNO (avg 4.75) and PENO (avg 5.0) suffered from unfair comparisons, weak baselines, or limited experimental scope, but their reported numbers were at least mathematically plausible. The paper under review is weaker than these because metric computation errors and apparent data mishandling in Figure 2 directly attack the integrity of the empirical record.
- **Low anchor:** Hartley Neural Operators (avg 3.0) failed to improve over FNO on key benchmarks and offered limited experimental scope. The paper under review is stronger than this anchor because its low-dimensional results (Tables 1–2, Figure 1, Figure 3) do show clear, meaningful improvements and address an important problem.

The paper has a genuine and well-motivated core idea, but the experimental record is too compromised by the impossible $R^2$ values in Tables 3–4, the implausibly identical per-parameter inversion metrics in Figure 2, and the unsubstantiated 82-parameter inversion claim to support acceptance. These issues are not minor clarifications; they strike at the credibility of the central empirical claims. I recommend **Reject** with encouragement to resubmit after correcting the metrics, providing verifiable per-parameter inversion results, and aligning the abstract's scalability claims with the actual experiments.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>