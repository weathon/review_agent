## Summary

The paper proposes Derivative Learning (DERL), a method that trains neural networks to represent solutions of dynamical systems and PDEs by matching the partial derivatives of the target solution plus initial/boundary conditions, without requiring the PDE structure itself. The key insight is that individually targeting derivatives decouples the optimization objective compared to the entangled PDE residual used by PINNs. The paper also introduces derivative distillation for transferring physical knowledge between models, showing that higher-order derivative distillation improves physical consistency.

## Strengths

- **Novel derivative distillation framework with demonstrated improvements**: Section 4.5 introduces derivative-based distillation for transferring physical knowledge between architecturally different models. On the KdV equation (Table 6), distilling from a PINN teacher to DERL/HESL students yields a BC loss reduction of over an order of magnitude (teacher: 0.33532 → student: 0.014197) while maintaining comparable PDE residual and solution accuracy. This is the paper's most original contribution and, to our knowledge, the first such attempt.

- **Clean conceptual insight about decoupled derivative objectives**: Figure 1 clearly illustrates that DERL treats each partial derivative as an independent target rather than entangling them in a single PDE residual like PINNs. This provides a well-motivated explanation for why DERL optimizes more easily — each gradient component has a direct target rather than a coupled constraint.

- **Higher-order derivative distillation demonstrably improves physical consistency**: Table 6 shows that DER+HESL (learning both ∇u and Hessians) achieves the best derivative matching (0.041988), best Hessian matching (0.85280), and strong PDE residual (0.19317), compared to DERL alone (PDE loss 0.32480) and OUTL (PDE loss 17.366). This directly validates the claim that distilling higher-order derivatives improves physical consistency.

- **DERL works with empirical derivatives**: The continuity equation experiment (Section 4.3) uses finite difference approximations (Δx = Δy = Δt = 0.01) without interpolation, yet DERL still achieves competitive performance. This is practically important since analytical derivatives are often unavailable.

- **Reasonable experimental breadth**: The evaluation spans ODEs (pendulum, E1), time-independent PDEs (Allen-Cahn, E2), time-dependent PDEs (continuity, E3), systems of PDEs (Navier-Stokes, E4), and distillation on third-order PDEs (KdV, E5) and Euler equations (E6), providing evidence across problem complexity classes.

## Weaknesses

### Fatal

None.

### Major

- **DERL and PINNs solve fundamentally different problems, but the paper conflates them**: DERL requires the true derivatives Du at interior collocation points (computed from the known solution via analytical formulas or numerical simulation), while PINNs require only the PDE structure + IC/BC and never access the solution at interior points. For well-posed problems, both types of information are sufficient to uniquely determine the solution, but they are different inputs. The paper frames DERL as outperforming PINNs (Abstract: "DERL outperforms PINNs"), implying they compete on the same task, but they receive qualitatively different information. The paper should clearly position DERL as solving a *neural approximation problem* (given derivative data + IC/BC) rather than a *forward PDE problem* (given PDE + IC/BC), and the comparison with PINNs should be reframed as comparing optimization landscapes under different information regimes, not as a direct performance contest.

- **Factually misleading claim "without ever seeing any data for t > 0"**: In Section 4.3, the paper states DERL "correctly learns the complete solution *without ever seeing any data for t > 0*." This is false — DERL's loss (Eq. 3) includes the derivative term ||Dû − Du||² over the full time-space domain [0,T] × Ω, meaning DERL sees derivative values at all interior points including t > 0. Derivative data is data. The paper's more carefully worded statement in Section 4.4 ("without having access to the solution in the interior") is accurate (DERL doesn't see function *values* in the interior), but the "no data for t > 0" claim exploits an artificial distinction between function values and derivative values to claim a surprising result that does not hold.

- **Unclear practical motivation for when DERL applies vs. PINNs**: The paper does not clearly articulate scenarios where one would have access to spatially and temporally dense derivative data but not the PDE itself. In practice, derivatives come from either (1) analytical solutions (trivial case), (2) high-resolution numerical simulations (but then why re-learn the solution as a neural network?), or (3) distillation from pre-trained models (the most compelling case, but limited in scope). The paper should honestly delineate when DERL is the right tool versus when PINNs are more appropriate, rather than presenting DERL as a universal alternative to PINNs.

### Minor

- **Selective presentation of mixed results**: In the continuity equation (Table 4), OUTL achieves lower L² error than DERL (0.02793 vs. 0.02883), and PINN achieves lower PDE residual (0.04107 vs. 0.07338). In Navier-Stokes (Table 5), OUTL again has lower L² error (0.01195 vs. 0.02169). Yet the paper claims DERL is "the most effective method" for the continuity equation. While the paper does note DERL is "comparable to OUTL" and "second best for PDE consistency," the overall framing overstates DERL's advantage. DERL's strength is primarily in physical consistency metrics, not L² accuracy, and this should be reflected more honestly.

- **Theoretical results are straightforward consequences of Poincaré-type inequalities**: Theorems 2.1–2.3 prove that matching derivatives + IC/BC yields the correct function, which follows from standard Poincaré inequality / fundamental theorem of calculus results. The bound ||û−u|| ≤ 2(C+1)ε in Theorem 2.2 provides a quantitative estimate, but the theorems say nothing about whether a finite-capacity network on finite data can achieve small ε, which is the practically relevant question.

- **No variance or confidence intervals reported**: All tables report single-run results to five decimal places with no standard deviations. Given the stochastic nature of neural network training and the small absolute differences between some methods (e.g., Allen-Cahn Table 3: DERL PDE L² = 0.0096 vs. SOB = 0.0165), statistical significance is unclear.

### Trivial

None significant.

## Nice-to-Haves

- A comparison where OUTL is given the same collocation points (function values where DERL sees derivatives) would isolate the effect of derivative vs. output learning with equal information, directly testing the paper's core claim.

- Analysis of failure modes: when does DERL fail? What happens with noisy or sparse derivative data, irregular domains, or chaotic systems?

- Training dynamics/loss curves comparing DERL vs. PINN convergence to substantiate the "simpler to train" claim beyond final accuracy numbers.

- A multi-teacher compositional distillation experiment to support the conclusion's vision of "continual composition and integration of physical information across different models."

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"DERL receives strictly more information than PINNs"** (Harsh Critic point #1): While DERL does receive different information than PINNs, the characterization as "strictly more" is inaccurate. For well-posed problems, both the PDE + IC/BC and derivatives + IC/BC are sufficient to uniquely determine the solution — they are informationally equivalent but qualitatively different. The real issue is that the paper conflates these different problem setups, not that DERL gets "more" information.

- **HNN/LNN comparison is misleading** (Harsh Critic Section 4.1 note): The paper explicitly states "We train them on the conservative part of the field" and notes HNN/LNN are "specifically designed for conservative fields." This transparency allows readers to interpret the results appropriately. The comparison shows DERL's broader applicability to non-conservative systems, which is a valid point even if the setup disadvantages HNN/LNN on their home turf.

- **"We are the first to consider a pure derivative approach" overlooks Sobolev training** (Harsh Critic Section 3 note): The paper includes SOB as a baseline and the distinction between "pure derivative" (λ for output = 0) and "derivative + output" (SOB) is valid and practically meaningful, as the experimental results show SOB performs differently from DERL.

- **Criticism of conclusion's "compositional learning" vision as "speculative and unsupported"**: Conclusions are expected to speculate about future directions. The paper's distillation experiments provide a foundation for this vision, even if they don't fully demonstrate it.

- **Strength "DERL dramatically outperforms PINNs on time-dependent PDEs"**: While numerically true, this strength is weakened by the fact that DERL and PINNs receive different inputs. Retained in the review as part of the experimental evidence but the framing issue is noted as a Major weakness.

- **Strength "General-purpose DERL outperforms physics-specialized methods on non-conservative systems"**: This is weakened because HNN/LNN are disadvantaged by being applied to non-conservative systems. Removed from main strengths.

- **Criticism about reproducibility / undisclosed hyperparameters**: The paper states hyperparameters are independently tuned and details are in Appendix A/D. Standard for the field.

## Novel Insights

The most insightful observation that emerges from synthesizing the reviews with the paper is that DERL's core contribution is not about being "better than PINNs" but about revealing a fundamental trade-off in how physical information is encoded in training objectives. The PDE residual (PINN's approach) and the derivative targets (DERL's approach) are both sufficient to determine the solution for well-posed problems, but they create radically different optimization landscapes: PINNs require satisfying a coupled constraint while DERL decomposes into independent per-derivative regression. The derivative distillation experiments (Section 4.5) are the clearest illustration of this principle — they work precisely because the teacher model's derivative outputs can serve as independent targets for the student, avoiding the optimization difficulties that would arise from re-imposing the PDE as a constraint. This suggests a broader design principle: when transferring or composing physical knowledge across models, derivative-based objectives may be more effective than constraint-based objectives, regardless of whether one method is "better" in absolute terms.

## Suggestions

- Rewrite the abstract and introduction to clearly position DERL as solving the *neural approximation problem with derivative data* rather than competing with PINNs on the forward PDE problem. State explicitly that DERL requires access to the solution's derivatives (from simulation, analytical formulas, or distillation), while PINNs require the PDE structure but no interior data.

- Correct the "without ever seeing any data for t > 0" claim to "without ever seeing function values at interior points for t > 0" — derivative data at t > 0 is still data.

- Add a focused discussion (even 1–2 paragraphs) of practical scenarios where DERL is applicable: (1) compact neural representation of expensive simulations, (2) knowledge distillation between physics-informed models, (3) systems where derivative measurements are available but the governing equations are unknown.

## Score and Decision

**Calibration anchors compared:**

- **Low anchors**: PDE-Diffusion (avg 2.2) — fundamentally flawed methodology and presentation; EPINN (avg 2.5) — overclaimed speed with no baselines; DeepFDM (avg 3.5) — unfair comparison where PDE structure was known to proposed method but not baselines. DERL is clearly better than these: it has a valid core idea, genuine experimental evidence, and a novel distillation contribution.

- **Medium anchors**: Sobolev acceleration (avg 4.5) — shallow theory on derivative-based training, limited practical validation; Hybrid Numerical PINNs (avg 3.33) — overclaimed speedup, insufficient baselines; Efficient Discrete PINNs (avg 4.0) — limited novelty, underwhelming compared to numerical methods. DERL is comparable to or slightly above these: it has more experimental breadth and the novel distillation angle, but faces similar overclaiming issues.

- **High anchors**: PhyMPGN (avg 8.0) — strong experimental results, comprehensive evaluation, clear novelty; AAS (avg 7.25) — novel minmax formulation with good theory; KAN (avg 7.2) — novel architecture with strong empirical results. DERL is clearly below these: the framing issues and unclear practical applicability prevent it from reaching this level.

DERL sits between the medium and high-scoring anchors for papers with genuine contributions but significant framing issues. The distillation experiments push it above purely theoretical contributions like Sobolev acceleration, but the overclaiming about PINN comparisons and the misleading "no data" claim pull it below the 6 threshold. The DeepFDM case (3.5) involved a clearly unfair comparison; DERL's situation is more nuanced since both DERL and PINN receive informationally sufficient but qualitatively different inputs, so DERL deserves a higher score than DeepFDM.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>