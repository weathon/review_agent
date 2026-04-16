Now I have sufficient calibration data. Let me compile the final review.

## Summary

This paper proposes NFFS (Neural Functional Flow on Surface), a framework for simulating incompressible fluid flow on geometric surfaces using neural implicit representations. The method constructs a divergence-free velocity field on surfaces by combining the Closest Point Method (CPM) with exterior calculus to parameterize a stream function via a neural network (Theorem 3.1), and advances the flow using a covariant-derivative-based advection scheme. The framework supports multiple surface representations—analytic, explicit mesh, and implicit neural surfaces—and is demonstrated on simulation, generation, and Helmholtz decomposition tasks.

## Strengths

- **Novel and principled problem setting:** Addressing incompressible surface flow on implicit neural representations is genuinely underexplored, and the combination of CPM with exterior calculus for a neural divergence-free construction is technically interesting. The paper claims to be the first to present simulation results of incompressible fluid on implicitly neural-represented surfaces with guaranteed divergence-free behavior, which appears to be correct.

- **Elegant theoretical framework:** The connection between differential forms (divergence-free = *dμ), the Closest Point Method for surface transport, and neural parametrization of σ to yield v(x) = j^*((∇(cp*σ) ∘ j(x)) × n(x)) is a coherent and principled construction. It naturally eliminates the need for pressure projection by enforcing incompressibility by design.

- **Memory efficiency advantages demonstrated:** Table 1 shows the method achieves significantly lower MSE (2.89e2) than PINN (1.73e5) and INSR (8.63e4) at comparable storage (~530KB), which is a meaningful comparison. The 5× memory savings claim vs. classical GT (2643KB) is also supported.

- **Practical robustness on implicit surfaces:** The observation that classical mesh-based solvers fail to converge on Armadillo/Lucy INR surfaces while NFFS produces smooth results is a genuinely useful finding, supported by supplementary analysis of crash times across resolutions.

- **Versatility beyond simulation:** The conditioning/generation experiment (Sec. 5.3) and Helmholtz decomposition (Sec. 5.4) demonstrate that the divergence-free parametrization has useful applications beyond forward simulation.

## Weaknesses

### Major:

- **Energy preservation claims are overstated relative to the method presented.** The paper repeatedly claims "low energy dissipation" and "energy preservation" (Abstract, Sec. 1, Sec. 4.1), but the advection scheme (Eq. 15) is derived by taking a **first-order approximation** of the exponential map (Eq. 14) and solving via per-step optimization with Adam (Eq. 16). This introduces both truncation error (first-order Taylor approximation) and optimization error (finite stochastic gradient steps), and there is **no theoretical guarantee** that the discrete scheme preserves energy, enstrophy, or circulation—even in the limit of vanishing optimization error. The appendix energy plots provide empirical evidence of low dissipation on specific examples, but this falls short of the broad claims made in the abstract and introduction. The comparison to Azencot et al. (2014) is partially misleading: Azencot's discrete exterior calculus operators do have provable conservation properties on meshes, while here the formal continuum expression is borrowed and then regressed through a neural network.

- **No numerical divergence error reported anywhere.** The core selling point of the framework is that the velocity field is divergence-free **by construction** via Theorem 3.1. However, in practice, the field is computed by a neural network approximating σ, gradients are obtained via automatic differentiation, and normals are estimated. None of these are exact. Despite this, the paper never reports ‖∇·v‖ over the surface at any time step for any method, making it impossible to verify the central structural claim holds numerically. This is a critical omission for a paper whose primary differentiator is exact incompressibility.

- **No ablation studies isolating key design choices.** It is unclear how much of the accuracy gain comes from: (a) the divergence-free parametrization vs. a naïve velocity field with a divergence penalty (as in PINN/INSR), (b) the specific covariant-derivative advection scheme vs. a simpler explicit/Euler advection, or (c) the CPM-based surface representation vs. direct parameterization. The "15× accuracy" claim conflates all these choices; without ablations, it is impossible to attribute the improvements.

- **Evaluation is narrow on the most novel cases.** Quantitative results (Table 1, Fig. 5) are limited to two canonical analytical surface problems (sphere jet, inclined plane Taylor vortices). On the most novel and practically important cases—explicit meshes and implicit neural surfaces (Sec. 5.2)—the paper provides **only visual results with no quantitative metrics and no neural baselines**. The footnote explaining the absence of PINN/INSR comparisons on meshes is understandable, but it leaves the strongest claimed contribution (simulation on INR surfaces) without quantitative validation.

### Minor:

- **Topological handling is incomplete.** The paper acknowledges (footnote 1 and Sec. 4.1) that non-zero homology requires an additional harmonic field η, but: (a) no constraint enforces that η is actually harmonic (i.e., Δη = 0), so the combined field v_σ + η may not be divergence-free; (b) all tested surfaces (sphere, plane, hand, spot, armadillo, Lucy) are genus-0, so the harmonic component is never meaningfully exercised. This limits confidence in the method's generality.

- **Computational cost is high and under-discussed.** Table 1 shows NFFS takes 16.5h vs. 0.8h for Small-F.S. (~20× slower) and even 2× slower than the ground-truth reference solver. The paper mentions optimization per-step via Adam but doesn't specify convergence criteria, number of iterations per step, or how these scale with surface complexity.

- **The proof of Theorem 3.1 is underdeveloped.** The "proof sketch" mostly restates the construction without rigorously showing how the CPM pullback preserves divergence-freeness on the surface. Key details—what specific (n−2)-form μ corresponds to σ, how the 3D curl/gradient/cross-product maps to surface differential forms, and the precise role of the closest-point extension—are left implicit or deferred to "similar discussions as Richter-Powell et al. (2022)."

- **Conditioning experiment (Sec. 5.3) is decoupled from physical validation.** The EMNIST-based vorticity generation produces visually appealing patterns but no PDE residual or incompressibility check, making it a proof-of-concept that doesn't reinforce the fluid simulation claims.

## Nice-to-Haves

- Report **numerical divergence error** ‖∇·v‖ over the surface vs. time as a standard metric across all experiments—this would directly validate the paper's central structural claim.

- Add a **proper ablation** comparing: (i) divergence-free parametrization + covariant advection (full method), (ii) penalty-based divergence + same advection, (iii) divergence-free parametrization + simple explicit Euler advection, to isolate what drives the accuracy gains.

- Provide **energy conservation curves** (kinetic energy vs. time) in the main paper, not just the appendix, for all simulation tasks.

- Include at least one **quantitative comparison on a non-trivial surface** (even if only against a high-resolution reference computed on a mesh).

- Analyze **scaling behavior**: how does accuracy/compute change with number of surface samples, network size, or time step refinement?

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released / availability concerns" about baselines or tools:** The cited methods (PINN, INSR, Functional Fluids on Surfaces, etc.) are treated as real and available; no availability concerns are valid.

- **"Unfair comparison favoring authors" claim regarding baselines:** The harsh critic suggested baselines were configured unfairly. However, the paper compares against PINN and INSR (which genuinely struggle with surface flows as acknowledged even by those methods' scopes) and uses Small-F.S. at matched storage with GT at 5× storage, which is a reasonable setup. The criticism about dropping neural baselines on INR surfaces has some merit but is already acknowledged by the paper (footnote 2) and is a scope limitation rather than an unfair advantage.

- **Pure formatting/style nitpicks:** The notation choices (e.g., calling cp* a "pullback" when it acts as an extension) and the organization of the proof sketch, while imperfect, are not fatal flaws that invalidate the work.

- **Demanding viscosity and boundary conditions as core contributions:** The paper explicitly scopes to inviscid flow (Euler equations) and notes viscosity as future work. Criticizing the absence of boundary conditions and viscosity is scope creep; the paper does not claim to handle these.

- **Demanding theoretical convergence analysis:** This is an empirical/frameworks paper in its core contribution (new simulation framework for surfaces). A full convergence proof would be valuable but is not standard for this type of contribution.

## Novel Insights

The most interesting insight emerging from this work is that the Closest Point Method, combined with a stream-function-based neural parametrization, can serve as a practical bridge between the continuous differential-forms perspective on surface PDEs and the neural implicit representation paradigm—enabling simulation directly on SDFs without meshing. The observation that classical mesh-based surface fluid solvers crash on INR-extracted meshes (Appendix E.4) highlights a genuine practical limitation of current approaches that NFFS sidesteps. However, the claim that this yields "energy preservation" is an overreach: what the method achieves is a divergence-free construction that reduces (but does not eliminate) energy dissipation relative to projection-based approaches, via a first-order implicit advection scheme.

## Suggestions

1. Add a table reporting **numerical divergence error** for each method across time steps—this is the single most important missing experiment.
2. Tone down language: replace "energy-preserving" with "reduced energy dissipation" or "approximately energy-preserving" throughout, and state explicitly that the advection is a first-order implicit scheme.
3. Include at least one ablation separating the divergence-free construction from the advection scheme.

## Score and Decision

**Calibration anchors:**

- **clawNOs (Reject, avg ~5)**: Divergence-free neural construction with overclaimed benefits, limited experiments, missing divergence metrics—very similar weakness profile.
- **HelmSim (Reject, avg ~5)**: Helmholtz decomposition for fluid learning, limited baselines on novel claims, data-driven rather than physics-guaranteed.
- **LFlows (Accept-spotlight, avg ~7.3)**: Conservative neural flow with provable PDE satisfaction, well-validated theory, clean experiments—stronger theoretical grounding than NFFS.
- **Symmetric Basis Convolutions (Accept-poster, avg ~5.75)**: Solid engineering contribution with good experiments but limited scope.
- **Accurate Differential Operators (Reject, avg ~5)**: Neural field operators with mixed novelty and validation concerns.

NFFS has genuine novelty in its CPM+exterior calculus+INR combination and demonstrates compelling results on a previously unsolved problem (flow on INR surfaces). However, it shares the same pattern as clawNOs: a divergence-free construction that is structurally elegant but whose quantitative validation is insufficient (no divergence error metrics, limited ablations, overclaimed energy preservation). The paper is somewhat stronger than clawNOs in that the problem setting (surface flow on INRs) is more novel and the theoretical construction is more substantial, but it is weaker in that the core claims about energy conservation and accuracy are less rigorously validated. Relative to HelmSim (score ~5), NFFS has a more principled theoretical foundation but similar issues with experimental validation scope. Overall, this paper falls below the acceptance threshold primarily due to the gap between its strong claims and its empirical validation, though the core idea is worthy of further development.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>