=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary
The paper studies flow matching for generative modeling on convex domains via mirror maps. Its main proposal is to pair a **regularized mirror map**—a modified log-barrier plus quadratic term that is strongly convex and controls the dual-space tail behavior—with a **Student-\(t\) prior** in the dual space, motivated both by training stability and by obtaining regularity/convergence guarantees. The paper also provides theory for Euclidean flow matching with \(t\)-priors under polynomial-tail assumptions and lifts those guarantees back to the constrained primal space, alongside synthetic constrained experiments and a watermarking image-generation application.

## Strengths
- **A specific and nontrivial co-design of geometry and prior.** The paper does more than “use a mirror map for constraints”: it identifies a concrete pathology of standard log-barriers (heavy-tailed dual distributions with potentially missing moments) and addresses it with a particular regularized potential
  \[
  \Psi(x)= -\sum_i \frac{(-\phi_i(x))^{1-\kappa}}{1-\kappa} + \frac12\|x\|^2,
  \]
  for which Proposition 2.2 establishes both strong convexity (\(\nabla^2 \Psi \succeq I\)) and a tail transfer bound \(P(\|\nabla \Psi(X)\|\ge R)\lesssim R^{-\beta/\kappa}\). This is a concrete mechanism, not just a generic appeal to mirror geometry.
- **The paper gives a coherent reason for Student-\(t\) priors beyond empirical preference.** Section 2.2 and Proposition 4.1 tie the prior choice to regularity of the FM vector field, rather than presenting \(t\)-priors as a heuristic. In particular, the main theoretical claim is not merely existence of a flow, but spatial Lipschitzness and a bound on \(\partial_t v_D\) on \(t\in[0,T]\), which is then used for discretization error in Theorem 3.
- **The primal/dual formulation is conceptually clean.** Proposition 3.1 makes explicit that straight-line interpolation in the dual corresponds to geodesic interpolation in the primal under the squared Hessian metric, giving a principled constrained-generation interpretation rather than an ad hoc “map out / map back” pipeline.
- **The synthetic constrained results do support the claim that both the mirror-map choice and the prior matter.** Although a more explicit factorial ablation would help, the current evidence is still meaningful: the appendix visualizations compare log-barrier vs proposed mirror map and Gaussian vs \(t\)-flow, and Figure 5 shows that neither log-barrier variant works well while \(t\)-flow with the proposed map recovers the target substantially better than Gaussian flow with the same map.
- **The real-data watermarking result is stronger than a toy demo.** Table 3 reports a sizable improvement over MDM under EDM checkpoint initialization (FID 4.27 vs 7.29; CMMD 0.023 vs 0.170) with shorter training time, suggesting the approach is not limited to low-dimensional synthetic settings.

## Weaknesses

### Major:
- **The main theory applies only under a fairly restrictive tail assumption whose severity grows with dimension, which weakens the paper’s “heavy-tailed” scope.** Assumption 3 requires \(\pi_1^D(x)\le C\|x\|^{-\alpha}\) for large \(\|x\|\), with Proposition 4.1 and Theorem 3 requiring \(\alpha \ge 2d+\nu+2\). This does still allow polynomial tails, so it is not a contradiction of the paper’s premise; however, it is a strong condition in moderate/high dimensions, and the paper does not adequately discuss how realistic this regime is for the kinds of high-dimensional generative tasks it motivates. The theory is therefore meaningful but narrower than the broad framing suggests.
- **The empirical evaluation does not fully isolate the contribution of the two proposed ingredients.** The paper’s central message is that *both* a regularized mirror map and a Student-\(t\) prior are needed. Yet the main tables compare “Mirror t-Flow” to “Mirror G-Flow” and to external baselines, while the most direct factorial ablation—e.g., regularized map + Gaussian, regularized map + \(t\), log-barrier + Gaussian, log-barrier + \(t\)—appears only partially through appendix visualizations rather than a systematic main-text study. Because the paper’s claim is explicitly about co-design, a cleaner decomposition of the gains would materially strengthen the evidence.
- **The benchmark scope is limited relative to the breadth of the claimed applicability to general convex domains.** The method is motivated with examples including polytopes, simplices, PSD matrices, robotics, and molecular settings, but the experiments cover two synthetic geometries (polytope and \(L_2\) ball) and one specialized real-world application (watermarked AFHQv2 images using polytope constraints). This is enough to show promise, but not enough to substantiate the broader claim of effectiveness across general convex-domain generative modeling.
- **The practical cost and stability of the inverse mirror map are underexplained.** The algorithm requires mapping back via \(x=\nabla \Psi^*(z)\), i.e., inverting \(\nabla\Psi\). The paper correctly notes, in discussing MDM, that inverse maps can be difficult for general polytopes; however, it gives little detail on how this inversion is performed for the proposed mirror map in the reported experiments, how expensive it is, or how numerical errors affect feasibility and sample quality as dimension/constraint count increase. Since exact feasibility in practice depends on this step, more implementation detail and scaling analysis would be valuable.
- **The early-stopping tradeoff is theoretically acknowledged but not practically characterized.** Theorem 3 includes an explicit early-stopping term proportional to \(1-T\), so the paper does not ignore the bias. Still, \(T<1\) is central to the method’s stability story, and the experiments do not provide a clear empirical study of how close one can push \(T\) to 1 before instability appears, or how this tradeoff interacts with \(\kappa,\nu\), and dimensionality. That leaves an important practical knob somewhat under-analyzed.

### Minor
- **Hyperparameter sensitivity appears real and deserves more systematic guidance.** Figure 3 and the surrounding discussion indicate nontrivial dependence on \(\kappa\) and \(\nu\), with larger \(\kappa\) inducing heavier dual tails and the need to balance \(\nu\) accordingly. This is not a flaw in itself, but the paper stops short of giving actionable rules for selecting these parameters beyond qualitative remarks.
- **The real-data evaluation is only at \(64\times 64\).** This does not invalidate the result, especially given the constrained watermarking setup, but it limits how strongly one can generalize the practical significance to larger-scale modern image generation.
- **Some of the presentation around the theory is harder to parse than necessary.** In particular, the primal-space section leans on Riemannian/isometry language and appendix formalism that may obscure the key message. The content seems correct, but a simpler exposition would help readers assess what is genuinely new versus inherited from the dual analysis.

### Trivial
- **The paper would benefit from explicitly reporting whether feasibility is exact up to numerical precision after inverse mapping and Euler discretization.** Tables report 100% feasibility for the proposed method, which is good; a brief clarification of the numerical tolerance/check would make this more informative.

## Nice-to-Haves
- Add a **systematic factorial ablation** separating the regularized mirror map from the Student-\(t\) prior in the main paper.
- Include at least one additional **non-polytope constrained benchmark** (e.g., simplex or PSD-valued data) to better support the “general convex domains” narrative.
- Report empirical diagnostics tied to the theory: e.g., velocity norm distributions over \(t\), empirical dual-tail plots beyond 2D projections, or sensitivity curves for \(T\).
- Provide more implementation detail for **inverse mirror-map computation** and its runtime/scaling with dimension and number of constraints.
- Clarify how \(\kappa\), \(\nu\), and \(T\) were selected in practice, and whether there are reliable default settings.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Gauge Flow Matching is unfairly omitted for the \(L_2\) ball task.”** The paper explicitly states: “For the \(L_2\) ball case, Gauge Flow Matching is omitted since it coincides with Reflected Flow Matching.” This is a substantive explanation, so the criticism should not be kept.
- **“Theorem 4 is weak because it is mostly a composition of earlier results.”** This is true descriptively but not a substantive weakness by itself; lifting dual-space guarantees back to the primal constrained space is part of the paper’s stated goal, and the theorem does deliver that.
- **“The 1D Example 2 is insufficient because it lacks a formal high-dimensional analogue.”** As a criticism of exposition this is fair, but not as a substantive flaw. The paper uses Example 2 as intuition, not as the sole basis of the theory.
- **“The claim that no prior framework ensures constraint satisfaction while providing convergence rates is overstated.”** Without external verification this cannot be confidently assessed, and the paper’s actual contribution is more specific anyway.
- **Generic complaints about missing standard image benchmarks or higher resolutions.** These are too broad on their own. The kept version is the narrower, verified point that the current benchmark suite is limited relative to the paper’s own broad claims.
- **Formatting/parser artifacts or proof-style nitpicks.** Excluded per instruction.

## Novel Insights
The most compelling synthesis is that the paper’s real contribution is not just “mirror flow matching for constraints,” but a **tail-aware redesign of the entire constrained FM pipeline**: the mirror map shapes the dual-space tail behavior, and the prior determines whether the conditional FM velocity field remains regular enough to analyze and train. This shifts the constrained-generation design problem from purely geometric feasibility to a joint **geometry–tail compatibility** problem. At the same time, the paper’s theory reveals a tension the authors should foreground more clearly: while the method is motivated by heavy tails, the current regularity guarantees still require tails to be sufficiently light relative to dimension, so the practical promise presently outstrips the formal scope.

## Suggestions
- Add a main-paper ablation matrix crossing **{log-barrier, regularized mirror map} × {Gaussian, Student-\(t\) prior}** on at least one synthetic task, with both sample metrics and velocity/dual-tail diagnostics.
- Discuss the practical meaning of Assumption 3 more candidly: where it is likely to hold, where it may fail, and whether the experiments are intended as evidence beyond the present theory.
- Provide a concrete algorithmic description for computing \(x=\nabla\Psi^*(z)\) for general constraints, along with runtime and stability measurements.
- Include an empirical study of the **early-stopping parameter \(T\)** showing quality/stability tradeoffs and interaction with \(\kappa,\nu\).
- Broaden the empirical scope with at least one additional constrained domain beyond polytopes/\(L_2\) balls and one more realistic high-dimensional constrained dataset if feasible.
- Improve exposition in Sections 4.1–4.2 by separating the genuinely new analytical step (regularity under \(t\)-priors) from the geometric lifting argument back to the primal space.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
