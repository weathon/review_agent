## Summary

This paper proposes NFFS (Neural Functional Flow on Surface), a neural implicit framework for simulating incompressible fluid on geometric surfaces. The core idea combines the Closest Point Method (CPM) with exterior calculus to construct divergence-free velocity fields by construction via a parameterized stream function (Theorem 3.1, Eq. 4), eliminating the need for pressure projection. A covariant-derivative-based advection scheme (Eq. 15) is derived from the implicit midpoint rule to minimize energy dissipation. The framework operates on surface samples rather than meshes, enabling simulation on implicit neural representations (e.g., SDFs) where classical mesh-based solvers fail to converge.

## Strengths

- **Hard enforcement of divergence-free property by construction**: Theorem 3.1 constructs velocity as $v(x) = j^*((\nabla(cp^*\sigma) \circ j(x)) \times n(x))$ (Eq. 4), which is provably divergence-free via $d^2 = 0$. This architectural guarantee yields dramatic accuracy improvements over soft-constraint methods: Table 1 reports MSE of 2.89e2 for NFFS vs. 8.63e4 for INSR and 1.73e5 for PINN on sphere jet flow — roughly two orders of magnitude lower error at comparable storage (~530 KB).

- **First demonstration of divergence-free fluid simulation on implicit neural surfaces**: Fig. 7 shows jet flow results on Armadillo and Lucy represented as SDFs (Siren/DeepSDF), while Section 5.2 reports that the classical Functional Fluids on Surfaces method fails to converge on meshes extracted from these representations (with crash details in Appendix E.4). This fills a concrete capability gap for the neural geometry ecosystem.

- **Principled energy-preserving advection via covariant derivatives**: The advection loss in Eq. 15, derived from the implicit midpoint rule on covariant derivatives (Eqs. 12–14), avoids the velocity advection + pressure projection pipeline that causes energy dissipation in classical and neural baselines. Fig. 5 (Taylor vortices) qualitatively shows that INSR and PINN suffer visible energy dissipation while NFFS preserves vortex structure.

- **Versatility across geometry representations and downstream tasks**: The framework handles analytical surfaces (sphere, plane), explicit meshes (hand, spot), and implicit neural surfaces (Armadillo, Lucy) with the same construction. It also supports Helmholtz decomposition on real atmospheric data (Fig. 8b) and conditional vorticity generation via a VAE (Fig. 8a).

## Weaknesses

### Fatal
None.

### Major

- **No ablation study isolating the two claimed contributions**: The paper claims two distinct contributions: (i) the CPM/exterior-calculus divergence-free construction (Theorem 3.1) and (ii) the covariant-derivative advection scheme (Section 4.1). There is no experiment testing them independently. It is impossible to determine whether the accuracy gains over baselines come from the divergence-free-by-construction property, from the advection scheme, from the neural representation's continuity, or from spending 16.5 hours of optimization (vs. 0.8h for Small-F.S. in Table 1). Without ablation, the contribution cannot be properly attributed — this is a significant gap for a paper claiming two contributions.

- **Headline claims rest on a single quantitative experiment; other experiments are qualitative only**: The "approximately 15 times higher accuracy" and "5 times memory savings" claims (Abstract, Contributions) are derived entirely from the sphere jet experiment in Table 1. The Taylor vortex experiment (Fig. 5) and the mesh/implicit surface experiments (Figs. 6–7) present only qualitative visual comparisons with no quantitative error tables in the main text. Section 5.2 even excludes comparison with INSR and PINN (footnote 2). For a paper making strong quantitative claims in its abstract, having only one experiment with numerical error metrics is insufficient. The quantitative results referenced in Appendices E.2–E.3 would strengthen the paper if included in the main text.

- **Energy preservation claim lacks evidence in main experiments**: The paper claims "energy-preserving" simulation as a key contribution (Abstract: "energy preservation"; Contributions: "low energy dissipation"), yet the only energy-over-time evidence is a sphere rotation case in Appendix E.1. The main experiments (sphere jet, Taylor vortices, mesh flows) provide only qualitative visual evidence that vortices are preserved. For a method whose central differentiator is energy preservation, this should be demonstrated quantitatively in the primary experiments — e.g., kinetic energy vs. time step curves alongside the baselines.

### Minor

- **20× computational cost tradeoff not discussed**: Table 1 shows NFFS requires 16.5h versus 0.8h for Small-F.S. — approximately 20× more compute for ~18× better accuracy. The paper does not discuss this tradeoff or analyze what drives the cost (optimization iterations per time step, sample count, network forward/backward passes). The paper acknowledges "time efficiency" as a limitation (Section 6) but defers details to the appendix. A brief cost analysis in the main text would help readers assess practical viability.

- **Section 5.2 lacks quantitative evaluation**: The explicit mesh (hand, spot) and implicit surface (Armadillo, Lucy) experiments are shown only through qualitative visualizations. No error metrics are reported, making it difficult to assess how well the method quantitatively performs beyond analytical surfaces. This is particularly important since these experiments showcase the paper's claimed advantage of geometry adaptivity.

### Trivial
None.

## Nice-to-Haves

- Accuracy-vs-storage Pareto curves comparing NFFS at multiple network sizes against Functional Fluids at multiple resolutions would replace the single-point comparison with a proper characterization of the memory-accuracy tradeoff.
- Reporting optimization statistics (iterations per time step, typical $\mathcal{L}_i$ residual values after convergence) in the main text would strengthen reproducibility and allow assessment of whether the energy-preservation property holds in practice despite approximate optimization.
- Energy-over-time curves for all primary experiments would directly substantiate the "low energy dissipation" claim.

## Removed Points

*These points are flagged to be removed, treated with caution.*

- **Incomplete proof of Theorem 3.1**: The harsh critic flagged the "proof sketch" as incomplete. However, the paper provides a proof sketch in the main text (line 123) and likely contains a full proof in the appendix (Appendix D/E references), which was stripped during parsing. Per review rules, weaknesses about missing appendix proofs are removed.

- **Reproducibility concerns about Adam hyperparameters and iteration counts**: The critic demanded reporting of learning rates, number of Adam iterations per step, and convergence behavior. Per review rules, nitpicks about undisclosed hyperparameters and implementation details are removed. (The concern about not discussing computational cost is retained above as a minor weakness since it affects the practical assessment of the method, not just reproducibility.)

- **"Eliminates velocity advection" is misleading**: The critic claimed this is "a reframing, not an elimination." The paper is technically correct — it advects vorticity (not velocity) and does not need pressure projection. The vorticity advection is a different operation from velocity advection, and the elimination of pressure projection is genuine. The wording could be more precise but is not misleading.

- **Introduction Eq. 1 written as 3D Euler equations**: The critic noted this is misleading because surface Euler equations have different forms. The paper motivates with the 3D form and then transitions to the vorticity formulation (Eq. 6) for the surface case. This is a common presentation choice and the paper correctly handles the surface dynamics through the vorticity formulation.

- **VAE experiment is a weak test of the simulation framework**: The paper presents this as a conditioning/generation application (Section 5.3), not as a test of simulation quality. Criticizing it for not testing simulation quality misses its stated purpose.

- **Helmholtz decomposition lacks ground truth**: This is presented as an application demo, not a quantitative evaluation. It demonstrates a practical use case.

- **Topology handling mentioned in passing / harmonic component is significant complication**: The paper addresses this in Section 4.1 with a dedicated MLP $\eta$ for harmonic components and references Appendix F.1 for further discussion. The treatment is reasonable for the paper's scope.

- **NFFS abbreviation not defined in abstract**: Minor presentation nitpick; the term is defined in the contributions section immediately after.

## Novel Insights

The paper's approach of combining CPM with exterior calculus for neural surface fields is genuinely novel — it bridges the discrete exterior calculus tradition (Azcencot et al., 2014) with the continuous neural representation paradigm, gaining the best of both: divergence-free guarantees from the former and mesh-free flexibility from the latter. A key insight that emerges from the review is that the paper's real contribution may be more unified than the two-part claim suggests: the CPM construction and the covariant-derivative advection are deeply intertwined through the stream function parameterization, making ablation potentially non-trivial but all the more necessary to understand what drives the impressive accuracy gains.

## Suggestions

- **Add an ablation study** testing: (a) divergence-free construction + standard semi-Lagrangian advection, (b) non-divergence-free neural field + covariant-derivative advection, and (c) the full method. This would directly attribute the accuracy and energy gains to specific components.
- **Move quantitative results from appendices to the main text** for Taylor vortices and other experiments, or at minimum include a summary table with error metrics across all experimental settings.
- **Add energy-over-time curves** for the sphere jet and Taylor vortex experiments alongside baselines. Even a single plot showing kinetic energy vs. time step for NFFS vs. INSR vs. PINN would substantiate the energy preservation claim far more effectively than the current qualitative-only evidence.
- **Discuss the compute-accuracy tradeoff** explicitly — acknowledge the 20× cost increase and identify which components dominate runtime, so readers can assess whether the method is practical for their use case.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| clawNOs | /home/wg25r/review_agent/human_reviews/KEpR8hFzvO.md | 5.0 | Topically similar (divergence-free neural operators, hard constraint). clawNOs had more baselines but was dinged for missing ablations, overclaimed "significantly outperform" language, and lack of divergence-free metric. NFFS has a stronger core idea (surface simulation is harder) and more dramatic accuracy gains, but similarly lacks ablations and overclaims. |
| Simulation-free NCL | /home/wg25r/review_agent/human_reviews/jIOBhZO1ax.md | 5.5 | Similar pattern (novel hard-constraint framework, limited scalability evidence). NFFS is more complete in its experimental showcase but similarly overclaims relative to evidence. |
| HelmSim | /home/wg25r/review_agent/human_reviews/8HG2QrtXXB.md | 5.0 | Very similar topic (Helmholtz-based fluid simulation). HelmSim was rejected for missing related work and unclear advantages. NFFS has clearer advantages (2 orders of magnitude improvement) but also lacks ablation. |
| NMS | /home/wg25r/review_agent/human_reviews/uL1H29dM0c.md | 7.0 | Accepted with provably energy-conserving architecture and approximation results. NFFS lacks the theoretical rigor (no convergence analysis, proof sketch only) that elevated NMS. |
| Pan for Gold | /home/wg25r/review_agent/human_reviews/1gqR7yEqnP.md | 2.2 | Low anchor — fundamentally flawed claims with near-zero evidence. NFFS is far above this; it has a real contribution with dramatic accuracy improvements. |

NFFS sits in the 5.0–5.5 range occupied by clawNOs (5.0) and simulation-free NCL (5.5). It has a genuinely novel and sound core idea with dramatic empirical improvements, but the evidence doesn't match the strength of the claims: no ablation, one quantitative experiment for headline numbers, and energy preservation shown only in an appendix case. The paper is clearly above the low anchors (no fundamental flaws) but below the high anchors (lacks theoretical rigor and comprehensive evaluation of NMS at 7.0). I place it at 5.0, matching clawNOs as the closest topically similar anchor — slightly below simulation-free NCL (5.5) because that paper had a cleaner theoretical framework despite similar empirical limitations.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>