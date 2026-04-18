Now I have enough context. Let me synthesize the final review.

Let me think carefully about each of the harsh critic's points by cross-checking against the actual paper content:

1. **Divergence-free guarantee on curved surfaces**: The critic argues the proof is hand-wavy. Let me check what the paper actually provides:
- Theorem 3.1 gives the explicit formula v(x) = j*((∇(cp*σ)∘j(x))×n(x))
- A proof sketch is provided
- The construction follows from standard exterior calculus: *dμ with d²=0 implies div(v)=0
- The close relationship to "surface curl" / Helmholtz decomposition is noted
- This is a legitimate mathematical construction; the CPM + stream function approach for divergence-free fields on surfaces has precedent in the literature (Azcencot et al. 2014, which they build on). The core identity (gradient cross normal = tangent divergence-free field) is well-known in surface vector field decomposition. However, the paper does not provide a complete rigorous proof - just a sketch. This is a valid concern but perhaps not "fatal" since the construction follows established mathematical principles.

2. **Advection scheme energy preservation**: The critic argues that the "energy-preserving" claim is not substantiated. The paper:
- Uses a first-order approximation (Eq. 15) derived from the covariant derivative
- Minimizes residuals with Adam (not exact optimization)
- Claims "low energy dissipation" and "energy-preserving"
- The paper does reference Appendix E.1 for sphere rot energy analysis
- But no quantitative energy tracking is shown in the main text
- The advection scheme Eq. 15 is essentially a Crank-Nicolson-like implicit midpoint approximation, but implemented via optimization, not as a proper variational integrator

This is a substantive concern but needs to be calibrated. The paper does have energy analysis in the appendix, though it's not in the main text.

3. **Unfair baselines**: The critic has multiple sub-points:
- "15x" claim: The paper's abstract claims "approximately 15 times higher accuracy than other methods with the same storage cost." Looking at Table 1: Ours 2.89e2 vs Small-F.S. 5.34e3 - this is ~18.5x. The comparison is with "other methods" which they argue includes PINN and INSR. This is a reasonable point about selective comparison.
- PINN/INSR configuration: Not fully specified. Fair concern.
- FFS convergence failure: The paper does state this but with limited detail.
- No variance/statistical analysis: Valid but common in this field.

4. **ω-v relationship not enforced**: The critic argues that ω and v are independently parameterized with no constraint enforcing ω = (∇×v)·n. This is actually an important structural concern. Looking at the paper:
- Eq. 5 defines ω in terms of v
- But in the optimization (Eq. 16), both ω_{i+1} and v_{i+1} are parameterized separately
- No explicit loss term enforces their consistency
- The critic is correct that this is under-specified

Let me also think about what's genuinely strong about this paper:
- First neural simulation framework for incompressible flow on surfaces, particularly implicit neural surfaces
- Novel combination of CPM, exterior calculus, and INR for guaranteed divergence-free fields
- The method demonstrably works on challenging implicit surfaces where classical methods fail
- Memory efficiency advantage is real
- The conditioning/generation application is interesting

Now let me calibrate against similar papers:
- LFlows (Accept spotlight, scores 6,8,8): Novel parameterization for conservation laws, rigorous proofs, good experiments. More rigorous than this paper.
- clawNOs (Reject, scores 5,6,3,6,5): Divergence-free neural operators, overclaimed results, weak baselines. Similar pattern of hard constraint + limited evaluation.
- HelmSim (Reject, scores 3,5,6,6): Fluid decomposition, missing baselines, short horizons. Similar weaknesses.
- Geometric Constraints (Reject, scores 6,3,3): Physical constraints in neural PDEs, limited baselines.

This paper has real novelty (first neural surface flow with divergence-free guarantee on INR surfaces) and addresses an important niche. However, the theoretical gaps (incomplete proof, unenforced ω-v consistency) and empirical gaps (no energy tracking in main text, limited quantitative comparisons) are substantial. The claims are somewhat overclaimed relative to what's demonstrated.

I think a fair score would be around 5 - marginally below acceptance. The idea is novel and important, but the execution has several weaknesses that prevent fully supporting the claims, particularly the theoretical gaps and the incomplete empirical evidence for energy preservation.

Actually, let me reconsider. The paper does show results on INR surfaces - which is a genuine novelty. And the divergence-free construction via stream functions on surfaces using CPM is a legitimate mathematical approach. The proof sketch issue is real but not fatal - the construction is standard exterior calculus. The ω-v consistency gap is more concerning. The lack of energy plots in the main text is a significant omission but the concept works.

Let me settle on 5. The paper is borderline - interesting idea with real novelty but significant gaps in verification of claims.</think>## Summary

This paper proposes NFFS, a neural physical simulation framework for incompressible fluid flow on geometric surfaces. The key idea is to construct a divergence-free velocity field on surfaces by combining Implicit Neural Representations (INR) with the Closest Point Method and exterior calculus (via a stream function $\sigma$, yielding $v = j^*((\nabla(cp^*\sigma)\circ j) \times n)$), which guarantees divergence-freeness by construction without pressure projection. The vorticity is then advected using a covariant-derivative-based scheme (Eq. 15). The method applies to analytic, explicit mesh, and implicit neural representations alike, and also supports conditioning for generation tasks and Helmholtz decomposition.

## Strengths

- **Novel and important problem setting**: First neural framework for incompressible surface flow simulation that operates directly on implicit neural representations (e.g., Siren, DeepSDF) without requiring meshing. This is timely given the prevalence of neural surface representations and addresses a genuine gap — classical solvers break down on these representations (Sec. 5.2 provides evidence).

- **Guaranteed divergence-free construction by design**: Rather than enforcing incompressibility through loss functions (as in PINN/INSR), the stream function parameterization ensures divergence-free velocity fields as a structural property. This eliminates projection error accumulation, a known practical problem with cascading effects (demonstrated visually in Fig. 4).

- **Demonstrated robustness on implicit surfaces**: The experiments on armadillo and Lucy INR surfaces (Fig. 7) show the method produces smooth flows where classical Functional Fluids on Surfaces fails to converge. The supplementary further documents crash time steps across resolutions, substantiating this robustness claim.

- **Memory efficiency优势**: Table 1 shows the method achieves MSE 2.89e2 with ~532KB storage, compared to the ground truth requiring 2643KB. The compact INR representation genuinely provides a memory advantage over mesh-based methods for the same accuracy.

- **Flexibility of applications**: The conditioning framework (Sec. 4.2) and Helmholtz decomposition (Sec. 3.2, 5.4) demonstrate that the representation class extends naturally beyond forward simulation, which is a practical benefit of the neural parameterization.

## Weaknesses

### Fatal
None.

### Major

- **ω–v consistency is not explicitly constrained during optimization**: The paper defines vorticity via Eq. 5 ($\omega = (\nabla \times v)\cdot n$) but in the time integration (Eq. 16), both $\omega_{i+1}$ and $v_{i+1}$ are separately parameterized neural fields optimized to minimize $\mathcal{L}_i$. No loss term enforces the physical constraint $\omega = (\nabla \times v)\cdot n$ at each time step. The vorticity field can thus drift from being the true vorticity of the velocity field, breaking the Hamiltonian structure of Euler equations and undermining the energy-preservation narrative. This is noted in Sec. 4.1 only through the advection residual $\mathcal{L}_i$, which is an approximate PDE enforcement — not a hard constraint on the ω–v relationship.

- **Energy preservation claims lack quantitative support in the main text**: The paper repeatedly claims "low energy dissipation" and "energy-preserving" (Abstract, Sec. 1, Sec. 6), but no figure or table in the main text plots kinetic energy (or enstrophy) vs. time for any experiment. The "sphere rot" energy analysis is relegated to Appendix E.1 and never summarized quantitatively in the main body. For a paper whose central narrative emphasizes energy preservation over classical methods, this omission is a significant evidential gap. The advection scheme (Eq. 15) is a first-order approximation of the exponential map, solved inexactly via Adam optimization — claiming energy preservation without quantitative verification is unsupported.

- **Limited quantitative comparisons and selective "15×" claim**: The abstract states "approximately 15 times higher accuracy than other methods with the same storage cost." However, Table 1 shows the most relevant same-paradigm comparison is against Small-F.S. (MSE 5.34e3 vs 2.89e2), which is ~18×. Against PINN and INSR the ratio is 300–600×, but these baselines are not designed for surface simulation (the paper itself notes in footnote 2 that they "can not be simply adopted to the flow on various surfaces"). No quantitative error metrics are provided for the inclined plane, implicit surface, or explicit mesh experiments in the main text. The impressive visual results on INR surfaces lack numerical grounding.

### Minor

- **Incomplete proof for Theorem 3.1**: The paper acknowledges providing only a "proof sketch" connecting the exterior calculus formulation to Eq. (4), mixing ambient ($\mathbb{R}^3$) and intrinsic surface operators without making all geometric identities precise. The construction is plausible and builds on established CPM/exterior calculus machinery, but the lack of a complete derivation — especially showing that $\text{div}_S(v) = 0$ in the surface sense — requires the reader to fill in non-trivial steps. The paper notes the relationship to "surface curl" in Helmholtz decomposition, which lends credibility, but a full derivation should be in the appendix.

- **Harmonic component handling is under-specified**: For surfaces with non-trivial topology (non-zero homology), the method introduces a time-invariant MLP $\eta$ to capture the harmonic component (Sec. 4.1). This is frozen after initialization and never re-optimized. The main text states "we actually can address the issue" but defers details to Appendix F.1. No experiments on surfaces with non-trivial topology (e.g., torus) demonstrate that this handling is correct.

- **Computational cost is high with limited discussion**: Table 1 reports 16.5 hours for the proposed method vs. 0.8 hours for Small-F.S., a ~20× slowdown. While memory efficiency is highlighted, this time cost receives minimal discussion. The paper's own Sec. 6 acknowledges "time efficiency" as a limitation but provides no breakdown (iterations per time step, scaling analysis, or potential acceleration avenues).

- **First-order advection approximation**: Eq. 15 truncates the operator exponential to first order. The paper itself notes (Sec. 6) that "high order approximation" is future work, which is fair, but the impact of this truncation on long-time accuracy and the choice of time step $h$ are not analyzed.

### Trivial

- Notation switches between differential forms ($\mu$, $\omega$) and vector calculus ($\nabla$, $\times$) between Sections 3.1 and 3.2, which can be mildly confusing for readers unfamiliar with exterior calculus.

## Nice-to-Haves

- Quantitative energy tracking plots (kinetic energy vs. time) for all experiments, which would directly substantiate the energy-preservation claim.
- Ablation of the covariant-derivative advection against simpler alternatives (e.g., semi-Lagrangian) with the same divergence-free construction, to isolate the contribution of the advection scheme.
- Measured divergence error ($\int |\nabla_S \cdot v|^2 dS$) across time steps to confirm that the theoretical guarantee holds numerically after neural approximation and finite sampling.
- An experiment on a surface with non-trivial topology (e.g., torus) demonstrating the harmonic component handling.

## Removed Points

- **"PINN/INSR are not designed for surface flows, making comparison unfair"** — The paper itself acknowledges this (footnote 2) and excludes them from mesh/INR experiments. The comparison on the sphere/inclined plane is appropriate as both methods can in principle be adapted to analytic surfaces. The paper is transparent about the limitations of this comparison.

- **"No statistical variance or ablation across random seeds"** — While desirable, single-run evaluation is standard in neural PDE papers. This is a generic critique that does not specifically undermine the paper's claims.

- **"Storage accounting is opaque"** — The paper provides specific storage numbers in Table 1 in KB. While the exact accounting (parameters vs. cached states) could be clearer, this is a minor presentation detail.

- **"FFS convergence failure is under-explained"** — The paper states this result and points to Appendix E.4 with crash time steps. The claim that a classical method fails on complex INR-derived meshes is plausible and well-documented in the graphics literature. While more detail would help, this is not a fairness issue.

- **"No comparison against recent neural methods on implicit surfaces"** — The paper addresses a novel niche (neural surface flow on INR geometries). There are no directly comparable neural baselines for this specific setting, so this absence is expected rather than a gap.

## Novel Insights

The most novel insight is that the Closest Point Method provides a principled bridge between the surface exterior calculus formulation of divergence-free fields and the continuous $\mathbb{R}^3$ gradient operations that neural networks can easily compute, enabling a construction that is simultaneously exact in the divergence-free sense and amenable to neural parameterization. This is a genuine technical contribution distinct from prior neural flow work (which either operates in flat domains or requires mesh discretization). However, the ω–v decoupling during optimization is an architectural gap that partially undermines the physical consistency the construction promises.

## Suggestions

1. Add an explicit constraint $\|\omega - (\nabla \times v)\cdot n\|^2$ to the optimization loss (Eq. 16), or at minimum, empirically measure the magnitude of this discrepancy over time steps to assess how much drift occurs.
2. Include kinetic energy vs. time plots in the main text for at least one experiment (e.g., sphere rot or jet flow) to substantiate energy-preservation claims.
3. Provide a complete derivation of Theorem 3.1 in the appendix, showing $\text{div}_S(v) = 0$ explicitly on the surface using the CPM machinery.
4. Add quantitative error metrics (MSE or similar) for the inclined plane and at least one mesh/implicit surface experiment in the main text.
5. Discuss the time cost and potential for acceleration more explicitly (e.g., warm-starting from $\theta_i$, reducing Adam iterations, or using second-order optimizers).

## Score and Decision

**Calibration**: I compared against several related papers:
- **LFlows** (Accept spotlight, avg ~7.3): Novel neural parameterization with conservation guarantees, rigorous proofs, strong experiments. More theoretically rigorous and with better empirical support than this paper.
- **clawNOs** (Reject, avg ~5): Divergence-free construction in neural operators, overclaimed "significant" improvements, limited baselines, incomplete energy analysis. Similar pattern of hard constraint + limited verification.
- **HelmSim** (Reject, avg ~5): Helmholtz decomposition for fluid learning, missing important baselines, short prediction horizons. Structurally similar weaknesses.
- **Geometric Constraints** (Reject, avg ~4): Physical constraints in neural PDEs, limited baselines, narrow scope.

This paper has genuine novelty in its surface flow + INR combination and demonstrates practical advantages on implicit surfaces. However, the ω–v consistency gap and the unsubstantiated energy-preservation claims are significant issues that weaken the core narrative. The experimental evidence, while visually compelling, lacks quantitative depth for most experiments. The paper is above the clearly rejected papers (which have fatal flaws or negligible novelty) but below LFlows (which has both novelty and rigorous verification). I place it in the marginal range, slightly below acceptance, similar to clawNOs.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>