Now let me search for calibration anchors.Now I have sufficient calibration data to write the final review.

---

## Summary

NFFS (Neural Functional Flow on Surface) is a neural simulation framework for incompressible fluid flow on geometric surfaces. The core contribution is a provably divergence-free velocity field construction via the Closest Point Method (CPM) and exterior calculus (Theorem 3.1), paired with a covariant-derivative-based advection scheme for energy-preserving vorticity transport. The paper demonstrates results on analytic surfaces (sphere, inclined plane), explicit meshes, and — as a first-of-its-kind claim — implicit neural representation surfaces.

---

## Strengths

- **Theorem 3.1 with CPM-based divergence-free guarantee (Eq. 4):** The construction $v(x) = j^*((\nabla(cp^*\sigma) \circ j(x)) \times n(x))$ provably satisfies the divergence-free constraint by design, rather than penalizing it in a loss. This eliminates pressure-projection error accumulated by PINN and INSR at every time step — a meaningful architectural advantage with a clean theoretical foundation.

- **Strong quantitative accuracy on analytic surfaces (Table 1, Fig. 5):** At the same storage budget (~530 KB), NFFS achieves an MSE of 2.89e2 on the sphere jet benchmark — approximately 18× lower than Small-F.S. (5.34e3) and ~300× lower than PINN (1.73e5). Fig. 5 shows the same advantage on Taylor vortices on an inclined plane, with NFFS closely matching the reference GT while all baselines show significant vortex dissipation by time step 40.

- **First demonstration of incompressible surface flow on implicit neural surfaces (Sec. 5.2, Fig. 7):** The framework's CPM-based normal computation avoids marching-cubes meshing entirely, enabling simulation on DeepSDF/SIREN-parameterized Armadillo and Lucy surfaces. The paper substantiates that classical FFS crashes on marching-cubes-derived meshes at multiple resolutions (Appendix E.4), making this a genuine contribution rather than a superficial extension.

- **Covariant-derivative advection for energy preservation (Eq. 15):** The implicit symmetric scheme couples forward and backward advection around the midpoint, avoiding the energy drift inherent in operator-splitting approaches. Fig. 5 visually confirms that INSR and PINN dissipate vortex energy over time while NFFS maintains structure comparable to the reference GT.

- **Generality across surface representations (Secs. 5.1–5.2):** The same theoretical framework applies to analytic surfaces, discretized explicit meshes, and INR surfaces without method-specific modifications — a practical breadth advantage.

---

## Weaknesses

### Fatal
None.

### Major

- **No quantitative evaluation for the headlining novel contribution (implicit surface simulation, Sec. 5.2):** The paper's most distinctive claim — first-ever simulation on implicit neural surfaces — rests entirely on qualitative figures (Fig. 7) and a competitor's crash log. No divergence error $\|\nabla \cdot v\|$, no energy trace, no self-consistency measure of any kind is reported for the Armadillo or Lucy experiments. Given that there are no baselines to compare against here (since all competitors fail), some form of self-evaluation (e.g., energy conservation over time, divergence residual on sampled points) is the minimum required to substantiate the claim. Without it, the novel contribution is unverifiable.

- **No ablation separating the two stated contributions:** The paper presents two separable components: (a) CPM divergence-free construction (Theorem 3.1), and (b) covariant-derivative advection (Eq. 15). The quantitative gains in Table 1 and the energy preservation evidence in Fig. 5 could stem from either or both components. All comparative experiments pit full NFFS against PINN and INSR, which differ from NFFS on both dimensions simultaneously. Without a variant that isolates each contribution (e.g., CPM construction + standard semi-Lagrangian advection, or divergence-penalized loss + covariant advection), it is impossible to determine where the gains originate.

### Minor

- **Main quantitative benchmark uses FFS-as-GT rather than analytic ground truth:** Table 1 measures MSE against a higher-resolution FFS simulation (5× storage), which shares the same mathematical family as NFFS. This is a common practice for complex flows without analytic solutions, and the sphere jet case may not admit a closed-form GT. Still, this makes the central accuracy claim somewhat self-referential. The inclined plane case does have analytic Taylor vortex solutions, but quantitative results for that case are deferred to Appendix E.2 and not included in the main body. Moving the analytic-GT comparison to the main paper would substantially strengthen the accuracy claim.

- **Conditioning and Helmholtz decomposition sections lack quantitative evaluation (Secs. 5.3–5.4):** The VAE generation experiment (Fig. 8a) reports no metric (no divergence residual, no FID, no energy check on generated fields). The real-world wind decomposition (Fig. 8b) is purely qualitative. These sections demonstrate framework versatility but do not constitute scientific results in their current form.

- **Runtime (16.5 hours) is understated as a limitation:** Table 1 shows NFFS is ~20× slower than Small-F.S. (0.8 h). The discussion acknowledges time efficiency as a limitation but provides no breakdown of where the cost lies, making it hard to assess whether the bottleneck is fundamental to the optimization-per-step design or addressable with engineering improvements.

### Trivial

- The first-order truncation of the exponential map in Eq. 14–15 introduces truncation error that grows with step size $h$. The paper does not report the time-step size used in experiments, nor analyze the temporal accuracy order. This is acknowledged as a future direction (higher-order approximations), but stating the time-step size in the main experiments would aid reproducibility.

---

## Nice-to-Haves

- An energy-over-time plot (kinetic energy vs. time step) comparing NFFS, INSR, PINN, and Small-F.S. on the sphere jet flow would directly and quantitatively substantiate the energy preservation claim, complementing the visual evidence in Fig. 5.
- Even a qualitative demonstration that the harmonic component (handled via the time-invariant residual MLP, Sec. 4.1) contributes meaningfully on a surface with non-trivial topology (e.g., a torus) would help readers assess the completeness of the topological treatment.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"5× memory savings" framing is misleading (Harsh Critic):** The critic argues comparing NFFS (532.8 KB) against GT (2643.0 KB) is unfair because GT runs at 5× resolution. This is removed as a standalone weakness: the paper's claim is not that NFFS matches GT quality for 5× less memory, but that NFFS at budget X achieves higher accuracy than other methods at the same budget X, and that budget X is 5× smaller than the high-resolution reference. The framing is slightly loose but the underlying comparison in Table 1 is clear and legitimate.

- **Demand for mesh-based baselines on explicit meshes (Harsh Critic, referencing Elcott et al. 2007):** The critic notes that Elcott et al. 2007 could in principle be run on the hand and spot models. Removed because (a) Elcott et al. is included in Fig. 5 for the Taylor vortex case, (b) the explicit mesh section primarily demonstrates generality rather than numerical competition, and (c) INSR/PINN are appropriately excluded by footnote 2 given their architectural limitations on non-parameterized surfaces — this is a reasonable scoping decision, not an unfair omission.

- **Non-zero homology is incomplete in main text (Harsh Critic):** Removed. The paper explicitly addresses the harmonic component in Sec. 4.1 with the time-invariant MLP and references Appendix F.1. Per rules, the appendix cannot be penalized for being absent in the extract.

- **Demand for theoretical convergence/stability analysis (Harsh Critic):** Moved to Nice-to-Haves. The paper is an empirical systems paper; demanding formal convergence proofs is not the norm for this community at ICLR, and the paper already acknowledges this as future work.

---

## Novel Insights

The most conceptually interesting contribution — not fully explored in either review — is the interplay between CPM and the exterior calculus framework. By pulling back the surface calculation into $\mathbb{R}^3$ via the closest-point map, the method sidesteps the need for surface parameterization entirely, making it geometry-representation-agnostic by construction rather than by engineering. This is in contrast to prior methods that either require explicit parameterization (which distorts) or enforce constraints via loss terms (which drift). The practical consequence — that the same code path works on analytic, mesh, and INR surfaces without modification — is a genuine architectural elegance that the paper itself somewhat undersells by distributing it across sections.

---

## Suggestions

1. **Add a quantitative self-evaluation for INR experiments:** Report (a) mean $\|\nabla \cdot v\|$ on sampled surface points at each time step, and (b) kinetic energy $\frac{1}{2}\|v\|^2$ over time steps, for the Armadillo/Lucy cases. This requires no baselines and directly substantiates the method's physical plausibility on its novel test cases.

2. **Add an ablation run:** One additional experiment — CPM divergence-free construction + standard semi-Lagrangian advection — would resolve the attribution ambiguity between the two contributions and is likely straightforward to implement.

3. **Promote analytic-GT quantitative comparison to main text:** Move the Taylor vortex MSE comparison (Appendix E.2) to Table 1 or a companion table. This grounds the accuracy claim in analytic truth and reduces reliance on FFS-as-GT.

---

## Score and Decision

**Calibration Anchors:**

| Path | Avg Score | Comparison |
|------|-----------|------------|
| `/human_reviews/kIZcruKmBg.md` | 3.25 | Low anchor: PINNs on manifolds, poor validation, withdrawn. NFFS is clearly stronger with a real theorem and quantitative results. |
| `/human_reviews/gz8Rr1iuDK.md` | 4.00 | Low anchor: Hard-constraint neural PDE, rejected. Less theoretical depth than NFFS, weaker empirical wins. |
| `/human_reviews/8HG2QrtXXB.md` | 5.00 | Medium anchor: HelmSim fluid simulation with Helmholtz decomposition, rejected. Similar missing ablations and baseline gaps, but NFFS has stronger theory and better numeric results. |
| `/human_reviews/5LvTfc4fBz.md` | 5.00 | Medium anchor: Physics-enhanced neural operator for turbulent flow, rejected. Less novelty than NFFS. |
| `/human_reviews/sYAFiHP6qr.md` | 6.50 | High anchor: Implicit neural surface deformation with velocity fields, accepted poster. Shares the "first-on-INR" flavor; has ablations and quantitative evals that NFFS lacks. NFFS has more theoretical depth but weaker experimental completeness. |
| `/human_reviews/4yaFQ7181M.md` | 7.60 | High anchor: Space-time continuous physics simulation, accepted spotlight. Substantially stronger empirical validation with comprehensive baselines. |

**Positioning:** NFFS clearly sits above the low anchors (3.25–4.00) due to Theorem 3.1 and solid quantitative results on analytic surfaces. It is roughly comparable to the 5.0 medium anchors but has meaningfully stronger theoretical foundations. It falls below the 6.5 accepted poster (sYAFiHP6qr) because that paper includes ablations and quantitative support for all its claims, whereas NFFS's primary novel contribution (INR surfaces) has zero quantitative validation. The missing ablation further weakens the attributional clarity.

The paper is a genuine contribution — the divergence-free construction is clean, the analytic surface results are compelling, and the INR extension is novel in concept — but the experimental case for the most novel claim is substantially incomplete. This puts it in the borderline zone, landing at **5.5**.

**Axis-by-axis:**
- *Originality:* Good — CPM + exterior calculus for guaranteed divergence-free surface fields is a novel combination; INR-surface simulation is first-of-its-kind.
- *Importance of research question:* High — surface fluid simulation is practically relevant for graphics, climate science, and bio-physics.
- *Claims well supported:* Partially — analytic surface claims are well-supported; INR-surface claims are only qualitatively supported.
- *Soundness of experiments:* Moderate — strong for analytic surfaces, notably incomplete for the novel INR contribution, missing ablation.
- *Clarity of writing:* Good — mathematical framework is clearly presented; limitations are acknowledged.
- *Value to research community:* Moderate-to-High — the unified framework and INR compatibility are practically useful, though the time cost (16.5h) limits immediate applicability.

**Final Score: 5.5 (borderline, leaning toward weak accept)**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>