=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary
This paper addresses constrained generative modeling on convex domains via flow matching. It identifies that standard log-barrier mirror maps induce heavy-tailed dual distributions (causing ill-posed dynamics) and that Gaussian priors poorly match heavy-tailed targets. The solution combines a regularized mirror map (to control dual tails and ensure finite moments) with a Student-t prior (to align with heavy tails). Theoretically, the work establishes spatial Lipschitzness and temporal regularity of the velocity field under polynomial tail assumptions, yielding Wasserstein convergence rates for flow matching with Student-t priors and primal-space guarantees for constrained generation.

## Strengths
- **Novel co-design of mirror map and prior**: The paper insightfully links two previously separate issues—heavy-tailed dual distributions from log-barrier maps and prior mismatch—and addresses them jointly via a regularized mirror map and Student-t prior. This is a principled advance over simply using existing components.
- **Substantial theoretical contributions**: The analysis provides the first Lipschitzness and convergence guarantees for flow matching with Student-t priors under polynomial tail conditions (Assumption 3), extending prior work that required bounded support. The primal‑space guarantee (Theorem 4) that transfers dual‑space convergence to the constrained domain is non‑trivial and valuable.
- **Clear empirical gains on synthetic and real tasks**: On synthetic polytope and L₂‑ball tasks, the method consistently outperforms Gauge Flow, Reflected Flow Matching, and Mirror Diffusion Models in MMD and KL divergence while maintaining 100% feasibility. On watermarked AFHQv2 images, it achieves competitive FID/CMMD with significantly reduced training time when initialized from a pre‑trained checkpoint.

## Weaknesses
### Major
- **Theoretical assumptions are difficult to verify in practice**: Assumption 3 (polynomial tail bound on the dual target) and Assumption 4 (boundary decay of the primal density) are central to the convergence theorems. While Lemma G.5 gives a sufficient condition linking them, the paper offers no practical procedure to check these assumptions for an arbitrary constrained distribution. For users, this makes the guarantees more conceptual than actionable.
- **Limited demonstration on real‑world constrained tasks**: The only real‑data experiment is a specialized watermarked‑image task on AFHQv2 at 64×64. The method is not evaluated on established constrained‑generation benchmarks (e.g., molecular design, robotics constraints) or on standard image datasets with simple box constraints. Without broader validation, the claim of “competitive sample quality on real‑world constrained generative tasks” is only partially supported.

### Minor
- **Hyperparameter selection lacks clear guidance**: Performance depends critically on the mirror‑map regularization parameter κ and the Student‑t degrees of freedom ν. The theory gives abstract conditions (e.g., κ ≤ γ/(2d+ν+2), κ < β/2), but the paper provides no principled way to choose κ and ν for a new problem. Figure 3 shows sensitivity on one synthetic task, but a more systematic ablation and discussion of adaptive selection are missing.
- **Early‑stopping time T is an unguided choice**: Algorithm 1 requires choosing T < 1 to avoid temporal singularities, and the error bound contains a term (1−T)M. The paper does not discuss how to select T in practice or how to balance early‑stopping error against discretization/approximation errors.

### Trivial
- **Exponential dependence on Lipschitz constant in bounds**: Theorem 3’s error bound scales as e^(6L₁)/L₁. The authors correctly note this also appears in prior analyses and may be improvable via probabilistic coupling, so this does not detract from the contribution.

## Nice-to-Haves
- A more comprehensive ablation study of κ and ν across different constraint geometries and target distributions, along with heuristic guidelines for choosing them.
- Extension to non‑convex constraints or integration with adaptive parameter selection (e.g., learning κ or ν from data).
- Comparison to reflected flow matching on the watermarked‑image task, and evaluation on additional constrained benchmarks (e.g., molecular generation).

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **“Tail assumptions unrealistic for uniform distributions”**: The harsh critic claimed that for a uniform distribution on a cube, γ=0 forces κ=0, making the theory inapplicable. However, Example 1 in the paper shows P(K\Kδ) ≈ dδ, giving β=1, and the condition only requires κ < β/2 = 0.5. Hence the criticism misreads the paper.
- **“Empirical evaluation insufficient to support claims of outperforming baselines”**: The paper shows clear improvements in Tables 1 and 2 on synthetic tasks and competitive results on a real‑data task. Demanding statistical significance tests or many more application domains is scope‑creep; the presented evidence is adequate for the claims made.
- **“Missing comparison to state‑of‑the‑art constrained generative models beyond MDM”**: The paper compares to Gauge Flow, RFM, and MDM—representative recent works. Requiring an exhaustive survey of all constrained methods is not necessary for establishing the contribution.
- **“The inverse mirror map is non‑trivial and computationally costly”**: The paper explicitly uses a regularized mirror map with a simple closed‑form gradient; the computational cost of inversion is not discussed but is not a core methodological flaw.
- **“Assumption 1 (neural‑network approximation error) is optimistic”**: This assumption is standard in flow‑matching theory (cited from Benton et al., Zhou & Liu); questioning it is not a paper‑specific weakness.

## Suggestions
- Add a practical discussion (possibly in an appendix) on how one might estimate or check the tail exponent α (Assumption 3) or boundary decay γ (Assumption 4) from samples, even if approximately.
- Include a sensitivity analysis of the early‑stopping time T and step size h on synthetic tasks to empirically validate the error bound’s trade‑offs.
- For the real‑data experiment, report the percentage of generated samples that satisfy the watermark constraints (beyond just stating feasibility is checked) to quantify constraint adherence.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
