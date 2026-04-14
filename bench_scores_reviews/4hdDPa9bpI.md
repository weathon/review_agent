## Summary

Graph Fourier Neural Kernels (G-FuNK) is a neural operator framework for learning solution generators of nonlinear diffusive PDEs across multiple geometries and anisotropic diffusion fields. The core innovation is a physics-informed weighted graph Laplacian (Eq. 8) whose eigenbasis approximates the highest-order diffusive term in the PDE, enabling transfer across domains without retraining. The framework is validated on the anisotropic heat equation, 2D reaction-diffusion on random rectangles, and 3D cardiac electrophysiology on patient-specific left atrial geometries, achieving significant speedup over finite-element solvers.

---

## Strengths

- **Physics-informed spectral basis construction (Eq. 8):** The edge-weighting scheme encodes the full diffusion tensor K into the graph Laplacian, causing its eigenbasis to naturally encode the directional dependence of the PDE's principal term. This is a specific and non-trivial departure from geometry-only spectral bases. The consequence — demonstrated in Table 1 — is that G-FuNK achieves rotation invariance on the reaction-diffusion task (error 0.1189 under 90° rotation) while GeoFNO degrades to 0.5681, and G-FuNK trains on *fewer parameters* (~135K vs GeoFNO's ~2.5M) while matching GeoFNO accuracy on non-rotated test data.

- **Applicability to topologically complex 3D geometries where grid-based methods fail:** The left atrial surface has five holes (removed valves/veins) and cannot be diffeomorphically mapped to a cube or torus, making FNO/GeoFNO inapplicable. G-FuNK naturally handles this, which is a concrete and practically important distinction from prior art.

- **Dramatically fewer parameters with competitive performance:** G-FuNK achieves competitive or superior accuracy with 7–18× fewer parameters than FNO/GeoFNO baselines (Table 1), which is a meaningful practical advantage for the data-scarce cardiac setting.

- **Well-motivated hybrid spectral-spatial architecture:** The decomposition into a spectral branch (GFT → eigenvalue expansion → learned spectral filter → IGFT) handling diffusion and a pointwise spatial branch (W_n) handling local reaction terms aligns well with the structure of reaction-diffusion PDEs, and the intuition is clearly articulated.

---

## Weaknesses

### Fatal
None.

### Major

- **No derivation, convergence proof, or citation for the edge-weight formula (Eq. 8).** The central claim is that weighting edges by $w_{ij}^{-1} = \frac{1}{2}(x_j - x_i)^T(\mathbf{K}(x_i)^{-1} + \mathbf{K}(x_j)^{-1})(x_j - x_i)$ causes the graph Laplacian to approximate $\nabla\cdot(\mathbf{K}\nabla)$. The paper offers only an intuitive description ("we weight the edges at $x_i$ proportionally to the direction of the diffusion coefficient") but provides no formal justification. Rigorously establishing this approximation — even for the scalar $k$-NN case — is non-trivial; the passage from $k$-nearest-neighbor graphs to accurate PDE approximation differs significantly from cotangent-weight constructions for the isotropic Laplacian on Delaunay meshes. This formula is the mathematical core of the adaptation claim, and without a derivation or supporting reference, the approximation quality remains unverified. This is the paper's most important theoretical gap.

- **Cardiac EP generalization is evaluated on a single unseen geometry.** The most important claim — that G-FuNK generalizes to new patient anatomies without retraining — is backed by a single held-out left atrial mesh (24 training, 1 test). A single test case cannot establish statistical generalization; the 1.62 ms wavefront lag and relative ℓ₂ of 0.1642 could reflect idiosyncratic properties of that one geometry. The authors acknowledge data scarcity (>1 day per patient simulation), but a leave-one-out cross-validation on the 25 available patients would provide substantially more robust evidence for the generalization claim and should be feasible without additional data collection.

- **Missing ablations on core design choices.** Several design decisions are unexplored and could significantly affect the reader's understanding of what drives performance:
  - *Effect of the physics-adapted edge weights (Eq. 8) vs. standard geometric weights:* Is anisotropy-encoding in the eigenbasis (vs. simply providing fiber fields as input features) actually responsible for the rotation invariance result? This ablation is crucial to substantiate the core claim.
  - *Sensitivity to k_max:* The number of retained eigenpairs is presumably critical for approximation accuracy, but no sweep is reported.
  - *Effect of polynomial eigenvalue expansion (p):* The heat equation uses p=1; values used in other experiments are not stated, which affects reproducibility and prevents understanding the contribution of this component.

### Minor

- **Mesh-independence claim is asserted, not demonstrated.** Section 2.2 states G-FuNK "maintains a high degree of mesh-independence" appealing to asymptotic graph Laplacian convergence theory, but no resolution-transfer experiment (e.g., train on coarse mesh, evaluate on fine) is presented. For an ICLR methods paper making this claim, at least one resolution sweep should be shown.

- **Eigendecomposition overhead is excluded from speedup benchmarks.** The reported speedup (13 CPU minutes for FEM vs. <1 GPU second for G-FuNK) omits the eigendecomposition preprocessing cost for new domains, the GPU used for inference, and training cost amortization. Since eigendecomposition is cited as $O(k_{\max}^2 n_\alpha j)$, it could be non-trivial for large 3D meshes, and excluding it may overstate the practical speedup for new patients.

- **Training setup underspecified.** No training time, hardware, or convergence information is reported for any of the three experiments. This prevents practitioners from assessing practical cost.

- **No per-trajectory variance reported.** All errors in Tables 1–3 are point estimates. For the cardiac EP experiment with multiple test trajectories from one geometry, reporting mean ± std over trajectories would be straightforward and would quantify prediction variability under varying initial conditions.

### Tiny

- The abstract states the method "significantly speeds up prediction capabilities compared to traditional finite-element solvers" without clarifying the CPU vs. GPU comparison. The main text does note this, but abstract precision would prevent misreading.
- The value of $p$ used in the cardiac EP and reaction-diffusion experiments is not stated, hindering exact reproducibility.
- The phrase "can be viewed as performing dimension reduction and homogenization" (page 2) is presented as an interpretive frame without formal justification; clarifying this is a conceptual analogy rather than a derived theorem would prevent misunderstanding.

---

## Nice-to-Haves

- **Visualization of adapted eigenvectors vs. standard (isotropic) Laplacian eigenvectors** to illustrate concretely how the physics-informed basis differs from a purely geometric one — this would be compelling for readers.
- **Pointwise error heatmaps** (rather than solution field comparisons only) to reveal whether errors concentrate at wavefronts, boundaries, or fiber discontinuities.
- **Eigenvector-matching procedure** to address the ordering-mismatch issue already identified by the authors as a source of error; even a simple experiment comparing magnitude-sorted vs. unmatched eigenvectors would strengthen the discussion.
- **Extended temporal evaluation** (beyond 90ms) to assess whether integration errors grow unstably over longer time horizons relevant to clinical AF simulation.
- **Accuracy vs. total time Pareto frontier** (including eigendecomposition preprocessing and training amortization) comparing G-FuNK against FEM at varying accuracy targets.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[Removed] G-FuNK underperforming FNO on the heat equation as evidence of weakness.** The paper explicitly states that G-FuNK was intentionally *not* given the fiber field as input (learning anisotropy from eigenvectors alone), while FNO received the fiber field directly. This asymmetry deliberately favors the baseline (FNO), not G-FuNK, making the point stronger that G-FuNK can operate without explicit fiber input. Per the review rules, comparisons where the unfairness is beneficial to the baseline and not the proposed method should be removed from weaknesses.

- **[Removed] GINO omitted from 3D cardiac EP comparison.** The paper explicitly justifies GINO's inapplicability: the left atrium with five holes cannot be mapped diffeomorphically to shapes GINO handles (the paper also notes GINO requires abundant geometrically varying training samples, not available in this clinical setting). This is a clear and domain-justified exclusion, not a gap. Additionally, per review rules, claims about cited methods' unavailability or inapplicability should not be penalized without external evidence.

- **[Removed] Missing code/data availability statement.** While reproducibility is important, the paper is under double-blind review and absence of a code repository is a standard condition at submission time, not a scientific weakness. This is a venue-logistics concern, not a technical flaw.

- **[Removed] Boundary conditions limited to Neumann.** The paper explicitly scopes its contribution to no-flux Neumann BCs (Eq. 2), which matches all three experimental settings. Criticizing the absence of Dirichlet/Robin BC handling is scope creep for a paper that does not claim generality beyond its stated PDE family.

- **[Removed] "Reduced Models and Homogenization" paragraph is informal.** This is a pure style/framing nitpick. The paper does clarify this is an interpretive frame, not a theorem.

---

## Novel Insights

The most genuinely insightful observation across the three reviews — not simply restating the paper's contributions — concerns the **theoretical status of Eq. 8**. The edge weight formula has the structure of a harmonic mean of anisotropic resistances (motivated by resistor network analogies for graph discretizations of divergence-form operators), and there exists a body of literature on consistent discretizations of $\nabla\cdot(\mathbf{K}\nabla)$ on point clouds and irregular meshes (related to meshfree methods and anisotropic random walks). If the authors can connect Eq. 8 to known consistent discretization theory — even for the $k$-NN graph case — the method's justification would move from "physically motivated heuristic" to "principled approximation," which would substantially elevate the theoretical contribution. A second non-obvious insight is the relationship between the $\mathcal{L}_n$ polynomial eigenvalue expansion and spectral GNN filter design (ChebNet, GPRGNN), which the paper does not acknowledge; making this connection explicit would both clarify the architecture's role and position the work within a richer theoretical context.

---

## Suggestions

1. **Provide a derivation or literature citation** establishing that Eq. 8 weights produce a consistent approximation of $\nabla\cdot(\mathbf{K}\nabla)$ on $k$-NN graphs, at least for uniformly distributed nodes; even an asymptotic result or connection to known meshfree discretizations would substantially strengthen the paper's core claim.

2. **Add an ablation comparing physics-adapted edge weights (Eq. 8) against standard geometric weights** (e.g., $w_{ij} = 1/\|x_i - x_j\|^2$) on the cardiac EP task to demonstrate that the specific diffusion-encoding drives performance, not merely graph density.

3. **Perform leave-one-out cross-validation on the 25 available cardiac geometries** to provide statistically robust evidence for generalization; this requires no new data and directly addresses the single-test-geometry limitation.

4. **State the value of $p$ (polynomial power in $\mathcal{L}_n$) and $k_{\max}$ explicitly for all experiments**, and add a brief sensitivity analysis (even a 3-point sweep) on $k_{\max}$ to justify the chosen truncation.

5. **Add a mesh-resolution transfer experiment**: train on a 2D heat equation mesh at one density, evaluate on 2× finer mesh, to empirically substantiate the mesh-independence claim in Section 2.2.

6. **Separate eigendecomposition time from neural network inference time** in the benchmarking section to give a complete and honest picture of per-patient prediction cost.

---

**Overall evaluation:**
- *Novelty:* High — the physics-adapted graph spectral basis for PDE operator learning is a specific and non-obvious contribution that distinguishes G-FuNK from both GNNs and FNOs.
- *Technical soundness:* Moderate — the architecture is well-designed and self-consistent, but the core approximation (Eq. 8) lacks formal justification, which is a real gap for a theoretically-positioned contribution.
- *Empirical support:* Moderate — results are promising but the cardiac EP claim rests on a single test geometry, ablations are absent, and mesh-independence is unverified experimentally.
- *Significance:* High — the cardiac EP application is clinically impactful, the parameter efficiency advantage is meaningful, and the rotation-invariance result is a concrete win.
- *Clarity:* Good — the architecture is reasonably clearly described, though the $\mathcal{L}_n$ expansion and its hyperparameters need clarification; the motivation for each component is generally well-articulated.