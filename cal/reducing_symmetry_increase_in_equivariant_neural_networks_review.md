=== CALIBRATION EXAMPLE 39 ===

# Final Consolidated Review
## Summary

This paper provides a rigorous mathematical framework to characterize and mitigate "symmetry increase" in equivariant neural networks—the phenomenon where outputs become invariant to transformations beyond the input's inherent symmetries. The core contribution is the concept of a "symmetry infimum," a unique lower bound on symmetry increase determined by the algebraic structure of the feature space, along with algorithms to compute it and density results showing that almost-isovariant maps are generic under standard regularity assumptions.

## Strengths

- **Novel conceptual framework:** The symmetry infimum (Thm 3.1) and its uniqueness provide the first precise, computable bound on how much symmetry *must* increase for a given feature space and input symmetry. This generalizes prior observations (e.g., Cen et al. 2024's "collapse-to-zero" is a special case of full degeneration) into a predictive, structural theory.
- **Bridge from abstract theory to practical computation:** The paper translates orbit-type classification—a classical topic from bifurcation theory—into concrete algorithms (Algo 1 & 2) and produces exhaustive tables of symmetry infima for all closed subgroups of SO(3) and O(3) (Appendix E). This makes the framework actionable for the most common symmetry groups in scientific ML.
- **Clear taxonomy of degeneration types:** The classification into full, axial/continuous, and half/discrete degeneration (Fig 2, §C.4) gives practitioners a precise vocabulary for diagnosing specific expressivity failures, replacing vague prior observations with a structured theory that predicts exactly *which* feature degrees cause *which* type of information loss.

## Weaknesses

- **Experimental validation is observational, not interventional.** The QM9 experiment (§6.3) shows that molecules with symmetries causing full degeneration exhibit elevated MAE—a post-hoc confirmation of the predicted degradation. However, the paper never compares a model *designed according to the proposed guidelines* against an unconstrained baseline trained from scratch under identical conditions. Without this controlled ablation, the claim that the framework provides "practical guidelines for designing more reliable ENNs" (§7) is not empirically substantiated—only the prediction of degradation is confirmed, not the remedy.

- **Theory–experiment architecture mismatch on the most practical experiment.** The theoretical density result (Thm 5.1) and its proof (Appendix D.2) are specialized to TFN's tensor-product parameterization. The QM9 experiment, however, uses HEGNN because "TFN is computationally prohibitive" (§F.3.1). HEGNN uses spherical scalarization, a fundamentally different parameterization. The paper does not establish that the symmetry infimum predictions or the C∞-density result transfer to scalarization-based architectures, leaving the strongest practical claim disconnected from the theory that supports it.

- **Evaluation scope is narrow and does not test the core orientation-preservation claim.** The only real-world experiment uses a single dataset (QM9) and a single target (isotropic polarizability α), which is an orientation-invariant scalar. The guidelines in §4.2 distinguish between "orientation-dependent tasks" (where isovariance is crucial) and "general tasks" (where certain symmetry increases may be acceptable). Testing only an invariant scalar target cannot validate the framework's primary claim about preserving orientational information. Vector/tensor targets (dipole moments, forces) are natural next steps that go untested.

- **Missing comparison to alternative symmetry-handling methods.** The paper proposes feature-space design as a remedy for symmetry increase, but does not compare against existing practical solutions such as frame averaging (Puny et al. 2022), random reference frames, or symmetry-breaking via noise injection. Without such comparisons, the practical advantage of the proposed guidelines over established remedies is unknown.

- **Genericity result does not address trained networks.** Theorem 5.2 guarantees that almost-isovariant maps are dense in the space of smooth equivariant maps, given C∞ approximation capability. However, gradient-based training on finite data does not sample uniformly from this function space; training dynamics and implicit regularization can systematically steer learned functions away from generic configurations. The paper provides no evidence that networks trained by SGD on realistic datasets actually achieve near-isovariance in practice.

- **The high-multiplicity assumption requires clarification relative to single-representation claims.** Prop 4.2 (sufficiency of Michel's Criterion) requires multiplicity r > dim G = 3 for O(3), yet §4.1 states predictions are "identical for the single representation case (r = 1)." In §C.4, the paper acknowledges differences for specific subgroups (C_k, S_{2k}, C_{kh}) due to non-zero Ihrig–Golubitsky correction terms. The main-text claim of equivalence is therefore an oversimplification; the boundary conditions where the high-multiplicity regime diverges from r = 1 should be stated upfront.

- **No computational complexity analysis for the proposed algorithms.** Algorithms 1 and 2 require enumerating all adjacent closed supergroups of H in G and computing fixed-point subspace dimensions. For O(3), the subgroup lattice is intricate (Tables 6–7). Without any complexity analysis, it is unclear whether these algorithms are tractable for practitioners designing models with high-degree features or unusual symmetry groups, or whether they scale to larger groups.

- **The C_∞ bottleneck limitation is buried in the appendix.** Section C.5 establishes that for SO(3), all subgroups satisfy the bottleneck condition and the composition property O_G(V_1 ⊕ V_2) = O_G(V_1) ∪ O_G(V_2) holds. For O(3), C_∞ does *not* satisfy this condition, meaning the simple composition rule fails and constructing representations containing C_∞ as an orbit type requires selecting components of both parities simultaneously. This practical limitation on the guidelines is discussed only in the appendix (§C.5, last paragraph), where it belongs in §4.2.

- **The manifold hypothesis is strong and its satisfaction by molecular data is unverified.** Theorem 5.2's sufficiency result assumes data is supported on a finite union of smooth, compact G-submanifolds. Molecular datasets like QM9 involve discrete atom types, fixed atom counts, and potential near-degenerate geometries. The paper acknowledges this assumption in Appendix A.3 but does not discuss what happens to the density result when it is violated (e.g., if the data manifold has singularities or corners).

## Nice-to-Haves

- A software tool or library extension that automates symmetry infimum computation for common point groups would substantially lower the adoption barrier for practitioners without representation-theory expertise.
- An ablation on representation multiplicity (channel count) explicitly testing Theorem 5.2's prediction that increasing r beyond dim M_j enables full isovariance.
- Discussion of robustness to approximate symmetries (noisy inputs, thermal fluctuations) that break exact group structure, which is common in molecular dynamics.
- Evaluation on orientation-dependent tasks (e.g., dipole moment or force prediction) where the framework's distinction between orientation-dependent and general-task guidelines can be directly tested.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Example 2.2 indexing ambiguity"** — The paper explicitly states "x_i = (cos(2iπ/k), sin(2iπ/k), 0) for i > 0" with "x_0 at the origin" in Eq. 2. The indexing is unambiguous; the reviewer misread it.
- **"Curie's Principle attribution"** — Citing Kaba & Ravanbakhsh (2023) for a specific mathematical formulation of a classical principle is standard practice. Formatting nitpick.
- **"Compact Lie group assumption limits applicability to finite groups"** — Finite groups are 0-dimensional compact Lie groups; they are included in the assumption. The reviewer's concern is based on a misunderstanding.
- **"Figure/Table formatting issues"** — All garbled figures/tables are PDF parsing artifacts, not paper problems.
- **"Notation inconsistency (O_G vs O(3))"** — O_G(X) denotes orbit types; O(3) denotes the orthogonal group. The subscript/superscript distinction is clear in context. Style nitpick.
- **"Table 21 statistical significance"** — 50% accuracy on binary classification unambiguously indicates complete failure; no significance test is needed.
- **"Paper too mathematically dense"** — This is a theoretical paper at a top ML venue; mathematical depth is expected for this type of contribution. Style/formatting nitpick.
- **"When symmetry increase is beneficial"** — The paper explicitly states in §3.2: "This increase can be an intentional design choice" and distinguishes designed vs. unintended symmetry increase. The reviewer missed this.
- **"Trivial kernel assumption should not be the default"** — The paper handles both cases (§3.1 trivial, §3.2 non-trivial) in a standard mathematical exposition order. This is a presentation preference.
- **"Generic ethics statement"** — This is a theoretical math paper with no human subjects or deployed systems; a generic ethics statement is appropriate.

## Novel Insights

The symmetry infimum framework reveals a striking structural fact: for any equivariant map, the *minimum* achievable output symmetry is not a property of the map but is entirely determined by the feature space's orbit type structure. This means the expressivity limit is an architectural constraint, not a training failure—no amount of optimization can make an equivariant map preserve input symmetries that the feature space cannot support. The practical consequence is that feature-space design (choosing which irreps to include) is a zeroth-order decision that must precede any optimization, and the paper's tables provide a lookup for making this decision correctly for SO(3)/O(3).

## Suggestions

- Run a controlled ablation on QM9: train two models from scratch—one following the feature selection guidelines (e.g., excluding degrees predicted to cause full degeneration for each molecular symmetry) and one using a standard feature set—and compare MAE across symmetry groups. This directly tests whether the guidelines provide practical value.
- Add at least one orientation-dependent target (e.g., dipole moment from QM9) to validate the framework's core claim about preserving orientational information under the "orientation-dependent task" guidelines.
- Include a brief discussion in §4.2 about the C_∞ bottleneck limitation for O(3), noting that the simple composition property fails for this subgroup and that both parity components must be selected simultaneously.

## Evaluation by Axis

- **Novelty:** High. The symmetry infimum concept and its uniqueness proof are novel contributions that generalize prior empirical observations and partial theories into a unified, predictive framework.
- **Technical soundness:** Good. The theoretical development is rigorous with complete proofs in the appendix. However, the disconnect between TFN-specific theory and HEGNN-based experiments, and the oversimplified claim about r=1 equivalence, create gaps between what is proven and what is claimed.
- **Empirical support:** Moderate. Visualizations and synthetic experiments convincingly validate theoretical predictions about *degradation*, but the practical *remedy* (guidelines) lacks controlled interventional validation, and the most important experiment uses an architecture not covered by the theory.
- **Significance:** High if the practical gap is addressed. The framework provides the first principled way to predict and control symmetry increase in ENNs, which is a foundational issue in geometric deep learning. The current gap between theoretical predictions and controlled empirical validation of the remedy limits immediate practical impact.
- **Clarity:** Moderate. The paper is mathematically dense and requires significant background in representation theory and equivariant topology. Key practical limitations (C_∞ bottleneck, r=1 discrepancies) are deferred to the appendix. The main text could be more self-contained for the ML audience.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 8.0]
Average score: 6.7
Binary outcome: Accept
