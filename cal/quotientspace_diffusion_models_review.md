=== CALIBRATION EXAMPLE 88 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title "Quotient-Space Diffusion Models" accurately reflects the core contribution: a diffusion modeling framework for quotient spaces arising from symmetry.
- The abstract clearly states the problem (symmetry in diffusion models), the proposed solution (quotient-space diffusion), and claims advantages: reduced learning difficulty, guaranteed correct sampling, and empirical improvements. These claims are substantiated in the paper.

### Introduction & Motivation
- The motivation is strong: many scientific tasks (e.g., molecular generation) exhibit symmetry, and existing treatments (equivariant models, alignment heuristics) have limitations. The need for a principled approach is well-argued.
- Contributions are clearly listed: a general framework for diffusion on quotient spaces, derivation of the diffusion process and horizontal lift, specialization to SE(3), practical algorithms, and empirical validation. The contributions are novel and significant.

### Background (Section 2)
- Section 2.1 provides a concise review of diffusion models (stochastic interpolant framework) and sets up notation. It is clear and sufficient.
- Section 2.2 introduces necessary concepts from differential geometry (manifolds, Lie groups, quotient spaces, tangent vectors, Wiener processes). This section is dense but necessary. The paper assumes a high level of mathematical maturity, which is appropriate for ICLR but may challenge some readers. No major issues.

### Methods (Section 3)
- **Section 3.1 (Theoretical Framework):** Theorems 1 and 2 are the core theoretical contributions. Theorem 1 shows that projecting a diffusion process onto the quotient space yields an SDE with a curvature correction term. Theorem 2 provides the horizontal lift back to the total space. The proofs are deferred to the appendix and appear sound. The intuition (removing unnecessary vertical movements) is well-explained.
- **Section 3.2 (Specialization to Shape Space):** Theorem 4 gives explicit formulas for the horizontal projection \(P_{\mathbf{x}}\) and the mean curvature vector \(\tilde{\mathbf{h}}(\mathbf{x})\) for the \(\mathbb{R}^{3N}/SE(3)\) case. These formulas are essential for implementation and seem correct. A minor concern: the derivation assumes the group action is free, which requires excluding degenerate configurations (e.g., collinear points). The paper notes this is a measure-zero set and handles it numerically.
- **Section 3.3 (Practical Implementations):** The training objective (Eq. 10) projects the denoising error onto the horizontal subspace, allowing arbitrary vertical components. The ODE and SDE samplers incorporate the horizontal projection and curvature correction. Algorithms 1-3 are clear. A hyperparameter \(\gamma\) is introduced for the SDE sampler; its choice is motivated by prior work but not ablated.
- **Section 3.4 (Analysis of Existing Methods):** This is a strong section. Table 1 compares conventional equivariant diffusion, GeoDiff alignment, AF3 alignment, and the proposed quotient-space diffusion on learning difficulty and sampling compatibility. The analysis correctly identifies that alignment methods lack proper samplers, while the proposed method guarantees correct sampling and reduces learning difficulty. The discussion is insightful and well-supported by Figure 3.
- **Potential Concerns:**
    1. The curvature correction term \(\tilde{\mathbf{h}}(\mathbf{x})\) in the SDE sampler: its necessity and impact are not empirically ablated. Does it significantly affect performance?
    2. Numerical stability: computing \(\mathbf{J}^{-1}\) can be unstable for near-degenerate configurations. The appendix mentions adding \(\epsilon \mathbf{I}\), which is standard, but sensitivity is not discussed.
    3. The training objective Eq. (10) uses a projection that depends on \(\mathbf{x}_t\). This is valid, but it might be worth noting that the gradient through \(P_{\mathbf{x}_t}\) is straightforward (it is a linear projection).

### Experiments & Results (Section 4)
- **Section 4.1 (Small Molecules):** Experiments on GEOM-QM9 and GEOM-DRUGS show consistent improvements over baselines (including ET-Flow, GeoDiff, MCF) and heuristic alignments. The improvements are modest but meaningful (e.g., median AMR reduction from 0.030 to 0.024 on QM9). The experimental setup is standard and fair. The paper transparently notes that reproduced ET-Flow results differ due to data processing changes.
- **Section 4.2 (Protein Backbone Design):** Comparisons with Proteına demonstrate that the quotient-space diffusion outperforms the 60M parameter baseline and even the 200M model on many metrics when using SDE sampling. For ODE sampling, distributional metrics (FPSD, fJSD) improve, though designability remains low (consistent with the baseline). The failure of alignment methods (AF3) aligns with the theoretical analysis. The results are compelling.
- **Missing Ablations:**
    1. The contribution of the curvature term \(\tilde{\mathbf{h}}(\mathbf{x})\) is not isolated. An ablation study (e.g., removing it from the SDE) would clarify its importance.
    2. Separately evaluating the effect of horizontal projection during training vs. sampling would be insightful.
    3. Sensitivity to the hyperparameter \(\gamma\) in SDE sampling is not explored.
- **Additional Experiments:** Appendix F includes training/sampling convergence speed comparisons (Figure 5), showing that the quotient-space method converges faster in training and performs better at all NFEs. This is valuable.

### Writing & Clarity
- The paper is well-written but mathematically dense. The authors provide good intuitions (Figure 1, Table 1) to guide readers. The structure is logical.
- Minor clarity issues:
    - In Table 1, the notation \(A_{\mathbf{y}}(\mathbf{x})\) is introduced before its definition (Eq. 11). This is acceptable.
    - The notation switches between bold and non-bold for manifold points; consistency could be improved.
    - Some readers may struggle with the differential geometry terminology, but the paper includes a detailed appendix.

### Limitations & Broader Impact
- Limitations are not explicitly discussed in the main text. The appendix notes numerical stability and the exclusion of degenerate configurations. Other limitations: the framework assumes a free and proper group action; for non-compact groups (translation), a CoM subspace is used. Generalization to other symmetries is discussed but not tested.
- Broader impact: The work advances generative modeling for scientific applications (molecules, proteins). Potential negative impacts (e.g., misuse for drug design) are common to the field and are acknowledged in the ethics statement.

### Reproducibility
- The paper provides detailed algorithms, theoretical derivations in the appendix, and experimental settings. Code and checkpoints will be released. Reproducibility appears strong.

## Overall Assessment

This paper introduces a principled framework for diffusion models on quotient spaces, addressing the important problem of symmetry in generative modeling. The theoretical contributions are novel and sound, providing a rigorous foundation that unifies and improves upon existing heuristic approaches. The analysis of existing methods (Table 1) is particularly insightful. Empirical results on small molecules and proteins demonstrate consistent improvements over strong baselines, validating the practical benefits. While some ablations (e.g., the curvature term) and deeper empirical analysis would strengthen the paper, the core contributions are significant and well-supported. The paper meets ICLR's standards for novelty, technical soundness, and empirical validation. It is likely to influence the field of geometric deep learning and generative modeling for science. **Recommendation: Accept, pending minor revisions to address the missing ablations and limitations discussion.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a principled framework for building diffusion models on quotient spaces to handle inherent symmetries in generative tasks. Theoretically, it derives the correspondence between diffusion processes on the total space and the quotient manifold, and provides a practical implementation via horizontal lifts, removing the need to learn unnecessary degrees of freedom associated with group actions. The method is instantiated for SE(3) symmetry in molecular structure generation, and empirical results on small molecules and protein backbones demonstrate improved performance over equivariant diffusion baselines and heuristic alignment strategies.

### Strengths
1. **Novel theoretical foundation**: The paper provides rigorous mathematical derivations (Theorems 1, 2, 4) that formally connect diffusion processes on the total space and the quotient space, offering a principled way to handle symmetry. This framework generalizes beyond SE(3) and addresses limitations of heuristic alignment methods.
2. **Clear practical algorithms and efficient implementation**: For the SE(3) case, explicit formulas for the horizontal projection and mean curvature vector are derived (Theorem 4), leading to concrete training and sampling algorithms (Algorithms 1-3). The computational overhead is minimal (linear in the number of atoms) and comparable to alignment operations.
3. **Strong empirical validation**: The method shows consistent improvements on standard benchmarks: on GEOM-QM9 and GEOM-DRUGS, it outperforms ET-Flow and other baselines (Table 2); for protein backbone generation, a 60M-parameter model surpasses the comparable Proteína model and even the larger 200M model on key distributional metrics (Table 3).
4. **Insightful analysis of existing methods**: Table 1 provides a clear comparison of symmetry-handling strategies, highlighting that heuristic alignments (GeoDiff, AF3) either do not fully reduce learning difficulty or lack sampling compatibility, while the quotient-space approach addresses both.
5. **Reproducibility**: The paper includes detailed appendices with proofs, implementation notes, and commitments to release code. Experiments use standard datasets and follow established evaluation protocols.

### Weaknesses
1. **Limited experimental scope beyond SE(3)**: While the theory is general, the experiments focus solely on SE(3) symmetry for molecular structures. Applications to other symmetries (e.g., U(1), SU(2), periodic translations) are mentioned but not empirically validated, limiting the demonstrated breadth of the framework.
2. **Theoretical assumptions may not always hold**: The construction requires the group action to be smooth, free, proper, and isometric. For SE(3), degenerate configurations (e.g., collinear points) are excluded to ensure freeness. The practical impact of these degeneracies and the robustness of the epsilon-regularization used in matrix inversions are not thoroughly analyzed.
3. **Modest gains on some benchmarks**: On GEOM-DRUGS, improvements over the strong ET-Flow baseline are relatively small (e.g., Coverage 78.50 vs. 78.18, Table 2). For protein generation, while the method outperforms the comparable Proteína model, the margin on some metrics (e.g., designability) is not dramatic.
4. **Accessibility challenge**: The heavy reliance on differential geometry and stochastic calculus may hinder understanding for a broader machine learning audience. Key concepts like the mean curvature vector and horizontal lift could benefit from more intuitive explanations in the main text.
5. **Incomplete comparison with recent work**: The protein experiments do not include a direct comparison with the full Proteína model (200M) under all metrics, and other recent methods like Boltz-1 are only discussed theoretically without empirical comparison.

### Novelty & Significance
**Novelty**: The formulation of diffusion models on quotient spaces via horizontal lifts is a novel and principled approach to symmetry handling. It distinguishes itself from prior heuristic alignments and equivariant architectures by providing a theoretical guarantee for sampling correctness while reducing learning difficulty.
**Significance**: The work addresses a fundamental challenge in generative modeling for scientific domains where symmetries are inherent. By offering a rigorous framework that improves performance and simplifies learning, it has the potential to influence how symmetry is incorporated not only in diffusion models but also in other generative approaches. The empirical gains on molecule and protein generation tasks demonstrate its practical relevance.

### Suggestions for Improvement
1. **Broaden experimental validation**: Apply the framework to at least one additional symmetry (e.g., periodic translation in crystals or U(1) in quantum systems) to better demonstrate generality. Also, consider testing on larger protein complexes or other biomolecular structures.
2. **Conduct ablation studies**: Analyze the contribution of the mean curvature correction term in the SDE sampler to understand its necessity in practice. Also, investigate the sensitivity to the epsilon parameter used for numerical stability in matrix inversion.
3. **Improve accessibility**: Add a more intuitive, example-driven explanation of key geometric concepts (e.g., horizontal projection, mean curvature) in the main text, possibly with a simple 2D illustration beyond Figure 1.
4. **Extend empirical comparisons**: Include a full comparison with the 200M-parameter Proteína model on all metrics, and if possible, compare with other recent symmetry-aware methods like Boltz-1 on protein tasks.
5. **Ensure prompt code release**: As promised, release the full codebase and trained models upon acceptance to facilitate reproducibility and adoption by the community.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Convergence speed comparison under *sampling-correct* baselines.** The paper compares training speed against AF3/GeoDiff alignment, which the authors argue are incompatible. To truly show "reduced learning difficulty," they must compare against a sampling-compatible baseline (e.g., a standard equivariant model or data-augmented model) using identical sampling steps, measuring performance vs. training iterations/epochs. Without this, the claim of easier learning is unsupported.
2. **Ablation on the necessity of the mean curvature term (`h̃(x)`).** Theorem 4 introduces a correction term. An ablation study removing this term during training/sampling is essential to demonstrate its practical importance for distributional recovery, especially on the protein task where they introduce a hyperparameter `γ`. Its omission could undermine the theoretical guarantee.
3. **Evaluation with invariant metrics and more sampling steps.** The molecular generation results rely heavily on RMSD-based metrics (Coverage, AMR), which are sensitive to global rotations. They should report metrics that are SE(3)-invariant by construction (e.g., after optimal alignment) to isolate shape modeling performance. Additionally, sampling is done with only 50 NFEs; a plot of metric vs. NFE is needed to confirm their method's efficiency advantage isn't just an artifact of this fixed, low-NFE setting.
4. **Direct comparison against other symmetry-reducing methods.** The paper lacks comparison to conceptually related methods like the periodic-translation invariant diffusion for crystals (Lin et al., 2024) or the CoM-subspace method (Cornet et al., 2025, mentioned in Appendix A). A comparison on a shared task (e.g., molecular generation) would better contextualize the contribution.

### Deeper Analysis Needed (top 3-5 only)
1. **Attribution of performance gains.** The improvement over the baseline (e.g., ET-Flow) could stem from: (a) reduced learning difficulty (as claimed), (b) the mean curvature correction, or (c) a more stable sampler. The paper must disentangle these by analyzing, e.g., the variance of the model's vertical output during training and its correlation with final performance.
2. **Analysis of failure modes and the degenerate set `D`.** The method excludes a measure-zero set `D` (e.g., collinear points) for mathematical rigor. A practical analysis is needed: How often do training/generation trajectories approach this set? Does the `ϵI` stabilization (Appendix F.1) handle these cases robustly? Visualizing generated structures near degeneracy would build confidence.
3. **Generalization analysis to other groups.** The theoretical framework is general, but experiments are only on SE(3)/SO(3). A minimal case study on a simpler group (e.g., 2D rotations SO(2) as in Fig. 1, or translations) with a synthetic distribution would validate the general theory and show the method works beyond the specific `P_x` and `h̃(x)` derived for SE(3).

### Visualizations & Case Studies
1. **Visual decomposition of the learned velocity field.** For a set of generated molecules/proteins, plot the horizontal (`v^H`) and vertical (`v^V`) components of the learned velocity `v_θ` over time. This would visually confirm the claim that the model learns arbitrary vertical components while the sampler projects them out.
2. **Sampling trajectory visualization in the quotient space.** Similar to Fig. 1 but for real data (e.g., a small molecule), show the projection of sampling trajectories (`π(x_t)`) for the baseline equivariant model vs. their quotient-space model. This would illustrate the "shorter trajectory" claim (Corollary 3) and show less intra-class motion in practice.
3. **Failure case study for alignment methods.** The paper argues GeoDiff/AF3 alignment leads to incorrect sampling. A clear case study on a simple synthetic distribution (e.g., a diatomic molecule as in Fig. 3) showing the empirical distribution generated by these methods' samplers diverging from the true invariant target would powerfully support their criticism.

### Obvious Next Steps
1. **Controlled experiment isolating "learning difficulty".** Train two models: (A) with the proposed horizontal projection loss (Eq. 10), and (B) with a standard MSE loss but using an *equivariant architecture* that forces `D_θ(x_t)` to be horizontal. Both guarantee sampling correctness. Comparing their convergence would directly test if the benefit comes from the relaxed output space versus the architecture constraint.
2. **Application to a different symmetry group.** Implement the framework for another group mentioned in Appendix F.4 (e.g., U(1) symmetry on periodic angles) to demonstrate generality beyond molecular SE(3). This would significantly strengthen the paper's scope.
3. **Open-source code and benchmark with quotient space metrics.** Release code with the core projection operators (`P_x`, `h̃(x)`) to facilitate adoption. Furthermore, compute and report key metrics (like FPSD, fJSD for proteins) directly on the quotient space representation (e.g., after alignment to a canonical frame) to provide a purer evaluation of shape generation.

# Final Consolidated Review
## Summary
This paper develops a principled framework for constructing diffusion models on quotient spaces to handle inherent symmetries in generative tasks. Theoretically, it establishes the correspondence between diffusion processes on the total space and the quotient manifold via horizontal lifts. Practically, it derives explicit formulas for the SE(3) case (molecular structures), yielding a training objective that relaxes learning in the symmetric degrees of freedom and a sampler that guarantees recovery of the target distribution. Empirical results on small molecule and protein backbone generation demonstrate consistent improvements over strong equivariant baselines and heuristic alignment methods.

## Strengths
- **Novel and rigorous theoretical foundation:** Theorems 1, 2, and 4 formally derive the diffusion process on a general quotient space and its horizontal lift back to the total space, providing a principled solution that unifies and improves upon heuristic symmetry treatments. The analysis in Section 3.4 and Table 1 clearly delineates why previous alignment methods lack sampling compatibility while the proposed approach guarantees it.
- **Effective instantiation and implementation:** For the practically crucial SE(3) case, Theorem 4 provides explicit, efficient-to-compute formulas for the horizontal projection \(P_{\mathbf{x}}\) and the mean curvature vector \(\tilde{\mathbf{h}}(\mathbf{x})\). The resulting training (Eq. 10) and sampling algorithms (Algorithms 1-3) are concrete, add minimal overhead, and are shown to be numerically stable in practice.
- **Compelling empirical validation:** The method consistently improves generation quality on standard benchmarks. On GEOM-QM9/DRUGS, it outperforms ET-Flow and heuristic alignments (Table 2). In protein backbone design, a 60M-parameter model surpasses the comparable Proteína model and even a larger 200M model on key distributional metrics under SDE sampling (Table 3), demonstrating the practical benefit of reduced learning difficulty and correct sampling.

## Weaknesses
- **Lack of ablation for the curvature correction term:** The mean curvature vector \(\tilde{\mathbf{h}}(\mathbf{x})\) appears in the derived SDE sampler (Theorem 4, Eq. 9). While theoretically necessary for the distributional guarantee, its practical impact on performance is not isolated. An ablation study (e.g., removing it or varying its weight \(\gamma\)) would clarify its empirical importance and strengthen the validation of the theoretical construction.
- **Empirical scope is currently narrow:** The theoretical framework is general, but all experiments focus on the SE(3)/SO(3) symmetry for molecular structures. While this is a highly relevant and complex case, demonstrating the method on at least one other symmetry group (even a simpler synthetic example) would better substantiate the claim of generality and help the community apply it to new domains.

## Nice-to-Haves
- A more extensive analysis of the degenerate set \(D\) (e.g., collinear points) in practice, beyond the mention of \(\epsilon\)-regularization, could further assure robustness.
- Including a direct comparison of the horizontal projection loss (Eq. 10) against a standard MSE loss with a strictly horizontal, equivariant architecture could help disentangle the benefit of the relaxed output space from the architectural constraint.

## Novel Insights
The paper's core novel insight is the formal construction of a diffusion model that operates directly on the quotient space (the space of essential states) via a horizontally lifted process in the total space. This yields a training loss that does not penalize the model for arbitrary output within the symmetric (vertical) degrees of freedom, genuinely reducing learning difficulty, while the sampler—by projecting updates to the horizontal subspace—provably recovers the correct target distribution. This insight cleanly explains the incompatibility of heuristic alignment methods (their training target differs from what their sampler requires) and provides a unified, correct solution.

## Suggestions
- Conduct an ablation study to quantify the effect of the mean curvature term \(\tilde{\mathbf{h}}(\mathbf{x})\) on sampling fidelity and generation quality, especially for the protein SDE sampler where the hyperparameter \(\gamma\) is introduced.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 10.0, 6.0]
Average score: 7.5
Binary outcome: Accept
