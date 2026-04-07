=== CALIBRATION EXAMPLE 90 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: a diffusion modeling framework on quotient spaces. The abstract clearly states the motivation (symmetry in scientific tasks), the approach (formal quotient-space framework, horizontal lift), and key advantages (reduced learning difficulty, sampler correctness). All claims are substantiated in the paper.

### Introduction & Motivation
The problem is well-motivated by the prevalence of symmetries in scientific domains, particularly molecular generation. The limitations of existing treatments (equivariant models, heuristic alignments) are clearly explained. The contributions are explicitly listed: a formal framework for quotient-space diffusion, reduction of learning difficulty, guaranteed sampling correctness, and application to SE(3) symmetry with empirical validation. The introduction effectively sets up the paper.

### Method / Approach
The theoretical development is rigorous. Theorems 1 and 2 establish the diffusion process on a general quotient space and its horizontal lift. Theorem 4 provides the explicit specialization to the SE(3) case (shape space), deriving the horizontal projection \(P_x\) and mean curvature vector \(\tilde{h}(x)\). The practical implementation (training objective Eq. 10, Algorithms 1–3) is clearly presented and reproducible.

**Key strengths:** The mathematical derivation is sound, with proofs detailed in the appendix. The analysis of existing methods (Sec. 3.4) is insightful, correctly identifying that GeoDiff alignment changes the learning target incompatibly with standard samplers, and that AF3 alignment introduces arbitrariness breaking sampling guarantees.

**Potential concerns / questions:**
1. **Numerical stability:** The expression for \(P_x\) involves the inversion of \(\mathbf{J}\), which can be singular for collinear point configurations. The authors mention adding \(\epsilon \mathbf{I}\) for regularization, which is standard, but a brief discussion of how sensitive results are to \(\epsilon\) would be helpful.
2. **Ablation of the mean curvature term:** The term \(\tilde{h}(x)\) appears in the SDE sampler (Algorithm 3). Its effect on performance is not ablated; it would be useful to know if it is essential or merely a minor correction.
3. **Assumptions:** The framework assumes a smooth, free, proper, and isometric group action. The paper notes that for SE(3), degenerate (collinear) configurations are measure-zero and can be handled, but a short discussion on robustness to near-degenerate cases in practice would be beneficial.
4. **Computational overhead:** The projection \(P_x\) requires \(O(N)\) operations per step. The authors argue this is negligible compared to neural network costs, which is reasonable for the tested tasks, but could become relevant for extremely large \(N\) (e.g., full-atom protein systems). A brief comment on scalability is warranted.

### Experiments & Results
Experiments are comprehensive, covering small molecule (GEOM-QM9, GEOM-DRUGS) and protein backbone generation. The baselines are appropriate and state-of-the-art. The experimental setup is clearly described, and the authors carefully match architectures, training data, and sampling steps for fair comparisons.

**Strengths:**
- Consistent improvements over strong baselines (ET-Flow, Proteına) and heuristic alignment methods.
- On proteins, the 60M parameter quotient-space model outperforms the 200M parameter Proteına model on several distributional metrics, demonstrating efficiency gains.
- The failure of AF3/GeoDiff alignments in distributional metrics (Table 3) validates the theoretical incompatibility argument.

**Potential concerns / questions:**
1. **Statistical significance:** Metrics are reported as point estimates without confidence intervals or standard deviations across multiple runs. For the GEOM test sets (n=1000), reporting error bars or significance tests would strengthen the claims.
2. **Ablation studies:** Missing ablations on the importance of the horizontal projection alone versus the full framework (including the curvature term), and the effect of the hyperparameter \(\gamma\) in the SDE sampler.
3. **Broader applicability:** The experiments focus solely on SE(3) symmetry. While this is a major application, a small experiment on another symmetry group (e.g., a toy example with a different group) could better demonstrate the generality of the framework.
4. **Protein evaluation caveat:** The authors transparently note a bug in prior Foldseek versions affecting TM-score metrics and provide corrected results in the appendix (Table 5). This is good practice, but the main text results are limited primarily to designability. Including the corrected diversity/novelty metrics in the main text would provide a more complete picture.

### Writing & Clarity
The paper is well-structured and clearly written, despite the mathematically dense sections. The figures and Table 1 effectively illustrate concepts and comparisons. The algorithms are presented clearly. Some readers not deeply familiar with Riemannian geometry may find Sections 3.1–3.2 challenging, but the key results are summarized accessibly, and the appendix provides necessary background.

### Limitations & Broader Impact
Limitations are acknowledged: the assumptions on group action, the handling of degenerate cases via regularization, and the focus on SE(3) symmetry with a note on future extensions. The broader impact statement is appropriate, noting no ethical concerns beyond standard research practices. A more explicit discussion of potential negative societal impacts (e.g., misuse for generating harmful biomolecules) could be considered, though this is common for methodological papers.

### Overall Assessment
This paper presents a principled and theoretically sound framework for diffusion models on quotient spaces, addressing significant limitations in existing symmetry-handling approaches. The theoretical analysis is rigorous, the practical algorithm is clearly derived, and the empirical results demonstrate consistent improvements on challenging real-world tasks. While some experimental details could be strengthened (error bars, ablations), the core contributions—a novel framework, insightful analysis of prior methods, and strong empirical performance—are substantial and meet the high standards of ICLR. The work provides a solid foundation for future research on symmetry-aware generative models.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a principled framework for building diffusion models on quotient spaces to handle inherent symmetries in generative tasks. The authors derive the diffusion process on a general quotient manifold, project it back to the original space via horizontal lift, and specialize to the SE(3) symmetry for molecular structure generation. The approach reduces learning difficulty by removing unnecessary predictions along symmetric directions and guarantees sampling correctness. Empirical results on small molecule (GEOM) and protein backbone generation demonstrate consistent improvements over equivariant and alignment-based baselines.

### Strengths
1. **Theoretical foundation**: The paper provides rigorous derivations of the diffusion process on quotient spaces (Theorem 1) and its horizontal lift (Theorem 2), culminating in explicit expressions for the SE(3) case (Theorem 4). This offers a solid mathematical framework for symmetry-aware generative modeling.
2. **Clear analysis of existing methods**: Table 1 effectively compares conventional equivariant diffusion, GeoDiff alignment, AF3 alignment, and the proposed quotient-space diffusion, highlighting the advantages in learning difficulty reduction and sampling compatibility.
3. **Practical algorithms and implementation**: The paper translates theory into concrete training and sampling algorithms (Section 3.3, Appendix E) with efficient O(N) projections for the SE(3) case, making the method readily applicable.
4. **Strong empirical validation**: Experiments on GEOM-QM9 and GEOM-DRUGS show consistent gains over ET-Flow and alignment baselines. In protein backbone generation, the quotient-space diffusion with a 60M parameter model outperforms a 200M parameter Proteína model on key distributional metrics, demonstrating practical impact.
5. **Well-structured presentation**: The paper is logically organized, with clear motivations, background, theoretical development, and experiments. Figures 1 and 3 help illustrate the core ideas.

### Weaknesses
1. **Limited empirical demonstration of sampler correctness**: While the paper theoretically proves sampling compatibility, it does not empirically validate that alignment-based methods fail to recover the target distribution in a controlled setting (e.g., a simple synthetic invariant distribution). Such an experiment would strengthen the criticism of heuristic alignments.
2. **Narrow scope of empirical evaluation**: All experiments focus on SE(3) symmetry for molecular generation. The generality of the framework is not tested on other symmetry groups (e.g., U(1), SU(2), or periodic translations), which would better showcase its broad applicability.
3. **Assumptions and practical limitations**: The theory requires free, proper, isometric group actions. For SE(3), degenerate configurations (collinear points) are excluded as a measure-zero set, but this is not explicitly handled in implementation. Additionally, the method relies on centering (CoM-free subspace) for translation invariance, which, while common, is a special case not fully derived from the general quotient construction.
4. **Insufficient comparison to Riemannian diffusion literature**: While Appendix A mentions related work, the discussion is brief. A more thorough comparison with prior Riemannian diffusion models (e.g., De Bortoli et al., 2022; Huang et al., 2022) would better position the novelty of the quotient-space approach.
5. **Computational overhead not quantified**: Although the authors claim O(N) overhead is negligible, no runtime or memory comparisons with baselines are provided to substantiate this claim in practice.

### Novelty & Significance
**Novelty**: The paper introduces a novel framework for diffusion models on quotient spaces, formalizing the reduction of learning difficulty and guaranteeing correct sampling. The theoretical analysis of diffusion projection and horizontal lift, along with the explicit SE(3) instantiation, appears original. The comparative analysis of symmetry-handling strategies (Table 1) is also a novel contribution.

**Significance**: The work provides a principled alternative to heuristic alignment methods, addressing a critical issue in symmetry-aware generative modeling. The empirical improvements on molecular and protein generation tasks suggest tangible benefits for scientific applications. The theoretical grounding could influence future developments in geometric deep learning and diffusion models, promoting more rigorous handling of symmetries.

### Suggestions for Improvement
1. **Add a controlled synthetic experiment** to empirically demonstrate that alignment-based samplers fail to recover the target distribution, while the quotient-space method succeeds (e.g., on a simple rotation-invariant 2D distribution).
2. **Expand experiments to other symmetries** beyond SE(3), even if only on toy datasets, to showcase the generality of the framework. This could include discrete symmetries or other Lie groups.
3. **Discuss limitations more thoroughly**: Address how degenerate cases (e.g., collinear points) are handled in practice, the implications of non-free actions, and scalability to very large systems (e.g., multi-chain proteins).
4. **Provide runtime and memory comparisons** with baselines to quantify the overhead of horizontal projection and mean curvature terms, ensuring practicality for large-scale problems.
5. **Enhance the related work section** (Appendix A) to more comprehensively relate the work to Riemannian diffusion models and other symmetry-aware generative approaches, clarifying the distinct contributions.
6. **Include a simple illustrative example** (e.g., 2D rotation symmetry) in the main text to build intuition for readers less familiar with differential geometry.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validate sampling correctness on a simple synthetic distribution.** The paper claims alignment methods (GeoDiff, AF3) have incompatible samplers. This must be tested empirically: define a known symmetric distribution (e.g., on ℝ² with SO(2) symmetry), train models with these methods, and measure distributional distance (e.g., MMD) to the true target. Without this, the core claim that quotient-space diffusion guarantees correct sampling while others fail is not empirically substantiated.
2. **Ablate the mean curvature term.** The derived SDE includes a curvature correction term. An ablation study must show whether this term is necessary for correct sampling or if it can be omitted without performance loss. Without this, it is unclear if the full theoretical complexity is required in practice.
3. **Demonstrate training efficiency gains.** The paper claims reduced learning difficulty, but only final generation metrics are shown. Plot training loss vs. epochs for conventional, alignment, and quotient-space methods on the same architecture to show faster convergence or lower loss. This is direct evidence for the claimed learning advantage.
4. **Compare to a broader set of baselines for protein generation.** The paper only compares to Prote´ına. To establish state-of-the-art, include comparisons to other strong protein diffusion models (e.g., FrameDiff, RFDiffusion) using the same evaluation protocol.

### Deeper Analysis Needed (top 3-5 only)
1. **Empirically analyze the sampling incompatibility of alignment methods.** The theoretical argument that GeoDiff/AF3 alignment leads to incorrect samplers should be backed with empirical analysis. Measure the distribution of samples generated by these methods on a controlled setup and show systematic bias (e.g., orientation preference). This is critical to justify the need for the proposed method.
2. **Analyze the learned model's output components.** To verify that the model does not learn a specific vertical mapping, compute the variance of the model's output in the vertical vs. horizontal directions during training. High vertical variance would support the claim of reduced learning difficulty.
3. **Study the effect of the projection on optimization.** The horizontal projection changes the gradient flow. Analyze gradient norms, conditioning, or loss landscape sharpness to show that the projection simplifies optimization, as claimed.
4. **Assess sensitivity to hyperparameters (e.g., γ in SDE).** The method introduces a hyperparameter for stochasticity in protein sampling. A sensitivity analysis is needed to show how performance varies with γ and to justify the chosen values.

### Visualizations & Case Studies
1. **Visualize sampling trajectories in the total space.** For a concrete example (e.g., a small molecule), plot the diffusion sampling paths of quotient-space vs. conventional diffusion. Show that quotient-space trajectories exhibit no net rotation/translation (only horizontal movement), while conventional trajectories wander in the vertical directions. This would visually confirm the theoretical behavior.
2. **Case study on a specific molecule or protein.** Show side-by-side generated structures from all methods for a few representative examples, highlighting where quotient-space diffusion produces more accurate or diverse conformations. This makes the performance gains tangible.
3. **Visualize failure modes of alignment methods.** Generate samples using GeoDiff/AF3 alignment on a toy 2D distribution and plot them to show incorrect concentration or bias (e.g., all samples aligned to a reference orientation). This would clearly illustrate the sampler incompatibility discussed theoretically.

### Obvious Next Steps
1. **Validate on a simple synthetic dataset with known symmetry.** This is a critical sanity check that should have been included to confirm the theoretical framework works as intended before applying it to complex real-world data.
2. **Conduct an ablation study on the curvature term.** The paper presents the full theory but does not show whether the curvature term is necessary in practice. This ablation is essential to understand the practical contribution of each component.
3. **Test the framework on other symmetry groups (e.g., periodic translations).** While SE(3) is the primary application, a demonstration on another group (even on a toy problem) would strengthen the claim of generality for the quotient-space framework.
4. **Provide runtime and memory overhead analysis.** The projection and curvature computations add overhead. Quantify the additional time per training/ sampling step and memory usage, especially for large proteins, to assess practicality.

# Final Consolidated Review
## Summary
This paper introduces a principled framework for building diffusion models on quotient spaces to handle inherent symmetries in generative tasks. The authors derive the diffusion process on a general quotient manifold, project it back to the original space via horizontal lift, and specialize to SE(3) symmetry for molecular structure generation. The approach reduces learning difficulty by removing unnecessary predictions along symmetric directions and guarantees sampling correctness. Empirical results on small molecule and protein backbone generation demonstrate consistent improvements over equivariant and alignment-based baselines.

## Strengths
- **Rigorous theoretical foundation:** The paper provides a formal derivation of the diffusion process on a general quotient space (Theorem 1) and its horizontal lift (Theorem 2), culminating in explicit, implementable expressions for the SE(3) case (Theorem 4). This offers a solid mathematical framework for symmetry-aware generative modeling.
- **Clear analysis of existing methods:** Table 1 effectively compares conventional equivariant diffusion, GeoDiff alignment, AF3 alignment, and the proposed quotient-space diffusion, highlighting the advantages in learning difficulty reduction and sampling compatibility. This analysis correctly identifies the limitations of heuristic alignments.
- **Practical algorithms and efficient implementation:** The framework translates theory into concrete training and sampling algorithms (Section 3.3, Appendix E) with an efficient O(N) projection for the SE(3) case, making the method readily applicable. The computational overhead is shown to be negligible in practice (Appendix F.1).
- **Strong empirical validation:** Experiments on GEOM-QM9 and GEOM-DRUGS show consistent gains over ET-Flow and alignment baselines. In protein backbone generation, the quotient-space diffusion with a 60M parameter model outperforms a 200M parameter Prote´ına model on key distributional metrics, demonstrating practical impact and efficiency.

## Weaknesses
- **Lack of empirical validation of sampler correctness for alignment methods:** The paper theoretically argues that alignment methods (GeoDiff, AF3) do not guarantee correct sampling, but does not provide a controlled experiment (e.g., on a simple synthetic symmetric distribution) to empirically demonstrate this failure. This is a missed opportunity to strengthen the critique and solidify the motivation for the proposed method.
- **Limited ablation studies:** The paper does not ablate the components of their method, such as the effect of the mean curvature term in the SDE or the impact of using only the horizontal projection without the curvature correction. This leaves uncertainty about which aspects are necessary for the observed performance gains.
- **Narrow empirical evaluation:** All experiments focus on SE(3) symmetry for molecular generation. While this is an important application, the framework is general, and a demonstration on at least one other symmetry group (even a toy example) would better support the claim of generality and showcase broader applicability.
- **Insufficient analysis of training efficiency:** The paper claims reduced learning difficulty, but does not provide training curves or metrics to show faster convergence or lower loss compared to baselines. Direct evidence of the learning advantage would strengthen the core contribution.

## Nice-to-Haves
- Reporting confidence intervals or standard deviations across multiple runs for the evaluation metrics would increase robustness, though single-run evaluation is common in the field.
- A more thorough discussion of handling near-degenerate configurations (e.g., nearly collinear points) in practice, beyond the mention of ε-regularization, would be helpful for robustness.
- A brief runtime and memory comparison with baselines could further quantify the overhead of the projection and curvature terms, especially for very large systems.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength:** "The paper is well-structured and clearly written." (Generic strength)
- **Weakness:** "The CoM-free subspace for translation invariance is a special case not fully derived from the general quotient construction." (The paper explicitly derives this as an instantiation of the general framework for R^{3N}/SE(3).)
- **Weakness:** "Insufficient comparison to Riemannian diffusion literature." (The related work section in Appendix A covers key works; this is a minor stylistic issue.)
- **Weakness:** "Computational overhead not quantified." (Addressed in Appendix F.1 with training speed comparisons.)

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct a controlled synthetic experiment to empirically demonstrate that alignment-based samplers fail to recover the target distribution (e.g., on a simple 2D rotation-invariant distribution), while the quotient-space method succeeds.
- Perform an ablation study to isolate the contribution of the mean curvature term and the horizontal projection in both training and sampling.
- Include a small experiment on another symmetry group (e.g., a toy problem with a different Lie group) to showcase the generality of the framework.
- Plot training loss versus epochs for the proposed method and key baselines using the same architecture to provide direct evidence of reduced learning difficulty.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 10.0, 6.0]
Average score: 7.5
Binary outcome: Accept
