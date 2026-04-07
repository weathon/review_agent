=== CALIBRATION EXAMPLE 59 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and accurately reflects the core contribution: Multi-Object System Identification from Videos (MOSIV). The abstract succinctly states the problem, limitations of prior work, the proposed solution, and key results. Claims of substantial improvements in grounding accuracy and simulation fidelity are supported by the quantitative results presented later. The commitment to releasing code and data is positive for reproducibility.

### Introduction & Motivation
The introduction effectively motivates the problem, highlighting the gap between single-object methods and the chaotic, contact-rich reality of multi-object scenes. The goal of creating a "digital twin" is well-stated. The contributions (task formalization, new dataset, and a framework combining object-aware Gaussians with differentiable MPM) are clearly listed at the end. A minor weakness is the comparative discussion of CoupNeRF; while it's noted as being designed for a "free-fall regime," the introduction could more sharply articulate why its implicit NeRF representation fundamentally struggles with the temporal consistency required for the *contact-rich* interactions that are central to this work's focus.

### Method / Approach
The three-stage pipeline (geometric reconstruction, continuum lifting, joint fitting) is logically sound.
- **Section 3.1 & 3.2**: The problem statement and overview are clear.
- **Section 3.3 (Gaussian-to-Continuum Lifting)**: The process is described but contains heuristic elements (e.g., "randomly sampling particles within the bounding box," smoothing and thresholding a density field). While likely necessary, the description lacks precise algorithmic details that would aid full reproducibility. The handling of initial interpenetration ("assigning overlapping voxels to the nearest object surface") is a pragmatic solution but its impact on gradient flow during optimization is not analyzed.
- **Section 3.4 (Parameterization & Contact)**: The per-object, continuous parameterization is a core strength. However, the choice of a simple average `g(a,b) = (a+b)/2` for inter-material friction coefficient is a significant simplification. The authors note a fully pairwise parameterization is possible, but the implications of using this simplified model—especially for asymmetric friction scenarios—are not discussed. This is a notable limitation of the current modeling.
- **Section 3.5 (Objectives)**: The geometry-aligned losses (per-object Chamfer and silhouette) are well-justified. The ablation in Section 4.4 powerfully demonstrates their necessity over scene-wise losses. Equation (3) is unfortunately garbled by the parser, but the surrounding text clarifies the components.
- **Section 3.6 (Optimization)**: The use of a horizon curriculum and state re-synchronization is a practical detail for stabilizing training with long rollouts through a differentiable simulator.
**Overall**: The method is innovative and technically sound, combining modern representations (4D Gaussians) with a powerful simulator (MPM). The primary concerns are the reproducibility of some heuristic steps in the lifting process and the simplified friction model.

### Experiments & Results
The experimental design is comprehensive and generally strong.
- **Section 4.1 (Setting)**: The new synthetic dataset (45 scenes, 5 materials, 10 geometries) is a valuable contribution. The adaptation of baselines (OmniPhysGS-RGB, CoupNeRF) for a fair, video-driven comparison is appropriate. The oracle variant of OmniPhysGS is a clever ablation to isolate the effect of discrete model selection.
- **Section 4.2 (Quantitative Results)**: Tables 1 and 2 show MOSIV's clear and substantial superiority over both baselines across all metrics (PSNR, SSIM, CD, EMD) for both observable and future state simulation. The fact that MOSIV even surpasses the **oracle** OmniPhysGS (which has ground-truth material models) is a compelling result, highlighting the advantage of continuous parameter optimization over discrete selection even when the category is known. This is a key finding.
- **Section 4.3 & 4.4 (Qualitative & Ablation)**: The qualitative figures (4, 5, 6) and extensive appendix figures visually support the quantitative claims, showing better geometry preservation and contact handling. The trajectory visualization (Fig. 6) is particularly effective. The ablation study (Table 3) convincingly proves the critical importance of *object-aware* supervision over scene-wise losses.
- **Appendix Results**: The additional tables (5, 7) and sensitivity analysis (8, 9) strengthen the paper. The sensitivity analysis shows graceful degradation with noise, which is reassuring. The runtime/memory comparison (Table 6) shows MOSIV is efficient, though all methods are computationally intensive.

**Critical Questions/Concerns**:
1.  **Statistical Significance**: The results are presented as averages over material pairs. It would be beneficial to report standard deviations or confidence intervals to gauge variance, especially given the relatively small number of scenes per category (e.g., only one E-F interaction?).
2.  **Baseline Implementation Details**: For the reproduced CoupNeRF*, how closely does the implementation match the original? Are all adaptations (e.g., loss functions, training schedule) documented to ensure a perfectly fair comparison? The footnote "* for reproduced implementation" warrants a brief description in the main text or appendix.
3.  **Parameter Accuracy**: While simulation fidelity (CD, EMD) is thoroughly evaluated, there is no direct quantitative analysis of the *identified physical parameters* (Young's modulus, viscosity, etc.) against ground truth. For a "system identification" paper, reporting mean absolute error or correlation for key parameters would significantly strengthen the claim of accurate identification. The current evidence is indirect via simulation outcomes.

### Writing & Clarity
The paper is generally well-written and logically structured. The pipeline diagram (Fig. 2) is helpful. Some sections, particularly the method, are dense and could benefit from more sub-headings or summarized pseudocode to improve readability. The parser-induced errors in equations are distracting but understandable. The core ideas remain discernible.

### Limitations & Broader Impact
Section 5 appropriately lists key limitations: reliance on predefined constitutive models, computational intensity, sensitivity to initial geometry, and the sim-to-real gap. These are honest and relevant. The broader impact section is absent; a brief statement on potential positive (robotics, content creation) and negative (synthesis of misleading physical content) impacts would be expected for ICLR, though its absence is not a major flaw.

## Overall Assessment
This paper presents a strong, well-executed contribution to the challenging problem of multi-object system identification from video. The formalization of the task and the release of a tailored dataset are valuable to the community. The proposed MOSIV framework is novel and effective, convincingly outperforming adapted state-of-the-art baselines by leveraging object-aware dynamic Gaussians, a differentiable MPM simulator, and crucially, geometry-aligned per-object supervision. The most significant concerns are the lack of direct evaluation of the identified physical parameters (beyond simulation outcome) and the simplified symmetric friction model. The methodological heuristics in the lifting process could also be more precisely defined for reproducibility. Despite these issues, the core contribution—demonstrating that continuous parameter optimization with object-level supervision far outperforms discrete material classification in complex multi-object settings—is clear, supported by rigorous experiments, and likely to influence future work. The paper meets the high standards of ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces MOSIV, a framework for multi-object system identification from multi-view videos. The method reconstructs object-aware 4D geometry using dynamic Gaussians, lifts this to a continuum representation, and jointly optimizes continuous, per-object physical parameters (e.g., stiffness, friction) via a differentiable Material Point Method (MPM) simulator guided by geometry-aligned losses. The authors also formalize the task and release a new synthetic benchmark featuring contact-rich, multi-object interactions.

### Strengths
1. **Well-defined and novel problem formulation**: The paper clearly articulates the under-explored challenge of multi-object system identification from video, moving beyond prior single-object or discrete material classification works. This is a meaningful step for applications like robotic manipulation and scene editing.
2. **Comprehensive benchmark and evaluation**: The authors create and release a new synthetic dataset (MOSIV) with diverse objects, materials, and interactions, providing ground-truth physical parameters for rigorous evaluation. This is a valuable contribution to the community.
3. **Effective integration of components**: The pipeline synergistically combines object-aware dynamic Gaussian reconstruction, Gaussian-to-continuum lifting, and differentiable MPM simulation. The ablation study (Table 3) demonstrates that object-aware supervision (vs. scene-wise) is critical for stable optimization in multi-object contact settings.
4. **Strong empirical results**: MOSIV substantially outperforms adapted baselines (OmniPhysGS, CoupNeRF) on both observable and future state simulation across multiple metrics (PSNR, SSIM, Chamfer Distance, EMD) on the new dataset. The qualitative results (Figures 4-6, 9-18) visually support the quantitative gains.

### Weaknesses
1. **Limited real-world validation**: The evaluation is entirely on a controlled synthetic dataset. While the dataset is comprehensive, the paper does not demonstrate the method's performance on real-world videos with challenges like complex lighting, motion blur, or imperfect segmentation masks, leaving the sim-to-real gap unaddressed.
2. **High computational cost and sensitivity**: The method requires multi-view video, object masks, and a computationally intensive optimization process (differentiable MPM rollouts). The sensitivity analysis (Appendix A.8) shows performance degrades with reconstruction noise, and the approach relies on accurate initial geometry, which may be hard to obtain in cluttered real scenes.
3. **Simplified material and interaction model**: The method assumes a fixed set of pre-defined constitutive models (elastic, plastic, fluid, sand) and uses a simple symmetric composition for inter-material friction. This may limit its ability to handle materials with unknown or highly complex rheologies.
4. **Incomplete baseline adaptation details**: The description of how baselines (OmniPhysGS-RGB, CoupNeRF) were adapted for a "fair comparison" is somewhat brief. While efforts were made (e.g., replacing SDS loss with photometric loss), more details on the adaptation process and potential limitations would strengthen the comparative analysis.

### Novelty & Significance
**Novelty**: The work is novel in its problem formulation (multi-object system identification from video) and its technical approach, which integrates object-aware dynamic Gaussians with differentiable physics for continuous parameter learning. The release of a dedicated benchmark also provides a new resource for the field.
**Significance**: The task is highly relevant for robotics, simulation, and scene understanding. The method's ability to recover continuous physical parameters and generalize to novel interactions (Figure 3) is a step towards creating "digital twins" of dynamic scenes. The performance improvements over strong baselines are clear and substantial.
**Clarity**: The paper is generally well-written, with a clear pipeline description (Figure 2) and methodology. Some equations have formatting artifacts from the parser, but the core ideas remain understandable.
**Reproducibility**: The authors promise to release source code and the dataset. The method description, experimental settings, and implementation details (Section 4.1) appear sufficient for reproduction, contingent on the code release.

### Suggestions for Improvement
1. **Conduct experiments on real-world data**: To strengthen the paper's impact, include a validation on a small set of real-world multi-object interaction videos, even if qualitative, to demonstrate feasibility outside simulation. Discuss challenges and potential adaptations needed (e.g., for mask estimation).
2. **Improve efficiency and robustness**: Explore strategies to reduce the computational burden of the optimization, such as more efficient MPM implementations or learning a surrogate model. Also, investigate ways to make the pipeline more robust to imperfect reconstructions or masks (e.g., through robust losses or iterative refinement).
3. **Deepen analysis and ablations**: Provide a more detailed analysis of failure cases or scenarios where parameter identifiability is challenging (e.g., distinguishing stiffness from friction in specific interactions). An ablation on the number of views required or the impact of mask accuracy would be insightful.
4. **Expand discussion of limitations and future work**: The discussion section (Section 5) could be expanded. For instance, discuss the potential to learn neural constitutive laws instead of relying on predefined models, and elaborate on plans to handle a broader range of materials and more complex interactions (e.g., >3 objects).
5. **Clarify baseline adaptations and comparisons**: In the main text or appendix, provide a more detailed account of the adaptations made to OmniPhysGS and CoupNeRF, justifying why these adaptations are appropriate and fair for the new task. This would preempt concerns about comparison validity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validation on real-world videos.** The entire method is validated only on a synthetic dataset. Without experiments on real video sequences (with noise, imperfect masks, complex lighting), the core claim of "system identification from videos" is not substantiated for real-world applicability. This is a critical gap for ICLR.
2. **Comparison to more relevant and recent baselines.** The primary comparison is to an adapted version of OmniPhysGS (a generative model) and CoupNeRF. The field includes other direct video-to-physics inference methods (e.g., GIC, PhysGaussian, Pac-NeRF). The paper does not establish that these adapted baselines are the strongest points of comparison, undermining the claim of state-of-the-art performance.
3. **Ablation on the necessity of accurate 2D instance masks.** The method assumes pre-defined 2D material/instance masks. An experiment corrupting or automatically generating these masks (e.g., with an off-the-shelf segmenter) is needed to assess the method's robustness to a realistic, imperfect input.
4. **Longer-horizon rollout stability analysis.** Predictions are shown for a limited future horizon. Quantitative and qualitative results for much longer rollouts (e.g., 2-3x the observed sequence length) are required to properly evaluate the claim of "accurate prediction of future behaviors" and the simulator's stability.

### Deeper Analysis Needed (top 3-5 only)
1. **Parameter identifiability and ambiguity analysis.** With multiple interacting objects, parameters like stiffness and friction can be coupled. The paper must analyze whether the optimization uniquely recovers ground-truth parameters or finds plausible but incorrect combinations, especially for similar-looking interactions (e.g., sliding vs. deforming).
2. **Breakdown of error by material type and interaction complexity.** Tables 1 & 2 show performance varies across material pairs. A detailed analysis is needed to explain why certain materials (e.g., fluids vs. sand) are harder to identify and how error correlates with contact complexity (e.g., sliding, rolling, plastic deformation).
3. **Sensitivity to initial geometric reconstruction quality.** The Gaussian-to-continuum lifting and parameter optimization depend on the initial 4DGS reconstruction. A systematic study is needed to show how errors in the initial geometry (e.g., from occlusion) propagate to parameter identification and simulation fidelity.

### Visualizations & Case Studies
1. **Visualization of estimated vs. ground-truth physical parameters.** For a set of scenes, plot the recovered continuous parameters (E, ν, µ, etc.) against their true values. This is the most direct way to validate the core "system identification" claim, which is currently only indirectly assessed via geometry metrics.
2. **Qualitative failure cases.** The paper shows successful examples. To build trust, it must visually illustrate where and how the method fails—e.g., under extreme deformation, with nearly identical materials, or when the contact model is insufficient.
3. **Particle trajectory comparisons for *all* objects in a scene.** Figure 6 shows trajectories for one object. Visualizing the simulated paths for all objects in a multi-object scene, superimposed on the ground truth, would clearly demonstrate the accuracy of the identified multi-body dynamics.

### Obvious Next Steps
1. **Real-world video experiments.** This is not a "next step" but a mandatory validation for a paper making claims about learning from video. A small-scale study with real-world multi-object interactions (even if only qualitative) should have been included.
2. **Ablation on the continuum lifting process.** The multi-object lifting heuristic (disjoint supports, voxel assignment) is critical but not ablated. An experiment comparing against simpler strategies (e.g., direct particle sampling from Gaussians) would justify its design.
3. **Analysis of computational bottlenecks.** The method is noted as computationally intensive. A profile identifying which stages (4DGS reconstruction, MPM rollout, optimization) dominate cost would guide future efficiency improvements and is expected for a methods paper.

# Final Consolidated Review
## Summary
This paper formalizes the task of multi-object system identification from multi-view videos and proposes MOSIV, a framework that reconstructs object-aware 4D geometry via dynamic Gaussians, lifts it to a continuum representation, and optimizes continuous per-object physical parameters using a differentiable Material Point Method simulator. The authors also release a synthetic benchmark with diverse materials and contact-rich interactions. Experiments show MOSIV substantially outperforms adapted baselines in both observable and future-state simulation fidelity.

## Strengths
- **Novel problem formulation and benchmark:** The paper clearly defines the under-explored challenge of multi-object system identification from video and provides a valuable synthetic dataset with ground-truth parameters, enabling rigorous evaluation and future research.
- **Effective integration of modern representations with differentiable physics:** The pipeline synergistically combines object-aware dynamic Gaussians, Gaussian-to-continuum lifting, and a differentiable MPM simulator, demonstrating that continuous parameter optimization with per-object supervision is critical for handling complex multi-object contact.
- **Strong empirical validation:** MOSIV achieves significant quantitative improvements over adapted baselines (OmniPhysGS, CoupNeRF) across multiple metrics (PSNR, SSIM, Chamfer Distance, EMD) for both observable and future-state simulation, with compelling qualitative results and an ablation that confirms the necessity of object-aware losses.

## Weaknesses
- **No direct evaluation of identified physical parameters:** The paper assesses system identification only indirectly through simulation fidelity metrics. For a method whose core claim is recovering continuous material parameters (e.g., Young’s modulus, viscosity), reporting direct error between estimated and ground-truth parameters would substantially strengthen the validation.
- **Simplified inter-material friction model:** The friction coefficient between two materials is computed as a simple average of their individual coefficients, which assumes symmetry and may not capture asymmetric friction phenomena. The implications of this modeling choice are not discussed, limiting the model’s generality for real-world interactions.
- **Heuristic steps in the Gaussian-to-continuum lifting process:** The lifting procedure involves heuristic operations (random sampling, density smoothing, thresholding, and ad-hoc interpenetration resolution). While practical, these steps lack a thorough analysis of their impact on gradient flow, optimization stability, and reproducibility.
- **Lack of real-world video validation:** The evaluation is conducted entirely on synthetic data. Demonstrating the method’s performance on even a small set of real-world multi-object videos (with challenges like noise, lighting, and imperfect masks) would better support the claim of learning “from videos” and address the sim-to-real gap.

## Nice-to-Haves
- Reporting variance or confidence intervals for the quantitative results to better convey the consistency of performance across scenes.
- A more detailed ablation of the continuum lifting process to justify its design choices compared to simpler alternatives.
- Further analysis of parameter identifiability under ambiguous interactions (e.g., stiffness vs. friction) and failure cases to better understand the method’s limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Request for comparison to additional baselines (e.g., GIC, PhysGaussian):** The paper adequately justifies its choice of adapted baselines (OmniPhysGS and CoupNeRF) as relevant prior works for the new task; expanding the comparison is not required for the core contribution.
- **Criticism of parser-induced equation formatting:** This is an artifact of the review extraction process, not a flaw in the paper.
- **Demand for a broader impact statement:** While useful, its absence does not undermine the technical contribution.
- **Request for longer-horizon rollout analysis beyond the provided future-state simulation:** The paper already evaluates prediction over a substantial horizon; longer rollouts are an extension rather than a requirement.

## Novel Insights
The paper’s key insight is that in multi-object contact-rich settings, continuous per-object parameter optimization coupled with object-aware geometry-aligned supervision is far more effective than discrete material classification—even when the discrete category is known (as shown by MOSIV outperforming the oracle OmniPhysGS). This highlights the limitations of fixed material libraries and demonstrates that fine-grained, identity-preserving losses are essential for stable gradient-based identification in complex multi-body dynamics.

## Suggestions
- Include a quantitative evaluation of the recovered physical parameters (e.g., mean absolute error or correlation with ground truth) to directly validate the system identification claim.
- Extend the friction model to allow asymmetric coefficients or analyze the impact of the symmetric average assumption on simulation accuracy.
- Provide a more detailed description of the baseline adaptations (especially for CoupNeRF*) in the appendix to ensure full reproducibility and fairness of comparisons.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
