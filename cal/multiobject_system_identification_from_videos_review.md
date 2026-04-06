=== CALIBRATION EXAMPLE 53 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the core contribution: multi-object system identification from videos (MOSIV). The abstract succinctly presents the problem, the proposed framework's key components (continuous parameter optimization via differentiable simulation guided by geometry), the creation of a new benchmark, and the main claim of superior performance. All claims are supported in the body.

**Introduction & Motivation:** The motivation is clear and well-argued. The limitations of prior single-object or discrete material classification methods for chaotic, contact-rich multi-object scenes are convincingly laid out. The goal of creating a "digital twin" is well-stated. The contributions are explicitly listed, aligning with the abstract. The choice of baselines (OMNIPHYSGS, COUPNERF) is justified based on their relevance and shortcomings for the proposed task.

**Method / Approach:**
*   **Clarity & Reproducibility (Sections 3.1-3.6):** The overall pipeline is clearly illustrated in Figure 2. The problem statement (3.1) and overview (3.2) are good. However, several critical details are under-specified or rely on citations without sufficient explanation for a novel combination.
    1.  **Gaussian-to-Continuum Lifting (3.3):** The description is high-level and procedural ("We generate a rough internal shape...", "We construct a density field..."). The specific algorithm (e.g., number of sampling iterations, threshold values, exact method for resolving interpenetrations) is not provided, making exact reproduction difficult. While citing GIC (Cai et al.) for the idea, the multi-object extensions are novel and need more precise formulation.
    2.  **Multi-Material Parameterization (3.4):** The per-object parameterization is clearly stated. However, the composition function for friction `g(a,b) = 1/2(a+b)` is presented without justification. Is an arithmetic mean physically justified for all material pairs? A brief discussion or citation would strengthen this choice.
    3.  **Optimization Details (3.6):** The description of the "horizon curriculum" and "alternating update strategy" is too vague. How is the rollout length increased? What triggers re-synchronization? These are likely important for stability but are not specified.
    4.  **Equation Presentation:** Several equations (e.g., Eq. 1, 2, parts of 3) suffer from significant LaTeX parsing artifacts (`\`, `{`, misplaced symbols), which impede understanding. While this is noted as a parser issue, it critically damages the readability of the core methodological section. The reviewer must infer the intended equations, which is unacceptable for a final submission.
*   **Logical Soundness:** The core idea—using differentiable MPM to optimize continuous parameters via alignment between simulated and reconstructed (Gaussian-based) geometry—is sound and builds appropriately on related work. The key insight of using *object-aware* losses (Chamfer and silhouette computed per-object) to avoid association ambiguity at contact (explained in Sec. 4.4) is a significant and well-motivated contribution to the multi-object setting.

**Experiments & Results:**
*   **Dataset (4.1):** The new synthetic benchmark is a valuable contribution. The description of object geometries, materials, and scene generation is clear. Providing parameter ranges in Appendix A.3 is good. A minor note: the text says "10 unique geometries" and lists 10, but the table caption in A.3 says "(10 Geometry Shapes)" while listing only 4 material parameter sets. This is slightly confusing but not a major issue.
*   **Baselines (4.1):** The adaptation of OMNIPHYSGS-RGB and the creation of an "oracle" variant are reasonable and well-explained, ensuring a focused comparison on the parameter identification aspect. The inclusion of CoupNeRF* (reproduced) is appropriate.
*   **Quantitative Results (4.2, Tables 1, 2, 5):** The results are comprehensive, showing clear and substantial improvements across all metrics (PSNR, SSIM, CD, EMD) for both observable and future simulation. The breakdown by material-pair type is insightful. The consistent outperformance of MOSIV even over the "oracle" baseline (which knows the true material model) strongly supports the claim that continuous parameter identification is superior to discrete model selection for this task.
*   **Ablation Study (4.4, Table 3):** This is a **crucial and excellent** ablation. It cleanly demonstrates the necessity of both geometric terms (CD and α) and, more importantly, the *object-wise* granularity of supervision. The results and explanation of "association ambiguity" directly validate a core design decision and significantly strengthen the paper.
*   **Qualitative Results (4.3, Figs. 4-6, 9-18):** The figures effectively illustrate MOSIV's advantages in preserving object identity, contact boundaries, and material-specific behavior (e.g., sand vs. plasticine) over baselines, which show blur, leakage, and unrealistic dynamics. The trajectory visualization (Fig. 6) is a good addition.
*   **Limitations of Evaluation:**
    1.  **Synthetic-Only:** The entire evaluation is on a synthetic dataset. While this is necessary for ground-truth parameter evaluation, it is a significant limitation. The paper would be substantially stronger with at least a qualitative demonstration on a few real-world video clips to show sim-to-real potential, even without parameter ground truth. The discussion section mentions this challenge but does not attempt to address it.
    2.  **Parameter Error:** Surprisingly, the paper does **not** report direct error metrics for the identified physical parameters (e.g., Young's modulus `E`, friction `µ`). The evaluation is entirely indirect via simulation fidelity (CD, PSNR, etc.). For a "system identification" paper, reporting the Mean Absolute Percentage Error (MAPE) or similar for the recovered parameters `Θ` against ground truth (available in the synthetic benchmark) is a critical missing analysis. This makes it hard to assess if the correct parameters were found or if different parameters led to similarly good visual fits.

**Writing & Clarity (excluding parser artifacts):** The paper is generally well-written, with a logical flow. However, the severe equation formatting issues in the Method section (Sec. 3) are a major barrier to understanding. The appendix is thorough, adding necessary implementation and model details.

**Limitations & Broader Impact:** The discussion section (5) appropriately acknowledges key limitations: reliance on predefined constitutive models, computational intensity, sensitivity to initial geometry, and the sim-to-real gap. Broader impact is not discussed, which is acceptable for this technical work.

### Overall Assessment

This paper presents a timely and well-executed study on the important problem of multi-object system identification from video. The core contributions are substantial: a clear task formulation, a novel framework combining dynamic Gaussians with differentiable MPM and object-aware losses, and a valuable new synthetic benchmark. The ablation study on supervision granularity is particularly compelling. The empirical results convincingly demonstrate state-of-the-art performance on the proposed task against thoughtfully adapted baselines.

The most significant weaknesses are: (1) the **lack of direct parameter error evaluation**, which is essential for a system ID paper, and (2) the **exclusive reliance on synthetic data** for validation. Additionally, the **severe formatting artifacts in the methodology section** must be fixed. Addressing the first point with a table of parameter errors and expanding the discussion on the second point are necessary for acceptance at ICLR. Despite these issues, the paper's novel insights, strong experimental design, and demonstrated performance make it a meaningful contribution that stands above the adapted baselines.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces the task of multi-object system identification from videos, where the goal is to reconstruct 4D geometry and identify continuous physical parameters (e.g., stiffness, friction) for each interacting object. The authors propose MOSIV, a framework that integrates object-aware 4D Gaussian reconstruction with a differentiable Material Point Method (MPM) simulator and geometry-aligned optimization. A new synthetic benchmark of contact-rich, multi-object interactions is presented, and experiments show that MOSIV outperforms adapted baselines in parameter identification and simulation fidelity.

### Strengths
1. **Well-defined Problem and Benchmark**: The paper clearly formalizes the challenging and under-explored problem of multi-object system identification. The release of a new synthetic dataset (generated with the Genesis engine) with ground-truth parameters is a significant contribution that will facilitate future research.
2. **Comprehensive Evaluation**: The experimental evaluation is thorough, including comparisons against two strong adapted baselines (OmniPhysGS and CoupNeRF), extensive quantitative results (PSNR, SSIM, Chamfer Distance, EMD), and compelling qualitative visualizations. The ablation study on object-aware vs. scene-wise supervision provides strong evidence for a key design choice.
3. **Strong Technical Innovation**: The integration of object-aware dynamic Gaussians, differentiable MPM simulation, and a per-object, geometry-aligned loss is a novel and technically sound approach for learning continuous physical parameters. The demonstration of "novel interactions" via parameter swapping shows the framework's ability to generalize.

### Weaknesses
1. **Limited Real-World Validation**: The evaluation is conducted entirely on a controlled synthetic dataset. While this is appropriate for establishing a baseline, the paper does not demonstrate performance on real-world videos, leaving the sim-to-real transfer and robustness to real-world noise/occlusions as open questions. The limitations section acknowledges this, but empirical validation is missing.
2. **High Computational Cost and Assumptions**: The method relies on multi-view video, pre-computed object instance masks, and a computationally intensive optimization pipeline (differentiable MPM rollouts). This limits its practical applicability in many real-world scenarios (e.g., monocular video, unknown segmentation). The computational overhead, while compared favorably to baselines in the appendix, remains substantial.
3. **Incomplete Baseline Comparison**: The main comparison for CoupNeRF is against the authors' reproduced/adapted version (CoupNeRF*). A direct comparison with the original CoupNeRF method on this new task and dataset is not provided, making it difficult to fully assess the improvement over the state-of-the-art in a like-for-like manner.

### Novelty & Significance
**Novelty**: The paper makes several novel contributions: (1) the formalization of the multi-object system identification from video task, (2) the MOSIV framework that jointly performs object-aware reconstruction and continuous parameter identification via differentiable simulation, and (3) the release of a corresponding synthetic benchmark. The shift from discrete material classification (baselines) to continuous per-object parameter optimization is a key advance.
**Significance**: The work is significant for the fields of vision, graphics, and robotics. Faithful physical understanding of multi-object scenes is a core challenge for applications like robotic manipulation, AR/VR, and content creation. The paper provides a strong foundation and a valuable benchmark for future research.

### Suggestions for Improvement
1. **Include Real-World Experiments**: To strengthen the paper's impact, the authors should include a small-scale evaluation on real-world multi-view video sequences (even if captured in a lab setting). This would demonstrate the method's practical viability and help characterize the sim-to-real gap.
2. **Ablate on Mask Dependency**: The method assumes the availability of object masks. An ablation or discussion on the performance degradation with noisy or automatically generated masks (e.g., from an off-the-shelf segmenter) would be valuable to assess robustness and practicality.
3. **Expand Discussion on Limitations and Future Work**: The limitations section could be expanded. For instance, the reliance on pre-defined constitutive models could be discussed in the context of recent "neural physics" approaches. A clearer roadmap for improving computational efficiency would also be helpful.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct evaluation of recovered physical parameters.** The paper only reports rendered geometry and image metrics. To substantiate the core claim of "system identification," it must include a table comparing estimated continuous parameters (Young's modulus, viscosity, yield stress) against ground truth values. Without this, it's unclear if the method actually identifies the correct physics or just matches appearance.
2. **Real-world video experiments.** All validation is on a clean synthetic dataset. To claim the method works on "real-world scenes," at least one real video experiment (even with approximate ground truth) is needed to demonstrate robustness to imperfect masks, lighting, and reconstruction noise.
3. **Comparison to a true multi-object system identification baseline.** The chosen baseline, OmniPhysGS, is adapted from a generative task and uses discrete material selection. A more direct comparison should be made with methods like GIC (Cai et al., 2024) or PhysGaussian, extended to multi-object settings, to isolate the contribution of the proposed object-aware framework.
4. **Ablation on the number of objects and occlusion severity.** The paper tests only 2-3 objects. To claim generalization to "cluttered spaces," experiments with 4+ objects and varying occlusion levels are necessary to show the method's scalability and robustness to missing observations.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of parameter identifiability and coupling.** The paper states that ambiguities (e.g., stiffness vs. friction) are resolved through motion, but no analysis is provided. A study should show how individual parameter errors correlate and whether the optimization reliably converges to the true values, or if there are degenerate solutions that still match the visuals.
2. **Quantitative breakdown of failure cases.** The qualitative results show successes, but a quantitative analysis of when and why the method fails (e.g., specific material pairs, high deformation, fast motion) is missing. This is critical for understanding the method's limitations.
3. **Sensitivity to initial geometry and mask quality.** The method depends on object masks and Gaussian reconstruction. An ablation systematically degrading mask accuracy and reconstruction noise (beyond the simple noise addition in the appendix) is needed to show robustness, as real-world inputs are imperfect.

### Visualizations & Case Studies
1. **Visualization of per-object parameter optimization trajectories.** Plotting how each object's estimated parameters evolve during training would reveal whether they converge stably and uniquely, or if they oscillate or trade off with other objects' parameters.
2. **Side-by-side videos of simulated vs. real trajectories for novel interactions.** The paper claims generalization to novel material assignments. Providing video comparisons (as supplementary) for several such swaps would convincingly show the physical accuracy beyond a single example.
3. **Visualization of contact forces and friction effects.** Since contact modeling is key, visualizing inferred contact pressures or friction forces during interactions would help validate whether the physics is correctly captured, rather than just geometric alignment.

### Obvious Next Steps
1. **Incorporate uncertainty estimates for the recovered parameters.** Given the ill-posed nature of inverse physics, reporting confidence intervals or posterior distributions for the estimated parameters would strengthen the claims of identifiability and inform downstream use (e.g., in robotics).
2. **Test on a broader set of material models, including learned neural constitutives.** The method relies on pre-defined constitutive models. As a next step, integrating a neural constitutive model (as mentioned in the discussion) would demonstrate extensibility to unknown materials, a key direction for real-world application.
3. **Evaluate on dynamic scenes with topological changes (e.g., breaking, splitting).** The current experiments involve objects that remain coherent. Testing on scenes where objects fracture or merge (e.g., granular materials spreading) would push the method's limits and better assess its generality.

# Final Consolidated Review
## Summary
This paper introduces the task of multi-object system identification from videos, aiming to reconstruct 4D geometry and recover continuous physical parameters (e.g., stiffness, friction) for each object in contact-rich scenes. It proposes MOSIV, a framework that integrates object-aware dynamic Gaussian reconstruction with a differentiable Material Point Method simulator and geometry-aligned optimization, and releases a new synthetic benchmark for evaluation.

## Strengths
- **Task formalization and benchmark:** The paper clearly defines a novel and challenging problem, and contributes a well-crafted synthetic dataset with ground-truth parameters, which will enable meaningful comparisons in future research.
- **Effective framework and strong empirical results:** The integration of object-aware dynamic Gaussians, differentiable physics simulation, and per-object supervision yields substantial improvements over adapted baselines across all metrics, demonstrating the approach's superiority for this task.
- **Insightful ablation study:** The ablation cleanly validates a core design choice, showing that object-level geometric and silhouette losses are essential to avoid cross-object association errors during contact, which is critical for stable optimization and accurate parameter identification.

## Weaknesses
- **Missing direct parameter evaluation:** As a system identification paper, it does not report error metrics for the recovered physical parameters (e.g., Young's modulus, viscosity) against ground truth, relying instead on indirect simulation fidelity. This omission makes it impossible to assess whether the correct physics is identified or if degenerate solutions achieve visual fit.
- **Exclusively synthetic validation:** All experiments are conducted on a controlled synthetic dataset. While this allows for precise evaluation, the absence of any demonstration on real-world videos leaves the method's robustness to noise, occlusions, and sim-to-real transfer unproven, limiting the claim of applicability to "real-world scenes."
- **Under-specified methodological details:** Key components such as the horizon curriculum for rollout length, the alternating update strategy, and the precise algorithm for multi-object Gaussian-to-continuum lifting are described only at a high level, hindering reproducibility and independent verification.

## Nice-to-Haves
- Analysis of parameter identifiability and coupling under different interaction regimes.
- Investigation into sensitivity to imperfect object masks or monocular video inputs.
- Extension to neural constitutive models to handle materials outside the predefined set.

## Novel Insights
The paper demonstrates that in multi-object system identification, supervision must be applied at the object level rather than scene-wide to prevent optimization from exploiting cross-object associations during contact, which leads to physically inaccurate parameter estimates. This insight is crucial for stable training and faithful physics recovery in complex, interactive settings.

## Suggestions
- Add a table reporting direct error metrics (e.g., MAPE) for the estimated physical parameters against ground truth in the synthetic benchmark.
- Include a qualitative case study on a real-world multi-view sequence to illustrate practical challenges and the method's behavior under realistic conditions.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
