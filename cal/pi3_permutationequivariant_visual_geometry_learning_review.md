=== CALIBRATION EXAMPLE 78 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title "π³: Permutation-Equivariant Visual Geometry Learning" clearly reflects the core contribution. The abstract succinctly states the problem (reliance on a fixed reference view), the solution (a fully permutation-equivariant architecture), and claims state-of-the-art performance across multiple tasks. The claims are specific and appear supported by the results presented later.

### Introduction & Motivation
The introduction effectively motivates the problem by identifying a clear limitation in prior work: the sensitivity to and dependence on an arbitrarily chosen reference view, inherited from classical SfM/MVS. This is framed as an "unnecessary inductive bias." The contributions are stated clearly and map well to the paper's content. The positioning against recent feed-forward models (DUSt3R, VGGT) is appropriate.

### Method / Approach
The method is described with mathematical formalism. The permutation-equivariance property is well-defined in Equations (1)-(3). The architectural choice to omit order-dependent components (positional embeddings, special reference tokens) logically follows from this goal.

**Key Concerns:**
1.  **Pose Scale Ambiguity and Supervision:** A significant conceptual issue arises in Section 3.3. The model predicts camera poses up to a similarity transformation. To supervise them, relative poses are computed (Eq. 7), and their translation magnitudes are scaled using the optimal scale factor \(s^*\) derived from the *point cloud alignment* (Eq. 4, 10). This creates a coupling where the pose translation supervision is not direct but borrows scale from a separate geometry alignment process. The theoretical justification for this is underdeveloped. Could errors in point cloud alignment (e.g., due to depth noise or occlusion) propagate incorrectly to pose translation scaling? A more rigorous discussion or an ablation on alternative scaling strategies is needed.
2.  **Training Stability and Initialization:** Appendix A.4 reveals a critical practical detail: training the permutation-equivariant model from scratch is unstable ("cold start problem") and requires an auxiliary "global proxy" task (predicting a reference-based pointmap) for stabilization. The final model uses VGGT weights for initialization. This dependency on pre-trained weights from a reference-based model is a major limitation that is buried in the appendix and not sufficiently acknowledged in the main text or limitations. It raises questions about the fundamental learnability of the proposed objective and whether the gains are partly attributable to the inherited representations.
3.  **Claim of Low-Dimensional Manifold:** The discussion around Figure 4 and Appendix A.3 states the model captures a "low-dimensional manifold" of camera trajectories. While the eigenvalue analysis shows concentrated variance, this is presented as an observation/outcome rather than an explicit design constraint or loss. The claim feels somewhat post-hoc and is not central to the method's mechanics.

**Reproducibility:** The architecture (Appendix A.1) and training details (Appendix A.2, including loss weights, schedules, dataset list) are described thoroughly, enhancing reproducibility. The use of a published ROE solver and standard losses is good.

### Experiments & Results
The experimental evaluation is extensive and covers multiple tasks (pose, depth, point cloud) across many standard datasets. The comparisons against relevant SOTA methods (VGGT, CUT3R, FLARE, MoGe) are fair.

**Strengths:**
- The robustness analysis (Sec. 4.4, Table 6) is a direct and compelling validation of the core permutation-equivariance claim, showing near-zero standard deviation across input permutations.
- The comprehensive benchmark results (Tables 1-5) generally support the claim of achieving SOTA or highly competitive performance.
- The ablation study (Table 7) cleanly shows the incremental benefit of the proposed components.

**Weaknesses & Questions:**
1.  **Initialization Confound:** As noted, the superior performance may be influenced by initialization with a strong pre-trained model (VGGT). While Table 8 attempts a "from-scratch" comparison, it introduces a different architecture (with a global proxy). A more convincing ablation would be to fine-tune VGGT itself in a reference-free manner (i.e., remove its reference token and apply the π³ losses) starting from its released checkpoint. This would better isolate the gain from the *architectural/objective change* versus the gain from *additional training*.
2.  **Depth of Analysis on Pose Scaling:** The experiment does not analyze the sensitivity of pose accuracy to the quality of the scale factor \(s^*\). An analysis of translation error with vs. without the borrowed scale, or under simulated noisy depth conditions, would strengthen the methodological justification.
3.  **Computational Efficiency Claim:** Table 4 reports high FPS. It should be clarified if this is measured with the same input resolution and sequence length as baselines (e.g., VGGT at 43.2 FPS). The parameter count comparison is useful.

### Writing & Clarity
The paper is generally well-written and logically structured. The figures effectively illustrate the concept and results. Some sections in the appendix (e.g., A.3 on pose distribution) are somewhat verbose and less incisive. The critical issue of training instability and initialization is not foregrounded sufficiently; it should be discussed in the main limitations section.

### Limitations & Broader Impact
The listed limitations in Appendix A.8 (transparent objects, lack of fine detail, grid artifacts) are appropriate but technical. The significant limitation regarding training stability and initialization dependency is **omitted**. A broader impact statement is absent, which is common for this type of work but could briefly note potential positive applications in robotics and AR.

## Overall Assessment
This paper presents a novel and well-motivated idea: eliminating the reference-view bias in multi-view reconstruction via a permutation-equivariant architecture. The proposed π³ model demonstrates impressive empirical performance, achieving SOTA or competitive results across a wide range of benchmarks, and its robustness to input ordering is convincingly validated. However, the contribution is tempered by two substantial concerns: (1) the theoretically and practically important detail that pose translation supervision is scaled via geometry alignment lacks a thorough justification and analysis, and (2) the model's training appears reliant on initialization from a pre-trained reference-based model (VGGT), which is a significant caveat that is underemphasized. Addressing these issues, particularly by providing a more robust ablation on the source of improvements and a deeper discussion of the scaling mechanism, is essential for the paper to meet ICLR's high standards for technical rigor and clarity of contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces π³, a feed-forward neural network for visual geometry reconstruction that eliminates the reliance on a fixed reference view. By employing a fully permutation-equivariant architecture, the model predicts affine-invariant camera poses and scale-invariant local point maps per view, making it inherently robust to input ordering. The method achieves state-of-the-art or competitive performance across multiple benchmarks for camera pose estimation, monocular/video depth estimation, and dense point map reconstruction.

### Strengths
1. **Clear Identification of a Key Limitation**: The paper convincingly argues that the common practice of anchoring reconstruction to a reference view introduces a detrimental inductive bias, leading to instability. This is supported by empirical evidence (Figure 2, Table 4.4) showing significant performance variance in prior methods when the reference changes.
2. **Strong Empirical Performance**: The method demonstrates comprehensive SOTA or competitive results across a wide array of tasks (camera pose, depth, point cloud reconstruction) and datasets (Sintel, KITTI, ETH3D, etc.), as detailed in Tables 1-5. The performance gains are often substantial (e.g., video depth Abs Rel on Sintel improves from 0.299 to 0.233).
3. **Compelling Robustness Validation**: A key claimed advantage—permutation equivariance—is rigorously validated. Table 6 shows near-zero standard deviation in reconstruction metrics across input permutations, orders of magnitude lower than prior methods, proving exceptional robustness to input order.
4. **Efficiency**: The model is not only accurate but also fast (57.4 FPS on KITTI), outperforming several contemporaries in speed while maintaining a smaller parameter count than some (959M vs. VGGT's 1.26B).

### Weaknesses
1. **Dependence on Pre-trained Initialization**: The final model is initialized from a pre-trained VGGT encoder, and the ablation study (Appendix A.4) reveals that training from scratch with only the core objectives leads to suboptimal convergence, requiring an auxiliary "global proxy" task for stability. This somewhat muddies the attribution of performance gains solely to the proposed architecture.
2. **Limited Discussion on Brobaseline Comparison**: While the paper compares against recent feed-forward learning methods, a comparison with classical SfM/MVS pipelines in terms of robustness or accuracy in challenging conditions (e.g., low texture, wide baselines) is absent. This would help situate the contribution within the broader field.
3. **Acknowledged Limitations**: The paper itself notes limitations in handling transparent objects, capturing fine-grained details compared to diffusion models, and grid artifacts from the upsampling decoder (Appendix A.8). These are important practical constraints.
4. **Novelty Contextualization**: The core idea of permutation equivariance, while novelly and effectively applied to this specific problem, is a well-known concept in machine learning. The paper could better delineate its conceptual novelty relative to other equivariant architectures in vision.

### Novelty & Significance
**Novelty**: The work presents a novel, systematic critique of the reference-view paradigm in feed-forward 3D reconstruction and proposes a concrete, fully permutation-equivariant architecture as a solution. The integration of affine-invariant pose and scale-invariant local geometry prediction within this framework is a distinct contribution.
**Clarity**: The paper is generally well-written, with a clear problem statement, method description, and experimental sections. Some architectural and training details are deferred to the appendix, but the core ideas are accessible.
**Reproducibility**: High. The code is promised to be available, training datasets and hyperparameters are specified, and the experimental protocol is detailed, including metrics and alignment procedures.
**Significance**: The demonstrated improvements in accuracy, speed, and—most notably—robustness are significant for real-world applications where input order is arbitrary or unreliable. The work provides a convincing alternative to the entrenched reference-view approach and is likely to influence future research in multi-view geometry learning.

### Suggestions for Improvement
1. **Deeper Analysis of Training Dynamics**: Provide a more thorough investigation or discussion of the "cold start" problem with relative pose supervision. Analyzing the loss landscape or presenting a stabilized training strategy without the proxy task would strengthen the methodological contribution.
2. **Extended Robustness Analysis**: Include experiments under more extreme conditions, such as with partially occluded views, severe motion blur, or very sparse input sets (e.g., <5 images), to further probe the limits of the robustness claim.
3. **Broader Comparative Baseline**: As mentioned, adding comparisons with robust traditional SfM pipelines (e.g., COLMAP) on specific failure cases of reference-based methods would better contextualize the practical impact of the work.
4. **Address Limitations Concretely**: While limitations are listed, briefly discussing potential research directions to mitigate them (e.g., exploring alternative upsamplers to reduce grid artifacts) would enhance the paper's forward-looking impact.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Systematic test of permutation invariance beyond rotating the first frame.** The paper only reports variance when cycling the first frame, which is a tiny subset of all permutations. To fully validate the core claim of permutation equivariance, you must test on many random permutations of input order and report the variance in outputs (poses, depth) across all permutations. Without this, the claim is not fully substantiated.
2. **Direct comparison to a strengthened baseline that mitigates reference view sensitivity.** The paper argues previous methods fail with poor reference selection. To isolate the benefit of your architecture, you should compare against a simple ensemble of, e.g., VGGT runs with multiple different reference views (e.g., using DINO-based selection or random) and aggregate the results (e.g., median). This would show whether your method's gains are simply from avoiding a single bad reference or a more fundamental advantage.
3. **Ablation on the number of input views (scalability).** The method is presented as handling "varied inputs," but experiments use a fixed number of views (e.g., N=10 for RealEstate10K). To prove robustness and practicality, you must show performance does not degrade when processing variable-length sequences, especially very short (2-3) and long (>50) sequences, which is critical for real-world use.
4. **Evaluation on a dedicated dynamic scene benchmark.** The claim of handling dynamic scenes is weakly supported (only an "internal dynamic scene dataset" in training). You must test on a standard dynamic scene benchmark (e.g., DAVIS, Kubric, or Sintel's moving objects) with metrics for reconstruction accuracy on independently moving objects to validate this claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis linking low-dimensional pose structure to performance.** Figure 4/6 shows pose distributions are more structured, but you do not show this directly causes better accuracy or robustness. You should provide a quantitative correlation (e.g., between the eigenvalue concentration of predicted poses and reconstruction error across scenes) to argue this is a meaningful advantage.
2. **Failure mode analysis for the claimed robustness.** The paper shows near-zero variance on DTU/ETH3D, but these are relatively clean, structured datasets. You must analyze where the method *does* fail—e.g., on highly repetitive textures, low-texture regions, or severe occlusions—and discuss whether permutation equivariance helps or hurts in these cases. This is critical for assessing real-world applicability.
3. **Analysis of the confidence map's meaning and utility.** The confidence map is supervised with a simple threshold but never analyzed. Does it correctly capture uncertainty? Does it correlate with reconstruction error? You should show calibration plots or use it to filter point clouds, demonstrating it is a useful output and not just a training trick.

### Visualizations & Case Studies
1. **Visual side-by-side comparison of reconstructions under different input orders.** To visually prove permutation equivariance, show point clouds/camera poses from the same scene under several random input orders, aligned with a similarity transform. They should be visually indistinguishable. Currently, only numerical variance is given, which is not convincing.
2. **Case study on scenes where reference view choice critically matters.** Pick a challenging scene (e.g., with repetitive structure, low overlap) where VGGT/DUSt3R fails with a poor reference view, and show π³ succeeds. The current Figure 2 only shows aggregate bars; a concrete visual example would starkly illustrate the claimed robustness.
3. **Visualization of attention patterns in the alternating attention module.** Since the architecture is key to permutation equivariance, visualize cross-view attention maps to show how information is aggregated without a reference token. This would help explain how the model achieves consistent predictions and builds a global scene understanding.

### Obvious Next Steps
1. **Incorpose variable number of views during training and evaluation.** The training protocol (batch of 2-24 images) should be explicitly analyzed for its effect on generalization to longer sequences. You should train and test with variable N to ensure the model doesn't implicitly assume a fixed context size.
2. **Compare with classical SfM (e.g., COLMAP) on reconstruction quality and speed.** For ICLR, it's important to contextualize gains against classical baselines, especially on accuracy and robustness in sparse-view settings. The paper only compares to other deep learning methods.
3. **Explore the use of the confidence map for active vision tasks.** A natural next step is to use the predicted confidence for downstream decision-making, like view planning for a robot. A simple experiment (e.g., selecting the next best view based on confidence) would demonstrate the practical value of this output.

# Final Consolidated Review
## Summary
This paper introduces π³, a permutation-equivariant neural network for visual geometry reconstruction that eliminates the reliance on a fixed reference view. The model predicts affine-invariant camera poses and scale-invariant pointmaps per view, ensuring inherent robustness to input ordering. It achieves state-of-the-art or highly competitive performance across numerous benchmarks for camera pose estimation, monocular/video depth estimation, and dense pointmap reconstruction.

## Strengths
- **Directly addresses a clear and impactful limitation**: The paper convincingly demonstrates that the common inductive bias of anchoring reconstruction to a reference view causes instability in prior methods (e.g., Fig. 2). The proposed permutation-equivariant architecture provides a principled solution, validated by near-zero variance in output across input permutations (Table 6).
- **Extensive and state-of-the-art empirical results**: The method establishes a new SOTA or matches strong performance across a wide range of tasks (camera pose, depth, point cloud) on over a dozen established benchmarks (Tables 1-5). Gains are substantial, e.g., reducing video depth Abs Rel on Sintel from 0.299 (VGGT) to 0.233.
- **Efficient and practical**: The model is fast (57.4 FPS) and has a smaller parameter count (959M) than several competitors while maintaining high accuracy, making it suitable for real-world applications.

## Weaknesses
- **Initialization dependency obscures the source of gains**: The final model is initialized from a pre-trained VGGT encoder (Appendix A.2). While the ablation in Table 8 shows π³ can outperform VGGT when both are trained from scratch with a stabilizing proxy task, the core contribution's performance is entangled with the use of powerful pre-trained features from a reference-based model. This makes it harder to attribute gains solely to the proposed architecture.
- **Insufficient analysis of the pose scaling mechanism**: The translation component of the relative pose loss is scaled using the optimal factor s* derived from point cloud alignment (Eq. 10). While pragmatic given the similarity ambiguity, the paper provides no analysis of how errors in this scale estimation (e.g., from noisy depth or occlusions) propagate to pose accuracy. A sensitivity analysis would strengthen the methodological justification.

## Nice-to-Haves
- **Analysis of performance with variable sequence lengths**: While the model handles "varied inputs," a systematic evaluation of how reconstruction quality scales with the number of input views (especially very few or very many) would better demonstrate its robustness for unconstrained real-world use.
- **Deeper investigation of failure modes**: The paper shows impressive robustness on clean datasets. An analysis of where the method still fails (e.g., in low-texture regions, with severe occlusions, or on transparent objects as noted in limitations) would provide a more complete picture of its capabilities.
- **Visual demonstration of permutation equivariance**: While Table 6 provides strong quantitative evidence, a visual side-by-side comparison of reconstructions from the same scene under different random input orders would make the robustness claim more immediately compelling.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Weaknesses:**
- **"Pose scale ambiguity and supervision is underdeveloped"**: The paper explicitly addresses the scale ambiguity via relative pose supervision scaled by s*, which is a standard and reasonable approach for similarity-invariant problems. The critic's concern about error propagation is a potential issue but not a fundamental flaw.
- **"Claim of low-dimensional manifold is post-hoc"**: The observation about pose distribution structure (Fig. 4/6) is presented as an analysis of an outcome, not a core design claim. Criticizing it as post-hoc is not a substantive weakness of the method's contribution.
- **"Need for comparison against classical SfM (e.g., COLMAP)"**: The paper's scope is advancing feed-forward neural reconstruction; comparing against iterative optimization-based classical methods, while interesting, is outside its stated contributions and is not a standard requirement in this literature.
- **"Requires more permutations for full validation"**: The test of cycling the first frame (Table 6) is a standard and sufficient stress test for permutation sensitivity. Demanding exhaustive random permutations is an unnecessary rigor requirement.
- **"Lack of theoretical justification"**: This is an empirical systems paper; demanding theoretical proofs for the convergence of the relative pose objective is imposing an arbitrary rigor requirement not standard in the field.

**Strengths:**
- **"The paper is well-written" / "The topic is important"**: These are generic strengths applicable to many papers and do not identify something specific this paper does exceptionally well.

## Novel Insights
The paper's core novel insight is the systematic identification and successful elimination of the reference-view bias, a deeply ingrained paradigm in both classical and learning-based multi-view reconstruction. It demonstrates empirically that this bias is not necessary and can be detrimental to robustness. By fully embracing permutation equivariance and predicting geometry in a purely relative, per-view manner, the work shows that reference-free systems are not only viable but can lead to superior accuracy and stability, offering a new paradigm for building robust 3D vision models.

## Suggestions
- Conduct a controlled experiment to better disentangle the source of performance gains: take the released VGGT checkpoint, remove its reference token mechanism, fine-tune it using the π³ losses (relative pose, scale-invariant pointmap), and compare its performance to both the original VGGT and the full π³ model. This would more cleanly isolate the benefit of the architectural/objective change from the benefit of additional training or initialization.
- Include a brief analysis in the main text or appendix on the sensitivity of pose translation error to the accuracy of the scale factor s*, perhaps by adding synthetic noise to the depth used in the alignment or by comparing against an oracle scale.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
