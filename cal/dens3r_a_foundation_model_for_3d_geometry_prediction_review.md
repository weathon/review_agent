=== CALIBRATION EXAMPLE 58 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title “DENS3R: A FOUNDATION MODEL FOR 3D GEOMETRY PREDICTION” accurately reflects the paper’s core contribution. The abstract clearly states the goal of unified geometric prediction, the two-stage training framework, key components (shared backbone, position-interpolated RoPE, intrinsic-invariant pointmap), and claims of superior performance. The abstract’s claims are supported by the experiments section, but a reader would need to check if the term “foundation model” is justified (e.g., scale of training, zero-shot generalization)—this is partially addressed in the appendix with the large training dataset and downstream task adaptation.

### Introduction & Motivation
The introduction effectively motivates the problem: isolated prediction of geometric quantities (depth, normals) lacks consistency, and a unified model is needed. The critique of diffusion models for geometric tasks (Sec. 1, paragraph 2) is somewhat one-sided. The authors state geometric prediction is “fundamentally different” from generation because it requires a “deterministic task” and “strict one-to-one correspondence.” However, recent diffusion-based methods (e.g., StableNormal, GeoWizard) have shown strong deterministic regression capabilities via modes of the diffusion prior. This argument could be nuanced; dismissing them as inherently unsuitable may overstate the case. The introduction of “intrinsic-invariant” pointmaps via normals is well-motivated, but the exact meaning of “intrinsic-invariance” could be clearer upfront (is it affine invariance? scale/shift invariance?). Contributions are listed clearly.

### Method / Approach
**Sec. 3.1 Model Formulation:** The shared encoder-decoder backbone is a sensible design for parameter efficiency. The position-interpolated RoPE (Eq. 2) is adapted from LLM context window extension to the image domain—this is a clever and justified solution for high-resolution robustness. However, the description is brief; a short discussion on why interpolation is more stable than extrapolation for RoPE in images would strengthen the justification.

**Sec. 3.2 Foundation Model Training:** The two-stage strategy is the core novelty. Stage 1 uses losses from MASt3R to learn scale-invariant pointmaps. Stage 2 introduces normal supervision to create an “intrinsic-invariant” pointmap. A significant concern is the **justification and clarity of the “intrinsic-invariant” concept**. The text says it’s “inspired by the affine-invariant formulation of MoGe” but then focuses on normals providing a “locally deterministic geometric property.” The connection between “affine-invariant” (MoGe) and “intrinsic-invariant” (this work) is not rigorously defined. Equation 9 simply concatenates pointmap and normal features; it’s unclear how this operation yields an invariant representation mathematically. The claim that this representation “anchors the pointmap to a more deterministic geometric interpretation” is intuitive but lacks formal grounding.

Additionally, the training objective for Stage 2 (Eq. 11) removes the matching loss \(L_{\text{match}}\) and adds a normal loss \(L_n\). The authors state that confidence loss in prior works causes models to “ignore complex scenarios,” and that using normals obviates the need for additional views. This is an interesting claim, but the paper does not provide an ablation or analysis to validate that the removal of confidence weighting is safe *because* of the normal supervision. Without this, the training stability claim is somewhat anecdotal.

The coarse-to-fine training strategy is mentioned but lacks details: what exactly is “high-resolution data” and how is it filtered? The appendix (Tab. 5) shows data types A/B/C, but the split for fine-stage is not explicit.

**Sec. 3.3 Model Inference:** The shared decoder design that removes the need for a fixed reference view is a practical improvement. The multi-view post-processing pipeline is described at a high level (“constructing and optimizing a dense correspondence network”) but lacks algorithmic details, making it difficult to assess or reproduce.

### Experiments & Results
**Sec. 4.1 Normal and Matching Prediction:** Quantitative results (Tab. 1, Tab. 2) show clear improvements over strong baselines. However, a critical issue is the **comparison fairness**. For normal prediction, baselines like DSINE, StableNormal, and Lotus are monocular methods, while Dens3R uses multi-view cues during training (via pointmaps from image pairs). This is an inherent advantage, not an architectural one. The paper should explicitly discuss this and perhaps include an ablation where Dens3R is trained only on monocular data to isolate the benefit of the unified framework. Similarly, for matching (Tab. 2), comparing against MASt3R (which Dens3R builds upon) is appropriate, but the gains should be contextualized: how much comes from the added normal supervision versus other improvements?

**Sec. 4.2 Pointmap and Depth Prediction:** Qualitative figures (Fig. 5) show impressive results, but quantitative depth evaluation is only in the appendix (Tab. 7). The depth metrics are competitive but not always best. The paper would benefit from a consolidated table in the main text for depth, as it’s a key claimed output.

**Appendix Ablations (Sec. A.1):** The ablation studies are crucial but have weaknesses. Tab. 3 shows improvements from intrinsic-invariant training and coarse-to-fine strategy, but the two factors are combined; a full ablation isolating each component (position-interpolated RoPE, shared decoder, two-stage training, coarse-to-fine) would be more convincing. Fig. 8a shows high-resolution inference benefits, but it’s unclear if the comparison baseline (DUSt3R/VGGT) uses the same training data and strategy. The claim that position-interpolated RoPE alone is insufficient without high-res training (Sec. A.7) is important but not backed by a controlled experiment.

**Downstream Applications (Sec. A.2):** The segmentation and surface reconstruction results are promising but presented as proof-of-concept only. No quantitative evaluation is provided for segmentation, and surface reconstruction uses only qualitative visualization.

### Writing & Clarity
The paper is generally well-structured and readable. However, some key concepts are under-explained:
- “Intrinsic-invariant pointmap”: The term is used throughout but never formally defined. It appears to mean a pointmap representation that, when combined with normals, becomes more stable to monocular ambiguities (shift/scale). A clearer definition or mathematical formulation would help.
- The relationship between “scale-invariant” (Stage 1) and “intrinsic-invariant” (Stage 2) could be elaborated. How does adding normal supervision transform the representation’s invariance properties?
- The training dataset composition (Tab. 5) is detailed, but the “quality” types A/B/C are heuristic. The impact of this curation on final performance is not analyzed.

### Limitations & Broader Impact
The limitations section (Sec. A.8) is very brief, only mentioning difficulties with thin structures. Other potential limitations should be discussed: the model’s reliance on large-scale synthetic data (which may have a sim-to-real gap), the computational cost of two-stage training, the fact that multi-view inference requires a post-processing pipeline (not end-to-end), and potential failure modes in textureless or highly reflective scenes. Broader impact is not discussed; given the potential for 3D reconstruction in AR/VR and robotics, a brief statement on ethical use would be appropriate.

### Overall Assessment
This paper presents a solid and comprehensive approach to unified 3D geometric prediction. The core idea—leveraging normals to refine pointmaps and achieve consistent multi-task prediction—is novel and well-motivated. The experimental results demonstrate strong performance across multiple benchmarks. However, the paper has notable weaknesses: the “intrinsic-invariant” concept lacks rigorous grounding; the ablation studies are not fully conclusive; and some comparisons (especially for normal estimation) conflate the benefits of multi-view training with architectural advances. For ICLR, the contribution is significant but requires clearer justification of the key innovations and more thorough ablation analysis to isolate their effects. With revisions to address these concerns—particularly a more formal treatment of the representation and more controlled experiments—the paper could meet the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Dens3R, a feed-forward visual foundation model designed to jointly predict multiple 3D geometric quantities—including point maps, depth, surface normals, and matching features—from unposed single or multi-view images. The core innovation is a two-stage training framework that first learns a scale-invariant pointmap and then refines it into an "intrinsic-invariant" pointmap by incorporating surface normal supervision. The architecture employs a shared encoder-decoder backbone and a position-interpolated Rotary Positional Encoding (RoPE) to enhance robustness to high-resolution inputs.

### Strengths
1. **Unified Multi-Task Prediction**: Dens3R successfully regresses multiple correlated geometric outputs (pointmaps, depth, normals, matching) in a single forward pass, addressing a key limitation of prior works that focus on isolated tasks. Evidence: Quantitative results in Tables 1, 2, 6, and 7 show state-of-the-art or competitive performance across diverse benchmarks (NYUv2, ScanNet, ZEB, etc.) for normal estimation, depth, and image matching.
2. **Novel Training Strategy and Representation**: The proposed two-stage training, culminating in an "intrinsic-invariant" pointmap, effectively leverages the deterministic nature of surface normals to resolve monocular ambiguities and improve geometric consistency. Evidence: Ablation studies in Table 3 and Figure 8b demonstrate that both the intrinsic-invariant training and the coarse-to-fine strategy contribute significantly to improved normal accuracy.
3. **Technical Innovations for Scalability**: The introduction of position-interpolated RoPE effectively mitigates performance degradation on high-resolution inputs, a known issue in prior models like DUSt3R. Evidence: Figure 8a and Figure 21 show stable, high-quality pointmap predictions at 2K resolution, unlike comparison methods.
4. **Extensive Empirical Validation**: The paper provides comprehensive evaluations across numerous datasets and tasks (normal, depth, matching, pose estimation, surface reconstruction), including detailed ablation studies (Appendix A.1) and demonstrations of downstream applicability (segmentation, reconstruction in Figures 8c, 8d, 9).

### Weaknesses
1. **Computational Cost and Training Complexity**: The two-stage, coarse-to-fine training paradigm and the use of a large, curated dataset (Table 5) imply significant computational resources and engineering effort, potentially limiting accessibility for many researchers. Evidence: Training used 32 H20 GPUs for approximately two weeks per stage.
2. **Incomplete Baseline Comparisons**: While comparisons are made against strong baselines (DUSt3R, MASt3R, VGGT), the paper does not include an ablation or comparison with a simple multi-task baseline trained end-to-end without the proposed two-stage strategy. This makes it harder to isolate the contribution of the training framework versus the architectural changes.
3. **Limitations in Handling Thin Structures**: The authors acknowledge that the model struggles with thin structures (Figure 12), a common challenge in dense prediction tasks. This suggests the representation or receptive field may still be insufficient for high-frequency geometry.
4. **Clarity on "Intrinsic-Invariance"**: The core concept of the "intrinsic-invariant pointmap" is motivated but could be more precisely defined and differentiated from related concepts like affine invariance (MoGe). The description of how normals provide this invariance is intuitive but could be bolstered with more formal analysis.

### Novelty & Significance
**Novelty**: The work presents a clear advance over existing 3D foundation models (e.g., DUSt3R, MASt3R) by unifying multiple geometric predictions and introducing the intrinsic-invariant pointmap representation. The integration of normals as a key to resolving ambiguity and the adaptation of position-interpolated RoPE for vision transformers are novel and well-motivated contributions.

**Significance**: The model demonstrates strong potential as a versatile backbone for 3D vision. Achieving high performance across a spectrum of tasks from a single model is a significant step towards general-purpose geometric understanding. The results are compelling and, if reproducible, could influence the design of future 3D perception systems. The work aligns well with ICLR's focus on foundational models and innovative learning methodologies.

### Suggestions for Improvement
1. **Conduct a Parameter/Performance Trade-off Analysis**: To better understand the model's efficiency, report FLOPs, parameter counts, and inference speed comparisons with key baselines (DUSt3R, VGGT) on standard resolutions. This is crucial for assessing its practicality as a "foundation model."
2. **Strengthen the Ablation Study**: Include a baseline where all prediction heads (point, depth, normal, match) are trained jointly from scratch in a single stage to disentangle the benefit of the two-stage strategy from the benefit of simply having a multi-task architecture.
3. **Provide More Formal Justification**: Add a brief theoretical discussion or more rigorous experiments to clarify why the "intrinsic-invariant" formulation centered on normals is more effective than other potential constraints for unifying geometry.
4. **Address the Thin Structure Limitation**: Discuss potential architectural modifications (e.g., different loss functions on edges, higher-resolution feature maps) or data strategies that could mitigate this failure mode, framing it as future work.
5. **Clarify Reproducibility**: While datasets and hyperparameters are listed, explicitly state the license for the proposed model weights and code, and confirm the intention to release them to ensure full reproducibility, a key expectation for ICLR.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation of the two-stage training and normal integration.** The paper claims the intrinsic-invariant pointmap and normal integration are key innovations. However, there is no controlled ablation comparing: (a) single-stage training without normals, (b) Stage 1 only, (c) the full two-stage model. Without this, it's impossible to attribute performance gains to the proposed training strategy versus simply using a larger/more diverse dataset.
2. **Comparison with true multi-view foundation models.** The method is promoted as a foundation model for "single-view to multi-view inputs," but quantitative results for depth/normal are all on single-image benchmarks. To support the multi-view claim, experiments on true multi-view datasets (e.g., DTU, Tanks & Temples) comparing against methods like Sparse3R, Fast3R, or traditional MVS are essential.
3. **Ablation on training data composition.** The model is trained on a massive, mixed-quality dataset (Type A/B/C). No analysis shows the contribution of high-quality synthetic (Type A) vs. real (Type C) data. For a foundation model claim, it's critical to know if performance is driven by data scale/quality or architectural novelty.
4. **Efficiency and speed comparison.** The paper mentions a "lightweight" backbone but provides no comparison of inference speed (FPS), FLOPs, or parameters against key baselines like DUSt3R, MASt3R, or VGGT. For a "versatile backbone," practical deployment metrics are necessary.

### Deeper Analysis Needed (top 3-5 only)
1. **What does "intrinsic-invariant" actually mean and prove?** The core concept of an "intrinsic-invariant pointmap" is poorly defined and not quantitatively validated. The paper should analyze the variance of pointmap predictions for the same scene under different lighting/viewpoints, demonstrating improved invariance compared to the scale-invariant baseline.
2. **Mechanism of normal improving geometry.** The claim that normals "significantly improve the accuracy of point maps" is speculative. An analysis correlating normal prediction error with subsequent depth/pointmap error across samples is needed to establish a causal link, rather than just a concurrent improvement.
3. **Analysis of the shared decoder's impact.** The shared encoder-decoder is a design choice, but its impact on cross-view consistency versus task-specific performance is not analyzed. A simple experiment comparing a shared vs. non-shared decoder on matching and depth tasks would validate its necessity.
4. **Failure mode analysis for the post-processing pipeline.** The multi-view inference pipeline is described at a high level but not evaluated. An analysis of its failure cases (e.g., with repetitive textures, low overlap) and a comparison of consistency metrics (e.g., chamfer distance between fused views) before/after fusion are missing.

### Visualizations & Case Studies
1. **Visualizations of failure cases and limitations.** The paper shows only successful predictions (Figs. 4, 5, 6, 7). To properly assess the method, visualizations are needed where it fails: e.g., on the mentioned thin structures, transparent objects, or highly ambiguous monocular scenes. This defines the method's boundaries.
2. **Cross-view consistency visualization.** For a model predicting geometry from multiple views, a critical visualization is showing the same 3D point cloud or mesh generated from different input view pairs of the same scene, overlayed to show consistency (or lack thereof).
3. **Case study comparing unified vs. specialist models.** A dedicated case study on a few complex scenes comparing Dens3R's unified predictions against the current SOTA single-task models (e.g., Depth Anything V2 for depth, DSINE for normals) would strongly demonstrate the value of a unified foundation model.

### Obvious Next Steps
1. **Quantitative evaluation on multi-view reconstruction.** The paper should have included standard 3D reconstruction metrics (Chamfer Distance, F-score) on benchmarks like DTU or ScanNet, using their pipeline to generate a mesh/point cloud from multiple images. This is a direct application and a critical test of geometric consistency.
2. **Robustness test on extreme resolutions.** The position-interpolated RoPE is motivated by high-resolution support. The obvious next step is to systematically test performance degradation from 512 to 2K+ resolutions, comparing against baselines like DUSt3R with and without the proposed encoding.
3. **Expand downstream task evaluation.** The segmentation head experiment (Fig. 8c) is a promising start but is only a qualitative visualization. A quantitative benchmark (e.g., on ADE20K) for this head, trained with the frozen backbone, is needed to substantiate the "foundation model" claim.
4. **Release code and pre-trained models.** For a foundation model paper at ICLR, the obvious and expected next step is to release code and model weights to enable community validation, reproduction, and extension. The absence of this commitment in the paper is a significant gap.

# Final Consolidated Review
## Summary
Dens3R is a feed-forward visual foundation model that predicts multiple 3D geometric quantities—point maps, depth, surface normals, and matching features—from unposed single or multi-view images. Its core innovation is a two-stage training strategy that first learns a scale-invariant pointmap and then refines it into an "intrinsic-invariant" representation by incorporating surface normal supervision, alongside architectural choices like position-interpolated rotary positional encoding and a shared encoder-decoder backbone.

## Strengths
- **Unified multi-task prediction with state-of-the-art performance:** Dens3R jointly regresses pointmaps, depth, normals, and matching features in a single forward pass, outperforming specialized baselines across multiple benchmarks (Tables 1, 2, 6, 7). This addresses a key limitation of prior works that handle these tasks in isolation.
- **Effective training strategy and representation:** The two-stage training, culminating in the intrinsic-invariant pointmap, demonstrably improves geometric consistency and normal accuracy, as shown in ablation studies (Table 3, Figure 8b). The integration of normals helps resolve monocular ambiguities.
- **Robust high-resolution inference:** The novel position-interpolated rotary positional encoding (RoPE) mitigates the performance degradation on high-resolution inputs common in prior models like DUSt3R, enabling stable 2K inference (Figures 8a, 21).

## Weaknesses
- **Insufficient validation of the core "intrinsic-invariant" concept:** The paper's central innovation—the intrinsic-invariant pointmap enabled by normal supervision—is motivated intuitively but lacks a rigorous definition, formal analysis, or controlled experiments isolating its contribution. It is unclear how this representation differs mathematically from related concepts like affine invariance (MoGe) and how much of the performance gain is uniquely attributable to it versus other factors like the two-stage training or data scale.
- **Missing critical multi-view reconstruction evaluation:** The model is promoted for "single-view to multi-view" inputs, but quantitative evaluation is limited to single-image benchmarks for depth and normals. There is no standard 3D reconstruction evaluation (e.g., Chamfer Distance, F-score on DTU/ScanNet) using its multi-view pipeline, which is essential to validate its geometric consistency and utility as a 3D foundation model.
- **Incomplete ablation of the two-stage training framework:** While ablations show combined benefits, there is no controlled comparison against a strong single-stage, multi-task baseline trained end-to-end. This makes it difficult to disentangle the contribution of the proposed two-stage strategy from the benefit of simply having a multi-task architecture with normal supervision.
- **Unfair comparison baseline for normal estimation:** The normal prediction benchmarks compare Dens3R (which benefits from multi-view cues during training) against purely monocular methods (DSINE, StableNormal). This confounds the advantage of architectural unification with the inherent benefit of using multi-view information. A monocular-only ablation of Dens3R or a discussion of this asymmetry is needed for a fair assessment.

## Nice-to-Haves
- Analysis of parameter efficiency, inference speed (FPS), and FLOPs compared to key baselines (DUSt3R, VGGT) to better contextualize its practicality as a "lightweight" foundation model.
- A more detailed failure case analysis, particularly for the acknowledged challenge of thin structures, suggesting potential architectural or data-driven mitigations.
- Quantitative evaluation of the demonstrated downstream applications (e.g., segmentation accuracy on a standard benchmark) to substantiate the foundation model claim beyond qualitative proof-of-concept.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The critique of diffusion models for geometric tasks is one-sided and overstates the case." *Removed as it is a matter of perspective/emphasis, not a factual error. The paper's methodological choice is justified within its framework.*
- **Weakness:** "The multi-view post-processing pipeline lacks algorithmic details, making it difficult to assess or reproduce." *Weakened to a clarity issue; the paper follows the established MASt3R pipeline, and full implementation details are expected in code.*
- **Weakness:** "The confidence loss removal in Stage 2 is not validated by ablation." *Weakened; the paper provides an intuitive justification (normals provide deterministic signal), and requiring a full ablation on this specific design choice is overly granular.*
- **Weakness:** "The term 'foundation model' may not be fully justified." *Removed; the paper demonstrates large-scale training, multi-task capability, and downstream task adaptation, which aligns with common usage in the field.*
- **Weakness:** "Broader impact statement is missing." *Removed as a generic requirement not central to the technical contribution.*

## Novel Insights
The key novel insight is the effective integration of surface normals into a unified 3D geometry prediction framework. By treating normals not as an isolated output but as a source of "intrinsic" geometric invariance, Dens3R demonstrates that jointly modeling this correlated property can anchor and refine other geometric quantities (pointmaps, depth), leading to more consistent and accurate predictions than processing them separately. This moves beyond prior work that either predicts single quantities or treats multi-task learning as a simple parallel head addition.

## Suggestions
- Conduct a clear ablation isolating the two-stage training: compare (a) single-stage multi-task training, (b) Stage 1 only, (c) full two-stage model to definitively show the contribution of the intrinsic-invariant pointmap learning.
- Perform a standard quantitative evaluation of 3D reconstruction quality (e.g., on DTU or ScanNet) using the model's multi-view pipeline to substantiate claims about geometric consistency and multi-view capability.
- Add a brief discussion or experiment to address the comparison fairness in normal estimation, clarifying the role of multi-view training data in Dens3R's performance versus purely monocular baselines.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
