=== CALIBRATION EXAMPLE 60 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Foundation Model for 3D Geometry Prediction" overclaims. "Foundation model" in the current literature connotes large-scale pre-training with emergent generalization and broad downstream adaptability—properties only cursorily demonstrated here. The abstract states the model achieves "superior performance" across tasks, but the experimental section lacks quantitative depth results entirely (only Figure 5 qualitative comparisons are provided for that task), which means the abstract's promise is not fulfilled in the paper body. The claim that the two-stage framework builds a representation that is "both generalizable and intrinsically invariant" is vague and never given a precise mathematical meaning.

---

### Introduction & Motivation

The core motivation—that surface normals provide a deterministic, intrinsically invariant signal that can anchor and improve pointmap regression—is scientifically reasonable and worth investigating. The argument against diffusion models for geometric regression (non-deterministic, no one-to-one mapping) is well-articulated.

However, several claims in the introduction lack grounding:

- "Introducing normal information during geometric prediction can **significantly** improve the accuracy of point maps" (p. 2) — this is the central claim of the paper, but it is stated without forward reference to ablation data, and the actual quantitative evidence for this specific claim never materializes: there is no ablation that compares a pointmap model trained with vs. without the normal head/Stage 2.
- "DUSt3R-based methods overlook a crucial geometric information—surface normals" (p. 3) — VGGT (Wang et al., 2025a) is cited in Related Work and explicitly uses multiple prediction heads including normals; the characterization of the whole class as ignoring normals is imprecise.
- The contributions bullet points are stated at a high level of abstraction. The claim of "high-quality performance in various 3D tasks" is the *finding*, not a technical contribution.

---

### Related Work

The coverage is thorough. The positioning against MASt3R (no depth/normal heads) and MoGe (affine-invariant but monocular, no matching) is fair. The paper appropriately credits VGGT's multi-head prediction design but argues it "overlooks the normal attribute"—yet VGGT does include geometric prediction heads; the paper needs to clarify precisely which attributes VGGT lacks or under-predicts.

---

### Method

**Shared Backbone (Sec. 3.1).** The shared-weight decoder across views is presented as a key architectural novelty. However, the paper does not quantify the parameter count relative to DUSt3R/MASt3R, does not report wall-clock training time, and there is no ablation comparing shared vs. separate decoders on any metric. The efficiency claim is asserted but unverified. Furthermore, the operational detail of how two-view features are routed through the same decoder—and whether this impacts the cross-view attention that is central to DUSt3R's design—is not explained.

**Position-Interpolated RoPE (Sec. 3.1, Eq. 2).** This is directly taken from the context-window extension work of Chen et al. (2023) and applied to ViT. While the adaptation to image resolution is reasonable, it is a straightforward transfer of an existing technique. The paper does not report how much performance degrades at 1024× without PI-RoPE vs. with it—again, no ablation.

**Two-Stage Training (Sec. 3.2).**

*Stage 1 — Scale-invariant Pointmap:* The losses (Eqs. 3–7) are largely copied from MASt3R. The Pointmap Normal Loss $L_{\text{pts\_n}}$ (Eq. 6) is new and interesting: it regularizes the pointmap so that the normals derived from it match ground-truth normals. However, the loss weight $\eta_2 = 0.1$ is given with no ablation study to justify its magnitude. The four losses in Eq. 8 interact in non-trivial ways, and no analysis of their individual contributions is provided.

*Stage 2 — Intrinsic-invariant Pointmap:* The paper defines this in Eq. 9 as:
$$P_i^n = P_i \oplus n$$
This is simply *feature concatenation* of the predicted normal into the pointmap token. Naming this operation "intrinsic-invariant pointmap" is misleading—concatenation of a normal vector to a pointmap does not by itself make the representation invariant to anything. The theoretical justification given—that normals are a "deterministic, locally invariant" property—applies to ground-truth normals, not to the model's current normal prediction during training. The actual mechanism by which Stage 2 induces intrinsic invariance is unexplained.

The switch from "one-to-many" (multi-view) to "one-to-one" (single-view) supervision in Stage 2 is stated to "reduce ambiguity" (p. 7) but the causal mechanism is never explained. If the model is a two-image encoder, how is "single-viewpoint optimization" actually implemented architecturally? Is the second image discarded? Is self-pairing used? This is a critical implementation detail that is absent.

The removal of the confidence loss (used in DUSt3R/MASt3R) is claimed to be enabled by normal supervision. No ablation tests this—it is possible that normal supervision without removing confidence loss performs even better.

**Multi-view Inference (Sec. 3.3).** The "post-processing pipeline" reduces to: run the model pairwise (one-vs-all), collect matches, and run the existing MASt3R-SfM triangulation pipeline. This is not a novel contribution; it is using Dens3R as a drop-in feature extractor for MASt3R's existing reconstruction stack.

---

### Experiments & Results

This is the section with the most severe problems.

**Training Data — Missing Entirely.** The paper never states what datasets are used for training Dens3R. DUSt3R and MASt3R both enumerate their training data in detail (Habitat, Co3D, ARKitScenes, Megadepth, etc.) as this is central to understanding generalization. Without this, the paper is not reproducible and comparisons to baselines may be confounded (e.g., if Dens3R is trained on the test sets of competitors).

**Architecture Details — Missing.** No model size, number of parameters, ViT variant (Base/Large/Huge), or number of decoder layers is reported. It is impossible to assess whether performance gains come from the proposed design choices or simply from having more parameters.

**Normal Prediction (Table 1).** The comparison includes DSINE, GeoWizard, StableNormal, and Lotus. Critically, **VGGT is absent** despite being cited throughout the paper as a closely related method that also regresses geometric quantities from images. Similarly, Metric3Dv2 and Omnidata v2 are missing. The gains over the second-best method range from ~1.5° (NYUv2) to ~4.2° (Sintel) mean angular error, which are non-trivial but modest. No statistical significance or confidence intervals are reported.

**Depth and Pointmap (Section 4.2 / Figure 5).** This section presents **only qualitative results**. There is no Table showing depth metrics (AbsRel, RMSE, δ<1.25, etc.) on any dataset (not NYUv2, not ScanNet, not ETH3D, not KITTI). For a paper whose second core claim is improved depth estimation, this is an inexcusable omission. Standard depth evaluation datasets and protocols exist; the authors chose not to use them.

**Ablation Studies — Entirely Absent.** The paper makes at least five separable design choices:
1. Shared vs. separate decoder
2. Stage 1 alone vs. Stage 1 + Stage 2
3. With vs. without PI-RoPE at high resolution
4. With vs. without confidence loss
5. Normal head contribution to pointmap/depth accuracy

None of these is ablated. Without ablations, it is impossible to attribute the observed improvements to specific claimed contributions.

**Image Matching (Table 2).** The ZEB benchmark comparison with MASt3R shows a meaningful 4.6-point improvement in mean AUC (64.5 vs. 59.9). This is the paper's strongest quantitative result. However, MASt3R's reported number should be reproduced with the same evaluation code—it's unclear whether the numbers are taken from the original paper or re-evaluated.

**Downstream Applications.** The paper mentions segmentation and object detection as possible extensions (p. 8) but provides zero evidence for these capabilities. This appears to be speculation.

---

### Writing & Clarity

Section 3.2's introduction of "intrinsic invariance" is circular and confusing. The term is defined via an appeal to the determinism of ground-truth normals but then applied to the learned representation without justification. The phrase "the model to independently attend to each viewpoint" (p. 7) contradicts the stated cross-view architecture that is central to Stage 1. The transition from Stage 1 to Stage 2 training—specifically, what is frozen, what is fine-tuned, and how data is resampled—is described at a level that precludes reproduction.

---

### Limitations & Broader Impact

The paper has no limitations section. Obvious failure modes that go unacknowledged:
- Single-view ambiguity in textureless regions (mentioned for baselines but not addressed for Dens3R itself)
- Behavior under extreme domain shift (the training data is undisclosed, making out-of-distribution behavior impossible to assess)
- Computational cost of the coarse-to-fine training at 1024 resolution
- Failure cases of the post-processing (when do matching triangulation pipelines fail?)

---

### Overall Assessment

Dens3R proposes a two-stage training scheme that augments a DUSt3R/MASt3R-style dense geometric regression model with surface normal prediction, motivated by the deterministic nature of normals as a stabilizing signal. The conceptual insight—that normals are intrinsically less ambiguous than pointmaps and can anchor multi-task training—is potentially valuable. However, the paper falls well short of ICLR's bar in its current form. The experimental evaluation has critical gaps: there are **no quantitative depth results at all**, training data is never disclosed, architecture specifications are absent, and there are no ablation studies to isolate which design choices drive the improvements. The core technical contribution—the "intrinsic-invariant pointmap"—reduces to feature concatenation (Eq. 9) with a changed supervision regime, and the name is not mathematically justified. The paper reads as a preliminary technical report rather than a complete scientific contribution. To be competitive at ICLR, the authors need: (1) full training data disclosure, (2) quantitative depth benchmarks alongside normal benchmarks, (3) ablations for each claimed design decision, (4) a rigorous definition of "intrinsic invariance," and (5) comparison with VGGT on all evaluated tasks.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents Dens3R, a dense 3D vision foundation model designed for unified geometric prediction, jointly regressing depth, surface normals, and pointmaps from unposed images. The authors propose a two-stage training framework that transitions from scale-invariant to an "intrinsic-invariant" pointmap representation, leveraging surface normals to resolve geometric ambiguity. Extensive experiments demonstrate superior performance over existing DUSt3R-based methods and diffusion-based geometry estimators across multiple benchmarks for depth, normal, and matching tasks.

### Strengths
1.  **Unified Multi-Task Framework:** The model addresses the inconsistency issue inherent in predicting geometric quantities in isolation. By explicitly coupling normal estimation with pointmap prediction, the method achieves higher geometric consistency, as evidenced by improved performance on surface normal benchmarks (Table 1, Tab. 6) compared to diffusion-based baselines like StableNormal.
2.  **High-Resolution Robustness:** The introduction of position-interpolated rotary positional encoding effectively mitigates the performance degradation common in ViT-based geometric models when handling inputs beyond training resolution. Section A.7 and Figure 21 provide qualitative evidence that the model maintains structural integrity at 2K resolution, a feat the authors note previous methods lack.
3.  **Architectural Efficiency:** The use of a shared encoder-decoder backbone significantly reduces memory usage and parameter counts compared to separate decoders for reference and main views. Table 4 explicitly quantifies this reduction (decreasing memory cost from 4.6 GB to 4.1 GB and params to 624M), making inference more accessible.
4.  **Comprehensive Evaluation:** The paper includes rigorous testing across diverse datasets (indoor, outdoor, object-centric) and tasks (matching, depth, normal, segmentation). The inclusion of ablation studies on training strategies (Section A.1) and downstream applications (Segmentation, Surface Reconstruction) adds credibility to the model's versatility.

### Weaknesses
1.  **Novelty of "Intrinsic-Invariant" Representation:** The concept of intrinsic-invariant pointmaps closely resembles the "affine-invariant" formulation in MoGe (Wang et al., 2025b) and standard normalization techniques. The distinction between MoGe’s affine invariance and Dens3R’s intrinsic invariance is not rigorously delineated in Section 3.2, leaving some ambiguity about whether this is a structural novelty or a methodological refinement.
2.  **Training Complexity vs. Single-Stage Alternatives:** While the two-stage training strategy (Scale-Invariant -> Intrinsic-Invariant) improves accuracy, it increases training complexity and time compared to end-to-end multi-task training. The paper does not provide a detailed analysis of the convergence speed trade-off versus the accuracy gains (e.g., time-to-convergence metrics vs. AUC improvements).
3.  **Performance on Thin Structures:** The authors acknowledge a significant limitation in predicting thin structures (Figure 12, Section A.8). For a model claiming to be a "Foundation Model," this specific geometric failure mode is concerning, as it limits applicability in scenarios requiring fine-detail reconstruction (e.g., foliage, wires).
4.  **Dependency on External Post-Processing:** While claimed as a feed-forward model, the multi-view inference section (Sec 3.3) relies on a post-processing pipeline involving triangulation and potentially MASt3R-SfM extensions. This suggests the full pipeline is not purely feed-forward in a strict sense for multi-view consistency, which slightly contradicts the abstract's emphasis on single-pass efficiency for multi-view.

### Novelty & Significance
**Novelty:** The paper presents moderate-to-high novelty. While it builds heavily on the DUSt3R/MASt3R lineage (pointmap representations), the specific integration of normal maps to stabilize pointmap learning via an intrinsic-invariant formulation is a distinct contribution. The transfer of position-interpolated RoPE from LLMs to 3D ViTs is a smart engineering adaptation but not entirely new; however, the demonstration of its utility in 3D geometry is valuable.

**Significance:** The significance is high for the 3D vision community. Consistency between depth and normals is a long-standing problem. By achieving unified outputs without requiring separate specialized models, Dens3R simplifies the pipeline for downstream tasks like SLAM and scene reconstruction. The performance gains on matching (ZEB) and normals (DIODE) are substantial enough to warrant attention.

### Suggestions for Improvement
1.  **Clarify Theoretical Distinction:** Provide a clearer mathematical or conceptual comparison between "intrinsic-invariant" (Dens3R) and "affine-invariant" (MoGe) representations. A subsection explaining why normals specifically induce intrinsic invariance in this context would strengthen the theoretical foundation.
2.  **Ablation on Training Stage:** Include a quantitative comparison of training time and FLOPs for the two-stage approach versus a single-stage joint training. This would help readers evaluate whether the performance gain justifies the additional training complexity.
3.  **Address Thin Structure Limitation:** Provide an analysis of why thin structures fail (e.g., lack of normal information or resolution limits) and discuss potential modifications (e.g., hierarchical decoding) to mitigate this in future work.
4.  **Multi-View Feed-Forward Clarification:** Clearly distinguish between the core prediction pipeline (single view/fixed pair) and the post-processing multi-view inference. If possible, discuss plans to integrate geometric consistency directly into the network weights rather than relying on post-processing triangulation for future multi-view extensions.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Zero-Shot Transfer Evaluation:** Validate the "foundation model" claim by testing on unseen downstream tasks (e.g., semantic segmentation, object detection) using a frozen backbone with linear probing, rather than full fine-tuning.
2. **Out-of-Distribution Generalization:** Test on domain-shifted data (e.g., synthetic-to-real, artistic renderings, extreme weather) to prove the "intrinsic-invariant" representation generalizes beyond standard RGB-D training distributions.
3. **Efficiency vs. SOTA Baselines:** Provide latency and memory benchmarks against DUSt3R and MASt3R on identical hardware; comparing only against a "w/o Shared" ablation (Tab 4) is insufficient to claim efficiency.
4. **Multi-View Consistency Quantification:** Report cycle-consistency errors or reprojection errors across N>2 views to quantitatively verify the claimed "geometrically consistent multi-view inference" beyond pairwise matching AUC.

### Deeper Analysis Needed (top 3-5 only)
1. **Multi-Task Interference Analysis:** Provide gradient similarity or task-performance trade-off curves to demonstrate that joint regression of normals and pointmaps actually reduces interference rather than degrading individual task performance.
2. **Invariance Robustness Test:** Quantitatively measure output stability under photometric transformations (exposure, color jitter) to empirically verify the "intrinsic-invariant" property claimed in the method name.
3. **Stage Dependency Ablation:** Isolate the contribution of Stage 1 features to Stage 2 performance by freezing Stage 1 weights completely vs. fine-tuning, to prove the two-stage dependency is necessary.

### Visualizations & Case Studies
1. **Feature Representation Visualization:** Use t-SNE plots to compare feature distributions between Stage 1 (scale-invariant) and Stage 2 (intrinsic-invariant) to visually confirm the representation shift.
2. **High-Resolution Artifact Inspection:** Include side-by-side crop comparisons at 2K+ resolution against DUSt3R to explicitly show the specific geometric artifacts prevented by the position-interpolated RoPE.
3. **Systematic Failure Analysis:** Expand the limitations section with a quantitative breakdown of failure rates across specific scene attributes (e.g., transparency, motion blur, thin structures) rather than selective qualitative examples.

### Obvious Next Steps
1. **Standardized Foundation Benchmarks:** Evaluate on emerging 3D foundation model benchmarks (e.g., 3D-Bench) to contextualize performance against broader vision foundation models, not just 3D reconstruction baselines.
2. **Data Harmonization Protocol:** Detail the specific transformations applied to unify normal/depth coordinate systems across the 20+ heterogeneous datasets, as this is critical for reproducibility and claim validity.
3. **Pose-Free Robustness Analysis:** Evaluate geometry prediction accuracy as a function of input viewpoint baseline distance to determine the operational limits of the "unposed" claim.

# Final Consolidated Review
## Summary
Dens3R proposes a dense 3D vision foundation model that jointly regresses pointmaps, depth maps, and surface normals from unposed images. The key contributions are a two-stage training framework (scale-invariant → intrinsic-invariant pointmap), a shared encoder-decoder architecture, and position-interpolated RoPE for high-resolution inference. The model demonstrates strong performance across normal estimation, depth prediction, and image matching benchmarks.

## Strengths
- **Unified Multi-Task Geometric Prediction:** Unlike prior methods that predict geometric quantities in isolation, Dens3R explicitly models structural coupling between pointmaps and normals. The intrinsic-invariant formulation leverages the deterministic nature of surface normals (one-to-one correspondence per surface) to stabilize multi-task training, yielding consistent geometric outputs from a single forward pass.

- **Strong Quantitative Performance:** The model achieves state-of-the-art results on surface normal prediction across multiple benchmarks (NYUv2: 16.1° mean error vs. 17.5° for StableNormal; DIODE-outdoor: 20.8° vs. 24.7° for StableNormal, Table 1) and image matching (ZEB benchmark: 64.5 AUC@5° vs. 59.9 for MASt3R, Table 2). Depth results (Table 7) show competitive performance against recent baselines.

- **High-Resolution Robustness via Position-Interpolated RoPE:** The adaptation of context-window extension techniques from LLMs to ViT-based geometric models effectively handles resolution extrapolation. Figure 21 and Appendix A.7 demonstrate that Dens3R maintains structural coherence at 2K resolution where DUSt3R/VGGT exhibit degenerate artifacts.

- **Architectural Efficiency:** The shared encoder-decoder design reduces parameters from 737M to 624M and memory from 4.6GB to 4.1GB (Table 4) while maintaining prediction quality, enabling practical deployment on single GPUs.

- **Comprehensive Training Data Disclosure:** Table 5 explicitly documents 28 training datasets with quality tiers (Type A/B/C), ratios, and image pair counts—addressing reproducibility concerns common in foundation model papers.

- **Ablation Studies Provided:** Section A.1 (Table 3, Figures 8, 21) ablates key components including intrinsic-invariant training, coarse-to-fine strategy, and position-interpolated RoPE, demonstrating their individual contributions.

## Weaknesses
- **Inconsistent Baseline Comparisons:** VGGT is included in depth (Table 7) and pose estimation (Table 8) comparisons but absent from the normal prediction evaluation (Table 1). Since VGGT also predicts geometric quantities including normals, this omission weakens the comprehensive SOTA claim. The paper should either include VGGT in normal benchmarks or explain the exclusion.

- **Overclaimed Terminology:** The term "intrinsic-invariant pointmap" (Eq. 9) denotes feature concatenation (P^n = P ⊕ n). While the intuition—leveraging normals' deterministic one-to-one property—is sound, the name suggests mathematical invariance properties not rigorously established. The paper would benefit from clearer formalization of what invariance means in this context.

- **Thin Structure Limitation:** The authors acknowledge failure on thin structures (Figure 12, Section A.8). This is a notable limitation for a "foundation model" targeting general 3D reconstruction, as thin structures (foliage, wires, railings) appear frequently in real-world scenes.

- **Multi-View Pipeline Not Purely Feed-Forward:** The multi-view inference (Section 3.3) relies on MASt3R-SfM triangulation post-processing rather than end-to-end prediction. While this inherits established reconstruction capabilities, it contradicts the abstract's framing of "single forward pass" inference for multi-view scenarios.

- **No Efficiency Benchmarks Against Baselines:** Table 4 compares the shared vs. non-shared design internally but provides no latency/throughput comparison against DUSt3R, MASt3R, or VGGT on identical hardware. The efficiency claim relative to prior work remains unsubstantiated.

## Nice-to-Haves
- Zero-shot transfer evaluation on downstream tasks (e.g., semantic segmentation with frozen backbone + linear probing) to substantiate the "foundation model" framing
- Convergence time analysis quantifying the accuracy/efficiency trade-off of two-stage vs. single-stage training
- Out-of-distribution robustness tests (synthetic-to-real domain shift, extreme illumination)
- Cycle-consistency metrics for multi-view geometric coherence beyond pairwise matching AUC

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **"No quantitative depth results"** — This claim is factually incorrect. Table 7 provides full depth evaluation on NYUv2, DIODE-indoor, and DIODE-outdoor with REL, RMSE, and δ metrics.

- **"Training data not disclosed"** — Table 5 comprehensively lists 28 datasets across three quality tiers with image pair counts and ratios.

- **"No ablation studies"** — Section A.1 provides ablations on intrinsic-invariant training (Table 3), coarse-to-fine strategy (Table 3), PI-RoPE (Figure 8a, 21), and shared encoder-decoder (Table 4).

- **"VGGT not compared"** — VGGT appears in depth (Table 7) and pose (Table 8) comparisons. The valid critique is its absence from normal evaluation, not blanket omission.

- **"Architecture details missing (no model size, ViT variant)"** — Table 4 reports 624M parameters. ViT variant specification would be helpful but is not a critical omission.

- **"No failure mode acknowledgment"** — Section A.8 and Figure 12 explicitly discuss thin structure failures.

## Novel Insights
Beyond the paper's contributions, the synthesis reveals an interesting tension: the two-stage training essentially decouples the representation learning of the pointmap (Stage 1's multi-view matching objective) from the normal-guided refinement (Stage 2's single-view optimization). This suggests that geometric foundations may benefit from curriculum-style training where cross-view constraints establish coarse 3D structure before per-view intrinsic properties sharpen local details. The intrinsic-invariant pointmap can be viewed as embedding surface orientation priors directly into the 3D representation, which could inspire similar hybrid representations for other geometric modalities (curvature, material properties).

## Suggestions
- Add VGGT to the normal prediction benchmark (Table 1) or justify its exclusion to ensure comprehensive baseline coverage across all evaluated tasks.
- Clarify the single-view optimization in Stage 2: specify whether the second image input is masked, replaced with a blank token, or processed differently architecturally to enable "one-to-one" supervision.
- Report inference latency (ms/frame) and throughput metrics against DUSt3R and MASt3R on standardized hardware to substantiate practical efficiency claims.
- Consider renaming "intrinsic-invariant" to reflect that it's a training strategy leveraging normal determinism rather than a mathematically proven invariant representation.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
