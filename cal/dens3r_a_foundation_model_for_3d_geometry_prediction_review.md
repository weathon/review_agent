=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary

Dens3R presents a unified dense 3D visual foundation model that jointly regresses depth, surface normals, 3D pointmaps, and image matching features from unposed images. The method introduces a two-stage training framework progressing from scale-invariant pointmaps to what the authors term "intrinsic-invariant" pointmaps, a shared encoder-decoder backbone for efficiency, and position-interpolated rotary positional encoding for stable high-resolution inference. Experiments demonstrate strong performance on normal prediction, image matching, and depth estimation benchmarks.

## Strengths

- **Unified multi-quantity geometric prediction with explicit coupling**: Unlike prior DUSt3R-lineage methods that focus primarily on pointmap reconstruction or matching, Dens3R jointly regresses normals, depth, pointmaps, and matching features in a single framework. The integration of surface normals as a geometric anchor that reduces monocular ambiguity is a well-motivated design choice, and Tables 1 and 6 show consistent SOTA-level normal prediction results across five benchmarks (e.g., NYUv2 mean error 16.1° vs. 17.5° for StableNormal, DIODE-outdoor δ11.25 of 43.0 vs. 36.1).

- **Effective high-resolution inference via position-interpolated RoPE**: The paper identifies a concrete failure mode of DUSt3R-based methods—prediction degradation at resolutions beyond training—and adapts LLM context window interpolation to 2D ViT positional encoding. Figure 8a and Figure 21 show qualitatively that Dens3R produces well-structured pointmaps at 2K resolution where DUSt3R and VGGT produce degenerated outputs. This addresses a practical limitation for real applications.

- **Shared encoder-decoder reduces parameters while maintaining quality**: Table 4 shows the shared-weight design reduces parameters from 737M to 624M and memory from 4.6GB to 4.1GB, a non-trivial efficiency gain that also eliminates the need for view-swapping during inference (Section 3.3: "This design removes the need to explicitly define main and reference views").

- **Versatile downstream transfer**: The paper demonstrates extension to segmentation (Fig 8c), surface reconstruction with NeuS (Fig 8d), and automated reconstruction pipelines (Fig 9), showing the backbone serves as a general-purpose geometric prior beyond its primary training objectives.

## Weaknesses

### Major:

- **Inference protocol for normal prediction benchmarks is ambiguous, potentially compromising comparison fairness**: Table 1 compares Dens3R against purely monocular methods (DSINE, StableNormal, GeoWizard, Lotus) on single-image normal estimation benchmarks (NYUv2, ScanNet, etc.). However, Dens3R's architecture inherently processes image pairs with cross-attention (Eq. 1, Section 3.1). The paper never explicitly states whether single images or image pairs were used at test time for Table 1. If pairs from the video sequences were available at inference, Dens3R would have access to multi-view geometric information that monocular baselines fundamentally lack, making the comparison unfair. This must be clarified. If single-view inference is used (e.g., feeding the same image as both inputs), this should be explicitly stated; if pairs are used, these results belong in a separate category from monocular methods.

- **"Intrinsic-invariant" terminology is conceptually misleading**: The paper's central concept—the "intrinsic-invariant pointmap"—claims to leverage the view-invariance of normals (Section 3.2: "surface normals provide an intrinsic, locally deterministic geometric property"). However, Equation 10 supervises the normal head with **view-space normals** ($N_{v,v}$), which are *not* invariant to viewpoint changes—they rotate with the camera. The "intrinsic" property the authors likely intend is that normals are *deterministic* given a surface (one-to-one mapping, unlike depth with scale/shift ambiguity), not that they are geometrically invariant across views. By explicitly linking the term to "affine-invariant formulation of MoGe" (Section 3.2), the paper invites comparison to established invariance concepts that the method does not actually implement. This creates conceptual confusion about what property is being exploited and whether the method truly achieves any form of geometric invariance. Renaming to something like "normal-regularized pointmap" would more accurately reflect the mechanism.

- **Two-stage training necessity is claimed but not empirically validated**: The paper asserts that "jointly training the pointmap and normal at the initial scale-invariant stage leads to instability and poor convergence" (Appendix A.1), motivating the two-stage design. However, no empirical evidence for this claim is provided—no failed single-stage training curve, no convergence comparison, no ablation showing a single-stage baseline underperforming. Table 3 ablates the intrinsic-invariant stage (removing it degrades normals from 62.5 to 50.6 δ11.25 on NYUv2), but this only shows that removing Stage 2 hurts, not that a properly designed single-stage multi-task training would also fail. Without this evidence, the two-stage strategy appears to be an engineering choice rather than a principled necessity, weakening the methodological contribution.

### Minor:

- **Depth results are not uniformly superior, and quantitative comparison is relegated to the appendix**: Table 7 shows MoGe outperforms Dens3R on NYUv2 depth (REL 0.035 vs. 0.042, RMSE 0.167 vs. 0.189) and on DIODE-outdoor δ1 (72.8 vs. 72.2). The main paper (Section 4.2) only presents qualitative depth comparisons (Fig. 5), creating an incomplete picture. The paper's claim of "superior performance across various 3D prediction benchmarks" in the conclusion should be qualified.

- **Position-interpolated RoPE contribution is confounded with coarse-to-fine training**: The paper claims RoPE interpolation enables stable high-resolution inference, but the fine stage also trains directly on 1024-resolution images (Section 3.2). Figure 22 partially addresses this by showing RoPE alone (without high-res training) is insufficient, but the reverse—training at 1024 without RoPE interpolation—is not tested. Without this ablation, it remains unclear whether RoPE interpolation is necessary or whether simply training at high resolution would suffice.

- **Multi-view reconstruction claims lack quantitative evaluation**: Section 3.3 and Figure 9 showcase multi-view reconstruction results using the MASt3R-SfM pipeline, but only qualitative visualizations are provided. Standard multi-view benchmarks (e.g., DTU, Tanks and Temples) with Chamfer distance or F-score metrics would substantiate the geometric consistency claims.

- **Bidirectional pointmap–normal benefit is claimed but not isolated**: The paper states "the pointmap–normal interaction as a bidirectional mechanism" (Appendix A.1), and Figure 11 shows Stage 2 pointmaps improve over Stage 1. However, no ablation isolates the two directions: (a) training normals with frozen pointmap features (does multi-view info help normals?) and (b) training pointmaps with frozen normal predictions (do normals refine pointmaps?). Without these, the claimed bidirectionality is an assertion rather than a demonstrated finding.

### Trivial:

- **Inference latency not reported for baselines**: Table 4 reports internal parameter and memory comparisons but no wall-clock inference time comparison against DUSt3R, MASt3R, or VGGT on identical hardware, making it difficult to assess real-world efficiency claims.

## Nice-to-Haves

- **Systematic resolution scaling curves**: Report quantitative metrics at 512, 768, 1024, and 2048 resolutions to precisely characterize the stability enabled by position-interpolated RoPE, rather than relying solely on qualitative figures.

- **Feature probing of the intrinsic-invariant representation**: Validate the core hypothesis that Stage 2 produces representations encoding different geometric properties than Stage 1, e.g., via linear probing or t-SNE visualization of intermediate features.

- **Cross-dataset generalization (synthetic-only → real)**: Train on Type A/B synthetic data only and test on real-world benchmarks without fine-tuning to evaluate true transfer capability as a foundation model.

- **Broader failure mode analysis**: Beyond thin structures (Fig. 12), systematically evaluate on textureless regions, transparent/specular surfaces, and extreme viewpoint changes to define operational limits.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **[Harsh Critic] "Foundation model" claim is overblown due to modest training scale**: The paper demonstrates multi-task prediction, downstream transfer to segmentation and reconstruction, and a shared backbone architecture. Whether the training scale qualifies as "foundation model" level is subjective; the paper's usage aligns with common practice in the 3D vision community for multi-task pretrained models.

- **[Harsh Critic] "Feed-forward" characterization contradicts post-processing pipeline**: The feed-forward claim clearly refers to the backbone's single-forward-pass inference for single/pair inputs. The multi-view post-processing (Section 3.3) is a separate inference-time extension, not a contradiction.

- **[Harsh Critic] Broader impact / dual-use concerns missing**: ICLR does not mandate broader impact sections. This is a standard vision algorithm paper; flagging surveillance or deepfake risks is speculative and not a core flaw.

- **[Harsh Critic] Parser artifact complaints about Equation 3**: Formatting artifacts from PDF extraction are explicitly excluded from review per instructions.

- **[Harsh Critic] Missing confidence intervals / statistical significance**: Single-run evaluation is standard practice for large-scale 3D vision benchmarks; demanding confidence intervals is scope creep for this type of work.

- **[Spark Finder] Missing related works**: Per instructions, I cannot confirm existence of uncited works and should not flag missing citations.

- **[Spark Finder] Reproducibility concerns about undisclosed hyperparameters**: The paper provides loss weights (η₁=1.0, η₂=0.1, η₃=0.075, λ₁=1.0, λ₂=0.1, λ₃=1.0), training resolution, GPU count, and training duration. Demanding every implementation detail is impractical for this scale of work.

## Novel Insights

The most insightful observation across the reviews is the tension between what Dens3R *actually does* and how it is *framed*. The method's genuine contribution is demonstrating that surface normals—when used as a training signal with one-to-one mapping constraints—serve as an effective regularizer for pointmap representations, improving both normal accuracy and downstream depth/pointmap quality. However, the paper wraps this practical finding in theoretically loaded language ("intrinsic-invariant") that the method does not deliver on: the normals are view-space (not invariant), the pointmap representation gains come from multi-task regularization (not from enforcing a formal invariance property), and the two-stage design appears driven by training stability rather than a principled decomposition of invariant vs. non-invariant properties. Reframing the contribution around what is empirically validated—normals as a deterministic geometric regularizer that stabilizes multi-task dense prediction—would strengthen rather than weaken the paper, as it more honestly represents the mechanism and avoids inviting scrutiny on invariance claims the method doesn't make.

## Suggestions

1. **Explicitly state the inference protocol**: Add a sentence in Section 4.1 clarifying whether single images or image pairs are used at test time for each benchmark. If pairs are used, separate these results from monocular baselines or add a single-image ablation.

2. **Rename "intrinsic-invariant pointmap"**: Replace with "normal-regularized pointmap" or similar terminology that accurately describes the mechanism (normals as deterministic supervision for multi-task regularization) without implying geometric view-invariance that the method does not enforce.

3. **Add a single-stage training baseline**: Include at least one experiment attempting joint training of all heads from scratch (even if it fails or underperforms), to empirically justify the two-stage design rather than asserting it without evidence.

4. **Include quantitative depth results in the main paper**: Move Table 7 results to the main paper and qualify the "superior performance" claims to acknowledge benchmarks where MoGe outperforms.

5. **Add a resolution scaling ablation without RoPE interpolation**: Train at 1024 resolution with standard (non-interpolated) RoPE to isolate whether the interpolation or the high-res training data drives the high-resolution performance gains.

## Quality Assessment

- **Novelty**: Moderate. The core idea—using normals to regularize pointmap representations—is valuable and well-motivated, but the architectural lineage from DUSt3R/MASt3R is strong, and the position-interpolated RoPE is adapted from LLM techniques. The two-stage training strategy is practical but not rigorously shown to be necessary.

- **Technical soundness**: Mixed. The empirical results are strong and the method works well in practice, but the central "intrinsic-invariant" concept is terminologically misleading, and critical experimental details (inference protocol) remain ambiguous. The ablation coverage has notable gaps.

- **Empirical support**: Good for normal prediction and matching; weaker for depth (non-uniform improvements, relegated to appendix) and multi-view reconstruction (qualitative only). The missing inference protocol clarification is a significant gap that affects interpretability of the main normal prediction results.

- **Significance**: The work addresses an important problem—unified geometric prediction—and demonstrates practical gains. If the inference protocol concern is resolved favorably, this is a meaningful contribution to the DUSt3R ecosystem. However, the contribution is more incremental than the "foundation model" framing suggests.

- **Clarity**: The paper is generally well-organized with clear figures, but suffers from imprecise terminology ("intrinsic-invariant") and ambiguous experimental descriptions that undermine the clarity of the claims.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
