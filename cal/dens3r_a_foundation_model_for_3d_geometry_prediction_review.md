=== CALIBRATION EXAMPLE 63 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the scope. The abstract summarizes the problem, approach, and claims, but it does not sharply differentiate Dens3R from prior foundation models like DUSt3R and MASt3R. The claimed novelty of "intrinsic-invariant pointmap" and two-stage training is mentioned, but why these are necessary beyond existing affine-invariant or scale-invariant representations remains vague. Abstract claims of "high-quality performance" are supported later, but the abstract itself lacks concrete quantitative highlights.

### Introduction & Motivation
The introduction effectively motivates unified geometric prediction and identifies a gap: existing regression models (DUSt3R, MASt3R) do not jointly predict surface normals, which could improve consistency. However, the argument that normals provide "intrinsic invariance" and simplify training is not sufficiently developed. The introduction states that prior methods "overlook a crucial geometric information—surface normals," but does not explain why adding a normal head is non-trivial or why previous architectures cannot easily incorporate it. The contributions are listed but their incremental nature over MASt3R (which already uses a dense transformer and pointmap losses) is not critically addressed.

### Method / Approach
This section contains significant ambiguities that hinder reproducibility and understanding.

**Architecture details:** The shared encoder-decoder backbone is described only at a high level. Critical details are missing: transformer configuration (number of layers, heads, feature dimensions), how weight-sharing is implemented across decoders, and the exact form of the "lightweight" DPT heads. Without these, replication is difficult.

**Position-interpolated RoPE:** Equation (2) is presented, but its integration into the ViT is not explained. How are 2D image positions mapped to sequence indices? The claim that this "significantly enhances robustness" is not ablated in the main paper; only appendix figures show benefits.

**Two-stage training:** 
- Stage 1 losses are adopted from MASt3R, but the derivation of ground-truth normals for \(L_{pts\_n}\) is unclear (are they computed from ground-truth point clouds?). The transformation of normals between camera frames (for \(N^{v,t}\)) is not specified.
- Stage 2 introduces "intrinsic-invariant pointmap." The transition from scale-invariant to intrinsic-invariant is poorly defined. Equation (9) simply concatenates pointmap and normal features; this does not inherently impose invariance. The text mentions switching from "one-to-many" to "one-to-one" mapping, but the training objective \(L_{stage2}\) still includes the global loss \(L_{pts\_glb}\), which requires multiple views. This contradiction needs clarification. The claim that confidence loss is removed due to normal deterministicism is interesting but not justified theoretically or empirically.

**Multi-view inference:** The post-processing pipeline is described only as "constructing and optimizing a dense correspondence network," referencing MASt3R. This is too vague; the novel components of Dens3R’s multi-view handling are not specified.

Overall, the method raises more questions than it answers. The core innovations (two-stage training, intrinsic-invariant representation) are not rigorously derived or justified, and the training procedure lacks clarity.

### Experiments & Results
The experimental section has notable weaknesses:

**Normal prediction:** Table 1 shows strong results, but the evaluation protocol is unclear. Are comparison methods trained on the same data? For fairness, methods like DSINE or StableNormal might be trained on different datasets; the paper should specify if comparisons are zero-shot or fine-tuned. Qualitative results (Fig. 4) are convincing but lack quantitative depth.

**Depth and pointmap evaluation:** Quantitative depth results are only in the appendix (Table 7). Pointmap quality is assessed only qualitatively (Fig. 5). For a foundation model claiming high-fidelity 3D geometry, quantitative reconstruction metrics on standard benchmarks (e.g., ScanNet, DTU) are essential and missing.

**Ablation studies:** Critical ablations (position-interpolated RoPE, intrinsic-invariant training, coarse-to-fine strategy) are relegated to the appendix (Tables 3, 6, and figures). These should be in the main paper to validate design choices. The ablation on shared encoder-decoder (Table 4) only reports memory and parameters, not performance impact.

**Training data:** The dataset composition (Table 5) is extensive but arbitrary. The split into types A/B/C and the chosen ratios are not justified or ablated. The model's performance likely depends heavily on this mix, but no analysis is provided.

**Computational cost:** Training requires 32 H20 GPUs for two weeks, which is substantial but expected for a foundation model. However, inference efficiency compared to DUSt3R/MASt3R is not discussed.

**Downstream applications:** Segmentation and surface reconstruction are shown only in appendix figures, with no quantitative evaluation. The claim of being a "versatile backbone" is under-supported.

### Writing & Clarity
The writing is generally clear, but the method section is confusing. Terms like "intrinsic-invariant pointmap" are used without precise definition. The two-stage training procedure is described in a way that makes it difficult to understand what changes between stages. Figures 2 and 3 help, but more detailed diagrams of architecture and training flow would improve clarity.

### Limitations & Broader Impact
Limitations are briefly mentioned in the appendix (thin structures), but other critical limitations—such as performance on transparent/reflective surfaces, generalization to entirely unseen domains, dependence on large-scale synthetic data, and environmental impact of training—are not discussed. A broader impact statement is missing, which is expected for ICLR submissions.

## Overall Assessment
Dens3R proposes a unified foundation model for dense 3D geometry prediction, integrating surface normals into a pointmap representation via a two-stage training strategy. The idea is promising, and experimental results show strong performance on normal estimation and matching. However, the paper suffers from significant methodological ambiguities, insufficient quantitative evaluation (especially for 3D reconstruction), and key ablations relegated to the appendix. The novelty over the DUSt3R/MASt3R lineage is incremental, and the core concept of "intrinsic-invariant pointmap" is not clearly defined or justified. For ICLR, where technical rigor, clarity, and empirical thoroughness are paramount, the current version falls short. Major revisions are needed to clarify the method, provide comprehensive quantitative results, and situate the contributions more distinctly relative to prior work. Without these, the paper is unlikely to meet the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Dens3R, a foundation model for dense 3D geometric prediction from unposed images. Its core contribution is a unified framework that jointly regresses multiple geometric quantities—pointmaps, depth, surface normals, and image-pair matching features—through a two-stage training strategy. The first stage learns a scale-invariant pointmap, while the second refines it into an "intrinsic-invariant" representation by incorporating surface normal supervision to resolve monocular ambiguities and improve consistency.

### Strengths
1. **Unified Multi-Task Prediction**: Dens3R successfully predicts multiple correlated geometric outputs (depth, normals, pointmaps, matching) in a single forward pass, moving beyond prior works that typically focus on one or two tasks. This is evidenced by state-of-the-art or competitive results across diverse benchmarks (Tab. 1, 2, 7).
2. **Effective Training Strategy**: The proposed two-stage training (scale-invariant → intrinsic-invariant pointmap) and the integration of normal supervision are well-motivated. The paper shows qualitatively (Fig. 3, 4) and quantitatively (Tab. 3, 6) that this leads to more accurate normals and refined geometry compared to training without this stage.
3. **Scalability and Practical Design**: The introduction of position-interpolated rotary positional encoding (RoPE) effectively mitigates performance degradation on high-resolution inputs (Fig. 8a, 21). The shared encoder-decoder backbone reduces parameters and memory cost (Tab. 4), supporting multi-view and multi-resolution inference.

### Weaknesses
1. **Incremental Novelty**: The core architecture and loss functions are heavily based on the DUSt3R/MASt3R lineage. The primary novelty—the intrinsic-invariant pointmap and two-stage normal integration—feels like a natural, incremental extension rather than a foundational shift. The position-interpolated RoPE is a direct adaptation from LLMs to vision.
2. **Insufficient Ablation and Analysis**: While components are presented, the paper lacks thorough ablations to disentangle the contribution of each. For instance, the impact of the shared decoder vs. separate decoders, the sensitivity to loss weights (η, λ), and the necessity of the two-stage setup versus joint training are not rigorously studied. The claim that normal prediction "simplifies" the training of other quantities is not quantitatively verified.
3. **Reproducibility Concerns**: The training relies on a massive, meticulously curated mixture of 30+ datasets divided into quality tiers (Tab. 5). The exact data mixing ratios, preprocessing, and the "quality" classification criteria are not fully specified, making exact replication challenging. Training for two weeks on 32 H20 GPUs is also a high resource barrier.

### Novelty & Significance
**Novelty**: Moderate. The work synthesizes ideas from DUSt3R (pointmap regression), MoGe (affine-invariance), and normal-estimation literature into a unified multi-task model. The intrinsic-invariant pointmap formulation and the two-stage normal incorporation are the main novel conceptual contributions. The position-interpolated RoPE application to vision is a sensible but not groundbreaking technical adaptation.

**Significance**: High. Unified 3D geometry prediction is a critical and active area. Demonstrating that jointly modeling multiple geometric properties improves performance and consistency is a valuable finding. The model's strong performance and flexibility as a backbone for downstream tasks (segmentation, reconstruction) align well with ICLR's interest in foundational models.

### Suggestions for Improvement
1. **Conduct Comprehensive Ablation Studies**: Isolate and quantify the contribution of each key component: the intrinsic-invariant training stage, the normal loss, the shared decoder, and the position-interpolated RoPE. A table showing performance gains at each stage would strengthen the claims.
2. **Strengthen the Novelty Narrative**: Clearly differentiate from VGGT (which also predicts multiple quantities) and MoGe. A deeper discussion on the theoretical or empirical advantages of the "intrinsic-invariant" pointmap over prior invariant representations would elevate the contribution.
3. **Improve Reproducibility and Efficiency Reporting**: Provide more details on dataset curation (e.g., scripts for quality tier classification). Include a clear summary of computational costs (FLOPs, memory, inference time) compared to key baselines like DUSt3R and VGGT, which is crucial for assessing the model's practical utility.
4. **Deepen Analysis on Limitations**: The mentioned failure on thin structures (Fig. 12) warrants further investigation. Analyzing whether this is due to network architecture, training data bias, or the representation itself would provide valuable insights for future work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on multi-task joint training vs. independent baselines.** The core claim is unified, consistent prediction. However, there is no direct comparison against an ensemble of independent state-of-the-art single-task models (e.g., a top depth model + a top normal model) or a multi-task baseline like VGGT trained under the same conditions. Without this, the benefit of the proposed joint framework over simply running separate models is not quantified.
2. **Quantitative evaluation of multi-view inference post-processing.** The "geometrically consistent multi-view inference" pipeline is a claimed contribution. Yet, there are no quantitative results (e.g., reconstruction accuracy, pose error, consistency metrics) on standard multi-view datasets (e.g., DTU, Tanks & Temples) comparing Dens3R's full pipeline against DUSt3R/MASt3R's. Only mentioning it follows MASt3R's pipeline undermines its novelty and impact.
3. **Systematic high-resolution ablation.** The paper claims high-resolution robustness via position-interpolated RoPE and coarse-to-fine training. However, quantitative metrics (e.g., depth RMSE, normal error) on standard benchmarks at multiple resolutions (e.g., 512, 1024, 2048) are missing. Figure 8a only shows pointmaps without error metrics, making the claim of "preventing performance degradation" unsupported.
4. **Component-wise ablation study.** The model introduces several components (shared encoder-decoder, two-stage training, intrinsic-invariant pointmap). A comprehensive ablation table is missing that shows the individual contribution of each component to final performance across all tasks (pointmap, depth, normal, matching). The provided ablations (Tab 3, Fig 8b) are incomplete and lack depth/depth comparisons.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of "intrinsic invariance" and its effect.** The paper claims normals provide "intrinsic invariance" that simplifies learning. There is no analysis (e.g., feature space visualization, gradient norms, training dynamics curves) to demonstrate *how* adding normals leads to better, more stable convergence or disentangles representations compared to Stage 1 training alone.
2. **Failure mode and generalization analysis.** The paper shows success cases but lacks a systematic analysis of where and why Dens3R fails (beyond the brief "thin structures" mention in the appendix). A study on challenging categories (e.g., textureless regions, specular surfaces, dynamic objects) across different datasets is needed to understand the model's real-world limits.
3. **Computational efficiency and scaling analysis.** As a "foundation model," its practical utility depends on efficiency. There is no analysis of inference speed, memory footprint vs. input resolution/number of views, or parameter count comparison against simpler baselines. Table 4 is insufficient as it only compares shared vs. non-shared decoder within their own architecture.

### Visualizations & Case Studies
1. **Side-by-side visualization of all predicted quantities for the same scene.** To validate "unified" and "consistent" prediction, the paper needs visualizations showing input image, predicted depth, predicted normal, *and* the derived pointmap in a single, aligned view. Current figures (e.g., Fig 5, 7) show depth and pointmaps separately, making it impossible to judge their geometric consistency.
2. **Visualization of failure cases and error maps.** To build trust, the paper should include visual examples where predictions are poor (e.g., on thin structures, transparency) with per-pixel error heatmaps for depth and normals, contrasting them with successes.
3. **Visualization of cross-view correspondence.** For the matching task, showing dense correspondence fields or matched points between challenging image pairs (with large viewpoint/illumination change) would be more convincing than just reporting AUC scores.

### Obvious Next Steps
1. **Establish a true multi-task benchmark.** The authors should have created a consolidated benchmark evaluating all claimed tasks (depth, normal, pointmap, matching, pose) on a common set of scenes/datasets to demonstrate comprehensive superiority over prior "single-task" or "few-task" foundations like DUSt3R, MASt3R, and VGGT.
2. **Perform cross-dataset generalization tests.** The model is trained on a massive blended dataset. A critical next step is to evaluate zero-shot or few-shot performance on completely held-out datasets not in their training mix (e.g., specialized medical or aerial imagery) to test its true foundational capabilities.
3. **Provide interpretability/analysis of the learned pointmap representation.** A key scientific question is what the "intrinsic-invariant pointmap" actually represents. An analysis (e.g., via probing tasks, clustering) of its feature space compared to scale-invariant or affine-invariant representations would strengthen the methodological contribution.

# Final Consolidated Review
## Summary
Dens3R is a foundation model that jointly predicts multiple 3D geometric quantities—pointmaps, depth, surface normals, and image-pair matching—from unposed images. Its core contribution is a two-stage training strategy that first learns a scale-invariant pointmap and then refines it into an "intrinsic-invariant" representation by incorporating surface normal supervision, aiming to improve geometric consistency and accuracy.

## Strengths
- **Effective unified multi-task prediction:** Dens3R demonstrates strong, often state-of-the-art, performance across multiple benchmarks for normal estimation (Table 1, 6), image matching (Table 2), and depth estimation (Table 7), validating its ability to jointly regress correlated geometric properties in a single forward pass.
- **Well-motivated training strategy:** The two-stage training, which integrates normal supervision to create an intrinsic-invariant pointmap, is shown qualitatively (Fig. 3, 4) and quantitatively (Table 3) to improve normal accuracy and refine the underlying geometric representation compared to the initial scale-invariant stage.

## Weaknesses
- **Insufficient quantitative evaluation of core claims:** The paper lacks quantitative metrics for several key contributions. Most critically, there is no quantitative evaluation of the multi-view inference pipeline (e.g., reconstruction accuracy on standard multi-view datasets) to substantiate the claim of "geometrically consistent multi-view inference." Similarly, the high-resolution robustness claim via position-interpolated RoPE is supported only by qualitative pointmaps (Fig. 8a, 21) without resolution-dependent error metrics.
- **Incomplete ablation and component analysis:** While some ablations are provided in the appendix, a comprehensive component-wise ablation study is missing. The individual contributions of the intrinsic-invariant training stage, the shared decoder design, and the coarse-to-fine strategy to final performance across all predicted tasks (depth, normal, pointmap, matching) are not quantified, making it difficult to assess the necessity of each design choice.

## Nice-to-Haves
- A direct comparison against an ensemble of independent state-of-the-art single-task models would better quantify the benefit of the unified framework.
- An analysis of the learned "intrinsic-invariant" representation (e.g., via feature space probing) could provide deeper insight into how normal supervision shapes the pointmap.

## Novel Insights
The paper's key insight is that surface normals provide a form of intrinsic geometric invariance. By explicitly modeling normals and using them to anchor the pointmap representation, the model mitigates monocular ambiguity and improves the consistency of other predicted quantities like depth. This demonstrates that jointly regressing correlated geometric properties can be more effective than estimating them in isolation.

## Suggestions
- Provide quantitative results on standard multi-view reconstruction benchmarks (e.g., DTU) using the proposed multi-view inference pipeline to validate its geometric consistency.
- Conduct a full ablation study in the main paper, presenting a table that quantifies the performance impact of each major component (two-stage training, normal loss, shared decoder, position-interpolated RoPE) across all evaluation tasks.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
