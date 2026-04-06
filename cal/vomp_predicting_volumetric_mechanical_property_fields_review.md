=== CALIBRATION EXAMPLE 81 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the core contribution: predicting volumetric mechanical property fields. The abstract makes several strong claims: first feed-forward model, cross-representation generalization, physically valid outputs, and volumetric prediction. These are substantiated in the paper, though the "first" claims should be tempered in light of the concurrent Pixie (Le et al., 2025), which is also feed-forward, as noted in the text. The abstract accurately summarizes the method, contributions, and key results.

### Introduction & Motivation
The problem is well-motivated, highlighting the labor-intensive nature of setting up physics simulations and the lack of mechanical property annotations in 3D datasets. The contributions are clearly stated: (1) the first feed-forward model for volumetric mechanical property fields, (2) a material latent space (MatVAE), (3) an automatic data annotation pipeline and benchmark, and (4) thorough evaluation. The introduction effectively positions VoMP against prior work, distinguishing it from optimization-based methods, simulator-specific parameter predictors, and surface-only approaches. A minor concern is the repeated use of "first," which can be contentious with concurrent work (Pixie). However, the authors acknowledge Pixie and differentiate their focus on volumetric properties and physical validity.

### Method / Approach
**MatVAE (§3):** The design of a VAE for material triplets (E, ν, ρ) is novel and well-justified. The modifications (normalizing flow, total correlation penalty, capacity constraint) address specific challenges like heavy-tailed distributions and latent collapse. The ablation study (Table 7) validates these choices. However, the choice of a 2D latent space is not deeply motivated; while it enables visualization, it may limit expressiveness. The authors should discuss whether higher dimensionality could capture more complex material relationships without harming interpolation properties.

**Feature Aggregation & Geometry Transformer (§4):** The method for aggregating multi-view DINOv2 features into voxels is robust and representation-agnostic. A key strength is the explicit voxelization of object interiors, enabling true volumetric prediction. The Geometry Transformer, built on TRELLIS, is appropriate. The stochastic voxel sampling for large assets is a practical solution for variable-sized inputs. However, the use of nearest-neighbor interpolation to transfer voxel materials back to the original geometry (§4.2, G.1) is a simplification that may introduce artifacts, especially for high-frequency material boundaries. The authors should discuss potential errors from this approximation.

**Training Data Pipeline (§5):** The automatic annotation pipeline combining part-segmented 3D assets, material databases, textures, and a VLM is a significant contribution. It cleverly uses the VLM with rich context (renders, material names, real-world value ranges) to generate plausible material assignments. The resulting GVM dataset (37M annotated voxels) is a valuable resource. However, the assumption that each part has isotropic material is a notable limitation, as acknowledged in §7. For materials like wood or composites, this is inaccurate. The authors could discuss how this assumption might bias the model.

**Reproducibility:** The method is described in sufficient detail, with additional implementation specifics in the appendix. The release of code, models, and the benchmark dataset (except the vegetation subset) will facilitate reproduction.

### Experiments & Results
**Benchmarks & Metrics (§6.3):** The new GVM benchmark (4.9M point annotations) is a substantial contribution, far larger than prior datasets. The metrics (ALDE, ADE, ALRE, ARE) are standard and appropriate. The comparison with baselines (NeRF2Physics, PUGS, Phys4DGen*) is fair, though note that NeRF2Physics and PUGS do not predict Poisson's ratio, so the comparison is incomplete for ν. The authors properly convert hardness to Young's modulus for NeRF2Physics. The inclusion of an early comparison with Pixie is appropriate, but the results are limited (only validity metrics in Fig. 6d). A more comprehensive comparison with Pixie on the GVM benchmark would strengthen the paper, but given its concurrent nature, this is understandable.

**Quantitative Results:** VoMP significantly outperforms all baselines across all properties and metrics (Fig. 6b, Tables 2-4). The validity analysis (Fig. 6d) is a particularly strong evaluation, showing VoMP's predictions are much closer to real-world material ranges. The mass estimation on ABO-500 (Fig. 6c) shows competitive or better performance, though this is a proxy task.

**Speed:** The wall-clock comparison (Table 1) shows VoMP is 1-2 orders of magnitude faster than optimization-based baselines and even faster than the concurrent feed-forward Pixie. The breakdown indicates rendering and DINOv2 computation are the bottlenecks, suggesting further optimization is possible.

**Qualitative Evaluation & Simulation (§6.2):** The end-to-end simulations (Figs. 5, 8) are compelling and demonstrate practical utility. The ability to handle various representations (meshes, splats, SDFs, NeRFs) is well-illustrated. The supplemental video likely strengthens this.

**Ablations (§C):** The ablations (Tables 7, 8) are thorough and validate key design choices: MatVAE vs. vanilla VAE, DINOv2 features, the importance of log scaling, and ℓ2 loss. The ablation on image features shows DINOv2 and CLIP perform similarly when trained from scratch, but DINOv2 with TRELLIS initialization is best.

**Limitations of Evaluation:** The test set is a hold-out from the GVM dataset, which is generated via the same VLM pipeline as the training set. This may lead to optimistic performance if the VLM has systematic biases. Evaluating on a separate, manually annotated dataset (even if small) would provide stronger evidence of generalization. The authors do evaluate validity against real material ranges (MTD), which partially addresses this.

### Writing & Clarity
The paper is generally well-written and structured. The figures are informative. Some sections are dense (e.g., MatVAE objective in Eq. 2), but the appendix provides additional explanation. The method description is clear enough for an expert reader to follow. No major clarity issues impede understanding.

### Limitations & Broader Impact
The limitations section (§7) is candid: fixed-grid voxelization limits resolution and may oversmooth; the isotropic part-level assumption; and the scope limited to (E, ν, ρ). These are reasonable. Broader impact is not discussed explicitly, but the paper mentions applications in digital twins, robotics, and interactive content creation. Potential negative societal impacts (e.g., misuse for generating realistic but unsafe simulations) are not addressed but could be noted.

## Overall Assessment
This is a strong paper that makes significant contributions to the problem of automatic mechanical property estimation for 3D objects. The core ideas—a feed-forward model trained on a large, automatically annotated dataset, coupled with a material latent space ensuring physical validity—are novel and well-executed. The experiments are comprehensive, demonstrating clear improvements in accuracy, speed, and material realism over prior art. The new benchmark dataset is a valuable resource for the community. The main concerns are relatively minor: the "first" claims should be nuanced given concurrent work, the isotropic part assumption is a known limitation, and evaluation on a manually annotated test set would strengthen generalizability. Nonetheless, the paper meets ICLR's standards for novelty, technical rigor, and impact. It is likely to be influential in bridging 3D vision and physics-based simulation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces VoMP, a feed-forward method to predict volumetric mechanical property fields (Young’s modulus, Poisson’s ratio, and density) for 3D objects across multiple representations (e.g., meshes, Gaussian splats, NeRFs, SDFs). The core contributions are: (1) a material VAE (MatVAE) that learns a physically valid latent space of material triplets, (2) a Geometry Transformer that aggregates multi-view DINOv2 features within voxelized objects and predicts per-voxel material latents, (3) an automated data annotation pipeline combining segmented 3D assets, material databases, and a vision-language model, and (4) a new benchmark for volumetric physics materials. Experiments show VoMP outperforms prior art in accuracy and speed, enabling realistic elastodynamic simulations.

### Strengths
1. **Novel and practical contribution**: VoMP is the first feed-forward model for volumetric mechanical property estimation that generalizes across 3D representations and guarantees physically valid outputs via MatVAE. This significantly lowers the barrier for creating simulation-ready assets.
2. **Strong empirical evaluation**: The paper provides thorough qualitative simulations (Fig. 5, 8) and quantitative comparisons on both existing (mass estimation) and new benchmarks (Tables 1-4), demonstrating clear improvements over prior methods (NeRF2Physics, PUGS, Phys4DGen) in accuracy and run-time (100x speedup over optimization-based approaches).
3. **High-quality data and benchmark**: The authors introduce a large-scale dataset (GVM) with 37M voxels annotated via a carefully designed pipeline that leverages multiple knowledge sources (VLM, material databases, part segmentations). This dataset is a valuable resource for future research.
4. **Well-designed components**: The MatVAE latent space ensures physical validity and enables smooth interpolation (Fig. 7). The voxel feature aggregation and transformer architecture effectively capture interior material composition, addressing a key limitation of surface-focused prior work.

### Weaknesses
1. **Limitations acknowledged but not fully addressed**: The method assumes isotropic materials per part, which fails for anisotropic materials like wood. Fixed-grid voxelization can oversmooth fine details, and the output is limited to three properties (E, ν, ρ). While discussed, these limitations are not quantitatively analyzed (e.g., how anisotropy affects simulation fidelity).
2. **Dataset availability and potential noise**: The vegetation subset of GVM is withheld, limiting reproducibility. The annotation pipeline relies on a VLM and part segmentations, which may introduce label noise; the paper only provides a small manual validation (Table 9) without deeper analysis of error sources.
3. **Incomplete comparison with concurrent work**: The comparison to Pixie (Le et al., 2025) is limited due to its concurrent nature and unavailability. The analysis in Appendix B is preliminary and does not include quantitative results on the same benchmark.
4. **Evaluation gaps**: The method’s performance on highly heterogeneous objects (e.g., objects with intricate internal structures) is only qualitatively shown. There is no analysis of how voxel resolution affects accuracy, nor of failure cases (e.g., when the VLM annotation fails).

### Novelty & Significance
The paper presents several novel contributions: the first feed-forward model for volumetric mechanical property fields, the first latent space for material triplets (MatVAE), and a new benchmark. The work is significant for physics-based simulation, digital twins, and robotics, as it automates a labor-intensive step in simulation pipelines. The approach is representation-agnostic and produces physically valid parameters compatible with accurate simulators (FEM). The paper meets ICLR’s expectations for novelty and potential impact.

### Suggestions for Improvement
1. **Extend the method and analysis**: Address the isotropic assumption by discussing or experimenting with anisotropic materials. Explore higher-resolution outputs (e.g., adaptive voxelization) and extension to additional properties (yield strength, thermal expansion). Provide a quantitative analysis of how voxel resolution affects accuracy and simulation results.
2. **Improve dataset transparency and robustness**: Release the full dataset or provide a detailed description of the withheld vegetation subset. Conduct a deeper analysis of annotation noise (e.g., per-material error statistics) and its impact on model performance. Consider ablating the VLM component to understand its contribution.
3. **Strengthen comparisons and evaluation**: When possible, include a full quantitative comparison with Pixie on the same benchmark. Evaluate on more challenging cases (e.g., objects with complex internal heterogeneity) and report failure modes. Consider a user study or simulation-based metric beyond property error (e.g., trajectory similarity in dynamic scenarios).
4. **Enhance reproducibility**: Provide explicit pseudocode or code for the Gaussian splat voxelizer (Section 6.1) and the feature aggregation pipeline. Clarify the training details (e.g., how many views are rendered, camera distribution) and computational requirements for training and inference.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to stronger or more relevant baselines is missing.** The paper primarily compares against NeRF2Physics, PUGS, and Phys4DGen. It should include comparisons to more recent concurrent works (like SOPHY, PhysX-3D, and especially Pixie) using their official implementations and metrics to properly establish state-of-the-art. The current early comparison to Pixie is insufficient.
2. **Evaluation on a broader set of 3D representations is incomplete.** The claim of generalization across representations is supported only by a few qualitative examples (Fig. 8a). A quantitative ablation study measuring performance degradation across mesh, splat, NeRF, and SDF inputs on a shared benchmark is necessary to substantiate this claim.
3. **No end-to-end simulation error metric is provided.** The paper evaluates material property errors and shows qualitative simulations, but lacks a quantitative metric (e.g., trajectory error, final state difference) comparing simulations run with predicted properties versus ground-truth or plausible proxy properties. This is critical for the core claim of producing "simulation-ready" properties.

### Deeper Analysis Needed (top 3-5 only)
1. **Error analysis across material categories and object classes is absent.** The average errors reported hide performance on critical subsets. The method likely performs poorly on materials with complex internal structures (e.g., anisotropic wood, composites) or on novel object categories not well-represented in the training data. A breakdown of errors is needed to understand limitations.
2. **The contribution of the VLM-based annotation pipeline is not isolated.** The training data is created using a pipeline combining segmented 3D data, material databases, and a VLM. An ablation study quantifying the impact of each component (especially the VLM) on final model accuracy is missing. Without it, it's unclear if the complex pipeline is necessary.
3. **The claim of "physically valid" properties lacks rigorous analysis.** While MatVAE ensures outputs are within the convex hull of the training data, there is no analysis of whether the *spatial distribution* of properties (e.g., gradations inside an object) is physically plausible or just a smoothed average of part labels.

### Visualizations & Case Studies
1. **Side-by-side comparative simulations are needed.** To convincingly show that the predicted properties are accurate, the paper should include visualizations (e.g., video frames) of the same simulation scenario using ground-truth (or expert-assigned) properties versus VoMP-predicted properties, highlighting differences.
2. **Cross-sectional visualizations of internal property fields are insufficient.** Figure 9 shows a few slices, but systematic visualizations for complex, multi-material objects (e.g., a chair with cushion, frame, and screws) are needed to demonstrate that internal boundaries and heterogeneous materials are captured correctly, not just smoothed.

### Obvious Next Steps
1. **Validation on real-world objects with measured properties is essential.** The entire evaluation is on a synthetic benchmark (GVM) created via the authors' pipeline. To prove real-world utility, the model must be tested on a set of real objects (e.g., 3D scans) where mechanical properties have been physically measured at several locations.
2. **Portability across simulators should be demonstrated quantitatively.** The paper argues that real-world parameters are portable (Fig. 2) but only shows one qualitative example. A quantitative experiment running the same VoMP-annotated object in multiple high-fidelity simulators (e.g., FEM, MPM) and comparing results would strongly support the claim of simulator-agnostic outputs.

# Final Consolidated Review
## Summary
VoMP is a feed-forward model that predicts volumetric mechanical property fields (Young's modulus, Poisson's ratio, density) for 3D objects across representations like meshes, Gaussian splats, NeRFs, and SDFs. It introduces a material VAE (MatVAE) to ensure physical validity, a transformer-based architecture for per-voxel prediction, and a new benchmark dataset generated via an automated pipeline combining part segmentations, material databases, and a vision-language model.

## Strengths
- The model is feed-forward and achieves a dramatic speed-up over optimization-based prior art (e.g., 3.59 seconds per object vs. minutes for NeRF2Physics or PUGS), making it practical for integration into simulation workflows (Table 1).
- MatVAE learns a compact latent space of material triplets that guarantees physically plausible outputs and enables smooth, valid interpolation—a novel contribution in this domain (Fig. 7, ablation in Table 7).
- The automatic annotation pipeline produces a large-scale benchmark (GVM) with 37 million voxels annotated, which is released to the community and significantly advances the field by providing high-quality training and evaluation data (§5, Appendix E).

## Weaknesses
- The method assumes isotropic materials per segmented part, which is inaccurate for common anisotropic materials like wood; while acknowledged, this limitation is not quantitatively assessed and may degrade prediction fidelity for many real-world objects.
- Fixed-grid voxelization bounds spatial resolution and can oversmooth fine or heterogeneous material regions; the paper does not analyze how voxel size affects prediction error or simulation outcomes, leaving the precision limits unclear.
- Evaluation relies heavily on the synthetic GVM benchmark generated via the same VLM pipeline used for training, which risks circular bias; validation against physically measured properties on real objects is limited to proximity to material ranges rather than spatial accuracy of the fields.

## Nice-to-Haves
- Error breakdown by material category (e.g., metals, plastics, woods) or object class to identify where the model excels or struggles.
- Quantitative simulation error metrics (e.g., trajectory differences) comparing dynamics using predicted versus ground-truth properties.
- Ablation study isolating the contribution of the VLM component in the data annotation pipeline to understand its necessity.

## Novel Insights
The paper demonstrates that a low-dimensional latent space of material triplets can be learned to enforce physical validity and enable smooth interpolation, and that multi-view visual features aggregated over voxelized object interiors are sufficient for a feed-forward transformer to infer spatially-varying mechanical properties. This shows that fast, representation-agnostic augmentation of 3D assets with simulation-ready parameters is feasible without per-object optimization.

## Suggestions
- Conduct a sensitivity analysis of prediction accuracy and simulation fidelity to voxel grid resolution, and explore adaptive or hierarchical voxelization to preserve fine details.
- Release the vegetation subset of the GVM dataset or provide a detailed synthetic alternative to ensure full reproducibility and benchmarking.
- When feasible, perform a comprehensive quantitative comparison with concurrent feed-forward methods like Pixie on the same benchmarks to clearly establish state-of-the-art.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 8.0]
Average score: 7.0
Binary outcome: Accept
