## Summary

VoMP proposes a feed-forward model for predicting volumetric mechanical property fields (Young's modulus E, Poisson's ratio ν, density ρ) from 3D objects across multiple representations. The method combines a MatVAE (trained on real-world material triplets) to ensure physically valid outputs with a Geometry Transformer that aggregates multi-view DINOv2 features to predict per-voxel material latents. The authors introduce an annotation pipeline using VLMs and material databases to create training data, achieving 5-100× speedup over optimization-based baselines.

## Strengths

- **Feed-forward efficiency**: Table 1 shows VoMP runs in 3.59s versus 1000+ seconds for NeRF2Physics and PUGS. This practical speedup enables scalable deployment for simulation workflows, addressing a genuine bottleneck.

- **Physical validity guarantee via MatVAE**: The latent space design (§3) ensures decoded material triplets fall within real-world material ranges. Figure 7 demonstrates smooth interpolation between valid materials, and Figure 6d shows VoMP outputs stay within MTD ranges while baselines often produce implausible values. This is a meaningful architectural contribution.

- **Volumetric prediction**: Unlike prior work that only predicts surface properties, VoMP voxelizes object interiors (§4.1, Figure 9). The qualitative example of correctly inferring dirt inside a flower pot demonstrates learned priors about common internal structures.

- **Representation-agnostic design**: The method accepts meshes, Gaussian splats, NeRFs, and SDFs through a common rendering-voxelization pipeline (§4.1). Figure 8 shows results across representations.

- **Benchmark contribution**: The GVM dataset (37M voxels across 1,624 objects) and MTD (100,562 real-world material triplets) are valuable resources for a field lacking standardized evaluation.

## Weaknesses

- **Evaluation uses VLM-generated ground truth**: Both training and test data are annotated by the same VLM pipeline (Qwen2.5-VL 72B). The paper validates against VLM judgments (Tables 2-4), not physically measured properties. This creates a circular evaluation: VoMP is trained to reproduce VLM annotations and evaluated on how well it reproduces VLM annotations. The only partially independent benchmark (ABO-500 mass estimation) shows mixed results (VoMP loses on MnRE-mass: 0.887 vs PUGS's 0.767). A validation set with physically measured material properties would significantly strengthen the claims of "physically accurate" prediction.

- **VLM annotation quality as performance ceiling**: Table 9 reports VLM annotation errors (log(E) = 0.0295, ν = 0.0426), which suggests the VLM introduces systematic error that training cannot overcome. The paper does not analyze how VLM annotation noise propagates to final predictions.

- **No quantitative evaluation across 3D representations**: Tables 2-4 evaluate only on meshes. The claim of representation agnosticism is supported only qualitatively (Figure 8, §A.2) without quantitative metrics comparing accuracy on Gaussian splats, NeRFs, or SDFs against a common ground truth.

- **Training data domain specificity**: The 1,624 training objects all come from NVIDIA professional asset packs (commercial, residential, vegetation, simready). These are high-quality assets with clean segmentation and realistic PBR textures—quite different from noisy real-world captures. No evaluation on real scanned objects or objects outside the NVIDIA asset distribution is provided.

- **Interior inference from surface features alone**: For fully occluded interior voxels, features are simply averaged projections from surface views (Eq. 3). The model must infer internal composition from surface appearance, which works for objects with predictable internal structure (hollow pots contain air/dirt) but is fundamentally underconstrained for objects with complex internal heterogeneity. The paper frames this as "predicting internal material composition" without adequately discussing this limitation.

## Nice-to-Haves

- Ablation over MatVAE latent dimensionality (why 2D rather than 3D or 4D?)

- Resolution-accuracy tradeoff analysis for voxel grid size

- Quantitative simulation validation: compare simulated deformations against real object videos

- Anisotropic material extension for wood, composites, textiles

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Critic claimed the "validity" metric is circular because VoMP is designed to output valid materials.* While technically true, this metric correctly demonstrates that the architectural constraint (MatVAE) is functioning as intended—it's not claiming prediction accuracy, just validity. The distinction is reasonably clear in context.

- *Critic objected to "physically accurate" claim in abstract.* The paper appropriately tempers this in the body, describing outputs as "physically valid" (falling within real material ranges) rather than "accurate" for specific objects.

- *Positive reviewer's concern about 64³ fixed resolution.* The paper mentions stochastic sampling with LN = 32,768 voxels for large objects (§4.2), which partially addresses this. The voxelization resolution question is covered in limitations (§7).

- *Spark finder requested "naive baseline that assigns most common material".* This would be informative but is not a standard baseline in this area, and the MatVAE + geometry transformer approach clearly provides meaningful signal beyond global priors (Figure 9 shows spatial variation).

## Novel Insights

The MatVAE latent space enables a form of physics-aware regularization that prevents the common failure mode of generative models predicting implausible material combinations (e.g., extremely high stiffness with near-zero density). The latent space smoothness correlates with physical behavior—Figure 13 shows that interpolating between materials produces corresponding changes in FEM simulation outcomes. This suggests the 2D latent space has learned meaningful physical axes, not just compressed the input space.

## Suggestions

- Validate on a small set of objects with physically measured material properties (e.g., standardized material samples), even if only for a few materials. This would directly test the "physically accurate" claim.

- Report results on out-of-distribution objects (photogrammetry scans, real-world NeRFs) to establish generalization beyond professional 3D assets.

- For cross-representation evaluation, create a common test set rendered/processed through each representation type with known ground truth, then report quantitative metrics for each.

- Discuss more explicitly the limitations of surface-only information for interior prediction, including failure cases for objects with unexpected internal structure.