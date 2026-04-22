# CylinderSplat: 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Feed-forward 3D Gaussian Splatting (3DGS) has shown great promise for real-time novel view synthesis, but its application to panoramic imagery remains challenging. Existing methods often rely on multi-view cost volumes for geometric refinement, which struggle to resolve occlusions in sparse-view scenarios. Furthermore, standard volumetric representations like Cartesian Triplanes are poor in capturing the inherent geometry of $360^\circ$ scenes, leading to distortion and aliasing.

In this work, we introduce CylinderSplat, a feed-forward framework for panoramic 3DGS that addresses these limitations. The core of our method is a new {cylindrical Triplane} representation, which is better aligned with panoramic data and real-world structures adhering to the Manhattan-world assumption. We use a dual-branch architecture: a pixel-based branch reconstructs well-observed regions, while a volume-based branch leverages the cylindrical Triplane to complete occluded or sparsely-viewed areas. Our framework is designed to flexibly handle a variable number of input views, from single to multiple panoramas. Extensive experiments demonstrate that CylinderSplat achieves state-of-the-art results in both single-view and multi-view panoramic novel view synthesis, outperforming previous methods in both reconstruction quality and geometric accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes CylinderSplat, a novel feed-forward framework for panoramic 3D Gaussian Splatting. The key idea is to replace standard Cartesian triplanes with cylindrical triplanes, which are geometrically aligned with panoramic imagery and Manhattan-world structures. The method employs a dual-branch design: a pixel branch for reconstructing well-observed regions and a volume branch using cylindrical triplanes for hallucinating geometry in occluded areas. Extensive experiments across multiple panoramic datasets demonstrate state-of-the-art performance in both single- and multi-view settings, with particularly strong results in challenging sparse-view scenarios. Rich ablations validate the architectural decisions, including the cylindrical coordinate choice, RGB retrieval mechanism, and staged training strategy.

### Strengths
1. Clear novelty and motivation: Introducing cylindrical triplanes for panoramic 3DGS is both intuitive and technically meaningful, improving geometric alignment and addressing limitations of Cartesian/spherical triplanes.

2. Strong empirical results: The method consistently outperforms existing feed-forward panoramic reconstruction approaches in both quantitative metrics and visual quality, including single-view settings.

3. Comprehensive experimental suite: Experiments span multiple synthetic and real-world datasets, various view counts, and a detailed ablation study, increasing confidence in the robustness and generalizability.

4. Solid architectural design: The dual-branch structure and curriculum training pipeline are conceptually sound and empirically effective, enabling accurate pixel-aligned estimation and robust geometry completion.

5. Efficient and scalable: Avoiding cost-volume construction improves efficiency, and the model flexibly supports arbitrary numbers of input panoramas.

### Weaknesses
Limited discussion on memory efficiency: Although triplane is used for compression, the method still constructs multiple local triplanes and can scale in memory with view count; a clearer complexity breakdown would help.

### Questions
1. Fusion strategy: Could more advanced Gaussian merging (e.g., density clustering, depth-guided pruning) further improve seamless blending between branches? Have such strategies been experimented with?

2. Non-Manhattan-world scenes: How does performance degrade in highly unstructured outdoor scenes (e.g., forests, caves)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CylinderSplat, a feed-forward framework for panoramic novel view synthesis that marries a pixel branch (attention-based multi-view aggregation predicting per-pixel Gaussians without cost volumes) with a volume branch that completes occluded/sparse regions using a new cylindrical Triplane representation. The key contribution is replacing standard Cartesian triplane representations with cylindrical triplanes, motivated by the Manhattan-world assumption common in indoor/urban scenes. Features from both are trained in a three-stage curriculum and fused for direct equirectangular 3DGS rendering with an RGB-retrieval step for high-frequency detail. Unlike prior panoramic feed-forward methods that depend on multi-view cost volumes or Cartesian triplanes, the cylindrical triplane reduces distortion/aliasing and enables flexible inputs from single to multiple panoramas. Across Matterport3D, Replica, Residential, and 360Loc, CylinderSplat shows state-of-the-art image quality. Ablations show the benefit of the cylindrical triplane. The method is lightweight and fast (≈11.3M params; ~0.29s inference).

### Strengths
1. The cylindrical coordinate system choice is well-justified through both theoretical analysis (Manhattan-world assumption) and comprehensive ablations showing clear advantages over Cartesian and spherical alternatives.
2. Extensive experiments are shown across multiple datasets with thorough ablations examining triplane resolution, coordinate systems, initialization strategies, and rendering methods. Consistent improvements are shown in both single-view and multi-view settings, with particularly notable gains in geometric accuracy.
3. Unlike prior cost-volume methods, the attention-based pixel branch handles variable numbers of input views (1 to multiple) without retraining.
4. Good implementation details, clear mathematical formulations, and promise of code release for reproducibility.

### Weaknesses
1. Limited novelty: The core contribution is essentially a coordinate system change for existing triplane methods. The dual-branch architecture, attention mechanisms etc. seem to be borrowed from prior work.
2. Manhattan-world assumption limitations: The method seems to be explicitly designed for environments with orthogonal surfaces (indoor/urban scenes). Applicability to natural outdoor scenes, curved structures, or non-Manhattan environments is unclear
3. Training complexity: The three-stage curriculum (pixel branch → volume branch → joint fine-tuning) suggests potential brittleness: Table 4 shows end-to-end training is inferior.

### Questions
1. What are some of the scenes/settings where the proposed method is likely to fail?
2. How does performance degrade with larger baselines between input views (>2m)?
3. What is the training time comparison with baselines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CylinderSplat, a feed‑forward 3D Gaussian Splatting framework for 360 panoramas. The system has two branches: (i) a pixel branch that uses attention to aggregate multi‑view features and predict Gaussians for well‑observed pixels, and (ii) a volume branch that introduces per‑camera cylindrical triplanes to complete occlusions and sparsely observed regions. A three‑stage curriculum trains pixel, then volume, next joint fine‑tuning. Color for volume‑branch Gaussians is obtained by RGB retrieval from input panoramas with a simple visibility weighting derived from a depth prior. The paper also implements direct equirectangular 3DGS rasterization without doing cubemap rasterization.

### Strengths
- a practical pipeline that avoids heavy cost‑volumes while supporting a variable number of input panoramas via attention.

- Cylindrical triplane is a sensible geometric choice for Manhattan‑style scenes; the paper gives qualitative intuition and ablations showing it outperforms Cartesian and spherical alternatives in this setting.

- The paper measures the effect of each key component: coordinate system, RGB retrieval, multiple per‑camera triplanes, and training curriculum.

- Results are broadly stronger on synthetic datasets, and the method scales from single‑view to multi‑view inputs

### Weaknesses
- While the cylindrical triplane is well‑motivated, it can be viewed as a coordinate adaptation of established triplane+3DGS paradigms, rather than a new representation class or learning principle. Much of the performance gain appears to come from engineering side and somehow lacks of technical novelty.

- The transformation from cylindrical local scales to Cartesian scales is kind of approximation without derivation—raising concerns about correctness and whether R and S are consistently transformed.

- PCC is computed against DepthAnywhere (no GT), while UniK3D is used for depth supervision/priors and visibility weighting. Although the models differ, this model‑to‑model comparison weakens claims of geometric accuracy and could bias design choices toward correlating with a particular depth estimator.

- On 360Loc two‑view, gains over a strong panoramic baseline are small (e.g., WS‑PSNR 28.35 vs. 28.14; LPIPS 0.095 vs. 0.127), which tempers the “SOTA” claim outside synthetic domains. This should be carefully justified.

- Competitors like Splatter360 and PanSplat are designed for ≥2 views; duplicating a single view for training creates a non‑native setting that can amplify CylinderSplat’s advantage.

- What exactly is counted in the “inference time” of Table 6 (one forward pass to Gaussians? including rendering of how many targets)?

- Several sampling hyperparameters (e.g.,  N_r, N'_r are left unspecified in the main text; these matter for memory/time.

### Questions
- Please provide a derivation (or empirical validation) showing why S' is appropriate for anisotropic Gaussians in 3DGS.

- Can you report geometry vs. ground‑truth depth on any subset (e.g., Replica), or provide alternate metrics less tied to a particular depth estimator?

- How does the method fare on scenes with curved walls or outdoor panoramas where cylindrical alignment is less appropriate?

- Have you tried a learned Gaussian deduplication/merging module (e.g., confidence‑weighted thinning or non‑maximum suppression in splat space) rather than concatenation?

- Since you introduce a direct panoramic rasterizer, can you re‑evaluate OmniScene with the same rasterizer (rather than a cubemap adaptation) to isolate representation vs. rendering pipeline effects?

- Please specify precisely what Table 6 “Inference Time” measures

- Since UniK3D underpins depth supervision and visibility, can you ablate different depth priors (e.g., DepthAnywhere, ZoeDepth) to show robustness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a panoramic novel view synthesis method using 3D Gaussian Splatting (3DGS). They introduce a cylindrical triplane representation combined with a dual-branch design: an attention-based, cost-volume-free pixel branch that parameterizes Gaussians from the inputs, and a per-camera local cylindrical-triplane volume branch that handles occlusions. On both synthetic and real datasets, the method achieves competitive image quality (WS-PSNR, SSIM, LPIPS) and geometry metrics (PCC) with strong efficiency.

### Strengths
1. Originality. Introduces a cylindrical triplane representation specifically designed for panoramic 3D Gaussian Splatting. Proposes a three-stage curriculum training strategy (pixel, volume, joint) that significantly enhances reconstruction in occluded and sparsely viewed regions.

2. Quality. Provides comprehensive evaluations across multiple panoramic and indoor datasets with consistent improvements over prior methods. Includes thorough ablation studies demonstrating the effectiveness of the cylindrical triplane and each component.

3. The paper is well-organized and easy to follow.

4. Significance. Effectively addresses the blurriness and occlusion artifacts found in previous sparse-view panoramic methods.

### Weaknesses
1. The method relies heavily on a strong depth prior, but the paper does not analyze its robustness or potential failure cases.
2. The domain bias is significant—experiments focus primarily on indoor, Manhattan-world panoramas. Analysis of non-Manhattan or outdoor environments would strengthen the generality claims.
3. The paper does not discuss or address artifacts near seams and poles, which are common in panoramic representations and could impact real-world performance.

### Questions
Using a foundation model–based depth prior as a reference for geometry evaluation raises concerns about circular validation and fairness in comparisons.

### Soundness
3

### Presentation
3

### Contribution
3
