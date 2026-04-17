# Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
We introduce a diffusion-based framework that generates aligned novel view images and geometries via a warping‐and‐inpainting methodology. Unlike prior methods that require dense posed images or pose-embedded generative models limited to in‐domain views, our method leverages off‐the‐shelf geometry predictors to predict partial geometries viewed from reference images, and formulates novel view synthesis as an inpainting task for both image and geometry. To ensure accurate alignment between the generated image and geometry, we propose cross-modal attention instillation where the attention maps from an image diffusion branch are injected into a parallel geometry diffusion branch during both training and inference. This multi-task approach achieves synergistic effects, facilitating both geometrically robust image synthesis and geometry prediction. We further introduce proximity‐based mesh conditioning to integrate depth and normal cues, interpolating between point cloud and filtering erroneously predicted geometry from influencing the generation process. Empirically, our method achieves high-fidelity extrapolative view synthesis, delivers competitive reconstruction under interpolation settings, and produces geometrically aligned point clouds as 3D completion.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper looks at how to perform improved view synthesis, especially in the case of more extreme viewpoint changes where older methods would usually break down (due to slightly incorrect geometry or large reasons for outpainting).

The authors solution is to leverage a predicted 3D point cloud and image generation methodology together in order to improve on the quality of either in isolation by use of a x-attention map that leverages information from both modalities.

The authors propose a pipeline, leveraging prior work (e.g. for point cloud extraction and viewpoint prediction) in order to achieve this.

They achieve good results in comparison to prior work.

### Strengths
1. The paper is clearly written and easy enough to follow.

2. The idea is reasonable and makes sense -- to enforce consistency between predicted geometry and viewpoint synthesis. They also propose to use 3D techniques (e.g. conversion to a mesh w/ corresponding normals and depth to allow the model to determine which points are more reliable.

3. The authors evaluate both in/out of distribution and include ablations and standard baselines.

### Weaknesses
1. Some of the described intuition is unclear to me: "The image denoising U-Net receives deterministic training signals from the geometry completion network," --> This implies to me that the image-inpainting and geometry prediction models are trained separately but from reading the paper, I don't think that they are. I'm also unsure as to what losses are used to train the geometry prediction model and what it is expected to look like -- my understanding is the loss is on the depmap which the point cloud is projected to. But this is unclear to me from reading the paper.

2. Ablations on the geometry results.
The paper has an ablation in Table 3 on the impact of the geometry losses + cross model instillation for the view synthesis but not on how those same choices impact the geometric results in Table 4.

3. Further comparisons to SOTA
The authors could have included results / comparisons (in terms of speed / performance) to the methods that they say they are improving against by not having to do a separate NERF optimization (e.g. CAT3D). In that case, it would be imp. to compare the performance by considering the need of the authors' work to generate the 3D point cloud in the first place using the other method.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a diffusion-based inpainting framework to enhance novel view synthesis (NVS) quality, leveraging the 3D foundation model as geometric prior. Specifically, the method utilizes the off-the-shelf geometry predictors and fills both the geometry and image by framing NVS as a completion problem. To ensure alignment between the generated image and geometry, it presents MoAI where the attention maps from the diffusion branch are injected into a parallel geometry diffusion branch. The experiments validate the effectiveness of the proposed approach on several datasets.

### Strengths
* The paper is well structured and easy to follow.
* Experiments on Co3D and DTU datasets demonstrate the effectiveness of the proposed method.
* The cross-modal attention installation is interesting to me and has been shown to be effective via the ablation study.

### Weaknesses
* The integration of diffusion priors to help NVS has been extensively explored in recent works, like diffusion-aided NeRF/3DGS  reconstruction (Deceptive-NeRF/3DGS [Liu et al.]) and unified frameworks such as ReconX, ViewCrafter, and ZeroNVS. Moreover, the proximity with mesh conditioning is quite standard.  Although the MoAI is interesting, it cannot support the whole paper for a top conference. 
* The paper compares primarily against classical or earlier NVS methods (e.g., NeRF variants), but omits several strong diffusion-based or geometry-conditioned NVS works, like ZeroNVS, ViewCrafter, Rand econX. 
* Only the quantitative comparison with LVSM is also missing in the paper, which makes it difficult to assess the actual progress relative to the state of the art.
* The runtime and memory consumption are missing.
* The mesh generation with incomplete and noisy points is not easy itself. The robustness of the method is not well evaluated, especially considering that it adopts a rather classical mesh generation.

### Questions
Please refer to the weaknesses in the weaknesses part. I am open to being persuaded based on the feedback from the authors.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a diffusion-based warping-and-inpainting framework that jointly synthesizes novel-view images and geometry from unposed references. It uses off-the-shelf pose/pointmap predictors to project partial 3D into the target view, then completes both modalities via conditional denoising. The core idea, cross-Modal Attention Instillation (MoAI), replaces the geometry U-Net’s spatial attention with that of the image U-Net during training and inference to enforce image–geometry alignment and leverage semantic correspondences. To handle noisy projections, proximity-based mesh conditioning converts sparse point clouds to meshes, augments conditions with depth/normal cues, and masks implausible surfaces. Experiments on DTU and RealEstate10K show strong extrapolative NVS, competitive interpolation, aligned pointmaps enabling 3D completion, and consistent gains from pointmap/mesh conditioning and MoAI.

### Strengths
- Joint diffusion for images and geometry with cross-modal attention instillation (MoAI) is a simple, original coupling; using image attention to guide geometry and vice versa is creative, and proximity-based mesh conditioning is a practical improvement over raw pointmaps.

- Empirical results are strong in extrapolative settings on DTU and RealEstate10K, with clear additive gains in ablations (pointmap → mesh → MoAI) and the ability to benefit from more input views at test time despite two-view training.

- The method is clearly described with equations and rationale; the motivation for MoAI is well illustrated, and training/inference details are transparent enough to reproduce.

- The work is significant for pose-free extrapolative NVS and image–geometry alignment, producing aligned outputs that enable 3D completion and benefit downstream reconstruction and content creation.

### Weaknesses
- Reliance on off-the-shelf geometry/pose predictors   
The pipeline depends on VGGT/MASt3R-style predictors for both training supervision (pseudo GT) and inference conditioning. This creates a ceiling tied to those models’ biases and errors, and complicates fairness: improvements may partially reflect better use of VGGT rather than intrinsic advances. Please include: (i) a robustness study under degraded pointmaps/poses (noise, sparsity, biased scale), (ii) an alternative backbone (e.g., DUSt3R/MASt3R vs. VGGT) to show generality.

- MoAI ablations are underspecified   
The paper replaces geometry U-Net attention with image U-Net attention, but lacks: (i) analysis of which layers/stages matter most, (ii) partial vs. full instillation, and (iii) compute/memory overhead.

- Geometry alignment claims hinge on pseudo GT and camera-space normalization  
Depth/pointmap “alignment” is evaluated with pseudo labels derived from the same family of predictors, and benefits from target-camera-space normalization. To avoid confirmation bias, add: (i) cross-predictor validation (e.g., train with VGGT pseudo GT, evaluate with MASt3R/Colmap), and (ii) ablations that remove target-camera normalization to quantify true contribution and residual misalignment.

- Limited analysis of multi-view scaling and consistency across targets  
The model can accept more reference views at test time, but consistency across multiple simultaneously generated target views is not evaluated. Add: (i) cross-target consistency metrics (e.g., photometric and geometric cycle consistency), and (ii) performance scaling plots vs. number of references with compute trade-offs.

### Questions
- Scope of “pose-free”: The method estimates poses/pointmaps from references via an external model and then conditions the diffusion process. In practical deployments, how are target cameras specified? Are you assuming the user provides an arbitrary camera, or do you sample target poses? If target intrinsics/extrinsics are perturbed, how robust is synthesis?
- MoAI mechanism and design choices: Which attention layers are instilled (early/mid/late; which resolutions)? Did you try partial instillation (e.g., only shallow or only deep blocks) or mixing a fraction of the image U-Net attention instead of full replacement? Please include layer-wise ablations and attention visualizations before/after MoAI on several scenes to clarify what correspondences are transferred and where the largest gains arise.
- Comparisons to more baselines: Could you add AnySplat and a DUSt3R+PnP+splatting pipeline as additional pose-free baselines under 2–3 views, at least on DTU/RE10K subsets? If they fail at larger extrapolation, a controlled comparative failure analysis (with runtime and failure modes) would strengthen your positioning.
- Failure cases and qualitative diagnostics: A short taxonomy of failure modes and their link to upstream geometry errors vs. diffusion inpainting would be very helpful, along with attention visualizations that reveal where MoAI succeeds or fails.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a diffusion-based framework for few-shot novel view synthesis (NVS) that generates both aligned novel view images and corresponding geometries. Unlike prior approaches (e.g., PixelSplat, MVSplat, or diffusion-based GenWarp) that require dense camera poses or handle only interpolative view synthesis, this work aims to address the pose-free extrapolative NVS setting.

The key innovation is Cross-Modal Attention Instillation (MoAI): a mechanism that transfers attention maps from an image diffusion branch to a geometry diffusion branch. This encourages alignment between synthesized appearance and predicted geometry, mitigating inconsistencies that arise in conventional inpainting-based pipelines.

### Strengths
- Novel and technically elegant contribution:
The cross-modal attention instillation mechanism is conceptually clean and effectively bridges the gap between image synthesis and geometry completion. It’s a natural extension of diffusion-based correspondence learning and could influence future cross-domain conditioning designs.

- Comprehensive evaluation:
The experiments cover both extrapolative and interpolative regimes, multiple datasets (Co3D, DTU, RealEstate10K), and ablation analyses (Table 3–5) that clearly isolate contributions of each design component (pointmap conditioning, mesh proximity, MoAI).

- Strong empirical results:
The method consistently outperforms existing approaches across metrics (e.g., PSNR↑17.4 vs 14.3 on DTU extrapolative), demonstrating robust generalization, especially in pose-free and single-view setups.

- Interpretability and coherence:
The qualitative figures (Fig. 3, 6, 9) convincingly show improvements in structural alignment, artifact reduction, and cross-view consistency. The framework maintains geometric realism without explicit NeRF optimization.

- Scalability and flexibility:
The model generalizes to variable numbers of input views (Table 5), highlighting strong architectural modularity and robustness.

### Weaknesses
- Limited exploration of generalization across domains and semantics:
It remains unclear whether the approach scales to in-the-wild scenes (e.g., urban/street-level data) or semantic diversity (humans, animals, etc.).

- Dependence on pretrained geometry estimators (e.g., VGGT, Marigold):
The framework relies heavily on the quality of external predictors. Errors in these priors can propagate, which partially undermines the claim of being “pose-free” or “self-contained.”

- Lack of detailed comparison to recent large-scale diffusion baselines:
While comparisons to Zero123, GenWarp, and MVDream are discussed, newer world-consistent 3D diffusion models should be quantitatively evaluated to position the work more clearly.

- Compute complexity and practicality:
Training with dual-branch diffusion networks (image + geometry) and multi-view attention is computationally expensive; inference speed and memory footprint are not reported.

- Missing discussion of failure cases:
The paper does not analyze scenarios where MoAI might fail, e.g., inconsistent shadows or hallucinated surfaces due to semantic-geometry conflicts.

### Questions
- How does MoAI compare to simply concatenating geometry and image features before attention?
Could this mechanism be replaced with a more general form of cross-attention sharing?

- What happens if the pretrained geometry network is replaced with a lightweight learned depth prior, does performance drop significantly?

- Does the model maintain physical consistency (e.g., shading or occlusion correctness) when extrapolating beyond 90° view differences?

- Could the method be extended to temporal coherence (e.g., video novel view synthesis)?

### Soundness
3

### Presentation
2

### Contribution
2
