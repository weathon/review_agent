# Restore3D: Breathing Life into Broken Objects with Shape and Texture Restoration

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Restoring incomplete or damaged 3D objects is crucial for cultural heritage preservation, occluded object reconstruction, and artistic design. Existing methods primarily focus on geometric completion, often neglecting texture restoration and struggling with relatively complex and diverse objects. We introduce Restore3D, a novel framework that simultaneously restores both the shape and texture of broken objects using multi-view images. To address limited training data, we develop an automated data generation pipeline that synthesizes paired incomplete-complete samples from large-scale 3D datasets. Central to Restore3D is a multi-view model, enhanced by a carefully designed Mask Self-Perceiver module with a Depth-Aware Mask Rectifier. The rectified masks, learned through the self-perceiver, facilitate an image integration and enhancement phase that preserves shape and texture patterns of incomplete objects and mitigates the low-resolution limitations of the base model, yielding high-resolution, semantically coherent, and view-consistent multi-view images. A coarse-to-fine reconstruction strategy is then employed to recover detailed textured 3D meshes from refined multi-view images. Comprehensive experiments show that Restore3D produces visually and geometrically faithful 3D textured meshes, outperforming existing methods and paving the way for more robust 3D object restoration. Project Page: https://iclr-subx.github.io/Restore3D/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Restore3D, a framework designed to simultaneously recover the shape and texture of fragmented objects from multi-view images. The core component is a multi-view model enhanced by a Mask Self-Perceiver and a Depth-Aware Mask Rectifier. A coarse-to-fine reconstruction strategy is further employed to generate a complete 3D model with detailed textures from the refined multi-view inputs.
Overall, this is a technically comprehensive and systematic work that demonstrates promising results. However, the experimental validation appears limited, comparisons with related 3D reconstruction methods are missing, and ablation studies for key components are not sufficiently conducted.

### Strengths
1. Introduce an automated data synthesis pipeline that generates paired incomplete and complete shapes and texture.
2. A technically comprehensive and systematic work for restoring 3D assets that demonstrates promising results.

### Weaknesses
1. The mask itself does not provide any 3D geometric constraints, unlike depth or normal maps. How does the method ensure 3D consistency in the inpainted regions when relying solely on the mask?

2. There is a missing comparison with relevant 3D generation and completion baselines (e.g., SD-Fusion), which makes the reported results less convincing.

3. The ablation study is not sufficiently comprehensive. What is the effectiveness of the Masked Self-Perceiver module and the image integration and enhancement process? The overall framework also appears rather complex, which hyperparameters can be tuned, and how sensitive are the results to these settings?

4. Beyond the reported low-resolution limitation, are there other constraints on the method’s ability to complete missing regions? For example, how does the size of the missing area affect reconstruction quality?

### Questions
1. The mask itself does not provide any 3D geometric constraints, unlike depth or normal maps. How does the method ensure 3D consistency in the inpainted regions when relying solely on the mask?

2. There is a missing comparison with relevant 3D generation and completion baselines (e.g., SD-Fusion), which makes the reported results less convincing.

3. The ablation study is not sufficiently comprehensive. What is the effectiveness of the Masked Self-Perceiver module and the image integration and enhancement process? The overall framework also appears rather complex, which hyperparameters can be tuned, and how sensitive are the results to these settings?

4. Beyond the reported low-resolution limitation, are there other constraints on the method’s ability to complete missing regions? For example, how does the size of the missing area affect reconstruction quality?

I would revise my rating if the authors solved my concerns

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes *Restore3D*, a pipeline that simultaneously restores the shape and texture of broken 3D objects from multi-view images. The authors build an automated data synthesis pipeline to generate paired broken/complete data, design a mask self-perceiver with a depth-aware mask rectifier to solve multi-view inpainting without manually provided masks, and further refine geometry and texture using normal priors and image enhancement. Experiments demonstrate clear improvements over baselines in both inpainting quality and reconstructed mesh fidelity.

### Strengths
1. The proposed mask self-perceiver with depth-aware mask rectifier is well-motivated and eliminates the need for manually-defined masks in multi-view settings.
2. The experimental results show substantial improvements over existing baselines across multiple datasets, including real and physically simulated broken objects.
3. The paper is overall well-written and should be easy to follow.

### Weaknesses
1. The overall pipeline consists of many sequential components (e.g., multi-view inpainting, mask rectification, image integration, and refinement), and this multi-stage dependency may raise concerns regarding robustness to failures in intermediate stages. In addition, it would be helpful to provide a runtime breakdown for each stage to better assess the practicality and deployment cost of the system.
2. The paper would benefit from a comparison against **Amodal3R**, a Trellis-based method for object completion under occlusions. In this setting, the occluded regions predicted by the proposed mask self-perceiver could naturally serve as the amodal masks required by Amodal3R.

[1] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images. (ICCV25)

### Questions
How does the method behave in cases where the broken region is extremely large (e.g., >60% missing)?

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
3

### Summary
Authors propose a framework to restore geometry and texture from damaged 3D object. Specifically, authors use a multi-view model enhanced with a so-called mask-perceiver (which takes object incomplete mask) and depth-aware mask rectifier (which reconstructs occluded mask of interest). Afterwards, authors enhance multiview image from 256x256 and apply geometry-guided texture and shape optimization to refine the result.

### Strengths
1. Authors proposed a method which lets determine masks for multi-view inpainting, while other methods often require user-specified mask.
2. The design of mask self-perceiver and depth-aware mask rectifier is novel, to my knowledge.
3. Authors propose a novel way of generating synthetic dataset of broken objects.

### Weaknesses
1. Authors employ image enhancement and geometry + texture refinement as means to enhance overall quality of their pipeline. After careful review, I do not see any novelty in their proposed methodology, and consider it as test-time enhancement strategy as it has to be applied per object.
2. It is furthermore not clear whether the same texture / geometry optimization was used when inferring other methods.
3. Important baselines are missing (see question 3).
4. The quality of the output of mask self-perceiver is not convincing based on Fig 2. (see question 4).

### Questions
1. L088 "as simpler methods often fall short" - please elaborate.
2. Please provide more information on whether image enhancement or texture / geometry optimization technique was used when comparing with other methods, especially in table 1b.
3. Please add missing comparisons: MVInpainter in Table 1a, Amodal3R, Step1X or Hunyuan3D in Table 1b, ObjFiller3D in Table 2a.
4. Based on the visuals in Figure 2, the quality of the mask produced by the perceiver appears suboptimal. I therefore suggest that the authors (1) include additional visual examples and (2) consider performing an ablation study comparing their design with the straightforward DiffEdit approach (see Figure 2 in the main paper), which determines the masking region using, for instance, the difference between noise conditioned and unconditioned on text.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Restore3D, a system for joint restoration of geometry and texture for damaged objects from multi-view images. The pipeline first performs mask-free multi-view inpainting using a mask self-perceiver together with a depth-aware mask rectifier to decide what to keep versus synthesize, alleviating cross-view inconsistency and occlusion issues. The completed views are then integrated and enhanced before a coarse-to-fine 3D reconstruction stage: a large reconstruction model produces a coarse mesh, which is refined with normal priors for shape and UV back-projection with optimization for texture. Training pairs are generated automatically from G-Objaverse via Blender Boolean operations. Experiments on GSO, OmniObject3D, Objaverse, and out-of-domain case studies show consistent gains on PSNR/LPIPS/FID/SSIM for inpainting and CD/F-score for geometry, with ablations supporting the component choices and a reported ~20 s/object refinement time.

### Strengths
1. Originality: Tackling shape and texture restoration jointly from multi-view inputs—without user masks—is timely. The combination of a mask self-perceiver with a depth-aware rectifier is a sensible departure from single-view, mask-driven inpainting and geometry-only completion.
2. Design Rationality: The system is end-to-end and reasonably complete: synthetic data generation, mask-free inpainting, view integration/enhancement, and coarse-to-fine reconstruction with normal-prior refinement and UV-space texture optimization.
3. Broad Application: If the robustness holds, the approach is potentially useful for cultural-heritage restoration and occluded-object recovery, and it shows encouraging cross-domain generalization despite training primarily on synthetic data.

### Weaknesses
1 Data realism and domain gap: Breakage is synthesized via Boolean operations on G-Objaverse. This is practical but may not capture fracture statistics, material properties, or abrasion typical of real artifacts. The paper would benefit from a more systematic analysis of failure modes and long-tail categories.
2 Stacked dependencies and attribution: The method leans on several strong priors (Depth-Anything/StableNormal, Real-ESRGAN, ControlNet-Tile, LRMs). It is difficult to disentangle where the gains come from or to assess reproducibility under tighter compute budgets. More controlled diagnostics (e.g., removing Real-ESRGAN or swapping ControlNet-Tile) would clarify attribution.
3 Evaluation breadth: Beyond standard PSNR/FID/LPIPS/SSIM and CD/F-score, there is no assessment of texture/lighting fidelity from a perceptual standpoint, view-consistency scoring, or user preferences. Texture quality is not evaluated beyond SSIM or simple RGB losses.

### Questions
1. Data synthesis fidelity: How do Boolean-based fractures differ from real breakage (e.g., edge roughness, fragment size distribution), and does adding procedural noise narrow the gap? Any quantitative comparison on real fragments versus synthetic pairs?
2. Mask self-perceiver supervision: How are targets obtained for the rectified masks during training, and how often does the rectifier fail under depth errors? Diagnostics stratified by mask size and occlusion severity would be helpful.
3. Enhancement attribution: Could you ablate Real-ESRGAN versus ControlNet-Tile and the blending strategy, reporting view-consistency and texture sharpness before/after integration to pinpoint the source of gains?

### Soundness
3

### Presentation
3

### Contribution
3
