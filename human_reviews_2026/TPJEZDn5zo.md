# Constructing a 3D Scene from a Single Image

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes.
In this work, we introduce SceneFuse-3D, a training-free framework designed to synthesize coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning.
Extensive experiments across diverse scenes show that SceneFuse-3D outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, TripoSG, and LGM, in terms of geometry quality, spatial coherence, and texture fidelity. Our results demonstrate that high-quality unstructured 3D scene-level asset generation is achievable from a single top-down image using a principled, training-free pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of scene-level 3D generation from (bird-eye-view) images. The motivation and contribution is clear.

### Strengths
The proposed solution is intuitive and effective. Using point-cloud from monocular depth estimation as a guidance  helps improve the scene-level consistency. Regional alignment and extrapolation/inpainting is simple and effective.

### Weaknesses
1) Since the top-down images are generated, this whole data pipeline is ‘text-to-image -> image-to-3D’, which looks similar to syncity’s ‘text-to-3D’ pipeline. The contribution of this paper could be further strengthened by providing real-world experiments and assessments.

2) Runtime analysis should be provided. This pipeline takes longer time compared to directly generating the scene-scale mesh in one-run, it is critical to report run time (though it depends on scene complexity, which makes the analysis more critica).

### Questions
1. The monocular depth estimation results is expressed under the camera coordinate system. However, the camera often has a pitch angle, i.e., the camera’s optical axis is not parallel with the gravity. This means that the projected point clouds are tilted. When we generate scene-level 3D meshes, we would like the up-axis aligned with the gravity, how to resolve this issue?
2. Recommended citations: 
VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space (inpainting with Trellis)
Frankenstein: Generating semantic-compositional 3d scenes in one tri-plane (scene generation)
NuiScene: Exploring efficient generation of unbounded outdoor scenes (scene generation)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an image-to-3D method by leveraging 3D generative model, trellis, as a backbone to generate   3D scenes. It starts with depth estimation to obtain the sparse structure of the scene, and then a structured latent is generated region-by-region using masked rectified flow, which is then fed into decoder to generate the 3D scene.  The overall pipeline is training-free.

### Strengths
This paper provides a sound pipeline that leverages 3D generation model, Trellis, to generate 3D scenes.
1. It adapts a 2D diffusion method to 3D generation, and develops a region-by-region structured latent generation method.
2. It presents a masked rectified flow method to retain the latent feature at know voxels. 
3. The experimental results verify the advantage of the proposed method.

### Weaknesses
1. the proposed method relies on the top-down view of a 3D scene as an condition in the 3D scene generation, thus the generated ground plane is generally flat.  It might be difficult to handle terrains. 
2. The experimental results contain 4 scenes, which is not enough to verify the stability of the proposed pipeline.  In addition, how the method is influenced by different depth estimation method?  I would like to see how this pipeline works with the STOA depth estimation methods.

### Questions
In the step of structured latent generation,  how the latent feature of active voxels are obtained?  Since the proposed method depends on Trellis,  after obtaining the sparse structure, why not directly leverage Trellis to generate the structured latent?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a training-free method for creating 3D scenes from single images by fusing region-based generations. The key problem identified is in maintaining the structural consistency within partial generations during generation and completion. This is resolved by applying a sequence of techniques derived from various fields, including landmark bounding boxes (Florence2), instance masks (SAM2), point cloud-based mesh alignment (ICP), training-free inpainting from masked rectified flow (RePaint, updated to 3D diffusion). The method is mainly compared with training-based, and training-free 3D generation methods using human/GPT-4o preference scores and rendered view metrics.

### Strengths
- The proposed method effectively modifies and aggregates existing solutions from different subproblems into a single pipeline to solve general problem of generating a 3D scene.
- The proposed method shows higher qualitative and quantitative performances than previous training-free and model-based generation results as elaborated in multiple tables in the manuscript.
- The ablation study shows the two originality of the paper, i.e., region-based generation and landmark conditioning do help the generation process.

Overall, I believe the paper is nicely written, with sufficient amount of evaluations for the target task, and therefore is above acceptance threshold.

### Weaknesses
Although I believe the current version of the manuscript is above acceptance threshold, there are some limitations that prevents me recommending for higher honor (e.g., Highlight/Oral).

1. First of all, since the paper utilizes various methods that have been proposed beforehand, some modified and some unmodified, it would be much better to have the summarization table (somewhere in the appendices) that shows which part of the pipeline is operated by which method, thereby implying modular upgrades to gain overall performance boost. This also helps the reader clarify which part of the algorithm is novel in this paper.
2. As far as I read, the main problem at hand is to gain consistency while allowing partial generations, and the immediate desired consequence is infilling of holes within the final generated output. We may either quantitatively (maybe by calculating the size/number of holes created around the reference shot) or qualitatively (if the quantitative analysis is hard; by visually highlighting the holes created by the different types of methods within the same camera view) compare this geometric quality.
3. By changing the size of region being generation, the generation time may vary. It would be better reporting this generating time of different methods, as well as for the ablation studies.
4. Since the method sells for being training-free, we may also think about 3D generator-agnostic behavior. The paper will be more complete if the authors perform ablation study on the core generation model of the method.

### Questions
Here are some minor points that I did not count in my scoring.

1. In Table 2, quantitative comparison can be improved by adding reference consistency scores measured by LPIPS (PSNR and SSIM can also be considered), i.e., rendering the scene with the exact same camera parameters with reference image and then comparing them.
2. Likewise, it would be better to have a figure that compares the reference camera view rendering to gain consistency in comparison between various methods.

### Soundness
4

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
The paper proposes SceneFuse-3D, a training-free, modular pipeline for turning a single top-down image into a coherent, editable 3D scene. The core contributions are: (1) region-based generation that splits a scene-level structured latent into overlapping patches; (2) a spatial prior built from monocular depth and landmark instances; (3) masked rectified-flow completion that regenerates only unknown latent parts while preserving already-consistent content. The final scene is decoded with pretrained object decoders. The authors constructed a test dataset comprising 100 synthesized top-down images spanning diverse styles. The experimental results demonstrate that SceneFuse-3D outperforms existing object generation models.

### Strengths
* SceneFuse-3D employs a training-free approach, which utilizes existing models to accomplish the scene generation task without requiring fine-tuning of the base models.
* Using existing foundation models (e.g., depth estimation, Florence2, and SAM2) to provide spatial priors and effectively stabilize global layout and cross-region consistency.
* The paper is well structured in general.

### Weaknesses
* The method appears to rely heavily on external priors (e.g., monocular depth, Florence2, SAM2, ICP), which may propagate errors throughout the pipeline.
* Some of the generated scenes appear to contain holes (e.g., in Figure 1 and the supplementary materials).
* The proposed method seems to support only top-down views from specific angles as input images.

### Questions
1. How robust is the proposed pipeline to errors in depth estimation and landmark detection? For instance, during the initialization phase, the base model might fail to accurately detect all landmarks.

2. The authors attribute the holes in the generated scenes to occlusion; however, in the examples shown, the hole regions do not appear to be occluded. I remain curious whether these holes may instead result from the object generation backbone itself, as the regions near object voxel boundaries are often empty.

3. In the Masked Rectified Flow for Completion stage, what is the conditioning input? Does it make use of image patches from other regions?

Although I am now slightly negative, I am open to being persuaded based on the feedback from the authors.

### Soundness
3

### Presentation
2

### Contribution
2
