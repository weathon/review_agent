# Terra: Explorable Native 3D World Model with Point Latents

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
World models have garnered increasing attention for comprehensive modeling of the real world.
However, most existing methods still rely on pixel-aligned representations as the basis for world evolution, neglecting the inherent 3D nature of the physical world.
This could undermine the 3D consistency and diminish the modeling efficiency of world models.
In this paper, we present Terra, a native 3D world model that represents and generates explorable environments in an intrinsic 3D latent space.
Specifically, we propose a novel point-to-Gaussian variational autoencoder (P2G-VAE) that encodes 3D inputs into a latent point representation, which is subsequently decoded as 3D Gaussian primitives to jointly model geometry and appearance.
We then introduce a sparse point flow matching network (SPFlow) for generating the latent point representation, which simultaneously denoises the positions and features of the point latents. 
Our Terra enables exact multi-view consistency with native 3D representation and architecture, and supports flexible rendering from any viewpoint with only a single generation process.
Furthermore, Terra achieves explorable world modeling through progressive generation in the point latent space.
We conduct extensive experiments on the challenging indoor scenes from ScanNet v2.
Terra achieves state-of-the-art performance in both reconstruction and generation with high 3D consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Terra, a native 3D world model designed to represent, generate, and progressively explore 3D environments. The authors argue that conventional world models, which rely on 2D pixel-aligned representations, struggle with 3D consistency and modeling efficiency. Terra addresses this by operating directly in an intrinsic 3D latent space using point latents. Terra proposes a Point-to-Gaussian Variational Autoencoder (P2G-VAE) that transforms colored 3D point cloud to 3D Gaussians, and proposes a Sparse Point Flow matching network (SPFlow) to learn the latent point distribution. Authors show Terra capabilities to reconstruction the scene, do uncondition generation and image-conditioned generation.

### Strengths
1. Authors propose several novel techniques to facilitate the VAE and flow-matching learning, like Robust position perturbation, Adaptive upsampling and refinement, etc. 
2. The final 3D Gaussian representation naturally supports multi-view consistency
3. The model can progressively generate a large-scale, coherent world simulation step-by-step.

### Weaknesses
1. The accuracy and completeness of the input point cloud significantly affect the model performance, no matter in the reconstruction (point to Gaussian) task or the generation task. As shown in Figure 4 and Figure 5, even Terra can learn to complete the partial objects caused by the sensor failure in dark regions, the output Gaussians still have holes. 
2. Continuing from the previous one, in your image-conditioned generation, the accuracy of depth estimation may directly affect the quality of the generation. Once the depth estimator fails or the input image is out of domain, the model might fail as well.
3. Another baseline can be SCube[1], which uses voxels instead of points as an intermediate representation, decoding per-voxel Gaussians for rendering. It would be interesting to see their comparison or theoretical analysis.

[1] SCube: Instant Large-Scale Scene Reconstruction using VoxSplats, NeurIPS 2024

### Questions
1. Can the author elaborate on the scalability of this method? 3D data is not easily obtainable and may contain noise. Yet this method strongly relies on 3D input data. 
2. training time is not reported.
3. What is the maximum range of generation supported in a single inference and step-by-step exploration?
4. In 4.3 main results - Reconstruction, are you reporting the metrics on novel views or just input views?

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
3

### Summary
This paper introduces Terra, a framework for generating explorable 3D environments, where each environment is represented by point latents. The overall pipeline consists of two main components:
1. A Point-to-Gaussian VAE (P2G-VAE) based on PTv3, which encodes colored 3D points into a compact latent space.
2. A Sparse Point Flow Matching (SPFlow) model that learns the distribution of these point latents in the latent space.

The method is trained and evaluated on the ScanNet v2 dataset.
- For reconstruction, Terra achieves better depth accuracy than PixelSplat, MVSplat, Prometheus, and Can3Tok, though it performs worse on the LPIPS metric.
- For unconditional/image-conditioned generation, its geometric quality surpasses Trellis and Prometheus, while its visual quality is higher than Trellis but below Prometheus.

Ablation studies show that:
- Robust Position Perturbation reduces reconstruction quality in P2G-VAE but significantly enhances generative capability.
- Adaptive Upsampling and Refinement and Explicit Color Supervision both improve reconstruction and generation performance.
- Distance-Aware Trajectory Smoothing plays a key role in stabilizing training for generation tasks.

### Strengths
1. The proposed Distance-Aware Trajectory Smoothing is novel and demonstrates clear effectiveness in the context of sparse point flow matching models.

### Weaknesses
1. The term “world model” is conceptually broad. Using it as the paper’s main title may be misleading, as the method focuses more narrowly on 3D world generation and exploration, primarily within indoor scenes.
Moreover, most compared methods do not explicitly position themselves as world models — e.g., Prometheus (text-to-3D generation), Can3Tok (3D scene-level generation), and PixelSplat/MVSplat (3D reconstruction).

### Questions
1. What is the motivation for removing all residual connections in PTv3?
2. What exactly is the ground-truth distribution $\mathbf{P}$ of point latents, and how is it sampled? During inference, are the positions in $\mathbf{P}$ also randomly sampled from Gaussian noise?
3. How do the authors position Terra relative to recent approaches such as WorldMem [Xiao et al., 2025], VMem [Li et al., 2025], and Voyager [Huang et al., 2025]? From a visual standpoint, Terra’s generated results appear somewhat blurry, incomplete, or low-resolution compared to these models. While it remains an open question what the ideal representation for world models should be (e.g., 3D Gaussian Splatting, video-based, or otherwise), it would strengthen the paper if the authors clarified why comparisons to these methods were omitted and articulated Terra’s distinct advantages or future potential.

Things to improve the paper that did not impact the score:
- Please report the GPU hours required to train P2G-VAE and SPFlow.
- Table 1 appears far from its first citation — consider adjusting its placement for readability.
- Consider adding a section on the use of large language models (LLMs).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a method that, given an input colored point cloud, generates shapes in a latent space and allows for progressive exploration. The method uses a Point-to-Gaussian VAE to compress 3D inputs into sparse point latents and decodes them into 3D Gaussian primitives for rendering. It then uses a sparse point flow matching model to jointly denoise point positions and features for generative modeling.

### Strengths
The paper attempt to tackle the reconstruction problem from a native 3D generation perspective. The idea makes sense.

### Weaknesses
1) The method utilizes an input point cloud from fused multi-view depth sensors, which should provide high-quality shapes and textures. However, the generated results appear to make both the shapes and texture blurrier.


2) The comparison with Trellis is unfair. A more appropriate baseline would be to add the point cloud condition to Trellis, for instance, by voxelizing the point cloud to serve as the sparse grid for trellis's structured latent.

3) It seems that the paper is regenerating things that is already available from the input. In Fig. 1. What portion of the generated scene is not present in the input?   It is recommend to visualize the difference between input point cloud and the generated one.  


4) The paper is missing comparisons with important RGB-D reconstruction baselines, such as classic depth map fusion methods (e.g., BundleFusion [1], ElasticFusion [2]) or methods based on neural fields [3]. The reconstruction results reported in this paper appear to be much worse than those achieved by the aforementioned baselines.

[1] BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online Surface Re-integration

[2] ElasticFusion: Real-time dense visual SLAM system

[3] Neural RGB-D Surface Reconstruction

### Questions
- Is this method trained on random 3D crops?
- Does this method complete occluded geometry?
- Line 462. Can this method explorable unseen geometry in the input point clouds?

### Soundness
1

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
2

### Summary
The paper addresses the fundamental limitation of existing world models that rely on pixel-aligned representations. The authors introduce Terra, a native 3D world model, that represents and generates explorable environments using an intrinsic 3D latent space through two key technical innovations: a Point-to-Gaussian Variational Autoencoder (P2G-VAE) that encodes 3D inputs into latent point representations and decodes them as 3D Gaussian primitives to jointly model geometry and appearance, and a Sparse Point Flow Matching Network (SPFlow) that generates latent point representations by simultaneously denoising positions and features of point latents.

### Strengths
1. The paper is well-structured and easy to follow
2. Point-to-Gaussian Variational Autoencoder (P2G-VAE) effectively reduces redundancy in 3D input data while creating a compact latent space that jointly models both geometry and appearance through 3D Gaussian primitives, making it highly efficient for generative modeling.
3. Flexible rendering from any arbitrary viewpoint with only a single generation process
4. Progressive training strategy with three well-designed stages (reconstruction, unconditional pretraining, masked conditional generation)

### Weaknesses
1. No inference / training time comparison
2. No memory usage analysis
3. I believe that performance relates more to the method timing and suggest to use terms "Reconstruction Accuracy" and "Generation Accuracy" in tables 1 and 2.
4. I suggest a couple of high resolution renders in appendix or videos in supplementary materials to evaluate a visual quality of Terra


Overall, I'd be glad to increase the score if the authors address the above issues

### Questions
see weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
