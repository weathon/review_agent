# Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
Due to the real-time rendering performance, 3D Gaussian Splatting (3DGS) has emerged as the leading method for radiance field reconstruction. However, its reliance on spherical harmonics for color encoding inherently limits its ability to separate diffuse and specular components, making it challenging to accurately represent complex reflections. To address this, we propose a novel enhanced Gaussian kernel that explicitly models specular effects through view-dependent opacity. Meanwhile, we introduce an error-driven compensation strategy to improve rendering quality in existing 3DGS scenes. Our method begins with 2D Gaussian initialization and then adaptively inserts and optimizes enhanced Gaussian kernels, ultimately producing an augmented radiance field. Experiments demonstrate that our method not only surpasses state-of-the-art NeRF methods in rendering performance but also achieves greater parameter efficiency. Project page at: \url{https://xiaoxinyyx.github.io/augs}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Augmented Radiance Field (ARF), a plug-and-play framework that enhances 3D Gaussian Splatting (3DGS) for realistic view synthesis. While 3DGS achieves real-time rendering, its spherical harmonic color model struggles with specular reflections and material separation. ARF introduces a Gaussian kernel with view-dependent opacity to explicitly model specular effects. It further proposes an error-driven refinement that inserts and optimizes additional Gaussians in high-error regions. Experiments on multiple benchmarks show that ARF outperforms state-of-the-art NeRF and 3DGS variants.

### Strengths
1. The paper introduces a new Gaussian kernel with view-dependent opacity that effectively models specular reflections and high-frequency lighting effects.

2. The proposed 2D-to-3D error compensation mechanism adaptively adds and optimizes supplementary Gaussians in challenging regions.

3. Extensive experiments across several benchmarks demonstrate consistent improvements over state-of-the-art NeRF and 3DGS methods, even with lower-order spherical harmonics.

### Weaknesses
1. Since the method refines pre-trained 3DGS scenes rather than learning end-to-end, it may accumulate suboptimal biases from the base model and depend on prior scene quality. The image-space optimization and 2D-to-3D projection introduce extra computation and memory costs compared to standard 3DGS.

2. The paper does not analyze training stability or convergence behavior when optimizing new Gaussians, especially in highly reflective or complex lighting conditions.

3. The rendering quality heavily depends on the tuning of the opacity lobe parameters beta; the paper does not provide an analysis of how these parameters influence optimization.

### Questions
See weakness.

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
This paper addresses limitations of 3D Gaussian Splatting (3DGS) in handling complex reflections due to its reliance on spherical harmonics for color encoding. The authors propose an enhanced Gaussian kernel that models specular effects via view-dependent opacity, combined with an error-driven compensation strategy to improve rendering quality. Their method starts with 2D Gaussian initialization, followed by adaptive insertion and optimization of enhanced kernels to produce an augmented radiance field.

### Strengths
This paper proposes a novel post-enhancement method for Gaussian splatting based on the Phong shading model, aiming to improve the modeling of view-dependent color.

### Weaknesses
- The paper leverages geometric information from depth maps of a pre-trained 3DGS and back-projects the screen-space 2D Gaussians into world space. However, due to the limitation of low-order spherical harmonics in 3DGS, the reconstructed scene geometry tends to be quite poor. As is well known, more accurate geometry modeling typically leads to more reliable view-dependent color estimation. Unfortunately, the paper does not provide any comparison between the optimized depth results and those from the pre-trained 3DGS. For instance, in the garden scene of the Mip-NeRF 360 dataset, the flat tabletop region could have been used as a clear example for such a comparison.
- The proposed method does not fundamentally address the limitation of using spherical harmonics in 3DGS for separating diffuse and specular components. For example, Spec-Gaussian leverages anisotropic spherical Gaussians to better model view-dependent appearance. Instead of optimizing a pre-trained 3DGS, it might be more meaningful to design a more effective approach for modeling view-dependent effects directly, since the use of low-order SH inherently reflects a trade-off between rendering speed and quality.

The paper has some typos:
1. Line 101, "prossesses"
2. Line 155-156, "view-dependently"
3. Line 188, "...  reconstructs intricate radiance field"

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper augments 3D Gaussian Splatting (3DGS) with a view-dependent opacity lobe to better reproduce specular effects. It introduces a post-enhancement pipeline that (i) detects high-residual regions in image space and optimizes sparse 2D Gaussians there, (ii) back-projects these into 3D via an “inverse splatting” procedure, and (iii) jointly optimizes the enhanced set with the original scene. The module is positioned as plug-and-play for existing 3DGS pipelines and aims to improve quality with modest runtime overhead.

### Strengths
1. The two-stage 2D residual fixing, inverse splatting, and joint optimization could improve the final rendering quality.
2. The proposed method could be served as a post-hoc module on top of the 3DGS-based methods.

### Weaknesses
1. The primary concern is the design choice to separate diffuse and specular components across different primitives. In real scenes, most surfaces exhibit a mixture of both (with mirrors as a special extreme), and prior work that models these components within unified primitives enables shared optimization, coherent regularization, and cleaner support for inverse rendering and relighting. By contrast, the proposed error-driven initialization of dedicated “specular” primitives breaks this unification, ties specular capacity to residual patterns rather than material properties, and risks uneven coverage and missed interactions. As a result, the approach sacrifices some of the interpretability and downstream utility.
2. Lobe-based models for specular appearance (notably spherical Gaussians) are well established in rendering and have been adopted within 3DGS frameworks (e.g., SpecGaussian [A]). Beyond these, there are various prior works tailored to specular scenes, including methods with broader applications such as inverse rendering and relighting. The manuscript would benefit from direct comparisons with these baselines to demonstrate the performance of the proposed design.
3. Some statements about "SH inherently fails to decouple diffuse and specular components" on line 070 conflict with the definitions of the specular and diffuse components on line 138. SH0 could capture view-dependent colors, with higher orders capturing view-dependent colors. Some explanations about why SH inherently fails to decouple diffuse and specular components are needed.

[A] Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting

### Questions
1. How to compute the gradients when performing the optimization in image space? As described in 209-210, the depth is updated via the nearest-neighbor sampling during optimization. However, since the emulated 3DGS differentiable rendering (described on line 201) requires depth sorting (in both forward and backward pass), this depth updating strategy will influence the gradients computation. Specifically, how many rasterization passes are performed during an optimization step? when to perform the depth updating, like before or after updating other parameters?
2. What are the end-to-end training time and rendering FPS, compared with baselines (including baselines designed for specular scenes, such as SpecGaussian)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a strategy to enhance the under-reconstructed regions of 3DGS. The method begins by learning a 3DGS model to fit the given multi-view images. However, due to limitations in the original densification and optimization processes of 3DGS, certain regions may not be reconstructed effectively.

To address this, the method identifies under-reconstructed regions by comparing the rendered images with the GT images. An error-weighted adaptive strategy is then employed to sample 2DGS on each view screen, targeting these problematic regions. Using the original rendered images as backgrounds, the 2DGS in each view is optimized to correct the image and align it with the GT.

Although 2DGS is defined in screen space, the order of the splats is determined by the depth of the nearest rendered pixel. After optimization, clustering is performed for each 2D Gaussian to identify inlier pixels. These inliers are back-projected into 3D space, and a weighted PCA is applied to analyze their Gaussian distribution. The resulting 3D points are then directly used to initialize the supplemented 3DGS.

Finally, these supplemented 3DGS points, along with the opacity of the original 3DGS, are jointly optimized to fit the input images, resulting in a more accurate 3D reconstruction.

### Strengths
1. The paper addresses the under-reconstruction problem of 3D Gaussian Splatting (3DGS) directly and effectively by employing a smart solution: correcting problematic regions through 2D fitting and back-projection. The entire process carefully accounts for various factors in the pipeline that could lead to artifacts, making the approach both robust and practical. By "reintegrating missing details into 3D space," the method achieves improved reconstruction. The idea is plausible, insightful, and highly meaningful.

2. The rendered images produced by this method significantly outperform the baselines. Highlighted and specular regions are reconstructed with remarkable accuracy. This improvement is largely attributed to the proposed directional decaying opacity, which enables the method to better capture and fit high-frequency details in these challenging areas.

### Weaknesses
The primary weakness of the paper lies in the limited comparison with baselines. The main baseline used in the study is deformable beta splatting, which explores alternative primitive representation functions to replace Gaussians. However, two additional relevant baselines are worth considering for comparison:

1. **Spec-Gaussian**: Spec-Gaussian introduces a more advanced lighting function to replace SHs, enabling better handling of commonly observed anisotropic appearances. It achieves superior results in specular and highlight regions, making it closely related to the view-dependent opacity proposed in the paper.

2. **AbsGS**: AbsGS leverages the absolute gradient norm accumulation as a metric to densify 3DGS. By using the absolute gradients from the 2D screen, it can effectively identify under-reconstructed regions, thereby improving the densification of Gaussians in problematic areas. This is closely related to the enhancement-on-the-bad idea proposed in the paper.

### Questions
- I hold a positive attitude toward the proposed method; however, my main concern is the limited comparison with the closely related baselines mentioned, which weakens the overall persuasiveness of the paper. I would be willing to raise the score to accept if these comparisons are included. :)

- The proposed method does not appear to be restricted to a specific representation, such as 3DGS. Beyond 3DGS, other flexible primitives, such as Deformable Beta Splatting (DBS), 3D Convex Splatting, and Deformable Radial Kernel (DRK), can also efficiently capture sharp regions and varying boundaries. These primitives could be incorporated into the 2D fixing step as well. Among them, DRK stands out as a strict projection-based representation, making it much easier to back-project and achieve accurate depth rendering. I suggest including a discussion on the potential application of the method to these more flexible primitives, as it could provide valuable insights and inspiration for future research.

### Soundness
4

### Presentation
4

### Contribution
4
