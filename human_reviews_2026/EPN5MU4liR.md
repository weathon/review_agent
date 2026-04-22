# A^2TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Gaussian Splatting has emerged as a powerful representation for high-quality, real-time 3D scene rendering. While recent works extend Gaussians with learnable textures to enrich visual appearance, existing approaches allocate a fixed square texture per primitive, leading to inefficient memory usage and limited adaptability to scene variability. In this paper, we introduce adaptive anisotropic textured Gaussians (A$^2$TG), a novel representation that generalizes textured Gaussians by equipping each primitive with an anisotropic texture. Our method employs a gradient-guided adaptive rule to jointly determine texture resolution and aspect ratio, enabling non-uniform, detail-aware allocation that aligns with the anisotropic nature of Gaussian splats. This design significantly improves texture efficiency, reducing memory consumption while enhancing image quality. Experiments on multiple benchmark datasets demonstrate that A$^2$TG consistently outperforms fixed-texture Gaussian Splatting methods, achieving comparable rendering fidelity with substantially lower memory requirements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets the inefficiency of assigning uniformly sized, square per-primitive textures in Gaussian-splatting pipelines. The authors propose A2TG on top of 2DGS: (1) start from a trained 2DGS model and attach per-Gaussian RGBA texture patches; (2) tie texture upscaling and aspect ratio to positional-gradient signals; (3) implement a lightweight, discrete allocation over {1,2,4}×{1,2,4} with a thresholded trigger and a periodic upscaling schedule. Experiments under fixed-memory and fixed-#Gaussians protocols on Mip-NeRF360, Tanks&Temples, and DeepBlending show comparable or better rendering quality at lower memory.

### Strengths
- This paper is well-written and easy to understand.
- This paper introduces anisotropic textures and achieves a slight reduction in memory footprint without compromising rendering quality.
- The authors tie texture resolution to the gradient, enabling detail-aware texture optimization, which makes sense.
- The ablations are appreciated to show the contribution of each part.

### Weaknesses
- Missing references for some important related works:
  - [MM 2024] AbsGS: Recovering Fine Details for 3D Gaussian Splatting, by Zongxin Ye et al. AbsGS, similar to the `absolute value of its positional gradient` described in L229–230, is the key mechanism by which GS-based methods eliminate floating floaters.
  - HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering, by Yunzhou Song et al. Like the present work, HDGS adopts a Texture-GS framework built on 2DGS, and therefore needs detailed discussion and comparison in the **Related Work** section.
- While I appreciate the core idea proposed by the authors, the ablation in Table 3 suggests that the upscaling strategy and anisotropic textures contribute only marginally to the rendering quality. Their benefits are reflected primarily in reduced memory overhead. The improvements reported in Table 1 likely stem more from the inherently efficient memory management of 2DGS and the densification mechanism of 3DGS-MCMC.
- There is no visualization of the textures. I understand it may be difficult to visualize textures at different levels of detail, but I do not think rendering quality is the primary aspect to focus on when introducing texture maps. We should care more about the quality of the textures themselves.
- Also, no comparison on FPS and training time.
- Minor typo errors:
  - L53: `As the results` -> `As a result`
  - L81: `build upon 3DGS` -> `builds upon 3DGS`
  - L184-185: `are now calculated` -> `is now calculated`
  - L197: `the pixel and alpha value` -> `the pixel and alpha values`
  - L261: `the memory size of each algorithms resulting parameters`

### Questions
1. Could you specify the exact values of the thresholds (e.g., the anisotropy thresholds **kA, kS**)?
2. What motivates restricting texture sizes to the discrete set **{1,2,4}×{1,2,4}**? Did you try larger grids or continuous allocation, and how do they affect quality vs. memory?

### Soundness
3

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
1

### Summary
The paper proposes Adaptive Anisotropic Textured Gaussians (A2TG), which equips each Gaussian primitive with an anisotropic, gradient-adaptive texture, improving memory efficiency and visual fidelity over fixed-square textures in real-time 3D scene rendering.

### Strengths
- The paper introduces Adaptive Anisotropic Textured Gaussians (A2TG), a novel extension of textured Gaussian splatting that assigns each Gaussian an anisotropic texture whose resolution and aspect ratio are adaptively determined.
- The paper is well-structured, with a clear motivation for adaptive texture allocation, detailed explanations of the method, and thorough evaluation of results.

### Weaknesses
While the proposed A2TG effectively improves texture allocation and memory efficiency, the method relies on several heuristically chosen thresholds for gradient-driven selection and anisotropy-based upscaling, which may require dataset-specific tuning. Additionally, the iterative upscaling procedure introduces extra computational overhead, potentially limiting real-time performance for very large or highly detailed scenes. Finally, the approach does not explicitly handle occlusion interactions beyond local gradients, which could affect texture fidelity in heavily occluded regions.

The paper has some typos:
- Line 261, "LPIPs"
- Line 314, "... across varies textured-based 2DGS...."
- Line 322, "... latter having more nubmer"
- Line 364, "the visaul quality"
- Line 368-369, "Textured Guassians*"
- Line 428, " (w/o Uscaling)"
- Line 435-436, "methods typical introduce"
- Line 441, "among the-state-of-the-art methods"

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Adaptive Anisotropic Textured Gaussians (A2TG) to enhance texture representations by extending fixed square textures to adaptively allocated anisotropic resolutions, reducing memory consumption while improving image quality.
The resolution and aspect ratio are controlled by an adaptive gradient-based strategy during optimization.

### Strengths
1. This paper clearly identifies the problem of memory redundancy in textured Gaussians.
2. The proposed anisotropic textures demonstrate good results in terms of storage efficiency.
3. Using gradients to guide texture scaling is clear and well-motivated.
4. The paper is well-organized and readable.

### Weaknesses
1. Some experiments could be added for further validation:
1.1. The gradient-based densification follows the spirit of the adaptive density control procedure in 3DGS. Experiments re-enabling similar densification procedures in 2DGS or baseline models are desired and could further demonstrate the effectiveness of the proposed method.
1.2. Since the performance of Textured Gaussians also depends on the resolution of textures, how does the proposed method perform compared with Textured Gaussians using 2×2 resolutions? Would the proposed method still achieve consistent advantages?
1.3. What about results with more Gaussians (e.g. 1M)?
2. Some mathematical notations are unclear, making it difficult to evaluate the correctness of the equations.
2.1. For example, the c_{\cdot}(x_j) notations in Eq.6 and Eq.7 refer to different concepts but lack explanations. Could the authors update the equations consistently for easier evaluation? From the current version, I suspect Eq. 7 may correspond to dL/d\alpha instead of dc/d\alpha. Since the labels are used interchangeably, a consistent version with clear explanations would help the final evaluation.
The second bullet on line 242: should it be r < 1/k_A?
2.2. Various typos and hard-to-understand sentences remain; see Questions below.

### Questions
1. There are some contradictory claims about image-quality improvements (lines 53, 55–56, 66–67, and the Experiments section). Considering the proposed method is a special case of the fixed square resolution (albeit with more parameters), why does it improve image quality?
2. How is the number of 100 K Gaussians guaranteed, since SfM initialization may exceed 100 K? Do you use low-resolution images or other strategies?
3. According to the logic from lines 239–244, does the upscaling double the resolution when r is within [1/K_A, K_A], regardless of s_x or s_y values? This seems to contradict intuition.
4. Does the anisotropic texture introduce runtime overhead?

Typos
1. Use "texture-based" instead of "textured-based".
2. Line 314: replace “varies” with “various.”
3. Line 318: change “resulting” to “resulting in,” and add “a” before “fixed.”
4. Lines 322–323: the sentences are fragmented and hard to follow—please clarify why A2TG is the best.
5. Line 367, 368, then to than

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
5

### Summary
The paper introduces an enhanced 2D Gaussian splatting technique that incorporates texture maps with adaptively controlled resolutions. The intersection point of a ray with the tangent plane of the 2D Gaussian splat is computed within the CUDA rendering pipeline. Using the tangent plane coordinates (uv), the corresponding texture color is retrieved from the texture map. The resolution of the texture maps is dynamically adjusted based on the accumulated gradients of the local uv coordinates. This approach supports 'anisotropic' resolutions, enabling different levels of detail along the u and v axes. Experimental results demonstrate that the proposed method achieves superior rendering quality compared to state-of-the-art Textured Gaussians, while also reducing storage requirements.

### Strengths
- Vanilla 3DGS and 2DGS require a large number of primitives to capture the high-frequency appearance and geometry of 3D scenes. This limitation arises from their restricted texture representation, which uses only a single color from a fixed viewpoint. Textured Gaussians, such as GStex, HDGS, Billboard Splatting, and SuperGaussians, represent significant advancements in this domain, enhancing Gaussian splatting with textures for improved 3D reconstruction. However, these methods do not account for the adaptive control of GS textures, leading to inefficiencies, as different primitives require varying levels of texture detail. Building on this insight, this paper introduces adaptive textures for GS, enabling variable resolutions for each primitive and along different axes. This approach is both innovative and practical.

- The results are impressive and highly satisfactory. As demonstrated in Tables 1 and 2 and Figure 2, A^2GS achieves superior performance with reduced storage requirements thanks to its adaptive control of GS texture resolutions. Additionally, Figure 3 highlights the method’s superiority in reconstructing fine details.

### Weaknesses
The first weakness lies in the methodology and is quite fundamental. The paper does not clarify how textures are fetched based on UV coordinates. Is it through bilinear interpolation or nearest neighbor? If bilinear interpolation is used, Equation 6 omits the terms \(\frac{\partial c_l(x_j)}{\partial \mu_{i,x}}\) and \(\frac{\partial c_l(x_j)}{\partial \mu_{i,y}}\), as \(c_l(x_j)\) is linearly related to the UV coordinates. On the other hand, if bilinear interpolation is not used, the optimization process and model capability are constrained. Since the positions of Gaussian splats (GS) are jointly optimized, if the gradients of textures are not related to the intersection coordinates, the texture map gradients cannot effectively guide GS movements for better fitting. Furthermore, using nearest neighbor (NN) queries results in discontinuous textures, which could reduce overall quality.

The second weakness concerns the experiments. The paper does not evaluate rendering speed, an essential aspect of the approach. Because GS textures have varying resolutions, the indexing of each GS texture is non-uniform. As a result, within each CUDA grid, it is impossible to define a fixed-length shared array for threads within the grid and move the data in parallel from the global to the shared. Consequently, textures must be fetched from the global CUDA memory, which is slower. Additionally, calculating the cumulative sum of each GS texture resolution to obtain the global base index for each GS texture adds computational overhead. These factors likely impact rendering speed and should be evaluated to demonstrate the tradeoff between performance and speed. While a reduction in speed is acceptable to an extent, maintaining real-time performance (>30 FPS) is crucial. If the speed drops below this threshold, the limitation should be explicitly acknowledged as a significant constraint.

The third weakness pertains to the conclusion section. A visible but minor limitation is that the method does not address reducing texture resolution dynamically. For instance, if a GS is optimized to have finer textures during an incorrect stage of the process but later becomes smaller or moves to textureless regions, the high-resolution texture may no longer be necessary. Addressing this issue could further improve efficiency. Additionally, while the proposed method makes substantial contributions to GS textures, it raises the possibility of combining adaptive textures with flexible 2D primitives, such as Deformable Radial Kernel Splatting. This combination could enable both flexible boundary shapes and sharpness while preserving richer texture representations. These two points should be considered as limitations and opportunities for future work.

### Questions
My suggestions are as follows: (1) provide a detailed explanation of the texture color query process using UV coordinates; (2) include an evaluation of the rendering speed; and (3) revise the conclusion section. I believe addressing these points will significantly enhance the quality of the paper, and I would be inclined to raise the score. While I acknowledge the novelty and contribution of the paper, improving these aspects is essential to further strengthen its overall quality.

### Soundness
3

### Presentation
3

### Contribution
3
