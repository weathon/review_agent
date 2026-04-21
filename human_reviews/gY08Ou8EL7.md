# LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Generalizable rendering of an animatable human avatar from sparse inputs relies on data priors and inductive biases extracted from training on large data to avoid scene-specific optimization and to enable fast reconstruction. This raises two main challenges: First, unlike iterative gradient-based adjustment in scene-specific optimization, generalizable methods must reconstruct the human shape representation in a single pass at inference time.
Second, rendering is preferably computationally efficient yet of high resolution.
To address both challenges we augment the recently proposed 
dual shape representation, which combines the benefits of a mesh and Gaussian points, in two ways. 
To improve reconstruction, we propose an iterative feedback update framework, which successively improves the canonical human shape representation during reconstruction.
To achieve computationally efficient yet high-resolution rendering, we study a coupled-multi-resolution Gaussians-on-Mesh representation.
We evaluate the proposed approach on the challenging THuman2.0, XHuman and AIST++ data. Our approach reconstructs an animatable  representation from sparse inputs in less than 1s, renders views with 95.1FPS at $1024 \times 1024$, and achieves  PSNR/LPIPS*/FID of 24.65/110.82/51.27 on THuman2.0, outperforming the state-of-the-art in rendering quality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a generalizable human rendering approach using a learned iterative feedback update mechanism. The method leverages a coupled multi-resolution Gaussian-on-mesh representation and demonstrates state-of-the-art results on benchmark datasets such as THuman 2.0 and AIST++.

### Strengths
1.	The proposed iterative update net is a novel idea and is clearly motivated.
2.	The writing is clear and easy to follow, with a well-structured introduction and related work section.
3.	Extensive experimentation on benchmark datasets shows clear performance improvements over previous methods (Table 1, Table 2, Table 3).
4.	The authors provide a comprehensive ablation study, which helps justify the effectiveness of iterative step choice and subdivision.

### Weaknesses
1. In the related works section, the authors categorize human rendering into "per-scene optimized" and "generalizable" approaches. However, there is no discussion on "large reconstruction model-based" human rendering approaches [1,2]. It would be valuable to compare the proposed method’s performance (in terms of inference time, training time, resolution and PSNR) with such large reconstruction models.

2. The proposed method employs a multi-resolution Gaussian-on-mesh representation, but a direct comparison with traditional mesh-based human representations [3, 4, 5] is missing. It would strengthen the paper if the authors could discuss the advantages or trade-offs between these representations ((in terms of rendering quality, animation flexibility, FPS).

3. The method is built on SMPL/SMPLX models, which might limit its effectiveness in rendering loose clothing, such as skirts or dresses (see  examples in https://zhaofuq.github.io/NeuralAM/). Including examples of such scenarios would enhance the evaluation of the method’s generalizability.

[1] Weng, Zhenzhen, et al. "Single-view 3d human digitalization with large reconstruction models." arXiv preprint arXiv:2401.12175 (2024).

[2] Chen, Jinnan, et al. "Generalizable Human Gaussians from Single-View Image." arXiv preprint arXiv:2406.06050 (2024).

[3] Liao, Tingting, et al. "Tada! text to animatable digital avatars." 2024 International Conference on 3D Vision (3DV). IEEE, 2024.

[4] Zhang, Xuanmeng, et al. "Getavatar: Generative textured meshes for animatable human avatars." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[5] Liao, Tingting, et al. "High-fidelity clothed avatar reconstruction from a single image." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
While the paper demonstrates promising results, I am curious about the method’s generalizability across diverse scenarios. In particular, on the project page, it would be insightful to evaluate the animation quality of cross-domain examples (e.g., drastically different body types, dynamic poses, or motion sequences). Addressing these aspects would provide a clearer understanding of the robustness and adaptability of the proposed approach

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
4

### Summary
This paper presents a method for generating high-quality, animatable 3D human avatars from sparse images. It proposes a multi-resolution Gaussians-on-Mesh representation, where low-resolution mesh is used for efficient geometry and high-resolution Gaussians for high-quality avatars. This paper also introduces an iterative feedback update mechanism to refine the model.

### Strengths
1. The method achieves fast 3D human avatar reconstruction from sparse inputs, making it suitable for real-time applications.
2. The multi-resolution Gaussians-on-Mesh representation balances efficiency and rendering quality.

### Weaknesses
1. The experimental comparisons are incomplete without recent works[1,2,3] that report better metrics. A comparative analysis or discussion of technical differences is needed.
2. The cross-domain generalization shows concerning facial artifacts despite the proposed iterative feedback. What causes these failures? If it's a data bias issue, why showcase this case? Consider discussing limitations and potential solutions (e.g., face-aware regularization).
3. The ablation study focuses only on hyperparameters rather than analyzing the effectiveness of core components. Need evaluation of:
Iterative feedback mechanism; Coupled-multi-resolution representation; Module-wise contribution analysis.


[1] 3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting, CVPR2024
[2] GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh, CVPR2024
[3] HUGS: Human Gaussian Splats, CVPR2024

### Questions
Please refer to my weakness part. I also have some questions about the videos supplied.
1. Why not show all the input images in Cross-domain generalization part? Without all input frames, it is hard to evaluate the performance. 
2. Please make the input number of images clear in Novel pose synthesis part. The novel pose performance is impressive if it is achieved by a single reference image.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles the problem of fast, sparse-view 3D articulated avatar modeling. Specifically, the paper proposes to improve Gaussian-on-Mesh method by using a dual-resolution mesh: the low-resolution mesh is used for efficient feed-forward computation, while the high-resolution mesh anchors Gaussians to represent fine details. Furthermore, the approach the sparse input images as cues to extract pixel-aligned features to iteratively improve the 3D avatar. After training, the network can reconstruct 3D avatars for a given identity in under 1-2 seconds.

The experimental results show quantitative and qualitative improvement over previous methods.

After reviewing the rebuttal: my concerns are sufficiently addressed by the new quantitative results provided by the authors. Overall, this submission presents an effective method for generalizable 3D human modeling, and shows non-trivial improvement over the prior art. 
I therefore raise my score to 8 -- accept, good paper.

### Strengths
The presented method has the following strengths:
- The method shows fast, efficient reconstruction, and this is always a very welcome property for 3D avatar modeling.
- The method shows promising results on reconstructing out-of-distribution avatars.
- The method gets rid of the Gaussian split/growing heuristics, which are sometimes annoying to tune/control.
- The qualitative and quantitative results verify the effectiveness of their proposed design choice.

3D avatar modeling, the problem tackled in this paper, is an important topic in vision/graphics due to its wide impact on many fields (entertainment, sports, virtual try-on, etc). Overall, the presented approach makes sense and the results look promising,

### Weaknesses
The paper has the following weakness:
- Missing comparisons to GoMAvatar: as mentioned in line 121-123, the presented method is derived from GoMAvatar. It therefore makes sense to compare against GoMAvatar. The iterative-feedback bears a close similarity to [1], which also utilizes pixel/texel align features to improve fidelity. It would be great if the paper could discuss about these relevant works.
- The writing can be further improved: for example, the sentence in line 143-145: ```As a fix, to regularize the Gaussians and to ease the animation, pairing with parametric models like FLAME or SMPL helps``` may not be as straightforward as saying ```Prior work [cite] regularizes the Gaussians and enables animation using parametric models such as FLAME and SMPL```. Also, in line 180-181 ```given as additional input the target camera ...``` seems unnecessary, as line 175-179 already described the input. 
- Mesh topology limits the geometry that can be captured, as shown by the hairs in Figure 4 (also pointed out in Appendix C).
- Missing memory profile: perhaps I have missed it, but I did not see discussions related to the memory/GPU vram consumptions for the proposed method. How much VRAM does it take to train/render the avatars? Does it work on commodity-level GPUs?
- Potentially susceptible to body pose misalignment issues: when the body pose is imperfect/does not overlap with the correct body part, the feedback network will extract features from incorrect locations, and thus impact the reconstruction quality.

[1] Drivable Volumetric Avatars using Texel-Aligned Features, SIGGRAPH 2022

### Questions
Please address/discuss about the weakness mentioned above if they are not already in the paper.

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
This paper tackles the challenge of creating an animatable human avatar from just a few images. To make reconstruction faster and more accurate, they introduce an iterative feedback approach that gradually refines the results. For high-quality, efficient rendering, they also develop a multi-resolution Gaussians-on-Mesh technique, which balances detail and speed.

### Strengths
- The feedback mechanism in this generalizable method is notably innovative, refining predictions by addressing visible errors from input views, resulting in more accurate and detailed reconstructions within a second.
- Achieving human avatar reconstruction from images in different poses is highly practical, offering greater flexibility than multi-view approaches.
- The paper is well-written, well-structured, and clear.

### Weaknesses
- The model does not appear to have been trained on two datasets and can support both tasks simultaneously, limiting its ability to handle mixed input types effectively.
- The method's heavy dependence on SMPL priors poses a risk; if the estimations are inaccurate, especially for loose clothing, it can lead to significant errors
- The method shows significant performance degradation when applied across domains, particularly affecting the facial region, as in the video. Besides, the novel pose representation displays black hands in the video. 
- Minor issues: line 156 "estimate" to "estimation", line 234 translation -> scale?

### Questions
- In Figure 1, the mesh appears to be a GT-mesh rather than an SMPL model. Is the GT mesh utilized in your approach？
- Given that Gaussian splatting has a strong tendency to overfit, how is pose generalization achieved in your method? Are the displayed novel poses primarily from the dataset, or do they include truly unseen poses?

### Soundness
3

### Presentation
3

### Contribution
3
