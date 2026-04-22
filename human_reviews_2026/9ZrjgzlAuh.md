# Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We introduce Mono4DGS-HDR, the first system for reconstructing renderable 4D high dynamic range (HDR) scenes from unposed monocular low dynamic range (LDR) videos captured with alternating exposures. To tackle such a challenging problem, we present a unified framework with two-stage optimization approach based on Gaussian Splatting. The first stage learns a video HDR Gaussian representation in orthographic camera coordinate space, eliminating the need for camera poses and enabling robust initial HDR video reconstruction. The second stage transforms video Gaussians into world space and jointly refines the world Gaussians with camera poses. Furthermore, we propose a temporal luminance regularization strategy to enhance the temporal consistency of the HDR appearance. Since our task has not been studied before, we construct a new evaluation benchmark using publicly available datasets for HDR video reconstruction. Extensive experiments demostrate that Mono4DGS-HDR significantly outperforms alternative solutions adapted from state-of-the-art methods in both rendering quality and speed. The project page for this paper is available at https://liujf1226.github.io/Mono4DGS-HDR.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a system for reconstructing 4D HDR scenes from alternating-exposure monocular LDR videos without known camera poses. The method builds on 3D Gaussian Splatting and proposes a two-stage optimization framework: 
- A video-space stage, which learns dynamic HDR Gaussians in orthographic camera coordinates, eliminating the need for camera poses.
- A world-space stage, which transforms and refines these Gaussians jointly with camera poses using HDR photometric reprojection.
Additionally, a Temporal Luminance Regularization is introduced to ensure temporal consistency of HDR appearance.

### Strengths
-  it is the first to handle alternating-exposure monocular HDR reconstruction.

- Well-designed two-stage optimization, effectively bridging unposed monocular input to HDR Gaussian representation.

- Good experiment results, outperforming both 3DGS- and NeRF-based HDR methods in quality and speed, and comprehensive ablation studies, validating each component’s contribution.

### Weaknesses
- Dependence on multiple vision foundation models (DepthCrafter, RAFT, etc.) makes the pipeline complex and may limit real-time applicability.

- The approach assumes alternating exposure patterns; performance under random or adaptive exposure schedules is not analyzed.

- Large novel view rendering is not demonstrated. It remains unclear whether the reconstructed HDR scenes maintain geometric and photometric consistency under wide view changes

### Questions
- How sensitive is the method to inaccuracies in the exposure timing or camera response function estimation?

- Could the framework generalize to arbitrary (non-periodic) exposure sequences?

- How large is the performance drop if vision priors (depth/flow) are noisy or absent?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes HDR-4DGS, a two-stage framework for reconstructing HDR scenes from monocular alternating exposure videos. Unlike previous works that handle dynamic Gaussians implicitly, this method explicitly parameterizes the motion of Gaussians, enabling more accurate transformation of video Gaussians into the world coordinate system. Additionally, various optimization losses are introduced to ensure high-quality final rendering.

### Strengths
1. Explicitly parameterizing the motion of Gaussians not only improves rendering quality but also maintains a relatively fast rendering speed.
2. The invariance of 2D Gaussian covariance serves as a simple yet effective tool introduced by the authors, which is validated through ablation studies in the paper.
3. The entire paper is clear and easy to understand.

### Weaknesses
1. The division between dynamic and static regions relies on epipolar error maps, so the final results are heavily influenced by them.
2. The selection of dynamic Gaussians depends on threshold settings, which reduces the generalizability of the pipeline, as determining appropriate thresholds for each scene is not straightforward.

### Questions
One of my concerns is that if there is a large viewpoint difference between two frames, then when warping from frame t-1 to frame t, some regions may appear black because certain content in frame t-1 is not visible from the viewpoint of frame t. Would these regions affect the supervision of the TLR loss?

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
3

### Summary
HDR-4DGS presents a well-motivated approach for reconstructing renderable 4D HDR scenes from unposed monocular LDR videos with alternating exposures. The authors propose a two-stage Gaussian Splatting optimization framework, where dynamic HDR video Gaussians are first learned in orthographic camera space and then transformed to world space for joint optimization with camera poses. The method is complemented by temporal luminance regularization, ensuring temporal consistency of HDR appearance. The experimental evaluation is thorough, including both synthetic and real-world datasets, and demonstrates that HDR-4DGS outperforms  state-of-the-art methods.

### Strengths
1. This paper presents the first system to address 4D HDR reconstruction from unposed, single-camera, alternating-exposure LDR videos.

2. The proposed two-stage optimization is effective.

3. HDR-4DGS effectively handles varying brightness across frames, which would break conventional photometric reprojection losses.

4. The paper constructs a new benchmark for HDR video reconstruction including real and synthetic scenes.

### Weaknesses
1. Although HDR reconstruction is the core contribution, the paper primarily evaluates PSNR/SSIM on tone-mapped images. No HDR-specific metrics (e.g., PQ-PSNR, HDR-VDP).

2. While quantitative results are extensive, the paper provides limited qualitative discussion on typical failure cases.

3. The approach has not been evaluated on low-light, reflective, or transparent surfaces, which may limit applicability in certain real-world conditions.

### Questions
I would like to know how the proposed method performs when dealing with scenes that involve extremely fast and complex motion, where motion blur and ambiguity are present.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
HDR-4DGS tackles 4D HDR reconstruction from unposed monocular LDR videos with alternating exposures. This problem hasn't been tackled exactly before. Their temporal regularization and 2 stage training to model the world shows superior performance qualitatively and quantitatively over adapted baselines.

### Strengths
1. The problem setting is well defined and motivated.
2. The paper is well written and clear.
3. The evaluations done are adequate, both quantitatively and qualitatively.
4. The paper comprehensively ablates all the design features showing the importance/visual effect of each modification.

### Weaknesses
1. How does the optmization/loss curves look like with that many losses? Would it be possible to show the curves?
2. Are all scenes at 24-30fps? Have the authors tried any more challenging settings like faster motion (ex - moving cars for autonomous driving applications?), non-lambertian surfaces etc? Those would be nice to haves but not necessary of course.

### Questions
1. Just curious as to why GaussHDR tends to remove the foreground object altogether. Any insights?

### Soundness
3

### Presentation
3

### Contribution
3
