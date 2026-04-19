# SVG: 3D Stereoscopic Video Generation via Denoising Frame Matrix

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Video generation models have demonstrated great capability of producing impressive monocular videos, however, the generation of 3D stereoscopic video remains under-explored. We propose a pose-free and training-free approach for generating 3D stereoscopic videos using an off-the-shelf monocular video generation model. Our method warps a generated monocular video into camera views on stereoscopic baseline using estimated video depth, and employs a novel frame matrix video inpainting framework. The framework leverages the video generation model to inpaint frames observed from different timestamps and views. This effective approach generates consistent and semantically coherent stereoscopic videos without scene optimization or model fine-tuning. Moreover, we develop a disocclusion boundary re-injection scheme that further improves the quality of video inpainting by alleviating the negative effects propagated from disoccluded areas in the latent space. We validate the efficacy of our proposed method by conducting experiments on videos from various generative models, including Sora [4], Lumiere [2], WALT [8], and Zeroscope [12]. The experiments demonstrate that our method has a significant improvement over previous methods. Project page at https://daipengwa.github.io/SVG_ProjectPage/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a training-free framework to generate stereo video from single video sequence by using retrained video generation model. The proposed frame matrix leverages the power of video generation model and the joint optimization idea to generate semantically consistent and temporally smooth content, which offers valuable insights and can provide reference directions for future work.. However, besides the frame matrix, the other modules mentioned in methods are more like tricks and are all followed by existing works. The extensive results show that this paper achieve SOTA performance in training-free manner. However, the drawbacks are obvious, the performance is significantly worse than the training methods, and the process is slow due to the need for multiple iterations.

### Strengths
1. This paper proposes a training-free manner to generate stereo video from monocular video and achieve SOTA performance in training-free manner.

2. The proposed denoising frame matrix uses pre-trained video generation model as the inpainting model, which is the first one to do this in stereo video generation field and offers insight for leveraging video generation model to assist this task.

### Weaknesses
1. The analysis for denoising frame matrix is insufficient. Although the authors provide the reason for using video generation model, the theory and the high-level reason of why it works are not clear, please show the theoretical analysis of the proposed frame matrix.

2. Lack citations for the methodology followed by other methods. The presentation in the paper contains certain misleading and deceptive elements. Line 153-161, the viewpoint transfer part is widely used in novel view synthesis [R1, R2] and stereo image generation [R3]. Additionally, the proposed boundary re-injection is just a trick, which can not be a contribution. These parts should be revised for better presentation.

[R1] Tucker, R.; and Snavely, N. 2020. Single-view view synthesis with multiplane images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 551–560.

[R2] Han, Y.; Wang, R.; and Yang, J. 2022. Single-view view synthesis in the wild with learned adaptive multiplane images. In ACM SIGGRAPH 2022 Conference Proceedings, 1–8.

[R3] Wang, X.; Wu, C.; Yin, S.; Ni, M.; Wang, J.; Li, L.; Yang, Z.; Yang, F.; Wang, L.; Liu, Z.; et al. 2023b. Learning 3D Photography Videos via Self-supervised Diffusion on Single Images. arXiv preprint arXiv:2302.10781.

3. Insufficient evaluation metics. In [R1], [R2], [R3], they will use SSIM, PSNR, LPIPS evaluation metrics in novel-view image, which can also be used in stereo video generation. What's more, the temporal consistency is just justified in user study, the quantitative results such as Fréchet Video Distance should also be given.

4. Lack comparison with highly related methods. The methods compared in this paper is too simple. As mentioned in Q2, [R1], [R2], [R3] are the highly related stereo image generation methods. Although they need to train, they own the advantages of high performance and fast inference speed, the authors should give the result in the paper and clarify the strength of their methods.

### Questions
See the weakness.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses a gap in the generation of stereoscopic videos, particularly corresponding to the advancements in VR/AR technologies. The authors introduce an interesting pose-free and training-free framework that aims to improve the generation of high-quality 3D stereoscopic videos from monocular video inputs. The author utilizes video generation models to enhance 3D consistency while addressing challenges such as occlusion and temporal stability.

### Strengths
- The paper is well-written and easy to follow. 
- The proposed frame matrix representation is reasonable, and the extensive experiments support its functionality. Therefore, this paper is sufficiently novel. 
- The proposed method demonstrates a strong understanding of the challenges specific to 3D video generation, including issues with depth estimation and video inpainting. 
- I believe that sufficient experiments and ablation studies are presented to support the approach.

### Weaknesses
The main weaknesses of the proposed method are the disocclusion boundary artifacts, slightly lower temporal consistency compared to Deep3D, and the need for further improvements in holistic perceptual consistency, especially for certain subjects like human faces.

### Questions
Missing some reference about multi-view synthesis, please consider reference them:

[1] Chen, Zilong, et al. "V3d: Video diffusion models are effective 3d generators." arXiv preprint arXiv:2403.06738 (2024).
[2] Zuo, Qi, et al. "Videomv: Consistent multi-view generation based on large video generative model." arXiv preprint arXiv:2403.12010 (2024).

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
This paper aims to synthesize 3D stereoscopic videos using an off-the-shelf monocular video generation model without finetuning. Concretely, the authors leverage the estimated video depth and propose a novel frame matrix to denoise both spatially and temporally, and thus the model is aware of the exsiting information on the left view and synthesizes consistent right-view videos accordingly. A disocclusion boundary re-injection scheme is also proposed to solve the boundary problem.

### Strengths
1. The proposed method is pose-free and training-free.

2. This paper is clearly written and easy to understand.

3. Extensive experiments demonstrate the effectiveness of the proposed frame matrix and the disocclusion boundary re-injection scheme.

### Weaknesses
1. As the model denoises along both temporal and spatial dimensions, one experiment that is missing is the investigation of varying the number of cameras between the left and right views. How does this variation impact the final quality and overall efficiency of the process? Is it feasible to use fewer internal camera views to save time?
2. Currently all the experiments have been conducted on the synthesized videos. It would be beneficial to explore how the results look like when applied to real-world videos.
3. The model heavily depends on a pre-trained depth estimation model, which can overlook thin structures and sometimes produce inaccurate results.

### Questions
1. Most videos feature a single movable object or minor movements. How does the proposed method perform when applied to more complex scenes, such as those with multiple moving objects and significant motions?
2. Typos:

L225 Denosing -> Denoising

L266 refence → reference

### Soundness
3

### Presentation
3

### Contribution
3
