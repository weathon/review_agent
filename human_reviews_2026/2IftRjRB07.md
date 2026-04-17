# FlashWorld: High-quality 3D Scene Generation within Seconds

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 6

## Abstract
We propose FlashWorld, a generative model that produces 3D scenes from a single image or text prompt in seconds, $10 \sim 100\times$ faster than previous works while possessing superior rendering quality.
Our approach shifts from the conventional multi-view-oriented (MV-oriented) paradigm, which generates multi-view images for subsequent 3D reconstruction, to a 3D-oriented approach where the model directly produces 3D Gaussian representations during multi-view generation.
While ensuring 3D consistency, 3D-oriented method typically suffers poor visual quality.
FlashWorld includes a dual-mode pre-training phase followed by a cross-mode post-training phase, effectively integrating the strengths of both paradigms.
Specifically, leveraging the prior from a video diffusion model, we first pre-train a dual-mode multi-view diffusion model, which jointly supports MV-oriented and 3D-oriented generation mode. 
To bridge the quality gap in 3D-oriented generation, we further propose a cross-mode post-training distillation by matching distribution from consistent 3D-oriented mode to high-quality MV-oriented mode. 
This not only enhances visual quality while maintaining 3D consistency, but also reduces the required denoising steps for inference.
Also, we propose a strategy to leverage massive single-view images and text prompts during this process to enhance the model's generalization to out-of-distribution inputs.
Extensive experiments demonstrate the superiority and efficiency of our method.
Our code is released at https://github.com/imlixinyang/FlashWorld.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses text- and image-to-scene generation. Previous methods that leverage powerful video foundation models (VDMs) face a trade-off: multi-view-oriented approaches lack 3D consistency, while 3D-oriented methods often yield poor visual quality. This paper introduces pre-training and post-training strategies for VDMs to improve 3D consistency and accelerate video generation. For pre-training, the authors add a 3DGS decoder (the "3D-oriented" branch) to inject 3D priors and enhance the latent video diffusion model’s 3D consistency. For post-training, they distill a multi-view-oriented teacher into a 3D-oriented student to speed up generation. Comprehensive experiments show the proposed methods outperform prior state-of-the-art approaches in both qualitative and quantitative evaluations.

### Strengths
- The two proposed training strategies are novel, efficient, and effective for 3D scene generation.

- The authors present comprehensive experiments that convincingly support their claims.

- Comparison results show the proposed method outperforms both image-to-3D and text-to-3D approaches, producing finer-detail renderings and faster generation times.

### Weaknesses
- The writing flow is poor and Sec. 3.2 feels chaotic. For example, the sentence To generate a 3D scene, the 3D-oriented multi-view generation process alternates between denoising and noise injection steps to enhance sample quality.'' is confusing: is the goal to generate a 3D scene, to enhance sample quality, or both? 

- For Sec. 3.3, the reason the model improves the quality of out-of-domain (OOD) data is unclear. As I understand it, the approach uses camera-trajectory augmentation to enhance the model’s generalizability and also discards the GAN loss during training. Which of these two changes is more important for OOD performance? Data augmentation is easy to see as a way to improve generalization, but why does including a GAN loss lead to poor generalizability?

### Questions
1. There may be a typo on lines 254--255. You refer to λ in relation to Eq.(6), but Eq.(6) does not contain λ. 

2. Missing some relative references:
- VideoMV: Consistent Multi-View Generation Based on Large Video Generative Model,  which proposes a 3D-aware sampling strategy;
- AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction, using 4DGS to alleviate the inconsistency from multi-view video generation. 

Please consider add the above relative references.

In summary, the authors propose interesting pre-training and post-training strategies for video-diffusion model training to improve 3D consistency and quality in scene generation, achieving state-of-the-art results in both quantitative and qualitative experiments. However, the writing flow is poor: some sentences are confusing, and I have concerns about the model’s generalizability. As a result, I am inclined to give a borderline-accept score, though I would be happy to raise it if the authors address these main concerns.

### Soundness
3

### Presentation
2

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
The paper tackles scene generation from text and images. Existing video foundation model approaches trade off between multi-view methods, which struggle with 3D consistency, and 3D-focused methods, which often compromise visual fidelity. To overcome this, the authors propose pre-training and post-training techniques for VDMs: a 3DGS decoder (the “3D-oriented” branch) is added during pre-training to inject geometric priors and improve 3D coherence, and a teacher–student distillation step transfers knowledge from a multi-view-oriented teacher to a 3D-oriented student to speed up generation.

### Strengths
- The paper is clearly written and easy to follow.

- The organization is logical, and most design choices are validated with ablation studies.

- The method shows notable improvements over prior work, particularly in preserving fine-grained details.

### Weaknesses
The paper is novel and effective; the following points are offered as constructive suggestions for further improvement.

- The diversity and scale of generated scenes are still constrained by the coverage of existing datasets.

- The model currently struggles with accurately generating fine-grained geometry, mirror reflections, and articulated objects. These issues may be alleviated by incorporating depth priors and more 3D-aware structural information

- Although FlashWorld does not use explicit depth supervision, its 3D Gaussian Splatting (3DGS) outputs can be used to extract depth maps. However, the quality of the resulting depth information could be improved.

### Questions
There may be an error in Eq. (6), where is \lambda ?

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
This work presents a generative model that produces high-quality 3D scenes from a single image or text prompt in just seconds—10-100× faster than prior methods. FlashWorld’s key innovation is a distillation strategy that transfers high visual fidelity from a multi-view-oriented diffusion teacher to a 3D-oriented student, improving 3D consistency. The authors also introduce an out-of-distribution co-training strategy to improve generalization to scenes beyond the training distribution. Extensive experiments show FlashWorld outperforms state-of-the-art methods in both generation quality and inference speed.

### Strengths
- The paper is clearly written and easy to follow.
- Experimental validation is thorough, with a comprehensive benchmark against many competing methods.
- The out-of-distribution co-training approach is an effective and efficient way to improve generalization and robustness.

### Weaknesses
-  Although the model demonstrates strong and generalizable generation capabilities, its video-rendering performance is not extensively evaluated in the main paper.
- As noted in the limitations, the model still struggles with fine-grained geometry, mirror reflections, and articulated objects.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FlashWorld, a framework for fast and high-quality 3D scene generation that addresses the critical speed-quality trade-off in current methods. The authors propose a cross-mode distillation approach that leverages both MV-oriented (multi-view) and 3D-oriented generation modes. The key innovation lies in using dual-mode pre-training followed by cross-mode post-training, where the MV-oriented mode serves as a teacher to provide visual quality while the 3D-oriented mode acts as a student to ensure geometric consistency.

### Strengths
- The idea of training a dual-mode model and using cross-mode distillation for post-training is creative and well-motivated. Using the MV-oriented mode as teacher for visual quality while training the 3D-oriented mode for consistency is an elegant solution to the speed-quality trade-off.

- Achieving ~9 second generation time while maintaining SOTA quality represents a significant practical contribution. The 10-100× speedup over prior work (CAT3D, Wonderland, etc.) makes this approach much more suitable for real-world applications.

### Weaknesses
- Section 3.2 (cross-mode post-training) needs more detail. This stage appears to integrate DMD2 with the dual-mode diffusion model, but the step-by-step procedure is unclear. Please add a training algorithm box or pseudocode to clarify the process.
- Include NVS comparisons with video-based methods such as TrajectoryCrafter, GEN3C, and ViewCrafter.
- Provide depth visualizations of generated scenes and compare against baselines.

### Questions
- Does the Out-of-Distribution (OOD) data improve only text-to-3D, or does it also help image-to-3D? The gains for text-to-3D are clear, but the impact on image-to-3D is not.
- Line 249 is confusing: “we additionally update an MV-oriented student model at a lower frequency.” From Fig. 3, it seems the 3D- and MV-oriented modes share the same DiT backbone, whose latent output feeds the 3DGS decoder. If you update the 3D-oriented model, does that also update the DiT (and therefore the MV-oriented model)? Please clarify which parameters are shared or frozen during training and how updates are applied.
- How does the method handle scenes with significant occlusions or transparent objects?

### Soundness
3

### Presentation
3

### Contribution
3
