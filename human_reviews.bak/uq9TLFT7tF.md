# EG4D: Explicit Generation of 4D Object without Score Distillation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
In recent years, the increasing demand for dynamic 3D assets in design and gaming applications has given rise to powerful generative pipelines capable of synthesizing high-quality 4D objects.
  Previous methods generally rely on score distillation sampling (SDS) algorithm to infer the unseen views and motion of 4D objects, thus leading to unsatisfactory results with defects like over-saturation and Janus problem.
  Therefore, inspired by recent progress of video diffusion models, we propose to optimize a 4D representation by explicitly generating multi-view videos from one input image.
  However, it is far from trivial to handle practical challenges faced by such a pipeline, including dramatic temporal inconsistency, inter-frame geometry and texture diversity, and semantic defects brought by video generation results.
  To address these issues, we propose EG4D, a novel multi-stage framework that generates high-quality and consistent 4D assets without score distillation.
  Specifically, collaborative techniques and solutions are developed, including an attention injection strategy to synthesize temporal-consistent multi-view videos, a robust and efficient dynamic reconstruction method based on Gaussian Splatting, and a refinement stage with diffusion prior for semantic restoration.
  The qualitative comparisons and quantitative results demonstrate that our framework outperforms the baselines in generation quality by a considerable margin.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an explicit 4D object generalization method without using SDS loss. The method leverages SVD and SV3D to optimize the 4D representation from the generated videos. Experimental results demonstrate superior performance over previous baselines.

### Strengths
1.	The proposed pipeline is clearly motivated and reasonable.
2.	The writing is clear and easy to follow, with a well-structured introduction and related work section
3.	Extensive experiments shows clear performance improvements over previous methods (Table 1, Table 2).
4.	The authors provide a comprehensive ablation study, which helps justify the effectiveness of attention injection, color transformation, multi-scale renderer, refinement strategies.
5.	The supplementary materials are sufficient.

### Weaknesses
1.	As shown in the video, the results are flickering and blurry. It seems that the quality is not as good as Diffusion4D [1] (see the examples of the project page of Diffusion4D).  It would be valuable to add a discussion and analysis about the proposed method with Diffusion4D (e.g., generation time, temporal consistency, image fidelity, 3D consistency).

2.	The method is built on SVD and SV3D for 4D generation. It would strengthen the paper if the authors could discuss the different designs between SV4D [2] and proposed method. The authors are suggested to provide both qualitative and quantitative comparisons. In specific, the authors can quantitative comparison table between their method and SV4D on metrics like overall quality (view alignment, 3D consistency, multi-view consistency,  temporal consistency), animation flexibility (e.g. motion range, motion fidelity), and computational efficiency (e.g. FPS, generation time). 

3.	I noticed the examples are some simple images. I wonder if the proposed can handle more complex images like examples in https://mrtornado24.github.io/DreamCraft3D/ and other cases (with multiple objects, complex backgrounds, or challenging lighting conditions).

[1] Liang, Hanwen, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N. Plataniotis, Yao Zhao, and Yunchao Wei. "Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models." arXiv preprint arXiv:2405.16645 (2024).
[2] Xie, Yiming, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. "Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency." arXiv preprint arXiv:2407.17470 (2024).

### Questions
Another way is to first generate 3D models and then animate the mesh [4]. I wonder if this line of methods are more effective and flexible.  Could you discuss the potential advantages and limitations of their approach compared to the 3D-then-animate pipeline. Specifically, the authors are  suggested they address aspects like computational efficiency, animation quality, and flexibility in handling different types of objects or motions.

[4] Jiang, Yanqin, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. "Animate3d: Animating any 3d model with multi-view video diffusion." arXiv preprint arXiv:2407.11398 (2024).

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
5

### Summary
The paper proposes a 4D generation framework that avoids the need of score distillation. Instead, multi-view videos are generated from one input image and a 4D representation is optimized directly.

### Strengths
- A multiscale augmentation is proposed for rendering more details.
- A training-free attention injection strategy is introduced to ensure consistency.
- A Diffusion Refinement stage is included to refine the details.

### Weaknesses
- In Fig. 1, it seems that the multiscale renderer is not used for the "Diffusion Refinement" stage. What is the motivation for this change? Is it necessary?
- What is the difference of the proposed Diffusion Refinement stage and that in DreamGaussian4D?

### Questions
- For the main comparison results. Did the authors use the same video to compare the proposed method and Dreamgaussian4D? It seems that the generated object motions are very different. If not, is it possible to additionally compare a variant of DreamGausisan4D? In other words, can we use the foreground video rendered by the proposed method and reconstruct to 4D with the framework of DreamGausisan4D?

- For the Mario case, as shown in the supplementary video, contains flickering artifacts on the face regions. Why is this happening? Is it because the refinement stage suffers from the Janus issue?

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
The paper introduces EG4D, a multi-stage framework designed for generating 4D objects from a single image input without relying on score distillation sampling (SDS). EG4D employs three key stages: view generation using video diffusion models, coarse 4D reconstruction using Gaussian Splatting, and refinement with diffusion priors. The framework addresses common issues like temporal inconsistencies, over-saturation, and the Janus problem, achieving superior performance over existing 4D generation models. The authors demonstrate the effectiveness of EG4D through both qualitative and quantitative analyses, showing significant improvements in 4D object realism and consistency.

### Strengths
1. EG4D introduces a unique multi-stage approach that successfully avoids issues associated with score distillation, such as over-saturation and Janus artifacts.
2. The attention injection mechanism helps ensure temporal consistency.
3. Quantitative and qualitative results demonstrate that EG4D produces high-quality 4D content, achieving superior alignment with the reference view and more realistic motion realism compared to baselines like DreamGaussian4D and Animate124.
4. The evaluation is comprehensive. The evaluation metrics, including CLIP-I, PSNR, SSIM, FVD, and CD-FVD, along with a user study, offer robust evidence of the framework's advantages.

### Weaknesses
1. Lack of baseline methods. Recent approaches such as L4GM and efficient4D ( which also leverage 4D Gaussian Splatting) should definitely be included for comparison.
2. Lack of Comparative Analysis. The attention injection technique is novel, but without comparisons to other methods, its effectiveness remains uncertain.
3. Texture Inconsistencies. Despite color transformation, subtle color and texture shifts remain, questioning the robustness of temporal consistency.
4. Limited Support for High-Dynamic Motion. The model struggles with high-dynamic motion, restricting its applicability in scenarios needing greater flexibility.

### Questions
1. Could the authors provide more quantitative insights into the effects of the attention injection mechanism on temporal consistency?
2. How sensitive is the model’s performance to variations in the blending weight (α) during attention injection?

### Soundness
2

### Presentation
3

### Contribution
3
