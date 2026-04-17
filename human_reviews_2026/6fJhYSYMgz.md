# DS-VTON: An Enhanced Dual-Scale Coarse-to-Fine Framework for Virtual Try-On

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
n/a

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents DS-VTON, an enhanced dual-scale coarse-to-fine framework designed to overcome the two core challenges in virtual try-on: achieving accurate garment-body alignment and preserving fine-grained textures and patterns. 

The proposed method first generates a low-resolution try-on image that establishes robust semantic correspondence between the garment and the target human pose, leveraging reduced detail to facilitate structural alignment. 

In the second stage, a novel blend-refine diffusion process reconstructs high-resolution results by denoising the residual between scales through noise–image blending, which emphasizes texture fidelity and corrects fine-detail errors inherited from the coarse stage. 

Notably, DS-VTON operates without any human parsing maps or segmentation masks, offering a fully mask-free generation pipeline.

### Strengths
- 1. This paper proposed a novel dual-scale, mask-free framework that enhances the coarse-to-fine process and is particularly well-suited for the try-on task.

- 2. The mask-free formulation is a clear practical advantage—eliminating dependence on potentially brittle human-parsing or segmentation modules improves robustness and simplifies deployment.

- 3. The blend-refine diffusion re-formulation is novel and well-motivated; explicitly bridging low- and high-resolution distributions with a tunable $α / β$ mixture gives the model tighter control over the coarse-to-fine transition and consistently reduces texture artifacts.

- 4. Despite its additional stage, the method remains computationally reasonable: both stages share the same lightweight SD-1.5 U-Net backbone, inference is only ~5 s per image on a single A6000, and runtime is on par with or faster than most recent competitors.

### Weaknesses
1. Training data are synthetically amplified with IDM-VTON generations; although the paper acknowledges the risk, visible entanglement still occurs—hair, accessories, or background sometimes change, indicating less-than-perfect disentanglement that could undermine identity preservation in real applications.

2. The low-resolution stage is constrained to a fixed σ = 2; no adaptive or content-aware schedule is explored, so structural detail can be lost for unusually complex garments, and the choice feels ad-hoc given limited ablation (only σ ∈ {1, 2, 4} tested).

3. Coefficients α and β are set to a constant 0.5 for all images; this rigid trade-off may be sub-optimal when garment or pose complexity varies, yet no mechanism to predict or learn sample-specific values is provided.

4. The method inherits the gender, body-type, and skin-tone biases of the SD-1.5 backbone; failure cases on out-of-distribution body shapes are shown, and no bias-mitigation strategy or inclusive evaluation is offered.

5. Runtime, although reasonable, is still sequential-two-stage; latency doubles compared with single-pass competitors, and no distillation or joint-training scheme is investigated to enable real-time deployment on edge devices.

### Questions
- 1. In the authors’ Eq. (1), α and β appear to be crucial, yet Table 3 only gives the values found by ablation on a single split. Do these numbers generalize to every dataset, or must they be re-tuned whenever the data distribution changes? The paper offers neither statement nor evidence.
- 2. AAAI 2025 [1] also presents an identical coarse-to-fine VTON framework—why has no comparison been made?

[1] Li G, Wang Y, Luan J, et al. Cascaded diffusion models for virtual try-on: Improving control and resolution[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(5): 4689-4697.

- 3. Since the Low-Resolution Stage only produces a coarse result, why not drop ReferenceNet and adopt a CatVTON-like pipeline without the garment-encoding branch, instead of deliberately slowing the model down?
- 4. How computationally complex is this two-stage optimization architecture, which incorporates such a sophisticated ReferenceNet?
- 5. Mask-free methods can still be evaluated using SSIM and LPIPS.
- 6. Quantitative results for the Low-Resolution Stage are missing.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
summary:
This article proposes a coarse-to-fine generation framework that generates coarse results at low resolution and uses them as high-resolution generation input to generate fine results.

### Strengths
strength：
1.The coarse-to-fine framework is interesting and effective
2. The experiments are sufficient and the experimental results are excellent

### Weaknesses
weakness：
1. The overall approach is not innovative and is similar to IMAGDressing and MagicCloth.
2. The results in Table 1 are quite different from those of FiTDiT.
3. The generated results of the anime character in Figure 8 have obvious defects, such as the blue long sleeves.
4. The work based on SD is slightly behind, and experiments based on DiT or Flux may be a better choice

### Questions
See weakness

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
4

### Summary
This paper presents a method that perform image-based virtual try-on using two different diffusion models that deal two scale image content. The demonstrated results are nice compared to previous work. Another advantage is that the proposed method does not require an additional human body parsing/segmentation mask.

### Strengths
- The overall proposed method is simple and straightforward.
- The proposed method does not require a human parsing mask, makes it easier to deploy to real-world usage.
- The empirical results are nice against other existing methods.

### Weaknesses
- The overall method novelty is limited since the multi-scale image processing has been studied for a very long time. And the main technical difference of this method is to use two different diffusion models to handle input images that captures content under two different scales.
- The justification and evaluation of the use of human parsing learned at the diffusion model is not demonstrated in the paper.
- The necessity of the Blend-refine diffusion reformulation is doubtful. It is recommended to have comparison with other guidance methods to justify the needs of proposing a new method for this.
- Although the results might not be simialrly nice, I am curious about what is the results of latest large image editing models, such as Nano Banana, Qwen image edit, and other similar models the authors can access before submission? Based on my understanding, these models also can generate virtual try-on methods. I think extensive comparison with these models are also valuable.

### Questions
- I am wondering whether the proposed method can only work with SDXL? How to extend it to other more modern models, such as FLUX?
- I am curious whether it is possible to compare the high resolution stage with other guided generation methods, such as controlnet? For example, it could be possible to generate many low resolution and high resolution garment pairs for training a controlnet?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DS-VTON, a dual-scale framework to address the problem of single-stage diffusion model for virtual try-on tasks. DS-VTON attempts to disentangle the global structure alignment process from the fine-grained texture restoration. For this purpose, DS-VTON employs a two-stage paradigm: 1) In the low-resolution stage, the model generates a coarse try-on result by suppressing high-frequency content, while 2) in the high-resolution stage, DS-VTON transforms the low-resolution output into high resolution, restoring fine textures and correcting fine-detail errors from the first stage.

### Strengths
1. The writing of this paper is easy to follow. The motivation is well clarified, and the proposed method is easy to understand.

2. The quantitative and qualitative comparisons with state-of-the-art methods on two public virtual try-on datasets demonstrate the effectiveness of the proposed method.

### Weaknesses
1. One of the major problem of this paper lies in the novelty of the proposed DS-VTON method. The idea of first generating low-resolution images and then transforming them into a high-resolution version has already been widely explored in the field of high-resolution image generation. The multi-scale latent upsampling technique used in [1][2][3] is quite similar to the dual-scale DS-VTON method. Could the authors make a comparison with these approaches to elaborate more clearly on their technical contributions?

[1] Du R, Chang D, Hospedales T, et al. Demofusion: Democratising high-resolution image generation with no $$$[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024: 6159-6168.

[2] Guo L, He Y, Chen H, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation[C]//European conference on computer vision. Cham: Springer Nature Switzerland, 2024: 39-55.

[3] Jeong J, Han S, Kim J, et al. Latent space super-resolution for higher-resolution image generation with diffusion models[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 2355-2365.

2.  Another problem of DS-VTON is that it relies on the paired data to perform the model training. Although a generative model (i.e., IDM-VTON) is used to construct the input data, the quality of the generated pseudo-reference images will heavily affect the performance of DS-VTON. For example, IDM-VTON may preserve some clothing areas of the target samples in the pseudo-reference images. Training with these imprecise inputs, the model will consider that these areas should be preserved as background regions, thus establishing incorrect associations between the target clothing information and the human body or background pixels. This will lead to occlusions and clothing ghosts in the final results.

3. proposed DS-VTON requires to train/fine-tune two diffusion models separately, and also needs to perform inference on these two models. This may introduce additional computational complexity and inference time. Could the authors provide some computational efficiency analysis of the training and inference stages?

4. Could the authors present some failure cases and the related discussions to demonstrate the limitations of the proposed method?

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
