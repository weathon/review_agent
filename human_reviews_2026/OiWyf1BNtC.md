# Realtime Video Frame Interpolation using One-Step Diffusion Sampling

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Video Frame Interpolation (VFI) involving large, complex motions remains a significant challenge due to the difficulty of modeling diverse pixel trajectories from limited inputs. Traditional methods struggle with low-order approximations, and recent Latent Video Diffusion Models (LVDM) improve it through a conditional generation modeling. Still, current LVDMs often prioritize pixel fidelity over motion coherence in their reconstruction objective, leading to artifacts in extreme motion scenarios. To address this, we propose RDVFI, a novel approach that leverages an LVDM to generate sparse latent keyframes which define high-order, continuous pixel trajectories. The estimated continuous pixel trajectories accurately index pixel movements from inputs to arbitrary timestamps, generating optical flows to warp input pixels into the target frame. By decoupling sequence motion generation from high-resolution rendering, RDVFI operates on a fixed, lower resolution, and fewer diffusion sampling steps, introducing significant efficiency gains. Extensive experiments demonstrate that RDVFI achieves state-of-the-art visual and numerical performance, with over 75\% of viewers selecting it as the best method in terms of motion and frame quality compared to leading baselines. Furthermore, RDVFI is the first LVDM-based VFI method to achieve real-time performance (17 FPS at $1024\times 576$), offering a $\times 44$ acceleration over the current state-of-the-art and also robustly handling challenging motions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work introduces an efficient video interpolation approach based on a video diffusion model, enabling real-time frame interpolation. The proposed pipeline consists of three main stages: (1) estimating sparse, low-resolution keyframes through a one-step video diffusion process; (2) extracting a complex motion field from these keyframes using a continuous motion estimator; and (3) synthesizing full-resolution intermediate frames from the motion field and the input frames ($I_0$ and $I_1$) using a frame synthesis network. In addition, the authors propose a two-stage training framework to address the unstable convergence may observed in simple end-to-end training.

### Strengths
Traditional non-generative methods preserve the appearance of input frames well but struggle to interpolate dynamic motions. In contrast, diffusion-based methods effectively model and interpolate complex motions but are computationally expensive and often fail to maintain the appearance consistency of input frames.

This paper proposes a novel approach that disentangles motion from the video diffusion model (VDM) and applies the estimated sparse motion to achieve efficient and effective video interpolation.
The main strengths of the paper can be summarized as follows:

1) Disentangling motion field from a one-step VDM is both novel and effective.

2) Integrating the motion field extracted from the VDM into a frame synthesis framework improves efficiency compared to full VDM-based methods.

3) The qualitative and quantitative results are impressive and demonstrate the effectiveness of the proposed method.

### Weaknesses
1) Insufficient experimental analysis on efficiency:
Although the proposed framework emphasizes efficiency as a key contribution, the paper lacks detailed quantitative evidence. The authors should include explicit efficiency metrics in Table 1, such as inference time (s/frame) and memory usage (VRAM in GB), to substantiate their claims.

2) Lack of detailed baseline categorization:
For video interpolation methods based on video diffusion models (VDMs), it is crucial to clearly categorize baselines into zero-shot, fine-tuned, and fully trained settings. Without this distinction, the fairness and clarity of the comparison become ambiguous, potentially undermining the paper’s original motivation. Reporting the total training compute—e.g., approximate FLOPs or other comparable measures—would further improve the transparency and fairness of the evaluation.

3) Limited ablation studies:
The initial estimation of $k$ keyframes using the VDM plays a central role in determining the motion field and thus in the overall effectiveness of the proposed method. However, the paper lacks sufficient analysis of how the number of keyframes ($k$) affects both efficiency and interpolation quality. A detailed ablation study on this factor would greatly enhance the understanding of the method’s underlying principles.

### Questions
1) Please include explicit efficiency metrics — such as inference time (s/frame), memory usage (GB), and training compute (e.g., approximate FLOPs or other comparable measures) — in the main manuscript to better support the claimed efficiency of the proposed framework.

2) Please clearly categorize the baselines into zero-shot, fine-tuned, and fully trained settings for a fair comparison. Additionally, consider including more zero-shot video diffusion interpolation works such as TRF [1] and ViBiDSampler [2], which are missing from the current submission.

3) If possible, add a brief ablation study on the number of keyframes ($k$). This would significantly improve the clarity and completeness of the paper by illustrating how $k$ influences both interpolation quality and computational efficiency.

4) It would be helpful to include failure case analysis. Presenting representative failure cases and describing in which situations such samples frequently occur would provide deeper insight into the limitations and behavior of the proposed method.

[1] Explorative Inbetweening of Time and Space, ECCV 2024.

[2] ViBiDSampler: Enhancing Video Interpolation Using Bidirectional Diffusion Sampler, ICLR 2025.

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
The authors propose a novel one-step diffusion method for video frame interpolation. The authors present a two-stage strategy. Specifically, it first decomposes the interpolation process into motion modeling and frame synthesis. The motion modeling and frame synthesis module are trained together in an end-to-end format, with the ground truth frame latents as the additional input. In the second training stage, they train the diffusion model for denoising the pseudo ground truth frame latents for motion prediction. In their experiments, the proposed method outperforms the baselines with fast speed.

### Strengths
- The authors propose a novel framework for video frame interpolation, achieving fast speed and competitive performance.
- The proposed continuous motion field representation enables more flexible motion modeling and generates plausible flow samples, as evidenced by the visualizations.

### Weaknesses
- The paper lacks visualizations on more challenging scenarios, such as the “breaking dance” case in the DAVIS dataset.
- The flow sampling mechanism is not sufficiently explained; more details on how samples are generated and utilized would strengthen the paper.
- The paper is missing discussions of closely related works that also integrate motion or optical flow modeling in video generation/interpolation, such as *VideoJAM* [1], *Motion-I2V* [2], and *GIMM-VFI* [3].
- Continuous temporal results. More visualizations like Figure 5 to present the continuous motion modeling and interpolation ability.

[1] VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models.

[2] Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling.

[3] Generalizable Implicit Motion Modeling for Video Frame Interpolation.

### Questions
- Is it necessary to use the VAE encoder? Given the recent progress in the academy, it would be interesting to replace the VAE encoder with other encoders, such as DINOv2.
- Please add more visualizations for the cases indicated in the weakness section, including more challenging scenarios and continuous temporal results, to support the claimed interpolation ability.
- It is necessary to have discussions with previous closely related work to enhance the clarity of the paper's motivation and contribution. 
- Please add more detailed descriptions for the core motion modeling part, especially the flow sampling operation, for better presentation.

### Soundness
2

### Presentation
2

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
1. The core idea of RDVFI is to disentangle VFI into two stages: motion prediction and appearance generation. Generating new frames based on low-resolution motion information enables RDVFI to perform effectively in both inference speed and generation authenticity.
2. This is the first diffusion-based VFI method with one-step inference, which achieves 50× acceleration compared to SOTA with also better results.

### Strengths
1. In the diffusion stage, this method generates low-resolution optical flow as an intermediate result to both accelerate the diffusion process and improve the stability of interpolation.
2. Bi-directional interpolation significantly improves generation authenticity, making RDVFI get SOTA performance on DAVIS and FCVG benchmarks.
3. The video demo shows significantly better results than existing methods.

### Weaknesses
1. Optical flow is the key information for the entire pipeline, as it is utilized for both image warping and feature warping in a bidirectional manner. However, in the first training stage, optical flow results are trained in an unsupervised way. The rationality of this setting requires further verification.
2. The number of testsets is limited and lacks diversity in their sources. The authors evaluate their method on DAVIS-7 and FCVG, however FCVG is sampled from DAVIS and RealEstate10K.
3. For the RDVFI-U and RDVFI-D models, the authors set different numbers of key frames, but did not provide a reasonable explanation and lacked corresponding ablation experiments.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a video frame interpolation method using a one-step diffusion model, aiming to improve inference efficiency by eliminating the need for multi-step denoising. The approach decomposes intermediate frame generation into two stages: the first estimates a continuous motion field between the input frames, and the second synthesizes intermediate frames by warping the inputs according to the predicted motion field. Experimental results are compared against conventional video frame interpolation methods as well as diffusion-based approaches.

### Strengths
The paper aims to improve the efficiency of diffusion models for video frame interpolation by disentangling in-between frame synthesis into two stages: continuous motion prediction and warping. The continuous motion (which use a spline interpolation curve) prediction stage is lightweight, operating on latent features from spatially and temporally downsampled video representations, which reduces memory consumption. In addition, the method employs a one-step diffusion model to predict the latent features during inference time further increasing inference speed.

### Weaknesses
The main weakness is the experiment does not convince me of the effectiveness of the method:
1. There are only six qualitative video comparisons in the supplementary material, which are insufficient to demonstrate the superior quality of the proposed method. Moreover, I did not observe a clear visual difference between Wan and the proposed approach.
2.  Lacks baselines on direct one/few-step distillation of video in-betweening diffusion models, rather than decomposing the process into two stages as done in the paper.

### Questions
1. How does  the number of keyframes in the continuous motion representation affect the inbetweening results especially in 24x interpolation?
2. Directly fine-tuning the original diffusion denoiser to perform full noise removal in a single step seems rather ambitious. Incorporating a distillation-based loss might help improve stability and performance.

### Soundness
2

### Presentation
3

### Contribution
2
