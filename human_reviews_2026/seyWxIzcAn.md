# Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Diffusion-based video super-resolution (VSR) methods have recently demonstrated remarkable perceptual quality; however, their reliance on future-frame information and computationally expensive iterative denoising has restricted their application in latency-sensitive contexts. We present Stream-DiffVSR, a causally conditioned diffusion VSR framework designed for efficient online inference. Our method operates strictly with past frames and integrates three key components: a four-step distilled denoiser, an auto-regressive temporal guidance (ARTG) module that injects motion-aligned temporal cues into the denoising process, and a lightweight temporal-aware decoder with temporal processor module (TPM) that enhances spatial detail and temporal consistency. Stream-DiffVSR processes 720p frames in just 0.328 seconds on an RTX 4090 GPU, significantly outperforming previous diffusion-based methods. Compared with state-of-the-art online methods such as TMP, Stream-DiffVSR achieves a substantial improvement in perceptual quality (LPIPS improved by 0.095) while reducing inference latency by more than 130X relative to previous diffusion-based VSR approaches. These results demonstrate the potential of diffusion models for practical deployment in time-sensitive rendering pipelines and real world video super-resolution systems. Notably, Stream-DiffVSR achieves the lowest latency ever reported among diffusion-based VSR methods, reducing the initial delay from over 4600 seconds to just 0.328 seconds. This makes it the first diffusion-based solution viable for real-time online deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a ControlNet-based video super-resolution (SR) architecture. It starts from an image SR model, i.e., StableDiffusion ×4 Upscaler, and makes the following modifications:

- A rollout distillation to distill the 50-step model to a 4-step one.
- Making the image model an autoregressive one via taking both the current noisy latent at timestep t and warped image at timestep t-1 as conditional input.
- Making the VAE decoder a temporal one via incorporating temporal context into decoding to enhance spatial fidelity and temporal consistency.

The proposed approach presents online potential with faster speed compared with previous ControlNet SR baselines.

### Strengths
- The paper is easy to follow, with clear details for each component.
- The efficiency of the proposed approach is clear compared with other baselines in the paper.
- Extensive experiments are conducted on both synthesis and real-world datasets, though lacking some commonly used metrics for evaluation.

### Weaknesses
- Given the fast development of existing generative models, especially for video generation models, the necessity of continuing to modify an image model for VSR does not seem reasonable. To harness an image model for VSR, a large cost lies in improving the temporal consistency, i.e., adding temporal cues, including optical flow and the temporal decoder. Such a cost can be largely avoided when turning to adopt a video generation model, such as CogVideoX[1] and Wan2.1[2], as a base prior.

- The temporal consistency of the proposed approach largely relies on optical flow from the previous frame, which suffers from a limited capability for long-term information capturing. Given the advances of recent DiT-based architectures[3,4,5], the paper lacks the discussion and comparison with this new branch of VSR approaches.

- The paper mostly relies on known technologies, such as ControlNet[6], 4-step distillation from SDXL Turbo[7], and temporal decoder from Upscale-A-Video[8], making the novelty kind of incremental.

[1] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. ICLR 2025.

[2] Wan: Open and Advanced Large-Scale Video Generative Models. ArXiv 2025.

[3] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution. NeurIPS 2025.

[4] SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training. ArXiv 2025.

[5] InfVSR: Breaking Length Limits of Generic Video Super-Resolution. ArXiv 2025.

[6] Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023.

[7] Adversarial Diffusion Distillation. ECCV 2024.

[8] Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution. CVPR 2024.

### Questions
My main concerns are as follows:

1. The weaknesses above. Note that InfVSR [1] can be seen as concurrent work, and there is no need to compare with it. It is listed because the KV-Cache technology has become a popular way for DiT-based architecture for autoregressive generation. Considering that this paper also focuses on the autoregressive manner, the authors are expected to make a theoretical discussion between the proposed approach and the KV-Cache technology, which should be a more natural way to achieve online autoregressive VSR, in my view.

2. The paper lacks some important baselines for comparison, including DOVE[2] and SeedVR2[3], which both focus on one-step VSR for high efficiency.

3. Synthetic datasets are just toy examples. The authors should add more quantitative and qualitative results following previous methods. Commonly used metrics such as CLIP-IQA[4], MUSIQ[5] on real-world datasets, and warping error[6] on synthetic data should be added for better comparison.


[1] InfVSR: Breaking Length Limits of Generic Video Super-Resolution. ArXiv 2025.

[2] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution. NeurIPS 2025.

[3] SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training. ArXiv 2025.

[4] Exploring CLIP for Assessing the Look and Feel of Images. AAAI 2023.

[5] MUSIQ: Multi-scale Image Quality Transformer. ICCV 2021

[6] Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution. CVPR 2024.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a low-latency VSR framework termed Stream-DiffVSR. The model achieves 0.3s per frame with proposed conponents of a four-step distilled denoiser, an auto-regressive temporal guidance (ARTG) module that injects motion-aligned temporal cues into the denoising process, and a lightweight temporal-aware decoder with temporal processor module (TPM) that enhances spatial detail and temporal consistency while maintaining SOTA performance with improvement of 0.095 in LPIPS index.

### Strengths
1.	This paper proposes a streamable VSR framework with auto-regressive manner, which is straightforward and easy to follow.
2.	The proposed model achieves 0.3s per frame in only one RTX 3090 GPU while maintaining SOTA performance.

### Weaknesses
1.	The novelty is limited since most of the techniques are proposed in previous works. The authors should discuss in detail the contribution and novelty of the paper and enhancement that specifically developed for VSR task.
2.	The model comparison is insufficient since the paper does not provide the model size of all competitors.
3.	The visualization performance shows little improvements comparing to other SOTA models. And the authors should provide VSR results on more complicated textures like text or signs restoration in video.
4.	The reading flow is problematic. E.g., it is better to provide exact section number of the supplement when you want to quote it in the manuscripts.

### Questions
Refer to the weaknesses. The authors should discuss in detail the contribution and novelty of the paper and enhancement that related to VSR. More comparisons on model size is necessary.

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
5

### Summary
Stream-DiffVSR proposes an autoregressive diffusion pipeline for online video super-resolution. By distilling a 50-step teacher into 4-step sampling and conditioning solely on past frames, the paper reports >100× latency reduction relative to prior diffusion VSR methods. Experimental results on REDS4, Vimeo-90K-T, Vid4 and VideoLQ show that the proposed Stream-DiffVSR achieves a better trade-off between performance and latency among diffusion-based methods.

### Strengths
1. The paper is well-written and easy to follow.
2. Extensive ablations on step count, rollout distillation, ARTG and TPM verify component contributions.
3. The proposed method demonstrates significantly improved runtime performance compared to previous diffusion-based approaches.

### Weaknesses
1. Core ideas (U-Net distillation, auto-regressive temporal guidance) are existing techniques; this paper presents a careful system integration rather than a fundamentally new diffusion formulation.
2. The employed distillation strategy is similar to ADD, with no clear technical distinctions.
3. Distillation is only compared with the original 50-step teacher; modern fast samplers are not evaluated, leaving the optimality of 4-step design unclear.

### Questions
1. “Low-latency” is misleading. Paper repeatedly uses “real-time”, “practical deployment”, “first diffusion solution viable for real-time online VSR” – these statements are unsupported by absolute runtime numbers.
2. Table 4 does not include comparisons with unidirectional methods.
3. Are the results tested in Section A4 obtained under different degradation settings? If so, please provide detailed results for each degradation level.

### Soundness
3

### Presentation
3

### Contribution
2
