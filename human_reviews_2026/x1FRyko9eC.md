# SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Recent advances in diffusion-based video restoration (VR) demonstrate significant improvement in visual quality, yet yield a prohibitive computational cost during inference.
While several distillation-based approaches have exhibited the potential of one-step image restoration, extending existing approaches to VR remains challenging and underexplored, particularly when dealing with high-resolution video in real-world settings.
In this work, we propose a one-step diffusion-based VR model, termed as SeedVR2, which performs adversarial VR training against real data.
To handle the challenging high-resolution VR within a single step, we introduce several enhancements to both model architecture and training procedures.
Specifically, an adaptive window attention mechanism is proposed, where the window size is dynamically adjusted to fit the output resolutions, avoiding window inconsistency observed under high-resolution VR using window attention with a predefined window size.
To stabilize and improve the adversarial post-training towards VR, we further verify the effectiveness of a series of losses, including a proposed feature matching loss without significantly sacrificing training efficiency.
Extensive experiments show that SeedVR2 can achieve comparable or even better performance compared with existing VR approaches in a single step.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes AnonymousVR, a one-step diffusion-based video restoration (VR) model that aims to overcome the high computational cost of traditional diffusion-based VR methods. The paper introduce an adaptive window attention mechanism that dynamically adjusts window sizes according to output resolution to address inconsistency issues in high-resolution VR. They also design an adversarial post-training strategy incorporating a feature matching loss to enhance stability and quality without significantly increasing training cost. Experimental results suggest that AnonymousVR achieves competitive or superior performance compared with existing multi-step VR methods while requiring only a single inference step.

### Strengths
The experimental results are promising, showing a substantial improvement.

### Weaknesses
1. The technical contributions of this paper appear rather weak, as the work seems to rely more on engineering efforts than on genuine technical innovation. The proposed window attention and adversarial diffusion training methods lack clear novelty and do not demonstrate substantial methodological advancement beyond existing approaches.

2. Please clarify the core improvements of the proposed adaptive window attention compared with prior window attention mechanisms, such as those in SeedVR. Additionally, please elaborate on the key advancements of your adversarial diffusion training strategy relative to existing adversarial diffusion methods.

3. The authors claim that their method is the first one-step video restoration approach; however, one-step super-resolution and restoration methods already exist in the literature (e.g., [1][2]). A more detailed discussion and comparison with these prior works are necessary to justify the claimed novelty.

[1] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution. NeurIPS 2025.

[2] One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution. NeurIPS 2025.

### Questions
see Weaknesses.

### Soundness
2

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
4

### Summary
This paper proposes a one-step video restoration (VR) model that leverages a diffusion adversarial post-training (APT) framework to perform high-resolution video restoration. The key contributions include an adaptive window attention mechanism for high-resolution inputs, and a robust feature matching loss for improved training stability. The model is evaluated against state-of-the-art VR methods, demonstrating competitive performance while being four times faster than existing diffusion-based VR methods. The results suggest that it achieves comparable or even superior performance in real-world scenarios, especially with AI-generated content and high-resolution videos.

### Strengths
1. The paper introduces a novel one-step VR method by applying APT to diffusion-based models, reducing the computational burden significantly compared to traditional multi-step approaches.

2. The adaptive window attention mechanism for handling high-resolution videos and the feature matching loss for training stability are key contributions that improve the model's performance and robustness across varying video resolutions.

3. The method shows promising quantitative and qualitative results, outperforming existing VR approaches in real-world and synthetic benchmarks, demonstrating significant gains in speed and restoration quality.

### Weaknesses
1. The paper lacks comparisons with the latest VSR methods presented at NeurIPS 2025 (such as DLoraL [1] and DOVE [2]). The authors should include comparisons with these methods to better demonstrate the competitiveness of the proposed approach.

2. The paper does not provide results trained on public datasets (such as REDS). The reported improvements might stem from using a larger private dataset. Will the authors make the dataset publicly available?

3. Despite achieving faster inference, the training requires 72 H100 GPUs and significant resources, which raises concerns about scalability and accessibility for broader research adoption.

[1] One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution. NeurIPS2025

[2] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution. NeurIPS2025

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes AnonymousVR, a one-step diffusion-based video restoration method trained with adversarial post-training. It starts from a strong diffusion transformer, then applies progressive distillation and full adversarial tuning to remove the multi-step sampling cost. In addition, an adaptive window attention that adjusts window size to input resolution to avoid block boundaries at high resolutions, and a feature-matching loss taken from discriminator layers to replace expensive LPIPS during training are proposed. Experiments on synthetic, real, and AIGC videos demonstrate the effectiveness of the proposed method.

### Strengths
- The introduction of adaptive window attention effectively reduces boundary artifacts when processing high-resolution frames.

- The training strategy which combines RpGAN, approximate R2 regularization, feature-matching losses, and progressive distillation to ensure stable convergence and high perceptual quality is comprehensive.

- The experiments are extensive and include both synthetic and real-world data, multiple objective and perceptual metrics, as well as a well-organized user study.

### Weaknesses
- My main concern is that the novelty of the method is somewhat limited, as it largely builds upon the existing Adversarial Post-Training (APT) framework, and the paper does not clearly explain the fundamental differences or new contributions beyond APT.

- The training process is extremely resource-intensive, requiring 72 H100 GPUs, which significantly limits reproducibility and practical accessibility.

- The method’s robustness under challenging conditions, such as heavy degradations, large motion, or complex real-world dynamics, appears limited.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
I think the paper tackles an important and timely problem: high-resolution one-step video restoration (VR). The method “AnonymousVR” initializes from a diffusion transformer (SeedVR) and then performs adversarial post-training (APT) to convert it into a single-step generator. The paper’s two main technical levers are:  

(1) an adaptive window attention to avoid high-res window boundary artifacts.  

(2) an adversarial post-training recipe with progressive distillation and a feature-matching loss (taken from the discriminator) to stabilize training while avoiding pixel-space LPIPS cost. Experiments suggest competitive or better perceptual quality vs multi-step diffusion VR at much lower latency.

### Strengths
- I think the jump to truly one-step VR with a diffusion transformer (initialized from SeedVR) plus APT is a meaningful step beyond prior one-step image restoration; prior works are mostly teacher-distillation or rely on fixed diffusion priors that cap quality. This work claims distillation-free adversarial post-training against real data after a lightweight progressive distillation stage to bridge the gap, which is interesting for video. 

- The adaptive window attention to handle arbitrary resolutions with dynamic window size feels practical and addresses a real artifact at 2K/1080p; to my knowledge, such resolution-consistent windowing for VR in a one-step setting is new. 

- Using the discriminator’s multi-layer features as an LPIPS surrogate in latent / discriminator space for high-res VR is a reasonable, efficiency-motivated tweak (not conceptually new, but well-justified here).  

Overall, I think the contribution is incremental-to-moderate in theory but practically impactful for high-res, fast VR.

### Weaknesses
- I am concerned about the compute-heaviness. I think the approach relies heavily on significant compute (72×H100, 10M/5M pairs), which limits reproducibility in typical academic labs despite code release plans. Claims of “largest-ever VR GAN” underscore this.   

- Scope of degradations. While synthetic degradations follow prior work, I think the paper could better characterize real-world degradation diversity and robustness (e.g., compression artifacts, rolling shutter, severe motion blur) beyond VideoLQ/AIGC28; the method’s failure cases are not deeply analyzed.  

- Fairness of baselines. Diffusion baselines are run with 50 steps “to maintain stable performance”; I think it would be fair to include their fastest-setting curves (e.g., 10/25/50 steps trade-off plots) to contextualize speed-quality trade-offs.  

- Temporal metrics and consistency are missing. The paper mostly emphasizes frame-wise perceptual metrics; I would expect temporal consistency metrics (e.g., t-LPIPS variants or VMAF-like temporal terms) or user study questions specific to flicker/temporal stability. Current user study aggregates “overall quality” but not explicitly “temporal coherence.”

### Questions
- Temporal stability: How does one-step AnonymousVR compare to SeedVR on temporal coherence (quantitative & qualitative)? Any metric beyond a user study that isolates flicker?   

- Feature-matching loss: Why pick discriminator blocks 16/26/36 specifically? Did you try earlier/later layers or a learned weighting per layer? Impact on training speed?    

- Progressive distillation details: What are the teacher/student schedules and hyper-params across strides (64→32→…→1)? How much of the final gain comes from progressive distillation vs APT?

### Soundness
3

### Presentation
3

### Contribution
3
