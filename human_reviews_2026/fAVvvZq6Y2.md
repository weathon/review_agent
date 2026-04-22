# Taming Diffusion Transformer for Efficient Mobile Video Generation in Seconds

- Avg Score: 5.20
- Decision: Reject
- Scores: 4, 8, 4, 6, 4

## Abstract
Diffusion Transformers (DiT) have shown strong performance in video generation tasks, but their high computational cost makes them impractical for resource-constrained devices like smartphones, and practical on-device generation is even more challenging. 
In this work, we propose a series of novel optimizations to significantly accelerate video generation and enable practical deployment on mobile platforms. First, we employ a highly compressed variational autoencoder (VAE) to reduce the dimensionality of the input data without sacrificing visual quality. Second, we introduce a KD-guided, sensitivity-aware tri-level pruning strategy to shrink the model size to suit mobile platform while preserving critical performance characteristics. Third, we develop an adversarial step distillation technique tailored for DiT, which allows us to reduce the number of inference steps to four. Combined, these optimizations enable our model to achieve approximately 15 frames per second (FPS) generation speed on an iPhone 16 Pro Max, demonstrating the feasibility of efficient, high-quality video generation on mobile devices.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents several modifications to DiT-based video generation models to enable them to run on mobile devices. Specifically, for the model architecture, the paper adopts a high-compression video VAE to reduce the number of tokens and a lightweight DiT obtained through structured pruning. For training, adversarial distillation is used to reduce the number of timesteps required during inference. For efficient inference, a tiled GEMM strategy is applied to alleviate memory bottlenecks. The experimental results show that the proposed model can generate a 49-frame video clip within 4 seconds on an iPhone 16 Pro Max.

### Strengths
- The motivation for this work is clear and convincing. Nowadays, many video generation models or services have been released, but almost all of them require a huge amount of computational resources to run. Reducing this cost could have a significant impact on user-generated content creation and also contribute to greater sustainability.
- The base model used in this work, DiT in latent space, is a popular choice for video generation tasks. Therefore, the proposed modifications are likely to be widely applicable to existing or future video generation models.

### Weaknesses
- The novelty in methodology is marginal.
  - Increasing the compression ratio in VAE is a common choice (e.g., [R1]) for efficient latent diffusion models. To enhance the novelty, it would be beneficial if the authors could provide empirical findings that are particularly important in the context of video generation.
    - [R1] "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers," ICLR 2025.
  - The idea of tri-level pruning sounds interesting, but the optimization process is greedy and not specifically designed for tri-level pruning. If the authors want to argue for the benefits of the tri-level pruning search space, the ablation study in Table 4 should include results for all single-level pruning strategies (not only block pruning, but also attention-head and linear pruning).
  - Knowledge distillation and adversarial distillation are also common approaches for obtaining lightweight diffusion models. Using part of the teacher model as a feature extractor for the discriminator is interesting, but a similar idea has already been explored in prior work [R2].
    - [R2] "SoundCTM: Unifying Score-based and Consistency Models for Full-band Text-to-Sound Generation," ICLR 2025.

- The main quantitative results shown in Table 2 lack several important metrics, such as dynamic degree and motion smoothness. For a comprehensive comparison, it is highly encouraged to provide the full list of results.

### Questions
- Could you list the empirical findings obtained through this work that are specifically important for text-to-video generation? It would be beneficial to demonstrate that this work offers sufficient novelty beyond simply combining existing techniques.

### Soundness
3

### Presentation
3

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
The paper proposes a series of novel optimizations to accelerate video generation and enable practical deployment on mobile
platforms. These optimizations allow the model to achieve approximately 15 frames per second (FPS) generation speed on an iPhone 16 Pro Max, demonstrating the feasibility of efficient, high-quality video generation on mobile devices.

### Strengths
1. The paper achieved the deployment of a video generation model on mobile devices (iPhone 16 Pro Max). 
2. The experiments were extensive, involving extensive and complex work.
3. The final performance of the method is impressive.

### Weaknesses
1. It is recommended to verify the performance of the method on more mobile devices. There are currently various edge hardware environments; for example, low-power intelligent chips from Qualcomm and NVIDIA have been widely adopted, and they differ significantly from Apple's chip architectures. It is recommended to verify the performance on more types of chips or specify the hardware applicability scope of the method in the limitation section.
2. It is recommended to explore the causes and solutions for performance degradation induced by different VAEs, and strengthen the analysis of VAEs' impact on performance. For instance, it should be clarified which aspects of visual quality degradation correspond to PSNR loss, and how such degradation propagates to affect the sub-items in VBench scores. Additionally, efforts should be made to establish a theoretical connection between VAEs and DiT optimization, and to discuss the effectiveness and underlying principles of existing DiT optimization methods for compensating for performance losses when applied to models under current VAE conditions. 
3. Given that this research has achieved significant improvements in video generation performance, it is recommended to make the code for reproducible experimental results publicly available to enhance the reproducibility of the paper's experimental results.

### Questions
The paper is generally clearly presented; for specific issues, please refer to the points listed in the Weaknesses.

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
The paper proposes a set of optimizations that make video Diffusion Transformers (DiTs) viable for a video generation on mobile devices. By combining a high-compression video VAE, KD-guided tri-level pruning, adversarial step distillation, and operator-level improvements via tiled GEMM, the authors achieve high-quality video synthesis at 15 FPS on an iPhone 16 Pro Max. The approach maintains strong visual fidelity while significantly reducing inference steps and model size, offering a practical solution for on-device video generation.

### Strengths
1) The paper is well written and motivated; it is easy to read and understand.
2) One of the pioneering works in the niche application: one of the first video diffusion transformer models running on-device with decent quality.
3) Comprehensive evaluation, both automated quality metrics and user studies are conducted, and good results are reported.
4) Somewhat novel design of adversarial step distillation setup adapted to the new architecture (DiTs).
5) Interesting discussions about quality-vs-efficiency trade-off in VAEs and pruning strategies of the original model.

### Weaknesses
1) My main concern with this work is limited novelty: while the engineering contributions are solid and well-executed, the paper primarily combines existing techniques, compression via VAE, pruning, distillation, and operator-level optimization, without introducing fundamentally new ideas or theoretical insights. The novelty lies more in integration than in conceptual advancement.
2) The pruning and distillation strategies, although tailored for DiTs, follow well-established paradigms. The tri-level pruning and adversarial step distillation are adaptations rather than breakthroughs, and similar approaches have been explored in UNet-based models and mobile optimization literature.
3) Another concern is about applicability of the proposed methodology to other mobile devices and platforms. Will the model perform as well on other IPhone devices for example? How about Android devices, will this approach extend to this platform in principle?

### Questions
1) Can the authors elaborate on what they consider the core novel contribution of this work beyond engineering integration? Specifically, how does the tri-level pruning or adversarial step distillation differ fundamentally from prior work in UNet-based or transformer-based models?
2) The paper focuses on text-to-video generation. Could the proposed pipeline be adapted for other tasks such as video editing, inpainting, or conditional generation (e.g., image-to-video)? If so, what modifications would be needed?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an efficient video generation framework that uses DiT to generate high-quality videos on mobile devices. The authors accelerate inference through techniques such as high-compression VAE, latency-aware pruning, adversarial step distillation, and GEMM-based tiling. Ultimately, the model achieves real-time video generation at 15 frames per second on the iPhone 16 Pro Max.

### Strengths
1.The authors propose an innovative model acceleration method that addresses the deployment challenges of DiT on mobile devices. By combining VAE compression, pruning, distillation, and other techniques, they achieve real-time video generation.

2.The experiments are thorough, with strong supporting arguments, and the writing is clear and easy to understand.

### Weaknesses
1.While the model performance is improved after three layers of pruning and distillation, the distillation process requires significant computational power, which makes training more challenging.

2.The framework was tested on the iPhone 16 Pro Max, but its performance may depend on the specific hardware architecture and optimization strategies. The differences in memory bandwidth and computational power across various edge devices could affect the model’s performance, especially on older devices.

### Questions
1.Regarding the memory bottleneck addressed by GEMM, is this a problem unique to mobile devices, or would using GEMM on a server-side system provide similar acceleration?

2.Distillation training requires substantial resources; how much time is required to train the lightweight model discussed in the paper?

3.During training, both synthetic and real data are used. Could you provide more details on the ratio between these two types of data, and how different ratios might affect the model’s performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a pipeline to run Diffusion Transformer-based video generation efficiently on mobile devices. It combines a high-compression video VAE, tri-level pruning with KD, 4-step adversarial distillation, and operator-level optimizations, achieving over 15 FPS video synthesis on an iPhone 16 Pro Max with competitive quality.

### Strengths
S1. This paper tackles a practical problem of on-device DiT video generation and provides good deployment results.

S2. The proposed pruning with a KD-guided framework yields substantial speedups with moderate quality drop.

S3. Demonstrating real-time performance on mobile hardware is a meaningful empirical result.

### Weaknesses
W1. The novelty is somewhat limited, as the proposed approach mainly combines existing compression, pruning, and distillation techniques, rather than introducing new algorithmic ideas.

W2. The contribution is engineering-driven, focusing on system and deployment optimizations, with relatively limited new ML insights or principles that generalize beyond this specific application.

W3. The evaluation is not fully convincing, as it lacks comparisons with recent efficient video diffusion and on-device methods, and provides limited analysis of efficiency-quality trade-offs.

### Questions
Q1. What is the key methodological novelty beyond adapting existing compression, pruning, and distillation techniques?

Q2. What ML insights or generalizable design principles does this work provide beyond the specific engineering optimizations for on-device DiT?

Q3. Can you add recent efficient diffusion and/or on-device baselines and deeper efficiency-quality trade-off analysis to better support the claims? Is it possible or not?

### Soundness
3

### Presentation
3

### Contribution
2
