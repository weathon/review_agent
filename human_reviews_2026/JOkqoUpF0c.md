# AvatarSync: Rethinking Talking-Head Animation through Phoneme-Guided Autoregressive Perspective

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4, 4

## Abstract
Talking-head animation focuses on generating realistic facial videos from audio input. Following Generative Adversarial Networks (GANs), diffusion models have become the mainstream, owing to their robust generative capability. However, inherent limitations of the diffusion process often lead to inter-frame flicker and slow inference, hindering their practical use in talking-head animation. To address this, we introduce AvatarSync, an autoregressive framework on phoneme representations that generates realistic and controllable talking-head animations from a single reference image, driven by text or audio input. To mitigate flicker and ensure continuity, AvatarSync leverages an autoregressive pipeline that enhances temporal modeling. In addition, to ensure controllability, we introduce phonemes that are the basic units of speech sounds, and construct a many-to-one mapping from text/audio to phonemes, enabling precise phoneme-to-visual alignment. To further accelerate inference, we adopt a two-stage generation strategy that decouples semantic modeling from visual dynamics, incorporating a Phoneme-Frame Causal Attention Mask and a timestamp-aware adaptive strategy to support parallel inference. Extensive experiments conducted on Chinese (CMLR) and English (HDTF) benchmarks show that AvatarSync substantially reduces inter-frame flicker and outperforms existing methods in visual fidelity, temporal consistency, and computational efficiency, providing a scalable solution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an autoregressive generation framework for the talking-head video generation task. To alleviate the one-to-many relationship between speech and motion, as well as the flicker issue in previous autoregressive methods, the framework converts audio input into phonemes and then adopts a two-stage generation strategy: the first stage generates key frames, and the second stage generates interpolated frames between the key frames. Experiments demonstrate that the proposed method achieves promising performance.

### Strengths
- The paper employs an Autoregressive Method (instead of the commonly used diffusion methods). Such an attempt is relatively rare, and the method achieves excellent efficiency and quality.
- The paper conducts extensive experimental demonstrations, covering multiple aspects of metrics, which is quite comprehensive.
- The paper is written very clearly and is easy to understand.

### Weaknesses
- The proposed two-stage "key frame + interpolation" method lacks novelty, as this is actually an existing generation strategy (the "Two temporal res." mentioned in [1]). The authors need to clarify the core differences between this two-stage generation strategy and previous methods.
- Using phonemes may lead to information loss, such as emotion-related information and stress-related information. This may result in the lack of vividness in head movements, eye blinks, and facial expressions. The authors need to demonstrate the necessity of using phonemes (e.g., through ablation studies comparing with the commonly used audio embedding features in previous works) or provide theoretical proof that this modification is essential.
- Additionally, the paper lacks subjective metrics, such as user studies. It is recommended that the authors supplement such experiments or explain the reasons for being unable to do so.
Reference


[1] Harvey, William, et al. "Flexible diffusion modeling of long videos." Advances in neural information processing systems 35 (2022): 27953-27965.

### Questions
Please refer to the weakness.

### Soundness
3

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
This paper proposes AvatarSync, an autoregressive framework for talking-head animation that leverages phoneme representations and a two-stage generation strategy to produce high-quality, lip-synchronized facial videos from a single image and audio input.

### Strengths
（1）Structurally, the paper is well-organized, with clear explanations of the methodology, theoretical analysis of inter-frame flicker, and illustrative figures.
（2）In terms of quality, extensive experiments on two benchmark datasets demonstrate clear improvements in both quantitative metrics (e.g., FID, FVD) and qualitative results.
（3）The paper introduces a novel autoregressive framework that effectively leverages phoneme representations for talking-head generation, with the phoneme-frame causal attention mask and two-stage hierarchical generation strategy being both creative and well-motivated.

### Weaknesses
（1）The method primarily focuses on comparisons with strong baselines like Hallo, while insufficient attention is given to including newer approaches such as VASA-1. It also lacks direct comparisons with other autoregressive methods (e.g., VideoPoet, Teller) in terms of speed and resource consumption. Although comparisons with diffusion models demonstrate certain advantages, they fail to adequately establish its leading position within the autoregressive paradigm.（2）The experimental evaluation mainly concentrates on overall video quality and lip synchronization, while lacking specialized assessment and analysis of subtle facial expressions such as natural blinking and lip pursing.
（3）The approach relies heavily on multiple high-performance GPUs, which may limit its reproducibility for users with consumer-grade hardware. Demonstrating its performance on more accessible computing resources would be valuable.

### Questions
（1）How well does the model scale with longer phoneme sequences? Does performance degradation occur when generating extended videos?
（2）The paper states that the interpolation module can process different keyframe pairs in parallel. Does this imply that the entire second stage is fully parallelizable? If so, what is the theoretical speedup ratio? Please elaborate on the specific parallelization strategy and its practical benefits in real-world deployment.
（3）Have you considered incorporating more diverse scenarios in experiments, such as adding noise, to further test the model's robustness in real-world environments?
（4）The phoneme-frame causal attention mask forces each keyframe to focus solely on its corresponding phoneme. Could this strict one-to-one mapping limit the model's ability to learn coarticulation effects across adjacent phonemes? Would this limitation affect the fluency and naturalness of the resulting lip movements?
（5）The experiments involve a 4× super-resolution preprocessing step on the CMLR dataset. Is there any quantitative evidence demonstrating the specific contribution of this operation to the final model performance compared to using the original data?

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
4

### Summary
The paper proposes AvatarSync, an autoregressive (AR) framework on phoneme presentations with a two-stage pipeline : (1) Facial KeyFrame Generation (FKG) module that maps phoneme tokens (from text or audio) to sparse keyframes under Phoneme-Frame Causal Attention Mask, and (2) a tilmestep-aware, selective state-space modeling, which enables temporally coherent inference. The paper argues autoregressive framework mitigates flicker issues and improves efficiency. Extensive experiments cover performances on both CMLR (Chinese) and HDTF (English).

### Strengths
1.  Phoneme-conditioned keyframes and timestamp-aware interpolation is a clean split, and one-to-one causal attention is motivated and ablated. 
2. The paper cross-lingual settings. 
3. Attention variants and composite losses are well-documented.

### Weaknesses
1. Baseline Coverage 
Table1 includes several known baselines, but newer strong systems (e.g., StableAvatar, HunyuanAvatar, Hallo 3) are missing. 

2. Missing human evaluation and demos. 
The paper presents no user study and no supplementary videos, leaving perceptual claims (realism, temporal stability, lip sync, identity) unsupported by human evidence or visual inspection. As results rely solely on automatic metrics, the basis for performance judgment is insufficient. 

3. Metric weakness
Identity/facial similarity are used as losses, but test-time identity-preservation metrics are not surfaced alongside the metrics used in the main paper. Also, the paper emphasizes "responsive user experience in the real-world setting", but does not report the metrics of autoregressive efficiency. Please provide complementary identity-related metrics (e.g., ArcFace, CSIM), and efficiency-related metrics (e.g., FPS, latency, VRAM). 
Table2 omits any lip-sync metrics, and across the paper LSE-C (Sync-C) is largely absent too. Please explain the omission. 

4. Attribution gap (what component actually helps?) 
It is unclear how much each stage contributes. While the paper ablate the attention mechanism within FKG, there is no ablation isolating the timestamp-aware adaptive strategy in the inter-frame module;  hence the relative contributions of FKG vs. the temporal module are not established. Please add a controlled ablation that (i) disables timestamps, (ii) replaces the adaptive strategy with simpler temporal heads and (iii) final version (timestamp-aware adaptive strategy).

5. Phoneme timing and coarticulation.
A one-to-one causal mask may ignore coarticulation or context span. There is no analysis of context window , lip sync or forced alignment quality. 

6. Robustness gaps 
Would the authors comment on whether any cases involving occlusions, accessories (e.g., hats, glasses), pronounced head-pose variation, or mouth-region artifacts (e.g., teeth, tongue) were observed? If so, a brief analysis or examples would be helpful.

### Questions
I would appreciate responses to the questions in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on the inter-frame flickering and slow inference problems of existing GAN and diffusion models, and proposes an autoregressive framework that leverages a many-to-one mapping from text/audio to phonemes and a two-stage strategy to enhance temporal coherence and visual smoothness. Experimental results show improved temporal stability and efficiency of the proposed method.

### Strengths
1. The phoneme tokenization is inspiring, and the representation is easy to follow.
2. Empirical results show improved video quality, temporal coherence, and efficiency, which support the claims of the paper.

### Weaknesses
1. The paper only considers one reference image for the framework, while in practice, multiple reference images with different facial angles might be provided. How will the framework scale to this case? And will the diversity among reference images introduce noise into the tokens?
2. Humans are sensitive to inter-frame flicker, but there is a lack of user study in the experiments.
3. Providing synthesized video examples is recommended to further support the claims of the paper.

### Questions
Please answer the questions in the weakness.

### Soundness
2

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
4

### Summary
This paper presents AvatarSync, a novel framework for audio-driven talking-head animation that proposes a paradigm shift away from the prevailing GAN and diffusion-based approaches. The authors identify critical limitations in existing methods: GANs often suffer from visual artifacts and instability, while diffusion models, despite their high fidelity, are plagued by slow inference speeds and a characteristic inter-frame flicker that breaks temporal consistency.
To address these issues, AvatarSync reframes the task as a phoneme-guided autoregressive sequence generation problem, inspired by recent successes of large language models in video synthesis. The core of the method is a clever two-stage generation strategy designed to balance quality and efficiency:
Facial Keyframe Generation (FKG): An autoregressive model generates a sparse set of high-quality facial keyframes conditioned on phoneme sequences extracted from the input audio/text. A key innovation here is the "Phoneme-Frame Causal Attention Mask," which enforces a strict alignment between phonemes and their corresponding visual frames, ensuring precise lip synchronization.
Inter-Frame Interpolation: A second, lightweight module then fills in the intermediate frames between the generated keyframes. This module uses a timestamp-aware strategy to ensure the final video is temporally coherent, smooth, and computationally inexpensive to render.
The authors provide extensive experimental validation on the CMLR and HDTF datasets, demonstrating that AvatarSync achieves state-of-the-art results across multiple metrics, including visual fidelity (FID, FVD), lip-sync accuracy (Sync-D), and identity preservation. Crucially, the method is shown to be significantly faster than diffusion-based competitors and qualitatively free from the inter-frame flicker problem.

### Strengths
1. The paper does not merely propose an incremental improvement but "rethinks" the entire problem. By identifying the fundamental limitations of GANs and diffusion models for this specific task and proposing a coherent autoregressive alternative, the authors make a significant conceptual contribution. The framing is clear, well-motivated, and timely.
2. The FKG + Interpolation architecture is a highly intelligent solution to the quality-efficiency trade-off. It leverages the power of a large autoregressive model for the most critical parts of the animation (the keyframes) while delegating the more repetitive task of in-betweening to a faster, specialized module. This design is a key reason for the method's success.
3. A major strength of AvatarSync is its ability to generate temporally stable videos. The autoregressive generation of keyframes inherently builds upon previous frames, and the interpolation further ensures smoothness. The flicker visualization in Figure 5(b) is a powerful and convincing piece of evidence that directly showcases the superiority of this approach over diffusion models in eliminating distracting artifacts.

### Weaknesses
1. The paper's core motivation rests heavily on the premise that diffusion-based methods are inherently slow due to their iterative denoising process. While this is true for traditional DDPM/DDIM samplers, this argument is becoming outdated. Recent advancements in diffusion model distillation have enabled real-time, one-step generation. For instance, [1] demonstrate a diffusion-based approach that achieves real-time performance. The paper's efficiency claims would be much stronger if they were contextualized against these modern, distilled diffusion models, rather than just standard, slow samplers. Without this discussion, the claimed efficiency advantage feels overstated.
2. The paper introduces a two-stage generation process but fails to compare against other highly relevant and state-of-the-art two-stage methods. A critical omission is AniPortrait [2], a very popular and powerful framework that also employs a two-stage approach (first generating keyframes/poses and then rendering the video). Given the conceptual similarity in using a two-stage pipeline to tackle the problem, a direct comparison—or at least a detailed discussion—is essential to properly situate AvatarSync's contribution. Without it, it is difficult to assess whether the novelty and performance gains come from the autoregressive formulation itself or simply from adopting a two-stage strategy.

Reference:
[1].Guo, Hanzhong, et al. "Real-time One-Step Diffusion-based Expressive Portrait Videos Generation." arXiv preprint arXiv:2412.13479 (2024).
[2].Wei, Huawei, Zejun Yang, and Zhisheng Wang. "Aniportrait: Audio-driven synthesis of photorealistic portrait animation." arXiv preprint arXiv:2403.17694 (2024).

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
