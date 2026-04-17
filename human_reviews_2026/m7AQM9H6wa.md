# MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation

- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
Character image animation has rapidly advanced with the rise of digital humans. However, existing methods rely largely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 4D information for open-world animation. To address this, we propose MTVCraft (Motion Tokenization Video Crafter), the first framework that directly models raw 3D motion sequences (i.e., 4D motion) for character image animation. Specifically, we introduce 4DMoT (4D motion tokenizer) to quantize 3D motion sequences into 4D motion tokens. Compared to 2D-rendered pose images, 4D motion tokens offer more robust spatial-temporal cues and avoid strict pixel-level alignment between pose images and the character, enabling more flexible and disentangled control. Next, we introduce MV-DiT (Motion-aware Video DiT). By designing unique motion attention with 4D positional encodings, MV-DiT can effectively leverage motion tokens as 4D compact yet expressive context for character image animation in the complex 4D world. We implement MTVCraft on both CogVideoX-5B (small scale) and Wan-2.1-14B (large scale), demonstrating that our framework is easily scalable and can be applied to models of varying sizes. Experiments on the TikTok and Fashion benchmarks demonstrate our state-of-the-art performance. Moreover, powered by robust motion tokens, MTVCraft showcases unparalleled zero-shot generalization. It can animate arbitrary characters in full-body and half-body forms, and even non-human objects across diverse styles and scenarios. Hence, it marks a significant step forward in this field and opens a new direction for pose-guided video generation. Our project page is available at https://github.com/DINGYANB/MTVCrafter. A scaled version has been commercially deployed and is available at https://telestudio.teleagi.cn/generatevideo/creativeWorkshop.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a new framework that directly models raw 3D motion sequences to animate arbitrary characters from reference images. Unlike previous methods that rely on 2D pose renderings, MTVCraft encodes 3D joint trajectories into compact 4D motion tokens using a 4D Motion Tokenizer (4DMoT), which preserves spatial-temporal dynamics and eliminates the need for pixel-level pose alignment. These tokens are then integrated into a Motion-aware Video Diffusion Transformer (MV-DiT) featuring 4D motion attention and 4D positional encodings, enabling expressive, disentangled control of motion and appearance. Implemented on both CogVideoX-5B and Wan-2.1-14B backbones, MTVCraft achieves state-of-the-art results, with superior zero-shot generalization to unseen characters, styles, and non-human subjects.

### Strengths
A key strength of integrating a motion-generation pipeline into video generation lies in its ability to provide explicit temporal and structural control over motion, resulting in more coherent and realistic dynamics than end-to-end pixel-based video synthesis. By introducing intermediate motion representations, such as in motion generation domain [1,2], the framework captures fine-grained spatial-temporal cues that general video models often overlook. This separation of motion from appearance allows the generator to synthesize consistent, expressive, and physically plausible movements while preserving identity and style, effectively bridging human motion understanding with visual generation. Moreover, as demonstrated in previous literature [3], coupling motion modeling within video diffusion enables strong generalization to arbitrary characters. Also due to the modular paradigm, the method scales naturally across different backbones, whether smaller transformer-based models like CogVideoX or Wan-2.1, since the motion tokens are from unified SMPL.

[1] Momask : MoMask: Generative Masked Modeling of 3D Human Motions, cvpr 2024
[2] SALAD : salad skeleton-aware latent diffusion for text-driven motion generation and editing, cvpr 2025
[3] AnyMoLe : AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models, cvpr 2025

### Weaknesses
A potential weakness of this paradigm is that the 4D motion compression via 4DMoT is conceptually straightforward and not architecturally novel. The encoder-decoder with vector quantization closely follows standard VQVAE formulations, and while it effectively transforms SMPL joint trajectories into compact motion tokens, it does not introduce fundamentally new techniques in motion encoding or representation learning. However, despite this structural simplicity, the usage and integration of such motion tokenization within a large-scale video generation framework remains a meaningful contribution.

### Questions
The codebook size of 8,192 with a 3072-dimensional embedding is relatively large compared to those commonly used in motion generation models. Would decreasing its dimension or size help reduce unused codes or improve resolution?

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
4

### Summary
This paper proposes MTVCraft (Motion Tokenization Video Crafter), a novel framework for character image animation that directly models raw 3D motion sequences (referred to as 4D motion) rather than relying on traditional 2D-rendered pose images. This is achieved by introducing a 4D Motion Tokenizer (4DMoT) to quantize 3D motion into compact tokens, which are then used to condition a Motion-aware Video DiT (MV-DiT). The approach effectively decouples motion from pixel-level alignment and appearance biases, showing strong zero-shot generalization to arbitrary characters, diverse styles, and non-human subjects. MTVCraft achieves state-of-the-art performance on benchmarks like TikTok and Fashion. The shift from 2D pose to 4D motion tokens is a significant and positive step forward for controllable video generation.

### Strengths
1.	The paper addresses a limitation of current methods by replacing fragile 2D pose images with robust, compact 4D motion tokens derived directly from SMPL joint coordinates.
2.	Experimental results indicate the effectiveness of the proposed method.

### Weaknesses
1.	The method introduces a new, independently trained component: the 4D Motion Tokenizer (4DMoT). Training this additional encoder (a VQVAE) adds complexity to the overall pipeline and represents an extra component that must be learned, stored, and maintained, potentially limiting the ability to scale the entire framework compared to methods that use off-the-shelf 2D pose estimators. During inference, the system requires an additional forward pass through the 4DMoT encoder to generate the motion tokens from the raw SMPL joint coordinates. While the tokens are compact, the initial encoding step introduces additional inference latency and computational cost that is not present in 2D-based methods (which often use pre-calculated 2D maps). The paper should provide a detailed breakdown of the latency and resource consumption of the 4DMoT during inference compared to the latency of the main MV-DiT model to justify this extra computational step.
2.	The entire pipeline is contingent upon the accuracy of the upstream SMPL joint sequence estimation (using NLF-Pose in this work ). Errors or noise in the initial 3D pose data will directly impact the quality of the 4D motion tokens and thus the final animation quality.
3.	The teaser figure shows multi-character animation results. But the motions of different characters are same. How can we animate only one character in the multi-character image?

### Questions
Why 3D/4D information is more useful than 2D? Can the authors provide some evidence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes MTVCraft, which shifts pose guidance from 2D renderings to direct tokenization of raw 4D motion (SMPL joint coordinates) via a VQ-VAE motion tokenizer (4DMoT). These motion tokens condition a Motion-aware Video DiT (MV-DiT) equipped with 4D motion attention and 4D rotary positional encoding, aligning spatio-temporal structure during generation. The system scales from a 6B (CogVideoX-5B) to an 18B backbone (Wan-2.1-14B) and supports multi-control (text + motion) with simple integration, delivering state-of-the-art results on TikTok and Fashion benchmarks. Design ablations and negative results further motivate coordinate-space tokenization over parameter-space alternatives.

### Strengths
1. Clear paradigm shift from 2D renderings to discrete 4D motion tokens, with a well-articulated rationale (coordinates vs. parameters) and an architecture that uses motion tokens natively (4D RoPE + motion attention).

2. Strong empirical gains on both TikTok and Fashion. The 18B model improves FID/FVD while modestly raising SSIM/PSNR over strong baselines. 

3. Scalable & practical. The 18B integration is straightforward (zero-padding alignment), and the paper documents unsuccessful alternatives (linear/MLP projection, SMPL-parameter tokenizer), which is valuable for reproducibility and future work. 

4. This paper shows cross-identity animation results across different species, e.g., human pose to bird, fish or even chair.

### Weaknesses
1. Camera view handling are implicit. There’s no explicit camera-parameter conditioning. The method relies on data diversity and 4D tokens. This is workable, but leaves questions about view transitions or long-term 3D consistency.

2. As SMPL joint are hard to accurately estimated, how the authors ensure that annotation quality? Besides, why don't use SMPL-X which includes hands?

3. For the heavy 18B model, will the inference cost of the proposed model be 10x or even 100x of the previous U-net based models, so that the comparison is not fair enough? It will be better to report the inference cost in performance comparison.

### Questions
Please see the weaknesses.

Typos: "Fahsion" should be "Fashion" in Table.4's header.

Overall, this paper offers a clean, well-justified shift from 2D renderings to discrete 4D motion tokens, demonstrates superior results on two standard benchmarks (with scaling to 18B), and provides enough architectural detail to be useful for practitioners. The main remaining gaps ( camera parameterization, brief data quality discussion) are addressable in the camera-ready and do not undermine the core contribution. I believe this paper will contribute to the community, so I vote clear acceptance.

### Soundness
4

### Presentation
3

### Contribution
4
