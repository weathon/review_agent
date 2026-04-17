# Bridging Implicit-Explicit Representations for Ultra-Low Bitrate Image Compression

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
While recent VAE-based neural codecs achieve impressive results at low bitrates when optimized for perceptual quality, their performance degrades significantly under ultra-low bitrate conditions. To address this, generative methods that exploit semantic priors from pretrained models have emerged, revolutionizing ultra-low bitrate compression. However, these approaches remain constrained by a fundamental tradeoff between semantic faithfulness and perceptual realism. Methods relying on explicit semantic guidance preserve content accuracy but often lack textural fidelity, while those based on implicit representation can generate convincing details but may suffer from semantic drift. In this work, we introduce a unified framework that bridges this gap by coherently integrating explicit and implicit semantic representations. We condition a diffusion model with explicit high-level semantics while using reverse-channel coding to implicitly encode fine-grained information. In addition, a novel plugin encoder provides flexible control over the distortion-perception balance. Extensive experiments demonstrate that our framework achieves state-of-the-art rate–perception performance, outperforming existing approaches and surpassing DiffC by 23.49\%, 12.25\%, and 23.09\% DISTS-BD-Rate on the Kodak, DIV2K, and CLIC2020 datasets, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a dual semantic compression framework for ultra-low bitrate image compression. An explicit semantic encoder with tag-style prompts and plugin implicit semantic extractor capture high-level semantics and fine-grained visual details, respectively, achieving  distortion-perception tradeoff and flexible quality control. The proposed method consistently surpasses state-of-the-art approaches at extremely low bitrates.

### Strengths
This paper aims to compress the images under ultra-low bitrate conditions. Specifically, an unified implicit-explicit compression framework is proposed which achieves state-of-the-art rate-perception performance. In addition, the paper is well-organized and well-written.

### Weaknesses
1、	The paper lacks implementation details for the proposed dual representation compression framework. What loss functions are used in the paper and how is the framework trained? How to set the value of mixing coefficient during the training?
2、	In the proposed framework, are the MSE and VAE encoders based on a similar network structure? Are z and z_wave both 4-channel features?
3、	How can the conditions (c and y_hat) be embedded into the diffusion process? Is a CLIP model required to extract the semantic representation from the tag-style prompts?
4、	What are the encoding and decoding times of the proposed framework, which are important for image compression? The authors need to demonstrate how the inference time compares with that of other competitive compression approaches.
5、	We believe that the CLIC_2020 dataset, rather than the DIV2K dataset, is the most widely used benchmark for image compression tasks. However, the authors do not present any comparisons based on the CLIC_2020 dataset, which comprises 428 images with diverse content. Please include comparisons in the manuscript.
6、	In Fig. 4, why do the authors show the FID rather than the LPIPS metric for the DIV2K dataset? In Section 5.2, the authors do not analyse the comparisons in terms of FID. Additionally, Figures 8 and 9 in the supplementary file should be included in the main body of the manuscript.
7、	The paper lacks comparisons to recent extreme image compression methods including RDEIC[1], StableCodec[2], DLF[3], ResULIC[4], and OSCAR[5].
[1]. Li Z, Zhou Y, Wei H, et al. RDEIC: Accelerating Diffusion-Based Extreme Image Compression with Relay Residual Diffusion[J]. TCSVT2025.
[2]. Zhang T, Luo X, Li L, et al. StableCodec: Taming One-Step Diffusion for Extreme Image Compression[J]. ICCV2025.
[3]. Xue N, Jia Z, Li J, et al. DLF: Extreme Image Compression with Dual-generative Latent Fusion[J]. ICCV2025.
[4]. Ke A, Zhang X, Chen T, et al. Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion[J]. ICML2025.
[5]. Guo J, Ji Y, Chen Z, et al. OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates[J]. NeurIPS2025.

### Questions
Please see the weaknesses.

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
3

### Summary
The article presents an ultra-low bitrate image compression scheme based on conditional diffusion, which demonstrates optimal performance. Through extensive experiments conducted across multiple datasets, remarkable results have been achieved—further highlighting the scheme’s promising potential in terms of transferability.

### Strengths
The application of RCC technology in diffusion-based compression schemes has not been thoroughly explored. This article presents an intriguing solution. Specifically, I believe the strengths of this article are as follows:
1. Clear motivation and writing, allowing readers to follow the author's train of thought easily.
2. The proposed solution is backed by compelling and thorough experiments, yielding satisfactory performance results.
3. Following in the footsteps of DIFFC, this article represents a meaningful attempt at utilizing RCC technology in the field of image compression. The high transferability of the method holds significance for the advancement of this domain.

### Weaknesses
1. Some architectures appear less novel, for instance, the introduction of dual branches has been previously explored in various diffusion-based methods, blending semantic and image controls.
2. The article introduces ‘Tile-based Processing,’ from which the model benefits, yet this module lacks thorough elaboration, including aspects such as block quantity and complexity. Furthermore, the concepts of image segmentation and parallelism lack appeal.
3. The article lacks in discussing the complexity of its experiments. Notably, the significant encoding latency introduced by RCC, combined with the use of a tile structure for image segmentation, raises doubts about the necessity of multiple RCC encodings. Additionally, segmenting images and separately extracting prompts via RAM might exacerbate encoding latency. Given the inherent decoding latency of diffusion architectures, a detailed exploration of the complexity of encoding and decoding is crucial for elucidating the feasibility of this algorithm.

### Questions
1. What are the details of the Tile-based Processing scheme? Does the method of tiling vary for images of different resolutions? How does using different tiling methods affect the model's performance?
2. Even though the increase in encoding complexity due to RCC is unavoidable, the authors should discuss the model's complexity in the paper. This discussion would provide valuable insights for the applicability and reference value of this research.

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
This paper proposes a dual semantic compression framework that integrates explicit semantic representations (quantized latents and tag-style prompts) with implicit semantic representations (noise-corrupted latents via reverse-channel coding, RCC) for ultra-low bitrate image compression. The method conditions a diffusion model on explicit semantics while using RCC to encode fine-grained details, and introduces a plugin encoder to control the distortion–perception tradeoff without modifying the decoder. Experiments show strong performance, with 23.49% and 12.25% DISTS-BD-Rate improvements over DiffC on Kodak and DIV2K, respectively.

### Strengths
1) The paper clearly identifies the tradeoff between explicit approaches (semantic faithfulness but texture loss) and implicit ones (rich textures but semantic drift), and provides a principled framework to bridge this gap.
2) The framework is compatible with various base codecs (e.g., DiffEIC, PerCo), and the plugin encoder enables controllable distortion–perception balance without retraining the decoder.
3) The method achieves substantial gains in DISTS-BD-Rate over DiffC and produces visually pleasing reconstructions at extremely low bitrates.

### Weaknesses
1) The paper should include a comparison with ResULIC (ICML 2025), which also explores diffusion-based ultra-low bitrate compression.
2) Encoding and decoding times should be reported to assess real-world usability.
3) Bitrate allocation for different components would clarify efficiency.
4) Results on the CLIC dataset are missing, which is more commonly used in recent compression works.
5) To substantiate claims of controllable perception, the authors should include FID variation curves and report FID scores in Table 1.
6) The computation of CLIPSim should be clarified—since CLIP typically supports 224×224 inputs, how was it applied to full-resolution images?
7) While the overall combination is effective, the individual techniques are largely borrowed from prior work: RCC for image compression originates from DiffC (Theis et al., 2022; Vonderfecht & Liu, 2025), Conditioning diffusion models on latent features follows PerCo and DiffEIC, Tag-style prompts are derived from RAM (Zhang et al., 2024), the primary contribution lies in integrating these existing components rather than introducing fundamentally new mechanisms.

Ke, A., Zhang, X., Chen, T., Lu, M., Zhou, C., Gu, J., & Ma, Z. Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion. In Forty-second International Conference on Machine Learning.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2
