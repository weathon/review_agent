# Autoregressive-based Progressive Coding for Ultra-Low Bitrate Image Compression

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Generative models have demonstrated significant results in ultra-low bitrate image compression, owing to their powerful capabilities for content generation and texture completion. Existing works primarily based on diffusion models still face challenges such as limited bitrate adaptability and high computational complexity for encoding and decoding. Inspired by the success of Visual AutoRegressive model (VAR), we introduce AutoRegressive-based Progressive Coding (ARPC) for ultra-low bitrate image compression, a progressive image compression framework based on next-scale prediction visual autoregressive model. Based on multi-scale residual vector quantizer, ARPC efficiently encodes the image into multi-scale discrete token maps and controls the bitrates by selecting different scales for transmission. For decompression, ARPC leverages the prior knowledge inherent in the visual autoregressive model to predict the unreceived scales, which is naturally the autoregressive generation process. To further increase the compression ratio, we target the VAR as a probability estimator for lossless entropy coding and propose group-masked bitwise multi-scale residual quantizer to adaptively allocate bits for different scales. Extensive experiments show that ARPC achieves state-of-the-art perceptual fidelity at ultra-low bitrates and high decompression efficiency compared with existing diffusion-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a novel framework for ultra-low bitrate image compression , which utilizes a Visual AutoRegressive model (VAR) to encode images into multi-scale discrete tokens. Decompression is achieved by autoregressively generating (predicting) the unreceived scales.

### Strengths
This paper achieves variable bitrate by combining VAR with lossless compression across different scales, effectively integrating image compression with existing pre-trained models.

The paper is well-organized and easy to understand.

### Weaknesses
1. The encoding time increases with the bitrate. Additionally, the encoding process requires a caption model to generate captions, making the encoder relatively "heavy." Could the authors provide the parameter counts for the models used at both the encoder and decoder ends?

2. The ablation study in Figure 7 shows that while each proposed module contributes some improvement, the individual gains are relatively modest. Could it be that directly using VAR for encoding and decoding already offers a performance advantage over other generative compression methods?

3. Further experiments are needed to investigate the impact of different specific mask choices for the Group-Masked (GM-BMSRQ) method.

### Questions
see weakness. My main point of interest is whether VAR itself is already sufficiently effective when used as a codec, and what the performance improvement of this paper's method is compared to a baseline VAR.

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
This paper proposes an ultra-low bitrate image compression method based on the VAR model. Specifically, it introduces group-masked bitwise multi-scale residual quantization (GM-BMSRQ) and lossless re-encoding (LRE) techniques to improve compression ratio. The scale random dropout (SRD) strategy enhances the representation capability of earlier scales. The proposed ARPC method achieves superior perceptual and statistical fidelity on benchmark datasets.

### Strengths
This paper proposes an innovative extreme image compression method based on VAR. The manuscript is well-organized and original.

### Weaknesses
1.The Kodak dataset is a commonly used benchmark in image compression. However, the authors do not provide quantitative or qualitative comparisons of it. Please add these comparisons.
2.The paper does not compare the ARPC method with representative methods, such as token-based and one-step diffusion methods. The former includes the GLC [1] and DLF [2] methods, while the latter includes the RDEIC [3], OSCAR [4], and StableCodec [5] methods. Please add these comparisons for a comprehensive evaluation.
[1]Jia Z, Li J, Li B, et al. Generative latent coding for ultra-low bitrate image compression[C]. CVPR 2024.
[2]Xue N, Jia Z, Li J, et al. DLF: Extreme Image Compression with Dual-generative Latent Fusion[J]. ICCV 2025.
[3]Li Z, Zhou Y, Wei H, et al. RDEIC: Accelerating Diffusion-Based Extreme Image Compression with Relay Residual Diffusion[J]. TCSVT2025.
[4]Guo J, Ji Y, Chen Z, et al. OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates[J]. NeurIPS2025.
[5]Zhang T, Luo X, Li L, et al. StableCodec: Taming One-Step Diffusion for Extreme Image Compression[J]. ICCV2025.
3.The related work lacks one-step, diffusion-based extreme image compression methods.
4.What is the bitwise multi-scale residual quantizer? Please explain how it works in detail.
5.Please explain how the scale random dropout strategy work and introduce it in detail.
6.Please explain the symbol d() in line 221 of the manuscript. Is it the MSE function?
7.For qualitative comparisons, the authors should select results for which the ARPC has the smallest BPP values compared to other competing methods. Please modify it (Fig. 4 and 5.).
8. Would you release the source code and pretrained models? We hope you can also release them to help us understand the ARPC.

### Questions
See the part of weaknesses.

### Soundness
4

### Presentation
2

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
This paper proposes ARPC (Autoregressive-based Progressive Coding) for ultra-low bitrate image compression. An image is quantized into K multi-scale residual token maps; the encoder transmits only the first k (coarse→fine), and a visual autoregressive (VAR) model predicts the untransmitted scales at the decoder (next-scale generation). ARPC also treats the VAR as a probability estimator for arithmetic (lossless) entropy coding of the transmitted tokens and introduces a group-masked bitwise multi-scale residual quantizer (GM-BMSRQ) to reduce bits at coarse scales.

### Strengths
- Dual use of VAR . Using VAR both to compress transmitted bits losslessly (arithmetic coding) and to generate untransmitted scales is technically neat and principled.
- Good progressive design. Encoding into hierarchical residual scales and stopping transmission at k aligns naturally with bitrate adaptation; the decoder’s next-scale VAR completes missing scales, yielding progressive reconstruction. The pipeline (lossless arithmetic decode → VAR generation → image decoder) is clearly illustrated.

### Weaknesses
- Complexity accounting. The claim of 2–6× faster decompression than diffusion is compelling, but a fuller wall-clock/compute-memory breakdown across image sizes and k values would strengthen the efficiency story. 
- The impact analysis of missing text on reconstruction is lacking. The paper mentions using BLIP2 to extract image captions to assist reconstruction, but it does not analyze how the absence of text affects the final results, which is an incomplete approach.
- A more detailed baseline comparison is needed. Recently, numerous diffusion-based image compression methods have emerged, such as StableCodec[1] and ResULIC[2]. These methods have significantly improved decoding speed and performance, particularly StableCodec, which can complete image generation in a single step. This demonstrates that multi-step denoising is no longer a common limitation of diffusion-based methods. The authors need to conduct a more detailed performance and complexity comparison with these methods to highlight the significance of VAR-based approaches.

[1] Zhang, Tianyu, et al. "StableCodec: Taming One-Step Diffusion for Extreme Image Compression." arXiv preprint arXiv:2506.21977 (2025).

[2] A. Ke, X. Zhang, T. Chen, M. Lu, C. Zhou, J. Gu, and Z. Ma, “Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion,” in Proc. Int. Conf. Mach. Learn. (ICML), 2025.

### Questions
- In your introduction, I noticed the term "Infinite shared randomness." What exactly does this refer to? Is it about the impact of random seeds in diffusion models on the denoising process? This doesn’t seem to have a very significant impact.

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
This paper introduces ARPC (AutoRegressive-based Progressive Coding), which leverages a Visual Autoregressive Model (VAR) based on next-scale prediction to for image compression. The VAR also serves as a probability estimator for near-lossless entropy coding. To further improve compression efficiency and semantic representation, the authors propose a Group-Masked Bitwise Multi-Scale Residual Quantizer (GM-BMSRQ) and a Scale Random Dropout (SRD) strategy. Experiments on DIV2K-val and CLIC2020 demonstrate that ARPC surpasses both diffusion-based and VQ/GAN-based baselines in perceptual metrics.

### Strengths
By transmitting only the first k scales and autoregressively completing the rest with VAR, it enables progressive transmission and adaptive bitrate control. The VAR module serves both as a generator and a probability estimator, reducing overall bitrate while supporting lossy and lossless compression. GM-BMSRO further enhances the bitwise multi-scale residual quantizer by group masking to early scales, while SRD encourages these scales to capture richer semantic information. Extensive evaluations across diverse datasets and metrics demonstrate the robustness and practical effectiveness of these techniques.

### Weaknesses
(1) The structural diagram (Figure 2) fails to explicitly show the image caption’s presentation form and functional mechanism. While the text states captions (generated via BLIP2) act as global semantic context to guide VAR’s autoregressive prediction of unreceived scales, Figure 2 lacks labels for the caption generation module.
(2) The grouping logic and parameter settings of GM-BMSRQ lack sufficient justification. The paper proposes dividing the K scales into three groups and masking the last c/2 channels of the first group and the last c/4 channels of the second group to reduce bit cost. However, the rationale behind these key design choices is not explained: (i) the basis for selecting the specific channel masking numbers (e.g., c/2) is unclear, as no experiments compare different masking configurations; (ii) the logic for dividing scales into three groups rather than two or four, and the criteria for allocating scales to each group, is not discussed; (iii) the effect of different channel configurations (e.g., 8, 12, 16 channels) on compression performance is only partially explored through comparisons of c=32 and c=16, leaving the module’s design insufficiently validated and its optimality unproven.
(3) The paper exhibits notable deficiencies in evaluating decoding efficiency and comparing with baseline methods. On one hand, although the ARPC decoding time is reported as 5.39s and claimed to be 2–6× faster than diffusion-based methods, the breakdown of this runtime is not provided—key steps such as VAR autoregressive prediction, arithmetic decoding, and image reconstruction are not individually profiled, making it difficult to identify efficiency bottlenecks and guide further optimization. More importantly, this decoding time is not compared against the current state-of-the-art in ultra-low bitrate image compression. On the other hand, the selection of baseline methods is limited and does not include recent mainstream approaches. To fully validate ARPC’s performance, comparisons should be extended to methods such as DLF[1] (Extreme Image Compression with Dual-generative Latent Fusion), GLC[2], and single-step diffusion methods like StableCodec[3], thereby providing a more comprehensive evaluation and clarifying ARPC’s position relative to the current research frontier.

[1] N. Xue, Z. Jia, J. Li, B. Li, Y. Zhang, and Y. Lu, “DLF: Extreme Image Compression with Dual-generative Latent Fusion,” ICCV, 2025.
[2] Jia, Zhaoyang, et al. "Generative latent coding for ultra-low bitrate image compression." CPVR, 2024.
[3] Zhang, Tianyu, et al. "StableCodec: Taming One-Step Diffusion for Extreme Image Compression." ICCV, 2025.

### Questions
(1)	Are the bits used for text encoding included in the reported bpp statistics? How much bitrate do the captions contribute to the total bpp?
(2)	The paper states that scale random dropout (SRD) is applied with a probability of 0.2 from the fourth scale during training—why is the fourth scale chosen as the starting point for SRD, and how does adjusting the dropout probability (e.g., 0.1 or 0.3) affect the model’s ability to preserve semantic information in earlier scales?
(3)	For images with complex textures (e.g., dense text or fine-grained patterns), does ARPC require adjusting hyperparameters (e.g., number of scales K, channel dimensions of GM-BMSRQ groups) to maintain reconstruction quality, and if so, what guidelines exist for such adjustments?

### Soundness
3

### Presentation
3

### Contribution
2
