# CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
As 3D Gaussian Splatting (3DGS) is increasingly adopted in various academic and commercial applications due to its high-quality and real-time rendering capabilities, the need for copyright protection is growing. At the same time, its large model size requires efficient compression for storage and transmission. However, compression techniques, especially quantization-based methods, degrade the integrity of existing 3DGS watermarking methods, thus creating the need for a novel methodology that is robust against compression. To ensure reliable watermark detection under compression, we propose a compression-tolerant 3DGS watermarking method that preserves watermark integrity and rendering quality. Our approach utilizes an anchor-based 3DGS, embedding the watermark into anchor attributes, particularly the anchor feature, to enhance security and rendering quality. We also propose a quantization distortion layer that injects quantization noise during training, preserving the watermark after quantization-based compression. Moreover, we employ a frequency-aware anchor growing strategy that enhances rendering quality by effectively identifying Gaussians in high-frequency regions, and an HSV loss to mitigate color artifacts for further rendering quality improvement. Extensive experiments demonstrate that our proposed method preserves the watermark even under compression and maintains high rendering quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes CompMarkGS, a watermarking method for 3D Gaussian Splatting (3DGS) that is specifically designed to remain robust under quantization-based compression. The method leverages anchor-based 3DGS, embedding watermarks into anchor features rather than direct Gaussian attributes. Key technical contributions include: (1) a quantization distortion layer (QDL) that simulates quantization noise during training, (2) a frequency-aware anchor growing (FAG) strategy that selectively densifies high-frequency regions, and (3) an HSV loss to mitigate color artifacts. Experiments demonstrate robustness to compression and various attacks.

### Strengths
## Strengths

1. The paper addresses a critical gap in 3DGS watermarking - robustness to model compression. As the abstract correctly notes, existing watermarking methods fail dramatically under quantization-based compression, making this a timely and valuable contribution.

2. The choice of anchor-based 3DGS for watermarking is well explained - the implicit representation through MLPs provides additional security compared to direct attribute modification. The quantization distortion layer can adapt differentiable distortion layers to the 3DGS compression context.

3. Well-structured paper with clear motivation and problem statement

### Weaknesses
## Weaknesses

1. The method requires a pre-trained HiDDeN decoder, which is a significant limitation and can be vulnerable if any malicious user applies a specific adversarial attack specially effective against the HiDDeN decoder.

2. The choice of the decoder is basically following the current mainstream 3DGS watermark decoder, although the paper tests with the SSL decoder, but it underperforms with the HiDDeN decoder. The contribution of this work on the message decoding side is trivial.

3. At the message embedding side, the anchor-based hiding is built on top of the While the frequency-based growing is similar to the Frequency Guided Densification (FGD) in 3D-GSW. So, although technically the CompMarkGS sounds, it still relies on the existing well-established technology of Scaffold-GS[1] and 3DGSW[2].

4. No capacity analysis. What's the theoretical maximum bit length? Is it limited by the pre-trained message decoder?

[1] Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

[2] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

### Questions
## Questions

1. What is the relationship between noise magnitude in QDL and actual compression distortion?

2. CompMarkGS extracts watermarks from rendered images, but can the message be directly from the 3D Gaussians? What if adversaries extract and reuse anchor subsets without rendering? 

3. What happens if adversaries know watermarks are embedded in anchor features and specifically target them, such as adding noise to anchor features, or re-training MLPs?

4.  In Appendix E, can you explain the difference between the robustness toward pruning and quantization? What can be the key property against such two compressions, and why can CompMarkGS outperform the baselines? In addition, the baselines also show some robustness towards these two compressions, which show over 90% accuracy.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes CompMarkGS, a watermarking method for 3D Gaussian Splatting (3DGS) designed to remain robust under quantization-based compression. The method builds upon anchor-based 3DGS architecture by embedding a learnable watermark feature into anchor attributes, which are then processed through MLPs to generate Gaussian parameters. To preserve watermark integrity during compression, the authors introduce a Quantization Distortion Layer (QDL) that simulates quantization noise during training. Additionally, a Frequency-Aware Anchor Growing (FAG) strategy selectively densifies Gaussians in high-frequency regions to mitigate quality degradation, and an HSV loss addresses color artifacts. Experiments on multiple datasets demonstrate that the method maintains high bit accuracy and rendering quality after compression with various quantization schemes, outperforming existing 3DGS watermarking baselines that suffer severe degradation under compression.

### Strengths
S1: The paper identifies a critical gap by tackling watermark robustness against quantization-based compression, which is essential for real-world deployment but has been overlooked by existing methods that focus primarily on image-domain distortions.

S2: The paper validates the method across multiple datasets, compression schemes, image distortions, model distortions, and message capacities, demonstrating broad applicability and robustness.

S3: The method maintains high rendering quality while embedding watermarks, achieving competitive PSNR and SSIM scores compared to baselines, demonstrating that the watermarking process does not significantly degrade the visual fidelity of 3DGS models.

### Weaknesses
W1: The core contribution, the Quantization Distortion Layer (QDL), is a straightforward application of differentiable distortion layers from digital watermarking and quantization-aware training from model compression. Injecting noise during training to simulate downstream distortions is standard practice in watermarking and QAT literature. The adaptation merely changes the noise source from JPEG/cropping to quantization, which is incremental rather than novel. 

W2: The paper does not explicitly state whether watermark embedding requires training a new model for each scene or uses a universal encoder. If the method requires per-scene optimization, then for a content creator with 1000 scenes, they must train 1000 separate watermarked models. What is the amortized training cost per scene?

W3: The paper does not clearly specify how watermark messages are sampled during training. Specifically, it remains unclear whether each scene trains with a single fixed message throughout all 30,000 iterations or different random messages sampled at each iteration. If the method requires training each scene with a predetermined fixed message, this introduces significant scalability concerns for real-world deployment.

W4: The QDL (Eq. 4) uses uniform random noise U(-1/2, 1/2)·q_i with per-anchor independent sampling to simulate quantization errors. However, real compression methods (HAC and ContextGS) employ non-uniform quantization with spatial dependencies. This mismatch raises questions about whether the watermark is actually robust to real compression or merely to the specific noise pattern seen during training.

W5: Technically incorrect use of SSIM in Equation 5. The paper computes "pixel-wise SSIM" as P_error(p) = 1 - SSIM(I_hf(p), I'_hf(p)), which fundamentally misunderstands SSIM. The SSIM operates over local windows, not individual pixels. Computing SSIM on single pixel values is meaningless and violates the metric's core design. This error propagates through the entire FAG strategy, invalidating the median-based thresholding mechanism that relies on P_error(p).

W6: The paper uses DWT for watermark extraction but switches to DFT for high-frequency detection without justification. More critically, the claimed separation between "low-frequency watermark" and "high-frequency quality enhancement" is contradictory: all anchor features (including those from FAG) contain the watermark via f^w = f + tanh(f_wf) (Eq. 3), and the resulting Gaussians affect all frequency bands. If FAG-grown anchors in high-frequency regions also embed watermarks in low frequencies, how does this resolve the claimed conflict? The core motivation for FAG appears flawed.

W7: The binary masks M_c(p) are constructed based on whether pixels fall within predefined hue ranges, but the loss penalizes all pixels within these ranges regardless of whether they represent artifacts or legitimate content. A genuinely red pixel in both the rendered and ground truth images still contributes to L_hsv, which seems to be incorrect. Additionally, it remains unclear whether |Ω| in Equation 8 represents total pixels or only those where M_c(p)=1. Most critically, Table 5 shows that removing L_hsv only degrades LPIPS by 0.002, suggesting negligible practical benefit despite the added complexity.

W8: While Appendix E.3 tests one pruning-based method (LightGaussian), the main evaluation (Table 1) focuses exclusively on two quantization schemes from the same architectural family (HAC and ContextGS), both of which are anchor-based compression methods with similar quantization mechanisms. Other vector quantization methods mentioned in related work (Lee et al., 2024a; Navaneet et al., 2024) remain untested, limiting claims about generalizability across diverse compression paradigms.

### Questions
Q1: How do you map 3D Gaussians to 2D high-frequency pixels? When multiple Gaussians project to one pixel, which determines labeling? When an anchor's K Gaussians partially overlap high-frequency regions, how is growth decided? What are the coverage rates and how sensitive is performance to different mapping strategies?

### Soundness
3

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
Targeting on copyright protection of 3D Gaussian Splatting (3DGS) models after compression, this paper proposes an anchor-based 3DGS watermarking framework, CompMarkGS, which directly adds learnable watermark-embedding features to the anchor features to enhance robustness against quantization. During training, a Quantization Distortion Layer (QDL) injects uniform noise with a learnable scale to explicitly simulate quantization rounding errors, aligning the model to be robust against post-compression. In parallel, a Frequency-Aware Anchor Growth (FAG) strategy adds anchors only in high-frequency error regions to balance visual quality, and an HSV loss is used to suppress color artifacts. Experimental results under anchor-based pipelines on Blender, LLFF, and Mip-NeRF360 evaluate performance before and after two mainstream quantization-based compressors (HAC, ContextGS), showing advantages over WateRF, GaussianMarker, 3D-GSW, and GuardSplat in both bit accuracy and visual quality; the trend holds across different message lengths and when swapping in an alternative decoder (self-supervised DINO). It is claimed that this is the first work to systematically address quantization-compression robustness for 3DGS watermarking with end-to-end validation.

### Strengths
The paper injects learnable quantization noise into anchor features via QDL, bringing the compressor’s distortion statistics forward into training, which is distinct from prior differentiable perturbation layers that only model image-domain distortions. By focusing squarely on robustness to quantization-based compression, the proposed FAG separates, in the frequency domain, the tension between visual enhancement (high frequencies) and watermark carrying (low-frequency LL), making apt use of anchor-based 3DGS structural priors. Overall, it is a new combination of structural choice + distortion modeling + frequency-aware growth. 
The work offers comprehensive validation across three datasets and two compressors, evaluates both pre/post compression, covers image- and model-domain perturbations, varies message length, and tests multiple decoders; ablations show QDL/FAG/HSV each contributes substantially. After ContextGS/HAC compression, it outperforms strong baselines in bit accuracy and PSNR/SSIM. 
In general, the method diagram and the training/extraction flow are explained clearly. As the anchor-based designs and compression are mainstream in the 3DGS ecosystem, addressing compression-induced watermark failure carries clear practical importance.

### Weaknesses
First, compression generalization could be strengthened. The current evaluation mainly covers HAC/ContextGS. The authors may consider adding different quantization bit-widths, per-channel/per-layer non-uniform quantization, and compressors with error feedback or post-entropy coding, to better match industrial diversity. Also, the QDL calibration strategy and mismatch-sensitivity analysis need to be reported for each setting. 
Regarding baseline adaptation, GuardSplat should be moved from the SH space to an anchor-feature bias, which may introduce out-of-domain effects. It would help to report the native setting and a compatibility setting separately, or add a comparison axis of “native 3DGS + native compression”. 
More targeted model-domain attacks and structural perturbations may be added to assess the limits of QDL/FAG under stronger adversaries.

### Questions
The frequency-aware anchor growth (FAG) sounds effective, but I’d like a more direct view of its costs and benefits. Could you add a page comparing model size, the number of anchors/Gaussians, and rendering latency (FPS or ms/frame) with and without FAG, and include a few figures showing where the added anchors are placed (aligned with your error/frequency heatmaps)? This would help assess its practical engineering overhead.

For “unknown/recompression” scenarios, double compression and bitrate switching across platforms are common in practice. I suggest adding a set of black-box re-compression experiments (e.g., compress with ContextGS then HAC, or multiple bitrate chains within the same compressor) to examine changes in bit accuracy and visual quality; also include stronger model-level attacks to help readers understand the method’s resilience under less friendly pipelines.

Regarding baselines, some methods need to be ported from their native representation space to the anchor-feature space. To avoid bias from implementation shifts, could you report the “native setting” (as faithful to the original paper as possible) and the “compatibility-modified” version separately, with key hyper-parameters and implementation details? This would make reproduction easier and fairness easier to judge.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel compression-tolerant anchor-based framework for 3DGS watermarking that protects the ownership of 3DGS model. The authors focus on preserving watermark integrity and rendering quality effectively both before and after compression. The framework is evaluated on multiple datasets, demonstrating that it can achieve state-of-the-art performance.

### Strengths
1.	A compression-tolerant anchor-based watermarking method is proposed to embed a learnable watermark embedding feature into the anchor feature, which improves robustness against model compression.
2.	The authors perform experiments on different datasets and investigate robustness by measuring bit accuracy under various distortions before and after compression.

### Weaknesses
1.	What is the true innovation of the quantization distortion layer? Is it merely a transfer of technique from the 3DGS compression method?
2.	What is the true advantage of HSV loss? Can you compare HSV loss with the effects of using perceptual loss?
3.	The experiments were primarily validated on anchor-based 3DGS. Does it exhibit the same performance for standard 3DGS?
4.	In Table 2, for model distortion robustness, how does the method perform under additional 3D geometric attacks, such as 3D translation, rotation, or crop-out (which is evaluated in GaussianMarker) ?
5.	The text in Figure 2 may need to be revised for clearer expression.

### Questions
See the weakness

### Soundness
2

### Presentation
2

### Contribution
2
