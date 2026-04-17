# Texture Vector-Quantization and Reconstruction Aware Prediction for Generative Super-Resolution

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 8

## Abstract
Vector-quantized based models have recently demonstrated strong potential for visual prior modeling. However, existing VQ-based methods simply encode visual features with nearest codebook items and train index predictor with code-level supervision. Due to the richness of visual signal, VQ encoding often leads to large quantization error. Furthermore, training predictor with code-level supervision can not take the final reconstruction errors into consideration, result in sub-optimal prior modeling accuracy. 
 In this paper we address the above two issues and propose a Texture Vector-Quantization and a Reconstruction Aware Prediction strategy. The texture vector-quantization strategy  leverages the task character of super-resolution and only introduce codebook to model the prior of missing textures. While the reconstruction aware prediction strategy makes use of the straight-through estimator to directly train index predictor with image-level supervision. Our proposed generative SR model TVQ&RAP is able to deliver photo-realistic SR results with small computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper enhances vector-quantized (VQ) models for image super-resolution by addressing the limitations of high quantization errors and sub-optimal supervision. The authors introduce Texture Vector-Quantization (TVQ), which focuses the codebook on modeling missing texture priors, and a Reconstruction-Aware Prediction (RAP) strategy that directly trains the index predictor using image-level supervision via a straight-through estimator. Experimental results demonstrate improvement over previous methods.

### Strengths
Clarity: The paper is well-written and easy to follow.

### Weaknesses
Originality, quality, and significance: The paper aims to establish its novelty against "vanilla VQ-based method", which is a trivial baseline that overlooks recent advances, e.g., those mentioned in the related work like AdaCode (Liu et al., 2023) and VARSR (Qu et al., 2025). It is very unclear how the proposed method compares with the "multi-codebook quantization" in AdaCode and the "multi-scale residual quantization mechanism and a large pretrained autoregressive predictor" in VARSR. Why it performs better than state-of-the-art VQ-based methods? Without a detailed analysis and associated experiments with them, it is very difficult to determine the originality, quality and significance of the proposed method.

### Questions
As mentioned above, comparing with "vanilla VQ-based method" is trivial and insufficient. Why does the method outperform SOTA VQ-based methods that adopt more advanced methods? A detailed and thorough analysis against SOTA VQ-based methods is required to justify the contribution of the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a generative super-resolution framework that improves vector-quantization (VQ)-based models by addressing their limitations in texture representation and perceptual alignment. The authors introduce Texture Vector Quantization (TVQ), which decomposes a high-resolution image into a structural component reconstructed directly and a texture component encoded via a compact codebook, allowing the codebook to specialize in modeling fine textures rather than full image content. In addition, a Reconstruction-Aware Prediction (RAP) module is presented to train the code index predictor not only with categorical cross-entropy but also with image-level reconstruction loss, enabling gradients from perceptual and adversarial losses to influence code selection. Together, TVQ and RAP improve texture realism and perceptual quality while maintaining efficient inference. Experiments on ImageNet-Test, RealSR, and RealSet65 demonstrate competitive perceptual metrics (LPIPS, DISTS, CLIPIQA) compared to diffusion-based and GAN-based SR methods, with faster runtime and reduced codebook complexity.

### Strengths
The proposed VQ-based super-resolution framework performs image restoration with a single forward pass, which makes it highly practical for real-time or low-latency applications. In addition, the use of a multi-scale or pyramid-based VQ scheme is conceptually sound and aligns well with the hierarchical nature of image structures, offering an efficient way to capture both global context and fine textures.

### Weaknesses
The proposed TVQ module constructs an image pyramid and learns an autoencoder to model only the high-frequency (texture) components. However, similar multi-scale or hierarchical vector-quantization schemes have already been widely explored in the literature. In particular, the structure of VQ-VAE-2 (Razavi et al., NeurIPS 2019) also employs a coarse-to-fine multi-level quantization strategy that closely resembles the idea of separating and quantizing texture-like information. Although the present work adapts this design for the super-resolution task, several more general and prior VQ-based architectures already realize comparable hierarchical encoding mechanisms. Therefore, the algorithmic novelty of TVQ itself appears moderate rather than substantial.

Recent diffusion-based methods are not included in the experimental comparison. For instance, I²SB (Image-to-Image Schrödinger Bridge, Liu et al., ICML 2023) achieves very low FID on the ImageNet super-resolution benchmark. Moreover, I²SB requires only a small number of denoising steps, which makes its inference significantly faster than conventional diffusion SR models. Therefore, it would be valuable to compare the proposed method against such recent approaches to better contextualize its performance.

In addition, verifying whether Schrödinger-bridge-based SR methods have been evaluated on the same benchmarks (e.g., DIV2K, RealSR, Urban100) would strengthen the paper’s experimental validity.

Some additional clarifications are necessary to improve the manuscript. Details are listed below in the Questions section.

### Questions
Abstract clarity:
The abstract states that “VQ methods do not take the final reconstruction errors into consideration.” However, in standard VQ frameworks, reconstruction loss (e.g., MSE, perceptual loss) is typically a major objective used to train the encoder, decoder, and quantizer jointly.


Page 2 (alignment with image quality):
The manuscript claims that existing VQ-based methods “do not strictly align with image quality.” However, methods such as VQ-GAN and its successors explicitly introduce GAN loss and perceptual loss terms to improve reconstruction realism and perceptual fidelity. Please clarify in what sense those methods “do not align” with image quality.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces TVQ&RAP, a novel generative SR framework that addresses limitations in existing code prediction-based models. These models often suffer from high quantization error when encoding complex visual features and sub-optimal prior modeling because their code-level training objectives are not aligned with the final goal of image quality. The proposed solution, TVQ, first disentangles the image into a structure component and a texture component, applying VQ only to the simpler texture space. This is complemented by Reconstruction Aware Prediction, a strategy that trains the index predictor using direct image-level supervision by backpropagating final reconstruction losses via STE. Experiments on synthetic and real-world datasets demonstrate that the method achieves competitive generative SR results while maintaining high computational efficiency.

### Strengths
- The paper clearly identifies two key issues in VQ-based SR: quantization error and sub-optimal training; and proposes two intuitive solutions: TVQ and RAP.
- The model achieves competitive perceptual quality (LPIPS, MUSIQ, etc.) while being remarkably efficient, significantly faster than diffusion-based competitors.
- The benefits of both TVQ and RAP are validated by strong, separate ablation studies.

### Weaknesses
- The paper could better articulate the key advantages of its texture-only codebook (TVQ) compared to the direct u-net skip connection mechanisms used in other VQ-SR methods like FeMaSR or CodeFormer.

- Although it is claimed than TVQ space is simpler, the final code prediction accuracy is still quite low. A deeper discussion on why this occurs and its implications would be beneficial.

- It would be better if the paper could discuss how TVQ is distinguished and superior to other VQ enhancement techniques, such as the RQ-VAE mentioned in the related works. For example, comparison and analysis with VARSR might be better to highlight in main text.

In conclusion, this paper is well-written and well-motivated. The method design is simple and clear, and the model shows good performance with remarkable efficiency. The two proposed components, TVQ and RAP, are independently validated by strong ablation studies.

### Questions
Please see weakness points above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a Texture Vector-quantization (TVQ) method to mitigate vector-quantization encoding errors by disentangling structure and texture features, thereby reducing codebook complexity. Furthermore, the paper introduces a Reconstruction Aware Prediction (RAP) strategy, which enables the direct optimization of perceptual quality by incorporating image-level supervision to train the predictor. As a result, the proposed model achieves state-of-the-art performance. It also demonstrates advantages over competing methods in terms of inference speed and computational efficiency.

### Strengths
1. The core design is intuitive. It directly leverages the inherent characteristics of the super-resolution task. The low-resolution (LR) input already contains the majority of the low-frequency structure information. Therefore, it only needs to realistically synthesize the missing high-frequency texture details. 
2. This paper clearly identify two fundamental weaknesses in traditional VQ-based models and propose a specific solution for each. TVQ is proposed to solve the problem of traditional VQ methods attempt to encode the entire complex visual space, which leads to difficult discretization and significant quantization errors. RAP is proposed to train the predictor and force it to optimize for the final perceptual quality of the reconstructed image.
3. Comprehensive experiments and ablation studies validate the method's effectiveness and efficiency.

### Weaknesses
1. Some existing methods, such as Tokenflow, have already explored the concept of improving VQ through feature disentanglement. Therefore, the novelty of the TVQ component might be considered somewhat lacking.
2. Check grammatical and spelling errors (e.g., "structrue").

### Questions
1. The paper's core claim is that removing structure information reduces feature space redundancy. To what extent is this redundancy reduced? I would suggest providing a more quantitative analysis or visualization results of feature space to substantiate this claim.
2. How exactly does the texture codebook improve codebook efficiency? Traditional VQ methods often face challenges with low codebook utilization or codebook collapse. I would like to see an analysis to demonstrate that TVQ achieves a higher and more effective utilization of its codebook items compared to the baseline.

### Soundness
3

### Presentation
3

### Contribution
2
