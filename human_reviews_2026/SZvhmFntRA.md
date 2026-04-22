# Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Arbitrary-scale super-resolution (ASSR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs with arbitrary upsampling factors using a single model, addressing the limitations of traditional SR methods constrained to fixed-scale factors (\textit{e.g.}, $\times$ 2). Recent advances leveraging implicit neural representation (INR) have achieved great progress by modeling coordinate-to-pixel mappings. However, the efficiency of these methods may suffer from repeated upsampling and decoding, while their reconstruction fidelity and quality are constrained by the intrinsic representational limitations of coordinate-based functions. To address these challenges, we propose a novel ContinuousSR framework with a Pixel-to-Gaussian paradigm, which explicitly reconstructs 2D continuous HR signals from LR images using Gaussian Splatting. This approach eliminates the need for time-consuming upsampling and decoding, enabling extremely fast ASSR. Once the Gaussian field is built in a single pass, ContinuousSR can perform arbitrary-scale rendering in just 1ms per scale. Our method introduces several key innovations. Through statistical analysis, we uncover the Deep Gaussian Prior (DGP) and propose DGP-Driven Covariance Weighting, which dynamically optimizes covariance via adaptive weighting. Additionally, we present Adaptive Position Drifting, which refines the positional distribution of the Gaussian space based on image content, further enhancing reconstruction quality. Extensive experiments on seven benchmarks demonstrate that our ContinuousSR delivers significant improvements in SR quality across all scales, with an impressive 19.5× speedup when continuously upsampling an image across forty scales.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a 2D Gaussian splatting based arbitrary-scale SR model. With the help of Deep Gaussian Prior and Adaptive Position Drifting, the proposed ContinuousSR achieves excellent performance and ultra-fast inference speed.

### Strengths
1.	The author organized a soundly description about the motivation, the detailed technique and experimental implementations, which makes the method looks valuable especially when implementing the method into industrial applications. 

2.	The DGP proposed in this paper appears to be quite novel and reliable, effectively capturing the prior information of images, which indeed helps the model generalize across images with different content.

3.	The author provides large quantities of experiments to validate the effectiveness of the proposed method. The results reported in the paper also seem highly competitive in the current super-resolution field, where performance is approaching saturation.

### Weaknesses
1.	Please unify the style of the references. The authors neglected to maintain consistency in the references regarding conference name capitalization, full conference names, abbreviated conference names, and title capitalization. The ununified samples in the paper are listed in the following, 

“Xiaoyi Liu and Hao Tang. Difffno: Diffusion fourier neural operator. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 150–160, 2025. 2, 3” 

“Jaewon Lee and Kyong Hwan Jin. Local texture estimator for implicit representation function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1929–1938, 2022a. 2, 4”

2.	The inference time in Table.1 seems not soundly. From my point of view, the author seems not to indicate the acceleration strategy in the paper, then how could ContinuousSR runs even much faster than the very lightweight model Meta-SR?

3.	Please indicate the detailed settings in training progress, especially the data preparation. The author says they fix the GT size to 256, while randomly sampling the scaling factor, then if want to do parallel training, in the code implementation, the author must pad the LR image to a fixed size. However, in this way, it will waste huge computational resources since the padding area will introduce additional computational burden. What’s more, why not adopting the same strategy in most the existing method? They just fix the LR image size while randomly sampling the scaling factor.

Minor:

1.	Is there any GPU memory usage comparison results with state-of-the-art models?

2.	Although different ASSR models might adopt different training settings, but I hope the author could provide some fair experimental settings, such as the fixed LR size, the same scaling factors (1, 4).

### Questions
This paper proposed a novel and valuable method in ASSR task. While I am going to give weak accept towards the whole quality of the paper, I hope the author could provide some reliable answers to the weakness of the paper. I am willing to increase the rating score if the author can address the questions I raised in a satisfactory manner.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. This paper proposes ContinuousSR, a Pixel-to-Gaussian paradigm that reconstructs a continuous 2D high-resolution (HR) signal (a Gaussian field) from a low-resolution (LR) image in a single pass, enabling fast arbitrary-scale rendering via Gaussian rendering
2. The paper identifies a Deep Gaussian Prior (DGP) on Gaussian covariance parameters from statistics over ~40k images and uses it to constrain the covariance space.

### Strengths
1. Combining ideas from 3D Gaussian Splatting to model continuous 2D image signals is very interesting, and the paper convincingly demonstrates the effectiveness of this idea。
2. Strong empirical efficiency: consistent large speedups, stable memory across scales, and favorable FLOPs/runtime on Manga109 single-scale tests and multi-scale averages.
3. The upsampling/rendering component is an explicit, trainable parameter-free module at inference, which is both elegant and practical.

### Weaknesses
1. I think the theoretical connection between GMM-based formulation and the actual HR rendering needs clarification (see Question 1).
2. Dependence on the prior dictionary: performance hinges on the size and sampling of the DGP-derived kernel dictionary; generalization to out-of-distribution (OOD) content or non-natural images is unclear.

### Questions
1. In Figure 3 you mention using differentiable rendering to render the HR image from a set of Gaussians. Are you using the 3DGS rendering pipeline? If so, I think the GMM-related theory in Eq. (3) may be inappropriate, because 3DGS rasterization is based on alpha-blending theory rather than GMM theory. If not, how is your differentiable rendering implemented, and how does it match your GMM formulation?

2. Since there is a “Deep Gaussian Prior,” is there also a “Deep Fourier Prior”? Is it possible to simply replace Gaussians with Fourier bases of different frequencies and linearly combine them to obtain the final HR image? Using Fourier bases to describe the composition of a continuous HR function may be more in line with traditional intuition. I recall some works a few years ago that applied similar ideas to arbitrary-scale super-resolution, but I forgot the names; it would be better to add them in the related work section.

3. I think the Gaussian rendering parts can be further refined. For example, the theory and methods of “Mip-Splatting” (CVPR 2024) might be applicable to your current framework. I’d like to hear your thoughts on this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ContinuousSR, a novel framework for arbitrary-scale super-resolution that leverages 2D Gaussian modeling to directly reconstruct continuous high-resolution signals from low-resolution images in a single pass. By introducing the Deep Gaussian Prior (DGP) and key modules such as DGP-Driven Covariance Weighting and Adaptive Position Drifting, the method achieves both high-quality reconstruction and ultra-fast rendering speed, outperforming state-of-the-art methods in both efficiency and performance across multiple benchmarks.

### Strengths
1. The paper derives a Deep Gaussian Prior (DGP) from large-scale natural image statistics, effectively addressing local minima in Gaussian kernel optimization and proposing a novel continuous modeling framework for arbitrary-scale super-resolution.
2. Extensive experiments on multiple benchmarks and scales, evaluated by PSNR, SSIM, FID, and DISTS, clearly verify the effectiveness of ContinuousSR and show its potential in other low-level vision tasks such as deraining.
3. The method fully utilizes the continuity of Gaussian kernels, achieving one-pass generation and multi-scale rendering with high efficiency and speed.

### Weaknesses
1. A spelling error appears on line 175 of the paper: “the theresponding discrete pixel grids” should be “the corresponding discrete pixel grids.”
2. The paper leaves several important implementation details insufficiently described. In particular, the sampling strategy used to generate Gaussian covariances from the Deep Gaussian Prior and the construction of the predefined Gaussian kernel dictionary are not clearly explained. Without clarification of how the sampling and dictionary size were determined, it is difficult to assess the reproducibility and design rationale of these components.

### Questions
1. What specific sampling strategy was used to draw Gaussian covariances from the DGP in Eq. (6)? 
2. Each low-resolution pixel is assigned four Gaussian kernels, but the reason for this specific choice is unclear. Why are four Gaussian kernels assigned per LR pixel? 
3. The predefined Gaussian kernel dictionary plays a key role in covariance weighting, yet its size and the rationale behind it are not specified. How large is the predefined Gaussian kernel dictionary, and how was this number determined?
4. The model is trained on data generated by bicubic downsampling, and the evaluation is also conducted on synthetic datasets created in the same way. Has the performance of ContinuousSR been tested on real-world arbitrary-scale super-resolution datasets such as COZ[1]?

[1] Fu, Huiyuan, et al. "Continuous optical zooming: A benchmark for arbitrary-scale image super-resolution in real world." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ContinuousSR, an approach to arbitrary-scale super-resolution (ASSR) that reconstructs continuous high-resolution (HR) signals from low-resolution (LR) inputs using 2D Gaussian Splatting. The core idea is a Pixel-to-Gaussian paradigm: instead of repeatedly upsampling and decoding features as in coordinate-based implicit neural representations (e.g., LIIF, CiaoSR), the method builds a single continuous Gaussian field from which HR images at arbitrary scales can be rendered in ~1 ms per scale.

The authors propose three technical components:

1. Deep Gaussian Prior (DGP) – a statistical observation that Gaussian parameters from natural images follow a bounded Gaussian distribution.

2. DGP-Driven Covariance Weighting (DDCW) – a mechanism that learns adaptive weights over a pre-sampled dictionary of Gaussian kernels to stabilize training.

3. Adaptive Position Drifting (APD) – a bounded offset learning scheme that adjusts Gaussian centers based on image content to refine spatial alignment.

Evaluated on seven benchmarks (Set5, Set14, B100, Urban100, Manga109, DIV2K, LSDIR) with SwinIR and HAT backbones, ContinuousSR achieves slightly higher PSNR (up to +0.18 dB on Manga109) and SSIM than state-of-the-art GSASR, while being up to 19.5× faster when rendering across 40 scales. Memory usage is also markedly lower, enabling high-scale rendering (×32–×48) where prior INR methods run out of memory.

### Strengths
Ultra-Fast Arbitrary Scaling: The proposed method eliminates iterative upsampling, achieving real-time rendering at arbitrary scales (≈1 ms per scale after one forward pass). This is a ~20× speedup over implicit models when generating a continuous zoom or multiple outputs, making it highly attractive for practical use where efficiency is crucial.

High Reconstruction Quality: It delivers superior super-resolution fidelity across a wide range of scales. On multiple benchmarks (e.g. Urban100, Manga109), the approach surpasses previous state-of-the-art methods by a significant margin (~0.8–0.9 dB PSNR improvement at 4× upsampling), and also achieves better SSIM and FID scores, indicating improvements in perceptual quality. The Gaussian representation better preserves structure and details (sharper textures, as shown in qualitative examples) compared to coordinate-MLP approaches.

Innovative Use of Gaussian Splatting with Deep Prior: The paper cleverly integrates Gaussian splatting (recently popular in 3D/NeRF tasks) into image SR, and addresses its training difficulties via a Deep Gaussian Prior. By pre-characterizing the typical covariance range of natural images, the method avoids converging to poor local optima and reduces the solution space. This DGP-driven covariance weighting mechanism and the adaptive position drifting are novel contributions that improve modeling flexibility (allowing anisotropic Gaussian shapes and content-aware placement) while keeping the optimization stable.

Excellent Memory Efficiency: The continuous Gaussian framework uses memory proportional to the number of Gaussians (which is tied to input resolution) rather than the output pixel count. Consequently, the method can handle very large scale factors (e.g. 16×, 32×) without running out of memory, unlike prior methods (LIIF, CiaoSR) that exhaust GPU memory at high scales. This efficient scaling is beneficial for ultra-high-resolution outputs and demonstrates a well-designed pipeline.

Thorough Evaluation: The authors validate their approach on seven diverse datasets and include extensive comparisons to nine prior methods (including recent ones like GaussianSR and GSASR). They also perform ablation studies to justify each component, and even test a joint deraining+SR scenario to show the model’s robustness in adverse conditions. This thorough experimentation strengthens confidence in the method’s effectiveness and generality.

### Weaknesses
Complex Training Pipeline: The solution introduces multiple components (DGP-based kernel dictionary, adaptive weighting network, position drift module, etc.), making the overall pipeline more complex than some prior approaches. Training requires careful setup – for instance, deriving the DGP involved 700 GPU-hours of optimization on external data, and the model must learn to combine predefined kernels and offsets correctly. This complexity might make the approach harder to reproduce or extend without the provided code, and it suggests a heavy computational cost in pre-processing and training (though inference is efficient).

Heavily Engineered Solution: While effective, the novelty is somewhat incremental in that it builds upon known ideas from related domains (e.g. using Gaussian splats instead of coordinate MLPs, which was explored by GaussianSR and GSASR). The main contributions lie in engineering a workable solution (using a learned prior and adaptive modules) rather than fundamentally new theory. The approach may be viewed as a clever combination of existing techniques (INR networks, Gaussian rendering, learned priors) tailored to overcome a training hurdle, which, despite being valuable, might not be conceptually groundbreaking.

Assumptions of the Deep Gaussian Prior: The method’s efficacy leans on the assumption that natural image content adheres to the learned Gaussian parameter distributions. If an input deviates from this prior (e.g. very sparse or non-photographic images), the fixed kernel dictionary might become suboptimal. The paper does not explore how sensitive the model is to the DGP assumptions or how it would perform on out-of-distribution data (such as medical imagery or vector graphics). In other words, the generalization of the DGP to all image types remains a potential concern.

Limited Discussion of CGM and Color Fidelity: The Color Gaussian Mapping component is not described in detail, leaving some ambiguity about how color from the LR image is transferred or refined in the continuous representation. If color assignment to Gaussians is naive (e.g. directly using LR pixel colors), it could limit the method’s ability to add high-frequency color details. More explanation is needed on how color is handled and whether the model can recover color nuances at high scales. This lack of clarity is a minor presentation issue but also a technical point that could affect the perceived color fidelity of results.

Potential Minor Artifacts: Representing an image as a sum of Gaussians could inherently introduce smoothing, especially if a Gaussian’s covariance is large. The paper focuses on PSNR/SSIM, which favor fidelity, but it’s unclear if there are any artifacts such as slight blurring or spatial shifts due to APD. For example, Adaptive Position Drifting moves Gaussian centers for better alignment; however, without constraints, this might risk slight geometric distortions (if many Gaussians shift from their original pixel locations). The authors do not report any artifacts, but a discussion on how APD balances flexibility with spatial accuracy would strengthen the work.

Marginal quantitative gains: The reported PSNR improvements over GSASR (≈0.08–0.18 dB) are small; hence claims of a “new paradigm” feel overstated given access to those prior results.

Ablation gaps: Table 4 compares DGP to uniform sampling but omits a baseline with directly learned covariances (no dictionary), leaving uncertainty about DDPW’s absolute necessity.

### Questions
Q1. Novelty vs Prior Work: How does ContinuousSR fundamentally differ from GaussianSR (which already models each pixel as a Gaussian field) and GSASR (which performs scale-aware 2D Gaussian Splatting)? Beyond eliminating per-scale decoding, what conceptual innovation justifies a new paradigm claim?

Q2. Can you clarify whether DGP is fixed or fine-tuned during training? How would results change if covariances were learned end-to-end without the predefined dictionary?

Q3: How is the Color Gaussian Mapping (CGM) implemented and how crucial is it for final image quality? For instance, do you simply assign each Gaussian an RGB value from the LR feature map, or is there an additional refinement to predict high-frequency color details? A clearer explanation of CGM’s role would help understand how color fidelity is maintained for large upscaling factors.

Q4: Does each input pixel strictly correspond to one Gaussian, or can multiple Gaussians represent a single pixel/region? The current description implies one Gaussian per LR pixel. If so, have you considered allowing a more adaptive number of Gaussians (e.g., splitting a pixel’s Gaussian to capture complex textures, or merging in flat areas)? This might further improve representation capability for highly textured regions.

Q5: How sensitive is the model to the assumed Deep Gaussian Prior if applied to very different data distributions? For example, would a model trained on natural images struggle with domain-specific content (such as medical images or line drawings) because the covariance distributions differ? Would fine-tuning the DGP (or the predefined kernel set) be necessary in such cases?

Q6: What measures are in place to ensure that Adaptive Position Drifting does not distort the image structure? Since APD shifts Gaussian centers based on content, do you impose any regularization or limits on these offsets? Clarification on how much drift is typically learned (e.g., fraction of a pixel) and its effect on alignment with ground-truth details would be helpful.

Q7: Can you elaborate on the rendering performance in extreme cases? The reported 1 ms rendering is impressive – what output resolution and hardware does this refer to? If one were to render an extremely high-resolution image (say 4K or 8K) from a small LR input, does the rendering remain near-instantaneous, and are there any memory or precision considerations with the GPU rasterizer at those scales?

Q8. Reproducibility: Will you release the code, pretrained models, and the 40 000-image covariance statistics used to construct the DGP?

### Soundness
3

### Presentation
4

### Contribution
2
