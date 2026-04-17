# Spatial-Spectral Binarized Neural Network for Panchromatic and Multi-spectral Images Fusion

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Remote sensing pansharpening aims to reconstruct spatial-spectral properties during the fusion of panchromatic (PAN) images and low-resolution multi-spectral (LR-MS) images, finally generating the high-resolution multi-spectral (HR-MS) images. Although deep learning-based models have achieved excellent performance, they often come with high computational complexity, which hinder their applications on resource-limited devices. In this paper, we explore the feasibility of applying the binary neural network (BNN) to pan-sharpening. Nevertheless, there are two main issues with binarizing pan-sharpening models: (i) the binarization will cause serious spectral distortion due to the inconsistent spectral distribution of the PAN/LR-MS images; (ii) the common binary convolution kernel is difficult to adapt to the multi-scale and anisotropic spatial features of remote sensing objects, resulting in serious degradation of contours.
To address the above issues, we design the customized spatial-spectral binarized convolution (S2B-Conv), which is composed of the Spectral-Redistribution Mechanism (SRM) and Gabor Spatial Feature Amplifier (GSFA). Specifically, SRM employs an affine transformation, generating its scaling and bias parameters through a dynamic learning process. GSFA, which randomly selects different frequencies and angles within a preset range, enables to better handle multi-scale and-directional spatial features.
A series of S2B-Conv form a brand-new binary network for pan-sharpening, dubbed as S2BNet. Extensive quantitative and qualitative experiments have shown our high-efficiency binarized pan-sharpening method can attain a promising performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles remote sensing pansharpening—fusing a high-resolution panchromatic (PAN) image with a low-resolution multispectral (LR-MS) image to produce a high-resolution multispectral (HR-MS) image—under tight compute and memory constraints. The authors propose S2BNet, a largely binarized U-shaped network for efficient pan-sharpening. Its core is a Spatial-Spectral Binarized Convolution (S2B-Conv) that combines with Spectral-Redistribution Mechanism (SRM) and Gabor Spatial Feature Amplifier (GSFA).
S2BNet places these modules within a U-shaped architecture (dual encoders, bottleneck, dual decoders with skip connections), using binary convolutions for most layers and a small number of full-precision layers. Experiments on GaoFen-2, WorldView-2, and QuickBird (4-band) with standard Wald protocol and metrics (e.g., PSNR, SSIM, SAM, ERGAS, Q-index, QNR and its components) show that S2BNet outperforms other binary approaches and is competitive with some full-precision baselines. Ablation studies indicate both SRM and GSFA contribute measurable gains, and binarizing different parts trades accuracy for parameter/OPs reduction. The authors claim code will be released.

### Strengths
- Practical and targeted adaptation of BNNs to pansharpening, addressing two concrete pain points (spectral mismatch and anisotropic/multi-scale spatial textures) via SRM and GSFA.
- Extensive experiments on three common satellites (GF-2, WV-2, QB), with strong baselines including recent CNN/Transformer and binary methods; both quantitative and qualitative results; ablations showing each component’s effect and binarization placement trade-offs.
- Architecture and modules are explained clearly; equations for SRM and Gabor filters help reproducibility.

### Weaknesses
Limited novelty: SRM is essentially a bounded affine calibration and GSFA leverages classical Gabor filtering with randomized parameterization. While these yield practical gains, the conceptual advance is incremental relative to recent pansharpening models introducing stronger spatial–spectral mechanisms, such as transformer-based fusion [1], content-adaptive non-local convolution [2], adaptive kernel-shape learning [3], and frequency-domain mixture-of-experts [4].                    

Generalization and robustness: No cross-sensor generalization (train on one sensor, test on another). In full-resolution evaluation, per-band spectral fidelity (e.g., band-wise SAM) and spectral index errors (e.g., NDVI) are not reported. There is no sensitivity analysis for GSFA parameter ranges nor an exploration of learnable Gabor parameters, despite recent emphasis on adaptivity in spatial kernels and frequency components [2][3][4].

[1] Zhou, H., Liu, Q., & Wang, Y. (2022). PanFormer: A transformer based model for pan-sharpening. 2022 IEEE International Conference on Multimedia and Expo (ICME). arXiv:2203.02916

[2] Duan, Y., Wu, X., Deng, H., & Deng, L.-J. (2024). Content-adaptive non-local convolution for remote sensing pansharpening. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27738–27747.

[3] Wang, X., Zheng, Z., Shao, J., Duan, Y., & Deng, L.-J. (2025). Adaptive rectangular convolution for remote sensing pansharpening. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17872–17881.

[4] He, X., Yan, K., Li, R., Xie, C., Zhang, J., & Zhou, M. (2024). Frequency-adaptive pan-sharpening with mixture of experts. Proceedings of the AAAI Conference on Artificial Intelligence, 38(3), 2121–2129. https://doi.org/10.1609/aaai.v38i3.27984

### Questions
Robustness and generalization:
- Add cross-sensor generalization (train on GF-2, test on WV-2/QB) and cross-resolution tests. These would strengthen claims that S2BNet is not sensor-specific.
- Report multiple seeds (mean±std) and significance tests for main results and ablations to contextualize ~0.3–0.5 dB gains.

Spectral fidelity and full-resolution analysis
- Provide per-band SAM and errors on spectral indices (e.g., NDVI) to directly assess spectral preservation.
- For full-resolution, include qualitative failure cases (e.g., thin structures, haze, strong PAN–MS mismatch).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces S2BNet, a binary neural network tailored for panchromatic–multispectral pansharpening. The core innovation is the S2B-Conv that contains two novel sub-modules:
Spectral-Redistribution Mechanism – learnable affine re-scaling of each spectral band to combat spectral distortion caused by heterogeneous PAN/LR-MS distributions;
Gabor Spatial Feature Amplifier – randomly-selected Gabor kernels that enrich binary filters with multi-scale / anisotropic cues before binarization.
Stacking S2B-Conv units into a light U-Net yields S2BNet, which is trained end-to-end with pure L1 loss. Extensive experiments on WorldView-2, GaoFen-2 and QuickBird show S2BNet outperforming every published BNN (+2.5 dB PSNR gap) and many full-precision models, while keeping ≈ 1/32 parameters and ≈ 1/60 FLOPs of the latter.

### Strengths
1. First BNN for PAN-MS fusion; SRM & GSFA are novel in the binary context.
2. multi-sensor, multi-metric evaluation; ablation of each submodule
3. Overall pipeline easy to follow; equations complete. The presentation is good.

### Weaknesses
1.This paper is not the first to apply binarized neural networks to pansharpening.
The paper titled "Binarized Neural Network for Multi-spectral Image Fusion" holds that distinction.
2.The authors claim that their method surpasses other BNNs by more than 2 dB in PSNR; however, Table 1 shows that it is only about 0.2 dB higher than BiSRNet.
3.Likewise, Table 1 shows that the proposed method has three times the parameters of BiSRNet and significantly higher FLOPs. Is the reported performance gain entirely due to this larger model capacity rather than to the novel SRM/GSFA components?
4.From Figure 2 it is completely impossible to tell which method produces higher visual-quality fused images, and the method names in the figure are somewhat blurry.
5.The motivation behind the proposed core modules—the Spectral-Redistribution Mechanism and the Gabor Spatial Feature Amplifier—is unclear. There is only some textual explanation, but no other intuitive or quantifiable evidence is provided.

### Questions
see the Weaknesses

### Soundness
2

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
5

### Summary
This paper proposes a Spatial-Spectral Binarized Neural Network (S2BNet) for efficient panchromatic and multispectral image fusion (pan-sharpening). The method introduces a custom Spatial-Spectral Binarized Convolution (S2B-Conv) layer, consisting of two modules:
(1) Spectral-Redistribution Mechanism (SRM) — dynamically adjusts spectral distribution through data-driven affine scaling and bias;
(2) Gabor Spatial Feature Amplifier (GSFA) — captures multi-scale, multi-directional spatial patterns using Gabor kernels.
S2BNet is designed to achieve comparable reconstruction quality to full-precision networks while being lightweight enough for resource-limited satellite platforms. Experiments on GaoFen-2, WorldView-2, and QuickBird datasets are used to demonstrate that S2BNet outperforms other binary networks and approaches full-precision models in PSNR, SSIM, and QNR metrics

### Strengths
Motivation relevance: The work targets a meaningful problem—deploying deep pan-sharpening models on resource-limited devices such as satellites—where efficiency is crucial.

Architecture clarity: The proposed S2B-Conv structure (with SRM and GSFA) is conceptually simple and well integrated into a binarized U-Net-like framework.

Empirical completeness: The paper provides quantitative and qualitative experiments, including ablation on both SRM and GSFA and comparisons with state-of-the-art (SOTA) binary and full-precision models.

Energy efficiency: The use of bitwise operations (XNOR, bit-count) is suitable for low-power environments, aligning with practical needs in embedded EO applications.

### Weaknesses
Limited originality / incremental contribution.
The technical innovation is minimal. Both SRM and GSFA are straightforward combinations of existing ideas: adaptive scaling (used in SE, FiLM) and Gabor filters (widely applied in image enhancement and texture analysis). Their integration into BNNs does not represent a fundamental advance in pan-sharpening or network design. The approach is essentially a routine adaptation of BiSRNet (Cai et al., 2023) with Gabor initialization.

Lack of theoretical or methodological depth.
There is no formal justification for why SRM or GSFA mitigates spectral distortion or anisotropy introduced by binarization. The discussion is qualitative and lacks rigorous analysis of spectral error propagation or frequency response behavior.

Overstated performance claims.
While numerical improvements over other BNNs (~0.3–0.5 dB PSNR) are reported, these are minor and may not be statistically significant. Results still lag behind top full-precision models (e.g., FAMENet, CANNet). The claim that S2BNet “outperforms most full-precision baselines” is exaggerated.

Insufficient novelty compared with prior lightweight or quantized models.
The paper overlooks related quantization-aware pan-sharpening or efficient Transformer works (e.g., HyperTransformer, LitePNN). The positioning of S2BNet as a novel direction in efficient fusion is weak.

Poor figure readability.
Figures (e.g., Fig. 1, 2, 3) are too small, and the text is illegible. Important architectural components and qualitative comparisons are difficult to interpret, limiting reproducibility.

Limited generalization and scalability.
All experiments are conducted on moderate-resolution datasets; there is no evidence that S2BNet scales to very large images or complex multimodal datasets. The efficiency discussion also lacks actual inference latency or hardware deployment results.

### Questions
How does S2B-Conv differ mathematically from existing spectral recalibration (SE, FiLM) and orientation-sensitive convolutions (GaborNet, DCTNet)?

Can the authors provide quantitative analyses on spectral distortion reduction introduced by SRM, beyond PSNR/SSIM metrics?

What is the real computational saving in milliseconds or watts on embedded hardware compared to full-precision models?

How robust is S2BNet under cross-sensor transfer (e.g., training on GF-2, testing on WV-2)?

Figures should be enlarged and restructured for readability—particularly the architecture and ablation visualizations.

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
5

### Summary
This paper introduces S2BNet, a Spatial–Spectral Binarized Neural Network tailored for remote sensing pan-sharpening. The network incorporates two novel components: a Spectral Redistribution Mechanism (SRM) and a Gabor Spatial Feature Amplifier (GSFA). Evaluated on four-band datasets—GaoFen-2, WorldView-2, and QuickBird—S2BNet achieves performance that is competitive with or superior to both full-precision and other binarized models, while maintaining significantly higher computational efficiency.

### Strengths
1.	Substantially reduces model size and FLOPs, making it suitable for deployment on resource-limited satellite or embedded platforms without significant performance degradation.
2.	Enhances the recovery of both spectral and spatial information.

### Weaknesses
* Limited parameter analysis is provided.
* Generalization to datasets with more spectral bands remains unexplored.
* No runtime comparison is included. While “efficiency” is emphasized, inference time benchmarks on actual hardware are missing.
* Citation formatting should be corrected.
* The related work section should more systematically review prior binarized methods.
* Metrics and methods are tested with inconsistent presentation across tables. For example, the QNR is not comprehensive for full-resolution cases.
* The reference for metrics is missing. 
* The simulated process is missing.

### Questions
* How does the adaptive SRM perform on datasets with greater spectral diversity?
* How does S2BNet’s binarization affect color fidelity and edge preservation in visually demanding applications? In ablation studies where numerical differences are subtle, visual comparisons would be more informative.
* Why is binarization not applied in conjunction with linear layers? What trade-offs exist between binarization and module design? Could partial binarization offer a better balance?
* Have the SRM and GSFA modules been tested in other network architectures? Without such validation, the effectiveness of these modules lacks statistical support and may be difficult to substantiate.
In summary, the effectiveness of this method remains unclear.

### Soundness
2

### Presentation
2

### Contribution
3
