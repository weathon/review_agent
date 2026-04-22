# WavePolyp: Video Polyp Segmentation via Hierarchical Wavelet-Based Feature Aggregation and Inter-Frame Divergence Perception

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Automatic polyp segmentation from colonoscopy videos is a crucial technique that assists clinicians in improving the accuracy and efficiency of diagnosis, preventing polyps from developing into cancer.
However, video polyp segmentation (VPS) is a challenging task due to (1) the significant inter-frame divergence in videos, (2) the high camouflage of polyps in normal colon structures and (3) the clinical requirement of real-time performance.
In this paper, we propose a novel segmentation network, WavePolyp, which consists of two innovative components: a hierarchical wavelet-based feature aggregation (HWFA) module and inter-frame divergence perception (IDP) blocks.
Specifically, HWFA excavates and amplifies discriminative information from high-frequency and low-frequency features decomposed by wavelet transform, hierarchically aggregating them into refined spatial representations within each frame.
This module enhances the representation capability of intra-frame spatial features, effectively addressing the high camouflage of polyps in normal colon structures.
Furthermore, IDP perceives and captures inter-frame polyp divergence through a temporal divergence perception mechanism, enabling accurate polyp tracking while mitigating temporal inconsistencies caused by the significant inter-frame variations across frames.
Extensive experiments conducted on the SUN-SEG and CVC-612 datasets demonstrate that our method outperforms other state-of-the-art methods.
Codes are available at \url{https://github.com/FishballZhang/WavePolyp.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes WavePolyp, a novel framework for video polyp segmentation (VPS) that addresses two key challenges: intra-frame feature discrimination in highly camouflaged polyps and inter-frame temporal inconsistency. The method introduces two core components: a Hierarchical Wavelet-based Feature Aggregation (HWFA) module that decomposes and enhances multi-level features in the frequency domain, and Inter-frame Divergence Perception (IDP) blocks that explicitly model temporal changes between frames. Extensive experiments on SUN-SEG and CVC-612 benchmarks show that WavePolyp outperforms existing state-of-the-art methods while maintaining near real-time inference speed.

### Strengths
1. Comprehensive Experiments and Ablations: The paper provides thorough ablation studies to validate the contribution of each component (HFC, LFC, AFA, IDA, IDD). Results show consistent improvements across multiple metrics and datasets, with particularly strong gains on the challenging SUN-SEG-Hard subset.
2. Efficiency-Aware Design: The model achieves a good balance between accuracy and inference speed (23.04 FPS), making it suitable for real-time clinical applications.

### Weaknesses
1. Limited Conceptual Justification for Frequency-Based Design:
While the HWFA module is motivated by the idea that polyps exhibit discriminative features in both high- and low-frequency domains, the paper does not provide strong empirical or theoretical evidence for this claim. For example, visualizations of frequency-domain feature maps or analyses of polyp characteristics in the frequency space are missing.
2. Limited Conceptual Justification for Frequency-Based Design:
While the HWFA module is motivated by the idea that polyps exhibit discriminative features in both high- and low-frequency domains, the paper does not provide strong empirical or theoretical evidence for this claim. For example, visualizations of frequency-domain feature maps or analyses of polyp characteristics in the frequency space are missing.
3. Narrow Evaluation Scope: All experiments are limited to polyp segmentation. While the method is claimed to be general, no experiments on other video object segmentation benchmarks (e.g., DAVIS, YouTube-VOS) are provided to demonstrate broader applicability. The comparison with non-medical VPS methods is limited, making it difficult to assess the generalizability of the proposed components.
4. Lack of Novelty in Wavelet-Based Design and Inadequate Related Work: 
The related work in Section 2.2 fails to establish a clear and substantive distinction between the proposed HWFA module and prior wavelet-based methods in vision tasks (e.g., SDWNet, LAAT, FEDER). The claim of novelty appears overstated. The application of DWT for feature decomposition, followed by attention mechanisms and hierarchical aggregation, is a known paradigm. The paper does not convincingly answer: What is the fundamental architectural or methodological leap here? The HWFA module, while well-engineered, comes across as a straightforward application of existing frequency-domain concepts to a new dataset, rather than a conceptual breakthrough.
5. The "Divergence Perception" in IDP is Unfocused and Potentially Noisy. The core mechanism of the IDP block is to compute a frame difference Diff = Shift(V) - V and modulate it via attention. However, this is a generic, low-level motion signal that captures all changes between frames, not just those pertaining to the polyp. This design is highly susceptible to noise from irrelevant background motion, such as camera jitter, fluid flow, or intestinal wall movements. The paper provides no evidence that the IDP block selectively focuses on polyp-specific divergence. Without an explicit mechanism to ground the divergence signal to the polyp region (e.g., using the current prediction as a guide), it is likely that the model is learning from a noisy signal, which could harm robustness in complex scenes. The failure cases in Fig. 7 (e-f) with overlapping intestinal walls might be a direct consequence of this issue.
6. The Overall Architecture Lacks Conceptual Novelty. The combination of DWT for multi-scale frequency analysis and attention mechanisms for feature refinement is a well-established design pattern in contemporary literature. The paper does not demonstrate that the proposed HWFA or IDP blocks represent a significant departure from this pattern. The HWFA module can be largely viewed as a specific instantiation of a multi-scale feature refinement network that uses DWT as its decomposition tool, rather than a fundamentally new operator. The incremental nature of the architectural design is a significant drawback for a paper aiming for a top-tier conference.
7.  Performance Gains Are Marginal and Not Compelling. A close inspection of Table 1 and Table 3 reveals that the performance advantage of WavePolyp is not decisive. On SUN-SEG-Hard, WavePolyp (Dice: 87.55) outperforms ZoomNeXt (Dice: 85.22) by 2.33 points, which is a solid but not groundbreaking improvement. More critically, on SUN-SEG-Easy, the gap is much smaller (WavePolyp: 88.96 vs. ZoomNeXt: 87.55, a 1.41-point difference).  When considering efficiency in Table 3, WavePolyp (23.04 FPS, 86.63M Params) and ZoomNeXt (22.48 FPS, 84.78M Params) are nearly identical in speed and model size, yet the performance delta is minimal. This suggests that the proposed complex modules (HWFA + IDP) offer only a marginal benefit over a much simpler and more general baseline, which severely undermines the claim of a significant advancement.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes WavePolyp, a method for video polyp segmentation (VPS) in colonoscopy videos. The model targets two key challenges: (i) high camouflage of polyps within surrounding tissue and (ii) significant inter-frame divergence (shape, size, position, boundary changes) that causes temporal inconsistency. WavePolyp contains two main components:

•	Hierarchical Wavelet-based Feature Aggregation (HWFA): Applies discrete wavelet transform (DWT) to multi-level encoder features, splits into low-frequency (LF) and high-frequency (HF) components, refines them via bespoke LF/HF calculation units (LFC/HFC) with attention and normalization strategies, and aggregates them across scales by an Ascending Frequency-guided Aggregation (AFA) unit. The goal is to excavate fine-grained intra-frame discriminative cues (edges, textures in HF; illumination/color in LF).

•	Inter-frame Divergence Perception (IDP): A temporal module that computes frame-wise differences via a temporal shift (TSM-like) operation, modulates them with a learnable projection, and applies a time-only attention to emphasize regions with meaningful temporal changes. A two-layer temporal convolution diffuses the divergence signal, and features are fused in a coarse-to-fine decoder.
Experiments on SUN-SEG (Easy/Hard splits) and CVC-612 show improvements over a set of baselines, including SLT-Net, ZoomNeXt, SALI, VP-SAM, and others, with competitive FPS on RTX 3090. Ablations attribute gains to both HWFA and IDP, and to specific internal design choices (normalization in LFC, AFA, divergent attention/diffusion). The paper reports favorable Dice, S-measure, E-measure, and weighted F across datasets, and discusses failure cases.

### Strengths
Originality:
•	Using DWT to explicitly separate HF/LF feature bands and tailor distinct processing (HFC/LFC) in a VPS context is a thoughtful design. While wavelets have been used in other CV tasks, positioning them to address polyp camouflage with a hierarchical aggregation (AFA) is a creative application.

•	The IDP block frames temporal modeling as learning inter-frame divergence via a simple yet principled TSM-based difference, time-only attention, and diffusion. It departs from heavier 3D/transformer-temporal designs by focusing on divergence emphasis with relatively low complexity.

•	The multi-scale zooming scheme via ZoomNeXt-style feature merging complements the frequency-domain design, potentially improving robustness to size variation.

Quality:
•	Comprehensive experimental protocol on standard VPS datasets (SUN-SEG with Easy/Hard splits; CVC-612), multiple metrics, and a reasonably broad set of recent baselines, including both image and video methods.

•	Ablation studies that dissect both macro-components (HWFA vs. IDP) and micro-components (HFC/LFC normalization, AFA, IDP’s attention/diffusion), plus exploration of clip length.

•	Efficiency analysis (params, GFLOPs, FPS) against video methods provides a useful view of the accuracy-efficiency trade-off.

Clarity:
•	The high-level motivation and module roles (camouflage handled by HWFA; temporal divergence by IDP) are clearly articulated and connected to the challenges illustrated in Fig. 1.

•	The pipeline overview (encoder with zoomed scales, HWFA, IDP decoder, final predictor) is easy to follow.

•	Equations and data flow for IDP are described sufficiently to reproduce the idea. The HFC/LFC and AFA figures are helpful.

Significance:
•	VPS is a clinically meaningful and actively researched area; improvements that bring both accuracy and near-real-time speed are valuable.

•	The approach offers a perspective shift: deliberately leveraging frequency-domain cues for medical temporal segmentation, which could transfer to other endoscopic or camouflaged lesion videos.

### Weaknesses
Novelty vs. prior art:
•	Wavelet/decomposition-driven feature processing has been employed in medical segmentation (e.g., FEDER and related frequency-decomposition works) and image generation/denoising; the paper cites some but does not thoroughly differentiate its technical novelty beyond the specific arrangement (HFC/LFC/AFA) and application to VPS. More rigorous positioning against frequency-aware segmentation works (e.g., learnable wavelet bases, frequency-aware attention, Fourier-based filtering such as FFC/AFNO/FFT-based U-Nets) is needed to substantiate novelty.

•	The IDP block conceptually resembles several recent “temporal difference + attention” modules: TSM/TDN-like frame differencing, TEA-style excitation of motion, and time-attention in transformer decoders. The paper cites TSM but not closely related temporal-difference attention models. A tighter comparative discussion is necessary to establish conceptual distinctiveness.


Methodological specification:
•	The AFA unit is central, but the mathematical description is incomplete. The appendix starts an optimization objective for LF AFA but cuts off; the main text mentions a “window-based linear model” inspired by Swin Transformer, yet does not concretely define the window filtering/weighting, parameterization, loss, or computations. Reproducibility of AFA as described is difficult.

•	The LFC normalization stack mixes PN/IPN with BN in Eq. (2), although the text claims BN in HFC and IN in LFC; there is an inconsistency: the formula shows BN in LFC after PN, contrary to the stated IN replacement. If IN is truly used, the equation should reflect it. Moreover, the learnable µ′, σ′ details (how they are produced per position, per channel? with which conv kernel sizes? affine parameters?) are underspecified.

•	For IDP, the shapes and exact axes for attention are a bit ambiguous: K, Q in R^{HW × T} suggests flattening spatial into tokens and attending over time only, but the normalization over sqrt(HW) is unusual (typically sqrt(d)). If d=HW, this is extremely large and may cause scale issues; justification or reparameterization (e.g., linear projections to lower d) is not provided. Complexity claims should be reconciled with potentially large HW × T multiplications.

•	The learnable WV is a T × T matrix. If T is variable at test time, how is WV handled? If fixed T=5 always, this should be stated as a constraint. Also, the two T × 3 × 3 convolutions imply grouping or depthwise temporal convolution per frame with spatial 3x3 kernels, but the notation is nonstandard; more detail would help.

Experimental rigor:
•	The baseline list is decent, but misses some recent strong VPS/medical video temporal models (e.g., DINOv2-temporal adaptations, token mixers with long-range memory, or optical-flow-based medical segmentation), and does not compare to simple but strong baselines like “per-frame SOTA IPS + temporal smoothing/post-processing.” Including a naive temporal regularization baseline could calibrate the added value of IDP.

•	Statistical significance or variability is not reported (no std/CI across runs or seeds). Given close margins over VP-SAM and SALI on some metrics, significance analysis is warranted.

•	The clip length ablation is limited to a single dataset and reports only a curve; numeric values and an analysis of inference-time latency/memory vs. T would be useful for practitioners.

•	The speed benchmark uses batch size 1 on RTX 3090; ensuring comparable precision settings (FP32 vs. AMP), input size, and consistent pre- and post-processing is important. Also, the model depends on PVTv2-B5 and ZoomNeXt merging; an ablation on the backbone and the multi-scale merging would contextualize where the gains originate.


Generalization and data:
•	Only two datasets (one large, one small) are used; cross-dataset generalization (train on SUN-SEG, test on CVC-612 without fine-tuning) would be informative.

•	Robustness to common endoscopic artifacts (specular highlights, motion blur, occlusions), lighting changes, and strong peristalsis is only qualitatively discussed; quantitative robustness tests (e.g., controlled corruption benchmarks) are missing.

•	The method benefits from training on three zoom scales; at inference, is the multi-scale encoder always used (computational overhead)? The text implies that multi-scale features are merged at 1.0x; an explicit statement of the inference-time pipeline and cost would help.


Clarity/consistency issues:
•	Some figures/equations appear truncated or inconsistent (e.g., AFA math in appendix). Equation (8) references DWTh/DWTl applied to fk; it’s not entirely clear whether DWT is applied at each level to encoder features vs. features already processed by prior steps. A tidy pseudo-code or module-by-module dimensionality flow would resolve ambiguities.

•	The reliance on ZoomNeXt’s multi-scale merging is nontrivial; a short recap or an ablation “w/o multi-scale merging” is needed to attribute gains correctly to HWFA/IDP vs. upstream feature enrichment.

### Questions
1.	AFA specification and reproducibility:
•	Please provide the complete mathematical definition of AFA, including the “window-based linear model” parameters, the guidance weighting using W^HF/LF, and how the downsampled aggregated feature f_k^dh is computed. Is it a learned linear filter per window guided by W? Any regularization?

•	In Appendix A.1, the optimization objective for LF AFA is truncated. Can you provide the full objective and solution or implementation details?

2.	Normalization details in LFC:
•	The text says BN in HFC and IN in LFC, but Eq. (2) shows BN in LFC. Which is correct? If IN is used, please provide the exact formula and where PN/IPN sit relative to IN. How are µ′, σ′ computed (per-position, per-channel), and with what kernel sizes/parameterization?

3.	IDP attention scaling and complexity:
•	You use Softmax(K^T Q / sqrt(HW)). Why is sqrt(HW) the appropriate scale? Do you use linear projections to reduce dimensionality before attention? If not, what are the memory/time costs for HW × T tokens, especially with 352×352 inputs? Please include exact tensor shapes and a complexity analysis.

•	WV is T × T and appears to hardcode the clip length. How does the module adapt to different T at test time, or is T fixed? Would a content-dependent dynamic filtering (e.g., 1D temporal conv) be preferable?

4.	Baseline completeness and significance:
•	Could you include comparisons with: (i) per-frame SOTA polyp segmenter plus temporal smoothing (e.g., Savitzky–Golay, EMA), (ii) an optical-flow-guided refinement baseline, and (iii) a temporal transformer with time-only attention but without divergence computation? This would isolate the incremental value of your divergence design.

•	Please report mean±std across at least 3 runs for key metrics to ascertain the significance of improvements, especially over VP-SAM and SALI.

5.	Multi-scale pipeline at inference:
•	Do you run the tri-scale encoder (0.75×/1.0×/1.25×) at inference? If yes, what is the added latency and memory compared to single-scale? If not, how is multi-scale training transferred to single-scale inference?

6.	Wavelet choices and learned variants:
•	Which wavelet family (e.g., Haar, Daubechies) is used? Have you explored learnable wavelet filters or alternative frequency decompositions (e.g., DCT/Fourier), and how do results compare?

•	How sensitive is performance to DWT level, number of scales, and the balance between HF/LF weighting?

7.	Generalization and robustness:
•	Please include cross-dataset generalization results (e.g., train on SUN-SEG, test on CVC-612 without fine-tuning) and robustness to common perturbations (blur, noise, brightness shifts), to validate claims of robustness to camouflage and inter-frame divergence.

8.	Clinical practicality:
•	VP-SAM requires a point prompt; your method does not. However, your reliance on multi-scale processing and wavelet decomposition adds overhead. Can you provide a detailed breakdown of latency per module and avenues for lightweight deployment (e.g., replacing PVTv2-B5, pruning, quantization), including any preliminary results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a video polyp segmentation framework composed of two main modules: Hierarchical Wavelet-based Feature Aggregation (HWFA) and Inter-frame Divergence Perception (IDP) blocks. In the encoder, the HWFA module amplifies information from both high- and low-frequency features to address the camouflage property of polyps. The IDP module in the decoder captures inter-frame divergence to enhance temporal consistency.
Main contributions:
- The framework considers both intra-frame discriminative features and inter-frame divergence for improved segmentation performance.
- It accounts for both segmentation accuracy and real-time efficiency by evaluating GFLOPs and FPS.
- An extensive ablation study is conducted to validate the effectiveness of each component.

### Strengths
- Extensive experiments are conducted across multiple metrics, including GFLOPs and FPS, along with visualizations such as t-SNE plots for clearer interpretability.
- A thorough ablation study is performed, analyzing the impact of the presence or absence of different modules.
- Results are comprehensively compared against recent SoTA methods.

### Weaknesses
- The author mentions that they compared their results on two video polyp datasets, one of which is CVC-612. However, it is a static dataset. This dataset contains individual frames extracted from colonoscopy videos, which are static images and do not include temporal relationships. Therefore, validating a video model on such a static dataset may not effectively evaluate the video-specific contributions of the proposed model.
- The train-test split for CVC-612 is not described and should be specified.
- Since the IDP module is designed to capture inter-frame information, some frame-wise qualitative results are needed to validate its effectiveness.
- In Fig. 3, the CA+SA module is shown, but it is not discussed in detail in the text.
- In Table 2, it appears that when all modules except AFA are used, performance drops even below the baseline (top row). This behavior requires justification.

Minor:
In Section 3.1, the sentence “Due to the dynamic nature of video” should be rephrased, as it seems incomplete or unclear.

### Questions
- It should be explained how evaluating the model on CVC-612, a static dataset without temporal relationships, effectively reflects the video-specific contributions of the proposed framework.
- The train-test split for CVC-612 should be provided to clarify how the dataset was used for training and evaluation.
- Frame-wise qualitative results or visualizations should be provided to demonstrate the effectiveness of the IDP module in capturing inter-frame divergence.
- Description for the CA+SA module should be provided, as it is shown in Fig. 3.
- Justification should be provided for why performance drops below the baseline in Table 2 when all modules except AFA are used.

### Soundness
2

### Presentation
2

### Contribution
2
