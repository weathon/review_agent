# ExGS: Extreme 3D Gaussian Compression with Diffusion Priors

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios.
In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality.
We introduce \textbf{ExGS}, a novel feed-forward framework that unifies \textbf{Universal Gaussian Compression} (UGC) with \textbf{GaussPainter} for \textbf{Ex}treme 3D\textbf{GS} compression. \textbf{UGC} performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas \textbf{GaussPainter} leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes.
Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration.
Our framework can even achieve over $100\times$ compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a framework for compressing 3DGS using a voxel-based pruning strategy combined with a diffusion model to guide reconstruction. Empirical results demonstrate that ExGS achieves high compression ratios with minimal loss of rendering quality, outperforming existing baselines.

### Strengths
1. This paper introduces an effective framework that combines voxel-based pruning with diffusion-based priors for 3DGS compression.
2. It demonstrates superior performance over existing baselines on both indoor and outdoor scenes.

### Weaknesses
1. The Global Significance Score in Eq. 7 requires computing the intersection between camera rays and Gaussian primitives. Does this mean that the input images or at least their camera parameters are needed for 3DGS compression? If so, this limit the method’s applicability. Additionally, the computation seems to involve all pixels and all primitives, raising concerns about efficiency. The exact formulation of the intersection between a Gaussian distribution and a ray is also unclear.

2. The use of high-order omission and FP16 precision appears to significantly aid compression. An ablation study is needed to quantify the contribution of these techniques relative to the GS score and voxel-based pruning.

3. The paper claims improved efficiency, so a test-time experiment evaluating the compression process itself (not just the diffusion-based reconstruction) is necessary to substantiate this claim.

4. Quantitative comparisons with the optimization-based compression methods discussed in the introduction should also be provided.

### Questions
The neural predictor in Eq. 10 is not clearly defined. Does it require additional training, and if so, how is it trained?

### Soundness
3

### Presentation
2

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
ExGS introduces a feed-forward 3D Gaussian Splatting compression pipeline that combines a training-free Universal Gaussian Compression (UGC) stage with a diffusion-prior restorer, GaussPainter. UGC prunes and compacts Gaussian primitives using global significance scoring and voxel-aware selection (with lightweight amplification and SH simplification). GaussPainter then performs mask-guided, one-step diffusion in a VAE latent space—using latent supervision—to both complete missing content and refine preserved regions.

Main contributions：
UGC (training-free) compression: Global significance scoring + voxel-aware selection with adaptive amplification; simplifies appearance (e.g., SH) and packs parameters for compact storage.
GaussPainter (efficient generative restoration): Mask guidance from 3DGS opacity and latent supervision enable one-step diffusion that jointly inpaints and enhances results.
Validated pipeline design: Ablations show each component’s effect and the combined benefits of UGC + GaussPainter across diverse scenes.

### Strengths
### Originality
* Frames 3DGS compression as a hybrid “compress + generative restore” paradigm; the UGC (training-free) + GaussPainter (diffusion prior) pairing is novel.
* Uses opacity-guided one-step diffusion with latent-space supervision to curb hallucinations while keeping decoding efficient.
* Voxel-aware significance scoring with adaptive amplification preserves structure under aggressive pruning.
### Quality
* Clear modular pipeline; ablations disentangle contributions of UGC, mask guidance, and latent supervision.
### Clarity
* Straightforward narrative with effective figures; two-stage roles and I/O are easy to follow.
* Clearly separates geometry/opacity from appearance, defining where the generative prior should and shouldn’t act.
### Significance
* Direct practical value for 3DGS storage/streaming: smaller footprint with fast decode.
* Provides a general recipe—“compact representation + learned restoration”—that can transfer to other scene formats (including 4D).
* Offers a more controllable quality–compression trade-off, maintaining usable quality even at extreme compression.

### Weaknesses
Missing information. Please clarify the training data sources and splits for UGC and GaussPainter (did each dataset participate in training? train/val/test splits? combined training?), and whether TAESD / the diffusion backbone were pretrained on generic image corpora and then fine-tuned for this task, including the fine-tuning ratio and freezing strategy.

Generalization validations needed:
Out-of-Domain (OOD) evaluation: Choose sampling protocols/scenes absent from training (e.g., Tanks&Temples, LLFF, KITTI-360, or self-captured data). Report PSNR/SSIM/LPIPS and show failure cases.

One-line summary: As of now, the paper has not demonstrated generalization under train–test isolation across datasets; we recommend adding LODO/OOD and distribution-shift evaluations to upgrade empirical stability into reproducible evidence of generalization.

### Questions
Question:
Can you add Leave-One-Dataset-Out (LODO) results—train on A+B and evaluate zero-shot on C—and Out-of-Domain (OOD) results on datasets not seen during training (e.g., Tanks&Temples, LLFF, KITTI-360, or self-captured scenes)?

Why it matters (Importance):
This directly tests cross-domain robustness and guards against overfitting to the evaluation domains.

Expected evidence:
PSNR/SSIM/LPIPS curves across compression levels or pruning strengths.
A gallery of failure cases with brief diagnostics.
Qualitative comparisons under extreme sampling/texture/geometry shifts to illustrate robustness (or lack thereof).

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
This paper introduces ExGS, a novel framework for extreme compression of 3D Gaussian Splatting by combining a pruning-based compression module (UGC) with a diffusion-based refinement module (GaussPainter). The method achieves compression ratios exceeding 100× while maintaining high rendering quality. Unlike prior optimization-based or training-free compression methods, ExGS leverages generative diffusion priors to restore and enhance heavily pruned scenes, enabling robust performance across indoor and outdoor benchmarks.

### Strengths
1. Novel Integration of Compression and Generative Modeling: The combination of UGC for aggressive pruning and GaussPainter for diffusion-based restoration is innovative and effectively bridges the gap between extreme compression and high-quality rendering.

2. Impressive Compression Ratios: The method achieves impressive compression levels while preserving visual fidelity.

3. Comprehensive Evaluation: The paper provides thorough experiments across multiple datasets, compression ratios, and metrics, including ablation studies that validate the contribution of each component.

### Weaknesses
1. While the diffusion-based refinement module significantly improves rendering quality, this process substantially increases rendering complexity compared to the original Gaussian Splatting overhead, limiting its applicability in real-world scenarios.

2. The experimental data still needs to be supplemented. For example, the main experimental results and ablation studies (especially Table 4(a)) should ideally include rate-distortion curves. Additionally, as a feed-forward method, the paper lacks discussion on encoding speed and decoding speed.

### Questions
1. How does the UGC module perform compared to other models, particularly feed-forward models (such as FCGS)? A related question is: what would be the effect of applying the GaussPainter to other compression models?

2. What datasets were used during training?

3. The results of LightGS on Mip-NeRF360 are significantly lower than those reported in the original paper. What differences exist in the experimental setup?

4. How large is the neural predictor in Section Adaptive Amplification?

If these issues are addressed, will consider increasing the score.

### Soundness
3

### Presentation
2

### Contribution
2
