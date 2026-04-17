# FreeAdapt: Unleashing Diffusion Priors for Ultra-High-Definition Image Restoration

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Latent Diffusion Models (LDMs) have recently shown great potential for image restoration owing to their powerful generative priors. However, directly applying them to ultra-high-definition image restoration (UHD-IR) often results in severe global inconsistencies and loss of fine-grained details, primarily caused by patch-based inference and the information bottleneck of the VAE. To overcome these issues, we present FreeAdapt, a plug-and-play framework that unleashes the capability of diffusion priors for UHD-IR. The core of FreeAdapt is a training-free Frequency Feature Synergistic Guidance (FFSG) mechanism, which introduces guidance at each denoising step during inference time. It consists of two modules: 1) Frequency Guidance (FreqG) selectively fuses phase information from a reference image in the frequency domain to enforce structural consistency across the entire image; 2) Feature Guidance (FeatG) injects global contextual information into the self-attention layers of the U-Net, effectively suppressing unrealistic textures in smooth regions and preserving local detail fidelity. In addition, FreeAdapt includes an optional VAE fine-tuning module, where skip connection further enhances the reconstruction of fine-grained textures. Extensive experiments demonstrate that our method achieves superior quantitative performance and visual quality compared to state-of-the-art UHD-IR approaches, and consistently delivers strong gains across multiple LDM-based backbones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes FreeAdapt, a plug-and-play framework designed to enhance ultra-high-definition image restoration (UHD-IR) using diffusion priors. To address the global inconsistency and detail loss issues of existing latent diffusion models (LDMs) caused by patch-based inference and VAE bottlenecks, the paper introduce a training-free Frequency Feature Synergistic Guidance  mechanism that provides guidance at each denoising step. FFSG includes two modules: Frequency Guidance, which fuses phase information from a reference image in the frequency domain to ensure structural consistency, and Feature Guidance (FeatG), which injects global contextual information into U-Net self-attention layers to suppress artifacts and preserve fine details. Additionally, an optional VAE fine-tuning module with skip connections is proposed to further improve texture reconstruction. Experiments show that FreeAdapt achieves superior quantitative and qualitative performance compared to existing UHD-IR and diffusion-based adaptation methods.

### Strengths
1. It achieves promising results in terms of perceptual quality metrics.  

2. Exploring the application of diffusion models in UHD image restoration is an interesting direction.

### Weaknesses
1. The second claimed contribution — the frequency and feature guidance modules — appears highly similar to the approach proposed in [1]. Please clearly describe the differences between your method and [1]. If there are no substantial distinctions, these components should not be presented as core contributions.

2. How does the proposed method compare in terms of computational efficiency with existing UHD-IR approaches? Although I understand this may be a limitation of the current method, the paper does not provide any relevant data. Please report a quantitative comparison including GPU memory usage, inference speed, number of parameters, and FLOPs.

3. Applying diffusion models to UHD image restoration incurs substantial computational costs, making deployment on edge devices challenging. Compared with existing methods such as UHDformer and ERR, the practical applicability and significance of the proposed approach remain unclear. A discussion of its real-world deployment potential, efficiency trade-offs, and specific use cases is necessary.

4. The paper lacks comparative results on UHD image deraining.

[1] FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion. CVPR 2025.

### Questions
If the authors can address the aforementioned issues, I would consider raising my evaluation score.

### Soundness
2

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
3

### Summary
This paper introduces a training-free method FreeAdapt to utilize pre-trained diffusion priors for ultra-high-definition image restoration. The core of the framework is a training-free frequency-feature synergistic guidance mechanism, which enforces global structural consistency through frequency guidance and suppresses artifacts while preserving local detail fidelity via feature guidance. By fine-tuning VAE decoder, it further enhances the reconstruction of texture and detail. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
● The proposed frequency and feature guidance strategies effectively resolves the structure inconsistencies and artifacts in patch-based inference results.
● By fine-tuning VAE decoder, it further enhances the reconstructed texture and mitigates the lossy compression characteristic.
● Extensive experiments demonstrate the effectiveness of the proposed method on three synthetic benchmark.

### Weaknesses
● Training the VAE decoder conflicts with the authors' claim of ``training-free approach''. In fact, in Fig. 3, using only FreqG/FeatG yields only marginal performance gains. I highly doubt the validity of the restoration results, which stems from fine-tuning the VAE decoder network across three synthetic degradation scenarios.
● The part of ``reference image generation'' is confusing. Transforming a degraded image to a clean image require performing sdedit process or utizling a pre-trained I2I backbone. The authors need to provide more details about it.
● The testing scenarios are limited. Current benchmark are constructed by applying several synthetic degradation factors, which can be fitted by only fine-tuning the decoder network with the corresponding datasets. The authors are suggested to perform experiments on AI-generated or real-world benchmark.

### Questions
● More experiments need to be conducted to demonstrate the proposed ``training-free'' approaches without fine-tuning decoder network.
● The authors are recommended to provide image results in supplementary material.

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
The paper addresses Ultra-High-Definition Image Restoration (UHD-IR), i.e. restoring 4K/8K images degraded by noise, blur, haze, low-light etc. The authors argue that existing UHD-IR methods (e.g. UHDFormer, ERR, Wave-Mamba) rely on architectural tricks and still hit bottlenecks due to the ill-posed nature of restoration. To overcome this, the paper proposes to leverage pretrained latent diffusion models (LDMs) as powerful generative priors. Directly applying LDMs to UHD images is problematic (patch-based inference causes stitching artifacts, loss of global context and detail due to VAE bottleneck). The core contribution is FreeAdapt, a training-free, plug-and-play framework that adapts off-the-shelf LDMs (and extensions like ControlNet) to UHD-IR. During inference, FreeAdapt applies a novel Frequency-Feature Synergistic Guidance (FFSG) mechanism and an optional VAE fine-tuning (VAE-FT). Specifically, at each denoising step, Frequency Guidance (FreqG) replaces the phase spectrum of the current latent patch with that of a low-resolution reference image, enforcing cross-patch structural consistency, while Feature Guidance (FeatG) injects global contextual features (from the reference) into the U-Net's self-attention to suppress hallucinated textures. The reference is obtained by downsampling the input and running one LDM denoising pass, yielding coherent low-frequency structure. Finally, a lightweight fine-tuning of the VAE decoder (with skip connections) is optionally applied to alleviate compression losses and sharpen details.

### Strengths
1. The paper's approach appears novel. While diffusion priors have been used in low-level vision, adapting them to ultra-high-res restoration with plug-and-play guidance is new.
2. The method adopts a Plug-and-play design, which can be compatible with different diffusion models.
3. The paper is generally well-organized and clearly written.

### Weaknesses
1. The paper should further clarify the distinction between high-resolution image restoration and high-resolution image generation. To my knowledge, DemoFusion is a method for high-resolution generation rather than restoration. The key difference lies in that restoration tasks emphasize maintaining consistency between the low-resolution and high-resolution images, whereas training-free high-resolution generation methods do not impose such a constraint.
2. Computational cost: diffusion inference with dozens of steps and many patches is slow and resource-intensive. Indeed, patch-based diffusion on 8K images is inherently heavy.
3. FFSG is designed for U-Net-based LDMs (with self-attention). Emerging diffusion architectures (e.g. Diffusion Transformers) may not readily accommodate the same attention-based feature injection. A more important point is that the DiT architecture is currently the mainstream (such as SD3 and FLUX), which limits the application of the method presented in this paper.
4. This method introduces several new hyperparameters. The paper only provides default values for these parameters, but does not offer any sensitivity analysis on how these hyperparameters affect the final results. This makes it impossible for readers to determine whether the method is easy to adjust and whether a large amount of cumbersome parameter tuning is required when applied to new tasks or backbone models.
5. The paper lacks comparisons with the latest state-of-the-art super-resolution or image restoration models, which makes it difficult to fully assess the effectiveness of the proposed method.

### Questions
See Weaknesses

### Soundness
2

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
The paper proposes FreeAdapt, a training-free framework to adapt pre-trained diffusion models (e.g., LDM, DiffBIR) to ultra-high-definition image restoration (UHD-IR) tasks. The method first generates a structure-preserving reference image at a lower resolution, then during UHD latent denoising, applies (1) frequency-domain guidance that fuses low-frequency phase information from the reference to enforce global structural and color consistency across patches, and (2) feature-level guidance that injects reference features into self-attention to provide global context and suppress hallucinated textures. In addition, an optional lightweight VAE decoder fine-tuning module with multi-scale skip connections and LoRA is introduced to better reconstruct high-frequency details. Experiments on several UHD-IR benchmarks show consistent gains over both traditional UHD restoration networks and other training-free diffusion-based adapters, especially in perceptual metrics and visual fidelity.

### Strengths
1. The paper proposes a novel training-free framework that adapts diffusion priors to ultra-high-resolution image restoration via designed FFSG mechanism.
2. The plug-and-play design can be seamlessly applied to different diffusion-based restoration models.
3. The paper is clearly written and well structured.

### Weaknesses
1. While the paper correctly reviews MultiDiffusion, DemoFusion, and PixelSmith as training-free methods for high-resolution generation, these are later treated as baselines for UHD restoration without a dedicated discussion of the additional consistency constraints that restoration requires. It would be helpful to more explicitly clarify in what sense these generation-oriented methods are adapted to restoration, and how input–output consistency is enforced or evaluated beyond standard full-reference metrics.
2. This paper lacks comparisons with the latest state-of-the-art uper-resolution or image restoration methods; many works mentioned in the related work section (e.g., DreamUHD, StableSR) are not included in the experimental evaluation, which makes it difficult to fully assess the competitiveness of the proposed approach.
3. FFSG is implemented by directly modifying the self-attention blocks in the U-Net decoder, and the paper does not provide any discussion or experiments on how to extend the proposed guidance mechanism to diffusion transformers. Given that DiT-style models are becoming the mainstream choice for strong diffusion priors, this U-Net dependency may substantially restrict the method’s impact.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
