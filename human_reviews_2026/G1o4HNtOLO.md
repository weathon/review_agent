# SSDD: Single-Step Diffusion Decoder for Efficient Image Tokenization

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Tokenizers are a key component of state-of-the-art generative image models, extracting the most important features from the signal while reducing data dimension and redundancy. Most current tokenizers are based on KL-regularized variational autoencoders (KL-VAE), trained with reconstruction, perceptual and adversarial losses. Diffusion decoders have been proposed as a more principled alternative to model the distribution over images conditioned on the latent. However, matching the performance of KL-VAE still requires adversarial losses, as well as a higher decoding time due to iterative sampling. To address these limitations, we introduce a new pixel diffusion decoder architecture for improved scaling and training stability, benefiting from transformer components and GAN-free training. We use distillation to replicate the performance of the diffusion decoder in an efficient single-step decoder. This makes SSDD the first diffusion decoder optimized for single-step reconstruction trained without adversarial losses, reaching higher reconstruction quality and faster sampling than KL-VAE. In particular, SSDD improves reconstruction FID from 0.87 to 0.50 with 1.4⨉ higher throughput and preserve generation quality of DiTs with 3.8⨉ faster sampling. As such, SSDD can be used as a drop-in replacement for KL-VAE, and for building higher-quality and faster generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SSDD, an efficient single-step diffusion decoder, which achieves improvements in both reconstruction quality and sampling speed without relying on GAN loss.

### Strengths
1. The paper introduces SSDD, a single-step decoder, and provides extensive experiments demonstrating its effectiveness in improving reconstruction fidelity and training efficiency. 


2. The exploration of removing GAN losses to achieve more stable training is valuable and practically relevant for large-scale diffusion modeling.

### Weaknesses
1. This is primarily an engineering-focused work. The proposed SSDD architecture is largely based on U-ViT, with modifications to the resolution hierarchy and the inclusion of REPA loss, LPIPS loss, and a distillation technique. While these design choices lead to strong empirical performance, the methodological novelty and conceptual insights remain limited. Therefore, although this work demonstrates improvements in training efficiency, the insights are relatively limited, which leads me to keep my overall rating at the marginal level.

2. The experiments are conducted mainly on ImageNet. It would strengthen the paper to include additional evaluations on out-of-distribution datasets such as COCO or other domains, to verify the generalization of both reconstruction and generation performance.

### Questions
See Weaknesses.

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
3

### Summary
This paper introduces SSDD (Single-Step Diffusion Decoder), a new image tokenizer designed to replace traditional KL-regularized VAEs. The method's key contributions include: 1) A new pixel diffusion decoder architecture that combines a convolutional U-Net with a central transformer block, designed for improved scalability and stability. 2) A GAN-free training scheme that relies on a combination of a flow-matching objective, a perceptual LPIPS loss, and a REPA feature regularization loss. 3) A distillation process that transfers the high-quality output of a multi-step diffusion decoder into a fast, single-step model. The authors demonstrate that SSDD achieves state-of-the-art reconstruction quality on perceptual metrics like rFID and LPIPS, surpassing existing VAEs and diffusion decoders while offering significantly higher throughput. They also show that using SSDD as a decoder for a DiT model preserves generation quality while drastically speeding up inference.

### Strengths
- This paper performs good improvements against baseline.
- Extensive experiments are conducted.

### Weaknesses
- The baseline is just $\epsilon$-VAE which is limited.
- What is the difference between train regular decoder and then train a refiner against diffusion decoder?
- Diffusion decoder is more like a generation task instead of reconstruction. The usage of it is questionable.
- There is no novelty in the method side.

### Questions
see above

### Soundness
2

### Presentation
2

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
This paper introduces SSDD (Single-Step Diffusion Decoder), a novel diffusion-based autoencoder for image tokenization that addresses key limitations of existing approaches. The authors propose a hybrid U-Net-transformer architecture that leverages flow matching with perceptual alignment (LPIPS) and REPA regularization to achieve state-of-the-art reconstruction quality without adversarial training. Through a distillation strategy, they compress multi-step diffusion behavior into a single-step decoder, achieving 3.8× faster sampling while maintaining generation quality. SSDD improves reconstruction FID from 0.87 to 0.50 compared to KL-VAE with 1.4× higher throughput, making it suitable as a drop-in replacement for existing tokenizers in generative models.

### Strengths
-  SSDD achieves impressive performance improvements across multiple metrics, particularly in reconstruction FID (0.87→0.50) and generation speed (3.8× faster) compared to baselines.
-  Successfully eliminates the need for adversarial training in both encoder training and decoder distillation, while achieving competitive or superior results.
- The paper includes extensive experiments with thorough ablations, multiple baselines, and evaluation across various resolutions (128×128 to 512×512) and model sizes (13.4M to 345.9M parameters).

### Weaknesses
- The core architecture builds heavily on existing components (U-ViT, REPA loss, flow matching). While the combination is effective, the individual components are not novel. 
- Although SSDD is the first work to demonstrate that a single-step diffusion decoder can match the performance of multi-step diffusion decoders, similar work has already been thoroughly explored in the context of conditional diffusion models (text-to-image or image-to-image). Replacing the condition from text with latents seems not particularly special. This significantly diminishes the contribution of this work.

### Questions
See Weaknesses.

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
This paper presents SSDD, a single-step diffusion decoder designed for efficient image tokenization. The goal is to overcome the inefficiency of KL-regularized VAEs and multi-step diffusion decoders, which typically require adversarial losses and iterative sampling. SSDD introduces a U-ViT–based pixel diffusion decoder trained under a GAN-free flow-matching objective and perceptual regularization, followed by a distillation stage that compresses a multi-step diffusion process into a single-step model. The method claims to improve reconstruction FID from 0.87 to 0.50 and increase decoding throughput by 1.4×, while maintaining the generation quality of DiT models with 3.8× faster sampling. Experiments on ImageNet demonstrate quantitative advantages over KL-VAE, SD-VAE, LiteVAE, ε-VAE, and VA-VAE, across multiple compression setups.

### Strengths
- The work targets a practical and relevant limitation in generative pipelines — the trade-off between reconstruction quality and decoding speed. The single-step distillation from a multi-step diffusion decoder is an intuitive and useful engineering contribution that could simplify downstream diffusion training.

- The model design is computationally efficient (U-ViT backbone, GAN-free objective) and can potentially serve as a drop-in replacement for VAEs in large-scale text-to-image systems.

### Weaknesses
**Soundness**

My main concern is with the quantitative evaluation. Unless I’m missing something, several reported numbers—especially in Table 3—look inconsistent with established baselines. For instance, the paper claims substantial gaps for the KL-VAE f8c4 tokenizer on ImageNet 256×256; the no-CFG FIDs in the teens and with-CFG FIDs around 6.x are noticeably worse than what is typically reported for comparable setups. This discrepancy makes it difficult to interpret the claimed SSDD gains. Please clarify:

- the exact evaluation pipeline (data preprocessing, Inception network/version, number of samples, seeds),
- the CFG scale search protocol and whether scores are reported with best-searched CFG,
- whether your KL-VAE/SD-VAE checkpoints and training strictly reproduce the original implementations.*

**Technical contribution**

The “new” elements are largely combinations of existing ideas. The authors are essentially applying standard distillation methods to diffusion models for diffusion decoders, without providing significant insights. 

**Presentation**

The presentation of this paper has significant room for improvement. I highly recommend the authors to polish their paper to a high standard of English. Sentences in this paper are often incomplete or ungrammatical. Examples:

- 'Common tokenizers such as the KL-VAE from Rombach et al. (2022) are optimized with L1 reconstruction loss, LPIPS (Zhang et al., 2018), and a GAN discriminator (Goodfellow et al., 2014), to which a KL-regularization of the latent space is added.' What does the 'which' refer to here is unclear. It will not be a major obstacle for an experienced reader to comprehend, but it is not grammatically correct. 

- 'Pixel-space diffusion decoders mainly leverage the same convolutional U-Net architectures (Zhao et al., 2025a) that were found successful in early pixel-space diffusion models (Dhariwal & Nichol, 2021).' I believe there is a typo.

### Questions
- Baseline Consistency:
The reported rFID improvement over KL-VAE is substantial, but the baseline result (rFID = 0.87) appears considerably weaker than the commonly cited performance of the official Stable Diffusion VAE. Could the authors clarify their training setup—including optimizer, loss weights, and data preprocessing—and verify whether their reproduction matches the official KL-VAE checkpoint performance on ImageNet 256×256? Without this confirmation, the strength of the reported improvement is difficult to assess.

- Perceptual Quality After Distillation:
Does the single-step distillation process lead to any loss of perceptual fidelity or fine-detail degradation (e.g., texture aliasing, local blurring, or over-smoothing)? The qualitative examples shown in Figure 3 (right) are not convincing, since the reference image is already very blurry and lacks high-frequency details. Including sharper examples or zoomed-in patches from higher-detail regions would make the comparison more informative.

- Generalization and Scalability:
Could the authors discuss whether the proposed decoder generalizes to higher-resolution images (e.g., 512×512 or 1024×1024) or to cross-domain datasets (e.g., faces, artworks, medical images) without retraining? If retraining is required, how sensitive is the single-step distillation process to scale and domain shifts?

### Soundness
2

### Presentation
1

### Contribution
2
