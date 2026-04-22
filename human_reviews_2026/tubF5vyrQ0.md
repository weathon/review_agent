# VUGEN: Visual Understanding priors for GENeration

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Recent advances in Vision-Language Models (VLMs) have enabled unified understanding across text and images, yet equipping these models with robust image generation capabilities remains  challenging. Existing approaches often rely on reconstruction-oriented autoencoders or complex bridging mechanisms, leading to misalignment between understanding and generation representations, or architectural complexity. In this work, we propose VUGEN, a novel framework that explicitly leverages VLM's pretrained visual understanding priors for efficient and high-quality image generation. Our approach first transforms the high-dimensional latent space of the VLM’s native vision encoder into a lower-dimensional, tractable distribution that maximally preserves visual information. The VLM is then trained to sample within this reduced latent space, ensuring alignment with its visual understanding capabilities. Finally, a dedicated pixel decoder maps these generated latents back to the image space. We find that a VAE-free pixel diffusion decoder to be on par or better than commonly used complex latent diffusion decoders that internally rely on VAE latents. Extensive experiments  demonstrate that VUGEN achieves superior image generation performance, improving DPG Bench from 71.17 to 74.32 and FID from 11.86 to 9.06 on COCO, while fully preserving the VLM’s original understanding capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes VUGEN, a framework that leverages pretrained visual understanding embeddings (from a frozen VLM) as priors for image generation. A learnable dimension reducer is introduced to map the high-dimensional understanding space to a lower-dimensional latent space, making it easier to model with a generative flow-matching network. The model then decodes the latent space into images using either a lightweight pixel diffusion decoder (PDD) or a latent diffusion decoder (LDM). Experiments on StockMix and ImageNet show that VUGEN outperforms VLM-based generation baselines.

### Strengths
The idea of reusing pretrained visual understanding embeddings as generative priors is both intuitive and meaningful, offering potential to bridge understanding and generation tasks in multimodal modeling. The use of a learnable dimension reducer to create a smoother and more compact latent space is technically sound and empirically validated. Additionally, the lightweight and fast PDD provides a practical alternative to standard latent diffusion decoders.

### Weaknesses
- This work primarily focuses on generation, with an emphasis on training and evaluating on generation tasks. Therefore, it should be compared to state-of-the-art generative models, rather than unified multimodal models (UMMs). Additionally, even when compared to UMMs, the generation performance of VUGEN is not particularly outstanding.

- If the paper aims to argue for VUGEN as a unified multimodal model (UMM), it falls short in terms of unification. Also, evaluations on visual understanding and reasoning tasks are necessary to fully justify its claim as a UMM.

- The core idea of the framework is to leverage understanding priors for generation, so further clarification and more analyses are necessary regarding the contribution of these priors, theoretically, qualitatively and quantitatively.

- The motivation is not clearly articulated enough in the abstract and introduction. The claim that generating in the understanding latent space is challenging requires more theoretical and empirical analyses.

- Clarifying the computational cost and scalability of the two-stage training would facilitate the cost-benefit comparisons.

### Questions
Please see weaknesses.

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
This paper introduces VUGEN, a unified vision language model for multimodal understanding and generation. To equip a pre-trained VLM with image generation capability, VUGEN transforms the high-dimensional latent space of the VLM's native vision encoder into a lower-dimensional one to simplify VLM's training for generation while preserving visual information.

### Strengths
- The paper is well-written and easy to follow.
 - The generation experiments are done on diverse datasets and evaluated with compreshensive metrics.
 - The ablation experiments on dimension reduction ratio provides valuable insights into the trade-off between the generation task and the reconstruction tasks.

### Weaknesses
- Reconstruction metrics like PSNR, SSIM, LPIPS, rFID are not reported to compare the proposed pretrained vision encoder + dimension reducer + pixel decoder with existing autoencoders (e.g., SD-VAE and Flux-VAE) used in other diffusion- or autoregressive-based image generation models.
 - The paper only considers two pixel decoder designs, LDM and PDD, both of which are diffusion based models. Another simple but important baseline would be a standard convolutional decoder commonly used in autoencoders like SD-VAE and Flux-VAE.
 - The image understanding task uses the original vision encoder features, while the image generation task uses the compressed representations from the dimension reducer, resulting a gap between the two spaces. What is the advantage of this unified design compared to decoupled vision encoders baselines if semantically aligned VAEs like VA-VAE[1] are used?
 - The baselines reported in Table 1 for ImageNet generation appear relatively weak. The current state-of-the-art FID score on ImageNet 256x256 is below 2. Also, only the SD3 VAE is considered in the decoupled vision encoders baseline. Exploring autoencoders like VA-VAE[1], which incorporate semantic information, would provide a more complete comparison.
 - As VUGEN introduces a separate module for image generation, it would be helpful to clarify its advantages over previous methods like LaVIT[2] and Emu[3] which introduce another diffusion model for image generation based on VLM output features?

[1] Taming Optimization Dilemma in Latent Diffusion Models

[2] Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization

[3] Emu: Generative Pretraining in Multimodality

### Questions
Please refer to the weakness session.

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
The paper proposes VUGEN, a two-stage approach to equip a unified VLM with image generation by directly leveraging its native visual understanding features. Stage 1 learns a dimension reducer g that projects the high-dimensional understanding embeddings z from the VLM’s vision encoder into a reduced, tractable space ˜Z optimized jointly with a pixel decoder d (either a finetuned LDM or a lightweight pixel-space diffusion decoder, PDD). Stage 2 freezes g and trains a generative head (Mixture-of-Transformers tower) via rectified flow matching to sample ˜z ∼ P(˜z|c), then decodes to pixels x via d(˜z). The key idea is to align generation with the model’s understanding priors, avoiding representation mismatch from separate (VQ-)VAE tokenizers and the complexity of bridging to external diffusion models.

  Empirically, VUGEN improves prompt-following and fidelity on COCO (models trained on StockMix): DPG-Bench 71.17→74.32 and FID 11.86→9.06 vs. a REPA-aligned decoupled baseline; and outperforms baselines trained on ImageNet (FID 5.40→4.15, Density/Coverage up). Ablations show: (i) directly generating in Z is hard; a jointly learned reducer outperforms PCA; (ii) pixel-space diffusion decoder (PDD) achieves comparable reconstructions to LDM but with far fewer params (48M vs. 794M) and higher throughput; (iii) a reduction ratio r≈16 balances generative tractability and decoding difficulty. Understanding performance is preserved at base VLM levels (Table 3).

### Strengths
- Clear, well-motivated design: samples in a reduced version of the VLM’s understanding space, preserving alignment between understanding and generation; strong rationale and ablations (PCA vs. learned reducer; r trade-off).
- Competitive results with careful baselines sharing architecture/data/training: improves FID/DPG/GenEval across StockMix→COCO and ImageNet settings; analysis over guidance scale clarifies realism–consistency trade-offs.
- Practical decoder findings: pixel-space diffusion decoder rivals LDM while being far smaller and faster; avoids dependence on VAE latents, reducing complexity.
- Preserves understanding: retains PLM-1B’s comprehension performance; shows a path to unified MLLMs without decoupled vision tokenizers.

### Weaknesses
- Data provenance and comparability: the main training uses a mixed StockMix (YFCC100M, CC12M, and a proprietary S320M recaptioned with Florence-2). While baselines share this setup, cross-paper comparability is limited; clearer licensing/availability statements for S320M would help.
- Limited scope of understanding preservation: while Table 3 suggests parity on standard benchmarks, it would be useful to test for regressions in more fine-grained or long-context visual reasoning after generative training, especially under higher r.
- Generative scaling and distribution shift: results are at 256×256; how do trends hold at higher resolution and for out-of-domain prompts? Also, how stable is training when swapping in different base VLM encoders (e.g., DINOv2 or SigLIP variants)?
- Theoretical underpinnings: the choice of rectified flow matching is reasonable; adding a brief justification vs. diffusion loss and showing a small apples-to-apples comparison would strengthen claims of sample efficiency.
- Decoder choice vs. alignment: PDD and LDM are “similar” in reconstructions; however, prompt alignment (DPG/GenEval) contributions per component (reducer vs. generator vs. decoder) could be clarified via controlled ablations.

### Questions
- Does training the reducer jointly with PDD ever reduce understanding robustness? Can you report pre/post shifts on more challenging understanding tasks (e.g., MMMU categories requiring fine localization)?
- How sensitive are results to r and g’s architecture across datasets? Is there a principled way (e.g., information bottleneck or spectral metrics) to set r per vision encoder?
- Could you show a small table isolating “generate-in-Z” vs. “generate-in-˜Z” under the same compute, to quantify the tractability gap, beyond anecdotal FID>200?
- For external comparability: do you have COCO metrics when training only on public data (e.g., YFCC100M+CC12M without S320M), to contextualize gains relative to models trained purely on public datasets?
- PDD details: you mention distillation to a single-step decoder; can you quantify speedups at sample time for end-to-end T2I, not just reconstruction throughput?

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
5

### Summary
The authors propose leveraging visual understanding priors for both visual perception and generation tasks. They transform a high-dimensional semantic latent space into a low-dimensional, tractable distribution that preserves essential visual information. A pixel diffusion model is then trained to generate images from these latent representations. Experimental results demonstrate that, by utilizing a unified semantic visual representation, the method achieves superior image generation performance.

### Strengths
1. The proposed method is intriguing and demonstrates that generating semantic visual latent features can lead to improved image generation performance.
2. To make the generation process feasible, the authors introduce a dimension reduction module, which is a simple yet effective design.
3. The paper is well-written and easy to follow.

### Weaknesses
1. There is no quantitative comparison between the proposed pixel decoder and other mainstream tokenizers, such as VAE and latent diffusion decoders. The authors should include relevant metrics (e.g., rFID) to assess the performance. It remains unclear how well the proposed pixel decoder performs. If its results are significantly worse, it would suggest substantial information loss when generating images from the semantic latent space.
2. The image generation module in the mixture-of-transformers contains only 0.2B parameters. This relatively small capacity raises concerns about the reliability and persuasiveness of the results. I encourage the authors to increase the model size for the image generation component to validate the scalability and robustness of the approach.
3. The dataset S320M is not widely adopted in the community, particularly for unified understanding and generation tasks. The authors should clarify their motivation for using this dataset. Furthermore, the use of such datasets may contribute to the relatively weaker performance on more modern benchmarks, such as GenEval and DPGBench, which diminishes the overall persuasiveness of the work.

### Questions
Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
