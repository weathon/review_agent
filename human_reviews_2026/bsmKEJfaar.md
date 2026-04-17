# Boosting Latent Diffusion Models via Semantic-Disentangled VAE

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Latent Diffusion Models (LDMs) rely on image tokenizers, typically implemented as Variational Autoencoders (VAEs), to compress high-dimensional images into compact latent space, facilitating efficient generative modeling. We contend that VAEs trained solely on pixel-level reconstruction objective struggle to capture rich semantic information, which poses challenges for the modeling of downstream diffusion models. In this paper, we propose that a generation-friendly VAE should have the ability of semantic disentanglement, which means it can encode attribute-level semantic information more effectively. To address this, we introduce Semantic-disentangled VAE (Send-VAE), which leverages the rich semantic knowledge from pre-trained vision foundation models to improve the VAE’s ability to disentangle semantics. Specifically, we employ a sophisticated non-linear mapper network to transform VAE’s latent representations, then align them with the representations from vision foundation models. The mapper network is designed to bridge the representation gap between VAE and vision foundation models, thus facilitating effective guidance for VAE learning. Additionally, we implement linear probing on attribute prediction tasks to assess the VAE’s semantic disentanglement ability, demonstrating a strong correlation with downstream generation performance. Finally, utilizing on the proposed Send-VAE, we train popular flow-based transformers SiTs, and experimental results indicate that our proposed Send-VAE can significantly speed up SiT training and achieves a new state-of-the-art FID score of 1.21 and 1.75 with and without classifier free guidance on ImageNet 256 × 256 resolution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses the limitations of conventional VAEs in Latent Diffusion Models, which often lack semantic understanding due to their focus on pixel reconstruction. The authors identify semantic disentanglement as the key attribute for a "generation-friendly" VAE. They introduce Send-VAE, a method that employs a sophisticated mapper network to align the VAE's latent space with a pre-trained Vision Foundation Model (VFM), thereby bridging their "representation gap."

### Strengths
1. The paper delves into the fundamental question of which properties of a VAE are crucial for the generative process. It proposes a novel approach to evaluate the semantic disentanglement of the VAE's latent space based on attribute prediction tasks.
2. The proposed method achieves state-of-the-art (SOTA) gFID on ImageNet 256x256 while also demonstrating a significant acceleration in convergence speed.
3. The authors provide comprehensive ablation studies on the mapper network's depth, noise injection, choice of VFM, and VAE initialization, which thoroughly validate their design choices.

### Weaknesses
1. The distinction from VA-VAE needs further clarification. VA-VAE also proposed enhancing the semantics of the latent space by aligning it with a vision foundation model. The current work builds upon this by introducing noise injection and a more complex mapper network. The connection between these specific technical improvements and the core motivation of enhancing the VAE's semantic disentanglement ability is not sufficiently explained and requires more elaboration.
2. A majority of the experiments are based on fine-tuning a pre-existing VA-VAE. To provide a more controlled and fair comparison, an additional experiment is needed. Specifically, starting from the same VAE initialization, the authors should compare the results of fine-tuning using the original VA-VAE methodology versus their proposed Send-VAE methodology under identical conditions.
3. The paper could be strengthened by including more qualitative comparisons, such as latent space visualizations (e.g., t-SNE plots or attribute-based interpolations) against other VAEs. This would provide more intuitive visual evidence to support the claim of improved semantic disentanglement.

### Questions
Please check the weaknesses

### Soundness
3

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
3

### Summary
The paper proposes Send-VAE, which aligns VAE latents to those of a pretrained vision foundation model (VFM) to inject richer semantics into the VAE space. Concretely, the authors add a patch-wise cosine alignment loss to the standard VAE objective and train a non-linear mapper to bridge the VAE–VFM gap. They argue that a VAE’s semantic disentanglement is key to improving latent diffusion models (LDMs). On ImageNet-256, Send-VAE reports faster convergence and new SOTA FID: 1.21 (with CFG) / 1.75 (without CFG). The problem is important, but the core contribution—adding a non-linear mapper for VAE–VFM alignment—appears incremental relative to existing alignment/distillation approaches, and the empirical scope (ImageNet-256 only) limits the broader significance.

### Strengths
- The paper explains why several popular VAE-space metrics fail to predict generative quality and motivates attribute-level linear probing as a more reliable proxy.
- The paper is well structured and easy to follow.

### Weaknesses
- **Limited novelty:** Aligning VAE latents to a frozen VFM follows prior work (REPA/REPA-E, VA-VAE). Send-VAE’s main change is a non-linear mapper, which is an architectural tweak rather than a new principle. 
- **Sensitivity to tuning:** Although the mapper is presented as bridging the representation gap, performance still depends on careful fine-tuning of this network (cf. Table 2). 
- **Narrow experimental scope:** Experiments are restricted to ImageNet-256; there are no results at higher resolutions or for text-to-image settings.
- **Reproducibility:** No code is provided.

### Questions
- How is Gaussian noise injected in Eq. (1) during Send-VAE training (fixed variance or scheduled)?
- Without REPA loss, does Send-VAE still perform well?
- In the “system-level comparison on ImageNet 256×256 conditional and unconditional generation” (Table 1), are the reported numbers unconditional, or conditional without CFG?
- What is the baseline referenced in Table 4?
- What are the mapper details beyond depth—e.g., width, number of heads, and patch size of the ViT block?
- Did you try other VFMs such as SigLIP or MAE? Given their competitive representations vs. DINOv2, does Send-VAE still have similar gains?
- Please fix citation style: use \citet only when the authors’ names are part of the sentence.
- “Ablation on Vision Foundation Models … present the ablation results in Table 2” appears to reference Table 4—please check the table cross-reference.

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
- They investigate the generation-friendly latent space of VAE. Based on the analysis, they proposed a sophisticated mapping network which aligns the representation of the vision foundation model with the latent of VAE while training. Specifically, they employ various measurements for linear probing and found that the low-level attributes are the key factor that the VAE should disentangle within the latent space. Through the analysis, they alternate a simple MLP mapping network to a patch-level mapping network, and introduce patch-wise alignment loss. Qualitative and quantitative results support their claims

### Strengths
- The author has well defined the semantics of the DINO feature in terms of generation performance.
- Through comparison of existing and recent methods
- The proposed method is well related to the preliminary experiments
- Qualitative results are visually pleasing.

### Weaknesses
- Most of the experiments are conducted with 256x256
- Lack of results with the text-to-image setting.

### Questions
- While training VAE, why do you use the mapping function rather than directly using similarity? Initializing the model with the original vision foundation model could be an alternative, which has the same latent dimensions and

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies what makes a VAE effective when used as the tokenizer in latent diffusion models. The authors argue that the key property is how well the VAE’s latent space separates different semantic attributes (i.e., its semantic disentanglement ability).

Based on this idea, the paper introduces Send-VAE, which improves semantic structure in the VAE’s latent space by aligning it with features from a pretrained vision foundation model (DinoV2). To do this, the authors use a 2-depths non-linear mapper network between the VAE and the vision model, which helps bridge the gap between their representations more effectively than direct alignment.

Using Send-VAE leads to faster convergence and better generative performance in diffusion models such as SiT, achieving state-of-the-art FID scores on ImageNet 256×256, both with and without classifier-free guidance. The paper also provides thorough ablations studying the mapper depth, noise injection, different VAEs for initialization, and different vision foundation models.

### Strengths
1. The paper provides an interesting and clear intuition: instead of treating the VAE purely as a reconstruction model, it should be trained in a way that directly supports downstream generation. Framing semantic disentanglement as the key property of a “generation-friendly” VAE is insightful and well-motivated.
2. The proposed method achieves strong empirical performance. In particular, Send-VAE leads to faster convergence and improved ImageNet 256×256 generation quality, reaching competitive or state-of-the-art FID scores.
3. The experimental validation is thorough. The paper includes comprehensive ablations (e.g., mapper depth, noise injection, VAE initialization choices, and different vision foundation models), which help isolate the contributions of each component and strengthen the overall claims.

### Weaknesses
1. Although the paper focuses on building a VAE that is more suitable for generation, it would still be helpful to report reconstruction performance more explicitly. Since VAEs traditionally balance semantic structure and pixel-level fidelity, showing the reconstruction trade-offs would make the argument more complete and allow the reader to better understand what is being sacrificed or preserved. (PSNR, LPIPS, SSIM, ... FID is not enough.) 
2. While the proposed approach is conceptually sound, similar ideas have appeared in prior work. In particular, VA-VAE and related methods also align the VAE latent space with representations from pretrained vision encoders. The paper does introduce a non-linear mapper, but the conceptual difference from earlier representation alignment approaches is somewhat incremental. Clarifying the novelty relative to those prior works would strengthen the contribution.

### Questions
1. In Table 1, are all the baseline VAEs trained under the same diffusion backbone (SiT) and using the same training data and protocol? Confirming this is important to ensure that the improvements shown are attributable to the proposed VAE rather than differences in diffusion training setups.
2. The paper shows a correlation between semantic disentanglement and downstream generation performance. Did the authors attempt any controlled intervention to demonstrate causality (e.g., artificially altering disentanglement levels while keeping reconstruction constant)?
3. Since the mapper has significant capacity, how can we be sure that the performance improvement comes from a better VAE latent space rather than the mapper effectively compensating for VAE weaknesses at inference time?

### Soundness
3

### Presentation
2

### Contribution
3
