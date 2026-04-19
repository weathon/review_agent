# Boosting Latent Diffusion with Perceptual Objectives

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 5

## Abstract
Latent diffusion models (LDMs) power state-of-the-art high-resolution generative image models. LDMs learn the data distribution in the latent space of an autoencoder (AE) and produce images by mapping the generated latents into RGB image space using the AE decoder. While this approach allows for efficient model training and sampling, it induces a disconnect between the training of the diffusion model and the decoder, resulting in a loss of detail in the generated images. To remediate this disconnect, we propose to leverage the internal features of the decoder to define a latent perceptual loss (LPL). This loss encourages the models to create sharper and more realistic images. Our loss can be seamlessly integrated with common autoencoders used in latent diffusion models, and can be applied to different generative modeling paradigms such as DDPM with epsilon and velocity prediction, as well as flow matching. Extensive experiments with models trained on three datasets at 256 and 512 resolution show improved quantitative -- with boosts between 6% and 20% in  FID -- and qualitative results when using our perceptual loss.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces "latent perceptual loss" (LPL) to improve the training of diffusion models. According to the paper, LPL enhances diffusion models performance across various datasets (Table 2) and methods (Table 3). Additionally, extensive analyses, e.g., comparing the generation quality with and without LPL loss, and ablation studies, deepen our understanding of LPL.

### Strengths
**1. Extensive analysis**

This work provides extensive analysis to demonstrate the validity of LPL. For instance, the major performance gain from LPL arises from generating more precise high-frequency components, albeit at the expense of some low-frequency details.

**2. Ablation study**

It gives us various ablation study to provide the motivation behind design choices. The ablation study includes various components including feature depth for LPL, SNR threshold value, Reweighting strategy, or etc. This level of extensive ablation study is rare.

### Weaknesses
**1. Efficiency**

In the paper, it is mentioned several times that utilizing the VAE decoder features to compute the LPL loss is costly. This is why recent research has explored alternative perceptual losses, such as latent LPIPS [1]. While the paper claims that LPL incurs minimal computation since it’s only used post-training, this claim is far from acceptable, as the post-training phase in this paper involves iterations amounting to as much as one-third, or at minimum one-fifth, of the pretraining iterations.

[1]: Distilling Diffusion Models into Conditional GANs ([ECCV24](https://mingukkang.github.io/Diffusion2GAN/))

**2. Novelty**

Although subjective, I believe the novelty of LPL is somewhat lacking. This loss trick slightly enhances quality during training but seems more heuristic than principled. Additionally, it doesn’t contribute much new knowledge about diffusion models, which weaken the paper's novelty. Furthermore, it does not appear that any non-trivial trick was devised in the process of introducing the perceptual loss to the diffusion models.

### Questions
**1. Efficiency**

- How does the result compare when using FLOPS as the x-axis? Comparing by iteration seems somewhat unfair. If the performance improvement looks promising when compared in terms of FLOPS, I'm considering raising the score.

**2. Novelty**

- Is there a specific reason why LPL was only used in post-training? I feel that LPL might provide a good gradient signal early in training, but it seems that experiment wasn't conducted, which is unfortunate. I'm curious about the performance changes if LPL is applied from the start of the pretraining.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The author introduces latent perceptual loss which acts on the decoder's intermediate features to enrich the training signal of LDM. The experiments showcased the effect of the added perceptual loss.

### Strengths
- The author proposed latent perceptual loss (LPL) which shows the efficacy over various tasks datasets.
- The qualitative results and the quantitative metrics seem promising
- The frequency analysis showcases that the methods works
- The ablations are carried out in a systematic way

### Weaknesses
- The paper seems unpolished and rushed towards the deadline
- Sometimes the notation is a bit confusing, see the third point in the questions section
- Applying perceptual loss in T2I generation isn’t novel; Lin and Yang [1] calculates the perceptual loss in middle blocks to reduce computation which bypasses the computational constraint, whereas this paper requires passing results to the decoder for intermediate features. If the authors can provide evidence that the method proposed is better and distant themselves to the literature then I would consider raising my score.

---
[1] Lin, S., & Yang, X. (2023). Diffusion Model with Perceptual Loss. arXiv preprint arXiv:2401.00110.

### Questions
- Line 179, Many SOTA T2I diffusion models use a basic L2 training objective. Since only a VAE citation is given to explain blurriness, could you add citations or evidence that diffusion training leads to blurriness? Typically, adversarial losses are used in diffusion distillation, not in standard diffusion training.
- Line 185, "a perceptual loss cannot be used directly on the predicted latents", there is already accepted literature [2] that train another model which efficiently compute LPIPS losses in the latent space. It would be interesting to compare and contrast to the LatentLPIPS method.
- Line 193, it would be better if the authors could stick to the canonical DDPM/EDM [3] notations to avoid confusion.
- Line 289, do you enforce zero-terminal SNR here? It is known that the SD diffusion schedules are flawed [4].
---
[2] Kang, M., Zhang, R., Barnes, C., Paris, S., Kwak, S., Park, J., ... & Park, T. (2024). Distilling Diffusion Models into Conditional GANs. In ECCV 2024

[3] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35, 26565-26577.

[4] Lin, S., Liu, B., Li, J., & Yang, X. (2024). Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 5404-5411).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes applying perceptual losses to latent diffusion models during training to improve the quality of generated images. Their proposed perceptual loss uses features from the VAE decoder, with additional contributions such as outlier filtering being proposed to enable this setup to work. The authors extensively validate the benefits of their approach over standard training of MMDiTs in different diffusion formulations and datasets.

### Strengths
The comparisons with baselines are extensive and clearly show that the proposed latent perceptual loss objective improves metrics across different diffusion formulations and datasets. I really appreciate the authors demonstrating the improvements for both eps-pred diffusion and flow matching, demonstrating that the proposed loss potentially has a general significance.

The paper ablates over/explores a large number of parameters and their influence.

The paper is generally reasonably well-written and accessible.

### Weaknesses
I think framing the proposed loss as a perceptual loss is likely incorrect. Perceptual losses typically try to incorporate human perception-based invariances into the loss, such as weighting the presence of the correct texture (e.g., grass) as more important than getting every detail of the instance of the texture (e.g., the exact positions of individual blades of grass) right. This is directly opposed to losses such as the MSE in pixel space. This is typically accomplished by taking features of a deep pre-trained discriminative model. In this case, you use features of the VAE latent decoder, which, potentially even puts the features at a lower abstraction level than the original latents. As far as I could see, there was no investigation of whether these features have the qualities of a perceptual loss. In the end, optimizing this loss seems to result in improved FIDs, which makes it a valuable contribution. Still, I think calling it a perceptual loss without further investigation goes against what people expect "perceptual losses" to refer to, which could lead to confusion. Could it be that the improvement actually doesn't come from perceptual qualities of the loss but rather from other qualities, such as a different implicit weighting of timesteps, which has previously been shown to substantially improve FIDs as well [https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf].

You also claim that "the autoencoder’s latent space has a highly irregular structure and is not equally influenced by the different pixels in the latent code" as an important part of your motivation. However, it has been shown [e.g., in https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204] that, at least for the very commonly used SD VAEs, the autoencoder latents effectively correspond to downsampled colors, differing fundamentally from that statement. Similarly, noisy latents from intermediate steps of the sampling process often already show a good approximation of the image being generated. A.2 shows an investigation of the effect of spatial interpolation for some (unspecified) AE, but is inherently very limited in its scope. It would be nice to see better backing up of this central claim that goes against common assumptions using standard methods, such as introducing small (random) deviations and showing how their effect on the decoded image is disproportionally large compared to larger ones. General claims should also be verified with standard VAEs.

### Questions
In addition to the addressing the weaknesses, I'd appreciate the authors answering the following questions:

What is the difference in training FLOPS per step of using LPL vs. just the normal objective? I assume it is a minor difference, but if it is not, just comparing to normal objectives at the same step count would not be fair. Similarly, the effect of the substantial increase in VRAM usage on the training speed (via the maximum achievable local batch size) should be quantified properly.

Why do you use a CFG scale of 2? Standard choices for latent diffusion are typically much lower for ImageNet class-conditional generation and much higher for text-to-image.

Why is the frequency cut off at 360 in Fig. 6?

Suggestion: while not technically required according to ICLR's rules, as it hasn't been published yet (and therefore also not taking an influence on my rating of this paper), I think a discussion of the almost one year old seminal work [https://arxiv.org/abs/2401.00110] in the area of applying perceptual losses to diffusion models would help improve the paper, especially if this work had any influence on the submitted paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies the perceptual loss function in the training of diffusion models and proposes to compare the features between $z_0$ and $\hat{z}_0$ by sending them into the latent autoencoder's decoder model. The loss is only applied during the post-training stage and only applied in the time steps that have a SNR higher than the preset threshold. The results on CC3M, ImageNet-1k and S320M show that the introduction of this training strategy could help improve the generation quality of the model.

### Strengths
1. The motivation is clear and straightforward. And the proposed method is simple and can be easily applied to the training of other diffusion models.
2. Under the same training iterations during the post-training stage, the method can improve the FID over the baseline method which only adopts the MSE loss.

### Weaknesses
1. The paper only shows the performance increase over the baseline model. I feel like it's better to clearly demonstrate the effectiveness and performance gain over the previous state-of-the-art methods, to show that the perceptual loss can achieve what the widely used MSE loss cannot achieve.
2. The introduction of the perceptual loss would increase the computation cost during the training stage. Could the authors provide a clear comparison on this? 
3. Following the last one, what would be the performance comparison for the same training time instead of the same training iterations?
4. The authors mention the outliers in the features of the autoencoder's decoder which is not ideal for the computation of the perceptual loss. I'm wondering if the authors have tried other ways instead of simply masking those features out as this might cause information lost. Or has the authors tried using some other models to compute the perceptual loss to avoid those outliers?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
