# Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Diffusion models have achieved impressive results in generating high-quality images. Yet, they often struggle to faithfully align the generated images with the input prompts. This limitation is associated with synchronous denoising, where all pixels simultaneously evolve from random noise to clear images. As a result, during generation, the prompt-related regions can only reference the unrelated regions at the same noise level, failing to obtain clear context and ultimately impairing text-to-image alignment. To address this issue, we propose asynchronous diffusion models, a novel framework that allocates distinct timesteps to different pixels and reformulates the pixel-wise denoising process. By dynamically modulating the timestep schedules of individual pixels, prompt-related regions are denoised more gradually than unrelated regions, thereby allowing them to leverage clearer inter-pixel context. Consequently, these prompt-related regions achieve better alignment in the final images. Extensive experiments demonstrate that our asynchronous diffusion models can significantly improve text-to-image alignment across diverse prompts. The code repository for this work is available at https://github.com/hu-zijing/AsynDM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an asynchronous diffusion framework for text-to-image generation. The core idea is to create a pixel-level timestep scheduler and let prompt-related regions be decoded more slowly. Extensive experiments demonstrate AyncDM achieves better performance among other training-free text-to-image alignment approaches.

### Strengths
1. The idea is novel and well-motivated. The proposed approach, using cross-attention as a mask indicator, is quite intuitive and easy to follow.
2. This paper is clearly written and well-organized.
3. The comparison results are very promising, showing clear advantages over relevant baselines.
4. The authors conduct comprehensive experiments across multiple model baselines, sampler choices, and ablation settings, which provide strong and convincing evidence for the proposed approach.

### Weaknesses
1. My main concern is the inconsistency between the training and inference stages of AyncDM. From my understanding, during training, noise is added synchronously into all pixels, and the diffusion model predicts $f_\theta(x_t)$ where $x_t$ is a noised latent with uniform noise levels. However, during inference, the input of the diffusion model is asynchronous or spatially varying noise levels across pixels. How can the model reliably decode latents that contain uneven noise distributions, given that it was never explicitly trained under such conditions?  Some marginal artifacts may occur in these scenarios. 
2. The distracted attention mask relies heavily on cross-attention maps. However, in the early denoising steps, cross-attention maps are often noisy and unstable, which may lead to unreliable or ambiguous guidance when determining which regions should be denoised faster or slower.
3. Similarly, for more advanced T2I models such as SD3.5 or Flux that adopt the MMDiT framework instead of conventional cross-attention, deriving reliable spatial masks becomes more challenging, since text and image latents are concatenated within a self-attention module.

### Questions
1. How do the authors address the training–inference gap in AsynDM? Is there any theoretical or empirical evidence showing that this discrepancy can be safely ignored, or that it does not occur during sampling?
2. How does AsynDM apply or adapt the distracted cross-attention mechanism to SD3.5-Medium or other architectures based on MMDiT? The process seems non-trivial and could benefit from further clarification.

I would be happy to raise my score if these concerns are properly addressed.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The work deals with text-to-image (T2I) diffusion models. The authors argue that the misalignment of the generated image and the textual prompt results from a spatially and temporally uniform (synchronous) change of the denoised image pixels. Hence they propose to denoise some regions differently, at different time steps during the reverse denoising process, the timestep of every pixel being dynamically determined. For this a timestep is assigned to each pixel. The regions of the image that are significantly associated to some tokens of the prompt are thus scheduled according to a specific (decreasing concave) function. The approach is compared to recent baseline models on four prompt datasets, reporting four alignment metrics.

### Strengths
* the initial motivation -- the fact that it might be worthwhile to denoise differently various regions of an image -- is intuitive and makes sense. It is clearly explained both in the introduction (section 1) and the method itself (section 3).

* the quantitative and qualitative results are reported with three different backbones, including the recent SD 3.5 (in the appendix). The proposed approach is compared to four recent baselines, published in 2024 or 2025 in top-tier venues (CVPR, NeurIPS, ICLR).

* the quantitative results are computed with a significant number of 1280 images per prompt set, with the same random seed for all models.

### Weaknesses
* the paper ignores an important part of the literature relating to generative semantic nursing. Following Attend-and-Excite [d] several papers investigated at optimizing alignment (cross attention) between the prompt and the noise during the backward process e.g Divide and Bind [g] or Syngen [h]. Similarly to the proposed work, these works showed that the "regions" of the image are indeed decoded at various timestep, but the conclusion was rather than the diffusion models first reconstruct the high-power, low-frequency components at early denoising stages before adding low-power, high-frequency details at later stages [i,j]. Positioning with regards to these works would thus have been relevant.

* The quantitative results are not convincing
  - the authors do not adopt previous protocol. In particular for GenEval, they use the same 553 prompt but do not use the metrics of the GenEval benchmark itself, making hard to compare to previous published results. For Drawbench, the metrics are not the same as e.g (Bai LiChen et al, 2025), making hard to compare directly with previous published results
  - the used BERTscore relies on image description obtained with an ad-hoc model, Qwen2.5-VL-7B-Instruct in this paper. The resulting scores thus evaluates both the models tested and the model used to generate the "ground truth". Similar remarks applies to the QwenScore. However, the two other metrics reflects also alignment (see below)
  - all the metrics deal with text-to-image alignment, ignoring other aspects of image generation. One can understand that this aspect is important to evaluate for this paper, but it should have been asserted that, for example, the image quality is -- at least -- maintained.
  - the quantitative results are reported without any standard deviation, while the quality of generated images (both for aesthetic and alignment) is known to be variable w.r.t the seed. Given the limited difference in performances in comparison to baselines (in particular ofor the most recent backbones in Table 4 and Table 5), one can have doubt regarding the significance.
    - by the way, it is hard to understand why the results in the main paper are based on a old model (SD 2.1) while results on more recent SDXL

* the human study is poorly described
  - there no detail on the 22 participants: are they diversified in gender and age ? Is a majority of them student of the same university as the authors? Or even colleagues? Or the author themselves? Is there some diversity in terms of native language? if so, how the prompt were presented (in English or in their native language) ?
  - it is not clear how many triplet were shown to the participants, nor how these triplets were chosen. For the automatic evaluation of Table 1 it is said that $4\times 1280$ images are considered for each of the prompt sets. Just after, on line 376, it is reported that the participant evaluate "for each group of three candidate", suggesting that they evaluate 5120 triplets. One can doubt that participant actually made as many evaluation.
  - there is no inter-annotator agreement, making hard to estimate the relevance of the study
  - for good practice on human studies, one can refer to [f]

* the qualitative results in Figure 4 are poor for the baselines, but it seems to be mainly due to the old SD 2.1 backbone used. If one uses the [online inference available for SDXL on huggingface](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) -- while it is itself quite old since released in July 2023 -- it is quite easy to get much better results for the base DM than those reported in Figure 4 (with the same prompts).
  - in the Appendix, on Figure 10 and 11, the qualitative results for SDXL and SD 3.5 looks often better for the baselines than the proposed model.

* minor
  - the definition of $x_i\in\mathbb{R}^{n_c\times h\times w}$ on line 194 should be introduced earlier, before equation (4) around line 184. It is indeed a crucial change for the proposed method since it reflects the *pixel-level* aspect.
  - the references for "text-to-image misalignment" on lines 057-058 are recent but inappropriate since this phenomenon has been identified well before for diffusion models, e.g in DALL-e [a] released as a preprint, Imagen (Saharia et al, 2022), DAAM [b], Structured Guidance [c] and Attend&Excite [d]. Not to mention previous works with GANs e.g SOA [e]. The reference (Liu et al, 2025) may be relevant since it is a review, although it seems to be only a preprint and not (yet?) published. However (Hu et al, 2025a) is just a recent paper dealing with a problem already known.
  - which "SD 2.1 base" (line 300) is used ? SD 2.1-512 or SD 2.1-768? 

[a] A. Ramesh et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents". In: arXiv 2204.06125 (2022). arXiv: 2204.0612

[b] R. Tang et al. "What the DAAM: Interpreting Stable Diffusion Using Cross Attention". ACL 2023.

[c] W. Feng et al. "Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis". ICLR 2023.

[d] H. Chefer et al. "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models". In: ACM Trans. Graph. 42.4 (July 2023)

[e] Hinz et al "Semantic Object Accuracy for Generative Text-to-Image Synthesis", TPAMI 2022 (and arXiv:1910.13321 in 2019)

[f] M. Otani et al. "Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation". CVPR, 2023 

[g] Li et al "Divide & Bind Your Attention for Improved Generative Semantic Nursing", BMVC 2023

[h] Rassin et al "Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment" ,NeurIPS 2023

[i] S. Rissanen, M. Heinonen, and A. Solin. “Generative modelling with inverse heat dissipation”. ICLR 2023

[j] Y.-H. Park et al. “Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry”. NeurIPS 2023

### Questions
- which "SD 2.1 base" (line 300) is used ? SD 2.1-512 or SD 2.1-768? 
- why the results on SDXL and SD 3.5  have not been reported in the main paper (and those with SD 2.1 in appendix) ?
- Could we have more details about the study conducted with humans, in particular on the cohort of 22 participants (see above)?

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
This paper proposes a novel method that utilizes masks (getting from network attention or using fixed masks) to decompose the diffusion time of every pixels in an image, resulting clearer inter-pixel context and significant improvement.

### Strengths
**S1: Very good innovation.** I believe asynchronous diffusion can bring a lot of inspiration to subsequent work.

**S2: Significant improvement.** Figure 4 (I was looking forward to discovering more in the supplementary materials, but I couldn't find them) and human survey show extremely good improvements.

### Weaknesses
**W1: Claim issue**. In intro., the authors claim that the text-to-image misalignment is caused primarily by synchronous denoising. However, in many other single-step generative models, including GANs and VAEs, the misalignment also is a key problem. Thus, I think this statement lacks strong support. In other words, I believe that the proposed asynchronous denoising method can alleviate this misalignment problem to some extent, but this misalignment may not necessarily be caused by this synchronous denoising. Therefore, I believe that this statement needs to be revised and the author needs to provide a broader discussion on other methods to address this issue (including discussing existing methods [1] [2] [3] in other generative models to address this issue and other potential solutions).

[1] Liao W, Hu K, Yang M Y, et al. Text to image generation with semantic-spatial aware gan[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 18187-18196.

[2] Zhang C, Peng Y. Stacking VAE and GAN for context-aware text-to-image generation[C]//2018 IEEE Fourth International Conference on Multimedia Big Data (BigMM). IEEE, 2018: 1-5.

[3] Wang H, Lin G, Hoi S C H, et al. Cycle-consistent inverse GAN for text-to-image synthesis[C]//Proceedings of the 29th ACM international conference on multimedia. 2021: 630-638.

### Questions
Q1:
This proposed method uses different time steps for different pixels. However, current research on diffusion focuses on reducing the time step of diffusion. Therefore, when this method is applied to models with small-time diffusion, e.g., T=4 in DDGAN [4], can the effect still be significantly improved?

Q2:
This proposed method can be further combined with patchDiff [5] to improve the generation effect?

Q3:
From the Fig.4, we can find there are clear improvement between the proposed methods and other methods. However, the quantitative performance improvement in Table1 are not obvious. I believe this is a question worth explaining.

[4] Xiao Z, Kreis K, Vahdat A. Tackling the Generative Learning Trilemma with Denoising Diffusion GANs[C]//International Conference on Learning Representations.

[5] Wang Z, Jiang Y, Zheng H, et al. Patch diffusion: Faster and more data-efficient training of diffusion models[J]. Advances in neural information processing systems, 2023, 36: 72137-72154.

### Soundness
3

### Presentation
3

### Contribution
3
