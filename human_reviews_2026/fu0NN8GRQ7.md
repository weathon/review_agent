# VAE-CycleGAN: Variational Latent Representation for Unpaired Image-to-Image Translation

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 2, 4, 2

## Abstract
Image-to-image translation plays a central role in computer vision, enabling applications such as style transfer, domain adaptation, and image enhancement. While recent advances have achieved strong paired translation results, learning mappings in unpaired settings remains challenging. In this work, we present a systematic comparison of autoencoder and variational autoencoder (VAE) variants for unpaired image-to-image translation, using paired data solely as a reference baseline. To capture distributional uncertainty, we introduce VAE-CycleGAN, a unified probabilistic framework that integrates variational inference into the CycleGAN architecture. Our method combines adversarial training and cycle-consistency with a VAE’s probabilistic latent space, allowing the model to approximate the true posterior distribution. Further, the architecture achieves a 256$\times$ spatial compression, efficiently compressing the input into a compact latent representation. Empirical results across the satellite-to-map benchmark dataset demonstrate that VAE-CycleGAN generates high-quality translated images (FID: 67.75) and achieves superior reconstruction fidelity (MSE: 0.0010, PSNR: 29.85 dB, SSIM: 0.7873) comparable to state-of-the-art deterministic approaches without hyperparameter tuning. For summer-to-winter and label-to-cityscape datasets, VAE-CycleGAN performs comparably with state-of-the-art UNSB at 1 step, and is far superior to UNIT-DDPM at 1000 steps, while the determistic AE-CycleGAN is comparable to the 5-step USNB variant.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper conducts a comparison of variational and adversarial approaches to perform image translation via cycle consistency. In addition, VAE-CycleGAN is proposed, which is a hybrid VAE-GAN model that combines variational and adversarial objectives for unpaired image translation.

### Strengths
* It is refreshing to see a GAN/VAE paper in the era of diffusion models! The paper offers a different perspective on variational cycle consistency.
* Theoretical justification.

### Weaknesses
* Introduction is very short and is not entirely convincing and motivating regarding the problem the paper tackles. It also misses a lot of recent work on generative modeling and ignores the diffusion models literature.
* Figure 1: should provide a short description of the pipeline, not just “VAE-CycleGAN”.
* The qualitative results are not entirely convincing, and the visual fidelity does not seem high.
* Quantitatively, it seems that the proposed VAE-CycleGAN is outperformed by the autoencoder variants, even on generative metrics like FID. In that case, what is the significance of the proposed model? What is its advantage and why would one choose to use it?
* Section 5 is poorly written and hard to understand.
* Only one dataset.
* Overall, the manuscript reads like a “technical project report”. It is not entirely clear what the contribution is.
* Missing related work:

[1] [Jha, Ananya Harsh, et al. "Disentangling factors of variation with cycle-consistent variational auto-encoders." Proceedings of the European conference on computer vision (ECCV). 2018.](https://arxiv.org/abs/1804.10469)

[2] [Kim, Beomsu, et al. "Unpaired Image-to-Image Translation via Neural Schrödinger Bridge." The Twelfth International Conference on Learning Representations.](https://openreview.net/forum?id=uQBW7ELXfO)

[3] [Zhao, Min, et al. "Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations." Advances in Neural Information Processing Systems 35 (2022): 3609-3623.](https://arxiv.org/abs/2207.06635)

[4] [Sasaki, Hiroshi, Chris G. Willcocks, and Toby P. Breckon. "Unit-ddpm: Unpaired image translation with denoising diffusion probabilistic models." arXiv preprint arXiv:2104.05358 (2021).](https://arxiv.org/abs/2104.05358)

### Questions
* What is the role of $\mathcal{L}_{\text{identity}}$? It seems very counter-intuitive. Has it been ablated?
* Stability: how stable is the training of the model? How prone is it to mode collapse?
* Open-source code and reproducibility: is it the authors’ intention to publish an open-source code?
* What is the role of this kind of model compared to recent diffusion models? Why would one choose this type of model instead of a diffusion model? I would like the authors to clarify their contribution with regard to the missing related work I mentioned above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposed VAE-CycleGAN for unpaired image-to-image translation. It combines VAE and cyclegan architecture to estimate posterior distribution for unpaired image-to-image translation task, which is a classical task for image generation. It evaluates various Autoencoder and VAE variants on satellite-to-image task. No diffusion-based methods are evaluated and compared.

### Strengths
- The paper provides a systematic and comprehensive comparison of different model variants (AE, VAE, AE-GAN, VAE-GAN, VAE-CycleGAN, and so on).

- Empirical results across the satellite-to-map benchmark show that VAE-CycleGAN generates high-quality translated images.

### Weaknesses
- missing comparison with SOTA. as diffusion models are current mainstream generative models and show unprecedent performance on image-to-image translation, why not compare with some diffusion-bsed methods? Are there any advantages over diffusion-based methods, such as Flux-Context and Qwen-ImageEdit? Motivations are needed to be further clarified.


- Insufficient evalution. this work proposeds an unpaired image-to-image translation method, but only perform evaluation on satellite to image task. I would suggest evaluating on more tasks (such as those in cyclegan paper) to justify the effectiveness of proposed method. The current experimental settings are not sufficicient to support the claim for unpaired image-to-image translation.


- writing can be improved. for example, the introduciton is too short, making it hard for readers to quickly understand. More insights and analysis are prefered in method section.

- As training process of GAN is usually unstable, how important of hyperparameter tuning in the proposed method?

### Questions
see above.

### Soundness
2

### Presentation
2

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
This paper considers the problem that existing deterministic models (such as CycleGAN) cannot handle ill-posed or multimodal tasks, and developed VAE-CycleGAN.

### Strengths
1. Integration of VAE into cycleGAN is technically solid;
2. Several analyses and visualization results are provided;
3. This paper conducts a comprehensive ablation study on 10 different AE and VAE variants.

### Weaknesses
1. Following the related work in the paper, integration of VAE with GAN helps to improve the generation quality. As a result, it might be straightforwad idea to introduce VAE into CycleGAN for improvement, which is the main technical contribution of the work. Therefore, the contribution and novelty of the work might not be sufficient for acceptance.
2. The experiments are performed on a single dataset (satellite to map), while most works also evaluate translation capacities on other datasets or tasks. Thus, the experimental evaluaiton is not sufficient.
3. The comparison methods are limited. For another, it seems that the proposed method does not achieve the superior performance on all the metrices, could the authors provide the corresponding explanation?
4. The organization of the work can be further improved. The Introduciton does not analyze the motivations of the work in detail.
5. Some statements are not clear,

### Questions
Refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper combines the VAE, GAN and cycle consistency ideas to create a VAE-Cycle-GAN. As such, the ideas are similar to MUNIT/UNIT, and various other works that featured thereafter (e.g. VQ-VAE). 

The general idea is to combine the best qualities of the VAE (latent space meaningfulness) with GANs (crisp samples), together with cycle consistency (for unpaired data). In that sense the paper succeeds in showing the effectiveness of the model. 

Results are shown for standard metrics (e.g. LPIPS, FID, SSIM) to show performance on standard datasets (CityScapes and Horse-Zebra/Monet, etc.) for this type of work.

### Strengths
In nearly every setting, the work shows improved results over AE/VAE/GAN/CycleGAN settings. The model is able to show crispness attributable to a GAN and keeps the latent space qualities of the VAE. 

The narrative generally makes sense.

### Weaknesses
- It is very unfortunate that I say this, but the work reads more like a recipe than an advancement. That is to say, novelty is quite limited. It might have been different if it were 2017-18. 
- I am generally in agreement with the points in the paper otherwise.

### Questions
Have the authors considered attempting this on more modern architectures (e.g. diffusion models)?

### Soundness
3

### Presentation
3

### Contribution
2
