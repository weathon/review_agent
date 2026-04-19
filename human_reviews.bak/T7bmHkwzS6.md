# You Only Sample Once: Taming One-Step Text-to-Image Synthesis by Self-Cooperative Diffusion GANs

- Decision: Accept (Poster)
- Scores: 5, 6, 5, 6, 6

## Abstract
Recently, some works have tried to combine diffusion and Generative Adversarial Networks (GANs) to alleviate the computational cost of the iterative denoising inference in Diffusion Models (DMs). 
However, existing works in this line suffer from either training instability and mode collapse or subpar one-step generation learning efficiency. 
To address these issues, we introduce YOSO, a novel generative model designed for rapid, scalable, and high-fidelity one-step image synthesis with high training stability and mode coverage. 
Specifically, we smooth the adversarial divergence by the denoising generator itself, performing self-cooperative learning. We show that our method can serve as a one-step generation model training from scratch with competitive performance. 
Moreover, we extend our YOSO to one-step text-to-image generation based on pre-trained models by several effective training techniques (i.e., latent perceptual loss and latent discriminator for efficient training along with the latent DMs; the informative prior initialization (IPI), and the quick adaption stage for fixing the flawed noise scheduler). Experimental results show that YOSO achieves the state-of-the-art one-step generation performance even with Low-Rank Adaptation (LoRA) fine-tuning.
In particular, we show that the YOSO-PixArt-$\alpha$ can generate images in one step trained on 512 resolution, with the capability of adapting to 1024 resolution without extra explicit training, requiring only \textasciitilde10 A800 days for fine-tuning. Our code is available at:  [https://github.com/Luo-Yihong/YOSO](https://github.com/Luo-Yihong/YOSO)

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a novel one-step generative model named YOSO, which offers higher training stability and model coverage compared to existing methods. This approach aims to achieve high-quality generation while maintaining performance metrics in a single step. Experiments are conducted on CIFAR-10 using HPS and AeS evaluation metrics.

### Strengths
1. Applying the Latent Discriminator leads to a stable training process with fast convergence.
2. The paper tackles a relevant application (i.e., ControlNet) with one-step.

### Weaknesses
1. As mentioned in the paper, "It is hard to extend GANs on large-scale datasets due to training challenges," yet there are existing works that successfully train GANs with large-scale datasets (e.g., GigaGAN with 2.7 billion training image-prompt pairs, Diffusion2GAN with 12 million training image-prompt pairs). It is important to compare these methods. While GigaGAN and Diffusion2GAN do not provide source code, the authors can still conduct comparisons using the COCO 2014 dataset, as GigaGAN provides inference results for this dataset.

GigaGAN: Scaling up GANs for Text-to-Image Synthesis (CVPR 2023)

Diffusion2GAN: Distilling Diffusion Models into Conditional GANs.

2. Why use the bottleneck layer of the UNet to compute the latent perceptual loss? Some works (e.g., DIFT, PnP, FasterDiffusion) suggest that encoder features are more important than both the encoder and bottleneck features.

DIFT: Emergent Correspondence from Image Diffusion (NeurIPS'23)

PnP: Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation (CVPR 2023)

FasterDiffusion: Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference (NeurIPS'24)

3. It’s better to provide experiments comparing with exiting SOTA one-step generative methods (e.g., SwiftBrush) to prove that YOSO enhances the inference efficiency and high-quality generation.

SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation (CVPR'24)

4. The effectiveness of IPI should be validated on datasets like COCO 2014 using FID and CLIPscore to avoid cherry-picking. FID and CLIPscore is wildy used in one-step generation methods like SD-Turbo, InstaFlow, SwiftBrush.

### Questions
1. The authors did not provide information about the dataset used for the test metrics in Table 2.  Although coco2014 is mentioned in the appendix, I am not certain if it is the dataset used for these metrics.
2. The ablation study is unconvincing, as the authors used CIFAR-10 for validation. However, CIFAR-10 has a low resolution and only consists of 10 simple categories, lacking the richness of text prompts. It would be better to conduct the ablation study on the COCO 2014 and COCO 2017 datasets, similar to existing one-step text-to-image generative methods (e.g., InstaFlow, SD-Turbo, SwiftBrush).

### Soundness
2

### Presentation
3

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
This paper, "You Only Sample Once: Taming One-Step Text-to-Image Synthesis by Self-Cooperative Diffusion GANs (YOSO)", proposes a novel generative method that merges diffusion models (DMs) with GANs to perform efficient, high-quality one-step text-to-image synthesis. YOSO leverages a self-cooperative learning framework, where a generator smooths adversarial divergence by referencing self-generated denoised samples, combined with a latent perceptual loss and a decoupled scheduler to stabilize training. The paper positions YOSO as an efficient alternative to multi-step latent diffusion methods (LDMs) with discriminator-based losses, aiming to achieve comparable quality while reducing computational overhead. Experimental results demonstrate competitive performance with state-of-the-art methods, though the model would benefit from clearer differentiation from similar single-step latent diffusion approaches.

### Strengths
1. Creative Approach to Stability in One-Step Synthesis: YOSO’s self-cooperative learning framework is an inventive attempt to stabilize adversarial training by smoothing divergence via self-generated data, a strategy distinct from standard GAN-DM hybrids. This approach may address some instability challenges faced by prior one-step latent diffusion models.
2. Efficient Resource Utilization: YOSO’s compatibility with LoRA allows it to produce high-resolution images without the significant computational burden typically associated with GAN-based models, making it adaptable for resource-constrained settings.

### Weaknesses
1. Clarity of Key Contribution: This paper lacks clarity on whether YOSO is a new model architecture that can be trained from scratch or a step distillation methodology that should be applied on pre-trained models. Further elaboration, possibly with additional diagrams, would make the paper more accessible and clear.

2. Experimental Scope and Robustness: This paper introduces YOSO both as a new model and a novel distillation techniques for pre-trained models. However, the former part was only evaluated on the CIFAR-10 dataset, additional benchmarks are necessary to substantiate its effectiveness across more diverse and complex datasets.

2. Novelty in Methodology: Besides self-cooperative learning, the adoption of decoupled scheduling, perceptual and consistency loss isn't novel to the field, and while the rationale is sound, these sections could benefit from more detailed comparison with previous works and how it is different from them.

### Questions
Please refer to the weakness section

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces YOSO, a novel generative model designed for rapid, scalable, and high-fidelity one-step image synthesis with high training stability and mode coverage. Specifically, the paper smooths the adversarial divergence by the denoising generator itself, performing self-cooperative learning. This method can serve as a one-step generation model training from scratch with competitive performance. Moreover, the authors extend YOSO to one-step text-to-image generation based on pre-trained models by several effective training techniques (i.e., latent perceptual loss and latent discriminator for efficient training along with the latent DMs; the informative prior initialization (IPI), and the quick adaption stage for fixing the flawed noise scheduler). Experimental results show that YOSO achieves state-of-the-art one-step generation performance even with Low-Rank Adaptation (LoRA) fine-tuning.

### Strengths
1. The article is well written and easy to understand.
2. The visualization results look very good.

### Weaknesses
1. There is a lack of important references in the Background, such as [1,2,3], which are all about combining Diffusion models and GANs.  The differences between the proposed method and these methods [1,2,3] need to be emphasized and discussed.
2. In the experimental part, it is also necessary to compare with these methods [1,2,3].
3. There is no comparison with the current SOTA methods in terms of model complexity (such as the number of network parameters and training inference time).
4. In the TEXT-TO-IMAGE GENERATION experiment, there is no comparison with the GAN-based methods, such as [4]. It is also worth discussing whether the proposed method is better or more efficient than the GAN-based methods.

[1]Wang, Zhendong, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. "Diffusion-gan: Training gans with diffusion." arXiv preprint arXiv:2206.02262 (2022).
[2]Kang, Minguk, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. "Distilling Diffusion Models into Conditional GANs." ECCV (2024).
[3]Xu, Yanwu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. "Ufogen: You forward once large scale text-to-image generation via diffusion gans." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8196-8206. 2024.
[4]Kang, Minguk, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. "Scaling up gans for text-to-image synthesis." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10124-10134. 2023.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper combines diffusion models and GANs to propose the YOSO framework for one-step image generation. It employs various training strategies to enhance training stability and has been validated on both unconditional generation and text-to-image tasks.

### Strengths
- In terms of qualitative metrics, YOSO demonstrates certain advantages across various tasks and metrics, and it is capable of transferring to higher resolutions, image editing, and conditional generation tasks.

- The simple and effective training strategies proposed during practical implementation have certain reference significance for generative tasks.

- The paper shows many analyses and easy to follow.

### Weaknesses
- The presentation of the visualization results in this paper is somewhat insufficient, including a comparison with YOSO-PixArt-alpha and other Few-Step models, such as PixArt-delta, and it might be worth attempting to compare with the original models as well.

- The clarity of the narrative in this paper needs improvement; currently, the narrative logic resembles an experimental report. When encountering issues in a new task (unconditional and text-to-image generation / finetuning) or a new model (U-Net or DiT based), a specific strategy is proposed to address the problem. This approach may not be conducive to the coherence expected in a complete paper.

- Regarding writing:
  - The abbreviation "NEF" is not explained.
  - Should the steps for Hyper-SD-LoRA and YOSO-LoRA in Table 2 be equal to 1?
  - Figure 5 does not specify the comparative configuration for model inference—does it keep the prompt unchanged while changing the same seed?
  - The reference formatting in the paper should be kept consistent as possible.

### Questions
- Why were only SD-LoRA and PixArt-Full experiments conducted in the experimental setup, without including SD-Full and PixArt-LoRA related experiments?

- I am curious about how the zero-shot one-step 1024 resolution is effective. As far as I know, 1024 has a different ar_size compared to 512, and the posemb scale is also different. Wouldn't these factors significantly impact the weight distribution of the 1024 model?

- Without retraining, will increasing the sampling steps, similar to LCM, improve the results? If further enhancements are desired for better generated image quality, what would be the subsequent optimization directions based on the proposed approach?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents YOSO (You Only Sample Once), a novel approach for high-quality, one-step text-to-image generation that enhances training stability and efficiency by combining elements from diffusion and GAN models. YOSO’s design includes Self-Cooperative Learning, where less noisy “teacher” samples guide more noisy “student” samples to stabilize training, along with a Decoupled Scheduler that applies separate optimization schedules to Adversarial Loss and Consistency Loss, improving sample quality. Additionally, Informative Prior Initialization (IPI) provides a more data-informed starting point for faster convergence and reduced artifacts, while Quick Adaptation facilitates smooth adjustments to v-prediction and zero terminal SNR, essential for stable, one-step sampling. Unlike prior models that require multiple steps, YOSO achieves single-step synthesis with refined adversarial learning, using teacher-student distributions within the model to mitigate instability and mode collapse risks. This efficient and stable framework positions YOSO as a significant advancement for rapid, high-fidelity text-to-image generation.

### Strengths
1. The proposed self-cooperative technique to train a one-step sampler seems novel in the diffusion literature, and the author proved it to be very effective. The fact that the less noisy (teacher) samples guide more noisy (student) samples during training is quite interesting.

2. It is interesting to see the connection between the self-cooperative objective and consistency models.

3. The experimental results are quite strong. They outperform other one-step baseline models on CIFAR-10 and also show improvement in the large-scale text-to-image setting.

4. The mathematical analysis is clear and robust, which supports each of its contributions. The derivations for Self-Cooperative Learning and the teacher-student dynamic seems well-structured.

### Weaknesses
1. Lack of the ablation study on training techniques. The paper introduces training techniques like Informative Prior Initialization (IPI) and v-prediction, which are valuable training methods that could also benefit other models. However, the lack of an ablation study on these components makes it difficult to assess their specific impact and to directly compare YOSO's performance with other models that could potentially leverage these techniques.

2. Hyperparameter sensitivity. The timestep difference between the teacher and student latents, a key factor in the Decoupled Scheduler, is a hyperparameter that directly impacts training stability and eventual performance. Does it need to be tuned for different datasets? Could you show how robust the method is to this hyperparameter? 

[minor comments]
1. Table 2 caption: text-to-to -> text-to
2. Eq (3): remove ‘x’
3. In Eq (3): why did you use different lambda for different t?

### Questions
Questions are embedded in the weakness section.

### Soundness
4

### Presentation
3

### Contribution
4
