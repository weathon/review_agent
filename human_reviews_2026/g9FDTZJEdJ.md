# Scalable GANs with Transformers

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Scalability has driven recent advances in generative modeling, yet its principles remain underexplored for adversarial learning. We investigate the scalability of Generative Adversarial Networks (GANs) through two design choices that have proven to be effective in other types of generative models: training in a compact Variational Autoencoder latent space and adopting purely transformer-based generators and discriminators. Training in latent space enables efficient computation while preserving perceptual fidelity, and this efficiency pairs naturally with plain transformers, whose performance scales with computational budget.
Building on these choices, we analyze failure modes that emerge when naively scaling GANs. 
Specifically, we find issues as underutilization of early layers in the generator and optimization instability as the network scales. Accordingly, we provide simple and scale-friendly solutions as lightweight intermediate supervision and width-aware learning-rate adjustment. Our experiments show that Generative Adversarial Transformers (GAT), a purely transformer-based and latent-space GANs, can be easily trained reliably across a wide range of capacities (S through XL). Moreover, GAT-XL/2 achieves state-of-the-art single-step, class-conditional generation performance (FID of 2.18) on ImageNet-256 in just 60 epochs, 4× fewer epochs than strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper introduces Generative Adversarial Transformers (GAT) — a fully transformer-based GAN framework trained in a compact VAE latent space, enabling efficient and scalable adversarial training.

- It identifies two major issues when scaling GANs, inactive early generator layers and training instability. The authors solve them with (1) Multi-level Noise-perturbed Guidance (MNG) for early-layer activation and (2) a width-aware adaptive learning rate rule to stabilize training.

- GAT scales reliably from small to extra-large models, achieving state-of-the-art single-step ImageNet-256 generation with an FID of 2.96 in just 40 epochs — about 6× fewer than strong diffusion baselines.

- The work demonstrates that GANs can benefit from the same scalability principles as diffusion and autoregressive transformers, retaining GANs’ strengths (like single-step generation and latent manipulation) while achieving strong performance at large scale.

### Strengths
### 1. Technical Strengths

- Introduces a pure transformer-based GAN framework (GAT) that operates in VAE latent space, combining transformer scalability with GAN efficiency.

- Proposes two simple yet effective techniques: 1) Multi-level Noise-perturbed Guidance (MNG) for activating early generator layers and 2) Adaptive learning rate scaling to stabilize training across different model widths.

- Maintains a consistent transformer structure for both generator and discriminator without requiring architectural changes across scales.

- Incorporates representation alignment (REPA) between the discriminator and pre-trained Vision Foundation Models (DINOv2), enhancing discriminator features and stability.

### 2. Experimental Strengths

- Achieves state-of-the-art single-step image generation on ImageNet-256 with an FID of 2.96 in only 40 epochs, far fewer than diffusion/flow baselines.

- Demonstrates near-monotonic performance improvements (better FID) with increased model capacity, confirming successful scaling behavior.

- Includes detailed ablations for MNG, adaptive LR, and REPA, showing each component’s impact on stability and performance.

- Retains GANs’ key advantage, single-step inference, while outperforming many multi-step models in both speed and data efficiency.

### Weaknesses
### 1. Technical Limitations

- Relies on pre-trained VAE encoders (e.g., Stable Diffusion VAE), which may limit generality or introduce biases from the tokenizer.

- Transformer-based architectures can be memory-intensive, especially at large scales, limiting accessibility for lower-resource setups.

- Uses existing adversarial losses (R3GAN) with modest innovation; potential remains for more advanced loss formulations or self-distillation.

- While MNG alleviates inactive early layers, it adds auxiliary outputs and supervision complexity, which might scale non-linearly in deeper models.

### 2. Experimental Limitations

- Experiments are limited to ImageNet-256, leaving open questions about generalization to higher resolutions or more diverse domains (e.g., text-to-image).

- No perceptual or user-study validation of image quality, relies solely on FID metrics.

- Even though training is more efficient than diffusion models, the largest model (GAT-XL) still requires ~12 days on 8×A6000 GPUs.

- Does not test transferability or robustness of learned latent spaces for tasks like editing or cross-domain synthesis.

### Questions
See the weakness section. In addition:

- How sensitive is the GAT architecture to the choice of VAE latent space? Would performance change significantly with a different latent tokenizer?

- The Multi-level Noise-perturbed Guidance (MNG) adds auxiliary supervision. How does this affect training complexity or gradient flow compared to MSG-GAN (https://arxiv.org/abs/1903.06048)?

- Can you elaborate on how MNG interacts with adversarial training — is the discriminator jointly trained on all perturbed outputs or aggregated losses?

- How does your adaptive learning rate rule compare empirically to other scaling rules (e.g., linear or square-root scaling used in transformers)?

- Since both generator and discriminator are transformers, how do you handle instability due to self-attention depth or gradient explosion at large scales?

- Is the REPA alignment objective necessary for all scales, or does it mainly benefit large-capacity models?

- How much does the VFM alignment loss contribute to computational cost during training?

- Does the model’s performance degrade if the VFM alignment is removed after pretraining (e.g., fine-tuning without REPA)?

- How well does GAT generalize to other data modalities (e.g., faces, artwork, text-conditional generation)?

- You claim scalability properties similar to diffusion and autoregressive models — could you provide a formal scaling law or empirical scaling curve?

- Why does transformer scalability translate better to GANs in the latent space than in pixel space?

- Are there theoretical reasons why early generator layers become inactive without MNG?

- Why did you choose ImageNet-256 as the only benchmark? Have you evaluated GAT on higher resolutions or other datasets (e.g., FFHQ, COCO)?

- Could you provide qualitative examples comparing GAT outputs to StyleGAN-XL or DiT at similar FID levels?

- How consistent are your results across random seeds? what is the variance of FID across runs?

- What is the computational cost comparison (in GPU-days) between GAT-XL/2 and diffusion models with comparable FID?

- Did you observe any mode collapse or overfitting patterns as model capacity increased?

- How much improvement does latent-space training provide over pixel-space training, quantitatively?

- How does GAT compare in memory footprint and inference latency against diffusion models at similar quality?

- Does the use of VAE latent space constrain the diversity of generated samples (e.g., limits on fine texture detail)?

- How transferable are your findings to conditional or text-driven GANs (e.g., T2I models like GigaGAN)?

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
2

### Summary
This works looks to bring in scaling law perspectives for GANs, noting that there is scant literature on showing how to scale GANs to large settings. The authors motivate the need based on the fact that GAN inference is a single shot, as opposed to say diffusion, as well as interpretability in the form of latent interpolation. They use transformer architectures for the the generator and discriminator, and highlight two key observations in scaling: use of intermediate feedback (multi-level noise perturbed guidance) to ensure all layers of the generator contribute, and a scaling rule to adjust hyper parameters to ensure training stability.  

While the proposed methodology is simple and well motivated, I am not particularly convinced by the experiments. For a paper on scaling, the image generated (the largest being 256 x 256) are not substantially large to demonstrate scaling effects. The paper also lacks a theoretical motive/analysis of the multi-level noise perturbed guidance. This is not a major issue, but mean I must focus primarily on the experiments. 

That said, this is slightly outside my area of expertise and happy to engage with the other reviewers on this.

### Strengths
See above.

### Weaknesses
See above.

### Questions
Can you comment on the weakness.

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
3

### Summary
This paper is the first to conduct a systematic study on the scalability of Generative Adversarial Networks (GANs) and proposes a framework named GAT. This framework combines two key design choices: training in the compact latent space of a VAE and using a pure Transformer architecture for both the generator and discriminator. To address issues such as underutilization of early layers and optimization instability that arise when scaling up the model, the authors propose two simple and effective solutions: a mechanism that provides lightweight supervision to the intermediate layers of the generator through multi-level noise perturbation, and a rule for adaptively adjusting the learning rate based on model width. Experiments show that GAT can be trained stably across a wide range of model capacities (from S to XL). Its largest model, GAT-XL/2, achieves state-of-the-art performance (FID 2.96) in class-conditional generation on ImageNet-256 after only 40 training epochs, demonstrating a 6x improvement in training efficiency compared to previous strong baselines.

### Strengths
+ This is the first work to systematically scale GANs across various Transformer model capacities and demonstrate stable performance improvement with increasing scale, laying the foundation for research into GAN scalability.

+ It effectively addresses the core challenges of scaling GANs through straightforward techniques (such as intermediate layer supervision and adaptive learning rates). The methods are simple and easy to implement, offering high practical value.

+ It achieves highly competitive performance in single-step generation tasks while retaining the inherent advantages of GANs, such as fast inference and latent space manipulability, proving the effectiveness of the framework.

### Weaknesses
- Despite training in latent space, the computational cost for the largest model remains very high, requiring 8 A6000 GPUs for 12 days, which limits its reproducibility and accessibility for further exploration.

- The experimental scope is relatively limited, focusing only on class-conditional generation on the ImageNet-256 dataset. It lacks validation on higher resolutions, cross-dataset generalization, or other conditional modalities (such as text-to-image generation).

- Although claimed to be a "pure Transformer" architecture, the generator incorporates style modulation techniques inspired by StyleGAN, which somewhat deviates from the original intention of "preserving the plain Transformer design."

### Questions
- Why does scaling the discriminator lead to greater performance improvements than scaling the generator? Does this imply that the key to enhancing GAN performance in the future lies in strengthening the discriminator?

- Can the adaptive learning rate rule proposed in the paper be further extended to form more comprehensive scaling laws for GANs, such as simultaneously considering other dimensions like model depth and batch size?

- Beyond class-conditional image generation, how can the GAT framework be applied to broader generative tasks, such as text-to-image generation? What new scalability challenges might arise in these tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a scalable GAN framework that includes architectural changes, loss function choices, learning in the latent space, and scaling laws that enable stable GAN training. Unlike other generative models like Diffusion models, GANs enable single step generation and semantic latent spaces due to a compact noise dimension. The authors present positive results with these modifications and acheive SoTA in FID for ImageNet 256x256.

### Strengths
1. The paper addresses fundamental challenges in scaling up GAN training that actually works and achieves SoTA FID scores on ImageNet 256x256. This is a significant contribution and strength of the paper.
2. The authors' present thorough ablations and the narrative of the paper builds from the challenges to the final SoTA results while focusing on scalability of various components, including, architecture, loss functions, auxiliary losses, etc.
3. The presented model GAT-XL/2 achieves a SoTA FID score on class-conditional generation on ImageNet 256x256
4. The motivation for reviving GANs over SoTA Diffusion Models is made quite clear in Section 2.1, helping readability.

### Weaknesses
Minor:
1. Please use abbreviations only after they have been expanded at least once before. E.g. "GAT" in the Abstract.
2. On Line 137, the quote `Layerscale` -- can you please provide citations or a definition of what is meant by Layerscale? Is it referring to the scale factor in the BN layer - if so, calling it the `scaling factor in BN layer` would be much more clarifying.
3. It is unclear to the reader how the VAE fits in to this scalable GAN framework since the "VAE Latent" word is used from the very beginning of the paper. Only later does it clear up that a pretrained VAE model is being used to apply GAN in its latent feature space. Other than that, there is not any connection conceptually to VAE/Variational Bounds, etc. Please clarify this early on to improve readability. Thank you.
4. Please include SD-VAE's performance in Table 1 too so it is clear how much of the gains are due to the latent VAE representation.
5. Line 376, please cite LPIPS.
6. Under Related Works, for GANs various works like SNGAN, WGAN, WGAN-GP have been cited, that are similar. I suggest citing a family of models that use Discontinuous Functions in their Discriminator (GraN-GAN and Gradient Normalization) by Wu et al. and Bhaskara et al. (See Papers [1] and [2] below).


Major:
1. On Line 86, the authors claim that the larger the network, the rapid are the changes in the outputs. Please clarify this statement and back it up with citations or empirical / toy experiment with plots. What are the "outputs" here? Do they refer to the discriminator's logits or the generators image tokens? How to measure "speed" of change of the outputs?
2.  One Line 152, the authors say, "the early layers of the generator remain largely inactive." The reader is instantly led to question where the evidence for this is. So please cite your ablation study in Fig 4a even though it is still early in the paper -- than leading the reader to wrongly believe early on in the paper that it does not present enough evidence to claims.
3. Fig 4a does not seem sufficient to argue that early layers are not active in the generator to motivate for the layer-wise aux losses. Can a quantitative plot be presented. Example: Average abs rate of change of weights per layer over the course of training and showing that the early layers don't change much unless the MNG aux losses are used. Something quantitative would be really helpful. For instance, along the lines of Figure 3 in Bhaskara et al. [1].
4. Equation 4 is a hueristic for scaling the learning rate. Please back this up with an empirical quantitative plot. In the ablation (Sec 3.3) only qualitative arguments are given. Please show numbers to back up the claim for this scaling law. Also clarify if this scaling law applies to both Generator and Discriminator or only one of them? Also please include the initial LRs used for D and G for your SoTA model.  


## References:
===  
[1] Bhaskara et. al., GraN-GAN: Piecewise Gradient Normalization for Generative Adversarial Networks (WACV 2022) - https://openaccess.thecvf.com/content/WACV2022/html/Bhaskara_GraN-GAN_Piecewise_Gradient_Normalization_for_Generative_Adversarial_Networks_WACV_2022_paper.html. 

[2] Wu et al., Gradient Normalization for Generative Adversarial Networks (ICCV 2021) - https://ieeexplore.ieee.org/document/9710427.

### Questions
In addition to the weaknesses above, I have the following additional questions:

1. Why was the pretrained VAE model used to extract the latent features for learning? Why not VGG/ResNet hidden features or BigGAN's hidden latent features? This is not clear and a clarification would greatly improve the cohesiveness of the paper
2. What is the module that remaps the generated latents back into the image space? This was not clear. Please describe this decoder mapping architecture.
3. In Figure 2, the list [x,x,x,x,] is labeled as copies of real images. This is confusing. Did you mean copies of the VAE latents of the real images instead? Please clarify.
4. It is not clear why the authors do not have any experiments on Unconditional Generation. Why is this? Is this because it is more challenging and the authors could not make it work (please share any negative results, as this strengthens the paper).

I am inclined to update my score given the authors can clarify and address some of the points raised. Thank you.

### Soundness
4

### Presentation
4

### Contribution
3
