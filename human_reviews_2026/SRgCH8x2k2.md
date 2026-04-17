# H3AE: High Compression, High Speed, and High Quality AutoEncoder for Video Diffusion Models

- Decision: Reject
- Scores: 2, 4, 2, 8, 8

## Abstract
Autoencoder (AE) is the key to the success of latent diffusion models for image and video generation, reducing the denoising resolution and improving efficiency. However, the power of AE has long been underexplored in terms of network design, compression ratio, and training strategy. In this work, we systematically examine the architecture design choices and optimize the computation distribution to obtain a series of efficient and high-compression video AEs that can decode in real time on mobile devices. We also unify the design of plain Autoencoder and image-conditioned I2V VAE, achieving multifunctionality in a single network. In addition, we find that the widely adopted discriminative losses, i.e., GAN, LPIPS, and DWT losses, provide no significant improvements when training AEs at scale. We propose a novel latent consistency loss that does not require complicated discriminator design or hyperparameter tuning, but provides stable improvements in reconstruction quality. Our AE achieves an ultra-high compression ratio and real-time decoding speed on mobile while outperforming prior art in terms of reconstruction metrics by a large margin. We finally validate our AE by training a DiT on its latent space and demonstrate fast, high-quality text-to-video generation capability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines the design of autoencoder architectures and proposes an Omni-AE training strategy to improve efficient video generation. A latent consistency loss function is designed to enhance the perceptual and pixel fidelity of the reconstructed results. The proposed method achieves ultra-high compression ratios and real-time decoding speeds on GPUs and mobile devices, surpassing other competing methods by a large margin.

### Strengths
The proposed method aims to enable real-time I2V and T2V on GPUs and mobile devices. The manuscript is well-organized.

### Weaknesses
[1].H3AE should be explained in line 24 of the manuscript because it appears for the first time in the paper.
[2].The paper lacks the details of H3AE architecture. For example, the kernel size used in the convolution. How to downsamle the features?
[3].In line 184, what is RoPE and QK-norm? Please explain this and cite the related papers.
[4].In line 191, what is 4ch? This content is unreadable.
[5].In Eq.(1), how to set the value of p during the training phase?
[6].How to use the hierarchical features of the first frame? Please explain this. How about using cross-attention or other ways can improve the performance?
[7].In Fig. 3, the visual comparison in the last column is not obvious. Please change this.

### Questions
See the part of weaknesses.

### Soundness
2

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
5

### Summary
This paper proposes a series of video auto-encoders with high spatial compression and with real-time mobile decoding capability.
It combines architectural engineering choices such as 2D spatial convs at highest resolution and 3D causal attention. It also proposes an "omni-objective" training scheme that allows the VAE to be used as plain as well as for I2V tasks. Finally it also proposes a latent-consistency loss to improve quality.  Along with quantitative and qualitative results for VAE, authors also train a text-to-video DiT model to showcase the latent space capability of the trained VAEs.

### Strengths
This paper is a very well executed engineering effort that proposes a new Video VAE architecture and improvement to the training pipeline. This is specifically also tailored towards real-time decoding on mobile devices.

1. The paper proposes a modified architecture design for practical VAEs.  The VAE's have significantly improved performance, particularly the one with compression ratio of $4 \times 16 \times 16$. 

2. The idea of training a single VAE that can be used both as a plain VAE and for I2V scenario is pretty neat.

3. The architecture ablation study is well-detailed and the final architecture proposed seems to be quite efficient on hardware with a high FPS (Table-1).

### Weaknesses
1. The idea of conditioning decoder on image information has been explored before. Authors propose to train a VAE that can be used for both T2V and I2V tasks, However, there are no experiments that verify the usefulness of the VAE for the I2V denoiser scenario.

2. The quality of VAE latents is mainly measured with quantitative and qualitative samples. However, since this is a Video VAE, there is not much discussion / analysis about the temporal consistency and any flickering artifacts that may arise from such extreme compression. 

3. A user study that compares the various compression ratios is also missing.

4. Comparision of SoTA models in Table-1 seems unfair since the compression ratios are quite different. It makes it difficult to really judge if techniques proposed in this paper really are much more beneficial than the other methods.

### Questions
1. Authors have provided ablation studies for each improvement in Tab. 2, 4, 5 . However, could the authors provide a clear explanation of which component (architecture improvements vs latent consistency vs omni-AE training ) provided the most gains in terms of performance (PSNR)? Is the performance of each component similar across different compression ratios?

2. SoTA comparisions with efficient decoders such as [Turbo-VAED Zou et. al](https://arxiv.org/pdf/2508.09136) are missing. 

3. The latent consistency loss is posed as a novel contribution. However, it bears a close resemblance to regularization applied in the paper [Consistency Regularisation for Variational Auto-encoders Sinha et. al](https://arxiv.org/pdf/2105.14859). Could the authors provide more clarity to this? It seems like a common regularisation technique being used in classifiers.

4. Did the authors also attempt high temporal compression along with spatial compression? Did they observe any significant performance changes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents H3AE, an efficient and high-compression video autoencoder. The authors systematically explore VAE architectural components and introduce a LC Loss and Omni-objective training strategy, achieving strong improvements in reconstruction quality.

### Strengths
1. The paper provides a systematic study of video VAE design choices e.g., attention and normalization.
2, H3AE achieves notable improvements in recosntruction over strong baselines.

### Weaknesses
1. This paper is an engineering work. The main contributions stem from combining existing components, e.g., 2D spatial convolutions in high-resolution stages, 3D causal attention, and increased latent channels—without introducing new conceptual insights.

2. The paper does not provide explanations for key findings, remaining at an empirical reporting level. For instance, it does not clarify why the proposed omni-objective training yields “free lunch” improvements instead of task conflict between t2v and I2V objectives.

3. While reconstruction metrics improve, generation results (e.g., Fig. 6) show little gain over baselines, suggesting limited impact on generative capability. Reconstruction and generation quality are not highly correlated; for generative tasks, improvements in generation metrics would be more meaningful.

4. It remains unexplained why the latent consistency loss is formulated as a KL divergence rather than a simpler distance such as MSE. Additional ablation studies are needed to justify this choice.

5. The code is not released, which reduces transparency and reproducibility for a work that is primarily engineering-oriented.

### Questions
See Weaknesses. Overall, I believe this paper presents valuable engineering exploration. However, it lacks sufficient justification for its methodological design, clear explanations for key findings, and meaningful improvements in generative metrics.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces H3AE, a highly efficient and expressive autoencoder for video diffusion models that supports both text-to-video and image-to-video generation. The authors show that by distributing 2D and 3D computation in video VAEs, one can design architectures that are more efficient while maintaining reconstruction quality under stronger compression. The paper also studies training techniques for video VAEs and suggests several improvements that further stabilize training and improve both efficiency and quality. Finally, the proposed VAE is ablated against several strong baselines, and the authors demonstrate that competitive video diffusion models can be trained in the latent space of H3AE.

### Strengths
- As video diffusion models become increasingly common, improving their efficiency to enable real-time, on-device applications is an important direction. This paper helps bridge the gap between the efficiency of current video VAEs and the requirements of real-time processing on smaller devices.

- The idea of distributing 2D and 3D computation based on resolution is novel and interesting.

- The modified training techniques, such as the use of consistency loss and image conditioning, are both interesting and valuable to the broader community.

- The experiments are well designed and clearly show the improvements and contributions presented in the paper.

- The paper is well written and easy to follow.

### Weaknesses
- The paper would benefit from a more detailed explanation of why the proposed consistency loss can replace the GAN loss. From my experience, the GAN loss is typically essential for achieving sharp results. This finding is both interesting and surprising, and therefore might need further discussion.

- In Equation (1), it is unclear what the product operation represents. A brief explanation or notation clarification would improve readability.

- The paper contains a few minor typos that should be corrected in the final version.

### Questions
- It would be helpful to explain why the FVD score increases when using the GAN loss in Table 5, and how the proposed approach manages to avoid blurry outputs without it.

- In Equation (3), it appears that the model assumes the latent variables follow a Gaussian distribution. It would be helpful if the authors could clarify the motivation or theoretical justification for this assumption. Since the KL divergence weight is often set to a small value in practice, the latent distribution may deviate from a true Gaussian, so a brief discussion on how this affects the validity of the loss formulation would strengthen the paper.


- How does the model scale when latency is not a limiting factor? For instance, could larger architectures or higher-resolution settings further improve performance under relaxed efficiency constraints?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce several improvements to VAE training and architecture that allow them to reach a new Pareto frontier on the reconstruction quality / compression ratio tradeoff. These improvements include:

- a new, computationally efficient architecture
- training the decoder to reconstruct the video both with and without being shown the first frame. They report that this improves decoder performance even when the first frame isn't shown.
- a "latent consistency loss" imposed on the decoder. While reconstruction loss encourages D(E(x)) to be close to x, latent consistency loss encourages E(D(E(x))) to be close to E(x). This improves reconstruction quality across all their metrics.

### Strengths
Very straightforward technology paper offering very straightforward improvements in VAE design and backing up those improvements with sensible empirical evidence. Well-written and presented.

### Weaknesses
No complaints. This paper is thorough and professional beyond my ability to find criticisms.

### Questions
N/a

### Soundness
3

### Presentation
3

### Contribution
3
