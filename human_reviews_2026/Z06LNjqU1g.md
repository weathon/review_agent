# Scale-wise Distillation of Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Recent diffusion distillation methods have achieved remarkable progress, enabling high-quality ${\sim}4$-step sampling for large-scale text-conditional image and video diffusion models. 
However, further reducing the number of sampling steps becomes more and more challenging, suggesting that efficiency gains may be better mined along other model axes. 
Motivated by this perspective, we introduce SwD, a scale-wise diffusion distillation framework that equips few-step models with progressive generation, avoiding redundant computations at intermediate diffusion timesteps. 
Beyond efficiency, SwD enriches the family of distribution matching distillation approaches by introducing a simple patch-level distillation objective based on Maximum Mean Discrepancy (MMD). 
This objective significantly improves the convergence of existing distillation methods and performs surprisingly well in isolation, offering a competitive baseline for diffusion distillation.
Applied to state-of-the-art text-to-image/video diffusion models, SwD approaches the sampling speed of two full-resolution steps and largely outperforms alternatives under the same compute budget, as evidenced by automatic metrics and human preference studies. Project page: https://yandex-research.github.io/swd.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Scale-wise Distillation (SwD), a method that distills a diffusion model into a single model capable of progressively increasing resolution while generating images with only a few sampling steps.

The key ideas are:
(i) Spectral analysis, which shows that in the early high-noise stages, high-frequency components are largely suppressed, thus computations can be performed at lower spatial/temporal resolutions to save cost;
(ii) A progressive sampling and training procedure, where at each step the resolution is increased, and the previous step’s
is upsampled → re-noised to serve as the input for the next step;
(iii) A simple distribution matching loss that aligns teacher and student distributions in the DM feature space using Maximum Mean Discrepancy (MMD), particularly a linear-kernel variant (LMMD).

### Strengths
-  **Method motivation**: Through RAPSD-based analysis, the paper convincingly demonstrates why low-resolution processing is safe during high-noise stages (Fig. 1). This motivation is appropriate as a methodological rationale; however, similar motivations have actually been used several times before from the diffusion efficiency literature. See Weakness.

- **Solid experiments**: The approach is also extended to both text and video domains. The comparison of efficiency in Tables 4 and 5 is very appropriate, and the performance comparison is clearly shown in Figure 7. Under the same number of steps, SWD and the full model seem to have comparable performance, but SWD shows better efficiency.

### Weaknesses
- Recently, many diffusion efficiency studies have discussed that it is reasonable to focus on low-frequency components at time steps close to the noise. For example, [1] uses a transformer with a larger patch size at earlier (noise-near) time steps, and [2] proposes a method to better capture low-frequency information at each time step. It would be better if the motivation of this work were discussed in connection with these findings.

- My greatest concern is whether the proposed method can effectively capture high-frequency details. Since the method focuses largely on low-frequency content except at the final step, this potentially poses a limitation. Moreover, because large-resolution images are involved, it may be difficult for the metrics to properly evaluate whether the high-frequency information has been preserved. I strongly recommend incorporating the measurement of FID-Patch as used in the SDXL‑Lightning paper to comprehensively assess fine detail preservation.


[1] (CVPR 25) FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute

[2] (CVPR 25) Autoregressive Distillation of Diffusion Transformers

[3] SDXL-Lightning: Progressive Adversarial Diffusion Distillation

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper advances diffusion distillation by introducing a scale-wise few-step diffusion model. Instead of operating entirely at a fixed resolution, the model begins at a lower scale and progressively increases to the final resolution, similar to standard few-step diffusion approaches. Additionally, the authors propose a Maximum Mean Discrepancy (MMD)-based distillation loss to complement the existing DMD and GAN losses, further improving training effectiveness. Experiments conducted on various architectures—including SD 3.5 and FLUX for image generation (evaluated on COCO2014 and MJHQ) and WAN 2.1 for video generation (evaluated on VBench 2.0)—demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper presents a detailed spectral analysis of the latent spaces across different diffusion models, offering valuable insights that motivate the first contribution—scale-wise distillation.

2. The proposed distillation framework is evaluated on multiple diffusion models, covering both image and video generation tasks. The results demonstrate the potential of scale-wise distillation to achieve higher efficiency while maintaining performance comparable to existing distillation methods.

3. The paper is clearly written and well-organized, with logical section flow and well-explained experimental setups that effectively support its main conclusions.

### Weaknesses
1. The experiment on different upsampling strategies (lines 192–212) lacks evaluations with alternative scale configurations (e.g., from/to 32, 80, or 96), which would provide a more comprehensive and reliable analysis.

2. The paper does not explain why the temporal dimension of SwD does not contribute to performance improvement when applied to Wan 2.1 (lines 375–377).

3. The human preference study charts in Figures 6, 7, and 8 lack sufficient descriptive captions or visual annotations, reducing interpretability.

4. In Table 3, the performance of SwD on 8B- and 12B-scale models shows only marginal gains over other distilled variants—for instance, SD3.5-L-Turbo (0.71 vs. 0.70) and FLUX-Schnell (0.71 vs. 0.69). Moreover, the human preference study (Figure 6) indicates that SD3.5-L-SwD and FLUX-SwD are outperformed by Turbo-L and FLUX-Schnell in most aspects, calling into question the practical advantage of SwD at similar scales.

5. The paper omits comparisons with other few-step video diffusion models, such as Video LCM [1], T2V-Turbo [2], and MagicDistillation [3], which are relevant baselines.

6. There is no discussion of Jenga [4], which shares a similar conceptual idea of adjusting scale with timestep (smaller scales for larger timesteps and larger scales for smaller timesteps).

7. Table 3 lacks a description of the scale schedule corresponding to each timestep, making the setup difficult to reproduce.
 
[1] https://arxiv.org/abs/2312.09109 

[2] https://arxiv.org/abs/2410.05677 

[3] https://arxiv.org/abs/2503.13319 

[4] https://arxiv.org/abs/2505.16864

### Questions
1. Why does the GenEval result for Infinity reported in this paper (0.69 in Table 3) differ from the value reported in the original Infinity paper (0.73)?

2. In Equation (2), how are the two presented kernels applied in practice?

3. Are the models initialized from pretrained weights or trained from scratch?

4. Could the authors provide additional explanation for why the temporal dimension in SwD does not lead to performance improvements in Wan 2.1? (lines 375–377)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the inefficiency of few-step diffusion distillation methods, which redundantly compute all steps at full resolution. The authors propose two main contributions.First, Scale-wise Distillation (SwD), a framework where a single generator progressively increases its operating resolution during the few-step sampling process. This avoids unnecessary computation in early, noisy steps, where high-frequency detail is absent. Second, a simple and effective MMD-based distillation loss ($\mathcal{L}_{MMD}$) that matches student and teacher feature distributions. This loss is shown to be a highly competitive baseline on its own and significantly speeds up training. Applied to SOTA models (SD3.5, FLUX, Wan2.1), SwD provides ~2-3x speedup over full-resolution models at the same step count and achieves significantly higher quality under the same computational budget

### Strengths
1. The core idea of unifying progressive generation with few-step distillation (SwD) is novel and elegant. The motivation is strong, grounded in a solid spectral analysis (Section 3) of VAE latents for both images and video;
2. The paper introduces a simple MMD-based loss that is surprisingly powerful. Ablations (Table 6) show it performs competitively on its own while being remarkably efficient. As it requires no extra trainable models (unlike GAN or DMD losses), it enables >7x faster training iterations (Table 5), making it a highly valuable standalone contribution;
3. The experiments are comprehensive. The key comparison in Section 5.2 (Figure 7, Tables 7-8) clearly demonstrates SwD's superiority: at an equivalent compute cost (e.g., 4-step SwD vs. 2-step Full-res), SwD produces significantly better-quality images with fewer defects . The main results (Table 3, Figure 6) show SOTA performance, even outperforming teacher models in human preference;

### Weaknesses
1. The framework's performance depends on the co-design of the timestep schedule $[t_i]$ and the scale schedule $[s_i]$. While the authors provide the schedules used (Appendix D), the paper offers limited intuition on the methodology for finding these optimal schedules or the model's sensitivity to them.
2. While the $ \mathcal{L} _ {MMD} $ only variant is exceptionally simple, the full $ \mathcal{L} _ {SwD} $ objective used to achieve the absolute SOTA results ($\mathcal{L} _ {MMD}+\mathcal{L} _ {DMD}+\mathcal{L} _ {GAN}$) inherits the training complexity of methods like DMD2 (e.g., training a "fake" DM). However, the paper wisely presents the $\mathcal{L} _ {MMD}$-only path as a highly effective, simpler alternative.

### Questions
see weaknesses

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
5

### Summary
The paper introduces Scale-wise Distillation (SwD), which progressively increases latent resolution during few-step diffusion distillation.
Motivated by spectral analysis showing that high-noise latents mainly contain low-frequency content, the method reduces redundant computation at early steps.
An additional MMD-based distillation loss is proposed and combined with DMD, showing competitive results on SD3.5, FLUX, and Wan2.1 models.

### Strengths
* Clear and well-motivated idea linking noise level and latent frequency spectrum.
* Practical framework that integrates smoothly with existing distillation methods.
* The proposed MMD loss is easy to use, and effective even alone.
* Good results with notable speedups for both text-to-image and text-to-video generation.
* The writing is clear and the figures (e.g., Figure 1 spectral plots) effectively communicate the intuition behind the framework.

### Weaknesses
1. **Limited and possibly unfair baseline comparisons (SDXL case):**  
Table 3 compares SwD results on SD3.5 and FLUX but omits SwD results on SDXL, even though Appendix B (Figure 9) shows that SDXL experiments were performed. For a fair evaluation, the authors should include SDXL-SwD and compare it directly to DMD2-SDXL and SDXL-Turbo. Moreover, other recent open-source distillation baselines such as Hyper-SD [Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis] should be incorporated.
Given that the paper emphasizes the synergy between MMD + DMD losses, fair quantitative evidence across identical model backbones is essential to support the claimed superiority.

2. **Insufficient validation for FLUX and Wan distillations:**  
The comparison for FLUX and Wan 2.1 models is not convincing, as no competing baselines are provided. Stronger open-source baselines such as Hyper-FLUX, CausVid, or LightX2V (all publicly available and Wan-based distillation frameworks) should be included. Without these comparisons, it is difficult to judge the SwD’s efficiency and quality againest other distillation methods.

3. **a minor issue** :  
In Table 3, the number of function evaluations (NFE) or diffusion steps used for each model is unclear, making speed–quality trade-offs hard to assess.


4. **Relationship between MMD loss and prior progressive distillation methods (ADD/LADD) is not well clarified:**  
The MMD loss seems similar with the progressive distillation in ADD/LADD with different training objective. (constrain intermediate latent representations via feature alignment.) The paper should discuss more explicitly:
* Whether the training procedure of “MMD-only distillation” is otherwise identical to ADD or DMD pipelines aside from the loss definition.
* What specific advantages (e.g., stability, generalization) MMD introduces beyond computational simplicity.

### Questions
Please see the part of Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
