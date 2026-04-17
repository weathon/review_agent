# Residual Diffusion Implicit Models

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Diffusion models achieve state-of-the-art results across multiple tasks. However, in inverse problems, standard initialization from pure Gaussian noise misaligns the generative process with real-world degradations. More recent methods such as diffusion bridges impose strict endpoint constraints and often require long reverse processes that are prone to hallucinations. Alternative consistency models provide noise-invariant, one-step mappings but lack inherent variance modeling and can degrade under severe corruption. Hence, residual diffusion implicit models (RDIMs) are proposed, constituting a generalized framework that explicitly models the residuals between high-quality (HQ) and low-quality (LQ) images, aligning the forward process with the actual degradation. A non-Markovian implicit reverse sampler is derived, which can skip intermediate timesteps, enabling accurate few-step or even single-step reconstruction, while mitigating the hallucinations inherent to long diffusion chains. RDIM also introduces a controllable variance mechanism that interpolates between deterministic and stochastic sampling, balancing fidelity and diversity. Furthermore, it enables the straightforward use of perceptual losses, when needed. Experiments on denoising and super-resolution benchmarks demonstrate that RDIMs consistently outperforms the state of the art, including bridge and consistency models, in terms of PSNR, SSIM, and LPIPS, reducing halucinations while requiring only a few sampling steps (often just one). The results position RDIMs as an efficient solution for a broad range of image restoration tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes *Residual Diffusion Implicit Models (RDIM)*, which aligns the *forward* process with real degradation (by removing the residual between HQ and LQ) and adopts a DDIM-style non-Markovian long-range skip sampling for the *reverse* process. Formally, the method strictly subsumes ResShift as a special case, making RDIM a generalization of it. Experiments report that RDIM-1 (1 step) and RDIM-10 (10 steps) outperform or are comparable to ResShift/DDPM while being faster.

### Strengths
1. The paper proves that ResShift is a special case of RDIM, enabling existing ResShift models to be directly used for RDIM sampling without retraining.
2. The forward process is aligned with degradation and includes a controllable variance mechanism.
3. Experiments demonstrate strong performance with significant acceleration.

### Weaknesses
**Lack of novelty:**

1. One of the authors’ contributions is generalizing ResShift, but from today’s perspective, RDIM is already outdated. The authors lack an adequate survey of recent works.
   The core formulation of the paper is: $q(x_t|x_0,\Delta)=\mathcal{N}(x_0 - \beta_t \Delta, \gamma^2 \beta_t I)$. However, expanding it gives: $q(x_t|x_0,\Delta)=\mathcal{N}((1-\beta_t)x_0+\beta_t y_0, \gamma^2 \beta_t I)$, which can also be written as: $q(x_t|x_0,y_0)=\mathcal{N}((1-\beta_t)x_0+\beta_t y_0, \gamma^2 \beta_t I)$. This is clearly a simple case of stochastic interpolation [1] (diffusion bridge) or essentially time varying Ornstein–Uhlenbeck (OU) process. The sampling formulas and training losses mentioned later are straightforward corollaries under this framework. The acceleration formula that follows is also a rather trivial conclusion of DDIM.

**Lack of baseline comparisons:**

2. Based on the above, the authors did not compare with several advanced diffusion-bridge-type models. For example, **GOUB[3] (ICML 2024)** achieves 28.5 PSNR on DIV2K ×4, and **UniDB [7](ICML 2025)** reaches 28.64 — both higher than RDIM, and belonging to the same family of methods. Other models such as **BBDM**[4], **I2SB**[5], , **DDBM**[6] and more advanced methods, should also be discussed in the *Related Work* section.

**Writing:**

3. The authors first introduce the goal as $q(x_T |x_0,\Delta) = q(x_T |y_0)$, but immediately follow it with the proof in Appendix 1. At that point, the forward process has not yet been defined, but the appendix proof directly uses the later-defined forward process. This creates a logical inconsistency and can confuse readers when reading sequentially.

**Overall:**
From today’s perspective, RDIM lacks innovation. I have some suggestions:

1. Conduct a deeper survey of diffusion-bridge-related works (2024–2025). The authors need to distinguish RDIM from existing models and establish a more advanced motivation.
2. Compare RDIM with more diffusion-bridge models through deeper experiments to validate its theoretical advantages.

# Reference:

[1] Michael S. Albergo, et al. "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions." (arXiv 2023)

[2] Luo, Ziwei, et al. "Image restoration with mean-reverting stochastic differential equations." (ICML 2023).

[3] Yue, Conghan, et al. "Image restoration through generalized ornstein-uhlenbeck bridge." (ICML 2024).

[4] Li, Bo, et al. "Bbdm: Image-to-image translation with brownian bridge diffusion models." (CVPR 2023).

[5] Liu, Guan-Horng, et al. "I $^ 2$ SB: Image-to-Image Schr\" odinger Bridge." (ICML 2023).

[6] Zhou, Linqi, et al. "Denoising diffusion bridge models." (ICLR 2024).

[7] Zhu, Kaizhen, et al. "UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control." (ICML 2025).

### Questions
See weaknesses.

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
4

### Summary
In this work, authors propose residual diffusion impicit models (RDIMs), a generalized framework to model the residuals between high-quality and low-quality images. In particular, the forward process of the model is aligned with the degradation model, which enables reconstructions that are faster and more accurate. Inspired by DDIMs, during the reverse process, some of the intermediate timesteps are skipped to accelerate the reconstruction. Akin to DDIM sampling, authors introduce a controllable variance mechanism to control the stochasticity of the sampling. Authors demonstrate the effectiveness of RDIMs on denoising and super-resolution benchmarks and further qualitative evidence on image inpainting, colorization, and deblurring tasks.

### Strengths
* The paper is written well and is easy to read.
* The ablation studies are conducted extensively, solidying the design choices. 
* The authors demonstrate their method on various degradation models (SR, denoising, deblurring, colorization, inpainting, etc.)
* In terms of reproducibility, authors did an excellent job in disclosing dataset and implementation details.

### Weaknesses
* The work is not contextualized well. There are several important works that investigate aligning the forward process of the model with degradation (see [1-4] below, note that [3] is cited but not discussed). These should be discussed and compared with RDIM. Whenever appropriate, it would be good to include some of these as part of baseline results. The only competitive baseline method is ResShift which shares too many similarities with RDIM already. 
* On a related note, I find the technical contributions of the paper to be limited. In its current version, it reads as a simple extension of ResShift combining it with (essence of) DDIM sampling.
* The central claim that RDIM matching or exceeding ResShift performance while requiring fewer samples is due to distortion metrics (PSNR and SSIM). Due to perception-distortion trade-off [5] (also noted by authors in Appendix C.7), it is possible that in terms of perceptual quality, ResShift (or other models that perform many steps) can outperform few/single-step solvers (also observed in [1,2] and also ResShift). If this is the case, limitations/benefits of RDIM against ResShift should be disclosed and discussed better.
* Please see questions below as well.

### Questions
***Questions and Suggestions:***
* In terms of evaluation metrics, providing perceptual quality scores such as LPIPS and FID would be good to complement the distortion metrics (PSNR/SSIM) and provide a more comprehensive view. As authors also comment on this in Appendix C.7, perception-distortion trade-off is a well known phenomenon [5].
* As mentioned in the weaknesses section, could the authors comment on similarities and differences between RDIM and existing work [1,2,3,4]? I would like to note that for RDDM [3], the authors cite this work saying "... requiring to traverse all diffusion timesteps" but their conclusion mentions that "RDDM achieves SOTA performance in no more than five sampling steps".
* Besides comparing with ResShift, which can be considered as an ablation study due to RDIM generalizing this work, I think comparison with more competitive baselines (which are dedicated diffusion based solver) would help clarify the strengths/weaknesses of the method. Some of the good candidates are DPS (Diffusion Posterior Sampling), CCDF (Come-Close-Diffuse-Faster), DDRM, Resample [6], etc.
* In the introduction, authors cite DDPM as a powerful class of models for image reconstruction (line 36-37). I think this is slightly misleading. Such models were introduced as a means for generative modeling/image synthesis. Although it is natural to consider them as useful priors, it took non-trivial effort in the community to actually use such priors and guide the generation process tailored towards arbitrary (linear/non-linear) reconstruction problems. For readers not familiar with the literature, I would recommend rephrasing this section for clarity.

***
***References:***

[1] Delbracio, Mauricio, and Peyman Milanfar. "Inversion by direct iteration: An alternative to denoising diffusion for image restoration." arXiv preprint arXiv:2303.11435 (2023).

[2] Fabian, Zalan, Berk Tinaz, and Mahdi Soltanolkotabi. "Diracdiffusion: Denoising and incremental reconstruction with assured data-consistency." Proceedings of machine learning research 235 (2024): 12754.

[3] Liu, Jiawei, et al. "Residual denoising diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[4] Heitz, Eric, Laurent Belcour, and Thomas Chambon. "Iterative α-(de) blending: A minimalist deterministic diffusion model." ACM SIGGRAPH 2023 Conference Proceedings. 2023.

[5] Blau, Yochai, and Tomer Michaeli. "The perception-distortion tradeoff." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[6] Song, Bowen, et al. "Solving inverse problems with latent diffusion models via hard data consistency." arXiv preprint arXiv:2307.08123 (2023).

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
4

### Summary
This paper introduces Residual Diffusion Implicit Models (RDIMs), a diffusion framework tailored for inverse problems such as image denoising and super-resolution. Unlike standard DDPMs that start from pure Gaussian noise, RDIMs explicitly model the residuals between high-quality and low-quality images, aligning the forward process with the actual degradation. The reverse process leverages implicit sampling to skip intermediate steps, enabling few-step or even single-step reconstructions. Additionally, RDIMs incorporate a controllable variance mechanism to balance deterministic and stochastic sampling. Experiments on denoising and super-resolution benchmarks demonstrate that RDIMs outperform DDPMs and match or surpass ResShift, while reducing inference steps by up to 100×.

### Strengths
1. This paper aligns the forward process with real degradations, addressing the mismatch of standard diffusion models in inverse problems.
2. This paper shows that ResShift is a special case of RDIM, highlighting the generality of the framework.

### Weaknesses
1. Experiments mainly compare against DDPM and ResShift; missing comparisons with other recent efficient diffusion approaches (e.g., consistency models, rectified flow).
2. This paper focuses on denoising and super-resolution; validation on other inverse problems (e.g., deblurring, inpainting, compressed sensing) is limited.

### Questions
1. How does RDIM ensure stability and avoid artifacts when using extremely few steps (e.g., single-step reconstruction)?
2. Can pretrained ResShift models be directly reused for RDIM across different tasks without retraining, or are task-specific adjustments required?

### Soundness
2

### Presentation
2

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
This paper proposes Residual Diffusion Implicit Models (RDIM), shifting the diffusion process from the traditional paradigm of ‘diffusing a clean image into pure noise and then reconstructing from the noise’ to directly modelling the residual $\Delta = x_0 − y_0$ between the clean image $x_0$ and the observed low-quality image $y_0$. Specifically, the forward (degradation) process progressively removes residuals while allowing controllable variance injection; the inverse (reconstruction) process recovers $x_0$ from the final latent variable $x_T$ aligned with the observation (rather than pure noise). Inverse sampling employs implicit sampling akin to DDIM, enabling skipping of intermediate steps to achieve restoration in fewer steps or even a single step. Experiments demonstrate significant reductions in sampling steps compared to standard methods like DDPM/ResShift for denoising and super-resolution tasks, while achieving superior PSNR/SSIM metrics.

### Strengths
1. The idea is very interesting, i.e. changing the diffusion objective from ‘noise’ to “residual” theoretically aligns better with inverse problems (where observations already contain substantial information), reducing unnecessary randomness and redundant ‘recovery from noise’ steps.

2. Employing implicit sampling (DDIM-style) with skip sampling enables the inverse process to be performed in $S<<T$ steps. This should reduce inference latency in practical applications, especially for natural image restoration.

3. Experimental results on natural image restorations (e.g. denoising, super-resolution) are good.

### Weaknesses
Overall, the paper builds upon and extends ResShift, representing a further advancement of the residual/residual shift concept when combined with DDIM-style implicit sampling. Similar approaches, such as replacing noise space with observation-aligned latent space or directly modelling residuals, have partial precedents in the literature and essentially overlap with certain conditional diffusion/restoration models, such as DDRM (Denoising Diffusion Restoration Models) and 'Residual denoising diffusion models'. 

1. Line 127. 'Subsequently, the Markovian formulation is relaxed to derive a non-Markovian process that preserves the same
marginal distributions. The author does not provide a complete mathematical proof or rigorous conditions. In particular, when stepwise sampling is employed, the claim that the target marginal distribution is preserved (or that the inverse process remains valid) needs further theoretical justification.

2. The results are not particularly promising. The experiments lack a systematic comparison with recent works on fast or accelerated diffusion models, e.g. Consistency Models (Song et al., ICML'23). 

3. Furthermore, the paper does not sufficiently demonstrate performance degradation when the forward model is mismatched.

4. The performance of RDIM needs to be demonstrated on real or more challenging applications, such as medical image reconstruction (e.g. accelerated MRI or sparse-view CT).

### Questions
Is the forward degradation model used for training and testing consistent with the assumptions made during inference? If the degradation operator known during training differs from that encountered during testing, how robust is the method? 

In the context of 'single-step reconstruction', the constraints and generalisability (i.e. feasibility across tasks/noise intensities) must be explicitly stated.

Regarding real inverse problems (e.g. sparse-view CT), where the observations (sinograms) are generally corrupted by a mixture of Gaussian and Poisson processes, will RDIM still work?

### Soundness
3

### Presentation
3

### Contribution
3
