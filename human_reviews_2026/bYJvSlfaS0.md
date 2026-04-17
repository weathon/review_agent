# Normalization-equivariant Diffusion Models: Learning Posterior Samplers From Noisy And Partial Measurements

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Diffusion models (DMs) have rapidly emerged as a powerful framework for image generation and restoration, achieving remarkable perceptual quality. However, existing DMs are primarily trained in a supervised manner by using a large corpus of clean images. This reliance on clean data poses fundamental challenges in many real-world scenarios, where acquiring noise-free data is hard or infeasible, and only noisy and potentially incomplete measurements are available. While some methods are capable of training DMs using noisy data, they are generally effective only when the amount of noise is very mild or when some additional noise-free data is available. In addition, existing methods for training DMs from incomplete measurements require access to multiple complementary acquisition processes, an assumption that poses a significant practical limitation. Here we introduce the first approach for learning DMs for image restoration using only noisy measurement data from a single operator. As a first key contribution, we show that DMs, and more broadly minimum mean squared error denoisers, exhibit a weak form of scale equivariance linking rescaling in signal amplitude to changes in noise intensity. We then leverage this theoretical insight to develop a denoising score-matching strategy that generalizes robustly to noise levels lower than those present in the training data, thereby enabling the learning of DMs from noisy measurements. To further address the challenges of incomplete and noisy data, we integrate our method with equivariant imaging, a complementary self-supervised learning framework that exploits the inherent invariants of imaging problems, in order to train DMs for image restoration from single-operator measurements that are both incomplete and noisy. We validate the effectiveness of our approach through extensive experiments on image denoising, demosaicing, and inpainting, along with comparisons with the state of the art.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenging problem of training diffusion models from a single set of noisy and incomplete measurements. The authors introduce a normalization equivariance property to modify the SURE loss, enabling a denoiser trained only at noise level $\sigma_n$ to generalize to levels $\sigma < \sigma_n$. This key insight, when combined with the Equivariant Imaging framework, allows the model to learn a posterior sampler from a single, fixed, and rank-deficient measurement operator. The method is validated on denoising, inpainting, and demosaicing, showing strong perceptual quality against self-supervised baselines.

### Strengths
1. The paper tackles an important problem of training generative models from single-operator, noisy, and incomplete measurements.
2. The identification and exploitation of "normalization equivariance" as a mechanism to bridge the gap and train a denoiser for $\sigma < \sigma_n$ using only data at $\sigma_n$ is a clever and original contribution.
3. The empirical results are comprehensive and promising.

### Weaknesses
1. The authors propose a loss-based enforcement and state, "we find that this leads to better performance for self-supervised learning". This is a strong claim, but it is not substantiated with an experimental comparison. A key missing ablation would be to train the architecturally-equivariant network from Herbreteau et al. (2024) using the SURE loss and compare it against the proposed $\mathcal{L}_{NE,SURE}$.
2. The proposed loss $\mathcal{L}_{NE,SURE}$ samples $\alpha \sim \mathcal{U}(0,1)$ and $\mu \sim \mathcal{U}(0,1)$. This choice seems arbitrary. How sensitive is the method to this choice? Would a different distribution (e.g., log-uniform for $\alpha$) be more effective? Some justification or ablation on this design choice is needed.
3. In Figure 4, the authors claim the MSE scales approximately as $\sigma_t^2/\sqrt{N}$. However, the $C/\sqrt{N}$ line does not appear parallel to several of the empirical lines. This claim seems to be a slight over-simplification of the empirical result shown.

### Questions
1. What is the rationale for sampling $\alpha, \mu \sim \mathcal{U}(0,1)$ in Eq. 8? Have the authors explored the sensitivity of the method to this sampling distribution?
2. Is there a more competitive generative self-supervised baseline for the single-operator setting apart from Ambient Diffusion?
3. Learning from corrupted data is very timely. I would recommend that the authors also discuss their method in the context of other recent works tackling similar problems, which may offer complementary insights:
    - Restoration Score Distillation: From Corrupted Diffusion Pretraining to One-Step High-Quality Generation, arxiv, 2025
    - An Expectation-Maximization Algorithm for Training Clean Diffusion Models from Corrupted Observations, NeurIPS 2024
    - Learning Diffusion Priors from Observations by Expectation Maximization, NeurIPS 2024

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a diffusion-based image restoration framework that learns entirely from noisy measurement data obtained through a single operator. By embedding the normalization-equivariance property into the standard SURE loss, the method enables training without clean ground truth and achieves competitive performance across denoising, demosaicing, and inpainting, along with comparisons with the state of the art.

### Strengths
- Good motivation and no ground-truth requirement. The paper presents a theoretical framework to train diffusion-based denoisers and posterior samplers without any access to clean data, offering a practical and scalable solution for real-world denoising and inverse problems where ground-truth supervision is infeasible.

- Novel theoretical insight and finding (Normalization-Equivariance). The authors contribute by formalizing a new normalization-equivariant property of MMSE denoisers, providing both analytical justification and architectural grounding for learning across unseen noise levels.

### Weaknesses
- Theoretical assumptions are idealized. The normalization-equivariance assumption relies on the prior being approximately positively homogeneous and factorable into radial and angular components. This theoretical assumptions may not be practical in real world setting. 

- Restricted to gaussian noise and linear operators. The method is formulated specifically for additive Gaussian noise and linear forward operators. Its applicability to non-Gaussian, structured, or data-dependent degradations (e.g., Poisson, compression artifacts, MRI nonlinearity) remains unclear.

- Limited real-world dataset validation and ablation.
All experiments are conducted on synthetic corruption processes (Gaussian noise, masking, demosaicing) using curated datasets like FFHQ, AFHQ, and NBU. The method has not been evaluated on real-world restoration benchmarks (e.g., RainDrop, AllWeather, SIDD, BSD-Denoise), where uncontrolled degradations and sensor characteristics could significantly affect performance. More ablation and cross-domain testing would strengthen claims of generalization.

### Questions
Could the authors evaluate how the proposed normalization-equivariant diffusion framework performs on real-world degradation datasets (e.g., AllWeather, RainDrop, SIDD) where noise characteristics deviate from the assumed Gaussian model and amplitude–structure independence does not strictly hold?

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
3

### Summary
The paper introduces a novel approach for training diffusion models (DMs) for image restoration using only noisy measurement data from a single operator. The authors show that DMs and MMSE denoisers exhibit a weak form of scale equivariance, linking signal rescaling to changes in noise intensity. Building on this insight, they develop a denoising score-matching strategy that generalizes to lower noise levels than those seen during training. To handle incomplete and noisy data, their method is integrated with equivariant imaging, and experiments on denoising, demosaicing, and inpainting demonstrate its effectiveness compared to state-of-the-art approaches.

### Strengths
1. The method tackles a challenging problem of image restoration using only corrupted data which can be useful in some scenarios.

2. The method achieves better results compared to the existing methods.

### Weaknesses
1. I think the method can only be applied to some specific degradations like noise and masks.

### Questions
1. Can the method be extended in the case of data corrupted with multiple degradations?

2. Can the method be applied to more challenging degradations like blur for example?

### Soundness
3

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
3

### Summary
The paper proposes **Normalization-Equivariant Diffusion Models** for learning posterior samplers using only noisy and/or partial measurements from a single operator. The key idea is to exploit a *scale (normalization) equivariance* property of MMSE denoisers to transfer supervision from the measurement noise level to lower noise levels, enabling conditional score learning without clean data. Concretely, the authors embed this property into a modified SURE objective (NE-SURE) and then plug the resulting denoiser into a diffusion sampler. They further combine this with Equivariant Imaging to handle rank-deficient single-operator inverse problems (e.g., inpainting, demosaicing). Experiments on FFHQ/AFHQ/NBU show improved perceptual metrics over self-supervised baselines and competitive results to supervised counterparts.

### Strengths
Shows that a weak form of scale equivariance links amplitude rescaling and noise level, and operationalizes it via NE-SURE (Eq. 8) to learn below the measurement noise—critical for diffusion posterior sampling without clean data. A theorem formalizes when MMSE denoisers are approximately normalization-equivariant under a factorized prior, lending plausibility to the rescaling argument. On denoising, the method maintains strong performance below the training noise and improves perceptual metrics over self-supervised baselines; on inverse problems, it outperforms EI/Ambient Diffusion in FID/LPIPS while EI remains better in distortion metrics—consistent with perception–distortion trade-offs.

### Weaknesses
1. The approximate scale-equivariance relies on structural assumptions about the image prior (factorization into norm and direction); discussion is persuasive but still idealized, and empirical ablations that probe violation of these assumptions (e.g., strongly non-homogeneous priors) are limited.
2. Training and evaluations assume additive Gaussian noise with known level; robustness to modest model mismatch (e.g., mis-specified σ, signal-dependent noise) is not studied.
3. While the EI fusion is elegant, demos focus on synthetic masks/Bayer patterns at one noise level; broader operators (e.g., blur, Fourier subsampling) or real sensor pipelines would strengthen claims of generality.

### Questions
1. How sensitive is NE-SURE to under/over-estimating the measurement noise? Please add curves where training assumes ωₙ′ ≠ true ωₙ.
2. Beyond (ε, μ) sampling in Eq. (8), what happens if μ=0 or if ε sampling range narrows? Does performance degrade smoothly?
3. Can you show results for additional single operators (e.g., spatial blur, partial Fourier) to complement inpainting/demosaicing?
4. How do FID/LPIPS trade with number of reverse-SDE steps K and ω-schedules for the same trained denoiser?

### Soundness
2

### Presentation
3

### Contribution
2
