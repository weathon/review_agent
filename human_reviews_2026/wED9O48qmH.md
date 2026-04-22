# KernelFusion: Zero-Shot Blind Super-Resolution via Patch Diffusion

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Traditional super-resolution (SR) methods assume an "ideal'' downscaling SR-kernel (e.g., bicubic downscaling) between the high-resolution (HR) image and the low-resolution (LR) image. Such methods fail once the LR images are generated differently. Current blind-SR methods aim to remove this assumption, but are still fundamentally restricted to rather simplistic downscaling SR-kernels (e.g., anisotropic Gaussian kernels), and fail on more complex (out of distribution) downscaling degradations. However, using the correct SR-kernel is often more important than using a sophisticated SR algorithm. In "KernelFusion'', we introduce a zero-shot diffusion-based method that uses an unrestricted kernel. Our method recovers the unique image-specific SR-kernel directly from the LR input image, while simultaneously recovering its corresponding HR image.  KernelFusion exploits the principle that the correct SR-kernel is the one that  maximizes patch similarity across different scales of the LR image.  We first train an image-specific patch-based diffusion model on the single LR input image, capturing its unique internal patch statistics. We then reconstruct a larger HR image with the same learned patch distribution, while simultaneously recovering the correct downscaling  SR-kernel that maintains this cross-scale relation between the HR and LR images. Empirical results demonstrate that KernelFusion handles complex downscaling degradations where existing Blind-SR methods fail, achieving robust kernel recovery and superior SR quality. By breaking free from predefined kernel assumptions and training distributions, KernelFusion establishes a new paradigm of zero-shot Blind-SR that can handle unrestricted, image-specific kernels previously thought impossible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a zero-shot blind image super-resolution method that leverages a diffusion model to learn patch distribution and then further use it for kernel estimation and super-resolution. The overall approach contains two stages: patch diffusion training and blind super-resolution. In the second stage, an image-specific kernel is estimated with implicit kernel representation. In experiments, the proposed method is evaluated on three blind SR datasets with complex kernels.

### Strengths
- The proposed zero-shot training and inference is interesting, and the predicted kernels look good.
- The results on Blind144 and DIV2KFK is promising.

### Weaknesses
Main weaknesses:
- In Figure 2, the HR images are sent to the UNet twice at each iteration. The training and inference of diffusion model s also makes this algorithm inefficient. Could the authors provide a comparison on model complex and inference time?
- Sec. 4 is not very clear to me, particular for the Phase 2. Could you rephrase it and provide a figure (or some math equations) for the kernel estimation part?
- The experiments are only performed on x4 SR, it would be better to also evaluate the method on x2 or x8 SR datasets.
- The written can be improved.
- Small issue and suggestion: please use "\citep" rather than "\cite" for the citations.

### Questions
- The diffusion is trained with T=1000 steps, have you tried to use some acceleration algorithms such as DDIM to reduce the inference time?
- In Eq. (1), the degradation is design without noise, but what if the LR image contains small noise?
- How do you design the kernel size you want to learn in real world applications?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper lacks sufficient motivation and fair evaluations. The proposed method tends to resolve complex kernel, however, the simulations are carried on unfair settings. The complex kernels are not real in application.

### Strengths
The paper is well-written and organised in a good structure.

### Weaknesses
1 This paper focuses on synthetic kernels (e.g., L-shape, empty squares in Fig. 5) with no evidence of their occurrence in real-world data. Why should the community care about reconstructing images degraded by a square kernel? The authors must justify whether such kernels exist beyond synthetic stress tests.
2 Training a diffusion model for ​20 minutes per image​ on an L40S GPU is impractical for real applications (e.g., mobile or large-scale processing). The authors do not discuss trade-offs between cost and performance gains.
​3  The authors claim diffusion models excel at capturing patch statistics but fail to compare against simpler internal-learning methods (e.g., GANs in KernelGAN). What advantages does diffusion offer over GANs for this task? The justification is weak.

4 This paper omits critical details: How are the patch diffusion and INR networks initialized? What is the role of the "consistency loss" in Eq. 2? How does it interact with the diffusion prior? Why is a U-Net applied twice in Phase 2 (Sec. 4)? The ablation (Table 3) hints at its importance but lacks explanation. Kernel Estimation Stability: The INR-based kernel estimator is praised for flexibility but not evaluated for stability (e.g., sensitivity to noise or image content).

5 The authors highlight performance on non-Gaussian kernels but ​ignore standard scenarios​ (e.g., isotropic Gaussian, motion blur). Does KernelFusion underperform on common kernels? Table 2 shows it is "comparable" on DIV2KRK (Gaussian kernels) but lacks head-to-head comparisons with SOTA methods like SwinIR or Real-ESRGAN in these settings.
​6 On DIV2KFK and Blind144, KernelFusion surpasses SOTA methods by only ​0.1-0.3 dB in PSNR (Table 2), while bicubic interpolation outperforms many blind SR methods. Are these gains meaningful given the high computational cost?
​7 Fig. 7 shows real-image results but without ground truth, making it impossible to assess accuracy. The kernels estimated for real images (e.g., DSLR without stabilization) appear unstructured—are they physically plausible?

### Questions
See weaknesses

### Soundness
1

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
This work proposes KernelFusion, which primarily focuses on accurate estimation of non-restricted SR kernels to effectively tackle blind SR problems. In the first phase of KernelFusion, it learns local patch distribution of a single LR image with a zero-shot PatchDiffusion (PD) model, by optimizing PD at test time. After PD is sufficiently trained, it is frozen and acts as a strong prior for patch similarity across different scales. In the second phase, the SR result and according SR kernel is simultaneously estimated, with each an UNet and an INN. The method effectively captures extreme non-gaussian SR kernels, and outperforms baselines under various blind SR settings.

### Strengths
- The overall writing is very clear and easy to follow, with well supported citations.
- The core idea is both intuitive and sound.
- The performance gain is significant for unrestricted complex kernels (e.g., Blind144, DIV2KFK).
- The method does not require pretraining, and thus, can adapt to arbitrary kernels.

### Weaknesses
The reviewer sincerely appreciates the authors’ efforts in this work. In its current state, I lean toward a borderline reject due to the weaknesses below. However, **I am willing to revise my evaluation to accept if these concerns are adequately addressed**. Please, counter-argue about my concerns and provide according experimental results.

---

**Weakness 1: Limited performance on conventional kernels**

In Table 2, the performance gain of **KernelFusion** is significant for *Blind144* and *DIV2K-FancyKernels (DIV2KFK)*. Since these test sets include extreme kernels (e.g., K0, K1, K2 in Fig. 5), these results align well with the aim of the paper: accurately estimating extreme **unrestricted** kernels.

However, **KernelFusion** shows limited performance on *DIV2K-RK*, which uses random Gaussian kernels. Considering that such kernels are more common (and thus more practical), I have concerns about the performance of **KernelFusion** in real-world use cases. Because **blind SR** primarily focuses on practical scenarios (as bicubic SR tasks), this limitation is critical.

---

**Weakness 2: Missing analysis for the choice of INR**

> *“We suspect that this limitation stems from the implicit bias of the CNN and MLP architectures (Line 105).”*

One of the key contributions of this work lies in the use of **INR architectures** to estimate complex non-Gaussian kernels. However, an analysis justifying this architectural choice is missing. It is necessary to include both (1) the final SR performance and (2) the kernel estimation results under alternative architectural designs for kernel estimation.

The reviewer suggests reporting SR performance and kernel estimation results

* for the extreme kernels K0, K1, and K2 (shown in Fig. 5), and
* under various architectural configurations, including (1) direct kernel optimization and (2) DIP-style architectures.

---

**Weakness 3: Lack of comparison with diffusion-based kernel estimation methods**

Accurately estimating complex kernels with diffusion models **without any prior (i.e., pretraining)** is an interesting aspect of this work. However, kernel estimation using diffusion models is **not entirely new**. Prior diffusion-based works are not sufficiently discussed in this paper.

For instance, **BlindDPS [1]** employs diffusion models to estimate non-linear kernels for solving arbitrary blind inverse problems. The reviewer highlights the need to compare and discuss this work in two respects:
* Although blind **SR** is not directly addressed in *BlindDPS*, extending it to SR is straightforward, as it only requires an additional subsampling step.
* The central objective of both works is accurate kernel estimation using diffusion models.

The reviewer recommends comparing **KernelFusion** with *BlindDPS* (and possibly other related works), both theoretically and experimentally.

[1] Chung, Hyungjin, et al. “Parallel diffusion models of operator and image for blind inverse problems.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023.

### Questions
**Question 1: Confusing experimental setting**

The experimental configuration of **Ablation (iii, PD+UNet)** is confusing.
Does *“acting as a denoiser”* indicate that the UNet is used only once (before the reverse diffusion step by PD)?
In other words, is this configuration equivalent to using **line 15 of Algorithm 1** only, while skipping **line 20 of Algorithm 1**?

If this interpretation is correct, the reviewer suggests adding another ablation experiment: skipping **line 15** while *not* skipping **line 20**. This experiment would help validate the necessity of applying the UNet twice, while also addressing the concern mentioned below:

> “Due to the small receptive field, the network does not know global structure, effectively destroying it …” (Line 430)

---

**Question 2: Necessity of using a shared UNet**

KernelFusion employs the same UNet twice (once before and once after PD).
Please discuss (and experimentally compare if possible) a configuration that uses independent network weights for each UNet step.

---

**Question 3: KernelFusion is slow**

> *"Our aim is not to deliver a production-ready Blind-SR system, but to establish feasibility of unrestricted kernel estimation ... "* (Line 123)

KernelFusion currently requires approximately **20 minutes** to perform SR for a single image, which remains excessively slow in practice: a clear limitation. However, the reviewer agrees with the academic necessity of analyzing unrestricted kernel estimation (despite being slow) and also considers the statement above fair.

However, please discuss (and experimentally compare if possible) configurations that reduce the number of diffusion steps (or explore techniques designed to accelerate diffusion processes).

### Soundness
3

### Presentation
4

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
This paper proposes KernelFusion, a zero-shot blind super-resolution method that simultaneously estimates an image-specific degradation kernel and the high-resolution image using a patch-based diffusion model. It addresses the limitation of existing methods that fail on complex, non-Gaussian kernels by leveraging internal learning, avoiding out-of-distribution problems.

### Strengths
- The combination of internal patch learning via diffusion models with implicit neural representations for kernel estimation is novel and creative.

- Successfully handles complex, non-Gaussian kernels where state-of-the-art methods fail, pushing blind SR into a more assumption-free paradigm.

- Strong experimental results on both synthetic kernels and real-world images convincingly demonstrate the advantage.

### Weaknesses
1. **Computational Cost and Practicality:** The most significant weakness is the computational cost. A 20-minute training time per image on a high-end GPU (L40S) makes it impractical for any real-time or even interactive application. While the authors correctly state their goal is to establish feasibility, this limitation is severe and should be a central point of discussion. Comparisons to other methods should ideally mention their relative runtimes to provide context.

2. **Limitation to Global Kernels:** The method is built on the assumption of a single, globally uniform kernel. As noted in the limitations, this prevents it from handling spatially varying blur, which is a common and important problem in real-world scenarios (e.g., motion blur in parts of an image, lens aberrations). This restricts the scope of its applicability.

3. **Ablation on Kernel Estimation Component:** While the ablation study in Table 3 effectively dissects the contributions to the final HR reconstruction quality (PSNR), it does not explicitly ablate the kernel estimation component. How crucial is the INR architecture compared to a simpler parameterization? A quantitative analysis (e.g., using kernel PSNR or MSE) of the estimated kernels in the different ablated settings would strengthen the claim that the INR is key to recovering complex kernels.

4. **Comparison to Other Zero-Shot Methods:** The comparison to the two-step process of KernelGAN+ZSSR is excellent and shows a clear advantage. However, it would be even more compelling to see a comparison against other recent internal-learning or zero-shot SR methods that are not based on diffusion models, to better isolate the benefit of the proposed diffusion-based approach.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
