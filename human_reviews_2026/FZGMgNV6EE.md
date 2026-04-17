# Wavelet Diffusion Posterior Sampling with Frequency Domain Guidance

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Inverse imaging problems often involve the reconstruction of high-fidelity signals from noisy and incomplete measurements. Recent advances in diffusion models have achieved strong results for these tasks, yet most approaches operate in the spatial domain and struggle to preserve high-frequency details under noise. We introduce Wavelet diffusion posterior sampling (WDPS), a frequency domain framework that integrates wavelet transforms with posterior sampling. By decomposing images into multiscale frequency subbands, WDPS performs posterior updates adaptively across low- and high-frequency components, enabling more stable sampling trajectories and improved detail recovery. To further enhance robustness, we propose a wavelet-regularized diffusion strategy that dynamically adjusts the influence of frequency-domain constraints during sampling. We demonstrate our approach on both linear and nonlinear inverse problems. We also extend our task to the lensless camera task to show the applicability of our approach. Our results highlight the effectiveness of frequency-domain posterior diffusion as a general and efficient solution to noisy inverse problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Wavelet Diffusion Posterior Sampling (WDPS), which integrates wavelet transforms into diffusion posterior sampling to perform frequency-aware updates for inverse imaging tasks. While the idea of frequency-guided sampling is conceptually reasonable, the novelty over existing diffusion-based inverse problem solvers is limited, and the technical contribution is incremental. The paper also suffers from serious formatting issues, incomplete quantitative evaluations, and weak baseline comparisons. Overall, despite some interesting intuition, the work is not sufficiently strong or polished for acceptance at a top-tier venue.

### Strengths
The paper introduces a moderately novel idea of incorporating wavelet-based frequency guidance into diffusion posterior sampling, offering a limited but interesting extension of existing diffusion inverse problem frameworks.

### Weaknesses
1. **Formatting Issues:**
   The manuscript suffers from several serious formatting problems. For instance, some tables are stored as images and then placed alongside other figures using the *subfigure* (or similar) command, without providing an overall caption. As a result, the document lacks clear and consistent figure/table captions. This leads to confusion—for example, in the *Ablation Study* section, the first sentence states “We conduct an ablation study on the proposed dynamic wavelet regularization schedule. Results are provided in Tables 2b,” yet there is actually no Table 2b. In addition, captions appear inconsistently—sometimes above, sometimes below, and sometimes missing altogether.

2. **Baseline Comparison Issues:**
   The selection of baseline algorithms is problematic. On one hand, the paper does not compare with the latest state-of-the-art diffusion-based inverse problem solvers such as **SITCOM** or **DAPS** [1,2], whose performance is significantly superior to the reported methods. On the other hand, including algorithms like *Score-SDE*, which is a pure diffusion model rather than an inverse-problem solver, is confusing and makes the comparison less meaningful.

3. **Quantitative Metric Issues:**
   The quantitative evaluation lacks key metrics. The main body of the paper does not report **PSNR** or **SSIM**, which are standard in diffusion-based inverse problem literature. These metrics appear only in the appendix and only for the DPS baseline, which makes the comparison incomplete and less convincing.

[1] Alkhouri, Ismail, et al. *“SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems.”* Proceedings of the **42nd International Conference on Machine Learning (ICML)**.

[2] Zhang, Bingliang, et al. *“Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.”* Proceedings of the **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, 2025.

### Questions
1. **Formatting issues** – Can the authors provide a corrected version where all tables and figures have proper captions and numbering (e.g., fixing the missing “Table 2b” in the Ablation Study)?

2. **Baseline selection** – Why were recent SOTA diffusion-based inverse solvers not included? Please justify the inclusion of Score-SDE, which is not an inverse solver.

3. **Quantitative metrics** – Could the authors report standard metrics (PSNR, SSIM) for all methods, and compare them with SOTA diffusion-based inverse solvers to ensure fair evaluation?

### Soundness
2

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
4

### Summary
This paper tackles inverse problems of the form $Y=A(X)+\epsilon$ using diffusion models. Building on Diffusion Posterior Sampling (DPS), which enforces measurement consistency via a gradient update $\nabla_{X_t}\|Y-A(\hat{X}_0)\|^2_2$ during sampling, the authors propose performing the posterior update in the frequency (wavelet) domain instead of the spatial domain.

Specifically, each intermediate sample $X_i$ is decomposed into its wavelet representation $W_i$ using the discrete wavelet transform (DWT), and the posterior update is computed on $W_i$. After the update, the inverse DWT is applied to obtain the updated spatial representation.
To achieve coarse-to-fine restoration, the authors introduce a wavelet-strength scheduling function $r(i;a,b)$ that preserves the low-frequency subband while gradually amplifying the high-frequency components, scaling $W^{LH, HL, HH}_i$ as $r(i;a,b)\cdot W^{LH, HL, HH}_i$. By integrating this wavelet posterior update and wavelet-strength scheduling, the proposed method demonstrates improved performance over DPS across both linear and non-linear degradation scenarios.

### Strengths
1. Novelty

The application of posterior updates in the wavelet domain is novel and provides fresh insight into how frequency-domain representations can be leveraged for diffusion-based inverse problem solving. This idea can inspire further research in frequency-aware diffusion methods.

2. Technical soundness

The proposed strategy of controlling individual wavelet subbands to achieve coarse-to-fine restoration is conceptually sound and empirically effective, enabling better separation of global and local structures during reconstruction.

3. Experimental validation

The paper demonstrates consistent improvements across diverse degradation types, including both linear and non-linear cases, as well as challenging lensless camera setups, showing the robustness and generality of the proposed approach.

### Weaknesses
1. Posterior update in the wavelet domain itself is not effective.

The main claim of the paper is that performing the posterior update in the wavelet domain is more effective than doing so in the spatial domain. However, the experimental results (Table on page 9) show that the posterior update in the wavelet domain without wavelet-strength scheduling performs worse than the spatial-domain update. This indicates that the proposed “wavelet-strength scheduling” is the actual key component.

I think applying wavelet-strength scheduling after using original "spatial-domain" posterior update might be a more effective design according to the presented results.

2. Limited comparative methods.

The paper primarily compares the proposed approach with DPS, which was introduced in 2023.
As DPS is now considered a baseline, more recent zero-shot inverse problem solvers such as DAPS [1] or ReSample [2] should be included for a fair evaluation of current performance.

3. Evaluation limited to low resolution.

All experiments are conducted only at 256x256 resolution.
Since frequency-based posterior updates are potentially sensitive to image resolution, additional experiments at higher resolutions (e.g., 512x512) are needed to demonstrate robustness and scalability.

4. Potential diffusion off-manifold issue.

The proposed scaling of high-frequency wavelet subbands $W_{i}$ with a factor less than 1 will reduce the noise level of the current sample $X_{i}$.
This operation may make the noise variance is inconsistent with the expected variance at the corresponding diffusion timestep, potentially causing the sample to deviate from the learned diffusion manifold.
Such deviation could lead to error accumulation or unstable sampling behavior.
A clearer theoretical justification or empirical evidence is needed to demonstrate why this scaling does not compromise the diffusion process’s stability.

Reference

[1] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing (CVPR 2025)

[2] Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency (ICLR 2024)

### Questions
1. Ablation on spatial-domain posterior update with wavelet-strength scheduling

It would be valuable to examine how performance changes when applying the proposed wavelet-strength scheduling after the original spatial-domain posterior update. This comparison would clarify whether the key improvement stems from operating in the wavelet domain or from the wavelet-strength scheduling itself.

2. Inclusion of more recent comparative methods

Incorporating recent zero-shot inverse problem solvers, such as DAPS or ReSample, would provide a fairer and more comprehensive evaluation of the proposed method’s effectiveness relative to the current state-of-the-art.

3. Evaluation on higher-resolution images

Extending experiments to higher resolutions (e.g., 512x512) would strengthen the paper by demonstrating the robustness and scalability of the frequency-based posterior update.

4. Clarification on noise variance inconsistency

Additional explanation is needed regarding how scaling high-frequency wavelet subbands affects the noise variance at each diffusion step. Specifically, clarifying why this operation does not push the sample off the learned diffusion manifold would improve theoretical soundness and reader confidence.

### Soundness
2

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
4

### Summary
The authors propose Wavelet Diffusion Posterior Sampling (WDPS), which modifies DPS to operate in the wavelet domain rather than pixel space. This helps stabilize sampling, especially with particularly ill-posed or nonlinear forward models. Wavelet-based regularization helps stabilize sampling by starting with low-frequency structure and then refining high-frequency details when sampling is stabilized. The authors show qualitative and quantitative improvements across a variety of challenging inverse imaging tasks compared to DPS and popular baselines. They highlight lensless imaging as an application where their method shines.

### Strengths
* The results are quite impressive, especially for lensless imaging.
* The algorithm is a simple yet effective way to solve the problem of unstable sampling with DPS. The wavelet-based regularization is particularly effective.
* Although the wavelet strength schedule introduces hyperparameters, the authors provide some amount of theoretical justification for it.

### Weaknesses
* The chosen baselines are somewhat out-of-date. Consider adding a newer baseline like DAPS (Zhang et al. CVPR 2025).
* The wavelet strength schedule introduces another hyperparameter in addition to the hyperparameters that are already tricky to calibrate for DPS. I noted in the strengths, though, that this is less of a problem since the authors provide some theoretical justification.

### Questions
Please comment on the two weaknesses I mentioned.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets the issue of poor high-frequency reconstruction in training-free diffusion-based inverse imaging. Unlike standard Diffusion Posterior Sampling (DPS), which applies spatial-domain guidance uniformly across pixels, the authors propose Wavelet Diffusion Posterior Sampling (WDPS)—a multiscale, frequency-domain approach that performs posterior updates separately across wavelet subbands. By decomposing the intermediate latent into $(LL, LH, HL, HH)$ bands, WDPS applies distinct step sizes and dynamic weighting per frequency band. A wavelet-regularized diffusion scheme further stabilizes the process by gradually relaxing high-frequency constraints as denoising progresses.

Experiments span inpainting, deblurring, and super-resolution on FFHQ and ImageNet, plus a simulated lensless imaging setup. WDPS consistently improves sharpness and quantitative scores (PSNR, SSIM, LPIPS, FID) compared to DPS and similar plug-and-play baselines. The improvement is most prominent in high-frequency detail restoration (e.g., textures, edges). However, the paper lacks any analysis on sample diversity or how the frequency decomposition affects the stochasticity and bias of the posterior sampling trajectory.

### Strengths
Addresses a key limitation of DPS—uniform spatial guidance—by enabling per-frequency adaptation.

The approach is simple, plug-and-play, and computationally light since DWT/IDWT operations are orthonormal transforms.

Experiments demonstrate clear perceptual and quantitative gains, especially in high-frequency recovery under noise.

The intuition that guidance strength should differ per subband is sound and relevant for inverse problems with structured degradation.

### Weaknesses
The method is tailored to DPS and doesn’t generalize to variational oroptimization-based samplers such as RED-Diff, MPGD, or TMPD.

The so-called “stability analysis” (Theorem 1) is a conventional Lipschitz argument for deterministic inverse problems and doesn’t address the stochastic reverse process or the approximation bias of Tweedie-based guidance.

The paper does not analyze or quantify diversity. Since different subbands use different step sizes and strength schedules, the diversity of generated reconstructions could be heavily impacted—potentially either improving variety (through relaxed high-frequency control) or collapsing modes (through over-regularization).

Missing discussion of prior related works that also exploit multiscale or frequency decomposition in diffusion guidance (e.g., Sadat, Seyedmorteza, et al. "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales." arXiv:2506.19713 (2025).)

The paper does not demonstrate whether frequency-domain separation alleviates the known likelihood approximation bias in DPS, i.e., whether it reduces the variance or error of $\nabla_{x_t}\log p(y|x_t)$ estimates.

### Questions
Can the authors formally show that operating in wavelet space improves the conditioning or variance of the likelihood score approximation? This could explain the observed stability gains beyond heuristic intuition.

How does the dynamic wavelet strength schedule $r(i;a,b,C)$ affect convergence and sample diversity? Has its sensitivity been studied?

Does the per-band step size scheme effectively perform an anisotropic preconditioning of the posterior gradient? If so, can this be linked to improved bias-variance properties?

The method heavily relies on orthonormal wavelets (e.g., Haar). What happens with biorthogonal or learned transforms that break orthogonality?

The paper includes no metrics on sample diversity (e.g., variance, FID dispersion, or multi-sample PSNR spread). Since frequency-specific control can influence diversity significantly, such analysis is critical.

The theoretical section should clarify whether the stability result applies to the full stochastic diffusion chain or only to the deterministic gradient descent substep.

For completeness, the authors should discuss compatibility or contrast with RED-Diff (Mardani et al., 2023), MPGD (He et al., 2024), and TMPD (Boys et al., 2023)—all of which provide alternative corrections to the biased DPS surrogate and use momentum or covariance adjustments.

### Soundness
2

### Presentation
2

### Contribution
2
