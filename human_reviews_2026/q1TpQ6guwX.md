# Integrated Forward–Inverse Network for Physics-Guided Image Reconstruction

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Inverse modeling plays a central role across computational optical imaging problems, including microscopy, imaging through scattering media, and lensless cameras, where the forward model often manifests as a severe blur. Discrepancies between the model and the actual imaging process further aggravate the ill-posed nature of the inverse problem. Physics-enabled methods that integrate analytical forward models with data-driven networks have been explored, but most incorporate physics only in a one-sided manner—either operating purely in the measurement space or only after inversion—thereby discarding complementary cues and reducing robustness to calibration errors.
Here, we propose the Integrated Forward–Inverse Network (IFIN), a physics-guided deep neural network that interleaves differentiable forward operators with learnable inverse modules at every stage of the hierarchy. This design preserves physical consistency while shaping richer feature representations by jointly leveraging information from both measurement and image domains. A physics-guided kernel adaptation further compensates for inaccurate or unavailable PSF calibration, dynamically refining the kernel for blind deconvolution under system constraints.
IFIN is especially effective when measurements are severely blurred by large point-spread functions, where conventional CNN-based inversion is limited by local receptive fields and underutilizes the measurement signal. On challenging lensless imaging benchmarks—including our newly introduced dataset, IFIN achieves state-of-the-art reconstruction quality and improved robustness under noise and model mismatch.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Integrated Forward–Inverse Network (IFIN), an encoder–decoder architecture that embeds a Forward System Operator (FSO) and a learnable Inverse System Operator (ISO) at every feature scale, conditioned by a shared, learnable spatially varying PSF field. The goal is to keep raw measurement information useful  throughout the network while enforcing forward–inverse consistency, which the authors argue is crucial under severe, large-kernel blur in lensless imaging. The method initializes with a PSF-aware inverse, then runs two coupled streams, measurement to and from images,  through multi-scale IFIB blocks. Experiments on DiffuserCam, an SV-lensless dataset, and a MultiWienerNet benchmark show improvements over classical, data-driven, and physics-guided baselines.

### Strengths
* The problem motivation is clear. Conventional CNN/ViT backbones under-utilize information under large-kernel blur; one-sided physics pipelines discard complementary cues. The paper articulates this trade-off well.  

* Robust empirical evaluation is provided. Diverse baselines (classical, data-driven, physics-guided) and three benchmarks dataset are used to report the results.

### Weaknesses
* The score of the work is too narrow. The title suggests that the work could be applied to image reconstruction, while the paper and the contribution only targets the lensless imaging. It is not clear what is the use of the proposed method on the general family of inverse problems. 

* The contribution is very limited. While the authors describe the integration of forward and inverse operators at every scale as novel, this idea parallels well-established unrolled or feature-space deconvolution frameworks. In these methods, forward and adjoint physics are embedded at every iteration or scale. The “forward–inverse coupling at each layer” is not new; it’s conceptually equivalent to "Unrolled primal–dual or ADMM networks (Adler & Öktem 2018 and  Zhang et al. 2020)", which apply both forward and adjoint operators at every iteration, and "Deep Wiener and Multi-Wiener Deconvolution Networks (DWDN/MWDN) (Dong et al and  Li et al.)", which already apply learned deconvolutions at multiple scales in feature space and can incorporate a physical forward model. 

* The use of a learnable PSF field is a straightforward extension of prior PSF-grid and blind deconvolution frameworks. Encoding the PSF and conditioning the network hierarchically is an implementation detail, not a conceptual advance, given that hypernetwork conditioning on optical parameters is widely explored in blind or self-calibrating deconvolution literature.

* The method is described as physics-integrated, yet both the forward and inverse modules reduce to convolutional operations without explicit physics enforcement or constraints. Thus, the “integration” is architectural rather than physical or theoretical in nature.

* The contribution claim on “integration” improving performance, but there is no ablation showing the involvement of each component in the final performance. 

* The performance comparison of couple of baselines is missing ( FISTA-Net, MoDL, and newer deep unfolding methods such as "Robust unrolled network for lensless imaging with enhanced resistance to model mismatch and noise").

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
IFIN is a physics-guided network that interleaves differentiable forward operators with learnable inverse modules at each stage, jointly exploiting measurement and image domains to keep physical consistency and enrich features. With a physics-guided kernel adaptation for PSF mismatch, it achieves state-of-the-art lensless imaging reconstruction and improved robustness to noise and model errors, especially under severe blur.

### Strengths
Interleaving differentiable forward operators with learnable inverse modules enforces physical consistency while enriching representations in both measurement and image domains.
﻿The physics-guided kernel adaptation mitigates PSF mismatch/incompleteness, enabling constrained blind deconvolution and reducing sensitivity to model errors.
﻿Demonstrates state-of-the-art reconstruction quality on challenging lensless benchmarks (including a new dataset).

### Weaknesses
Physics grounding is insufficient. Although the paper claims to be physics-inspired, it does not substantiate the physical modeling in depth (assumptions, operator derivations, constraints, or validation against instrument physics).
Requires accurate, differentiable operators; robustness claims under strong model misspecification (nonlinear aberrations, spatially variant PSFs, misalignment) aren’t quantified.
Blind PSF refinement can be ill-posed; constraints, regularizers, and failure modes (e.g., texture transfer, PSF–image ambiguity) are not discussed.

### Questions
1.The paper claims to be physics-inspired, but the analysis is superficial, providing only a brief explanation of the forward and inverse processes in lensless imaging.
2.It is unclear how the learnable PSF is obtained—the paper should specify the initialization, optimization strategy, and physical constraints involved.
3.In Figures 3–5 and Table 1, the qualitative comparisons do not include recent methods from the past two years.
4.For in-the-wild lensless imaging, there is no quantitative comparison. Additionally, it is unclear how many images are included in this dataset and how many images are contained in the proposed SV Lensless dataset.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes IFIN for physics-guided image reconstruction. IFIN embeds a pair of operators at every encoder–decoder stage: a FSO that maps current estimates into the measurement domain and an ISO that maps measurements back toward the image domain. A learnable, spatially varying PSF field conditions both operators across scales, supporting blind or mis-calibrated settings. Experiments on DiffuserCam and MultiWienerNet report consistent improvements over classical and recent learned/physics-guided baselines in PSNR/SSIM and LPIPS.

### Strengths
1. The paper presents a well-motivated architecture that tightly integrates forward and inverse physics across network stages to address limitations of one-sided models.
2. It demonstrates consistent performance gains on multiple benchmarks, including real-world spatially varying data.

### Weaknesses
The contribution reads more like a careful system integration than a genuinely new paradigm; the paper offers little formal grounding—there is no error-propagation or identifiability analysis, and the bias from using an averaged PSF under spatial variance is not quantified; ablations and robustness checks are thin (no systematic removal of modules or scale coupling, limited study of PSF tiling and losses, no variance/significance reporting or cross-device transfer); baseline fairness is uncertain because prior physics-guided methods are retrained under a unified loss without parallel results from their original settings or stronger recent baselines; presentation leaves gaps in where SI vs. SV operators are used and lacks clear captions/notation and a limitations section; and there is no clear commitment to release code, models, or data, which undercuts reproducibility.

### Questions
1.The innovation mainly lies in system-level integration (multi-scale forward–inverse coupling with a learnable SV-PSF) rather than a conceptually new paradigm; the paper lacks a principled justification for why coupling both domains at all scales is necessary or theoretically superior to top-level or one-sided physics.

2.The distinction from prior unfolded optimization or feature-domain deconvolution frameworks (e.g., UPDN, MWDN) is insufficiently clarified, making the novelty appear incremental without controlled counterexamples or failure analyses.

3.Key design choices (where SI vs. SV operators are applied, how PSF encodings are injected, and the gating/residual paths) are scattered between main text and appendix, making the architectural logic difficult to reconstruct from the figures alone.

4.The paper lacks a dedicated “Limitations” or “Failure Cases” section and does not discuss boundary conditions such as sampling mismatch, noise robustness, or nonlinear forward models, which would help frame applicability.

5.The description of “physics-guided” components is mostly heuristic; it does not clearly connect to physical principles such as energy conservation, boundedness, or optical transfer constraints, which weakens the claimed physical interpretability.

6.Some notation and figure captions are incomplete or inconsistent—operator modes (SI/SV), PSF-tile granularity k, and cross-scale data flow are not explicitly annotated, affecting readability and reproducibility.

7.Computational trade-offs (accuracy–cost–latency) are unexplored; practitioners cannot judge when to switch between the SV and surrogate FSO modes or how scaling affects efficiency.

### Soundness
3

### Presentation
3

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
The paper proposes IFIN, an encoder–decoder that interleaves differentiable forward operators and learnable inverse modules at every scale, guided by a learnable spatially varying PSF field and a frequency-selective Wiener-like regularizer to maintain physics consistency under severe blur and shift variance. Across DiffuserCam, shift-variant lensless, and MultiWienerNet, IFIN achieves SOTA or near-SOTA reconstructions with improved robustness to noise and model mismatch.

### Strengths
- Introduces a multi-scale forward-inverse architecture (FSO $\leftrightarrow$ ISO) with a learnable spatially varying PSF field (ROI blending and multi-scale conditioning), enabling blind or weak-calibration reconstruction under shift variance.

- Thorough evaluation on DiffuserCam, shift-variant lensless, and MultiWienerNet, achieving state-of-the-art or near-SOTA performance, especially at peripheral FoV and under noise/model mismatch.

- Clear problem setup and figures (IFIB, FSO/ISO variants) with mathematical formulation, making design choices and the pipeline easy to follow.

### Weaknesses
This paper proposes an integrated forward–inverse network for physics-guided image reconstruction. While the approach is promising and shows gains on several benchmarks, there remain notable concerns:
- Limited novelty. The core design, feature-space deconvolution with a learnable regularizer embedded in a multi-scale encoder-decoder, closely resembles DWDN/MWDN and unrolled physics-guided pipelines (Dong et al., 2021; Li et al., 2023; Monakhova et al., 2019; Kingshott et al., 2022; Yanny et al., 2022; Poudel & Nakarmi, 2024). The paper should clarify what is fundamentally new beyond interleaving FSO/ISO at each scale and support this with ablations.

- Limited comparisons. The proposed model uses measurement consistency and cross-domain alignment losses, whereas retrained baselines use only image and perceptual losses; some baselines rely on calibrated PSFs without access to learned PSF fields. The authors are encouraged to retrain physics-guided baselines with matched loss terms and PSF priors/initializations and to report their own model without the extra losses to isolate architectural gains.

- Missing ablations: (a) learned ROI maps vs. fixed grids (and smoothness priors), (b) multi-scale PSF conditioning/resampling policy, (c) primary results with a fully shift-variant forward operator, (d) removal of measurement/cross-domain losses, (e) fixed vs. learnable $\epsilon(u,v)$, including sharing across scales/ROIs. These analyses are needed to substantiate design choices.

### Questions
Based on the weaknesses above, the following issues also require discussion:
- How sensitive are results to the number of PSFs $k=s^2$, ROI initialization (centers, $\sigma_r$), and smoothness of the ROI weights $w_r$? Are there seam artifacts at ROI boundaries?

- How are multi-scale PSF embeddings $h^{(n)}$ generated (down/upsampling policy, anti-aliasing, parameter sharing across scales)? Please detail the PSF encoder and provide ablations.

- The proposed model uses measurement-consistency and cross-domain alignment losses. How much of the gain derives from these terms? Could you retrain physics-guided baselines (e.g., MWNet, MWDN, UPDN) with matched losses and PSF priors/initializations to isolate architectural benefits?

- FSO uses zero padding, whereas ISO uses replicate padding with a Gaussian window. Does this discrepancy bias measurement consistency, especially under sensor truncation? Please unify the treatment and quantify its impact.

- For the shift-variant Lensless dataset, how robust are the results across different capture conditions (lighting, display spectra, geometry drift)? Are there scene types where IFIN fails?

### Soundness
2

### Presentation
3

### Contribution
2
