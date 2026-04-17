# ART-VITON: Measurement-Guided Latent Diffusion for Artifact-Free Virtual Try-On

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Virtual try-on (VITON) aims to generate realistic images of a person wearing a target garment, requiring precise garment alignment in try-on regions and faithful preservation of identity and background in non-try-on regions. While latent diffusion models (LDMs) have advanced alignment and detail synthesis, preserving non-try-on regions remains challenging. A common post-hoc strategy directly replaces these regions with original content, but abrupt transitions often produce boundary artifacts. To overcome this, we reformulate VITON as a linear inverse problem and adopt trajectory-aligned solvers that progressively enforce measurement consistency, reducing abrupt changes in non-try-on regions. However, existing solvers still suffer from semantic drift during generation, leading to artifacts. We propose $\textsf{ART-VITON}$, a measurement-guided diffusion framework that ensures measurement adherence while maintaining artifact-free synthesis. Our method integrates residual prior-based initialization to mitigate training-inference mismatch and artifact-free measurement-guided sampling that combines data consistency, frequency-level correction, and periodic standard denoising. Experiments on VITON-HD, DressCode, and SHHQ-1.0 demonstrate that $\textsf{ART-VITON}$ effectively preserves identity and background, eliminates boundary artifacts, and consistently improves visual fidelity and robustness over state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes ART-VITON, a measurement-guided latent diffusion framework for artifact-free virtual try-on. The central idea is to recast virtual try-on as a linear inverse problem and introduce a latent diffusion-based inverse solver that incorporates measurement adherence progressively during the sampling process. ART-VITON combines residual prior-based initialization, artifact-free measurement-guided sampling utilizing data consistency and frequency-level correction, and periodic standard denoising.

### Strengths
1. Clear Motivation: The paper addresses a well-recognized, often neglected issue in virtual try-on—preserving non-try-on regions and avoiding boundary artifacts between try-on and non-try-on areas. 
2 Sufficient Ablation Study: Ablation studies (Table 5 & 6, Figure 5) provide convincing evidence for their necessity and complementary effects.

### Weaknesses
1. The baseline methods does not cover the latest approaches like IDM-VTON[1] or OOTDiffusion[2].
[1] Improving diffusion models for authentic virtual try-on in the wild.
[2] Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on.

2. Limited theoretical novelty in inverse solvers.  Although reducing abrupt changes innon-try-on regions is useful in Try-on, the core of the proposed high-frequency correction is to restrict the editing region with the mask, which is a common technique. It would be much better if the idea of partial editing in LDM is validated in more complex tasks.

3. The overall quality is not good enough for ICLR.

### Questions
See weakness.

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
5

### Summary
This paper addresses the challenge in Virtual Try-On (VITON) of preserving non-try-on regions and eliminating boundary artifacts in latent diffusion models by reformulating the problem as a linear inverse problem. 

The authors propose ART-VITON, a measurement-guided diffusion framework that ensures measurement adherence using residual prior-based initialization and an artifact-free sampling strategy. 

Experiments demonstrate that ART-VITON effectively preserves identity and background, eliminates boundary artifacts, and consistently improves visual fidelity and robustness over state-of-the-art baselines.

### Strengths
1. Reformulates the virtual try-on task as a **linear inverse problem**, offering a principled approach to handle measurement consistency and reduce boundary artifacts.
2. Introduces a **measurement-guided diffusion framework** with residual prior initialization and frequency-level correction to ensure artifact-free, identity-preserving synthesis.
3. Demonstrates consistent performance gains and visual improvements across multiple benchmarks (VITON-HD, DressCode, SHHQ-1.0) compared to state-of-the-art methods.

### Weaknesses
1. Several implementation details remain unclear. For instance, Line 64 mentions *“violates measurements (M)”*—it is not explained how **M** is defined or computed. The example showing *t = 835* also lacks justification for this choice, and results for other timesteps are not discussed. Moreover, training details and computational overhead compared to baselines are missing, making it difficult to assess reproducibility and efficiency.

2. The claim of producing *“artifact-free”* results seems overstated. In Figure 1, minor artifacts are still visible, even if reduced compared to baselines. Similarly, Figures 4(b), 8, 9, and 10 show that improvements over baselines are sometimes marginal, suggesting the enhancement is not consistently substantial.

3. The evaluation lacks comparisons with **recent diffusion transformer-based VITON methods**, such as ITA-MDT [1] and IMAGDressing-v1 [2], which would provide a fairer and more comprehensive validation of the proposed plug-and-play module.

[1] *ITA-MDT: Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On*, CVPR 2025

[2] *IMAGDressing-v1: Customizable Virtual Dressing*, arXiv:2407.12705

### Questions
See weaknesses

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
This paper proposes ART-VITON, a framework for virtual try-on aimed at resolving the boundary artifacts produced by existing methods when preserving non-try-on regions. The paper reformulates VITON as a linear inverse problem, seeking to reconstruct the full image conditioned on the given non-try-on regions (the "measurement"). The authors design an "artifact-free measurement-guided sampling" strategy that progressively guides the LDM's sampling trajectory during reverse diffusion to match these measurements.

### Strengths
The artifact-free measurement-guided solver combines multiple complementary techniques: (i) a residual prior-based initialization to mitigate the training-inference mismatch , (ii) data consistency to maintain semantic coherence , (iii) frequency-level correction to restore high-frequency details lost during VAE encoding , and (iv) periodic standard denoising to harmonize regions. Furthermore, the framework is model-agnostic, meaning it can be applied to various existing LDM-based VITON models without retraining, consistently improving their performance and demonstrating strong generalization on multiple datasets.

### Weaknesses
My primary concern stems from the paper's comparison against post-hoc replacement. The authors motivate their complex measurement-guided sampling framework (ART-VITON) by highlighting the "abrupt transitions" and "boundary artifacts" produced by standard "post-hoc replacement". However, the "post-hoc replacement" method described appears to be a naive pixel-wise copy-paste. To my knowledge, methods like CAT-DM[1] have effectively utilized Poisson Blending as a "post-hoc replacement"to resolve these exact artifacts. The authors should therefore discuss blending techniques, especially Poisson Blending, to situate their contribution more accurately.

Furthermore, the proposed sampling method seems extremely complex. Each sampling step involves: (1) a VAE decode and encode (Eq. 5), (2) a latent-space optimization (Eq. 6), and (3) both a Fourier Transform and an Inverse Fourier Transform (Eq. 7). This process is almost certainly much slower than standard DDIM sampling or even other solvers like TReg. The authors should include a discussion of the model's inference time and computational overhead.

[1] Zeng J, Song D, Nie W, et al. Cat-dm: Controllable accelerated virtual try-on with diffusion model[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024: 8372-8382.


Finally, Figure 2, as the main flowchart for the method, is overly dense and difficult to understand.

### Questions
Please address the issues raised in the weaknesses section. Including Poisson blending and inference time in the analysis would help address my concerns.

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
4

### Summary
This paper considers virtual try-on. The focus of this paper is on the improvement in generation of the non-garment regions that should be preserved. The proposed approach adapts the intermediate generated image so that the inpainted and existing regions are mode continuous. At regular intervals during the diffusion process, the predicted latent $\hat{z}_0^{(t)}$ image is decoded and fused with the masked model image (the masked region comes from the decoded image, the non-masked region comes from the conditioning image). This image is again encoded into a latent $\hat{z}_y$. To insure consistency, interpolation between $\hat{z}_y$ and  $\hat{z}_0^{(t)}$ is done and a further frequency correction is done to preserve detail. This process is interchangeably done with regular diffusion steps for a more stable diffusion process.

### Strengths
[S1] The proposed approach is inference-time, and may therefore be used to improve existing VTON pipelines. 

[S2] Good quantitative and qualitative results.

### Weaknesses
[W1] Missing ablations for important choices in the method (e.g. the strength of interpolation in equations 6 and 7, the number of steps for C).

[W2] Prior-based initialization is not novel and I am not completely certain how it relates to reformulating VTON as inverse problem and equation 4.

[W3] I thought that the paper is badly written and difficult to follow. See questions. I also thought some of the naming/symbol conventions made the paper more difficult to follow. For example, naming the masked area as a measurement. A measurement would imply some sort of injection of physical constraints into to model. Similarly, tradinally in VTOM paper M indicates the masked region that should be inpainted, but here it is reversed.

### Questions
[Q1] What is being shown by the Figure 3? Is the determination of artefacts quantitative and measurable or just qualitative? How is failure/success determined?  

[Q2] Some elements of the method are not completely clear, e.g. how is $\hat{z}'_y$ obtained?

### Soundness
3

### Presentation
2

### Contribution
3
