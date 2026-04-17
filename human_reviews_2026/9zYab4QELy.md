# OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Emerging deep-learning-based lens library pre‑training (LensLib-PT) pipeline offers a new avenue for blind lens aberration correction by training a universal neural network, demonstrating strong capability in handling diverse unknown optical degradations. This work proposes the OmniLens++ framework, which resolves two challenges that hinder the generalization ability of existing pipelines: the difficulty of scaling data and the absence of prior guidance characterizing optical degradation. To improve data scalability, we expand the design specifications to increase the degradation diversity of the lens source, and we sample a more uniform distribution by quantifying the spatial-variation patterns and severity of optical degradation. In terms of model design, to leverage the Point Spread Functions (PSFs), which intuitively describe optical degradation, as guidance in a blind paradigm, we propose the Latent PSF Representation (LPR). The VQVAE framework is introduced to learn latent features of LensLib's PSFs, which is assisted by a PSF-conditioned regularizer modeling the optical degradation process to constrain the learning of degradation priors. Experiments on diverse aberrations of real-world lenses and synthetic LensLib show that OmniLens++ exhibits state‑of‑the‑art generalization capacity in blind aberration correction. Beyond performance, the AODLibpro is verified as a scalable foundation for more effective training across diverse aberrations, and LPR can further tap the potential of large‑scale LensLib. The source code and datasets will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an image restoration method for optical aberration removal that generalizes well across diverse optical lenses. By overfitting existing optical aberrations (PSFs) using a large-scale lens dataset, the network is exposed to a variety of aberrations, thereby improving generalization performance. The key improvement of this work compared to the prior arXiv paper (OmniLens) lies in its scalability. To achieve this, the paper expands the lens design specifications and introduces a latent representation of the PSF.

### Strengths
I agree that training an “all-in-one” network for various lens aberrations is an interesting academic exploration, but which requires a more comprehensive benchmark.

### Weaknesses
1. Regarding the paper's writing, two points need improvement:
  - The baseline for this paper is the arXiv paper “OmniLens: Towards Universal Lens Aberration Correction via LensLib-to-Specific Domain Adaptation.” However, the OmniLens framework is not clearly described in the main text. Readers may also be confused about whether there is an "OmniLens+" between OmniLens and OmniLens++.
  - There are too many uncommon abbreviations, for example, OD for optical degradation, LPR for latent PSF representation, CAC for computational aberration correction, and OIQ for optical image quality. These abbreviations are usually unnecessary and significantly increase reading difficulty.

2. Regarding the experiments:
  - The baseline (OmniLens) remains on arXiv, and this paper is a clear follow-up to it. Benchmarking against an un-peer-reviewed work is risky and may be hard to convince readers. The evaluation cases and dataset are highly customized by the authors, whereas a more comprehensive, standardized benchmark should be proposed.
  - Similar to Chen et al., this paper only considers plane object scenes. However, in real scenarios, different objects appear at different depths, introducing non-uniform defocus effects. The corresponding defocus datasets and simulation works should not be ignored. Examples include:
    - “Defocus Deblurring Using Dual-Pixel Data”
    - “Learning to Deblur Using Light Field Generated and Real Defocused Images”
    - “Aberration-Aware Depth-from-Focus”
    - “Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur”

3. Debate on “all-in-one” image restoration:
  - I agree that training an “all-in-one” network for various lens aberrations is an interesting academic exploration, and there are related works on this topic for image denoising, motion deblurring, and deraining. However, different optical lenses can exhibit very different aberrations. Training on such a large-scale lens dataset may improve network performance on general camera lenses, but the results on metalenses (Figure 3) are not promising enough. In short, it is challenging to achieve zero-shot generalization to customized optics if they are not included in the dataset—this seems like a clear failure case for the overall idea of this paper. Examples include:
    - “Perspective-Aligned AT Mirror with Under-Display Camera”
    - "Removing Diffraction Image Artifacts in Under-Display Camera via Dynamic Skip Connection Networks"
  - In practice, different camera sensors have distinct noise profiles, which often require image restoration networks to be retrained or fine-tuned for different camera systems. As the dataset scale increases, overfitting all existing PSFs becomes significantly more difficult and demands larger network sizes and more complex architectures, posing challenges for deployment. In such cases, the practicality of an “all-in-one” image restoration network is marginal.

4. Regarding novelty: 
  - The original idea of OmniLens is new, while scaling it up (e.g., by using aspherical surfaces, which is a standard approach) seems marginal.

### Questions
Please check weaknesses

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
This work proposes OmniLens++, a framework for blind computational aberration correction build on Lens Library Pre-Training (lensLib-PT). 
The authors claim two main contributions. First, they introduce a larger, better-balanced library (AODLibpro) that expands the design space with aspheric surfaces and image-plane perturbations to broaden degradation diversity. 
Second, they propose a Latent PSF Representation (LPR) that injects PSF knowledge into a blind correction pipeline. 
Using LPR as guidance, the authors train a foundational aberration-correction model (FoundCAC). 
Across RealLens-Sim and real photographs, OmniLens++ delivers consistent improvements over prior LensLib-PT variants and strong deconvolution baselines.

### Strengths
Combining a broadened, more uniformly sampled synsthetic lens library with a learned latent prior is technically sound and novel.
The experimental validation is extensive, spanning diverse simulated aberration settings, and includes ablation studies showing that AODLibpro improves over earlier AODLib-EAOD settings and LPR guidance helps as data scale increases.

### Weaknesses
The presentation is needlessly hard to parse. The paper is overloaded with abbreviations (OD, CAC, ODN, OIQ, etc.), many of which describe concepts that could be expressed directly or formally. Optical degradation (OD) is essentially a forward operator in an inverse problem. Computational Aberration Correction could simply be described as image reconstruction and an Optical Degradation Network is, at its core, a neural network modeling the forward process. This naming density obscures rather than clarifies the contributions.
The method section is dense and very hard to follow. It introduces multiple interdependent modules (VQVAE, ODN, LPR, FoundCAC) in quick succession, often without intuitive explanation or guiding diagrams. Figures 1 and 2 are overly complex and fail to provide a clear overview of the system. They require significant prior knowledge of the text to decode. They are repeatedly cited as explanatory ("as shown in Figure 1 (a)...", "as shown in Figure 2 (a)...") when they in fact demand the textual explanation to be understood. Key design flows and data construction steps should be broken into simpler, sequential diagrams. The baseline OmniLens method is introduced too late in the text. Readers unfamiliar with the earlier work will struggle to contextualize what is actually new. The phrasing "constructing the large LensLib AODLibpro" reads as though the dataset already existed. Clarifying that AODLibpro is newly proposed and constructed in this work would strengthen the narrative. The writing, particularly in the methodology section, reads like a technical report rather than a scientific paper. Many sentences could be made clearer by explaining the motivation and intuition behind each modeling choice.

### Questions
- The Optical Image Quality (OIQ) metric is central to AODLibpro and the evaluation, but its motivation and validation are unclear. Could the authors justify why OIQ is needed for assessing blind lens aberration correction and show that it correlates better with perceptual or optical quality than PSNR or SSIM?
- The experiments seem simulation-heavy. Are there quantitative evaluations on real aberrated captures, not just qualitative examples?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces OmniLens++, a new framework for blind lens aberration correction that leverages a large-scale LensLib pre-training pipeline and a Latent PSF Representation (LPR) module. This paper aims to overcome two limitations in existing approaches: insufficient scalability of optical degradation data and the lack of explicit degradation priors in blind correction models. To address these, the authors construct AODLibpro, a uniformly sampled lens library with enriched optical specifications, and design an LPR-guided CAC (Computational Aberration Correction) model using a VQVAE-based latent PSF codebook and an Optical Degradation Network for prior regularization. Extensive experiments on synthetic and real lenses demonstrate state-of-the-art performance and strong generalization to unseen aberrations. While the technical pipeline is well-executed and the experimental evaluation is compelling, the paper would benefit from a more 
thorough explanation of the motivation behind the latent PSF representation and the specific advantages of the VQVAEODN combination.

### Strengths
The paper presents a technically solid and conceptually novel contribution by linking large-scale lens data construction with degradation-prior learning in a blind setting. The experimental evaluation is comprehensive and compelling, demonstrating state-of-the-art performance across both simulation and real-world benchmarks. The writing is clear and well-structured, making the technical contributions straightforward to follow.

### Weaknesses
Although the framework is comprehensive and experimentally solid, I find the overall technical novelty is relatively limited. 
The proposed VQVAE-based design represents a straightforward structural adaptation rather than a fundamentally new methodological contribution. While the integration of the Optical Degradation Network (ODN) to regularize the latent space is a thoughtful addition, it largely follows standard practice in generative modeling and degradation-aware representation learning. The author should provide more detailed discussion or ablation study.

### Questions
1. Could the authors provide a more detailed ablation study on the hyperparameter settings? For example, on page 
17, the LPR module is described as using a VQVAE with a codebook size of K=1024 and a latent feature 
dimension of n_z=256. It would be helpful to understand how these choices affect model performance and whether 
the results are sensitive to variations in these parameters.

2. While the paper demonstrates strong generalization across various lens types, it remains unclear how OmniLens++ 
performs under more extreme imaging conditions, such as low-light environments or high dynamic range scenes.

3. In line 362, the authors mention the suppression of purple fringing caused by chromatic aberration. Could they 
provide a more detailed explanation of how this is achieved? This would help reviewers and readers better 
understand the model’s capability in handling chromatic aberrations and its implications for real-world applications.

### Soundness
3

### Presentation
4

### Contribution
3
