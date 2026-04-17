# Controllable Video Generation with Provable Disentanglement

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Controllable video generation remains a significant challenge, despite recent advances in generating high-quality and consistent videos. Most existing methods for controlling video generation treat the video as a whole, neglecting intricate fine-grained spatiotemporal relationships, which limits both control precision and efficiency. In this paper, we propose \textbf{Co}ntrollable \textbf{V}ide\textbf{o} \textbf{G}enerative \textbf{A}dversarial \textbf{N}etworks (\ourmes) to disentangle the video concepts, thus facilitating efficient and independent control over individual concepts. Specifically, following the \textbf{minimal change principle}, we first disentangle static and dynamic latent variables. We then leverage the \textbf{sufficient change property} to achieve component-wise identifiability of dynamic latent variables, enabling independent control over motion and identity. To establish the theoretical foundation, we provide a rigorous analysis demonstrating the identifiability of our approach. Building on these theoretical insights, we design a \textbf{Temporal Transition Module} to disentangle latent dynamics. To enforce the minimal change principle and sufficient change property, we minimize the dimensionality of latent dynamic variables and impose temporal conditional independence. To validate our approach, we integrate this module as a plug-in for GANs. Extensive qualitative and quantitative experiments on various video generation benchmarks demonstrate that our method significantly improves generation quality and controllability across diverse real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CoVoGAN, a GAN-based framework for controllable video generation with provable disentanglement between content (static) and motion (dynamic) factors. the authors propose to add a light Temporal Transition Module (GRU + component-wise flow) in front of a StyleGAN-like generator, plus a video discriminator and an MI term to tie dynamics to what is being produced.

### Strengths
- The authors provided solid theory to back their claims
- The paper is well-written and easy to follow

### Weaknesses
- The authors provide a disentanglement comparisons between all method, yet it is likely favor methods with compact latents (i.e Diffusion models vs GANs)
- while the author demonstrate independence between content and motion, I could not find any table quantifying this improvement/ 
- The method is based of GANs. However, due the popularity of diffusion models recently, the contribution on GANs might be undermind.

### Questions
See above.

### Soundness
3

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
2

### Summary
This paper introduces CoVoGAN, a video generation model that achieves fine-grained control through theoretically-grounded disentanglement. The approach separates videos into static content (identity) and dynamic factors (motion), with component-wise separation of individual motion concepts (e.g., head movement vs. eye blinking). Experiments across multiple datasets show superior video quality and controllability compared to baselines, enabling precise independent manipulation of different video aspects.

### Strengths
1. This paper provides strong theoretical foundation. It provide rigorous identifiability theorems for video generation, offering mathematical guarantees for disentanglement at both block-wise and component-wise levels.

2. This paper enables independent manipulation of specific motion concepts (e.g., eye blinking, head movement) without affecting content/identity, demonstrated across multiple datasets.

3. The evaluation is thorough across different metrics and datasets.

### Weaknesses
1. The paper assumes a strong assumption: videos decompose cleanly into content (which is static) and dynamic motion. However, many real-world vidoes don't fit this dichotomy. For example, deformable objects have constant but moving parts. The identity and motion are entangled. Lighting changes, and shadow moving cannot fit into these two categories either. I am concerned about the limited generalizability of the proposed method to broader video domains.

2. Performance on complex human motion (TaiChi dataset) shows room for improvement, suggesting challenges with highly dynamic scenes.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose Controllable Video Generative Adversarial Networks (CoVoGAN), an approach designed to disentangle video concepts to facilitate efficient and independent control over individual factors. CoVoGAN is built upon rigorous theoretical insights derived from nonlinear Independent Component Analysis (ICA), establishing conditions for identifiability. It used to achieve component-wise identifiability of the dynamic latent variables, allowing for disentangled control within the motion itself. It proposed the Temporal Transition Module (TTM), integrated as a plug-in into a GAN architecture (based on StyleGAN2-ADA). Extensive experiments across various datasets (including FaceForensics, SkyTimelapse, RealEstate, and CelebV-HQ) demonstrate that CoVoGAN improves both generative quality (measured by FVD) and controllability (measured by SAP, modularity, and MCC) compared to existing video generation models.

### Strengths
1. The model successfully achieves both block-wise identifiability (separating content from motion) and component-wise identifiability (separating distinct factors within motion).
2. CoVoGAN achieves the best performance across disentanglement metrics (MCC, SAP, Modularity) compared to strong baselines like StyleGAN-V, MoStGAN-V, LVDM, and Latte. 
3. CoVoGAN is computationally efficient, having a generator parameter count (24.98M) comparable to StyleGAN2-ADA and significantly smaller than other baselines.

### Weaknesses
1. The primary implementation and theoretical guarantees are grounded in the GAN framework, how to integrate the idea into current diffusion-based models? Compared to diffusion models, GAN-based approaches still have large visual fidelity gap.
2. Experiments only focus on domain-specific datasets, how will be the performance of CoVoGAN on UCF101 or even larger dataset such as Kinetics? It would be better if the author could show the abilities of the model on more complex datasets.
3. Can CoVoGAN be used in long-term video generation?

### Questions
see weaknesses

### Soundness
2

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
2

### Summary
This paper presents Controllable Video Generative Adversarial Networks , a novel framework designed to address the challenge of fine-grained control in video generation by leveraging provable disentanglement. The authors introduce a Temporal Transition Module that separates static content elements and dynamic style dynamics, grounded in theoretical guarantees of block-wise and component-wise identifiability. By integrating TTM into a GAN architecture, CoVoGAN achieves disentangled representations through the "minimal change principle" and "sufficient change property," enabling independent manipulation of motion components like head movement and eye blinking. Extensive experiments on datasets including FaceForensics and RealEstate demonstrate superior performance in both generation quality  and controllability  compared to baselines.

### Strengths
The paper provides a strong theoretical foundation by proving block-wise and component-wise identifiability under mild assumptions. This formalizes why disentanglement between motion and content is achievable, addressing gaps in prior heuristic-based methods. The TTM effectively combines GRU for temporal dependency modeling and Deep Sigmoid Flow for conditional independence, ensuring dynamic variables are disentangled and controllable. Ablation studies validate the critical role of GRU and flow components in maintaining performance.

### Weaknesses
While CoVoGAN excels in unsupervised disentanglement, it does not address how to align latent dimensions with semantic labels (e.g., explicitly mapping a dimension to "eye blinking"). Integrating weak supervision or text guidance could enhance interpretability and usability. 
Additionally, the article lacks experiments on the UCF101 dataset, and it is recommended to supplement them.

### Questions
The method in this paper belongs to unconditional video generation. Can it be applied to text-to-video generation models? The latter has greater practical application value.

### Soundness
3

### Presentation
3

### Contribution
2
