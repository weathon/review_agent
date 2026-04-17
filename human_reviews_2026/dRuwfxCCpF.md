# Focused Diffusion GAN: Object-Centric Image Generation Using Integrated GAN and Diffusion Frameworks

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Generative Adversarial Networks (GANs) and Diffusion Models (DMs) have shown significant progress in synthesizing high-quality object-centric images. However, generating realistic object-centric images remains challenging when training datasets are limited or contain degraded images (e.g., privacy-induced face blurring). Under these conditions, existing generative models frequently produce images that lack perceptual quality or exhibit overfitting to the training examples. To overcome these limitations, we propose a novel hybrid generative model, \textit{Focused Diffusion-GAN (FDGAN)}, targeting low-data object-centric regimes, which integrates a GAN discriminator directly into the diffusion model at intermediate denoising stages. Central to FDGAN is an Additional Noise Perturbation Module (ANPM) that selectively activates the GAN component only for images sufficiently denoised, ensuring the discriminator receives meaningful input. Additionally, ANPM applies targeted noise perturbations within predefined bounding-box regions, implicitly guiding the model’s focus toward key objects. FDGAN differs from other models like LayoutDiffusion, which explicitly conditions synthesis on fixed bounding-box layouts, or Diffusion-GAN and StyleGAN2-ADA, which employ noise augmentation throughout the entire training process, by combining adversarial training with targeted noise perturbations at specific intermediate diffusion steps. We evaluate FDGAN on three small object-centric datasets (Cityscapes subset, Traffic-Signs, and MS-COCO ``potted plant'') and, against strong GAN, diffusion, and object-centric baselines, show improved perceptual quality (Fréchet Distance) and reduced overfitting (Feature Likelihood Score). Ablation studies indicate that selective mid-timestep adversarial guidance together with ANPM improves the realism–overfitting trade-off in limited-data generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces FDGAN, a method that combines GAN and diffusion models for low-data, object-aware synthesis aimed at augmenting downstream object detectors such as YOLO and DETR. While the idea of integrating these models is potentially interesting, the paper fails to provide a clear big picture or logical foundation for the approach. It primarily focuses on implementation details without explaining the underlying principles, and the experimental validation does not adequately support the claims made in the introduction.

### Strengths
The attempt to merge GAN and diffusion models for data augmentation in low-data scenarios is a relevant and timely topic.

The paper presents a structured method with multiple loss functions, which could be a basis for further development.

### Weaknesses
Lack of Conceptual Clarity and Big Picture
- The paper does not sufficiently explain the core principles behind fusing GAN and diffusion models. For instance, it describes how the models are combined but fails to justify why this fusion is theoretically sound or beneficial. This omission makes it difficult to assess the novelty and contribution of the work.

Excessive Repetition in Citations
- The paper suffers from redundant citations, which reduce its readability and professionalism. For example, in the first paragraph of page 2, "Karras et al., 2020a" is cited four times. This indicates a need for better citation management to avoid clutter.

Insufficient Experimental Validation
- The introduction claims that FDGAN is an object-aware synthesizer for augmenting detectors like YOLO and DETR, but the experiments do not provide evidence to support this. There are no results demonstrating improved performance on downstream detection tasks, which undermines the paper's main motivation.

Methodological Justification
- The method section introduces three loss functions but does not explain the rationale for their selection or combination. Without a principled discussion of why these losses are chosen and how they interact, the approach appears ad hoc and lacks theoretical grounding.

### Questions
Can the authors provide a more detailed theoretical explanation for the fusion of GAN and diffusion models? 

How does the object-aware synthesis specifically benefit downstream detectors? The authors should include experiments that evaluate FDGAN's impact on detector performance (e.g., using metrics like mAP for YOLO or DETR) to validate the claims.

Please justify the combination of the three loss functions: what is the principle behind each loss, and how do they collectively contribute to the model's objectives?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Focused Diffusion-GAN (FDGAN), a hybrid generative model designed for object-centric image generation in low-data regimes. The key innovation is integrating a GAN discriminator into intermediate denoising stages of a diffusion model through an Additional Noise Perturbation Module (ANPM). ANPM selectively activates adversarial training at specific timesteps and applies targeted Gaussian noise within bounding-box regions to guide the model's attention toward objects. The authors evaluate FDGAN on three small datasets: Cityscapes-Pedestrian, Traffic-Signs, and MS-COCO potted plants, demonstrating improvements in perceptual quality and reduced overfitting compared to GAN-only, diffusion-only, and hybrid baselines.

### Strengths
- The selective integration of adversarial training at intermediate diffusion timesteps (t < t_early) is an interesting approach that differs from prior hybrid methods.
- Detailed ablation studies demonstrating the effectiveness of each component (GAN/ANPM, reconstruction losses, weighting schemes).

### Weaknesses
- The evaluation is restricted to only three small datasets, all at 256×256 resolution. The generalizability to other domains, higher resolutions, or multi-class scenarios remains unclear. 
- The main part of the method is performing GAN training on intermediate diffusion timesteps, which can be regarded as a hyper-parameter tuning. And the justification (theory / empirical investigation) is insufficient, resulting in limited novelty. 
- Although the BB noise is highlighted in the abstract, there is no ablation study on it.

### Questions
- See Weaknesses.
- Why use diffusion loss instead of consistency loss in training? I think the consistency loss aligns closer with the GAN, I feel strange about the usage of diffusion loss.

### Soundness
3

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
3

### Summary
This paper proposes Focused Diffusion-GAN (FDGAN), a hybrid generative model that integrates a GAN discriminator into a diffusion model at intermediate denoising stages. The method introduces an Additional Noise Perturbation Module (ANPM) that selectively activates the adversarial branch when samples are sufficiently denoised and applies localized noise within bounding-box regions to guide object-centric focus. The paper targets low-data object-centric regimes, evaluating on three small datasets (Cityscapes–Pedestrian, Traffic-Signs, COCO “potted plant”). Experimental results demonstrate improved perceptual fidelity and reduced overfitting compared to diverse baselines.

### Strengths
1. Task focus: The focus on limited-data, object-centric scenarios is well-motivated and practical (e.g., privacy-blurred faces, small datasets).
2. Comprehensive evaluation: Benchmarks include both GANs and DMs, using DINOv2-based metrics and traditional FID/Precision/Recall.

### Weaknesses
1. Marginal FID improvements: The proposed method performs worse than Diffusion-GAN on FID across all datasets.
2. Novelty scope: The hybridization of diffusion and GANs has been explored. The core novelty lies mainly in localized noise perturbation (ANPM) and timestep scheduling, which might be seen as incremental.
3. Effectiveness evidence: Since FDGAN aims to be "a low-data, object-aware synthesizer for augmenting downstream detectors (e.g., YOLO/DETR)", including downstream detection fine-tuning results would strengthen claims.

### Questions
1. How sensitive is FDGAN to the choice of the timestep threshold $t_\text{early}$ and noise strength $\gamma$ in ANPM?
2. Can the ANPM mechanism generalize to non-bounding-box settings (e.g., segmentation masks or text prompts)?
3. Sections 4.1 and 4.2 share the same table (Table 1) without explicit reference to it, and the order of the models in the table is chaotic. It is not friendly to performance comparison and analysis. Improvements are recommended.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript investigates how to enhance the quality of object-centric image generation when training data is limited(e.g. <3k) or contains degraded images. The authors propose a hybrid GANs-Diffusion framework that integrates a discriminator into the intermediate denoising steps of the diffusion process to improve visual fidelity. An Additional Noise Perturbation Module is also introduced to steer the model's focus toward predefined bounding boxes containing key objects. The proposed method has been experimentally validated on complex scene datasets—including Cityscapes-pedestrian, Traffic-Signs, and MS-COCO(Potted Plant)—demonstrating its effectiveness in generation tasks.

### Strengths
The research problem addressed in this manuscript—generation with limited data—is highly meaningful. The approach of integrating a GANs discriminator to enhance quality is well-justified, and the idea of leveraging bounding boxes to prioritize the generation quality of key objects is particularly suitable for complex scene generation. Experimental results demonstrate a clear improvement in generated quality compared to existing methods.

### Weaknesses
The experimental analysis appears somewhat fragmented and would benefit from consolidation and restructuring. The current evaluation is incomplete, as it fails to demonstrate the method's effectiveness in downstream tasks—particularly as data augmentation. Moreover, the study lacks intuitive assessments of generation quality, such as visual comparisons of generated images. Additionally, discussions and comparisons with existing methods in the field of generation with limited data are notably absent.

### Questions
1.The manuscript should discuss recent work on few-shot sample generation, which is highly relevant to the presented approach.

2.Several notation issues are present in Equations (7) and (8). For instance, the time step 't' is missing in Equation (8), and the origin of the variable x^is not defined.

3.Both the diffusion loss and the reconstruction loss pertain to reconstruction. Please clarify the distinct roles and motivations for including both terms in the objective function.

4.While the introduction claims that the method is intended for augmenting downstream detectors, no experiments are conducted to evaluate the utility of the generated samples in such downstream tasks.

5.How does the performance vary with different scales of training data (e.g., 100, 1,000 samples)? An analysis of the method's sensitivity to training set size is needed.

6.The experiments primarily follow a single-objective-per-dataset setting (e.g., pedestrians, traffic signs, potted plants). The applicability of the method to multi-object generation scenarios should be discussed, as this is critical for complex real-world applications.

### Soundness
2

### Presentation
2

### Contribution
2
