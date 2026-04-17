# CoDi: Subject-Consistent and Pose-Diverse Text-to-Image Generation

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Subject-consistent generation (SCG)-aiming to maintain a consistent subject identity across diverse scenes-remains a challenge for text-to-image (T2I) models. Existing training-free SCG methods often achieve consistency at the cost of layout and pose diversity, hindering expressive visual storytelling. To address the limitation, we propose subject-Consistent and pose-Diverse T2I framework, dubbed as CoDi, that enables consistent subject generation with diverse pose and layout. Motivated by the progressive nature of diffusion, where coarse structures emerge early and fine details are refined later, CoDi adopts a two-stage strategy: Identity Transport (IT) and Identity Refinement (IR). IT operates in the early denoising steps, using optimal transport to transfer identity features to each target image in a pose-aware manner. This promotes subject consistency while preserving pose diversity. IR is applied in the later denoising steps, selecting the most salient identity features to further refine subject details. Extensive qualitative and quantitative results on subject consistency, pose diversity, and prompt fidelity demonstrate that CoDi achieves both better visual perception and stronger performance across all metrics. The code is provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work focuses on subject-consistent generation. It preserves pose diversity by leveraging Identity Transport in the early stage of denoising and promotes subject consistency through Identity Refinement in the late stage of denoising. Both qualitative and quantitative experiments demonstrate the effectiveness of the proposed method.

### Strengths
- The paper is well written and easy to follow.
- The method proposed in the paper is training-free and can be directly applied during inference.
- The paper includes comprehensive comparative experiments and ablation studies. The design of evaluation metrics also demonstrates certain insights, especially regarding "pose diversity".

### Weaknesses
- Optimal Transport (OT) essentially addresses the optimization problem of transforming one probability distribution into another with minimal cost. However, the problem in **IT** is to find the feature matching relationship between the reference image and the target image. Clearly, a straightforward ranking based on cosine similarity would be simpler and more efficient. In contrast, the "globally optimal transport" property of OT not only complicates the problem but may also introduce redundancies. To illustrate this with a simple though imperfect analogy: The IT task aims for direct and semantically consistent matches. For instance, it seeks to align the "eye" feature in the reference image with the "eye" feature in the target image, and the "nose" feature with the "nose" feature. Introducing OT could disrupt this consistency. Suppose the "eye" feature in the reference image has a higher correlation than other features. Under OT’s global optimization logic, this could lead to the "eye" feature being inappropriately "transported" to the "nose" region in the target image. This outcome violates the need for direct and semantically meaningful feature matching. In summary, I argue that the introduction of Optimal Transport in this work serves more as a mathematical formality rather than a practically useful component.
- The design of Identity Refinement lacks innovation, as its feature fusion approach bears significant similarities to that of prior works.
- SDXL is built on the U-Net architecture, which is considered outdated in the current field. Most state-of-the-art base models now adopt the DiT architecture, and the effectiveness of the proposed method when applied to DiT-based models remains unproven.

### Questions
Given that current state-of-the-art base models such as XVerse, UNO, Flux-Kontext and Nano Banana have already achieved the three key objectives outlined in the introduction with excellent performance, and further support unlimited multi-subject consistent generation as well as strong generalization capabilities, what is the core significance of this work?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To solve the trade-off between subject consistency and pose diversity in text-to-image subject-consistent generation (SCG), the paper proposes the CoDi framework. Inspired by diffusion models’ progressive nature, this paper includes two stages: Identity Transport and Identity Refinement.  Extensive experiments have demonstrated the effectiveness of the model.

### Strengths
1. This work proposes an effective method to improve pose diversity.
2. This work is clearly expressed and easy to understand.
3. This work introduces Optimal transport into Subject-consistent generation.

### Weaknesses
1. This model was tested on SDXL, but its effectiveness was not verified on the DiT architecture.
2. The qualitative and quantitative experimental results of this work did not show significant improvement.
3. The long description in lines L126-L131 seems informal in the main text.

### Questions
1. This work was conducted on the SDXLl, which is a relatively old base model. Could the proposed method in this paper work on DiT model (e.g., FLUX and SD 3.5)?
2. In the experimental results shown in Figure 3(a) of this paper, the styles of the three scientists are not very consistent (and all are anime characters), and the portrait similarity is also lower compared to ConsiStory and StoryDiffusion. Will the proposed model in this paper affect the style?
3. Could this method generate images with consistent subject appearance and different style?

### Soundness
3

### Presentation
4

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
This study introduces a novel training-free framework for subject-consistent image generation. The framework aims to address the limitation of existing methods that sacrifice pose and layout diversity to maintain consistency. Its core design employs a staged processing strategy: during the early generation phase, optimal transport is utilized for coarse-grained identity feature transfer to preserve subject consistency; in the later generation phase, fine-grained refinement is applied to the image features. Experimental results demonstrate that the proposed method outperforms existing mainstream approaches across three key metrics—subject consistency, pose diversity, and text alignment—achieving state-of-the-art performance on public benchmarks.

### Strengths
1.The key innovation is the explicit decoupling of identity alignment into coarse-grained transport in early steps and fine-grained refinement in later steps, which is a well-motivated approach based on the progressive nature of diffusion models.
2.A significant strength is the superior balance it achieves. As claimed, the paper provides strong evidence that the method outperforms existing training-free baselines in subject consistency while preserving significantly greater pose diversity and text alignment, addressing a well-known trade-off in the field.

### Weaknesses
1.For long-story generation scenarios, it is crucial to maintain consistency in both character identity and their apparel. However, the results presented in the paper demonstrate that the method primarily ensures identity consistency, while the consistency of clothing remains inadequate. In my opinion, this limitation would significantly restrict the method's practicality in long-story applications.

2.The method has a core reliance on binary masks derived from cross-attention maps to extract identity information from the reference image. However, recent and emerging generative models, such as SD3 and other DiT-based architectures, have moved away from using cross-attention mechanisms. This fundamental incompatibility means that the proposed facial extraction technique faces significant challenges in being adapted to these mainstream, state-of-the-art generative models, thereby limiting its generalizability and future relevance.

### Questions
1.How is the iterative generation performance? Can it maintain identity consistency?
2.Why does the style of the generated images vary? Since style should also be correlated with features from certain layers of the U-Net, why do the other two models (ConsiStory and StoryDiffusion) not exhibit style changes (based on the observation from Figure 3 in the paper)? Could this limitation affect the model's practical applicability?

### Soundness
3

### Presentation
3

### Contribution
3
