# CLoSeR: Continual Learning in VQ-GAN for Test-time Style Refinement

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
While existing artistic style transfer methods enable cross-domain image synthesis, they often struggle to strike a balance among stylistic realism, inference efficiency, and geometric consistency. To address this limitation, we propose a test-time refinement (TTR) framework that universally enhances stylistic fidelity through a self-supervised VQ-GAN, without requiring any gradient updates to the pre-trained generator. Our primary contribution is a continual learning framework for VQ-GAN, which combines Low-Rank Adaptation (LoRA) with incremental codebook expansion.  This design enables efficient adaptation to diverse artistic styles while preserving previously learned knowledge, significantly reducing the computational and memory overhead of deploying models across multiple domains. Notably, our approach reduces the number of trainable parameters by up to 94% compared to full-model fine-tuning, offering a highly parameter-efficient solution for test-time refinement. Furthermore, we introduce positional embeddings into the latent embedding space, which strengthens the model's geometry awareness and improves structural coherence in the generated results. We name our approach CLoSeR (Continual Learning in VQ-GAN for Style Refinement), and evaluate it across multiple style transfer benchmarks under a test-time adaptation setting. Experimental results show that CLoSeR improves style fidelity and structural consistency, achieving a maximum relative reduction of 44% in Fréchet Inception Distance (FID), demonstrating significant gains in generation quality. The code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an universal test-time-refinement (TTR) framework for improving generation quality and style alignment of image artistic style transfer (AST). Specifically, this method employs a self-supervised VQ-GAN model as a post-processing module which is trained to reconstruct the input, effectively enhancing the outputs from any AST generator. To adapt to a new artistic style, this VQ-GAN model can be trained at test-time through continual learning via LoRA and incremental codebook expansion. To enhance the geometric prior of the model, the quantization in VQ-GAN is augmented with 2D positional embeddings. Together, the proposed method CLoSeR can be used as an universal post-processing model for any AST models including neural style-transfer and diffusion-based stylized image generation.

### Strengths
- The proposed method is efficient and is orthogonal to other AST model designs. As presented in Section 4.2.3 and Table 1, CLoSeR contains only 4.74 MB trainable parameters. Notably, it takes 0.055 seconds for inference, which is negligible compared to those generators with diffusion models.
- The effectiveness of CLoSeR is demonstrated through extensive experiments on different AST generators including neural style-transfer and diffusion-based stylized image generation. CLoSeR shows consistent improvements on multiple quantitative metrics.
- The proposed framework is robust against novel artistic styles outside the VQ-GAN training data. The VQ-GAN can be trained on the new style data in a self-supervised manner. Moreover, this continual learning process is implemented through LoRA and incremental codebook updates, enhancing the efficiency of the proposed method.
- The ablation experiments’ configuration is reasonable.

### Weaknesses
- Despite the constant improvements of CLoSeR on different generators, its performances rely on the baseline. As presented in Figure 3, CLoSeR on more powerful base generators like AttenDis has better quantitative metrics.
- The connection between a VQ-GAN refiner (for improving style fidelity) and continual learning framework is somehow weak. The improved geometric consistency in CLoSeR is justified through ablation, while the benefit of continual learning in this task is unclear. According to my understanding, the refiner is simply trained to ‘overfit’ to one artistic style at a time.
- There are some typos in equation 2.

### Questions
- The quantization in VQ-GAN is known to introduce some discretization error in reconstruction, potentially suffering the risk of reduced variety in the generation outputs. How could the proposed method handle the reconstruction error in VQ-GAN?
- Is it possible to let the CLoSeR refine the outputs for multi rounds? Like using the first refinement output as the input to the second round and so on.

### Soundness
3

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
4

### Summary
This paper proposes a lightweight test-time refinement framework, called CLoSeR, that improves style fidelity and geometric consistency in artistic style transfer. CLoSeR consists of three key components: LoRA-based continual adaptation that injects trainable low-rank matrices to modulate features in a style-specific manner, codebook expansion that enables the model to encode style-specific visual primitives while preserving previously learned representations, and positional encoding that enables geometry-aware refinement without introducing any additional learnable parameters.

### Strengths
1. The proposed method can be applied to current style transfer approaches to enhance the quality of the generated images.
2. This paper is well-written and well-organized.
3. Extensive experiments are conducted to evaluate the performance of the proposed method.

### Weaknesses
1. As shown in Figure 4, while the method proposed in this paper offers some improvement over the original baseline, the difference is not significant, and the extent of the improvement is relatively modest and not very satisfactory.

2. Many state-of-the-art style transfer methods are not introduced and compared in this paper, such as SaMam [1], HIS [2], OmniStyle [3], and StyleSSP [4]. \
[1] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer. CVPR 2025. \
[2] HSI: A Holistic Style Injector for Arbitrary Style Transfer. CVPR 2025. \
[3] OmniStyle: Filtering High Quality Style Transfer Data at Scale. CVPR 2025. \
[4] StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer. CVPR 2025.

3. In the qualitative results shown in Figure 5, the effects under the three settings—+ Vanilla VQ-GAN, + VQ-GAN PE, and + CLoSeR—are very similar, with no significant differences. If the differences are too minor, they are not significant enough to impact the final generated quality in a meaningful way. This raises my concerns about the necessity and effectiveness of these components.

4. User study is an important evaluation metric commonly used in style transfer, as it helps assess user preferences for different generated results. It is recommended that the authors include experiments involving a user study.

### Questions
Please see **Weaknesses**.

Others:

Section 4.1 of the paper mentions, 'All images are resized to 256 × 256 before training and evaluation.' Does this imply that all the experiments were conducted at a 256 × 256 resolution? Could this resolution be too low? Why wasn't a higher resolution used for the experiments?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a method of artistic style transfer by adapting VQ-GAN with test-time style refinement. In particular, the whole framework combines LoRA with incremental codebook expansion for VQ-GAN. In addition, positional embedding is also introduced into the latent embedding space to improve geometry awareness and structural coherence in the stylization process. Experimental results validated the effectiveness of the proposed method.

### Strengths
(1) The motivation is well presented of combining LoRA with incremental codebook expansion for test-time style refinement.

(2) The explanations and illustrations are mostly clear and intuitive of the test-time refinement framework, the codebook expansion, the positional embedding in the latent space.

### Weaknesses
(1) The contribution is limited. In particular, the core efficiency improvement comes from LoRA. However, there are no results of only using LoRA for AST with VQ-GAN.

(2) Considering AST itself, there are many works that leverage pretrained T2I models which show impressive stylization results and in a any-to-any manner without any case-by-case training. There is no discussion on this line of works and the proposed method is not effective and generable compared to them.

(3) On Table 3, the ablation study of with or without positional embedding cannot validate the effectiveness since the measure gap is not significant.

(4) On Page 1, Line 015-016, the claim of “without requiring any gradient updates to the pre-trained generator” is confusing. In particular, the generator’s LoRA training requires gradient updates from the objective function including the discriminator loss.

(5) Besides, from the computational aspect, the refinement LoRA is added to the generator during inference thus has the same inference cost. The only benefit seems to be lower training cost, however, inherits from LoRA which is not the contribution of this paper.

### Questions
No.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
CLoSeR is a test-time refinement framework that bridges the distributional gap between coarse stylized outputs and authentic target artistic domains without retraining the generator. Motivated by persistent mismatches in style fidelity and geometric consistency observed across GAN-, attention-, and diffusion-based translation models, CLoSeR leverages VQ-GAN as a domain anchor: it reconstructs generated images in the embedding space by aligning their features to a pre-learned target representation in the latent codebook. To make adaptation scalable across multiple styles, CLoSeR introduces continual learning via Low-Rank Adaptation (LoRA) and codebook expansion, reducing trainable parameters by over 94% compared to full fine-tuning while preserving prior styles. Additionally, it augments vanilla VQ-GAN with 2D sine-cosine positional embeddings to inject spatial awareness into the codebook and decoder, improving geometric consistency. As illustrated by t-SNE feature visualizations and FID comparisons, CLoSeR refines outputs from base stylization models (e.g., StyleID) to better align with target domains, achieving high-fidelity stylization and robust structure across diverse artistic styles.

### Strengths
- High fidelity without retraining the generator: Performs test-time refinement in VQ-GAN’s embedding space, avoiding costly generator fine-tuning while closing the distribution gap to the target artistic domain.
- Strong alignment to target style distribution: Uses VQ-GAN as a domain anchor to pull coarse stylized outputs toward a pre-learned target codebook, improving style fidelity (lower FID) and visual coherence.
- Minimal overhead: Incorporates LoRA and codebook expansion to incrementally adapt to new styles, reducing trainable parameters by >94% vs. full fine-tuning and preserving knowledge of prior styles.
- Scalable to multiple domain and models: Can refine outputs from various generators (GAN-, attention-, diffusion-based, e.g., StyleID), making it broadly applicable across artistic domains.

### Weaknesses
- Dependence on a high-quality VQ-GAN anchor. The refinement quality hinges on how well the VQ-GAN codebook captures the target domain. If the target style is underrepresented or highly diverse, the codebook may induce over-smoothing or mode bias. More studies on sensitivity (to codebook size, training data coverage, and codebook learning objectives) are necessary.
- Generalization: Although “plug-and-play,” performance may vary with the artifacts of different base models (GAN vs. diffusion vs. attention-based). Certain artifacts may be hard to correct by codebook alignment alone. Please consider to ablate across diverse generators and degradation modes.
- Robustness to domain shifts and OOD inputs: If test images fall outside the codebook’s learned manifold, refinement may produce artifacts or collapse. Stress-test with OOD inputs, mixed styles, and low-light/noisy conditions will help audience understand the limit.

### Questions
Please check Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
