# Dual Contrastive Inversion with Distributional Priors for Diversity-Aware Data-Free Knowledge Distillation

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Model inversion (MI) has emerged as a key paradigm for data-free knowledge distillation (DFKD), yet existing MI methods suffer from limited diversity in synthetic data due to simplistic unimodal priors and the lack of explicit mechanisms for instance separability. We propose **D2CIP** (*Dual Contrastive Inversion with Distributional Priors*), a two-stage framework that enhances diversity by first recovering a class-conditional distributional prior with a Gaussian Mixture Model (GMM) aligned to teacher predictions and batch-normalization statistics, and then applying dual contrastive learning at both latent and instance levels with memory banks to enlarge the set of negatives. We further formalize data diversity as expected pairwise separability and establish its monotonic relationship with the contrastive loss, providing a principled justification for diversity maximization. Experiments on **CIFAR-10**, **CIFAR-100**, and **Tiny-ImageNet** demonstrate that D2CIP consistently outperforms state-of-the-art MI-based DFKD methods in both synthetic data diversity and distillation accuracy. The code is available at [https://anonymous.4open.science/r/gmdmi4dfkd-53E3](https://anonymous.4open.science/r/gmdmi4dfkd-53E3).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new framework, D2CIP, to enhance data diversity in data-free knowledge distillation (DFKD) by addressing the limitations of existing model inversion (MI) approaches that rely on unimodal priors and lack separability constraints. D2CIP introduces a two-stage process: (1) Distributional Prior Recovery, which learns a class-conditional GMM aligned with the teacher’s predictions and batch-normalization statistics to capture multimodal data structure; and (2) Dual Contrastive Inversion, which applies contrastive learning at both latent and instance levels using memory banks to enlarge negative samples and encourage diversity. Theoretically, the paper formalizes data diversity as expected pairwise separability and proves its monotonic relation with contrastive loss, providing a principled basis for diversity maximization. Experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-100 show that D2CIP outperforms previous DFKD methods (e.g., CMI, TA-DFKD) in both synthetic data diversity (lower FID and JS divergence) and distillation accuracy, confirming that its combination of distributional priors and dual contrastive learning yields robust and generalizable data-free distillation

### Strengths
1. The use of a class-conditional Gaussian Mixture Model (cGMM) captures multimodal latent structures aligned with teacher statistics, addressing mode collapse; ablation results show large accuracy drops when this component is removed (–1.0% on CIFAR-10, –3.8% on CIFAR-100). This means it’s indeed important to model different classes separately.  
2. Quantitative evaluations show consistent superiority in both JS divergence (0.498 vs 0.531) and FID (110.56 vs 112.32) over CMI and TA-DFKD, confirming that the dual contrastive design leads to more diverse and semantically aligned synthetic data.

### Weaknesses
1. While the paper presents a well-organized framework, several core components, such as contrastive learning, batch normalization alignment, and adversarial distillation, are closely related to existing works like CMI and DeepInversion. As a result, the boundary between adaptation and innovation could be articulated more clearly to highlight the unique conceptual advances of D2CIP.   
2. The paper provides strong quantitative evidence through FID and JS divergence but includes few visual examples of the generated samples. Additional visual comparisons would help readers better appreciate whether D2CIP produces more realistic and diverse images than existing methods.   
3. In Figure 3, the generated distributions of D2CIP and CMI appear visually similar, suggesting comparable data manifolds. A more detailed qualitative analysis or higher-resolution visualization could better illustrate the claimed improvements in diversity and distributional richness.

### Questions
Please see the weaknesses.

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
5

### Summary
The paper proposes a novel data-free knowledge distillation method that controls the diversity and quality of synthetic images using a Gaussian Mixture Model for input generation instead of random noise. It also introduces a dual contrastive inversion strategy that applies contrastive learning at both latent and instance levels, supported by memory banks to expand the negative sample space.

### Strengths
1. The paper is well written and easy to follow.
2. The idea of using a prior distribution to generate inputs for the generator is sound and clearly explains why it can improve performance.

### Weaknesses
1. The main novelty of this paper lies in controlling the diversity and quality of the generator’s outputs by using a prior distribution to create the input, rather than relying on random noise. However, this idea is quite similar to the approach in [1], where fixed inputs (based on label text embeddings) are used, and randomness is introduced through additional layers. It would be beneficial if the authors could discuss the advantages and disadvantages of these two techniques.
2. The experiments lack comparisons with some state-of-the-art methods, such as [1] and [2].
3. The paper reports results only on a small dataset and focuses solely on the image classification task.

[1] Nayer: Noisy layer data generation for efficient and effective data-free knowledge distillation. CVPR 2025.
[2] Coupling the Generator with Teacher for Effective Data-Free Knowledge Distillation. ICCV 2025.

### Questions
Please see the weakness section. I will consider raising my score if Question 1 is clearly explained.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper points out that the existing model inversion-based data-free knowledge distillation suffers limited synthetic data diversity problem, which from unimodal prior and implicit mechanisms for instance separability. To solve this issue, this paper proposes a two-stage framework D2CIP to enhance the diversity of synthetic data, which includes Distributional Prior Recovery and Dual Contrastive Inversion stages. For the first stage, Distributional Prior Recovery, it learns a class-conditional prior and adopt GMM to jointly optimized with the generator to align with the predictions of the teacher and batch-normalization statics, which can capture the multi-modal nature of real data. In second stage, it applies dual contrastive learning at both latent and instance levels, and adopts memory banks to expand negative samples, enhancing instance discrimination and maximizing data diversity. It further defines data diversity as expected pairwise separability. For downstream distillation, it proposes a decision-adversarial strategy for boundary samples. The experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1.	This paper is well written with clear and illustrative figures and tables, and well-motivated.
2.	This paper defines data diversity as expected pairwise separability and theoretically proves the monotonic relationship with contrastive loss.
3.	The experimental results demonstrate the SOTA performance of the proposed method on CIFAR-10, CIFAR-100, Tiny-ImageNet datasets.

### Weaknesses
1.	[major] The paper introduces key components, e.g., the class-conditional GMM prior, dual contrastive objectives, and memory banks, which improve the performance of the synthetic data for downstream distillation. However, they also lead to a notable increase in training time and GPU memory usage as increased learnable parameters. As shown in Table 11, D2CIP requires 1.5 times higher memory usage and longer training time per epoch compared with baseline. Although, the paper reports only the comparison of runtime and memory requirement per epoch, without further analysis of the total number of epochs required for training convergence and the overall training cost, it is clear that the proposed framework results in an increase in computational resource requirements.
2.	[minor] Due to the major weakness in computational efficiency, the proposed method may have scaling-up problems, for larger datasets, e.g., ImageNet-1K, or more complex teacher-student architectures.
3.	[major] The proposed framework introduces too many learnable or tunable parameters but does not clarify how these values are determined or how they should be tuned for different datasets or architectures.
4.	[minor] All experiments are conducted on CNN-based architectures, ResNet, VGG, WRN, and the method has not been evaluated on more diverse or modern backbones such as Vision Transformers.

### Questions
1.	As mentioned in weakness [major] 1, this paper (Table 11) only reports runtime and memory usage per epoch. Could the authors provide the total training cost, including number of epochs to convergence and total GPU hours, to better support the overall efficiency of D2CIP?
2.	The performance appears sensitive to parameter choices, could the authors provide clearer guidance or for parameter setting without extensive tuning?
3.	Could the proposed method apply on larger-scale dataset, e.g., ImageNet-1K, and more complex architectures, e.g., ViT?

### Soundness
3

### Presentation
3

### Contribution
2
