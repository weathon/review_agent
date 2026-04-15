# D^3: Distributional Dataset Distillation with Latent Priors

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6

## Abstract
Dataset distillation, the process of condensing a dataset into a smaller synthetic version while retaining downstream predictive performance, has gained traction in diverse machine learning applications, including neural architecture search, privacy-preserving learning and continual learning. Existing methods face challenges in scaling efficiently beyond toy datasets. They also suffer from diminishing returns when increasing the distilled dataset size. We present Distributional Data Distillation (D$^3$), a novel approach that reframes data distillation problem into a distributional one. In contrast to existing methods that distill a dataset into a finite set of real or synthetic examples, D$^3$ produces a probability distribution and a decoder from which the original dataset can be approximately regenerated. We use Deep Latent Variable Models (DLVMs) to parametrize the condensed data distribution and introduce a new training objective that combines a trajectory-matching distillation loss with a distributional discrepancy term, such as Maximum Mean Discrepancy, to encourage alignment between original and distilled distributions. Experimental results across various computer vision datasets show that our method effectively distills with minimal performance degradation. Even for large high-resolution datasets like ImageNet, our method consistently outperforms sample-based distillation methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Distributional Data Distillation (D3) is a groundbreaking approach that transforms dataset distillation into a distribution-based problem. Unlike traditional methods that create a finite set of real or synthetic examples, D3 generates a probability distribution and a decoder to approximate the original dataset. Using Deep Latent Variable Models (DLVMs), it combines a trajectory-matching distillation loss with a distributional discrepancy term, resulting in strong alignment between original and distilled data. Across various computer vision datasets, D3 demonstrates effective distillation with minimal performance loss, even excelling with large datasets like ImageNet, surpassing sample-based methods consistently.

### Strengths
1. simple and intuitive idea
2. presentation is smooth and easy to understand

### Weaknesses
1. I highly disagree with the statement claimed in this authors, "Existing methods face challenges in scaling efficiently beyond toy datasets.", "More generally, these methods lack fine-grained control over distillation strength and often struggle to scale beyond smaller datasets like CIFAR-10 and MNIST, experiencing diminished performance when compressing larger or higher-dimensional datasets, such as ImageNet."[1, 2] cited in this paper, presented in CVPR 2023 has its main results in ImageNet on its cover.

2. Lack of novelty and significant contributions, the idea is of sampling from a latent distribution has been thoroughly explored since VAE came out. It's unclear what is the significant contribution or additional innovation the authors are intending to propose.

3. Experiment results in tables seem incomplete, it's hard to have holistic picture on how good this method is.


[1] Cazenavette, George, et al. "Generalizing Dataset Distillation via Deep Generative Prior." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[2] Cui, Justin, et al. "Scaling up dataset distillation to imagenet-1k with constant memory." International Conference on Machine Learning. PMLR, 2023.
[3] Wu, Xindi, Zhiwei Deng, and Olga Russakovsky. "Multimodal Dataset Distillation for Image-Text Retrieval." arXiv preprint arXiv:2308.07545 (2023).

### Questions
1. Is it a typo where in the abstract, it claims to have done experiments on imageNet, but in the actual experiments, it runs ImageNette, which is a 10 class subset and a much easier problem to solve.
2. Why does ConvNet have better accuracy than more complex and sophisticated networks?
3. Why are Imagenette and imagewoof results not available for DM in table 1?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Distributional Data Distillation (D3), a novel approach to dataset distillation. Unlike existing methods that condense datasets into smaller versions, D3 focuses on creating a conditional latent distribution $p(z)$ and a decoder $Q_\mathcal{S}^\theta (x|z)$. The paper utilizes the resulting data distribution $Q_\mathcal{S}^\theta (x) = \int Q_\mathcal{S}^\theta (x|z) p(z) dz$, called Deep Latent Variable Models (DLVMs), and a new training objective, combining trajectory-matching distillation with a distributional discrepancy term like Maximum Mean Discrepancy (MMD). Experimental results across various computer vision datasets, including the challenging ImageNet, demonstrate that D3 effectively condenses datasets with minimal performance loss. Notably, it consistently outperforms traditional sample-based distillation methods, even for large high-resolution datasets.

### Strengths
- The proposed method is simple and the description is easy-to-follow.

- This paper proposes improved MMD loss over previous work which only matches mean of feature vectors.

### Weaknesses
- The idea of utilizing a generative prior has already been explored in several papers, including HaBa, LinBa, KFS, IT-GAN, and GLaD.

- The comparison to the original literature on dataset distillation is entirely unfair. The proposed method outputs a distribution; therefore, it should be compared to deep generative models. Deep generative models can perform the exact same tasks as the proposed method.

### Questions
- In the loss of $\mathcal{L}_\texttt{MTT}$, why do we need the KL penalty? How does this regularization effects the performance?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Dataset distillation is a technique used to condense large datasets into smaller synthetic versions while maintaining predictive performance. It has applications in various machine learning domains, but existing methods face challenges when scaling beyond small datasets and experience diminishing returns as the distilled dataset size increases. To address these limitations, a novel approach called Distributional Data Distillation (D3) is introduced.

Unlike previous methods that distill datasets into finite sets of real or synthetic examples, D3 frames the data distillation problem as a distributional one. Instead of producing individual examples, D3 generates a probability distribution and a decoder that can approximately regenerate the original dataset. Deep Latent Variable Models (DLVMs) are used to parameterize the condensed data distribution.

D3 introduces a new training objective that combines a trajectory-matching distillation loss with a distributional discrepancy term, such as Maximum Mean Discrepancy. This objective encourages alignment between the original dataset distribution and the distilled distribution.

Experimental results on various computer vision datasets demonstrate that D3 effectively distills datasets with minimal performance degradation. Even for large high-resolution datasets like ImageNet, D3 consistently outperforms sample-based distillation methods.

### Strengths
1) This paper challenge the conventional approach of distilling into a finite set of samples, instead
casting the problem as a distributional one: finding a synthetic probability distribution which, when
sampled to produce training data, yields performance comparable to training on the original dataset.

2) To make this optimization problem tractable, this paper parametrize the distribution using Deep Latent
Variable Models (Kingma & Welling, 2013), and design a loss function that combines a state-of-theart gradient-matching criterion (Cazenavette et al., 2023) with a distributional loss (e.g., MMD or
Wasserstein distance) — a natural choice for our distributional framework.

3) This novel distributional dataset distillation perspective is appealing and it could addresses many of the limitations of prior distillation methods.

### Weaknesses
1)  The design in LEARNING THE DISTILLED DISTRIBUTION Matching is simply borrowed from [1]. Please clarify the difference.

2) The comparison in Table 1 is confusing.  Is Comp. rate good when this rate is high or low?

3) More comparison with generative-based dataset distillation methods could be added.

[1] Dataset distillation by matching training trajectories.

### Questions
Please see weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
