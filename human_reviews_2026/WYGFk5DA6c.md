# MIST: Multiple Stochastic Prompt Tuning for Few-shot Adaptation under Extreme Domain Shift

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Foundation Vision-Language Models (VLMs) like CLIP generalize well due to
large-scale pretraining, but their performance degrades under significant distribution
shifts in appearance and label semantics. Few-shot adaptation via adapter or
prompt tuning addresses limited-data tasks, but are not specifically designed to
handle such extreme domain shifts. Some cross-domain few-shot methods consider
such domain-shifts but often use episodic settings with fixed classes, limiting
real-world applicability. To address this gap, we propose a novel framework MIST
(Multiple Stochastic Prompt Tuning), which adapts CLIP to extreme domain shifts
with few labeled examples across all classes simultaneously. MIST uses multiple
learnable prompts per class to capture diverse modes in visual features, modeled
as Gaussian distributions to improve generalization and reduce overfitting. Extensive
experiments show the effectiveness of the proposed framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a prompt tuning method for cross-domain few-shot adaptation using CLIP. Existing approaches face two key limitations: (1) they often overlook domain shifts that involve both visual appearance and label semantics, and (2) their evaluations typically exclude part of the target classes. To address these issues, the authors introduce two key techniques. First, they model prompt parameters stochastically using Gaussian distributions rather than deterministic vectors, improving robustness to domain and semantic variation. Second, they employ multiple prompts per class to better capture the diverse semantic representations within each category. The proposed method is extensively evaluated across multiple standard benchmarks

### Strengths
- The paper is well-written, allowing readers to easily follow the logical flow and understand the motivation as well as the proposed method at both the conceptual and technical levels. For example, in **Section 4.2**, the authors clearly introduce the limitation that motivates their approach, then explain the conceptual idea (“why” and “what”) before presenting the detailed technical design (“how”). In addition, **Figure 4** effectively visualizes the overall pipeline of the proposed framework.

- The experimental evaluation is comprehensive, including comparisons with state-of-the-art methods, detailed ablation studies, and hyperparameter analyses such as class imbalance, number of prompts, and prompt length.

### Weaknesses
- The problem setting in the Introduction is not clearly defined. Adapting CLIP to downstream target datasets can correspond to several existing problem formulations (e.g., few-shot adaptation, domain generalization, or open-world transfer). For instance, models such as Few-Shot Test-Time Domain Adaptation (FSTT-DA)[1] address few-shot adaptation with unlabeled target data under challenging domain and semantic shifts [2]. In this paper, the stated task is cross-domain few-shot learning, where both the domains and classes differ, but this distinction is not explicitly articulated. The authors are encouraged to include one or two sentences clarifying the setting in terms of domain, class, and label availability, and to highlight how it differs from related formulations. In addition, in Line 41, the definition of cross-domain few-shot learning may confuse readers unfamiliar with the meta-learning paradigm. It would help to explain K-way-N-shot evaluation in simple terms.

- The conceptual idea of using multiple learnable prompt embeddings to represent knowledge for a specific domain or class is not entirely novel. For example, prior works [3] have introduced prompt pools containing multiple learnable prompts whose weighted combination captures domain-specific knowledge. The same concept can be directly extended to class-specific representations. The authors should clarify these conceptual overlaps in the Related Work section and explicitly emphasize how their method differs or improves upon these earlier approaches.

- The proposed method is evaluated only with CLIP ViT-B/16, leaving uncertainty about whether the improvements generalize to other backbones. While computational constraints are understandable, testing on at least one additional architecture—such as ViT-L/14—on a representative benchmark would strengthen the claim of generality.

[1] Meta-dmoe: Adapting to domain shift by meta-distillation from mixture-of-experts. NeurIPS 2022

[2] WILDS: A Benchmark of in-the-Wild Distribution Shifts. ICML 2021

[3] Adapting to Distribution Shift by Visual Domain Prompt Generation. ICLR 2024

### Questions
- Could the authors explicitly clarify the problem setting of this work? How are domain, class, and label availability defined in this setting?

- How does this problem formulation differ from existing setups such as FSTT-DA or other few-shot adaptation methods?

- In Line 41, the definition of cross-domain few-shot learning may be unclear to readers without a meta-learning background. Could the authors briefly explain K-way-N-shot evaluation in simpler terms for clarity?

- Did the authors compare against or draw inspiration from existing methods that use multiple prompts for domain or class-specific adaptation? If so, please clarify these differences in Related Work or Ablation Studies.

- The experiments are conducted only on CLIP ViT-B/16. Can the authors comment on whether similar performance trends are expected on larger or different architectures such as ViT-L/14?

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
4

### Summary
This paper proposes MIST (Multiple Stochastic Prompt Tuning) to address few-shot adaptation of Vision-Language Models (VLMs) like CLIP under domain shifts and label semantic misalignment. MIST introduces multiple learnable prompt vectors per class modeled as Gaussian distributions to capture multimodal visual features, enhancing generalization and reducing overfitting in data-scarce scenarios. Experiments demonstrate significant improvements over SOTA methods, particularly in ultra-low-data regimes (1-shot).

### Strengths
1. MIST diagnoses fragmented feature representations under extreme domain shifts and proposes multiple Gaussian-distributed prompts per class to form diverse decision boundaries—a principled approach beyond standard prompt tuning.
2.  Significant improvements over methods across multiple benchmarks, especially in 1-shot scenarios, with demonstrated robustness to class imbalance and good generalization.

### Weaknesses
1. Weak Motivation and Limited Novelty: The method lacks clear motivation for why Gaussian-distributed multiple prompts address extreme domain shifts. The approach resembles existing GMM-based strategies without sufficient differentiation. Why Gaussian over other probabilistic models?

2. Incomplete Experiments: Baselines are outdated; missing recent few-shot adaptation methods. Limited to four benchmarks (EuroSAT, ISIC, Plant Disease, ChestX). No evaluation on standard OOD benchmarks (ImageNet-R, ImageNet-A, CIFAR-10-C, DomainNet, VisDA, etc.). Missing cross-dataset generalization experiments.

### Questions
Have you evaluated on other OOD benchmarks? How does MIST compare to recent domain generalization and OOD adaptation methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MIST (Multiple Stochastic Prompt Tuning), a method for adapting CLIP-like vision-language models to few-shot tasks under extreme domain and label semantic shifts. MIST introduces (i) stochastic prompt learning (each prompt represented as a Gaussian distribution) to improve generalization and (ii) multiple prompts per class to model multimodal feature distributions. The method is evaluated on the BSCDFSL benchmark (EuroSAT, ISIC, PlantDisease, ChestX) and shows improved performance on most datasets over SOTA methods published up to 2024.

### Strengths
+ The integration of stochastic prompt learning, deep prompting, and multiple prompts per class represents a novel contribution. As shown in Table 1, the proposed approach outperforms several state-of-the-art baselines.

### Weaknesses
- Although this integration has not been explicitly explored in prior work, all three underlying ideas have already been proposed in the existing literature. Therefore, the proposed approach appears somewhat straightforward.
- While the “k-shot all-class” setup (Lines 94-96) is interesting, the paper does not clearly explain why it is considered “closer to real-world deployment” than the widely used N-way k-shot (episodic) setting. Moreover, there seems to be no reason not to include comparisons between MIST and existing methods under the episodic setting as well.
- More recent methods should be included in the comparison. For example, although PromptMargin [1] (published in 2025) is cited in §2, its results are not reported in Table 1.
- Several ablation studies are missing. For instance, the effect of deep prompting and its configuration has not been evaluated. In addition, while the mean of the first prompt (i.e., ``A photo of a [CLS]'') is fixed throughout the experiments, it should also be tested as a learnable parameter.
- Bayesian prompt learning for vision–language models has already been explored in the literature (e.g., [2, 3]). Although the experimental settings differ, these works should be discussed to better contextualize and highlight the novelty of the proposed method.
- The listed contributions (Lines 56-65) are not orthogonal; at least points 1, 3, and 4 seem to overlap.
- While my understanding might be incomplete, it is unclear why the summation in Equation (5) (Lines 291-294) is taken over 1 to 2C. j takes only the values 1 or 2, doesn’t it?

[1] D. Brahma et al., Prompt Tuning Vision Language Models with Margin Regularizer for Few-Shot Learning under Distribution Shifts, in TMLR, 2025.

[2] M.M. Derakhshani et al., Bayesian Prompt Learning for Image-Language Model Generalization, in ICCV, 2023.

[3] X. Liu et al., Patch-Prompt Aligned Bayesian Prompt Tuning for Vision-Language Models, in UAI, 2024.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
