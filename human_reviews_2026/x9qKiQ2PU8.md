# Fully Kolmogorov-Arnold Deep Model in Medical Image Segmentation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Deeply stacked KANs are practically impossible due to high training difficulties and substantial memory requirements. Consequently, existing studies can only incorporate few KAN layers, hindering the comprehensive exploration of KANs. This study overcomes these limitations and introduces the first fully KA-based deep model, demonstrating that KA-based layers can entirely replace traditional architectures in deep learning and achieve superior learning capacity. Specifically, (1) the proposed Share-activation KAN (SaKAN) reformulates Sprecher’s variant of Kolmogorov-Arnold representation theorem, which achieves better optimization due to its simplified parameterization and denser training samples, to ease training difficulty, (2) this paper indicates that spline gradients contribute negligibly to training while consuming huge GPU memory, thus proposes the Grad-Free Spline to significantly reduce memory usage and computational overhead.	(3) Building on these two innovations, our ALL U-KAN is the first representative implementation of fully KA-based deep model, where the proposed KA and KAonv layers completely replace FC and Conv layers. Extensive evaluations on three medical image segmentation tasks confirm the superiority of the full KA-based architecture compared to partial KA-based and traditional architectures, achieving all higher segmentation accuracy. Compared to directly deeply stacked KAN, ALL U-KAN achieves ${10\times}$ reduction in parameter count and reduces memory consumption by more than ${20\times}$, unlocking the new explorations into deep KAN architectures. Code is available at: https://github.com/****.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ALL U-KAN, a fully KAN-based deep model for medical image segmentation. The key challenges are excessive training difficulty and GPU memory consumption. Thus, this paper proposes two contributions: Shared-activation KAN and Grad-Free Spline. Built upon these, the paper constructs the first fully KA-based U-shaped segmentation network (ALL U-KAN), replacing all FC and convolutional layers with KA/KAonv layers. Experiments on three datasets demonstrate superior segmentation performance and 20× lower memory usage compared to baseline KANs.

### Strengths
1. The first work for a fully KA-based network architecture.
2. Writing and figures are clear enough for reproducibility.
3. Experimental evaluation includes multiple datasets and ablation studies.

### Weaknesses
1. Improvements over U-KAN and other baselines are modest, sometimes within error margins.
2. Experiments focus solely on 2D medical segmentation. Generalization to other modalities or tasks is untested.

### Questions
1. Is the model stable for deeper stacks (e.g., >20 layers)?
2. Does detaching spline gradients introduce any measurable bias in gradient flow or convergence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge that deeply stacked Kolmogorov-Arnold Networks known as KANs face excessive parameter counts and memory requirements, preventing their practical application in deep architectures as discussed in the Introduction and Sec. 2. The authors propose three innovations: First, Share-activation KAN named SaKAN, which reduces parameters by sharing activation functions across input dimensions while maintaining representational capacity. Second, Grad-Free Spline, which significantly reduces memory consumption by detaching spline gradients during backpropagation. Third, ALL U-KAN, the first fully KA-based deep model where all fully connected and convolutional layers are replaced with KA-based layers.

### Strengths
• Novel fully KA-based deep architecture that replaces all FC and convolutional layers with KA-based layers, surpassing prior partial or shallow KAN implementations.

• Significant parameter reduction via SaKAN maintains performance with approximately 57% fewer parameters compared to vanilla KAN.

• Substantial memory efficiency achieved through Grad-Free Spline.

### Weaknesses
• Scope-claim mismatch: The paper frames contributions as universally applicable to deep learning, yet experiments are limited exclusively to medical image segmentation, leading to overgeneralization in conclusions.

• FLOPs reporting inconsistency: KAonv configurations show approximately 70× discrepancy with approximately 25M versus 1752M without methodological explanation or derivation.


• Limited baseline coverage and tuning fairness: No comparisons with recent KAN variants such as FastKAN and ChebyKAN; unclear whether all baselines received equivalent hyperparameter tuning budgets.

### Questions
• Enhance reproducibility: Release anonymized code, complete training configurations, preprocessing scripts, and dataset split specifications to enable independent verification of results.

• Expand baseline comparisons with fair tuning: Include recent KAN variants such as FastKAN, ChebyKAN, and others; explicitly document hyperparameter tuning budget and search strategy for all methods to ensure fair comparison.

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
This paper proposes ALL U-KAN, claimed as the first fully Kolmogorov–Arnold (KA)-based deep network for medical image segmentation. It builds upon the Kolmogorov–Arnold representation theorem, which states any multivariate continuous function can be represented as sums of univariate functions, previously used in Kolmogorov–Arnold Networks (KANs). While prior KANs demonstrated strong representation power, they were limited to a few layers due to excessive parameters and GPU memory requirements. The authors address these barriers through two main innovations:
1) Shared-activation KAN (SaKAN):
- Reformulates Sprecher’s variant of the KA theorem so that all input dimensions share a single learnable activation function rather than per-input splines. This reduces parameters and increases sample efficiency.

2) Grad-Free Spline:
- Detaches spline gradients during backpropagation, arguing (with a small theoretical justification) that spline gradients contribute negligibly to learning, thus cutting GPU memory by >20×.

Combining these yields the KA and KAonv layers, replacing fully connected and convolutional layers, respectively. The resulting ALL U-KAN achieves better segmentation accuracy than U-Net, U-KAN, and other baselines on BUSI, GlaS, and CVC-ClinicDB datasets, with a ≈10× parameter reduction and ≈20× memory reduction.

### Strengths
1. Clear and well-motivated problem statement
- The paper correctly identifies the key bottlenecks preventing deep stacking of KANs: training instability and GPU memory explosion and tackles them directly.
2. Practical engineering contributions
- Introduces two concrete, implementable techniques (Shared-activation KAN (SaKAN) and Grad-Free Spline) that make deep KAN architectures trainable on commodity GPUs without excessive resource cost. The proposed methods are simple to adopt and can be generalized to other spline-based models.
3. Substantial efficiency improvement
- Achieves up to 10x parameter reduction and >20x memory savings compared with conventional deep KANs, while maintaining or improving accuracy, showing tangible computational benefits.

### Weaknesses
1. Novelty and Attribution
- SaKAN’s shared-activation design is a straightforward adaptation of Sprecher’s 1965 theorem; the paper does not sufficiently contrast it with KAN 2.0, LoKi, or GKAN, which already explore shared or parameter-efficient KA forms. Also, Grad-Free Spline is effectively “stop-gradient on spline basis”, which many practitioners might already do as a memory optimization. The “Theorem 1” proof is heuristic.
2. Empirical scope
- Evaluations are limited to 2D biomedical datasets (≤ 600 images). It will be great if it can be further tested on large-scale 3D or cross-domain data.
3. Empirical comparison between different network backbones
- Detailed performance comparisons have been shown in Table 1. However, it is difficult to show the innovation that "why we need KANs in segmentation". It will be great to perform ablation experiments that compares the network block design with pure convolution (e.g. 3x3), vision transformer, swin transformer, large kernel convolution, deformable convolution and sparse convolution in the proposed network structure. If KANs demonstrate the best performance with best efficiency, that will be a strong signal for using KANs as the segmentation backbone.

### Questions
1. Motivation and necessity of KANs:
The paper motivates KANs via the Kolmogorov–Arnold theorem, but it remains unclear why such a backbone is necessary for segmentation tasks already well-handled by CNNs, ViTs, or Mamba-based models. Can the authors articulate what fundamental limitation in existing architectures KANs are designed to overcome (e.g., functional interpretability, sample efficiency)?
Moreover, can they provide empirical evidence (e.g., limited data robustness) demonstrating a capability that existing architectures lack?

2. Mathematical distinction of SaKAN:
The proposed Shared-activation KAN (SaKAN) reformulates Sprecher’s variant of the Kolmogorov–Arnold theorem, but its novelty relative to KAN 2.0’s shared spline basis and LoKi’s low-dimensional KAN remains unclear. Both prior works also reduce the number of learned univariate spline functions via parameter sharing or dimensionality reduction. Can the authors clarify the mathematical difference or additional representational advantage of SaKAN beyond these existing formulations?

3. Comparison to existing gradient-efficiency techniques:
The proposed Grad-Free Spline seems conceptually similar to gradient checkpointing or selective detachment strategies already used to save GPU memory in deep networks. Have the authors conducted quantitative comparisons against these standard techniques under equivalent compute and memory budgets? If not, it would be valuable to demonstrate that Grad-Free Spline offers measurable benefits beyond these established approaches.

4. Effect of gradient detachment on optimization and fine-tuning:
Since Grad-Free Spline explicitly detaches spline gradients, does this introduce bias or degradation in gradient flow, convergence stability, or fine-tuning performance? Have the authors evaluated whether this detachment affects downstream adaptation or sensitivity to learning rate schedules? A deeper theoretical justification or diagnostic experiment would help clarify whether this design preserves full optimization fidelity.

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
This paper introduces a novel U-Net–based architecture for 2D medical image segmentation, leveraging the Kolmogorov–Arnold framework to enable deeply stacked KANs. It extends from U-KAN. The authors propose the Shared-Activation KAN (SaKAN), which reformulates Sprecher’s variant of the Kolmogorov–Arnold representation theorem to achieve a more compact parameterization and improved training efficiency through reduced parameter count and sample requirements. In addition, Grad-Free Spline was proposed to reduce the parameter count and memory consumption further. The proposed method is evaluated on three 2D medical segmentation datasets (BUSI, GlaS, CVC), with comprehensive ablation studies conducted to validate the effectiveness of each component.

### Strengths
This paper solved several issues that KAN faced. In order to address this problem. Two approach was proposed, [1] SaKAN reformulates Sprecher’s variant of the KA theorem into a computationally efficient deep-learning form. [2] Grad-Free Spline offers a memory-efficient strategy supported by theoretical analysis. It shows a 10x reduction in parameter count and  reduces memory consumption by more than 20x.

### Weaknesses
Limited Novelty: The overall architectural design appears similar to U-KAN, with the primary difference being the introduction of the proposed SaKAN module. While SaKAN and the Grad-Free Spline contribute to training stability and memory efficiency, the paper would benefit from a clearer discussion of how these innovations fundamentally advance beyond prior U-KAN architectures.

Need for 3D Experiments: Since one of the key claims is improved parameter efficiency and training stability, it would be valuable to include 3D medical image segmentation experiments to validate the scalability and robustness of the proposed approach.

Training Efficiency Comparison: The paper highlights that the proposed method reduces training difficulty compared to U-KAN. However, a comprehensive comparison of training times across models—including traditional CNN-based U-Nets—would provide a clearer picture of computational efficiency. Even if KAN-based architectures remain slower, such a comparison would provide a better overview.

Architectural Details: The description of the ALL U-KAN architecture lacks sufficient implementation details. The paper should specify the number of KAN layers used, the depth of the network, and how these components correspond to the encoder–decoder stages of the original U-Net structure. 

Other Related Work: Recently, there has been a work [r1], which uses a fully continuous approach and is similar to your architecture. It should be added as a baseline.
[r1] Cheng, Chun-Wun, et al. "Implicit U-KAN2. 0: Dynamic, efficient and interpretable medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2025.

Experiment Result: The result outperforms U-KAN in BUSI while having a very similar result with GlaS and CVC.

### Questions
See the Weakness

### Soundness
2

### Presentation
2

### Contribution
2
