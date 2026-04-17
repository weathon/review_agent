# DRO-Augment Framework: Robustness by Synergizing Wasserstein Distributionally Robust Optimization and Data Augmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
In many real-world applications, ensuring the robustness and stability of deep neural networks (DNNs) is crucial, particularly for image classification tasks that encounter various input perturbations. While data augmentation techniques have been widely adopted to enhance the resilience of a trained model against such perturbations, there remains significant room for improvement in robustness against corrupted data and adversarial attacks simultaneously. 
To address this challenge, we introduce DRO-Augment, a novel framework that integrates Wasserstein Distributionally Robust Optimization (W-DRO) with various data augmentation strategies to improve the robustness of the models significantly across a broad spectrum of corruptions. 
Our method outperforms existing augmentation methods under severe data perturbations and adversarial attack scenarios while maintaining the accuracy on the clean datasets on a range of benchmark datasets, including but not limited to CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, and Fashion-MNIST.  
On the theoretical side, we establish novel generalization error bounds for neural networks trained using a computationally efficient, variation-regularized loss function with augmented data, closely related to the W-DRO problem. Furthermore, we introduce a refined CIFAR-C benchmark that corrects inconsistencies in corruption intensities, providing a more reliable evaluation for future robustness research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a novel training framework that integrates Wasserstein Distributionally Robust Optimization (W-DRO) with data augmentation to improve robustness against both adversarial attacks and corrupted data.

### Strengths
The proposed unified framework aims to enhance robustness against both common corruptions and adversarial attacks.

### Weaknesses
* The overall contribution of this work appears to be marginal. The objective function defined in Eq. (2.1) is adopted from prior work, and the data augmentation strategies employed are common and well-established.
* According to the results reported in RobustBench [1], accuracies against adversarial examples and corrupted data are evaluated on two distinct leaderboards. Methods such as NoisyMix and AugMix have already achieved strong performance on corruption benchmarks. Simply combining these data augmentation techniques with adversarial training does not seem to present a novel contribution, which diminishes the originality of this work.
* The authors claim that “The proposed method can enhance robustness against both common corruptions and adversarial attacks” (lines 95–97). To substantiate this claim, the authors should evaluate the accuracy of existing adversarially trained models reported in RobustBench on corrupted datasets for comparison. In its current form, the empirical results are not convincing. A stronger baseline should be established using standard datasets such as MNIST, CIFAR-10/100, or ImageNet, rather than Fashion-MNIST or Tiny-ImageNet, to ensure fair and comparable results with existing works. I recommend that the authors include a direct comparison with these established models.

[1] Croce, Francesco, et al. "Robustbench: a standardized adversarial robustness benchmark." arXiv preprint arXiv:2010.09670 (2020).

### Questions
What is the relationship between the proposed Wasserstein Distributionally Robust Optimization approach and existing methods that introduce gradient flow regularization (i.e., regularization based on the alignment or flow of gradients) [2]?

[2] Xia, Pengfei, and Bin Li. "Improving resistance to adversarial deformations by regularizing gradients." Neurocomputing 455 (2021): 38-46.

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
2

### Summary
This paper proposes DRO-Augment, a framework that combines Wasserstein Distributionally Robust Optimization with data augmentation techniques to improve neural network robustness against both natural corruptions and adversarial attacks. The authors provide theoretical generalization bounds for neural networks trained with variation-regularized loss on augmented data and introduce a refined CIFAR-C benchmark with corrected severity levels.

### Strengths
- The combination of W-DRO and data augmentation is well-motivated and technically sound, effectively merging two complementary robustness strategies.
- The paper establishes generalization error bounds for neural networks trained with W-DRO on augmented data, achieving a faster convergence rate compared to prior work.
- Extensive experiments across multiple benchmark datasets (CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, Fashion-MNIST) with various attack types (PGD, AutoAttack, C&W, FAB-T, Square) provide convincing evidence of the method's effectiveness.

### Weaknesses
- While the paper mentions small additional time costs, there is no systematic analysis of computational overhead compared to baselines, memory requirements, or scalability to larger datasets/models. Actually, this is very critical in practice.
- The ablation studies, mainly in Table 3, only examine CIFAR-100-C and Fashion-MNIST. It should cover more datasets and analyze the sensitivity to key hyperparameters (for example, the mixing ratios \frac{\alpha}{\beta}) more thoroughly. 
- The experiments use only PreActResNet-18, which limits understanding of how the method generalizes to other modern architectures (Vision Transformers, EfficientNets, etc.). The refined CIFAR-C evaluation only includes ResNet variants, not validating performance on the architectures mentioned as motivation.

### Questions
- Can you provide results on modern architectures (ViT, ConvNeXt, EfficientNet) to demonstrate the method's broader applicability? 
- How does performance scale with model capacity?
- What is the sensitivity of the method to the W-DRO radius?
- Can you provide a detailed computational analysis, including training time, memory usage, and wall-clock time comparisons across all baselines?

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
The paper introduces DRO-Augment, a training framework that combines Wasserstein Distributionally Robust Optimization (W-DRO) with data augmentation techniques to improve neural network robustness against both natural corruptions and adversarial attacks. The authors provide theoretical analysis establishing generalization error bounds for their approach and demonstrate empirical improvements on standard corruption benchmarks (CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C) and adversarial robustness tests. Additionally, they propose a refined CIFAR-C benchmark to address inconsistencies in the original corruption severity settings

### Strengths
1. The paper tackles both natural corruptions and adversarial attacks simultaneously, which is a practical consideration often overlooked in papers that focus on only one type of robustness. 

2. The paper provides generalization error bounds for neural networks trained with W-DRO and augmented data (Theorem 4.1), achieving an improved convergence rate compared to previous work. 

3. The authors identify and address a real issue with CIFAR-C severity calibration, proposing a more consistent evaluation framework based on ResNet performance.

### Weaknesses
1. The main contribution is essentially combining two existing techniques (W-DRO and data augmentation) without fundamental algorithmic innovation.

2. The paper admits DRO-Augment adds overhead due to gradient-norm evaluation but dismisses it as small. However, no measurements (FLOPs, time comparison) are given. Given that W-DRO involves per-sample gradients, cost may scale poorly with model size.

3. Only PreActResNet-18 is tested. Without scaling to transformers, larger CNNs, or ImageNet-level datasets, the method’s generality and computational feasibility remain uncertain.

### Questions
Please refer to the weaknesses section.

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
3

### Summary
The paper proposes a combined data augmentation and distributionally robust optimization framework DRO-Augment, aimed at improving both adversarial and natural corruption robustness of neural networks. The authors approximate the Wasserstein DRO objective using a variation-regularization surrogate (a gradient-norm penalty) and integrate it with popular augmentations (Mixup, AugMix, and NoisyMix). The work claims that this combination captures both worst-case perturbations (via DRO penalty) and diverse real-world corruptions (via augmentations). Theoretical contributions include an asymptotic robust generalization bound for mixup-trained models under a sparse ReQU network class, with explicit dependence on the Wasserstein radius ρ. Empirical results on CIFAR-10/100-C, Tiny-ImageNet-C, and adversarial settings (Fashion-MNIST-ε and Tiny-ImageNet-ε) show consistent robustness improvements with minimal accuracy loss. The paper also introduces a refined “severity scale” benchmark for CIFAR-C datasets.

### Strengths
1. Combines two complementary robustness paradigms, Wasserstein DRO in training and data augmentation before optimization, within a single unified and implementable framework. And effectiveness is validated by consistent empirical gains across multiply common datasets.

2. The generalization bound includes explicit ρ-dependence and recovers the expected nonparametric rate under sparse ReQU networks, improving interpretability of robustness–sample trade-offs.

3. The proposed refinement of CIFAR-C severity scales improves evaluation consistency and could serve as a useful benchmark extension.

4. Writing quality and experimental reproducibility are strong overall. Tables and figures are clear and well-structured.

### Weaknesses
1. The claimed L∞-Wasserstein DRO formulation conflicts with the L2-based implementation for the gradient penalty (P. 8, L.400-402). This inconsistency weakens the claim that the model optimizes L∞ W-DRO.

2. The theoretical contribution on adversarial risk bounds is largely incremental. it mainly differs in applying mixup data and sparse ReQU architectures rather than introducing a new bounding method. Also, the network class smoothness bounds for norm of gradient  and (operator) norm of the Hessian (P. 8, L.425-426) are described informally as “almost bounded,” lacking explicit uniform inequalities or norm definitions needed for formal correctness.

3. The authors note that NoisyMix has strong baseline robustness. But there is no per-augmentation analysis clarifying how noise-heavy augmentation interacts with DRO regularization and why improvements are limited.

### Questions
1. Could you clarify the claim of focusing on the L∞-Wasserstein DRO while the proxy loss (P. 8, L.400-402) applies an L2 gradient penalty? Is the use of the L2 norm intended as an approximation for the L∞-Wasserstein ball, or should the formulation instead use an L1 penalty (dual of L∞)?

2. Could you clarify the symbol consistency between Eq. 2.1 (P. 3) and Algorithm 1 (P. 4, L172). Equation 2.1 defines the gradient norm using the dual exponent q* with an outer 1/q power, while Algorithm 1 applies norm q without that outer power. Is this a typo inconsistency or an intentional change in implementation?

3. Could you provide  additional  ablations to verify the relation between NoisyMix’s robustness by noise injections and the limited incremental benefit of the DRO regularization?

### Soundness
2

### Presentation
3

### Contribution
2
