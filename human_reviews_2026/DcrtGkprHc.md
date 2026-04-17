# Geometric Conformal Outlier Synthesis

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Deep neural networks for image classification often exhibit overconfidence on out-of-distribution (OOD) samples. To address this, we introduce Geometric Conformal Outlier Synthesis (GCOS), a training-time regularization framework aimed at improving OOD robustness during inference. GCOS addresses a limitation of prior synthesis methods by generating virtual outliers in the hidden feature space that respect the learned manifold structure of in-distribution (ID) data. The synthesis proceeds in two stages: (i) PCA on training features identifies geometrically-informed, off-manifold directions; (ii) a Conformally-Inspired Shell, defined by the empirical quantiles of a nonconformity score from a calibration set, adaptively controls the synthesis magnitude to produce boundary samples. The shell ensures that generated outliers are neither trivially detectable nor indistinguishable from in-distribution data, facilitating smoother learning of robust features. This is combined with a contrastive regularization objective that promotes separability of ID and OOD samples in a chosen score space, such as Mahalanobis or energy-based. Experiments show that GCOS improves OOD detection relative to baselines using the standard energy-based inference approach. As an exploratory extension, the framework naturally transitions to conformal OOD inference, which translates uncertainty scores into statistically valid p-values and enables thresholds with formal error guarantees, providing a pathway toward more predictable and reliable OOD detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Geometric Conformal Outlier Synthesis (GCOS), a training-time regularization framework for improving out-of-distribution (OOD) robustness. GCOS combines geometric information from PCA with a conformal prediction heuristic to generate synthetic outliers in the latent space. The method is evaluated on several datasets and compared against the baseline Virtual Outlier Synthesis (VOS).

### Strengths
The paper considers the application of conformal prediction in the context of outlier synthesis, which is an interesting attempt to connect uncertainty quantification with OOD data generation.

### Weaknesses
1. The experimental section is extremely limited. The manuscript only compares the proposed GCOS method with VOS and a no-regularization baseline on a few datasets for the main results. If the authors aim to position GCOS as an improvement over VOS, they should conduct evaluations on a broader range of benchmarks and tasks, similar to VOS including object detection. Moreover, VOS includes comparisons with multiple baselines, whereas the current experiments are insufficient to substantiate the claimed effectiveness of GCOS.

2. The authors note that the online calibration stage inherently violates the exchangeability assumption central to conformal prediction. The subsequent post-training calibration mitigates this theoretical limitation. As a result, the conformal component of GCOS lacks full statistical validity, weakening the theoretical rigor of the proposed framework.

### Questions
1. What is the computational overhead introduced by GCOS, particularly compared to VOS, given its additional PCA computations and conformal prediction?

2. How well does GCOS generalize to diverse OOD scenarios, like object detection or other tasks?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to combine conformal learning with outlier synthesis. The outliers are synthesized by sampling from off-manifold directions, as defined by low-variance principal components. During training, these outliers are encouraged by a regularization loss to move farther away from the class centers. At inference time, the standard energy-based metric is used to detect outliers. The method is compared against VOS on some near-OOD experiments.

### Strengths
- The application of conformal prediction to outlier synthesis is interesting.
- Sampling in off-manifold directions is a sensible approach.
- The method is clearly explained, and the idea of using conformal prediction during inference is interesting.

### Weaknesses
The method fails to mention a number of papers published since VOS that effectively address its Gaussian assumption shortcoming. One example, NPOS [1], is even explicitly cited as justification for the weaknesses of VOS, yet it is not compared against, despite addressing the exact same pitfalls. Other examples include Dream-OOD [2], BOOD [3], and NCIS [4], which all (especially [3,4]) take the same approach of off-manifold sampling. The paper should both compare to and discuss these methods.

Furthermore, while moving towards Near-OOD experiments for evaluation is sensible, the method should still be compared on standard experiments (such as far-OOD on ImageNet) to be able to place it within the wider field. The chosen Near-OOD experiments are also very simple, such as MVTec, which has been almost perfectly solved for a long time [5], and the models reach close to perfect scores on both MNIST and Stanford Dogs. More standard Near-OOD experiments would be appropriate, such as ImageNet:SSB-Hard and CIFAR100:CIFAR10.

Overall, due to these aspects, the current manuscript does not convincingly demonstrate that the proposed method is superior to the other methods proposed for the same purpose.

[1] Tao, Leitian, et al. "Non-parametric outlier synthesis." International Conference on Learning Representations 2023.
[2] Du, Xuefeng, et al. "Dream the impossible: Outlier imagination with diffusion models." Advances in Neural Information Processing Systems 36 (2023): 60878-60901.
[3] Liao, Qilin, et al. "BOOD: Boundary-based Out-Of-Distribution Data Generation." International Conference on Machine
Learning 2025.
[4] Doorenbos, Lars, Raphael Sznitman, and Pablo Márquez-Neila. "Non-Linear Outlier Synthesis for Out-of-Distribution Detection." arXiv preprint arXiv:2411.13619 (2024).
[5] Roth, Karsten, et al. "Towards total recall in industrial anomaly detection." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Questions
- The authors mention multiple times the need to tune energy-based scores on validation data. However, their approach utilizes two separate calibration sets. Why is this not a similar weakness for the proposed method?
- What are the mean and standard deviation of the results over multiple runs?
- The sentence "we start with the discussion of Virtual Outlier Synthesis (VOS) (Du et al., 2022), thereby shaping more robust decision boundaries," on L38-40 is grammatically incorrect. The same holds for L233-235.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Geometric Conformal Outlier Synthesis (GCOS), a training-time framework aimed to improve out-of-distribution (OOD) detection and overconfidence. The core idea is to synthesize virtual outliers in the feature space that
respect the learned geometry of the in-distribution (ID) data. This should improve over simple parametric (e.g., Gaussian) assumptions of prior work like VOS.

The GCOS synthesis process has two stages: Direction by applying PCA to the training features to identify geometrically-informed, off-manifold directions, and Magnitude during which a "Conformally-Inspired Shell" is used to adaptively control the synthesis magnitude. The shell is defined by the empirical quantiles (e.g., 95th and 99th) of a nonconformity score (Mahalanobis distance) from a calibration set.

The goal is to generate hard outliers that are not trivially easy from ID data. This synthesis is paired with a contrastive regularization loss. The authors show that GCOS improves OOD detection on several Near-OOD benchmarks compared to a baseline and
VOS.

### Strengths
- one clear strength is the move away from simple parametric assumptions for outlier synthesis. Using PCA to find low-variance directions is an intuitive and non-parametric way to explore the odd regions of the learned feature space.

- the paper argues  that the choice of outlier synthesis is key, and that outliers should not be too easy or too hard. The use of a shell based on quantiles of the Mahalanobis score for adaptively calibrating this difficulty based on the model’s current state is original, well justified and, in my opinion, clever

- a positive point is also the focus on near-OOD datasets which are harder, more critical for real world tasks, and the good performance is encouraging

### Weaknesses
- the contribution feels incremental, by combining several existing concepts : virtual outlier synthesis, using PCA for geometric analysis, and using Mahalanobis distance for OOD detection. GCOS is a novel combination of these parts, but it does not feel like a fundamentally new approach. In its defense, the combination is well justified and seems effective.- the main results (Table 1) only compare GCOS against VOS and a No Reg. baseline. Authors should ideally benchmark their method against related modern SOTA OOD detection methods on Near-OOD datasets, completing Table 1. While the authors emphasis on Near-OOD is well-motivated, a discussion to support the fact that a method generalizing well to Near-OOD will also perform well on Far-OOD would be appreciated.

 - the central claim is that difficult outliers are superior for training. To prove that this claim is grounded, the authors should compare the strategy to a trivial synthesis method (e.g., generating points with a large range of fixed $\alpha$, or sampling random vectors near the origin).

- the method rests on two strong assumptions, linearity and cluster separation. Using PCA assumes the feature manifold is locally linear and that linear paths along low-variance eigenvectors are meaningful directions. This is a strong structure assumption for deep, high-dimensional feature spaces. Given the known nonlinear nature of deep features, a discussion on the validity of this local linearity assumption or its potential limitations for GCOS is necessary. Secondly, the use of Conformal in the title is questionable. The synthesis method is inspired by conformal prediction but provides no statistical guarantees. The actual formal conformal hypothesis testing (Contribution 3) is presented as Future Work and yields mixed results and can collapse to a nearly random classifier.

### Questions
- Could the authors provide results comparing GCOS against one or two more related modern SOTA methods (selected by themselves) on the Near-OOD benchmarks from Table 1 ?

- Can the authors provide an ablation study comparing GCOS to a trivial synthesis baseline (e.g., synthesis with a very large range of $\alpha$, or sampling random latent vectors far from the class custers) to demonstrate that the hard negative calibration from the conformal shell is truly necessary?

- How does GCOS handle cases where class manifolds are adjacent ? What prevents the hidden representation sampled from Class A generating a sample that is easily mistaken for Class B, thereby potentially confusing the classifier? Could they also discuss datasets with high uncertainty possibly leading to hard linear separation in the feature space ?

### Soundness
3

### Presentation
3

### Contribution
2
