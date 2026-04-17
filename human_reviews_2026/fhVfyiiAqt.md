# Point-UQ: An Uncertainty-Quantification Paradigm for Point Cloud Few-Shot Class Incremental Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
3D few-shot class-incremental learning (3D FSCIL) requires effectively integrating novel classes from limited samples while preserving base-class knowledge, without succumbing to catastrophic forgetting the learned knowledge or overfitting the novel ones. 
Current 3D FSCIL approaches predominantly focus on fine-tuning feature representations yet retain static decision boundaries. 
This leads to a critical trade-off: excessive adaptation to new samples tends to erase previously learned knowledge, while insufficient adaptation hinders novel-class recognition. 
We argue that the key to effective incremental learning lies not only in feature enhancement but also in adaptive decision-making. 
To this end, we introduce **Point-UQ**, an incremental training-free paradigm for 3D **point** clouds based on **u**ncertainty **q**uantification, which shifts the focus from feature tuning to dynamic decision optimization. 
Point-UQ comprises two co-designed modules: *Attention-driven Adaptive Enhancement (AAE)* and *Uncertainty-quantification Decision Decoupling (UDD)*.
The former module fuses multi-scale features into calibrated representations, where prediction entropy serves as a reliable measure of per-sample epistemic uncertainty while preserving original feature semantics. Building on AAE-derived calibrated entropy, the UDD module dynamically arbitrates between semantic classifiers and geometric prototypes—enabling robust base-class knowledge retention and accurate novel-class recognition in 3D FSCIL without retraining. 
Extensive experiments on ModelNet, ShapeNet, ScanObjectNN, and CO3D demonstrate that our approach outperforms state-of-the-art methods by 4% in average accuracy, setting a new standard for robust 3D incremental learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Point-UQ, a training-free uncertainty-quantification framework for 3D few-shot class-incremental learning. Unlike existing approaches that rely on fine-tuning feature representations with static decision boundaries, Point-UQ focuses on decision optimization. The framework integrates two key modules: Attention-driven Adaptive Enhancement and Uncertainty-Quantification Decision Decoupling. Extensive experiments show that Point-UQ outperforms state-of-the-art baselines across multiple benchmarks.

### Strengths
1. The paper is well-motivated, clearly contrasting conventional feature-tuning paradigms with the proposed decision-centric approach and providing novelty to this task.
2. The design of AAE and UDD is technically sound: AAE enhances representation quality while providing uncertainty cues, and UDD leverages entropy to arbitrate between semantic and geometric reasoning.
3. The experimental evaluation is extensive, covering both intra-dataset and cross-dataset benchmarks, and demonstrates the effectiveness of the proposed framework.

### Weaknesses
1. Some ablations on hyperparameters are missing. For example, how sensitive is the method to the choice of the entropy scaling factor (Eq 12) in UDD?
2. The uncertainty estimation relies solely on entropy from the semantic classifier. It would be useful to investigate alternative uncertainty measures and analyze their impact on performance.
3. Missing related work. Several recent works on 3D few-shot learning are relevant and should be cited:
   - Rethinking few-shot 3d point cloud semantic segmentation (CVPR 2024)
   - Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation (ICLR 2025)
   - Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model (CVPR 2025)

### Questions
Please refer to the weakness part and address the concerns there.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Point-UQ, a novel framework designed for 3D few-shot class-incremental learning (FSCIL). The key innovation of Point-UQ lies in its approach to dynamic decision optimization through uncertainty quantification. The framework integrates two co-designed modules: the Attention-driven Adaptive Enhancement (AAE) for fusing multi-scale features and generating epistemic uncertainty, and the Uncertainty-quantification Decision Decoupling (UDD) module, which helps retain base-class knowledge while facilitating novel-class recognition.

### Strengths
1. This paper presents a well-structured pipeline from multi-scale feature fusion to decision-making, addressing core challenges in 3D FSCIL, such as catastrophic forgetting and overfitting on novel classes.
2. The use of uncertainty quantification in decision-making (AAE and UDD modules) is a novel and effective strategy to tackle the trade-offs in incremental learning.
3. The method outperforms existing approaches in various metrics, including average accuracy, forgetting rate, and harmonic accuracy, demonstrating its robustness across different datasets and evaluation setups.

### Weaknesses
1. Fairness of Experiments: This paper is based on a strong pre-trained model, Uni3D, for initialization. As shown in Table 3, even without the addition of AAE and UDD, the model's performance is already better than other methods. This raises concerns about the primary sources of performance improvement in the proposed approach.
2. Outdated Benchmark: The benchmark datasets used in this paper are relatively old. I suggest the authors additionally evaluate the model’s performance on the FSCIL3D-XL benchmark [1] to provide a more up-to-date comparison.

[1] FILP-3D: Enhancing 3D few-shot class-incremental learning with pre-trained vision-language models. Pattern Recognition, 2025, 165: 111558.

### Questions
My questions are mainly based on the above weaknesses:
1. If a different base model is used to extract point features, would the performance of the proposed method still be strong?
2. How does the model perform on the FSCIL3D-XL benchmark?
3. Could the authors consider testing the model on newer datasets, such as Objaverse [1,2], to further evaluate its performance?

[1] Objaverse: A universe of annotated 3d objects. CVPR, 2023: 13142-13153.
[2] Objaverse-xl: A universe of 10m+ 3d objects. NIPS, 2023, 36: 35799-35813.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a unified method to estimate and interpret uncertainty in 3D point cloud tasks such as classification and segmentation. It introduces a lightweight probabilistic module that models both epistemic and aleatoric uncertainties directly within point-based neural networks without requiring multiple forward passes. The framework leverages a dual-branch design—one for feature learning and another for uncertainty estimation—to identify unreliable regions in 3D space, enabling more robust predictions and better calibration under noisy or sparse data conditions. Experimental results on benchmark datasets demonstrate that Point-UQ improves model reliability and interpretability while maintaining competitive accuracy.

### Strengths
1. Adoption of decision optimization for incremental learning instead of conventional fine-tuning approaches.

2. Uncertainty-based quantification for decision-making, providing higher weight to geometric prototypes or semantic representations accordingly.

3. Training-free and computationally efficient method.

### Weaknesses
1. The authors did not discuss the effect of replacing the sigmoid function in Equation (12) with other activation functions.

2. The paper mentions handling overfitting and catastrophic forgetting, but comparative results showing model success versus baseline failures are missing.

3. Tables could be clearer and more reader-friendly. For example, in table headings, the numbers listed below the dataset names are not explained and should be clarified.

4. While the method claims to improve reliability without multiple forward passes, the paper provides limited quantitative analysis of calibration metrics (e.g., ECE, NLL) and does not fully benchmark computational trade-offs compared to ensemble-based or Bayesian baselines.

### Questions
1. Can the authors experiment with different functions instead of the sigmoid function in Equation (12)?

2. Can the authors display the intermediate output of Equation (12) during model tuning to verify whether it correctly weights according to uncertainty?

3. The paper does not mention the model’s performance with respect to seed sensitivity. Methods of this kind often vary significantly with random seeds—can the authors comment on this?

4. Pl comment on the calibration point.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Point-UQ, a training-free uncertainty-guided paradigm for 3D few-shot class-incremental learning that replaces feature fine-tuning with dynamic decision optimization. Experiments on four benchmarks report roughly 4% average accuracy gains over existing methods while reducing catastrophic forgetting.

### Strengths
1. Point-UQ introduces a paradigm shift from feature fine-tuning to decision optimization, resolving the core trade-off between retaining base-class knowledge and adapting to novel classes in 3D FSCIL while avoiding the high training costs and catastrophic forgetting of conventional methods.

2. Point-UQ’s AAE and UDD modules are tightly integrated, as AAE generates calibrated features and reliable uncertainty estimates, and UDD employs these signals for adaptive decision-making, effectively combining geometric structure modeling and semantic consistency that are critical for 3D point cloud analysis.

3. Point-UQ shows improvements across multiple benchmarks and metrics (average accuracy, harmonic mean, forgetting rate), and its theoretical analysis formally proves advantages over fine-tuning in error bounds, complexity, and feature stability.

### Weaknesses
1. The intra-dataset evaluation omits comparison with the strongest baseline, FoundationModel. The authors explain that reproducible intra-dataset numbers could not be obtained, but no detailed comparison between their reproduced values and the original paper is provided. The claimed improvements may therefore partly reflect implementation differences.

2. No qualitative analysis links prediction entropy to actual sample ambiguity, making it hard to verify that the UDD module switches between semantic and geometric cues as intended.

3. All experiments adopt a 5-shot protocol and performance in more extreme 1- or 2-shot regimes, where prototype construction is fragile, is not examined.

### Questions
1. Can intra-dataset results for FoundationModel be supplied, or at least the numerical discrepancies encountered during reproduction, so that readers can judge whether performance gaps are methodological or implementational?

2. Can high- and low-entropy examples be visualized together with the corresponding UDD decisions to confirm that uncertainty estimates capture genuine ambiguity?

3. How does accuracy degrade under 1-shot or 2-shot conditions, and is the K-means prototype still reliable?

4. Does the approach maintain low forgetting and modest resource usage when the number of incremental stages grows well beyond ten?

### Soundness
3

### Presentation
3

### Contribution
3
