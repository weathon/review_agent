# acuSimNet: Multi-View Self-Occlusion-Awared Visibility Learning for Cranio-Cervical Acupuncture Points

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
The localization of acupuncture points (acupoints) in Traditional Chinese Medicine (TCM) presents unique challenges since they are defined by abstract principles rather than distinct anatomical landmarks. Existing approaches are typically constrained to single view or rely on indirect calculations of relative positions with respect to other landmarks, thereby overlooking human anatomical variations. Furthermore, acupoint visibility assessment, which determins whether points are occluded by human itself, hair, clothing, or other objects, has received limited attention in practical applications. Acupoint localization itself does not require 3D reconstruction, but inferring their occlusion relationships with anatomical surfaces does, which adds computational cost and limits real-time inference. In this work, we introduce acuSimNet, an efficient hierarchical multi-task learning architecture for multi-view, self-occlusion-aware visibility prediction of acupoints, achieving 99.97% accuracy on the validation set. Our approach also addresses the challenges of high-dimensional classification (174 acupoints in cervicocranial region), negative convergence issues for visible acupoints, and inter-task scheduling optimization, resulting in substantially accelerated convergence. We improved the training efficiency from the exisiting methods of 3000 epochs to achieve 99% validation accuracy, to our optimized framework achieving 90% accuracy in 39 epochs and 99% accuracy in 86 epochs. This architecture overcomes the limitations of existing methods, could enable practical applications in acupoints detection and visualization, advancing the automation of TCM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ACUSimNet, a hierarchical multi-task framework for multi-view, self-occlusion-aware visibility learning of 174 cranio-cervical acupuncture points. The model uses a DenseNet backbone with nine meridian-specific branches and jointly learns (1) acupoint visibility classification, (2) coordinate regression, and (3) acupoint identity classification. The design includes per-meridian uncertainty weighting, soft visibility mask padding, and a decaying weight function to suppress false positives. Experiments on the synthetic acuSim dataset show extremely high visibility accuracy (up to 99.97%) and a reported 35× faster convergence compared with prior baselines (86 vs. 3000 epochs). Ablation studies analyze different backbones, soft-masking, and uncertainty initialization.

### Strengths
1）Clear motivation and clinically relevant context: Occlusion-aware visibility reasoning is important for safety-critical acupuncture localization, where false positives may lead to physical harm.
2）Systematic engineering integration: The meridian-based hierarchy, uncertainty-weighted multi-task formulation, and soft masking are well-motivated and implemented coherently.

### Weaknesses
1）Evaluation limited to synthetic data: All experiments are conducted on the acuSim synthetic dataset. There is no evaluation on real-world or clinical imagery, where lighting, hair, skin tone, and occlusion patterns differ significantly. The claimed “clinical applicability” therefore lacks empirical support. The near-perfect results (99.97% accuracy, 0% false positives) suggest strong domain overfitting to the synthetic environment.
2）Weak baseline comparisons: The study mainly compares internal ablations and network backbones but does not benchmark against strong, existing visibility- or keypoint-based baselines (e.g., HRNet, MediaPipe, Transformer-based keypoint detectors). Without these comparisons, it is difficult to assess the true performance gain.
3）Limited methodology insights: Although the topic is interesting, the methodological contribution is modest for ICLR. The paper overemphasizes clinical relevance and convergence speed without demonstrating generality or new learning insights.
4）Limited analysis of causal interaction between tasks: The hierarchical structure assumes visibility should precede localization/classification, but no ablation or comparison validates this causal design. It remains unclear whether visibility prediction genuinely improves localization beyond simple confidence-weighted baselines.
5）Writing: The mathematical descriptions in Equations (6)–(10) are not expressed in a fully standard or rigorous way. Also, Since the submission is intended for ICLR, quantitative and qualitative experiments are very essential for evaluating the proposed model’s performance. Thus, they should appear in the main submission, not the appendix.
6）Failure cases: Since experimental results are extremely good, it will be better to discuss some failure cases to enhance the comprehensiveness of the proposed framework.

### Questions
•  Add real-world validation (even a small-scale dataset with manual annotations) to test generalization beyond synthetic data.
•  Include strong baselines (e.g., HRNet, DETR-like keypoint models, Transformer-based detectors with occlusion simulation).
•  Clarify contribution positioning—emphasize this work as a domain-driven, system-level engineering improvement, not as a fundamentally new learning approach.
•  Discuss failure cases (misclassified visibility, localization drift) and potential mitigation strategies.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces acuSimNet, a multi-task deep learning framework designed for cranio-cervical acupuncture point localization under multi-view and self-occlusion conditions.
The model uses DenseNet201 as the backbone and incorporates Traditional Chinese Medicine (TCM) prior knowledge by grouping 174 acupuncture points into 9 meridian-specific branches. The proposed framework jointly optimizes three objectives: visibility prediction, coordinate regression, and point classification.

Key design elements include:

1. An explicit visibility prediction head to detect occluded points;

2. A soft visibility mask that dynamically modulates regression and classification losses;

3. Learnable uncertainty weighting for automatic task balancing;

4. A decaying weight function to reduce the impact of occluded samples;

5. Meridian-based hierarchical structure inspired by domain priors.

Experiments on a synthetic dataset (acuSim) show high reported performance (99.97% visibility accuracy, 0% false positives) and faster convergence (86 epochs vs. 3000 in the baseline).
While the engineering execution is solid, the overall contribution relies heavily on combining well-established components rather than proposing a fundamentally novel algorithm or theory.

### Strengths
**1. Significant performance improvements:**  high accuracy, low false-positive rate, and 35× training speed-up.

**2. Well-defined tasks**: Tailored to the specific needs of acupuncture point localization (visibility and safety), the training objectives are appropriately designed.

**3. Cross-disciplinary integration**：Good motivation from a real-world application, showing potential practical value for clinical.

### Weaknesses
**1. Lack of Novelty:** The paper's main methodological components—multi-task learning with uncertainty weighting, visibility-aware modeling, and domain-specific hierarchical grouping—have all been proposed and extensively explored in prior literature. The present work mainly integrates these well-known ideas into a specific medical application without introducing new theoretical mechanisms or learning paradigms.

- **Uncertainty-based task weighting:** The idea of dynamically adjusting loss weights across tasks using task-dependent uncertainty has become a standard technique for multi-task learning. The current paper follows the same formulation with minor adaptation to a medical dataset, offering limited methodological novelty.
  - Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," CVPR 2018.

- **Visibility-aware or occlusion modeling:** Similar strategies for modeling keypoint visibility have appeared in human pose estimation and 3D reconstruction. These works already treated visibility prediction as an auxiliary or explicit branch to improve robustness under self-occlusion. The visibility branch proposed in this paper is conceptually similar, without introducing a fundamentally different mechanism.
  - Mehta et al., "VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera," TOG 2017
  - Zhao et al., "Visibility-Aware Human Pose Estimation via Self-Supervised Occlusion Learning," CVPR 2020
  - Sun et al., "Integral Human Pose Regression," ECCV 2018

- **Hierarchical or anatomically structured modeling:** Organizing prediction heads by body parts or semantic groups has been used widely in keypoint detection and medical landmark localization. The proposed "meridian-specific branches" follow the same principle of anatomical partitioning, adapted to the TCM domain but not conceptually novel.
  - Payer et al., "Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization," MICCAI 2019
  - Papandreou et al., "PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model," ECCV 2018

- **Soft-masking for uncertain targets:** Soft or visibility-aware masks have also been used to modulate losses for uncertain or occluded keypoints. The proposed soft visibility mask appears to reuse this idea with domain-specific naming.
  - Zeng et al., "PoseFlow: Efficient Online Pose Tracking," BMVC 2018
  - Tang et al., "Learning Visibility for Robust Dense Optical Flow," CVPR 2019

**2. Weak experimental persuasiveness:** No comparison with mainstream baseline models, but only ablation studies.

**3. No validation on real data:** Experiments rely entirely on synthetic data; domain adaptation or generalization analysis is absent.

**4. Domain-specific innovation lacks generalizability:** The design is tailored to acupuncture point localization and offers little transferable insight to broader vision or learning communities.

**5. The experiments are overly focused on the visibility task:**  Visibility prediction is highlighted, while detailed analyses of coordinate-regression performance, error distributions, and cross-view generalization are lacking.

**6. The ablation study is incomplete:** No experiments ablating the number of groups were conducted, so it remains unclear whether the current number of meridian-specific branches actually incorporates meaningful medical prior knowledge.

**7. Thin theoretical contribution:** No new derivations, theoretical analyses, or generalization guarantees are provided.

### Questions
**1.Novelty Question:**  How does acuSimNet differ fundamentally from existing visibility-aware multi-task frameworks?

**2.Experimental Persuasiveness Question:**  Can you provide more baselines for comparison?

**3.Medical Prior-Based Meridian Layering Question:**  Does the meridian-based grouping offer statistically significant improvement over non-grouped models? And have you tested other numbers of groups to verify that the current grouping scheme indeed incorporates meaningful medical prior knowledge?

**4.Generalization Question:**  Have you tested generalization on real or clinical images? If not, what measures mitigate the synthetic–real domain gap?

**5.Theoretical Contribution Question:**

- Can you provide any theoretical intuition or formal analysis showing why the proposed visibility masking and uncertainty weighting lead to faster convergence or improved generalization?
- If the contribution is primarily empirical, please clarify how the framework advances understanding of multi-task coordination or uncertainty modeling, rather than only reapplying known formulations.

### Soundness
2

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
5

### Summary
The paper proposes a method called acuSimNet to accurately locate acupoints by designing an efficient hierarchical multi-task learning
architecture for multi-view and self-occlusion-aware visibility prediction. The approach is validated by comprehensive experiments in public dataset.

### Strengths
1. The problem of automatically locating acupoint is interesting in traditional Chinese medicine.
2. The overall structure of the paper is complete.

### Weaknesses
1. The problem of automatic acupoint localization lacks importance and interest in ICLR community
2. The proposed method is too trivial and with few novelty.
3. Line 24, what is "negative convergence issues".
4. Line 50, how to come to the conclusion that "...resulting in training instability and suboptimal convergence characteristics", any evidence to support?

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
