# PhysTTT: Accurate and Lightweight Cross-Domain Heart Rate Measurement with Test-Time Training

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 6, 8

## Abstract
Remote photoplethysmography (rPPG), a contactless technology for measuring physiological signals, holds significant promise for smart healthcare and affective computing. However, a key challenge for existing deep learning methods is the paradox between maintaining high measurement accuracy and ensuring low computational cost, especially in cross-domain scenarios. To address this, we propose PhysTTT, a novel and lightweight framework for heart rate measurement that integrates multiple 1D-CNNs with residual structures and a Test-Time Training (TTT) layer. Multi-time frame differences fusion and 1D-CNNs extract spatio-temporal features from facial video sequences by modeling subtle brightness variations, the TTT layer compresses the context information into a learnable vector space, enhancing the temporal modeling capability. Crucially, the TTT mechanism enables the model to adapt to unseen data distributions during inference, significantly boosting cross-domain generalization. Extensive experiments demonstrate that PhysTTT achieves state-of-the-art accuracy in both in-domain and cross-domain evaluations, offering an optimal balance of high performance, strong generalization, and low computational cost. Our code is publicly available at https://anonymous.4open.science/r/PhysTTT-B605/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a lightweight rPPG measurement framework that leverages test-time training to enhance adaptability in cross-domain scenarios. The method focuses on enforcing temporal consistency through multi-objective losses that align waveform trends and spectral components with physiological patterns. The approach shows promise for practical deployment due to its low complexity and ability to update online at inference time (Table 5).

### Strengths
1. The model achieves competitive performance with a compact architecture, demonstrating a strong balance between computational efficiency and prediction accuracy suitable for real-time or mobile deployment.

2. By updating parameters during inference, the method adapts dynamically to unseen conditions and distributions, improving robustness when deployment environments differ from the training domain.

### Weaknesses
1. The frequency and waveform stability is mainly enforced by the definition of the lost function. While the model itself have simple architectures without explicit ability to help on the temporal structure of rPPG. A better design or an explanation of why keeping the model architecture simple is needed.

2. The cross-domain evaluation currently focuses only on UBFC-rPPG and VIPL-HR. These datasets share limited diversity in terms of illumination, subject motion, and demographic variations. As a result, the reported cross-domain gains may largely reflect spectral alignment improvements rather than true robustness to heterogeneous real-world conditions. Including more challenging and diverse datasets such as MMPD or BUAA would provide a more comprehensive assessment of generalization performance, particularly regarding sensitivity to skin tone variation, lighting shifts, and motion artifacts, which are key factors in practical deployment.

### Questions
1. What is the reason of separating the optimization process into an inner-outer loop? Does the optimization of W and theta be split to ensure convexity or smoothness? Or it is simply based on empirical observation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a novel Test-Time Training (TTT) framework for remote physiological signal measurement. The proposed approach employs multi-frame difference fusion techniques and leverages 1D-CNNs to extract spatio-temporal features from facial video sequences. The TTT layer subsequently compresses contextual information from video frame sequences into a self-supervised learning model, enabling adaptation to unseen domain distributions during the testing phase. Experimental results demonstrate that the method outperforms existing state-of-the-art approaches, exhibiting precise and efficient measurement capabilities.

### Strengths
1. The method systematically introduces Test-Time Training into the Test-Time Adaptation framework for remote photoplethysmography (rPPG), effectively addressing domain shift issues in this field.
2. The proposed approach achieves a favorable balance between measurement accuracy and computational cost, thereby enhancing the generalizability of remote physiological signal measurement, which is particularly valuable for real-world deployment.
3. Comprehensive experiments validate the exceptional performance of the method in both in-domain and cross-domain scenarios.
4. The paper is well-written with a clear narrative structure, adequately covers related work, and distinctly highlights its contributions.

### Weaknesses
1. In lines 084–086 of the introduction, the authors state that “While SSMs offer a better balance between linear complexity and long-range modeling, they can face limitations in generalization and parallel processing, particularly in cross-domain scenarios where model robustness is critical.” This claim lacks sufficient justification or empirical evidence.
2. In line 088 of the introduction, there are several citation errors. Specifically, Dual-GAN is a standard supervised learning method, while Dual-bridging belongs to the domain generalization category, not domain adaptation, as stated by the authors.
3. Section 2.2 of the related work is unclear and lacks logical structure. It is not evident what the authors intend to convey, nor why “PhysTTT effectively captures spatial information.” This section needs a clearer focus and stronger justification.
4. The Frame Stem module described in Section 3.2 is nearly identical to those used in RhythmFormer and RhythmMamba. This component should not be claimed as a novel contribution, and the authors must explicitly clarify this overlap in the paper.
5. All formulas in the paper are missing serial numbering, which violates the standard formatting and submission requirements of academic papers.
6. Both the intra-dataset and cross-dataset experiments only consider illumination variations, without evaluating robustness under other common noise factors such as head motion. This is a significant limitation in the experimental design.
7. The paper reports cross-domain experiments only on UBFC-rPPG and VIPL-HR, which are not sufficiently diverse. It is recommended to validate the proposed method on a broader range of datasets (e.g., MMPD, MR-NIRP-Car) to demonstrate robustness and generalization.
8. Table 5 mentions “Throughput” in its caption, but the corresponding quantitative results are missing. 
9. The paper lacks crucial ablation studies on these core modules and the self-supervised learning strategy, as well as validation on alternative backbones to verify the generality of the proposed approach.

### Questions
1. The paper does not clearly explain the conceptual and practical differences between TTT (Test-Time Training) and TTA (Test-Time Adaptation). What are the respective advantages and disadvantages of TTT compared to TTA, and why is TTT considered more suitable for rPPG tasks?
2. The source of the model’s strong cross-domain performance remains unclear. To what extent does it stem from the carefully designed 1D-ResNet and frame-difference backbone, and to what extent from the TTT adaptation layer?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose PhysTTT, a Test Time Training framework for remote heart rate estimation. The authors claim this to be a first time experiment with TTT in the domain and this seeks to address cross-domain generalization.

### Strengths
- The authors aim to tackle a known problem of cross-domain generalization in the rPPG domain. This is crucial as rPPG signals are extremely sensitive to environment settings, and also to skin tones.

### Weaknesses
- The authors can explore more datasets where the experiments settings are different.
  - For example, PURE has 6 types of movement in the dataset. COHFACE has natural and artificial lightning and is a challenging dataset.
- The authors should also compare their work with other approaches on the task to further elucidate their results and contributions.

### Questions
- I understand that some of the design choices around using 1D CNNs was around efficiency, however do the authors think too much information is being lost in the pooling layer in the Frame-Stem, and then subsequently utilizing 1D CNNs in the pipeline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
To address the paradox between high accuracy and low computational cost in cross-domain remote photoplethysmography (rPPG)-based heart rate measurement, this paper proposes PhysTTT, a lightweight framework integrating multiple 1D-CNNs with residual structures and a Test-Time Training (TTT) layer. The frame stem amplifies subtle skin color variations through multi-time frame differences fusion, 1D-ResNet layers extract local spatio-temporal features, and the TTT layer dynamically adapts to unseen data distributions during inference by updating model weights, thereby enhancing cross-domain generalization. Equipped with a multi-dimensional loss function (trend pattern alignment, frequency domain alignment, and waveform feature alignment) for fine-grained BVP signal recovery, PhysTTT is evaluated on UBFC-rPPG and VIPL-HR datasets, achieving state-of-the-art performance in both in-domain and cross-domain (including cross-dataset and cross-illumination) scenarios while maintaining low computational cost (42.64M MACs and 7.08M peak GPU memory usage), demonstrating great potential for real-world healthcare applications.

### Strengths
1. It is the first work to introduce the Test-Time Training (TTT) paradigm into rPPG research, effectively solving the challenge of adapting to unseen data distributions in cross-domain scenarios that traditional methods struggle with.
2. The multi-dimensional loss function designed for rPPG tasks (integrating negative Pearson correlation loss, power spectral density loss, and peak alignment loss) enables precise alignment of predicted and ground truth BVP signals, significantly improving the accuracy of heart rate measurement.
3. The comprehensive experimental validation (covering in-domain, cross-dataset, and cross-illumination evaluations) and ablation study fully demonstrate the effectiveness of each module, while the lightweight design (low parameters, MACs, and GPU memory usage) ensures its applicability on resource-constrained devices.

### Weaknesses
1. The paper does not provide detailed analysis on the real-time performance of PhysTTT, especially the additional latency introduced by the TTT layer’s parameter updates during inference, which is critical for practical deployment in real-time health monitoring.
2. Cross-domain evaluations are limited to two datasets (UBFC-rPPG and VIPL-HR) and three illumination conditions, lacking validation in more diverse scenarios such as different camera devices, extreme motion, or varied skin tones, which may restrict the generalization of the conclusions.
3. The ablation study only verifies the overall contribution of key modules (frame stem, 1D-ResNet, TTT layer) but fails to explore the impact of critical hyperparameters (e.g., the balance factor α in peak alignment loss) or compare with different TTA variants, leaving room for optimizing the model’s design.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
