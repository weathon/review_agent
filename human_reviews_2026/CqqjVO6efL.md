# Cross-modal Transfer Through Time for Human Activity Recogntion

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Cross-modal knowledge transfer between time-series sensors remains a critical challenge for robust Human Activity Recognition (HAR) systems.
Effective cross-modal transfer exploits knowledge from one modality to train models for a completely unlabeled target modality—a problem setting we refer to as Unsupervised Modality Adaptation (UMA).
Existing methods typically compress continuous-time data samples into single latent vectors during alignment, limiting their ability to transfer temporal information through real-world temporal distortions.
To address this, we introduce Cross-modal Transfer Through Time (C3T), which preserves fine-grained temporal information during alignment to handle dynamic sensor data better.
C3T achieves this by aligning a set of temporal latent vectors across sensing modalities.
Our extensive experiments on various camera+IMU datasets demonstrate that C3T outperforms existing methods in UMA by over 8\% in accuracy and shows superior robustness to temporal distortions such as time-shift, misalignment, and dilation.
Our findings suggest that C3T has significant potential for developing generalizable models for time-series sensor data, opening new avenues for various multimodal applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Cross-modal Transfer Through Time (C3T) for Unsupervised Modality Adaptation (UMA) in Human Activity Recognition (HAR), focusing on transferring knowledge from labeled RGB videos to unlabeled IMU data. Existing methods compress time-series into single latent vectors, losing temporal details; C3T aligns temporal latent vectors from convolution feature maps using an extended contrastive loss (LC3T), followed by self-attention for classification.

### Strengths
1. **Relevant Problem and Innovation**: The UMA setting is practical for unlabeled sensors in wearables/IoT, addressing real-world challenges like labeling scarcity and temporal variations. 
2. **Empirical Rigor**: Comprehensive evaluations on diverse datasets (UTD-MHAD, CZU-MHAD, MMACT, MMEA-CL) demonstrate consistent outperformance of baselines, with ablations on latent size and noise. 
3. **Potential Impact**: Opens avenues for generalizable time-series models in healthcare, smart homes, and HCI, with implications for emerging modalities beyond RGB-IMU.

### Weaknesses
1. **Limited Novelty**: While effective, C3T feels incremental. Temporal alignment echoes recent frameworks like COMODO, and multi-modal contrastive learning extensions (e.g., CoMM). The paper cites a 2024 IMU cross-modal survey but overlooks broader 2024-2025 surveys on generalizable HAR and multi-modal CL advances, potentially understating similarities to sequence-based CL in video/HAR. Baselines are appropriate but not SOTA.
2. **Scope and Assumptions**: Relies on paired RGB-IMU data; no handling of unpaired scenarios common in real-world settings. Limited to RGB-to-IMU; lacks tests on other pairs (e.g., WiFi, radar). Datasets are small/old.
3. **Analytical Depth**: No theoretical justification for LC3T's robustness (e.g., invariance bounds). Privacy/ethics in sensor data ignored. Broader impacts mentioned but not explored deeply.

### Questions
See above.

### Soundness
2

### Presentation
2

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
This paper presents a novel method (C3T) and tackles an important problem, namely Unsupervised Modality Adaptation for cross-modal human activity recognition.

### Strengths
1. The paper clearly defines and formalizes the "Unsupervised Modality Adaptation" (UMA) setting, which is a meaningful and practical contribution for real-world sensor-based HAR systems.

2. The core idea of C3T—aligning fine-grained temporal latent vectors instead of a single global vector, which is intuitively appealing for time-series data.

3. The results demonstrate that C3T consistently and significantly outperforms strong baselines (ST and CA) in the UMA setting.

### Weaknesses
1. The paper only compares C3T against two adapted baseline families (Student-Teacher and Contrastive Alignment). There is no comparison with recent, powerful, and published methods designed for cross-modal learning, missing modality problems, or domain adaptation for time-series data.The paper fails to situate C3T within the current landscape of SOTA cross-modal HAR or video-IMU fusion techniques like Ts2act or Vi2ACT, to name a few. A reviewer cannot be confident that C3T represents a true advancement without these comparisons.
2. The ST and CA baselines, while reasonable starting points, are not pushed to their limits. For instance, was the temperature parameter (
τ) in the contrastive loss optimized for the CA baseline? Were more advanced distillation techniques explored for the ST baseline? Weak baselines may weaken the perceived performance improvement of C3T. 
3. Performance on larger, more challenging datasets like MMACT (32.4% top-1 accuracy) is still very low, indicating that the method struggles with high diversity and is far from solving the general UMA problem.

### Questions
1. The performance improvement is attributed to the temporal alignment loss (L C3T ). However, C3T also uses a different HAR decoder (self-attention) compared to CA (MLP). The ablation in Table 4 attempts to address this but is confusing. It shows that a C3T variant with an MLP head can achieve higher accuracy (70.5%) on clean data than the chosen class-token attention head (62.5%). This critically undermines the claim that the self-attention mechanism is a key component for performance. It suggests that the primary benefit comes from the temporal alignment strategy itself, and the chosen architecture might even be suboptimal. A clearer ablation is needed.

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
This paper introduces Cross-modal Transfer Through Time (C3T), a novel method for Unsupervised Modality Adaptation (UMA) in human activity recognition. C3T aligns temporal latent vectors across modalities (e.g., RGB and IMU) to preserve fine-grained temporal information, outperforming existing Student-Teacher and Contrastive Alignment methods.

### Strengths
Extensive experiments across four diverse datasets validate the method’s superiority in both accuracy and robustness. Ablation studies and visualizations (e.g., t-SNE, attention maps) provide strong empirical support.

### Weaknesses
This paper applies Unsupervised Modality Adaptation (UMA) to human activity recognition (HAR) and proposes a Cross-modal Transfer Through Time (C3T) method. The authors demonstrate that C3T outperforms ST and CA on a self-designed benchmark under the UMA setting, but I have several major concerns: 
1.UMA has already been studied in other tasks, and merely applying it to HAR does not constitute significant novelty; 
2.The baselines are overly simple, including only ST and CA, lacking comparisons with more advanced methods;
3.The benchmark is self-designed, and basic UMA methods are not evaluated on it, lacking comparison with existing work, so the claimed SOTA results are not convincing; 
4.The method itself shows limited novelty, and the overall contribution is insufficient to justify a full-length paper.

### Questions
Please refer to Weakness.

### Soundness
2

### Presentation
2

### Contribution
2
