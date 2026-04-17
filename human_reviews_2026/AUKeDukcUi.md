# MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Multimodal learning requires integrating heterogeneous modalities, such as structured records, visual imagery, and temporal signals. It has been revealed that this heterogeneity causes modality encoders to converge at different rates, making the multimodal learning imbalanced. We empirically observe that such an imbalance is related to the sharpness of the solution. Modality encoders that converge faster could be dragged into sharp regions due to inter-modal interference, degrading the generalization capability of unimodal features learned. Sharpness-Aware Minimization is effective in improving generalization via finding solutions in flat regions. However, its application in multimodal scenarios is challenging: 1) SAM overemphasizes the dominant modality, inducing misaligned perturbations in weaker modalities, and 2) the perturbation gradient calculation is affected by interference from other modalities. To address these issues, we propose Multimodal Adaptive Sharpness-Aware Minimization (MASAM), which optimizes different modalities based on their dominance. We design an Adaptive Perturbation Score (APS) using convergence speed and gradient alignment to identify dominant modalities for SAM application. Our Modality-Decoupled Perturbation Scaling (MDPS) then reduces inter-modal interference during optimization, better aligning each modality with shared information. Extensive empirical evaluations on five multimodal datasets and six downstream tasks demonstrate that MASAM consistently attains flatter solutions, achieves balanced multimodal learning, and subsequently surpasses state-of-the-art methods across diverse datasets and tasks. Code is available at https://github.com/Orange2107/MASAM-Multimodal-Adaptive-SAM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MASAM, a multimodal optimization framework that adapts Sharpness-Aware Minimization (SAM) to address modality imbalance in multimodal learning. The authors identify two main issues when applying SAM to multimodal settings—amplified modality imbalance and modality-agnostic perturbation—and propose two components: the Adaptive Perturbation Score (APS) and Modality-Decoupled Perturbation Scaling (MDPS). APS dynamically identifies dominant modalities based on convergence speed and gradient alignment, while MDPS decouples perturbations to reduce inter-modal interference. The method demonstrates improved performance across multiple multimodal datasets, achieving flatter minima and better balance among modalities.

### Strengths
1. The paper presents an interesting and well-motivated attempt to extend SAM to multimodal learning, where convergence dynamics differ across modalities.
2. The introduction of APS and MDPS is intuitive and addresses the challenge of applying sharpness-aware optimization in multimodal settings.
3. The experimental results are extensive and consistent, showing that MASAM achieves improved generalization and balanced performance across diverse datasets.
4. The visualizations are impressive.

### Weaknesses
1. Lines 90–92 state that “SAM’s uniform sharpness minimization thus disproportionately benefits dominant modalities with faster convergence.” However, in the proposed MASAM framework, SAM is applied only to dominant modality-specific encoders. This design choice appears contradictory to the stated motivation—if SAM inherently benefits dominant modalities, applying it exclusively to them seems counterintuitive. The logic connecting the problem statement and the solution needs clearer justification.
2. The introduction of unimodal auxiliary objectives and the APS mechanism represents another gradient-based modulation approach, which has been explored in existing works such as MMPareto, OGM, and related gradient alignment strategies. The paper would benefit from a clearer differentiation from these prior methods, especially in how APS or MDPS uniquely contributes beyond existing gradient-based optimization schemes.
3. In the discussion section (Line 422), the authors claim that “MASAM enables all modalities to converge jointly to flatter minima.” However, if a modality remains consistently weak (non-dominant) throughout training, no perturbation is applied to it. This raises doubts about whether MASAM truly ensures joint convergence across all modalities or only benefits dominant ones. A more detailed discussion of this limitation is warranted.

### Questions
1. The SAM technique seems to require multiple passes of gradient computation, which raises concerns about the efficiency of the proposed method.

I would consider raise my rating if the above concerns could be sufficiently addressed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors presented MASAM, a novel method that applies sharpness-aware minimization (SAM) to multimodal setting. The paper identifies the issue with directly applying SAM to multimodal setting where modality imbalance could be exacerbated by SAM, and devised a method to decouple the modalities when determining the scales of the perturbations. The proposed method, MASAM, first determines the dominant modality using APS, then applies perturbation on the dominant modality using MDPS. The proposed method was evaluated over 6 tasks across 5 very different datasets from different domains against a large set of baseline fusion methods, and MASAM achieves top performance across all tasks. Additional analysis was conducted to highlight MASAM's robustness to label noise and ability to allow all modalities to converge jointly to a flatter minima, and ablation studies was conducted to demonstrate the necessity for both APS and MDPS modules.

### Strengths
1. The proposed method outperforms many baselines over 5 very different domains, demonstrating MASAM's generalizability across many multimodal application domains.

2. The overall presentation quality of the paper is good. It is overall easy to follow.

3. There is theoretical analysis to show that naive SAM does not work well under multimodal settings, and that MASAM can better converge all modalities.

4. There is additional visualizations and empirical analysis to demonstrate MASAM's more balanced learning across modalities and allows all modalities to converge to a flatter minimum. The additional analysis also shows MASAM's better robustness to label noise.

5. There is ablation study to demonstrate the necessity of both APS and MDPS modules.

### Weaknesses
The paper could be improved by including more justification/analysis for the necessity of both Decay and $\gamma$ in APS. In figure 10a, it seems like low $\alpha$ generally performs quite well, so is Decay actually necessary? There needs to be an ablation study to justify this (i.e. what happens if you set $\alpha$ to 0?)

### Questions
I am a bit confused about the setup for calculating Decay. Can you explain why it is calculated as the difference between the previous round's absolute loss and the current round's moving average? Also, in Figure 10b, it seems like a high $\beta$ works better, but wouldn't a higher $\beta$ make the Decay smaller and more likely to be zero?

### Soundness
3

### Presentation
4

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
This paper presents MASAM (Multimodal Adaptive Sharpness-Aware Minimization) to address the challenges of modality imbalance and heterogeneous loss geometries in multimodal learning. The authors introduce Adaptive Perturbation Score (APS) and Modality-Decoupled Perturbation Scaling (MDPS) to ensure balanced multimodal learning while benefiting from Sharpness-Aware Minimization (SAM). Their approach is demonstrated to achieve flatter loss landscapes for each modality, resulting in improved generalization across diverse datasets and tasks.

### Strengths
- The paper presents both qualitative and quantitative experiments across five multimodal datasets, with six downstream tasks, demonstrating MASAM's effectiveness in diverse domains, including clinical, video, and image-text datasets.
- The problem of modality imbalance is well-motivated, and the proposed solution is thoroughly explained. The use of SAM in multimodal learning is a novel approach to addressing this problem.
- The writing is clear, logical, and easy to follow.

### Weaknesses
- The loss landscape analysis is only conducted on the small clinical MIMIC dataset. It would be more convincing if similar visualizations were provided for larger and more diverse "in-the-wild" datasets, ensuring the scalability and generalizability of the approach across various real-world data settings.
- The sensitivity of hyperparameters, such as $\alpha$ and $\rho$, is not sufficiently explored. A more detailed analysis of how these parameters vary across different datasets and how they affect the method performance is necessary.
- The paper focuses on experiments with two modalities (e.g., EHR + CXR, audio-video). An exploration of how MASAM performs with more than two modalities would be valuable for assessing its scalability in more complex multimodal fusion tasks.

Minor Issues:

The used "Kinetics" dataset should be called the "Kinetics-Sounds" dataset. Kinetics-Sounds dataset is a subset of Kinetics.

### Questions
Please check the above section.

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
This paper proposes the MASAM method, which introduces the Sharpness Aware Minimization (SAM) strategy from machine learning into multimodal learning. By adjusting the loss function to prioritize model robustness, it enables each modality to
explore flatter minima along directions that align with shared information while remaining robust to heterogeneous loss landscapes, thereby improving overall performance.

### Strengths
⦁	This paper introduces Sharpness-Aware Minimization into the multimodal domain and further designs it specifically for the multimodal context, enhancing its applicability.

### Weaknesses
⦁	The baselines are weak. Only 1 work published in 2024 is compared and no works published in 2025 are compared. This comparision is not enough to show its SOTA performance.
⦁	 The writing is confusing. Fig 1b is not mentioned in the manuscript. Fig 2 is used until in page 6.
⦁	Tha ablation study should be performed on more datasets. Also, APS and MDPS should be compared to other choices to verift their advantages. 
⦁	The experimental section primarily focuses on interpreting dataset performance while neglecting an in-depth analysis of the method’s effectiveness.
⦁	The authors argue that MDPS aligns gradients between unimodal and fusion objectives, please give detailed verification.

### Questions
Refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
