# AEMP: Autoregressive-Enhanced Masked Pre-training for Robust Indoor Localization

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6

## Abstract
The major obstacle for learning-based Channel State Information (CSI) localization is to obtain a high-quality large-scale annotated dataset. However, unlike visual datasets that can be easily annotated by human workers, CSI signals are  RF signal is non-intuitive and non-interpretable, making the annotation process both time-consuming and labor-intensive. Considering the potential of self-supervised learning to reduce reliance on labeled data, masked reconstruction has emerged as a promising alternative. However, directly applying existing designs to large-scale CSI scenarios faces unique challenges, including unstable representations in unmasked regions, inability to preserve long-range channel correlations, and high sensitivity to variations in access point layouts and propagation environments. To address these issues, we propose an autoregressive-enhanced masked pre-training (AEMP) framework. AEMP employs a hierarchical Transformer architecture where spatial subnetworks perform masked reconstruction to capture local channel features, while a temporal network enforces consistency through autoregressive prediction. In addition, multi-view fusion and span masking improve robustness under dynamic deployment conditions. Extensive experiments demonstrate that AEMP yields stable and transferable representations, achieving superior performance and strong generalization on downstream indoor localization tasks. To the best of our knowledge, this is the first pre-training framework for wireless sensing that integrates temporal prediction to complement masked reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the issue that traditional masked pre-training methods are unsuitable for WiFi CSI indoor localization signals, pointing out that these methods produce unstable representations and fail to capture global spatio-temporal correlations, making models highly sensitive to real-world deployment changes . To solve this, the paper proposes a novel hybrid framework called AEMP, which utilizes a spatial subnetwork for Masked Reconstruction while using a temporal subnetwork for Autoregressive Prediction. AEMP achieves state-of-the-art localization performance  and demonstrates strong generalization, especially in low-data scenarios. Additionally, the authors contribute a new large-scale real-world dataset called ISACLoc for this research.

### Strengths
This paper addresses the issue that traditional masked pre-training methods are unsuitable for WiFi CSI indoor localization signals, pointing out that these methods produce unstable representations and fail to capture global spatio-temporal correlations, making models highly sensitive to real-world deployment changes . To solve this, the paper proposes a novel hybrid framework called AEMP, which utilizes a spatial subnetwork for Masked Reconstruction while using a temporal subnetwork for Autoregressive Prediction  . AEMP achieves state-of-the-art localization performance and demonstrates strong generalization, especially in low-data scenarios. Additionally, the authors contribute a new large-scale real-world dataset called ISACLoc for this research.

### Weaknesses
“We employ a multi-view fusion strategy to reduce the reliance on specific AP combinations. In addition, we introduce a span masking mechanism (Joshi et al., 2020) to simulate dynamic deployment conditions in real-world scenarios.”
I dont see the ablation study  on these two.
Besides, I'm not sure the novelty of the MR and AP combination.

### Questions
Is the MR+AP combination really that useful? In other words, in your specific scenario, I feel the distinction between the two is not that significant. Also, I suggest experimenting on one or two public datasets to demonstrate the generalization performance.

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
3

### Summary
This paper proposes a self-supervised pre-training mechanism for CSI-based data-driven indoor localization systems that aims to capture the spatiotemporal dependencies of wireless signals, providing robust representations that enhance the robustness and generalization capabilities of downstream indoor localization systems. The Spatial component is learned by training the model to perform 2D mask reconstruction as well as the model's confidence in the mask prediction, represented by the predicted variance, while temporal consistency is taught to the model by auto-regressively predicting the future frames. Moreover, the authors collect a large-scale dataset of CSI measurements from a large number of APs coming from a diverse set of mobile phones, and from different geographical locations over several days. Several studies and evaluations are performed to quantify the impact of the two proposed sub-components, as well as some experiments to quantify the proposed approach's capability to make the downstream systems robust to varying deployment settings. Moreover, quantitative analysis shows that the proposed approach outperforms other pretraining methods in terms of median and tail localization error.

### Strengths
- The proposed work maintains both spatial and temporal properties of wireless signal propagation
- The proposed methodology is well motivated and decently evaluated.

### Weaknesses
- The work does not fully evaluate the impact of the different proposed components on the mentioned challenges. For example, it is not clear the contribution of the Masked Reconstruction (MR) on generalizing to new areas or across different devices
- The work is only evaluated on the constructed dataset, and it is not clear how well the performance gains from the proposed pre-training methods carry over to other well-established datasets.

Overall, this proposed work can be a significant contribution in enhancing indoor localization systems, but would benefit from some additional clarifications and experiments. Based on these clarifications and experiments from the authors, I would be willing to revise this score.

Clarifying the following points in writing or by performing minor additional experiments would help quantify the effects and impacts of several design choices:
- Which part of the dataset was used as a validation split?
- What is the impact of multi-view fusion in the MR component on the generalization capability of downstream models? What is the impact of an increasing number of combinations by dropping more than 1 AP?
- I think the fact that almost all the baselines are underperforming compared to systems without pretraining is counterintuitive and warrants discussing and possibly evaluating.
- Since the position of the APs are encoded in the input, can a model trained on one test area be directly evaluated on another? I believe including the performance without additional fine-tuning in Table 4 would be a good addition.
- It would be interesting to see how reproducible the results are on other existing and well-established datasets, and to show that the proposed approach is not over-fitting on some nuanced features of the environments where the data was collected. This would help strengthen the proposed approach, as well as validate the proposed dataset.
- What is the structure of the MLP used as a task-specific head?


Minor comments:
- In lines 216 and 217, the N-1 and N should be swapped.

### Questions
Please focus on the weakenesses.

### Soundness
2

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
3

### Summary
This paper proposes AEMP (Autoregressive-Enhanced Masked Pre-training), a self-supervised framework designed to improve robustness and generalization in indoor localization using WiFi Channel State Information (CSI). Traditional masked modeling fails to capture the spatiotemporal dependencies of CSI signals and is sensitive to environmental variations. To address these issues, AEMP integrates a hierarchical Transformer architecture with two complementary pretext tasks: masked reconstruction for learning spatial features and autoregressive prediction for temporal consistency. The authors also introduce span masking and multi-view fusion strategies to enhance resilience to deployment changes. Extensive experiments on the proposed ISACLoc dataset show that AEMP achieves superior localization accuracy and generalization compared with state-of-the-art baselines.

### Strengths
(1) The first pre-training framework for wireless sensing that combines masked modeling with autoregressive prediction to improve temporal coherence in CSI representation learning.

(2) A well-designed hierarchical Transformer architecture that separates spatial and temporal modeling via parameter-shared subnetworks.

(3) Introduction of span masking and multi-view fusion to simulate real-world dynamics and improve robustness to varying access point configurations.

(4) Comprehensive experiments, including cross-region, cross-device, and low-label scenarios, showing consistent improvements in accuracy and stability. Clear ablation and fine-tuning analyses demonstrating the role of each module in performance gains.

### Weaknesses
(1) The abstract does not effectively highlight the research gap or the motivation. Readers must infer the challenges being addressed from the introduction.

(2) Figure 2 is difficult to interpret due to small and cluttered text; visualization clarity is limited.

(3) While the gap is well-defined, the proposed solution lacks strong novelty, primarily integrating known ideas (masked modeling and autoregression) with moderate architectural adjustments.

(4) Although the method section is detailed, the flow is dense, and the conceptual link between the two tasks (masked reconstruction and autoregression) could be better articulated. 

(5) while thorough, results mostly compare against conventional pretraining baselines. Broader comparisons (e.g., contrastive or diffusion-based pretraining methods) would better support novelty claims.

(6) The ISACLoc dataset setup is interesting, but reproducibility could be improved by including open-source plans or quantitative data statistics.

### Questions
(1) The combination of Gaussian NLL loss for masking and weighted MSE for prediction is sensible, but how is the stability of training? 

(2) How sensitive is AEMP to the weighting and scheduling parameters (e.g., λ and η) in Equation (9)? Were these tuned heuristically or via validation?

(3) Since span masking and multi-view fusion both modify spatial inputs, how do they interact? Could one suffice without the other?

(4) How scalable is AEMP to different wireless protocols or hardware with different CSI formats—does the pretraining transfer effectively?

### Soundness
3

### Presentation
2

### Contribution
2
