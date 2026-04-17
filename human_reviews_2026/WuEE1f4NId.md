# Generalizable Diabetic Retinopathy Grading via Knowledge Constrained Concept Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
Diabetic retinopathy (DR) grading models often suffer a significant performance drop when deployed to unseen clinical domains. A promising strategy is to mirror the diagnostic process of clinicians, who rely on identifying specific pathological signs to make judgments. Concept-based models (CBMs) are well-suited for this, but their effectiveness often hinges on concept supervision, which is rarely available in medical imaging. To address this, we propose Knowledge Constrained Concept Learning (KCCL), a novel framework that achieves robust domain generalization through concept learning under knowledge constraints. We first curate DRL6k, a dataset of 6,000 fundus images with lesion annotations, and train a lesion detection model to provide concept supervision via knowledge distillation. However, directly using this supervision may introduce noise and inconsistencies. Therefore, KCCL employs a knowledge constraint mechanism: it leverages medical priors to correct implausible concept predictions and reduce the influence of those deviating from clinical expectations during distillation, while also directly penalizing the model for producing clinically inconsistent concept predictions. Extensive experiments on multiple unseen target datasets demonstrate that KCCL significantly outperforms state-of-the-art domain generalization and DR grading methods, achieving generalization by producing clinically coherent and interpretable predictions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the domain generalization problem in DR grading—a critical issue where models trained on one dataset fail to generalize across diverse clinical settings. The authors propose Knowledge Constrained Concept Learning (KCCL), a framework that integrates concept-based modeling, knowledge distillation, and medical knowledge constraints to achieve clinically consistent and generalizable predictions. The authors first curate a new `DRL6k` dataset, containing 6,000 fundus images annotated with four key lesion types (i.e. hard/soft exudates, microaneurysms, hemorrhages). They then train a lesion detection model as a teacher, and then constrain the student concept bottleneck model (CBM) via a knowledge-constrained distillation.

### Strengths
- The proposed method is timely and clinically meaningful. The paper tackles a major bottleneck in medical AI—domain generalization—with a clinically guided approach that aligns with human diagnostic reasoning.
- The integration of concept bottlenecks with knowledge-constrained distillation is interesting. The proposed three mechanisms (i.e. Self-Correction, Reweighting, Regularization) seamlessly integrate medical knowledge and balance robustness and interpretability.
- Evaluations span six unseen domains and include detailed ablations isolating each component’s contribution. The improvements are consistent and significant across both AUC and F1 metrics. The correlation heatmaps between learned lesion concepts and DR grades demonstrate clinically coherent reasoning.

### Weaknesses
- My biggest concern is the dependency on the curated lesion detection model. While DRL6k partially mitigates annotation scarcity, the overall performance still depends on the teacher model’s quality. The resting of the framework assumes access to a reasonably accurate lesion detector. Unfortunately, the curated DRL6k dataset appears to be quite small (~6,000 samples), which does not lead to a high accuracy of the lesion detection model (F1=83.3). The curation of the DRL6k dataset also relies on image-level labels. 

- Following my previous point, with the proposed settings, it seems that all the methods do not achieve a high accuracy in DR grading (with the highest F1=47.8). But training solely on e.g., Eyepacs and evaluating on APTOS can easily achieve an F1 score of 80. Then what is the point of applying the proposed framework ? 

- The use of only four lesion types in the curated dataset omits key proliferative indicators (e.g., neovascularization), potentially constraining the grading accuracy for advanced DR.

- Currently, the lesion detection model is a standard ResNet50. Can the authors try different teacher models like ViTs to see the effect of the teacher?

### Questions
- It appears that the proposed method is tailored to DR grading. Can the method be generalized to other tasks, such as lesion segmentation.
- The authors should provide some visualization of the lesion localization to verify if the model indeed focuses on lesion regions.
- While Figure 3 provides interpretability analysis, qualitative failure examples (e.g., false lesion activations under domain shift) would enhance transparency.

### Soundness
2

### Presentation
3

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
The paper proposes a framework for diabetic retinopathy (DR) grading that aims to improve cross-domain generalization. The authors integrate concept-based models (CBMs) with medical knowledge constraints, leveraging lesion detection and knowledge distillation to ensure clinically consistent and interpretable feature learning.

They construct a new DRL6k dataset with 6,000 annotated fundus images and introduce three components, including Self-Correction (SC), Distillation Reweighting (DR), and Knowledge Constrained Regularization (KCR), which are used to refine concept learning under medical priors. Experiments across multiple unseen datasets reportedly show performance improvements compared to prior domain generalization methods.

### Strengths
(1).Interpretability emphasis
The visual analyses showing better alignment between learned lesion concepts and disease grades highlight some degree of explainability.

(2). Structured integration of medical priors
The proposed knowledge constraints (hard and soft) are well-formulated and embedded in the training objective, enhancing interpretability.

(3). Relatively comprehensive experiments
Multiple datasets and ablation studies are included to assess contributions from different modules.

(4). Clear motivation and clinical grounding
The work identifies the domain shift problem in DR grading and builds from the observation that ophthalmologists rely on lesion-level reasoning.

### Weaknesses
(1). While the authors frame their method as introducing a “knowledge-constrained concept learning” approach, the underlying paradigm largely replicates well-established lines of work: for instance, Integrating Clinical Knowledge into Concept Bottleneck Models aligns CBMs with clinical knowledge for improved out-of-distribution (OOD) generalization [1]; several label-free or post-hoc CBM methods reduce reliance on concept annotations and perform concept distillation [2, 3]. Hence, the methodological contribution appears incremental rather than fundamentally novel.

[1] W. Pang, X. Ke, S. Tsutsui, and B. Wen, Integrating Clinical Knowledge into Concept Bottleneck Models, in Proc. MICCAI 2024.

[2] T. Oikarinen, S. Das, L. M. Nguyen, and T.-W. Weng, Label-Free Concept Bottleneck Models, ICLR 2023.

[3] D. Srivastava, G. Yan, and T.-W. Weng, VLG-CBM: Training Concept Bottleneck Models with Vision–Language Guidance, NeurIPS 2024.

(2). Underwhelming performance
The reported results are significantly lower than recent related paper [1], even simpler CNN-based networks, such as MobileNet and ResNet50, combined with data augmentation and regularization, can likely achieve comparable or better results[2,3].

[1].Che, Haoxuan, et al. "Towards generalizable diabetic retinopathy grading in unseen domains." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.

[2].Zhu, Wenhui, et al. "nnMobileNet: rethinking CNN for retinopathy research." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[3].Huang, Yijin, et al. "Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation." Diagnostics 13.10 (2023): 1664.


(3). The curated DRL6k dataset is small (6k images) and includes only four lesion types (EX, SE, MA, HE), omitting key indicators like neovascularization (NV) that are crucial for proliferative DR. Complex dataset like EyePACS (Hard to identify grading 0 and 1) is used only for training rather than for evaluation, making the claim of generalization less convincing[1].

[1] Sun, Rui, et al. "Lesion-aware transformers for diabetic retinopathy grading." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

(4). The paper does not compare against recent retinal foundation models such as RetFound [1], which already show strong cross-domain generalization in retinal and medical imaging tasks without explicit domain adaptation.

[1] Li et al., RetFound: Foundation Model for Retinal Disease Detection, Nature, 2023.

### Questions
While the motivation and interpretability goals are valuable, the overall contribution is incremental and empirically weak. A lot of recent literature should be included.

### Soundness
3

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
3

### Summary
This paper addresses domain generalization in diabetic retinopathy (DR) grading by proposing Knowledge Constrained Concept Learning (KCCL), which integrates concept-based models with knowledge distillation. The authors curate a DRL6k dataset of 6,000 fundus images with lesion annotations to train a lesion detection model. This model provides concept supervision through knowledge distillation to a concept-based model (CBM). To ensure clinically consistent concept learning, they introduce a dual-constraint mechanism: Self Correction directly fixes predictions violating medical knowledge, Distillation Reweighting adjusts loss weights based on deviation from clinical patterns, and Knowledge Constrained Regularization penalizes implausible concept predictions. The method is evaluated on CEM and CLAT architectures across six unseen datasets, achieving average AUC of 84.1% and F1-score of 46.1%, outperforming state-of-the-art by 4.0% and 3.9% respectively.

### Strengths
Originality: Novel combination of concept-based models with knowledge-constrained distillation specifically for DR grading. The dual correction-regularization mechanism is creative.

Quality: Extensive experiments across six datasets with two different CBM architectures. Comprehensive ablation studies validate each component. The achieved F1-scores surpass 0.44 on average, which is competitive with recent hybrid models achieving 0.99 on single datasets.

Clarity: Clear motivation and problem formulation. The knowledge constraint design (hard vs soft) is intuitive and well-explained.

Significance: Addresses a critical challenge in medical AI deployment. The 10.6% F1-score improvement on IDRID is particularly notable. The interpretability aspect through concept visualization adds clinical value.

### Weaknesses
Limited concept vocabulary: Only four lesion types are considered (EX, HE, MA, SE), missing important indicators like neovascularization for proliferative DR. Recent reviews emphasize the importance of comprehensive lesion detection including various DR features
.
Dependency on initial supervision: While claiming to alleviate annotation requirements, the method still requires the curated DRL6k dataset. The quality ceiling is bounded by the lesion detection model's 91.3% AUC.

Statistical prior limitations: The priors P and C may not generalize to populations with different disease distributions. No analysis of prior transferability across ethnicities or imaging protocols is provided.

### Questions
How does the method handle images with subtle early-stage lesions that may be missed by the 91.3% accurate lesion detector? Could this create a systematic bias against mild DR detection?

How sensitive is the method to the threshold τ=0.5? Did you experiment with adaptive or learnable thresholds?

The co-occurrence matrix C is learned from DRL6k. How would this transfer to populations with different disease patterns (e.g., different ethnic groups with varying DR progression patterns)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Knowledge Constrained Concept Learning (KCCL), a framework designed to improve domain generalization for diabetic retinopathy (DR) classification by leveraging concept-based reasoning. To support concept learning, the authors construct DRL6k, a curated dataset of fundus images annotated with key retinal lesions. A lesion detection model is first trained as a teacher, providing soft concept supervision. KCCL then employs a dual-constraint mechanism—combining medical knowledge–based correction and regularization—to refine the teacher’s concept predictions and enforce clinically consistent concept outputs in the student Concept based Model (CBM).

### Strengths
1. The method mirrors real clinical practice by explicitly using lesion detection to guide DR grading. This is a more principled approach than prior works, which typically demonstrate lesion awareness indirectly via attention maps rather than making lesions explicit reasoning units.
2. The self-correction and distillation-reweighting modules effectively handle noisy teacher outputs. This ensures that incorrect lesion detections do not degrade the student model’s reasoning, improving stability and preventing error propagation.
3. The method shows consistent and significant improvements across multiple unseen domains, demonstrating both generalization strength and interpretability benefits over existing domain generalization and concept bottleneck baselines.

### Weaknesses
1.The paper is somewhat difficult to follow, and the presentation can be improved. The introduction should include more method-specific motivation, and the system diagram should be revised for clarity and readability.
2.The approach leverages an additional curated dataset (DRL6k), whereas other domain-generalization baselines rely only on publicly available datasets. For a fully fair comparison, DRL6k should also be annotated and used to train the competing DG methods.
3.Incorporating stronger modern backbones (e.g., ViT or CLIP-based architectures) would make the empirical evaluation more compelling, given current trends in medical imaging and foundation models.
4.Domain generalization performance can be sensitive to random seeds. Reporting standard deviations across multiple runs would improve confidence in the robustness of the results.

Minor:
5.There are spelling errors (e.g., “systematically” in Line 51).
6.Table 3 should bold the best and underline the second-best results for easier interpretation.

### Questions
1. What are the training costs and overhead?
2. Weaknesses 2,3,4

### Soundness
3

### Presentation
2

### Contribution
3
