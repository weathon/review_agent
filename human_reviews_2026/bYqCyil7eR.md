# Towards Unified Image Deblurring using a Mixture-of-Experts Decoder

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
Image deblurring, removing blurring artifacts from images, is a fundamental task in computational photography and low-level computer vision. Existing approaches focus on specialized solutions tailored to particular blur types, thus, these solutions lack generalization. This limitation in current methods implies requiring multiple models to cover several blur types, which is not practical in many real scenarios. In this paper, we introduce the first all-in-one deblurring method capable of efficiently restoring images affected by diverse blur degradations, including global motion, local motion, blur in low-light conditions, and defocus blur. We propose a mixture-of-experts (MoE) decoding module, which dynamically routes image features based on the recognized blur degradation, enabling precise and efficient restoration in an end-to-end manner. Our unified approach not only achieves performance comparable to dedicated task-specific models, but also shows promising generalization to unseen blur scenarios, particularly when leveraging appropriate expert selection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes the first all-in-one deblurring method capable of efficiently restoring images affected by diverse blur degradations, including global motion, local motion, blur in low-light conditions, and defocus blur. To achieve this, the authors propose a mixture-of-experts (MoE) decoding module, which dynamically routes image features based on the recognized blur degradation, enabling precise and efficient
restoration in an end-to-end manner. The proposed approach not only achieves performance comparable to dedicated task-specific models, but also shows promising generalization to unseen blur scenarios.

### Strengths
1. This paper proposes the first all-in-one deblurring method that can efficiently restore any blurry image. The motivation and contributions are technically sound, and the design of the mixture-of-experts (MoE) decoding module is novel.

2. The subsection “Deblurring Similarity Analysis” effectively introduces the importance of an all-in-one network for deblurring, enabling readers to easily understand the motivation behind the proposed method.

3. The paper is well written and easy to follow.

### Weaknesses
1. Although the motivation and idea are technically sound, the performance of the proposed DeMoE seems unsatisfactory. As shown in Table 1, the performance of DeMoE$_{k=1}$ works similar to the baseline NAFNet while with two times number of parameters. 

2. As the authors propose a new all-in-one model, it would be better to compare the proposed DeMoE with more recent all-in-one methods in Table 1, such as AdaIR [A] (ICLR 2025), MoCE-IR [B] (CVPR 2025), and DFPIR [C] (CVPR 2025).

3. Although the authors claim to use multiple experts, the experimental results in Tables 1, 2, and 3 only report the performance of a single expert, DeMoE$_{k=1}$, which appears somewhat inconsistent with the proposed multi-expert framework.

[A] AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation. In CVPR 2025.

[B] Complexity Experts are Task-Discriminative Learners for Any Image Restoration: . In CVPR 2025.

[C] Degradation-Aware Feature Perturbation for All-in-One Image Restoration. In CVPR 2025

### Questions
1. In Tables 1, 2, and 3, why do you report only the performance of a single expert, DeMoE$_{k=1}$.
 
 How about experts such as DeMoE$_{k=5}$?

2. Could your compare the proposed method with the latest all-in-one models, such as AdaIR [A] (ICLR 2025), MoCE-IR [B] (CVPR 2025), and DFPIR [C] (CVPR 2025).

3. I am curious about how the proposed method handles images that contain both motion blur and defocus blur. In some cases, one may wish to preserve the defocus blur while removing only the motion blur. However, your method appears to remove both types of blur simultaneously.

[A] AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation. In CVPR 2025.

[B] Complexity Experts are Task-Discriminative Learners for Any Image Restoration: . In CVPR 2025.

[C] Degradation-Aware Feature Perturbation for All-in-One Image Restoration. In CVPR 2025

### Soundness
3

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
This paper aims to propose a unified image deblurring model to tackle various blur types in the real world. The paper first investigated the blur type difference of deblurring datasets by the network similarity study.  Then, a mixture-of-experts decoder is introduced to deal with various blur types in a divide-and-conquer way. The performance is evaluated across several deblurring benchmarks, including synthetic and real blur scenarios.

### Strengths
- Network similarity study provides a novel and insightful analysis of the blur types in different deblurring datasets. The conclusion also contributes to the community.
- The MoE structure yields an efficient method compared with previous sotas.
- The paper is well-written and easy to follow.

### Weaknesses
- The effect of general deblurring is not satisfactory. In Table 2, DeMoE without manual selection cannot surpass previous methods in both RealDOF and Real-LOLBlur. Meanwhile, the reviewer considers manually selecting experts as a task-specific method, since the type of blur in the input image should be unknown in the unified deblurring scenario.
- The MoE router is trained with the ground-truth degradation label. Where does the label come from? 
- The allocation of the router in the MoE module needs to be analyzed using various deblurring datasets to verify whether it draws consistent conclusions with the network similarity study section.
- DeMoE is trained by the proposed AIO-Blur dataset. How about other rivals listed in the experimental results? A fair comparison is supposed to use the same training data.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

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
This paper proposes a unified image deblurring framework, DeMoE (Mixture-of-Experts Decoder), which employs a mixture-of-experts mechanism to handle multiple blur types within a single model. Through network similarity analysis, the authors find strong parameter correlations among task-specific deblurring models, motivating a router-controlled MoE decoder that dynamically selects experts for different blur types. Additionally, the paper constructs the AIO-Blur dataset, integrating multiple blur scenarios for unified training and evaluation.

### Strengths
1. Comprehensive experiments: The study includes AIO-Blur and OOD testing, task-specific comparisons, ablation studies, and efficiency analyses, offering broad coverage and credible conclusions.
2. Practical significance: DeMoE serves as a unified framework applicable to diverse blur scenarios, reducing the need for multiple specialized models and showing potential for real-world deployment.

### Weaknesses
1. This article lacks significant innovation, and its multi expert strategy is very common in all-in-one image restoration, only changing the focus of different restoration tasks in all-in-one image restoration to different scenes under the single task of deblurring.

2. Lack of algorithmic overview: Although Figure 3 shows the network architecture, a concise workflow summarizing differences between training and inference stages is missing. 

3. Limited router generalization: The router performs poorly on OOD datasets, leading to incorrect expert selection. While manual expert control mitigates this, it undermines the “dynamically unified model” objective.

4. Unclear expert sharing and independence mechanism : The paper does not clarify whether expert parameters are fully independent or partially shared, nor analyze how this design affects model capacity and generalization.

5. In Table 1, DeMoE fails to reach SOTA on nearly half of the six datasets. On RealBlur, DeMoE is inferior to SFHFormer which has less parameters; on ReLoBlur and DPDD, DeMoE shows no advantage over NAFNet which has less Computational Cost.

6. In Table 3, DeMoE performs near the bottom among compared methods, indicating unsatisfactory results.

7. Labeling error: In Table 3 (ReLoBlur results), two different SSIM values are both marked as SOTA. SSIM(LBAG)=0.9249 and SSIM(DeMoEk=1)=0.925, but the former is marked as 2nd best.

### Questions
1. What are the differences between DeMoE’s training and inference stages? Consider adding pseudocode or a system flowchart for clarity.

2. Has the router’s classification accuracy been evaluated on OOD datasets? Would uncertainty-based or entropy-based gating improve its generalization?

3. Since the encoder is shared while each expert in the decoder has independent convolutional modules, please analyze this “partially shared + partially independent” design trade-off. Specifically, does full independence improve performance? Could partial sharing enhance generalization or parameter efficiency? An ablation or comparative study is recommended.

### Soundness
2

### Presentation
3

### Contribution
2
