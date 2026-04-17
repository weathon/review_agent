# Pretrain then Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Text-based person retrieval faces inherent limitations due to data scarcity, driven by stringent privacy constraints and the high cost of manual annotation. To mitigate this, existing methods usually rely on a \textbf{Pretrain-then-Finetune} paradigm, where models are first pretrained on synthetic person-caption data to establish cross-modal alignment, followed by fine-tuning on labeled real-world datasets. However, this paradigm lacks practicality in real-world deployment scenarios, where large-scale annotated target-domain data is typically inaccessible. 
In this work, we propose a new \textbf{Pretrain-then-Adapt} paradigm that eliminates reliance on extensive target-domain supervision. 
The key underpinning our approach is Uncertainty-Aware Test-Time Adaptation (UATTA), a framework enabling dynamic model adaptation using only unlabeled test data, with minimal computational overhead. 
UATTA introduces a bidirectional retrieval disagreement mechanism to estimate uncertainty, \ie, low uncertainty is assigned when an image-text pair ranks highly in both image-to-text and text-to-image retrieval, indicating high alignment; otherwise, high uncertainty is detected. 
This indicator drives test-time model recalibration without labels, effectively mitigating domain shift. 
We validate UATTA on four benchmarks, \ie,  CUHK-PEDES, ICFG-PEDES, RSTPReid, and PAB, showing consistent improvements across both CLIP-based (one-stage) and XVLM-based (two-stage) frameworks. 
Ablation studies confirm that UATTA outperforms existing test-time adaptation strategies, establishing a new benchmark for label-efficient, deployable person retrieval systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper pioneers the exploration of the text-based person retrieval task in the absence of labeled target-domain data. The authors propose an uncertainty-aware test-time adaptation framework based on a bidirectional retrieval disagreement mechanism, which measures the discrepancy between image-to-text and text-to-image matching. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is well written, and I am able to clearly understand most of its contributions.
2. This work pioneers the exploration of text-based person retrieval without relying on labeled target-domain data, which could alleviate the dependence on expensive human annotations.

### Weaknesses
1. **Logical Flaw**: The proposed bidirectional retrieval pipeline assumes access to the entire set of test textual queries in advance. In real-world scenarios, however, text-based person retrieval is typically performed in an online manner, where user queries arrive sequentially and are not known beforehand. This assumption significantly limits the practical applicability of the method, as it does not align with the dynamic nature of actual deployment environments.
2. **There is inconsistency in terminology**: the title refers to "text-based person search," whereas the main text uses "text-based person retrieval." Although both terms are acceptable, I recommend maintaining consistency throughout the paper.
3. **The paper lacks a unique and in-depth discussion of the text-based person retrieval task itself.** In fact, many existing vision-language cross-modal tasks (e.g., image/video-text retrieval) face similar challenges with expensive text annotation. Is the proposed method applicable to these tasks as well?
4. **The paper does not thoroughly discuss existing test-time adaptation methods.** While related work is mentioned, it remains unclear whether current approaches can be directly applied to text-based person retrieval, and what specific issues the proposed method addresses when adapting test-time adaptation techniques to this task.
5. **The experimental section lacks citations to several key related works.**

### Questions
1. Equation 2 uses ground-truth labels from the test set for model optimization. Does this violate standard protocol? Typically, ground-truth labels are used only for evaluation, not for model training or adaptation.
2. Can X-VLM be combined with other test-time adaptation methods to further improve performance? In Table 1(a), only results for X-VLM + Ours are shown, and comparisons with other related methods are missing.

### Soundness
2

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
3

### Summary
This paper proposes Pretrain-then-Adapt, an innovative framework designed for uncertainty-aware test-time adaptation (UATTA) in text-based person retrieval. The authors aim to address the domain shift problem, which is often observed in models trained on synthetic data but deployed in real-world environments. The method uses unlabeled test data at test time to adapt the model with minimal computational overhead. The core idea is to estimate prediction uncertainty through a bidirectional retrieval disagreement mechanism, which allows the model to recalibrate itself without relying on labeled data. The paper evaluates this approach across multiple benchmarks, showing improvements over traditional Pretrain-then-Finetune methods and existing test-time adaptation strategies.

### Strengths
1. The idea of uncertainty-aware adaptation for test-time adaptation (TTA) in text-based retrieval is interesting and addresses a significant gap in existing research. The use of unlabeled test data to adapt models in an efficient manner is a valuable contribution to practical deployment scenario.

2. The framework is designed for real-world applications where fine-tuning on labeled data is infeasible. This makes it a promising alternative to traditional pretrain-then-finetune paradigms.

### Weaknesses
1. The confidence filtering mechanism used in the framework is rather simple and somewhat “tricky”. It relies on basic engineering methods, such as disagreement between text-to-image and image-to-text retrieval. While effective, it does not appear to bring significant methodological innovation to the table. This aspect could be seen as a relatively basic engineering approach rather than a novel contribution to the field.

2. Despite its efficiency, the proposed method shows comparatively larger performance gaps when evaluated against fine-tuning-based methods, making it difficult to validate its effectiveness.

### Questions
See weakness

### Soundness
3

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
4

### Summary
This paper proposes a new paradigm for text-based person retrieval called “Pretrain-then-Adapt,” which replaces the traditional “Pretrain-then-Finetune” approach. The core contribution is an Uncertainty-Aware Test-Time Adaptation (UATTA) framework that allows models to adapt to new target domains using only unlabeled test data, without requiring labeled target-domain data. It offers a label-free, uncertainty-driven test-time adaptation method that enhances the robustness and deployability of text-based person retrieval systems in real-world scenarios.

### Strengths
1. This paper introduces a label-free "Pretrain-then-Adapt" framework, eliminating the need for annotated target data.

2. This paper proposes a simple yet effective bidirectional retrieval disagreement to reliably detect uncertainty and prevent overconfidence.

3. The experiments demonstrate some improvements across multiple models and benchmarks with minimal computational cost.

### Weaknesses
1. The proposed "Pretrain-then-Adapt" paradigm appears functionally similar to unsupervised domain adaptation (UDA) or unsupervised learning for text-based ReID. The core distinction—adapting only a few parameters (LN layers) at test time—is an incremental engineering contribution rather than a foundational shift in paradigm, as claimed.

2. The core innovation, using bidirectional retrieval disagreement as an uncertainty signal, is a straightforward application of cross-modal consistency. This concept is a well-established principle in cross-modal and unsupervised learning. The method essentially uses this consensus as a pseudo-label quality signal, an approach that is not novel in the broader machine learning context.

3. The empirical evaluation lacks critical comparisons with state-of-the-art unsupervised text-based ReID methods. This omission makes it difficult to assess the true contribution of UATTA versus simply applying established unsupervised learning techniques to the test set.

4. The paper relies solely on quantitative results. Qualitative visualizations (e.g., t-SNE plots of features before/after adaptation, examples of successful/failed adaptations) are missing. Such analysis would greatly strengthen the claim of mitigating domain shift and provide intuitive insights into the method's behavior.

5. The theoretical derivation linking parameter variance to bidirectional disagreement, while a nice addition, relies on strong assumptions (e.g., symmetric consistency for an ideal model) and first-order approximations. A more rigorous analysis or an ablation on the impact of this specific theoretical motivation is needed.

### Questions
What is the fundamental conceptual and methodological distinction between the proposed "Pretrain-then-Adapt" paradigm and standard Unsupervised Domain Adaptation (UDA) or unsupervised learning for text-based ReID? The process of adapting a pre-trained model using unlabeled target data appears to align closely with the core objective of UDA. Please clarify the novel theoretical or practical contribution of the paradigm itself, beyond the specific mechanism of UATTA.

For other questions, please refer to the points raised in the Weaknesses section.

### Soundness
2

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
This paper proposes UATTA (Uncertainty-Aware Test-Time Adaptation), a novel framework for text-based person retrieval that eliminates the need for labeled target-domain data. The authors introduce a new paradigm — Pretrain-then-Adapt, replacing the traditional Pretrain-then-Finetune pipeline. The key idea is to leverage Bidirectional Retrieval Disagreement (BRD) as an uncertainty measure between text-to-image and image-to-text retrieval rankings. This uncertainty guides test-time entropy minimization to avoid overconfident false positives during adaptation.The method is theoretically grounded through an analysis showing proportionality between parameter variance and retrieval disagreement, and empirically validated on four datasets (CUHK-PEDES, ICFG-PEDES, RSTPReid, and PAB) using both CLIP and XVLM frameworks. UATTA improves retrieval accuracy while requiring minimal computational cost and no labeled data.

### Strengths
1. The proposed Pretrain-then-Adapt pipeline directly addresses the impracticality of relying on labeled target-domain data in text-based person retrieval. This design is timely and pragmatic for real-world deployment under privacy and resource constraints.

2. Extensive evaluations on several benchmarks confirm consistent gains, and the proposed method can achieve competitive results while requiring only 0.08 GPU-hours for adaptation. The ablation studies are thorough and insightful.

### Weaknesses
1.This paper adopts many hand-crafted parameters and designs, like K and uncertainty formulation. Would these designs affect the generalization performance across different models?

2.The authors only update LayerNorm affine parameters to ensures stability, but it might limit adaptability. Hence, could lightweight tuning modules (e.g., LoRA, adapters) provide a better performance–efficiency tradeoff?

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2
