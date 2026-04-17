# EvoProto: Evolving Prototypes with Class Similarity for Weakly Incremental Segmentation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Weakly Supervised Incremental Learning for Semantic Segmentation (WILSS) seeks to segment new classes using only image-level labels, without access to old class data, which challenges the stability-plasticity balance. The absence of pixel-level annotations for new classes and historical data for old classes often leads to class overwriting, where predictions for new classes misclassify or override regions belonging to semantically similar previously learned classes. We observe that such overwriting frequently arises from class confusion, where visually similar classes are entangled due to weak supervision and limited feature discrimination. To address this, we propose EvoProto, a framework that explicitly models and mitigates class confusion through the dynamic evolution of trainable class prototypes. We begin by introducing a confusion score that quantifies semantic similarity between new and old classes. Computed from CAM-derived predictions after a warm-up phase, this score is transformed into adaptive weights that guide both contrastive prototype learning and prototype-level knowledge distillation, thereby reinforcing inter-class separability during continual updates. Besides, each class in EvoProto is associated with a learnable prototype vector, which is continuously updated during training to better capture class-specific semantics and improve discriminability under weak supervision. Additionally, to counter the degradation in classification capability and the resulting pseudo-label noise during incremental steps in weak supervision, we propose a CAM Channel Selection mechanism that emphasizes confident and consistent activations as more reliable supervision. Extensive experiments on Pascal VOC and COCO benchmarks demonstrate that EvoProto effectively alleviates class overwriting and achieves state-of-the-art performance under various incremental scenarios. The code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of Weakly Supervised Incremental Learning for Semantic Segmentation (WILSS), where models must learn to segment new classes with only image-level labels, without accessing previous class data. The authors identify class overwriting， the misclassification of old classes as new ones due to class confusion, as a key failure mode in WILSS. This paper provide a framework that explicitly models and mitigates class confusion through dynamic prototype evolution guided by inter-class similarity. Experiments on Pascal VOC  and COCO-to-VOC under multiple incremental protocols (10-10, 15-5, 10-5, 10-2) show state-of-the-art results.

### Strengths
After reading this paper carefully, I think this paper has the following strengths:

- The idea of quantifying inter-class confusion and evolving prototypes accordingly is novel and well-motivated for the WILSS setting. Integrating contrastive learning and prototype-level distillation under confusion-aware reweighting is a creative and non-trivial combination.

- The paper is generally well-written and structured, with logical flow from motivation to implementation.

- The method is rigorously formulated, with mathematical clarity on confusion scoring, reweighting, and objective functions.

- The paper targets an emerging and challenging area (WILSS) with practical importance: learning segmentation models from limited supervision under continual updates.

### Weaknesses
The following weaknesses should be solved to improve the quality of this paper:

- While the confusion-aware weighting is new, the prototype evolution concept shares similarities with prior prototype refinement works such as RePRIand PLOP. The contribution could be perceived as incremental rather than fundamentally novel.

- The paper defines a symmetric confusion score from pseudo-labels but does not analyze its robustness to CAM noise or how it behaves across steps. It’s unclear whether confusion scores remain stable or amplify errors over time; this could undermine reliability.

- The framework introduces multiple prototype-level losses and dynamic confusion estimation per epoch. Runtime or memory overhead is not discussed.

- The ALD module uses activation thresholds but the rationale for the averaging bias “+1” and its hyperparameter sensitivity is underexplained. The section could be clearer on its interaction with confusion-aware objectives.

### Questions
I have some questions fior this paper:

Q1: How does the model prevent confusion scores from being corrupted by noisy pseudo-labels, especially early in incremental steps? Have the authors tried smoothing or memory-based averaging?

Q2: Would EvoProto generalize to instance segmentation or open-vocabulary segmentation tasks?

### Soundness
3

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
4

### Summary
The paper introduces EvoPrototype to address the problem of class overwriting in weakly supervised incremental learning for semantic supervision. Class overwriting stems from the confusion between semantically similar old and new classes. EvoPrototype addresses this by learning the prototypes with contrastive learning to maximize the discrimnation and distillation to preserve old class performance. EvoProtoype constructs a confusion matrix M to indicate the bidirectional confusion measurement between classes. M is then (1) uses to reweight the prototype contrastive loss with the most confusing class (CO_i); and (2) determines how much to distill from the previous step, where it assigns low weight to confusing classes. Furthermore, the authors introduce Activation-based label denoising to refines the image-level labels of old classes in new steps, using the average of the maximal activation across different activation maps as the threshold to binarize. EvoPrototype demonstrates state of the art on multiple datasets in different settings on both ViT and ResNet backbone.

### Strengths
1. Although the use of confusion matrix has been explored to reweight class-wise loss in [1], this paper utilize the matrix in a clever way for the task of WILSS.
2. The method demonstrates consistently strong results accross benchmarks.
3. The writing is easy to follow.
4. Extensive experiments are conducted.

[1] https://proceedings.mlr.press/v238/zhang24e/zhang24e.pdf

### Weaknesses
1. Although being introduced and cleverly used, the contribution of the confusion matrix should be ablated (results when all indicies is 1) to quantify its effect.
2. The reviewer is not convinced on using the average across class activation as the threshold for all classes because the classes should be independent.
3. Currently, there are many strong general natural open-vocal segmentation/grounding models, which makes this task less relevant. unless it is applied in specific domain that those foundation models underperform (e.g., medical imaging). It could be more relevant to see how this adapts in these special domain rather on natural images.

### Questions
1. Is there any results to quantify the effectiveness of the confusion matrix?
2. In figure 1, the reviewer finds the bottom plots ambiguous. Why is the y-axis cuttof, and what is the value of sheeps and trains?
3. In Equation 4, w_{ij} = 0 for classes that are not the most confused counter part, why do we need this and can we set it to the original value because it naturally lower than the CO class?
4. Could the author elaborate how equation 5 ensure that i and CO_i are always selected at the old-new boundary?
5. The threshold thre is obtained as the average of maximum activation of each class i in equation (8). Reviewer finds that the activation of each class should be indepdent, could the author justify how the average across classes is a good value for threshold?
6. The paper claims that class overwriting is caused by class confusion, is there any quantitative analysis on this cause and effect? (e.g., the correlation between pair-wise class semantic similarity and their mIoU, where low mIoU indicates class overwriting)

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenging problem of Weakly Supervised Incremental Learning for Semantic Segmentation (WILSS), where new categories must be segmented using only image-level labels, and old data are unavailable. The authors identify class overwriting—the misclassification of old regions as new ones—as a key obstacle caused by class confusion under weak supervision. To address this, the paper proposes EvoProto, a framework that dynamically evolves learnable class prototypes guided by a confusion score quantifying semantic similarity between classes. The confusion-aware adaptive weights regulate both contrastive prototype learning and prototype-level distillation, enabling better inter-class separation. Additionally, an Activation-based Label Denoising (ALD) module enhances pseudo-label reliability. Experiments on Pascal VOC and COCO show consistent improvements over prior WILSS baselines, confirming the framework’s effectiveness.

### Strengths
Clearly identifies class confusion as the root cause of class overwriting in WILSS, providing a strong conceptual motivation.

The EvoProto framework is well-designed and technically sound, integrating prototype evolution, adaptive reweighting, and knowledge distillation in a coherent manner.

The confusion score is intuitive and bridges semantic similarity and optimization dynamics, offering interpretability.

The ALD module effectively complements the prototype evolution mechanism, improving pseudo-label quality under weak supervision.

Extensive experiments on Pascal VOC and COCO, along with ablation and visualization, convincingly demonstrate the model’s advantages over state-of-the-art methods.

Writing is clear, and figures (especially confusion visualization) help convey the intuition.

### Weaknesses
While the proposed EvoProto framework is well-structured, several concerns limit its novelty and practicality. First, the mechanism by which the adaptive weights evolve across incremental stages remains insufficiently explained, and it is unclear whether such adaptive reweighting may introduce error accumulation over time. Second, the overall innovation appears moderate, as the paper lacks an in-depth discussion of related works in fine-grained recognition and continual incremental learning. Third, the framework involves numerous hyperparameters (e.g., k, γ, τ), yet their sensitivity is not analyzed, making the method complex and potentially difficult to reproduce. In addition, without any released implementation, the computational overhead introduced by prototype evolution and the activation-based label denoising (ALD) module may be nontrivial; a complexity or runtime comparison would substantially strengthen the contribution. Finally, the paper does not clarify how the model behaves when the ratio between new and old classes varies, which is important for assessing its robustness under different incremental settings.

### Questions
N/A

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
4

### Summary
This work focuses on weakly supervised incremental learning for semantic segmentation, where pixel-level annotations for new classes and historical data for old classes are unavailable. To address this, it employs CAM to generate pseudo labels for training. However, due to the lack of precise annotations, the model tends to suffer from overwriting and class confusion. To mitigate this issue, a confusion matrix is introduced to model inter-class relationships. The proposed method is evaluated on the Pascal VOC and COCO benchmarks and compared with multiple baselines, achieving state-of-the-art performance and even surpassing some methods that rely on pixel-level annotations.

### Strengths
1、This paper achieves state-of-the-art performance on short-task, long-task, and cross-dataset incremental learning settings, demonstrating further improvements over existing methods.

2、The proposed approach is not limited to a specific model architecture and has been validated on both ResNet and ViT backbones.

### Weaknesses
1、The changes in the “Train” mIoU curves in Figure 1 are not clearly presented.

2、In Eq. 1, the meanings of u and v are not clearly defined.

3、The framework diagram in Figure 2 lacks clarity and appears somewhat confusing. For example, the “class prototype pool” in the upper-right corner seems unnecessary, and among the two BCE losses, the one shown at the bottom of the figure is not clearly explained.

### Questions
1、Weakly Incremental Learning for Semantic Segmentation (WILSS) incrementally learns new classes using only image-level supervision. However, in Figure 2, it is unclear where the GT (ground truth) labels come from.

2、Eq. 3 implies M_{i, j} = M_{j, i}. However, the confusion matrix M in Figure 2 appears asymmetric, which might be due to certain normalization or visualization processing. Could you please clarify this?

3、Because the class activation map A ∈ [0, 1]^p obtained from CAM cannot be directly used as pseudo labels for semantic segmentation, the authors introduce a thresholding scheme in Eq. 8 to generate binary masks. However, the effectiveness of this threshold selection of concerning, as it remains unclear whether the chosen threshold is accurate or optimal.

4、In the overall objective shown in Eq. 11, multiple loss terms are included, but the BCE loss mentioned in Figure 2 and Section 4.1.2 is missing, which appears to be inconsistent.

5、The overall objective involves multiple hyperparameters, which makes parameter selection challenging.  It is unclear how the optimal parameters were determined, and corresponding ablation studies on these hyperparameters are needed to support the choice.

6、The proposed method relies on CAM to generate pseudo labels, which are then used for training. As the quality of these pseudo labels intuitively determines the overall performance of the method, would adopting more advanced class activation mapping techniques, such as Grad-CAM or Grad-CAM++, help improve the pseudo-label quality and consequently enhance model performance?

### Soundness
3

### Presentation
2

### Contribution
3
